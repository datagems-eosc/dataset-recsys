"""Command-line demo of the MathE recommendation API.

Sends the request MathE would send for a question to the document and video
endpoints, times each call, and shows each recommended material with its title,
topic and the keywords it shares with the question.

    poetry run python demo/mathe_api_demo.py --base-url "$BASE_URL"
    poetry run python demo/mathe_api_demo.py --base-url "$BASE_URL" --exact-text --repeat 3

Credentials come from `.env` (DATAGEMS_AUTH_URL, DATAGEMS_CLIENT_ID,
DATAGEMS_USER, DATAGEMS_PASSWORD, DATAGEMS_SCOPE), as in
docs/source/mathe_recsys_testing.md. The token is never printed.

Titles and the exact question text are read from the MathE mirror database
(read-only) when it is reachable (VPN / port-forward and DATAGEMS_POSTGRES_*
in `.env`); otherwise the demo shows material IDs only.

The API serves a stored ranking (fast) only when the question text matches the
text the ranking was built from; any other text is scored live (slower).
`--exact-text` sends the question text stored in MathE for that ID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_env() -> None:
    """Load the repository's .env, wherever the demo is started from."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def base_url(value: Optional[str] = None) -> str:
    """The API base URL from the argument or BASE_URL, with a clear error if missing."""
    load_env()
    url = (value or os.getenv("BASE_URL") or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        sys.exit("Set BASE_URL to the dev API address, e.g. https://<host> "
                 "(in .env, the environment, or the notebook's setup cell)")
    return url

DEFAULT_QUESTION_ID = "82"
DEFAULT_QUESTION = r"Differentiate $\displaystyle{y = 2x\sin(x)}$."
BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def _style(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def get_token() -> str:
    """Password-grant token from the DataGEMS identity provider."""
    load_env()
    names = ["DATAGEMS_AUTH_URL", "DATAGEMS_CLIENT_ID", "DATAGEMS_USER", "DATAGEMS_PASSWORD", "DATAGEMS_SCOPE"]
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    body = urllib.parse.urlencode({
        "grant_type": "password",
        "client_id": os.environ["DATAGEMS_CLIENT_ID"],
        "username": os.environ["DATAGEMS_USER"],
        "password": os.environ["DATAGEMS_PASSWORD"],
        "scope": os.environ["DATAGEMS_SCOPE"],
    }).encode()
    request = urllib.request.Request(os.environ["DATAGEMS_AUTH_URL"], data=body, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            token = json.load(response).get("access_token", "")
    except urllib.error.HTTPError as error:
        sys.exit(f"Could not get a token: HTTP {error.code}")
    if not token:
        sys.exit("Could not get a token: no access_token in the response")
    return token


def recommend(base_url: str, token: str, content: str, payload: dict) -> tuple[dict, float]:
    """POST one recommendation request; return the response and its time in ms."""
    url = f"{base_url.rstrip('/')}/dataset-recsys/mathe/recommend/{content}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"{content}: HTTP {error.code} {error.read().decode(errors='replace')[:300]}")
    return body, (time.perf_counter() - start) * 1000


class Mirror:
    """Optional read-only access to the MathE mirror: question and material metadata."""

    def __init__(self) -> None:
        self.conn = None
        try:
            from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient

            self.conn = MatheMirrorClient().conn
        except Exception as error:  # demo still runs with IDs only
            print(_style(f"(MathE mirror not reachable, showing IDs only: {type(error).__name__})", YELLOW))

    def question_text(self, question_id: str) -> Optional[str]:
        info = self.question_info(question_id)
        return info.get("question") if info else None

    def question_info(self, question_id: str) -> dict:
        """Text, topic, subtopic and keywords of a question."""
        if self.conn is None:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT q.question, t.name, s.name,
                       ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL)
                FROM platform__sna__questions q
                LEFT JOIN platform__topic t ON q.topic = t.id
                LEFT JOIN platform__subtopic s ON q.subtopic = s.id
                LEFT JOIN platform_keyword_snaquestion qk ON q.id = qk.platformsnaquestionid
                LEFT JOIN platform__keywords k ON qk.platformkeywordid = k.id
                WHERE q.id = %s
                GROUP BY q.question, t.name, s.name
                """,
                (int(question_id),),
            )
            row = cur.fetchone()
        if not row:
            return {}
        return {"question": row[0], "topic": row[1] or "", "subtopic": row[2] or "", "keywords": sorted(row[3] or [])}

    def materials(self, material_ids: list[str]) -> dict[str, dict]:
        """Title, kind, topics, subtopics and keywords of each material."""
        ids = [int(i) for i in material_ids if str(i).isdigit()]
        if self.conn is None or not ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.title, m.type, LOWER(m.file_ext),
                       ARRAY_REMOVE(ARRAY_AGG(DISTINCT t.name), NULL),
                       ARRAY_REMOVE(ARRAY_AGG(DISTINCT s.name), NULL),
                       ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL)
                FROM platform_materials m
                LEFT JOIN material_top_sub mts ON m.id = mts.platformmaterialid
                LEFT JOIN platform__topic t ON mts.platformtopicid = t.id
                LEFT JOIN platform__subtopic s ON mts.platformsubtopicid = s.id
                LEFT JOIN platform_material_keyword mk ON m.id = mk.platformmaterialid
                LEFT JOIN platform__keywords k ON mk.platformkeywordid = k.id
                WHERE m.id = ANY(%s)
                GROUP BY m.id, m.title, m.type, m.file_ext
                """,
                (ids,),
            )
            rows = cur.fetchall()
        out = {}
        for mid, title, mtype, ext, topics, subtopics, keywords in rows:
            kind = "video" if mtype in (1, 2) else (ext.upper() if ext else "document")
            out[str(mid)] = {"title": (title or "").strip(), "kind": kind, "topics": topics or [],
                             "subtopics": subtopics or [], "keywords": keywords or []}
        return out

    def titles(self, material_ids: list[str]) -> dict[str, str]:
        return {i: m["title"] for i, m in self.materials(material_ids).items()}


def relation(question: dict, material: dict) -> dict:
    """Why a material relates to the question: shared topic/subtopic and keywords."""
    q_kw = {k.lower(): k for k in question.get("keywords", [])}
    shared = [q_kw[k.lower()] for k in material.get("keywords", []) if k.lower() in q_kw]
    same_topic = question.get("topic") in material.get("topics", []) and (
        not question.get("subtopic") or question.get("subtopic") in material.get("subtopics", [])
    )
    return {"same_topic": same_topic, "shared_keywords": sorted(set(shared))}


def show(content: str, body: dict, times_ms: list[float], details: dict[str, dict], question: dict) -> None:
    recs = [str(r.get("material_id")) for r in body.get("recommendations", [])]
    timing = " · ".join(f"{t:,.0f} ms" for t in times_ms)
    print(f"\n{_style(content.capitalize(), BOLD)}  {_style('returned in ' + timing, GREEN)}")
    if not recs:
        print("  no recommendations (no material of this kind fits the question's topic)")
    for rank, material_id in enumerate(recs, 1):
        m = details.get(material_id)
        if not m:
            print(f"  {rank}. [{material_id}]")
            continue
        why = relation(question, m)
        tag = f"[{m['kind']} · id {material_id}]"
        print(f"  {rank}. {_style(m['title'] or '(untitled)', BOLD)}  {_style(tag, DIM)}")
        topic = "same topic and subtopic" if why["same_topic"] else " / ".join(m["topics"])
        kws = ", ".join(why["shared_keywords"]) or "none (ranked on content similarity)"
        print(f"     {topic} · shared keywords: {kws}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="API base URL (or BASE_URL env var)")
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="question text (LaTeX)")
    parser.add_argument("--exact-text", action="store_true", help="send the question text stored in MathE")
    parser.add_argument("-n", type=int, default=4, help="maximum number of recommendations")
    parser.add_argument("--repeat", type=int, default=1, help="calls per endpoint, to show stable timing")
    args = parser.parse_args()
    args.base_url = base_url(args.base_url)

    mirror = Mirror()
    info = mirror.question_info(args.question_id)
    question = args.question
    if args.exact_text:
        if not info:
            print(_style("(could not read the stored question text; using --question)", YELLOW))
        else:
            question = info["question"]

    token = get_token()
    payload = {"question_id": args.question_id, "question": question, "n": args.n}
    print(_style("Request (as MathE sends it):", BOLD))
    print(f"  question_id  {payload['question_id']}")
    print(f"  question     {payload['question']}")
    print(f"  n            {payload['n']}")
    if info:
        print(f"  topic        {info['topic']} / {info['subtopic']}")
        print(f"  keywords     {', '.join(info['keywords']) or '-'}")

    for content in ["documents", "videos"]:
        times, body = [], {}
        for _ in range(max(args.repeat, 1)):
            body, elapsed = recommend(args.base_url, token, content, payload)
            times.append(elapsed)
        ids = [r.get("material_id") for r in body.get("recommendations", [])]
        show(content, body, times, mirror.materials(ids), info)
    print()


if __name__ == "__main__":
    main()
