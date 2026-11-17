import os
import time

import requests
import urllib3
from dotenv import load_dotenv

from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile

# load_dotenv()

# The SCAYLE server uses a self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SCAYLE_BASE_URL = os.getenv("SCAYLE_BASE_URL", "").rstrip("/")
SCAYLE_USERNAME = os.getenv("SCAYLE_USERNAME")
SCAYLE_PASSWORD = os.getenv("SCAYLE_PASSWORD")
VERIFY_SSL = os.getenv("SCAYLE_VERIFY_SSL", "false").lower() in {"1", "true", "yes"}

# (connect, read): fail fast if the VPN is down, allow long completions
TIMEOUT = (10, 300)
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2

# Model ids are case-sensitive and the deployed list changes over time.
# Check with list_models() / test_scayle_model_access() before adding new ones.
# Reasoning models spend tokens on hidden reasoning, so they need a higher max_tokens.
MODEL_CONFIG = {
    "qwen3.6": {"model_id": "qwen3.6", "reasoning": False, "max_tokens": 400},
    "qwen3": {"model_id": "qwen3", "reasoning": False, "max_tokens": 400},
    "kimi-k2.6": {"model_id": "kimi-k2.6", "reasoning": False, "max_tokens": 400},
    "glm-5.3-flash": {"model_id": "glm-5.3-flash", "reasoning": False, "max_tokens": 400},
    "qwen3.8-flash-next": {"model_id": "qwen3.8-flash-next", "reasoning": True, "max_tokens": 2000},
    "deepseek-v4-flash-dspark": {"model_id": "deepseek-v4-flash-dspark", "reasoning": True, "max_tokens": 2000},
}
DEFAULT_LLM = "qwen3.6"

# Cached bearer token, reused across calls (it should not expire)
_TOKEN: str | None = None


# --------------------------------------------------------------------------- #
# Auth + HTTP helpers
# --------------------------------------------------------------------------- #
def get_token(force_refresh: bool = False) -> str:
    """Logs in via LDAP and returns a bearer token (cached after the first call)."""
    global _TOKEN
    if _TOKEN and not force_refresh:
        return _TOKEN

    if not SCAYLE_BASE_URL or not SCAYLE_USERNAME or not SCAYLE_PASSWORD:
        raise RuntimeError(
            "Missing SCAYLE_BASE_URL / SCAYLE_USERNAME / SCAYLE_PASSWORD in the environment."
        )

    r = requests.post(
        f"{SCAYLE_BASE_URL}/v1/auths/ldap",
        json={"user": SCAYLE_USERNAME, "password": SCAYLE_PASSWORD},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        verify=VERIFY_SSL,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _TOKEN = r.json()["token"]
    return _TOKEN


def _request(method: str, path: str, **kwargs) -> requests.Response:
    """
    Sends an authenticated request to SCAYLE.
    - Retries connection errors, timeouts and 5xx with exponential backoff.
    - On 401, logs in again and retries once. Other 4xx errors are not retried.
    """
    relogged = False
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        headers = {"Authorization": f"Bearer {get_token()}", **kwargs.pop("headers", {})}
        try:
            r = requests.request(
                method,
                f"{SCAYLE_BASE_URL}{path}",
                headers=headers,
                verify=VERIFY_SSL,
                timeout=TIMEOUT,
                **kwargs,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
        else:
            if r.status_code == 401 and not relogged:
                relogged = True
                get_token(force_refresh=True)
                continue
            if r.status_code < 500:
                return r
            last_error = RuntimeError(f"SCAYLE server error ({r.status_code}): {r.text}")

        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE_SECONDS ** (attempt + 1)
            print(f"SCAYLE request failed ({last_error}); retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"SCAYLE request failed after {MAX_RETRIES + 1} attempts: {last_error}")


def list_models() -> list[str]:
    """Returns the ids of the models currently deployed on SCAYLE."""
    r = _request("GET", "/models")
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


# Test SCAYLE access
def test_scayle_model_access():
    try:
        models = list_models()
        print(f"\nAvailable SCAYLE models at {SCAYLE_BASE_URL}:")
        for mid in models:
            print(f"  - {mid}")
    except Exception as e:
        print(f"\nFailed to query SCAYLE at {SCAYLE_BASE_URL} (are you on the VPN?): {e}")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def build_catalog_summary_prompt(
    title: str,
    headline: str,
    description: str,
    keywords,
    field_of_science,
) -> str:
    """Prompt for generating a public-facing catalog summary."""
    return f"""
    Based on this dataset's metadata, write a concise paragraph (80–150 words) that could serve as its summary in a scientific data catalog or registry.
    
    The paragraph should clearly state:
    - What the dataset contains
    - How it was created or structured (if known)
    - Its main value
    - Who it is intended for
    - Specific use cases or example analytical tasks it supports

    Dataset metadata:
    Title: {title}
    Subtitle: {headline}
    Description:
    {description}
    Keywords: {', '.join(keywords) if isinstance(keywords, list) else keywords}
    Field: {', '.join(field_of_science) if isinstance(field_of_science, list) else field_of_science}

    Use domain-specific terminology where appropriate.
    """


PROMPT_BUILDERS = {
    "catalog_summary_v1": build_catalog_summary_prompt,
}


def build_prompt(
    title: str,
    headline: str,
    description: str,
    keywords,
    field_of_science,
    prompt_version: str = "catalog_summary_v1",
) -> str:
    """Build a prompt from the selected prompt version."""
    prompt_builder = PROMPT_BUILDERS.get(prompt_version)
    if prompt_builder is None:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")

    return prompt_builder(
        title=title,
        headline=headline,
        description=description,
        keywords=keywords,
        field_of_science=field_of_science,
    )


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def call_scayle(prompt: str, llm: str = DEFAULT_LLM, max_tokens: int | None = None) -> str:
    """
    Calls an LLM on SCAYLE (OpenAI-style chat completions) and returns the generated text.
    """
    config = MODEL_CONFIG.get(llm)
    if not config:
        raise ValueError(f"Unsupported SCAYLE model alias: {llm}")

    model_id = config["model_id"]
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or config["max_tokens"],
        "temperature": 0.7,
    }

    print(f"Invoking SCAYLE model: {model_id} at {SCAYLE_BASE_URL}")

    r = _request("POST", "/chat/completions", json=body, headers={"Content-Type": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"SCAYLE chat failed ({r.status_code}): {r.text}")

    output = r.json()
    content = output["choices"][0]["message"].get("content") or ""

    # Reasoning models start with blank lines, kimi-k2.6 with a space
    return content.strip()


def enrich_data_profile(
    profile: DatasetProfile,
    llm: str = DEFAULT_LLM,
    prompt_version: str = "catalog_summary_v1",
) -> DatasetProfile:
    """
    Enriches a single dataset profile using an LLM on SCAYLE.
    """
    title = profile.title or ""
    headline = profile.headline or ""
    description = profile.description or ""
    keywords = profile.keywords or ""
    field_of_science = profile.field_of_science or ""

    prompt = build_prompt(
        title=title,
        headline=headline,
        description=description,
        keywords=keywords,
        field_of_science=field_of_science,
        prompt_version=prompt_version,
    )

    catalog_summary = call_scayle(prompt, llm=llm)

    return DatasetProfile(
        id=profile.id,
        title=profile.title,
        headline=profile.headline,
        description=profile.description,
        keywords=profile.keywords,
        field_of_science=profile.field_of_science,
        catalog_summary=catalog_summary,
    )


def enrich_batch(
    profiles: list[DatasetProfile],
    llm: str = DEFAULT_LLM,
    prompt_version: str = "catalog_summary_v1",
) -> list[DatasetProfile]:
    """
    Enriches a list of dataset profiles using an LLM on SCAYLE.
    """
    enriched_profiles = []

    for profile in profiles:
        title = profile.title or "<untitled>"
        print(f"Enriching: {title}")
        enriched_profiles.append(
            enrich_data_profile(profile, llm=llm, prompt_version=prompt_version)
        )

    return enriched_profiles


# --------------------------------------------------------------------------- #
# OCR (PDFs and images)
# --------------------------------------------------------------------------- #
# SCAYLE only takes images (OpenAI-style image_url), so PDFs are rendered page by page.
# Only qwen3.6 and kimi-k2.6 accept images; the other models return a 400.
# Requires: pip install pymupdf
import base64
from pathlib import Path

OCR_LLM = "qwen3.6"
OCR_PROMPT = (
    "Transcribe all text in this image exactly as written. Preserve reading order, "
    "line breaks, headings and tables (as Markdown tables). Output only the transcription."
)
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".gif": "image/gif"}


def ocr_image(image_bytes: bytes, mime: str = "image/png", llm: str = OCR_LLM,
              prompt: str = OCR_PROMPT, max_tokens: int = 4000) -> str:
    """Sends one image to a vision model on SCAYLE and returns the transcribed text."""
    if llm not in MODEL_CONFIG:
        raise ValueError(f"Unsupported SCAYLE model alias: {llm}")

    b64 = base64.b64encode(image_bytes).decode()
    body = {
        "model": MODEL_CONFIG[llm]["model_id"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,  # deterministic transcription
    }
    r = _request("POST", "/chat/completions", json=body, headers={"Content-Type": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"SCAYLE OCR failed ({r.status_code}): {r.text}")
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def ocr_file(path: str, llm: str = OCR_LLM, dpi: int = 200,
             prefer_text_layer: bool = True, min_text_chars: int = 50,
             prompt: str = OCR_PROMPT) -> str:
    """
    Extracts text from a PDF or image file.
    - PDFs: for each page, uses the embedded text layer if it has at least `min_text_chars`
      characters (fast, exact, free); otherwise renders the page at `dpi` and OCRs it.
    - Images (png/jpg/webp/gif): OCR directly.
    Pages are joined with a '--- Page N ---' separator. Pass `prompt` to customise the OCR
    instruction (e.g. LaTeX for equations). For maths documents use prefer_text_layer=False,
    since a PDF text layer garbles equations.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in IMAGE_MIME:
        return ocr_image(p.read_bytes(), mime=IMAGE_MIME[suffix], llm=llm, prompt=prompt)

    if suffix == ".pdf":
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(p) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if prefer_text_layer and len(text) >= min_text_chars:
                    print(f"{p.name} p.{i}/{len(doc)}: using text layer")
                else:
                    print(f"{p.name} p.{i}/{len(doc)}: OCR with {llm}")
                    png = page.get_pixmap(dpi=dpi).tobytes("png")
                    text = ocr_image(png, mime="image/png", llm=llm, prompt=prompt)
                pages.append(f"--- Page {i} ---\n{text}")
        return "\n\n".join(pages)

    raise ValueError(
        f"Unsupported file type '{suffix}'. Convert office files first, "
        "e.g. `soffice --headless --convert-to pdf file.docx`."
    )


__all__ = [
    "MODEL_CONFIG",
    "ocr_file",
    "ocr_image",
    "PROMPT_BUILDERS",
    "build_catalog_summary_prompt",
    "build_prompt",
    "call_scayle",
    "enrich_batch",
    "enrich_data_profile",
    "get_token",
    "list_models",
    "test_scayle_model_access",
]

if __name__ == "__main__":
    test_scayle_model_access()
    # print(call_scayle("Say hello in one short sentence.", llm="qwen3.6"))