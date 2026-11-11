"""The recommendation management CLI must start both as a module and by path."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "dataset_recsys" / "storage" / "manage_recommendations.py"


# Run by path, the script's folder comes first on sys.path, where the sibling
# qdrant_client.py used to hide the installed qdrant_client package.
@pytest.mark.parametrize(
    "command",
    [
        [str(SCRIPT), "--help"],
        ["-m", "dataset_recsys.storage.manage_recommendations", "--help"],
    ],
    ids=["by-path", "as-module"],
)
def test_cli_starts_and_prints_its_help(command, tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}

    result = subprocess.run(
        [sys.executable, *command],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT if command[0] == "-m" else tmp_path,
        env=environment,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "list-entities" in result.stdout
