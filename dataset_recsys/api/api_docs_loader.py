import json
from functools import lru_cache
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/valid_examples.json")
DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/error_examples.json")
AP_DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_valid_examples.json")
AP_DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_error_examples.json")
AP_REQUEST_EXAMPLE_PATH = Path("dataset_recsys/api/api_docs/ap_request_example.json")


@lru_cache
def load_json_file(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"File '{path}' does not exist.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load file from {path}: {e}")
        return {}
