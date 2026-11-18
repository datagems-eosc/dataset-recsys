"""The service calls LLMs on SCAYLE only.

Bedrock code is kept under ``dataset_recsys/legacy/`` for reference; nothing
the API or the workflows run may import it or the AWS SDK.
"""

import ast
from pathlib import Path

import dataset_recsys
from dataset_recsys.utils import scayle
from dataset_recsys.workflows import full_batch_rebuild, incremental_update

PACKAGE_DIR = Path(dataset_recsys.__file__).parent
FORBIDDEN_IMPORTS = ("boto3", "botocore", "dataset_recsys.legacy", "dataset_recsys.utils.bedrock")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_active_code_does_not_import_bedrock_or_the_aws_sdk():
    offenders = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR)
        if relative.parts[0] == "legacy":
            continue
        for module in _imported_modules(path):
            if any(module == name or module.startswith(f"{name}.") for name in FORBIDDEN_IMPORTS):
                offenders.append(f"{relative}: {module}")

    assert offenders == []


def test_dataset_workflows_enrich_with_scayle():
    assert full_batch_rebuild.enrich_batch is scayle.enrich_batch
    assert incremental_update.enrich_batch is scayle.enrich_batch
