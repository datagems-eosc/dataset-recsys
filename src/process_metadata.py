import json
from pathlib import Path
from collections import defaultdict

def extract_format_and_fields(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Get encoding format(s) from distribution
    encoding_formats = set()
    distributions = data.get("distribution", [])
    if isinstance(distributions, list):
        for dist in distributions:
            fmt = dist.get("encodingFormat")
            if fmt:
                encoding_formats.add(fmt.strip().lower())

    # 2. Extract field names from recordSet
    record_sets = data.get("recordSet", [])
    field_names = []

    for record in record_sets:
        fields = record.get("field", [])
        for field in fields:
            name = field.get("name")
            if name:
                field_names.append(name)

    return {
        "file": json_path.name,
        "encoding_formats": list(encoding_formats),
        "fields": field_names
    }

def extract_formats_and_files(folder_path: str) -> dict:
    """
    Scans all JSON metadata files in a folder and returns a mapping of
    encoding formats to list of dataset JSON filenames.
    """
    folder = Path(folder_path)
    format_to_files = defaultdict(list)

    for file in folder.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            encoding_formats = set()
            distributions = data.get("distribution", [])
            if isinstance(distributions, list):
                for dist in distributions:
                    fmt = dist.get("encodingFormat")
                    if fmt:
                        encoding_formats.add(fmt.strip().lower())

            if not encoding_formats and "encodingFormat" in data:
                encoding_formats.add(data["encodingFormat"].strip().lower())

            if not encoding_formats:
                encoding_formats.add("unknown")

            for fmt in encoding_formats:
                format_to_files[fmt].append(file.name)

        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

    print("Dataset Encoding Formats Summary")
    for fmt, files in sorted(format_to_files.items()):
        print(f"\nFormat: {fmt} ({len(files)} datasets)")
        for fname in sorted(files):
            print(f"  - {fname}")

    return dict(format_to_files)

def check_datasets_without_recordset(folder_path: str) -> list:
    """
    Scans all JSON metadata files in a folder and returns a list of files
    that do not contain a 'recordSet' key or have an empty 'recordSet'.
    """
    folder = Path(folder_path)
    missing_recordset = []

    for file in folder.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            record_set = data.get("recordSet")
            if not record_set:
                missing_recordset.append(file.name)

        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

    print("\nDatasets without recordSet:")
    for fname in missing_recordset:
        print(f"  - {fname}")

    return missing_recordset

if __name__ == "__main__":
    
    # # Examples:
    # result = extract_format_and_fields(Path("data/gems_datasets_metadata/mathe_material.json"))
    # print(json.dumps(result, indent=2))

    # formats_map = extract_formats_and_files(
    #     "data/gems_datasets_metadata/"
    # )

    missing = check_datasets_without_recordset("data/gems_datasets_metadata/")