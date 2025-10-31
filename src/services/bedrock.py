import boto3
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# AWS setup: 
MODEL_CONFIG = {
    "mistral": {
        "model_id": "mistral.mistral-7b-instruct-v0:2",
        "region": "eu-west-1"
    },
    "claude": {
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "region": "eu-central-1"
    }
}
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Bedrock client
def get_bedrock_client_for_model(model_name: str):
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise ValueError(f"No Bedrock config found for model '{model_name}'")

    return boto3.client(
        service_name="bedrock-runtime",
        region_name=config["region"],
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    ), config["model_id"]

# Test Bedrock access
def test_bedrock_model_access(region: str):
    try:
        mgmt_client = boto3.client(
            service_name="bedrock",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        response = mgmt_client.list_foundation_models()
        models = [m['modelId'] for m in response.get('modelSummaries', [])]
        print(f"\nAvailable Bedrock models in region {region}:")
        for mid in models:
            print(f"  - {mid}")
    except Exception as e:
        print(f"\nFailed to query Bedrock in region {region}: {e}")

# Prompt template
def build_prompt(description, headline, keywords, field_of_science, encoding_formats, structure_summary):
    """
    Constructs the prompt string to send to the LLM.
    """
    encoding_line = f"Encoding format(s): {', '.join(encoding_formats)}\n" if encoding_formats else ""
    structure_line = f"Structure Summary:\n{structure_summary}\n" if structure_summary else ""

    return f"""You are given the following metadata about a dataset:

    Description:
    {description}

    Headline:
    {headline}

    {encoding_line}{structure_line}
    Keywords: {', '.join(keywords)}
    Scientific domain: {', '.join(field_of_science)}

    Write a short, well-structured paragraph that could serve as a public-facing abstract for this dataset in a scientific data catalog or registry. 
    The paragraph should be 100–200 words, in fluent academic English. It should explain:
    - What the dataset contains
    - Its structure or origin, if applicable (e.g., data format, how it was collected, or which organization produced it)
    - Why the dataset is valuable (its potential benefits or advantages)
    - Who the dataset is intended for (target users or communities)
    - Relevant use cases or application scenarios where the dataset could be effectively used

    Use the structure summary only if it helps convey how the data is organized or what types of information are included.
    Do not assume the structure summary lists all records and their fields; it is based on a sample (i.e., up to 3 records and 5 fields each).
    """

def call_bedrock(prompt: str, model_name: str) -> str:
    """
    Unified Bedrock call supporting Claude 3.5 Sonnet and Mistral.
    """
    client, model_id = get_bedrock_client_for_model(model_name)

    # Claude 3.5
    if model_id == "anthropic.claude-3-5-sonnet-20240620-v1:0":
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9
        }

    # Mistral
    else:
        body = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9
        }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    output = json.loads(response["body"].read())

    # Parse model output
    if model_id == "anthropic.claude-3-5-sonnet-20240620-v1:0":
        return output["content"][0]["text"].strip()
    else:
        return output["outputs"][0]["text"].strip()

def enrich_dataset_from_json(json_path: Path, llm="mistral"):
    """
    Reads a JSON dataset metadata file, constructs a prompt, 
    calls Bedrock to enrich it, and returns the result.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    description = data.get("description", "")
    headline = data.get("headline", "")
    keywords = data.get("keywords", [])
    field_of_science = data.get("fieldOfScience", [])
    record_sets = data.get("recordSet", [])

    # Extract encoding formats
    encoding_formats = {
        dist["encodingFormat"].strip().lower()
        for dist in data.get("distribution", [])
        if "encodingFormat" in dist
    }

    # Take up to 3 record entries, and within each, list up to 5 field names for summarization
    # Build structure summary from recordSet
    structure_snippets = [
        f"'{record.get('name', '')}' with fields: {', '.join(f.get('name', '') for f in record.get('field', [])[:5] if f.get('name'))}"
        for record in record_sets[:3]
        if record.get("name") and record.get("field")
    ]

    structure_summary = (
        "The dataset contains the following records:\n" +
        "\n".join(f"- {s}" for s in structure_snippets)
    ) if structure_snippets else ""

    prompt = build_prompt(description, headline, keywords, field_of_science, list(encoding_formats), structure_summary)

    text_enriched = call_bedrock(prompt, model_name=llm)

    return {
        "file": json_path.name,
        "enriched_description": text_enriched
    }
    # return prompt

def batch_enrich(json_folder: str, output_file: str, llm="mistral"):
    input_paths = list(Path(json_folder).glob("*.json"))
    results = []

    for path in input_paths:
        print(f"Processing {path.name}...")
        try:
            result = enrich_dataset_from_json(path, llm=llm)
            results.append(result)
        except Exception as e:
            print(f"Failed for {path.name}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    test_bedrock_model_access("eu-central-1")
    
    # batch_enrich(
    #     "development/datagems_dataset_recs/datasets_metadata/",
    #     "development/datagems_dataset_recs/datasets_metadata/enriched_datasets_mistral.jsonl",
    #     llm="mistral"
    # )

    # batch_enrich(
    #     "development/datagems_dataset_recs/datasets_metadata/",
    #     "development/datagems_dataset_recs/datasets_metadata/enriched_datasets_claude.jsonl",
    #     llm="claude"
    # )

    # test_path = Path("development/datagems_dataset_recs/datasets_metadata/encyc_net.json")
    # prompt = enrich_dataset_from_json(test_path)
    # print("\nGENERATED PROMPT\n")
    # print(prompt)

    # result = enrich_dataset_from_json(test_path, llm="mistral")
    # print("\nENRICHED DESCRIPTION\n")
    # print(result["enriched_description"])
