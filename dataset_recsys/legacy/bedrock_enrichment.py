import boto3
import json
from pathlib import Path
# from dotenv import load_dotenv
import os
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile

# load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
MODEL_CONFIG = {
    "claude-sonnet-4-6": {
        "region": "eu-central-1",
        "model_id": "eu.anthropic.claude-sonnet-4-6",
    }
}

def get_bedrock_client(llm: str):
    config = MODEL_CONFIG.get(llm)
    if not config:
        raise ValueError(f"Unsupported Bedrock model alias: {llm}")

    return boto3.client(
        service_name="bedrock-runtime",
        region_name=config["region"],
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

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

def call_bedrock(prompt: str, llm: str = "claude-sonnet-4-6") -> str:
    """
    Calls an LLM on Amazon Bedrock and returns the generated text.
    Currently supports Claude via the configured model alias.
    """
    config = MODEL_CONFIG.get(llm)
    if not config:
        raise ValueError(f"Unsupported Bedrock model alias: {llm}")

    client = get_bedrock_client(llm)
    model_id = config["model_id"]
    region = config["region"]

    if llm.startswith("claude"):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        }
    else:
        raise ValueError(f"No request body defined yet for llm='{llm}'")

    print(f"Invoking Bedrock model/profile: {model_id} in region {region}")

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    output = json.loads(response["body"].read())

    if llm.startswith("claude"):
        return output["content"][0]["text"].strip()

    raise ValueError(f"No response parser defined yet for llm='{llm}'")

def enrich_data_profile(
    profile: DatasetProfile,
    llm: str = "claude-sonnet-4-6",
    prompt_version: str = "catalog_summary_v1",
) -> DatasetProfile:
    """
    Enriches a single dataset profile using Claude on Bedrock.
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

    catalog_summary = call_bedrock(prompt, llm=llm)

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
    llm: str = "claude-sonnet-4-6",
    prompt_version: str = "catalog_summary_v1",
) -> list[DatasetProfile]:
    """
    Enriches a list of dataset profiles using Claude on Bedrock.
    """
    enriched_profiles = []

    for profile in profiles:
        title = profile.title or "<untitled>"
        print(f"Enriching: {title}")
        enriched_profiles.append(
            enrich_data_profile(profile, llm=llm, prompt_version=prompt_version)
        )

    return enriched_profiles


__all__ = [
    "MODEL_CONFIG",
    "PROMPT_BUILDERS",
    "build_catalog_summary_prompt",
    "build_prompt",
    "call_bedrock",
    "enrich_batch",
    "enrich_data_profile",
    "get_bedrock_client",
    "test_bedrock_model_access",
]

if __name__ == "__main__":
    test_bedrock_model_access("eu-central-1")
    # print(call_bedrock("Say hello in one short sentence.", llm="claude-sonnet-4-6"))
