import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
import re

def load_repo_env():
    """Walk up directories until .env is found and load it."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(".env file not found in any parent directory.")
        current_dir = parent_dir

# Load .env once when the module is imported
load_repo_env()

def sanitize_filename(filename: str) -> str:
    # 1. Remove the file extension (e.g., '175.pdf' -> '175')
    name_without_ext = Path(filename).stem
    
    # 2. Replace everything not alphanumeric, space, hyphen, parens, or brackets with a hyphen
    # This regex matches anything NOT in the allowed set
    sanitized = re.sub(r'[^a-zA-Z0-9 \-\(\)\[\]]', '-', name_without_ext)
    
    # 3. Replace multiple consecutive spaces with a single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()

# Provided Configuration
MODEL_CONFIG = {
    "mistral": {
        "model_id": "mistral.mistral-7b-instruct-v0:2",
        "region": "eu-west-1"
    },
    "claude": {
        "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "region": "eu-central-1"
    }
}

# AWS credentials from env
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

def get_bedrock_client(model_name: str):
    """
    Creates a Boto3 Session and returns the Bedrock Runtime client 
    using the provided credentials and region.
    """
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise ValueError(f"No config found for '{model_name}'")

    # Use a session to keep credentials localized to this client
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=config["region"]
    )
    
    return session.client("bedrock-runtime"), config["model_id"]

def ocr_pdf_with_claude(pdf_path: Path):
    client, model_id = get_bedrock_client("claude")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Apply sanitization here
    clean_name = sanitize_filename(pdf_path.name)

    response = client.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [
                {
                    "document": {
                        "name": clean_name, # Use sanitized name
                        "format": "pdf",    # Keep format as 'pdf' here
                        "source": {"bytes": pdf_bytes}
                    }
                },
                {"text": "Extract all text from this math document. Use LaTeX for equations. Follow the native language of the document. Do not add any commentary or explanations, just return the raw extracted text."}
            ]
        }],
        inferenceConfig={"temperature": 0.0}
    )
    
    return response['output']['message']['content'][0]['text']

def run_batch_process(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}...")
        try:
            text = ocr_pdf_with_claude(pdf_file)
            
            # Save output
            with open(output_path / f"{pdf_file.stem}_ocr.txt", "w", encoding="utf-8") as f:
                f.write(text)
            print("Done.")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    # Ensure these paths exist
    run_batch_process("/home/mpkostas/Documents/data/cross-dataset-assets/mathe/materials", "/home/mpkostas/Documents/data/cross-dataset-assets/mathe/ocr_output")