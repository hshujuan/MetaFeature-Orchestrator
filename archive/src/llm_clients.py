import os
from azure.identity import AzureCliCredential
from openai import AzureOpenAI

def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

def call_responses(model: str, input_text: str) -> str:
    client = get_azure_openai_client()
    resp = client.responses.create(
        model=model,
        input=input_text,
    )
    return resp.output_text
