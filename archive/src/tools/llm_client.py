"""
OpenAI client wrapper and configuration.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class OpenAIClient:
    """Singleton wrapper for OpenAI client"""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = OpenAI(
                base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        return cls._instance
    
    @property
    def client(self) -> OpenAI:
        return self._client
    
    @staticmethod
    def get_deployment_name() -> str:
        return os.getenv("DEPLOYMENT_NAME", "gpt-4o")


def get_openai_client() -> OpenAI:
    """Get the OpenAI client instance"""
    return OpenAIClient().client


def get_deployment_name() -> str:
    """Get the model deployment name"""
    return OpenAIClient.get_deployment_name()
