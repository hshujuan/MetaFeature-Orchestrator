"""
LLM Client - Azure OpenAI / OpenAI client wrapper
Provides unified interface for LLM interactions with proper configuration.
"""
from __future__ import annotations
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from openai import OpenAI


class LLMClient:
    """
    Singleton wrapper for OpenAI/Azure OpenAI client.
    Supports both direct Azure endpoint and standard OpenAI API.
    """
    
    _instance: Optional['LLMClient'] = None
    _client: Optional[OpenAI] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_client()
        return cls._instance
    
    @classmethod
    def _initialize_client(cls):
        """Initialize the OpenAI client based on environment configuration"""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if endpoint:
            # Azure OpenAI with base_url
            cls._client = OpenAI(
                base_url=endpoint,
                api_key=api_key
            )
        else:
            # Standard OpenAI
            cls._client = OpenAI(api_key=api_key)
    
    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client instance"""
        return self._client
    
    @staticmethod
    def get_deployment_name() -> str:
        """Get the model deployment name"""
        return os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        **kwargs
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model/deployment name (defaults to env config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for the API
            
        Returns:
            The response content as a string
        """
        model = model or self.get_deployment_name()
        
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_evaluation_prompt(
        self,
        system_prompt: str,
        user_request: str,
        temperature: float = 0.0
    ) -> str:
        """
        Generate an evaluation prompt using the LLM.
        
        Args:
            system_prompt: System instructions for the LLM
            user_request: The user's request with feature details
            temperature: Sampling temperature
            
        Returns:
            Generated evaluation prompt
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=8000
        )


# Convenience functions for backward compatibility
def get_llm_client() -> LLMClient:
    """Get the LLM client singleton"""
    return LLMClient()


def get_openai_client() -> OpenAI:
    """Get the raw OpenAI client"""
    return LLMClient().client


def get_deployment_name() -> str:
    """Get the model deployment name"""
    return LLMClient.get_deployment_name()


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000
) -> str:
    """Convenience function for chat completion"""
    return LLMClient().chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
