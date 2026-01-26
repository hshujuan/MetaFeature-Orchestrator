"""
MetaFeature AI Agent - Intelligent evaluation prompt generation using Microsoft Agent Framework

This module provides a true AI agent that can:
- Understand natural language feature descriptions
- Dynamically select appropriate metrics
- Handle complex multi-locale requirements
- Validate RAI compliance
- Build optimized evaluation prompts

This replaces the deterministic FeaturePromptWriterAgent with an intelligent,
tool-equipped agent that can reason about feature requirements.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from agent_framework import Agent, AgentConfig
from agent_framework.models import AzureOpenAIConfig, OpenAIConfig

from .agent_tools import ALL_TOOLS
from .schemas import FeatureSpec, PromptOutput
from .prompt_templates import get_language, get_privacy_framework

logger = logging.getLogger(__name__)


# =============================================================================
# Agent System Prompt
# =============================================================================

METAFEATURE_AGENT_SYSTEM_PROMPT = """You are MetaFeature Agent, an expert AI evaluation specialist.

Your role is to help users create high-quality evaluation prompts for GenAI features.
You follow these core principles:

## Core Principles
1. **Metric-First**: Always identify evaluation metrics before building prompts
2. **Grounded**: Only use metrics and criteria that exist in our registry
3. **RAI by Design**: Ensure safety, privacy, and fairness are always considered
4. **Locale-Aware**: Respect cultural and regulatory differences across regions

## Your Capabilities (Tools)
You have access to these tools to help users:

- **lookup_metrics**: Find available metrics for a feature category
- **suggest_metrics**: Get recommendations for additional metrics
- **search_metric_by_name**: Find metrics by keyword
- **get_locale_info**: Get cultural and regulatory info for a locale
- **list_supported_locales**: See all supported locales
- **search_similar_features**: Find existing similar features
- **build_prompt**: Generate the evaluation prompt
- **validate_rai_compliance**: Check if metrics meet RAI requirements
- **get_code_metrics**: Get programmatic metrics (ROUGE, BLEU, etc.)
- **analyze_feature_description**: Analyze natural language descriptions

## Workflow
When a user describes a feature:
1. Use `analyze_feature_description` to understand the feature
2. Use `lookup_metrics` to find appropriate metrics
3. Use `validate_rai_compliance` to ensure RAI coverage
4. Use `build_prompt` to generate the evaluation prompt
5. Optionally use `get_code_metrics` for programmatic evaluation

## Important Guidelines
- Always check RAI compliance before finalizing
- For privacy-sensitive features, ensure privacy metrics are included
- For safety-critical features, ensure groundedness is checked
- Respect locale-specific regulations (GDPR, CCPA, PIPL, etc.)
- If unsure about something, ask the user for clarification

Be helpful, thorough, and always prioritize responsible AI practices.
"""


# =============================================================================
# Agent Configuration
# =============================================================================

def _get_model_config():
    """Get model configuration from environment"""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    
    if azure_endpoint and azure_key:
        return AzureOpenAIConfig(
            endpoint=azure_endpoint,
            api_key=azure_key,
            deployment=deployment,
            api_version="2024-02-15-preview"
        )
    elif openai_key:
        return OpenAIConfig(
            api_key=openai_key,
            model=deployment
        )
    else:
        raise ValueError(
            "No API credentials found. Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY "
            "or OPENAI_API_KEY in environment variables."
        )


# =============================================================================
# MetaFeature AI Agent
# =============================================================================

@dataclass
class AgentResponse:
    """Response from the MetaFeature Agent"""
    success: bool
    message: str
    evaluation_prompt: Optional[str] = None
    feature_name: Optional[str] = None
    category: Optional[str] = None
    locale: Optional[str] = None
    metrics_used: List[str] = field(default_factory=list)
    rai_checks: List[str] = field(default_factory=list)
    code_metrics_sample: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


class MetaFeatureAgent:
    """
    AI-powered agent for generating evaluation prompts.
    
    This agent uses Microsoft Agent Framework to provide intelligent,
    tool-equipped prompt generation that can handle complex requirements.
    
    Example:
        agent = MetaFeatureAgent()
        
        # Natural language request
        response = agent.chat(
            "I need an evaluation prompt for a medical document summarizer "
            "that will be used by doctors in Germany. It needs to be very "
            "careful about accuracy and patient privacy."
        )
        
        print(response.evaluation_prompt)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the MetaFeature Agent.
        
        Args:
            verbose: If True, log detailed agent reasoning
        """
        self.verbose = verbose
        self._agent = None
        self._thread = None
        
    def _ensure_agent(self):
        """Lazily initialize the agent"""
        if self._agent is None:
            config = AgentConfig(
                name="MetaFeatureAgent",
                instructions=METAFEATURE_AGENT_SYSTEM_PROMPT,
                model=_get_model_config(),
                tools=ALL_TOOLS,
                temperature=0.0  # Consistent outputs
            )
            self._agent = Agent(config)
            logger.info("MetaFeature Agent initialized")
    
    async def chat_async(self, message: str) -> AgentResponse:
        """
        Send a message to the agent and get a response (async version).
        
        Args:
            message: User's natural language request
            
        Returns:
            AgentResponse with evaluation prompt and metadata
        """
        self._ensure_agent()
        
        try:
            # Create or continue thread
            if self._thread is None:
                self._thread = await self._agent.create_thread()
            
            # Send message
            response = await self._agent.chat(
                thread=self._thread,
                message=message
            )
            
            # Parse response to extract structured data
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return AgentResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    def chat(self, message: str) -> AgentResponse:
        """
        Send a message to the agent and get a response (sync version).
        
        Args:
            message: User's natural language request
            
        Returns:
            AgentResponse with evaluation prompt and metadata
        """
        import asyncio
        return asyncio.run(self.chat_async(message))
    
    def _parse_response(self, response) -> AgentResponse:
        """Parse agent response to extract structured data"""
        # The response object contains the agent's final message
        # and any tool call results
        
        text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract evaluation prompt from response
        evaluation_prompt = None
        if "```" in text:
            # Extract code block
            parts = text.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are code blocks
                    if "Evaluation Prompt" in part or "# " in part:
                        evaluation_prompt = part.strip()
                        break
        
        return AgentResponse(
            success=True,
            message=text,
            evaluation_prompt=evaluation_prompt
        )
    
    def reset_conversation(self):
        """Reset the conversation thread"""
        self._thread = None
        logger.info("Conversation reset")
    
    def generate_from_spec(self, spec: FeatureSpec, locale: str = "en-US") -> AgentResponse:
        """
        Generate an evaluation prompt from a FeatureSpec.
        
        This provides a more structured interface similar to the legacy agent.
        
        Args:
            spec: Feature specification
            locale: Target locale
            
        Returns:
            AgentResponse with evaluation prompt
        """
        # Build a natural language request from the spec
        request = f"""Please generate an evaluation prompt for this feature:

**Feature Name:** {spec.name}
**Description:** {spec.description}
**Category:** {spec.category}
**Target Locale:** {locale}
**Input Format:** {spec.input_format}
**Output Format:** {spec.output_format}
**Suggested Metrics:** {', '.join(spec.success_metrics) if spec.success_metrics else 'auto-detect'}
**Privacy Sensitive:** {spec.privacy_sensitive}
**Safety Critical:** {spec.safety_critical}

Please:
1. Validate the metrics are appropriate
2. Check RAI compliance
3. Generate the evaluation prompt
4. Include code-based metrics if available
"""
        return self.chat(request)


# =============================================================================
# Quick Generation Function (Legacy Compatibility)
# =============================================================================

def generate_with_agent(
    feature_name: str,
    feature_description: str,
    category: str = "other",
    locale: str = "en-US",
    metrics: Optional[List[str]] = None,
    privacy_sensitive: bool = False,
    safety_critical: bool = False
) -> PromptOutput:
    """
    Generate an evaluation prompt using the AI agent.
    
    This function provides backward compatibility with the legacy agent API
    while using the new AI-powered agent under the hood.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of what the feature does
        category: Feature category
        locale: Target locale (BCP 47)
        metrics: Optional list of metrics to use
        privacy_sensitive: Whether feature handles PII
        safety_critical: Whether feature is safety-critical
        
    Returns:
        PromptOutput compatible with legacy code
    """
    spec = FeatureSpec(
        name=feature_name,
        description=feature_description,
        category=category,
        locales_supported=[locale],
        success_metrics=metrics or [],
        privacy_sensitive=privacy_sensitive,
        safety_critical=safety_critical
    )
    
    agent = MetaFeatureAgent()
    response = agent.generate_from_spec(spec, locale)
    
    if response.success and response.evaluation_prompt:
        return PromptOutput(
            feature_name=feature_name,
            category=category,
            locale=locale,
            metrics_used=response.metrics_used or metrics or [],
            evaluation_prompt=response.evaluation_prompt,
            rai_checks_applied=response.rai_checks
        )
    else:
        # Fall back to legacy agent on failure
        from .agent import FeaturePromptWriterAgent
        legacy_agent = FeaturePromptWriterAgent()
        return legacy_agent.generate(spec, locale=locale)


# =============================================================================
# Interactive Chat Interface
# =============================================================================

def interactive_chat():
    """
    Start an interactive chat session with the MetaFeature Agent.
    
    Useful for exploring capabilities and testing.
    """
    print("=" * 60)
    print("MetaFeature AI Agent - Interactive Mode")
    print("=" * 60)
    print("I can help you create evaluation prompts for AI features.")
    print("Type 'quit' to exit, 'reset' to start a new conversation.")
    print("=" * 60)
    
    agent = MetaFeatureAgent(verbose=True)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                print("Conversation reset. How can I help you?")
                continue
            
            print("\nAgent: Thinking...")
            response = agent.chat(user_input)
            
            print(f"\nAgent: {response.message}")
            
            if response.evaluation_prompt:
                print("\n" + "=" * 40)
                print("GENERATED EVALUATION PROMPT:")
                print("=" * 40)
                print(response.evaluation_prompt[:500] + "..." if len(response.evaluation_prompt) > 500 else response.evaluation_prompt)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    interactive_chat()
