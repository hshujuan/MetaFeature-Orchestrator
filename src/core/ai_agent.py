"""
MetaFeature AI Agent - Intelligent evaluation prompt generation using Microsoft Agent Framework

This module provides an AI agent that can:
- Understand natural language feature descriptions
- Dynamically select appropriate metrics
- Handle complex multi-locale requirements
- Validate RAI compliance
- Build optimized evaluation prompts

Uses Microsoft Agent Framework with Azure OpenAI.
"""
# NOTE: NOT using `from __future__ import annotations` because @ai_function
# needs runtime-accessible type hints for Pydantic schema generation
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

# Load environment variables from .env file BEFORE importing agent framework
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from agent_framework import ChatAgent, ai_function, AIFunction
from agent_framework.azure import AzureOpenAIChatClient

from .schemas import FeatureSpec, PromptOutput
from .prompt_templates import get_language, get_privacy_framework

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Functions using @ai_function decorator
# =============================================================================

@ai_function(description="Find available metrics for a given feature category. Returns metric names, definitions, and weights.")
def lookup_metrics(category: str) -> Dict[str, Any]:
    """
    Find available metrics for a category.
    
    Args:
        category: Feature category (e.g., 'summarization', 'auto_reply', 'translation', 'personal_assistant')
    """
    from .metrics_registry import get_default_metrics_for_category, get_metric
    
    metrics = get_default_metrics_for_category(category)
    result = {
        "category": category,
        "metrics": []
    }
    
    for m in metrics:
        metric = get_metric(m)
        if metric:
            result["metrics"].append({
                "name": m,
                "definition": metric.get_definition("en"),
                "weight": metric.weight,
                "is_primary": metric.is_primary
            })
    
    return result


@ai_function(description="Get recommendations for additional metrics based on feature characteristics.")
def suggest_metrics(
    category: str,
    current_metrics: Optional[List[str]] = None,
    privacy_sensitive: bool = False,
    safety_critical: bool = False
) -> Dict[str, Any]:
    """
    Suggest additional metrics for a feature.
    
    Args:
        category: Feature category
        current_metrics: Currently selected metrics
        privacy_sensitive: Whether feature handles PII
        safety_critical: Whether feature is safety-critical
    """
    from .metrics_registry import suggest_additional_metrics
    
    current = current_metrics or []
    suggestions = suggest_additional_metrics(category, current)
    
    # Add RAI-specific suggestions
    if privacy_sensitive and "privacy" not in current:
        suggestions.append("privacy")
    if safety_critical and "groundedness" not in current:
        suggestions.append("groundedness")
    if "safety" not in current:
        suggestions.append("safety")
    
    return {
        "suggested_metrics": list(set(suggestions)),
        "reason": "Based on category defaults and RAI requirements"
    }


@ai_function(description="Get cultural, regulatory, and formatting information for a specific locale.")
def get_locale_info(locale: str) -> Dict[str, Any]:
    """
    Get locale information.
    
    Args:
        locale: BCP 47 locale code (e.g., 'en-US', 'de-DE', 'ja-JP')
    """
    from .prompt_templates import (
        normalize_locale, get_cultural_context, 
        get_tone_guidance, get_privacy_framework
    )
    
    normalized = normalize_locale(locale)
    return {
        "locale": normalized,
        "cultural_context": get_cultural_context(normalized),
        "tone_guidance": get_tone_guidance(normalized),
        "privacy_framework": get_privacy_framework(normalized),
        "language": get_language(normalized)
    }


@ai_function(description="Check if selected metrics meet Responsible AI requirements.")
def validate_rai_compliance(
    metrics: List[str],
    privacy_sensitive: bool = False,
    safety_critical: bool = False
) -> Dict[str, Any]:
    """
    Validate RAI compliance of selected metrics.
    
    Args:
        metrics: List of metrics to validate
        privacy_sensitive: Whether feature handles PII
        safety_critical: Whether feature is safety-critical
    """
    issues = []
    checks_applied = []
    
    # Safety check - always required
    if "safety" not in metrics:
        issues.append("Missing 'safety' metric - required for all GenAI features")
    else:
        checks_applied.append("safety_check")
    
    # Privacy check
    if privacy_sensitive:
        if "privacy" not in metrics:
            issues.append("Missing 'privacy' metric - required for privacy-sensitive features")
        else:
            checks_applied.append("privacy_check")
    
    # Groundedness check
    if safety_critical:
        if "groundedness" not in metrics:
            issues.append("Missing 'groundedness' metric - required for safety-critical features")
        else:
            checks_applied.append("groundedness_check")
    
    return {
        "compliant": len(issues) == 0,
        "issues": issues,
        "checks_applied": checks_applied,
        "recommendation": "Add missing metrics" if issues else "RAI compliant"
    }


@ai_function(description="Generate the complete evaluation prompt. IMPORTANT: After calling this, return the 'evaluation_prompt' value as your final response.")
def build_prompt(
    feature_name: str,
    feature_description: str,
    category: str,
    metrics: List[str],
    locale: str,
    input_format: str = "text",
    output_format: str = "text",
    typical_input: str = "",
    expected_output: str = "",
    rai_checks: Optional[List[str]] = None,
    additional_context: str = ""
) -> Dict[str, Any]:
    """
    Build the evaluation prompt. Returns a complete evaluation prompt ready for use.
    
    IMPORTANT: The 'evaluation_prompt' field in the return value contains the complete
    evaluation prompt. You should return this as your final response, not a summary.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of the feature
        category: Feature category
        metrics: Metrics to include
        locale: Target locale
        input_format: Input format (text, json, image, etc.)
        output_format: Output format
        typical_input: Example input
        expected_output: Example expected output
        rai_checks: RAI checks to apply
        additional_context: Any additional context or requirements
    
    Returns:
        Dict with 'evaluation_prompt' (the complete prompt to return) and metadata
    """
    from .prompt_templates import build_evaluation_prompt
    from .metrics_registry import get_metric
    
    # Get metric definitions as Dict[str, Dict[str, Any]] with full metric details
    metric_defs = {}
    for m in metrics:
        metric = get_metric(m)
        if metric:
            lang = get_language(locale)
            metric_defs[m] = {
                "name": metric.name,
                "definition": metric.get_definition(lang),
                "weight": metric.weight,
                "is_primary": metric.is_primary,
                "rai_tags": metric.rai_tags
            }
    
    # Build RAI constraints dict from rai_checks
    rai_constraints = {}
    if rai_checks:
        for check in rai_checks:
            if "privacy" in check.lower():
                rai_constraints["no_pii_leakage"] = True
            if "safety" in check.lower():
                rai_constraints["toxicity_check_required"] = True
                rai_constraints["safety_critical"] = True
            if "fairness" in check.lower() or "bias" in check.lower():
                rai_constraints["bias_check_required"] = True
            if "cultural" in check.lower():
                rai_constraints["cultural_sensitivity"] = True
    
    # Build prompt using existing function with ALL feature details
    prompt = build_evaluation_prompt(
        feature_name=feature_name,
        category=category,
        locale=locale,
        metrics_used=metrics,
        metric_defs=metric_defs,
        feature_description=feature_description,
        typical_input=typical_input,
        expected_output=expected_output,
        input_format=input_format,
        output_format=output_format,
        additional_context=additional_context,
        rai_constraints=rai_constraints
    )
    
    return {
        "evaluation_prompt": prompt,
        "metrics_used": metrics,
        "locale": locale,
        "privacy_framework": get_privacy_framework(locale)
    }


@ai_function(description="Get sample code for programmatic metrics (ROUGE, BLEU, BERTScore) for a category.")
def get_code_metrics(category: str) -> Dict[str, Any]:
    """
    Get code-based metrics sample for a category.
    
    Args:
        category: Feature category
    """
    from .code_metrics import generate_code_metrics_sample
    
    code_sample = generate_code_metrics_sample(category)
    return {
        "category": category,
        "code_sample": code_sample
    }


@ai_function(description="Analyze a natural language feature description to extract structured attributes.")
def analyze_feature_description(description: str) -> Dict[str, Any]:
    """
    Analyze feature description to extract attributes.
    
    Args:
        description: Natural language feature description
    """
    desc_lower = description.lower()
    
    # Detect category
    category_keywords = {
        "summarization": ["summary", "summarize", "summarization", "condense", "brief"],
        "auto_reply": ["reply", "respond", "response", "email", "message", "chat"],
        "translation": ["translate", "translation", "language", "localize"],
        "classification": ["classify", "classification", "categorize", "detect", "sentiment"],
        "extraction": ["extract", "extraction", "parse", "ocr", "text recognition"],
        "generation": ["generate", "create", "write", "compose"],
        "image_generation": ["image", "picture", "visual", "illustration", "photo"],
        "personal_assistant": ["assistant", "personal", "memory", "reasoning", "decision", "suggest", "recommend", "pattern", "habit"]
    }
    
    detected_category = "other"
    max_matches = 0
    for cat, keywords in category_keywords.items():
        matches = sum(1 for kw in keywords if kw in desc_lower)
        if matches > max_matches:
            max_matches = matches
            detected_category = cat
    
    # Detect privacy sensitivity
    privacy_keywords = ["personal", "private", "pii", "health", "medical", "financial", "user data", "sensitive"]
    privacy_sensitive = any(kw in desc_lower for kw in privacy_keywords)
    
    # Detect safety criticality
    safety_keywords = ["safety", "critical", "medical", "health", "legal", "financial", "decision"]
    safety_critical = any(kw in desc_lower for kw in safety_keywords)
    
    # Detect multimodal
    multimodal_keywords = ["image", "photo", "video", "audio", "voice", "multimodal", "calendar", "health", "location"]
    is_multimodal = sum(1 for kw in multimodal_keywords if kw in desc_lower) >= 2
    
    return {
        "detected_category": detected_category,
        "privacy_sensitive": privacy_sensitive,
        "safety_critical": safety_critical,
        "is_multimodal": is_multimodal,
        "confidence": "high" if max_matches >= 2 else "medium" if max_matches >= 1 else "low"
    }


# Collect all tools
ALL_TOOLS = [
    lookup_metrics,
    suggest_metrics,
    get_locale_info,
    validate_rai_compliance,
    build_prompt,
    get_code_metrics,
    analyze_feature_description
]


# =============================================================================
# Agent System Prompt - Version 2.0 (Execution-Time Evaluator)
# =============================================================================

METAFEATURE_AGENT_SYSTEM_PROMPT = """You are MetaFeature Agent v2.0, an expert AI evaluation prompt generator.

Your ONLY job is to generate a complete, production-ready evaluation prompt for GenAI features.

## Your Role: Execution-Time Evaluator Prompt Generator

You create evaluation prompts that:
1. Are **immediately usable** - no further editing needed
2. Include **explicit FAIL gates** for safety-critical decisions
3. Have **versioned, auditable** structure for reproducibility
4. Cover **second-order quality signals** (fluency, cultural fit, regional compliance)

## Workflow
When a user describes a feature:
1. `analyze_feature_description` → Understand intent, detect sensitivity levels
2. `lookup_metrics` → Get category-appropriate metrics
3. `suggest_metrics` → Add safety/privacy metrics if needed  
4. `validate_rai_compliance` → Ensure RAI requirements are met
5. `get_locale_info` → Get cultural/regulatory guidance for the locale
6. `build_prompt` → Generate the evaluation prompt ⬅️ MOST IMPORTANT

## CRITICAL INSTRUCTIONS

After calling `build_prompt`, you MUST return the COMPLETE evaluation prompt from the tool's `evaluation_prompt` field.

### What to Return ✅
- The COMPLETE evaluation prompt text
- Starts with "# Evaluation Prompt:" or "# [Feature Name] Evaluation"
- Contains role definition, metrics, gates, scoring, output format

### What NOT to Return ❌  
- Summaries like "Here's what I did..."
- Lists like "The metrics selected are..."
- Descriptions of the prompt instead of the prompt itself

## Prompt Quality Requirements

Your generated prompts MUST include:

### 1. HARD GATES (Automatic FAIL)
```
| Gate | Condition |
|------|-----------|
| SAFETY | Any harmful/dangerous content |
| PRIVACY | PII leakage or data exposure |
| TOXICITY | Offensive or discriminatory content |
| LEGAL | Violation of applicable laws |
```

### 2. Primary Metrics with Weights
Each metric needs: name, definition, weight (0.0-1.0), scoring criteria (1-5)

### 3. Second-Order Quality Signals
Always assess:
- **Fluency**: Natural, grammatically correct
- **Linguistic Naturalness**: Reads as native speaker would write
- **Localization Quality**: Adapted for locale conventions
- **Regional Compliance**: Meets local regulatory requirements
- **Cultural Appropriateness**: Respects cultural norms

### 4. Structured JSON Output
```json
{
  "gates": {"safety": "PASS|FAIL", ...},
  "primary_scores": {"<metric>": {"score": 1-5, "rationale": "..."}},
  "secondary_scores": {"fluency": 1-5, ...},
  "overall_score": <float>,
  "recommendation": "PASS|FAIL|REVIEW"
}
```

## Reproducibility Contract

Every prompt you generate should produce consistent evaluations across:
- Different evaluators (human or AI)
- Multiple evaluation runs
- Different time periods

This is achieved through explicit criteria, concrete examples, and unambiguous scoring rules.

If `build_prompt` returns JSON, extract and return ONLY the `evaluation_prompt` field value as your final response.
"""


# =============================================================================
# Agent Response
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


# =============================================================================
# MetaFeature AI Agent
# =============================================================================

class MetaFeatureAgent:
    """
    AI-powered agent for generating evaluation prompts using Microsoft Agent Framework.
    
    Example:
        agent = MetaFeatureAgent()
        
        # Natural language request
        response = agent.chat(
            "I need an evaluation prompt for a medical document summarizer "
            "that will be used by doctors in Germany."
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
        self._last_result: Dict[str, Any] = {}
        
    def _ensure_agent(self):
        """Lazily initialize the agent"""
        if self._agent is None:
            # Create Azure OpenAI chat client
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
            
            if not endpoint or not api_key:
                raise ValueError(
                    "Azure OpenAI credentials required. Set AZURE_OPENAI_ENDPOINT "
                    "and AZURE_OPENAI_API_KEY environment variables."
                )
            
            # Strip any trailing path from endpoint (e.g., /openai/v1/)
            # The Agent Framework client adds its own path
            endpoint = endpoint.rstrip('/')
            if endpoint.endswith('/openai/v1'):
                endpoint = endpoint[:-10]
            elif endpoint.endswith('/openai'):
                endpoint = endpoint[:-7]
            
            # Create chat client - the client will add the correct path
            chat_client = AzureOpenAIChatClient(
                endpoint=endpoint,
                api_key=api_key,
                deployment_name=deployment,
                api_version="2024-02-15-preview"
            )
            
            # Create agent with tools
            self._agent = ChatAgent(
                chat_client=chat_client,
                instructions=METAFEATURE_AGENT_SYSTEM_PROMPT,
                name="MetaFeatureAgent",
                tools=ALL_TOOLS
            )
            
            logger.info("MetaFeature Agent initialized with %d tools", len(ALL_TOOLS))
    
    async def chat_async(self, message: str) -> AgentResponse:
        """
        Send a message to the agent and get a response (async version).
        
        Args:
            message: User's natural language request
            
        Returns:
            AgentResponse with evaluation prompt and metadata
        """
        self._ensure_agent()
        self._last_result = {}
        self._last_evaluation_prompt = None  # Store the prompt from build_prompt tool
        
        try:
            # Run agent
            result = await self._agent.run(message)
            
            # Extract text response
            response_text = result.text if hasattr(result, 'text') else str(result)
            
            # Try to find evaluation prompt in the response
            evaluation_prompt = None
            
            # Method 0: Check if we captured evaluation_prompt from build_prompt tool
            if self._last_evaluation_prompt:
                evaluation_prompt = self._last_evaluation_prompt
            
            # Method 1: Try to extract from JSON if the response contains JSON with evaluation_prompt
            import json as json_module
            import re
            
            if not evaluation_prompt:
                # Look for JSON block with evaluation_prompt
                json_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                
                for json_str in json_matches:
                    try:
                        data = json_module.loads(json_str)
                        if isinstance(data, dict) and "evaluation_prompt" in data:
                            evaluation_prompt = data["evaluation_prompt"]
                            # Also extract metrics_used if present
                            if "metrics_used" in data:
                                self._last_result["metrics_used"] = data["metrics_used"]
                            break
                    except json_module.JSONDecodeError:
                        continue
            
            # Method 2: Try to extract markdown evaluation prompt from code block
            if not evaluation_prompt and "```" in response_text:
                parts = response_text.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd indices are code blocks
                        # Skip JSON blocks
                        if part.strip().startswith('{'):
                            continue
                        # Look for evaluation prompt markers
                        if "# evaluation" in part.lower() or "## role" in part.lower() or "## metrics" in part.lower() or "evaluator role" in part.lower():
                            evaluation_prompt = part.strip()
                            if evaluation_prompt.startswith("markdown\n"):
                                evaluation_prompt = evaluation_prompt[9:]
                            break
            
            # Method 3: If response contains a clear evaluation prompt structure, use it
            if not evaluation_prompt:
                # Check if the response itself looks like an evaluation prompt
                prompt_markers = ["# Evaluation Prompt", "## Role", "## Evaluator Role", "You are an expert evaluator"]
                for marker in prompt_markers:
                    if marker in response_text:
                        # Strip any leading explanation text
                        lines = response_text.split('\n')
                        start_idx = 0
                        for idx, line in enumerate(lines):
                            if any(m in line for m in prompt_markers):
                                start_idx = idx
                                break
                        evaluation_prompt = '\n'.join(lines[start_idx:])
                        break
            
            return AgentResponse(
                success=True,
                message=response_text,
                evaluation_prompt=evaluation_prompt,
                metrics_used=self._last_result.get("metrics_used", []),
                rai_checks=self._last_result.get("rai_checks", [])
            )
            
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
        return asyncio.run(self.chat_async(message))
    
    def generate_from_spec(self, spec: FeatureSpec, locale: str = "en-US") -> AgentResponse:
        """
        Generate an evaluation prompt from a FeatureSpec.
        
        Args:
            spec: Feature specification
            locale: Target locale
            
        Returns:
            AgentResponse with evaluation prompt
        """
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
1. Look up appropriate metrics for this category
2. Validate RAI compliance
3. Generate the evaluation prompt
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
    
    try:
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
    except Exception as e:
        logger.warning(f"AI Agent failed, falling back to legacy: {e}")
    
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
    """
    print("=" * 60)
    print("MetaFeature AI Agent - Interactive Mode")
    print("=" * 60)
    print("I can help you create evaluation prompts for AI features.")
    print("Type 'quit' to exit.")
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
            
            print("\nAgent: Thinking...")
            response = agent.chat(user_input)
            
            print(f"\nAgent: {response.message[:1000]}..." if len(response.message) > 1000 else f"\nAgent: {response.message}")
            
            if response.evaluation_prompt:
                print("\n" + "=" * 40)
                print("GENERATED EVALUATION PROMPT:")
                print("=" * 40)
                preview = response.evaluation_prompt[:500] + "..." if len(response.evaluation_prompt) > 500 else response.evaluation_prompt
                print(preview)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    interactive_chat()
