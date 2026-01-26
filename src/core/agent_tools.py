"""
Agent Tools - Tool definitions for the MetaFeature AI Agent

These tools wrap existing functionality and expose them to the AI agent
for dynamic decision-making during evaluation prompt generation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agent_framework import tool

from .metrics_registry import (
    METRICS_REGISTRY,
    get_metric_definition,
    get_metrics_for_category,
    suggest_additional_metrics,
    get_default_metrics_for_category,
    get_metric
)
from .prompt_templates import (
    SUPPORTED_LOCALES,
    get_language,
    get_region,
    get_cultural_context,
    get_privacy_framework,
    get_tone_guidance,
    build_evaluation_prompt
)
from .database import FeatureStore, PromptTemplateStore
from .code_metrics import get_code_metrics_for_category, generate_code_metrics_sample


# =============================================================================
# Tool 1: Metrics Lookup
# =============================================================================

@tool
def lookup_metrics(category: str) -> Dict[str, Any]:
    """
    Look up available metrics for a feature category.
    
    Use this tool when you need to find what metrics are available
    for evaluating a specific type of AI feature.
    
    Args:
        category: The feature category (e.g., 'summarization', 'auto_reply', 
                  'translation', 'image_generation', 'assistant')
    
    Returns:
        Dictionary with:
        - default_metrics: List of recommended metrics for this category
        - all_applicable: List of all metrics that apply to this category
        - metric_details: Detailed info about each metric
    """
    category = category.lower().strip()
    
    # Get default metrics
    default_metrics = get_default_metrics_for_category(category)
    
    # Get all applicable metrics
    all_applicable = get_metrics_for_category(category)
    
    # Get details for each
    metric_details = {}
    for m_id in all_applicable:
        metric = get_metric(m_id)
        if metric:
            metric_details[m_id] = {
                "name": metric.name,
                "definition": metric.get_definition("en"),
                "weight": metric.weight,
                "rai_tags": metric.rai_tags,
                "is_primary": metric.is_primary,
                "applies_to": metric.applies_to
            }
    
    return {
        "category": category,
        "default_metrics": default_metrics,
        "all_applicable": all_applicable,
        "metric_details": metric_details
    }


@tool
def suggest_metrics(category: str, current_metrics: List[str]) -> Dict[str, Any]:
    """
    Suggest additional metrics that might be useful for a feature.
    
    Use this tool when you want to improve coverage by finding metrics
    the user might have missed.
    
    Args:
        category: The feature category
        current_metrics: List of metrics already selected
    
    Returns:
        Dictionary with suggested metrics and reasons
    """
    suggestions = suggest_additional_metrics(category, current_metrics)
    
    # Get details for suggestions
    suggestion_details = []
    for m_id in suggestions:
        metric = get_metric(m_id)
        if metric:
            suggestion_details.append({
                "id": m_id,
                "name": metric.name,
                "reason": f"Recommended for {category} features",
                "definition": metric.get_definition("en")
            })
    
    return {
        "current_metrics": current_metrics,
        "suggested_additions": suggestions,
        "suggestion_details": suggestion_details
    }


@tool
def search_metric_by_name(query: str) -> Dict[str, Any]:
    """
    Search for metrics by name or keyword.
    
    Use this tool when the user mentions a specific quality attribute
    and you need to find if we have a matching metric.
    
    Args:
        query: Search term (e.g., 'accuracy', 'safety', 'hallucination')
    
    Returns:
        Dictionary with matching metrics
    """
    query_lower = query.lower()
    matches = []
    
    for m_id, metric in METRICS_REGISTRY.items():
        # Match on id, name, or rai_tags
        if (query_lower in m_id.lower() or 
            query_lower in metric.name.lower() or
            any(query_lower in tag.lower() for tag in metric.rai_tags)):
            matches.append({
                "id": m_id,
                "name": metric.name,
                "definition": metric.get_definition("en"),
                "rai_tags": metric.rai_tags,
                "applies_to": metric.applies_to
            })
    
    return {
        "query": query,
        "matches": matches,
        "match_count": len(matches)
    }


# =============================================================================
# Tool 2: Locale Information
# =============================================================================

@tool
def get_locale_info(locale: str) -> Dict[str, Any]:
    """
    Get detailed information about a locale including cultural context.
    
    Use this tool when you need to understand the cultural and regulatory
    requirements for a specific target audience.
    
    Args:
        locale: BCP 47 locale code (e.g., 'en-US', 'de-DE', 'ja-JP')
    
    Returns:
        Dictionary with locale details including:
        - language: ISO language code
        - region: ISO region code
        - cultural_context: Formality, directness settings
        - privacy_framework: Applicable privacy regulation
        - tone_guidance: Culture-specific communication guidance
    """
    if locale not in SUPPORTED_LOCALES:
        # Try to find closest match
        lang = locale.split("-")[0] if "-" in locale else locale
        closest = [l for l in SUPPORTED_LOCALES if l.startswith(lang)]
        if closest:
            locale = closest[0]
        else:
            locale = "en-US"  # Default
    
    return {
        "locale": locale,
        "language": get_language(locale),
        "region": get_region(locale),
        "cultural_context": get_cultural_context(locale),
        "privacy_framework": get_privacy_framework(locale),
        "tone_guidance": get_tone_guidance(locale),
        "supported": locale in SUPPORTED_LOCALES
    }


@tool
def list_supported_locales() -> Dict[str, Any]:
    """
    List all supported locales with their privacy frameworks.
    
    Use this tool when you need to help the user choose an appropriate
    locale or when handling multi-locale features.
    
    Returns:
        Dictionary with all supported locales grouped by region
    """
    by_region = {}
    for locale in SUPPORTED_LOCALES:
        region = get_region(locale)
        if region not in by_region:
            by_region[region] = []
        by_region[region].append({
            "locale": locale,
            "language": get_language(locale),
            "privacy_framework": get_privacy_framework(locale)
        })
    
    return {
        "total_locales": len(SUPPORTED_LOCALES),
        "by_region": by_region,
        "all_locales": list(SUPPORTED_LOCALES.keys())
    }


# =============================================================================
# Tool 3: Database Operations
# =============================================================================

@tool
def search_similar_features(
    query: str,
    category: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search the database for similar features that have been created before.
    
    Use this tool to find existing features that might serve as templates
    or to avoid duplicating work.
    
    Args:
        query: Search term (feature name or description keywords)
        category: Optional category filter
        limit: Maximum results to return
    
    Returns:
        Dictionary with matching features
    """
    try:
        store = FeatureStore()
        all_features = store.list_features()
        
        # Simple text matching (could be enhanced with embeddings)
        query_lower = query.lower()
        matches = []
        
        for feat in all_features:
            score = 0
            if query_lower in feat.get("name", "").lower():
                score += 2
            if query_lower in feat.get("description", "").lower():
                score += 1
            if category and feat.get("category", "").lower() == category.lower():
                score += 1
            
            if score > 0:
                matches.append({
                    "feature": feat,
                    "relevance_score": score
                })
        
        # Sort by relevance
        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "query": query,
            "category_filter": category,
            "matches": matches[:limit],
            "total_found": len(matches)
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "matches": []
        }


@tool
def get_feature_by_id(feature_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific feature from the database by ID.
    
    Use this tool when you need the full details of a known feature.
    
    Args:
        feature_id: The unique identifier of the feature
    
    Returns:
        The feature data or error message
    """
    try:
        store = FeatureStore()
        feature = store.get_feature(feature_id)
        if feature:
            return {"found": True, "feature": feature}
        return {"found": False, "error": f"Feature '{feature_id}' not found"}
    except Exception as e:
        return {"found": False, "error": str(e)}


# =============================================================================
# Tool 4: Prompt Building
# =============================================================================

@tool
def build_prompt(
    feature_name: str,
    category: str,
    locale: str,
    metrics: List[str]
) -> Dict[str, Any]:
    """
    Build an evaluation prompt for a feature.
    
    Use this tool to generate the actual evaluation prompt once you have
    gathered all the necessary information about the feature.
    
    Args:
        feature_name: Name of the feature being evaluated
        category: Feature category (summarization, auto_reply, etc.)
        locale: Target locale (e.g., 'en-US')
        metrics: List of metric IDs to evaluate
    
    Returns:
        Dictionary with the generated prompt and metadata
    """
    # Build metric definitions
    lang = get_language(locale)
    metric_defs = {}
    for m_id in metrics:
        metric = get_metric(m_id)
        if metric:
            metric_defs[m_id] = {
                "definition": metric.get_definition(lang),
                "weight": metric.weight,
                "rai_tags": metric.rai_tags
            }
        else:
            metric_defs[m_id] = {
                "definition": "(custom metric)",
                "weight": 1.0,
                "rai_tags": []
            }
    
    # Generate prompt
    prompt = build_evaluation_prompt(
        feature_name=feature_name,
        category=category,
        locale=locale,
        metrics_used=metrics,
        metric_defs=metric_defs
    )
    
    return {
        "feature_name": feature_name,
        "category": category,
        "locale": locale,
        "metrics_used": metrics,
        "evaluation_prompt": prompt,
        "prompt_length": len(prompt),
        "privacy_framework": get_privacy_framework(locale)
    }


# =============================================================================
# Tool 5: RAI Validation
# =============================================================================

@tool
def validate_rai_compliance(
    metrics: List[str],
    privacy_sensitive: bool = False,
    safety_critical: bool = False,
    locale: str = "en-US"
) -> Dict[str, Any]:
    """
    Validate that a metric selection meets RAI requirements.
    
    Use this tool to check if the selected metrics adequately cover
    Responsible AI concerns for the feature.
    
    Args:
        metrics: Currently selected metrics
        privacy_sensitive: Whether feature handles PII
        safety_critical: Whether feature is safety-critical
        locale: Target locale for regional compliance
    
    Returns:
        Dictionary with validation results and recommendations
    """
    issues = []
    recommendations = []
    
    # Check for safety metric
    if "safety" not in metrics:
        issues.append("Missing 'safety' metric - required for all GenAI features")
        recommendations.append("Add 'safety' metric to check for harmful content")
    
    # Check for privacy metric if privacy-sensitive
    if privacy_sensitive and "privacy" not in metrics:
        issues.append("Missing 'privacy' metric for privacy-sensitive feature")
        recommendations.append("Add 'privacy' metric to check for PII leakage")
    
    # Check for groundedness if safety-critical
    if safety_critical and "groundedness" not in metrics:
        issues.append("Missing 'groundedness' metric for safety-critical feature")
        recommendations.append("Add 'groundedness' metric to prevent hallucinations")
    
    # Check for fairness
    has_fairness = any(
        "fairness" in m or "bias" in m 
        for m in metrics
    )
    if not has_fairness:
        recommendations.append("Consider adding fairness/bias checks")
    
    # Regional compliance
    framework = get_privacy_framework(locale)
    compliance_notes = []
    if framework == "GDPR":
        compliance_notes.append("GDPR: Ensure data minimization and right to explanation")
    elif framework == "CCPA":
        compliance_notes.append("CCPA: Ensure opt-out mechanisms are respected")
    elif framework == "PIPL":
        compliance_notes.append("PIPL: Ensure data localization and consent requirements")
    
    return {
        "metrics_checked": metrics,
        "is_compliant": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "privacy_framework": framework,
        "compliance_notes": compliance_notes
    }


# =============================================================================
# Tool 6: Code Metrics
# =============================================================================

@tool
def get_code_metrics(category: str) -> Dict[str, Any]:
    """
    Get programmatic (code-based) metrics for a category.
    
    Use this tool to find deterministic metrics that can be computed
    programmatically alongside LLM-based evaluation.
    
    Args:
        category: Feature category
    
    Returns:
        Dictionary with available code metrics and sample code
    """
    metrics = get_code_metrics_for_category(category)
    sample_code = generate_code_metrics_sample(category, metrics)
    
    return {
        "category": category,
        "available_code_metrics": metrics,
        "sample_code": sample_code,
        "note": "Code metrics provide deterministic scores to complement LLM evaluation"
    }


# =============================================================================
# Tool 7: Feature Analysis
# =============================================================================

@tool
def analyze_feature_description(description: str) -> Dict[str, Any]:
    """
    Analyze a feature description to extract key attributes.
    
    Use this tool to understand what category, metrics, and constraints
    might be appropriate for a feature based on its description.
    
    Args:
        description: Natural language description of the feature
    
    Returns:
        Dictionary with suggested category, metrics, and flags
    """
    desc_lower = description.lower()
    
    # Detect category
    category_hints = {
        "summarization": ["summarize", "summary", "condense", "brief", "tldr"],
        "auto_reply": ["reply", "respond", "email", "message", "answer"],
        "translation": ["translate", "translation", "convert language", "multilingual"],
        "image_generation": ["generate image", "create image", "dall-e", "image generation"],
        "image_understanding": ["describe image", "image caption", "ocr", "image understanding"],
        "assistant": ["assistant", "chatbot", "help", "qa", "question"],
        "classification": ["classify", "categorize", "detect", "identify"]
    }
    
    suggested_category = "other"
    for cat, hints in category_hints.items():
        if any(hint in desc_lower for hint in hints):
            suggested_category = cat
            break
    
    # Detect privacy sensitivity
    privacy_hints = ["pii", "personal", "private", "sensitive", "medical", "health", 
                     "financial", "ssn", "email address", "phone number"]
    is_privacy_sensitive = any(hint in desc_lower for hint in privacy_hints)
    
    # Detect safety criticality
    safety_hints = ["medical", "health", "legal", "financial", "safety-critical",
                    "life-threatening", "diagnosis", "prescription"]
    is_safety_critical = any(hint in desc_lower for hint in safety_hints)
    
    # Detect locale hints
    locale_hints = {
        "german": "de-DE", "germany": "de-DE",
        "japan": "ja-JP", "japanese": "ja-JP",
        "china": "zh-CN", "chinese": "zh-CN",
        "spanish": "es-ES", "spain": "es-ES",
        "mexico": "es-MX", "mexican": "es-MX",
        "french": "fr-FR", "france": "fr-FR",
        "brazil": "pt-BR", "brazilian": "pt-BR"
    }
    
    detected_locales = []
    for hint, locale in locale_hints.items():
        if hint in desc_lower:
            detected_locales.append(locale)
    
    return {
        "description": description[:200] + "..." if len(description) > 200 else description,
        "suggested_category": suggested_category,
        "is_privacy_sensitive": is_privacy_sensitive,
        "is_safety_critical": is_safety_critical,
        "detected_locales": detected_locales or ["en-US"],
        "recommended_metrics": get_default_metrics_for_category(suggested_category),
        "analysis_confidence": "high" if suggested_category != "other" else "low"
    }


# =============================================================================
# Export all tools
# =============================================================================

ALL_TOOLS = [
    lookup_metrics,
    suggest_metrics,
    search_metric_by_name,
    get_locale_info,
    list_supported_locales,
    search_similar_features,
    get_feature_by_id,
    build_prompt,
    validate_rai_compliance,
    get_code_metrics,
    analyze_feature_description
]
