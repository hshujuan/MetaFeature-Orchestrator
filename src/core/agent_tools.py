"""
Agent Tools - Plain function implementations for MetaFeature operations

These functions are used by the AI Agent in ai_agent.py via @ai_function decorator.
They can also be called directly for programmatic use.
"""
from typing import List, Dict, Any, Optional

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


def lookup_metrics(category: str) -> Dict[str, Any]:
    """
    Look up available metrics for a feature category.
    
    Args:
        category: The feature category (e.g., 'summarization', 'auto_reply', 
                  'translation', 'image_generation', 'personal_assistant')
    
    Returns:
        Dictionary with default_metrics, all_applicable, and metric_details
    """
    category = category.lower().strip()
    
    default_metrics = get_default_metrics_for_category(category)
    all_applicable = get_metrics_for_category(category)
    
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


def suggest_metrics(
    category: str, 
    current_metrics: Optional[List[str]] = None,
    privacy_sensitive: bool = False,
    safety_critical: bool = False
) -> Dict[str, Any]:
    """
    Suggest additional metrics that might be useful for a feature.
    
    Args:
        category: The feature category
        current_metrics: List of metrics already selected
        privacy_sensitive: Whether feature handles PII
        safety_critical: Whether feature is safety-critical
    
    Returns:
        Dictionary with suggested metrics and reasons
    """
    current = current_metrics or []
    suggestions = suggest_additional_metrics(category, current)
    
    # Add RAI-specific suggestions
    if privacy_sensitive and "privacy" not in current:
        suggestions.append("privacy")
    if safety_critical and "groundedness" not in current:
        suggestions.append("groundedness")
    if "safety" not in current:
        suggestions.append("safety")
    
    suggestion_details = []
    for m_id in set(suggestions):
        metric = get_metric(m_id)
        if metric:
            suggestion_details.append({
                "id": m_id,
                "name": metric.name,
                "reason": f"Recommended for {category} features",
                "definition": metric.get_definition("en")
            })
    
    return {
        "current_metrics": current,
        "suggested_additions": list(set(suggestions)),
        "suggestion_details": suggestion_details
    }


def search_metric_by_name(query: str) -> Dict[str, Any]:
    """
    Search for metrics by name or keyword.
    
    Args:
        query: Search term (e.g., 'accuracy', 'safety', 'hallucination')
    
    Returns:
        Dictionary with matching metrics
    """
    query_lower = query.lower()
    matches = []
    
    for m_id, metric in METRICS_REGISTRY.items():
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


def get_locale_info(locale: str) -> Dict[str, Any]:
    """
    Get detailed information about a locale including cultural context.
    
    Args:
        locale: BCP 47 locale code (e.g., 'en-US', 'de-DE', 'ja-JP')
    
    Returns:
        Dictionary with locale details
    """
    if locale not in SUPPORTED_LOCALES:
        lang = locale.split("-")[0] if "-" in locale else locale
        closest = [l for l in SUPPORTED_LOCALES if l.startswith(lang)]
        if closest:
            locale = closest[0]
        else:
            locale = "en-US"
    
    return {
        "locale": locale,
        "language": get_language(locale),
        "region": get_region(locale),
        "cultural_context": get_cultural_context(locale),
        "privacy_framework": get_privacy_framework(locale),
        "tone_guidance": get_tone_guidance(locale),
        "supported": locale in SUPPORTED_LOCALES
    }


def list_supported_locales() -> Dict[str, Any]:
    """
    List all supported locales with their privacy frameworks.
    
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


def search_similar_features(
    query: str,
    category: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search the database for similar features.
    
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


def get_feature_by_id(feature_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific feature from the database by ID.
    
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
    rai_checks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build an evaluation prompt for a feature.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of the feature
        category: Feature category
        metrics: List of metric IDs to evaluate
        locale: Target locale
        input_format: Input format
        output_format: Output format
        typical_input: Example input
        expected_output: Example expected output
        rai_checks: RAI checks to apply
    
    Returns:
        Dictionary with the generated prompt and metadata
    """
    lang = get_language(locale)
    metric_definitions = {}
    for m_id in metrics:
        metric = get_metric(m_id)
        if metric:
            metric_definitions[m_id] = metric.get_definition(lang)
        else:
            metric_definitions[m_id] = "(custom metric)"
    
    prompt = build_evaluation_prompt(
        feature_name=feature_name,
        feature_description=feature_description,
        category=category,
        metrics=metrics,
        metric_definitions=metric_definitions,
        locale=locale,
        typical_input=typical_input,
        expected_output=expected_output,
        additional_context=""
    )
    
    return {
        "evaluation_prompt": prompt,
        "feature_name": feature_name,
        "category": category,
        "locale": locale,
        "metrics_used": metrics,
        "privacy_framework": get_privacy_framework(locale)
    }


def validate_rai_compliance(
    metrics: List[str],
    privacy_sensitive: bool = False,
    safety_critical: bool = False,
    locale: str = "en-US"
) -> Dict[str, Any]:
    """
    Validate that a metric selection meets RAI requirements.
    
    Args:
        metrics: Currently selected metrics
        privacy_sensitive: Whether feature handles PII
        safety_critical: Whether feature is safety-critical
        locale: Target locale for regional compliance
    
    Returns:
        Dictionary with validation results and recommendations
    """
    issues = []
    checks_applied = []
    recommendations = []
    
    if "safety" not in metrics:
        issues.append("Missing 'safety' metric - required for all GenAI features")
        recommendations.append("Add 'safety' metric")
    else:
        checks_applied.append("safety_check")
    
    if privacy_sensitive:
        if "privacy" not in metrics:
            issues.append("Missing 'privacy' metric for privacy-sensitive feature")
            recommendations.append("Add 'privacy' metric")
        else:
            checks_applied.append("privacy_check")
    
    if safety_critical:
        if "groundedness" not in metrics:
            issues.append("Missing 'groundedness' metric for safety-critical feature")
            recommendations.append("Add 'groundedness' metric")
        else:
            checks_applied.append("groundedness_check")
    
    framework = get_privacy_framework(locale)
    
    return {
        "metrics_checked": metrics,
        "compliant": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "checks_applied": checks_applied,
        "privacy_framework": framework
    }


def get_code_metrics(category: str) -> Dict[str, Any]:
    """
    Get programmatic (code-based) metrics for a category.
    
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
        "code_sample": sample_code
    }


def analyze_feature_description(description: str) -> Dict[str, Any]:
    """
    Analyze a feature description to extract key attributes.
    
    Args:
        description: Natural language description of the feature
    
    Returns:
        Dictionary with suggested category, metrics, and flags
    """
    desc_lower = description.lower()
    
    category_hints = {
        "summarization": ["summarize", "summary", "condense", "brief", "tldr"],
        "auto_reply": ["reply", "respond", "email", "message", "answer"],
        "translation": ["translate", "translation", "convert language", "multilingual"],
        "image_generation": ["generate image", "create image", "dall-e", "image"],
        "image_understanding": ["describe image", "caption", "ocr", "visual"],
        "personal_assistant": ["assistant", "personal", "memory", "reasoning", "decision", "suggest", "recommend", "pattern", "habit"],
        "classification": ["classify", "categorize", "detect", "identify", "sentiment"]
    }
    
    suggested_category = "other"
    max_matches = 0
    for cat, hints in category_hints.items():
        matches = sum(1 for hint in hints if hint in desc_lower)
        if matches > max_matches:
            max_matches = matches
            suggested_category = cat
    
    privacy_hints = ["pii", "personal", "private", "sensitive", "medical", "health", 
                     "financial", "ssn", "email address", "phone number", "user data"]
    is_privacy_sensitive = any(hint in desc_lower for hint in privacy_hints)
    
    safety_hints = ["medical", "health", "legal", "financial", "safety",
                    "critical", "diagnosis", "prescription"]
    is_safety_critical = any(hint in desc_lower for hint in safety_hints)
    
    multimodal_hints = ["image", "photo", "video", "audio", "voice", "multimodal", 
                        "calendar", "health", "location", "sensor"]
    is_multimodal = sum(1 for hint in multimodal_hints if hint in desc_lower) >= 2
    
    return {
        "detected_category": suggested_category,
        "is_privacy_sensitive": is_privacy_sensitive,
        "is_safety_critical": is_safety_critical,
        "is_multimodal": is_multimodal,
        "recommended_metrics": get_default_metrics_for_category(suggested_category),
        "confidence": "high" if max_matches >= 2 else "medium" if max_matches >= 1 else "low"
    }


# Export all functions
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
