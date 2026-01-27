"""
Agent Tools - Utility functions for MetaFeature operations

NOTE: Primary AI Agent tools are defined in ai_agent.py with @ai_function decorators.
This module contains additional utility functions for programmatic use that are
NOT part of the Microsoft Agent Framework tool set.

For AI Agent tools, see ai_agent.py:
- lookup_metrics, suggest_metrics, recommend_metrics
- get_locale_info, validate_rai_compliance
- build_prompt, get_code_metrics, analyze_feature_description
"""
from typing import Dict, Any, Optional

from .metrics_registry import METRICS_REGISTRY
from .prompt_templates import (
    SUPPORTED_LOCALES,
    get_language,
    get_region,
    get_privacy_framework
)
from .database import FeatureStore


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


# Export utility functions (not AI Agent tools - those are in ai_agent.py)
ALL_TOOLS = [
    search_metric_by_name,
    list_supported_locales,
    search_similar_features,
    get_feature_by_id,
]
