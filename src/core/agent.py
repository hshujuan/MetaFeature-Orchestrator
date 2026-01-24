"""
Feature Prompt Writer Agent - Core agent for generating evaluation prompts
Implements metric-first, grounded, RAI-by-design prompt generation.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .schemas import (
    FeatureMetadata, FeatureSpec, PromptOutput, 
    QualityMetric, ResponsibleAIConstraints
)
from .metrics_registry import (
    METRICS_REGISTRY, get_metric_definition, get_metrics_for_category,
    suggest_additional_metrics, get_default_metrics_for_category
)
from .prompt_templates import build_evaluation_prompt


class FeaturePromptWriterAgent:
    """
    Core agent for generating metric-driven evaluation prompts.
    
    Design Principles:
    - Metric-first: Evaluation criteria are explicit and measurable
    - Grounded: No hallucinated judgment criteria
    - RAI by design: Safety, bias, privacy checks embedded
    - Extensible: Easy to add new features without re-engineering
    - Stateless & reproducible: Same input -> same output
    """
    
    def __init__(self):
        self.registry = METRICS_REGISTRY
    
    def generate(
        self,
        feature: FeatureSpec | FeatureMetadata,
        language: Optional[str] = None
    ) -> PromptOutput:
        """
        Generate an evaluation prompt for the given feature.
        
        Args:
            feature: Feature specification or metadata
            language: Target language (defaults to first supported)
            
        Returns:
            PromptOutput with evaluation prompt and metadata
        """
        # Handle both FeatureSpec and FeatureMetadata
        if isinstance(feature, FeatureMetadata):
            spec = self._metadata_to_spec(feature)
        else:
            spec = feature
        
        # Determine target language
        lang = language or (spec.languages_supported[0] if spec.languages_supported else "en")
        
        # Resolve metrics
        metrics_used, metric_defs, suggested = self._resolve_metrics(
            category=spec.category,
            user_metrics=spec.success_metrics,
            language=lang
        )
        
        # Apply RAI constraints
        metrics_used, metric_defs, rai_checks = self._apply_rai_constraints(
            metrics_used=metrics_used,
            metric_defs=metric_defs,
            privacy_sensitive=spec.privacy_sensitive,
            safety_critical=spec.safety_critical
        )
        
        # Build the evaluation prompt
        prompt = build_evaluation_prompt(
            feature_name=spec.name,
            category=spec.category,
            language=lang,
            metrics_used=metrics_used,
            metric_defs=metric_defs
        )
        
        return PromptOutput(
            feature_name=spec.name,
            group=spec.group,
            category=spec.category,
            language=lang,
            metrics_used=metrics_used,
            metric_definitions=metric_defs,
            suggested_additional_metrics=suggested,
            evaluation_prompt=prompt,
            rai_checks_applied=rai_checks
        )
    
    def _resolve_metrics(
        self,
        category: str,
        user_metrics: Optional[List[str]],
        language: str
    ) -> tuple[List[str], Dict[str, Dict[str, Any]], List[str]]:
        """
        Resolve which metrics to use and gather their definitions.
        
        Returns:
            (metrics_used, metric_definitions, suggested_additional)
        """
        category = (category or "other").strip().lower()
        
        # Get metrics to use
        if user_metrics and any(m.strip() for m in user_metrics):
            metrics_used = [m.strip().lower() for m in user_metrics if m.strip()]
        else:
            metrics_used = get_default_metrics_for_category(category)
        
        # Build definitions dict
        metric_defs: Dict[str, Dict[str, Any]] = {}
        for m in metrics_used:
            if m in self.registry:
                metric = self.registry[m]
                metric_defs[m] = {
                    "definition": metric.get_definition(language),
                    "weight": metric.weight,
                    "rai_tags": metric.rai_tags,
                    "is_primary": metric.is_primary,
                }
            else:
                metric_defs[m] = {
                    "definition": "(custom metric - definition not in registry)",
                    "weight": 1.0,
                    "rai_tags": [],
                    "is_primary": False,
                }
        
        # Suggest additional metrics
        suggested = suggest_additional_metrics(category, metrics_used)
        
        return metrics_used, metric_defs, suggested
    
    def _apply_rai_constraints(
        self,
        metrics_used: List[str],
        metric_defs: Dict[str, Dict[str, Any]],
        privacy_sensitive: bool,
        safety_critical: bool
    ) -> tuple[List[str], Dict[str, Dict[str, Any]], List[str]]:
        """
        Apply Responsible AI constraints by ensuring required metrics are included.
        
        Returns:
            (updated_metrics, updated_defs, rai_checks_applied)
        """
        rai_checks = []
        
        # Always include safety for GenAI features
        if "safety" not in metrics_used and "safety" in self.registry:
            metrics_used = metrics_used + ["safety"]
            metric = self.registry["safety"]
            metric_defs["safety"] = {
                "definition": metric.get_definition("en"),
                "weight": metric.weight,
                "rai_tags": metric.rai_tags,
                "is_primary": metric.is_primary,
            }
            rai_checks.append("safety_check_added")
        
        # Add privacy if feature is privacy-sensitive
        if privacy_sensitive and "privacy" not in metrics_used and "privacy" in self.registry:
            metrics_used = metrics_used + ["privacy"]
            metric = self.registry["privacy"]
            metric_defs["privacy"] = {
                "definition": metric.get_definition("en"),
                "weight": metric.weight,
                "rai_tags": metric.rai_tags,
                "is_primary": metric.is_primary,
            }
            rai_checks.append("privacy_check_added")
        
        # Add groundedness for safety-critical features
        if safety_critical and "groundedness" not in metrics_used and "groundedness" in self.registry:
            metrics_used = metrics_used + ["groundedness"]
            metric = self.registry["groundedness"]
            metric_defs["groundedness"] = {
                "definition": metric.get_definition("en"),
                "weight": metric.weight,
                "rai_tags": metric.rai_tags,
                "is_primary": metric.is_primary,
            }
            rai_checks.append("groundedness_check_added")
        
        return metrics_used, metric_defs, rai_checks
    
    def _metadata_to_spec(self, metadata: FeatureMetadata) -> FeatureSpec:
        """Convert FeatureMetadata to FeatureSpec"""
        return FeatureSpec(
            group=metadata.group,
            name=metadata.feature_name,
            description=metadata.feature_description,
            category=metadata.category,
            input_format=metadata.input_description,
            output_format=metadata.output_description,
            languages_supported=metadata.supported_languages,
            success_metrics=metadata.success_metrics or [m.name for m in metadata.quality_metrics],
            privacy_sensitive=metadata.responsible_ai.no_pii_leakage,
            safety_critical=metadata.responsible_ai.safety_critical,
        )
    
    def export_feature_json(self, feature: FeatureSpec | FeatureMetadata) -> dict:
        """Export feature to JSON-serializable dict"""
        if isinstance(feature, FeatureMetadata):
            return feature.model_dump()
        return asdict(feature)
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported feature categories"""
        return ["summarization", "auto_reply", "translation", "assistant", "content_generation", "classification", "other"]
    
    def get_available_metrics(self) -> Dict[str, str]:
        """Get all available metrics with their definitions"""
        return {name: metric.get_definition("en") for name, metric in self.registry.items()}
