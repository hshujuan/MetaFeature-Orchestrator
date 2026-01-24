"""
Core package - MetaFeature Orchestrator core components
"""
from .schemas import (
    InputOutputFormat, FeatureCategory, QualityMetric, OutputExample,
    ResponsibleAIConstraints, FeatureMetadata, FeatureSpec, PromptOutput,
    GeneratedPromptResult, feature_metadata_to_spec, spec_to_feature_metadata
)
from .metrics_registry import (
    MetricDefinition, METRICS_REGISTRY, DEFAULT_METRICS_BY_CATEGORY,
    get_metric, get_metric_definition, get_default_metrics_for_category,
    get_metrics_for_category, get_metrics_by_rai_tag, suggest_additional_metrics,
    get_all_metrics, get_available_categories
)
from .prompt_templates import (
    EVALUATION_AGENT_SYSTEM_PROMPT, FEATURE_EVALUATION_REQUEST_TEMPLATE,
    template_auto_reply, template_summarization, template_translation,
    template_generic, get_template_for_category, build_evaluation_prompt
)
from .agent import FeaturePromptWriterAgent
from .llm_client import (
    LLMClient, get_llm_client, get_openai_client, get_deployment_name, chat_completion
)
from .database import FeatureStore, PromptTemplateStore, RunStore
from .app import create_app, main as run_app


__all__ = [
    # Schemas
    "InputOutputFormat", "FeatureCategory", "QualityMetric", "OutputExample",
    "ResponsibleAIConstraints", "FeatureMetadata", "FeatureSpec", "PromptOutput",
    "GeneratedPromptResult", "feature_metadata_to_spec", "spec_to_feature_metadata",
    # Metrics
    "MetricDefinition", "METRICS_REGISTRY", "DEFAULT_METRICS_BY_CATEGORY",
    "get_metric", "get_metric_definition", "get_default_metrics_for_category",
    "get_metrics_for_category", "get_metrics_by_rai_tag", "suggest_additional_metrics",
    "get_all_metrics", "get_available_categories",
    # Templates
    "EVALUATION_AGENT_SYSTEM_PROMPT", "FEATURE_EVALUATION_REQUEST_TEMPLATE",
    "template_auto_reply", "template_summarization", "template_translation",
    "template_generic", "get_template_for_category", "build_evaluation_prompt",
    # Agent
    "FeaturePromptWriterAgent",
    # LLM Client
    "LLMClient", "get_llm_client", "get_openai_client", "get_deployment_name", "chat_completion",
    # Database
    "FeatureStore", "PromptTemplateStore", "RunStore",
    # App
    "create_app", "run_app",
]
