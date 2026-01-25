"""
Unified Schemas - Pydantic models for feature metadata and prompt generation
Combines comprehensive feature metadata with Responsible AI constraints.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class InputOutputFormat(str, Enum):
    """Supported input/output modality combinations"""
    TEXT_TO_TEXT = "text-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    IMAGE_TO_IMAGE = "image-to-image"
    VOICE_TO_TEXT = "voice-to-text"
    TEXT_TO_VOICE = "text-to-voice"
    MULTIMODAL = "multimodal"


class FeatureCategory(str, Enum):
    """Standard feature categories"""
    SUMMARIZATION = "summarization"
    AUTO_REPLY = "auto_reply"
    TRANSLATION = "translation"
    ASSISTANT = "assistant"
    CONTENT_GENERATION = "content_generation"
    CLASSIFICATION = "classification"
    OTHER = "other"


# ═══════════════════════════════════════════════════════════════════
# QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════

class QualityMetric(BaseModel):
    """Individual quality metric for evaluation"""
    name: str = Field(..., description="Metric name (e.g., faithfulness, fluency)")
    description: str = Field(default="", description="What this metric measures")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Importance weight (0-1)")
    is_primary: bool = Field(default=False, description="Whether this is a primary success metric")
    rai_tags: List[str] = Field(default_factory=list, description="Associated RAI concerns")


# ═══════════════════════════════════════════════════════════════════
# EXAMPLES
# ═══════════════════════════════════════════════════════════════════

class OutputExample(BaseModel):
    """Example of good or bad output for few-shot learning"""
    input_text: str = Field(..., description="The input that was given")
    output_text: str = Field(..., description="The output that was produced")
    is_good_example: bool = Field(..., description="True if this is a good example, False if bad")
    explanation: str = Field(default="", description="Why this is considered good or bad")


# ═══════════════════════════════════════════════════════════════════
# RESPONSIBLE AI
# ═══════════════════════════════════════════════════════════════════

class ResponsibleAIConstraints(BaseModel):
    """Responsible AI considerations and constraints"""
    no_pii_leakage: bool = Field(default=True, description="Ensure no personal data is leaked")
    bias_check_required: bool = Field(default=True, description="Check for biased content")
    toxicity_check_required: bool = Field(default=True, description="Check for offensive content")
    cultural_sensitivity: bool = Field(default=True, description="Consider cultural norms")
    safety_critical: bool = Field(default=False, description="Feature has safety implications")
    custom_constraints: List[str] = Field(default_factory=list, description="Additional constraints")


# ═══════════════════════════════════════════════════════════════════
# FEATURE METADATA (Pydantic - for API/validation)
# ═══════════════════════════════════════════════════════════════════

class FeatureMetadata(BaseModel):
    """Comprehensive metadata about an AI feature for evaluation"""
    
    # Basic Information
    feature_name: str = Field(..., description="Name of the feature")
    feature_description: str = Field(default="", description="Detailed description")
    group: str = Field(default="General", description="Feature group for organization")
    category: str = Field(default="other", description="Category: summarization, auto_reply, etc.")
    
    # Input/Output Format
    io_format: InputOutputFormat = Field(default=InputOutputFormat.TEXT_TO_TEXT)
    input_description: str = Field(default="", description="Description of input format")
    output_description: str = Field(default="", description="Description of expected output")
    
    # Localization (using BCP 47 locale codes)
    supported_locales: List[str] = Field(default_factory=lambda: ["en-US"], description="Supported locales (BCP 47)")
    target_locale: str = Field(default="en-US", description="Target locale for evaluation (e.g., en-US, zh-CN, es-MX)")
    locale_considerations: str = Field(default="", description="Special locale/cultural requirements")
    
    # Legacy fields (for backward compatibility)
    supported_languages: List[str] = Field(default_factory=lambda: ["en"], description="[Deprecated] Use supported_locales")
    target_language: str = Field(default="en", description="[Deprecated] Use target_locale")
    target_location: str = Field(default="", description="[Deprecated] Use target_locale")
    
    # Success Criteria & Metrics
    quality_metrics: List[QualityMetric] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list, description="Metric names to use")
    
    # Context & Constraints
    responsible_ai: ResponsibleAIConstraints = Field(default_factory=ResponsibleAIConstraints)
    domain_constraints: str = Field(default="", description="Domain-specific requirements")
    
    # Examples
    good_examples: List[OutputExample] = Field(default_factory=list)
    bad_examples: List[OutputExample] = Field(default_factory=list)
    
    # Test Data
    input_data_sample: str = Field(default="", description="Sample input data to evaluate")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()


# ═══════════════════════════════════════════════════════════════════
# FEATURE METADATA (Dataclass - for lightweight internal use)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FeatureSpec:
    """Lightweight feature specification for internal processing"""
    group: str
    name: str
    description: str
    category: str
    input_format: str
    output_format: str
    locales_supported: List[str]  # BCP 47 locale codes (e.g., ['en-US', 'es-MX'])
    success_metrics: List[str]
    privacy_sensitive: bool = True
    safety_critical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def languages_supported(self) -> List[str]:
        """Legacy property - extracts language codes from locales"""
        return list(set(loc.split("-")[0] for loc in self.locales_supported))


# ═══════════════════════════════════════════════════════════════════
# PROMPT OUTPUT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PromptOutput:
    """Output from prompt generation"""
    feature_name: str
    group: str
    category: str
    locale: str  # Full locale code (e.g., 'en-US', 'zh-CN')
    metrics_used: List[str]
    metric_definitions: Dict[str, Dict[str, Any]]
    suggested_additional_metrics: List[str]
    evaluation_prompt: str
    rai_checks_applied: List[str] = None
    
    def __post_init__(self):
        if self.rai_checks_applied is None:
            self.rai_checks_applied = []
    
    @property
    def language(self) -> str:
        """Legacy property - extracts language code from locale"""
        return self.locale.split("-")[0] if "-" in self.locale else self.locale


class GeneratedPromptResult(BaseModel):
    """Structured result from evaluation prompt generation"""
    feature_id: str = Field(default="")
    feature_name: str
    locale: str = Field(default="en-US", description="Full locale code (BCP 47)")
    category: str
    metrics_used: List[str]
    evaluation_prompt: str
    scoring_rubric: str = Field(default="")
    rai_checks: List[str] = Field(default_factory=list)
    locale_adaptations: List[str] = Field(default_factory=list)
    suggested_metrics: List[str] = Field(default_factory=list)
    
    # Legacy field
    language: str = Field(default="en", description="[Deprecated] Use locale")


# ═══════════════════════════════════════════════════════════════════
# CONVERSION HELPERS
# ═══════════════════════════════════════════════════════════════════

def feature_metadata_to_spec(metadata: FeatureMetadata) -> FeatureSpec:
    """Convert FeatureMetadata (Pydantic) to FeatureSpec (dataclass)"""
    # Use new locale field, fall back to legacy language field
    locales = metadata.supported_locales if metadata.supported_locales != ["en-US"] else \
              [f"{lang}-US" if lang == "en" else lang for lang in metadata.supported_languages]
    
    return FeatureSpec(
        group=metadata.group,
        name=metadata.feature_name,
        description=metadata.feature_description,
        category=metadata.category,
        input_format=metadata.input_description,
        output_format=metadata.output_description,
        locales_supported=locales,
        success_metrics=metadata.success_metrics or [m.name for m in metadata.quality_metrics],
        privacy_sensitive=metadata.responsible_ai.no_pii_leakage,
        safety_critical=metadata.responsible_ai.safety_critical,
    )


def spec_to_feature_metadata(spec: FeatureSpec) -> FeatureMetadata:
    """Convert FeatureSpec (dataclass) to FeatureMetadata (Pydantic)"""
    return FeatureMetadata(
        feature_name=spec.name,
        feature_description=spec.description,
        group=spec.group,
        category=spec.category,
        input_description=spec.input_format,
        output_description=spec.output_format,
        supported_locales=spec.locales_supported,
        supported_languages=spec.languages_supported,  # Derived property
