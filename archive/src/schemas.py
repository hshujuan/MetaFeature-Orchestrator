from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class InputOutputFormat(str, Enum):
    TEXT_TO_TEXT = "text-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    IMAGE_TO_IMAGE = "image-to-image"
    VOICE_TO_TEXT = "voice-to-text"
    TEXT_TO_VOICE = "text-to-voice"
    MULTIMODAL = "multimodal"


class QualityMetric(BaseModel):
    """Individual quality metric for evaluation"""
    name: str = Field(..., description="Metric name (e.g., accuracy, fluency)")
    description: str = Field(..., description="What this metric measures")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Importance weight (0-1)")
    is_primary: bool = Field(default=False, description="Whether this is a primary success metric")


class OutputExample(BaseModel):
    """Example of good or bad output for few-shot learning"""
    input_text: str = Field(..., description="The input that was given")
    output_text: str = Field(..., description="The output that was produced")
    is_good_example: bool = Field(..., description="True if this is a good example, False if bad")
    explanation: str = Field(..., description="Why this is considered good or bad")


class ResponsibleAIConstraints(BaseModel):
    """Responsible AI considerations and constraints"""
    no_pii_leakage: bool = Field(default=True, description="Ensure no personal data is leaked")
    bias_check_required: bool = Field(default=True, description="Check for biased content")
    toxicity_check_required: bool = Field(default=True, description="Check for offensive content")
    cultural_sensitivity: bool = Field(default=True, description="Consider cultural norms")
    custom_constraints: List[str] = Field(default_factory=list, description="Additional constraints")


class FeatureMetadata(BaseModel):
    """Comprehensive metadata about an AI feature for evaluation"""
    
    # Basic Information
    feature_name: str = Field(..., description="Name of the feature (e.g., Auto-summarize News)")
    feature_description: str = Field(..., description="Detailed description of what the feature does")
    category: str = Field(..., description="Category: Productivity, Communication, Intelligence, Creative")
    
    # Input/Output Format
    io_format: InputOutputFormat = Field(..., description="The input/output format type")
    input_description: str = Field(..., description="Description of input (e.g., 'Email text with subject and body')")
    output_description: str = Field(..., description="Description of expected output (e.g., 'Concise summary in 2-3 sentences')")
    
    # Localization
    supported_languages: List[str] = Field(default=["English"], description="Languages the feature supports")
    target_language: str = Field(default="English", description="Language being evaluated")
    target_location: str = Field(default="United States", description="Specific locale/region for evaluation")
    locale_considerations: str = Field(default="", description="Special locale considerations (e.g., honorifics, date formats)")
    
    # Success Criteria & Metrics
    quality_metrics: List[QualityMetric] = Field(default_factory=list, description="Quality metrics for evaluation")
    
    # Context & Constraints
    responsible_ai: ResponsibleAIConstraints = Field(default_factory=ResponsibleAIConstraints)
    domain_constraints: str = Field(default="", description="Domain-specific constraints or requirements")
    
    # Examples
    good_examples: List[OutputExample] = Field(default_factory=list, description="Examples of good outputs")
    bad_examples: List[OutputExample] = Field(default_factory=list, description="Examples of bad outputs")
    
    # Test Data
    input_data_sample: str = Field(..., description="Sample input data to evaluate")


class GeneratedPrompt(BaseModel):
    """Structured output from the evaluation prompt generator"""
    system_instruction: str = Field(..., description="System prompt for the evaluator")
    evaluation_prompt: str = Field(..., description="The main evaluation prompt")
    scoring_rubric: str = Field(..., description="Detailed scoring criteria")
    metrics_to_evaluate: List[str] = Field(default_factory=list, description="Metrics being evaluated")
    locale_adaptations: List[str] = Field(default_factory=list, description="Locale-specific adaptations applied")
    responsible_ai_checks: List[str] = Field(default_factory=list, description="RAI checks included")


# Predefined metric templates for common use cases
METRIC_TEMPLATES = {
    "summarization": [
        QualityMetric(name="Faithfulness", description="Output contains only information present in the source, no hallucinations", weight=1.0, is_primary=True),
        QualityMetric(name="Completeness", description="Captures all main points from the source", weight=0.8, is_primary=True),
        QualityMetric(name="Conciseness", description="Summary is appropriately brief without redundancy", weight=0.6, is_primary=False),
        QualityMetric(name="Fluency", description="Natural, grammatically correct language", weight=0.5, is_primary=False),
    ],
    "translation": [
        QualityMetric(name="Accuracy", description="Meaning is preserved correctly from source to target", weight=1.0, is_primary=True),
        QualityMetric(name="Fluency", description="Reads naturally in the target language", weight=0.9, is_primary=True),
        QualityMetric(name="Terminology", description="Domain-specific terms are correctly translated", weight=0.7, is_primary=False),
        QualityMetric(name="Style Preservation", description="Tone and style match the original", weight=0.5, is_primary=False),
    ],
    "auto_reply": [
        QualityMetric(name="Relevance", description="Reply directly addresses the question/topic", weight=1.0, is_primary=True),
        QualityMetric(name="Tone Appropriateness", description="Tone matches context (formal/casual)", weight=0.8, is_primary=True),
        QualityMetric(name="Completeness", description="Reply provides sufficient information", weight=0.7, is_primary=False),
        QualityMetric(name="Brevity", description="Appropriately concise for the medium", weight=0.5, is_primary=False),
    ],
    "content_generation": [
        QualityMetric(name="Prompt Adherence", description="Output follows the given instructions", weight=1.0, is_primary=True),
        QualityMetric(name="Creativity", description="Output is original and engaging", weight=0.7, is_primary=False),
        QualityMetric(name="Coherence", description="Logical flow and consistency", weight=0.8, is_primary=True),
        QualityMetric(name="Safety", description="No harmful, biased, or inappropriate content", weight=1.0, is_primary=True),
    ],
    "classification": [
        QualityMetric(name="Precision", description="Correctly identified positives among predictions", weight=1.0, is_primary=True),
        QualityMetric(name="Recall", description="Correctly identified all actual positives", weight=1.0, is_primary=True),
        QualityMetric(name="Confidence Calibration", description="Confidence scores reflect actual accuracy", weight=0.6, is_primary=False),
    ],
}
