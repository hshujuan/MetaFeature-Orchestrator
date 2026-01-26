"""
MetaFeature Orchestrator - Gradio Application
Automatically generates high-quality evaluation prompts with metric-first design.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Handle both direct execution and module import
if __name__ == "__main__":
    # Add src to path for direct execution
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    from src.core.schemas import (
        FeatureMetadata,
        FeatureCategory,
        InputOutputFormat,
        ResponsibleAIConstraints,
        QualityMetric
    )
    from src.core.metrics_registry import (
        get_metric,
        get_metric_definition,
        suggest_additional_metrics,
        DEFAULT_METRICS_BY_CATEGORY,
        METRICS_REGISTRY
    )
    from src.core.agent import FeaturePromptWriterAgent
    from src.core.database import FeatureStore, PromptTemplateStore, RunStore
    from src.core.code_metrics import generate_code_metrics_sample, get_code_metrics_for_category
    from src.core.llm_client import LLMClient
    # Try to import AI agent (optional)
    try:
        from src.core.ai_agent import MetaFeatureAgent
        AI_AGENT_AVAILABLE = True
    except ImportError:
        MetaFeatureAgent = None
        AI_AGENT_AVAILABLE = False
else:
    from .schemas import (
        FeatureMetadata,
        FeatureCategory,
        InputOutputFormat,
        ResponsibleAIConstraints,
        QualityMetric
    )
    from .metrics_registry import (
        get_metric,
        get_metric_definition,
        suggest_additional_metrics,
        DEFAULT_METRICS_BY_CATEGORY,
        METRICS_REGISTRY
    )
    from .agent import FeaturePromptWriterAgent
    from .database import FeatureStore, PromptTemplateStore, RunStore
    from .code_metrics import generate_code_metrics_sample, get_code_metrics_for_category
    from .llm_client import LLMClient
    # Try to import AI agent (optional)
    try:
        from .ai_agent import MetaFeatureAgent
        AI_AGENT_AVAILABLE = True
    except ImportError:
        MetaFeatureAgent = None
        AI_AGENT_AVAILABLE = False


# Initialize stores
feature_store = FeatureStore()
template_store = PromptTemplateStore()
run_store = RunStore()


# --- Predefined Feature Templates (from app2.py) ---
GROUPED_FEATURES: Dict[str, Dict[str, dict]] = {
    "Summarization": {
        "Summarize News": {
            "description": "Summarize a news article into a short, accurate summary for the user.",
            "category": "summarization",
            "input_format": "text",
            "output_format": "text",
            "typical_input": "A long news article about recent events...",
            "expected_output": "A 2-3 sentence summary capturing key facts.",
            "privacy_sensitive": False,
            "safety_critical": True,
        },
        "Summarize Email Thread": {
            "description": "Summarize an email thread and highlight action items.",
            "category": "summarization",
            "input_format": "text",
            "output_format": "text",
            "typical_input": "Email thread with multiple replies discussing a project...",
            "expected_output": "Summary with bullet points and action items.",
            "privacy_sensitive": True,
            "safety_critical": True,
        },
        "Summarize Document": {
            "description": "Create an executive summary of a long document.",
            "category": "summarization",
            "input_format": "text",
            "output_format": "markdown",
            "typical_input": "A 10-page business report...",
            "expected_output": "Executive summary with key findings and recommendations.",
            "privacy_sensitive": False,
            "safety_critical": False,
        },
    },
    "Auto Reply": {
        "Auto-Reply Email": {
            "description": "Draft a polite, helpful reply to an inbound email.",
            "category": "auto_reply",
            "input_format": "text",
            "output_format": "text",
            "typical_input": "Hi, I was wondering about the status of my order #12345...",
            "expected_output": "A professional response addressing the customer's inquiry.",
            "privacy_sensitive": True,
            "safety_critical": True,
        },
        "Auto-Reply Message": {
            "description": "Draft a short friendly reply to an inbound chat message.",
            "category": "auto_reply",
            "input_format": "text",
            "output_format": "text",
            "typical_input": "Hey, are you free for a call tomorrow?",
            "expected_output": "Sure! What time works best for you?",
            "privacy_sensitive": True,
            "safety_critical": False,
        },
    },
    "Translation": {
        "Translate Document": {
            "description": "Translate a document while preserving formatting and meaning.",
            "category": "translation",
            "input_format": "text",
            "output_format": "text",
            "typical_input": "Original document in source language...",
            "expected_output": "Accurately translated document in target language.",
            "privacy_sensitive": False,
            "safety_critical": False,
        },
    },
    "Classification": {
        "Sentiment Analysis": {
            "description": "Classify the sentiment of a text as positive, negative, or neutral.",
            "category": "classification",
            "input_format": "text",
            "output_format": "json",
            "typical_input": "I absolutely loved this product! Best purchase ever.",
            "expected_output": '{"sentiment": "positive", "confidence": 0.95}',
            "privacy_sensitive": False,
            "safety_critical": False,
        },
        "Intent Detection": {
            "description": "Detect the user's intent from their message.",
            "category": "classification",
            "input_format": "text",
            "output_format": "json",
            "typical_input": "I want to cancel my subscription",
            "expected_output": '{"intent": "cancel_subscription", "confidence": 0.92}',
            "privacy_sensitive": True,
            "safety_critical": False,
        },
    },
    # ─────────────────────────────────────────────────────────────────
    # Apple Intelligence - Image Features
    # ─────────────────────────────────────────────────────────────────
    "Image Understanding": {
        "Visual Look Up": {
            "description": "Identify objects, plants, pets, landmarks, and artwork in photos. Provide relevant information about the identified subject.",
            "category": "classification",
            "input_format": "image",
            "output_format": "json",
            "typical_input": "Photo of a Golden Gate Bridge at sunset",
            "expected_output": '{"subject": "Golden Gate Bridge", "category": "landmark", "location": "San Francisco, CA", "info": "Suspension bridge spanning the Golden Gate strait, completed in 1937"}',
            "privacy_sensitive": False,
            "safety_critical": False,
        },
        "Image Captioning": {
            "description": "Generate natural language descriptions of images for accessibility (VoiceOver) and search indexing.",
            "category": "generation",
            "input_format": "image",
            "output_format": "text",
            "typical_input": "Photo of a family having a picnic in a park",
            "expected_output": "A family of four sitting on a red checkered blanket in a sunny park, with a picnic basket and sandwiches spread out before them. Two children are playing with a golden retriever.",
            "privacy_sensitive": True,
            "safety_critical": True,
        },
        "Live Text (OCR)": {
            "description": "Extract and recognize text from images, screenshots, and camera feed. Make text selectable, searchable, and translatable.",
            "category": "extraction",
            "input_format": "image",
            "output_format": "json",
            "typical_input": "Photo of a restaurant menu or business card",
            "expected_output": '{"text_blocks": [{"text": "Joe\'s Coffee Shop", "bounds": [10, 20, 200, 50]}, {"text": "555-123-4567", "bounds": [10, 60, 150, 80], "type": "phone"}], "language": "en"}',
            "privacy_sensitive": True,
            "safety_critical": False,
        },
        "Photo Search": {
            "description": "Enable natural language search across photo library. Find photos by describing content, people, places, events, or objects.",
            "category": "extraction",
            "input_format": "text",
            "output_format": "json",
            "typical_input": "Find photos of my dog at the beach from last summer",
            "expected_output": '{"query_understanding": {"subject": "dog", "location": "beach", "time_range": "summer 2025"}, "search_filters": {"has_pet": true, "scene_type": "beach", "date_range": ["2025-06-01", "2025-08-31"]}}',
            "privacy_sensitive": True,
            "safety_critical": False,
        },
    },
    "Image Generation": {
        "Image Playground": {
            "description": "Generate images from text descriptions in various styles (animation, illustration, sketch). Create personalized images featuring people from photo library.",
            "category": "generation",
            "input_format": "text",
            "output_format": "image",
            "typical_input": "A cat astronaut floating in space with Earth in the background, illustration style",
            "expected_output": "Generated illustration of a cartoon cat in a spacesuit floating among stars with a blue Earth visible below",
            "privacy_sensitive": True,
            "safety_critical": True,
        },
        "Genmoji": {
            "description": "Create custom emoji from text descriptions. Generate personalized emoji that match user's intent and can feature people from photos.",
            "category": "generation",
            "input_format": "text",
            "output_format": "image",
            "typical_input": "A happy T-Rex wearing a party hat",
            "expected_output": "Custom emoji-style image of a smiling green T-Rex dinosaur with a colorful striped party hat",
            "privacy_sensitive": False,
            "safety_critical": True,
        },
        "Memory Movie": {
            "description": "Automatically create narrative photo/video montages from library. Select relevant media, arrange chronologically, add music and transitions based on theme.",
            "category": "generation",
            "input_format": "text",
            "output_format": "json",
            "typical_input": "Create a memory movie of my trip to Japan",
            "expected_output": '{"title": "Japan Adventures", "duration": "2:30", "clips": [{"type": "photo", "id": "img_001", "duration": 3}, {"type": "video", "id": "vid_002", "start": 0, "end": 5}], "music": {"mood": "uplifting", "tempo": "medium"}, "transitions": ["fade", "slide"]}',
            "privacy_sensitive": True,
            "safety_critical": False,
        },
    },
    "Image Editing": {
        "Clean Up (Object Removal)": {
            "description": "Identify and remove unwanted objects, people, or distractions from photos using generative AI inpainting.",
            "category": "generation",
            "input_format": "image",
            "output_format": "image",
            "typical_input": "Photo of a beach with a trash can that needs to be removed",
            "expected_output": "Same beach photo with the trash can seamlessly removed and the background naturally filled in",
            "privacy_sensitive": False,
            "safety_critical": True,
        },
        "Subject Lift (Background Removal)": {
            "description": "Automatically detect and separate the main subject from the background. Enable drag-and-drop of subjects into other apps.",
            "category": "extraction",
            "input_format": "image",
            "output_format": "image",
            "typical_input": "Photo of a dog sitting on grass",
            "expected_output": "PNG image of the dog with transparent background, cleanly separated from the grass",
            "privacy_sensitive": False,
            "safety_critical": False,
        },
        "Portrait Mode Enhancement": {
            "description": "Apply depth-based blur effects to photos. Adjust aperture, lighting, and focus point post-capture.",
            "category": "generation",
            "input_format": "image",
            "output_format": "image",
            "typical_input": "Portrait photo with flat background",
            "expected_output": "Same portrait with natural bokeh effect, subject in sharp focus, background artistically blurred",
            "privacy_sensitive": False,
            "safety_critical": False,
        },
    },
    "Image Safety": {
        "Sensitive Content Warning": {
            "description": "Detect and blur sensitive or explicit content in images before display. Provide content warnings with option to view.",
            "category": "classification",
            "input_format": "image",
            "output_format": "json",
            "typical_input": "Incoming image in Messages app",
            "expected_output": '{"is_sensitive": true, "categories": ["nudity"], "confidence": 0.95, "action": "blur", "warning_message": "This image may contain sensitive content"}',
            "privacy_sensitive": True,
            "safety_critical": True,
        },
        "Face Detection & Privacy": {
            "description": "Detect faces in images for privacy protection, photo organization, and People album features.",
            "category": "extraction",
            "input_format": "image",
            "output_format": "json",
            "typical_input": "Group photo at a party",
            "expected_output": '{"faces": [{"bounds": [100, 50, 150, 120], "person_id": "person_1", "confidence": 0.98}, {"bounds": [250, 60, 300, 130], "person_id": null, "confidence": 0.91}], "face_count": 2}',
            "privacy_sensitive": True,
            "safety_critical": True,
        },
    },
    # ─────────────────────────────────────────────────────────────────
    # Personal Intelligence - Advanced Reasoning & Memory
    # ─────────────────────────────────────────────────────────────────
    "Personal Intelligence": {
        "Apple LifeGraph": {
            "description": """Apple LifeGraph™ — A Private, On-Device Multimodal Memory & Reasoning Engine.

Apple LifeGraph™ is an on-device, privacy-first, multimodal AI system that builds a personal causal graph of a user's life across time—events, habits, people, health, tasks, places, emotions—and uses reasoning (not just prediction) to assist with decisions.

Data Sources: 📅 Calendar + 📍 Location + ❤️ Health + 📸 Photos + ✍️ Notes + 🧠 Intent → structured life understanding

Key Capabilities:
- Temporal reasoning across multiple data modalities
- Causal inference for habit and pattern detection
- Privacy-preserving on-device processing
- Personalized decision support with explainable suggestions""",
            "category": "personal_assistant",
            "input_format": "json",
            "output_format": "json",
            "typical_input": '''{
  "calendar": {"pattern": "Tue/Thu: Meetings until 8:30pm"},
  "healthkit": {"sleep_avg_tue": "5h 12m", "knee_pain_wed": "elevated"},
  "activity": {"soccer_practice": "Wed evenings"},
  "historical": {"workout_skip_rate_low_sleep": 0.78}
}''',
            "expected_output": '''{
  "insight": "On weeks with late Tuesday meetings, your Wednesday workout success rate drops from 71% to 22%.",
  "suggestions": [
    {"action": "shift_workout", "detail": "Move Wednesday workouts to Friday"},
    {"action": "reduce_intensity", "detail": "Lower intensity after late meetings"},
    {"action": "dismiss", "detail": "Ignore this suggestion"}
  ],
  "reasoning": {"causal_chain": ["late_meeting", "reduced_sleep", "fatigue", "skipped_workout"], "confidence": 0.87}
}''',
            "privacy_sensitive": True,
            "safety_critical": True,
        },
        "Proactive Suggestions": {
            "description": "Surface timely, context-aware suggestions based on user patterns, location, time, and upcoming events. Anticipate needs before users ask.",
            "category": "personal_assistant",
            "input_format": "json",
            "output_format": "json",
            "typical_input": '{"time": "8:45am", "location": "home", "calendar": "Meeting at 9am downtown", "traffic": "heavy", "usual_commute": "25min"}',
            "expected_output": '{"suggestion": "Leave now to arrive on time for your 9am meeting. Traffic is heavier than usual.", "urgency": "high", "action": "show_directions"}',
            "privacy_sensitive": True,
            "safety_critical": False,
        },
        "Personal Context Memory": {
            "description": "Maintain and query a personal knowledge graph of user preferences, relationships, and past interactions to provide contextual assistance.",
            "category": "personal_assistant",
            "input_format": "text",
            "output_format": "json",
            "typical_input": "What restaurant did I go to with Sarah last month?",
            "expected_output": '{"answer": "You went to Chez Marie on December 15th with Sarah.", "source": ["calendar_event", "photos_location"], "confidence": 0.94}',
            "privacy_sensitive": True,
            "safety_critical": False,
        },
    },
    "Custom": {}
}


def list_groups() -> List[str]:
    """Get sorted list of feature groups"""
    return sorted(GROUPED_FEATURES.keys())


def list_features(group: str) -> List[str]:
    """Get sorted list of features in a group"""
    if not group or group not in GROUPED_FEATURES:
        return []
    return sorted(GROUPED_FEATURES[group].keys())


def get_suggested_metrics(category: str) -> str:
    """Get suggested metrics for a category as a formatted string"""
    if not category:
        return ""
    
    cat = category.lower().replace(" ", "_")
    metrics = DEFAULT_METRICS_BY_CATEGORY.get(cat, DEFAULT_METRICS_BY_CATEGORY.get("generic", []))
    
    result = []
    for m in metrics:
        metric = get_metric(m)
        if metric:
            result.append(f"• **{metric.name}**: {metric.get_definition('en')}")
    
    return "\n".join(result)


def get_all_metrics_choices() -> List[str]:
    """Get all available metrics as choices"""
    return list(METRICS_REGISTRY.keys())


# Import locale utilities from prompt_templates
from .prompt_templates import (
    SUPPORTED_LOCALES, 
    get_language, 
    get_region, 
    normalize_locale,
    get_cultural_context,
    get_tone_guidance,
    get_privacy_framework
)


def get_locale_choices() -> List[str]:
    """Get all supported locales as dropdown choices"""
    return list(SUPPORTED_LOCALES.keys())


def get_locale_info_text(locale: str) -> str:
    """Generate informational text about the selected locale"""
    locale = normalize_locale(locale or "en-US")
    ctx = get_cultural_context(locale)
    framework = get_privacy_framework(locale)
    tone = get_tone_guidance(locale)
    
    # Extract key info
    formality = ctx.get("formality", "neutral")
    directness = ctx.get("directness", "direct")
    
    return f"""**{locale}**: {formality.capitalize()} tone, {directness} feedback, {framework} privacy framework
    
*Tone guidance: {tone[:100]}...*"""


def detect_feature_complexity(
    description: str,
    category: str,
    metrics: List[str]
) -> tuple[bool, str]:
    """
    Detect if a feature is complex enough to warrant AI agent processing.
    
    Returns:
        (is_complex, reason)
    """
    reasons = []
    complexity_score = 0
    
    desc_lower = description.lower()
    
    # Check for multimodal indicators
    multimodal_keywords = ["multimodal", "image", "photo", "video", "audio", "voice", 
                          "calendar", "health", "location", "sensor", "biometric"]
    multimodal_count = sum(1 for kw in multimodal_keywords if kw in desc_lower)
    if multimodal_count >= 2:
        complexity_score += 2
        reasons.append(f"multimodal ({multimodal_count} modalities)")
    
    # Check for novel/advanced concepts
    advanced_keywords = ["reasoning", "causal", "temporal", "graph", "memory", "context",
                        "personalization", "adaptive", "learning", "prediction", "inference",
                        "private", "on-device", "federated", "edge"]
    advanced_count = sum(1 for kw in advanced_keywords if kw in desc_lower)
    if advanced_count >= 2:
        complexity_score += 2
        reasons.append(f"advanced concepts ({advanced_count})")
    
    # Check description length (complex features have detailed descriptions)
    if len(description) > 500:
        complexity_score += 1
        reasons.append("detailed description")
    
    # Check if category is "other" or "generic" (not fitting standard templates)
    if category.lower() in ["other", "generic", ""]:
        complexity_score += 1
        reasons.append("novel category")
    
    # Check for privacy/safety critical indicators
    critical_keywords = ["medical", "health", "financial", "legal", "safety-critical",
                        "privacy-first", "sensitive", "personal data", "pii"]
    if any(kw in desc_lower for kw in critical_keywords):
        complexity_score += 1
        reasons.append("privacy/safety critical")
    
    # Check for integration complexity
    integration_keywords = ["integrate", "connect", "sync", "api", "cross-platform",
                          "ecosystem", "multiple sources", "data fusion"]
    if any(kw in desc_lower for kw in integration_keywords):
        complexity_score += 1
        reasons.append("integration complexity")
    
    is_complex = complexity_score >= 3
    reason = f"Complexity score: {complexity_score}/8 ({', '.join(reasons)})" if reasons else "Standard feature"
    
    return is_complex, reason


def get_locale_display_choices() -> List[tuple]:
    """Get locales with display names for dropdown"""
    return [(f"{code} - {name}", code) for code, name in SUPPORTED_LOCALES.items()]


# Legacy mapping (for backward compatibility)
LOCATION_TO_LOCALE = {
    "US": "en-US",
    "GB": "en-GB",
    "AU": "en-AU",
    "IN": "en-IN",
    "SG": "en-SG",
    "CA": "en-CA",
    "CN": "zh-CN",
    "TW": "zh-TW",
    "HK": "zh-HK",
    "JP": "ja-JP",
    "KR": "ko-KR",
    "DE": "de-DE",
    "FR": "fr-FR",
    "ES": "es-ES",
    "MX": "es-MX",
    "AR": "es-AR",
    "BR": "pt-BR",
    "PT": "pt-PT",
    "GLOBAL": "en-US",
}


def get_language_for_location(location: str) -> str:
    """[Deprecated] Get the primary language for a location. Use get_locale_for_location instead."""
    locale = LOCATION_TO_LOCALE.get(location, "en-US")
    return get_language(locale)


def get_locale_for_location(location: str) -> str:
    """Get the locale code for a location"""
    return LOCATION_TO_LOCALE.get(location, "en-US")


def format_metric_definitions(metric_ids: List[str], locale_or_language: str = "en-US") -> str:
    """Format metric definitions for display"""
    language = get_language(locale_or_language) if "-" in locale_or_language else locale_or_language
    lines = []
    for m_id in metric_ids:
        metric = get_metric(m_id)
        if metric:
            lines.append(f"• **{metric.name}**: {metric.get_definition(language)}")
    return "\n".join(lines) if lines else "(none)"


def update_feature_choices(group: str):
    """Update feature dropdown when group changes"""
    features = list_features(group)
    return gr.update(choices=features, value=features[0] if features else None)


def load_feature_template(group: str, feature_name: str):
    """Load a predefined feature template into the form"""
    if not group or not feature_name or group not in GROUPED_FEATURES:
        return [gr.update()] * 8 + ["❌ Please select a group and feature first."]
    
    if feature_name not in GROUPED_FEATURES[group]:
        return [gr.update()] * 8 + ["❌ Feature not found."]
    
    feat = GROUPED_FEATURES[group][feature_name]
    
    # Get default metrics for category
    cat = feat.get("category", "generic")
    default_metrics = DEFAULT_METRICS_BY_CATEGORY.get(cat, DEFAULT_METRICS_BY_CATEGORY.get("other", []))
    
    return [
        gr.update(value=feature_name),  # feature_name
        gr.update(value=feat.get("description", "")),  # feature_description
        gr.update(value=feat.get("category", "generic")),  # category
        gr.update(value=feat.get("input_format", "text")),  # input_format
        gr.update(value=feat.get("output_format", "text")),  # output_format
        gr.update(value=feat.get("typical_input", "")),  # typical_input
        gr.update(value=feat.get("expected_output", "")),  # expected_output
        gr.update(value=default_metrics),  # selected_metrics
        f"✅ Loaded **{feature_name}** template! Go to **📝 Feature Definition** tab to review and edit.",  # status
    ]


def add_custom_feature(
    new_group: str,
    new_name: str,
    description: str,
    category: str,
    input_format: str,
    output_format: str,
    typical_input: str,
    expected_output: str,
    privacy_sensitive: bool,
    safety_critical: bool
):
    """Add a new custom feature template"""
    group = (new_group or "Custom").strip() or "Custom"
    name = (new_name or "").strip()
    
    if not name:
        return "❌ Please provide a Feature Name.", gr.update(), gr.update()
    
    if group not in GROUPED_FEATURES:
        GROUPED_FEATURES[group] = {}
    
    GROUPED_FEATURES[group][name] = {
        "description": description or "",
        "category": (category or "generic").strip(),
        "input_format": input_format or "text",
        "output_format": output_format or "text",
        "typical_input": typical_input or "",
        "expected_output": expected_output or "",
        "privacy_sensitive": bool(privacy_sensitive),
        "safety_critical": bool(safety_critical),
    }
    
    return (
        f"✅ Added feature '{name}' under group '{group}'.",
        gr.update(choices=list_groups(), value=group),
        gr.update(choices=list_features(group), value=name)
    )


def generate_prompt(
    feature_name: str,
    feature_description: str,
    category: str,
    input_format: str,
    output_format: str,
    typical_input_example: str,
    expected_output_example: str,
    selected_metrics: List[str],
    additional_context: str,
    # RAI constraints
    check_safety: bool,
    check_privacy: bool,
    check_fairness: bool,
    check_transparency: bool,
    rai_additional_notes: str,
    locale: str = "en-US",
    use_ai_agent: str = "auto"  # "auto", "always", "never"
) -> tuple[str, str, str, str, str, str, str]:
    """
    Generate an evaluation prompt using the agent.
    
    Args:
        locale: Full locale code (e.g., 'en-US', 'zh-CN', 'es-MX')
        use_ai_agent: "auto" (detect complexity), "always" (force AI), "never" (legacy only)
    
    Returns:
        Tuple of (evaluation_prompt, metric_definitions, suggested_metrics, 
                  metrics_used, json_spec, code_metrics, status_message)
    """
    # Handle None values
    feature_name = feature_name or ""
    feature_description = feature_description or ""
    category = category or "other"
    input_format = input_format or "text"
    output_format = output_format or "text"
    typical_input_example = typical_input_example or ""
    expected_output_example = expected_output_example or ""
    additional_context = additional_context or ""
    rai_additional_notes = rai_additional_notes or ""
    locale = normalize_locale(locale or "en-US")
    language = get_language(locale)
    
    if not feature_name.strip():
        return "", "", "", "", "", "", "❌ Please provide a feature name"
    
    if not feature_description.strip():
        return "", "", "", "", "", "", "❌ Please provide a feature description"
    
    if not selected_metrics:
        return "", "", "", "", "", "", "❌ Please select at least one quality metric"
    
    # Determine whether to use AI agent
    use_ai = False
    complexity_reason = ""
    
    if use_ai_agent == "always" and AI_AGENT_AVAILABLE:
        use_ai = True
        complexity_reason = "AI Agent mode: always"
    elif use_ai_agent == "auto" and AI_AGENT_AVAILABLE:
        is_complex, complexity_reason = detect_feature_complexity(
            feature_description, category, selected_metrics
        )
        use_ai = is_complex
    # "never" or AI_AGENT_AVAILABLE = False → use_ai stays False
    
    try:
        if use_ai and AI_AGENT_AVAILABLE and MetaFeatureAgent is not None:
            # === AI AGENT PATH ===
            logger.info(f"Using AI Agent for complex feature: {complexity_reason}")
            
            ai_agent = MetaFeatureAgent()
            
            # Build a detailed request for the AI agent
            ai_request = f"""Generate a comprehensive evaluation prompt for this feature:

**Feature Name:** {feature_name.strip()}
**Description:** {feature_description.strip()}
**Category:** {category}
**Target Locale:** {locale}

**Input Format:** {input_format}
**Output Format:** {output_format}

**Example Input:** {typical_input_example[:500] if typical_input_example else 'Not provided'}
**Expected Output:** {expected_output_example[:500] if expected_output_example else 'Not provided'}

**Selected Metrics:** {', '.join(selected_metrics)}
**Additional Context:** {additional_context if additional_context else 'None'}

**RAI Requirements:**
- Safety checks: {check_safety}
- Privacy checks: {check_privacy}  
- Fairness checks: {check_fairness}
- Transparency: {check_transparency}
- Additional notes: {rai_additional_notes if rai_additional_notes else 'None'}

Please:
1. Analyze this feature and suggest any additional metrics that would be valuable
2. Identify any privacy/safety concerns specific to this feature
3. Generate a comprehensive evaluation prompt with detailed rubrics
4. Include locale-specific considerations for {locale}
"""
            
            response = ai_agent.chat(ai_request)
            
            if response.success and response.message:
                evaluation_prompt = response.message
                
                # Try to extract structured prompt if available
                if response.evaluation_prompt:
                    evaluation_prompt = response.evaluation_prompt
                
                # Format outputs
                metric_defs = format_metric_definitions(selected_metrics, locale)
                
                cat_str = category.lower() if category else "other"
                suggested = suggest_additional_metrics(cat_str, selected_metrics)
                suggested_str = ", ".join(suggested) if suggested else "(AI Agent may suggest more in prompt)"
                
                metrics_used_str = ", ".join(selected_metrics)
                
                # Create JSON spec
                import uuid
                feature_id = str(uuid.uuid4())
                json_spec = json.dumps({
                    "feature_id": feature_id,
                    "feature_name": feature_name.strip(),
                    "category": category,
                    "metrics": selected_metrics,
                    "locale": locale,
                    "privacy_framework": get_privacy_framework(locale),
                    "generated_by": "ai_agent",
                    "complexity_reason": complexity_reason
                }, indent=2)
                
                # Generate code-based metrics sample
                code_metrics_sample = generate_code_metrics_sample(cat_str)
                
                return (
                    evaluation_prompt,
                    metric_defs,
                    suggested_str,
                    metrics_used_str,
                    json_spec,
                    code_metrics_sample,
                    f"✅ Generated with **AI Agent** ({complexity_reason})"
                )
            else:
                # AI agent failed, fall back to legacy
                logger.warning(f"AI Agent failed, falling back to legacy: {response.message}")
                # Continue to legacy path below
        
        # === LEGACY AGENT PATH ===
        # Build quality metrics
        quality_metrics = []
        for metric_id in selected_metrics:
            metric = get_metric(metric_id)
            if metric:
                quality_metrics.append(QualityMetric(
                    name=metric.name,
                    description=metric.get_definition(language),
                    weight=metric.weight,
                    is_primary=metric.is_primary,
                    rai_tags=metric.rai_tags
                ))
        
        # Build RAI constraints
        custom_constraints = [rai_additional_notes.strip()] if rai_additional_notes and rai_additional_notes.strip() else []
        rai = ResponsibleAIConstraints(
            no_pii_leakage=check_privacy,
            bias_check_required=check_fairness,
            toxicity_check_required=check_safety,
            cultural_sensitivity=check_transparency,
            safety_critical=check_safety,
            custom_constraints=custom_constraints
        )
        
        # Parse formats - InputOutputFormat uses TEXT_TO_TEXT style values
        io_format = InputOutputFormat.TEXT_TO_TEXT  # default
        
        # Parse category
        category_map = {
            "auto_reply": FeatureCategory.AUTO_REPLY,
            "summarization": FeatureCategory.SUMMARIZATION,
            "translation": FeatureCategory.TRANSLATION,
            "classification": FeatureCategory.CLASSIFICATION,
            "assistant": FeatureCategory.ASSISTANT,
            "content_generation": FeatureCategory.CONTENT_GENERATION,
            "generation": FeatureCategory.CONTENT_GENERATION,
            "extraction": FeatureCategory.OTHER,
            "generic": FeatureCategory.OTHER,
            "other": FeatureCategory.OTHER,
        }
        cat = category_map.get(category.lower(), FeatureCategory.OTHER) if category else FeatureCategory.OTHER
        
        # Build feature metadata
        metadata = FeatureMetadata(
            feature_name=feature_name.strip(),
            feature_description=feature_description.strip(),
            category=category.lower() if category else "other",
            io_format=io_format,
            input_description=f"Input format: {input_format}",
            output_description=f"Output format: {output_format}",
            quality_metrics=quality_metrics,
            success_metrics=selected_metrics,
            responsible_ai=rai,
            input_data_sample=typical_input_example.strip() if typical_input_example else "",
            domain_constraints=additional_context.strip() if additional_context else ""
        )
        
        # Save feature to database
        import uuid
        feature_id = str(uuid.uuid4())
        feature_store.upsert_feature(feature_id, metadata.model_dump())
        
        # Generate using agent
        agent = FeaturePromptWriterAgent()
        result = agent.generate(metadata, locale=locale)
        
        # Format outputs
        evaluation_prompt = result.evaluation_prompt
        
        # Metric definitions used
        metric_defs = format_metric_definitions(selected_metrics, locale)
        
        # Suggested additional metrics
        cat_str = category.lower() if category else "other"
        suggested = suggest_additional_metrics(cat_str, selected_metrics)
        suggested_str = ", ".join(suggested) if suggested else "(none - you have good coverage!)"
        
        # Final metrics used
        metrics_used_str = ", ".join(selected_metrics)
        
        # Create JSON spec
        json_spec = json.dumps({
            "feature_id": feature_id,
            "feature_name": metadata.feature_name,
            "category": metadata.category,
            "metrics": [m.name for m in metadata.quality_metrics],
            "rai_constraints": {
                "no_pii_leakage": rai.no_pii_leakage,
                "bias_check_required": rai.bias_check_required,
                "toxicity_check_required": rai.toxicity_check_required,
                "cultural_sensitivity": rai.cultural_sensitivity,
                "safety_critical": rai.safety_critical
            },
            "locale": locale,
            "privacy_framework": get_privacy_framework(locale),
            "grading_rubric": result.grading_rubric if hasattr(result, 'grading_rubric') else {},
            "rai_checks": result.rai_checks if hasattr(result, 'rai_checks') else []
        }, indent=2)
        
        # Log run
        run_store.log_run(
            feature_id=feature_id,
            template_id="",
            language=language,  # Legacy field - still using language for DB compatibility
            metrics=selected_metrics,
            output_prompt=evaluation_prompt,
            result={"json_spec": json_spec, "locale": locale}
        )
        
        # Generate code-based metrics sample
        cat_str = category.lower() if category else "other"
        code_metrics_sample = generate_code_metrics_sample(cat_str)
        
        # Build status message
        status_msg = "✅ Generated with **Template Agent** (deterministic)"
        if use_ai_agent == "auto" and AI_AGENT_AVAILABLE:
            status_msg += f" — Feature did not meet complexity threshold"
        elif not AI_AGENT_AVAILABLE:
            status_msg += " — AI Agent not available (install agent-framework)"
        
        return (
            evaluation_prompt,
            metric_defs,
            suggested_str,
            metrics_used_str,
            json_spec,
            code_metrics_sample,
            status_msg
        )
        
    except Exception as e:
        return "", "", "", "", "", "", f"❌ Error generating prompt: {str(e)}"


def update_metrics_on_category_change(category: str) -> List[str]:
    """Update selected metrics when category changes"""
    cat = category.lower().replace(" ", "_") if category else "generic"
    return DEFAULT_METRICS_BY_CATEGORY.get(cat, DEFAULT_METRICS_BY_CATEGORY.get("generic", []))


# ═══════════════════════════════════════════════════════════════════
# SIMULATION SCENARIOS
# ═══════════════════════════════════════════════════════════════════

SIMULATION_SCENARIOS = {
    "summarization": {
        "name": "📰 News Article Summarization",
        "description": "Generate and evaluate a summary of a news article. The AI will summarize the article, then the system will evaluate the summary using both LLM-based and code-based metrics.",
        "category": "summarization",
        "task_prompt": "Summarize the following news article in 2-3 sentences, capturing the key facts:",
        "input": """ORIGINAL ARTICLE:
Scientists at MIT have developed a new type of solar cell that can convert sunlight into electricity with 
unprecedented efficiency. The breakthrough, published in Nature Energy, shows the cells achieving 47.1% 
efficiency under concentrated sunlight - nearly double the efficiency of conventional silicon solar panels.

The new cells use a multi-junction design with layers of different semiconductor materials, each optimized 
to capture different wavelengths of light. Lead researcher Dr. Sarah Chen said the technology could 
revolutionize renewable energy production within the next decade.

"This is a game-changer for solar energy," Dr. Chen stated in a press conference. "While there are still 
manufacturing challenges to overcome, we believe commercial deployment could begin as early as 2028."

The research was funded by the Department of Energy and private investors totaling $15 million. Industry 
analysts predict the technology could reduce the cost of solar energy by up to 40% once mass production begins.""",
        "ai_output_good": """MIT researchers have developed solar cells achieving 47.1% efficiency under concentrated sunlight - 
nearly double conventional silicon panels. The multi-junction design uses layered semiconductors to capture 
different light wavelengths. Lead researcher Dr. Sarah Chen expects commercial deployment by 2028, 
potentially reducing solar energy costs by 40%.""",
        "ai_output_bad": """Scientists have created amazing new solar panels that are 100% efficient and will solve all 
our energy problems immediately. The technology was invented last week and is already being sold in stores. 
Dr. John Smith, who won a Nobel Prize for this work, says everyone should buy these panels today because 
they're practically free.""",
        "metrics": ["faithfulness", "coverage", "groundedness", "fluency", "brevity", "safety"],
    },
    "auto_reply": {
        "name": "📧 Professional Email Reply",
        "description": "Generate and evaluate a professional customer service email reply. The AI will compose a helpful response, then the system will evaluate it for tone, relevance, and helpfulness.",
        "category": "auto_reply",
        "task_prompt": "Write a professional, empathetic, and helpful customer service reply to the following email:",
        "input": """FROM: john.doe@customer.com
SUBJECT: Problem with my order

Hi,

I ordered a laptop 2 weeks ago (Order #98765) but it still hasn't arrived. The tracking shows it's been 
stuck in transit for 5 days. I need this for work and I'm getting very frustrated.

Can someone please help me figure out what's going on?

Thanks,
John""",
        "ai_output_good": """Dear John,

Thank you for reaching out, and I sincerely apologize for the frustration caused by this delay with your 
laptop order (#98765).

I've looked into your order and can see it has been delayed in transit. Here's what I'm doing to help:

1. I've opened an investigation with our shipping partner to locate your package
2. I've escalated this to our logistics team for priority handling
3. You'll receive an update within 24 hours with the resolution

If the package cannot be located within 48 hours, I'll arrange for a replacement to be shipped via 
express delivery at no extra cost.

Please feel free to reply to this email or call us at 1-800-XXX-XXXX if you have any questions.

Best regards,
Sarah
Customer Support Team""",
        "ai_output_bad": """John,

Yeah, shipping is slow sometimes. Not much we can do about it. You should have ordered earlier if you 
needed it for work. 

Check the tracking again in a few days. If it's still not there, maybe file a claim or something. 
We're pretty busy so I can't really look into every order that's a little late.

- Support""",
        "metrics": ["relevance", "tone", "fluency", "brevity", "safety", "privacy"],
    },
    "translation": {
        "name": "🌍 English to Chinese Translation",
        "description": "Generate and evaluate a Chinese translation. The AI will translate the English text, then the system will evaluate the translation quality.",
        "category": "translation",
        "task_prompt": "Translate the following English text into Chinese (Simplified), maintaining a professional tone:",
        "input": """Dear Customer,

Thank you for your recent purchase. Your order #12345 has been shipped and will arrive within 3-5 business days.

If you have any questions about your order, please don't hesitate to contact our support team at support@example.com or call us at 1-800-555-0123.

We appreciate your business and look forward to serving you again.

Best regards,
Customer Service Team""",
        "ai_output_good": """尊敬的客户，

感谢您最近的购买。您的订单 #12345 已发货，预计将在 3-5 个工作日内送达。

如果您对订单有任何疑问，请随时通过 support@example.com 联系我们的客服团队，或拨打 1-800-555-0123。

感谢您的惠顾，期待再次为您服务。

此致敬礼，
客户服务团队""",
        "ai_output_bad": """亲爱的顾客，

谢谢你买东西。你的订单号码一二三四五已经发出去了，可能三到五天到，也可能更久，不好说。

有问题的话，发邮件到 support@example.com 吧，或者打电话。

再见，
服务的人""",
        "metrics": ["faithfulness", "accuracy", "fluency", "cultural_appropriateness", "localization_quality"],
    },
}


def generate_ai_output_for_scenario(scenario_choice: str, custom_input: str) -> tuple:
    """Generate BOTH good and bad AI outputs for evaluation comparison"""
    logger.info(f"Generating AI outputs for scenario: {scenario_choice}")
    
    if not scenario_choice:
        logger.warning("No scenario selected for generation")
        return "❌ Please select a scenario first.", "", ""
    
    # REQUIRE input from user
    if not custom_input or not custom_input.strip():
        logger.warning("No input provided")
        return "❌ Please provide input/source content first before generating outputs.", "", ""
    
    # Extract scenario key
    key = scenario_choice.split("(")[-1].rstrip(")")
    if key not in SIMULATION_SCENARIOS:
        return "❌ Invalid scenario selected.", "", ""
    
    scenario = SIMULATION_SCENARIOS[key]
    input_text = custom_input.strip()
    task_prompt = scenario.get("task_prompt", "Process the following input:")
    category = scenario.get("category", "other")
    
    try:
        llm = LLMClient()
        
        # Generate GOOD output - with quality-focused prompt
        logger.info("Generating GOOD quality output...")
        good_system_prompt = get_good_output_system_prompt(category)
        good_output = llm.chat_completion(
            messages=[
                {"role": "system", "content": good_system_prompt},
                {"role": "user", "content": f"{task_prompt}\n\n{input_text}"}
            ],
            temperature=0.0,  # Low temperature for consistent output
            max_tokens=1000
        )
        
        # Generate BAD output - with intentional quality issues
        logger.info("Generating BAD quality output (for comparison)...")
        bad_system_prompt = get_bad_output_system_prompt(category)
        bad_output = llm.chat_completion(
            messages=[
                {"role": "system", "content": bad_system_prompt},
                {"role": "user", "content": f"{task_prompt}\n\n{input_text}"}
            ],
            temperature=0.0,  # Low temperature for consistent output
            max_tokens=1000
        )
        
        status = f"""✅ Generated BOTH outputs for comparison!
- Scenario: {scenario['name']}
- Good output: High-quality, accurate response
- Bad output: Intentionally flawed for evaluation testing

Now select which output to evaluate and click 'Evaluate Output'."""
        
        logger.info("Both outputs generated successfully")
        return status, good_output, bad_output
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return f"❌ Error generating outputs: {str(e)}", "", ""


def get_good_output_system_prompt(category: str) -> str:
    """Get system prompt for generating HIGH-QUALITY output"""
    base = """You are an expert AI assistant producing HIGH-QUALITY output. Follow these guidelines:
- Be accurate, factual, and well-structured
- Use appropriate tone and formatting
- Be concise but complete
- Follow best practices for this type of task"""
    
    category_additions = {
        "summarization": "\n- Capture all key points faithfully\n- Do not add information not in the source\n- Maintain logical flow\n- Use clear, professional language",
        "auto_reply": "\n- Be professional and courteous\n- Address all points raised\n- Use appropriate greeting and closing\n- Show empathy where appropriate",
        "translation": "\n- Translate accurately preserving meaning\n- Use natural, idiomatic language\n- Maintain tone and register of original\n- Preserve cultural nuances",
    }
    return base + category_additions.get(category, "")


def get_bad_output_system_prompt(category: str) -> str:
    """Get system prompt for generating INTENTIONALLY FLAWED output for testing"""
    base = """CRITICAL: You MUST generate a POOR QUALITY response for AI evaluation testing.

Your task is to DELIBERATELY produce flawed output. This is for testing evaluation systems.

YOU MUST INCLUDE AT LEAST 3 OF THESE FLAWS:
1. HALLUCINATE: Add 1-2 fake facts/details NOT in the source (e.g., wrong numbers, made-up names, fictional events)
2. MISS KEY INFO: Omit 1-2 important points from the source
3. BE VAGUE: Use generic filler phrases like "very important", "significant impact", "various reasons" without specifics
4. WRONG TONE: Use overly casual language ("Hey!", "BTW", "gonna") or robotic corporate speak
5. GRAMMAR ISSUES: Include 1-2 awkward sentence structures
6. FACTUAL ERROR: Get a detail slightly wrong (wrong date, wrong name, wrong number)

REMEMBER: This is intentionally bad output for testing. Do NOT produce high-quality output."""
    
    category_additions = {
        "summarization": """

FOR SUMMARIZATION - YOU MUST:
- Add at least ONE claim not in the original (hallucination)
- Miss at least ONE key fact
- Use vague phrases like "very significant" or "extremely important" without specifics
- Make it either too short (missing details) or too verbose (padding with fluff)""",
        
        "auto_reply": """

FOR EMAIL REPLY - YOU MUST:
- Use inappropriate tone (too casual like "Hey buddy!" or too robotic)
- Miss addressing at least ONE concern from the original email
- Include generic unhelpful phrases like "We value your feedback" without substance
- Sound like a template response, not personalized""",
        
        "translation": """

FOR TRANSLATION - YOU MUST:
- Use awkward literal translations that sound unnatural
- Get at least ONE phrase wrong or use wrong register (too formal/informal)
- Include at least ONE grammar error in the target language
- Miss cultural context or idioms""",
    }
    return base + category_additions.get(category, "")


def get_simulation_scenarios() -> List[str]:
    """Get list of simulation scenario names"""
    return [f"{v['name']} ({k})" for k, v in SIMULATION_SCENARIOS.items()]


def load_simulation_scenario(scenario_choice: str) -> tuple:
    """Load a simulation scenario"""
    logger.info(f"Loading scenario: {scenario_choice}")
    if not scenario_choice:
        logger.warning("No scenario selected")
        return "", "", "", "", []
    
    # Extract the key from the choice
    key = scenario_choice.split("(")[-1].rstrip(")")
    if key not in SIMULATION_SCENARIOS:
        logger.error(f"Invalid scenario key: {key}")
        return "", "", "", "", []
    
    scenario = SIMULATION_SCENARIOS[key]
    logger.info(f"Loaded scenario '{key}' with {len(scenario['metrics'])} metrics")
    logger.info(f"Good example length: {len(scenario['ai_output_good'])} chars")
    logger.info(f"Bad example length: {len(scenario['ai_output_bad'])} chars")
    return (
        scenario["description"],
        scenario["input"],
        scenario["ai_output_good"],
        scenario["ai_output_bad"],
        scenario["metrics"]
    )


def run_simulation(
    scenario_choice: str,
    input_text: str,
    ai_output: str,
    selected_metrics: List[str],
    use_good_output: bool,
    locale: str,
    generated_prompt: str = ""
) -> tuple:
    """Run evaluation on the provided AI output using the generated prompt from the Generate tab"""
    logger.info(f"Running evaluation for scenario: {scenario_choice}")
    logger.info(f"Selected metrics: {selected_metrics}")
    logger.info(f"Evaluating {'GOOD' if use_good_output else 'BAD'} output, Locale: {locale}")
    logger.info(f"Using generated prompt: {bool(generated_prompt)}")
    
    if not scenario_choice:
        return "❌ Please select a scenario first.", "", "", ""
    
    if not input_text or not input_text.strip():
        return "❌ Please provide input/source content first.", "", "", ""
    
    if not ai_output or not ai_output.strip():
        return "❌ No output to evaluate. Please generate outputs first by clicking 'Generate Good & Bad Outputs'.", "", "", ""
    
    if not generated_prompt or not generated_prompt.strip():
        return "❌ No evaluation prompt found. Please go to the Generate tab and generate a prompt first, then return here to run the simulation.", "", "", ""
    
    # Extract scenario key
    key = scenario_choice.split("(")[-1].rstrip(")")
    if key not in SIMULATION_SCENARIOS:
        return "❌ Invalid scenario selected.", "", "", ""
    
    scenario = SIMULATION_SCENARIOS[key]
    category = scenario.get("category", "other")
    
    output_to_evaluate = ai_output.strip()
    output_type = "Good Output" if use_good_output else "Bad Output"
    final_input = input_text.strip()
    
    # Use the generated prompt from the Generate tab, replacing placeholders with actual content
    # The generated prompt should be used as the base evaluation criteria
    evaluation_prompt = f"""## Generated Evaluation Prompt (from Generate tab):
{generated_prompt.strip()}

---

## Content to Evaluate:

### Input:
{final_input}

### AI Output ({output_type}):
{output_to_evaluate}

---

## Instructions:
Using the evaluation criteria defined in the Generated Evaluation Prompt above, evaluate the AI Output.

1. Score each metric on a 1-5 scale:
   - 1 = Very poor (fails completely)
   - 2 = Poor (major issues)
   - 3 = Acceptable (some issues)
   - 4 = Good (minor issues)
   - 5 = Excellent (meets all criteria)

2. Provide specific rationale for each score with evidence from the text
3. Flag any RAI (Responsible AI) concerns
4. Give an overall recommendation: PASS, FAIL, or NEEDS_REVIEW

## Output Format:
Provide your evaluation in the following JSON format:
```json
{{
  "scores": {{
    "<metric_name>": {{"score": <1-5>, "rationale": "<explanation>"}}
  }},
  "overall_score": <weighted_average>,
  "rai_concerns": ["<list any concerns>"],
  "recommendation": "PASS|FAIL|NEEDS_REVIEW",
  "summary": "<brief overall assessment>"
}}
```"""

    # Run LLM evaluation
    logger.info("Calling LLM for evaluation...")
    try:
        llm = LLMClient()
        evaluation_result = llm.chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert AI evaluation assistant. Provide thorough, fair, and evidence-based evaluations."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.0,  # Low temperature for consistent evaluation
            max_tokens=2000
        )
        logger.info("LLM evaluation complete")
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        evaluation_result = f"Error calling LLM: {str(e)}\n\nPlease check your Azure OpenAI configuration."
    
    # Run code-based metrics
    logger.info("Running code-based metrics...")
    code_metrics_results = run_code_metrics_simulation(category, final_input, output_to_evaluate)
    logger.info("Code metrics complete")
    
    status = f"✅ Evaluation complete!\n- Scenario: {scenario['name']}\n- Output Type: {output_type}\n- Metrics evaluated: {len(selected_metrics)}"
    
    return status, evaluation_prompt, evaluation_result, code_metrics_results


def run_code_metrics_simulation(category: str, input_text: str, output_text: str) -> str:
    """Run code-based metrics for simulation"""
    results = []
    results.append("# Code-Based Metrics Results\n")
    results.append("These metrics are computed programmatically using open-source libraries.\n")
    
    try:
        # Always compute readability
        import textstat
        results.append("## 📊 Readability Metrics (textstat)\n")
        results.append(f"- **Flesch Reading Ease**: {textstat.flesch_reading_ease(output_text):.1f}")
        results.append(f"  - 90-100: Very Easy | 60-70: Standard | 0-30: Very Difficult")
        results.append(f"- **Flesch-Kincaid Grade**: {textstat.flesch_kincaid_grade(output_text):.1f}")
        results.append(f"- **SMOG Index**: {textstat.smog_index(output_text):.1f}")
        results.append(f"- **Automated Readability Index**: {textstat.automated_readability_index(output_text):.1f}")
        results.append(f"- **Word Count**: {textstat.lexicon_count(output_text)}")
        results.append(f"- **Sentence Count**: {textstat.sentence_count(output_text)}")
        results.append("")
    except ImportError:
        results.append("⚠️ textstat not installed. Run: pip install textstat\n")
    except Exception as e:
        results.append(f"⚠️ Readability computation error: {str(e)}\n")
    
    # ROUGE for summarization
    if category == "summarization":
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(input_text, output_text)
            
            results.append("## 📊 ROUGE Scores (rouge-score)\n")
            results.append(f"- **ROUGE-1 F1**: {scores['rouge1'].fmeasure:.3f}")
            results.append(f"- **ROUGE-2 F1**: {scores['rouge2'].fmeasure:.3f}")
            results.append(f"- **ROUGE-L F1**: {scores['rougeL'].fmeasure:.3f}")
            results.append("  - Higher scores indicate better overlap with source")
            results.append("")
        except ImportError:
            results.append("⚠️ rouge-score not installed. Run: pip install rouge-score\n")
        except Exception as e:
            results.append(f"⚠️ ROUGE computation error: {str(e)}\n")
    
    # BLEU for translation
    if category == "translation":
        try:
            import sacrebleu
            # BLEU expects reference as list
            bleu = sacrebleu.sentence_bleu(output_text, [input_text])
            
            results.append("## 📊 BLEU Score (sacrebleu)\n")
            results.append(f"- **BLEU Score**: {bleu.score:.2f}")
            results.append("  - Higher scores indicate better translation quality")
            results.append(f"- **Brevity Penalty**: {bleu.bp:.3f}")
            results.append("")
        except ImportError:
            results.append("⚠️ sacrebleu not installed. Run: pip install sacrebleu\n")
        except Exception as e:
            results.append(f"⚠️ BLEU computation error: {str(e)}\n")
    
    # Fuzzy string matching
    try:
        from rapidfuzz import fuzz
        results.append("## 📊 Fuzzy Matching (rapidfuzz)\n")
        results.append(f"- **Ratio**: {fuzz.ratio(input_text[:500], output_text[:500]):.1f}%")
        results.append(f"- **Partial Ratio**: {fuzz.partial_ratio(input_text[:500], output_text[:500]):.1f}%")
        results.append(f"- **Token Sort Ratio**: {fuzz.token_sort_ratio(input_text[:500], output_text[:500]):.1f}%")
        results.append("  - Higher ratios indicate more similarity")
        results.append("")
    except ImportError:
        results.append("⚠️ rapidfuzz not installed. Run: pip install rapidfuzz\n")
    except Exception as e:
        results.append(f"⚠️ Fuzzy matching error: {str(e)}\n")
    
    return "\n".join(results)


def create_app() -> gr.Blocks:
    """Create the Gradio application"""
    
    with gr.Blocks(title="MetaFeature Orchestrator") as app:
        gr.Markdown("""
        # 🎯 MetaFeature Orchestrator
        
        **Automatically generate high-quality evaluation prompts for your AI features.**
        
        Design Principles:
        - **Metric-first**: Define evaluation criteria before generation
        - **Grounded**: No hallucinated judgments - clear rubrics and thresholds
        - **RAI by Design**: Responsible AI checks built-in
        - **Human-reviewable**: All outputs are transparent and auditable
        """)
        
        with gr.Tabs():
            # Tab 0: Feature Definition (merged with Quick Start)
            with gr.Tab("📝 Feature Definition"):
                gr.Markdown("### Quick Start: Select from Templates")
                gr.Markdown("Choose a feature template to auto-fill the form, or enter your own details below.")
                
                with gr.Row():
                    group_dd = gr.Dropdown(
                        label="Feature Group",
                        choices=list_groups(),
                        value="Auto Reply"
                    )
                    feature_dd = gr.Dropdown(
                        label="Feature Template",
                        choices=list_features("Auto Reply"),
                        value="Auto-Reply Email"
                    )
                
                with gr.Row():
                    load_template_btn = gr.Button("📥 Load Template", variant="secondary")
                    template_status = gr.Markdown("")
                
                with gr.Accordion("➕ Add Custom Feature Template", open=False):
                    with gr.Row():
                        new_group = gr.Textbox(label="Group Name", placeholder="e.g., Accessibility AI", value="Custom")
                        new_name = gr.Textbox(label="Feature Name", placeholder="e.g., Auto Rewrite Notes")
                    
                    new_description = gr.Textbox(label="Description", lines=2)
                    
                    with gr.Row():
                        new_category = gr.Dropdown(
                            label="Category",
                            choices=["summarization", "auto_reply", "translation", "classification", "extraction", "generation", "image_generation", "image_editing", "image_understanding", "image_safety", "personal_assistant", "generic"],
                            value="generic"
                        )
                        new_input_format = gr.Dropdown(
                            label="Input Format",
                            choices=["text", "json", "markdown", "code", "html", "xml", "image", "audio", "video"],
                            value="text"
                        )
                        new_output_format = gr.Dropdown(
                            label="Output Format",
                            choices=["text", "json", "markdown", "code", "html", "xml", "image", "audio", "video"],
                            value="text"
                        )
                    
                    with gr.Row():
                        new_typical_input = gr.Textbox(label="Typical Input Example", lines=2)
                        new_expected_output = gr.Textbox(label="Expected Output Example", lines=2)
                    
                    with gr.Row():
                        new_privacy = gr.Checkbox(label="Privacy Sensitive", value=True)
                        new_safety = gr.Checkbox(label="Safety Critical", value=False)
                    
                    add_feature_btn = gr.Button("➕ Add Feature Template", variant="secondary")
                    add_status = gr.Markdown("")
                
                gr.Markdown("---")
                gr.Markdown("### Feature Details")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feature_name = gr.Textbox(
                            label="Feature Name",
                            placeholder="e.g., Email Auto-Reply Suggester",
                            lines=1
                        )
                        
                        feature_description = gr.Textbox(
                            label="Feature Description",
                            placeholder="Describe what your feature does...",
                            lines=3
                        )
                        
                        category = gr.Dropdown(
                            label="Feature Category",
                            choices=[
                                "auto_reply", "summarization", "translation",
                                "classification", "extraction", "generation",
                                "image_generation", "image_editing", "image_understanding", "image_safety",
                                "personal_assistant", "generic"
                            ],
                            value="generic"
                        )
                        
                        with gr.Row():
                            input_format = gr.Dropdown(
                                label="Input Format",
                                choices=["text", "json", "markdown", "code", "html", "xml", "image", "audio", "video"],
                                value="text"
                            )
                            output_format = gr.Dropdown(
                                label="Output Format",
                                choices=["text", "json", "markdown", "code", "html", "xml", "image", "audio", "video"],
                                value="text"
                            )
                    
                    with gr.Column(scale=1):
                        typical_input = gr.Textbox(
                            label="Typical Input Example",
                            placeholder="Provide an example input...",
                            lines=4
                        )
                        
                        expected_output = gr.Textbox(
                            label="Expected Output Example",
                            placeholder="Provide the expected output for the input above...",
                            lines=4
                        )
                        
                        additional_context = gr.Textbox(
                            label="Additional Context (Optional)",
                            placeholder="Any additional context, constraints, or notes...",
                            lines=2
                        )
            
            # Tab 2: Metrics Selection
            with gr.Tab("📊 Quality Metrics"):
                gr.Markdown("### Select Evaluation Metrics")
                gr.Markdown("Choose the metrics that matter most for evaluating your feature's output quality.")
                
                suggested_metrics_display = gr.Markdown(
                    value="Select a category to see suggested metrics.",
                    label="Suggested Metrics"
                )
                
                with gr.Row():
                    select_all_btn = gr.Button("✅ Select All", variant="secondary", size="sm")
                    unselect_all_btn = gr.Button("⬜ Unselect All", variant="secondary", size="sm")
                    select_recommended_btn = gr.Button("⭐ Select Recommended", variant="secondary", size="sm")
                
                selected_metrics = gr.CheckboxGroup(
                    label="Selected Metrics",
                    choices=get_all_metrics_choices(),
                    value=["relevance", "fluency", "safety"]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🌍 Locale Settings")
                gr.Markdown("*Select the target locale for culture-aware evaluation with region-specific RAI checks.*")
                
                with gr.Row():
                    locale_dropdown = gr.Dropdown(
                        label="Target Locale",
                        choices=[
                            # English variants
                            ("🇺🇸 English (United States)", "en-US"),
                            ("🇬🇧 English (United Kingdom)", "en-GB"),
                            ("🇦🇺 English (Australia)", "en-AU"),
                            ("🇮🇳 English (India)", "en-IN"),
                            ("🇸🇬 English (Singapore)", "en-SG"),
                            ("🇨🇦 English (Canada)", "en-CA"),
                            # Chinese variants
                            ("🇨🇳 中文 (中国大陆)", "zh-CN"),
                            ("🇹🇼 中文 (台灣)", "zh-TW"),
                            ("🇭🇰 中文 (香港)", "zh-HK"),
                            # Spanish variants
                            ("🇪🇸 Español (España)", "es-ES"),
                            ("🇲🇽 Español (México)", "es-MX"),
                            ("🇦🇷 Español (Argentina)", "es-AR"),
                            # Portuguese variants
                            ("🇧🇷 Português (Brasil)", "pt-BR"),
                            ("🇵🇹 Português (Portugal)", "pt-PT"),
                            # Other languages
                            ("🇯🇵 日本語 (日本)", "ja-JP"),
                            ("🇰🇷 한국어 (대한민국)", "ko-KR"),
                            ("🇩🇪 Deutsch (Deutschland)", "de-DE"),
                            ("🇫🇷 Français (France)", "fr-FR"),
                            ("🇨🇦 Français (Canada)", "fr-CA"),
                        ],
                        value="en-US",
                        info="Locale affects tone, formality, and privacy compliance (e.g., GDPR, CCPA, PIPL)"
                    )
                    
                    locale_info = gr.Markdown(
                        """**en-US**: Casual tone, direct feedback, CCPA privacy framework""",
                        elem_id="locale-info"
                    )
            
            # Tab 3: RAI Settings
            with gr.Tab("🛡️ Responsible AI"):
                gr.Markdown("### Responsible AI Constraints")
                gr.Markdown("Enable checks to ensure your feature meets responsible AI standards.")
                
                with gr.Row():
                    with gr.Column():
                        check_safety = gr.Checkbox(
                            label="Safety Check",
                            value=True,
                            info="Check for harmful, violent, or dangerous content"
                        )
                        check_privacy = gr.Checkbox(
                            label="Privacy Check",
                            value=True,
                            info="Check for PII exposure or privacy violations"
                        )
                    with gr.Column():
                        check_fairness = gr.Checkbox(
                            label="Fairness Check",
                            value=True,
                            info="Check for bias and ensure equitable treatment"
                        )
                        check_transparency = gr.Checkbox(
                            label="Transparency Check",
                            value=True,
                            info="Ensure outputs are explainable and traceable"
                        )
                
                rai_notes = gr.Textbox(
                    label="Additional RAI Notes",
                    placeholder="Any specific RAI concerns or requirements...",
                    lines=2
                )
            
            # Tab 4: Generate & Review
            with gr.Tab("🚀 Generate"):
                gr.Markdown("### Generate Evaluation Prompt")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        generate_btn = gr.Button("🎯 Generate Evaluation Prompt", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        ai_agent_mode = gr.Radio(
                            choices=[
                                ("🤖 Auto (detect complexity)", "auto"),
                                ("⚡ Always use AI Agent", "always"),
                                ("📋 Template only", "never")
                            ],
                            value="auto",
                            label="Generation Mode",
                            info="AI Agent provides better results for complex/novel features" if AI_AGENT_AVAILABLE else "AI Agent not available - install agent-framework",
                            interactive=AI_AGENT_AVAILABLE
                        )
                
                status_message = gr.Markdown("")
                
                with gr.Tabs():
                    with gr.Tab("📝 Prompt-Based Evaluation"):
                        with gr.Row():
                            with gr.Column():
                                output_prompt = gr.Textbox(
                                    label="Generated Evaluation Prompt",
                                    lines=18
                                )
                                
                                with gr.Accordion("📊 Metrics Details", open=False):
                                    metric_defs_output = gr.Markdown(label="Metric Definitions Used")
                                    suggested_metrics_output = gr.Textbox(label="Suggested Additional Metrics", lines=2)
                                    used_metrics_output = gr.Textbox(label="Final Metrics Used", lines=1)
                            
                            with gr.Column():
                                output_json = gr.Code(
                                    label="Feature Specification (JSON)",
                                    language="json",
                                    lines=20
                                )
                    
                    with gr.Tab("💻 Code-Based Metrics"):
                        gr.Markdown("""
                        **Code-based metrics** provide deterministic evaluation using well-known open source packages.
                        
                        These complement prompt-based evaluation with measurable, programmatic scores.
                        """)
                        
                        code_metrics_output = gr.Code(
                            label="Sample Code for Metric Calculation",
                            language="python",
                            lines=30
                        )
            
            # Tab 5: Simulation (linked to Feature Definition)
            with gr.Tab("🧪 Simulation") as sim_tab:
                # State to store the generated evaluation prompt
                sim_stored_prompt = gr.State(value="")
                
                # Notice when category doesn't support simulation
                sim_category_notice = gr.Markdown("""
                ### ⚠️ Simulation Not Available for This Category
                
                **Simulation is only available for these feature categories:**
                - 📰 **Summarization** (Summarize News, Summarize Email Thread, Summarize Document)
                - 📧 **Auto Reply** (Auto-Reply Email, Auto-Reply Message)
                - 🌍 **Translation** (Translate Document)
                
                👉 **Go to Feature Definition tab** and select a feature from one of these groups.
                """, visible=True)
                
                # Notice when prompt hasn't been generated yet
                sim_prompt_notice = gr.Markdown("""
                ### ⚠️ Generate Evaluation Prompt First
                
                **You need to generate an evaluation prompt before running simulation.**
                
                👉 **Go to the Generate tab** and click "Generate Evaluation Prompt" button, then return here.
                
                The simulation will use your generated prompt to evaluate the AI outputs.
                """, visible=False)
                
                with gr.Column(visible=False) as sim_content:
                    gr.Markdown("""
                    ### 🎯 Test Your Feature End-to-End
                    
                    **Workflow:**
                    1. **Enter your input** - Provide the source content (REQUIRED)
                    2. **Generate AI Outputs** - Creates BOTH good and bad examples for comparison
                    3. **Select which to evaluate** - Choose Good or Bad output
                    4. **Evaluate** - Uses your generated prompt from the Generate tab
                    """)
                    
                    # Hidden field to store the scenario key (linked from Feature Definition)
                    sim_scenario = gr.Textbox(visible=False, value="")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            sim_linked_info = gr.Markdown("**📋 Scenario:** Not selected")
                            
                            sim_description = gr.Textbox(
                                label="📝 Scenario Description",
                                lines=2,
                                interactive=False
                            )
                            
                            sim_input = gr.Textbox(
                                label="📥 Input / Source Content (REQUIRED)",
                                lines=8,
                                info="Enter the content for the AI to process",
                                placeholder="Enter your input here before generating outputs..."
                            )
                            
                            sim_metrics = gr.CheckboxGroup(
                                label="📊 Metrics to Evaluate",
                                choices=get_all_metrics_choices(),
                                value=["faithfulness", "fluency", "safety"]
                            )
                            
                            sim_locale = gr.Dropdown(
                                label="🌐 Evaluation Locale",
                                choices=[
                                    ("🇺🇸 English (US)", "en-US"), 
                                    ("🇬🇧 English (UK)", "en-GB"),
                                    ("🇨🇳 中文 (中国)", "zh-CN"),
                                    ("🇹🇼 中文 (台灣)", "zh-TW"),
                                    ("🇪🇸 Español (España)", "es-ES"),
                                    ("🇲🇽 Español (México)", "es-MX"),
                                    ("🇯🇵 日本語", "ja-JP"),
                                ],
                                value="en-US",
                                info="Locale affects tone, cultural context, and compliance"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎮 Actions")
                            
                            sim_generate_btn = gr.Button("🤖 Generate Good & Bad Outputs", variant="primary", size="lg")
                            
                            sim_status = gr.Markdown("")
                            
                            gr.Markdown("---")
                            gr.Markdown("### 📤 Generated Outputs")
                            
                            sim_good_output = gr.Textbox(
                                label="✅ Good Output (AI Generated)",
                                lines=6,
                                interactive=False,
                                info="High-quality output for evaluation"
                            )
                            
                            sim_bad_output = gr.Textbox(
                                label="❌ Bad Output (AI Generated)", 
                                lines=6,
                                interactive=False,
                                info="Intentionally flawed output for comparison"
                            )
                            
                            gr.Markdown("---")
                            
                            sim_use_good = gr.Radio(
                                label="Select output to evaluate:",
                                choices=[("✅ Evaluate GOOD Output", True), ("❌ Evaluate BAD Output", False)],
                                value=True,
                                info="Choose which output to run evaluation on"
                            )
                            
                            sim_run_btn = gr.Button("📊 Evaluate Selected Output", variant="secondary", size="lg")
                    
                    gr.Markdown("---")
                    
                    with gr.Tabs():
                        with gr.Tab("📝 Generated Prompt (from Generate Tab)"):
                            sim_generated_prompt_display = gr.Textbox(
                                label="✅ Evaluation Prompt (Generated in Generate Tab)",
                                lines=12,
                                interactive=False,
                                info="This prompt will be used to evaluate the AI outputs"
                            )
                        
                        with gr.Tab("📋 Full Evaluation Prompt Sent to LLM"):
                            sim_prompt_output = gr.Textbox(
                                label="Complete prompt sent to LLM (includes generated prompt + content to evaluate)",
                                lines=15
                            )
                        
                        with gr.Tab("🤖 LLM Evaluation Results"):
                            sim_llm_output = gr.Markdown(
                                label="LLM Evaluation Output"
                            )
                        
                        with gr.Tab("📊 Code Metrics Results"):
                            sim_code_output = gr.Markdown(
                                label="Code-Based Metrics"
                            )
        
        # --- Event handlers ---
        
        # Quick Start tab: group/feature selection
        group_dd.change(
            fn=update_feature_choices,
            inputs=[group_dd],
            outputs=[feature_dd]
        )
        
        # Load template button
        load_template_btn.click(
            fn=load_feature_template,
            inputs=[group_dd, feature_dd],
            outputs=[feature_name, feature_description, category, input_format, output_format, typical_input, expected_output, selected_metrics, template_status]
        )
        
        # Add custom feature
        add_feature_btn.click(
            fn=add_custom_feature,
            inputs=[new_group, new_name, new_description, new_category, new_input_format, new_output_format, new_typical_input, new_expected_output, new_privacy, new_safety],
            outputs=[add_status, group_dd, feature_dd]
        )
        
        # Category change updates suggested metrics
        category.change(
            fn=lambda c: get_suggested_metrics(c),
            inputs=[category],
            outputs=[suggested_metrics_display]
        )
        
        category.change(
            fn=update_metrics_on_category_change,
            inputs=[category],
            outputs=[selected_metrics]
        )
        
        # Metrics selection buttons
        select_all_btn.click(
            fn=lambda: get_all_metrics_choices(),
            inputs=[],
            outputs=[selected_metrics]
        )
        
        unselect_all_btn.click(
            fn=lambda: [],
            inputs=[],
            outputs=[selected_metrics]
        )
        
        select_recommended_btn.click(
            fn=update_metrics_on_category_change,
            inputs=[category],
            outputs=[selected_metrics]
        )
        
        # Locale change updates info display
        locale_dropdown.change(
            fn=get_locale_info_text,
            inputs=[locale_dropdown],
            outputs=[locale_info]
        )
        
        # Generate button - also update the simulation tab's generated prompt display and enable simulation
        def generate_and_sync_to_simulation(
            feature_name_val, feature_description_val, category_val,
            input_format_val, output_format_val,
            typical_input_val, expected_output_val,
            selected_metrics_val, additional_context_val,
            check_safety_val, check_privacy_val, check_fairness_val, check_transparency_val,
            rai_notes_val, locale_val, ai_agent_mode_val
        ):
            """Generate prompt and return extra copy for simulation tab, also update visibility"""
            result = generate_prompt(
                feature_name_val, feature_description_val, category_val,
                input_format_val, output_format_val,
                typical_input_val, expected_output_val,
                selected_metrics_val, additional_context_val,
                check_safety_val, check_privacy_val, check_fairness_val, check_transparency_val,
                rai_notes_val, locale_val, ai_agent_mode_val
            )
            # result is (evaluation_prompt, metric_defs, suggested_str, metrics_used_str, json_spec, code_metrics_sample, status_message)
            evaluation_prompt = result[0]
            
            # Check if category supports simulation
            cat = category_val.lower() if category_val else ""
            category_supports_sim = cat in SIMULATION_SUPPORTED_CATEGORIES
            
            # If prompt was generated successfully and category supports simulation
            if evaluation_prompt and evaluation_prompt.strip() and category_supports_sim:
                # Show simulation content, hide both notices
                return result + (
                    evaluation_prompt,           # sim_generated_prompt_display
                    gr.update(visible=False),    # sim_category_notice - hide
                    gr.update(visible=False),    # sim_prompt_notice - hide
                    gr.update(visible=True),     # sim_content - show
                )
            elif evaluation_prompt and evaluation_prompt.strip() and not category_supports_sim:
                # Prompt generated but category doesn't support simulation
                return result + (
                    evaluation_prompt,           # sim_generated_prompt_display
                    gr.update(visible=True),     # sim_category_notice - show
                    gr.update(visible=False),    # sim_prompt_notice - hide
                    gr.update(visible=False),    # sim_content - hide
                )
            else:
                # No prompt generated
                return result + (
                    "",                          # sim_generated_prompt_display
                    gr.update(),                 # sim_category_notice - no change
                    gr.update(),                 # sim_prompt_notice - no change
                    gr.update(),                 # sim_content - no change
                )
        
        generate_btn.click(
            fn=generate_and_sync_to_simulation,
            inputs=[
                feature_name, feature_description, category,
                input_format, output_format,
                typical_input, expected_output,
                selected_metrics, additional_context,
                check_safety, check_privacy, check_fairness, check_transparency,
                rai_notes, locale_dropdown, ai_agent_mode
            ],
            outputs=[output_prompt, metric_defs_output, suggested_metrics_output, used_metrics_output, output_json, code_metrics_output, status_message, sim_generated_prompt_display, sim_category_notice, sim_prompt_notice, sim_content]
        )
        
        # --- Simulation Tab Event Handlers ---
        
        # Supported simulation categories
        SIMULATION_SUPPORTED_CATEGORIES = ["summarization", "auto_reply", "translation"]
        
        # Category to scenario mapping
        CATEGORY_TO_SCENARIO = {
            "summarization": "summarization",
            "auto_reply": "auto_reply",
            "translation": "translation"
        }
        
        def update_simulation_for_category(cat: str, current_prompt: str) -> tuple:
            """Update simulation tab based on selected category and whether prompt exists"""
            logger.info(f"Category changed to: {cat}")
            has_prompt = bool(current_prompt and current_prompt.strip())
            
            if cat in SIMULATION_SUPPORTED_CATEGORIES:
                # Category supports simulation
                scenario_key = CATEGORY_TO_SCENARIO[cat]
                scenario = SIMULATION_SCENARIOS.get(scenario_key, {})
                scenario_name = scenario.get("name", scenario_key)
                description = scenario.get("description", "")
                metrics = scenario.get("metrics", ["faithfulness", "fluency", "safety"])
                
                # Format the full scenario choice string
                scenario_choice = f"{scenario_name} ({scenario_key})"
                info_text = f"**📋 Linked Scenario:** {scenario_name}"
                
                logger.info(f"Simulation enabled for category: {cat} → scenario: {scenario_key}, has_prompt: {has_prompt}")
                
                if has_prompt:
                    # Category OK and prompt exists - show simulation content
                    return (
                        gr.update(visible=False),  # sim_category_notice - hide
                        gr.update(visible=False),  # sim_prompt_notice - hide
                        gr.update(visible=True),   # sim_content - show
                        scenario_choice,           # sim_scenario (hidden)
                        info_text,                 # sim_linked_info
                        description,               # sim_description
                        metrics,                   # sim_metrics
                        "",                        # sim_input (clear)
                        "",                        # sim_good_output (clear)
                        "",                        # sim_bad_output (clear)
                        "",                        # sim_status (clear)
                    )
                else:
                    # Category OK but no prompt yet - show prompt notice
                    return (
                        gr.update(visible=False),  # sim_category_notice - hide
                        gr.update(visible=True),   # sim_prompt_notice - show
                        gr.update(visible=False),  # sim_content - hide
                        scenario_choice,           # sim_scenario (hidden)
                        info_text,                 # sim_linked_info
                        description,               # sim_description
                        metrics,                   # sim_metrics
                        "",                        # sim_input (clear)
                        "",                        # sim_good_output (clear)
                        "",                        # sim_bad_output (clear)
                        "",                        # sim_status (clear)
                    )
            else:
                # Category doesn't support simulation - show category notice
                logger.info(f"Simulation not available for category: {cat}")
                return (
                    gr.update(visible=True),   # sim_category_notice - show
                    gr.update(visible=False),  # sim_prompt_notice - hide
                    gr.update(visible=False),  # sim_content - hide
                    "",                        # sim_scenario
                    "**📋 Scenario:** Not available for this category",
                    "",                        # sim_description
                    [],                        # sim_metrics
                    "",                        # sim_input
                    "",                        # sim_good_output
                    "",                        # sim_bad_output
                    "",                        # sim_status
                )
        
        # Link category dropdown to simulation tab
        category.change(
            fn=update_simulation_for_category,
            inputs=[category, output_prompt],
            outputs=[
                sim_category_notice, sim_prompt_notice, sim_content, sim_scenario, sim_linked_info,
                sim_description, sim_metrics, sim_input,
                sim_good_output, sim_bad_output, sim_status
            ]
        )
        
        # Generate BOTH good and bad AI outputs
        sim_generate_btn.click(
            fn=generate_ai_output_for_scenario,
            inputs=[sim_scenario, sim_input],
            outputs=[sim_status, sim_good_output, sim_bad_output]
        )
        
        # Run simulation/evaluation on selected output
        def run_evaluation_on_selected(
            scenario_choice: str,
            input_text: str,
            good_output: str,
            bad_output: str,
            selected_metrics: List[str],
            use_good: bool,
            locale: str,
            generated_prompt: str
        ) -> tuple:
            """Run evaluation on the selected output (good or bad) using the generated prompt"""
            output_to_evaluate = good_output if use_good else bad_output
            return run_simulation(scenario_choice, input_text, output_to_evaluate, selected_metrics, use_good, locale, generated_prompt)
        
        sim_run_btn.click(
            fn=run_evaluation_on_selected,
            inputs=[sim_scenario, sim_input, sim_good_output, sim_bad_output, sim_metrics, sim_use_good, sim_locale, sim_generated_prompt_display],
            outputs=[sim_status, sim_prompt_output, sim_llm_output, sim_code_output]
        )
    
    return app


def main():
    """Launch the application"""
    logger.info("Starting MetaFeature Orchestrator...")
    app = create_app()
    logger.info("App created, launching on http://127.0.0.1:7860")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()

