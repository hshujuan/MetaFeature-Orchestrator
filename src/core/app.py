"""
MetaFeature Orchestrator - Gradio Application
Automatically generates high-quality evaluation prompts with metric-first design.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

import gradio as gr

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


# Location to language mapping
LOCATION_TO_LANGUAGE = {
    "US": "en",
    "GB": "en",
    "CN": "zh-Hans",
    "TW": "zh-Hant",
    "HK": "zh-Hant",
    "JP": "ja",
    "KR": "ko",
    "DE": "de",
    "FR": "fr",
    "ES": "es",
    "MX": "es",
    "BR": "pt",
    "IN": "en",  # India - English is common for tech
    "SA": "ar",
    "AE": "ar",
    "ID": "id",
    "VN": "vi",
    "TH": "th",
    "IT": "it",
    "NL": "nl",
    "RU": "ru",
    "PL": "pl",
    "TR": "tr",
    "AU": "en",
    "CA": "en",  # Canada - default to English
    "GLOBAL": "en",
}


def get_language_for_location(location: str) -> str:
    """Get the primary language for a location"""
    return LOCATION_TO_LANGUAGE.get(location, "en")


def format_metric_definitions(metric_ids: List[str], language: str = "en") -> str:
    """Format metric definitions for display"""
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
    language: str = "en"
) -> tuple[str, str, str, str, str, str]:
    """
    Generate an evaluation prompt using the agent.
    
    Returns:
        Tuple of (evaluation_prompt, metric_definitions, suggested_metrics, 
                  metrics_used, json_spec, status_message)
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
    language = language or "en"
    
    if not feature_name.strip():
        return "", "", "", "", "", "", "❌ Please provide a feature name"
    
    if not feature_description.strip():
        return "", "", "", "", "", "", "❌ Please provide a feature description"
    
    if not selected_metrics:
        return "", "", "", "", "", "", "❌ Please select at least one quality metric"
    
    try:
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
        result = agent.generate(metadata, language=language)
        
        # Format outputs
        evaluation_prompt = result.evaluation_prompt
        
        # Metric definitions used
        metric_defs = format_metric_definitions(selected_metrics, language)
        
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
            "grading_rubric": result.grading_rubric if hasattr(result, 'grading_rubric') else {},
            "rai_checks": result.rai_checks if hasattr(result, 'rai_checks') else []
        }, indent=2)
        
        # Log run
        run_store.log_run(
            feature_id=feature_id,
            template_id="",
            language=language,
            metrics=selected_metrics,
            output_prompt=evaluation_prompt,
            result={"json_spec": json_spec}
        )
        
        # Generate code-based metrics sample
        cat_str = category.lower() if category else "other"
        code_metrics_sample = generate_code_metrics_sample(cat_str)
        
        return (
            evaluation_prompt,
            metric_defs,
            suggested_str,
            metrics_used_str,
            json_spec,
            code_metrics_sample,
            "✅ Evaluation prompt generated successfully!"
        )
        
    except Exception as e:
        return "", "", "", "", "", "", f"❌ Error generating prompt: {str(e)}"


def update_metrics_on_category_change(category: str) -> List[str]:
    """Update selected metrics when category changes"""
    cat = category.lower().replace(" ", "_") if category else "generic"
    return DEFAULT_METRICS_BY_CATEGORY.get(cat, DEFAULT_METRICS_BY_CATEGORY.get("generic", []))


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
                            choices=["summarization", "auto_reply", "translation", "classification", "extraction", "generation", "image_generation", "image_editing", "image_understanding", "image_safety", "generic"],
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
                                "generic"
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
                gr.Markdown("### 🌍 Localization Settings")
                
                with gr.Row():
                    location = gr.Dropdown(
                        label="Target Location/Region",
                        choices=[
                            ("🇺🇸 United States", "US"),
                            ("🇬🇧 United Kingdom", "GB"),
                            ("🇨🇳 China (Mainland)", "CN"),
                            ("🇹🇼 Taiwan", "TW"),
                            ("🇭🇰 Hong Kong", "HK"),
                            ("🇯🇵 Japan", "JP"),
                            ("🇰🇷 South Korea", "KR"),
                            ("🇩🇪 Germany", "DE"),
                            ("🇫🇷 France", "FR"),
                            ("🇪🇸 Spain", "ES"),
                            ("🇲🇽 Mexico", "MX"),
                            ("🇧🇷 Brazil", "BR"),
                            ("🇮🇳 India", "IN"),
                            ("🇸🇦 Saudi Arabia", "SA"),
                            ("🇦🇪 United Arab Emirates", "AE"),
                            ("🇮🇩 Indonesia", "ID"),
                            ("🇻🇳 Vietnam", "VN"),
                            ("🇹🇭 Thailand", "TH"),
                            ("🇮🇹 Italy", "IT"),
                            ("🇳🇱 Netherlands", "NL"),
                            ("🇷🇺 Russia", "RU"),
                            ("🇵🇱 Poland", "PL"),
                            ("🇹🇷 Turkey", "TR"),
                            ("🇦🇺 Australia", "AU"),
                            ("🇨🇦 Canada", "CA"),
                            ("🌐 Global/International", "GLOBAL"),
                        ],
                        value="US",
                        info="Select the target region for culture-aware evaluation"
                    )
                    
                    language = gr.Dropdown(
                        label="Evaluation Language",
                        choices=[
                            ("English", "en"),
                            ("Chinese (Simplified)", "zh-Hans"),
                            ("Chinese (Traditional)", "zh-Hant"),
                            ("Japanese", "ja"),
                            ("Korean", "ko"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Portuguese", "pt"),
                            ("Italian", "it"),
                            ("Russian", "ru"),
                            ("Arabic", "ar"),
                            ("Hindi", "hi"),
                            ("Indonesian", "id"),
                            ("Vietnamese", "vi"),
                            ("Thai", "th"),
                            ("Dutch", "nl"),
                            ("Polish", "pl"),
                            ("Turkish", "tr"),
                        ],
                        value="en",
                        info="Auto-filled based on location, but you can change it"
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
                
                generate_btn = gr.Button("🎯 Generate Evaluation Prompt", variant="primary", size="lg")
                
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
                        **Code-based metrics** provide deterministic, reproducible evaluation using well-known open source packages.
                        
                        These complement prompt-based evaluation with measurable, programmatic scores.
                        """)
                        
                        code_metrics_output = gr.Code(
                            label="Sample Code for Metric Calculation",
                            language="python",
                            lines=30
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
        
        # Location change updates language
        location.change(
            fn=get_language_for_location,
            inputs=[location],
            outputs=[language]
        )
        
        # Generate button
        generate_btn.click(
            fn=generate_prompt,
            inputs=[
                feature_name, feature_description, category,
                input_format, output_format,
                typical_input, expected_output,
                selected_metrics, additional_context,
                check_safety, check_privacy, check_fairness, check_transparency,
                rai_notes, language
            ],
            outputs=[output_prompt, metric_defs_output, suggested_metrics_output, used_metrics_output, output_json, code_metrics_output, status_message]
        )
    
    return app


def main():
    """Launch the application"""
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
