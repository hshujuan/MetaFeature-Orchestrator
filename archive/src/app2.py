from __future__ import annotations

import gradio as gr

from agent import FeatureMetadata, FeaturePromptWriterAgent


# --- Predefined grouped features (you can expand freely) ---
GROUPED_FEATURES = {
    "Summarization": {
        "Summarize News": FeatureMetadata(
            group="Summarization",
            name="Summarize News",
            description="Summarize a news article into a short, accurate summary for the user.",
            category="summarization",
            input_format="news article text",
            output_format="short summary text",
            languages_supported=["en"],
            success_metrics=[],  # allow agent to suggest
            privacy_sensitive=False,
            safety_critical=True,
        ),
        "Summarize Email Thread": FeatureMetadata(
            group="Summarization",
            name="Summarize Email Thread",
            description="Summarize an email thread and highlight action items.",
            category="summarization",
            input_format="email thread text",
            output_format="summary + action items",
            languages_supported=["en"],
            success_metrics=[],
            privacy_sensitive=True,
            safety_critical=True,
        ),
    },
    "Auto Reply": {
        "Auto-Reply Email": FeatureMetadata(
            group="Auto Reply",
            name="Auto-Reply Email",
            description="Draft a polite, helpful reply to an inbound email.",
            category="auto_reply",
            input_format="email body text",
            output_format="reply email text",
            languages_supported=["en"],
            success_metrics=[],
            privacy_sensitive=True,
            safety_critical=True,
        ),
        "Auto-Reply Message": FeatureMetadata(
            group="Auto Reply",
            name="Auto-Reply Message",
            description="Draft a short friendly reply to an inbound chat message.",
            category="auto_reply",
            input_format="message text",
            output_format="short reply text",
            languages_supported=["en"],
            success_metrics=[],
            privacy_sensitive=True,
            safety_critical=True,
        ),
    },
    "Other": {}
}


def list_groups():
    return sorted(GROUPED_FEATURES.keys())


def list_features(group: str):
    if not group or group not in GROUPED_FEATURES:
        return []
    return sorted(GROUPED_FEATURES[group].keys())


AGENT = FeaturePromptWriterAgent()


def update_feature_choices(group):
    return gr.update(choices=list_features(group), value=None)


def toggle_custom(show_custom: bool):
    return gr.update(visible=show_custom)


def generate_prompt_from_selection(group, feature_name, language, metrics_csv):
    if not group or not feature_name:
        return "Select a group + feature first.", "", "", ""

    feat = GROUPED_FEATURES[group][feature_name]
    metrics = [m.strip() for m in (metrics_csv or "").split(",") if m.strip()]
    feat.success_metrics = metrics

    out = AGENT.generate(feat, language=language or None)

    metric_defs_str = "\n".join(
        [f"- {m}: {out.metric_definitions[m].get('definition','')}" for m in out.metrics_used]
    )
    suggested_str = ", ".join(out.suggested_additional_metrics) if out.suggested_additional_metrics else "(none)"

    return out.evaluation_prompt, metric_defs_str, suggested_str, f"{out.metrics_used}"


def add_custom_feature(
    new_group,
    new_name,
    description,
    category,
    input_format,
    output_format,
    languages_csv,
    metrics_csv,
    privacy_sensitive,
    safety_critical
):
    group = (new_group or "Other").strip() or "Other"
    name = (new_name or "").strip()
    if not name:
        return "Please provide a Feature Name.", gr.update(), gr.update()

    if group not in GROUPED_FEATURES:
        GROUPED_FEATURES[group] = {}

    languages = [x.strip() for x in (languages_csv or "en").split(",") if x.strip()]
    metrics = [m.strip() for m in (metrics_csv or "").split(",") if m.strip()]

    GROUPED_FEATURES[group][name] = FeatureMetadata(
        group=group,
        name=name,
        description=description or "",
        category=(category or "other").strip(),
        input_format=input_format or "",
        output_format=output_format or "",
        languages_supported=languages,
        success_metrics=metrics,
        privacy_sensitive=bool(privacy_sensitive),
        safety_critical=bool(safety_critical),
    )

    return f"Added feature '{name}' under group '{group}'.", gr.update(choices=list_groups(), value=group), gr.update(choices=list_features(group), value=name)


with gr.Blocks(title="Feature Prompt Writer Agent Demo") as demo:
    gr.Markdown("# Feature Prompt Writer Agent (Demo)\nSelect an existing feature or add a new group/feature. The agent resolves metrics (lookup + suggestions) and builds an evaluation prompt.")

    with gr.Row():
        group_dd = gr.Dropdown(label="Feature Group", choices=list_groups(), value="Auto Reply")
        feature_dd = gr.Dropdown(label="Feature", choices=list_features("Auto Reply"), value="Auto-Reply Email")

    group_dd.change(fn=update_feature_choices, inputs=group_dd, outputs=feature_dd)

    with gr.Row():
        language_tb = gr.Textbox(label="Language (optional)", placeholder="e.g., en, fr, zh-Hans", value="en")
        metrics_tb = gr.Textbox(label="Metrics (comma-separated, optional)", placeholder="e.g., relevance,tone,fluency")

    gen_btn = gr.Button("Generate Evaluation Prompt")

    prompt_out = gr.Textbox(label="Generated Evaluation Prompt", lines=18)
    metric_defs_out = gr.Textbox(label="Metric Definitions Used", lines=8)
    suggested_out = gr.Textbox(label="Suggested Additional Metrics", lines=2)
    used_metrics_out = gr.Textbox(label="Final Metrics Used", lines=2)

    gen_btn.click(
        fn=generate_prompt_from_selection,
        inputs=[group_dd, feature_dd, language_tb, metrics_tb],
        outputs=[prompt_out, metric_defs_out, suggested_out, used_metrics_out],
    )

    gr.Markdown("---\n## Add New Group / Feature")
    custom_toggle = gr.Checkbox(label="Add a new group/feature", value=False)
    custom_group = gr.Group(visible=False)

    custom_toggle.change(fn=toggle_custom, inputs=custom_toggle, outputs=custom_group)

    with custom_group:
        with gr.Row():
            new_group = gr.Textbox(label="New Group Name (optional)", placeholder="e.g., Accessibility AI")
            new_name = gr.Textbox(label="Feature Name", placeholder="e.g., Auto Rewrite Notes")
        description = gr.Textbox(label="Description", lines=2)
        with gr.Row():
            category = gr.Dropdown(label="Category", choices=["summarization", "auto_reply", "translation", "assistant", "other"], value="other")
            languages_csv = gr.Textbox(label="Languages (comma-separated)", value="en")
        with gr.Row():
            input_format = gr.Textbox(label="Input Format", placeholder="e.g., note text")
            output_format = gr.Textbox(label="Output Format", placeholder="e.g., rewritten text")
        metrics_csv = gr.Textbox(label="Metrics (comma-separated, optional)", placeholder="leave blank for suggestions")
        with gr.Row():
            privacy_sensitive = gr.Checkbox(label="Privacy Sensitive", value=True)
            safety_critical = gr.Checkbox(label="Safety Critical", value=False)

        add_btn = gr.Button("Add Feature")
        add_status = gr.Textbox(label="Status", lines=1)

        add_btn.click(
            fn=add_custom_feature,
            inputs=[new_group, new_name, description, category, input_format, output_format, languages_csv, metrics_csv, privacy_sensitive, safety_critical],
            outputs=[add_status, group_dd, feature_dd],
        )


if __name__ == "__main__":
    demo.launch()
