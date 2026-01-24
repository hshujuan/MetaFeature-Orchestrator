"""
MetaFeature Orchestrator - Gradio Web Application
Intelligent Evaluation Prompt Generator for GenAI Features
"""
import gradio as gr

from schemas import InputOutputFormat
from generator import process_feature_evaluation
from tools import load_metric_template, get_available_templates, get_default_metrics_json


# Custom CSS for styling
CUSTOM_CSS = """
.main-header { text-align: center; margin-bottom: 20px; }
.section-header { 
    background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 15px 0 10px 0;
}
.output-box {
    border: 2px solid #2d5a87;
    border-radius: 10px;
    padding: 15px;
}
"""


def create_demo() -> gr.Blocks:
    """Create and configure the Gradio demo interface"""
    
    with gr.Blocks(title="MetaFeature Orchestrator - GenAI Evaluation Agent") as demo:
        
        # Header
        gr.Markdown("""
        <div class="main-header">
        
        # 🧠 MetaFeature Orchestrator
        ### Intelligent Evaluation Prompt Generator for GenAI Features
        
        *Dynamically synthesize comprehensive evaluation instructions across feature domains*
        
        </div>
        """)
        
        with gr.Tabs():
            # ═══════════════════════════════════════════════════════════
            # TAB 1: Feature Configuration
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("📋 Feature Configuration"):
                gr.Markdown('<div class="section-header">📌 Basic Feature Information</div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feature_name = gr.Textbox(
                            label="Feature Name",
                            placeholder="e.g., Auto-Summarize News Articles"
                        )
                        category = gr.Dropdown(
                            choices=["Communication", "Intelligence", "Productivity", "Creative", "Safety", "Accessibility"],
                            label="Category",
                            value="Intelligence"
                        )
                    with gr.Column(scale=2):
                        feature_description = gr.Textbox(
                            label="Feature Description",
                            placeholder="Describe what this feature does in detail...",
                            lines=3
                        )
                
                gr.Markdown('<div class="section-header">🔄 Input/Output Format</div>')
                
                with gr.Row():
                    io_format = gr.Dropdown(
                        choices=[e.value for e in InputOutputFormat],
                        label="I/O Format",
                        value="text-to-text"
                    )
                with gr.Row():
                    input_description = gr.Textbox(
                        label="Input Description",
                        placeholder="e.g., Full news article text (100-2000 words)"
                    )
                    output_description = gr.Textbox(
                        label="Output Description",
                        placeholder="e.g., 3-5 bullet point summary capturing key facts"
                    )
                
                gr.Markdown('<div class="section-header">🌍 Localization Settings</div>')
                
                with gr.Row():
                    supported_languages = gr.Textbox(
                        label="Supported Languages",
                        value="English, Spanish, French, German, Japanese, Chinese",
                        placeholder="Comma-separated list"
                    )
                    target_language = gr.Textbox(
                        label="Target Language for Evaluation",
                        value="English"
                    )
                with gr.Row():
                    target_location = gr.Textbox(
                        label="Target Location/Locale",
                        value="United States",
                        placeholder="e.g., Tokyo, Japan"
                    )
                    locale_considerations = gr.Textbox(
                        label="Locale-Specific Considerations",
                        placeholder="e.g., Use formal register, respect honorifics, date format MM/DD/YYYY"
                    )
            
            # ═══════════════════════════════════════════════════════════
            # TAB 2: Quality Metrics
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("📊 Quality Metrics"):
                gr.Markdown('<div class="section-header">📈 Success Criteria & Metrics</div>')
                
                gr.Markdown("""
                Define the quality metrics that will be used to evaluate this feature's outputs.
                Each metric should have a name, description, weight (0-1), and whether it's a primary metric.
                """)
                
                with gr.Row():
                    metric_template = gr.Dropdown(
                        choices=get_available_templates(),
                        label="Load Metric Template",
                        value="None"
                    )
                    load_template_btn = gr.Button("📥 Load Template", variant="secondary")
                
                metrics_json = gr.Code(
                    label="Quality Metrics (JSON)",
                    language="json",
                    value=get_default_metrics_json(),
                    lines=15
                )
                
                load_template_btn.click(
                    fn=load_metric_template,
                    inputs=[metric_template],
                    outputs=[metrics_json]
                )
            
            # ═══════════════════════════════════════════════════════════
            # TAB 3: Responsible AI
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("🛡️ Responsible AI"):
                gr.Markdown('<div class="section-header">🔒 Responsible AI Constraints</div>')
                
                with gr.Row():
                    with gr.Column():
                        no_pii = gr.Checkbox(
                            label="No PII Leakage",
                            value=True
                        )
                        bias_check = gr.Checkbox(
                            label="Bias Check Required",
                            value=True
                        )
                    with gr.Column():
                        toxicity_check = gr.Checkbox(
                            label="Toxicity Check Required",
                            value=True
                        )
                        cultural_sensitivity = gr.Checkbox(
                            label="Cultural Sensitivity",
                            value=True
                        )
                
                custom_constraints = gr.Textbox(
                    label="Custom Constraints",
                    placeholder="Enter additional constraints, one per line:\n- Must not generate medical advice\n- Should respect copyright",
                    lines=4
                )
                
                domain_constraints = gr.Textbox(
                    label="Domain-Specific Requirements",
                    placeholder="e.g., For legal documents, must maintain exact terminology. For children's content, must be age-appropriate.",
                    lines=2
                )
            
            # ═══════════════════════════════════════════════════════════
            # TAB 4: Examples
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("📝 Examples"):
                gr.Markdown('<div class="section-header">✅ Good Output Example</div>')
                gr.Markdown("*Provide an example of a high-quality output to calibrate the evaluator*")
                
                with gr.Row():
                    good_example_input = gr.Textbox(
                        label="Example Input",
                        placeholder="The input that was provided...",
                        lines=3
                    )
                    good_example_output = gr.Textbox(
                        label="Good Output",
                        placeholder="The high-quality output produced...",
                        lines=3
                    )
                good_example_why = gr.Textbox(
                    label="Why This is Good",
                    placeholder="Explain what makes this output high-quality..."
                )
                
                gr.Markdown('<div class="section-header">❌ Bad Output Example</div>')
                gr.Markdown("*Provide an example of a poor-quality output to show what to avoid*")
                
                with gr.Row():
                    bad_example_input = gr.Textbox(
                        label="Example Input",
                        placeholder="The input that was provided...",
                        lines=3
                    )
                    bad_example_output = gr.Textbox(
                        label="Bad Output",
                        placeholder="The poor-quality output produced...",
                        lines=3
                    )
                bad_example_why = gr.Textbox(
                    label="Why This is Bad",
                    placeholder="Explain what's wrong with this output..."
                )
            
            # ═══════════════════════════════════════════════════════════
            # TAB 5: Generate & Output
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("🚀 Generate Evaluation"):
                gr.Markdown('<div class="section-header">📥 Sample Input Data</div>')
                
                input_sample = gr.Textbox(
                    label="Input Data Sample",
                    placeholder="""Paste a sample input that the feature would process...

For example, if evaluating a news summarizer, paste a news article here.
If evaluating an email auto-reply, paste the email thread here.""",
                    lines=8
                )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "🎯 Generate Evaluation Prompt",
                        variant="primary",
                        scale=2
                    )
                    clear_btn = gr.Button("🗑️ Clear Output", variant="secondary", scale=1)
                
                status_box = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown('<div class="section-header">📤 Generated Evaluation Prompt</div>')
                
                output_box = gr.Markdown(
                    label="Generated Evaluation Prompt",
                    elem_classes=["output-box"]
                )
                
                # Wire up the generate button
                generate_btn.click(
                    fn=process_feature_evaluation,
                    inputs=[
                        feature_name, feature_description, category, io_format,
                        input_description, output_description,
                        supported_languages, target_language, target_location, locale_considerations,
                        metrics_json,
                        no_pii, bias_check, toxicity_check, cultural_sensitivity, custom_constraints,
                        domain_constraints,
                        good_example_input, good_example_output, good_example_why,
                        bad_example_input, bad_example_output, bad_example_why,
                        input_sample
                    ],
                    outputs=[output_box, status_box]
                )
                
                clear_btn.click(
                    fn=lambda: ("", ""),
                    outputs=[output_box, status_box]
                )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <b>MetaFeature Orchestrator</b> | Intelligent GenAI Evaluation Framework<br>
        Generating high-quality, localized evaluation prompts for AI features worldwide
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(css=CUSTOM_CSS)
