"""
Core evaluation prompt generation logic.
"""
import json
from typing import Tuple

from schemas import (
    FeatureMetadata, QualityMetric, OutputExample, 
    ResponsibleAIConstraints, InputOutputFormat
)
from tools import (
    get_openai_client, get_deployment_name,
    format_metrics, format_examples, format_rai_constraints,
    parse_comma_separated, parse_newline_separated
)
from prompt_store import (
    EVALUATION_AGENT_SYSTEM_PROMPT,
    FEATURE_EVALUATION_REQUEST_TEMPLATE
)


def generate_eval_prompt(metadata: FeatureMetadata) -> str:
    """Generate comprehensive evaluation prompt from feature metadata"""
    
    client = get_openai_client()
    deployment = get_deployment_name()
    
    # Format the request using the template
    user_request = FEATURE_EVALUATION_REQUEST_TEMPLATE.format(
        feature_name=metadata.feature_name,
        category=metadata.category,
        feature_description=metadata.feature_description,
        io_format=metadata.io_format.value,
        input_description=metadata.input_description,
        output_description=metadata.output_description,
        supported_languages=', '.join(metadata.supported_languages),
        target_language=metadata.target_language,
        target_location=metadata.target_location,
        locale_considerations=metadata.locale_considerations or 'None specified',
        metrics_text=format_metrics(metadata.quality_metrics),
        rai_text=format_rai_constraints(metadata.responsible_ai),
        domain_constraints=metadata.domain_constraints or 'None specified',
        good_examples_text=format_examples(metadata.good_examples, is_good=True),
        bad_examples_text=format_examples(metadata.bad_examples, is_good=False),
        input_sample=metadata.input_data_sample
    )
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": EVALUATION_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_request}
        ],
        temperature=0.7,
        max_tokens=4000
    )
    return response.choices[0].message.content


def process_feature_evaluation(
    # Basic Info
    feature_name: str, feature_description: str, category: str, io_format: str,
    input_description: str, output_description: str,
    # Localization
    supported_languages: str, target_language: str, target_location: str, locale_considerations: str,
    # Metrics (as JSON string)
    metrics_json: str,
    # RAI
    no_pii: bool, bias_check: bool, toxicity_check: bool, cultural_sensitivity: bool, custom_constraints: str,
    # Domain
    domain_constraints: str,
    # Examples
    good_example_input: str, good_example_output: str, good_example_why: str,
    bad_example_input: str, bad_example_output: str, bad_example_why: str,
    # Test Data
    input_sample: str
) -> Tuple[str, str]:
    """Process all inputs and generate evaluation prompt"""
    
    try:
        # Parse metrics JSON
        metrics = []
        if metrics_json.strip():
            metrics_data = json.loads(metrics_json)
            metrics = [QualityMetric(**m) for m in metrics_data]
        
        # Build examples
        good_examples = []
        if good_example_input.strip() and good_example_output.strip():
            good_examples.append(OutputExample(
                input_text=good_example_input,
                output_text=good_example_output,
                is_good_example=True,
                explanation=good_example_why or "Demonstrates expected quality"
            ))
        
        bad_examples = []
        if bad_example_input.strip() and bad_example_output.strip():
            bad_examples.append(OutputExample(
                input_text=bad_example_input,
                output_text=bad_example_output,
                is_good_example=False,
                explanation=bad_example_why or "Shows common failure mode"
            ))
        
        # Parse custom constraints and languages
        custom_list = parse_newline_separated(custom_constraints)
        lang_list = parse_comma_separated(supported_languages, default=["English"])
        
        # Build metadata
        metadata = FeatureMetadata(
            feature_name=feature_name,
            feature_description=feature_description,
            category=category,
            io_format=InputOutputFormat(io_format),
            input_description=input_description,
            output_description=output_description,
            supported_languages=lang_list,
            target_language=target_language,
            target_location=target_location,
            locale_considerations=locale_considerations,
            quality_metrics=metrics,
            responsible_ai=ResponsibleAIConstraints(
                no_pii_leakage=no_pii,
                bias_check_required=bias_check,
                toxicity_check_required=toxicity_check,
                cultural_sensitivity=cultural_sensitivity,
                custom_constraints=custom_list
            ),
            domain_constraints=domain_constraints,
            good_examples=good_examples,
            bad_examples=bad_examples,
            input_data_sample=input_sample
        )
        
        # Generate the evaluation prompt
        result = generate_eval_prompt(metadata)
        return result, "✅ Evaluation prompt generated successfully!"
        
    except json.JSONDecodeError as e:
        return "", f"❌ Error parsing metrics JSON: {str(e)}"
    except Exception as e:
        return "", f"❌ Error: {str(e)}"
