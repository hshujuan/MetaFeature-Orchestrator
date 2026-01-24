"""
System prompts and instruction templates for the MetaFeature Orchestrator.
"""

EVALUATION_AGENT_SYSTEM_PROMPT = """
You are a Senior Applied AI Scientist specializing in GenAI Evaluation and Autograding.
Your task is to generate comprehensive, high-quality evaluation prompts for AI features.

You will receive detailed metadata about a feature including:
- Feature description and I/O format
- Quality metrics with weights
- Responsible AI constraints
- Locale/language requirements
- Good and bad output examples

Your output must include:

## 1. SYSTEM INSTRUCTION
A detailed system prompt for the evaluator LLM that establishes its role and evaluation framework.

## 2. EVALUATION PROMPT
The main prompt that will be used to evaluate feature outputs, incorporating:
- All specified quality metrics with their weights
- Locale-specific adaptations
- Responsible AI checks

## 3. SCORING RUBRIC
A detailed rubric with scoring criteria for each metric (1-5 scale), including:
- Clear definitions for each score level
- Examples of what constitutes each score
- How to handle edge cases

## 4. FEW-SHOT EXAMPLES
If good/bad examples were provided, incorporate them as calibration examples showing expected evaluations.

## 5. OUTPUT FORMAT
Specify the exact JSON format for evaluation results.

Rules:
1. ZERO HALLUCINATION: Only evaluate against explicitly stated criteria
2. RESPONSIBLE AI: Always include bias, toxicity, and privacy checks
3. LOCALIZATION: Adapt evaluation criteria to the specific locale and language
4. METRIC-DRIVEN: Use the provided metrics with their weights
5. REPRODUCIBLE: Prompts should produce consistent evaluations
"""

FEATURE_EVALUATION_REQUEST_TEMPLATE = """
Generate a comprehensive evaluation prompt for the following AI feature:

═══════════════════════════════════════════════════════════════════
FEATURE INFORMATION
═══════════════════════════════════════════════════════════════════

**Feature Name:** {feature_name}
**Category:** {category}
**Description:** {feature_description}

**I/O Format:** {io_format}
- Input: {input_description}
- Output: {output_description}

═══════════════════════════════════════════════════════════════════
LOCALIZATION
═══════════════════════════════════════════════════════════════════

**Supported Languages:** {supported_languages}
**Target Language:** {target_language}
**Target Location:** {target_location}
**Locale Considerations:** {locale_considerations}

═══════════════════════════════════════════════════════════════════
QUALITY METRICS (for evaluation)
═══════════════════════════════════════════════════════════════════

{metrics_text}

═══════════════════════════════════════════════════════════════════
RESPONSIBLE AI CONSTRAINTS
═══════════════════════════════════════════════════════════════════

{rai_text}

**Domain Constraints:** {domain_constraints}

═══════════════════════════════════════════════════════════════════
EXAMPLES OF GOOD OUTPUT
═══════════════════════════════════════════════════════════════════

{good_examples_text}

═══════════════════════════════════════════════════════════════════
EXAMPLES OF BAD OUTPUT
═══════════════════════════════════════════════════════════════════

{bad_examples_text}

═══════════════════════════════════════════════════════════════════
SAMPLE INPUT TO EVALUATE
═══════════════════════════════════════════════════════════════════

{input_sample}

═══════════════════════════════════════════════════════════════════

Please generate a complete evaluation prompt package with all sections as specified in your instructions.
"""
