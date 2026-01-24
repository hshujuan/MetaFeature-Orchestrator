"""
Text formatting utilities for prompt generation.
"""
from schemas import QualityMetric, OutputExample, ResponsibleAIConstraints


def format_metrics(metrics: list[QualityMetric]) -> str:
    """Format quality metrics for the prompt"""
    if not metrics:
        return "  No specific metrics provided - use standard quality metrics"
    
    return "\n".join([
        f"  - {m.name} (Weight: {m.weight}, Primary: {m.is_primary}): {m.description}"
        for m in metrics
    ])


def format_examples(examples: list[OutputExample], is_good: bool) -> str:
    """Format output examples for the prompt"""
    if not examples:
        return f"  No {'good' if is_good else 'bad'} examples provided"
    
    label = "Why Good" if is_good else "Why Bad"
    return "\n".join([
        f"  Input: {ex.input_text}\n  Output: {ex.output_text}\n  {label}: {ex.explanation}"
        for ex in examples
    ])


def format_rai_constraints(rai: ResponsibleAIConstraints) -> str:
    """Format responsible AI constraints for the prompt"""
    rai_checks = []
    if rai.no_pii_leakage:
        rai_checks.append("- No PII/personal data leakage")
    if rai.bias_check_required:
        rai_checks.append("- Check for biased content")
    if rai.toxicity_check_required:
        rai_checks.append("- Check for toxic/offensive content")
    if rai.cultural_sensitivity:
        rai_checks.append("- Cultural sensitivity for target locale")
    rai_checks.extend([f"- {c}" for c in rai.custom_constraints])
    
    return "\n".join(rai_checks) if rai_checks else "  Standard RAI checks"


def parse_comma_separated(text: str, default: list = None) -> list[str]:
    """Parse comma-separated string into list"""
    if not text or not text.strip():
        return default or []
    return [item.strip() for item in text.split(',') if item.strip()]


def parse_newline_separated(text: str) -> list[str]:
    """Parse newline-separated string into list"""
    if not text or not text.strip():
        return []
    return [item.strip() for item in text.split('\n') if item.strip()]
