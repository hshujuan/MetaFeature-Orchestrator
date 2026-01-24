from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from prompt_templates import template_auto_reply, template_summarization, template_generic


AUTO_CREATE_METRICS = """\
\"\"\"suggested_metrics.py
v1.0 – GenAI Evaluation Metrics Registry
\"\"\"

METRICS = {
    "faithfulness": {
        "definition": "Output must not introduce information not present in the input (no hallucination).",
        "applies_to": ["summarization", "translation", "assistant", "auto_reply"],
        "rai": ["hallucination"],
    },
    "coverage": {
        "definition": "All key points from the input should be captured in the output (no major omissions).",
        "applies_to": ["summarization"],
        "rai": [],
    },
    "relevance": {
        "definition": "Output directly addresses the user intent and input context; no off-topic content.",
        "applies_to": ["auto_reply", "assistant", "summarization"],
        "rai": [],
    },
    "tone": {
        "definition": "Appropriate politeness, formality, and emotional alignment for the context.",
        "applies_to": ["auto_reply", "assistant"],
        "rai": ["harassment", "toxicity"],
    },
    "fluency": {
        "definition": "Grammatically correct, natural, and easy to read in the target language.",
        "applies_to": ["all"],
        "rai": [],
    },
    "brevity": {
        "definition": "Concise without sacrificing required information; avoids unnecessary verbosity.",
        "applies_to": ["auto_reply", "summarization"],
        "rai": [],
    },
    "safety": {
        "definition": "No toxic, biased, sexual, violent, or otherwise harmful content; follows policy.",
        "applies_to": ["all"],
        "rai": ["toxicity", "bias", "self_harm", "sexual_content", "violence"],
    },
    "privacy": {
        "definition": "Avoids leaking sensitive personal data; respects confidentiality of the input content.",
        "applies_to": ["auto_reply", "summarization", "assistant", "translation"],
        "rai": ["privacy"],
    },
    "groundedness": {
        "definition": "Claims are supported by the provided source content; no unsupported assertions.",
        "applies_to": ["summarization", "assistant"],
        "rai": ["hallucination"],
    },
    "format_compliance": {
        "definition": "Output follows requested format (JSON, bullets, length constraints, etc.).",
        "applies_to": ["all"],
        "rai": [],
    },
}
"""


@dataclass
class FeatureMetadata:
    group: str
    name: str
    description: str
    category: str  # e.g. "auto_reply", "summarization"
    input_format: str
    output_format: str
    languages_supported: List[str]
    success_metrics: List[str]  # can be empty
    privacy_sensitive: bool = True
    safety_critical: bool = False


@dataclass
class PromptOutput:
    feature_name: str
    group: str
    category: str
    language: str
    metrics_used: List[str]
    metric_definitions: Dict[str, dict]
    suggested_additional_metrics: List[str]
    evaluation_prompt: str


DEFAULT_METRICS_BY_CATEGORY = {
    "summarization": ["faithfulness", "coverage", "groundedness", "fluency", "brevity", "safety", "privacy", "format_compliance"],
    "auto_reply": ["relevance", "tone", "fluency", "brevity", "safety", "privacy", "format_compliance"],
    "translation": ["faithfulness", "fluency", "safety", "privacy", "format_compliance"],
    "assistant": ["relevance", "faithfulness", "groundedness", "tone", "safety", "privacy", "format_compliance"],
    "other": ["relevance", "fluency", "safety", "privacy", "format_compliance"],
}


def ensure_metrics_registry(path: str = "suggested_metrics.py") -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(AUTO_CREATE_METRICS)


def load_metrics_registry() -> Dict[str, dict]:
    ensure_metrics_registry()
    mod = importlib.import_module("suggested_metrics")
    importlib.reload(mod)
    metrics = getattr(mod, "METRICS", {})
    if not isinstance(metrics, dict):
        raise ValueError("suggested_metrics.METRICS must be a dict.")
    return metrics


def resolve_metrics(
    category: str,
    user_metrics: Optional[List[str]],
    registry: Dict[str, dict],
) -> Tuple[List[str], Dict[str, dict], List[str]]:
    """
    Returns:
      metrics_used: final list
      metric_definitions: subset dict for used metrics (including placeholders for unknown)
      suggested_additional_metrics: other recommended metrics
    """
    category = (category or "other").strip().lower()
    base = DEFAULT_METRICS_BY_CATEGORY.get(category, DEFAULT_METRICS_BY_CATEGORY["other"])

    if user_metrics and any(m.strip() for m in user_metrics):
        metrics_used = [m.strip() for m in user_metrics if m.strip()]
    else:
        metrics_used = list(base)

    metric_definitions: Dict[str, dict] = {}
    for m in metrics_used:
        metric_definitions[m] = registry.get(m, {"definition": "(definition not found in registry)", "applies_to": [], "rai": []})

    # Suggest additional metrics (category-aware) if user provided subset
    suggested = []
    if user_metrics and any(m.strip() for m in user_metrics):
        for m, meta in registry.items():
            applies = meta.get("applies_to", [])
            if m in metrics_used:
                continue
            if "all" in applies or category in applies:
                # avoid dumping too many: cap to 8 suggestions
                suggested.append(m)
            if len(suggested) >= 8:
                break

    # Add implicit RAI-driven metrics if feature flags require it
    # (Keep simple for demo; you can expand logic later.)
    return metrics_used, metric_definitions, suggested


def build_evaluation_prompt(feature: FeatureMetadata, language: str, metrics_used: List[str], metric_defs: Dict[str, dict]) -> str:
    cat = feature.category.lower().strip()
    if cat == "auto_reply":
        return template_auto_reply(feature.name, language, metrics_used, metric_defs)
    if cat == "summarization":
        return template_summarization(feature.name, language, metrics_used, metric_defs)
    return template_generic(feature.name, language, metrics_used, metric_defs)


class FeaturePromptWriterAgent:
    """
    A lightweight agent core. You can wrap this in Microsoft Agent Framework
    by mapping methods here to tools and planner steps.
    """

    def __init__(self):
        self.registry = load_metrics_registry()

    def generate(self, feature: FeatureMetadata, language: Optional[str] = None) -> PromptOutput:
        # pick first supported language if not provided
        lang = language or (feature.languages_supported[0] if feature.languages_supported else "en")

        metrics_used, metric_defs, suggested = resolve_metrics(
            category=feature.category,
            user_metrics=feature.success_metrics,
            registry=self.registry,
        )

        # Add default RAI metrics if flags set and missing
        if feature.privacy_sensitive and "privacy" not in metrics_used and "privacy" in self.registry:
            metrics_used.append("privacy")
            metric_defs["privacy"] = self.registry["privacy"]
        if (feature.safety_critical or True) and "safety" not in metrics_used and "safety" in self.registry:
            # keep safety broadly on by default for GenAI features
            metrics_used.append("safety")
            metric_defs["safety"] = self.registry["safety"]

        prompt = build_evaluation_prompt(feature, lang, metrics_used, metric_defs)

        return PromptOutput(
            feature_name=feature.name,
            group=feature.group,
            category=feature.category,
            language=lang,
            metrics_used=metrics_used,
            metric_definitions=metric_defs,
            suggested_additional_metrics=suggested,
            evaluation_prompt=prompt,
        )

    def export_feature_json(self, feature: FeatureMetadata) -> dict:
        return asdict(feature)
