"""
suggested_metrics.py
v1.0 – GenAI Evaluation Metrics Registry

You can extend this dictionary over time (add new metrics, categories, languages, etc.).
"""

from __future__ import annotations

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
