from typing import Dict, Any

METRICS_I18N: Dict[str, Dict[str, Any]] = {
    "faithfulness": {
        "definitions": {
            "en": "Output must not introduce information not present in the input (no hallucination).",
            "zh-Hans": "输出不得引入输入中不存在的信息（避免幻觉）。",
            "ja": "入力にない情報を付け加えない（幻覚を避ける）。",
        },
        "applies_to": ["summarization", "translation", "assistant", "auto_reply"],
        "rai": ["hallucination"],
    },
    "tone": {
        "definitions": {
            "en": "Appropriate politeness, formality, and emotional alignment for the context.",
            "zh-Hans": "语气得体：礼貌、正式程度与情绪契合场景。",
        },
        "applies_to": ["auto_reply", "assistant"],
        "rai": ["toxicity", "harassment"],
    },
    # ... extend ...
}

def get_metric_definition(metric: str, lang: str = "en") -> str:
    meta = METRICS_I18N.get(metric)
    if not meta:
        return "(definition not found)"
    defs = meta.get("definitions", {})
    return defs.get(lang) or defs.get("en") or "(definition not found)"
