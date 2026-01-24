import json
from typing import List, Dict, Any
from llm_clients import call_responses
from db import PromptTemplateStore

template_store = PromptTemplateStore()

PROMPT_CRITIC_SYSTEM = """You are a strict prompt engineer.
Score evaluation prompts for quality and safety.
Return JSON with: clarity, metric_coverage, groundedness, rai_coverage, overall, rationale.
"""

def propose_variants(feature_json: Dict[str, Any], base_prompt: str, k: int = 5) -> List[str]:
    req = {
        "feature": feature_json,
        "base_prompt": base_prompt,
        "k": k,
        "instructions": "Generate improved evaluation prompt variants. Keep JSON output format. Strengthen groundedness and RAI checks."
    }
    out = call_responses(model="gpt-4o-mini", input_text=json.dumps(req))
    # Expect model to return JSON list; add robust parsing in production
    try:
        variants = json.loads(out)
        return [v for v in variants if isinstance(v, str)]
    except Exception:
        return [base_prompt]  # fallback

def judge_prompt(prompt_text: str, feature_json: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "system": PROMPT_CRITIC_SYSTEM,
        "feature": feature_json,
        "candidate_prompt": prompt_text,
        "rubric": {
            "clarity": "Unambiguous, easy for a judge to follow",
            "metric_coverage": "All required metrics explicitly evaluated",
            "groundedness": "Explicitly forbids external knowledge / hallucinated judging",
            "rai_coverage": "Includes safety/privacy/bias where relevant",
        }
    }
    out = call_responses(model="gpt-4o-mini", input_text=json.dumps(payload))
    try:
        return json.loads(out)
    except Exception:
        return {"overall": 0, "rationale": "judge parse failed"}

def optimize_and_store(feature_id: str, language: str, category: str, metrics: List[str], base_prompt: str, feature_json: Dict[str, Any]):
    variants = propose_variants(feature_json, base_prompt, k=5)

    scored = []
    for v in variants:
        s = judge_prompt(v, feature_json)
        scored.append((s.get("overall", 0), v, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_prompt, best_meta = scored[0]

    tid = template_store.upsert_template(
        feature_id=feature_id,
        language=language,
        category=category,
        metrics=metrics,
        prompt=best_prompt,
        source=f"optimizer_best_overall_{best_score}"
    )
    return {"template_id": tid, "best_score": best_score, "judge": best_meta}
