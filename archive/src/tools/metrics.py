"""
Metric template utilities.
"""
import json
from schemas import METRIC_TEMPLATES


def load_metric_template(template_name: str) -> str:
    """Load a predefined metric template as JSON string"""
    if template_name == "None":
        return "[]"
    
    metrics = METRIC_TEMPLATES.get(template_name, [])
    return json.dumps([m.model_dump() for m in metrics], indent=2)


def get_available_templates() -> list[str]:
    """Get list of available metric templates"""
    return ["None"] + list(METRIC_TEMPLATES.keys())


def get_default_metrics_json() -> str:
    """Return default metrics JSON for the UI"""
    return """[
  {
    "name": "Accuracy",
    "description": "Output correctly represents the source information",
    "weight": 1.0,
    "is_primary": true
  },
  {
    "name": "Fluency",
    "description": "Output reads naturally and is grammatically correct",
    "weight": 0.7,
    "is_primary": false
  }
]"""
