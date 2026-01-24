"""
Prompt store package - Contains system prompts and instruction templates.
"""
from .system_prompts import (
    EVALUATION_AGENT_SYSTEM_PROMPT,
    FEATURE_EVALUATION_REQUEST_TEMPLATE
)

__all__ = [
    "EVALUATION_AGENT_SYSTEM_PROMPT",
    "FEATURE_EVALUATION_REQUEST_TEMPLATE"
]
