"""
src package - MetaFeature Orchestrator

All functionality is now in the `core` subpackage.
"""
from .core import *

__version__ = "0.1.0"
__all__ = [
    "create_app",
    "run_app",
    "FeaturePromptWriterAgent",
    "FeatureMetadata",
    "LLMClient",
]
