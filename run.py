#!/usr/bin/env python
"""
MetaFeature Orchestrator - Entry Point
Run this script to launch the application.
"""
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core import run_app

if __name__ == "__main__":
    run_app()
