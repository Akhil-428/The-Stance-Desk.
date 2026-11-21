"""Fake News Stance Detection: Streamlit entry point.

Run:  streamlit run app.py      ->  http://localhost:8501
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make project modules importable regardless of the working directory, and keep
# the app fully offline (models are loaded from local files only).
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ui.navigation import run_app  # noqa: E402

run_app()
