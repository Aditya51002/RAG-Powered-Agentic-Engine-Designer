"""The API entry point must not require optional optimizer/model dependencies at import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_api_main_import_does_not_load_optional_components() -> None:
    root = Path(__file__).parents[2]
    script = """
import builtins

blocked = {"optuna", "langgraph", "anthropic", "sentence_transformers", "chromadb"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"Optional dependency imported at API startup: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from rag_phy.api.main import app
assert app.title == "RAG Phy API"
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
