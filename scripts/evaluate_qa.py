#!/usr/bin/env python3
"""Run answer generation and configured RAGAS metrics against the versioned QA set."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rag_phy.agents import AnthropicLLMClient
from rag_phy.api.knowledge import ConfiguredKnowledgeSearch
from rag_phy.config import load_config, load_evaluation_config
from rag_phy.evaluation import (
    ConfiguredRagasBackend,
    GroundedRAGEvaluationTarget,
    evaluate_dataset,
    load_qa_dataset,
    save_evaluation_report,
)

ROOT = Path(__file__).resolve().parents[1]

try:
    from scripts.validate_qa_dataset import validate_qa_dataset
except ModuleNotFoundError:
    from validate_qa_dataset import validate_qa_dataset


def _ragas_anthropic_llm(config):
    """Create RAGAS's native Anthropic adapter from the configured provider credentials."""
    try:
        from anthropic import Anthropic
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError("Install rag-phy[llm,evaluation] to run answer-quality evaluation") from exc
    if not config.api_key or not config.api_key.strip():
        raise ValueError("Set RAG_PHY_LLM__API_KEY to run live answer-quality evaluation")
    client = Anthropic(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=0,
    )
    return llm_factory(config.model_name, provider="anthropic", client=client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/evaluation/answer_quality_draft.json",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    errors = validate_qa_dataset()
    if errors:
        print("Cannot evaluate against an invalid QA/corpus manifest:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    evaluation_config = load_evaluation_config(ROOT / "config/evaluation.yaml")
    dataset = load_qa_dataset(evaluation_config.qa_dataset_path)
    app_config = load_config(ROOT / "config/app.yaml")
    if app_config.llm is None:
        raise ValueError("config/app.yaml must define an LLM before answer evaluation")

    target = GroundedRAGEvaluationTarget(
        retrieve=ConfiguredKnowledgeSearch(ROOT / "config/models.yaml"),
        generator=AnthropicLLMClient(app_config.llm),
    )
    backend = ConfiguredRagasBackend(
        evaluation_config.ragas_metric_names,
        llm=_ragas_anthropic_llm(app_config.llm),
    )
    report = evaluate_dataset(dataset, target, backend)
    save_evaluation_report(report, args.output)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
