"""Offline tests for the structured design proposal agent."""

from __future__ import annotations

import json

import pytest

from rag_phy.agents import (
    CandidateRejection,
    DesignAgent,
    DesignCandidate,
    DuplicateCandidateError,
    ProposalExhaustedError,
)
from rag_phy.config import load_agent_prompts_config


class SequenceClient:
    """Deterministic injected client returning queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        """Capture the rendered prompt and return the next configured test response."""
        self.prompts.append(prompt)
        return next(self.responses)


def _candidate(pressure_ratio: float = 8.0, turbine_temperature_k: float = 1400.0) -> dict[str, float]:
    """Provide a schema-valid synthetic cycle point for isolated agent tests."""
    return {
        "ambient_temperature_k": 288.15,
        "ambient_pressure_pa": 101325.0,
        "flight_speed_m_per_s": 0.0,
        "air_mass_flow_kg_per_s": 10.0,
        "compressor_pressure_ratio": pressure_ratio,
        "turbine_inlet_temperature_k": turbine_temperature_k,
        "hot_section_material_name": "TEST_ALLOY_ALPHA",
    }


@pytest.fixture
def agent_config():
    """Load the actual checked-in agent configuration."""
    return load_agent_prompts_config("config/agent_prompts.yaml").design_agent


def test_invalid_json_then_valid_candidate_retries(agent_config) -> None:
    client = SequenceClient(["not json", json.dumps(_candidate())])
    agent = DesignAgent(client, agent_config)

    candidate = agent.propose("maximize preliminary thrust")

    assert candidate.compressor_pressure_ratio == 8.0
    assert len(client.prompts) == 2
    assert '"feedback":' not in client.prompts[0]
    assert "required schema" in client.prompts[1]


def test_malformed_output_exhausts_configured_retry_limit(agent_config) -> None:
    client = SequenceClient(["{}", "{}", "{}"])
    agent = DesignAgent(client, agent_config)

    with pytest.raises(ProposalExhaustedError, match="3 attempts"):
        agent.propose("evaluate a design")

    assert len(client.prompts) == agent_config.max_attempts


def test_previously_rejected_candidate_is_not_returned(agent_config) -> None:
    rejected = DesignCandidate.model_validate(_candidate())
    client = SequenceClient([json.dumps(_candidate())] * agent_config.max_attempts)
    agent = DesignAgent(client, agent_config)

    with pytest.raises(DuplicateCandidateError):
        agent.propose(
            "propose a better design",
            [CandidateRejection(candidate=rejected, reason="temperature constraint exceeded")],
        )

    assert len(client.prompts) == agent_config.max_attempts
    assert "temperature constraint exceeded" in client.prompts[0]
    assert "previously rejected parameter set" in client.prompts[1]


def test_duplicate_history_is_canonicalized_and_bounded(agent_config) -> None:
    candidate_a = DesignCandidate.model_validate(_candidate(8.0000001))
    candidate_b = DesignCandidate.model_validate(_candidate(8.0000002))
    short_config = agent_config.model_copy(update={"rejection_history_limit": 1})
    client = SequenceClient([json.dumps(_candidate(9.0))])
    agent = DesignAgent(client, short_config)

    agent.propose(
        "propose a different design",
        [
            CandidateRejection(candidate=candidate_a, reason="older reason"),
            CandidateRejection(candidate=candidate_b, reason="latest reason"),
        ],
    )

    prompt = client.prompts[0]
    assert "latest reason" in prompt
    assert "older reason" not in prompt


def test_design_candidate_rejects_unknown_and_invalid_fields() -> None:
    with pytest.raises(ValueError):
        DesignCandidate.model_validate({**_candidate(), "unmodeled_parameter": 2})
    with pytest.raises(ValueError):
        DesignCandidate.model_validate(_candidate(pressure_ratio=1.0))
    with pytest.raises(ValueError, match="exceed ambient"):
        DesignCandidate.model_validate(_candidate(turbine_temperature_k=288.15))
