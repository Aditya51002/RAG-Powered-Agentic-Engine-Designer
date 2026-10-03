"""Anthropic client contract tests using a no-network fake transport."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from rag_phy.agents import AnthropicLLMClient, LLMClientError
from rag_phy.config import LLMConfig, load_config


def _config(**updates) -> LLMConfig:
    values = {
        "provider": "anthropic",
        "api_key": "test-secret",
        "model_name": "test-model",
        "max_tokens": 128,
        "timeout_seconds": 3,
        "transient_retry_attempts": 2,
        "retry_initial_delay_seconds": 1,
        "retry_max_delay_seconds": 4,
        "input_cost_usd_per_million_tokens": 2,
        "output_cost_usd_per_million_tokens": 10,
    }
    values.update(updates)
    return LLMConfig.model_validate(values)


class _Messages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _response(text: str = "completion"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        _request_id="request-test",
    )


def test_completion_logs_usage_cost_and_latency(caplog) -> None:
    caplog.set_level(logging.INFO)
    messages = _Messages([_response("grounded response")])
    client = AnthropicLLMClient(
        _config(),
        sdk_client=SimpleNamespace(messages=messages),
        monotonic=iter((10.0, 10.25)).__next__,
    )

    assert client.complete("prompt") == "grounded response"
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Anthropic completion succeeded"
    )
    assert record.tokens_in == 100
    assert record.tokens_out == 20
    assert record.latency_ms == 250
    assert record.estimated_cost_usd == pytest.approx(0.0004)
    assert record.provider_request_id == "request-test"
    assert messages.calls[0]["model"] == "test-model"


def test_retry_after_is_honored_and_transient_retry_budget_is_separate() -> None:
    rate_limit = RuntimeError("rate limited")
    rate_limit.response = SimpleNamespace(status_code=429, headers={"retry-after": "2"})
    messages = _Messages([rate_limit, _response()])
    delays = []
    client = AnthropicLLMClient(
        _config(), sdk_client=SimpleNamespace(messages=messages), sleep=delays.append
    )

    assert client.complete("prompt") == "completion"
    assert delays == [2]
    assert len(messages.calls) == 2


def test_transient_failures_stop_at_configured_budget() -> None:
    connection_failure = type("APIConnectionError", (Exception,), {})
    messages = _Messages([connection_failure(), connection_failure(), connection_failure()])
    delays = []
    client = AnthropicLLMClient(
        _config(), sdk_client=SimpleNamespace(messages=messages), sleep=delays.append
    )

    with pytest.raises(LLMClientError, match="completion failed"):
        client.complete("prompt")
    assert len(messages.calls) == 3
    assert delays == [1, 2]


def test_non_transient_api_errors_are_not_retried() -> None:
    bad_request = RuntimeError("invalid request")
    bad_request.status_code = 400
    messages = _Messages([bad_request])
    delays = []
    client = AnthropicLLMClient(
        _config(), sdk_client=SimpleNamespace(messages=messages), sleep=delays.append
    )

    with pytest.raises(LLMClientError):
        client.complete("prompt")
    assert len(messages.calls) == 1
    assert not delays


def test_rate_limit_without_retry_after_uses_bounded_backoff() -> None:
    rate_limit = RuntimeError("usage tier cap reached")
    rate_limit.response = SimpleNamespace(status_code=429, headers={})
    messages = _Messages([rate_limit, _response()])
    delays = []
    client = AnthropicLLMClient(
        _config(), sdk_client=SimpleNamespace(messages=messages), sleep=delays.append
    )

    assert client.complete("prompt") == "completion"
    assert len(messages.calls) == 2
    assert delays == [1]


def test_rate_limit_retry_after_is_capped_by_configured_maximum() -> None:
    rate_limit = RuntimeError("rate limited")
    rate_limit.response = SimpleNamespace(status_code=429, headers={"retry-after": "30"})
    messages = _Messages([rate_limit, _response()])
    delays = []
    client = AnthropicLLMClient(
        _config(retry_max_delay_seconds=4),
        sdk_client=SimpleNamespace(messages=messages),
        sleep=delays.append,
    )

    assert client.complete("prompt") == "completion"
    assert delays == [4]


@pytest.mark.parametrize("retry_after", ["NaN", "inf", "-2", "not-a-number"])
def test_malformed_retry_after_uses_finite_nonnegative_backoff(retry_after: str) -> None:
    rate_limit = RuntimeError("rate limited")
    rate_limit.response = SimpleNamespace(
        status_code=429, headers={"retry-after": retry_after}
    )
    messages = _Messages([rate_limit, _response()])
    delays = []
    client = AnthropicLLMClient(
        _config(), sdk_client=SimpleNamespace(messages=messages), sleep=delays.append
    )

    assert client.complete("prompt") == "completion"
    assert delays == [1]


def test_missing_key_fails_before_client_creation() -> None:
    with pytest.raises(ValueError, match="RAG_PHY_LLM__API_KEY"):
        AnthropicLLMClient(_config(api_key=None))


def test_llm_key_comes_from_environment_and_is_excluded_from_serialization() -> None:
    config = load_config(
        "config/app.yaml", environ={"RAG_PHY_LLM__API_KEY": "test-env-secret"}
    )

    assert config.llm is not None
    assert config.llm.api_key == "test-env-secret"
    assert config.llm.model_name == "claude-sonnet-5-5"
    assert config.llm.input_cost_usd_per_million_tokens == 2
    assert config.llm.output_cost_usd_per_million_tokens == 10
    assert "test-env-secret" not in str(config.model_dump())
