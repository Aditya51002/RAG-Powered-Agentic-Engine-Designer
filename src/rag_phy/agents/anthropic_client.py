"""Anthropic implementation of the provider-neutral text completion protocol."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

from rag_phy.agents.design import LLMClientError
from rag_phy.config import LLMConfig

logger = logging.getLogger(__name__)


class _MessageAPI(Protocol):
    def create(self, **kwargs: Any) -> Any:
        """Send one Messages API request."""


class _AnthropicSDK(Protocol):
    messages: _MessageAPI


class AnthropicLLMClient:
    """Call Anthropic with bounded transport retries and token-cost telemetry.

    Malformed text/schema handling remains the caller's responsibility. This adapter retries
    only transient transport/API failures, using a separate configured retry budget.
    """

    _RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
    _RETRYABLE_EXCEPTION_NAMES = frozenset(
        {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"}
    )

    def __init__(
        self,
        config: LLMConfig,
        *,
        sdk_client: _AnthropicSDK | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an SDK client or bind a fake; fail before networking if the key is absent."""
        if not config.api_key or not config.api_key.strip():
            raise ValueError(
                "Anthropic API key is required; set RAG_PHY_LLM__API_KEY in the environment"
            )
        self._config = config
        self._sleep = sleep
        self._monotonic = monotonic
        self._sdk_client = sdk_client or self._create_sdk_client(config)

    @staticmethod
    def _create_sdk_client(config: LLMConfig) -> _AnthropicSDK:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support is optional; install rag-phy[llm] to use the live provider"
            ) from exc
        return Anthropic(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def complete(self, prompt: str) -> str:
        """Generate text, retrying transient failures independently of schema retries."""
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        started = self._monotonic()
        for attempt in range(self._config.transient_retry_attempts + 1):
            try:
                response = self._sdk_client.messages.create(
                    model=self._config.model_name,
                    max_tokens=self._config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = self._extract_text(response)
                elapsed_ms = (self._monotonic() - started) * 1000
                input_tokens = int(response.usage.input_tokens)
                output_tokens = int(response.usage.output_tokens)
                estimated_cost = (
                    input_tokens * self._config.input_cost_usd_per_million_tokens
                    + output_tokens * self._config.output_cost_usd_per_million_tokens
                ) / 1_000_000
                logger.info(
                    "Anthropic completion succeeded",
                    extra={
                        "provider": "anthropic",
                        "model": self._config.model_name,
                        "tokens_in": input_tokens,
                        "tokens_out": output_tokens,
                        "latency_ms": round(elapsed_ms, 3),
                        "estimated_cost_usd": estimated_cost,
                        "provider_request_id": getattr(response, "_request_id", None),
                        "transient_retry_count": attempt,
                    },
                )
                return text
            except Exception as exc:
                if not self._is_retryable(exc) or attempt >= self._config.transient_retry_attempts:
                    logger.exception(
                        "Anthropic completion failed",
                        extra={
                            "provider": "anthropic",
                            "model": self._config.model_name,
                            "transient_retry_count": attempt,
                        },
                    )
                    raise LLMClientError("Anthropic completion failed") from exc
                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "Retrying transient Anthropic API failure",
                    extra={
                        "provider": "anthropic",
                        "model": self._config.model_name,
                        "retry_number": attempt + 1,
                        "retry_delay_seconds": delay,
                        "status_code": self._status_code(exc),
                    },
                )
                self._sleep(delay)
        raise AssertionError("Retry loop exited without a result or exception")

    @staticmethod
    def _extract_text(response: Any) -> str:
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text" and isinstance(block.text, str)
        ]
        if not text_parts:
            raise LLMClientError("Anthropic response did not contain a text block")
        return "\n".join(text_parts)

    @classmethod
    def _status_code(cls, error: Exception) -> int | None:
        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(getattr(error, "response", None), "status_code", None)
        return status if isinstance(status, int) else None

    @classmethod
    def _is_retryable(cls, error: Exception) -> bool:
        status = cls._status_code(error)
        return (
            status in cls._RETRYABLE_STATUSES
            or type(error).__name__ in cls._RETRYABLE_EXCEPTION_NAMES
        )

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        headers = getattr(getattr(error, "response", None), "headers", {})
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after is None:
            requested_delay = self._config.retry_initial_delay_seconds * (2**attempt)
        else:
            try:
                requested_delay = float(retry_after)
            except (TypeError, ValueError):
                requested_delay = self._config.retry_initial_delay_seconds * (2**attempt)
            if not math.isfinite(requested_delay) or requested_delay < 0:
                requested_delay = self._config.retry_initial_delay_seconds * (2**attempt)
        return min(requested_delay, self._config.retry_max_delay_seconds)


__all__ = ["AnthropicLLMClient"]
