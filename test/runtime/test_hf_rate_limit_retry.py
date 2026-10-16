"""Tests for Hugging Face Hub 429 rate-limit retry behavior."""

import os
import sys

import httpx
import pytest
from huggingface_hub.utils import HfHubHTTPError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime import hf_hub_retry
from tokenspeed.runtime.hf_hub_retry import call_with_hf_rate_limit_retry


def _hf_error(status: int, retry_after: str | None = None) -> HfHubHTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    request = httpx.Request("GET", "https://huggingface.co/api/models/acme/model")
    response = httpx.Response(status, headers=headers, request=request)
    return HfHubHTTPError(f"{status} error", response=response)


def _flaky(errors: list[HfHubHTTPError], result: str = "ok"):
    queue = list(errors)

    def call() -> str:
        if queue:
            raise queue.pop(0)
        return result

    return call


@pytest.fixture
def recorded_sleeps(monkeypatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(hf_hub_retry.time, "sleep", sleeps.append)
    return sleeps


def test_success_skips_retry(recorded_sleeps) -> None:
    assert call_with_hf_rate_limit_retry("testing", _flaky([])) == "ok"
    assert recorded_sleeps == []


def test_rate_limit_waits_for_retry_after_then_succeeds(recorded_sleeps) -> None:
    errors = [_hf_error(429, "30"), _hf_error(429, "30")]

    assert call_with_hf_rate_limit_retry("testing", _flaky(errors)) == "ok"
    assert len(recorded_sleeps) == 2
    for slept in recorded_sleeps:
        assert 30 <= slept <= 30 * 1.25


def test_non_rate_limit_error_propagates_immediately(recorded_sleeps) -> None:
    with pytest.raises(HfHubHTTPError):
        call_with_hf_rate_limit_retry("testing", _flaky([_hf_error(404)]))
    assert recorded_sleeps == []


def test_missing_retry_after_uses_exponential_backoff(recorded_sleeps) -> None:
    errors = [_hf_error(429), _hf_error(429), _hf_error(429)]

    assert call_with_hf_rate_limit_retry("testing", _flaky(errors)) == "ok"
    assert len(recorded_sleeps) == 3
    for slept, base in zip(recorded_sleeps, (5.0, 10.0, 20.0), strict=True):
        assert base <= slept <= base * 1.25


def test_unparseable_retry_after_uses_exponential_backoff(recorded_sleeps) -> None:
    errors = [_hf_error(429, "Thu, 01 Jan 1970 00:00:00 GMT")]

    assert call_with_hf_rate_limit_retry("testing", _flaky(errors)) == "ok"
    assert len(recorded_sleeps) == 1
    assert 5.0 <= recorded_sleeps[0] <= 5.0 * 1.25


def test_gives_up_once_retry_budget_is_exhausted(recorded_sleeps, monkeypatch) -> None:
    monkeypatch.setattr(hf_hub_retry, "HF_RATE_LIMIT_RETRY_BUDGET_S", 40.0)

    def always_rate_limited() -> str:
        raise _hf_error(429, "30")

    with pytest.raises(HfHubHTTPError):
        call_with_hf_rate_limit_retry("testing", always_rate_limited)
    assert len(recorded_sleeps) == 1
