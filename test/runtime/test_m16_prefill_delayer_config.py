"""CPU-only checks for experimental M16 prefill-delayer config plumbing."""

from __future__ import annotations

import pytest

pytest.importorskip("tokenspeed_scheduler")

from tokenspeed_scheduler import SchedulerConfig

from tokenspeed.runtime.engine.scheduler_utils import make_config


def _make_config(*, enabled: bool = False):
    return make_config(
        num_device_pages=32,
        max_scheduled_tokens=64,
        max_batch_size=8,
        prefix_granularity=2,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        enable_experimental_m16_prefill_delayer=enabled,
    )


def test_binding_defaults_experimental_m16_prefill_delayer_off() -> None:
    assert SchedulerConfig().enable_experimental_m16_prefill_delayer is False
    assert _make_config().enable_experimental_m16_prefill_delayer is False


def test_make_config_enables_experimental_m16_prefill_delayer() -> None:
    assert _make_config(enabled=True).enable_experimental_m16_prefill_delayer is True
