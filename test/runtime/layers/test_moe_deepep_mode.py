from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.moe.utils import DeepEPMode, use_deepep_low_latency


def _ctx(forward_mode: ForwardMode, all_decode_or_idle: bool) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=forward_mode, all_decode_or_idle=all_decode_or_idle
    )


@pytest.mark.parametrize(
    "forward_mode,expected",
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.IDLE, True),
        (ForwardMode.EXTEND, False),
        (ForwardMode.MIXED, False),
    ],
)
def test_without_dp_attention_the_local_forward_mode_decides(
    forward_mode: ForwardMode, expected: bool
) -> None:
    # all_decode_or_idle is only populated when DP attention gathers it, so a
    # non-DP deployment must not depend on it -- including at EP > 1, where the
    # whole group still forwards one batch.
    ctx = _ctx(forward_mode, all_decode_or_idle=False)
    assert use_deepep_low_latency(ctx, attn_dp_size=1) is expected


@pytest.mark.parametrize("all_decode_or_idle", [True, False])
def test_with_dp_attention_the_replicated_flag_decides(
    all_decode_or_idle: bool,
) -> None:
    # Dispatch legs are collectives: a rank that is locally decoding while a peer
    # extends must still take the normal legs, or the group deadlocks.
    ctx = _ctx(ForwardMode.DECODE, all_decode_or_idle)
    assert use_deepep_low_latency(ctx, attn_dp_size=2) is all_decode_or_idle


def test_decode_graph_capture_context_selects_low_latency() -> None:
    """A decode graph must record the low-latency legs, at any DP degree.

    ``ForwardStepRunner._capture_one`` reports ``all_decode_or_idle=True`` because
    the replay guard only ever replays such a graph for an all-decode forward.
    If capture instead recorded the normal legs, their host-side receive counts
    would deadlock the capture.
    """
    capture_ctx = _ctx(ForwardMode.DECODE, all_decode_or_idle=True)
    for attn_dp_size in (1, 2, 8):
        assert use_deepep_low_latency(capture_ctx, attn_dp_size) is True


def test_missing_forward_mode_falls_back_to_normal() -> None:
    assert use_deepep_low_latency(_ctx(None, False), attn_dp_size=1) is False


@pytest.mark.parametrize(
    "mode,normal,low_latency",
    [
        (DeepEPMode.NORMAL, True, False),
        (DeepEPMode.LOW_LATENCY, False, True),
        (DeepEPMode.AUTO, True, True),
    ],
)
def test_deepep_mode_enabled_legs(
    mode: DeepEPMode, normal: bool, low_latency: bool
) -> None:
    assert mode.enable_normal() is normal
    assert mode.enable_low_latency() is low_latency
