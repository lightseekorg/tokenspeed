"""Shared helpers for the flat KV-cache binding tests.

Provides the Kimi-K3-shaped 4-group scheduler config, tiny request/event
drivers, and the flat-build skip probe used by ``test_flat_kvcache.py``.

The nanobind extension is build-identical between the radix and flat
builds -- ``FlatForwardOp.flat_block_tables`` is always exposed -- but it
is only *populated* when built with ``TOKENSPEED_FLAT_KVCACHE=ON``.
``requires_flat_build`` probes that behaviorally and skips otherwise.
"""

from __future__ import annotations

import pytest

try:
    import tokenspeed_scheduler as ts
except ImportError:  # test modules importorskip on their own
    ts = None

# Kimi-K3 shape: cfg.block_size == P (1536 in production; ratio semantics
# identical at P=2).
PAGE = 2
K3_HISTORY_GROUP = "full_attention"
K3_STATE_GROUPS = ("linear_attention_0", "linear_attention_1", "linear_attention_2")
K3_GROUP_IDS = (K3_HISTORY_GROUP,) + K3_STATE_GROUPS


def _make_k3_config() -> "ts.SchedulerConfig":
    """Kimi-K3-shaped 4-group config: one full-attention History group plus
    three linear-attention State groups with FullHistory retention --
    MakeSpecsFromConfig maps State+FullHistory to AttnKind::kMambaState ->
    MambaStateManager (SwaManager window=2), i.e. one page per
    state-snapshot slot, exactly the production ratio."""
    cfg = ts.SchedulerConfig()
    cfg.block_size = PAGE
    cfg.num_device_pages = 33  # page 0 = null sentinel, 32 usable
    cfg.num_host_pages = 0
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.paged_cache_groups = [
        ts.PagedCacheGroupConfig(
            group_id=group_id,
            rows_per_page=cfg.block_size,
            entry_stride_tokens=1,
            total_pages=cfg.num_device_pages,
            retention=ts.PagedCacheRetention.FullHistory,
            family=(
                ts.PagedCacheGroupFamily.History
                if group_id == K3_HISTORY_GROUP
                else ts.PagedCacheGroupFamily.State
            ),
        )
        for group_id in K3_GROUP_IDS
    ]
    return cfg


def _spec(request_id: str, tokens: list[int]) -> "ts.RequestSpec":
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    return spec


def _advance(scheduler, request_id: str, tokens: list[int]) -> None:
    event = ts.ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = tokens
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _finish(scheduler, request_id: str) -> None:
    event = ts.ForwardEvent.Finish()
    event.request_id = request_id
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _find_flat_op(plan):
    for op in plan.forward:
        if dict(op.flat_block_tables):
            return op
    return None


def _positive(row) -> list[int]:
    return [page for page in row if page > 0]


def _flat_build_available() -> bool:
    """Behavioral probe: the extension is flat-built iff a prefilled request's
    FlatForwardOp carries a non-empty ``flat_block_tables``. On the radix
    build the field is exposed but stays empty, so this returns False there
    *without raising* (an unexpected raise means a real regression and should
    surface as a collection error, not a silent skip)."""
    if ts is None:
        return False
    scheduler = ts.Scheduler(_make_k3_config())
    scheduler.submit_requests([_spec("probe", list(range(1, 2 * PAGE + 1)))])
    return _find_flat_op(scheduler.next_execution_plan()) is not None


requires_flat_build = pytest.mark.skipif(
    not _flat_build_available(),
    reason="extension not built with TOKENSPEED_FLAT_KVCACHE=ON",
)
