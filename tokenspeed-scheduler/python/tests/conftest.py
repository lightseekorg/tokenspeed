"""Shared helpers for cache-group scheduler binding tests."""

from __future__ import annotations

try:
    import tokenspeed_scheduler as ts
except ImportError:  # test modules importorskip on their own
    ts = None

# Kimi-K3 shape: cfg.prefix_granularity == P (1536 in production; ratio semantics
# identical at P=2).
PAGE = 2
K3_HISTORY_GROUP = "full_attention"
K3_STATE_GROUPS = ("linear_attention_0", "linear_attention_1", "linear_attention_2")
K3_GROUP_IDS = (K3_HISTORY_GROUP,) + K3_STATE_GROUPS


def _make_k3_config() -> "ts.SchedulerConfig":
    """Kimi-K3-shaped 4-group config: one full-attention History group plus
    three linear-attention State groups. A State group has no token span
    (block_granularity 0): its table is two slots, the input checkpoint and
    the live state, each one page -- exactly the production ratio."""
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = PAGE
    cfg.num_device_pages = 33  # page 0 = null sentinel, 32 usable
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=group_id,
            block_granularity=(
                cfg.prefix_granularity if group_id == K3_HISTORY_GROUP else 0
            ),
            total_pages=cfg.num_device_pages,
            attn_kind=(
                ts.AttnKind.Full if group_id == K3_HISTORY_GROUP else ts.AttnKind.State
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


def _find_forward_op(plan):
    for op in plan.forward:
        if dict(op.block_tables):
            return op
    return None


def _positive(row) -> list[int]:
    return [page for page in row if page > 0]


def storage_ops(plan):
    """The Host-only commands in a scheduler's unified cache plan."""
    Cache = ts.Cache

    types = (
        Cache.StorageQueryOp,
        Cache.StorageGetOp,
        Cache.StoragePutOp,
        Cache.StorageCancelOp,
    )
    return [operation for operation in plan.cache if isinstance(operation, types)]
