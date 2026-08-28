# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Helper functions for constructing scheduler specs and events."""

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tokenspeed_scheduler import (
    Cache,
    CacheGroupConfig,
    CacheGroupFamily,
    CacheRetention,
    CacheTransferPolicy,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    SchedulerConfig,
)

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    require_positive_int,
)

_CACHE_EVENT_TYPES = {
    "WriteBackDoneEvent": Cache.WriteBackDoneEvent,
}
# Emitted only by the host tier. Keep the lookup guarded so an older extension
# still imports this module and fails later with a targeted compatibility error.
if hasattr(Cache, "LoadBackDoneEvent"):
    _CACHE_EVENT_TYPES["LoadBackDoneEvent"] = Cache.LoadBackDoneEvent
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

# Pool-spec string -> scheduler enum (pool_to_cache_groups).
_RETENTION_MAP = {
    "full_history": CacheRetention.FullHistory,
    "sliding_window": CacheRetention.SlidingWindow,
}
_FAMILY_MAP = {
    "history": CacheGroupFamily.History,
    "state": CacheGroupFamily.State,
}
_TRANSFER_POLICY_MAP = {
    "full_suffix": CacheTransferPolicy.FullSuffix,
    "latest_snapshot": CacheTransferPolicy.LatestSnapshot,
}


@dataclass(frozen=True)
class SchedulerCacheGeometry:
    prefix_granularity: int
    num_device_pages: int
    num_usable_pages: int
    token_capacity: int


def scheduler_cache_geometry_from_pool(pool: Any) -> SchedulerCacheGeometry:
    """Project the arena's published contract onto scheduler page counts.

    The contract is the only source: the arena publishes it for every pool,
    so there is no pool-side copy to prefer and no server-args fallback to
    reconcile against.
    """
    contract = pool.arena.runtime_contract
    num_lcm_blocks = require_positive_int(
        "contract.num_lcm_blocks", contract.num_lcm_blocks
    )
    return SchedulerCacheGeometry(
        prefix_granularity=contract.prefix_granularity,
        # Parent 0 is reserved as the null LCM block.
        num_device_pages=num_lcm_blocks + 1,
        num_usable_pages=num_lcm_blocks,
        token_capacity=contract.token_capacity,
    )


def aligned_max_scheduled_tokens(
    max_scheduled_tokens: int,
    cache_groups,
) -> int:
    """Floor ``max_scheduled_tokens`` to the state-snapshot grain, if any.

    Recurrent-state groups (family=State, retention=FullHistory — the C++
    ``final_state_manager`` criterion) register their state snapshot only when
    a prefill chunk ends exactly on a CacheBlock boundary
    (``RegistersAlignedFinalPageOnly``); interior boundaries never received a
    state write. A chunk size that is not a multiple of every such group's
    CacheBlock token span therefore never registers a state block. Since the
    admission probe takes the minimum hit across groups, prefix-cache reuse
    silently degrades to zero for the whole model.

    Args:
        max_scheduled_tokens: Requested per-step token budget
            (``--chunked-prefill-size``).
        cache_groups: Scheduler ``CacheGroupConfig`` sequence, or
            None/empty when the model declares no cache groups.
    Returns:
        ``max_scheduled_tokens`` floored to the LCM of the state groups'
        CacheBlock token spans. Returned unchanged when no such group exists
        or the value is already aligned.

    Raises:
        ValueError: If the configured budget is smaller than one state CacheBlock.
            Raising is safer than increasing a limit that may already have
            sized executor buffers.
    """
    require_positive_int("max_scheduled_tokens", max_scheduled_tokens)
    grain = 1
    for group in cache_groups or ():
        if group.family != CacheGroupFamily.State:
            continue
        if group.retention == CacheRetention.SlidingWindow:
            continue
        grain = math.lcm(
            grain,
            int(group.rows_per_page) * int(group.entry_stride_tokens),
        )
    if grain == 1:
        return max_scheduled_tokens
    if max_scheduled_tokens < grain:
        raise ValueError(
            "chunked_prefill_size must be at least one recurrent-state CacheBlock: "
            f"got {max_scheduled_tokens}, minimum {grain}"
        )
    return max_scheduled_tokens - max_scheduled_tokens % grain


def make_spec(rid: str, tokens: list[int], max_new_tokens: int = 0) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = tokens
    spec.max_new_tokens = max_new_tokens
    return spec


def make_config(
    num_device_pages: int,
    max_scheduled_tokens: int,
    max_batch_size: int,
    prefix_granularity: int,
    num_host_pages: int,
    disable_l2_cache: bool,
    enable_l3_storage: bool,
    role: str,
    enable_kv_cache_events: bool = False,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    disable_prefix_cache: bool = False,
    cache_groups: Sequence["CacheGroupConfig"] | None = None,
    enable_mixed_prefill_decode: bool = False,
    prefix_replay_tokens: int = 0,
) -> SchedulerConfig:
    if not 0 <= prefix_replay_tokens <= (1 << 31) - 1:
        raise ValueError(
            "prefix_replay_tokens must fit a non-negative int32; "
            f"got {prefix_replay_tokens}."
        )
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.prefix_granularity = prefix_granularity

    cfg.num_host_pages = num_host_pages
    cfg.enable_l3_storage = enable_l3_storage
    cfg.enable_kv_cache_events = enable_kv_cache_events

    if role == "prefill":
        cfg.role = SchedulerConfig.Role.P
    elif role == "decode":
        cfg.role = SchedulerConfig.Role.D
    else:
        cfg.role = SchedulerConfig.Role.Fused
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = overlap_schedule_depth
    cfg.disable_prefix_cache = disable_prefix_cache
    cfg.prefix_replay_tokens = prefix_replay_tokens
    cfg.disable_l2_cache = disable_l2_cache

    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode
    if cache_groups:
        cfg.cache_groups = list(cache_groups)
    return cfg


def pool_to_cache_groups(pool: Any) -> list:
    """Convert a cache's published contract into scheduler group configs."""
    # The arena is the sole publisher, so there is exactly one source here --
    # no fallback to pool-side copies of the same specs.
    contract = pool.arena.runtime_contract
    specs = contract.group_specs
    counts = contract.group_page_counts
    packing = contract.group_packing
    out = []
    for spec in specs:
        retention = _RETENTION_MAP.get(spec.retention)
        if retention is None:
            raise ValueError(
                f"pool_to_cache_groups: unsupported retention "
                f"{spec.retention!r} for group {spec.group_id!r}"
            )
        family = _FAMILY_MAP.get(spec.family)
        if family is None:
            raise ValueError(
                f"pool_to_cache_groups: unsupported family "
                f"{spec.family!r} for group {spec.group_id!r}"
            )
        # The C++ scheduler config only carries row geometry. A snapshot-state
        # group folds to (rows=checkpoint_granularity, stride=1) at this single
        # mapping point; both encode the same block_granularity.
        if spec.checkpoint_granularity is not None:
            rows_per_page = int(spec.checkpoint_granularity)
            entry_stride_tokens = 1
        else:
            rows_per_page = int(spec.rows_per_page)
            entry_stride_tokens = int(spec.entry_stride_tokens)
        kwargs = dict(
            group_id=spec.group_id,
            rows_per_page=rows_per_page,
            entry_stride_tokens=entry_stride_tokens,
            total_pages=int(counts[spec.group_id]),
            retention=retention,
            family=family,
            cache_blocks_per_lcm_block=int(packing[spec.group_id]),
        )
        transfer_policy = spec.transfer_policy
        if transfer_policy is not None:
            mapped_policy = _TRANSFER_POLICY_MAP.get(transfer_policy)
            if mapped_policy is None:
                raise ValueError(
                    "pool_to_cache_groups: unsupported transfer policy "
                    f"{transfer_policy!r} for group {spec.group_id!r}"
                )
            kwargs["transfer_policy"] = mapped_policy
        if spec.retention == "sliding_window":
            kwargs["sliding_window_tokens"] = int(spec.sliding_window_tokens)
        out.append(CacheGroupConfig(**kwargs))
    return out


def should_use_overlap_schedule(
    *,
    disable_overlap_schedule: bool,
    disaggregation_mode: str,
) -> bool:
    """Return whether the runtime can use the overlapped scheduler loop."""

    if disable_overlap_schedule:
        return False
    if disaggregation_mode in ("prefill", "encode"):
        # prefill drain + KV send run only on the non-overlap loop; encode has no LM loop.
        return False
    return True


def resolve_dspark_prefix_replay_tokens(
    *,
    speculative_algorithm: str | None,
    enable_prefix_caching: bool,
    enable_kvstore: bool,
    disaggregation_mode: str,
    draft_model_path_use_base: bool,
    draft_model_config: Any | None,
) -> int:
    """Resolve the prompt tail needed to rebuild DSpark runtime state.

    DeepSeek V4 DSpark advertises the requirement through its draft
    ``ModelConfig``. Same-checkpoint DSpark configurations without that
    capability remain fail-closed. External generic DSpark configurations keep
    their existing scheduler behavior until they advertise an equivalent
    contract.
    """

    if not enable_prefix_caching or speculative_algorithm != "DSPARK":
        return 0
    if draft_model_config is None:
        raise ValueError(
            "DSPARK prefix caching requires a resolved draft model configuration."
        )

    replay_tokens = getattr(draft_model_config, "dspark_prefix_replay_tokens", None)
    if replay_tokens is None:
        if draft_model_path_use_base:
            raise ValueError(
                "DSPARK same-checkpoint prefix caching requires a draft model "
                "that advertises captured-context replay support."
            )
        return 0

    replay_tokens = int(replay_tokens)
    if not 0 < replay_tokens <= (1 << 31) - 1:
        raise ValueError(
            "DSPARK captured-context replay requirement must fit a positive int32; "
            f"got {replay_tokens}."
        )
    if enable_kvstore:
        raise ValueError(
            "DSPARK captured-context replay does not support KVStore; "
            "use --disable-kvstore."
        )
    if disaggregation_mode != "null":
        raise ValueError(
            "DSPARK captured-context replay does not support disaggregated "
            f"serving; got role {disaggregation_mode!r}."
        )
    return replay_tokens


def make_extend_result_event(
    request_id: str,
    tokens: Sequence[int] = (),
    spec_candidate_ids: Sequence[int] | None = None,
) -> "ForwardEvent.ExtendResult":
    fe = ForwardEvent.ExtendResult()
    fe.request_id = request_id
    fe.tokens = list(tokens)
    if spec_candidate_ids:
        # P-side final chunk: the drafter candidates ride to the scheduler so
        # its remote-decode operation is self-contained.
        fe.spec_candidate_ids = list(spec_candidate_ids)
    return fe


def make_finish_event(request_id: str) -> "ForwardEvent.Finish":
    fe = ForwardEvent.Finish()
    fe.request_id = request_id
    return fe


def make_abort_event(request_id: str) -> "ForwardEvent.Abort":
    """Finish without caching: AbortEvent skips the single-table-tree insert and
    never enters Draining, so no host-KV writeback (target or draft) is
    issued. Used for numerically-corrupted requests whose KV must not be
    reused.
    """
    fe = ForwardEvent.Abort()
    fe.request_id = request_id
    return fe


def make_update_reserve_tokens_event(request_id: str, new_reserve_num_tokens: int):
    fe = ForwardEvent.UpdateReserveNumTokens()
    fe.request_id = request_id
    fe.reserve_num_tokens_in_next_schedule_event = new_reserve_num_tokens
    return fe


def scheduler_cache_group_pages(scheduler):
    """Return a ``group_id -> (total, available)`` page-count query.

    Two counter reads, bound to the scheduler once, so the batch logger holds
    a query rather than a reference to the loop that owns the scheduler.

    Args:
        scheduler: The engine's C++ scheduler.

    Returns:
        A callable taking a cache-group id and returning its total and
        available page counts.
    """

    def pages(group_id: str) -> tuple[int, int]:
        return (
            scheduler.cache_group_total_pages(group_id),
            scheduler.cache_group_available_pages(group_id),
        )

    return pages


def advance_scheduler(scheduler, events: list) -> None:
    """Feed completion events (forward results or cache-op results) back into
    the C++ scheduler. The ONLY caller of ``scheduler.advance``.

    Design principle: this is only invoked explicitly and directly from the
    ``EventLoop.event_loop`` body — never from helpers. Helpers RETURN their
    events; the loop applies them, so every scheduler state change is visible
    by reading the loop alone.
    """
    ec = ExecutionEvent()
    for event in events:
        ec.add_event(event)
    scheduler.advance(ec)


def cache_event_to_payload(event) -> dict:
    kind = type(event).__name__
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    return {
        "kind": kind,
        "op_id": int(event.op_id),
    }


def cache_event_from_payload(payload: dict):
    kind = payload["kind"]
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    event = _CACHE_EVENT_TYPES[kind]()
    event.op_id = int(payload["op_id"])
    return event


def cache_event_key(payload: dict) -> tuple[str, int]:
    return payload["kind"], int(payload["op_id"])


def pop_common_cache_event_payloads(
    pending_payloads_by_rank: Sequence[Sequence[dict]],
) -> list[dict]:
    if not pending_payloads_by_rank:
        return []

    rank_maps = []
    common_keys = None
    for payloads in pending_payloads_by_rank:
        rank_map = {cache_event_key(payload): payload for payload in payloads}
        rank_maps.append(rank_map)
        rank_keys = set(rank_map)
        common_keys = rank_keys if common_keys is None else common_keys & rank_keys
        if not common_keys:
            return []

    ready_payloads = []
    for key in sorted(common_keys, key=lambda item: (item[1], item[0])):
        ready_payloads.append(dict(rank_maps[0][key]))
    return ready_payloads


def cache_sync_debug_enabled() -> bool:
    value = os.getenv("TS_DEBUG_CACHE_SYNC", "")
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Bridge the per-group block tables to GPU int32 tensors: absolute
    page indices, null hole = 0 preserved, ragged-row padding -1. No
    base-offset companion -- the cache path never compacts.

    All groups stage into ONE pinned buffer and ride ONE H2D copy; the
    returned per-group views share a single storage, which is the
    precondition of the backends' one-launch packed replay fill
    (``_try_packed_group_unpack``). Per-group uploads would fail its
    same-storage check and fall back to per-group copy/fill chains
    (~40 tiny transfers per decode step).

    Args:
        forward_op: Scheduler forward operation exporting CPU NumPy tables.
        device: Destination device for the packed tensor.
        num_reqs: Optional expected row count for every group.
        expected_group_ids: Optional contract order and exact key set.
        max_page_id: Optional common inclusive upper bound for page IDs.
        max_page_ids: Optional per-group inclusive upper bounds.

    Returns:
        Per-group tensor views in ``expected_group_ids`` order when supplied,
        otherwise preserving producer order.

    Raises:
        ValueError: If strict contract validation fails before device transfer.
    """
    if max_page_id is not None and max_page_ids is not None:
        raise ValueError("pass max_page_id or max_page_ids, not both")
    strict_validation = (
        expected_group_ids is not None
        or max_page_id is not None
        or max_page_ids is not None
    )
    if strict_validation and num_reqs is not None:
        require_positive_int("num_reqs", num_reqs)
    if expected_group_ids is not None:
        seen_group_ids: set[str] = set()
        duplicate_group_ids: set[str] = set()
        for group_id in expected_group_ids:
            if group_id in seen_group_ids:
                duplicate_group_ids.add(group_id)
            seen_group_ids.add(group_id)
        if duplicate_group_ids:
            raise ValueError(
                f"expected_group_ids contains duplicates: {sorted(duplicate_group_ids)}"
            )
    array_items = [
        (str(key), arr) for key, arr in forward_op.block_tables_arrays().items()
    ]
    if strict_validation:
        normalized_group_ids: set[str] = set()
        collided_ids: set[str] = set()
        for group_id, _ in array_items:
            if group_id in normalized_group_ids:
                collided_ids.add(group_id)
            normalized_group_ids.add(group_id)
        if collided_ids:
            raise ValueError(
                "cache group keys collide after string normalization: "
                f"{sorted(collided_ids)}"
            )
    arrays_by_id = dict(array_items)
    if expected_group_ids is not None:
        actual = set(arrays_by_id)
        expected = set(expected_group_ids)
        if actual != expected:
            raise ValueError(
                f"cache group keys disagree: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        ordered_items = [
            (group_id, arrays_by_id[group_id]) for group_id in expected_group_ids
        ]
    else:
        ordered_items = array_items
    if strict_validation:
        for group_id, arr in ordered_items:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"cache group {group_id!r} must be a NumPy array")
            if arr.dtype != np.int32:
                raise ValueError(f"cache group {group_id!r} must use int32")
            if arr.ndim != 2:
                raise ValueError(f"cache group {group_id!r} has invalid shape")
            if arr.shape[0] == 0:
                raise ValueError(f"cache group {group_id!r} has zero rows")
            # rows-vs-num_reqs is checked once in the packing loop below.
            if arr.shape[1] == 0:
                raise ValueError(f"cache group {group_id!r} has zero width")
            group_max_page_id = (
                max_page_ids.get(group_id) if max_page_ids is not None else max_page_id
            )
            if max_page_ids is not None and group_max_page_id is None:
                raise ValueError(f"max_page_ids is missing cache group {group_id!r}")
            if group_max_page_id is not None:
                invalid = (arr < -1) | (arr > group_max_page_id)
                if bool(invalid.any()):
                    raise ValueError(
                        f"cache group {group_id!r} contains a page ID outside "
                        f"-1..{group_max_page_id}"
                    )
    device = torch.device(device) if isinstance(device, str) else device
    out: dict[str, torch.Tensor] = {}
    packable: list[tuple[str, Any, int]] = []
    total = 0
    for key, arr in ordered_items:
        if num_reqs is not None and arr.shape[0] != num_reqs:
            raise ValueError(
                f"block_tables_arrays[{key}] has {arr.shape[0]} rows "
                f"but forward op reported num_reqs={num_reqs}"
            )
        if arr.shape[0] == 0:
            continue
        if arr.shape[1] == 0:
            # Kept out of the pack: a zero-width table must stay loud in the
            # replay fill's cols >= 1 assert, not be silently tail-padded.
            out[key] = torch.empty((arr.shape[0], 0), dtype=torch.int32, device=device)
            continue
        packable.append((key, arr, total))
        total += arr.shape[0] * arr.shape[1]
    if not packable:
        return out
    # Fresh pinned stage per step (event-fenced; reuse races overlap).
    # arr is a read-only zero-copy view over the C++ buffer; np.copyto
    # reads it into our own writable pinned tensor (never writes back).
    staged = torch.empty(total, dtype=torch.int32, pin_memory=device.type == "cuda")
    staged_np = staged.numpy()
    for key, arr, offset in packable:
        np.copyto(staged_np[offset : offset + arr.size].reshape(arr.shape), arr)
    packed = staged.to(device, non_blocking=True)
    for key, arr, offset in packable:
        out[key] = packed[offset : offset + arr.size].view(arr.shape[0], arr.shape[1])
    return out


def _classify_param(name: str) -> str:
    """Bucket a parameter/buffer name into a weight group for the memory
    summary. Names follow the Kimi-K3 / DeepSeek module layout."""
    if "self_attn" in name or ".attn." in name or "kv_a" in name or "q_a" in name:
        return "attention_weights"
    if (
        "experts" in name
        or "block_sparse_moe" in name
        or "routed_expert" in name
        or "gate" in name
    ) and "shared_experts" not in name:
        return "moe_weights"
    if (
        "mlp" in name
        or "shared_experts" in name
        or "down_proj" in name
        or "up_proj" in name
        or "gate_proj" in name
    ):
        return "dense_mlp_weights"
    return "other_weights"


def _kv_pool_bytes(*pools) -> int:
    """Best-effort total KV-buffer bytes across the given pools, deduped.

    Target and draft are two compute views of the ONE arena (draft KV lives
    in the target's arena), so counting both naively would double the KV.
    Dedupe by arena identity -- the allocation each view reports on.

    Return types differ (int, or a tuple like MSA's (kv, index); a hybrid pool
    may nest several) -- sum any numeric leaves.
    """
    seen: set[int] = set()
    total = 0
    for pool in pools:
        if pool is None:
            continue
        arena = getattr(pool, "arena", pool)
        if id(arena) in seen:
            continue
        seen.add(id(arena))
        getter = getattr(pool, "get_kv_size_bytes", None)
        if getter is None:
            continue
        try:
            result = getter()
        except Exception:
            continue

        def _sum(x):
            if isinstance(x, (int, float)):
                return int(x)
            if isinstance(x, (tuple, list)):
                return sum(_sum(e) for e in x)
            return 0

        total += _sum(result)
    return total


def log_gpu_memory_summary(
    model,
    gpu_id: int,
    rank: int,
    logger,
    draft_model=None,
    kv_pool=None,
    draft_kv_pool=None,
) -> None:
    """Log a per-rank GPU memory breakdown after model + KV pool are built.

    Weight groups are summed from the model's parameters and buffers (deduped
    by storage pointer). A draft model (speculative decoding) is summed
    separately into its own row. KV cache / CUDA graphs / non-torch (context,
    NCCL, DeepEP) are derived from the torch allocator and driver views, so the
    summary is backend-agnostic. Best-effort: never raises into startup.
    """
    try:
        GB = 1024**3
        groups = {
            "attention_weights": 0,
            "moe_weights": 0,
            "dense_mlp_weights": 0,
            "other_weights": 0,
        }
        seen: set[int] = set()

        def _accumulate(name, tensor, sink):
            if tensor is None or not tensor.is_cuda:
                return
            ptr = tensor.data_ptr()
            if ptr in seen:
                return
            seen.add(ptr)
            sink[_classify_param(name)] += tensor.numel() * tensor.element_size()

        for n, p in model.named_parameters():
            _accumulate(n, p, groups)
        for n, b in model.named_buffers():
            _accumulate(n, b, groups)

        weights_total = sum(groups.values()) / GB

        # Draft model (MTP / DSpark) weights, deduped against target: shared
        # tensors (e.g. embed/lm_head lent from the target) are already in
        # ``seen`` and won't be double-counted.
        draft_total = 0
        if draft_model is not None:
            draft_sink = {k: 0 for k in groups}
            for n, p in draft_model.named_parameters():
                _accumulate(n, p, draft_sink)
            for n, b in draft_model.named_buffers():
                _accumulate(n, b, draft_sink)
            draft_total = sum(draft_sink.values())
        draft_gb = draft_total / GB

        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id) / GB
        reserved = torch.cuda.memory_reserved(gpu_id) / GB
        device_total = total_bytes / GB
        device_free = free_bytes / GB
        device_used = device_total - device_free
        # KV pool bytes read straight from the pool buffers. Draft KV lives in
        # the target's merged arena (its pool is a view), so dedupe to count once.
        kv_cache_gb = _kv_pool_bytes(kv_pool, draft_kv_pool) / GB
        # Allocated beyond the classified target+draft weights and the KV pool is
        # activations + captured-graph private pools; non-torch is
        # context/NCCL/DeepEP.
        activations_and_graphs = max(
            0.0, allocated - weights_total - draft_gb - kv_cache_gb
        )
        non_torch = max(0.0, device_used - reserved)

        rows = [
            ("Attention weights", groups["attention_weights"] / GB),
            ("MoE weights", groups["moe_weights"] / GB),
            ("Dense/MLP weights", groups["dense_mlp_weights"] / GB),
            ("Other weights (embed/head/norm)", groups["other_weights"] / GB),
        ]
        if draft_model is not None:
            rows.append(("Draft model weights", draft_gb))
        rows += [
            ("KV cache", kv_cache_gb),
            ("Activations + CUDA graphs", activations_and_graphs),
            ("Torch allocated (total)", allocated),
            ("Torch reserved (allocator pool)", reserved),
            ("Non-torch (context/NCCL/DeepEP)", non_torch),
            ("Device used (nvidia-smi view)", device_used),
            ("Device free", device_free),
            ("Device total", device_total),
        ]
        name_width = max(len(n) for n, _ in rows)
        sep = "+" + "-" * (name_width + 2) + "+" + "-" * 12 + "+"
        lines = [
            f"GPU memory summary (rank {rank}, gpu {gpu_id}, GB):",
            sep,
            f"| {'Component'.ljust(name_width)} | {'GB'.rjust(10)} |",
            sep,
        ]
        for n, v in rows:
            lines.append(f"| {n.ljust(name_width)} | {v:10.2f} |")
        lines.append(sep)
        logger.info("\n".join(lines))
    except Exception:
        logger.warning("Failed to log GPU memory summary", exc_info=True)
