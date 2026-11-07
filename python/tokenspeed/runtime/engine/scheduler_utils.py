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

"""What the data plane and the rank setup read off the C++ scheduler: its
config, its cache geometry and groups, the cache ops a plan carries and
their acknowledgments, and the block tables a forward reads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
from tokenspeed_scheduler import (
    Cache,
    CacheGroupConfig,
    SchedulerConfig,
    StorageConfig,
)

from tokenspeed.runtime.cache.transfer.ops import (
    CacheTransfer,
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    require_positive_int,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    cache_group_config,
    scheduler_role,
)

# The Host cache's ACKs, as the scheduler binds them: an L2 write-back or
# load-back ticket, a retraction image's snapshot store, a restore. The data
# plane reports them to the Rust control plane as ``(kind, op_id)`` payloads.
_CACHE_EVENT_TYPES = {
    "WriteBackDoneEvent": Cache.WriteBackDoneEvent,
    "LoadBackDoneEvent": Cache.LoadBackDoneEvent,
    "SnapshotDoneEvent": Cache.SnapshotDoneEvent,
    "RestoreDoneEvent": Cache.RestoreDoneEvent,
}
# make_config's host_write_policy names; ServerArgs validates the flag.
_HOST_WRITE_POLICIES = {
    "write_through": SchedulerConfig.HostWritePolicy.WriteThrough,
    "write_back": SchedulerConfig.HostWritePolicy.WriteBack,
}


def engram_context_len(text_config) -> int:
    """Select caller-owned Engram inputs, not Qwen4's LCM-owned PLE state."""
    if not getattr(text_config, "engram_layer_ids", ()):
        return 0
    if getattr(text_config, "ngram_context_len", None) != 3:
        raise ValueError("Engram requires hf_text_config.ngram_context_len = 3")
    return text_config.ngram_context_len


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


def make_config(
    num_device_pages: int,
    max_scheduled_tokens: int,
    max_batch_size: int,
    prefix_granularity: int,
    num_host_pages: int,
    disable_l2_cache: bool,
    enable_l3_storage: bool,
    role: str,
    *,
    num_snapshot_pages: int,
    host_group_packing: Sequence[int],
    max_retracted_requests: int,
    storage_config: StorageConfig | None,
    host_write_policy: str,
    prefill_decode_interval: int,
    debug_force_retraction_interval: int = 0,
    enable_kv_cache_events: bool = False,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    disable_prefix_cache: bool = False,
    cache_groups: Sequence[CacheGroupConfig] | None = None,
    enable_mixed_prefill_decode: bool = False,
    prefix_replay_tokens: int = 0,
    state_checkpoint_in_forward: bool = False,
    demote_superseded_state_checkpoints: bool = False,
    drop_short_prompt_end_checkpoints: bool = False,
    max_kv_page_num: int = 0,
    checkpoint_retention_requests: int | None = None,
    checkpoint_retention_leaf_distance: int | None = None,
    prefill_chunk_yield_tokens: int = 0,
) -> SchedulerConfig:
    """Build the C++ scheduler's configuration.

    Args:
        num_device_pages: Device LCM blocks including the null page.
        max_scheduled_tokens: Per-step prefill token budget.
        max_batch_size: Per-rank batch slots -- the most requests this rank
            runs at once (``--max-num-seqs`` over the attention-DP ranks).
        prefix_granularity: Prefix-hash page size in tokens.
        num_host_pages: Host L2 LCM blocks including the null page; 0 without
            the L2 tier.
        disable_l2_cache: No Host L2 tier.
        enable_l3_storage: An L3 store backs the L2 tier.
        role: ``server_args.disaggregation_mode``.
        num_snapshot_pages: The retraction snapshot pool's LCM blocks
            including the null page (``DeviceSpecs.num_snapshot_pages``).
            Required on every role: 1 -- the null page alone -- means no
            image can be taken, so a capacity-blocked round aborts the newest
            resident instead of suspending anyone; more enables suspend and
            restore on the fused and decode roles (the prefill role refuses a
            pool).
        host_group_packing: Virtual children per compact Host parent in
            scheduler group order; empty when both Host pools are absent.
        max_retracted_requests: Slot-state arena rows, hence the most
            requests suspended at once; positive exactly when a pool exists.
            A victim whose image does not fit the Host (L2 pins, pool, rows)
            is aborted by the scheduler instead of imaged.
        storage_config: Explicit asynchronous L3 budgets, or None without L3.
        host_write_policy: ``--kvstore-write-policy``: ``"write_through"``
            streams published entries to Host (at chunk ends, the prompt end
            and finish); ``"write_back"`` writes back what an admission or a
            restore evicts, in the same plan (``SchedulerConfig::Validate``
            refuses it where it cannot run). Required on every role; without a
            Host tier write_through has no effect, and the decode role's Host
            tier holds retraction images only.
        prefill_decode_interval: ``--prefill-decode-interval``: decode rounds
            owed to the running decodes after each round that schedules a
            prefill chunk, unless no decode can run, and the most rounds it
            holds any one prompt back in total; 0 is off. Fused role only.
        debug_force_retraction_interval: ``--debug-force-retraction-interval``:
            0 is off; every ``|N|`` plans the scheduler retracts the oldest
            quiescent Decoding (``N > 0``) or Prefilling (``N < 0``) request
            without capacity pressure. A test knob, never set in serving.
        enable_kv_cache_events: Publish KV cache events.
        decode_input_tokens: Verify width per decode step.
        overlap_schedule_depth: Dispatched-but-uncommitted plans allowed.
        disable_prefix_cache: No prefix matching at admission.
        cache_groups: The arena's scheduler group configs.
        enable_mixed_prefill_decode: Mixed prefill/decode batches.
        prefix_replay_tokens: Prompt tail recomputed after a prefix hit.
        state_checkpoint_in_forward:
            ``--enable-state-checkpoint-in-forward``: the prefill forward that
            spans a long prompt's last prefix-page boundary before its replay
            tail also writes the State checkpoint there, and no KV tail and no
            State checkpoint is published at a prompt end that checkpoint
            covers (prefill and fused roles) or at a fused finish past the
            prompt end. Requires prefix caching.
        demote_superseded_state_checkpoints:
            On with ``--enable-state-checkpoint-in-forward``: a finish that
            publishes an in-forward State checkpoint demotes the one its
            conversation published two turns back, unless another request
            shares it. Requires ``state_checkpoint_in_forward`` outside the
            decode role.
        drop_short_prompt_end_checkpoints:
            On a prefill server with ``--enable-state-checkpoint-in-forward``:
            when a prompt is too short for the in-forward checkpoint and
            ends inside a page, no KV tail and no State checkpoint is
            published at its end. Requires ``state_checkpoint_in_forward``
            and the prefill role.
        max_kv_page_num: Per-local-prefill-batch budget for the sum of
            ``ceil((cached prefix + current chunk) / prefix_granularity)``
            across requests. 0 leaves the budget unbounded. Derived from
            ``--chunk-prefill-kv-size``.
        checkpoint_retention_requests: ``--checkpoint-retention-requests``:
            under cache pressure a hybrid model's State checkpoint that a later
            request's checkpoint extends ranks as if last used this many new
            requests ago (0 ranks every cache entry by its own use). None keeps the
            scheduler's default, 16 times ``max_batch_size``, which is
            ``--max-num-seqs`` per attention-DP rank.
        checkpoint_retention_leaf_distance:
            ``--checkpoint-retention-leaf-distance``: checkpoints within this
            many later requests of the newest on their branch rank as it does. None keeps the
            scheduler's default, 0.
        prefill_chunk_yield_tokens:
            ``--prefill-chunk-yield-tokens``: the most tokens a prefill chunk
            that cannot finish its prompt gives up, page-aligned, so that
            bootstrapped prompts waiting behind it that fit whole run in the
            same forward. 0 is off; the prefill role only, at most half of
            ``max_scheduled_tokens`` (``SchedulerConfig::Validate`` refuses
            it elsewhere).
    """
    if not 0 <= prefix_replay_tokens <= (1 << 31) - 1:
        raise ValueError(
            "prefix_replay_tokens must fit a non-negative int32; "
            f"got {prefix_replay_tokens}."
        )
    if num_snapshot_pages < 1:
        raise ValueError(
            "num_snapshot_pages must include the null page (1 = no pool); "
            f"got {num_snapshot_pages}."
        )
    if (num_snapshot_pages > 1) != (max_retracted_requests > 0):
        raise ValueError(
            "a retraction snapshot pool and its slot-state rows go together; got "
            f"num_snapshot_pages={num_snapshot_pages}, "
            f"max_retracted_requests={max_retracted_requests}."
        )
    if (storage_config is not None) != enable_l3_storage:
        raise ValueError("storage_config is required exactly when L3 is enabled")
    if prefill_chunk_yield_tokens < 0:
        raise ValueError(
            "prefill_chunk_yield_tokens must be non-negative; "
            f"got {prefill_chunk_yield_tokens}."
        )
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    if max_kv_page_num:
        cfg.max_kv_page_num = max_kv_page_num
    cfg.max_batch_size = max_batch_size
    cfg.prefix_granularity = prefix_granularity

    cfg.num_host_pages = num_host_pages
    cfg.num_snapshot_pages = num_snapshot_pages
    cfg.host_group_packing = list(host_group_packing)
    cfg.max_retracted_requests = max_retracted_requests
    cfg.debug_force_retraction_interval = debug_force_retraction_interval
    cfg.storage_config = storage_config
    cfg.enable_l3_storage = enable_l3_storage
    cfg.enable_kv_cache_events = enable_kv_cache_events

    cfg.role = scheduler_role(role)
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = overlap_schedule_depth
    cfg.disable_prefix_cache = disable_prefix_cache
    cfg.prefix_replay_tokens = prefix_replay_tokens
    if state_checkpoint_in_forward:
        # Set only when on: off is the binding's default, so the runtime keeps
        # working with a scheduler wheel released before this field, where
        # turning the option on raises AttributeError here instead of being
        # ignored.
        cfg.state_checkpoint_in_forward = True
    if demote_superseded_state_checkpoints:
        # Set only when on, as above.
        cfg.demote_superseded_state_checkpoints = True
    if drop_short_prompt_end_checkpoints:
        # Likewise set only when on.
        cfg.drop_short_prompt_end_checkpoints = True
    if checkpoint_retention_requests is not None:
        # Set only when given, as above: an older scheduler wheel keeps its
        # plain LRU by default and refuses an explicit value.
        cfg.checkpoint_retention_requests = checkpoint_retention_requests
    if checkpoint_retention_leaf_distance is not None:
        cfg.checkpoint_retention_leaf_distance = checkpoint_retention_leaf_distance
    if prefill_chunk_yield_tokens:
        # Set only when on, for the same reason.
        cfg.prefill_chunk_yield_tokens = prefill_chunk_yield_tokens
    cfg.disable_l2_cache = disable_l2_cache
    cfg.host_write_policy = _HOST_WRITE_POLICIES[host_write_policy]

    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode
    if prefill_decode_interval:
        # Set only when on, like state_checkpoint_in_forward: a scheduler
        # wheel without the field still runs the default.
        cfg.prefill_decode_interval = prefill_decode_interval
    if cache_groups:
        cfg.cache_groups = list(cache_groups)
    return cfg


def pool_to_cache_groups(pool: Any) -> list:
    """Convert a cache's published contract into scheduler group configs."""
    # The arena is the sole publisher, so there is exactly one source here --
    # no fallback to pool-side copies of the same specs.
    contract = pool.arena.runtime_contract
    counts = contract.virtual_block_counts
    packing = contract.virtual_packing
    return [
        cache_group_config(
            spec,
            total_pages=counts[spec.group_id],
            cache_blocks_per_lcm_block=packing[spec.group_id],
        )
        for spec in contract.group_specs
    ]


def should_use_overlap_schedule(
    *,
    disable_overlap_schedule: bool,
    disaggregation_mode: str,
) -> bool:
    """Return whether the runtime runs the overlap schedule: a decode step
    dispatched before the previous one commits, with decode KV reserved for
    it (``overlap_schedule_depth``)."""

    if disable_overlap_schedule:
        return False
    if disaggregation_mode in ("prefill", "encode"):
        # The prefill role never decodes locally, so it has no decode step to
        # run ahead or reserve KV for; its chunks still run ahead of their
        # commits (resolve_in_flight_depth). Encode has no LM loop.
        return False
    return True


def resolve_in_flight_depth(
    *,
    disable_overlap_schedule: bool,
    disaggregation_mode: str,
    pp_size: int,
) -> int:
    """Return the event loop's in-flight depth: how many dispatched forwards
    may await commit at once (``docs/design/event-loop.md``, Principle 4).

    ``pp_size`` under pipeline parallelism, where the prefill chunk pipeline
    keeps one forward per stage in flight. At PP=1, 0 with
    ``disable_overlap_schedule`` (each forward commits in the round that
    dispatched it) or on an encode rank, which has no LM loop; otherwise 1:
    the control plane plans the next round, commits the previous forward and
    polls transfers while the GPU runs the dispatched one. On a role that
    decodes that is the overlap schedule. The prefill role runs ahead without
    it (``should_use_overlap_schedule``): its chunks need no decode KV
    reserved, and the scheduler holds a prompt's remote decode until the
    final chunk's result has landed.

    Args:
        disable_overlap_schedule: ``--disable-overlap-schedule``.
        disaggregation_mode: ``--disaggregation-mode``.
        pp_size: Pipeline-parallel stages.

    Returns:
        The in-flight depth.
    """
    if pp_size > 1:
        return pp_size
    if disable_overlap_schedule or disaggregation_mode == "encode":
        return 0
    return 1


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

    A same-checkpoint DSpark draft advertises the requirement through its
    draft ``ModelConfig``; DeepSeek V4.1 advertises zero because its windows
    are cache resident. Same-checkpoint DSpark configurations without that
    capability remain fail-closed. External generic DSpark configurations
    keep their existing scheduler behavior until they advertise an
    equivalent contract.
    """

    if speculative_algorithm != "DSPARK":
        return 0
    if draft_model_config is None:
        raise ValueError("DSPARK requires a resolved draft model configuration.")

    replay_tokens = getattr(draft_model_config, "dspark_prefix_replay_tokens", None)
    if replay_tokens is None:
        if draft_model_path_use_base:
            raise ValueError(
                "DSPARK same-checkpoint decoding requires a draft model that "
                "advertises captured-context replay support."
            )
        return 0

    replay_tokens = int(replay_tokens)
    if not 0 <= replay_tokens <= (1 << 31) - 1:
        raise ValueError(
            "DSPARK captured-context replay requirement must fit a non-negative "
            f"int32; got {replay_tokens}."
        )
    if replay_tokens == 0:
        # The draft's context lives in the KV cache and follows the prefix.
        return 0
    # A drafter-private context cannot be restored from the host tier, with or
    # without prefix reuse.
    if enable_kvstore:
        raise ValueError(
            "DSPARK captured-context replay does not support KVStore; "
            "use --disable-kvstore."
        )
    if not enable_prefix_caching:
        return 0
    if disaggregation_mode != "null":
        raise ValueError(
            "DSPARK captured-context replay does not support disaggregated "
            f"serving; got role {disaggregation_mode!r}."
        )
    return replay_tokens


def cache_event_to_payload(event) -> dict:
    """A cache ACK as the payload the host's data plane reports
    (``HostDataPlane.poll_cache_events``).

    ``(kind, op_id)`` identifies a completed Host copy.
    """
    kind = type(event).__name__
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    payload = {
        "kind": kind,
        "op_id": int(event.op_id),
    }
    return payload


def _check_batch_rows(op, columns: Sequence[Sequence]) -> int:
    """The row count of a wire batch whose per-op columns must all agree."""
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"ragged {type(op).__name__} batch: column lengths {sorted(lengths)}"
        )
    return lengths.pop()


def _wire_transfers(
    op_id: int,
    groups: Sequence[int],
    sources: Sequence[int],
    destinations: Sequence[int],
    hashes: Sequence[str] | None,
    offsets: Sequence[int] | None,
) -> tuple[CacheTransfer, ...]:
    """One op's rows of a wire batch as ``CacheTransfer`` values."""
    if hashes is None:
        hashes = [""] * len(groups)
        offsets = [0] * len(groups)
    if not (
        len(groups) == len(sources) == len(destinations) == len(hashes) == len(offsets)
    ):
        raise ValueError(f"ragged cache operation {op_id}")
    return tuple(
        CacheTransfer(
            group_id=int(group),
            source_page=int(source),
            destination_page=int(destination),
            content_hash=str(content_hash),
            page_offset=int(page_offset),
        )
        for group, source, destination, content_hash, page_offset in zip(
            groups, sources, destinations, hashes, offsets
        )
    )


def _snapshot_batch_columns(op) -> tuple[Sequence, ...]:
    return (
        op.op_ids,
        op.request_ids,
        op.request_pool_indices,
        op.snapshot_slots,
        op.group_ids,
        op.src_pages,
        op.dst_pages,
    )


def snapshot_ops_from_wire(op) -> list[SnapshotOp]:
    """One ``SnapshotOp`` per request of a ``Cache.SnapshotOp`` batch."""
    rows = _check_batch_rows(op, _snapshot_batch_columns(op))
    return [
        SnapshotOp(
            op_id=int(op.op_ids[i]),
            request_id=str(op.request_ids[i]),
            request_pool_index=int(op.request_pool_indices[i]),
            snapshot_slot=int(op.snapshot_slots[i]),
            transfers=_wire_transfers(
                int(op.op_ids[i]),
                op.group_ids[i],
                op.src_pages[i],
                op.dst_pages[i],
                None,
                None,
            ),
        )
        for i in range(rows)
    ]


def restore_ops_from_wire(op) -> list[RestoreOp]:
    """One ``RestoreOp`` per request of a ``Cache.RestoreOp`` batch."""
    rows = _check_batch_rows(
        op,
        _snapshot_batch_columns(op)
        + (op.content_hashes, op.page_offsets, op.source_tiers),
    )
    ops = []
    for i in range(rows):
        transfers = _wire_transfers(
            int(op.op_ids[i]),
            op.group_ids[i],
            op.src_pages[i],
            op.dst_pages[i],
            op.content_hashes[i],
            op.page_offsets[i],
        )
        tiers = tuple(HostTier(int(tier)) for tier in op.source_tiers[i])
        if len(tiers) != len(transfers):
            raise ValueError(f"ragged cache operation {op.op_ids[i]}")
        ops.append(
            RestoreOp(
                op_id=int(op.op_ids[i]),
                request_id=str(op.request_ids[i]),
                request_pool_index=int(op.request_pool_indices[i]),
                snapshot_slot=int(op.snapshot_slots[i]),
                transfers=transfers,
                source_tier=tiers,
            )
        )
    return ops


def is_device_cache_op(operation) -> bool:
    """Whether an adapted operation belongs to the GPU copy executor."""
    return isinstance(
        operation, (Cache.WriteBackOp, Cache.LoadBackOp, SnapshotOp, RestoreOp)
    )


class CacheOperations(NamedTuple):
    """One validated plan snapshot, partitioned by its execution owner.

    Each tuple keeps the plan's order. It defines no ordering between Host
    storage I/O and Device copies; tickets and completions express dependencies.
    """

    device: tuple[Cache.WriteBackOp | Cache.LoadBackOp | SnapshotOp | RestoreOp, ...]
    storage: tuple[
        Cache.StorageQueryOp
        | Cache.StorageGetOp
        | Cache.StoragePutOp
        | Cache.StorageCancelOp,
        ...,
    ]


def cache_ops_from_plan(execution_plan) -> CacheOperations:
    """Read and validate every cache op once before any owner submits work.

    L2 wire batches pass through; snapshot/restore batches become per-request
    operations. Host-only QUERY/GET/PUT/Cancel descriptors go to the storage
    tuple: the engine host's L3 executor runs them. Unknown kinds fail the
    whole conversion without partial submission.
    """
    device: list = []
    storage: list = []
    for op in execution_plan.cache:
        if isinstance(op, (Cache.WriteBackOp, Cache.LoadBackOp)):
            device.append(op)
        elif isinstance(op, Cache.SnapshotOp):
            device.extend(snapshot_ops_from_wire(op))
        elif isinstance(op, Cache.RestoreOp):
            device.extend(restore_ops_from_wire(op))
        elif isinstance(
            op,
            (
                Cache.StorageQueryOp,
                Cache.StorageGetOp,
                Cache.StoragePutOp,
                Cache.StorageCancelOp,
            ),
        ):
            storage.append(op)
        else:
            raise TypeError(f"unsupported cache op kind: {type(op).__name__}")
    return CacheOperations(tuple(device), tuple(storage))


class PackedBlockTables(NamedTuple):
    """One batch's per-group block tables, uploaded once.

    Attributes:
        tables: Per-group int32 views into the one device storage.
        tables_cpu: The same tables as views into one pinned host copy, for
            planning that must not wait on the device; empty unless asked for.
    """

    tables: dict[str, torch.Tensor]
    tables_cpu: dict[str, torch.Tensor]


def block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """The device tables of :func:`packed_block_tables_from_forward_op`."""
    return packed_block_tables_from_forward_op(
        forward_op,
        device,
        host_tables=False,
        num_reqs=num_reqs,
        expected_group_ids=expected_group_ids,
        max_page_id=max_page_id,
        max_page_ids=max_page_ids,
    ).tables


def packed_block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    host_tables: bool,
    num_reqs: int | None = None,
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> PackedBlockTables:
    """Bridge the per-group block tables to GPU int32 tensors: absolute
    page indices, null hole = 0 preserved, ragged-row padding -1. No
    base-offset companion -- the cache path never compacts.

    All groups ride ONE H2D copy and one kernel rebuilds them into ONE device
    storage; the returned per-group views share it, which is the
    precondition of the routers' one-launch fill of every group
    (``GroupTableStacks.fill``). Per-group uploads would fail its
    same-storage check and fall back to one launch per group. The copy
    carries each row's span from the scheduler (``block_table_spans``):
    before it the row holds its released prefix's null holes and after it
    padding, so a sliding-window row at a long context moves a few entries.
    With ``host_tables`` the tables are also copied into pinned host memory,
    so a backend planning an extend from them reads them without a D2H sync.

    Structural contract checks run whenever bounds or expected groups are
    supplied. Every page ID is scanned against the bounds, and every row
    against its span, when they are supplied; at long contexts the scan
    costs more than the packing, so callers pass bounds only when they mean
    to scan (``CacheBatchMetadata.from_forward_op``'s ``validate_page_ids``).

    Args:
        forward_op: Scheduler forward operation exporting CPU NumPy tables
            and their row spans.
        device: Destination device for the packed tensor.
        host_tables: Also return the tables in host memory (``tables_cpu``).
        num_reqs: Optional expected row count for every group.
        expected_group_ids: Optional pack order and exact key set; the
            packed buffer and the returned views follow it.
        max_page_id: Optional common inclusive upper bound for page IDs.
        max_page_ids: Optional per-group inclusive upper bounds.

    Returns:
        Device and (with ``host_tables``) host per-group tensor views in
        ``expected_group_ids`` order when supplied, otherwise preserving
        producer order.

    Raises:
        ValueError: If strict contract validation or a row span fails before
            device transfer.
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
    spans_by_id = {
        str(key): spans for key, spans in forward_op.block_table_spans().items()
    }
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
    out_cpu: dict[str, torch.Tensor] = {}
    # (key, table, offset in the packed storage, window start per row, width)
    packable: list[tuple[str, Any, int, np.ndarray, int]] = []
    total = 0
    for key, arr in ordered_items:
        if num_reqs is not None and arr.shape[0] != num_reqs:
            raise ValueError(
                f"block_tables_arrays[{key}] has {arr.shape[0]} rows "
                f"but forward op reported num_reqs={num_reqs}"
            )
        if arr.shape[0] == 0:
            continue
        spans = spans_by_id.get(key)
        if (
            not isinstance(spans, np.ndarray)
            or spans.dtype != np.int32
            or spans.shape != (arr.shape[0], 2)
        ):
            raise ValueError(f"block_table_spans[{key}] must be int32 [rows, 2]")
        released, pages = spans[:, 0], spans[:, 1]
        if bool(((released < 0) | (released > pages) | (pages > arr.shape[1])).any()):
            raise ValueError(f"block_table_spans[{key}] exceeds its table")
        if max_page_id is not None or max_page_ids is not None:
            column = np.arange(arr.shape[1])
            if bool(
                (arr[column < released[:, None]] != 0).any()
                or (arr[column >= pages[:, None]] != -1).any()
            ):
                raise ValueError(
                    f"cache group {key!r} holds entries outside its row spans"
                )
        if arr.shape[1] == 0:
            # Kept out of the pack: CacheBatchMetadata rejects a zero-width table.
            out[key] = torch.empty((arr.shape[0], 0), dtype=torch.int32, device=device)
            if host_tables:
                out_cpu[key] = torch.empty((arr.shape[0], 0), dtype=torch.int32)
            continue
        # One window width per group: the windows of a group form one block,
        # and a window ending at a span's end starts at or before it.
        width = int((pages - released).max())
        first = np.maximum(pages.astype(np.int64) - width, 0)
        packable.append((key, arr, total, first, width))
        total += arr.shape[0] * arr.shape[1]
    if not packable:
        return PackedBlockTables(out, out_cpu)
    from tokenspeed_kernel.ops.metadata import (
        BLOCK_TABLE_ROW_FIELDS,
        expand_block_table_rows,
    )

    num_rows = sum(arr.shape[0] for _, arr, _, _, _ in packable)
    fields = num_rows * BLOCK_TABLE_ROW_FIELDS
    num_values = sum(arr.shape[0] * width for _, arr, _, _, width in packable)
    # Fresh pinned stages per step (event-fenced; reuse races overlap).
    # arr is a read-only zero-copy view over the C++ buffer; np.copyto
    # reads it into our own writable pinned tensors (never writes back).
    pin_memory = device.type in ("cuda", "npu")
    upload = torch.empty(fields + num_values, dtype=torch.int32, pin_memory=pin_memory)
    upload_np = upload.numpy()
    row_fields = upload_np[:fields].reshape(num_rows, BLOCK_TABLE_ROW_FIELDS)
    values = upload_np[fields:]
    row_at = value_at = 0
    for key, arr, offset, first, width in packable:
        rows, columns = arr.shape
        # (offset, width, start, end, source): expand_block_table_rows' fields.
        group_fields = row_fields[row_at : row_at + rows]
        group_fields[:, 0] = offset + np.arange(rows) * columns
        group_fields[:, 1] = columns
        group_fields[:, 2] = first
        group_fields[:, 3] = first + width
        group_fields[:, 4] = value_at + np.arange(rows) * width
        windows = values[value_at : value_at + rows * width].reshape(rows, width)
        if not first.any():
            np.copyto(windows, arr[:, :width])
        elif width:
            np.copyto(
                windows,
                np.lib.stride_tricks.sliding_window_view(arr, width, axis=1)[
                    np.arange(rows), first
                ],
            )
        row_at += rows
        value_at += rows * width
    upload = upload.to(device, non_blocking=True)
    packed = torch.empty(total, dtype=torch.int32, device=device)
    expand_block_table_rows(
        upload[:fields].view(num_rows, BLOCK_TABLE_ROW_FIELDS), upload[fields:], packed
    )
    for key, arr, offset, _, _ in packable:
        out[key] = packed[offset : offset + arr.size].view(arr.shape)
    if host_tables:
        staged = torch.empty(total, dtype=torch.int32, pin_memory=pin_memory)
        staged_np = staged.numpy()
        for key, arr, offset, _, _ in packable:
            np.copyto(staged_np[offset : offset + arr.size].reshape(arr.shape), arr)
            out_cpu[key] = staged[offset : offset + arr.size].view(arr.shape)
    return PackedBlockTables(out, out_cpu)


def _classify_param(name: str) -> str:
    """Bucket a parameter/buffer name into a weight group for the memory
    summary. Names follow the Kimi-K3 / DeepSeek module layout."""
    if ".engram." in name:
        return "engram_weights"
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

    Arena-backed pools report actual device allocations, including residency
    workspace and excluding authoritative host history. Legacy pools may
    report nested numeric totals.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena

    seen: set[int] = set()
    total = 0
    for pool in pools:
        if pool is None:
            continue
        arena = getattr(pool, "arena", pool)
        if id(arena) in seen:
            continue
        seen.add(id(arena))
        if isinstance(arena, CacheArena):
            total += arena.allocated_device_bytes
            continue
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
    device: str = "cuda",
) -> None:
    """Log a per-rank accelerator memory breakdown after cache allocation.

    Weight groups are summed from the model's parameters and buffers (deduped
    by storage pointer). A draft model (speculative decoding) is summed
    separately into its own row. KV cache, graph allocations, and non-torch
    allocations are derived from the selected device's allocator and driver
    views. Best-effort: never raises into startup.
    """
    try:
        GB = 1024**3
        device_type = torch.device(device).type
        device_module = torch.get_device_module(device_type)
        groups = {
            "attention_weights": 0,
            "moe_weights": 0,
            "dense_mlp_weights": 0,
            "engram_weights": 0,
            "other_weights": 0,
        }
        seen: set[int] = set()

        def _accumulate(name, tensor, sink):
            if tensor is None or tensor.device.type != device_type:
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

        free_bytes, total_bytes = device_module.mem_get_info(gpu_id)
        allocated = device_module.memory_allocated(gpu_id) / GB
        reserved = device_module.memory_reserved(gpu_id) / GB
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
        # Engram tables are only listed when resident on the device; with
        # --engram-host-table they live in host memory and are not counted.
        if groups["engram_weights"]:
            rows.append(("Engram weights (tables/wkv)", groups["engram_weights"] / GB))
        if draft_model is not None:
            rows.append(("Draft model weights", draft_gb))
        rows += [
            ("KV cache", kv_cache_gb),
            ("Activations + device graphs", activations_and_graphs),
            ("Torch allocated (total)", allocated),
            ("Torch reserved (allocator pool)", reserved),
            ("Non-torch (context/collectives)", non_torch),
            ("Device used (driver view)", device_used),
            ("Device free", device_free),
            ("Device total", device_total),
        ]
        name_width = max(len(n) for n, _ in rows)
        sep = "+" + "-" * (name_width + 2) + "+" + "-" * 12 + "+"
        lines = [
            f"Device memory summary (rank {rank}, {device_type} {gpu_id}, GB):",
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
