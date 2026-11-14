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

"""Disjoint cache-page planning for startup decode profiling."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
    compute_max_logical_pages_for_capture,
)


def profile_decode_context(
    context_length: int, *, max_positions: int, verify_width: int, draft_width: int
) -> int:
    """Leave room for the draft following a fully accepted target block."""
    if min(context_length, max_positions, verify_width, draft_width) < 1:
        raise ValueError("profiling dimensions must be positive")
    context = min(context_length, max_positions - draft_width)
    if context < verify_width:
        raise ValueError("position capacity cannot fit target verification and draft")
    return context


@dataclass(frozen=True)
class ProfileCacheGroup:
    group_id: str
    first_page: int
    pages_per_request: int
    batch_size: int

    @property
    def page_stop(self) -> int:
        return self.first_page + self.pages_per_request * self.batch_size

    def block_tables(self) -> tuple[tuple[int, ...], ...]:
        rows = []
        for row in range(self.batch_size):
            first = self.first_page + row * self.pages_per_request
            rows.append(tuple(range(first, first + self.pages_per_request)))
        return tuple(rows)


def plan_profile_cache(
    specs: Sequence[CacheGroupSpec],
    *,
    virtual_packing: Mapping[str, int],
    page_counts: Mapping[str, int],
    batch_size: int,
    context_length: int,
    verify_width: int,
    overlap_schedule_depth: int,
) -> tuple[ProfileCacheGroup, ...] | None:
    """Reserve disjoint parents without mutating an arena or scheduler.

    Call only before serving starts. None means the synthetic working set
    cannot fit, or its sharding is unsupported; callers must not substitute
    aliased placeholder pages and publish that timing as a complete profile.
    """
    if batch_size < 1 or context_length < 1 or verify_width < 1:
        raise ValueError("profiling dimensions must be positive")
    if len({spec.group_id for spec in specs}) != len(specs):
        raise ValueError("duplicate profiling cache group")
    if not specs or any(spec.shard_count != 1 for spec in specs):
        return None
    parent_offset = 0
    groups = []
    for spec in specs:
        packing = virtual_packing[spec.group_id]
        if packing < 1 or page_counts[spec.group_id] < 1:
            raise ValueError("invalid profiling cache capacity")
        width = compute_max_logical_pages_for_capture(
            spec,
            max_context_len=context_length,
            max_tokens_per_req=verify_width,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        pages = width
        group = ProfileCacheGroup(
            spec.group_id,
            1 + parent_offset * packing,
            pages,
            batch_size,
        )
        if group.page_stop > page_counts[spec.group_id]:
            return None
        groups.append(group)
        parent_offset += (batch_size * pages + packing - 1) // packing
    return tuple(groups)


@contextmanager
def preserve_profile_tensors(tensors: Sequence[torch.Tensor]):
    """Restore startup scratch contents in place, including on profiling failure.

    The caller orders profiling and restoration on one stream and joins any
    auxiliary work before exiting. Tensor addresses must not change because
    the already captured graphs retain them.
    """
    snapshots = [tensor.clone() for tensor in tensors]
    try:
        yield
    finally:
        for tensor, snapshot in zip(tensors, snapshots, strict=True):
            tensor.copy_(snapshot)


def measure_startup_graph(
    replay: Callable[[], None],
    prepare: Callable[[], None],
    *,
    device: torch.device,
    host_group: torch.distributed.ProcessGroup | None,
    warmup: int,
    samples: int,
) -> tuple[float, ...]:
    """Measure a captured GPU step before serving, aligned across ranks.

    Preparation and rank barriers are outside the CUDA interval. Replay must
    join any auxiliary GPU streams before returning. A host budget operation
    inside replay contributes its exposed GPU idle interval; this is not a
    separate CPU wall-time measurement or an end-to-end request measurement.
    """
    if warmup < 0 or samples < 1:
        raise ValueError("invalid startup profiling sample counts")
    if device.type != "cuda":
        raise ValueError("startup graph profiling requires CUDA")
    events = []
    with torch.cuda.device(device):
        for index in range(warmup + samples):
            prepare()
            torch.cuda.synchronize(device)
            if host_group is not None:
                torch.distributed.barrier(group=host_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            replay()
            end.record()
            end.synchronize()
            if index >= warmup:
                events.append((start, end))
        elapsed = torch.tensor(
            [start.elapsed_time(end) for start, end in events],
            dtype=torch.float64,
            device="cpu",
        )
        valid = torch.tensor(
            int(bool(torch.isfinite(elapsed).all()) and bool((elapsed > 0).all())),
            dtype=torch.int32,
        )
        if host_group is not None:
            torch.distributed.all_reduce(
                valid, op=torch.distributed.ReduceOp.MIN, group=host_group
            )
            torch.distributed.all_reduce(
                elapsed, op=torch.distributed.ReduceOp.MAX, group=host_group
            )
        if not valid.item():
            raise ValueError("non-finite or non-positive startup graph timing")
        return tuple(elapsed.tolist())


def agree_startup_profile(
    *,
    available: bool,
    batch_size: int,
    context_length: int,
    capacities: tuple[int, ...],
    host_group: torch.distributed.ProcessGroup | None,
) -> bool:
    """All ranks either profile the same graph ladder or skip it together."""
    signature = (batch_size, context_length, capacities) if available else None
    if host_group is None:
        return signature is not None
    peers = [None] * torch.distributed.get_world_size(host_group)
    torch.distributed.all_gather_object(peers, signature, group=host_group)
    return signature is not None and all(peer == signature for peer in peers)


@torch.inference_mode()
def profile_dsv41_startup(executor) -> None:
    """Profile final DSpark decode graphs before admitting any live requests."""
    import numpy as np

    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        STARTUP_CONTEXT_ENDPOINTS,
        STARTUP_MEASURED_REPLAYS,
        STARTUP_WARMUP_REPLAYS,
        VerifyCalibration,
    )
    from tokenspeed.runtime.execution.drafter.verify_feedback import VerifyFeedback
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.sampling.backends.base import CUDA_GRAPH_VARIANT_DEFAULT
    from tokenspeed.runtime.utils import get_colorful_logger

    logger = get_colorful_logger(__name__)
    runner = executor.forward_step
    drafter = executor.drafter
    if drafter is None or runner.disable:
        return
    if not drafter._verify_budget_buffers:
        return
    arena = runner.token_to_kv_pool.arena
    device = torch.device(runner.device)
    group = (
        drafter._verification_host_group
        if len(drafter._verification_group) > 1
        else None
    )
    width = drafter.spec_num_tokens
    maximum = int(executor.config.context_len)
    endpoints = sorted({min(n, maximum) for n in (*STARTUP_CONTEXT_ENDPOINTS, maximum)})
    ib = runner.input_buffers
    old_feedback = drafter.verify_feedback
    installed = drafter._verify_calibrations
    old_policy_ms = drafter._verify_policy_ms
    try:
        for bs in sorted(drafter._verify_budget_buffers):
            capacities = drafter.decode_graph_query_capacities(bs)
            lower = 1
            for context in endpoints:
                if context < width:
                    continue
                if any(
                    p.covers(batch_size=bs, min_context=lower, max_context=context)
                    for p in installed
                ):
                    lower = context + 1
                    continue
                # A full step drafts beyond the last accepted target position.
                # At the model limit, measure just below the endpoint rather than
                # indexing outside RoPE; the calibration interval stays unchanged.
                measured_context = profile_decode_context(
                    context,
                    max_positions=min(
                        maximum, int(drafter.model.config.max_position_embeddings)
                    ),
                    verify_width=width,
                    draft_width=drafter.block_size,
                )
                plan = plan_profile_cache(
                    arena.cache_group_specs,
                    virtual_packing=arena.runtime_contract.virtual_packing,
                    page_counts=arena.cache_group_page_counts,
                    batch_size=bs,
                    context_length=measured_context,
                    verify_width=width,
                    overlap_schedule_depth=runner.overlap_schedule_depth,
                )
                keys = {
                    q: (runner._query_variant(CUDA_GRAPH_VARIANT_DEFAULT, q, bs), bs)
                    for q in capacities
                }
                available = plan is not None and all(
                    k in runner.graphs for k in keys.values()
                )
                if not agree_startup_profile(
                    available=available,
                    batch_size=bs,
                    context_length=context,
                    capacities=capacities,
                    host_group=group,
                ):
                    if drafter._verification_rank == 0:
                        logger.warning(
                            f"Adaptive DSpark startup profile unavailable: "
                            f"local_bs={bs}, context=[{lower}, {context}]; using full verification"
                        )
                    lower = context + 1
                    continue
                assert plan is not None
                tables = {
                    g.group_id: torch.tensor(
                        g.block_tables(), dtype=torch.int32, device=device
                    )
                    for g in plan
                }
                blocks = {
                    g.group_id: np.arange(g.first_page, g.page_stop, dtype=np.int32)
                    for g in plan
                }
                tensors = [
                    ib.input_ids_buf[: bs * width],
                    ib.positions_buf[: bs * width],
                    ib.req_pool_indices_buf[:bs],
                    ib.state_write_req_pool_indices_buf[:bs],
                    ib.seq_lens_buf[:bs],
                ]
                for owner in executor.slot_state_exporters():
                    for slot in range(bs):
                        tensors.extend(owner.slot_state_rows(slot))
                measured = {}
                try:
                    with preserve_profile_tensors(tensors):
                        arena.zero_blocks(blocks)
                        slots = torch.arange(
                            bs, device=device, dtype=ib.req_pool_indices_buf.dtype
                        )
                        ib.req_pool_indices_buf[:bs].copy_(slots)
                        ib.state_write_req_pool_indices_buf[:bs].copy_(slots)
                        ids = [f"startup-profile-{i}" for i in range(bs)]
                        feedback = VerifyFeedback(draft_width=width - 1)
                        for _ in range(3):
                            feedback.prepare(ids, list(range(bs)), num_extends=0)
                            feedback.publish(
                                feedback.snapshot(),
                                [[0.0] * (width - 1)] * bs,
                                context_lengths=[max(1, measured_context - 2 * width)]
                                * bs,
                            )
                        drafter.verify_feedback = feedback
                        for q in reversed(capacities):
                            # Force exactly this prepared tier through the normal
                            # host policy path; no acceptance result is overridden.
                            drafter._verify_calibrations = (
                                VerifyCalibration(
                                    bs,
                                    1,
                                    maximum + 2 * width,
                                    width,
                                    1e9,
                                    ((q, 1.0),) if q < bs * width else (),
                                    8,
                                ),
                            )
                            drafter._verify_policy_ms = 0.0

                            def prepare():
                                ib.input_ids_buf[: bs * width].copy_(
                                    (
                                        torch.arange(bs * width, device=device)
                                        + 17
                                        + runner.global_rank * 101
                                    )
                                    % runner.vocab_size
                                )
                                ib.positions_buf[: bs * width].copy_(
                                    torch.arange(
                                        measured_context - width,
                                        measured_context,
                                        device=device,
                                    ).repeat(bs)
                                )
                                ib.seq_lens_buf[:bs].fill_(measured_context)
                                states = executor.runtime_states
                                states.valid_cache_lengths[:bs].fill_(
                                    measured_context - width
                                )
                                states.spec_candidates_ready[:bs].fill_(True)
                                states.stopped[:bs].zero_()
                                states.stop_token_ids[:bs].fill_(-1)
                                states.token_limits[:bs].fill_(
                                    torch.iinfo(torch.int32).max
                                )
                                runner._prepare_sampling_capture(
                                    bs=bs, variant=CUDA_GRAPH_VARIANT_DEFAULT
                                )
                                runner._prepare_decode_metadata(
                                    bs,
                                    bs,
                                    ib.req_pool_indices_buf[:bs],
                                    ib.seq_lens_buf[:bs],
                                    ForwardMode.DECODE,
                                    use_graph=True,
                                    settle=lambda: None,
                                    block_tables=tables,
                                )
                                runner.deepep_adapter.replay()

                            def replay():
                                if q < bs * width:
                                    chosen = drafter._prepare_calibrated_budget(
                                        actual_bs=bs, padded_bs=bs
                                    )
                                    if chosen != q:
                                        raise RuntimeError(
                                            "startup profiling selected a different graph"
                                        )
                                runner.graphs[keys[q]].replay()

                            executor.nan_guard.reset(bs)
                            measured[q] = measure_startup_graph(
                                replay,
                                prepare,
                                device=device,
                                host_group=group,
                                warmup=STARTUP_WARMUP_REPLAYS,
                                samples=STARTUP_MEASURED_REPLAYS,
                            )
                            flags = executor.nan_guard.flags_cpu
                            torch.cuda.synchronize(device)
                            valid = flags is None or not bool(flags.any())
                            if not agree_startup_profile(
                                available=valid,
                                batch_size=bs,
                                context_length=context,
                                capacities=(q,),
                                host_group=group,
                            ):
                                raise RuntimeError(
                                    "non-finite synthetic startup profile"
                                )
                        torch.cuda.synchronize(device)
                finally:
                    torch.cuda.synchronize(device)
                    arena.zero_blocks(blocks)
                    executor.nan_guard.reset(0)
                    runner._init_capture_metadata(bs)
                    drafter._verify_calibrations = installed
                    drafter._verify_policy_ms = old_policy_ms
                profile = VerifyCalibration.measure(
                    batch_size=bs,
                    min_context=lower,
                    max_context=context,
                    verify_width=width,
                    native_samples=measured[bs * width],
                    adaptive_samples={
                        q: values for q, values in measured.items() if q != bs * width
                    },
                    minimum_samples=STARTUP_MEASURED_REPLAYS,
                )
                installed = (*installed, profile)
                drafter.install_verify_calibration(installed, policy_ms=0.0)
                drafter._save_verify_calibration()
                if drafter._verification_rank == 0:
                    logger.info(
                        f"Adaptive DSpark startup profile complete: local_bs={bs}, "
                        f"context=[{lower}, {context}], measured_context={measured_context}, "
                        f"full_ms={profile.native_step_ms:.3f}, "
                        f"adaptive_ms={dict(profile.adaptive_step_ms)}"
                    )
                lower = context + 1
    finally:
        torch.cuda.synchronize(device)
        drafter.verify_feedback = old_feedback
        drafter._verify_calibrations = installed
        drafter._verify_policy_ms = 0.0 if installed else old_policy_ms
