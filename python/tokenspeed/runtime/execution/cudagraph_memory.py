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

"""Project the CUDA-graph pool reserve from a throwaway capture measured at startup."""

from __future__ import annotations

import copy
import gc
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.memory_delta import (
    MemoryDeltaObserver,
    memory_delta_observer,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import ProbeBatch
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


CUDAGRAPH_FAMILIES = ("prefill", "decode")
PROBE_CAPTURES_PER_FAMILY = 4


@dataclass(frozen=True)
class CudagraphFamilyEstimate:
    """Projected bytes for one CUDA-graph pool."""

    first_capture: int
    extrapolated_rate: int | float
    total: int | float


@dataclass(frozen=True)
class CudagraphMemoryEstimate:
    """Projected bytes for the prefill and decode CUDA-graph pools."""

    prefill: CudagraphFamilyEstimate
    decode: CudagraphFamilyEstimate
    total: int | float


def _estimate_family(
    family: str, samples: Sequence[int], entry_count: int
) -> CudagraphFamilyEstimate:
    """Project one pool from the captures actually sampled out of its ladder."""
    if entry_count == 0:
        if samples:
            raise ValueError(f"{family} projection got samples for no entries")
        return CudagraphFamilyEstimate(0, 0, 0)

    required = min(2, entry_count)
    if not required <= len(samples) <= entry_count:
        raise ValueError(
            f"{family} projection got {len(samples)} samples for "
            f"{entry_count} entries, expected between {required} and {entry_count}"
        )

    first, *marginals = samples
    # The median keeps a one-off allocator growth step out of the extrapolated rate.
    rate = max(median(marginals), 0) if marginals else 0

    # Net samples go negative when the allocator releases segments mid-capture,
    # so only the extrapolation and the projection itself are floored.
    remaining = entry_count - len(samples)
    total = max(first + sum(marginals) + rate * remaining, 0)

    return CudagraphFamilyEstimate(first, rate, total)


def estimate_cudagraph_memory(
    samples: Mapping[str, Sequence[int]], entry_counts: Mapping[str, int]
) -> CudagraphMemoryEstimate:
    """Project the prefill and decode pools, which are disjoint and so add up."""
    unknown = sorted(set(samples) - set(CUDAGRAPH_FAMILIES))
    if unknown:
        raise ValueError(f"projection got samples for unknown families: {unknown}")

    prefill = _estimate_family(
        "prefill", samples.get("prefill", ()), entry_counts["prefill"]
    )
    decode = _estimate_family(
        "decode", samples.get("decode", ()), entry_counts["decode"]
    )

    return CudagraphMemoryEstimate(
        prefill=prefill, decode=decode, total=prefill.total + decode.total
    )


@dataclass(frozen=True)
class CudagraphProbeConfig:
    server_args: ServerArgs
    model_config: ModelConfig
    draft_model_config: ModelConfig | None
    target: ModelRunner
    draft: ModelRunner | None
    gpu_id: int
    global_rank: int
    gpu_memory: int
    overlap_schedule_depth: int
    decode_input_tokens: int
    max_batch_size: int


@dataclass(frozen=True)
class _CudagraphProbe:
    """A throwaway executor whose captures are measured and then released."""

    executor: ModelExecutor
    entry_counts: dict[str, int]

    def release(self, device: str) -> list[tuple[str, weakref.ReferenceType[object]]]:
        """Drop the probe's graphs and private pool, naming what must now be dead."""
        from tokenspeed.runtime.execution import cuda_graph_wrapper
        from tokenspeed.runtime.execution.workspace import workspace_pool

        liveness = [
            ("executor", weakref.ref(self.executor)),
            ("prefill_graph", weakref.ref(self.executor.prefill_graph)),
            ("decode_graph_wrapper", weakref.ref(self.executor.forward_step)),
        ]

        self.executor.shutdown()
        workspace_pool(device).unfreeze()
        cuda_graph_wrapper.global_graph_memory_pool = None

        return liveness


def _probe_batch(config: CudagraphProbeConfig) -> ProbeBatch:
    """The dummy batch the probe's forwards fabricate.

    Both the autotune forward and the captured buckets spread one
    chunked-prefill budget over ``ceil(tokens / context_len)`` requests, and
    ``PrefillGraph._dummy_group_tables`` gives each request a single block that
    every column of its row repeats. One token per request is therefore all
    the arena has to admit, which is what keeps a throwaway probe small.
    """
    tokens = max(1, int(config.server_args.chunked_prefill_size or 0))
    context_len = max(1, int(config.model_config.context_len))
    requests = max(1, -(-tokens // context_len))
    return ProbeBatch(requests=requests, tokens=requests)


def _build_cudagraph_probe(
    config: CudagraphProbeConfig, observer: MemoryDeltaObserver
) -> _CudagraphProbe:
    """Build an executor over a minimal arena, capturing a few graphs per family."""
    probe_batch = _probe_batch(config)
    from tokenspeed.runtime.execution.cuda_graph_wrapper import (
        get_batch_sizes_to_capture,
    )
    from tokenspeed.runtime.execution.factory import (
        ModelExecutorConfig,
        create_model_executor,
    )
    from tokenspeed.runtime.execution.prefill_graph import get_prefill_token_buckets
    from tokenspeed.runtime.layers.attention.registry import create_attn_components

    (
        attn_backend,
        token_to_kv_pool,
        draft_attn_backend,
        draft_token_to_kv_pool,
        _cache_storage,
    ) = create_attn_components(
        config.server_args,
        config.model_config,
        config.gpu_id,
        config.global_rank,
        config.gpu_memory,
        config.server_args.enable_memory_saver,
        config.draft_model_config,
        decode_input_tokens=config.decode_input_tokens,
        overlap_schedule_depth=config.overlap_schedule_depth,
        probe_batch=probe_batch,
    )
    if token_to_kv_pool.arena.plan.num_lcm_blocks < 1:
        raise RuntimeError(
            "CUDA-graph probe arena allocated no parent blocks, so the recipe "
            f"ignored the {probe_batch} the probe asked it to hold"
        )

    executor_config = ModelExecutorConfig.from_server_args(
        server_args=config.server_args,
        model_config=config.model_config,
        max_req_pool_size=config.max_batch_size + 1,
        gpu_id=config.gpu_id,
        global_rank=config.global_rank,
        prefix_granularity=token_to_kv_pool.arena.prefix_granularity,
        overlap_schedule_depth=config.overlap_schedule_depth,
    )
    decode_entries = get_batch_sizes_to_capture(executor_config)
    prefill_entries = get_prefill_token_buckets(executor_config)

    probe_config = copy.copy(executor_config)
    probe_config.cudagraph_capture_sizes = sorted(decode_entries, reverse=True)[
        :PROBE_CAPTURES_PER_FAMILY
    ]
    probe_config.prefill_graph_capture_sizes = sorted(prefill_entries, reverse=True)[
        :PROBE_CAPTURES_PER_FAMILY
    ]

    executor = create_model_executor(
        server_args=config.server_args,
        config=probe_config,
        model_runner=config.target,
        draft_model_runner=config.draft,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        draft_attn_backend=draft_attn_backend,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
        memory_observer=observer,
    )

    # Counts follow PrefillGraph/CudaGraphWrapper disable rules, not the ladder.
    entry_counts = {
        "prefill": 0 if executor.prefill_graph.disable else len(prefill_entries),
        "decode": (
            0
            if executor.forward_step.disable
            else len(decode_entries) * len(executor.forward_step.graph_variants)
        ),
    }

    return _CudagraphProbe(executor=executor, entry_counts=entry_counts)


def _reconcile_cudagraph_reserve(
    config: CudagraphProbeConfig, total: int | float
) -> int | float:
    """Size every rank's KV cache for the hungriest rank's projection."""
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    world_group = config.server_args.mapping.world_group
    if world_group is None or config.server_args.mapping.world_size == 1:
        return total

    reduced = torch.tensor(total, dtype=torch.float64)
    torch.distributed.all_reduce(
        reduced,
        op=torch.distributed.ReduceOp.MAX,
        group=pg_manager.get_process_group("gloo", world_group),
    )
    return int(reduced.item())


def _assert_probe_released(
    liveness: list[tuple[str, weakref.ReferenceType[object]]],
) -> None:
    """Fail loudly if the probe's graphs or private pool outlive its teardown."""
    alive = [name for name, ref in liveness if ref() is not None]
    if alive:
        raise RuntimeError(
            f"CUDA-graph memory probe leaked objects: {', '.join(alive)} "
            "survived teardown, so their captured graphs and private pool still "
            "hold device memory the real capture was projected to reuse"
        )


def probe_cudagraph_memory(config: CudagraphProbeConfig) -> int | float:
    """Capture a small throwaway graph set and project the runtime pool reserve."""
    device_module = torch.get_device_module(config.server_args.device)
    observer = memory_delta_observer(
        record=True, device_module=device_module, gpu_id=config.gpu_id
    )

    probe = _build_cudagraph_probe(config, observer)
    entry_counts = probe.entry_counts
    estimate = estimate_cudagraph_memory(observer.samples, entry_counts)
    liveness = probe.release(config.server_args.device)

    # This frame's own reference would keep the probe alive past the check below.
    del probe
    gc.collect()
    device_module.empty_cache()
    device_module.synchronize()
    _assert_probe_released(liveness)

    reserve = _reconcile_cudagraph_reserve(config, estimate.total)
    logger.info(
        "CUDA-graph memory reserve: %d bytes "
        "(prefill %d over %d entries, decode %d over %d entries)",
        reserve,
        estimate.prefill.total,
        entry_counts["prefill"],
        estimate.decode.total,
        entry_counts["decode"],
    )
    return reserve
