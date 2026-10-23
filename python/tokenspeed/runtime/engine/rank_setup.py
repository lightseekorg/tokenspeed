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

"""What building one engine rank needs before its data plane exists: the
model config, the process groups, the request length limits and the C++
scheduler's config. ``rust_host.bootstrap`` runs them in this order."""

import torch
from tokenspeed_scheduler import StorageConfig

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.engine.scheduler_utils import make_config
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedConfig,
    DistributedInitializer,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs, global_server_args_dict_update
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.startup_timing import startup_phase

logger = get_colorful_logger(__name__)


def rank_process_title(mapping) -> str:
    """The name ``ps`` shows for an engine rank (``docs/serving/parallelism.md``):
    ``tokenspeed::scheduler_tp{rank}``, then the rank in each other parallel
    dimension larger than one."""
    title = f"tokenspeed::scheduler_tp{mapping.attn.tp_rank}"
    if mapping.moe.has_ep:
        title += f"_ep{mapping.moe.ep_rank}"
    if mapping.attn.has_dp:
        title += f"_dp{mapping.attn.dp_rank}"
    if mapping.attn.has_kvp:
        title += f"_kvp{mapping.attn.kvp_rank}"
    if mapping.has_pp:
        title += f"_pp{mapping.pp_rank}"
    return title


def maybe_warm_cupti_for_graph_capture() -> None:
    """Preload CUPTI before any CUDA graph is captured. Opt-in.

    The warm-up guards a hazard reported on older stacks: a profiler that
    first attaches AFTER capture invalidates the captured graphs -- every
    later replay dies with cudaErrorLaunchFailure -- which would forbid
    runtime ``/start_profile`` on graph-mode servers. One empty profiler
    session loads CUPTI ahead of every capture.

    It is off by default because it defeats its own purpose: the empty session
    leaves activity collection dead for the life of the process, so every
    later ``/start_profile`` returns a trace with ``cpu_op`` entries and zero
    ``"cat": "kernel"`` events, on every rank, in eager and graph mode alike.
    That was documented for ROCm's roctracer and is CUPTI's behaviour too --
    on CUDA 13.0 / torch 2.13, 2xGB300 TP8, a profiled decode yields 0 kernel
    events with the warm-up and 933k across 340 graph replays without it. The
    hazard did not reproduce there either: all 340 replays ran after the
    attach with no launch failure.

    ``TOKENSPEED_CUPTI_GRAPH_WARMUP=1`` restores it on a stack that still
    needs it, at the cost of GPU profiling. AMD ignores that request: on
    roctracer the warm-up has the same cost and no upside.
    """
    from tokenspeed_kernel.platform import current_platform

    if not envs.TOKENSPEED_CUPTI_GRAPH_WARMUP.get():
        return
    if not torch.cuda.is_available() or current_platform().is_amd:
        return

    from torch.profiler._utils import _init_for_cuda_graphs

    _init_for_cuda_graphs()


@startup_phase("model.config")
def load_model_config(
    server_args: ServerArgs, model_path: str, *, draft_of: ModelConfig | None
) -> ModelConfig:
    """The config of the target model, or of the draft of ``draft_of``.

    Args:
        server_args: The resolved server arguments.
        model_path: The model's path.
        draft_of: The target's config when loading its draft, else None.
    """
    quantization = server_args.quantization
    dtype = server_args.dtype
    if draft_of is not None:
        quantization = server_args.speculative_draft_model_quantization
        if dtype == "auto":
            # A draft is fed the target's hidden states and borrows its
            # embedding and LM head, so the two dtypes have to agree.
            dtype = draft_of.dtype
    return ModelConfig(
        model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.max_model_len,
        model_override_args=server_args.hf_overrides,
        dtype=dtype,
        quantization=quantization,
        server_args=server_args,
        is_draft_worker=draft_of is not None,
    )


@startup_phase("distributed.init")
def init_distributed(
    server_args: ServerArgs,
    port_args: PortArgs,
    *,
    gpu_id: int,
    global_rank: int,
    hidden_size: int,
) -> float:
    """Initialize this rank's process groups; return the free-memory floor
    the KV budget is sized against."""
    # The communication backend reads the resolved launch (node spans,
    # RS/AG buffer capacity, the collective switches) from this dict, and
    # the distributed init already probes it; publish before probing. The
    # model runner republishes after its own resolution.
    global_server_args_dict_update(server_args)
    max_num_input_tokens = (
        server_args.chunked_prefill_size
        if server_args.chunked_prefill_size > 0
        else server_args.max_prefill_tokens + server_args.max_model_len
    )
    distributed_config = DistributedConfig.from_server_args(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        global_rank=global_rank,
        hidden_size=hidden_size,
        max_num_tokens=max_num_input_tokens,
    )
    return DistributedInitializer.initialize(distributed_config)


def request_length_limits(
    *, context_len: int, max_single_request_tokens: int, decode_input_tokens: int
) -> tuple[int, int]:
    """The longest request and the longest prompt this rank admits.

    Args:
        context_len: The model's context length.
        max_single_request_tokens: The most tokens the cache holds for one
            request.
        decode_input_tokens: Verify width per decode step.

    Returns:
        ``(max_model_len, max_req_input_len)``.

    Raises:
        RuntimeError: The cache cannot hold one request with its first
            decode window.
    """
    max_model_len = min(context_len, max_single_request_tokens)
    # Every role reserves the first decode/verify window behind the prompt:
    # a decoding role to verify into it, the prefill role to draft into it.
    input_reserve = max(decode_input_tokens, 1)
    max_req_input_len = max_model_len - input_reserve
    if max_req_input_len < 1:
        raise RuntimeError(
            "The cache cannot admit one request with the configured "
            f"decode reserve: max_single_request_tokens="
            f"{max_single_request_tokens}, reserve={input_reserve}"
        )
    logger.info(
        f"Single-request token limit: cache={max_single_request_tokens!s} "
        f"model={context_len!s} effective={max_model_len!s} "
        f"max_input={max_req_input_len!s}",
    )
    return max_model_len, max_req_input_len


def scheduler_config_from_args(
    server_args,
    specs,
    *,
    storage_config: StorageConfig | None,
    max_scheduled_tokens: int,
    max_batch_size: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
    enable_kv_cache_events: bool,
    prefix_replay_tokens: int,
):
    """Build scheduler config from Device facts and separate Host-storage limits.

    Args:
        server_args: The resolved server arguments.
        specs: The ``DeviceSpecs`` of the built device side: the cache
            geometry, the cache groups and the Host tiers' page counts.
        storage_config: Shared storage scheduling limits, or None
            when no storage backend is configured.
        max_scheduled_tokens: Per-step prefill token budget.
        max_batch_size: Rank-local batch slots.
        decode_input_tokens: Verify width per decode step.
        overlap_schedule_depth: Dispatched-but-uncommitted plans allowed.
        enable_kv_cache_events: Whether KV cache events are published.
        prefix_replay_tokens: DSpark prefix replay tokens, 0 without.
    """
    geometry = specs.cache_geometry
    has_l3 = server_args.kvstore_storage_backend is not None
    return make_config(
        num_device_pages=geometry.num_device_pages,
        max_scheduled_tokens=max_scheduled_tokens,
        max_batch_size=max_batch_size,
        prefix_granularity=geometry.prefix_granularity,
        num_host_pages=specs.num_host_pages,
        disable_l2_cache=not server_args.enable_kvstore,
        enable_l3_storage=has_l3,
        role=server_args.disaggregation_mode,
        num_snapshot_pages=specs.num_snapshot_pages,
        max_retracted_requests=specs.max_retracted_requests,
        storage_config=storage_config,
        debug_force_retraction_interval=server_args.debug_force_retraction_interval,
        enable_kv_cache_events=enable_kv_cache_events,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
        disable_prefix_cache=not server_args.enable_prefix_caching,
        prefix_replay_tokens=prefix_replay_tokens,
        cache_groups=specs.cache_groups,
        enable_mixed_prefill_decode=server_args.enable_mixed_batch,
        state_checkpoint_in_forward=server_args.writes_state_checkpoints_in_forward,
    )
