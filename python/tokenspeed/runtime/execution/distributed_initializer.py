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

from dataclasses import dataclass

import torch

from tokenspeed.runtime.distributed.comm_backend import set_global_backend
from tokenspeed.runtime.distributed.comm_backend.emulated import EmulatedRankBackend
from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import (
    get_available_gpu_memory,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.common import maybe_set_numa_aware_cpu_affinity
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

logger = get_colorful_logger(__name__)


def workspace_token_capacity(
    hidden_size: int, max_num_tokens: int, *, batch_invariant_collectives: bool
) -> int:
    """Rows the trtllm all-reduce workspace of a group is armed for.

    The one-shot window (2 MiB of bf16 rows) by default; configure_group
    widens cross-node groups. Under batch-invariant collectives every
    all-reduce folds ranks in order, and a call the workspace cannot hold
    falls back to all-gathering world_size copies of the partials, so the
    workspace also holds a forward's rows through the two-shot range.
    """
    from tokenspeed_kernel.ops.communication.trtllm import MNNVL_TWOSHOT_MAX_TOKEN

    oneshot_tokens = max(1, (2 * 1024 * 1024) // max(hidden_size * 2, 1))
    if not batch_invariant_collectives:
        return oneshot_tokens
    return max(oneshot_tokens, min(max_num_tokens, MNNVL_TWOSHOT_MAX_TOKEN))


@dataclass
class DistributedConfig:
    """Lightweight configuration for distributed initialization.

    Contains only primitive types (int, str, bool) to avoid heavy dependencies.
    All information needed for distributed setup is captured here.
    """

    # Device configuration
    device: str
    gpu_id: int

    # Distributed topology
    world_size: int
    global_rank: int
    local_rank: int

    # Tensor parallelism
    attn_tp_rank: int
    attn_tp_size: int

    # Data parallelism
    dp_size: int

    # Dense layer parallelism
    dense_tp_size: int

    # Expert parallelism (MoE)
    moe_ep_size: int
    moe_ep_rank: int

    # Network configuration
    nccl_port: int

    # Run this rank alone, with local stand-ins for its collectives
    # (--emulate-rank-zero).
    emulate_rank_zero: bool
    # Full enabled fusion window; zero when fused reductions are disabled.
    comm_fusion_token_capacity: int

    # --batch-invariant-collectives: every all-reduce folds ranks in order.
    batch_invariant_collectives: bool

    dist_init_addr: str | None = None
    distributed_timeout_seconds: int = 1800

    # Node configuration
    nnodes: int = 1
    nprocs_per_node: int = 1

    # Model configuration (needed for attention groups)
    hidden_size: int = 0
    max_num_tokens: int = 0

    # Feature flags
    force_deterministic_rsag: bool = False

    # The full Mapping object for pg_manager initialization
    mapping: object = None

    @classmethod
    def from_server_args(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        global_rank: int,
        hidden_size: int,
        max_num_tokens: int,
    ):
        mapping = server_args.mapping
        return cls(
            device=server_args.device,
            gpu_id=gpu_id,
            world_size=mapping.world_size,
            global_rank=global_rank,
            local_rank=global_rank % mapping.nprocs_per_node,
            attn_tp_rank=mapping.attn.tp_rank,
            attn_tp_size=mapping.attn.tp_size,
            dp_size=mapping.attn.dp_size,
            dense_tp_size=mapping.dense.tp_size,
            moe_ep_size=mapping.moe.ep_size,
            moe_ep_rank=mapping.moe.ep_rank,
            nccl_port=port_args.nccl_port,
            emulate_rank_zero=server_args.emulate_rank_zero,
            comm_fusion_token_capacity=(
                max(0, server_args.comm_fusion_max_num_tokens)
                if server_args.enable_allreduce_fusion
                else 0
            ),
            batch_invariant_collectives=server_args.batch_invariant_collectives,
            dist_init_addr=port_args.dist_init_addr,
            distributed_timeout_seconds=(
                server_args.distributed_timeout_seconds
                if server_args.distributed_timeout_seconds is not None
                else 1800
            ),
            nnodes=mapping.nnodes,
            nprocs_per_node=mapping.nprocs_per_node,
            hidden_size=hidden_size,
            max_num_tokens=max_num_tokens,
            force_deterministic_rsag=server_args.force_deterministic_rsag,
            mapping=mapping,
        )


class DistributedInitializer:
    @staticmethod
    def initialize(config: DistributedConfig) -> float:
        torch.get_device_module(config.device).set_device(config.gpu_id)
        logger.info(
            "Init torch distributed begin. Avail mem="
            f"{get_available_gpu_memory(config.device, config.gpu_id):.4f} GB",
        )
        if config.device == "cuda":
            maybe_set_numa_aware_cpu_affinity(config.gpu_id)

        # Determine backend
        if config.device == "cuda":
            backend = "nccl"
        elif config.device == "npu":
            backend = "hccl"
        else:
            raise ValueError(f"Unsupported device: {config.device}")

        # Build distributed init method
        if config.dist_init_addr:
            dist_init_method = f"tcp://{config.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{config.nccl_port}"

        # Pass the device so PyTorch binds the process group to it (eager NCCL
        # init) instead of inferring it later — this also mutes the c10d
        # "barrier(): using the device under current context" warning.
        device_id = (
            None if backend == "hccl" else torch.device(config.device, config.gpu_id)
        )

        if config.emulate_rank_zero:
            pg_manager.init_emulated_rank_zero(
                backend=backend,
                distributed_init_method=dist_init_method,
                timeout=config.distributed_timeout_seconds,
                device_id=device_id,
            )
            set_global_backend(EmulatedRankBackend(rank=config.mapping.rank))
        else:
            # Initialize distributed via the mapping-based process group manager
            pg_manager.init_distributed(
                config.mapping,
                backend=backend,
                distributed_init_method=dist_init_method,
                timeout=config.distributed_timeout_seconds,
                device_id=device_id,
            )
        pg_manager.init_process_group(config.mapping.world_group)
        pg_manager.init_process_group(config.mapping.attn.world_group)
        pg_manager.init_process_group(config.mapping.attn.tp_group)
        # The KVP group is the group the DCP decode collectives address; a
        # group of one still is. init_process_group is idempotent and handles
        # size 1.
        pg_manager.init_process_group(config.mapping.attn.kvp_group)
        if config.mapping.attn.has_kvp:
            pg_manager.init_process_group(
                config.mapping.attn.kvp_group,
                backend=backend,
                name=KVP_HISTORY_GROUP_NAME,
            )
        # The query-context-parallel group of a sharded extend; equal to the
        # attention TP group while qcp == tp, so this is idempotent there.
        pg_manager.init_process_group(config.mapping.attn.qcp_group)
        pg_manager.init_process_group(config.mapping.attn.dp_group)
        # No-op at the default linear_attn.tp == attn.tp (same group,
        # idempotent).
        pg_manager.init_process_group(config.mapping.linear_attn.tp_group)
        # Head-sharded attention projections and the vocab-sharded LM head
        # under attention DP; both default to groups created above.
        pg_manager.init_process_group(config.mapping.attn.head_tp_group)
        pg_manager.init_process_group(config.mapping.lm_head.tp_group)
        pg_manager.init_process_group(config.mapping.dense.tp_group)
        pg_manager.init_process_group(config.mapping.moe.tp_ep_group)
        if config.mapping.has_pp:
            # Cross-stage group for hidden-state P2P (nccl) and small control
            # broadcasts like the sampled first token (gloo).
            pg_manager.init_process_group(config.mapping.pp_group)

        from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

        gather_fabric_map()

        # Arm the trtllm AR workspaces; --force-deterministic-rsag overrides at dispatch.
        if config.hidden_size > 0:
            from tokenspeed.runtime.distributed.comm_backend import (
                get_global_backend,
            )

            backend = get_global_backend()
            trtllm_ar = getattr(backend, "trtllm_ar", None)
            if trtllm_ar is not None:
                # Plain one-shot reductions and fused norms share this
                # workspace. Prefill may never capture a decode graph, so
                # arm the full fusion window here, before serving begins.
                max_workspace_tokens = max(
                    workspace_token_capacity(
                        config.hidden_size,
                        config.max_num_tokens,
                        # The rsag override keeps the fold off the workspace.
                        batch_invariant_collectives=config.batch_invariant_collectives
                        and not config.force_deterministic_rsag,
                    ),
                    config.comm_fusion_token_capacity,
                )
                for group in {
                    config.mapping.attn.tp_group,
                    config.mapping.moe.tp_ep_group,
                }:
                    if len(group) > 1:
                        ok = trtllm_ar.configure_group(
                            rank=group.index(config.mapping.rank),
                            group=group,
                            max_token_num=max_workspace_tokens,
                            hidden_dim=config.hidden_size,
                        )
                        logger.info(
                            f"trtllm one-shot all-reduce for group {group!s}: "
                            f"{('enabled' if ok else 'unavailable (NCCL fallback)')!s}",
                        )

        logger.info(
            "Init comm buff end. Avail mem="
            f"{get_available_gpu_memory(config.device, config.gpu_id):.4f} GB",
        )
        mapping = config.mapping
        logger.info(
            f"Current Process distributed state:  global rank: {mapping.rank!s}  "
            f"attn_tp_rank: {mapping.attn.tp_rank!s}  attn_dp_rank: "
            f"{mapping.attn.dp_rank!s}",
        )

        # Get minimum available GPU memory across all ranks
        min_per_gpu_memory = get_available_gpu_memory(
            config.device,
            config.gpu_id,
            distributed=config.world_size > 1,
            cpu_group=pg_manager.get_process_group("gloo", mapping.world_group),
        )

        # Verify memory balance for tensor parallelism
        if config.world_size > 1:
            local_gpu_memory = get_available_gpu_memory(config.device, config.gpu_id)
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. "
                    "Some GPUs may be occupied by other processes."
                )

        return min_per_gpu_memory
