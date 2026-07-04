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
    dist_init_addr: str | None = None
    distributed_timeout_seconds: int = 1800

    # Node configuration
    nnodes: int = 1
    nprocs_per_node: int = 1

    # Model configuration (needed for attention groups)
    hidden_size: int = 0
    max_num_tokens: int = 0

    # Feature flags
    disable_custom_all_reduce: bool = False
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
            dist_init_addr=server_args.dist_init_addr,
            distributed_timeout_seconds=(
                server_args.distributed_timeout_seconds
                if server_args.distributed_timeout_seconds is not None
                else 1800
            ),
            nnodes=mapping.nnodes,
            nprocs_per_node=mapping.nprocs_per_node,
            hidden_size=hidden_size,
            max_num_tokens=max_num_tokens,
            disable_custom_all_reduce=server_args.disable_custom_all_reduce,
            force_deterministic_rsag=server_args.force_deterministic_rsag,
            mapping=mapping,
        )


class DistributedInitializer:
    @staticmethod
    def initialize(config: DistributedConfig) -> float:
        torch.get_device_module(config.device).set_device(config.gpu_id)
        logger.info(
            "Init torch distributed begin. Avail mem=%.4f GB",
            get_available_gpu_memory(config.device, config.gpu_id),
        )
        if config.device == "cuda":
            maybe_set_numa_aware_cpu_affinity(config.gpu_id)

        # Determine backend
        if config.device == "cuda":
            backend = "nccl"
        else:
            raise ValueError(f"Unsupported device: {config.device}")

        # Build distributed init method
        if config.dist_init_addr:
            dist_init_method = f"tcp://{config.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{config.nccl_port}"

        # Initialize distributed via the mapping-based process group manager
        pg_manager.init_distributed(
            config.mapping,
            backend=backend,
            distributed_init_method=dist_init_method,
            timeout=config.distributed_timeout_seconds,
            device_id=torch.device(config.device, config.gpu_id),
        )
        pg_manager.init_process_group(config.mapping.world_group)
        pg_manager.init_process_group(config.mapping.attn.tp_group)
        pg_manager.init_process_group(config.mapping.dense.tp_group)
        pg_manager.init_process_group(config.mapping.moe.tp_ep_group)

        logger.info(
            "Init comm buff end. Avail mem=%.4f GB",
            get_available_gpu_memory(config.device, config.gpu_id),
        )
        mapping = config.mapping
        logger.info(
            "Current Process distributed state:  global rank: %s  attn_tp_rank: %s  attn_dp_rank: %s",
            mapping.rank,
            mapping.attn.tp_rank,
            mapping.attn.dp_rank,
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
