# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/
# of https://github.com/vllm-project/vllm (Apache-2.0), unmodified. The
# tokenspeed orchestration lives in ops/moe/latent_tail.py.

"""CuTe DSL kernels for the Kimi-K3 latent-MoE tail fusion."""

from .allreduce_rmsnorm_reduce_scatter_early_exit import CollectiveKernel
from .fused_add_multicast_gemm import AdaptiveUpProjectionKernel
from .lamport_copy import LamportCopyKernel

__all__ = [
    "AdaptiveUpProjectionKernel",
    "CollectiveKernel",
    "LamportCopyKernel",
]
