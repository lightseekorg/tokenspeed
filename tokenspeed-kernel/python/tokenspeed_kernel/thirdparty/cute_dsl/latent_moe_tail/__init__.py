# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM; preserve the fused-residual gather and cluster-barrier phase fix.

"""CuTe DSL kernels for the Kimi-K3 latent-MoE tail fusion."""

from .allreduce_rmsnorm_reduce_scatter_early_exit import CollectiveKernel
from .fused_add_multicast_gemm import AdaptiveUpProjectionKernel
from .lamport_copy import LamportCopyKernel

__all__ = [
    "AdaptiveUpProjectionKernel",
    "CollectiveKernel",
    "LamportCopyKernel",
]
