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

"""TRT-LLM manifold-constrained hyper-connection kernels."""

from functools import cache

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.trtllm import has_mhc_kernels
from tokenspeed_kernel.thirdparty.trtllm import mhc_big_fuse as _mhc_big_fuse
from tokenspeed_kernel.thirdparty.trtllm import mhc_fused_hc as _mhc_fused_hc
from tokenspeed_kernel.thirdparty.trtllm import mhc_post_mapping as _mhc_post_mapping

_SUPPORTED_HIDDEN_SIZES = frozenset({4096, 7168})
_SUPPORTED_HC_MULT = 4
_MHC_KERNELS_AVAILABLE = has_mhc_kernels()
_BLACKWELL_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 9),
    vendors=frozenset({"nvidia"}),
)


def has_trtllm_mhc() -> bool:
    """Return whether the installed TRT-LLM extension exports the mHC ops."""

    return _MHC_KERNELS_AVAILABLE


@cache
def _is_blackwell_device(device_index: int) -> bool:
    major, _minor = torch.cuda.get_device_capability(device_index)
    return major == 10


def supports_trtllm_mhc(
    device: torch.device,
    hc_mult: int,
    hidden_size: int,
) -> bool:
    """Return whether TRT-LLM mHC kernels support a tensor configuration.

    Args:
        device: CUDA device that owns the mHC tensors.
        hc_mult: Number of hyper-connection lanes.
        hidden_size: Per-lane hidden dimension.

    Returns:
        ``True`` only when the extension is installed and its SM100 kernel
        contract covers the requested lane count and hidden size.
    """

    if (
        not _MHC_KERNELS_AVAILABLE
        or device.type != "cuda"
        or not torch.cuda.is_available()
        or hc_mult != _SUPPORTED_HC_MULT
        or hidden_size not in _SUPPORTED_HIDDEN_SIZES
    ):
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _is_blackwell_device(device_index)


# fused_hc launch table, selected by a B200 sweep over the vendored kernel's
# backend/tile/k-split parameters (all 768 combinations, graph-replay timed,
# parity-checked against the composed post_mapping + prenorm-GEMM + big_fuse
# path). backend=1 is the two-stage fma_ksplit + big_fuse organization; it
# beats the allinone default by ~31% at M=32 (15.2us vs 22.2us) and stays
# flat past the M>32 allinone-mma cliff. Two constraints shape the table:
#   * backend=1 accumulates num_k_splits x M partial rows into the y_acc and
#     r_acc workspaces; callers must size them accordingly (see
#     FUSED_HC_MAX_K_SPLITS). Undersized accumulators corrupt small-M
#     outputs and crash at scale.
#   * small token counts stay on the allinone-fma default, which is at
#     least as fast there.
_FUSED_HC_SMALL_M_MAX = 12
_FUSED_HC_MEDIUM_M_MAX = 32

FUSED_HC_MAX_K_SPLITS = 2
"""Largest ``num_k_splits`` the launch table may select.

The two-stage k-split backend writes ``num_k_splits * num_tokens`` partial
rows into the ``y_acc``/``r_acc`` accumulator workspaces, so callers must
allocate ``FUSED_HC_MAX_K_SPLITS * max_tokens`` rows for them.
"""


def _select_fused_hc_launch(num_tokens: int) -> tuple[int, int, int, int]:
    """Pick the fused_hc backend and tile configuration for a token count.

    Args:
        num_tokens: Number of tokens (rows) in the fused mHC call.

    Returns:
        Tuple of ``(backend, tile_n, num_k_splits, bigfuse_block_size)``
        kernel launch parameters.
    """

    if num_tokens <= _FUSED_HC_SMALL_M_MAX:
        return 3, 1, 1, 0  # fused_all_fma (allinone) default
    # bigfuse_block_size=512 over the kernel default: 15.1 -> 14.7us at M=32,
    # 11.9 -> 9.6us at M=16 (graph-replay microbench; endpoint-neutral).
    if num_tokens <= _FUSED_HC_MEDIUM_M_MAX:
        return 1, 2, 2, 512  # fma_ksplit + big_fuse
    return 1, 4, 2, 512


trtllm_mhc_big_fuse = error_fn
trtllm_mhc_fused_hc = error_fn
trtllm_mhc_post_mapping = error_fn

if _MHC_KERNELS_AVAILABLE and current_platform().is_nvidia:

    @register_kernel(
        "mhc",
        "pre_mapping",
        name="trtllm_mhc_big_fuse",
        solution="trtllm",
        capability=_BLACKWELL_CAPABILITY,
        signatures=format_signatures("residual", "dense", {torch.bfloat16}),
        traits={
            "hc_mult": frozenset({_SUPPORTED_HC_MULT}),
            "hidden_size": _SUPPORTED_HIDDEN_SIZES,
        },
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )
    def trtllm_mhc_big_fuse(
        y_acc: torch.Tensor,
        r_acc: torch.Tensor,
        residual: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> None:
        """Run the TRT-LLM mHC pre-mapping epilogue.

        Args:
            y_acc: FP32 GEMM accumulators.
            r_acc: FP32 residual square-sum accumulators.
            residual: BF16 hyper-connected residual input.
            hc_scale: FP32 pre/post/comb scale values.
            hc_base: FP32 pre/post/comb bias values.
            post_mix: FP32 post-mapping output buffer.
            comb_mix: FP32 combination-matrix output buffer.
            layer_input: BF16 reduced layer-input output buffer.
            rms_eps: RMS normalization epsilon.
            hc_eps: Hyper-connection normalization epsilon.
            sinkhorn_iters: Number of Sinkhorn normalization iterations.

        Returns:
            None. Output tensors are populated in place.
        """

        num_tokens, hc_mult, hidden_size = residual.shape
        _mhc_big_fuse(
            y_acc,
            r_acc,
            residual,
            hc_scale,
            hc_base,
            post_mix,
            comb_mix,
            layer_input,
            num_tokens,
            hc_mult * hidden_size,
            hidden_size,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_iters,
            y_acc.shape[0],
            256,
        )

    @register_kernel(
        "mhc",
        "post_mapping",
        name="trtllm_mhc_post_mapping",
        solution="trtllm",
        capability=_BLACKWELL_CAPABILITY,
        signatures=format_signatures("residual", "dense", {torch.bfloat16}),
        traits={
            "hc_mult": frozenset({_SUPPORTED_HC_MULT}),
            "hidden_size": _SUPPORTED_HIDDEN_SIZES,
        },
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )
    def trtllm_mhc_post_mapping(
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Run the TRT-LLM mHC post-mapping kernel.

        Args:
            residual: BF16 hyper-connected residual input.
            hidden_states: BF16 layer output to mix into the residual.
            post_mix: FP32 per-lane post-mapping weights.
            comb_mix: FP32 lane-combination matrices.
            output: BF16 hyper-connected residual output buffer.

        Returns:
            None. ``output`` is populated in place.
        """

        num_tokens, _hc_mult, hidden_size = residual.shape
        _mhc_post_mapping(
            residual,
            hidden_states,
            post_mix,
            comb_mix,
            output,
            num_tokens,
            hidden_size,
        )

    @register_kernel(
        "mhc",
        "fused_post_pre",
        name="trtllm_mhc_fused_hc",
        solution="trtllm",
        capability=_BLACKWELL_CAPABILITY,
        signatures=format_signatures(
            ("x_prev", "residual_prev"), "dense", {torch.bfloat16}
        ),
        traits={
            "hc_mult": frozenset({_SUPPORTED_HC_MULT}),
            "hidden_size": _SUPPORTED_HIDDEN_SIZES,
        },
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )
    def trtllm_mhc_fused_hc(
        x_prev: torch.Tensor,
        residual_prev: torch.Tensor,
        post_mix_prev: torch.Tensor,
        comb_mix_prev: torch.Tensor,
        weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        residual_cur: torch.Tensor,
        post_mix_cur: torch.Tensor,
        comb_mix_cur: torch.Tensor,
        layer_input_cur: torch.Tensor,
        y_acc_workspace: torch.Tensor,
        r_acc_workspace: torch.Tensor,
        done_counter_workspace: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> None:
        """Run TRT-LLM's fused previous-post/current-pre mHC kernel.

        Args:
            x_prev: BF16 previous layer output.
            residual_prev: BF16 previous hyper-connected residual.
            post_mix_prev: FP32 previous post-mapping weights.
            comb_mix_prev: FP32 previous lane-combination matrices.
            weight: FP32 current pre-mapping projection weight.
            hc_scale: FP32 current pre/post/comb scale values.
            hc_base: FP32 current pre/post/comb bias values.
            residual_cur: BF16 current residual output buffer.
            post_mix_cur: FP32 current post-mapping output buffer.
            comb_mix_cur: FP32 current combination output buffer.
            layer_input_cur: BF16 current layer-input output buffer.
            y_acc_workspace: FP32 projection accumulator workspace.
            r_acc_workspace: FP32 square-sum accumulator workspace.
            done_counter_workspace: INT32 kernel synchronization workspace.
            rms_eps: RMS normalization epsilon.
            hc_eps: Hyper-connection normalization epsilon.
            sinkhorn_iters: Number of Sinkhorn normalization iterations.

        Returns:
            None. Output tensors and workspaces are populated in place.
        """

        num_tokens, hc_mult, hidden_size = residual_prev.shape
        backend, tile_n, num_k_splits, bigfuse_block_size = _select_fused_hc_launch(
            num_tokens
        )
        _mhc_fused_hc(
            x_prev,
            residual_prev,
            post_mix_prev,
            comb_mix_prev,
            weight,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            y_acc_workspace,
            r_acc_workspace,
            done_counter_workspace,
            num_tokens,
            hidden_size,
            hc_mult,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_iters,
            backend,
            tile_n,
            num_k_splits,
            bigfuse_block_size,
            1,
            None,
            0.0,
        )


__all__ = [
    "has_trtllm_mhc",
    "supports_trtllm_mhc",
    "trtllm_mhc_big_fuse",
    "trtllm_mhc_fused_hc",
    "trtllm_mhc_post_mapping",
]
