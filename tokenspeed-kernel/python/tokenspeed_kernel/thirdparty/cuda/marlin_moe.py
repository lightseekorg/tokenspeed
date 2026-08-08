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

"""Marlin WNA16 MoE GEMM (weight-only 4-bit, grouped experts).

Wraps the pre-compiled vendored Marlin MoE kernel (see csrc/marlin_moe/). Only
the bf16-activation specializations are exported: MXFP4 with E8M0 group-32
scales is numerically valid only on the bf16 path.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# ScalarType id for float4_e2m1f, matching the C++ ScalarType bit encoding
# (exponent=2:8b | mantissa=1:8b | signed=1:1b | bias=0:32b |
# finite_values_only=1:1b | nan_repr=NONE(0):8b). Keep in sync with
# csrc/marlin_moe/include/scalar_type.hpp (kFE2M1f).
FLOAT4_E2M1F_ID = (2) | (1 << 8) | (1 << 16) | (0 << 17) | (1 << 49) | (0 << 50)

# Max thread_n across kernel configs; sizes the fp32-reduce scratch. Matches
# device::marlin_moe in csrc/marlin_moe/moe_wna16_marlin.cuh.
_MAX_THREAD_N = 256


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_marlin_moe_module():
    """Load the pre-compiled marlin_moe shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "marlin_moe" / "marlin_moe.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel marlin_moe library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def is_marlin_moe_available() -> bool:
    """Report whether the pre-compiled marlin_moe library is present."""
    return (_objs_dir() / "marlin_moe" / "marlin_moe.so").exists()


def _empty(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=dtype)


def marlin_make_workspace(
    device: torch.device, max_blocks_per_sm: int = 4
) -> torch.Tensor:
    """Allocate the zero-initialized inter-block sync workspace.

    Args:
        device: CUDA device to allocate on.
        max_blocks_per_sm: Upper bound of resident blocks per SM used to size
            the lock array.

    Returns:
        int32 zeros ``[sms * max_blocks_per_sm]``. Allocate per call when
        capturing CUDA graphs: captured graphs must not alias another graph's
        locks, or replay can deadlock on stale lock state.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def _get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    """Permute a 2D scale tensor ``[size_k/group, size_n]`` into Marlin order.

    Matches the layout the Marlin GEMM expects for its dequant path. When the
    group spans fewer rows than ``size_k`` (grouped quant, e.g. MXFP4 g32) the
    interleaved 8x8 perm is used; otherwise the per-channel single perm.
    """
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def mxfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Reorder + retype permuted MXFP4 scales for the bf16 Marlin path.

    The bf16 kernel consumes E8M0 scales in a 4-lane [0,2,1,3] order. Input is
    the ``marlin_permute_scales`` output as uint8 (raw E8M0 bytes).
    """
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    return marlin_scales.view(torch.float8_e8m0fnu)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort routes by expert and pad each expert's routes to ``block_size``.

    Args:
        topk_ids: int32 ``[tokens, top_k]`` global expert ids. Out-of-range
            ids (EP-masked experts) still get slots; the Marlin kernel skips
            blocks whose expert id falls outside the local range.
        block_size: Token block granularity of the downstream grouped GEMM.
        num_experts: Global expert count.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_padded)`` — flat
        route ids padded per expert, the expert id of each block, and the
        total padded route count (device scalar).
    """
    device = topk_ids.device
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    # One extra lane holds EP-invalid routes; expert ids are emitted as
    # ``eid - 1`` so real experts land back on [0, num_experts).
    cumsum_buffer = torch.empty(num_experts + 2, dtype=torch.int32, device=device)

    _load_marlin_moe_module().moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,  # pad_sorted_token_ids
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    size_m: int,
    size_n: int,
    size_k: int,
    use_fp32_reduce: bool = True,
) -> torch.Tensor:
    """Run one grouped Marlin W4A16 GEMM over routed expert tokens.

    Computes ``c[route] = a[token(route)] @ dequant(b_q_weight[expert(route)])``
    for every route produced by ``moe_align_block_size``-style scheduling.
    MXFP4 weights only (E2M1 packed, E8M0 group-32 scales); activation stays
    bf16 (W4A16 -- no FP4 tensor cores needed, runs on SM90).

    Args:
        a: bf16 activations ``[size_m, size_k]``.
        c: Optional output ``[size_m * top_k, size_n]``; allocated when None.
            Must be zero-filled by the caller if experts can be masked (EP).
        b_q_weight: Marlin-repacked weights ``[E, size_k/16, size_n*2]`` int32.
        b_scales: Marlin-permuted scales ``[E, size_k/32, size_n]``
            float8_e8m0fnu.
        workspace: int32 lock array from :func:`marlin_make_workspace`.
        sorted_token_ids: Routes padded per expert to ``moe_block_size``.
        expert_ids: Expert id per scheduled block.
        num_tokens_post_padded: Total scheduled routes after padding.
        topk_weights: ``[size_m, top_k]`` float32 route weights.
        moe_block_size: Token block size used by the alignment step (8-64).
        top_k: Routes per token.
        mul_topk_weights: Scale outputs by ``topk_weights`` (GEMM2 finalize).
        is_ep: Expert-parallel mode; out-of-range expert ids are skipped.
        size_m/size_n/size_k: GEMM problem shape.
        use_fp32_reduce: Reduce partial tiles in fp32 scratch (recommended).

    Returns:
        ``c`` with dtype of ``a``.
    """
    device = a.device
    if c is None:
        c = torch.zeros((size_m * top_k, size_n), dtype=a.dtype, device=device)
    if size_m == 0:
        return c

    num_groups = b_scales.size(1)
    group_size = size_k // num_groups if num_groups > 1 else -1

    # fp32 partial-tile reduction scratch (no atomic-add on the mxfp4 path:
    # E8M0 bf16 atomics accumulate in bf16 and lose precision).
    if use_fp32_reduce:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_c_tmp_size = min(
            size_n * sorted_token_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    empty_a = _empty(device, a.dtype)
    empty_i32 = _empty(device, torch.int32)

    module = _load_marlin_moe_module()
    fn = (
        module.moe_wna16_marlin_gemm_bf16_ep
        if is_ep
        else module.moe_wna16_marlin_gemm_bf16
    )
    fn(
        a,
        c,
        b_q_weight,
        empty_a,  # b_bias
        b_scales,
        empty_a,  # global_scale
        empty_a,  # b_zeros
        empty_i32,  # g_idx
        empty_i32,  # perm
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        empty_a,  # a_tmp (act_order only)
        c_tmp,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        FLOAT4_E2M1F_ID,
        size_m,
        size_n,
        size_k,
        False,  # has_act_order
        False,  # has_bias
        True,  # is_k_full
        False,  # has_zp
        num_groups,
        group_size,
        False,  # use_atomic_add
        use_fp32_reduce,
        False,  # is_zp_float
    )
    return c
