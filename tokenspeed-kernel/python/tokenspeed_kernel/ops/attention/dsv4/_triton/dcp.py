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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _normalize_dcp_partials_kernel(
    output,
    lse,
    swa_lens,
    extra_lens,
    normalized_lse,
    os_t,
    os_h,
    os_d,
    ls_t,
    ls_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    nonempty = tl.load(swa_lens + token) > 0
    if extra_lens is not None:
        nonempty = nonempty | (tl.load(extra_lens + token) > 0)
    value = tl.load(lse + token * ls_t + head * ls_h)
    # A kernel may report an empty row as +inf rather than -inf (FlashMLA
    # does); a non-finite LSE therefore marks an empty selection too, so the
    # weighting never sees exp(inf - inf).
    nonempty = nonempty & (value == value) & (value != float("inf"))
    tl.store(
        normalized_lse + token * HEADS + head, tl.where(nonempty, value, -float("inf"))
    )
    if not nonempty:
        offsets = tl.arange(0, BLOCK).to(tl.int64)
        tl.store(output + token * os_t + head * os_h + offsets * os_d, 0, offsets < DIM)


def normalize_dcp_partials(
    output: torch.Tensor,
    lse: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize empty attention partials for the runtime DCP combine.

    Args:
        output: FP16/BF16/FP32 CUDA partials [tokens, heads, dim], possibly
            strided, updated in place only for empty selections. Nonempty
            values (including NaNs) are preserved.
        lse: Natural-log FP32 [tokens, heads], possibly strided.
        swa_lens: Int32 valid SWA lengths [tokens], possibly noncontiguous.
        extra_lens: Optional int32 local valid compressed counts [tokens].
            These exclude masked holes, unlike the attention scan lengths.
            Noncontiguous length vectors are copied to contiguous storage.

    Returns:
        The input output tensor, and contiguous FP32 LSE. Empty rows become
        output=0 and LSE=-inf; nonempty rows retain their original bits. A row
        is empty when both lengths are zero or when its LSE is NaN or +inf,
        which is how a kernel reports attending to no key at all.
    """
    if output.ndim != 3 or lse.ndim != 2 or lse.shape != output.shape[:2]:
        raise ValueError("DCP normalization shapes disagree")
    if output.shape[-1] <= 0:
        raise ValueError("DCP normalization requires a positive head dimension")
    if (
        output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or lse.dtype != torch.float32
    ):
        raise TypeError("DCP normalization requires floating partials and FP32 LSE")
    if not output.is_cuda or lse.device != output.device:
        raise ValueError("DCP normalization requires tensors on one CUDA device")
    for name, lengths in (("swa_lens", swa_lens), ("extra_lens", extra_lens)):
        if lengths is None and name == "extra_lens":
            continue
        if lengths.shape != (output.shape[0],):
            raise ValueError(f"{name} must have shape [tokens]")
        if lengths.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if lengths.device != output.device:
            raise ValueError(f"{name} must be on the partials device")
    normalized = torch.empty(
        output.shape[:2], dtype=torch.float32, device=output.device
    )
    if normalized.numel() == 0:
        return output, normalized
    _normalize_dcp_partials_kernel[(output.shape[0], output.shape[1])](
        output,
        lse,
        swa_lens.contiguous(),
        extra_lens.contiguous() if extra_lens is not None else None,
        normalized,
        *output.stride(),
        *lse.stride(),
        HEADS=output.shape[1],
        DIM=output.shape[2],
        BLOCK=triton.next_power_of_2(output.shape[2]),
    )
    return output, normalized


@triton.jit
def _dcp_weight_kernel(
    output,
    all_lse,
    weighted,
    global_lse,
    os_t,
    os_h,
    os_d,
    ls_r,
    ls_t,
    ls_h,
    tokens,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    SHARDS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    shards = tl.arange(0, SHARDS).to(tl.int64)
    lse = tl.load(
        all_lse + shards * ls_r + token * ls_t + head * ls_h,
        shards < DEGREE,
        other=-float("inf"),
    )
    # +inf is a kernel's "attended to nothing", never a finite softmax; treat
    # it as the empty shard it is instead of producing exp(inf - inf).
    lse = tl.where(lse == float("inf"), -float("inf"), lse)
    maximum = tl.max(lse, 0)
    has_nan = tl.sum((lse != lse).to(tl.int32), 0) > 0
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.exp(lse - safe_max)
    denominator = tl.sum(mass, 0)
    rank = tl.full((), RANK, tl.int64)
    local_lse = tl.load(all_lse + rank * ls_r + token * ls_t + head * ls_h)
    local_lse = tl.where(local_lse == float("inf"), -float("inf"), local_lse)
    weight = tl.exp(local_lse - safe_max) / tl.maximum(
        denominator, 1.1754943508222875e-38
    )
    weight = tl.where(has_nan, float("nan"), weight)
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    values = tl.load(
        output + token * os_t + head * os_h + offsets * os_d, offsets < DIM, other=0
    ).to(tl.float32)
    values = tl.where(local_lse == -float("inf"), 0.0, values)
    # Program IDs already carry int64 address arithmetic. tokens can be the
    # Python constant 1 under Triton's scalar specialization.
    tl.store(
        weighted + (head * tokens + token) * DIM + offsets,
        values * weight,
        offsets < DIM,
    )
    local_heads = HEADS // DEGREE
    if head >= RANK * local_heads and head < (RANK + 1) * local_heads:
        combined = tl.where(
            denominator == 0.0, -float("inf"), maximum + tl.log(denominator)
        )
        combined = tl.where(has_nan, float("nan"), combined)
        tl.store(global_lse + token * local_heads + head - RANK * local_heads, combined)


@triton.jit
def _dcp_sink_kernel(
    output,
    lse,
    sink,
    result,
    os_t,
    os_h,
    os_d,
    ls_t,
    ls_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    normalizer = tl.load(lse + token * ls_t + head * ls_h)
    sink_logit = tl.load(sink + head).to(tl.float32)
    factor = tl.where(
        normalizer == -float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_logit - normalizer))
    )
    value = tl.load(
        output + token * os_t + head * os_h + offsets * os_d, offsets < DIM, other=0
    ).to(tl.float32)
    tl.store(
        result + (token * HEADS + head) * DIM + offsets, value * factor, offsets < DIM
    )


def dcp_weight_for_reduce_scatter(
    output: torch.Tensor, all_lse: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 head-major weighted O and TP-local global LSE for DCP AG+RS.

    Args:
        output: Local no-sink O [tokens, gathered_heads, dim], on CUDA.
        all_lse: Gathered FP32 natural-log LSE [shards, tokens, gathered_heads].
        rank: Context rank and destination TP head slice within the DCP group.

    Returns:
        FP32 [gathered_heads, tokens, dim] for reduce-scatter and FP32
        [tokens, gathered_heads / shards] global LSE, with empty rows at -inf.
    """
    if output.ndim != 3 or all_lse.ndim != 3 or all_lse.shape[1:] != output.shape[:2]:
        raise ValueError("DCP weight shapes disagree")
    degree, tokens, heads = all_lse.shape
    if degree <= 0 or not 0 <= rank < degree or heads % degree:
        raise ValueError("DCP weight topology is invalid")
    if (
        not output.is_cuda
        or all_lse.device != output.device
        or all_lse.dtype != torch.float32
    ):
        raise ValueError("DCP weights require matching CUDA output and FP32 LSE")
    dim = output.shape[-1]
    weighted = torch.empty(
        (heads, tokens, dim), device=output.device, dtype=torch.float32
    )
    lse = torch.empty(
        (tokens, heads // degree), device=output.device, dtype=torch.float32
    )
    if tokens == 0:
        return weighted, lse
    _dcp_weight_kernel[(tokens, heads)](
        output,
        all_lse,
        weighted,
        lse,
        *output.stride(),
        *all_lse.stride(),
        tokens,
        HEADS=heads,
        DIM=dim,
        DEGREE=degree,
        RANK=rank,
        BLOCK=triton.next_power_of_2(dim),
        SHARDS=triton.next_power_of_2(degree),
        enable_fp_fusion=False,
        num_warps=4,
    )
    return weighted, lse


def dcp_apply_sink(
    output: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """Scale CUDA no-sink O by one sink and cast directly to its output dtype.

    Args:
        output: FP32 O [tokens, TP-local heads, dim], possibly a transposed RS view.
        lse: FP32 natural-log LSE [tokens, TP-local heads].
        sink: Contiguous sink logits, covering at least the TP-local heads.
        dtype: Desired output dtype, BF16 or FP16.

    Returns:
        Contiguous [tokens, TP-local heads, dim] with the sink counted once.
    """
    if (
        output.ndim != 3
        or lse.shape != output.shape[:-1]
        or sink.numel() < output.shape[1]
    ):
        raise ValueError("DCP sink shapes disagree")
    if (
        not output.is_cuda
        or lse.device != output.device
        or sink.device != output.device
        or not sink.is_contiguous()
    ):
        raise ValueError("DCP sink tensors require one CUDA device and contiguous sink")
    if lse.dtype != torch.float32 or dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("DCP sink requires FP32 LSE and 16-bit output")
    tokens, heads, dim = output.shape
    result = torch.empty(output.shape, device=output.device, dtype=dtype)
    if tokens == 0:
        return result
    _dcp_sink_kernel[(tokens, heads)](
        output,
        lse,
        sink,
        result,
        *output.stride(),
        *lse.stride(),
        HEADS=heads,
        DIM=dim,
        BLOCK=triton.next_power_of_2(dim),
        enable_fp_fusion=False,
        num_warps=4,
    )
    return result
