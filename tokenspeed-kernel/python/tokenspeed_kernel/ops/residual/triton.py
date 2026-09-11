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

"""Triton residual kernels."""

from __future__ import annotations

import threading
from functools import cache

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_LOW_PRECISION_DTYPES = {torch.float16, torch.bfloat16}
_MIX_SIGNATURES = format_signatures(
    ("normalized", "projection_weight", "up_weight"), "dense", _DTYPES
)
_PERSISTENT_MIX_SIGNATURES = format_signatures(
    ("normalized", "projection_weight", "up_weight"),
    "dense",
    _LOW_PRECISION_DTYPES,
)
_COMBINE_SIGNATURES = format_signatures(
    ("block_output", "residual", "inject_logits"), "dense", _DTYPES
)


@triton.jit
def _projection_epilogue_kernel(
    projected_ptr,
    activated_ptr,
    inject_ptr,
    projected_row_stride,
    activated_row_stride,
    inject_row_stride,
    projection_scale,
    LOWRANK: tl.constexpr,
    ACTIVATION_WIDTH: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    down_mask = offsets < LOWRANK
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    value = tl.load(
        projected_ptr + row * projected_row_stride + offsets,
        mask=down_mask,
        other=0.0,
    ).to(tl.float32)
    value *= projection_scale
    activated = value * tl.sigmoid(value)
    tl.store(
        activated_ptr + row * activated_row_stride + offsets,
        activated,
        mask=offsets < ACTIVATION_WIDTH,
    )
    if HAS_INJECT:
        inject_mask = offsets < HC_COUNT
        inject = tl.load(
            projected_ptr + row * projected_row_stride + LOWRANK + offsets,
            mask=inject_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            inject_ptr + row * inject_row_stride + offsets,
            inject * projection_scale,
            mask=inject_mask,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _mix_epilogue_kernel(
    gate_ptr,
    normalized_ptr,
    out_ptr,
    gate_row_stride,
    normalized_row_stride,
    out_row_stride,
    hidden_size,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden_size
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    mixed = tl.zeros([BLOCK], dtype=tl.float32)
    for branch in tl.static_range(HC_COUNT):
        column = branch * hidden_size + offsets
        gate = tl.load(
            gate_ptr + row * gate_row_stride + column, mask=mask, other=0.0
        ).to(tl.float32)
        value = tl.load(
            normalized_ptr + row * normalized_row_stride + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mixed += tl.sigmoid(gate) * value
    tl.store(out_ptr + row * out_row_stride + offsets, mixed / HC_COUNT, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _combine_kernel(
    residual_ptr,
    block_ptr,
    inject_ptr,
    out_ptr,
    residual_row_stride,
    block_row_stride,
    inject_row_stride,
    out_row_stride,
    hidden_size,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden_size
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    value = tl.load(
        block_ptr + row * block_row_stride + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    for branch in tl.static_range(HC_COUNT):
        logit = tl.load(inject_ptr + row * inject_row_stride + branch).to(tl.float32)
        column = branch * hidden_size + offsets
        residual = tl.load(
            residual_ptr + row * residual_row_stride + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_ptr + row * out_row_stride + column,
            residual + value * 2.0 * tl.sigmoid(logit),
            mask=mask,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _persistent_mix_kernel(
    x_ptr,
    projection_weight_ptr,
    up_weight_ptr,
    projection_raw_ptr,
    mixed_ptr,
    inject_ptr,
    counters_ptr,
    K,
    num_rows,
    num_ctas,
    projection_scale,
    ROWS: tl.constexpr,
    PROJECTION_ROWS: tl.constexpr,
    LOWRANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """One-resident-grid decode mix with stream-private barrier state."""
    pid = tl.program_id(0)
    offsets_m = tl.arange(0, ROWS)
    mask_m = offsets_m < num_rows

    # A preceding PDL producer may still be draining. This kernel reuses a
    # stream-private scratch tensor across layers, so wait before even zeroing
    # it; consecutive persistent launches are therefore safe as well.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    zero_span = ROWS * PROJECTION_ROWS
    offsets_z = tl.arange(0, 256)
    for zero_start in range(pid * 256, zero_span, num_ctas * 256):
        indices = zero_start + offsets_z
        tl.store(projection_raw_ptr + indices, 0.0, mask=indices < zero_span)
    _grid_barrier(counters_ptr, num_ctas)

    offsets_k = tl.arange(0, BLOCK_K)
    offsets_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(PROJECTION_ROWS, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        n_block = tile % n_blocks
        k_chunk = tile // n_blocks
        n = n_block * BLOCK_N + offsets_n
        k = k_chunk * BLOCK_K + offsets_k
        mask_n = n < PROJECTION_ROWS
        x = tl.load(
            x_ptr + offsets_m[:, None] * K + k[None, :],
            mask=mask_m[:, None] & (k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            projection_weight_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None] & (k[None, :] < K),
            other=0.0,
        )
        partial = tl.dot(x, tl.trans(weight))
        tl.atomic_add(
            projection_raw_ptr + offsets_m[:, None] * PROJECTION_ROWS + n[None, :],
            partial,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    if HAS_INJECT:
        # Production HC=4 and ROWS=16 fit in one power-of-two vector. Only CTA
        # zero stores the tiny output, after every projection partial is visible.
        if pid == 0:
            offsets_i = tl.arange(0, 64)
            inject_row = offsets_i // HC_COUNT
            branch = offsets_i - inject_row * HC_COUNT
            inject_mask = (inject_row < num_rows) & (branch < HC_COUNT)
            logits = tl.load(
                projection_raw_ptr + inject_row * PROJECTION_ROWS + LOWRANK + branch,
                mask=inject_mask,
                other=0.0,
            )
            tl.store(
                inject_ptr + inject_row * HC_COUNT + branch,
                logits * projection_scale,
                mask=inject_mask,
            )

    offsets_j = tl.arange(0, BLOCK_J)
    offsets_r = tl.arange(0, BLOCK_R)
    offsets_g = tl.arange(0, HC_COUNT)
    j_blocks = tl.cdiv(HIDDEN_SIZE, BLOCK_J)
    for j_block in range(pid, j_blocks, num_ctas):
        j = j_block * BLOCK_J + offsets_j
        mask_j = j < HIDDEN_SIZE
        gj = offsets_g[:, None] * HIDDEN_SIZE + j[None, :]
        gj_flat = tl.reshape(gj, (HC_COUNT * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC_COUNT, BLOCK_J)),
            (HC_COUNT * BLOCK_J,),
        )
        gate_acc = tl.zeros((ROWS, HC_COUNT * BLOCK_J), dtype=tl.float32)
        for rank_start in range(0, LOWRANK, BLOCK_R):
            rank = rank_start + offsets_r
            mask_r = rank < LOWRANK
            down = tl.load(
                projection_raw_ptr
                + offsets_m[:, None] * PROJECTION_ROWS
                + rank[None, :],
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0,
            )
            down *= projection_scale
            activated = (down * tl.sigmoid(down)).to(x_ptr.dtype.element_ty)
            up = tl.load(
                up_weight_ptr + gj_flat[:, None] * LOWRANK + rank[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            gate_acc = tl.dot(activated, tl.trans(up), gate_acc)
        gate = tl.sigmoid(tl.reshape(gate_acc, (ROWS, HC_COUNT, BLOCK_J)))
        branches = tl.load(
            x_ptr
            + offsets_m[:, None, None] * (HC_COUNT * HIDDEN_SIZE)
            + offsets_g[None, :, None] * HIDDEN_SIZE
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        mixed = tl.sum(gate * branches, axis=1) / HC_COUNT
        tl.store(
            mixed_ptr + offsets_m[:, None] * HIDDEN_SIZE + j[None, :],
            mixed,
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _launch_projection_epilogue(
    projected: torch.Tensor,
    lowrank: int,
    hc_count: int,
    projection_scale: float,
    *,
    activation_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply projection scale and SiLU, optionally padding the activation."""
    rows = projected.shape[0]
    has_inject = projected.shape[1] != lowrank
    activation_width = lowrank if activation_width is None else activation_width
    activated = torch.empty(
        (rows, activation_width), dtype=projected.dtype, device=projected.device
    )
    inject = (
        torch.empty((rows, hc_count), dtype=projected.dtype, device=projected.device)
        if has_inject
        else None
    )
    block = triton.next_power_of_2(max(activation_width, hc_count))
    enable_pdl = pdl_enabled()
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _projection_epilogue_kernel[(rows,)](
        projected,
        activated,
        activated if inject is None else inject,
        projected.stride(0),
        activated.stride(0),
        0 if inject is None else inject.stride(0),
        projection_scale,
        LOWRANK=lowrank,
        ACTIVATION_WIDTH=activation_width,
        HC_COUNT=hc_count,
        HAS_INJECT=has_inject,
        ENABLE_PDL=enable_pdl,
        BLOCK=block,
        **launch_kwargs,
    )
    return activated, inject


def _launch_mix_epilogue(
    gate: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    rows = gate.shape[0]
    mixed = torch.empty(
        (rows, hidden_size), dtype=normalized.dtype, device=normalized.device
    )
    block = min(triton.next_power_of_2(hidden_size), 1024)
    enable_pdl = pdl_enabled()
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _mix_epilogue_kernel[(rows, triton.cdiv(hidden_size, block))](
        gate,
        normalized,
        mixed,
        gate.stride(0),
        normalized.stride(0),
        mixed.stride(0),
        hidden_size,
        HC_COUNT=hc_count,
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        **launch_kwargs,
    )
    return mixed


@register_kernel(
    "residual",
    "hyperconnection_mix",
    name="triton_hyperconnection_mix",
    solution="triton",
    signatures=_MIX_SIGNATURES,
    priority=Priority.PERFORMANT,
    tags={"determinism", "portability", "throughput"},
)
def triton_hyperconnection_mix(
    normalized: torch.Tensor,
    projection_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    projection_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GEMM plus Triton-epilogue path for general decode and prefill shapes."""
    projected = F.linear(normalized, projection_weight)
    activated, inject = _launch_projection_epilogue(
        projected, lowrank, hc_count, projection_scale
    )
    gate = F.linear(activated, up_weight)
    return _launch_mix_epilogue(gate, normalized, hc_count, hidden_size), inject


_PERSISTENT_ROWS = frozenset(range(1, 17))
_PERSISTENT_TRAITS = {
    "num_tokens": _PERSISTENT_ROWS,
    "hc_count": frozenset({4}),
    "hidden_size": frozenset({2560}),
    "lowrank": frozenset({320}),
    "has_inject": frozenset({False, True}),
    "contiguous": frozenset({True}),
    "folded_scale": frozenset({False, True}),
    "deterministic": frozenset({False}),
    "capturing": frozenset({False, True}),
}
_WORKSPACE_LOCK = threading.Lock()
_PERSISTENT_WORKSPACES: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


def _persistent_workspace(
    device: torch.device, projection_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    key = (device_index, stream_id, projection_rows)
    workspace = _PERSISTENT_WORKSPACES.get(key)
    if workspace is None:
        with _WORKSPACE_LOCK:
            workspace = _PERSISTENT_WORKSPACES.get(key)
            if workspace is None:
                raw = torch.empty(
                    (16, projection_rows), dtype=torch.float32, device=device
                )
                counters = torch.zeros(3, dtype=torch.int32, device=device)
                workspace = (raw, counters)
                _PERSISTENT_WORKSPACES[key] = workspace
    return workspace


@register_kernel(
    "residual",
    "hyperconnection_mix",
    name="triton_persistent_hyperconnection_mix",
    solution="triton_persistent",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(8, 0),
    ),
    signatures=_PERSISTENT_MIX_SIGNATURES,
    traits=_PERSISTENT_TRAITS,
    priority=Priority.SPECIALIZED + 2,
    tags={"decode", "latency"},
)
def triton_persistent_hyperconnection_mix(
    normalized: torch.Tensor,
    projection_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    projection_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Single persistent decode kernel with per-stream barrier workspaces."""
    rows, wide = normalized.shape
    if (
        not current_platform().is_nvidia
        or not 1 <= rows <= 16
        or hc_count != 4
        or hidden_size != 2560
        or lowrank != 320
        or wide != hc_count * hidden_size
        or normalized.dtype not in _LOW_PRECISION_DTYPES
        or not normalized.is_contiguous()
        or not projection_weight.is_contiguous()
        or not up_weight.is_contiguous()
    ):
        raise ValueError(
            "persistent HC mix requires NVIDIA and contiguous BF16/FP16 "
            "production shape T<=16, hc_count=4, hidden_size=2560, lowrank=320"
        )
    has_inject = projection_weight.shape[0] != lowrank
    projection_rows = int(projection_weight.shape[0])
    raw, counters = _persistent_workspace(normalized.device, projection_rows)
    mixed = torch.empty(
        (rows, hidden_size), dtype=normalized.dtype, device=normalized.device
    )
    inject = (
        torch.empty((rows, hc_count), dtype=normalized.dtype, device=normalized.device)
        if has_inject
        else None
    )
    num_ctas = torch.cuda.get_device_properties(normalized.device).multi_processor_count
    enable_pdl = pdl_enabled()
    launch_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _persistent_mix_kernel[(num_ctas,)](
        normalized,
        projection_weight,
        up_weight,
        raw,
        mixed,
        mixed if inject is None else inject,
        counters,
        wide,
        rows,
        num_ctas,
        projection_scale,
        ROWS=16,
        PROJECTION_ROWS=projection_rows,
        LOWRANK=lowrank,
        HC_COUNT=hc_count,
        HIDDEN_SIZE=hidden_size,
        HAS_INJECT=has_inject,
        ENABLE_PDL=enable_pdl,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
        **launch_kwargs,
    )
    return mixed, inject


@register_kernel(
    "residual",
    "hyperconnection_combine",
    name="triton_hyperconnection_combine",
    solution="triton",
    signatures=_COMBINE_SIGNATURES,
    priority=Priority.PERFORMANT,
    tags={"determinism", "portability"},
)
def triton_hyperconnection_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_logits: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """Triton gated residual-stream update without broadcast temporaries."""
    rows = block_output.shape[0]
    combined = torch.empty(
        (rows, hc_count * hidden_size),
        dtype=block_output.dtype,
        device=block_output.device,
    )
    block = min(triton.next_power_of_2(hidden_size), 1024)
    enable_pdl = pdl_enabled()
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _combine_kernel[(rows, triton.cdiv(hidden_size, block))](
        residual,
        block_output,
        inject_logits,
        combined,
        residual.stride(0),
        block_output.stride(0),
        inject_logits.stride(0),
        combined.stride(0),
        hidden_size,
        HC_COUNT=hc_count,
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        **launch_kwargs,
    )
    return combined


# ===-----------------------------------------------------------------------===#
# DeepSeek V4 mHC
# ===-----------------------------------------------------------------------===#


@cache
def _compute_num_split(
    device: torch.device, block_k: int, k: int | None, grid_size: int
) -> int:
    device_props = torch.cuda.get_device_properties(device)
    split_k = device_props.multi_processor_count // grid_size
    if k is not None:
        num_block_k = triton.cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _pre_reduce_apply_is_supported(
    pre_reduce_apply_impl,
    n_splits: int,
    *,
    hc_mult: int,
    hidden_size: int,
) -> bool:
    if pre_reduce_apply_impl is None:
        return False
    supported = getattr(pre_reduce_apply_impl, "supported_n_splits", None)
    if supported is not None and n_splits not in supported:
        return False
    supported_hc_mults = getattr(pre_reduce_apply_impl, "supported_hc_mults", None)
    if supported_hc_mults is not None and hc_mult not in supported_hc_mults:
        return False
    hidden_size_multiple = getattr(pre_reduce_apply_impl, "hidden_size_multiple", None)
    return hidden_size_multiple is None or hidden_size % hidden_size_multiple == 0


def _pre_reduce_apply_fuses_norm(
    pre_reduce_apply_impl, use_pre_reduce_apply: bool, has_norm_weight: bool
) -> bool:
    return bool(
        use_pre_reduce_apply
        and has_norm_weight
        and getattr(pre_reduce_apply_impl, "supports_fused_norm", False)
    )


@triton.jit
def _mhc_prenorm_gemm_triton_kernel(
    x,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    split_id = tl.program_id(0)
    token_block = tl.program_id(1)
    n_block = tl.program_id(2)
    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    split_start = split_id * SPLIT_K
    dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    square_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, SPLIT_K, BLOCK_K):
        offs_k = split_start + k_start + tl.arange(0, BLOCK_K)
        x_values = tl.load(
            x + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < num_tokens) & (offs_k[None, :] < K),
            other=0.0,
        )
        fn_values = tl.load(
            fn + offs_k[:, None] + offs_n[None, :] * K,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        dot_acc = tl.dot(
            x_values.to(tl.float32),
            fn_values,
            dot_acc,
            input_precision="ieee",
        )
        x_fp32 = x_values.to(tl.float32)
        square_acc += tl.sum(x_fp32 * x_fp32, axis=1)

    tl.store(
        out_mul + split_id * num_tokens * N + offs_m[:, None] * N + offs_n[None, :],
        dot_acc,
        mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < N),
    )
    tl.store(
        out_sqrsum + split_id * num_tokens + offs_m,
        square_acc,
        mask=(offs_m < num_tokens) & (n_block == 0),
    )


def _mhc_prenorm_gemm_launch_config(
    num_tokens: int, k: int, n: int, n_splits: int
) -> tuple[int, int, int, int, int]:
    if (num_tokens, k, n, n_splits) == (64, 16384, 24, 64):
        return 16, 16, 64, 2, 3
    return 16, 32, 64, 4, 1


def _mhc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out_mul: torch.Tensor,
    out_sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    num_tokens, k = x.shape
    n = fn.shape[0]
    block_m, block_n, block_k, num_warps, num_stages = _mhc_prenorm_gemm_launch_config(
        num_tokens, k, n, n_splits
    )
    split_k = triton.cdiv(triton.cdiv(k, n_splits), block_k) * block_k
    _mhc_prenorm_gemm_triton_kernel[
        (
            n_splits,
            triton.cdiv(num_tokens, block_m),
            triton.cdiv(n, block_n),
        )
    ](
        x,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        K=k,
        N=n,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _load_reduced_mix(
    gemm_out_mul,
    token_id,
    mix_id: tl.constexpr,
    num_tokens,
    hc_mult3: tl.constexpr,
    n_splits: tl.constexpr,
):
    value = tl.full((), 0.0, tl.float32)
    for split_id in tl.static_range(0, n_splits):
        offset = split_id * num_tokens * hc_mult3 + token_id * hc_mult3 + mix_id
        value += tl.load(gemm_out_mul + offset)
    return value


@triton.jit
def _mhc_pre_mix_triton_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    hc_mult: tl.constexpr,
    hc_mult2: tl.constexpr,
    hc_mult3: tl.constexpr,
    block_comb: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)

    rms_sum = tl.full((), 0.0, tl.float32)
    for split_id in tl.static_range(0, n_splits):
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)
    rms = tl.rsqrt(rms_sum / (hc_mult * hidden_size) + rms_eps)

    pre_scale = tl.load(hc_scale)
    for hc_id in tl.static_range(0, hc_mult):
        mix = _load_reduced_mix(
            gemm_out_mul,
            token_id,
            hc_id,
            num_tokens,
            hc_mult3,
            n_splits,
        )
        pre = tl.sigmoid(mix * rms * pre_scale + tl.load(hc_base + hc_id)) + hc_eps
        tl.store(pre_mix + token_id * hc_mult + hc_id, pre)

    post_scale = tl.load(hc_scale + 1)
    for hc_id in tl.static_range(0, hc_mult):
        mix = _load_reduced_mix(
            gemm_out_mul,
            token_id,
            hc_mult + hc_id,
            num_tokens,
            hc_mult3,
            n_splits,
        )
        post = (
            tl.sigmoid(mix * rms * post_scale + tl.load(hc_base + hc_mult + hc_id))
            * 2.0
        )
        tl.store(post_mix + token_id * hc_mult + hc_id, post)

    comb_offsets = tl.arange(0, block_comb)
    comb_mask = comb_offsets < hc_mult2
    comb_scale = tl.load(hc_scale + 2)
    comb_mix_values = tl.zeros((block_comb,), tl.float32)
    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * hc_mult3 + token_id * hc_mult3
        comb_mix_values += tl.load(
            gemm_out_mul + split_base + hc_mult * 2 + comb_offsets,
            mask=comb_mask,
            other=0.0,
        )
    comb_values = comb_mix_values * rms * comb_scale + tl.load(
        hc_base + hc_mult * 2 + comb_offsets, mask=comb_mask, other=0.0
    )
    rows = comb_offsets // hc_mult
    cols = comb_offsets - rows * hc_mult
    active = comb_mask

    for row_id in tl.static_range(0, hc_mult):
        row_values = tl.where((rows == row_id) & active, comb_values, -float("inf"))
        row_max = tl.max(row_values, axis=0)
        comb_values = tl.where(
            (rows == row_id) & active, tl.exp(comb_values - row_max), comb_values
        )
    for row_id in tl.static_range(0, hc_mult):
        row_sum = tl.sum(tl.where((rows == row_id) & active, comb_values, 0.0), axis=0)
        comb_values = tl.where(
            (rows == row_id) & active, comb_values / row_sum + hc_eps, comb_values
        )
    for col_id in tl.static_range(0, hc_mult):
        col_sum = tl.sum(tl.where((cols == col_id) & active, comb_values, 0.0), axis=0)
        comb_values = tl.where(
            (cols == col_id) & active,
            comb_values / (col_sum + hc_eps),
            comb_values,
        )

    for _ in tl.static_range(1, sinkhorn_iters):
        for row_id in tl.static_range(0, hc_mult):
            row_sum = tl.sum(
                tl.where((rows == row_id) & active, comb_values, 0.0), axis=0
            )
            comb_values = tl.where(
                (rows == row_id) & active,
                comb_values / (row_sum + hc_eps),
                comb_values,
            )
        for col_id in tl.static_range(0, hc_mult):
            col_sum = tl.sum(
                tl.where((cols == col_id) & active, comb_values, 0.0), axis=0
            )
            comb_values = tl.where(
                (cols == col_id) & active,
                comb_values / (col_sum + hc_eps),
                comb_values,
            )

    tl.store(
        comb_mix + token_id * hc_mult2 + comb_offsets,
        comb_values,
        mask=comb_mask,
    )


@triton.jit
def _mhc_pre_layer_triton_kernel(
    pre_mix,
    residual,
    layer_input,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)

    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    layer_acc = tl.zeros((block_h,), tl.float32)
    for hc_id in tl.static_range(0, hc_mult):
        pre = tl.load(pre_mix + token_id * hc_mult + hc_id).to(tl.float32)
        residual_offsets = (
            token_id * hc_mult * hidden_size + hc_id * hidden_size + hidden_offsets
        )
        residual_values = tl.load(
            residual + residual_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        layer_acc += pre * residual_values
    tl.store(
        layer_input + token_id * hidden_size + hidden_offsets,
        layer_acc,
        mask=hidden_mask,
    )


@triton.jit
def _mhc_post_triton_kernel(
    comb,
    residual,
    post,
    hidden_states,
    out,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    hidden_values = tl.load(
        hidden_states + token_id * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    for out_hc in tl.static_range(0, hc_mult):
        acc = tl.load(post + token_id * hc_mult + out_hc).to(tl.float32) * hidden_values
        for in_hc in tl.static_range(0, hc_mult):
            comb_value = tl.load(
                comb + token_id * hc_mult * hc_mult + in_hc * hc_mult + out_hc
            ).to(tl.float32)
            residual_values = tl.load(
                residual
                + token_id * hc_mult * hidden_size
                + in_hc * hidden_size
                + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            acc += comb_value * residual_values
        tl.store(
            out
            + token_id * hc_mult * hidden_size
            + out_hc * hidden_size
            + hidden_offsets,
            acc,
            mask=hidden_mask,
        )


@triton.jit
def _mhc_post_hc4_triton_kernel(
    comb,
    residual,
    post,
    hidden_states,
    out,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    token_hidden_offset = token_id * hidden_size
    token_residual_offset = token_id * 4 * hidden_size

    hidden_values = tl.load(
        hidden_states + token_hidden_offset + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    post_base = token_id * 4
    acc0 = tl.load(post + post_base).to(tl.float32) * hidden_values
    acc1 = tl.load(post + post_base + 1).to(tl.float32) * hidden_values
    acc2 = tl.load(post + post_base + 2).to(tl.float32) * hidden_values
    acc3 = tl.load(post + post_base + 3).to(tl.float32) * hidden_values

    comb_base = token_id * 16
    for in_hc in tl.static_range(0, 4):
        residual_values = tl.load(
            residual + token_residual_offset + in_hc * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        comb_row = comb_base + in_hc * 4
        acc0 += tl.load(comb + comb_row).to(tl.float32) * residual_values
        acc1 += tl.load(comb + comb_row + 1).to(tl.float32) * residual_values
        acc2 += tl.load(comb + comb_row + 2).to(tl.float32) * residual_values
        acc3 += tl.load(comb + comb_row + 3).to(tl.float32) * residual_values

    tl.store(out + token_residual_offset + hidden_offsets, acc0, mask=hidden_mask)
    tl.store(
        out + token_residual_offset + hidden_size + hidden_offsets,
        acc1,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size * 2 + hidden_offsets,
        acc2,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size * 3 + hidden_offsets,
        acc3,
        mask=hidden_mask,
    )


def _mhc_pre_impl(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    prenorm_gemm,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
    pre_mix_impl=None,
    pre_reduce_apply_impl=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("norm_weight and norm_eps must be provided together")
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise RuntimeError("fast mHC requires bf16 residual and fp32 weights")
    if not residual.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape,
                hc_mult,
                1,
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual.device,
            ),
        )

    n_splits = _compute_num_split(
        residual.device,
        64,
        hc_hidden_size,
        triton.cdiv(num_tokens, 64),
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    use_pre_reduce_apply = _pre_reduce_apply_is_supported(
        pre_reduce_apply_impl,
        n_splits,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )
    pre_mix = (
        None
        if use_pre_reduce_apply
        else torch.empty(
            num_tokens, hc_mult, dtype=torch.float32, device=residual.device
        )
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_hidden_size)
    prenorm_gemm(
        residual_2d,
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    block_h = 1024
    fused_norm = _pre_reduce_apply_fuses_norm(
        pre_reduce_apply_impl,
        use_pre_reduce_apply,
        norm_weight is not None,
    )
    if use_pre_reduce_apply:
        fused_norm_kwargs = {}
        if getattr(pre_reduce_apply_impl, "supports_fused_norm", False):
            fused_norm_kwargs = {
                "norm_weight": norm_weight if fused_norm else None,
                "norm_eps": norm_eps if fused_norm else 0.0,
                "block_size": 512,
                "enable_pdl": pdl_enabled(),
            }
        pre_reduce_apply_impl(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            layer_input,
            post_mix,
            comb_mix,
            hidden_size,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            n_splits,
            num_tokens,
            **fused_norm_kwargs,
        )
    else:
        if pre_mix_impl is None:
            _mhc_pre_mix_triton_kernel[(num_tokens,)](
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                pre_mix,
                post_mix,
                comb_mix,
                hidden_size=hidden_size,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                sinkhorn_iters=sinkhorn_iters,
                n_splits=n_splits,
                hc_mult=hc_mult,
                hc_mult2=hc_mult2,
                hc_mult3=hc_mult3,
                block_comb=triton.next_power_of_2(hc_mult2),
                num_tokens=num_tokens,
                num_warps=1,
            )
        else:
            pre_mix_impl(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                pre_mix,
                post_mix,
                comb_mix,
                hidden_size,
                rms_eps,
                hc_eps,
                sinkhorn_iters,
                n_splits,
                num_tokens,
            )
        _mhc_pre_layer_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
            pre_mix,
            residual_flat,
            layer_input,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            block_h=block_h,
            num_warps=4,
        )

    if norm_weight is not None and not fused_norm:
        if norm_eps is None:
            raise ValueError("norm_eps is required when norm_weight is provided")
        layer_input = torch.nn.functional.rms_norm(
            layer_input, (hidden_size,), norm_weight, norm_eps
        )

    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
    )


def _tiled_mhc_pre_hc4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the shared branch's tiled four-stream prefill projection."""
    outer_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, 4, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape, 4, 1, dtype=torch.float32, device=residual.device
            ),
            torch.empty(
                *outer_shape, 4, 4, dtype=torch.float32, device=residual.device
            ),
        )

    n_splits, block_m, block_k = _mhc_prefill_config_hc4(num_tokens)
    n_splits = min(n_splits, triton.cdiv(4 * hidden_size, block_k))
    projection = torch.empty(
        n_splits, num_tokens, 24, dtype=torch.float32, device=residual.device
    )
    square_sum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(num_tokens, 4, dtype=torch.float32, device=residual.device)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty(num_tokens, 16, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    mhc_prefill_project_hc4(
        residual_flat,
        fn,
        projection,
        square_sum,
        n_splits=n_splits,
        block_m=block_m,
        block_k=block_k,
    )
    mhc_pre_mix_hc4(
        projection,
        square_sum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
    )
    block_h = 1024
    _mhc_pre_layer_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        residual_flat,
        layer_input,
        hidden_size=hidden_size,
        hc_mult=4,
        block_h=block_h,
        num_warps=4,
    )
    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, 4, 1),
        comb_mix.view(*outer_shape, 4, 4),
    )


@register_kernel(
    "residual",
    "mhc_pre",
    name="triton_mhc_pre",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                residual=dense_tensor_format(torch.bfloat16),
                fn=dense_tensor_format(torch.float32),
                hc_scale=dense_tensor_format(torch.float32),
                hc_base=dense_tensor_format(torch.float32),
            )
        }
    ),
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the portable Triton mHC pre-mapping."""
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    if residual.shape[-2] == 4 and num_tokens > 256:
        layer_input, post_mix, comb_mix = _tiled_mhc_pre_hc4(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
        )
        if norm_weight is not None:
            if norm_eps is None:
                raise ValueError("norm_eps is required when norm_weight is provided")
            layer_input = torch.nn.functional.rms_norm(
                layer_input, (residual.shape[-1],), norm_weight, norm_eps
            )
        return layer_input, post_mix, comb_mix
    return _mhc_pre_impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        _mhc_prenorm_gemm_triton,
        norm_weight,
        norm_eps,
    )


@register_kernel(
    "residual",
    "mhc_post",
    name="triton_mhc_post",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                residual=dense_tensor_format(torch.bfloat16),
                post=dense_tensor_format(torch.float32),
                comb=dense_tensor_format(torch.float32),
            )
        }
    ),
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Run the portable Triton mHC post-mapping."""
    if not hidden_states.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")
    if residual.numel() == 0:
        return torch.empty_like(residual)

    out = torch.empty_like(residual)
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    hidden_states_flat = hidden_states.view(-1, hidden_size)
    post_flat = post.view(-1, hc_mult)
    comb_flat = comb.view(-1, hc_mult, hc_mult)
    num_tokens = residual_flat.shape[0]
    if hc_mult == 4:
        block_h = 256
        _mhc_post_hc4_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
            comb_flat,
            residual_flat,
            post_flat,
            hidden_states_flat,
            out,
            hidden_size=hidden_size,
            block_h=block_h,
            num_warps=4,
        )
        return out

    block_h = 1024
    _mhc_post_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        comb_flat,
        residual_flat,
        post_flat,
        hidden_states_flat,
        out,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        block_h=block_h,
        num_warps=4,
    )
    return out


# ===-----------------------------------------------------------------------===#
# DeepSeek V4 mHC prefill helpers
# ===-----------------------------------------------------------------------===#


@triton.jit
def _mhc_prefill_project_hc4_kernel(
    residual,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    hidden_size: tl.constexpr,
    split_k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    split_id = tl.program_id(0).to(tl.int64)
    token_block_id = tl.program_id(1).to(tl.int64)
    token_offsets = token_block_id * block_m + tl.arange(0, block_m).to(tl.int64)
    mix_offsets = tl.arange(0, 32).to(tl.int64)
    split_start = split_id * split_k
    hc_hidden_size: tl.constexpr = 4 * hidden_size

    projection = tl.zeros((block_m, 32), dtype=tl.float32)
    square_sum = tl.zeros((block_m,), dtype=tl.float32)
    for k_start in range(0, split_k, block_k):
        k_offsets = split_start + k_start + tl.arange(0, block_k).to(tl.int64)
        residual_values = tl.load(
            residual + token_offsets[:, None] * hc_hidden_size + k_offsets[None, :],
            mask=(token_offsets[:, None] < num_tokens)
            & (k_offsets[None, :] < hc_hidden_size),
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            fn + k_offsets[:, None] + mix_offsets[None, :] * hc_hidden_size,
            mask=(k_offsets[:, None] < hc_hidden_size) & (mix_offsets[None, :] < 24),
            other=0.0,
        )
        projection = tl.dot(
            residual_values,
            weight_values,
            projection,
            input_precision="ieee",
        )
        square_sum += tl.sum(residual_values * residual_values, axis=1)

    output_base = split_id * num_tokens.to(tl.int64)
    tl.store(
        out_mul + output_base * 24 + token_offsets[:, None] * 24 + mix_offsets[None, :],
        projection,
        mask=(token_offsets[:, None] < num_tokens) & (mix_offsets[None, :] < 24),
    )
    tl.store(
        out_sqrsum + output_base + token_offsets,
        square_sum,
        mask=token_offsets < num_tokens,
    )


def mhc_prefill_project_hc4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    out_mul: torch.Tensor,
    out_sqrsum: torch.Tensor,
    *,
    n_splits: int,
    block_m: int,
    block_k: int,
) -> None:
    """Project an hc=4 residual into split prefill mapping partials.

    Args:
        residual: Contiguous BF16 residual streams shaped ``[T, 4, H]``.
        fn: Contiguous FP32 projection weights shaped ``[24, 4 * H]``.
        out_mul: Contiguous FP32 projection output shaped
            ``[n_splits, T, 24]``.
        out_sqrsum: Contiguous FP32 squared-sum output shaped
            ``[n_splits, T]``.
        n_splits: Number of contiguous reduction-axis partitions.
        block_m: Number of token rows in each program tile.
        block_k: Reduction-axis tile width.

    Returns:
        None. ``out_mul`` and ``out_sqrsum`` are written in place.
    """
    if residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("mhc_prefill_project_hc4 requires residual shaped [T, 4, H]")
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise ValueError(
            "mhc_prefill_project_hc4 requires BF16 residual and FP32 weights"
        )
    if out_mul.dtype != torch.float32 or out_sqrsum.dtype != torch.float32:
        raise ValueError("mhc_prefill_project_hc4 requires FP32 output buffers")
    tensors = (residual, fn, out_mul, out_sqrsum)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("mhc_prefill_project_hc4 requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in tensors):
        raise ValueError("mhc_prefill_project_hc4 tensors must share one device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("mhc_prefill_project_hc4 tensors must be contiguous")
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    for name, value in (("block_m", block_m), ("block_k", block_k)):
        if not isinstance(value, int) or value < 16 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two of at least 16")

    num_tokens, _, hidden_size = residual.shape
    if hidden_size < 1:
        raise ValueError("hidden size must be positive")
    hc_hidden_size = 4 * hidden_size
    if fn.shape != (24, hc_hidden_size):
        raise ValueError(
            f"fn shape mismatch: expected {(24, hc_hidden_size)}, got {tuple(fn.shape)}"
        )
    if out_mul.shape != (n_splits, num_tokens, 24):
        raise ValueError(
            "out_mul shape mismatch: expected "
            f"{(n_splits, num_tokens, 24)}, got {tuple(out_mul.shape)}"
        )
    if out_sqrsum.shape != (n_splits, num_tokens):
        raise ValueError(
            "out_sqrsum shape mismatch: expected "
            f"{(n_splits, num_tokens)}, got {tuple(out_sqrsum.shape)}"
        )
    if num_tokens == 0:
        return

    split_k = triton.cdiv(triton.cdiv(hc_hidden_size, n_splits), block_k) * block_k
    _mhc_prefill_project_hc4_kernel[(n_splits, triton.cdiv(num_tokens, block_m))](
        residual,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        hidden_size=hidden_size,
        split_k=split_k,
        block_m=block_m,
        block_k=block_k,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _mhc_pre_mix_hc4_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)
    pre_post_offsets = tl.arange(0, 8)
    comb_offsets = tl.arange(0, 16)
    pre_post_values = tl.zeros((8,), tl.float32)
    comb_values = tl.zeros((16,), tl.float32)
    rms_sum = tl.full((), 0.0, tl.float32)

    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * 24 + token_id * 24
        pre_post_values += tl.load(gemm_out_mul + split_base + pre_post_offsets)
        comb_values += tl.load(gemm_out_mul + split_base + 8 + comb_offsets)
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)

    rms = tl.rsqrt(rms_sum / (4 * hidden_size) + rms_eps)
    pre_post_scale = tl.where(
        pre_post_offsets < 4,
        tl.load(hc_scale),
        tl.load(hc_scale + 1),
    )
    pre_post_values = tl.sigmoid(
        pre_post_values * rms * pre_post_scale + tl.load(hc_base + pre_post_offsets)
    )
    tl.store(
        pre_mix + token_id * 4 + pre_post_offsets,
        pre_post_values + hc_eps,
        mask=pre_post_offsets < 4,
    )
    tl.store(
        post_mix + token_id * 4 + pre_post_offsets - 4,
        pre_post_values * 2.0,
        mask=pre_post_offsets >= 4,
    )

    comb_values = comb_values * rms * tl.load(hc_scale + 2) + tl.load(
        hc_base + 8 + comb_offsets
    )
    comb_matrix = tl.reshape(comb_values, (4, 4))
    row_max = tl.max(comb_matrix, axis=1)
    comb_matrix = tl.exp(comb_matrix - tl.expand_dims(row_max, 1))
    row_sum = tl.sum(comb_matrix, axis=1)
    comb_matrix = comb_matrix / tl.expand_dims(row_sum, 1) + hc_eps
    col_sum = tl.sum(comb_matrix, axis=0)
    comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    for _ in tl.static_range(1, sinkhorn_iters):
        row_sum = tl.sum(comb_matrix, axis=1)
        comb_matrix = comb_matrix / (tl.expand_dims(row_sum, 1) + hc_eps)
        col_sum = tl.sum(comb_matrix, axis=0)
        comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    tl.store(
        comb_mix + token_id * 16 + comb_offsets,
        tl.reshape(comb_matrix, (16,)),
    )


def mhc_pre_mix_hc4(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Reduce split-K mHC projections and form hc=4 mixing coefficients.

    Args:
        gemm_out_mul: FP32 split-K projections shaped ``[n_splits, T, 24]``.
        gemm_out_sqrsum: FP32 split-K squared sums shaped ``[n_splits, T]``.
        hc_scale: FP32 scales for pre, post, and combination mappings.
        hc_base: FP32 biases shaped ``[24]``.
        pre_mix: FP32 output buffer shaped ``[T, 4]``.
        post_mix: FP32 output buffer shaped ``[T, 4]``.
        comb_mix: FP32 output buffer shaped ``[T, 16]``.
        hidden_size: Hidden width of one residual stream.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC mixing transforms.
        sinkhorn_iters: Number of row/column Sinkhorn normalization rounds.
        n_splits: Number of split-K projection partials.
        num_tokens: Number of token rows in each tensor.

    Returns:
        None. The three output buffers are written in place.
    """
    _mhc_pre_mix_hc4_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
        num_warps=1,
    )


@triton.jit
def _mhc_pre_only_hc4_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    n_splits: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)
    pre_offsets = tl.arange(0, 4)
    pre_values = tl.zeros((4,), tl.float32)
    rms_sum = tl.full((), 0.0, tl.float32)

    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * 24 + token_id * 24
        pre_values += tl.load(gemm_out_mul + split_base + pre_offsets)
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)

    rms = tl.rsqrt(rms_sum / (4 * hidden_size) + rms_eps)
    pre_values = (
        tl.sigmoid(
            pre_values * rms * tl.load(hc_scale) + tl.load(hc_base + pre_offsets)
        )
        + hc_eps
    )
    tl.store(pre_mix + token_id * 4 + pre_offsets, pre_values)


def mhc_pre_only_hc4(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Form only the pre-mapping coefficients for an hc=4 mHC layer.

    Args:
        gemm_out_mul: FP32 split-K projections shaped ``[n_splits, T, 24]``.
        gemm_out_sqrsum: FP32 split-K squared sums shaped ``[n_splits, T]``.
        hc_scale: FP32 scales for pre, post, and combination mappings.
        hc_base: FP32 biases shaped ``[24]``.
        pre_mix: FP32 output buffer shaped ``[T, 4]``.
        hidden_size: Hidden width of one residual stream.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC pre-mapping transform.
        n_splits: Number of split-K projection partials.
        num_tokens: Number of token rows in each tensor.

    Returns:
        None. ``pre_mix`` is written in place.
    """
    _mhc_pre_only_hc4_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        n_splits=n_splits,
        num_tokens=num_tokens,
        num_warps=1,
    )


@triton.jit
def _mhc_post_comb_hc4_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)
    post_offsets = tl.arange(0, 4)
    comb_offsets = tl.arange(0, 16)
    post_values = tl.zeros((4,), tl.float32)
    comb_values = tl.zeros((16,), tl.float32)
    rms_sum = tl.full((), 0.0, tl.float32)

    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * 24 + token_id * 24
        post_values += tl.load(gemm_out_mul + split_base + 4 + post_offsets)
        comb_values += tl.load(gemm_out_mul + split_base + 8 + comb_offsets)
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)

    rms = tl.rsqrt(rms_sum / (4 * hidden_size) + rms_eps)
    post_values = tl.sigmoid(
        post_values * rms * tl.load(hc_scale + 1) + tl.load(hc_base + 4 + post_offsets)
    )
    tl.store(post_mix + token_id * 4 + post_offsets, post_values * 2.0)

    comb_values = comb_values * rms * tl.load(hc_scale + 2) + tl.load(
        hc_base + 8 + comb_offsets
    )
    comb_matrix = tl.reshape(comb_values, (4, 4))
    row_max = tl.max(comb_matrix, axis=1)
    comb_matrix = tl.exp(comb_matrix - tl.expand_dims(row_max, 1))
    row_sum = tl.sum(comb_matrix, axis=1)
    comb_matrix = comb_matrix / tl.expand_dims(row_sum, 1) + hc_eps
    col_sum = tl.sum(comb_matrix, axis=0)
    comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    for _ in tl.static_range(1, sinkhorn_iters):
        row_sum = tl.sum(comb_matrix, axis=1)
        comb_matrix = comb_matrix / (tl.expand_dims(row_sum, 1) + hc_eps)
        col_sum = tl.sum(comb_matrix, axis=0)
        comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    tl.store(
        comb_mix + token_id * 16 + comb_offsets,
        tl.reshape(comb_matrix, (16,)),
    )


def mhc_post_comb_hc4(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Form post and combination coefficients for an hc=4 mHC layer.

    This is independent from the pre-mapping after the split-K projection, so
    callers may run it on a side stream while the layer consumes the pre path.

    Args:
        gemm_out_mul: FP32 split-K projections shaped ``[n_splits, T, 24]``.
        gemm_out_sqrsum: FP32 split-K squared sums shaped ``[n_splits, T]``.
        hc_scale: FP32 scales for pre, post, and combination mappings.
        hc_base: FP32 biases shaped ``[24]``.
        post_mix: FP32 output buffer shaped ``[T, 4]``.
        comb_mix: FP32 output buffer shaped ``[T, 16]``.
        hidden_size: Hidden width of one residual stream.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC mixing transforms.
        sinkhorn_iters: Number of row/column Sinkhorn normalization rounds.
        n_splits: Number of split-K projection partials.
        num_tokens: Number of token rows in each tensor.

    Returns:
        None. ``post_mix`` and ``comb_mix`` are written in place.
    """
    _mhc_post_comb_hc4_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
        num_warps=1,
    )


@triton.jit
def _mhc_pre_layer_norm_hc4_kernel(
    pre_mix,
    residual,
    weight,
    out,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_offsets = tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    residual_base = token_id * 4 * hidden_size

    layer_input = tl.zeros((block_h,), tl.float32)
    for hc_id in tl.static_range(0, 4):
        pre = tl.load(pre_mix + token_id * 4 + hc_id).to(tl.float32)
        residual_values = tl.load(
            residual + residual_base + hc_id * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        layer_input += pre * residual_values

    # Preserve the rounding point of the unfused path, which materializes the
    # weighted residual sum as BF16 before RMSNorm reads it back.
    layer_input = layer_input.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(layer_input * layer_input, axis=0) / hidden_size
    norm_scale = tl.rsqrt(variance + eps)
    norm_weight = tl.load(weight + hidden_offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        out + token_id * hidden_size + hidden_offsets,
        layer_input * norm_scale * norm_weight,
        mask=hidden_mask,
    )


def mhc_pre_layer_norm_hc4(
    pre_mix: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
) -> None:
    """Form an hc=4 mHC layer input and apply RMSNorm in one kernel.

    Args:
        pre_mix: FP32 mixing coefficients shaped ``[..., 4]``.
        residual: BF16 residual streams shaped ``[..., 4, hidden_size]``.
        weight: BF16 or FP32 RMSNorm weight shaped ``[hidden_size]``.
        out: BF16 output buffer shaped ``[..., hidden_size]``.
        eps: RMSNorm epsilon.

    Returns:
        None. ``out`` is written in place.
    """
    if residual.shape[-2] != 4 or pre_mix.shape[-1] != 4:
        raise ValueError("mhc_pre_layer_norm_hc4 requires exactly four streams")
    hidden_size = residual.shape[-1]
    if weight.shape != (hidden_size,):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size "
            f"{hidden_size}"
        )
    if out.shape != (*residual.shape[:-2], hidden_size):
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match residual prefix "
            f"{tuple(residual.shape[:-2])} and hidden size {hidden_size}"
        )
    if not (pre_mix.is_contiguous() and residual.is_contiguous()):
        raise ValueError("pre_mix and residual must be contiguous")
    if not (weight.is_contiguous() and out.is_contiguous()):
        raise ValueError("weight and out must be contiguous")

    num_tokens = residual.numel() // (4 * hidden_size)
    if num_tokens == 0:
        return
    block_h = triton.next_power_of_2(hidden_size)
    _mhc_pre_layer_norm_hc4_kernel[(num_tokens,)](
        pre_mix,
        residual,
        weight,
        out,
        hidden_size=hidden_size,
        eps=eps,
        block_h=block_h,
        num_warps=8,
    )


def _mhc_prefill_config_hc4(num_tokens: int) -> tuple[int, int, int]:
    if num_tokens <= 1024:
        return 8, 32, 256
    if num_tokens <= 2048:
        return 8, 64, 128
    if num_tokens <= 4096:
        return 4, 64, 256
    if num_tokens <= 8192:
        return 2, 64, 256
    return 1, 64, 256


def fused_mhc_prefill_hc4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the tiled hc=4 mHC prefill path with fused output RMSNorm.

    Args:
        residual: Contiguous BF16 residual streams shaped ``[T, 4, H]``.
        fn: Contiguous FP32 projection weights shaped ``[24, 4 * H]``.
        hc_scale: Contiguous FP32 pre, post, and combination scales shaped
            ``[3]``.
        hc_base: Contiguous FP32 projection bias shaped ``[24]``.
        norm_weight: Contiguous BF16 or FP32 RMSNorm weight shaped ``[H]``.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC mapping transforms.
        sinkhorn_iters: Number of row/column Sinkhorn normalization rounds.
        norm_eps: Epsilon used by the output RMSNorm.

    Returns:
        A tuple of BF16 normalized layer input ``[T, H]``, FP32 post
        coefficients ``[T, 4, 1]``, and FP32 combination coefficients
        ``[T, 4, 4]``.
    """
    if residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("fused_mhc_prefill_hc4 requires residual shaped [T, 4, H]")
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise ValueError(
            "fused_mhc_prefill_hc4 requires BF16 residual and FP32 weights"
        )
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise ValueError("fused_mhc_prefill_hc4 requires FP32 scales and biases")
    if norm_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("fused_mhc_prefill_hc4 requires BF16 or FP32 norm weight")
    tensors = (residual, fn, hc_scale, hc_base, norm_weight)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused_mhc_prefill_hc4 requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in tensors):
        raise ValueError("fused_mhc_prefill_hc4 tensors must share one device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused_mhc_prefill_hc4 tensors must be contiguous")
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be positive")

    num_tokens, _, hidden_size = residual.shape
    if hidden_size < 1:
        raise ValueError("hidden size must be positive")
    if fn.shape != (24, 4 * hidden_size):
        raise ValueError(
            f"fn shape mismatch: expected {(24, 4 * hidden_size)}, "
            f"got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (24,):
        raise ValueError("hc_scale and hc_base must have shapes [3] and [24]")
    if norm_weight.shape != (hidden_size,):
        raise ValueError(
            f"norm_weight shape mismatch: expected {(hidden_size,)}, "
            f"got {tuple(norm_weight.shape)}"
        )
    if num_tokens == 0:
        return (
            residual.new_empty(0, hidden_size),
            torch.empty(0, 4, 1, device=residual.device, dtype=torch.float32),
            torch.empty(0, 4, 4, device=residual.device, dtype=torch.float32),
        )

    n_splits, block_m, block_k = _mhc_prefill_config_hc4(num_tokens)
    n_splits = min(n_splits, triton.cdiv(4 * hidden_size, block_k))
    projection = torch.empty(
        n_splits, num_tokens, 24, device=residual.device, dtype=torch.float32
    )
    square_sum = torch.empty(
        n_splits, num_tokens, device=residual.device, dtype=torch.float32
    )
    pre_mix = torch.empty(num_tokens, 4, device=residual.device, dtype=torch.float32)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty(num_tokens, 16, device=residual.device, dtype=torch.float32)
    layer_input = torch.empty(
        num_tokens, hidden_size, device=residual.device, dtype=torch.bfloat16
    )

    mhc_prefill_project_hc4(
        residual,
        fn,
        projection,
        square_sum,
        n_splits=n_splits,
        block_m=block_m,
        block_k=block_k,
    )
    mhc_pre_mix_hc4(
        projection,
        square_sum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
    )
    mhc_pre_layer_norm_hc4(
        pre_mix,
        residual,
        norm_weight,
        layer_input,
        eps=norm_eps,
    )
    return layer_input, post_mix.unsqueeze(-1), comb_mix.view(num_tokens, 4, 4)
