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

"""Portable and persistent Triton gated-residual kernels."""

from __future__ import annotations

import threading

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
from tokenspeed_kernel.signature import format_signatures

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
    "hyperconnection",
    "mix",
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
    "hyperconnection",
    "mix",
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
    "hyperconnection",
    "combine",
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


__all__ = [
    "triton_hyperconnection_combine",
    "triton_hyperconnection_mix",
    "triton_persistent_hyperconnection_mix",
]
