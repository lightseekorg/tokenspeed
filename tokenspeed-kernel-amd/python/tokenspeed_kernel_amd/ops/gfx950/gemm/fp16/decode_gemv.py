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

"""Cold-weight BF16 GEMV for one or a few decode rows on gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_SUBGROUP_SIZE = 64


# (N, K) -> (BLOCK_K, wave count) where another geometry beat the K rule below
# by at least 1.5% in each of three cold-weight graph-replay sweeps on MI355X
# (median of six runs alternating with the K rule).
_CONFIGS: dict[tuple[int, int], tuple[int, int]] = {
    (160, 4096): (2048, 4),  # GLM 5.3 Flash attention TP4
    (288, 4096): (2048, 4),  # GLM 5.3 Flash MoE router gate
    (512, 8192): (1024, 8),  # Muse Spark pre-expert projection TP8
    (896, 21504): (1024, 2),
    (2112, 14336): (1024, 2),
    (2560, 3072): (512, 1),  # Qwen3.8-Flash-Next attention o_proj TP2
    (4608, 7168): (512, 1),
    (7168, 1536): (128, 1),  # Kimi K3 o_proj TP8
    (7168, 2304): (512, 1),
}


def _rowcta_config(n: int, k: int) -> tuple[int, int]:
    """Choose the cold-weight reduction geometry for a one-row projection."""
    if (n, k) in _CONFIGS:
        return _CONFIGS[(n, k)]
    if k == 8192:
        return (256, 2) if n <= 1024 else (1024, 2)
    if k == 2048:
        return 1024, 2
    if k <= 1536:
        # Short rows: one wave with 16-byte loads measured fastest.
        return 512, 1
    return 512, 4


def gluon_rowcta_bf16_gemv_geometry_gfx950(n: int, k: int) -> tuple[int, int]:
    """Return the ``(BLOCK_K, wave count)`` a single row of ``[N, K]`` runs with.

    A kernel that reproduces this GEMV's per-output reduction, bit for bit,
    must use the same geometry.
    """
    return _rowcta_config(n, k)


# (N, K) -> {power-of-two row bucket: (outputs per wave, waves per program)}
# where projecting a few rows with the single-row reduction beat the default
# selection of ``gemm.mv`` (Gluon MFMA decode GEMM) in cold-weight graph replay
# on MI355X (median of six alternating trials). Eight rows and the 8192x1024
# attention output measured slower than the MFMA GEMM.
_ROWS_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int]]] = {
    (512, 8192): {2: (2, 8), 4: (1, 8)},  # Muse Spark pre-expert TP8
}


def _rows_config(m: int, n: int, k: int) -> tuple[int, int, int, int] | None:
    """``(BLOCK_K, waves per output, outputs per wave, waves per program)``."""
    if m < 2:
        return None
    bucket = 1 << (m - 1).bit_length()
    config = _ROWS_CONFIGS.get((n, k), {}).get(bucket)
    if config is None:
        return None
    block_k, waves = _rowcta_config(n, k)
    outputs_per_wave, num_warps = config
    # This runs inside the selection predicate: a geometry the program does
    # not tile completely falls back to the default selection.
    if (
        num_warps % waves
        or block_k % (_SUBGROUP_SIZE * waves)
        or k % block_k
        or n % (num_warps // waves * outputs_per_wave)
    ):
        return None
    return block_k, waves, outputs_per_wave, num_warps


def supports_gluon_rowcta_bf16_gemv_gfx950(m: int, n: int, k: int) -> bool:
    """Return whether ``m`` rows of ``[N, K]`` run here.

    Any single row runs; more rows need a measured bucket in ``_ROWS_CONFIGS``.
    """
    return m == 1 or _rows_config(m, n, k) is not None


@gluon.jit
def gluon_rowcta_bf16_gemv_rows_sums_gfx950(
    x_ptr,
    weight_ptr,
    first,
    M,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    OUTPUTS_PER_WAVE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Return ``[BLOCK_M, OUTPUTS]`` sums of outputs ``first ..`` for every row.

    Each (row, output) keeps the single-row kernel's reduction for
    ``(BLOCK_K, WAVES_PER_OUTPUT)``: per-lane FP32 chunks accumulated in K order,
    then one sum over the output's subgroups, so row ``m`` is bitwise the
    single-row GEMV of ``x[m]``. Rows sit in each lane's registers, so every
    weight load serves all of them.
    """
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT * OUTPUTS_PER_WAVE
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M, OUTPUTS_PER_WAVE, BLOCK_K // (64 * WAVES_PER_OUTPUT)],
        [1, 1, 64],
        [1, NUM_WARPS // WAVES_PER_OUTPUT, WAVES_PER_OUTPUT],
        [2, 1, 0],
    )
    w_layout: gl.constexpr = gl.SliceLayout(0, layout)
    x_layout: gl.constexpr = gl.SliceLayout(1, layout)
    outputs = first + gl.arange(0, OUTPUTS, layout=gl.SliceLayout(1, w_layout))
    w_columns = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, w_layout))
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_layout))
    x_columns = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, x_layout))
    accumulator = gl.zeros((BLOCK_M, OUTPUTS, BLOCK_K), gl.float32, layout=layout)
    for start in range(0, K, BLOCK_K):
        weight = gl.amd.cdna4.buffer_load(
            weight_ptr,
            (outputs[:, None] * K + start + w_columns[None, :]).to(gl.int32),
            cache=".cg",
        ).to(gl.float32)
        x = gl.amd.cdna4.buffer_load(
            x_ptr,
            (rows[:, None] * K + start + x_columns[None, :]).to(gl.int32),
            mask=(rows < M)[:, None],
            other=0.0,
        ).to(gl.float32)
        accumulator += x[:, None, :] * weight[None, :, :]
    return gl.sum(accumulator, axis=2)


def _rows_launch_metadata(grid, kernel, args):
    """Report weight, input and output bytes to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "bytes": (n * k + m * k + m * n) * args["weight_ptr"].element_size(),
        "flops": 2 * m * n * k,
    }


@gluon.jit(launch_metadata=_rows_launch_metadata, do_not_specialize=("M",))
def gluon_rowcta_bf16_gemv_rows_gfx950(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    N,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    OUTPUTS_PER_WAVE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT * OUTPUTS_PER_WAVE
    first = gl.program_id(0) * OUTPUTS
    sums = gluon_rowcta_bf16_gemv_rows_sums_gfx950(
        x_ptr,
        weight_ptr,
        first,
        M,
        K,
        BLOCK_M,
        BLOCK_K,
        WAVES_PER_OUTPUT,
        OUTPUTS_PER_WAVE,
        NUM_WARPS,
    )
    layout: gl.constexpr = sums.type.layout
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    outputs = first + gl.arange(0, OUTPUTS, layout=gl.SliceLayout(0, layout))
    gl.store(
        out_ptr + rows[:, None] * N + outputs[None, :],
        sums.to(gl.bfloat16),
        mask=(rows < M)[:, None],
    )


@gluon.jit
def _rowcta_bf16_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    output = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_K // (SUBGROUP_SIZE * NUM_WARPS)],
        [SUBGROUP_SIZE],
        [NUM_WARPS],
        [0],
    )
    offsets = gl.arange(0, BLOCK_K, layout=layout)
    accumulator = gl.zeros((BLOCK_K,), gl.float32, layout=layout)
    for start in range(0, K, BLOCK_K):
        columns = start + offsets
        valid = columns < K
        x = gl.amd.cdna4.buffer_load(
            x_ptr,
            columns.to(gl.int32),
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        weight = gl.amd.cdna4.buffer_load(
            weight_ptr,
            (output * K + columns).to(gl.int32),
            mask=valid,
            other=0.0,
            cache=".cg",
        ).to(gl.float32)
        accumulator += x * weight
    gl.store(out_ptr + output, gl.sum(accumulator, axis=0))


def gluon_rowcta_bf16_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` for one or a few contiguous BF16 rows.

    Row ``m`` of the result is bitwise the single-row projection of ``x[m]``.
    More than one row needs a measured bucket (see
    ``supports_gluon_rowcta_bf16_gemv_gfx950``).

    Args:
        x: Contiguous BF16 input shaped ``[M, K]``.
        weight: Contiguous BF16 weight shaped ``[N, K]``.
        out: Contiguous BF16 destination shaped ``[M, N]``.

    Returns:
        ``out`` after the projection completes.
    """
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous BF16 GPU rows")
    if (
        weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous BF16 [N, K] beside x")
    m, k = x.shape
    n = weight.shape[0]
    if (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [M, N] beside x")
    if k <= 0:
        raise ValueError("K must be positive")
    if m == 1:
        block_k, num_warps = _rowcta_config(n, k)
        _rowcta_bf16_gemv_kernel[(n,)](
            x,
            weight,
            out,
            K=k,
            BLOCK_K=block_k,
            NUM_WARPS=num_warps,
            SUBGROUP_SIZE=_SUBGROUP_SIZE,
            num_warps=num_warps,
            waves_per_eu=0,
        )
        return out
    config = _rows_config(m, n, k)
    if config is None:
        raise ValueError(f"no measured {m}-row rowcta GEMV for [{n}, {k}]")
    block_k, waves, outputs_per_wave, num_warps = config
    gluon_rowcta_bf16_gemv_rows_gfx950[(n // (num_warps // waves * outputs_per_wave),)](
        x,
        weight,
        out,
        m,
        n,
        K=k,
        BLOCK_M=1 << (m - 1).bit_length(),
        BLOCK_K=block_k,
        WAVES_PER_OUTPUT=waves,
        OUTPUTS_PER_WAVE=outputs_per_wave,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out


# (N, K, heads) where one gated launch beat the gated norm plus this GEMV by
# at least 5% of the attention tail in 8-rank MI355X graph replay (median of
# six alternating trials); Muse Spark's TP4 shape measured slower fused.
_GATED_SHAPES = frozenset(
    {
        (8192, 1024, 16),  # Muse Spark TP8 attention output projection
    }
)
# Four-subgroup programs with two outputs per subgroup measured fastest.
_GATED_NUM_WARPS = 4
_GATED_OUTPUTS_PER_WAVE = 2
# triton_weightless_rmsnorm_gated_sigmoid launches four warps; its default
# layout, and so its per-head sums, follow from that count.
_GATE_NORM_NUM_WARPS = 4


@gluon.constexpr_function
def _gate_norm_layout(
    num_heads: int, head_dim: int, norm_warps: int, num_warps: int
) -> gl.constexpr:
    """Lay out a [heads, head_dim] tile as the Triton gated-norm kernel does.

    That kernel runs ``norm_warps`` warps; its per-head sums run over the
    columns of this layout, so matching them keeps every value bit for bit.
    Heads may span a different subgroup count.
    """
    heads = 1 << (num_heads - 1).bit_length()
    per_thread = min(8, max(1, heads * head_dim // (64 * norm_warps)))
    lanes = head_dim // per_thread
    return gl.BlockedLayout(
        [1, per_thread], [64 // lanes, lanes], [num_warps, 1], [1, 0]
    )


def _gated_launch_metadata(grid, kernel, args):
    """Report weight, input, gate and output bytes to Proton."""
    n, k = args["out_ptr"].numel(), args["K"]
    return {
        "name": kernel.name,
        "bytes": (n * k + 2 * k + n) * args["weight_ptr"].element_size(),
        "flops": 2 * n * k,
    }


@gluon.jit(launch_metadata=_gated_launch_metadata)
def gluon_gated_rowcta_bf16_gemv_gfx950(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    K: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    EPS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    OUTPUTS_PER_WAVE: gl.constexpr,
    NORM_WARPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    HEAD_DIM: gl.constexpr = 64
    HEADS: gl.constexpr = BLOCK_K // HEAD_DIM
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT * OUTPUTS_PER_WAVE
    # Each output keeps the rowcta reduction for (BLOCK_K, WAVES_PER_OUTPUT).
    layout: gl.constexpr = gl.BlockedLayout(
        [OUTPUTS_PER_WAVE, BLOCK_K // (64 * WAVES_PER_OUTPUT)],
        [1, 64],
        [NUM_WARPS // WAVES_PER_OUTPUT, WAVES_PER_OUTPUT],
        [1, 0],
    )
    head_layout: gl.constexpr = _gate_norm_layout(
        NUM_HEADS, HEAD_DIM, NORM_WARPS, NUM_WARPS
    )
    outputs = gl.program_id(0) * OUTPUTS + gl.arange(
        0, OUTPUTS, layout=gl.SliceLayout(1, layout)
    )
    offsets = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, layout))
    heads = gl.arange(0, HEADS, layout=gl.SliceLayout(1, head_layout))
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, head_layout))
    accumulator = gl.zeros((OUTPUTS, BLOCK_K), gl.float32, layout=layout)
    for chunk in gl.static_range(K // BLOCK_K):
        columns = chunk * BLOCK_K + offsets
        weight = gl.amd.cdna4.buffer_load(
            weight_ptr,
            (outputs[:, None] * K + columns[None, :]).to(gl.int32),
            cache=".cg",
        ).to(gl.float32)
        # The chunk's heads, normalized and gated as the Triton kernel rounds them.
        head_offsets = (chunk * HEADS + heads)[:, None] * HEAD_DIM + dims[None, :]
        gate = gl.amd.cdna4.buffer_load(gate_ptr, head_offsets).to(gl.float32)
        values = gl.amd.cdna4.buffer_load(x_ptr, head_offsets).to(gl.float32)
        variance = gl.sum(values * values, axis=1) / HEAD_DIM
        gated = values * gl.rsqrt(variance[:, None] + EPS) * (1 / (1 + gl.exp(-gate)))
        x = gl.convert_layout(
            gl.reshape(gated.to(gl.bfloat16), [BLOCK_K]), gl.SliceLayout(0, layout)
        ).to(gl.float32)
        accumulator += x[None, :] * weight
    gl.store(out_ptr + outputs, gl.sum(accumulator, axis=1).to(gl.bfloat16))


def _gated_geometry(
    n: int, k: int, num_heads: int, head_dim: int
) -> tuple[int, int] | None:
    """``(BLOCK_K, waves per output)`` of one gated launch, or None.

    The shape must be measured in ``_GATED_SHAPES`` and its single-row geometry
    must tile the four-subgroup programs completely, or the launch could not
    reproduce the separate GEMV.
    """
    if (
        (n, k, num_heads) not in _GATED_SHAPES
        or head_dim != 64
        or head_dim * num_heads != k
    ):
        return None
    block_k, waves = _rowcta_config(n, k)
    if _GATED_NUM_WARPS % waves or block_k % (64 * waves) or k % block_k:
        return None
    outputs_per_program = _GATED_NUM_WARPS // waves * _GATED_OUTPUTS_PER_WAVE
    if n % outputs_per_program:
        return None
    return block_k, waves


def supports_gluon_gated_rowcta_bf16_gemv_gfx950(
    n: int, k: int, num_heads: int, head_dim: int
) -> bool:
    """Whether one gated row of ``[N, K]`` runs in a single launch here."""
    return _gated_geometry(n, k, num_heads, head_dim) is not None


def launch_gluon_gated_rowcta_bf16_gemv_gfx950(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    eps: float,
) -> torch.Tensor:
    """Apply per-head weightless RMSNorm and sigmoid gating, then project one BF16 row.

    Every program recomputes the gated row from ``x`` and ``gate`` and keeps
    both the Triton gated norm's rounding and the single-row GEMV's
    reduction, so ``out`` is bitwise ``weightless_rmsnorm_gated_sigmoid``
    followed by ``gluon_rowcta_bf16_gemv_gfx950``.

    Args:
        x: Contiguous BF16 attention row ``[1, num_heads * 64]``.
        gate: BF16 gate logits with ``x``'s shape and a unit inner stride.
        weight: Contiguous BF16 weight ``[N, K]``.
        out: Contiguous BF16 destination ``[1, N]``.
        num_heads: Heads in ``x``; each head is 64 wide.
        eps: Epsilon added to each per-head variance.

    Returns:
        ``out`` after the projection completes.
    """
    n, k = weight.shape
    geometry = _gated_geometry(n, k, num_heads, x.shape[-1] // max(num_heads, 1))
    if geometry is None:
        raise ValueError(
            f"no single-launch gated GEMV for [{n}, {k}], {num_heads} heads"
        )
    if (
        x.shape != (1, k)
        or gate.shape != x.shape
        or x.dtype != torch.bfloat16
        or gate.dtype != x.dtype
        or weight.dtype != x.dtype
        or not x.is_contiguous()
        or gate.stride(1) != 1
        or not weight.is_contiguous()
        or out.shape != (1, n)
        or out.dtype != x.dtype
        or not out.is_contiguous()
        or len({x.device, gate.device, weight.device, out.device}) != 1
    ):
        raise ValueError(
            "expected one contiguous BF16 row, its gate, [N, K] and [1, N]"
        )
    block_k, waves = geometry
    outputs_per_program = _GATED_NUM_WARPS // waves * _GATED_OUTPUTS_PER_WAVE
    gluon_gated_rowcta_bf16_gemv_gfx950[(n // outputs_per_program,)](
        x,
        gate,
        weight,
        out,
        K=k,
        NUM_HEADS=num_heads,
        EPS=eps,
        BLOCK_K=block_k,
        WAVES_PER_OUTPUT=waves,
        OUTPUTS_PER_WAVE=_GATED_OUTPUTS_PER_WAVE,
        NORM_WARPS=_GATE_NORM_NUM_WARPS,
        NUM_WARPS=_GATED_NUM_WARPS,
        num_warps=_GATED_NUM_WARPS,
    )
    return out


# Single-row shapes (N, K) where this GEMV beat ``gemm.mm``'s own path (Torch's
# skinny GEMM or hipBLASLt) by at least 5% in cold-cache standalone sweeps on
# MI350X, and stayed faster in serving traces. ``gemm.mm`` hands a single row
# to this GEMV only on these shapes (the decode GEMV's own measured shapes
# route to ``gemm.mv`` separately); direct ``gemm.mv`` calls take this GEMV for
# any single row.
_PREFERRED_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {
        (160, 4096),
        (288, 4096),
        (512, 5120),
        (512, 8192),
        (896, 21504),
        (2112, 14336),
        (2304, 8192),
        (2560, 2560),
        (2560, 3072),
        (3648, 7168),
        (4608, 7168),
        (4608, 8192),
        (6656, 2560),
        (7168, 1536),
        (7168, 2304),
        (7168, 4224),
        (8192, 1024),
        (8448, 7168),
        (12800, 2560),
        (20480, 7168),
        (32320, 5120),
        (38720, 4096),
        (124160, 2560),
    }
)


def prefers_gluon_rowcta_bf16_gemv_gfx950(n: int, k: int) -> bool:
    """Whether a single row of ``[N, K]`` measured faster here than in mm."""
    return (n, k) in _PREFERRED_SHAPES


__all__ = [
    "launch_gluon_gated_rowcta_bf16_gemv_gfx950",
    "gluon_rowcta_bf16_gemv_geometry_gfx950",
    "gluon_rowcta_bf16_gemv_gfx950",
    "gluon_rowcta_bf16_gemv_rows_sums_gfx950",
    "prefers_gluon_rowcta_bf16_gemv_gfx950",
    "supports_gluon_gated_rowcta_bf16_gemv_gfx950",
    "supports_gluon_rowcta_bf16_gemv_gfx950",
]
