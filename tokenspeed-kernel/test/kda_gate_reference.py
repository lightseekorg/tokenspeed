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
# Original recurrence: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.


"""KDA gate precompute kernels kept for tests, excluded from the runtime package.

The per-token gate ``lower_bound * sigmoid(exp(A_log) * (f_a @ f_b.T + dt_bias))``
is computed two ways over one descriptor table: a reduction kernel and a
tensor-core kernel; test_kda_gate_precompute.py checks that they agree.
"""

import torch
from tokenspeed_kernel._triton import tl, triton

_SM_COUNT: dict[int, int] = {}


def _sm_count(device) -> int:
    """Multiprocessor count for the device, memoized per index."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SM_COUNT.get(index)
    if sms is None:
        sms = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = sms
    return sms


def _gate_tiling(rows: int, num_heads: int, head_dim: int, device, layers: int = 1):
    """Rows and gate channels per program: the widest tile the grid can afford.

    Both knobs only decide which rows a program owns -- every row still
    reduces over the whole D_FA axis -- but the tile shape does pick the
    reduction tree, so two tilings agree to a few ulp rather than bit for bit.
    What the choice really moves is the ``[BK, D_FA]`` weight traffic, which
    scales with ``cdiv(rows, block_t)``: a narrower row block buys parallelism
    by reading that tile again in every program, so taking one the grid did
    not need is pure amplification.

    ``layers`` is the batched caller's outer grid dimension. Omitting it made
    the estimate short by that factor and drove the all-layer launch to the
    narrowest tiling at every batch size.
    """
    sms = _sm_count(device)
    for block_t, block_k in ((8, 32), (4, 32), (2, 16), (1, 16)):
        if block_t > rows:
            continue
        programs = (
            layers
            * num_heads
            * triton.cdiv(rows, block_t)
            * triton.cdiv(head_dim, block_k)
        )
        if programs >= 4 * sms:
            return block_t, min(block_k, triton.next_power_of_2(head_dim))
    return 1, min(16, triton.next_power_of_2(head_dim))


def _gate_tiling_dot(rows: int, head_dim: int):
    """Tiling from shape alone, so looped and batched launches match bitwise."""
    return 32 if rows >= 32 else 16, min(32, triton.next_power_of_2(head_dim))


@triton.jit
def batched_kda_gate_precompute_kernel(
    addresses,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Descriptor-table form of the established KDA gate precompute."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    ab = i_l * 10
    f_a = tl.load(addresses + ab + 3).to(tl.pointer_type(tl.bfloat16))
    f_b = tl.load(addresses + ab + 4).to(tl.pointer_type(tl.bfloat16))
    A_log = tl.load(addresses + ab + 6).to(tl.pointer_type(tl.float32))
    dt_bias = tl.load(addresses + ab + 7).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    for j in range(BT):
        row = i_tb * BT + j
        if row < rows:
            fa = tl.load(f_a + row * stride_fa + o_fa).to(tl.float32)
            gate = tl.sum(wfb * fa[None, :], axis=1) + b_bias
            gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
            tl.store(gate_scratch + row * stride_gate + gc, gate, mask=mask_k)


@triton.jit
def batched_kda_gate_precompute_dot_kernel(
    addresses,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Tensor-core form of the batched KDA gate precompute."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    ab = i_l * 10
    f_a = tl.load(addresses + ab + 3).to(tl.pointer_type(tl.bfloat16))
    f_b = tl.load(addresses + ab + 4).to(tl.pointer_type(tl.bfloat16))
    A_log = tl.load(addresses + ab + 6).to(tl.pointer_type(tl.float32))
    dt_bias = tl.load(addresses + ab + 7).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    o_t = tl.arange(0, BT)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    row = i_tb * BT + o_t
    mask_t = row < rows
    # Left as bf16: tl.dot accumulates in fp32, and the product of two bf16
    # values is exact there, so upcasting first would only cost bandwidth.
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    )
    fa = tl.load(
        f_a + row[:, None] * stride_fa + o_fa[None, :],
        mask=mask_t[:, None],
        other=0.0,
    )
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    gate = tl.dot(fa, tl.trans(wfb)) + b_bias[None, :]
    gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
    tl.store(
        gate_scratch + row[:, None] * stride_gate + gc[None, :],
        gate,
        mask=mask_t[:, None] & mask_k[None, :],
    )
