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

"""V4.1 weighted SwiGLU with FP32 clamp and pre-down route scaling."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _weighted_swiglu(
    X,
    W,
    Y,
    H: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * XS0 + col * XS1, col < H, other=0).to(tl.float32)
    up = tl.load(X + row * XS0 + (H + col) * XS1, col < H, other=0).to(tl.float32)
    if LIMIT > 0:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    silu = gate / (1.0 + tl.exp(-gate))
    w = tl.load(W + row * WS).to(tl.float32)
    result = (silu * up) * w
    tl.store(Y + row * H + col, result, col < H)


def weighted_swiglu(gate_up, route_weights, limit=10.0, *, block_size=4096):
    if gate_up.device.type != "npu" or route_weights.device != gate_up.device:
        raise ValueError("weighted SwiGLU requires NPU tensors on the same device")
    if (
        gate_up.ndim != 2
        or gate_up.shape[1] % 2
        or route_weights.shape != (gate_up.shape[0],)
    ):
        raise ValueError("Expected gate_up [rows,2H] and route_weights [rows]")
    if gate_up.dtype != torch.bfloat16 or route_weights.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("Expected BF16 activations and FP32/BF16 route weights")
    rows, width = gate_up.shape
    hidden = width // 2
    if hidden % 32:
        raise ValueError("Hidden width must be a multiple of 32 for aligned NPU stores")
    result = torch.empty((rows, hidden), dtype=gate_up.dtype, device=gate_up.device)
    if rows and hidden:
        _weighted_swiglu[(rows, triton.cdiv(hidden, block_size))](
            gate_up,
            route_weights,
            result,
            hidden,
            *gate_up.stride(),
            route_weights.stride(0),
            float(limit),
            block_size,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return result


@triton.jit
def _weighted_swiglu_active(
    X,
    W,
    Counts,
    Y,
    H: tl.constexpr,
    E: tl.constexpr,
    N,
    LIMIT: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    experts = tl.arange(0, triton.next_power_of_2(E))
    valid = tl.sum(tl.load(Counts + experts, experts < E, other=0).to(tl.int32), 0)
    valid = tl.minimum(valid, N)
    col = tl.arange(0, BLOCK)
    # A fixed number of workers avoids launching a program for every reserved
    # row. Counts stay on device and can change on every graph replay.
    for row in range(tl.program_id(0), valid, CORES):
        gate = tl.load(X + row * 2 * H + col, col < H, other=0).to(tl.float32)
        up = tl.load(X + row * 2 * H + H + col, col < H, other=0).to(tl.float32)
        if LIMIT > 0:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        w = tl.load(W + row).to(tl.float32)
        result = ((gate / (1.0 + tl.exp(-gate))) * up) * w
        tl.store(Y + row * H + col, result, col < H)


def weighted_swiglu_active(gate_up, route_weights, expert_counts, limit):
    """Compute the valid sorted prefix; unused capacity remains unspecified.

    ``expert_counts`` contains counts, not cumulative offsets. The downstream
    GMM consumes precisely their sum and must never read the reserved tail.
    """
    if (
        expert_counts.device != gate_up.device
        or expert_counts.ndim != 1
        or expert_counts.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected device INT32/INT64 expert counts")
    if (
        gate_up.ndim != 2
        or gate_up.shape[1] % 64
        or route_weights.shape != (gate_up.shape[0],)
    ):
        raise ValueError("Expected aligned gate/up rows and one weight per row")
    hidden = gate_up.shape[1] // 2
    if (
        gate_up.device.type != "npu"
        or gate_up.dtype != torch.bfloat16
        or route_weights.device != gate_up.device
        or route_weights.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("Expected NPU BF16 activations and FP32/BF16 weights")
    if (
        not gate_up.is_contiguous()
        or not route_weights.is_contiguous()
        or not expert_counts.is_contiguous()
        or hidden > 4096
        or expert_counts.numel() == 0
    ):
        return weighted_swiglu(gate_up, route_weights, limit)
    out = torch.empty(
        (gate_up.shape[0], hidden), device=gate_up.device, dtype=gate_up.dtype
    )
    if out.numel():
        _weighted_swiglu_active[(48,)](
            gate_up,
            route_weights,
            expert_counts,
            out,
            hidden,
            expert_counts.numel(),
            gate_up.shape[0],
            float(limit),
            48,
            triton.next_power_of_2(hidden),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
