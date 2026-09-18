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


"""Small-message projection reduction over persistent symmetric GEMM storage."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication.triton import _alloc_symm, _peer_ptrs_dev
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def owner_reduce(
    PTRS,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for peer in tl.static_range(P):
        ptr = tl.load(PTRS + peer).to(tl.pointer_type(Y.dtype.element_ty))
        total += tl.load(ptr + R * M * H + i, mask=i < M * H, other=0).to(tl.float32)
    tl.store(Y + i, total, mask=i < M * H)


class ProjectionPeerState:
    def __init__(self, group, max_rows, hidden, device):
        """Allocate TP4 BF16 partials before capture; peers use one serialized stream."""
        if group.size() != 4 or max_rows <= 0 or hidden <= 0:
            raise ValueError(
                "Projection peer reduction requires TP4 and positive capacity"
            )
        self.max_rows = max_rows
        self.group = group
        self.p, self.r, self.hidden = group.size(), group.rank(), hidden
        self.buffer, self.handle = _alloc_symm(
            (self.p * max_rows, hidden), torch.bfloat16, device, group
        )
        self.ptrs = _peer_ptrs_dev(
            self.handle, self.buffer.shape, self.buffer.dtype, self.p, device
        )

    def input_buffer(self, rows):
        """Return the borrowed GEMM destination for equal physical rows per peer."""
        if not 0 < rows <= self.max_rows:
            raise ValueError("Projection peer reduction exceeds prepared capacity")
        return self.buffer[: self.p * rows]

    def reduce(self, partial, rows):
        """Return owned local rows; all peers must call, including padded ranks."""
        return self._reduce(partial, rows, True)

    def _reduce(self, partial, rows, synchronize_reuse):
        destination = self.input_buffer(rows)
        if partial.shape != destination.shape or partial.dtype != destination.dtype:
            raise ValueError("Projection partials have incompatible shape or dtype")
        # Direct-output GEMMs already wrote this exact view. Other linear
        # methods may still stage here without changing the collective protocol.
        if (
            partial.data_ptr() != destination.data_ptr()
            or partial.stride() != destination.stride()
        ):
            destination.copy_(partial)
        # Publish every peer's GEMM writes before reading remote partials.
        self.handle.barrier(channel=0)
        out = torch.empty(
            (rows, self.hidden), device=partial.device, dtype=partial.dtype
        )
        owner_reduce[(triton.cdiv(rows * self.hidden, 1024),)](
            self.ptrs, out, rows, self.hidden, self.p, self.r, 1024
        )
        # Standalone callers need an explicit fence before reusing partials.
        # The complete projection can instead use its next A2A entry barrier.
        if synchronize_reuse:
            self.handle.barrier(channel=1)
        return out


@register_kernel(
    "communication",
    "projection_reduce_scatter",
    name="triton_projection_reduce_scatter",
    solution="triton",
    signatures=format_signatures(("partial",), "dense", {torch.bfloat16}),
)
def triton_projection_reduce_scatter(state, partial, rows):
    """Reduce TP4 [4*rows,H] partials to owned [rows,H] local-token outputs.

    State is prepared collectively before capture. Calls and all consumers of
    its borrowed input buffer must be serialized on one stream per subgroup.
    """
    return state.reduce(partial, rows)


@register_kernel(
    "communication",
    "projection_reduce_scatter_after_a2a",
    name="triton_projection_reduce_scatter_after_a2a",
    solution="triton",
    signatures=format_signatures(("partial",), "dense", {torch.bfloat16}),
)
def triton_projection_reduce_scatter_after_a2a(state, partial, rows):
    """Reduce TP4 partials, deferring reuse synchronization to the next A2A.

    Arguments and output match triton_projection_reduce_scatter. Every future
    write to state.buffer must follow a same-subgroup A2A entry barrier or an
    explicit symmetric-memory pre-write barrier on the same serialized stream.
    That barrier observes all previous reduction reads complete before any
    peer launches its next GEMM. NCCL A2A itself does not supply this guarantee:
    callers using NCCL must fence before writing, including after intervening
    empty or NCCL-only batches. Standalone reductions use the fenced entry point.
    """
    return state._reduce(partial, rows, False)
