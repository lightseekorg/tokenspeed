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

"""Direct paged SFA over a row-aligned shared LCM arena."""

import torch
import triton as tr
import triton.language as tl
from tokenspeed_kernel_npu.ops.dsv41_attention_fusion import apply_sink, compact_indices


class PagedArenaContext:
    """Persistent constants and zero-copy BF16 view; owned by the cache pool."""

    def __init__(self, arena):
        if arena.buffer.device.type != "npu":
            raise ValueError("Direct paged attention requires NPU storage")
        plane = next(p for p in arena.plan.planes if p.plane_id == "flatkv")
        if plane.arena_offset_bytes % 1024:
            raise ValueError("flatkv plane must begin on a BF16-512 row")
        span = plane.bytes_per_lcm_block * (arena.plan.num_lcm_blocks + 1)
        padded_span = (span + 65535) // 65536 * 65536
        if plane.arena_offset_bytes + padded_span > arena.buffer.numel():
            raise ValueError("arena lacks a safe final PA block")
        self.rows = padded_span // 1024
        if self.rows >= 2**31:
            raise ValueError("PA physical row indices exceed int32")
        self.origin_elements = plane.arena_offset_bytes // 2
        self.key = (
            arena.buffer[
                plane.arena_offset_bytes : plane.arena_offset_bytes + padded_span
            ]
            .view(torch.bfloat16)
            .reshape(-1, 64, 1, 512)
        )
        self.key_rope = torch.zeros(
            (self.rows // 64, 64, 1, 64), dtype=torch.bfloat16, device=self.key.device
        )
        self.query_zeros = torch.zeros(
            (32768, 64), dtype=torch.bfloat16, device=self.key.device
        )
        self.block_table = torch.arange(
            self.rows // 64, dtype=torch.int32, device=self.key.device
        ).reshape(1, -1)
        self.seq_kv = torch.tensor(
            [self.rows], dtype=torch.int32, device=self.key.device
        )
        self.storage_ptr = arena.buffer.untyped_storage().data_ptr()
        self.extra_bytes = (
            self.key_rope.nbytes
            + self.query_zeros.nbytes
            + self.block_table.nbytes
            + self.seq_kv.nbytes
        )

    def geometry(self, cache):
        if (
            cache.dtype != torch.bfloat16
            or cache.shape[1:] != (64, 512)
            or cache.stride(1) != 512
            or cache.stride(2) != 1
        ):
            raise ValueError("PA cache must have dense BF16-512 rows")
        if cache.untyped_storage().data_ptr() != self.storage_ptr:
            raise ValueError("PA banks must share the registered arena")
        relative = cache.storage_offset() - self.origin_elements
        if relative % 512 or cache.stride(0) % 512:
            raise ValueError(
                "PA field origin and page stride must be 1024-byte aligned"
            )
        base = relative // 512
        stride = cache.stride(0) // 512
        if base < 0 or base + (cache.shape[0] - 1) * stride + 64 > self.rows:
            raise ValueError("PA field exceeds registered physical rows")
        return base, stride, cache.shape[0] * 64


@tr.jit
def _absolute_rows(
    S0,
    L0,
    S1,
    L1,
    O,
    T,
    W0,
    W1,
    S00,
    S01,
    S10,
    S11,
    L00,
    L10,
    B0,
    P0,
    C0,
    B1,
    P1,
    C1,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    first = c < W0
    local = tl.where(first, c, c - W0)
    # Keep row/index arithmetic int32: validated physical arena < 2**31 rows.
    a = tl.load(S0 + t * S00 + local * S01, (c < W0), other=-1).to(tl.int32)
    b = tl.load(S1 + t * S10 + local * S11, (c >= W0) & (c < W0 + W1), other=-1).to(
        tl.int32
    )
    id = tl.where(first, a, b)
    n0 = tl.load(L0 + t * L00).to(tl.int32)
    n1 = tl.load(L1 + t * L10).to(tl.int32)
    valid = (
        (local < tl.where(first, n0, n1))
        & (id >= 0)
        & (id < tl.where(first, C0, C1))
        & (c < W0 + W1)
    )
    physical = tl.where(first, B0, B1) + (id // 64) * tl.where(first, P0, P1) + id % 64
    tl.store(O + t * (W0 + W1) + c, tl.where(valid, physical, -1), c < W0 + W1)


def selected_paged_attention(q, segments, sink, scale, context):
    if len(segments) not in (1, 2):
        raise ValueError("PA expects SWA and optional global bank")
    first = segments[0]
    second = segments[1] if len(segments) == 2 else first
    c0, s0, l0 = first
    c1, s1, l1 = second
    w0 = s0.shape[1]
    w1 = s1.shape[1] if len(segments) == 2 else 0
    if q.shape[0] == 0 or w0 + w1 == 0:
        return torch.zeros_like(q)
    g0 = context.geometry(c0)
    g1 = context.geometry(c1)
    from tokenspeed_kernel_npu.ops.dsv41_paged_metadata_native import prepare_indices

    prepared = prepare_indices(s0, l0, s1, l1, g0, g1, w1, context.rows)
    if prepared is None:
        ids = torch.empty((q.shape[0], w0 + w1), dtype=torch.int32, device=q.device)
        _absolute_rows[(q.shape[0], tr.cdiv(w0 + w1, 256))](
            s0,
            l0,
            s1,
            l1,
            ids,
            q.shape[0],
            w0,
            w1,
            *s0.stride(),
            *s1.stride(),
            l0.stride(0),
            l1.stride(0),
            *g0,
            *g1,
            BLOCK=256
        )
        compact, counts = compact_indices(ids, context.rows)
    else:
        compact, counts = prepared
    t, h, _ = q.shape
    if t * h > context.query_zeros.shape[0]:
        raise ValueError("PA query exceeds persistent zero-rope slab")
    qr = context.query_zeros[: t * h].reshape(1, t, h, 64)
    import flash_ops  # noqa: F401 - registers custom NPU operators

    y, m, s = torch.ops.custom.npu_sparse_flash_attention(
        q.contiguous().unsqueeze(0),
        context.key,
        context.key,
        compact.reshape(1, t, 1, -1),
        scale,
        block_table=context.block_table,
        actual_seq_lengths_kv=context.seq_kv,
        query_rope=qr,
        key_rope=context.key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return apply_sink(y.reshape_as(q), m, s, sink, counts)
