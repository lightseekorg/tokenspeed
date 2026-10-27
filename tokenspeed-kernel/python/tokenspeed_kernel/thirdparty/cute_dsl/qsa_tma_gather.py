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

"""Explicit, caller-buffered entry for the in-tree SM107 sparse attention port."""

from __future__ import annotations

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.combine import (
    compile_combine,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.kernel import (
    TmaGatherSparseAttentionKernel,
)


@lru_cache(maxsize=None)
def _prepare(
    device: torch.device,
    page_size: int,
    split_kv: int,
    enable_pdl: bool,
    query_group_size: int,
):
    # Only model-static geometry and the explicit tactic key compilation.
    kernel = TmaGatherSparseAttentionKernel(
        page_size=page_size, split_kv=split_kv, query_group_size=query_group_size
    )
    with torch.cuda.device(device):
        core = kernel.compile(enable_pdl=enable_pdl)
        combine = (
            compile_combine(split_kv=split_kv, enable_pdl=enable_pdl)
            if split_kv > 1
            else None
        )
    return core, combine


class TmaGatherSparseAttention:
    """Prepared BF16 paged attention kernel with no request-owned state.

    The constructor compiles the selected static tactic outside CUDA capture.
    ``execute`` reads live metadata on device and uses caller-owned output and
    scratch. This low-level entry is not registered for runtime dispatch yet.

    Args:
        page_size: Positive physical page size, divisible by four.
        split_kv: Explicit split tactic: 1, 4, 8 or 16.
        query_group_size: One for independent selection rows, or two for pairs
            of consecutive queries with identical selected-block lists. The
            paired tactic requires an even query length; it does not check
            device list equality. Each query retains its own causal tail.
        enable_pdl: Whether both launches use programmatic dependency ordering.
        device: CUDA device whose primary context owns the compiled kernels.
    """

    def __init__(
        self,
        *,
        page_size: int,
        split_kv: int,
        enable_pdl: bool,
        device: torch.device,
        query_group_size: int,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("SM107 sparse attention requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 7):
            raise ValueError("SM107 sparse attention requires compute capability 10.7")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Prepare SM107 sparse attention before CUDA graph capture"
            )
        self.device = device
        self.page_size = page_size
        self.split_kv = split_kv
        self.enable_pdl = enable_pdl
        self.query_group_size = query_group_size
        self._core, self._combine = _prepare(
            device, page_size, split_kv, enable_pdl, query_group_size
        )

    def workspace_bytes(
        self, *, batch_size: int, query_length: int, num_query_heads: int
    ) -> int:
        """Return scratch bytes for split-major FP32 output and LSE slabs."""
        if min(batch_size, query_length, num_query_heads) < 0:
            raise ValueError("Workspace dimensions must be nonnegative")
        if self.split_kv == 1:
            return 0
        rows = self.split_kv * batch_size * query_length * num_query_heads
        return rows * (256 + 1) * 4

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_ids: torch.Tensor,
        seq_kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        *,
        scale: float,
        output: torch.Tensor,
        workspace: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute per-query sparse attention into ``output`` on the current stream.

        Args:
            q: BF16 ``[B,S,Hq,256]`` with aligned outer strides; S is 1..4.
            k: Contiguous BF16 NHD cache ``[slots,Hkv,256]``. Hq/Hkv is 12.
            v: Same shape and storage contract as k.
            block_ids: Contiguous int32 ``[B,S,512]`` logical four-token blocks.
                Each row uses min(512, floor((position+1)/4)) entries; negative
                entries are absent. IDs must be distinct and exclude the open tail.
                With query_group_size=2, both rows of a pair must contain the
                same list, including entries used only by the later query.
            seq_kv_lens: Contiguous int32 ``[B]`` live lengths. Query j is at
                position length-S+j; its incomplete tail is appended by the kernel.
            block_table: Int32 ``[B,max_pages]`` with contiguous columns. Its
                row stride may exceed max_pages. Page zero, negative entries,
                out-of-pool entries and missing logical pages contribute no weight.
            scale: Signed softmax scale, normally 1/sqrt(256).
            output: Caller-owned BF16 tensor with q's shape and aligned strides.
            workspace: Contiguous uint8 scratch of at least ``workspace_bytes``;
                pass None for split_kv=1. Buffers must remain live through graph replay.

        Returns:
            The same output tensor. All-invalid and zero-length rows output zero.
        """
        if q.ndim != 4 or q.shape[-1] != 256 or not 1 <= q.shape[1] <= 4:
            raise ValueError("q must have shape [B, S in 1..4, Hq, 256]")
        b, sq, h, _ = q.shape
        if sq % self.query_group_size:
            raise ValueError("query_length must be divisible by query_group_size")
        if b >= 65536:
            raise ValueError("Batch exceeds the CLC work-item encoding")
        if (
            k.ndim != 3
            or k.shape[2] != 256
            or k.shape[1] < 1
            or h != 12 * k.shape[1]
            or v.shape != k.shape
        ):
            raise ValueError("K/V must be [slots,Hkv,256] with Hq/Hkv=12")
        if k.shape[1] >= 65536:
            raise ValueError("KV heads exceed the CLC work-item encoding")
        page = self.page_size
        if k.shape[0] < page or k.shape[0] % page or k.shape[0] >= 2**31:
            raise ValueError(
                "Cache slots must be whole pages within the TMA row-coordinate range"
            )
        if output.shape != q.shape:
            raise ValueError("output must have q's shape")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("output", output)):
            if tensor.device != self.device or tensor.dtype is not torch.bfloat16:
                raise ValueError(f"{name} must be BF16 on {self.device}")
            if (
                tensor.stride(-1) != 1
                or tensor.data_ptr() % 16
                or any(s % 8 for s in tensor.stride()[:-1])
            ):
                raise ValueError(
                    f"{name} must have a contiguous head dimension and 16-byte aligned strides/base"
                )
        for name, tensor in (("q", q), ("output", output)):
            bs, qs, hs, _ = tensor.stride()
            if hs < 256 or qs < h * hs or bs < sq * qs:
                raise ValueError(f"{name} must have non-overlapping BSHD rows")
        if not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("K/V must use contiguous NHD cache storage")
        for name, tensor, shape in (
            ("block_ids", block_ids, (b, sq, 512)),
            ("seq_kv_lens", seq_kv_lens, (b,)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype is not torch.int32
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} has an invalid shape, dtype, device or layout"
                )
        if block_ids.data_ptr() % 16:
            raise ValueError("block_ids must be 16-byte aligned")
        if (
            block_table.ndim != 2
            or block_table.shape[0] != b
            or block_table.shape[1] < 1
            or block_table.dtype is not torch.int32
            or block_table.device != self.device
            or block_table.stride(1) != 1
            or block_table.stride(0) < block_table.shape[1]
        ):
            raise ValueError(
                "block_table must be int32 [B,max_pages] with contiguous columns"
            )
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        need = self.workspace_bytes(batch_size=b, query_length=sq, num_query_heads=h)
        if need:
            if (
                workspace is None
                or workspace.device != self.device
                or workspace.dtype is not torch.uint8
                or not workspace.is_contiguous()
                or workspace.numel() < need
                or workspace.data_ptr() % 16
            ):
                raise ValueError(
                    f"workspace must provide {need} aligned uint8 bytes on {self.device}"
                )
        elif workspace is not None:
            raise ValueError("split_kv=1 does not use workspace; pass None")
        if b == 0:
            return output
        gmem = cute.AddressSpace.gmem

        def ptr(tensor, dtype, align):
            return cute.runtime.make_ptr(
                dtype, tensor.data_ptr(), gmem, assumed_align=align
            )

        qptr, kptr, vptr = (ptr(t, cutlass.BFloat16, 16) for t in (q, k, v))
        optr = ptr(output, cutlass.BFloat16, 16)
        lseptr = None
        ostrides = output.stride()[:3]
        lstrides = (0, 0, 0)
        if need:
            partial_bytes = self.split_kv * b * sq * h * 256 * 4
            optr = cute.runtime.make_ptr(
                cutlass.Float32, workspace.data_ptr(), gmem, assumed_align=16
            )
            lseptr = cute.runtime.make_ptr(
                cutlass.Float32,
                workspace.data_ptr() + partial_bytes,
                gmem,
                assumed_align=16,
            )
            ostrides = (sq * h * 256, h * 256, 256)
            lstrides = (h * sq, sq, 1)
        with torch.cuda.device(self.device):
            stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
            self._core(
                qptr,
                kptr,
                vptr,
                optr,
                lseptr,
                ptr(block_ids, cutlass.Int32, 16),
                None,
                ptr(seq_kv_lens, cutlass.Int32, 4),
                (b, h, k.shape[1], sq, block_table.shape[1] * page),
                q.stride()[:3],
                (page * k.stride(0), k.stride(0), k.stride(1)),
                (page * v.stride(0), v.stride(0), v.stride(1)),
                ostrides,
                lstrides,
                cutlass.Float32(scale * math.log2(math.e)),
                ptr(block_table, cutlass.Int32, 4),
                (
                    k.shape[0] // page,
                    block_table.shape[1],
                    page,
                    0,
                    page,
                    0,
                    block_table.stride(0),
                ),
                stream,
            )
            if self._combine is not None:
                self._combine(
                    optr,
                    lseptr,
                    ptr(output, cutlass.BFloat16, 16),
                    b,
                    sq,
                    h,
                    output.stride()[:3],
                    stream,
                )
        return output
