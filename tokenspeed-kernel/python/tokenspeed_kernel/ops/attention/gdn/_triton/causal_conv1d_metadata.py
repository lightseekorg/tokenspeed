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

"""Per-forward, layer-shared scheduling for the short causal convolution."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton

CAUSAL_CONV1D_BLOCK_M = 8


@dataclass(frozen=True)
class CausalConv1dPrefillMetadata:
    """Read-only program-to-request and program-to-chunk maps for one forward."""

    batch_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    block_m: int


@triton.jit
def _build_causal_conv1d_metadata_kernel(
    query_start_loc,
    batch_indices,
    chunk_offsets,
    num_requests,
    BLOCK_M: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    STORE_BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    rows = tl.arange(0, REQUEST_BLOCK)
    starts = tl.load(query_start_loc + rows, mask=rows < num_requests, other=0)
    ends = tl.load(query_start_loc + rows + 1, mask=rows < num_requests, other=0)
    chunks = tl.cdiv(ends - starts, BLOCK_M)
    first_chunk = tl.sum(tl.where(rows < request, chunks, 0))
    start = tl.load(query_start_loc + request)
    end = tl.load(query_start_loc + request + 1)
    count = tl.cdiv(end - start, BLOCK_M)
    offsets = tl.arange(0, STORE_BLOCK)
    for base in range(tl.cdiv(count, STORE_BLOCK)):
        local = base * STORE_BLOCK + offsets
        tl.store(batch_indices + first_chunk + local, request, mask=local < count)
        tl.store(chunk_offsets + first_chunk + local, local, mask=local < count)


def build_causal_conv1d_prefill_metadata(
    query_start_loc: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_m: int,
) -> CausalConv1dPrefillMetadata:
    """Build convolution indices once; every layer consumes the same tensors.

    Args:
        query_start_loc: Contiguous int32/int64 device boundaries [requests+1],
            beginning at zero; their differences must equal seq_lens_cpu.
        seq_lens_cpu: CPU integer lengths [requests], from the same metadata
            build as query_start_loc. Used only to size the output, never
            uploaded or read back from the GPU.
        block_m: Tokens per convolution program; also used by the consumer.

    Returns:
        Two exactly sized int32 device index arrays and their block size.
        Storage belongs to this forward, not a mutable cross-forward cache.
        Empty requests produce no programs. CUDA/HIP uses one Triton launch
        with no index-table initialization or H2D; CPU provides a reference.
    """
    if block_m <= 0:
        raise ValueError("convolution block_m must be positive")
    if (
        seq_lens_cpu.device.type != "cpu"
        or seq_lens_cpu.ndim != 1
        or seq_lens_cpu.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("seq_lens_cpu must be a one-dimensional CPU integer tensor")
    if (
        query_start_loc.ndim != 1
        or not query_start_loc.is_contiguous()
        or query_start_loc.dtype not in (torch.int32, torch.int64)
        or query_start_loc.numel() != seq_lens_cpu.numel() + 1
    ):
        raise ValueError(
            "query_start_loc must contain one boundary per request plus one"
        )
    lengths = seq_lens_cpu.tolist()
    if any(length < 0 for length in lengths):
        raise ValueError("convolution sequence lengths must be nonnegative")
    counts = [(length + block_m - 1) // block_m for length in lengths]
    num_programs = sum(counts)
    indices = torch.empty(
        (2, num_programs), dtype=torch.int32, device=query_start_loc.device
    )
    metadata = CausalConv1dPrefillMetadata(
        batch_indices=indices[0], chunk_offsets=indices[1], block_m=block_m
    )
    if num_programs == 0:
        return metadata
    if query_start_loc.is_cuda:
        _build_causal_conv1d_metadata_kernel[(len(lengths),)](
            query_start_loc,
            metadata.batch_indices,
            metadata.chunk_offsets,
            len(lengths),
            BLOCK_M=block_m,
            REQUEST_BLOCK=triton.next_power_of_2(len(lengths)),
            STORE_BLOCK=256,
        )
    else:
        cursor = 0
        for request, count in enumerate(counts):
            metadata.batch_indices[cursor : cursor + count] = request
            metadata.chunk_offsets[cursor : cursor + count] = torch.arange(
                count, dtype=torch.int32
            )
            cursor += count
    return metadata
