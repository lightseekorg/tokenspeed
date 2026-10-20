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

"""Position preparation for pool-indexed sampling kernels."""

import torch


def prepare_sampling_offsets(
    valid_cache_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    extend_lengths: torch.Tensor | None,
    *,
    out: torch.Tensor,
) -> None:
    """Write absolute target positions for the requests this call samples.

    Args:
        valid_cache_lengths: Pool-indexed accepted input frontiers before forward.
        req_pool_indices: Batch-ordered slots; positive slots must be unique.
            Zero is the scheduler's reserved padding slot and is never written.
        extend_lengths: Input lengths of the leading prefill requests, or None
            for an all-decode call. Decode/verify starts at frontier + 1,
            independent of its candidate width; the verifier adds depth later.
        out: Contiguous int64 pool-sized scratch, initialized to zero by its
            owner. Only referenced positive slots are overwritten. The frontier
            must not alias it. Unreferenced slots are not valid sampling inputs.
    """
    if valid_cache_lengths.ndim != 1 or req_pool_indices.ndim != 1:
        raise ValueError("sampling frontiers and slot indices must be one-dimensional")
    if out.shape != valid_cache_lengths.shape or out.dtype != torch.int64:
        raise ValueError("sampling offsets need int64 scratch with the frontier shape")
    if out.data_ptr() == valid_cache_lengths.data_ptr():
        raise ValueError("sampling offsets must not overwrite the accepted frontier")
    if not out.is_contiguous() or not valid_cache_lengths.is_contiguous():
        raise ValueError("sampling pool buffers must be contiguous")
    rows = req_pool_indices.numel()
    extends = 0 if extend_lengths is None else extend_lengths.numel()
    if extends > rows or (extend_lengths is not None and extend_lengths.ndim != 1):
        raise ValueError("prefill lengths must describe a leading subset of requests")
    tensors = [out, req_pool_indices]
    if extend_lengths is not None:
        tensors.append(extend_lengths)
    integer_inputs = [valid_cache_lengths, req_pool_indices]
    if extend_lengths is not None:
        integer_inputs.append(extend_lengths)
    if any(t.dtype not in (torch.int32, torch.int64) for t in integer_inputs):
        raise ValueError("sampling frontiers, slots, and lengths must be int32/int64")
    if any(t.device != valid_cache_lengths.device for t in tensors):
        raise ValueError("sampling position inputs must share a device")
    if rows == 0:
        return
    if valid_cache_lengths.is_cuda:
        from tokenspeed_kernel.ops.sampling.triton.positions import prepare_offsets

        prepare_offsets(valid_cache_lengths, req_pool_indices, extend_lengths, out)
        return
    # CPU reference for the metadata contract; CUDA always uses the fused kernel.
    positions = valid_cache_lengths[req_pool_indices].to(torch.int64) + 1
    if extends:
        positions[:extends] = (
            valid_cache_lengths[req_pool_indices[:extends]].to(torch.int64)
            + extend_lengths
        )
    live = req_pool_indices != 0
    out.index_copy_(0, req_pool_indices[live].long(), positions[live])
