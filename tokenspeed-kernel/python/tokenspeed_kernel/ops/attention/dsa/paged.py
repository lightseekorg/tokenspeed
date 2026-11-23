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

"""Selected computations for BF16 paged DSA with owner-local caches.

These callables hold implementation choices only. Request metadata, scratch
storage and native communication registrations belong to their runtime owners.
No operation here gathers historical KV. Only the fused prologue writes the
main attention cache; Index-K storage has a separate operation.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _select(operation, dtype):
    return select_kernel(
        "attention",
        f"dsa_{operation}",
        format_signature(input=dense_tensor_format(dtype)),
    )


@dataclass(frozen=True)
class PagedDSAIndexerKernels:
    """Index-K storage and request-local sparse selection.

    store_index_keys(rows, cache, slots) writes only nonnegative owner-local
    slots. select_indices(q, cache, weights, q_ends, lengths, table, topk,
    initial, window) returns logical indices and full-scan chunk hints.
    select_candidates takes per-group initial/window counts instead, plus
    sparse_mode and per-query visible_lengths, and returns indices/scores.
    select_local_indices(indices, positions, rank) compacts the global TopK
    positions into this owner's indices and chunk counts. merge_topk performs
    the same selection using the explicitly bound native communicator.
    """

    store_index_keys: Callable
    select_indices: Callable
    select_candidates: Callable
    select_local_indices: Callable
    merge_topk: Callable


def select_paged_dsa_indexer() -> PagedDSAIndexerKernels:
    """Select BF16 paged Indexer operations once, outside forward/capture."""
    return PagedDSAIndexerKernels(
        store_index_keys=_select("store_index_keys", torch.bfloat16),
        select_indices=_select("paged_select_indices", torch.bfloat16),
        select_candidates=_select("paged_select_candidates", torch.bfloat16),
        select_local_indices=_select("select_local_indices", torch.int32),
        merge_topk=_select("merge_topk", torch.int32),
    )


@dataclass(frozen=True)
class PagedDSAAttentionKernels:
    """Sparse attention and LSE-weighted partial merging.

    attention(q, qr, cache, indices, chunks, q_ends, lengths, table, scale)
    returns BF16 [T,H,512]. attention_partial(packed_q, cache, indices, chunks,
    q_ends, lengths, table, scale, *, sparse_mode, out=None) consumes queries
    [rank,T,local_H,576] and returns FP32 O[H,T,512], max/sum[1,T,H].
    Optional out holds stable producer buffers borrowed from the native owner.
    merge_partials(O[P,N,D], lse[P,N]) returns BF16 [N,D]; callers sanitize empty
    shards before exchange. merge_lse_direct(O,max,sum,*,group,rank_size) merges
    via the native communicator and returns BF16 [local_H,T,512].
    """

    attention: Callable
    attention_partial: Callable
    merge_partials: Callable
    merge_lse_direct: Callable


def select_paged_dsa_attention() -> PagedDSAAttentionKernels:
    """Select the paged SFA and partial-merge computations outside capture."""
    return PagedDSAAttentionKernels(
        attention=_select("paged_attention", torch.bfloat16),
        attention_partial=_select("paged_attention_partial", torch.bfloat16),
        merge_partials=_select("merge_partials", torch.float32),
        merge_lse_direct=_select("merge_lse_direct", torch.float32),
    )


@dataclass(frozen=True)
class DSAProjectionKernels:
    """Prepared fused projection computations, separate from native resources.

    prepare_weight preserves coefficient bits in the provider's weight layout.
    prolog consumes prepared weights, router slots and a borrowed native window;
    it writes KV/Index-K and the supplied query/index-query/score output buffers.
    output_projection produces the local heads' O-projection contribution;
    the model's communication manager still owns TP aggregation.
    """

    prepare_weight: Callable
    prolog: Callable
    output_projection: Callable


def select_dsa_projection_kernels() -> DSAProjectionKernels:
    """Select fused projection and weight preparation without allocating buffers."""
    return DSAProjectionKernels(
        prepare_weight=_select("prepare_projection_weight", torch.bfloat16),
        prolog=_select("fused_prolog", torch.bfloat16),
        output_projection=_select("output_projection_partial", torch.bfloat16),
    )
