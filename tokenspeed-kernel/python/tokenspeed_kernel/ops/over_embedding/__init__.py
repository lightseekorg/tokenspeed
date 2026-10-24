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

"""LongCat over-embedding kernel APIs."""

from __future__ import annotations

import torch as _torch
from tokenspeed_kernel.ops.over_embedding.spec import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    longcat_pro_tp8_spec,
)
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def append_packed_lookup_(
    input_ids: _torch.Tensor,
    input_start_offsets: _torch.Tensor,
    req_pool_indices: _torch.Tensor,
    active_request_mask: _torch.Tensor,
    history_token_ids: _torch.Tensor,
    committed_lengths: _torch.Tensor,
    oe_tables: tuple[_torch.Tensor, ...],
    *,
    spec: OverEmbeddingSpec,
    out: _torch.Tensor | None = None,
    solution: str | None = None,
    enable_pdl: bool,
) -> _torch.Tensor:
    """Append target inputs and gather their TP-local OE fragments.

    The kernel writes each active request's packed inputs to
    ``history_token_ids[row, L:L+width]`` and computes its n-gram hashes in
    the same launch. ``L = committed_lengths[row]`` is read-only here; the
    caller publishes the accepted prefix after target verification. DFlash W8
    is represented by offsets ``[0, 8, 16, ...]``; prefill and mixed batches
    use arbitrary nondecreasing offsets.

    Args:
        input_ids: Contiguous int32 tensor ``[M]`` containing all request
            tokens in request-major order.
        input_start_offsets: Contiguous int32 tensor ``[Q+1]`` partitioning
            ``input_ids``. It must start at zero, be nondecreasing, and end at
            ``M``. Empty request intervals are supported. These value
            invariants are owned by the batch builder and are not read back on
            the host in this hot-path API.
        req_pool_indices: Contiguous int64 tensor ``[Q]`` mapping requests to
            their persistent history rows.
        active_request_mask: Contiguous bool tensor ``[Q]``. Inactive graph-padding
            requests skip the table append and produce zero activation.
        history_token_ids: Contiguous int32 tensor
            ``[slot_count, max_context_len]``. Its last row is reserved for
            graph padding.
        committed_lengths: Contiguous int32 publication pointers
            ``[slot_count]``. This kernel never modifies them.
        oe_tables: Compact BF16 OE fragment tables in ``spec.fragments`` order.
        spec: Static profile, TP rank, and fragment ownership contract.
        out: Optional caller-owned contiguous BF16
            ``[M, spec.local_width]`` destination.
        solution: Optional registered implementation name.
        enable_pdl: Enable Programmatic Dependent Launch in the selected lookup
            implementation.

    Returns:
        Packed local branch rows in the same token order as ``input_ids``.
    """
    request_count = input_start_offsets.numel() - 1
    token_count = input_ids.numel()
    expected_shape = (token_count, spec.local_width)
    if out is None:
        out = _torch.empty(
            expected_shape,
            dtype=oe_tables[0].dtype,
            device=input_ids.device,
        )
    elif (
        out.shape != expected_shape
        or out.dtype != oe_tables[0].dtype
        or out.device != input_ids.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must match the packed OE shape, dtype and device")
    if token_count == 0:
        return out

    signature = format_signature(table=dense_tensor_format(oe_tables[0].dtype))
    traits = {
        "fragment_count": len(spec.fragments),
        "fragment_widths": tuple(fragment.feature_width for fragment in spec.fragments),
    }
    kernel = select_kernel(
        "over_embedding",
        "append_packed_lookup",
        signature,
        traits=traits,
        solution=solution,
    )
    shape_params = {
        "request_count": request_count,
        "token_count": token_count,
        "profile": spec.profile,
        "tp_size": spec.tp_size,
        "fragment_count": len(spec.fragments),
        "local_width": spec.local_width,
        "enable_pdl": enable_pdl,
    }
    ShapeCapture.get().record(
        "over_embedding",
        "append_packed_lookup",
        kernel.name,
        oe_tables[0].dtype,
        shape_params,
    )
    with kernel_scope(
        "over_embedding",
        "append_packed_lookup",
        oe_tables[0].dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            input_ids=input_ids,
            input_start_offsets=input_start_offsets,
            req_pool_indices=req_pool_indices,
            active_request_mask=active_request_mask,
            history_token_ids=history_token_ids,
            committed_lengths=committed_lengths,
            oe_tables=oe_tables,
            out=out,
            spec=spec,
            enable_pdl=enable_pdl,
        )
    return out


def project_add_word_(
    word_partial: _torch.Tensor,
    activation: _torch.Tensor,
    projection: _torch.Tensor,
    *,
    scale: int,
    solution: str | None = None,
) -> _torch.Tensor:
    """Project local OE fragments and merge them into word storage in place.

    Computes ``word_partial / scale + activation @ projection / scale``.
    The checkpoint loader must provide the contiguous physical projection
    layout ``[K_local, hidden_size]``.

    Args:
        word_partial: Contiguous BF16 ``[M, hidden_size]`` input and output.
        activation: Contiguous BF16 ``[M, K_local]`` lookup result.
        projection: Contiguous BF16 ``[K_local, hidden_size]`` packed weight.
        scale: Positive global averaging divisor, 17 for LongCat-2.0.
        solution: Optional registered implementation name.

    Returns:
        ``word_partial`` after the in-place matrix multiply-add.
    """
    num_rows, hidden_size = word_partial.shape
    _, projection_k = activation.shape
    if num_rows == 0:
        return word_partial

    signature = format_signature(
        word=dense_tensor_format(word_partial.dtype),
        activation=dense_tensor_format(activation.dtype),
        projection=dense_tensor_format(projection.dtype),
    )
    traits = {
        "projection_k": projection_k,
        "hidden_size": hidden_size,
    }
    kernel = select_kernel(
        "over_embedding",
        "project_add_word",
        signature,
        traits=traits,
        solution=solution,
    )
    shape_params = {
        "num_rows": num_rows,
        "projection_k": projection_k,
        "hidden_size": hidden_size,
        "scale": scale,
    }
    ShapeCapture.get().record(
        "over_embedding",
        "project_add_word",
        kernel.name,
        word_partial.dtype,
        shape_params,
    )
    with kernel_scope(
        "over_embedding",
        "project_add_word",
        word_partial.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            word_partial=word_partial,
            activation=activation,
            projection=projection,
            scale=scale,
        )
    return word_partial


__all__ = [
    "OverEmbeddingSpec",
    "TableFragmentSpec",
    "longcat_pro_tp8_spec",
    "append_packed_lookup_",
    "project_add_word_",
]


import tokenspeed_kernel.ops.over_embedding.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.over_embedding.torch  # noqa: E402,F401
