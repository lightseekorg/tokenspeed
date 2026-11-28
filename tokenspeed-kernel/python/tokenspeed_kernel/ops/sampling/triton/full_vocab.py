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

"""Ordered FP32 block folding and binary full-vocabulary row packing."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _full_vocab_score_bits(value, normalizer, DIRECT: tl.constexpr):
    if DIRECT:
        return (value - normalizer).to(tl.int32, bitcast=True)
    difference = normalizer - value
    bits = difference.to(tl.int32, bitcast=True)
    # Torch arithmetic keeps the subtraction's NaN rather than flipping its sign.
    sign = tl.where(difference == difference, -2147483648, 0)
    return bits ^ sign


@triton.jit(
    do_not_specialize=["shifted_stride", "column_stride", "ids_stride", "log_z_stride"],
    do_not_specialize_on_alignment=[
        "shifted_stride",
        "column_stride",
        "ids_stride",
        "log_z_stride",
    ],
)
def _full_vocab_head(
    shifted,
    ids,
    log_z,
    out,
    shifted_stride,
    column_stride,
    ids_stride,
    log_z_stride,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    DIRECT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, K_PAD)
    token = tl.load(ids + row * ids_stride + col, col < K, other=0).to(tl.int64)
    value = tl.load(
        shifted + row * shifted_stride + token * column_stride,
        col < K,
        other=0.0,
    )
    normalizer = tl.load(log_z + row * log_z_stride)
    head = _full_vocab_score_bits(value, normalizer, DIRECT).to(
        tl.float32, bitcast=True
    )
    tl.store(out + row * K + col, head, col < K)


# AMD retains the Torch leaf: fused subtraction changes NaN payload/sign bits.
@register_kernel(
    "sampling",
    "full_vocab_head",
    name="triton_full_vocab_head",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
)
def triton_full_vocab_head(shifted, ids, log_z, *, form):
    """Gather head scores and normalize in one launch, preserving FP32 bits.

    Args:
        shifted: FP32 [rows, vocab] maximum-shifted logits; arbitrary strides.
        ids: Int64 [rows, K] head IDs, unit column stride.
        log_z: FP32 [rows] shifted log denominators.
        form: Explicit direct or negative-cross-entropy subtraction form.

    Returns:
        Contiguous FP32 [rows, K] head logprobs.
    """
    if form not in ("direct", "negative-cross-entropy"):
        raise ValueError(f"unknown full-vocabulary logprob form: {form!r}")
    rows, k = ids.shape
    out = torch.empty((rows, k), device=shifted.device, dtype=torch.float32)
    if rows:
        _full_vocab_head[(rows,)](
            shifted,
            ids,
            log_z,
            out,
            shifted.stride(0),
            shifted.stride(1),
            ids.stride(0),
            log_z.stride(0),
            K=k,
            K_PAD=triton.next_power_of_2(k),
            DIRECT=form == "direct",
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit(
    do_not_specialize=["rows", "stride_row", "stride_block"],
    do_not_specialize_on_alignment=["rows", "stride_row", "stride_block"],
)
def _full_vocab_sum(
    values,
    out,
    rows,
    stride_row,
    stride_block,
    BLOCKS: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * ROW_TILE + tl.arange(0, ROW_TILE)
    base = values + row * stride_row
    total = tl.load(base, row < rows, other=0.0)
    # No parallel reduction, reassociation or logarithm: this is the original
    # FP32 left fold. Torch still supplies the original log implementation.
    for block in range(1, BLOCKS):
        total = total + tl.load(
            base + block * stride_block.to(tl.int64), row < rows, other=0.0
        )
    tl.store(out + row, total, row < rows)


@register_kernel(
    "sampling",
    "full_vocab_sum",
    name="triton_full_vocab_sum",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
)
def triton_full_vocab_sum(values: torch.Tensor) -> torch.Tensor:
    """Return the FP32 left fold of [rows, blocks], with arbitrary strides.

    Args:
        values: FP32 block partials [rows, blocks], at least one block.

    Returns:
        FP32 [rows] in exactly ((p0 + p1) + p2) order.
    """
    rows, blocks = values.shape
    out = torch.empty((rows,), device=values.device, dtype=torch.float32)
    if rows:
        _full_vocab_sum[(triton.cdiv(rows, 128),)](
            values,
            out,
            rows,
            values.stride(0),
            values.stride(1),
            BLOCKS=blocks,
            ROW_TILE=128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit(
    do_not_specialize=[
        "shifted_stride",
        "log_z_stride",
        "ids_stride",
        "head_stride",
        "tail_stride",
        "token_stride",
        "path_stride",
    ],
    do_not_specialize_on_alignment=[
        "shifted_stride",
        "log_z_stride",
        "ids_stride",
        "head_stride",
        "tail_stride",
        "token_stride",
        "path_stride",
    ],
)
def _full_vocab_pack(
    shifted,
    log_z,
    ids,
    head,
    tail,
    tokens,
    path,
    out,
    shifted_stride,
    log_z_stride,
    ids_stride,
    head_stride,
    tail_stride,
    token_stride,
    path_stride,
    VOCAB: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    HAS_PATH: tl.constexpr,
    HAS_HEAD: tl.constexpr,
    DIRECT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    src = tl.load(path + row * path_stride).to(tl.int64) if HAS_PATH else row
    token = tl.load(tokens + row * token_stride)
    target = tl.minimum(tl.maximum(token, 0), VOCAB - 1).to(tl.int64)
    selected = tl.load(shifted + src * shifted_stride + target)
    normalizer = tl.load(log_z + src * log_z_stride)
    sampled = _full_vocab_score_bits(selected, normalizer, DIRECT)
    dst = out + row * (3 + 2 * K)
    tl.store(dst, token.to(tl.int32))
    tl.store(dst + 1, tl.load(tail + src * tail_stride).to(tl.int32, bitcast=True))
    cols = tl.arange(0, K_PAD)
    head_ids = tl.load(ids + src * ids_stride + cols, cols < K, other=0)
    if HAS_HEAD:
        scores = tl.load(head + src * head_stride + cols, cols < K, other=0.0)
        head_bits = scores.to(tl.int32, bitcast=True)
    else:
        values = tl.load(
            shifted + src * shifted_stride + head_ids.to(tl.int64),
            cols < K,
            other=0.0,
        )
        head_bits = _full_vocab_score_bits(values, normalizer, DIRECT)
    tl.store(dst + 2 + cols, head_ids.to(tl.int32), cols < K)
    tl.store(dst + 2 + K + cols, head_bits, cols < K)
    tl.store(dst + 2 + 2 * K, sampled)


@register_kernel(
    "sampling",
    "full_vocab_pack",
    name="triton_full_vocab_pack",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
)
def triton_full_vocab_pack(
    shifted,
    log_z,
    ids,
    head,
    tail,
    tokens,
    *,
    prediction_rows,
    form,
):
    """Pack binary records and sampled-score bits in output-token order.

    Args:
        shifted: FP32 [rows, vocab] original logits minus each row's maximum.
        log_z: FP32 [rows] log denominators in shifted-logit space.
        ids: Int64 [rows, K] head vocabulary IDs, unit column stride.
        head: FP32 [rows, K] head scores, or None to gather and score while packing.
        tail: FP32 [rows] tail log masses.
        tokens: Int32 or int64 [rows] output IDs; padded IDs are retained.
        prediction_rows: Int64 [rows] prediction-row mapping, or None.
        form: "direct" or "negative-cross-entropy", preserving zero sign.

    Returns:
        Int32 [rows, 3 + 2*K]: token, tail bits, IDs, head bits, sampled bits.
    """
    rows, vocab = shifted.shape
    k = ids.shape[1]
    out = torch.empty((rows, 3 + 2 * k), device=shifted.device, dtype=torch.int32)
    if rows:
        _full_vocab_pack[(rows,)](
            shifted,
            log_z,
            ids,
            head,
            tail,
            tokens,
            prediction_rows,
            out,
            shifted.stride(0),
            log_z.stride(0),
            ids.stride(0),
            head.stride(0) if head is not None else 0,
            tail.stride(0),
            tokens.stride(0),
            prediction_rows.stride(0) if prediction_rows is not None else 0,
            VOCAB=vocab,
            K=k,
            K_PAD=triton.next_power_of_2(k),
            HAS_PATH=prediction_rows is not None,
            HAS_HEAD=head is not None,
            DIRECT=form == "direct",
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit(
    do_not_specialize=[
        "rows",
        "num_prefill_outputs",
        "num_extends",
        "decode_width",
        "padding_index",
        "length_stride",
        "pool_stride",
        "path_stride",
    ],
    do_not_specialize_on_alignment=[
        "rows",
        "num_prefill_outputs",
        "num_extends",
        "decode_width",
        "padding_index",
        "length_stride",
        "pool_stride",
        "path_stride",
    ],
)
def _full_vocab_active_rows(
    lengths,
    pools,
    path,
    active,
    rows,
    num_prefill_outputs,
    num_extends,
    decode_width,
    padding_index,
    length_stride,
    pool_stride,
    path_stride,
    HAS_PATH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    decode = row - num_prefill_outputs
    request = tl.where(
        row < num_prefill_outputs, row, num_extends + decode // decode_width
    )
    position = tl.where(row < num_prefill_outputs, 0, decode % decode_width)
    length = tl.load(lengths + request * length_stride, row < rows, other=0)
    pool = tl.load(pools + request * pool_stride, row < rows, other=padding_index)
    valid = (row < rows) & (position < length) & (pool != padding_index)
    source = (
        tl.load(path + row * path_stride, row < rows, other=0).to(tl.int64)
        if HAS_PATH
        else row
    )
    # Invalid tree slots can alias an accepted source: they must never store.
    tl.store(active + source, True, valid)


@register_kernel(
    "sampling",
    "full_vocab_active_rows",
    name="triton_full_vocab_active_rows",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
)
def triton_full_vocab_active_rows(
    accept_lengths,
    req_pool_indices,
    *,
    num_prefill_outputs,
    num_extends,
    decode_width,
    padding_index,
    prediction_rows,
):
    """Return a fixed-size GPU mask; never download the accepted row count."""
    rows = num_prefill_outputs + (accept_lengths.numel() - num_extends) * decode_width
    active = torch.zeros(rows, dtype=torch.bool, device=accept_lengths.device)
    if rows:
        _full_vocab_active_rows[(triton.cdiv(rows, 128),)](
            accept_lengths,
            req_pool_indices,
            prediction_rows,
            active,
            rows,
            num_prefill_outputs,
            num_extends,
            decode_width,
            padding_index,
            accept_lengths.stride(0),
            req_pool_indices.stride(0),
            0 if prediction_rows is None else prediction_rows.stride(0),
            HAS_PATH=prediction_rows is not None,
            BLOCK=128,
            num_warps=4,
        )
    return active


@triton.jit(
    do_not_specialize=["elements", "stride_row", "stride_column"],
    do_not_specialize_on_alignment=["elements", "stride_row", "stride_column"],
)
def _full_vocab_source_shift(
    logits,
    maximum,
    source,
    shifted,
    elements,
    stride_row,
    stride_column,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = offset // VOCAB, offset % VOCAB
    value = tl.load(
        logits + row * stride_row + column * stride_column, offset < elements, other=0.0
    )
    largest = tl.load(maximum + row, offset < elements, other=0.0)
    tl.store(source + offset, value, offset < elements)
    tl.store(shifted + offset, value - largest, offset < elements)


@register_kernel(
    "sampling",
    "full_vocab_source_shift",
    name="triton_full_vocab_source_shift",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("values", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
)
def triton_full_vocab_source_shift(logits, maximum):
    """Read each original value once to retain it and write its FP32 shift."""
    rows, vocab = logits.shape
    storage = torch.empty((2, rows, vocab), dtype=torch.float32, device=logits.device)
    source, shifted = storage[0], storage[1]
    if logits.numel():
        _full_vocab_source_shift[(triton.cdiv(logits.numel(), 1024),)](
            logits,
            maximum,
            source,
            shifted,
            logits.numel(),
            logits.stride(0),
            logits.stride(1),
            VOCAB=vocab,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return source, shifted
