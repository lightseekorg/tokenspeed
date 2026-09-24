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

"""Triton kernels for the PLE (predictive latent embedding) block.

Kernels backing the public API in :mod:`tokenspeed_kernel.ops.ple`. All of
them read the packed-free layout directly -- a flat ``[total_tokens, ...]``
tensor plus per-token ``(req, col)`` coordinates and per-request ``starts`` --
so none of the padded ``[bs, max_len, ...]`` glue the eager path builds is ever
materialized. Where a kernel spans the request boundary it addresses the
virtual sequence ``[carried state | tokens]`` and picks the source per tap:

- ``_ngram_ids_kernel``: per-token n-gram hash ids, optionally emitting the
  raw trailing window from the same loads.
- ``_ple_page_gather_kernel`` / ``_ple_page_scatter_kernel``: one cache row per
  page id, treating page id 0 as the null page.
- ``_ple_dilated_conv_kernel``: dilated depthwise conv + SiLU with an epilogue
  that folds up to two full-width addends and the verify-scratch windows.
- ``_ple_conv_state_kernel``: one launch combining convolution and request
  state writers (conv and final-state helpers are inlined).
- ``_ple_gate_norm_kernel``: three grouped Gemma RMSNorms plus the query-key
  gate in one launch.

Block configurations come from static heuristics in the launchers, so the
kernels stay CUDA-graph friendly.
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _ngram_ids_kernel(
    ids_ptr,
    init_ptr,
    req_ptr,
    col_ptr,
    lengths_ptr,
    starts_ptr,
    mult_ptr,
    sizes_ptr,
    offsets_ptr,
    reciprocals_ptr,
    out_ptr,
    tail_ptr,
    total,
    batch_size,
    tail_block_rows,
    eos_token,
    N: tl.constexpr,
    HPN: tl.constexpr,
    H: tl.constexpr,
    UNIFORM_LENGTH: tl.constexpr,
    WRITE_TAIL: tl.constexpr,
    SCATTER_TAIL: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    USE_RECIPROCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused per-token n-gram hash ids straight from the flat token stream.

    The window matrix is never materialized: each row walks left from its
    anchor through the virtual layout ``[carried context (N-1) | tokens]``,
    zeroes tokens behind an EOS boundary, folds the SplitMix multipliers via
    XOR and emits every head's ``mixed % prime + offset`` id. Products stay
    below 2**63 by construction of ``layer_multipliers``, so C-style ``%``
    matches ``torch.remainder``. With ``WRITE_TAIL`` the raw trailing window
    (``contexts[:, 1:]`` of the legacy layout, i.e. the verify-scratch rows)
    is emitted from the same loads.
    """

    pid = tl.program_id(0)
    lanes = pid * BLOCK + tl.arange(0, BLOCK)
    rows = lanes // H
    head = lanes % H
    owner = head == 0
    mask = rows < total
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    if UNIFORM_LENGTH:
        req = (rows // UNIFORM_LENGTH).to(tl.int64)
        col = (rows % UNIFORM_LENGTH).to(tl.int64)
        start = req * UNIFORM_LENGTH
        tl.store(req_ptr + rows, req, mask=mask & owner)
        tl.store(col_ptr + rows, col, mask=mask & owner)
        request_mask = (rows < batch_size) & owner
        tl.store(lengths_ptr + rows, UNIFORM_LENGTH, mask=request_mask)
        tl.store(starts_ptr + rows, rows * UNIFORM_LENGTH, mask=request_mask)
    else:
        req = tl.load(req_ptr + rows, mask=mask, other=0).to(tl.int64)
        col = tl.load(col_ptr + rows, mask=mask, other=0).to(tl.int64)
        start = tl.load(starts_ptr + req, mask=mask, other=0).to(tl.int64)

    tail_row = rows.to(tl.int64)
    if SCATTER_TAIL:
        tail_row = req * tail_block_rows + 1 + col
        carried_mask = (rows < batch_size) & owner
        for s in tl.static_range(N - 1):
            carried = tl.load(init_ptr + rows * (N - 1) + s, mask=carried_mask, other=0)
            tl.store(
                tail_ptr + (rows * tail_block_rows) * (N - 1) + s,
                carried,
                mask=carried_mask,
            )

    anchor = tl.load(ids_ptr + start + col, mask=mask, other=0).to(tl.int64)
    if WRITE_TAIL:
        tl.store(tail_ptr + tail_row * (N - 1) + (N - 2), anchor, mask=mask & owner)
    mixed = anchor * tl.load(mult_ptr)
    blocked = anchor != anchor
    for p in tl.static_range(1, N):
        v = col + (N - 1) - p
        from_init = v < (N - 1)
        raw_init = tl.load(
            init_ptr + req * (N - 1) + v, mask=mask & from_init, other=0
        ).to(tl.int64)
        raw_tok = tl.load(
            ids_ptr + tl.maximum(start + (v - (N - 1)), 0),
            mask=mask & (~from_init),
            other=0,
        ).to(tl.int64)
        raw = tl.where(from_init, raw_init, raw_tok)
        if WRITE_TAIL and p <= N - 2:
            tl.store(
                tail_ptr + tail_row * (N - 1) + (N - 2 - p), raw, mask=mask & owner
            )
        tok = tl.where(blocked, eos_token, raw)
        mixed = tl.where(
            p <= head // HPN + 1, mixed ^ (tok * tl.load(mult_ptr + p)), mixed
        )
        blocked = blocked | (tok == eos_token)
    size = tl.load(sizes_ptr + head)
    offset = tl.load(offsets_ptr + head)
    if USE_RECIPROCAL:
        remainder = _exact_remainder(mixed, size, tl.load(reciprocals_ptr + head))
    else:
        remainder = mixed % size
    tl.store(out_ptr + rows * H + head, remainder + offset, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _exact_remainder(value, divisor, reciprocal):
    # For 0 <= value < 2**63 and 1 < divisor < 2**63, floor(2**64/d)
    # underestimates the quotient by at most one. No floating-point math.
    n = value.to(tl.uint64)
    d = divisor.to(tl.uint64)
    q = tl.umulhi(n, reciprocal.to(tl.uint64))
    r = n - q * d
    return tl.where(d == 1, 0, tl.where(r >= d, r - d, r)).to(tl.int64)


@triton.jit
def _ple_page_gather_kernel(
    field_ptr,
    page_ptr,
    out_ptr,
    default,
    field_row_stride,
    N,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Read one cache row per page id, substituting ``default`` for null pages.

    Page id 0 is the null page, so its row is replaced rather than read. The
    masked load supplies the fill, which keeps the replacement inside the one
    pass instead of allocating a full-size ``default`` block and selecting
    against it. ``field_row_stride`` carries the plan's page stride, which the
    arena is free to pad past the row's own extent.
    """

    row = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    page = tl.load(page_ptr + row).to(tl.int64)
    field_dtype = field_ptr.dtype.element_ty
    value = tl.load(
        field_ptr + tl.maximum(page, 0) * field_row_stride + off,
        mask=mask & (page > 0),
        other=default.to(field_dtype),
    )
    tl.store(out_ptr + row * N + off, value, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ple_page_gather_pair_kernel(
    context_ptr,
    conv_ptr,
    page_ptr,
    context_out_ptr,
    conv_out_ptr,
    context_default,
    context_stride,
    conv_stride,
    CONTEXT_N: tl.constexpr,
    CONV_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    page = tl.load(page_ptr + row).to(tl.int64)
    if tile == 0:
        context_dtype = context_ptr.dtype.element_ty
        context = tl.load(
            context_ptr + tl.maximum(page, 0) * context_stride + off,
            mask=(off < CONTEXT_N) & (page > 0),
            other=context_default.to(context_dtype),
        )
        tl.store(
            context_out_ptr + row * CONTEXT_N + off,
            context,
            mask=off < CONTEXT_N,
        )
    conv_dtype = conv_ptr.dtype.element_ty
    conv = tl.load(
        conv_ptr + tl.maximum(page, 0) * conv_stride + off,
        mask=(off < CONV_N) & (page > 0),
        other=0.0,
    )
    tl.store(
        conv_out_ptr + row * CONV_N + off,
        conv.to(conv_dtype),
        mask=off < CONV_N,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ple_page_scatter_kernel(
    field_ptr,
    page_ptr,
    values_ptr,
    field_row_stride,
    N,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write one row per page id, leaving null pages alone.

    Rows bound for page id 0 are simply not stored, so no placeholder row has
    to be built and copied over itself.
    """

    row = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    page = tl.load(page_ptr + row).to(tl.int64)
    value = tl.load(values_ptr + row * N + off, mask=mask, other=0)
    tl.store(
        field_ptr + tl.maximum(page, 0) * field_row_stride + off,
        value.to(field_ptr.dtype.element_ty),
        mask=mask & (page > 0),
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ple_host_gather_kernel(
    table_address,
    ids_ptr,
    scale_value,
    scale_ptr,
    out_ptr,
    head_dim,
    vocab_start,
    vocab_end,
    IS_FP8: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ROW_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather one n-gram row per program straight out of pinned host memory.

    ``table_address`` is the host allocation's base address, not a device
    tensor: under unified addressing a page-locked host pointer is a valid
    device address, so the load below is a PCIe/C2C read issued by the GPU
    itself. Rows outside this rank's shard emit zeros, matching the masked
    lookup that :class:`VocabParallelEmbedding` performs before its all-reduce.

    The dequant scale mirrors how the table was quantized, independent of
    offloading. An offline FP8 checkpoint publishes one whole-table factor, so
    ``ROW_SCALE`` is false and the scalar ``scale_value`` is used. A
    compute-dtype checkpoint is quantized online, one row at a time, so
    ``ROW_SCALE`` is true and each row's factor is read from ``scale_ptr`` at
    its local id (folded row-0 for shard-masked rows is harmless: the output
    is zeroed anyway). Either factor commutes with the all-reduce, so applying
    it here -- before the reduce -- stays correct under tensor parallelism.
    """

    row = tl.program_id(0)
    global_id = tl.load(ids_ptr + row)
    in_range = (global_id >= vocab_start) & (global_id < vocab_end)
    local_id = tl.where(in_range, global_id - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < head_dim
    out_dtype = out_ptr.dtype.element_ty
    if IS_FP8:
        table = table_address.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        table = table_address.to(tl.int64).to(tl.pointer_type(out_dtype))
    values = tl.load(
        table + local_id * head_dim + offsets, mask=mask & in_range, other=0.0
    ).to(tl.float32)
    if HAS_SCALE:
        if ROW_SCALE:
            values = values * tl.load(scale_ptr + local_id, mask=in_range, other=0.0)
        else:
            values = values * scale_value
    tl.store(
        out_ptr + row * head_dim + offsets,
        tl.where(in_range, values, 0.0).to(out_dtype),
        mask=mask,
    )


@triton.jit
def _ple_dilated_conv_kernel(
    values_ptr,
    initial_ptr,
    weights,
    req_ptr,
    col_ptr,
    starts_ptr,
    out_ptr,
    windows_ptr,
    gated_ptr,
    residual_ptr,
    windows_block_rows,
    gated_row_stride,
    residual_row_stride,
    C,
    D: tl.constexpr,
    K: tl.constexpr,
    STATE: tl.constexpr,
    WRITE_WINDOWS: tl.constexpr,
    SCATTER_WINDOWS: tl.constexpr,
    ADD_GATED: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused dilated depthwise conv + SiLU over the packed-free layout.

    Inlined by _ple_conv_state_kernel, which owns PDL synchronization.

    Each program covers one token and a channel block. The virtual per-request
    sequence is ``[carried state (STATE cols) | tokens]``; tap ``k`` of output
    column ``c`` reads virtual position ``c + k*D`` and sources it from the
    carried state or the flat token tensor directly, so the batched pack /
    transpose / unfold glue disappears. When ``WRITE_WINDOWS`` is set the
    per-token sliding state windows (verify scratch input) are emitted from
    the same loads; ``SCATTER_WINDOWS`` then places each one at its verify
    scratch row instead of a packed one-row-per-token buffer, so the rollback
    rows are filled in place and the full-size window tensor never exists.

    ``ADD_GATED`` / ``ADD_RESIDUAL`` fold the two full-width additions that
    follow the conv (the gated value stream and the incoming hidden states)
    into this epilogue, so neither the bare conv output nor the PLE delta ever
    reaches memory. Each addend is rounded to the store dtype before the next
    one is applied, which reproduces the separate ``gated + conv_output`` and
    ``hidden_states + delta`` tensor adds bit-for-bit whenever that dtype is
    narrower than the fp32 accumulator. A pure fp32 stream has no such
    rounding barrier, so the SiLU product stays unrounded into the first add
    and results may differ by one ulp at operand scale (in the more accurate
    direction).
    """

    token = tl.program_id(0)
    block = tl.program_id(1)
    ch = block * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = ch < C
    req = tl.load(req_ptr + token).to(tl.int64)
    col = tl.load(col_ptr + token).to(tl.int64)
    start = tl.load(starts_ptr + req).to(tl.int64)
    out_dtype = out_ptr.dtype.element_ty

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for k in tl.static_range(K):
        v = col + k * D
        from_state = v < STATE
        tok_idx = tl.maximum(start + (v - STATE), 0)
        x_state = tl.load(
            initial_ptr + (req * C + ch) * STATE + v,
            mask=cmask & from_state,
            other=0.0,
        ).to(tl.float32)
        x_tok = tl.load(
            values_ptr + tok_idx * C + ch,
            mask=cmask & (~from_state),
            other=0.0,
        ).to(tl.float32)
        x = tl.where(from_state, x_state, x_tok)
        w = weights[k]
        acc += w * x
    silu = acc * (1.0 / (1.0 + tl.exp(-acc)))
    result = silu.to(out_dtype)
    if ADD_GATED:
        gated = tl.load(
            gated_ptr + token * gated_row_stride + ch, mask=cmask, other=0.0
        )
        result = (result.to(tl.float32) + gated.to(tl.float32)).to(out_dtype)
    if ADD_RESIDUAL:
        residual = tl.load(
            residual_ptr + token * residual_row_stride + ch, mask=cmask, other=0.0
        )
        result = (result.to(tl.float32) + residual.to(tl.float32)).to(out_dtype)
    tl.store(out_ptr + token * C + ch, result, mask=cmask)

    if WRITE_WINDOWS:
        # The verify scratch holds one ``windows_block_rows`` row block per
        # request whose first row is the carried state, so token ``col`` of
        # request ``req`` owns row ``req * windows_block_rows + 1 + col``. The
        # packed layout is one row per token instead.
        wrow = token.to(tl.int64)
        if SCATTER_WINDOWS:
            wrow = req * windows_block_rows + 1 + col
        win_dtype = windows_ptr.dtype.element_ty
        for s in tl.static_range(STATE):
            v = col + 1 + s
            from_state = v < STATE
            tok_idx = tl.maximum(start + (v - STATE), 0)
            w_state = tl.load(
                initial_ptr + (req * C + ch) * STATE + v,
                mask=cmask & from_state,
                other=0.0,
            ).to(win_dtype)
            w_tok = tl.load(
                values_ptr + tok_idx * C + ch,
                mask=cmask & (~from_state),
                other=0.0,
            ).to(win_dtype)
            tl.store(
                windows_ptr + (wrow * C + ch) * STATE + s,
                tl.where(from_state, w_state, w_tok),
                mask=cmask,
            )


@triton.jit
def _ple_conv_final_kernel(
    values_ptr,
    initial_ptr,
    lengths_ptr,
    starts_ptr,
    final_ptr,
    windows_ptr,
    windows_block_rows,
    C,
    STATE: tl.constexpr,
    WRITE_CARRIED: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Trailing conv window per request (the state carried to the next step).

    Inlined by _ple_conv_state_kernel, which owns PDL synchronization.

    Reads virtual positions ``length .. length + STATE - 1``; zero-length
    requests naturally pass their carried state through unchanged.
    """

    req = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    ch = block * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = ch < C
    if WRITE_FINAL:
        length = tl.load(lengths_ptr + req).to(tl.int64)
        start = tl.load(starts_ptr + req).to(tl.int64)
    out_dtype = final_ptr.dtype.element_ty
    for s in tl.static_range(STATE if WRITE_FINAL else 0):
        v = length + s
        from_state = v < STATE
        tok_idx = tl.maximum(start + (v - STATE), 0)
        x_state = tl.load(
            initial_ptr + (req * C + ch) * STATE + v,
            mask=cmask & from_state,
            other=0.0,
        ).to(out_dtype)
        x_tok = tl.load(
            values_ptr + tok_idx * C + ch,
            mask=cmask & (~from_state),
            other=0.0,
        ).to(out_dtype)
        tl.store(
            final_ptr + (req * C + ch) * STATE + s,
            tl.where(from_state, x_state, x_tok),
            mask=cmask,
        )
    if WRITE_CARRIED:
        for s in tl.static_range(STATE):
            carried = tl.load(
                initial_ptr + (req * C + ch) * STATE + s,
                mask=cmask,
                other=0.0,
            )
            tl.store(
                windows_ptr + ((req * windows_block_rows) * C + ch) * STATE + s,
                carried,
                mask=cmask,
            )


@triton.jit
def _ple_conv_state_kernel(
    values_ptr,
    initial_ptr,
    weight_ptr,
    req_ptr,
    col_ptr,
    lengths_ptr,
    starts_ptr,
    out_ptr,
    final_ptr,
    windows_ptr,
    gated_ptr,
    residual_ptr,
    windows_block_rows,
    gated_row_stride,
    residual_row_stride,
    C,
    TOTAL,
    BATCH,
    D: tl.constexpr,
    K: tl.constexpr,
    STATE: tl.constexpr,
    WRITE_WINDOWS: tl.constexpr,
    SCATTER_WINDOWS: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
    ADD_GATED: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    WEIGHTS_INDEPENDENT: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One launch for token outputs and request states, including empty requests.

    Program i owns token i and request i independently. Neither state writer
    consumes conv outputs: both read the same immutable normalized input.
    """
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    if ENABLE_PDL and not WEIGHTS_INDEPENDENT:
        tl.extra.cuda.gdc_wait()
    weights = ()
    for k in tl.static_range(K):
        weights += (tl.load(weight_ptr + ch * K + k, ch < C, other=0.0).to(tl.float32),)
    if ENABLE_PDL:
        if WEIGHTS_INDEPENDENT:
            tl.extra.cuda.gdc_wait()
        # Successors may set up, but must wait before reading any output.
        tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(0) < TOTAL:
        _ple_dilated_conv_kernel(
            values_ptr,
            initial_ptr,
            weights,
            req_ptr,
            col_ptr,
            starts_ptr,
            out_ptr,
            windows_ptr,
            gated_ptr,
            residual_ptr,
            windows_block_rows,
            gated_row_stride,
            residual_row_stride,
            C,
            D,
            K,
            STATE,
            WRITE_WINDOWS,
            SCATTER_WINDOWS,
            ADD_GATED,
            ADD_RESIDUAL,
            BLOCK_C,
        )
    if WRITE_FINAL or SCATTER_WINDOWS:
        if tl.program_id(0) < BATCH:
            _ple_conv_final_kernel(
                values_ptr,
                initial_ptr,
                lengths_ptr,
                starts_ptr,
                final_ptr,
                windows_ptr,
                windows_block_rows,
                C,
                STATE,
                SCATTER_WINDOWS,
                WRITE_FINAL,
                BLOCK_C,
            )


@triton.jit
def _ple_gate_norm_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    key_gw_ptr,
    query_gw_ptr,
    conv_gw_ptr,
    gated_ptr,
    normalized_ptr,
    eps,
    inv_sqrt_d,
    key_stride,
    query_stride,
    value_stride,
    HC: tl.constexpr,
    D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused PLE gating: three grouped Gemma RMSNorms plus the query-key
    gate collapse into one launch.

    Everything between the key/value projections and the short conv is
    row-local per ``(token, hc branch)``: normalize the key and query
    slices, dot them into a signed-sqrt sigmoid gate, scale the shared
    value row, then re-normalize for the conv input. Norm math runs in
    fp32 with a store-dtype round-trip after each norm so the fused
    output bit-matches the unfused module chain.
    """

    token = tl.program_id(0)
    branch = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    row = token * (HC * D) + branch * D
    out_dtype = gated_ptr.dtype.element_ty

    # Static gamma buffers load before the PDL wait: they are not written by
    # the preceding kernel, so this prologue overlaps its tail.
    key_gw = tl.load(key_gw_ptr + branch * D + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    query_gw = tl.load(query_gw_ptr + branch * D + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    conv_gw = tl.load(conv_gw_ptr + branch * D + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    key = tl.load(
        key_ptr + token * key_stride + branch * D + offs, mask=mask, other=0.0
    ).to(tl.float32)
    query = tl.load(
        query_ptr + token * query_stride + branch * D + offs, mask=mask, other=0.0
    ).to(tl.float32)
    value = tl.load(value_ptr + token * value_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    key_norm = key * tl.rsqrt(tl.sum(key * key, 0) / D + eps) * key_gw
    query_norm = query * tl.rsqrt(tl.sum(query * query, 0) / D + eps) * query_gw
    key_norm = key_norm.to(out_dtype).to(tl.float32)
    query_norm = query_norm.to(out_dtype).to(tl.float32)

    gate = tl.sum(key_norm * query_norm, 0) * inv_sqrt_d
    magnitude = tl.sqrt(tl.maximum(tl.abs(gate), 1e-6))
    gate = tl.where(gate > 0, magnitude, tl.where(gate < 0, -magnitude, 0.0))
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))

    gated = (sigmoid * value).to(out_dtype)
    tl.store(gated_ptr + row + offs, gated, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    gated_f = gated.to(tl.float32)
    normalized = gated_f * tl.rsqrt(tl.sum(gated_f * gated_f, 0) / D + eps) * conv_gw
    tl.store(
        normalized_ptr + row + offs,
        normalized.to(normalized_ptr.dtype.element_ty),
        mask=mask,
    )
