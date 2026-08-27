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

"""Predictive latent embedding (PLE) layers for Qwen4-Exp."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn

from tokenspeed.runtime.configs.qwen4_exp_config import Qwen4ExpTextConfig
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.backends.qwen4_exp import (
    qwen4_exp_linear_backend,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_PLE_CONTEXT_FIELD,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.hyperconnection import GroupedGemmaRMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from tokenspeed.runtime.utils import add_prefix


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    limit = math.isqrt(value)
    while divisor <= limit:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _nth_prime_after(start: int, count: int) -> int:
    candidate = max(1, int(start))
    found = 0
    while found < count:
        candidate += 1
        if _is_prime(candidate):
            found += 1
    return candidate


_PLE_FP8_MAX = 448.0  # torch.float8_e4m3fn finite maximum


def quantize_ple_embedding_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online per-row FP8 quantization for the n-gram table.

    Checkpoint shards stream in row ranges, so each row is quantized
    independently (``scale = amax / 448``) -- no whole-table amax prescan is
    needed and no clipping can occur. Returns the FP8 rows and their fp32
    dequant scales.
    """

    values = rows.to(torch.float32)
    scale = (values.abs().amax(dim=1) / _PLE_FP8_MAX).clamp_min(1e-12)
    quantized = (values / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return quantized, scale


@triton.jit
def _ngram_ids_kernel(
    ids_ptr,
    init_ptr,
    req_ptr,
    col_ptr,
    starts_ptr,
    mult_ptr,
    sizes_ptr,
    offsets_ptr,
    out_ptr,
    tail_ptr,
    total,
    eos_token,
    N: tl.constexpr,
    HPN: tl.constexpr,
    H: tl.constexpr,
    WRITE_TAIL: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
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
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    mask = rows < total
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    req = tl.load(req_ptr + rows, mask=mask, other=0).to(tl.int64)
    col = tl.load(col_ptr + rows, mask=mask, other=0).to(tl.int64)
    start = tl.load(starts_ptr + req, mask=mask, other=0).to(tl.int64)

    anchor = tl.load(ids_ptr + start + col, mask=mask, other=0).to(tl.int64)
    if WRITE_TAIL:
        tl.store(tail_ptr + rows * (N - 1) + (N - 2), anchor, mask=mask)
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
            tl.store(tail_ptr + rows * (N - 1) + (N - 2 - p), raw, mask=mask)
        tok = tl.where(blocked, eos_token, raw)
        mixed = mixed ^ (tok * tl.load(mult_ptr + p))
        blocked = blocked | (tok == eos_token)
        for h in tl.static_range(0, HPN):
            head = (p - 1) * HPN + h
            size = tl.load(sizes_ptr + head)
            offset = tl.load(offsets_ptr + head)
            tl.store(out_ptr + rows * H + head, mixed % size + offset, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


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
def _ple_dilated_conv_kernel(
    values_ptr,
    initial_ptr,
    weight_ptr,
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
    ENABLE_PDL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused dilated depthwise conv + SiLU over the packed-free layout.

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
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
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
        w = tl.load(weight_ptr + ch * K + k, mask=cmask, other=0.0).to(tl.float32)
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
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ple_conv_final_kernel(
    values_ptr,
    initial_ptr,
    lengths_ptr,
    starts_ptr,
    final_ptr,
    C,
    STATE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Trailing conv window per request (the state carried to the next step).

    Reads virtual positions ``length .. length + STATE - 1``; zero-length
    requests naturally pass their carried state through unchanged.
    """

    req = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    ch = block * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = ch < C
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    length = tl.load(lengths_ptr + req).to(tl.int64)
    start = tl.load(starts_ptr + req).to(tl.int64)
    out_dtype = final_ptr.dtype.element_ty
    for s in tl.static_range(STATE):
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
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


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

    gated_f = gated.to(tl.float32)
    normalized = gated_f * tl.rsqrt(tl.sum(gated_f * gated_f, 0) / D + eps) * conv_gw
    tl.store(
        normalized_ptr + row + offs,
        normalized.to(normalized_ptr.dtype.element_ty),
        mask=mask,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashed, independently-sharded n-gram embedding used by PLE."""

    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PRIME_1 = 10007

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        mapping: Mapping,
        embedding_dim: int,
        ple_layer_index: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        self.eos_token_id = int(config.eos_token_id)
        if self.ngram_size < 2:
            raise ValueError("Qwen4-Exp ngram_size must be at least 2")
        if self.heads_per_ngram <= 0:
            raise ValueError("Qwen4-Exp heads_per_ngram must be positive")
        if self.embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by "
                "(ngram_size - 1) * heads_per_ngram"
            )

        self.head_dim = self.embedding_dim // self.ngram_heads
        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(
                self.ngram_size, int(getattr(config, "seed", 1234))
            ),
            persistent=True,
        )
        sizes = [
            _nth_prime_after(
                int(config.ngram_vocab_size_base) - 1,
                self.ple_layer_index * self.ngram_heads + index + 1,
            )
            for index in range(self.ngram_heads)
        ]
        offsets = []
        total = 0
        for size in sizes:
            offsets.append(total)
            total += size
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisible_by = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab = (total + divisible_by - 1) // divisible_by * divisible_by
        ple_embed_dtype = getattr(config, "ple_embed_dtype", None)
        if ple_embed_dtype not in (None, "float8_e4m3fn"):
            raise ValueError(
                "Qwen4-Exp ple_embed_dtype supports only 'float8_e4m3fn', "
                f"got {ple_embed_dtype!r}"
            )
        self.embed_store_dtype = torch.float8_e4m3fn if ple_embed_dtype else None
        # Lookups are dequantized back to the model compute dtype (layer
        # construction runs under the model's default dtype).
        self.embed_output_dtype = torch.get_default_dtype()
        # Source checkpoints may already store the table in FP8 and publish
        # one dequant scale beside all of its split shards. The loader updates
        # this value when that tensor arrives; keeping it as Python state lets
        # CPU-side shard copies apply the scale without a device sync.
        self._checkpoint_weight_scale = 1.0
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab,
            self.head_dim,
            org_num_embeddings=padded_vocab,
            params_dtype=self.embed_store_dtype,
            prefix=add_prefix("ngram_embedding", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        if self.embed_store_dtype is not None:
            # Per-local-row dequant scales, written by the loader's online
            # quantization. Ones (not zeros / empty): rows gathered before the
            # checkpoint lands, or shard-masked rows folded to local row 0,
            # must stay finite. Non-persistent: derived from the bf16
            # checkpoint, never round-tripped.
            self.register_buffer(
                "ngram_embedding_scale",
                torch.ones(self.ngram_embedding.num_embeddings_per_partition),
                persistent=False,
            )

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    def _build_layer_multipliers(self, size: int, seed: int) -> torch.Tensor:
        max_long = (1 << 63) - 1
        half_bound = max(1, max_long // max(self.unigram_vocab_size, 1) // 2)
        base_seed = seed + self._PRIME_1 * self.ple_layer_index
        values = []
        for index in range(size):
            value = (base_seed + self._SPLITMIX_GAMMA * (index + 1)) & self._MASK64
            values.append(2 * (self._splitmix64(value) % half_bound) + 1)
        return torch.tensor(values, dtype=torch.long)

    def _ngram_ids_torch(self, contexts: torch.Tensor) -> torch.Tensor:
        """Anchor-only n-gram hash ids (CPU / fallback path).

        The legacy implementation shifted and hashed every window column via
        ``_shift_right_ignore_eos`` even though only the anchor (last) column
        was consumed. Walking left from the anchor with a running EOS-boundary
        flag produces the identical ids in a handful of elementwise ops.
        """

        eos = self.eos_token_id
        anchor = contexts[:, -1]
        mixed = anchor * self.layer_multipliers[0]
        blocked = torch.zeros_like(anchor, dtype=torch.bool)
        blocks = []
        for position in range(1, self.ngram_size):
            tok = contexts[:, self.ngram_size - 1 - position]
            tok = torch.where(blocked, torch.full_like(tok, eos), tok)
            mixed = torch.bitwise_xor(mixed, tok * self.layer_multipliers[position])
            blocked = blocked | (tok == eos)
            head_start = (position - 1) * self.heads_per_ngram
            head_end = head_start + self.heads_per_ngram
            sizes = self.ngram_heads_vocab_sizes[head_start:head_end]
            offsets = self.ngram_heads_offsets[head_start:head_end]
            blocks.append(
                torch.remainder(mixed.unsqueeze(-1), sizes.view(1, -1))
                + offsets.view(1, -1)
            )
        return torch.cat(blocks, dim=-1)

    def _ngram_ids_flat_cuda(
        self,
        input_ids: torch.Tensor,
        initial: torch.Tensor,
        req: torch.Tensor,
        col: torch.Tensor,
        starts: torch.Tensor,
        need_tail: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        total = input_ids.shape[0]
        device = input_ids.device
        ids = torch.empty((total, self.ngram_heads), dtype=torch.long, device=device)
        tail = (
            torch.empty((total, self.ngram_size - 1), dtype=torch.long, device=device)
            if need_tail
            else None
        )
        if total == 0:
            return ids, tail
        block = 256
        use_pdl = pdl_enabled()
        _ngram_ids_kernel[(triton.cdiv(total, block),)](
            input_ids.contiguous(),
            initial.contiguous(),
            req,
            col,
            starts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            ids,
            tail if need_tail else ids,
            total,
            self.eos_token_id,
            N=self.ngram_size,
            HPN=self.heads_per_ngram,
            H=self.ngram_heads,
            WRITE_TAIL=need_tail,
            ENABLE_PDL=use_pdl,
            BLOCK=block,
            **({"launch_pdl": True} if use_pdl else {}),
        )
        return ids, tail

    def _dequant(self, raw: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Cast the FP8 lookup to compute dtype and apply per-row scales.

        Must run before the TP all-reduce: FP8 payloads cannot be reduced and
        each row's scale lives only on its owning rank. Shard-masked rows were
        zero-filled by the embedding forward, so gathering their folded row-0
        scale is harmless (0 * finite = 0).
        """

        module = self.ngram_embedding
        if module.tp_size > 1:
            local_ids, _ = get_masked_input_and_mask(
                ids,
                module.shard_indices.org_vocab_start_index,
                module.shard_indices.org_vocab_end_index,
                module.shard_indices.num_org_vocab_padding,
                module.shard_indices.added_vocab_start_index,
                module.shard_indices.added_vocab_end_index,
            )
        else:
            local_ids = ids.clamp(min=0, max=module.num_embeddings_padded - 1)
        scale = self.ngram_embedding_scale[local_ids]
        return (raw.to(torch.float32) * scale.unsqueeze(-1)).to(self.embed_output_dtype)

    def forward_flat(
        self,
        input_ids: torch.Tensor,
        initial: torch.Tensor,
        req: torch.Tensor,
        col: torch.Tensor,
        starts: torch.Tensor,
        need_tail: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """CUDA path: embed n-grams without materializing the window matrix."""

        ids, tail = self._ngram_ids_flat_cuda(
            input_ids, initial, req, col, starts, need_tail
        )
        # Reduce the flattened [tokens, heads * head_dim] view instead of the
        # 3D lookup: the lamport backend folds trailing dims into token count
        # ([T * heads, head_dim]), which blows past the mnnvl token cap and
        # demotes this all-reduce to the IPC/NCCL path. flatten(-2) is a
        # metadata-only view and was applied to the output anyway. FP8 tables
        # dequantize here, before the reduce.
        embeddings = self.ngram_embedding(ids, reduce_results=False)
        if self.embed_store_dtype is not None:
            embeddings = self._dequant(embeddings, ids)
        embeddings = embeddings.flatten(-2)
        if self.ngram_embedding.tp_size > 1:
            embeddings = all_reduce(embeddings, self.ngram_embedding.tp_group)
        return embeddings, tail

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Embed contexts shaped ``[tokens, ngram_size]`` (fallback path)."""

        contexts = contexts.to(torch.long)
        ids = self._ngram_ids_torch(contexts)
        embeddings = self.ngram_embedding(ids, reduce_results=False)
        if self.embed_store_dtype is not None:
            embeddings = self._dequant(embeddings, ids)
        if self.ngram_embedding.tp_size > 1:
            embeddings = all_reduce(embeddings, self.ngram_embedding.tp_group)
        return embeddings.flatten(-2)


class Qwen4ExpPLELayer(nn.Module):
    """PLE gating plus dilated depthwise short convolution.

    Persistent context and convolution windows live in the model's unified
    paged cache under :data:`QWEN4_EXP_PLE_CACHE_GROUP`.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        mapping: Mapping,
        layer_id: int,
        ple_layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ngram_size = int(config.ngram_size)
        self.context_len = self.ngram_size - 1
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.ngram_size
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            mapping,
            int(config.ple_embed_dim),
            ple_layer_index,
            add_prefix("ple_embedding", prefix),
        )
        # key_proj and value_proj consume the same embeddings; fuse them into
        # a single GEMM. Checkpoint shards are routed here by the stacked
        # mapping in load_qwen4_exp_weights: "key" fills rows
        # [0, hc_hidden) and "value" rows [hc_hidden, hc_hidden + hidden).
        # Quantization of the fused name follows the member projections via
        # the fused-module table in should_exclude_quant_module.
        self.kv_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size + self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_proj", prefix),
        )
        # Unquantized ReplicatedLinear hands back a plain Parameter whose
        # weight_loader is a writable attribute, while a quantized one returns a
        # BaseWeightParameter exposing it as a read-only property. Write to
        # whichever storage exists; picking the wrong one either raises or
        # silently drops the shard routing.
        weight = self.kv_proj.weight
        if hasattr(weight, "_weight_loader"):
            weight._weight_loader = self._load_kv_proj_shard
        else:
            weight.weight_loader = self._load_kv_proj_shard
        self.norm_key = GroupedGemmaRMSNorm(
            self.hc_hidden_size,
            config.rms_norm_eps,
            group_size=self.hidden_size,
        )
        self.norm_query = GroupedGemmaRMSNorm(
            self.hc_hidden_size,
            config.rms_norm_eps,
            group_size=self.hidden_size,
        )
        self.norm_conv = GroupedGemmaRMSNorm(
            self.hc_hidden_size,
            config.rms_norm_eps,
            group_size=self.hidden_size,
        )
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            dilation=self.ngram_size,
            groups=self.hc_hidden_size,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)
        self._verify_scratch: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._active_verify_key: tuple[int, int] | None = None

    def _load_kv_proj_shard(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> None:
        offsets = {
            "key": (0, self.hc_hidden_size),
            "value": (self.hc_hidden_size, self.hidden_size),
        }
        start, size = offsets[shard_id]
        if loaded_weight.shape[0] != size or param.shape[0] != (
            self.hc_hidden_size + self.hidden_size
        ):
            raise ValueError(
                f"kv_proj shard {shard_id} shape mismatch: param "
                f"{tuple(param.shape)}, loaded {tuple(loaded_weight.shape)}"
            )
        param.data[start : start + size].copy_(
            loaded_weight.to(param.device, param.dtype)
        )

    @staticmethod
    def _linear_backend(ctx: ForwardContext):
        return qwen4_exp_linear_backend(ctx.attn_backend)

    @staticmethod
    def _metadata(linear_backend):
        metadata = getattr(linear_backend, "forward_metadata", None)
        if metadata is None:
            raise RuntimeError("Qwen4-Exp PLE requires hybrid state metadata")
        return metadata

    @staticmethod
    def _page_row_stride(field: torch.Tensor) -> int:
        """Elements between page rows of a cache field.

        The plan is free to pad a page past the extent of the row it holds, so
        the stride cannot be derived from the shape. Everything inside a row is
        dense, which is what lets the page kernels address it flatly.
        """

        if not field[0].is_contiguous():
            raise RuntimeError("Qwen4-Exp PLE cache rows must be dense")
        return field.stride(0)

    @staticmethod
    def _read_pages(
        field: torch.Tensor,
        page_ids: torch.Tensor,
        default: int | float = 0,
    ) -> torch.Tensor:
        """One cache row per page id, with null pages read as ``default``."""

        rows = page_ids.shape[0]
        if not field.is_cuda:
            page_ids = page_ids.to(torch.long)
            valid = page_ids > 0
            values = field.index_select(0, page_ids.clamp_min(0))
            mask = valid.view(-1, *([1] * (values.ndim - 1)))
            return torch.where(mask, values, torch.full_like(values, default))
        out = field.new_empty((rows, *field.shape[1:]))
        if rows == 0:
            return out
        numel = out[0].numel()
        block = min(1024, triton.next_power_of_2(numel))
        use_pdl = pdl_enabled()
        _ple_page_gather_kernel[(rows, triton.cdiv(numel, block))](
            field,
            page_ids,
            out,
            default,
            Qwen4ExpPLELayer._page_row_stride(field),
            numel,
            ENABLE_PDL=use_pdl,
            BLOCK=block,
            **({"launch_pdl": True} if use_pdl else {}),
        )
        return out

    @staticmethod
    def _write_pages(
        field: torch.Tensor,
        page_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Store one row per page id, skipping null pages."""

        rows = page_ids.shape[0]
        if not field.is_cuda:
            valid = page_ids > 0
            safe_pages = page_ids.to(torch.long).clamp_min(0)
            stored_values = torch.where(
                valid.view(-1, *([1] * (values.ndim - 1))),
                values.to(field.dtype),
                field[0].unsqueeze(0),
            )
            field.index_copy_(
                0,
                safe_pages,
                stored_values,
            )
            return
        if rows == 0:
            return
        numel = values[0].numel()
        block = min(1024, triton.next_power_of_2(numel))
        use_pdl = pdl_enabled()
        _ple_page_scatter_kernel[(rows, triton.cdiv(numel, block))](
            field,
            page_ids,
            values.contiguous(),
            Qwen4ExpPLELayer._page_row_stride(field),
            numel,
            ENABLE_PDL=use_pdl,
            BLOCK=block,
            **({"launch_pdl": True} if use_pdl else {}),
        )

    def _lengths(self, metadata, total_tokens: int, bs: int) -> list[int]:
        """Per-request token counts for a flat batch of ``total_tokens`` rows.

        ``total_tokens`` is an upper bound, not an identity: a padded-bucket
        graph replay hands the layer bucket rows whose tail is filler, so the
        CPU lengths only have to fit inside them. They stay the single source
        of truth for what is real -- the caller slices to their sum.
        """
        cpu_lengths = metadata.extend_seq_lens_cpu
        if cpu_lengths is not None and cpu_lengths.numel() >= bs:
            lengths = [int(value) for value in cpu_lengths[:bs].tolist()]
            if sum(lengths) <= total_tokens:
                return lengths
        if bs == 0:
            return []
        if total_tokens % bs:
            raise RuntimeError(
                "Qwen4-Exp PLE cannot infer per-request token lengths from "
                f"{total_tokens} tokens and batch size {bs}"
            )
        return [total_tokens // bs] * bs

    @staticmethod
    def _batch_indices(
        lengths: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Vectorized ``(request, column)`` index bundle for a flat batch.

        Returns ``(req_indices, col_indices, lengths_t, starts, max_len, total,
        bs)``.
        ``lengths`` is the single source of truth: :meth:`_lengths` guarantees
        ``sum(lengths)`` covers the real tokens, so the derived indices always
        stay inside ``[0, bs) x [0, max_len)``. Uniform lengths (decode /
        target-verify) take the pure ``arange`` path, which needs no
        host->device copy; a ragged batch takes the general path instead, which
        is why :meth:`Qwen4ExpPLELayer.forward` has to run as an eager break
        rather than being captured.

        ``starts`` is each request's first flat token index. It belongs to the
        bundle because every consumer of ``req`` needs it to reach the tokens
        themselves, and the ragged path has to compute it for ``col`` anyway.
        """

        bs = len(lengths)
        total = int(sum(lengths))
        max_len = max(lengths) if lengths else 0
        positions = torch.arange(total, device=device, dtype=torch.long)
        if bs and max_len > 0 and max_len * bs == total:
            req = positions // max_len
            col = positions - req * max_len
            lengths_t = torch.full((bs,), max_len, device=device, dtype=torch.long)
            starts = torch.arange(bs, device=device, dtype=torch.long) * max_len
        else:
            lengths_t = torch.tensor(lengths, device=device, dtype=torch.long)
            ends = torch.cumsum(lengths_t, dim=0)
            starts = ends - lengths_t
            req = torch.searchsorted(ends.contiguous(), positions, right=True)
            col = positions - starts[req]
        return req, col, lengths_t, starts, max_len, total, bs

    def _token_contexts(
        self,
        input_ids: torch.Tensor,
        initial: torch.Tensor,
        lengths: list[int],
        index: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        req, col, lengths_t, _, max_len, total, bs = (
            index if index is not None else self._batch_indices(lengths, device)
        )
        if total == 0:
            return (
                input_ids.new_empty((0, self.ngram_size), dtype=torch.long),
                initial.clone(),
            )
        context_len = self.context_len
        eos = self.ple_embedding.eos_token_id
        # Augmented per-request sequence: [prefix(context_len) | tokens | pad].
        # The pad value is irrelevant: window rows for padded columns are never
        # emitted and the trailing-context gather never reads past the real end.
        aug = input_ids.new_full((bs, context_len + max_len), eos, dtype=torch.long)
        aug[:, :context_len] = initial.to(torch.long)
        aug[req, context_len + col] = input_ids.to(torch.long)
        # Sliding window of width ngram_size ending at each token.
        window = torch.arange(self.ngram_size, device=device)
        contexts = aug[req.unsqueeze(1), col.unsqueeze(1) + window.unsqueeze(0)]
        # Trailing context_len tokens per request (carried into the state page).
        rows = torch.arange(bs, device=device)
        tail = torch.arange(context_len, device=device)
        final_context = aug[
            rows.unsqueeze(1), lengths_t.unsqueeze(1) + tail.unsqueeze(0)
        ]
        return contexts, final_context

    def _conv_sequences(
        self,
        values: torch.Tensor,
        initial: torch.Tensor,
        lengths: list[int],
        index: tuple | None = None,
        need_intermediate: bool = True,
        *,
        add_terms: tuple[torch.Tensor, ...] = (),
        windows_out: torch.Tensor | None = None,
        windows_block_rows: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Short-conv over the flat batch.

        ``add_terms`` are full-width ``[tokens, channels]`` tensors folded into
        the conv output in order, letting callers skip separate tensor adds.
        ``windows_out`` receives the per-token state windows directly, one
        ``windows_block_rows`` row block per request with the carried state in
        row 0, so the verify scratch is filled without a packed intermediate.
        """
        device = values.device
        req, col, lengths_t, starts, max_len, total, bs = (
            index if index is not None else self._batch_indices(lengths, device)
        )
        state_len = self.conv_state_len
        if total == 0:
            return (
                values,
                initial.clone(),
                initial.new_empty((0, *initial.shape[1:])),
            )
        if values.is_cuda and state_len:
            return self._conv_sequences_cuda(
                values,
                initial,
                req,
                col,
                lengths_t,
                total,
                bs,
                need_intermediate,
                starts=starts,
                add_terms=add_terms,
                windows_out=windows_out,
                windows_block_rows=windows_block_rows,
            )
        conv_output, final_conv, intermediate = self._conv_sequences_torch(
            values,
            initial,
            req,
            col,
            lengths_t,
            max_len,
            total,
            bs,
            add_terms=add_terms,
        )
        if windows_out is not None:
            # The reference path has no kernel to scatter through, so it lands
            # the packed windows in the same rows afterwards.
            windows_out[req * windows_block_rows + 1 + col] = intermediate
            intermediate = windows_out
        return conv_output, final_conv, intermediate

    def _conv_sequences_cuda(
        self,
        values: torch.Tensor,
        initial: torch.Tensor,
        req: torch.Tensor,
        col: torch.Tensor,
        lengths_t: torch.Tensor,
        total: int,
        bs: int,
        need_intermediate: bool,
        *,
        starts: torch.Tensor | None = None,
        add_terms: tuple[torch.Tensor, ...] = (),
        windows_out: torch.Tensor | None = None,
        windows_block_rows: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(add_terms) > 2:
            raise ValueError("the fused conv epilogue folds at most two addends")
        channels = self.hc_hidden_size
        state_len = self.conv_state_len
        if starts is None:
            # Callers normally hand over the index bundle's copy; only direct
            # callers passing bare positional indices pay for the scan.
            starts = torch.cumsum(lengths_t, dim=0) - lengths_t
        weight = self.conv1d.weight.view(channels, self.conv_kernel_size)
        initial_c = initial.contiguous()
        values_c = values.contiguous()
        conv_output = torch.empty_like(values_c)
        # Row strides let strided addends feed the kernel without a contiguity
        # copy; only the last dim must be dense. Unused slots alias values_c,
        # which the ADD_* constexprs keep unread.
        addends = [
            term if term.stride(-1) == 1 else term.contiguous() for term in add_terms
        ]
        gated = addends[0] if len(addends) > 0 else values_c
        residual = addends[1] if len(addends) > 1 else values_c
        if windows_out is not None:
            if not need_intermediate:
                raise ValueError("windows_out needs need_intermediate=True")
            # The kernel addresses rows as a dense [rows, C, state_len] block,
            # and a copy would silently drop the writes, so refuse anything
            # else rather than repack.
            if not windows_out.is_contiguous():
                raise ValueError("windows_out must be contiguous")
            windows = windows_out
        elif need_intermediate:
            windows = values.new_empty((total, channels, state_len))
        else:
            # Dummy target: WRITE_WINDOWS=False never stores through it. Decode
            # and non-verify prefill skip the [T, C, state_len] materialization.
            windows = values.new_empty((0, channels, state_len))
        block_c = 256
        grid = (total, triton.cdiv(channels, block_c))
        use_pdl = pdl_enabled()
        pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
        _ple_dilated_conv_kernel[grid](
            values_c,
            initial_c,
            weight.contiguous(),
            req,
            col,
            starts,
            conv_output,
            windows,
            gated,
            residual,
            windows_block_rows,
            gated.stride(0),
            residual.stride(0),
            channels,
            D=self.ngram_size,
            K=self.conv_kernel_size,
            STATE=state_len,
            WRITE_WINDOWS=need_intermediate,
            SCATTER_WINDOWS=windows_out is not None,
            ADD_GATED=len(addends) > 0,
            ADD_RESIDUAL=len(addends) > 1,
            ENABLE_PDL=use_pdl,
            BLOCK_C=block_c,
            **pdl_kwargs,
        )
        final_conv = values.new_empty((bs, channels, state_len))
        _ple_conv_final_kernel[(bs, triton.cdiv(channels, block_c))](
            values_c,
            initial_c,
            lengths_t,
            starts,
            final_conv,
            channels,
            STATE=state_len,
            ENABLE_PDL=use_pdl,
            BLOCK_C=block_c,
            **pdl_kwargs,
        )
        return conv_output, final_conv, windows

    def _conv_sequences_torch(
        self,
        values: torch.Tensor,
        initial: torch.Tensor,
        req: torch.Tensor,
        col: torch.Tensor,
        lengths_t: torch.Tensor,
        max_len: int,
        total: int,
        bs: int,
        *,
        add_terms: tuple[torch.Tensor, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = values.device
        channels = self.hc_hidden_size
        state_len = self.conv_state_len
        weight = self.conv1d.weight.to(values.dtype)
        # Pack variable-length requests into [bs, state_len + max_len, C],
        # prepending each request's carried conv window, then run a single
        # grouped dilated conv instead of one launch per request. Padded
        # columns stay zero and never influence the gathered valid outputs
        # because each output only reads inputs within its own request.
        packed = values.new_zeros((bs, state_len + max_len, channels))
        if state_len:
            packed[:, :state_len, :] = initial.transpose(1, 2).to(values.dtype)
        packed[req, state_len + col, :] = values
        packed = packed.transpose(1, 2).contiguous()
        conv = F.conv1d(
            packed,
            weight,
            dilation=self.ngram_size,
            groups=channels,
        )
        conv = F.silu(conv)
        conv_output = conv[req, :, col]
        for term in add_terms:
            conv_output = conv_output + term
        if state_len:
            # windows[:, :, w, :] == packed[:, :, w : w + state_len]; window 0
            # reproduces the carried state, so token k maps to window k + 1 and
            # the trailing window per request sits at index length.
            windows = packed.unfold(2, state_len, 1)
            rows = torch.arange(bs, device=device)
            intermediate_conv = windows[req, :, col + 1, :]
            final_conv = windows[rows, :, lengths_t, :]
        else:
            intermediate_conv = values.new_empty((total, channels, 0))
            final_conv = values.new_empty((bs, channels, 0))
        return conv_output, final_conv, intermediate_conv

    def _verify_scratch_for(
        self,
        bs: int,
        width: int,
        backend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (bs, width)
        self._active_verify_key = key
        rows = bs * (width + 1)
        external = backend.ple_verify_scratch(self.layer_id)
        if external is None:
            raise RuntimeError("Qwen4-Exp PLE verify workspace was not preallocated")
        if external[0].shape[0] < rows or external[1].shape[0] < rows:
            raise RuntimeError(
                "Qwen4-Exp PLE verify workspace is smaller than the "
                f"captured batch: need {rows} rows"
            )
        scratch = (external[0][:rows], external[1][:rows])
        self._verify_scratch[key] = scratch
        return scratch

    def commit_verified(
        self,
        accepted_lengths: torch.Tensor,
        destination_pages: torch.Tensor,
    ) -> None:
        bs = accepted_lengths.shape[0]
        active_width = self._active_verify_key[1] if self._active_verify_key else None
        candidates = [
            key
            for key in self._verify_scratch
            if key[0] >= bs and (active_width is None or key[1] == active_width)
        ]
        if not candidates:
            return
        # Graph capture owns one scratch tensor per padded batch bucket. Model
        # Python does not run on replay, so `_active_verify_key` still names the
        # last captured graph; the smallest bucket covering the live batch is
        # the graph CudaGraphWrapper selected for this step.
        key = min(candidates, key=lambda value: value[0])
        _, width = key
        context_scratch, conv_scratch = self._verify_scratch[key]
        accepted = accepted_lengths.to(torch.long).clamp(1, width)
        source = torch.arange(bs, device=accepted.device) * (width + 1) + accepted
        pool = self._last_pool
        context_field = pool.arena.field(QWEN4_EXP_PLE_CONTEXT_FIELD)
        conv_field = pool.arena.field(qwen4_exp_ple_conv_field(self.layer_id))
        self._write_pages(
            context_field, destination_pages, context_scratch.index_select(0, source)
        )
        self._write_pages(
            conv_field, destination_pages, conv_scratch.index_select(0, source)
        )

    def _final_context(
        self,
        flat_ids: torch.Tensor,
        initial: torch.Tensor,
        lengths_t: torch.Tensor,
        starts: torch.Tensor,
    ) -> torch.Tensor:
        """Trailing ``context_len`` token ids per request (next step's prefix).

        Position ``j`` of request ``r`` reads virtual index ``length_r + j`` of
        ``[carried context | tokens]`` -- a handful of tiny [bs, context_len]
        ops instead of building the padded augmented matrix.
        """

        context_len = self.context_len
        offs = torch.arange(context_len, device=flat_ids.device)
        virtual = lengths_t.unsqueeze(1) + offs.unsqueeze(0)
        from_init = virtual < context_len
        init_part = initial.to(torch.long).gather(1, virtual.clamp_max(context_len - 1))
        token_idx = (starts.unsqueeze(1) + virtual - context_len).clamp_min(0)
        token_part = flat_ids.to(torch.long)[token_idx]
        return torch.where(from_init, init_part, token_part)

    def _gate_and_norm_torch(
        self,
        key: torch.Tensor,
        hidden_states: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unfused gating chain (CPU / fallback path)."""

        key_norm = self.norm_key(key).unflatten(-1, (self.hc_count, self.hidden_size))
        query_norm = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        gate = (key_norm * query_norm).sum(-1, keepdim=True) / math.sqrt(
            self.hidden_size
        )
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated = gated.flatten(-2)
        return gated, self.norm_conv(gated)

    def _gate_and_norm_cuda(
        self,
        key: torch.Tensor,
        hidden_states: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = key.shape[0]
        gated = key.new_empty((total, self.hc_hidden_size))
        normalized = torch.empty_like(gated)
        if total == 0:
            return gated, normalized
        # Row strides let kv_proj's split views feed the kernel without a
        # contiguity copy; only the last dim must be dense.
        if key.stride(-1) != 1:
            key = key.contiguous()
        if hidden_states.stride(-1) != 1:
            hidden_states = hidden_states.contiguous()
        if value.stride(-1) != 1:
            value = value.contiguous()
        use_pdl = pdl_enabled()
        _ple_gate_norm_kernel[(total, self.hc_count)](
            key,
            hidden_states,
            value,
            self.norm_key.gemma_weight,
            self.norm_query.gemma_weight,
            self.norm_conv.gemma_weight,
            gated,
            normalized,
            self.norm_key.variance_epsilon,
            1.0 / math.sqrt(self.hidden_size),
            key.stride(0),
            hidden_states.stride(0),
            value.stride(0),
            HC=self.hc_count,
            D=self.hidden_size,
            ENABLE_PDL=use_pdl,
            BLOCK_D=triton.next_power_of_2(self.hidden_size),
            **({"launch_pdl": True} if use_pdl else {}),
        )
        return gated, normalized

    @break_point
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Hidden states updated with this layer's PLE contribution.

        The gated value stream and the incoming ``hidden_states`` are folded
        into the short conv's epilogue, so the PLE delta is never materialized
        and the caller needs no residual add of its own.

        This runs as an eager break under a breakable capture: everything
        per-request here -- the index bundle, the state page ids, and the
        ``bs``-shaped page gather / scatter / conv-final grids -- would
        otherwise bake into the graph at the capture batch size (the prefill
        graph captures every bucket with a single dummy request) and then be
        replayed verbatim for a ragged multi-request batch.

        The break's handoff buffer is keyed by output shape and shared with
        same-shape breaks, and this output survives past the next break: it is
        the ``hyper_input`` residual that ``GatedResidualSimple.combine`` reads
        after the attention break. That is safe only because the width here is
        ``hc_count * hidden_size`` while an attention break emits
        ``heads * head_dim`` -- keep those distinct.
        """
        if ctx.forward_mode.is_idle() or hidden_states.shape[0] == 0:
            return hidden_states
        linear_backend = self._linear_backend(ctx)
        metadata = self._metadata(linear_backend)
        in_blocks_by_group = metadata.state_in_blocks_by_group or {}
        out_blocks_by_group = metadata.state_out_blocks_by_group or {}
        if QWEN4_EXP_PLE_CACHE_GROUP not in in_blocks_by_group:
            raise RuntimeError("Qwen4-Exp PLE cache group was not published")
        # A padded-bucket replay hands us bucket rows whose tail is filler, so
        # the lengths decide how many rows are real before anything reads them.
        lengths = self._lengths(metadata, input_ids.shape[0], ctx.bs)
        hidden_states, input_ids = slice_to_real_tokens(
            sum(lengths), hidden_states, input_ids
        )
        input_pages = in_blocks_by_group[QWEN4_EXP_PLE_CACHE_GROUP][: ctx.bs]
        output_pages = out_blocks_by_group[QWEN4_EXP_PLE_CACHE_GROUP][: ctx.bs]
        pool = ctx.token_to_kv_pool
        self._last_pool = pool
        context_field = pool.arena.field(QWEN4_EXP_PLE_CONTEXT_FIELD)
        conv_field = pool.arena.field(qwen4_exp_ple_conv_field(self.layer_id))
        initial_context = self._read_pages(
            context_field, input_pages, self.ple_embedding.eos_token_id
        )
        initial_conv = self._read_pages(conv_field, input_pages)
        index = self._batch_indices(lengths, input_ids.device)
        req, col, lengths_t, starts, _, total, bs = index
        flat_ids = input_ids.flatten()
        verify = metadata.mamba_output_indices is not None
        if flat_ids.is_cuda:
            # The n-gram windows are gathered inside the hash kernel; the
            # [tokens, ngram_size] context matrix is never materialized. The
            # raw verify-scratch rows (tail) are only emitted under verify.
            embeddings, context_tail = self.ple_embedding.forward_flat(
                flat_ids, initial_context, req, col, starts, need_tail=verify
            )
            final_context = self._final_context(
                flat_ids, initial_context, lengths_t, starts
            )
        else:
            contexts, final_context = self._token_contexts(
                flat_ids, initial_context, lengths, index
            )
            embeddings = self.ple_embedding(contexts)
            context_tail = contexts[:, 1:]

        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split([self.hc_hidden_size, self.hidden_size], dim=-1)
        if key.is_cuda:
            gated, normalized = self._gate_and_norm_cuda(key, hidden_states, value)
        else:
            gated, normalized = self._gate_and_norm_torch(key, hidden_states, value)
        context_scratch = conv_scratch = None
        scratch_stride = 0
        if verify:
            # Acquired before the conv so its kernel can scatter the per-token
            # state windows straight into the rollback rows.
            width = max(lengths, default=0)
            context_scratch, conv_scratch = self._verify_scratch_for(
                ctx.bs,
                width,
                linear_backend,
            )
            scratch_stride = width + 1
        conv_output, final_conv, _ = self._conv_sequences(
            normalized,
            initial_conv,
            lengths,
            index,
            need_intermediate=verify,
            add_terms=(gated, hidden_states),
            windows_out=conv_scratch,
            windows_block_rows=scratch_stride,
        )

        if verify:
            # Row 0 of every (width + 1)-strided block holds the carried state;
            # the conv already filled the token rows that follow it.
            init_rows = torch.arange(bs, device=input_ids.device) * scratch_stride
            context_scratch[init_rows] = initial_context
            conv_scratch[init_rows] = initial_conv
            if total:
                context_scratch[req * scratch_stride + 1 + col] = context_tail
        else:
            self._write_pages(context_field, output_pages, final_context)
            self._write_pages(conv_field, output_pages, final_conv)
        return conv_output


__all__ = [
    "QWEN4_EXP_PLE_CACHE_GROUP",
    "QWEN4_EXP_PLE_CONTEXT_FIELD",
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLELayer",
    "qwen4_exp_ple_conv_field",
    "quantize_ple_embedding_rows",
]
