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
from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tokenspeed_kernel.ops.ple import (
    ple_conv_sequences,
    ple_gate_norm,
    ple_ngram_ids,
    ple_page_gather,
    ple_page_scatter,
)
from torch import nn

from tokenspeed.runtime.configs.qwen4_exp_config import Qwen4ExpTextConfig
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    qwen4_exp_backend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
    PLEForwardMetadata,
    Qwen4ExpPLEBackend,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    qwen4_exp_ple_context_field,
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

_IndexBundle = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
# Uniform-batch index bundles captured into a CUDA graph. During capture the
# (bs, max_len) pair is fixed per bucket, so entries are bounded by the capture
# set. Entries are never evicted: a replay reads the addresses its capture
# recorded, so freeing one would let the allocator hand that memory to someone
# else and feed a replay another tensor's bytes. Other graphs sharing the
# capture pool can overwrite their contents, so eager calls neither read nor
# populate this cache. They compute fresh tensors instead.
_UNIFORM_INDEX_CACHE: dict[tuple[int, int, torch.device], _IndexBundle] = {}


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
        table + local_id * head_dim + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_SCALE:
        if ROW_SCALE:
            values = values * tl.load(scale_ptr + local_id)
        else:
            values = values * scale_value
    tl.store(
        out_ptr + row * head_dim + offsets,
        tl.where(in_range, values, 0.0).to(out_dtype),
        mask=mask,
    )


def materialize_ngram_table_on_host(embedding: VocabParallelEmbedding) -> None:
    """Give a meta-constructed n-gram table page-locked host storage."""

    source = embedding.weight
    host_weight = nn.Parameter(
        torch.empty(
            source.shape, dtype=source.dtype, device="cpu", pin_memory=True
        ),
        requires_grad=False,
    )
    for name, value in vars(source).items():
        setattr(host_weight, name, value)
    del embedding.weight
    embedding.register_parameter("weight", host_weight)


def host_gather_ngram_rows(
    embedding: VocabParallelEmbedding,
    ids: torch.Tensor,
    scale: float | None,
    out: torch.Tensor,
    row_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill ``out`` with the ``ids`` rows of a host-resident n-gram table.

    The dequant factor follows the checkpoint format (see
    :func:`_ple_host_gather_kernel`): ``row_scale`` is the device per-row buffer
    for an online-quantized table, ``scale`` the per-tensor scalar for an
    offline FP8 one, and both ``None`` for a compute-dtype table that needs
    none.
    """

    rows = ids.numel()
    if rows == 0:
        return out
    head_dim = out.shape[-1]
    has_row = row_scale is not None
    _ple_host_gather_kernel[(rows,)](
        embedding.weight.data_ptr(),
        ids.reshape(-1),
        float(scale) if scale is not None else 1.0,
        # Unused when ROW_SCALE is false; ``out`` is a valid device pointer to
        # keep triton's type inference happy without a throwaway allocation.
        row_scale if has_row else out,
        out.view(rows, head_dim),
        head_dim,
        embedding.shard_indices.org_vocab_start_index,
        embedding.shard_indices.org_vocab_end_index,
        IS_FP8=embedding.weight.dtype == torch.float8_e4m3fn,
        HAS_SCALE=has_row or scale is not None,
        ROW_SCALE=has_row,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=1,
    )
    return out


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
        # Construct offloaded tables on meta so production-sized tables never
        # transiently consume device memory before moving to pinned host RAM.
        self.offload_embedding = bool(
            getattr(config, "ple_offload_embedding", False)
        )
        with torch.device("meta") if self.offload_embedding else nullcontext():
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
        if self.offload_embedding:
            materialize_ngram_table_on_host(self.ngram_embedding)
        # Created on first use rather than here: a host table can be built on a
        # CUDA-less box, and only the prefetch path ever needs the stream.
        self._gather_stream: torch.cuda.Stream | None = None
        if self.embed_store_dtype is not None and not self.offload_embedding:
            # Per-local-row dequant scales, written by the loader's online
            # quantization. Ones (not zeros / empty): rows gathered before the
            # checkpoint lands, or shard-masked rows folded to local row 0,
            # must stay finite. Non-persistent: derived from the bf16
            # checkpoint, never round-tripped.
            #
            # An offloaded table is not built with this buffer: the offline FP8
            # checkpoint offloading targets carries a single per-tensor scale
            # (kept in _checkpoint_weight_scale), so a per-row buffer would be
            # ngram_vocab_size_base * 4 bytes of a repeated constant. A
            # compute-dtype checkpoint under offload is still quantized online
            # per row; the loader allocates this buffer lazily in that case.
            self.register_buffer(
                "ngram_embedding_scale",
                torch.ones(self.ngram_embedding.num_embeddings_per_partition),
                persistent=False,
            )

    def allocate_lookup_buffer(
        self, tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Destination for a host gather, shaped like a flattened lookup."""

        return torch.empty(
            (tokens, self.embedding_dim),
            dtype=self.embed_output_dtype,
            device=device,
        )

    def gather_host(self, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Read the ``ids`` rows from host memory into ``out``.

        ``out`` is ``[tokens, ngram_heads * head_dim]``. Row-major order makes
        the per-head rows land contiguously, so the flatten that the device
        path applies afterwards is already implicit here, and the FP8 dequant
        happens inside the kernel instead of materializing an fp32 temporary.
        The dequant factor follows the checkpoint format: a per-row device
        buffer for an online-quantized table, the per-tensor scalar for an
        offline FP8 one (see :meth:`__init__`), and none for a compute-dtype
        table.
        """

        row_scale = getattr(self, "ngram_embedding_scale", None)
        scale = (
            self._checkpoint_weight_scale
            if row_scale is None and self.embed_store_dtype is not None
            else None
        )
        host_gather_ngram_rows(
            self.ngram_embedding,
            ids,
            scale,
            out.view(-1, self.head_dim),
            row_scale=row_scale,
        )
        return out

    def reduce_lookup(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine per-rank shard contributions of a gathered lookup."""

        if self.ngram_embedding.tp_size > 1:
            return all_reduce(embeddings, self.ngram_embedding.tp_group)
        return embeddings

    def start_flat_gather(
        self,
        input_ids: torch.Tensor,
        initial: torch.Tensor,
        req: torch.Tensor,
        col: torch.Tensor,
        starts: torch.Tensor,
        need_tail: bool = False,
        tail_out: torch.Tensor | None = None,
        tail_block_rows: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Issue the host gather on a side stream and return before it lands.

        Only the gather itself is moved off the caller's stream. The hash kernel
        is microseconds of device work, so overlapping it would buy nothing,
        while keeping its outputs on the caller's stream means the ids and the
        destination are allocated there and only *read* across the boundary --
        the pair of ``record_stream`` calls is what tells the caching allocator
        the side stream still holds them.

        The returned buffer is not readable until :meth:`finish_flat_gather`.
        """

        ids, tail = self._ngram_ids_flat_cuda(
            input_ids,
            initial,
            req,
            col,
            starts,
            need_tail,
            tail_out,
            tail_block_rows,
        )
        out = self.allocate_lookup_buffer(ids.shape[0], ids.device)
        if self._gather_stream is None:
            self._gather_stream = torch.cuda.Stream()
        stream = self._gather_stream
        stream.wait_stream(torch.cuda.current_stream())
        ids.record_stream(stream)
        out.record_stream(stream)
        with torch.cuda.stream(stream):
            self.gather_host(ids, out)
        return out, tail

    def finish_flat_gather(self, gathered: torch.Tensor) -> torch.Tensor:
        """Wait for a :meth:`start_flat_gather` to land, then reduce it."""

        torch.cuda.current_stream().wait_stream(self._gather_stream)
        return self.reduce_lookup(gathered)

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
        tail_out: torch.Tensor | None = None,
        tail_block_rows: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return ple_ngram_ids(
            input_ids,
            initial,
            req,
            col,
            starts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
            eos_token_id=self.eos_token_id,
            need_tail=need_tail,
            tail_out=tail_out,
            tail_block_rows=tail_block_rows,
        )

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
        tail_out: torch.Tensor | None = None,
        tail_block_rows: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """CUDA path: embed n-grams without materializing the window matrix."""

        ids, tail = self._ngram_ids_flat_cuda(
            input_ids,
            initial,
            req,
            col,
            starts,
            need_tail,
            tail_out,
            tail_block_rows,
        )
        if self.offload_embedding:
            out = self.allocate_lookup_buffer(ids.shape[0], ids.device)
            return self.reduce_lookup(self.gather_host(ids, out)), tail
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
        if self.offload_embedding:
            out = self.allocate_lookup_buffer(ids.shape[0], ids.device)
            return self.reduce_lookup(self.gather_host(ids, out))
        embeddings = self.ngram_embedding(ids, reduce_results=False)
        if self.embed_store_dtype is not None:
            embeddings = self._dequant(embeddings, ids)
        if self.ngram_embedding.tp_size > 1:
            embeddings = all_reduce(embeddings, self.ngram_embedding.tp_group)
        return embeddings.flatten(-2)


class _PleLookupPlan(NamedTuple):
    """What a PLE forward derives before it touches the n-gram table.

    Extracted so :meth:`Qwen4ExpPLELayer.start_prefetch` and
    :meth:`Qwen4ExpPLELayer.forward` share a single derivation. Re-deriving it
    on both sides is the kind of duplication that fails silently: the lengths,
    the state page ids and the carried context all have to describe the same
    batch as the rows a prefetch already has in flight, and a mismatch yields
    plausible embeddings for the wrong tokens rather than an error.
    """

    lengths: list[int]
    num_real_tokens: int
    flat_ids: torch.Tensor
    index: tuple
    output_pages: torch.Tensor
    context_field: torch.Tensor
    conv_field: torch.Tensor
    initial_context: torch.Tensor
    initial_conv: torch.Tensor
    verify: bool
    backend: Qwen4ExpPLEBackend
    metadata: PLEForwardMetadata
    context_scratch: torch.Tensor | None
    conv_scratch: torch.Tensor | None
    scratch_stride: int


class Qwen4ExpPLELayer(nn.Module):
    """PLE gating plus dilated depthwise short convolution.

    Persistent context and convolution windows live in the model's unified
    cache under :data:`QWEN4_EXP_PLE_CACHE_GROUP`.
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
        context_layer_id = min(config.short_conv_layer_ids)
        self.context_field_id = qwen4_exp_ple_context_field(context_layer_id)
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
        # Set by start_prefetch, consumed by the next forward; see start_prefetch.
        self._prefetched: tuple | None = None

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
    def _ple_backend(ctx: ForwardContext) -> Qwen4ExpPLEBackend:
        backend = qwen4_exp_backend(ctx.attn_backend).ple_backend
        if backend is None:
            raise RuntimeError("Qwen4-Exp PLE requires its PLE backend")
        return backend

    @staticmethod
    def _metadata(backend: Qwen4ExpPLEBackend) -> PLEForwardMetadata:
        metadata = backend.forward_metadata
        if metadata is None:
            raise RuntimeError("Qwen4-Exp PLE metadata was not prepared")
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

        if not field.is_cuda:
            page_ids = page_ids.to(torch.long)
            valid = page_ids > 0
            values = field.index_select(0, page_ids.clamp_min(0))
            mask = valid.view(-1, *([1] * (values.ndim - 1)))
            return torch.where(mask, values, torch.full_like(values, default))
        return ple_page_gather(
            field,
            page_ids,
            Qwen4ExpPLELayer._page_row_stride(field),
            default,
        )

    @staticmethod
    def _write_pages(
        field: torch.Tensor,
        page_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Store one row per page id, skipping null pages."""

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
        ple_page_scatter(
            field,
            page_ids,
            values,
            Qwen4ExpPLELayer._page_row_stride(field),
        )

    def _lengths(self, metadata, total_tokens: int, bs: int) -> list[int]:
        """Per-request token counts for a flat batch of ``total_tokens`` rows.

        ``total_tokens`` is an upper bound, not an identity: a padded-bucket
        graph replay hands the layer bucket rows whose tail is filler, so the
        CPU lengths only have to fit inside them. They stay the single source
        of truth for what is real -- the caller slices to their sum.
        """
        lengths = metadata.query_lengths[:bs]
        if len(lengths) != bs or sum(lengths) > total_tokens:
            raise RuntimeError("Qwen4-Exp PLE query lengths exceed the supplied batch")
        return lengths

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

        The uniform bundle is memoized during CUDA graph capture, since
        ``(bs, max_len)`` is fixed for a capture bucket and recomputing it cost
        seven elementwise kernels per replay. Eager calls compute fresh tensors
        to avoid unbounded cache growth from varying prefill lengths. The
        cached tensors are shared between calls, so consumers must treat them
        as read-only.

        ``starts`` is each request's first flat token index. It belongs to the
        bundle because every consumer of ``req`` needs it to reach the tokens
        themselves, and the ragged path has to compute it for ``col`` anyway.
        """

        bs = len(lengths)
        total = int(sum(lengths))
        max_len = max(lengths) if lengths else 0
        if bs and max_len > 0 and max_len * bs == total:
            key = (bs, max_len, device)
            capturing = (
                device.type == "cuda" and torch.cuda.is_current_stream_capturing()
            )
            bundle = _UNIFORM_INDEX_CACHE.get(key) if capturing else None
            if bundle is None:
                positions = torch.arange(total, device=device, dtype=torch.long)
                req = positions // max_len
                col = positions - req * max_len
                lengths_t = torch.full((bs,), max_len, device=device, dtype=torch.long)
                starts = torch.arange(bs, device=device, dtype=torch.long) * max_len
                bundle = (req, col, lengths_t, starts)
                if capturing:
                    _UNIFORM_INDEX_CACHE[key] = bundle
            req, col, lengths_t, starts = bundle
        else:
            positions = torch.arange(total, device=device, dtype=torch.long)
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
        On CUDA, ``windows_out`` receives the carried row and per-token state
        windows directly. The fallback caller fills its carried row separately.
        """
        device = values.device
        req, col, lengths_t, starts, max_len, total, bs = (
            index if index is not None else self._batch_indices(lengths, device)
        )
        state_len = self.conv_state_len
        if total == 0 and not (
            values.is_cuda and state_len and windows_out is not None
        ):
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
        channels = self.hc_hidden_size
        state_len = self.conv_state_len
        if starts is None:
            # Callers normally hand over the index bundle's copy; only direct
            # callers passing bare positional indices pay for the scan.
            starts = torch.cumsum(lengths_t, dim=0) - lengths_t
        if windows_out is not None:
            if not need_intermediate:
                raise ValueError("windows_out needs need_intermediate=True")
            windows = windows_out
        elif need_intermediate:
            windows = values.new_empty((total, channels, state_len))
        else:
            # Decode and non-verify prefill skip the [T, C, state_len]
            # materialization entirely.
            windows = None
        return ple_conv_sequences(
            values,
            initial,
            self.conv1d.weight.view(channels, self.conv_kernel_size),
            req,
            col,
            lengths_t,
            starts,
            total_tokens=total,
            batch_size=bs,
            dilation=self.ngram_size,
            kernel_size=self.conv_kernel_size,
            state_len=state_len,
            add_terms=add_terms,
            windows=windows,
            windows_block_rows=windows_block_rows,
            scatter_windows=windows_out is not None,
        )

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
        return ple_gate_norm(
            key,
            hidden_states,
            value,
            self.norm_key.gemma_weight,
            self.norm_query.gemma_weight,
            self.norm_conv.gemma_weight,
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            eps=self.norm_key.variance_epsilon,
        )

    def _lookup_plan(
        self, input_ids: torch.Tensor, ctx: ForwardContext
    ) -> _PleLookupPlan | None:
        """Derive the per-request quantities a lookup needs; ``None`` when idle.

        Everything here is stream-agnostic bookkeeping (CPU lengths, index
        bundle, page reads) that both :meth:`forward` and :meth:`start_prefetch`
        must agree on. It stops just short of the n-gram hash so the two callers
        can drive the gather onto different streams from a single derivation.
        """

        if ctx.forward_mode.is_idle() or input_ids.shape[0] == 0:
            return None
        backend = self._ple_backend(ctx)
        metadata = self._metadata(backend)
        # A padded-bucket replay hands us bucket rows whose tail is filler, so
        # the lengths decide how many rows are real before anything reads them.
        lengths = self._lengths(metadata, input_ids.shape[0], ctx.bs)
        num_real_tokens = sum(lengths)
        (input_ids,) = slice_to_real_tokens(num_real_tokens, input_ids)
        input_pages = metadata.input_blocks[: ctx.bs]
        output_pages = metadata.output_blocks[: ctx.bs]
        pool = ctx.token_to_kv_pool
        load_tracker = getattr(pool, "layerwise_load_tracker", None)
        if load_tracker is not None:
            load_tracker.wait_for_layer(self.layer_id)
        context_field = pool.arena.field(self.context_field_id)
        conv_field = pool.arena.field(qwen4_exp_ple_conv_field(self.layer_id))
        initial_context = self._read_pages(
            context_field, input_pages, self.ple_embedding.eos_token_id
        )
        initial_conv = self._read_pages(conv_field, input_pages)
        index = self._batch_indices(lengths, input_ids.device)
        flat_ids = input_ids.flatten()
        verify = metadata.verify_width is not None
        context_scratch = conv_scratch = None
        scratch_stride = 0
        if verify:
            context_scratch, conv_scratch = backend.ple_verify_scratch(
                self.context_field_id, self.layer_id, ctx.bs
            )
            scratch_stride = metadata.verify_width + 1
        return _PleLookupPlan(
            lengths=lengths,
            num_real_tokens=num_real_tokens,
            flat_ids=flat_ids,
            index=index,
            output_pages=output_pages,
            context_field=context_field,
            conv_field=conv_field,
            initial_context=initial_context,
            initial_conv=initial_conv,
            verify=verify,
            backend=backend,
            metadata=metadata,
            context_scratch=context_scratch,
            conv_scratch=conv_scratch,
            scratch_stride=scratch_stride,
        )

    def start_prefetch(self, input_ids: torch.Tensor, ctx: ForwardContext) -> None:
        """Kick off the host gather so it overlaps the preceding decoder layers.

        Only meaningful when the table lives on the host and the ids are on the
        device -- otherwise there is nothing to hide behind PCIe/C2C latency and
        this is a no-op. The plan is derived once here and carried to
        :meth:`forward`, which reuses it verbatim instead of re-deriving (see
        :class:`_PleLookupPlan`). Called before the layer that owns this PLE
        runs, so its result is waited on, not recomputed.
        """

        if not self.ple_embedding.offload_embedding or self._prefetched is not None:
            return
        plan = self._lookup_plan(input_ids, ctx)
        if plan is None or not plan.flat_ids.is_cuda:
            return
        req, col, _, starts, _, _, _ = plan.index
        gathered, _ = self.ple_embedding.start_flat_gather(
            plan.flat_ids,
            plan.initial_context,
            req,
            col,
            starts,
            tail_out=plan.context_scratch,
            tail_block_rows=plan.scratch_stride,
        )
        self._prefetched = (plan, gathered)

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
        prefetched = getattr(self, "_prefetched", None)
        self._prefetched = None
        if prefetched is not None:
            plan, gathered = prefetched
        else:
            plan = self._lookup_plan(input_ids, ctx)
        (hidden_states,) = slice_to_real_tokens(plan.num_real_tokens, hidden_states)
        req, col, lengths_t, starts, _, total, _ = plan.index
        lengths = plan.lengths
        flat_ids = plan.flat_ids
        initial_context = plan.initial_context
        initial_conv = plan.initial_conv
        context_field = plan.context_field
        conv_field = plan.conv_field
        output_pages = plan.output_pages
        verify = plan.verify
        context_scratch = plan.context_scratch
        conv_scratch = plan.conv_scratch
        scratch_stride = plan.scratch_stride
        if prefetched is not None:
            embeddings = self.ple_embedding.finish_flat_gather(gathered)
            final_context = (
                None
                if verify
                else self._final_context(flat_ids, initial_context, lengths_t, starts)
            )
        elif flat_ids.is_cuda:
            # The n-gram windows are gathered inside the hash kernel; verify
            # writes their state rows directly instead of returning a packed tail.
            embeddings, _ = self.ple_embedding.forward_flat(
                flat_ids,
                initial_context,
                req,
                col,
                starts,
                tail_out=context_scratch,
                tail_block_rows=scratch_stride,
            )
            # Only the non-verify branch writes it back, and ``verify`` is fixed
            # when a graph is captured, so skipping here keeps the gather and
            # its ten elementwise kernels out of the captured verify graph.
            final_context = (
                None
                if verify
                else self._final_context(flat_ids, initial_context, lengths_t, starts)
            )
        else:
            contexts, final_context = self._token_contexts(
                flat_ids, initial_context, lengths, plan.index
            )
            embeddings = self.ple_embedding(contexts)
            context_tail = contexts[:, 1:]

        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split([self.hc_hidden_size, self.hidden_size], dim=-1)
        if key.is_cuda:
            gated, normalized = self._gate_and_norm_cuda(key, hidden_states, value)
        else:
            gated, normalized = self._gate_and_norm_torch(key, hidden_states, value)
        conv_output, final_conv, _ = self._conv_sequences(
            normalized,
            initial_conv,
            lengths,
            plan.index,
            need_intermediate=verify,
            add_terms=(gated, hidden_states),
            windows_out=conv_scratch,
            windows_block_rows=scratch_stride,
        )

        if verify and not flat_ids.is_cuda:
            # The fallback has no fused producers, so it fills the same rollback
            # layout after computing the packed state windows.
            context_scratch[::scratch_stride] = initial_context
            conv_scratch[::scratch_stride] = initial_conv
            if total:
                token_rows = req * scratch_stride + 1 + col
                context_scratch[token_rows] = context_tail
        elif not verify:
            self._write_pages(context_field, output_pages, final_context)
            self._write_pages(conv_field, output_pages, final_conv)
        return conv_output


__all__ = [
    "QWEN4_EXP_PLE_CACHE_GROUP",
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLELayer",
    "qwen4_exp_ple_context_field",
    "qwen4_exp_ple_conv_field",
    "quantize_ple_embedding_rows",
]
