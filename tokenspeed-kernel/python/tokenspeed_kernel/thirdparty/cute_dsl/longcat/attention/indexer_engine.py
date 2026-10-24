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

"""Host wrapper for the fused LongCat GPU0 indexer engine.

Compiles :class:`LongcatIndexerEngineKernel` per KV dtype, page count,
softmax scale, norm epsilon and quantized output count; reuses graph-stable buffers.

Scoring weights fold query quantization and softmax scales; the epilogue applies
the cached per-position index-K scale before weighted reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._indexer_engine_kernel import (
    CLUSTERS,
    CTAS_PER_CLUSTER,
    HEAD_DIM,
    HIDDEN,
    INDEX_K_PAGE_BYTES,
    KV_LORA_DIM,
    NUM_HEADS,
    NUM_THREADS,
    NUM_TICKETS,
    PAGE_SIZE,
    QKV_A_TILE_N,
    QKV_LATENT_OFFSET,
    QKV_OUT,
    ROPE_DIM,
    ROW_STATE_WORDS,
    SCRATCH_WORDS,
    SMEM_BUDGET_BYTES,
    TOPK,
    WK_OUT,
    WK_TILE_N,
    WQ_B_OUT,
    WQ_B_TILE_N,
    LongcatIndexerEngineKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._launch import (
    _cached_ptr,
    _cached_weight,
    assert_cluster_residency,
    row_major,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    prepare_stream_operand,
)

__all__ = [
    "IndexerEngineWorkspace",
    "IndexerWeights",
    "indexer_engine_supported",
    "make_indexer_engine_workspace",
    "run_indexer_engine",
]

_KV_DTYPE_NAMES = {
    torch.float8_e4m3fn: "fp8",
    torch.bfloat16: "bf16",
}
_KV_CUTLASS_NAMES = {"fp8": "Float8E4M3FN", "bf16": "BFloat16"}

NUM_TOKENS = 8
SOFTMAX_SCALE_DEFAULT = HEAD_DIM**-0.5 * NUM_HEADS**-0.5  # 1/64 folded


@dataclass
class IndexerWeights:
    """BF16 LongCat attention-0 indexer projection and norm weights.

    Attributes:
        w_qkv_a: fused QKV-A projection, BF16 [2112, 8192].
        wq_b: per-head Q-B projection, BF16 [4096, 1536].
        wk: index-K and scoring-weight projection, BF16 [160, 8192].
        rms_w: index-K RMSNorm weight, BF16 [128].
    """

    w_qkv_a: torch.Tensor
    wq_b: torch.Tensor
    wk: torch.Tensor
    rms_w: torch.Tensor


@dataclass
class IndexerEngineWorkspace:
    """Graph-stable buffers; shapes and dtypes are defined by the factory.

    Tickets are monotonic; P4 advances the generation. ``row_states`` contains
    filter counters and generation-parity fine/coarse score histograms.
    ``scratch`` holds per-tile projection publication slots. P0 fills only
    the q_lora prefix of ``qkv_out``, read directly by Q-B through TMA.
    ``scaled_w`` folds query and softmax scales into scoring weights.
    P5 writes the residual, normalized output and requested NVFP4 images.
    ``ptr_cache`` reuses wrappers until an argument's pointer or dtype changes.
    """

    tickets: torch.Tensor
    generation: torch.Tensor
    row_states: torch.Tensor
    scratch: torch.Tensor
    qkv_out: torch.Tensor
    raw_q: torch.Tensor
    raw_k: torch.Tensor
    q_fp8: torch.Tensor
    scaled_w: torch.Tensor
    topk: torch.Tensor
    lens: torch.Tensor
    logits: torch.Tensor
    residual_out: torch.Tensor
    norm_out: torch.Tensor
    quant_out: torch.Tensor
    quant_scale: torch.Tensor
    quant2_out: torch.Tensor
    quant2_scale: torch.Tensor
    ptr_cache: dict = field(default_factory=dict)


def indexer_engine_supported(hidden_states: torch.Tensor, kv_cache: torch.Tensor):
    """Whether the fused engine can serve these inputs on this device.

    Args:
        hidden_states: ``[8, 8192]`` BF16 hidden states.
        kv_cache: ``[slots, 576]`` FP8 or BF16 physical-slot KV rows.

    Returns:
        True on SM103 (B300) with contiguous supported dtypes.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(hidden_states.device)
    if (major, minor) != (10, 3):
        return False
    return (
        hidden_states.dtype == torch.bfloat16
        and tuple(hidden_states.shape) == (NUM_TOKENS, HIDDEN)
        and hidden_states.is_contiguous()
        and kv_cache.dtype in _KV_DTYPE_NAMES
        and kv_cache.dim() == 2
        and kv_cache.shape[1] == KV_LORA_DIM + ROPE_DIM
        and kv_cache.is_contiguous()
    )


def make_indexer_engine_workspace(
    device: torch.device, max_pages: int
) -> IndexerEngineWorkspace:
    """Allocate the graph-stable indexer engine buffers.

    Args:
        device: CUDA device for every buffer.
        max_pages: block-table width (logical pages per request); fixes the
            logits width (``max_pages * 64``) and the P3 radix chunk.

    Returns:
        An :class:`IndexerEngineWorkspace` with zeroed tickets and the
        generation set to 1.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be positive")

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=device)

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    return IndexerEngineWorkspace(
        tickets=zeros(NUM_TICKETS, dtype=torch.uint32),
        # Peers and ticket waits use generation 1 on the first step.
        generation=torch.ones((1,), dtype=torch.int32, device=device),
        row_states=zeros(NUM_TOKENS, ROW_STATE_WORDS, dtype=torch.int32),
        scratch=zeros(SCRATCH_WORDS, dtype=torch.uint32),
        qkv_out=empty(NUM_TOKENS, QKV_OUT),
        raw_q=empty(NUM_TOKENS, WQ_B_OUT),
        raw_k=empty(NUM_TOKENS, WK_OUT),
        q_fp8=empty(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn),
        scaled_w=empty(NUM_TOKENS, NUM_HEADS, dtype=torch.float32),
        # Zero initialization makes incomplete selection writes observable.
        topk=zeros(NUM_TOKENS, TOPK, dtype=torch.int32),
        lens=zeros(NUM_TOKENS, dtype=torch.int32),
        logits=zeros(NUM_TOKENS, max_pages * PAGE_SIZE),
        residual_out=zeros(NUM_TOKENS, HIDDEN),
        norm_out=zeros(NUM_TOKENS, HIDDEN),
        quant_out=zeros(NUM_TOKENS, 4096, dtype=torch.uint8),
        quant_scale=zeros(128 * 512, dtype=torch.float8_e4m3fn),
        quant2_out=zeros(NUM_TOKENS, 4096, dtype=torch.uint8),
        quant2_scale=zeros(128 * 512, dtype=torch.float8_e4m3fn),
    )


class _IndexerEntry:
    """JIT entry: build static layouts and plans from raw pointers, then launch."""

    def __init__(self, kernel: "LongcatIndexerEngineKernel", max_pages: int) -> None:
        self.kernel = kernel
        self.max_pages = max_pages

    @cute.jit
    def wrapper(
        self,
        hidden_ptr: cute.Pointer,
        w_qkv_a_ptr: cute.Pointer,
        wq_b_ptr: cute.Pointer,
        wk_ptr: cute.Pointer,
        rms_w_ptr: cute.Pointer,
        cos_sin_ptr: cute.Pointer,
        positions_ptr: cute.Pointer,
        out_loc_ptr: cute.Pointer,
        index_k_ptr: cute.Pointer,
        qkv_out_ptr: cute.Pointer,
        raw_q_ptr: cute.Pointer,
        raw_k_ptr: cute.Pointer,
        q_fp8_ptr: cute.Pointer,
        scaled_w_ptr: cute.Pointer,
        seq_lens_ptr: cute.Pointer,
        block_table_ptr: cute.Pointer,
        topk_ptr: cute.Pointer,
        lens_ptr: cute.Pointer,
        logits_ptr: cute.Pointer,
        tickets_ptr: cute.Pointer,
        row_states_ptr: cute.Pointer,
        scratch_ptr: cute.Pointer,
        peer_bases_ptr: cute.Pointer,
        generation_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        post_gamma_ptr: cute.Pointer,
        residual_out_ptr: cute.Pointer,
        norm_out_ptr: cute.Pointer,
        quant_out_ptr: cute.Pointer,
        quant_scale_ptr: cute.Pointer,
        quant_global_ptr: cute.Pointer,
        quant2_out_ptr: cute.Pointer,
        quant2_scale_ptr: cute.Pointer,
        quant2_global_ptr: cute.Pointer,
        transport_bases_ptr: cute.Pointer,
        rotation_ptr: cute.Pointer,
        stream: CUstream,
    ) -> None:
        max_pages = self.max_pages
        hidden = row_major(hidden_ptr, (NUM_TOKENS, HIDDEN))
        q_a = cute.make_tensor(
            qkv_out_ptr,
            cute.make_layout((NUM_TOKENS, QKV_LATENT_OFFSET), stride=(QKV_OUT, 1)),
        )
        hidden_tma = prepare_stream_operand(hidden, NUM_TOKENS)
        q_a_tma = prepare_stream_operand(q_a, NUM_TOKENS)
        qkv_operands = (
            *hidden_tma,
            *prepare_stream_operand(
                row_major(w_qkv_a_ptr, (QKV_LATENT_OFFSET, HIDDEN)), QKV_A_TILE_N
            ),
        )
        wk_operands = (
            *hidden_tma,
            *prepare_stream_operand(row_major(wk_ptr, (WK_OUT, HIDDEN)), WK_TILE_N),
        )
        qb_operands = (
            *q_a_tma,
            *prepare_stream_operand(
                row_major(wq_b_ptr, (WQ_B_OUT, QKV_LATENT_OFFSET)), WQ_B_TILE_N
            ),
        )
        self.kernel.kernel(
            row_major(rms_w_ptr, (HEAD_DIM,)),
            row_major(cos_sin_ptr, (_MAX_ROWS, ROPE_DIM)),
            row_major(positions_ptr, (NUM_TOKENS,)),
            row_major(out_loc_ptr, (NUM_TOKENS,)),
            row_major(index_k_ptr, (max_pages * INDEX_K_PAGE_BYTES,)),
            row_major(qkv_out_ptr, (NUM_TOKENS, QKV_OUT)),
            row_major(raw_q_ptr, (NUM_TOKENS, WQ_B_OUT)),
            row_major(raw_k_ptr, (NUM_TOKENS, WK_OUT)),
            row_major(q_fp8_ptr, (NUM_TOKENS, NUM_HEADS, HEAD_DIM)),
            row_major(scaled_w_ptr, (NUM_TOKENS, NUM_HEADS)),
            row_major(seq_lens_ptr, (1,)),
            row_major(block_table_ptr, (1, max_pages)),
            row_major(topk_ptr, (NUM_TOKENS, TOPK)),
            row_major(lens_ptr, (NUM_TOKENS,)),
            row_major(logits_ptr, (NUM_TOKENS, max_pages * PAGE_SIZE)),
            row_major(tickets_ptr, (NUM_TICKETS,)),
            row_major(row_states_ptr, (NUM_TOKENS, ROW_STATE_WORDS)),
            row_major(scratch_ptr, (SCRATCH_WORDS,)),
            row_major(peer_bases_ptr, (8,)),
            row_major(generation_ptr, (1,)),
            row_major(residual_ptr, (NUM_TOKENS, HIDDEN)),
            row_major(post_gamma_ptr, (HIDDEN,)),
            row_major(residual_out_ptr, (NUM_TOKENS, HIDDEN)),
            row_major(norm_out_ptr, (NUM_TOKENS, HIDDEN)),
            row_major(quant_out_ptr, (NUM_TOKENS, 4096)),
            cute.make_tensor(quant_scale_ptr, cute.make_layout((128 * 512,))),
            cute.make_tensor(quant_global_ptr, cute.make_layout((1,))),
            row_major(quant2_out_ptr, (NUM_TOKENS, 4096)),
            cute.make_tensor(quant2_scale_ptr, cute.make_layout((128 * 512,))),
            cute.make_tensor(quant2_global_ptr, cute.make_layout((1,))),
            row_major(transport_bases_ptr, (8,)),
            row_major(rotation_ptr, (9,)),
            qkv_operands,
            wk_operands,
            qb_operands,
        ).launch(
            grid=(CTAS_PER_CLUSTER, CLUSTERS, 1),
            block=(NUM_THREADS, 1, 1),
            cluster=(CTAS_PER_CLUSTER, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )


_COMPILED: dict[tuple, object] = {}
# Declared extent of indexed-only tables (cos_sin, kv); kept small enough
# that extent * row width stays below 2**31 in the layout arithmetic.
_MAX_ROWS = 1 << 20


def _compile(
    kv_name: str,
    max_pages: int,
    softmax_scale: float,
    norm_eps: float,
    ptrs: tuple,
    stream,
    quant_outputs: int = 0,
):
    """Cache one compiled kernel per configuration for the process lifetime.

    ``ptrs`` only type the compile; later calls accept fresh pointers.
    """
    key = (
        kv_name,
        max_pages,
        softmax_scale,
        norm_eps,
        int(quant_outputs),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        kv_dtype = cutlass.Float8E4M3FN if kv_name == "fp8" else cutlass.BFloat16
        kernel = LongcatIndexerEngineKernel(
            kv_dtype=kv_dtype,
            softmax_scale=softmax_scale,
            chunk_size=max_pages * PAGE_SIZE // 8,
            norm_eps=norm_eps,
            quant_outputs=quant_outputs,
        )
        compiled = cute.compile(
            _IndexerEntry(kernel, max_pages).wrapper, *ptrs, stream=stream
        )
        _COMPILED[key] = compiled
    return compiled


# All CLUSTERS * CTAS_PER_CLUSTER CTAs must be resident simultaneously:
# waits on peer CTA tickets otherwise deadlock.
_CTAS_PER_SM = 1  # the ``min_blocks_per_mp`` launch bound the kernel requests


@lru_cache(maxsize=None)
def _assert_cluster_residency(device_index: int) -> None:
    """Check SM/SMEM capacity, falling back to the active-cluster query.

    All 74 two-CTA clusters need residency: one CTA per SM on 148 SMs.

    Args:
        device_index: CUDA device ordinal to validate (cached per device).

    Raises:
        RuntimeError: when the fallback reports too few resident clusters.
    """
    props = torch.cuda.get_device_properties(device_index)
    smem_per_cta = props.shared_memory_per_multiprocessor // _CTAS_PER_SM
    sm_capacity = props.multi_processor_count * _CTAS_PER_SM
    if sm_capacity >= CLUSTERS * CTAS_PER_CLUSTER and smem_per_cta >= SMEM_BUDGET_BYTES:
        return
    assert_cluster_residency(
        device_index,
        CTAS_PER_CLUSTER,
        CLUSTERS,
        "LongCat indexer engine (all CTAs must be resident or the ticket waits "
        f"deadlock, and the {_CTAS_PER_SM} CTAs/SM fallback does not fit: "
        f"{props.multi_processor_count} SMs, "
        f"{props.shared_memory_per_multiprocessor} B SMEM/SM vs "
        f"{SMEM_BUDGET_BYTES} B arena)",
    )


def run_indexer_engine(
    hidden_states: torch.Tensor,
    weights: IndexerWeights,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    index_k_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    workspace: IndexerEngineWorkspace,
    *,
    softmax_scale: float = SOFTMAX_SCALE_DEFAULT,
    peer_bases: torch.Tensor,
    collective=None,
    residual: torch.Tensor | None = None,
    post_norm_gamma: torch.Tensor | None = None,
    norm_eps: float = 1e-5,
    quant_global: torch.Tensor | None = None,
    quant2_global: torch.Tensor | None = None,
) -> None:
    """Run one full GPU indexer step.

    Args:
        hidden_states: ``[8, 8192]`` BF16 input.
        weights: indexer projections/norms (:class:`IndexerWeights`).
        cos_sin_cache: ``[max_pos, 64]`` FP32 interleaved RoPE cache
            (``[cos(32) | sin(32)]`` rows).
        positions: ``[8]`` int32 positions of the verify rows.
        out_cache_loc: ``[8]`` int64 index-K destination slots.
        index_k_cache: ``uint8[max_pages * 8448]`` packed index-K plane
            (128 FP8 bytes + FP32 scale per 132-B row).
        kv_cache: ``[slots, 576]`` physical-slot KV rows, checked for format
            eligibility; worker ranks write these rows, not the indexer.
        seq_lens: ``[1]`` int32 candidate length of the (single) request.
        block_table: ``[1, max_pages]`` int32 logical-to-physical pages.
        workspace: buffers from :func:`make_indexer_engine_workspace`; the
            outputs land in ``logits``/``topk``/``lens`` (+ q_fp8/scaled_w
            and the P0 projections).
        softmax_scale: full scoring scale folded into the weights (production
            value ``128**-0.5 * 32**-0.5``).
        peer_bases: ``int64[8]`` live peer region byte offsets to publish to.
        collective: TP8 o_proj transport from
            ``make_o_proj_collective_workspace`` (``transport_bases`` and
            ``rotation_flags``); required for P5.
        residual: ``[8, 8192]`` BF16 residual rows added to the P5
            all-reduce; required for P5.
        post_norm_gamma: ``[8192]`` BF16 post-attention RMSNorm weight;
            required for P5.
        norm_eps: P5 RMSNorm epsilon.

    Returns:
        None. Unless the short-context selection shortcut is taken,
        ``workspace.logits`` holds the scoring rows (causal tail ``-inf``).
        ``workspace.topk``/``workspace.lens`` always hold the selection;
        ``workspace.residual_out``/``workspace.norm_out``
        hold the reduced rows.
    """
    if not indexer_engine_supported(hidden_states, kv_cache):
        raise ValueError(
            "unsupported indexer engine inputs:"
            f" hidden={tuple(hidden_states.shape)} {hidden_states.dtype},"
            f" kv={tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    if (
        positions.shape != (NUM_TOKENS,)
        or positions.dtype != torch.int32
        or out_cache_loc.shape != (NUM_TOKENS,)
        or out_cache_loc.dtype != torch.int64
        or seq_lens.shape != (1,)
        or seq_lens.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != 1
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
        or index_k_cache.dtype != torch.uint8
        or any(
            tensor.device != hidden_states.device or not tensor.is_contiguous()
            for tensor in (
                positions,
                out_cache_loc,
                seq_lens,
                block_table,
                index_k_cache,
            )
        )
    ):
        raise ValueError(
            "indexer position/slot/page metadata has the wrong shape or dtype"
        )
    max_pages = int(block_table.shape[1])
    if workspace.logits.shape != (NUM_TOKENS, max_pages * PAGE_SIZE):
        raise ValueError("indexer workspace must match the fixed block-table width")
    if index_k_cache.numel() < max_pages * INDEX_K_PAGE_BYTES:
        raise ValueError("index_k_cache must hold max_pages * 8448 bytes")
    missing = [
        name
        for name, value in (
            ("collective", collective),
            ("residual", residual),
            ("post_norm_gamma", post_norm_gamma),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"segment P5 requires {', '.join(missing)} (the o_proj"
            " collective workspace, the residual rows and the RMSNorm"
            " weight)"
        )
    if (
        residual.dtype != torch.bfloat16
        or tuple(residual.shape) != (NUM_TOKENS, HIDDEN)
        or not residual.is_contiguous()
    ):
        raise ValueError("residual must be a contiguous bf16 [8, 8192] tensor")
    if (
        post_norm_gamma.dtype != torch.bfloat16
        or tuple(post_norm_gamma.shape) != (HIDDEN,)
        or not post_norm_gamma.is_contiguous()
    ):
        raise ValueError("post_norm_gamma must be a contiguous bf16 [8192] tensor")
    if peer_bases is None or peer_bases.dtype != torch.int64:
        raise ValueError("peer_bases must supply the live int64 peer byte offsets")
    _assert_cluster_residency(hidden_states.device.index)

    # P5 packs NVFP4 for each supplied input scale. The compile-time output
    # count keeps absent-scale placeholders unread.
    quant_outputs = (quant_global is not None) + (quant2_global is not None)
    if quant2_global is not None and quant_global is None:
        raise ValueError("quant2_global requires quant_global")
    placeholder = workspace.tickets.view(torch.float32)[:1]
    quant_scales = (
        quant_global if quant_global is not None else placeholder,
        quant2_global if quant2_global is not None else placeholder,
    )
    for name, scale in (
        ("quant_global", quant_global),
        ("quant2_global", quant2_global),
    ):
        if scale is not None and (
            scale.numel() != 1
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous float32[1] tensor")

    peer_tensor = peer_bases
    residual_tensor = residual
    gamma_tensor = post_norm_gamma
    transport_tensor = collective.transport_bases
    rotation_tensor = collective.rotation_flags
    cache = workspace.ptr_cache
    w_qkv_a = _cached_weight(cache, "w_qkv_a", weights.w_qkv_a)
    wq_b = _cached_weight(cache, "wq_b", weights.wq_b)
    wk = _cached_weight(cache, "wk", weights.wk)
    ptrs = (
        _cached_ptr(cache, "hidden_states", hidden_states, cutlass.BFloat16),
        _cached_ptr(cache, "w_qkv_a", w_qkv_a, cutlass.BFloat16),
        _cached_ptr(cache, "wq_b", wq_b, cutlass.BFloat16),
        _cached_ptr(cache, "wk", wk, cutlass.BFloat16),
        _cached_ptr(cache, "rms_w", weights.rms_w, cutlass.BFloat16),
        _cached_ptr(cache, "cos_sin_cache", cos_sin_cache, cutlass.Float32),
        _cached_ptr(cache, "positions", positions, cutlass.Int32, 4),
        _cached_ptr(cache, "out_cache_loc", out_cache_loc, cutlass.Int64, 8),
        _cached_ptr(cache, "index_k_cache", index_k_cache, cutlass.Uint8),
        _cached_ptr(cache, "qkv_out", workspace.qkv_out, cutlass.BFloat16),
        _cached_ptr(cache, "raw_q", workspace.raw_q, cutlass.BFloat16),
        _cached_ptr(cache, "raw_k", workspace.raw_k, cutlass.BFloat16),
        # Raw e4m3 bits: P1 stores ``_quantize_bits`` bytes, P2 recasts to u32.
        _cached_ptr(cache, "q_fp8", workspace.q_fp8, cutlass.Uint8),
        _cached_ptr(cache, "scaled_w", workspace.scaled_w, cutlass.Float32),
        _cached_ptr(cache, "seq_lens", seq_lens, cutlass.Int32, 4),
        _cached_ptr(cache, "block_table", block_table, cutlass.Int32, 4),
        _cached_ptr(cache, "topk", workspace.topk, cutlass.Int32, 4),
        _cached_ptr(cache, "lens", workspace.lens, cutlass.Int32, 4),
        _cached_ptr(cache, "logits", workspace.logits, cutlass.BFloat16),
        _cached_ptr(cache, "tickets", workspace.tickets, cutlass.Uint32, 4),
        _cached_ptr(cache, "row_states", workspace.row_states, cutlass.Int32, 4),
        _cached_ptr(cache, "scratch", workspace.scratch, cutlass.Uint32, 4),
        _cached_ptr(cache, "peer_bases", peer_tensor, cutlass.Int64, 8),
        _cached_ptr(cache, "generation", workspace.generation, cutlass.Int32, 4),
        _cached_ptr(cache, "residual", residual_tensor, cutlass.BFloat16),
        _cached_ptr(cache, "post_norm_gamma", gamma_tensor, cutlass.BFloat16),
        _cached_ptr(cache, "residual_out", workspace.residual_out, cutlass.BFloat16),
        _cached_ptr(cache, "norm_out", workspace.norm_out, cutlass.BFloat16),
        _cached_ptr(cache, "quant_out", workspace.quant_out, cutlass.Uint8),
        _cached_ptr(cache, "quant_scale", workspace.quant_scale, cutlass.Float8E4M3FN),
        _cached_ptr(cache, "quant_global", quant_scales[0], cutlass.Float32),
        _cached_ptr(cache, "quant2_out", workspace.quant2_out, cutlass.Uint8),
        _cached_ptr(
            cache, "quant2_scale", workspace.quant2_scale, cutlass.Float8E4M3FN
        ),
        _cached_ptr(cache, "quant2_global", quant_scales[1], cutlass.Float32),
        _cached_ptr(cache, "transport_bases", transport_tensor, cutlass.Int64, 8),
        _cached_ptr(cache, "rotation", rotation_tensor, cutlass.Uint32, 4),
    )
    stream = CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
    compiled = _compile(
        _KV_DTYPE_NAMES[kv_cache.dtype],
        max_pages,
        float(softmax_scale),
        float(norm_eps),
        ptrs,
        stream,
        quant_outputs=quant_outputs,
    )
    compiled(*ptrs, stream=stream)
