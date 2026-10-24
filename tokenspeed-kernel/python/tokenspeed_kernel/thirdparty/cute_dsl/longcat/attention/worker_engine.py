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

"""Host launch and persistent workspace for attention-0 worker ranks 1-7.

The serving pipeline uses BF16 streaming QKV-A, composed Q-B/absorbed weights,
interleaved RoPE, tagged selection, and sparse MLA with 16 contiguous splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._launch import (
    _cached_ptr,
    _cached_weight,
    assert_cluster_residency,
    row_major,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._sparse_mla_hetero_kernel import (
    PARTIAL_FLOATS,
    TOPK,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    make_folded_q_b_weight,
    prepare_stream_operand,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    smem_bytes as gemv_smem_bytes,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._worker_engine_kernel import (
    NUM_TOKENS,
    P0_FAST_TILE_N,
    ROW_SUMSQ_WORDS,
    TICK_ABS_READY,
    TICK_EXIT,
    TICK_GEN,
    TICK_HEAD_READY,
    TICK_PREPARE_COMPLETED,
    TICK_QB_COMPLETED,
    TICK_QB_READY,
    TICK_QKV_COMP_Q,
    TICK_QKV_COMPLETED,
    TICKET_WORDS,
    LongcatWorkerEngineKernel,
)

_DTYPE_MAP = {
    torch.float8_e4m3fn: "fp8",
    torch.bfloat16: "bf16",
}

# Declared extent of indexed-only tables (cos_sin, kv); kept small enough
# that extent * row width stays below 2**31 in the layout arithmetic.
_MAX_ROWS = 1 << 20


@dataclass
class WorkerEngineWorkspace:
    """Graph-stable buffers; shapes and dtypes are defined by the factory.

    Tickets and P5 counters accumulate across steps; generation distinguishes
    arrivals. P0 stores q_a in the ordinary QKV output for direct TMA reads.
    P6 publishes the untagged selection for attention branch 1. P7 writes the
    residual, normalized output and both downstream NVFP4 activation images.
    ``ptr_cache`` reuses raw pointer wrappers and retains composed weights.
    """

    tickets: torch.Tensor
    partial: torch.Tensor
    counter: torch.Tensor
    p5_generation: torch.Tensor
    topk_out: torch.Tensor
    lens_out: torch.Tensor
    row_sumsq: torch.Tensor
    qkv_out: torch.Tensor
    q_b_out: torch.Tensor
    query: torch.Tensor
    attn_proj: torch.Tensor
    o_local: torch.Tensor
    residual_out: torch.Tensor
    norm_out: torch.Tensor
    quant_out: torch.Tensor
    quant_scale: torch.Tensor
    quant2_out: torch.Tensor
    quant2_scale: torch.Tensor
    ptr_cache: dict = field(default_factory=dict)


def make_worker_engine_workspace(
    device, local_heads: int, kv_dtype: torch.dtype
) -> WorkerEngineWorkspace:
    """Allocate the persistent workspace for one worker-engine instance.

    Args:
        device: CUDA device of this rank.
        local_heads: this rank's attention head count (9 or 10).
        kv_dtype: ``torch.float8_e4m3fn`` or ``torch.bfloat16``.

    Returns:
        The zero-initialized workspace with the generation word at 1.
    """
    if kv_dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported kv dtype {kv_dtype}")

    def rows(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=device)

    tickets = rows(TICKET_WORDS, dtype=torch.uint32)
    tickets[TICK_GEN] = 1  # the first step runs at generation 1
    return WorkerEngineWorkspace(
        tickets=tickets,
        partial=rows(NUM_TOKENS, 4, PARTIAL_FLOATS, dtype=torch.float32),
        counter=rows(NUM_TOKENS, dtype=torch.int32),
        p5_generation=rows(1, dtype=torch.int32),
        topk_out=rows(NUM_TOKENS, TOPK, dtype=torch.int32),
        lens_out=rows(NUM_TOKENS, dtype=torch.int32),
        row_sumsq=rows(2, 2, NUM_TOKENS, dtype=torch.float32),
        qkv_out=rows(NUM_TOKENS, 2112),
        q_b_out=rows(NUM_TOKENS, local_heads * 192),
        query=rows(NUM_TOKENS, local_heads, 576, dtype=kv_dtype),
        attn_proj=rows(NUM_TOKENS, local_heads * 128),
        o_local=rows(NUM_TOKENS, 8192),
        residual_out=rows(NUM_TOKENS, 8192),
        norm_out=rows(NUM_TOKENS, 8192),
        # P7 packs norm_out for two NVFP4 consumers with separate input scales.
        quant_out=rows(NUM_TOKENS, 4096, dtype=torch.uint8),
        quant_scale=rows(128 * 512, dtype=torch.float8_e4m3fn),
        quant2_out=rows(NUM_TOKENS, 4096, dtype=torch.uint8),
        quant2_scale=rows(128 * 512, dtype=torch.float8_e4m3fn),
    )


def worker_engine_supported(hidden: torch.Tensor, kv_cache: torch.Tensor) -> bool:
    """Whether the worker engine can serve these inputs on this device.

    Args:
        hidden: ``[8, 8192]`` bf16 hidden states.
        kv_cache: ``[slots, 576]`` fp8/bf16 KV cache rows.

    Returns:
        True on SM103 (B300) with supported shapes and dtypes.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(hidden.device)
    if (major, minor) != (10, 3):
        return False
    return (
        hidden.dim() == 2
        and tuple(hidden.shape) == (NUM_TOKENS, 8192)
        and hidden.dtype == torch.bfloat16
        and kv_cache.dim() == 2
        and kv_cache.shape[1] == 576
        and kv_cache.dtype in _DTYPE_MAP
        and kv_cache.is_contiguous()
    )


class _WorkerEngineEntry:
    """JIT launch entry: declares layouts, prepares routines, launches."""

    def __init__(self, kernel: LongcatWorkerEngineKernel) -> None:
        self.kernel = kernel

    @cute.jit
    def wrapper(
        self,
        hidden_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        qkv_a_weight_ptr: cute.Pointer,
        q_b_weight_ptr: cute.Pointer,
        kv_gamma_ptr: cute.Pointer,
        wkc_ptr: cute.Pointer,
        wvc_t_ptr: cute.Pointer,
        o_proj_ptr: cute.Pointer,
        post_gamma_ptr: cute.Pointer,
        cos_sin_ptr: cute.Pointer,
        positions_ptr: cute.Pointer,
        out_cache_loc_ptr: cute.Pointer,
        block_table_ptr: cute.Pointer,
        kv_cache_ptr: cute.Pointer,
        topk_slots_ptr: cute.Pointer,
        topk_lens_ptr: cute.Pointer,
        selection_flag_ptr: cute.Pointer,
        topk_out_ptr: cute.Pointer,
        lens_out_ptr: cute.Pointer,
        tickets_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        p5_gen_ptr: cute.Pointer,
        row_sumsq_ptr: cute.Pointer,
        qkv_out_ptr: cute.Pointer,
        q_b_out_ptr: cute.Pointer,
        query_ptr: cute.Pointer,
        attn_proj_ptr: cute.Pointer,
        o_local_ptr: cute.Pointer,
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
        """Build tensors from raw pointers and launch the engine kernel."""
        kernel = self.kernel
        heads = kernel.local_heads
        cos_rows = kv_rows = _MAX_ROWS
        tickets = row_major(tickets_ptr, (TICKET_WORDS,))

        def ticket(word, count=1):
            return row_major(tickets.iterator + cutlass.Int32(word), (count,))

        zone_a1_bytes = max(
            gemv_smem_bytes(P0_FAST_TILE_N),
            gemv_smem_bytes(kernel.p_b_tile_n, kernel.p_b_stages),
        )
        zone_a2_bytes = 104496 if kernel.is_fp8 else 71712
        zone_a2_offset, zone_c_offset, arena_bytes = kernel.arena_layout(
            zone_a1_bytes, zone_a2_bytes
        )
        qkv_weight = row_major(qkv_a_weight_ptr, (2112, 8192))
        qb_weight = row_major(q_b_weight_ptr, (kernel.p_b_width, 1536))
        hidden = row_major(hidden_ptr, (NUM_TOKENS, 8192))
        q_a = cute.make_tensor(
            qkv_out_ptr, cute.make_layout((NUM_TOKENS, 1536), stride=(2112, 1))
        )
        qkv_operands = (
            *prepare_stream_operand(hidden, NUM_TOKENS),
            *prepare_stream_operand(qkv_weight, P0_FAST_TILE_N),
        )
        qb_operands = (
            *prepare_stream_operand(q_a, NUM_TOKENS),
            *prepare_stream_operand(qb_weight, kernel.p_b_tile_n),
        )
        kernel.kernel(
            row_major(residual_ptr, (NUM_TOKENS, 8192)),
            row_major(kv_gamma_ptr, (512,)),
            cute.make_tensor(
                wkc_ptr, cute.make_ordered_layout((heads, 512, 128), order=(2, 1, 0))
            ),
            row_major(wvc_t_ptr, (heads, 128, 512)),
            row_major(o_proj_ptr, (8192, heads * 128)),
            row_major(post_gamma_ptr, (8192,)),
            row_major(cos_sin_ptr, (cos_rows, 64)),
            row_major(positions_ptr, (8,)),
            row_major(out_cache_loc_ptr, (8,)),
            row_major(block_table_ptr, (1, kv_rows)),
            row_major(kv_cache_ptr, (kv_rows, 576)),
            row_major(topk_slots_ptr, (NUM_TOKENS, TOPK)),
            row_major(topk_lens_ptr, (8,)),
            row_major(selection_flag_ptr, (1,)),
            row_major(topk_out_ptr, (NUM_TOKENS * TOPK,)),
            row_major(lens_out_ptr, (NUM_TOKENS,)),
            row_major(partial_ptr, (NUM_TOKENS, kernel.quarters, PARTIAL_FLOATS)),
            row_major(counter_ptr, (8,)),
            row_major(p5_gen_ptr, (1,)),
            row_major(row_sumsq_ptr, (ROW_SUMSQ_WORDS,)),
            row_major(qkv_out_ptr, (NUM_TOKENS, 2112)),
            # P4 reads Q-B by (token, head, channel).
            row_major(q_b_out_ptr, (NUM_TOKENS, heads * 192)),
            row_major(q_b_out_ptr, (NUM_TOKENS, heads, 192)),
            row_major(query_ptr, (NUM_TOKENS, heads, 576)),
            row_major(attn_proj_ptr, (NUM_TOKENS, heads * 128)),
            row_major(o_local_ptr, (NUM_TOKENS, 8192)),
            row_major(residual_out_ptr, (NUM_TOKENS, 8192)),
            row_major(norm_out_ptr, (NUM_TOKENS, 8192)),
            row_major(quant_out_ptr, (NUM_TOKENS, 4096)),
            row_major(quant_scale_ptr, (128 * 512,)),
            row_major(quant_global_ptr, (1,)),
            row_major(quant2_out_ptr, (NUM_TOKENS, 4096)),
            row_major(quant2_scale_ptr, (128 * 512,)),
            row_major(quant2_global_ptr, (1,)),
            row_major(transport_bases_ptr, (8,)),
            row_major(rotation_ptr, (9,)),
            ticket(TICK_GEN),
            ticket(TICK_QKV_COMPLETED),
            ticket(TICK_QKV_COMP_Q),
            ticket(TICK_QB_COMPLETED),
            ticket(TICK_QB_READY, heads * 4),
            ticket(TICK_ABS_READY, heads * 8),
            ticket(TICK_PREPARE_COMPLETED),
            ticket(TICK_EXIT),
            ticket(TICK_HEAD_READY, heads),
            qkv_operands,
            qb_operands,
            zone_a2_offset,
            zone_c_offset,
            arena_bytes,
        ).launch(
            grid=(4, kernel.num_clusters, 1),
            block=(384, 1, 1),
            cluster=(4, 1, 1),
            min_blocks_per_mp=1,
            use_pdl=True,
            stream=stream,
        )


_COMPILED: dict[tuple, object] = {}


def _get_compiled(key: tuple, kernel_kwargs: dict, ptrs: tuple, stream):
    """Cache the compiled engine entrypoint per key.

    Pointers only type the compile; later calls accept fresh pointers.
    """
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = LongcatWorkerEngineKernel(**kernel_kwargs)
        assert_cluster_residency(
            key[0], 4, kernel.num_clusters, "LongCat worker engine"
        )
        compiled = cute.compile(
            _WorkerEngineEntry(kernel).wrapper,
            *ptrs,
            stream=stream,
        )
        compiled.to(key[0])
        _COMPILED[key] = compiled
    return compiled


def _folded_q_b_weight(
    cache: dict, q_b_weight: torch.Tensor, q_gamma: torch.Tensor
) -> torch.Tensor:
    """``q_b_weight * q_gamma`` along K, rebuilt only when either pointer changes.

    Retain sources to prevent allocator address reuse while the key matches.
    """
    key = (q_b_weight.data_ptr(), q_gamma.data_ptr())
    entry = cache.get("q_b_folded")
    if entry is None or entry[0] != key:
        with torch.no_grad():
            folded = (q_b_weight.float() * q_gamma.float()[None, :]).to(torch.bfloat16)
        entry = (key, folded, q_b_weight, q_gamma)
        cache["q_b_folded"] = entry
    return entry[1]


def _composed_q_b_weight(
    cache: dict, q_b_folded: torch.Tensor, w_kc: torch.Tensor
) -> torch.Tensor:
    """Cache the composed BF16 Q-B matrix and retain its sources."""
    key = (q_b_folded.data_ptr(), w_kc.data_ptr())
    entry = cache.get("composed_q_b_weight")
    if entry is None or entry[0] != key:
        with torch.no_grad():
            weight = make_folded_q_b_weight(q_b_folded, w_kc)
        entry = (key, weight, q_b_folded, w_kc)
        cache["composed_q_b_weight"] = entry
    return entry[1]


def run_worker_engine(
    hidden: torch.Tensor,
    qkv_a_weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    q_b_weight: torch.Tensor,
    w_kc: torch.Tensor,
    w_vc_t: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_norm_gamma: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    block_table: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    selection_flag: torch.Tensor,
    workspace: WorkerEngineWorkspace,
    *,
    sm_scale: float,
    quant_global: torch.Tensor,
    quant2_global: torch.Tensor,
    residual: torch.Tensor | None = None,
    k_scale: float = 1.0,
    rank: int = 1,
    collective=None,
    norm_eps: float = 1e-5,
    selection_slot_words: int = 0,
) -> dict[str, torch.Tensor]:
    """Run one verify-8 attention step for this worker rank.

    Composed Q-B writes RoPE channels to ``q_b_out`` and the absorbed result
    directly to ``query``, skipping the NOPE intermediate.

    Args:
        hidden: ``[8, 8192]`` bf16 input hidden states (the QKV-A input).
        qkv_a_weight: ``[2112, 8192]`` bf16 QKV-A projection weight.
        q_gamma: ``[1536]`` bf16 q_a RMSNorm weight.
        kv_gamma: ``[512]`` bf16 kv_a RMSNorm weight.
        q_b_weight: ``[H*192, 1536]`` bf16 Q-B projection weight.
        w_kc: ``[H, 512, 128]`` bf16 absorbed-BMM weight (n-major rows).
        w_vc_t: ``[H, 128, 512]`` bf16, ``w_vc`` transposed per head.
        o_proj_weight: ``[8192, H*128]`` bf16 o_proj weight.
        post_norm_gamma: ``[8192]`` bf16 post-attention RMSNorm weight.
        cos_sin_cache: ``[max_pos, 64]`` bf16/fp32 rotary tables.
        positions: ``[8]`` int32 positions of the verify tokens
            (``positions[0] + 8 == seq_len``; token ``t`` attends to
            positions ``< positions[t] + 1``).
        out_cache_loc: ``[8]`` int32 KV slots this step writes.
        block_table: ``[1, pages]`` int32 page table, page size 64. Hosted P5
            gathers each token's forced first 16 and last 1024 positions before
            selection arrives. ``topk_slots`` must match the indexer's layout:
            forced slots in ascending position order, then score-selected slots.
        kv_cache: ``[slots, 576]`` fp8/bf16 KV cache.
        topk_slots: ``[8, 2048]`` int32 selected physical slots.
        topk_lens: ``[8]`` int32 valid selection lengths.
        selection_flag: ``[1]`` uint32 selection-exchange publication flag.
        workspace: buffers from :func:`make_worker_engine_workspace`.
        sm_scale: softmax scale of the sparse MLA.
        residual: ``[8, 8192]`` bf16 rows added by P7 to the all-reduced
            o_proj output; defaults to ``hidden`` (production feeds a
            normalized ``hidden`` and the untouched residual stream here).
        k_scale: KV dequantization scale (1.0 for BF16 caches).
        rank: this rank's id in the TP8 group (1..7), used by P7.
        collective: the ``LongcatDownReduceOneShotWorkspace`` of the TP8
            group.
        norm_eps: post-attention RMSNorm epsilon.
        quant_global / quant2_global: contiguous float32[1] reciprocal input
            scales for the two NVFP4 consumers of the normalized output.
        selection_slot_words: int32 stride between the exchange's two receive
            slots. If positive, ``topk_slots``/``topk_lens`` point at slot 0;
            the device picks ``generation & 1`` for captured-step replay.
            0 (default) uses the views as given.
    Returns:
        Dict with the ``qkv_out``, ``q_b_out``, ``query``,
        ``attn_proj``, ``o_local``, ``residual_out`` and ``norm_out``
        intermediates (the workspace tensors).
    """
    if not worker_engine_supported(hidden, kv_cache):
        raise ValueError(
            f"unsupported worker-engine inputs: hidden={tuple(hidden.shape)} "
            f"{hidden.dtype}, kv={tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    if residual is None:
        residual = hidden
    elif (
        residual.dtype != torch.bfloat16
        or tuple(residual.shape) != (NUM_TOKENS, 8192)
        or not residual.is_contiguous()
    ):
        raise ValueError("residual must be a contiguous bf16 [8, 8192] tensor")
    local_heads = workspace.query.shape[1]
    dtype_name = _DTYPE_MAP[kv_cache.dtype]
    if collective is None:
        raise ValueError("worker engine requires the TP8 o_proj collective workspace")
    if local_heads not in (9, 10):
        raise ValueError("worker requires 9 or 10 heads")
    if (
        tuple(q_b_weight.shape) != (local_heads * 192, 1536)
        or q_b_weight.dtype != torch.bfloat16
        or tuple(w_kc.shape) != (local_heads, 512, 128)
        or w_kc.dtype != torch.bfloat16
        or q_b_weight.device != w_kc.device
    ):
        raise ValueError(
            "fold_prepare needs BF16 [H*192,1536] Q-B and [H,512,128] w_kc"
        )
    if (
        topk_slots.shape != (NUM_TOKENS, TOPK)
        or topk_slots.dtype != torch.int32
        or topk_lens.shape != (NUM_TOKENS,)
        or topk_lens.dtype != torch.int32
        or positions.dtype != torch.int32
        or out_cache_loc.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != 1
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
        or selection_flag.dtype != torch.uint32
    ):
        raise ValueError("selection/position tensors have the wrong shape or dtype")
    if tuple(w_vc_t.shape) != (local_heads, 128, 512) or not w_vc_t.is_contiguous():
        raise ValueError("w_vc_t must be a contiguous [H, 128, 512] tensor")

    device = hidden.device
    tickets = workspace.tickets

    for name, scale in (
        ("quant_global", quant_global),
        ("quant2_global", quant2_global),
    ):
        if (
            scale is None
            or scale.numel() != 1
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous float32[1] tensor")

    transport_bases = collective.transport_bases
    rotation = collective.rotation_flags

    kv_cutlass = cutlass.Float8E4M3FN if dtype_name == "fp8" else cutlass.BFloat16
    cos_cutlass = (
        cutlass.BFloat16 if cos_sin_cache.dtype == torch.bfloat16 else cutlass.Float32
    )
    cache = workspace.ptr_cache
    qkv_a_weight = _cached_weight(cache, "qkv_a", qkv_a_weight)
    q_b_folded = _folded_q_b_weight(cache, q_b_weight, q_gamma)
    q_b_composed = _composed_q_b_weight(cache, q_b_folded, w_kc)
    ptrs = (
        _cached_ptr(cache, "hidden", hidden, cutlass.BFloat16),
        _cached_ptr(cache, "residual", residual, cutlass.BFloat16),
        _cached_ptr(cache, "qkv_a_weight_ptr", qkv_a_weight, cutlass.BFloat16),
        _cached_ptr(cache, "q_b_weight_ptr", q_b_composed, cutlass.BFloat16),
        _cached_ptr(cache, "kv_gamma", kv_gamma, cutlass.BFloat16),
        _cached_ptr(cache, "w_kc", w_kc, cutlass.BFloat16),
        _cached_ptr(cache, "w_vc_t", w_vc_t, cutlass.BFloat16),
        _cached_ptr(cache, "o_proj_weight", o_proj_weight, cutlass.BFloat16),
        _cached_ptr(cache, "post_norm_gamma", post_norm_gamma, cutlass.BFloat16),
        _cached_ptr(cache, "cos_sin_cache", cos_sin_cache, cos_cutlass),
        _cached_ptr(cache, "positions", positions, cutlass.Int32),
        _cached_ptr(cache, "out_cache_loc", out_cache_loc, cutlass.Int32),
        _cached_ptr(cache, "block_table", block_table, cutlass.Int32),
        _cached_ptr(cache, "kv_cache", kv_cache, kv_cutlass),
        _cached_ptr(cache, "topk_slots", topk_slots, cutlass.Int32),
        _cached_ptr(cache, "topk_lens", topk_lens, cutlass.Int32),
        _cached_ptr(cache, "selection_flag", selection_flag, cutlass.Uint32),
        _cached_ptr(cache, "topk_out", workspace.topk_out, cutlass.Int32),
        _cached_ptr(cache, "lens_out", workspace.lens_out, cutlass.Int32),
        _cached_ptr(cache, "tickets", tickets, cutlass.Uint32),
        _cached_ptr(cache, "partial", workspace.partial, cutlass.Float32),
        _cached_ptr(cache, "counter", workspace.counter, cutlass.Int32),
        _cached_ptr(cache, "p5_generation", workspace.p5_generation, cutlass.Int32),
        _cached_ptr(cache, "row_sumsq", workspace.row_sumsq, cutlass.Float32),
        _cached_ptr(cache, "qkv_out", workspace.qkv_out, cutlass.BFloat16),
        _cached_ptr(cache, "q_b_out", workspace.q_b_out, cutlass.BFloat16),
        _cached_ptr(cache, "query", workspace.query, kv_cutlass),
        _cached_ptr(cache, "attn_proj", workspace.attn_proj, cutlass.BFloat16),
        _cached_ptr(cache, "o_local", workspace.o_local, cutlass.BFloat16),
        _cached_ptr(cache, "residual_out", workspace.residual_out, cutlass.BFloat16),
        _cached_ptr(cache, "norm_out", workspace.norm_out, cutlass.BFloat16),
        _cached_ptr(cache, "quant_out", workspace.quant_out, cutlass.Uint8),
        _cached_ptr(cache, "quant_scale", workspace.quant_scale, cutlass.Float8E4M3FN),
        _cached_ptr(cache, "quant_global", quant_global, cutlass.Float32),
        _cached_ptr(cache, "quant2_out", workspace.quant2_out, cutlass.Uint8),
        _cached_ptr(
            cache, "quant2_scale", workspace.quant2_scale, cutlass.Float8E4M3FN
        ),
        _cached_ptr(cache, "quant2_global", quant2_global, cutlass.Float32),
        _cached_ptr(cache, "transport_bases", transport_bases, cutlass.Int64),
        _cached_ptr(cache, "rotation", rotation, cutlass.Uint32),
    )
    stream = CUstream(torch.cuda.current_stream(device).cuda_stream)
    kernel_kwargs = dict(
        local_heads=local_heads,
        kv_dtype=kv_cutlass,
        sm_scale=sm_scale,
        k_scale=k_scale,
        rank=rank,
        norm_eps=norm_eps,
        selection_slot_words=selection_slot_words,
    )
    key = (
        device.index,
        dtype_name,
        local_heads,
        "bf16" if cos_sin_cache.dtype == torch.bfloat16 else "fp32",
        rank,
        float(sm_scale),
        float(k_scale),
        float(norm_eps),
        False,
        True,
        int(selection_slot_words),
    )
    compiled = _get_compiled(key, kernel_kwargs, ptrs, stream)
    compiled(*ptrs, stream=stream)

    return {
        "qkv_out": workspace.qkv_out,
        "q_b_out": workspace.q_b_out,
        "query": workspace.query,
        "attn_proj": workspace.attn_proj,
        "o_local": workspace.o_local,
        "residual_out": workspace.residual_out,
        "norm_out": workspace.norm_out,
    }


__all__ = [
    "WorkerEngineWorkspace",
    "make_worker_engine_workspace",
    "worker_engine_supported",
    "run_worker_engine",
]
