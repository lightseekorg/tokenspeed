"""Pre-compile deep_gemm JIT kernels used by DeepSeek V4.

deep_gemm compiles CUDA kernels on first invocation for each unique
(M, N, K, tile) combination. This module provides warmup functions that
exercise representative shapes at startup so no JIT compilation happens
on the serving hot path.
"""

from __future__ import annotations

import logging
from math import ceil

import torch

logger = logging.getLogger(__name__)


def _warmup_m_values(max_tokens: int) -> list[int]:
    """Dense set of M (token counts) covering every deep_gemm tile.

    A cubin's JIT key includes ``block_m``, ``block_n`` and the ``swap_ab`` flag,
    all chosen by a C++ heuristic over ``(M, N, num_sms[, num_groups])`` that
    deep_gemm does NOT expose to Python (``get_best_config`` lives in ``_C.so``).
    Modelling that selection from Python is unreliable -- the choice flips with M
    in ways that depend on ``swap_ab`` and (for batched GEMMs) the group count.

    Instead we sample M densely and let the real heuristic pick at each point,
    compiling whatever it selects. Because the serving path only ever runs
    ``M in [1, max_tokens]``, sampling that whole interval provably covers every
    reachable config. The tiling changes most often at small M (``swap_ab`` flips
    and small ``block_m``), so we step by 1 there and coarsen for large M where
    the selected tile is stable.

    Args:
        max_tokens: largest M (token count) the serving path can reach.
    """
    dense = min(max_tokens, 2048)
    values: set[int] = set(range(1, dense + 1))
    values.update(range(dense, max_tokens + 1, 16))
    values.add(max_tokens)
    return sorted(values)


# ---------------------------------------------------------------------------
# mega_moe JIT warmup
# ---------------------------------------------------------------------------


def warmup_mega_moe_jit(
    num_experts: int,
    max_num_tokens: int,
    top_k: int,
    hidden_size: int,
    device: torch.device,
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor],
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor],
    symm_buffer: object,
    activation_clamp: float | None = None,
) -> None:
    """Pre-compile ``fp8_fp4_mega_moe`` kernel tiles.

    Moved from ``DeepseekV4MegaMoEExperts.warmup_jit_variants``.
    The caller must issue a ``torch.distributed.barrier(group)`` before
    invoking this so all EP ranks enter together.

    All heavy objects (weights, symmetric buffer) must be passed in from
    the already-initialized model to avoid duplicate GPU allocations.
    """
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import fp8_fp4_mega_moe
    except ImportError:
        logger.warning("deep_gemm mega_moe symbols unavailable, skipping warmup")
        return

    token_counts = _warmup_m_values(max_num_tokens)
    logger.info(
        "Warming up mega_moe JIT: %d token counts up to %d",
        len(token_counts),
        max_num_tokens,
    )

    for num_tokens in token_counts:
        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device=device,
        )

        y = torch.empty_like(hidden_states)
        symm_buffer.x[:num_tokens].copy_(hidden_states.to(torch.float8_e4m3fn))
        symm_buffer.x_sf[:num_tokens].fill_(1.0)
        symm_buffer.topk_idx[:num_tokens].copy_(topk_ids)
        symm_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        fp8_fp4_mega_moe(
            y,
            transformed_l1_weights,
            transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
        )

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Prefill kernel JIT warmup (compressor / indexer / attention projections)
# ---------------------------------------------------------------------------


def warmup_prefill_jit(
    *,
    hidden_size: int,
    num_attention_heads: int,
    head_dim: int = 128,
    hc_mult: int = 0,
    kv_lora_rank: int = 0,
    index_n_heads: int = 0,
    index_head_dim: int = 0,
    indexer_cache_block_size: int = 64,
    max_decode_tokens: int = 256,
    mxfp4_block_size: int = 32,
    tp_size: int = 1,
    max_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile deep_gemm prefill/decode kernels for DeepSeek V4.

    Derives kernel shapes from model config values and warms up
    ``tf32_hc_prenorm_gemm`` (compressor), the ragged ``fp8_fp4_mqa_logits``
    (prefill indexer), and the paged ``fp8_fp4_paged_mqa_logits`` +
    ``get_paged_mqa_logits_metadata`` (decode indexer) kernels.

    Args:
        hidden_size: model hidden dimension.
        num_attention_heads: total attention heads (before TP split).
        head_dim: per-head dimension.
        hc_mult: compressor head-coupling multiplier (0 = no compressor).
        kv_lora_rank: KV LoRA rank for indexer (0 = no indexer).
        index_n_heads: sparse-indexer head count (0 = no indexer). The indexer
            projections are replicated (not TP-split), so this is the full
            per-rank head count, not ``num_attention_heads // tp_size``.
        index_head_dim: sparse-indexer per-head dim (FP4-packed in the cache).
        indexer_cache_block_size: paged indexer KV-cache block size (BLOCK_KV).
        max_decode_tokens: largest decode batch the server can run; the paged
            decode-indexer warmup sweeps every 32-aligned bucket up to this so
            no in-range decode batch JIT-compiles the metadata kernel inline.
        mxfp4_block_size: MXFP4 quantization block size.
        tp_size: attention tensor-parallel size.
        max_tokens: maximum prefill token count to warm up to.
        device: CUDA device.
    """
    warmup_count = 0

    if hc_mult and hc_mult > 1:
        hc_hidden_size = hc_mult * hidden_size
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        _warmup_tf32_hc_prenorm_gemm(
            [{"hc_hidden_size": hc_hidden_size, "mix_hc": mix_hc, "hc_dim": hc_dim}],
            max_tokens,
            device,
        )
        warmup_count += 1

    if index_n_heads > 0 and index_head_dim > 0:
        # Sparse-indexer kernels -- distinct cubins, not covered elsewhere:
        #   prefill: ragged  fp8_fp4_mqa_logits
        #   decode:  paged   fp8_fp4_paged_mqa_logits + get_paged_mqa_logits_metadata
        # The indexer projections are replicated (config.index_n_heads heads,
        # NOT TP-split), so gate on index_n_heads -- NOT kv_lora_rank, which is
        # absent for V4-Flash. Without these the kernels JIT-compile inline on
        # the first prefill/decode and stall the engine past the gRPC health
        # probe.
        _warmup_fp8_fp4_mqa_logits(
            num_heads=index_n_heads,
            index_head_dim=index_head_dim,
            device=device,
        )
        _warmup_fp8_fp4_paged_mqa_logits(
            num_heads=index_n_heads,
            index_head_dim=index_head_dim,
            cache_block_size=indexer_cache_block_size,
            max_decode_tokens=max_decode_tokens,
            device=device,
        )
        warmup_count += 1

    if warmup_count > 0:
        logger.info("Warmed up %d deep_gemm prefill kernel families", warmup_count)
        torch.cuda.synchronize()


def _compute_num_split(block_k: int, k: int, grid_size: int) -> int:
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    split_k = num_sms // grid_size
    num_block_k = ceil(k / block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _warmup_tf32_hc_prenorm_gemm(
    shapes: list[dict],
    max_tokens: int,
    device: torch.device,
) -> None:
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import tf32_hc_prenorm_gemm
    except ImportError:
        logger.warning("deep_gemm tf32_hc_prenorm_gemm unavailable, skipping")
        return

    seen: set[tuple[int, ...]] = set()
    block_k = 64
    block_m = 64

    for params in shapes:
        hc_hidden_size = params["hc_hidden_size"]
        mix_hc = params["mix_hc"]
        hc_dim = params["hc_dim"]

        if (hc_hidden_size, mix_hc) in seen:
            continue
        seen.add((hc_hidden_size, mix_hc))

        fn = torch.ones(mix_hc, hc_dim, dtype=torch.float32, device=device)

        token_counts = _warmup_m_values(max_tokens)
        for num_tokens in token_counts:
            grid_size = ceil(num_tokens / block_m)
            n_splits = _compute_num_split(block_k, hc_hidden_size, grid_size)

            x = torch.zeros(
                num_tokens,
                hc_hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            out_mul = torch.empty(
                n_splits,
                num_tokens,
                mix_hc,
                dtype=torch.float32,
                device=device,
            )
            out_sqrsum = torch.empty(
                n_splits,
                num_tokens,
                dtype=torch.float32,
                device=device,
            )
            tf32_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits)


def _warmup_fp8_fp4_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    device: torch.device,
    max_kv_len: int = 4096,
) -> None:
    """Pre-compile the ragged prefill sparse-indexer ``fp8_fp4_mqa_logits``.

    Mirrors the prefill call (deepseek_v4.py:1392): q = (int8 values, int32
    scales), kv = (gathered int8 values, int32 scales), ragged ``cu_seq_len``
    of length ``num_tokens`` (not +1). FP4 packs 2 values per byte, so the
    per-head value dim is ``index_head_dim // 2``. Batch and kv length are not
    JIT keys, so a couple of token counts suffice.

    Args:
        num_heads: sparse-indexer head count (``index_n_heads``).
        index_head_dim: sparse-indexer per-head dim (e.g. 128).
        device: CUDA device.
        max_kv_len: representative KV length to warm up to.
    """
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import fp8_fp4_mqa_logits
    except ImportError:
        logger.warning("deep_gemm fp8_fp4_mqa_logits unavailable, skipping")
        return

    head_dim_bytes = index_head_dim // 2
    for num_tokens in (1, 256):
        q_vals = torch.zeros(
            num_tokens, num_heads, head_dim_bytes, dtype=torch.uint8, device=device
        ).view(torch.int8)
        q_scales = torch.zeros(num_tokens, num_heads, dtype=torch.int32, device=device)
        k_vals = torch.zeros(
            max_kv_len, head_dim_bytes, dtype=torch.uint8, device=device
        ).view(torch.int8)
        k_scales = torch.zeros(max_kv_len, dtype=torch.int32, device=device)
        weights = torch.ones(num_tokens, num_heads, dtype=torch.float32, device=device)
        cu_start = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        cu_end = torch.full((num_tokens,), max_kv_len, dtype=torch.int32, device=device)

        fp8_fp4_mqa_logits(
            q=(q_vals, q_scales),
            kv=(k_vals, k_scales),
            weights=weights,
            cu_seq_len_k_start=cu_start,
            cu_seq_len_k_end=cu_end,
            clean_logits=False,
            max_seqlen_k=max_kv_len,
            logits_dtype=torch.float32,
        )


def _warmup_fp8_fp4_paged_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    cache_block_size: int,
    max_decode_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile the paged decode sparse-indexer kernels.

    Mirrors the decode call ``fp8_fp4_paged_mqa_logits`` and its schedule-
    metadata builder ``get_paged_mqa_logits_metadata``. These are distinct
    cubins from the ragged prefill ``fp8_fp4_mqa_logits`` and are not otherwise
    warmed. The paged logits kernel is keyed on (num_heads, head_dim, block_kv)
    -- not the batch size -- so one call warms it, but the metadata kernel is
    keyed on the 32-aligned decode batch size, so sweep every 32-aligned bucket
    up to the runtime decode-batch ceiling.

    Args:
        num_heads: sparse-indexer head count (``index_n_heads``).
        index_head_dim: sparse-indexer per-head dim (e.g. 128).
        cache_block_size: paged indexer KV-cache block size (BLOCK_KV).
        max_decode_tokens: largest decode batch the server can run (e.g.
            ``max_cudagraph_capture_size`` / ``max_num_seqs``). The metadata
            kernel is swept over every 32-aligned bucket up to this value so no
            in-range decode batch hits an uncompiled cubin.
        device: CUDA device.
    """
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            get_num_sms,
            get_paged_mqa_logits_metadata,
        )
    except ImportError:
        logger.warning("deep_gemm paged MQA logits unavailable, skipping")
        return

    # FP4 packs 2 values per byte; the paged KV row stores the value bytes plus
    # a single int32 scale (head_dim / 2 + sizeof(int)).
    head_dim_bytes = index_head_dim // 2
    row_bytes = index_head_dim // 2 + 4
    num_sms = get_num_sms()

    # The metadata kernel is JIT-keyed on the 32-aligned decode batch size, so
    # cover every bucket up to the runtime ceiling (batches < 32 map to the 32
    # bucket, so they are covered too).
    top_bucket = max(32, ((max_decode_tokens + 31) // 32) * 32)
    decode_batch_sizes = range(32, top_bucket + 1, 32)

    for num_tokens in decode_batch_sizes:
        num_blocks = max(1, num_tokens)
        q_values = torch.zeros(
            num_tokens, num_heads, head_dim_bytes, dtype=torch.uint8, device=device
        )
        q_scales = torch.zeros(num_tokens, num_heads, dtype=torch.int32, device=device)
        cache_2d = torch.zeros(
            num_blocks, cache_block_size * row_bytes, dtype=torch.uint8, device=device
        )
        kv_cache = torch.as_strided(
            cache_2d,
            (num_blocks, cache_block_size, 1, row_bytes),
            (cache_2d.stride(0), row_bytes, row_bytes, 1),
        )
        weights = torch.ones(num_tokens, num_heads, dtype=torch.float32, device=device)
        context_lens = torch.full(
            (num_tokens, 1), cache_block_size, dtype=torch.int32, device=device
        )
        block_table = torch.arange(num_tokens, dtype=torch.int32, device=device).view(
            num_tokens, 1
        )
        schedule_meta = get_paged_mqa_logits_metadata(
            context_lens, cache_block_size, num_sms
        )
        fp8_fp4_paged_mqa_logits(
            q=(q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
            kv_cache=kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_table,
            schedule_meta=schedule_meta,
            max_context_len=cache_block_size,
            clean_logits=False,
            logits_dtype=torch.float32,
        )
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# FP8 GEMM warmup (attention / compressor linear projections)
# ---------------------------------------------------------------------------


def warmup_fp8_gemm_nt(
    shapes: list[tuple[int, int]],
    max_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile ``fp8_gemm_nt`` for FP8 block-scaled linear layers.

    Args:
        shapes: (N, K) weight shapes for layers using deep_gemm FP8.
        max_tokens: maximum prefill token count to warm up to.
        device: CUDA device.
    """
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import fp8_gemm_nt
    except ImportError:
        logger.warning("deep_gemm fp8_gemm_nt unavailable, skipping warmup")
        return

    block_size = 128
    seen: set[tuple[int, int]] = set()

    for n, k in shapes:
        if (n, k) in seen:
            continue
        seen.add((n, k))

        a = torch.zeros(max_tokens, k, dtype=torch.float8_e4m3fn, device=device)
        a_scales = torch.ones(
            max_tokens, k // block_size, dtype=torch.float32, device=device
        )
        b = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device=device)
        b_scales = torch.ones(
            n // block_size, k // block_size, dtype=torch.float32, device=device
        )
        out = torch.empty(max_tokens, n, dtype=torch.bfloat16, device=device)

        token_counts = _warmup_m_values(max_tokens)
        for num_tokens in token_counts:
            fp8_gemm_nt(
                (a[:num_tokens], a_scales[:num_tokens]), (b, b_scales), out[:num_tokens]
            )

        del a, a_scales, b, b_scales, out

    logger.info("Warmed up fp8_gemm_nt for %d weight shapes", len(seen))
    torch.cuda.synchronize()


def warmup_fp8_einsum(
    bmm_layers: list[tuple[torch.Tensor, torch.Tensor, int, int]],
    max_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile ``fp8_einsum("bhr,hdr->bhd")`` for is_bmm output projections.

    The attention output projection (V4 ``wo_a``) runs a per-group *batched* FP8
    GEMM via ``deep_gemm.fp8_einsum`` -- a distinct ``GemmType::Batched`` path that
    ``warmup_fp8_gemm_nt`` does NOT cover. Its ``block_m`` grows with the prefill
    token count M, so serving JITs new (N, K, block_m) tiles inline unless we
    sweep M offline. The activation operand is laid out exactly as
    ``deepseek_v4_fused_inv_rope_fp8_quant`` returns it (per-group, transposed,
    TMA-aligned INT32 UE8M0 scales); only its values are dummy -- the kernel's
    JIT key is shape-only.

    Args:
        bmm_layers: ``(weight, weight_scale_inv, n_groups, block_n)`` tuples taken
            from the loaded model (real weights/scales are reused so no scale
            layout is guessed). ``weight`` is ``[n_groups * N, K]``; per group the
            GEMM is ``(M, K) x (N, K) -> (M, N)``.
        max_tokens: maximum prefill token count to warm up to.
        device: CUDA device.
    """
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import fp8_einsum
    except ImportError:
        logger.warning("deep_gemm fp8_einsum unavailable, skipping bmm warmup")
        return

    for weight, weight_scale_inv, n_groups, block_n in bmm_layers:
        in_dim = weight.shape[1]  # K (== r == per-group quant dim)
        o_lora_rank = weight.shape[0] // n_groups  # N (per-group output)
        w = weight.view(n_groups, o_lora_rank, in_dim)
        recipe = (1, 1, block_n)
        num_scale_blocks = in_dim // block_n

        # N drives the batched GEMM's block_m heuristic, so sweep M against it.
        for num_tokens in _warmup_m_values(max_tokens):
            tma_aligned_t = ((num_tokens + 3) // 4) * 4
            scale_inner = (num_scale_blocks + 3) // 4  # tma-aligned INT32 scales
            o_fp8 = torch.zeros(
                (n_groups, num_tokens, in_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            ).transpose(0, 1)
            o_scale = (
                torch.zeros(
                    n_groups * scale_inner * tma_aligned_t,
                    dtype=torch.int32,
                    device=device,
                )
                .as_strided(
                    (n_groups, num_tokens, scale_inner),
                    (scale_inner * tma_aligned_t, 1, tma_aligned_t),
                )
                .transpose(0, 1)
            )
            z = torch.empty(
                (num_tokens, n_groups, o_lora_rank),
                dtype=torch.bfloat16,
                device=device,
            )
            fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8, o_scale),
                (w, weight_scale_inv),
                z,
                recipe=recipe,
            )

    logger.info("Warmed up fp8_einsum for %d bmm shapes", len(bmm_layers))
    torch.cuda.synchronize()


def warmup_fp8_gemm_nt_from_model(
    model: torch.nn.Module,
    max_tokens: int = 8192,
) -> None:
    """Scan a model for deep_gemm FP8 linear layers and warm their JIT tiles.

    Collects (N, K) weight shapes from all modules where
    ``_use_deep_gemm_fp8=True`` (set by ``Fp8LinearMethod.process_weights_after_loading``):
    plain projections are warmed via ``fp8_gemm_nt`` and is_bmm projections
    (V4 ``wo_a``) via ``fp8_einsum`` -- both grow ``block_m`` with M, so both must
    be swept offline or they JIT inline on the first long prefill.

    Call after ``quant_method.process_weights_after_loading()`` has run on all
    modules so the ``_use_deep_gemm_fp8`` flag is set.
    """
    if torch.cuda.get_device_capability()[0] < 10:
        return
    shapes: set[tuple[int, int]] = set()
    bmm_layers: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    bmm_seen: set[tuple] = set()
    for module in model.modules():
        if not getattr(module, "_use_deep_gemm_fp8", False):
            continue
        if getattr(module, "is_bmm", False):
            n_groups = getattr(module, "bmm_batch_size", 0)
            block_size = getattr(module, "_deep_gemm_block_size", None)
            if not n_groups or not block_size:
                continue
            key = (tuple(module.weight.shape), n_groups, block_size[0])
            if key in bmm_seen:
                continue
            bmm_seen.add(key)
            bmm_layers.append(
                (module.weight, module.weight_scale_inv, n_groups, block_size[0])
            )
        else:
            n, k = module.weight.shape
            shapes.add((n, k))
    if not shapes and not bmm_layers:
        return
    device = next(model.parameters()).device
    if shapes:
        logger.info("Pre-compiling %d deep_gemm FP8 GEMM shapes...", len(shapes))
        warmup_fp8_gemm_nt(list(shapes), max_tokens, device)
    if bmm_layers:
        logger.info(
            "Pre-compiling %d deep_gemm FP8 einsum (bmm) shapes...", len(bmm_layers)
        )
        warmup_fp8_einsum(bmm_layers, max_tokens, device)
