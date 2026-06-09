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


def _token_count_sweep(max_tokens: int) -> list[int]:
    """Token counts covering deep_gemm block_m tile boundaries.

    deep_gemm compiles one cubin per unique (N, K, block_m) combination.
    The block_m tile is selected by a heuristic based on M and the number
    of SMs. We only need one representative M per block_m to trigger
    compilation — sweeping more M values just wastes execution time.

    Block_m values range from 16 to 256 in steps of 16 (15 variants).
    We pick one M per block_m step plus a few small values for decode-like
    paths, keeping the total under 20 calls per weight shape.
    """
    values: set[int] = set()
    values.update([1, 2, 4, 8])
    for block_m in range(16, 257, 16):
        if block_m <= max_tokens:
            values.add(block_m)
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

    token_counts = _token_count_sweep(max_num_tokens)
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
    mxfp4_block_size: int = 32,
    tp_size: int = 1,
    max_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile deep_gemm prefill kernels for DeepSeek V4.

    Derives kernel shapes from model config values and warms up
    ``tf32_hc_prenorm_gemm`` (compressor) and ``fp8_fp4_mqa_logits``
    (indexer) kernels for representative token counts.

    Args:
        hidden_size: model hidden dimension.
        num_attention_heads: total attention heads (before TP split).
        head_dim: per-head dimension.
        hc_mult: compressor head-coupling multiplier (0 = no compressor).
        kv_lora_rank: KV LoRA rank for indexer (0 = no indexer).
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

    if kv_lora_rank > 0:
        num_heads = num_attention_heads // tp_size
        _warmup_fp8_fp4_mqa_logits(
            [
                {
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "kv_dim": kv_lora_rank,
                    "kv_scale_dim": kv_lora_rank // mxfp4_block_size,
                    "max_kv_len": 4096,
                }
            ],
            max_tokens,
            device,
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
    token_counts = _token_count_sweep(max_tokens)
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
    shapes: list[dict],
    max_tokens: int,
    device: torch.device,
) -> None:
    try:
        from tokenspeed_kernel.thirdparty.deep_gemm import fp8_fp4_mqa_logits
    except ImportError:
        logger.warning("deep_gemm fp8_fp4_mqa_logits unavailable, skipping")
        return

    token_counts = _token_count_sweep(max_tokens)

    for params in shapes:
        num_heads = params["num_heads"]
        head_dim = params["head_dim"]
        kv_dim = params["kv_dim"]
        kv_scale_dim = params["kv_scale_dim"]
        max_kv_len = params.get("max_kv_len", 4096)

        for num_tokens in token_counts:
            q_vals = torch.zeros(
                num_tokens,
                num_heads,
                head_dim,
                dtype=torch.int8,
                device=device,
            )
            q_scales = torch.ones(
                num_tokens,
                num_heads,
                dtype=torch.float32,
                device=device,
            )
            k_vals = torch.zeros(
                max_kv_len,
                kv_dim,
                dtype=torch.int8,
                device=device,
            )
            k_scales = torch.ones(
                max_kv_len,
                kv_scale_dim,
                dtype=torch.uint8,
                device=device,
            )
            weights = torch.ones(
                num_tokens,
                dtype=torch.float32,
                device=device,
            )
            cu_start = torch.zeros(
                num_tokens + 1,
                dtype=torch.int32,
                device=device,
            )
            cu_end = torch.full(
                (num_tokens + 1,),
                max_kv_len,
                dtype=torch.int32,
                device=device,
            )

            fp8_fp4_mqa_logits(
                q=(q_vals, q_scales),
                kv=(k_vals, k_scales),
                weights=weights,
                cu_seq_len_k_start=cu_start,
                cu_seq_len_k_end=cu_end,
                clean_logits=True,
                max_seqlen_k=max_kv_len,
                logits_dtype=torch.float32,
            )


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
    token_counts = _token_count_sweep(max_tokens)

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

        for num_tokens in token_counts:
            fp8_gemm_nt(
                (a[:num_tokens], a_scales[:num_tokens]), (b, b_scales), out[:num_tokens]
            )

        del a, a_scales, b, b_scales, out

    logger.info("Warmed up fp8_gemm_nt for %d weight shapes", len(seen))
    torch.cuda.synchronize()


def warmup_fp8_gemm_nt_from_model(
    model: torch.nn.Module,
    max_tokens: int = 8192,
) -> None:
    """Scan a model for deep_gemm FP8 linear layers and warm up ``fp8_gemm_nt``.

    Collects (N, K) weight shapes from all modules where
    ``_use_deep_gemm_fp8=True`` (set by ``Fp8LinearMethod.process_weights_after_loading``),
    excluding BMM layers (handled by ``fp8_einsum``).

    Call after ``quant_method.process_weights_after_loading()`` has run on all
    modules so the ``_use_deep_gemm_fp8`` flag is set.
    """
    if torch.cuda.get_device_capability()[0] < 10:
        return
    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if getattr(module, "_use_deep_gemm_fp8", False) and not getattr(
            module, "is_bmm", False
        ):
            n, k = module.weight.shape
            shapes.add((n, k))
    if not shapes:
        return
    device = next(model.parameters()).device
    logger.info("Pre-compiling %d deep_gemm FP8 GEMM shapes...", len(shapes))
    warmup_fp8_gemm_nt(list(shapes), max_tokens, device)
