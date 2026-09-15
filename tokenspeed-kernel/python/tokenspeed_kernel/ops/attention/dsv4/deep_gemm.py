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

from __future__ import annotations

import logging
from math import ceil

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
    prepare_cuda_toolkit_env,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_IS_HOPPER_PLUS = current_platform().is_hopper_plus
logger = logging.getLogger(__name__)

if _IS_HOPPER_PLUS:
    prepare_cuda_toolkit_env()
    import deep_ep  # noqa: F401
    import deep_gemm
    import trtllm_kernel  # noqa: F401
    from tokenspeed_kernel.ops._deep_gemm.mega_moe_bf16 import (
        prepare_mega_moe_bf16_jit,
    )
    from tokenspeed_kernel.ops.attention.dsv4.cuda import (
        has_indexer_mxfp4_paged_gather,
        has_indexer_topk_prefill,
        has_persistent_topk,
        indexer_mxfp4_paged_gather,
        indexer_topk_prefill,
        persistent_topk,
    )

    prepare_mega_moe_bf16_jit()

_MXFP4_BLOCK_SIZE = 32
_MXFP4_VALUE_BYTES_PER_BLOCK = _MXFP4_BLOCK_SIZE // 2
_MXFP4_SCALE_BYTES_PER_BLOCK = 1
_FP8_BLOCK_SIZE = 128
_FP8_SCALE_BYTES_PER_BLOCK = 4
_PERSISTENT_TOPK_WORKSPACE_BYTES = 1024 * 1024


def _allocate_topk(
    out: torch.Tensor | None,
    *,
    tokens: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    if out is None:
        return torch.empty((tokens, topk), dtype=torch.int32, device=device)
    if (
        out.ndim != 2
        or out.shape[0] < tokens
        or out.shape[1] != topk
        or out.dtype != torch.int32
        or out.device != device
    ):
        raise ValueError(
            "out must be int32 with at least shape "
            f"({tokens}, {topk}) on {device}, got "
            f"{tuple(out.shape)} {out.dtype} {out.device}"
        )
    return out[:tokens]


def _cache_row_bytes(cache_2d: torch.Tensor, page_size: int) -> int:
    if cache_2d.ndim != 2 or cache_2d.dtype != torch.uint8:
        raise ValueError(
            "index_k_cache must be a 2-D uint8 page matrix, got "
            f"{tuple(cache_2d.shape)} {cache_2d.dtype}"
        )
    if page_size <= 0 or cache_2d.shape[1] % page_size != 0:
        raise ValueError(
            "index-K cache row size must be divisible by page_size, got "
            f"shape={tuple(cache_2d.shape)}, page_size={page_size}"
        )
    return cache_2d.shape[1] // page_size


def _mxfp4_layout(cache_2d: torch.Tensor, page_size: int) -> tuple[int, int]:
    row_bytes = _cache_row_bytes(cache_2d, page_size)
    bytes_per_block = _MXFP4_VALUE_BYTES_PER_BLOCK + _MXFP4_SCALE_BYTES_PER_BLOCK
    if row_bytes % bytes_per_block != 0:
        raise ValueError(f"invalid MXFP4 index-K row size: {row_bytes} bytes")
    blocks = row_bytes // bytes_per_block
    return (
        blocks * _MXFP4_VALUE_BYTES_PER_BLOCK,
        blocks * _MXFP4_SCALE_BYTES_PER_BLOCK,
    )


def _fp8_layout(cache_2d: torch.Tensor, page_size: int) -> tuple[int, int]:
    row_bytes = _cache_row_bytes(cache_2d, page_size)
    bytes_per_block = _FP8_BLOCK_SIZE + _FP8_SCALE_BYTES_PER_BLOCK
    if row_bytes % bytes_per_block != 0:
        raise ValueError(f"invalid scaled-FP8 index-K row size: {row_bytes} bytes")
    blocks = row_bytes // bytes_per_block
    return blocks * _FP8_BLOCK_SIZE, blocks * _FP8_SCALE_BYTES_PER_BLOCK


def _mxfp4_cache_view(cache_2d: torch.Tensor, page_size: int) -> torch.Tensor:
    row_bytes = _cache_row_bytes(cache_2d, page_size)
    return torch.as_strided(
        cache_2d,
        (cache_2d.shape[0], page_size, 1, row_bytes),
        (cache_2d.stride(0), row_bytes, row_bytes, 1),
    )


def _gather_paged_mxfp4(
    cache_2d: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    page_size: int,
    workspace: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_bytes, scale_bytes = _mxfp4_layout(cache_2d, page_size)
    if workspace is None:
        rows = int(cu_seq_lens[-1].item()) if cu_seq_lens.numel() else 0
        values = torch.empty(
            (rows, value_bytes), dtype=torch.uint8, device=cache_2d.device
        )
        scales = torch.empty(
            (rows, scale_bytes), dtype=torch.uint8, device=cache_2d.device
        )
    else:
        values, scales = workspace
        if values.shape[0] != scales.shape[0]:
            raise ValueError(
                "MXFP4 gather workspace value/scale rows must match, got "
                f"values={values.shape[0]}, scales={scales.shape[0]}"
            )
        if (
            values.ndim != 2
            or scales.ndim != 2
            or values.shape[1] != value_bytes
            or scales.shape[1] != scale_bytes
            or values.dtype != torch.uint8
            or scales.dtype != torch.uint8
            or values.device != cache_2d.device
            or scales.device != cache_2d.device
        ):
            raise ValueError(
                "MXFP4 gather workspace has an incompatible shape, dtype, or device"
            )
        rows = values.shape[0]

    if rows == 0:
        return values.view(torch.int8), scales.view(torch.int32).squeeze(-1)
    if not (cache_2d.is_cuda and block_table.is_cuda and cu_seq_lens.is_cuda):
        raise RuntimeError(
            "DeepSeek V4 paged MXFP4 gather requires cache, block table, and "
            "sequence lengths on CUDA"
        )
    if not has_indexer_mxfp4_paged_gather():
        raise RuntimeError("DeepSeek V4 paged MXFP4 gather kernel is unavailable")
    indexer_mxfp4_paged_gather(
        kv_cache=cache_2d,
        values_out=values,
        scales_out=scales,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
        cache_block_size=page_size,
    )
    return values.view(torch.int8), scales.view(torch.int32).squeeze(-1)


def _gather_paged_fp8(
    cache_2d: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim, scale_bytes = _fp8_layout(cache_2d, page_size)
    device = cache_2d.device
    cu_seq_lens_i64 = cu_seq_lens.to(device=device, dtype=torch.int64)
    rows = int(cu_seq_lens_i64[-1].item()) if cu_seq_lens_i64.numel() else 0
    if rows == 0:
        return (
            torch.empty((0, head_dim), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    page_table_i64 = block_table.to(device=device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64)
    req = torch.searchsorted(cu_seq_lens_i64[1:].contiguous(), row_ids, right=True)
    req = req.clamp_max(page_table_i64.shape[0] - 1)
    local = row_ids - cu_seq_lens_i64[req]
    logical_page = torch.div(local, page_size, rounding_mode="floor")
    logical_page = logical_page.clamp_max(page_table_i64.shape[1] - 1)
    in_page = local % page_size
    physical_page = page_table_i64[req, logical_page]
    value_offsets = (
        in_page[:, None] * head_dim
        + torch.arange(head_dim, device=device, dtype=torch.int64)[None, :]
    )
    scale_offsets = (
        page_size * head_dim
        + in_page[:, None] * scale_bytes
        + torch.arange(scale_bytes, device=device, dtype=torch.int64)[None, :]
    )
    values = cache_2d[physical_page[:, None], value_offsets]
    scales = cache_2d[physical_page[:, None], scale_offsets]
    return (
        values.contiguous().view(torch.float8_e4m3fn),
        scales.contiguous().view(torch.float32).reshape(rows),
    )


def _prefill_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    out: torch.Tensor,
) -> torch.Tensor:
    lengths = lengths.to(device=logits.device, dtype=torch.int32).reshape(-1)
    out = out[: lengths.numel()]
    out.fill_(-1)
    if lengths.numel() == 0 or logits.shape[1] == 0:
        return out
    row_starts = torch.zeros_like(lengths)
    if has_indexer_topk_prefill():
        indexer_topk_prefill(logits, row_starts, lengths, out, topk)
        return out
    trtllm_ops = getattr(torch.ops, "trtllm", None)
    if trtllm_ops is None or not hasattr(trtllm_ops, "indexer_topk_prefill"):
        raise RuntimeError("DeepSeek V4 prefill top-k kernel is unavailable")
    trtllm_ops.indexer_topk_prefill(
        logits.contiguous(), row_starts, lengths.contiguous(), out, topk
    )
    return out


def _trtllm_decode_topk(
    values: torch.Tensor,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
) -> None:
    seq_lens = seq_lens.to(torch.int32).reshape(-1).contiguous()
    torch.ops.trtllm.indexer_topk_decode(values, seq_lens, indices, 1, topk)


def _decode_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    out: torch.Tensor,
    workspace: torch.Tensor | None,
) -> torch.Tensor:
    if topk not in (512, 1024, 2048):
        raise RuntimeError(
            "DeepSeek V4 decode top-k supports topk in {512, 1024, 2048}"
        )
    lengths = lengths.to(device=logits.device, dtype=torch.int32).contiguous()
    out.fill_(-1)
    if lengths.numel() == 0 or logits.shape[1] == 0:
        return out
    if (
        workspace is not None
        and workspace.is_cuda
        and workspace.device == logits.device
        and workspace.dtype == torch.uint8
        and workspace.numel() >= _PERSISTENT_TOPK_WORKSPACE_BYTES
    ):
        if not has_persistent_topk():
            raise RuntimeError(
                "DeepSeek V4 persistent top-k workspace was provided, but the "
                "persistent top-k kernel is unavailable"
            )
        persistent_topk(
            logits.contiguous(), lengths, out, workspace, topk, logits.shape[1]
        )
        return out
    _trtllm_decode_topk(logits.contiguous(), lengths, out, topk)
    return out


def _dsv4_prefill_topk(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_seqlen_k_start: torch.Tensor,
    cu_seqlen_k_end: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    max_seqlen_k: int,
    index_k_format: str,
    gathered_k: tuple[torch.Tensor, torch.Tensor] | None = None,
    gather_workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    q_values, q_scales = index_q
    result = _allocate_topk(
        out, tokens=q_values.shape[0], topk=topk, device=q_values.device
    )
    result.fill_(-1)
    if q_values.shape[0] == 0 or max_seqlen_k <= 0:
        return result, gathered_k
    if deep_gemm.get_pdl() != pdl_enabled():
        deep_gemm.set_pdl(pdl_enabled())

    if index_k_format == "mxfp4":
        if gathered_k is None:
            gathered_k = _gather_paged_mxfp4(
                index_k_cache,
                block_table,
                cu_seq_lens,
                page_size,
                gather_workspace,
            )
        k_values, k_scales = gathered_k
        logits = deep_gemm.fp8_fp4_mqa_logits(
            q=(q_values.contiguous().view(torch.int8), q_scales.contiguous()),
            kv=(k_values.contiguous(), k_scales.contiguous()),
            weights=weights.contiguous(),
            cu_seq_len_k_start=cu_seqlen_k_start,
            cu_seq_len_k_end=cu_seqlen_k_end,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
            logits_dtype=torch.float32,
        )
    else:
        if gathered_k is None:
            gathered_k = _gather_paged_fp8(
                index_k_cache, block_table, cu_seq_lens, page_size
            )
        logits = deep_gemm.fp8_mqa_logits(
            q_values.contiguous(),
            (gathered_k[0].contiguous(), gathered_k[1].contiguous()),
            weights.contiguous(),
            cu_seqlen_k_start,
            cu_seqlen_k_end,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
        )
    return _prefill_topk(logits, seq_lens, topk, result), gathered_k


def _dsv4_decode_topk(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    max_context_len: int,
    plan: object,
    index_k_format: str,
    out: torch.Tensor | None = None,
    persistent_topk_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if deep_gemm.get_pdl() != pdl_enabled():
        deep_gemm.set_pdl(pdl_enabled())
    q_values, q_scales = index_q
    result = _allocate_topk(
        out, tokens=q_values.shape[0], topk=topk, device=q_values.device
    )
    result.fill_(-1)
    if q_values.shape[0] == 0 or max_context_len <= 0:
        return result
    if plan is None:
        raise RuntimeError(
            "DeepSeek V4 decode top-k requires a plan returned by dsv4_plan"
        )

    kv_cache = _mxfp4_cache_view(index_k_cache, page_size)
    if index_k_format == "mxfp4":
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            q=(
                q_values.contiguous().view(torch.int8).unsqueeze(1),
                q_scales.contiguous().unsqueeze(1),
            ),
            kv_cache=kv_cache,
            weights=weights.contiguous(),
            context_lens=context_lens,
            block_table=block_table,
            schedule_meta=plan,
            max_context_len=max_context_len,
            clean_logits=False,
            logits_dtype=torch.float32,
        )
    else:
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_values.contiguous().unsqueeze(1),
            kv_cache,
            weights.contiguous(),
            context_lens,
            block_table,
            plan,
            max_context_len,
            clean_logits=False,
        )
    return _decode_topk(logits, context_lens, topk, result, persistent_topk_workspace)


_SIGNATURES = {
    "mxfp4": format_signature(
        q=dense_tensor_format(torch.uint8),
        weights=dense_tensor_format(torch.float32),
        index_k_cache=dense_tensor_format(torch.uint8),
    ),
    "fp8_scaled": format_signature(
        q=dense_tensor_format(torch.float8_e4m3fn),
        weights=dense_tensor_format(torch.float32),
        index_k_cache=dense_tensor_format(torch.uint8),
    ),
}


def _register(format_name: str, min_arch: ArchVersion) -> None:
    common = dict(
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=min_arch,
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({_SIGNATURES[format_name]}),
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({format_name}),
        },
        priority=Priority.SPECIALIZED,
        tags={"nvidia", "sparse", "latency"},
    )
    register_kernel(
        "attention",
        "dsv4_prefill_topk",
        name=f"deep_gemm_dsv4_{format_name}_prefill_topk",
        **common,
    )(_dsv4_prefill_topk)
    register_kernel(
        "attention",
        "dsv4_decode_topk",
        name=f"deep_gemm_dsv4_{format_name}_decode_topk",
        **{
            **common,
            "traits": {
                **common["traits"],
                "topk": frozenset({512, 1024, 2048}),
            },
        },
    )(_dsv4_decode_topk)


def _warmup_m_values(max_tokens: int) -> list[int]:
    """Return token counts covering every DeepGEMM tile reachable at runtime."""
    dense = min(max_tokens, 2048)
    values: set[int] = set(range(1, dense + 1))
    values.update(range(dense, max_tokens + 1, 16))
    values.add(max_tokens)
    return sorted(values)


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
        for num_tokens in _warmup_m_values(max_tokens):
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
            deep_gemm.tf32_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits)


def _warmup_fp8_fp4_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    device: torch.device,
    max_kv_len: int,
) -> None:
    """Pre-compile the ragged prefill sparse-indexer kernel."""
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

        deep_gemm.fp8_fp4_mqa_logits(
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
    """Pre-compile paged decode indexer and schedule-metadata kernels."""
    head_dim_bytes = index_head_dim // 2
    row_bytes = head_dim_bytes + 4
    num_sms = deep_gemm.get_num_sms()
    top_bucket = max(32, ((max_decode_tokens + 31) // 32) * 32)

    for num_tokens in range(32, top_bucket + 1, 32):
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
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, cache_block_size, num_sms
        )
        deep_gemm.fp8_fp4_paged_mqa_logits(
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


def warmup_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    cache_block_size: int,
    max_decode_tokens: int,
    device: torch.device,
) -> None:
    """Compile packed MQA scoring and paged metadata kernels before capture."""
    _warmup_fp8_fp4_mqa_logits(
        num_heads=num_heads,
        index_head_dim=index_head_dim,
        device=device,
        max_kv_len=4096,
    )
    _warmup_fp8_fp4_paged_mqa_logits(
        num_heads=num_heads,
        index_head_dim=index_head_dim,
        cache_block_size=cache_block_size,
        max_decode_tokens=max_decode_tokens,
        device=device,
    )


def _warmup_prefill_jit(
    *,
    hidden_size: int,
    num_attention_heads: int,
    head_dim: int,
    hc_mult: int,
    kv_lora_rank: int,
    index_n_heads: int,
    index_head_dim: int,
    indexer_cache_block_size: int,
    max_decode_tokens: int,
    mxfp4_block_size: int,
    tp_size: int,
    max_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile DeepSeek V4 compressor and indexer kernel families."""
    del num_attention_heads, head_dim, kv_lora_rank, mxfp4_block_size, tp_size
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
        _warmup_fp8_fp4_mqa_logits(
            num_heads=index_n_heads,
            index_head_dim=index_head_dim,
            device=device,
            max_kv_len=4096,
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


def deep_gemm_dsv4_warmup(**kwargs) -> None:
    if deep_gemm.get_pdl() != pdl_enabled():
        deep_gemm.set_pdl(pdl_enabled())
    _warmup_prefill_jit(**kwargs)


if _IS_HOPPER_PLUS:

    @register_kernel(
        "attention",
        "dsv4_plan",
        name="deep_gemm_dsv4_plan",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature()}),
        traits={"page_size": frozenset({64})},
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_dsv4_plan(
        *,
        page_size: int,
        seq_lens_2d: torch.Tensor,
        out: object | None = None,
    ) -> torch.Tensor:
        if deep_gemm.get_pdl() != pdl_enabled():
            deep_gemm.set_pdl(pdl_enabled())
        refreshed = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens_2d,
            page_size,
            deep_gemm.get_num_sms(),
        )
        if out is None:
            with torch.inference_mode(False):
                return refreshed.clone()
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != refreshed.shape
            or out.device != refreshed.device
            or out.dtype != refreshed.dtype
        ):
            actual = (
                f"{tuple(out.shape)} {out.dtype} {out.device}"
                if isinstance(out, torch.Tensor)
                else type(out).__name__
            )
            raise RuntimeError(
                "DeepSeek V4 decode indexer plan changed shape during CUDA graph "
                "replay; recapture or use eager for this batch. "
                f"captured={actual}, refreshed={tuple(refreshed.shape)} "
                f"{refreshed.dtype} {refreshed.device}"
            )
        with torch.inference_mode():
            out.copy_(refreshed)
        return out

    _register("fp8_scaled", ArchVersion(9, 0))
    _register("mxfp4", ArchVersion(10, 0))
    register_kernel(
        "attention",
        "dsv4_warmup",
        name="deep_gemm_dsv4_warmup",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature()}),
        traits={},
        priority=Priority.SPECIALIZED,
    )(deep_gemm_dsv4_warmup)
