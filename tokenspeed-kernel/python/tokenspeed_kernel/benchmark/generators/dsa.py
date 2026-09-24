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

import math
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "prepare_dsa_decode",
    "prepare_dsa_prefill",
    "prepare_kpool_decode_append",
    "prepare_kpool_decode_topk",
    "prepare_kpool_prefill_topk",
    "prepare_kpool_prefill_write",
]


_IMPLEMENTED_DTYPES = {
    "bfloat16": torch.bfloat16,
}
_IMPLEMENTED_KV_CACHE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
_IMPLEMENTED_MODEL_PROFILES = frozenset({"glm53_flash_tp4"})
_POOL_PERMUTATION_STRIDE = 1_000_003
_TOKEN_PERMUTATION_STRIDE = 7_919


@dataclass(frozen=True)
class _DSAConfig:
    model_profile: str
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    index_heads: int
    index_head_dim: int
    local_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    pool_size: int
    index_page_stride_bytes: int
    kv_page_size: int
    topk_tokens: int
    max_context: int
    max_logits_bytes: int

    @property
    def index_rows_per_page(self) -> int:
        return self.kv_page_size // self.pool_size

    @property
    def index_scale_groups(self) -> int:
        return self.index_head_dim // 128

    @property
    def topk_pools(self) -> int:
        return self.topk_tokens // self.pool_size

    @property
    def selected_width(self) -> int:
        return self.topk_tokens + self.pool_size - 1

    @property
    def qk_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kpool_softmax_scale(self) -> float:
        return self.index_head_dim**-0.5 * self.index_heads**-0.5

    @property
    def dsa_softmax_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


def _implemented_value(
    name: str,
    value: object,
    implemented: Collection[str],
) -> str:
    if value not in implemented:
        accepted = ", ".join(sorted(implemented))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Implemented DSA {name} values: {accepted}",
        )
    return value


def _parse_dtype(
    name: str,
    value: object,
    implemented: dict[str, torch.dtype],
) -> torch.dtype:
    dtype_name = _implemented_value(name, value, implemented)
    return implemented[dtype_name]


def _validate_config(config: _DSAConfig) -> None:
    if config.index_head_dim % 128:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "DSA index_head_dim must be divisible by 128",
        )
    if config.topk_tokens % config.pool_size:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "DSA topk_tokens must be divisible by pool_size",
        )
    if config.kv_page_size % config.pool_size:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "DSA kv_page_size must be divisible by pool_size",
        )
    value_bytes = config.index_rows_per_page * config.index_head_dim
    scale_bytes = (
        config.index_rows_per_page * config.index_scale_groups * torch.float32.itemsize
    )
    if (
        config.index_page_stride_bytes < value_bytes + scale_bytes
        or config.index_page_stride_bytes % torch.float32.itemsize
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "DSA index_page_stride_bytes must fit and align the packed index payload",
        )


def _resolve_config(
    request: BenchmarkRequest,
) -> _DSAConfig:
    if request.parameters.get("validation") is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "DSA benchmark correctness validation is not implemented yet",
        )
    parameters = request.parameters
    config = _DSAConfig(
        model_profile=_implemented_value(
            "model_profile",
            parameters["model_profile"],
            _IMPLEMENTED_MODEL_PROFILES,
        ),
        dtype=_parse_dtype("dtype", parameters["dtype"], _IMPLEMENTED_DTYPES),
        kv_cache_dtype=_parse_dtype(
            "kv_cache_dtype",
            parameters["kv_cache_dtype"],
            _IMPLEMENTED_KV_CACHE_DTYPES,
        ),
        index_heads=parameters["index_heads"],
        index_head_dim=parameters["index_head_dim"],
        local_attention_heads=parameters["local_attention_heads"],
        kv_lora_rank=parameters["kv_lora_rank"],
        qk_nope_head_dim=parameters["qk_nope_head_dim"],
        qk_rope_head_dim=parameters["qk_rope_head_dim"],
        pool_size=parameters["pool_size"],
        index_page_stride_bytes=parameters["index_page_stride_bytes"],
        kv_page_size=parameters["kv_page_size"],
        topk_tokens=parameters["topk_tokens"],
        max_context=parameters["max_context"],
        max_logits_bytes=parameters["max_logits_bytes"],
    )
    _validate_config(config)
    return config


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)


def _select_registration(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    *,
    signature_roles: dict[str, torch.dtype],
    traits: dict[str, object],
) -> KernelSpec:
    signature = format_signature(
        **{role: dense_tensor_format(dtype) for role, dtype in signature_roles.items()}
    )
    try:
        selected = select_kernel(
            request.family,
            request.mode,
            signature,
            platform=platform,
            traits=traits,
            solution=request.solution,
            override=request.registration,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(
            BenchmarkStatus.NOT_APPLICABLE,
            str(error),
        ) from error
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec


def _packed_index_cache(
    pages: int,
    *,
    config: _DSAConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_bytes = config.index_rows_per_page * config.index_head_dim
    scale_bytes = (
        config.index_rows_per_page * config.index_scale_groups * torch.float32.itemsize
    )
    storage = torch.empty(
        (pages, config.index_page_stride_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    cache = storage[:, : value_bytes + scale_bytes]
    values = (
        cache[:, :value_bytes]
        .view(torch.float8_e4m3fn)
        .view(pages, config.index_rows_per_page, config.index_head_dim)
    )
    scales = (
        cache[:, value_bytes:]
        .view(torch.float32)
        .view(pages, config.index_rows_per_page, config.index_scale_groups)
    )
    source = _randn(
        (pages, config.index_rows_per_page, config.index_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    source_groups = source.float().view(
        pages,
        config.index_rows_per_page,
        config.index_scale_groups,
        128,
    )
    scale_values = source_groups.abs().amax(dim=-1).clamp_min(1e-4)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale_values.div_(fp8_max)
    values.copy_(
        (source_groups / scale_values.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .view_as(source)
    )
    scales.copy_(scale_values)
    return cache, values, scales


def _page_tables(
    batch: int,
    sequence_length: int,
    *,
    config: _DSAConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pages_per_request = max(1, math.ceil(sequence_length / config.kv_page_size))
    table = torch.arange(
        batch * pages_per_request,
        dtype=torch.int32,
        device=device,
    ).view(batch, pages_per_request)
    return table, table, pages_per_request


def _decode_append_page_table(
    batch: int,
    sequence_length: int,
    *,
    config: _DSAConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, int]:
    table_columns = max(1, math.ceil(sequence_length / config.kv_page_size))
    table = (
        torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        .unsqueeze(1)
        .expand(-1, table_columns)
        .contiguous()
    )
    return table, batch + 1


def _decode_tail_width(q_len_per_req: int, *, config: _DSAConfig) -> int:
    return config.pool_size if q_len_per_req == 1 else config.pool_size + q_len_per_req


def _kpool_scoring_weights(
    tokens: int,
    *,
    config: _DSAConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    fused_key_weights = _randn(
        (tokens, config.index_head_dim + config.index_heads),
        generator=generator,
        dtype=config.dtype,
    )
    return fused_key_weights[:, config.index_head_dim :]


def _prefill_metadata(
    batch: int,
    prefix_tokens: int,
    query_tokens_per_sequence: int,
    *,
    config: _DSAConfig,
    device: torch.device | str,
) -> dict[str, torch.Tensor | int]:
    query_start_loc = torch.arange(
        batch + 1,
        dtype=torch.int32,
        device=device,
    ).mul_(query_tokens_per_sequence)
    token_offsets = torch.arange(
        query_tokens_per_sequence,
        dtype=torch.int32,
        device=device,
    )
    positions = token_offsets.add(prefix_tokens).repeat(batch).contiguous()
    causal_lens = positions + 1
    req_ids = torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(
        query_tokens_per_sequence
    )
    final_pools = (prefix_tokens + query_tokens_per_sequence) // config.pool_size
    max_num_pools = max(final_pools, 1)
    index_pages_per_request = max(
        1,
        math.ceil(final_pools / config.index_rows_per_page),
    )
    index_rows_per_request = index_pages_per_request * config.index_rows_per_page
    pool_workspace_slots = torch.cat(
        [
            torch.arange(final_pools, dtype=torch.int64, device=device).add_(
                request * index_rows_per_request
            )
            for request in range(batch)
        ]
    )
    workspace_starts = torch.arange(
        batch,
        dtype=torch.int32,
        device=device,
    ).mul_(final_pools)
    row_starts = workspace_starts.repeat_interleave(query_tokens_per_sequence)
    row_ends = row_starts + torch.div(
        causal_lens,
        config.pool_size,
        rounding_mode="floor",
    )
    return {
        "positions": positions,
        "query_start_loc": query_start_loc,
        "req_ids": req_ids,
        "causal_lens": causal_lens,
        "pool_workspace_slots": pool_workspace_slots,
        "row_starts": row_starts,
        "row_ends": row_ends,
        "max_num_pools": max_num_pools,
    }


def _selected_slots(
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    *,
    config: _DSAConfig,
    region_slots: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    completed_pools = torch.div(
        causal_lens,
        config.pool_size,
        rounding_mode="floor",
    ).to(torch.int32)
    selected_pools = torch.minimum(
        completed_pools,
        torch.full_like(completed_pools, config.topk_pools),
    )
    available_pools = completed_pools.clamp_min(1)
    pool_columns = torch.arange(config.topk_pools, dtype=torch.int32, device=device)
    token_phases = torch.arange(
        causal_lens.numel(),
        dtype=torch.int32,
        device=device,
    ).mul_(_TOKEN_PERMUTATION_STRIDE)
    selected_pool_ids = (
        pool_columns.unsqueeze(0)
        .mul(_POOL_PERMUTATION_STRIDE)
        .add(token_phases.unsqueeze(1))
        .remainder_(available_pools.unsqueeze(1))
    )
    pool_offsets = torch.arange(config.pool_size, dtype=torch.int32, device=device)
    pool_slots = (
        selected_pool_ids.unsqueeze(2)
        .mul(config.pool_size)
        .add(pool_offsets)
        .view(causal_lens.numel(), config.topk_tokens)
    )
    pool_valid = (
        pool_columns.unsqueeze(0) < selected_pools.unsqueeze(1)
    ).repeat_interleave(config.pool_size, dim=1)

    tail_columns = torch.arange(config.pool_size - 1, dtype=torch.int32, device=device)
    tail_lens = torch.remainder(causal_lens, config.pool_size).to(torch.int32)
    tail_slots = completed_pools.unsqueeze(1).mul(config.pool_size).add(tail_columns)
    tail_valid = tail_columns.unsqueeze(0) < tail_lens.unsqueeze(1)

    slots = torch.cat((pool_slots, tail_slots), dim=1)
    valid = torch.cat((pool_valid, tail_valid), dim=1)
    slots.add_(req_ids.to(torch.int32).unsqueeze(1) * region_slots)
    slots.masked_fill_(~valid, -1)
    valid_lens = selected_pools.mul(config.pool_size).add_(tail_lens)
    return slots.contiguous(), valid_lens.contiguous()


def _snapshot_reset(*tensors: torch.Tensor) -> Callable[[], None]:
    snapshots = tuple(tensor.clone() for tensor in tensors)

    def reset() -> None:
        for tensor, snapshot in zip(tensors, snapshots, strict=True):
            tensor.copy_(snapshot)

    return reset


def _common_parameters(config: _DSAConfig) -> dict[str, object]:
    return {
        "model_profile": config.model_profile,
        "dtype": str(config.dtype).removeprefix("torch."),
        "kv_cache_dtype": str(config.kv_cache_dtype).removeprefix("torch."),
        "index_heads": config.index_heads,
        "index_head_dim": config.index_head_dim,
        "attention_heads": config.local_attention_heads,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "pool_size": config.pool_size,
        "topk_tokens": config.topk_tokens,
        "index_rows_per_page": config.index_rows_per_page,
        "index_page_stride_bytes": config.index_page_stride_bytes,
        "kv_page_size": config.kv_page_size,
        "max_context": config.max_context,
        "max_logits_bytes": config.max_logits_bytes,
    }


def prepare_kpool_prefill_write(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one completed-pool compression call."""

    config = _resolve_config(request)
    rows = request.parameters["rows"]
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={"slot_k": config.dtype},
        traits={
            "head_dim": config.index_head_dim,
            "pool_size": config.pool_size,
            "index_k_format": "fp8_scaled",
            "rotate": True,
        },
    )

    generator = _generator(request.seed)
    slot_k = _randn(
        (rows, config.pool_size, config.index_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    slot_score = _randn(
        tuple(slot_k.shape),
        generator=generator,
        dtype=config.dtype,
    )
    pages = math.ceil(rows / config.index_rows_per_page)
    _, index_values, index_scales = _packed_index_cache(
        pages,
        config=config,
        generator=generator,
    )
    write_slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    ape = torch.randn(
        (config.pool_size, config.index_head_dim),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    from tokenspeed_kernel.ops.attention import kpool as kpool_ops

    def invoke() -> object:
        return kpool_ops.kpool_prefill_write(
            slot_k,
            slot_score,
            write_slots,
            index_values,
            index_scales,
            ape,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={**_common_parameters(config), "rows": rows},
        validation=None,
    )


def prepare_kpool_decode_append(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one decode KPool append call."""

    config = _resolve_config(request)
    batch = request.parameters["batch"]
    q_len_per_req = request.parameters["q_len_per_req"]
    sequence_length = request.parameters["sequence_length"]
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={"k": config.dtype},
        traits={
            "head_dim": config.index_head_dim,
            "pool_size": config.pool_size,
            "index_k_format": "fp8_scaled",
            "rotate": True,
        },
    )

    generator = _generator(request.seed)
    keys = _randn(
        (batch, q_len_per_req, config.index_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    gates = _randn(
        tuple(keys.shape),
        generator=generator,
        dtype=config.dtype,
    )
    tail_width = _decode_tail_width(q_len_per_req, config=config)
    tail_k = _randn(
        (batch + 1, tail_width, config.index_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    tail_gate = _randn(
        tuple(tail_k.shape),
        generator=generator,
        dtype=config.dtype,
    )
    index_block_table, index_pages = _decode_append_page_table(
        batch,
        sequence_length,
        config=config,
        device="cuda",
    )
    _, index_values, index_scales = _packed_index_cache(
        index_pages,
        config=config,
        generator=generator,
    )
    seq_lens = torch.full(
        (batch,),
        sequence_length,
        dtype=torch.int32,
        device="cuda",
    )
    request_slots = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    ape = torch.randn(
        (config.pool_size, config.index_head_dim),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    reset = _snapshot_reset(tail_k, tail_gate, index_values, index_scales)

    from tokenspeed_kernel.ops.attention import kpool as kpool_ops

    def invoke() -> object:
        return kpool_ops.kpool_decode_append(
            keys,
            gates,
            tail_k,
            tail_gate,
            seq_lens,
            request_slots,
            index_block_table,
            index_values,
            index_scales,
            ape,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            **_common_parameters(config),
            "batch": batch,
            "q_len_per_req": q_len_per_req,
            "sequence_length": sequence_length,
            "tail_width": tail_width,
            "index_pages": index_pages,
        },
        validation=None,
    )


def _prepare_kpool_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    *,
    prefill: bool,
) -> PreparedBenchmark:
    if prefill:
        config = _resolve_config(request)
        batch = request.parameters["batch"]
        prefix_tokens = request.parameters["prefix_tokens"]
        query_tokens_per_sequence = request.parameters["query_tokens_per_sequence"]
        sequence_length = prefix_tokens + query_tokens_per_sequence
        q_len_per_req = 1
    else:
        config = _resolve_config(request)
        batch = request.parameters["batch"]
        q_len_per_req = request.parameters["q_len_per_req"]
        sequence_length = request.parameters["sequence_length"]
        query_tokens_per_sequence = q_len_per_req

    traits: dict[str, object] = {
        "head_dim": config.index_head_dim,
        "pool_size": config.pool_size,
        "page_size": config.index_rows_per_page,
        "index_k_format": "fp8_scaled",
        "score_activation": "relu",
        "topk_layout": "global_slots",
        "topk_pools": config.topk_pools,
    }
    if prefill:
        traits.update({"index_heads": config.index_heads, "has_prefill_plan": True})
    else:
        traits["q_len"] = q_len_per_req
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={"q": config.dtype},
        traits=traits,
    )

    generator = _generator(request.seed)
    tokens = batch * query_tokens_per_sequence
    q = _randn(
        (tokens, config.index_heads, config.index_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    weights = _kpool_scoring_weights(
        tokens,
        config=config,
        generator=generator,
    )
    index_table, kv_table, pages_per_request = _page_tables(
        batch,
        sequence_length,
        config=config,
        device="cuda",
    )
    pooled_k_cache, _, _ = _packed_index_cache(
        batch * pages_per_request,
        config=config,
        generator=generator,
    )
    out = torch.empty(
        (tokens, config.selected_width),
        dtype=torch.int32,
        device="cuda",
    )
    lens_out = torch.empty(tokens, dtype=torch.int32, device="cuda")

    from tokenspeed_kernel.ops.attention import kpool as kpool_ops

    if prefill:
        metadata = _prefill_metadata(
            batch,
            prefix_tokens,
            query_tokens_per_sequence,
            config=config,
            device="cuda",
        )

        def invoke() -> object:
            return kpool_ops.kpool_prefill_topk(
                q,
                pooled_k_cache,
                weights,
                metadata["positions"],
                metadata["query_start_loc"],
                index_table,
                kv_table,
                pool_size=config.pool_size,
                page_size=config.index_rows_per_page,
                kv_page_size=config.kv_page_size,
                topk_pools=config.topk_pools,
                softmax_scale=config.kpool_softmax_scale,
                req_ids=metadata["req_ids"],
                causal_lens=metadata["causal_lens"],
                pool_workspace_slots=metadata["pool_workspace_slots"],
                row_starts=metadata["row_starts"],
                row_ends=metadata["row_ends"],
                max_num_pools=metadata["max_num_pools"],
                max_logits_bytes=config.max_logits_bytes,
                out=out,
                lens_out=lens_out,
            )

    else:
        seq_lens = torch.full(
            (batch,),
            sequence_length,
            dtype=torch.int32,
            device="cuda",
        )

        def invoke() -> object:
            return kpool_ops.kpool_decode_topk(
                q,
                pooled_k_cache,
                weights,
                seq_lens,
                index_table,
                kv_table,
                pool_size=config.pool_size,
                page_size=config.index_rows_per_page,
                kv_page_size=config.kv_page_size,
                topk_pools=config.topk_pools,
                softmax_scale=config.kpool_softmax_scale,
                q_len_per_req=q_len_per_req,
                max_seq_len=config.max_context,
                out=out,
                lens_out=lens_out,
            )

    parameters = {
        **_common_parameters(config),
        "batch": batch,
        "query_tokens_per_sequence": query_tokens_per_sequence,
        "sequence_length": sequence_length,
        "index_pages_per_request": pages_per_request,
    }
    if prefill:
        parameters["prefix_tokens"] = prefix_tokens
    else:
        parameters["q_len_per_req"] = q_len_per_req
    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters=parameters,
        validation=None,
    )


def prepare_kpool_prefill_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one planned prefill KPool selection call."""

    return _prepare_kpool_topk(request, platform, prefill=True)


def prepare_kpool_decode_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one decode KPool selection call."""

    return _prepare_kpool_topk(request, platform, prefill=False)


def _prepare_dsa_attention(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    *,
    prefill: bool,
) -> PreparedBenchmark:
    if prefill:
        config = _resolve_config(request)
        batch = request.parameters["batch"]
        prefix_tokens = request.parameters["prefix_tokens"]
        query_tokens_per_sequence = request.parameters["query_tokens_per_sequence"]
        sequence_length = prefix_tokens + query_tokens_per_sequence
        q_len_per_req = 1
        metadata = _prefill_metadata(
            batch,
            prefix_tokens,
            query_tokens_per_sequence,
            config=config,
            device="cuda",
        )
        causal_lens = metadata["causal_lens"]
        req_ids = metadata["req_ids"]
    else:
        config = _resolve_config(request)
        batch = request.parameters["batch"]
        q_len_per_req = request.parameters["q_len_per_req"]
        sequence_length = request.parameters["sequence_length"]
        query_tokens_per_sequence = q_len_per_req
        offsets = torch.arange(
            1 - q_len_per_req,
            1,
            dtype=torch.int32,
            device="cuda",
        )
        causal_lens = offsets.add(sequence_length).repeat(batch).contiguous()
        req_ids = torch.arange(
            batch,
            dtype=torch.int32,
            device="cuda",
        ).repeat_interleave(q_len_per_req)

    traits = {
        "page_size": config.kv_page_size,
        "q_len": q_len_per_req,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "topk": config.selected_width,
        "has_kv_cache": True,
        "has_sparse_kv_cache": False,
        "topk_layout": "global_slots",
        "logit_cap": False,
        "return_lse": False,
    }
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={"q": config.kv_cache_dtype},
        traits=traits,
    )

    generator = _generator(request.seed)
    tokens = batch * query_tokens_per_sequence
    # Production casts the query to the cache dtype for FP8 DSA configurations.
    q = _randn(
        (tokens, config.local_attention_heads, config.qk_head_dim),
        generator=generator,
        dtype=config.dtype,
    ).to(config.kv_cache_dtype)
    kv_cache = _randn(
        (batch * sequence_length, 1, config.qk_head_dim),
        generator=generator,
        dtype=config.dtype,
    ).to(config.kv_cache_dtype)
    topk_slots, topk_lens = _selected_slots(
        causal_lens,
        req_ids,
        config=config,
        region_slots=sequence_length,
        device="cuda",
    )
    out = torch.empty(
        (tokens, config.local_attention_heads, config.kv_lora_rank),
        dtype=config.dtype,
        device="cuda",
    )

    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    common_kwargs = {
        "q": q,
        "kv_cache": kv_cache,
        "sparse_kv_cache": None,
        "topk_slots": topk_slots,
        "topk_lens": topk_lens,
        "max_seqlen_k": sequence_length if prefill else config.max_context,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "softmax_scale": config.dsa_softmax_scale,
        "page_size": config.kv_page_size,
        "kv_seq_lens": causal_lens,
        "out": out,
    }
    if prefill:

        def invoke() -> object:
            return dsa_ops.dsa_prefill(**common_kwargs)

    else:

        def invoke() -> object:
            return dsa_ops.dsa_decode(
                **common_kwargs,
                q_len_per_req=q_len_per_req,
            )

    parameters = {
        **_common_parameters(config),
        "batch": batch,
        "query_tokens_per_sequence": query_tokens_per_sequence,
        "sequence_length": sequence_length,
        "selected_width": config.selected_width,
        "kv_region_slots_per_request": sequence_length,
    }
    if prefill:
        parameters["prefix_tokens"] = prefix_tokens
    else:
        parameters["q_len_per_req"] = q_len_per_req
        parameters["max_seqlen_k"] = config.max_context
    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters=parameters,
        validation=None,
    )


def prepare_dsa_prefill(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one sparse prefill attention call."""

    return _prepare_dsa_attention(request, platform, prefill=True)


def prepare_dsa_decode(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one sparse decode attention call."""

    return _prepare_dsa_attention(request, platform, prefill=False)
