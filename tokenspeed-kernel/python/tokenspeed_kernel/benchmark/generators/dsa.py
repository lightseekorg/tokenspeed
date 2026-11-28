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

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.ops.kvcache.index_k_scale import (
    encode_index_k_scale,
    index_k_scale_dtype,
    round_index_k_scale,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "prepare_dsa_decode",
    "prepare_dsa_decode_dcp_records",
    "prepare_dsa_decode_projected_value",
    "prepare_dsa_decode_topk",
    "prepare_dsa_index_candidates",
    "prepare_dsa_prefill",
    "prepare_dsa_select_owned_candidates",
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
        .view(index_k_scale_dtype())
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
    scale_values = round_index_k_scale(scale_values / fp8_max)
    values.copy_(
        (source_groups / scale_values.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .view_as(source)
    )
    scales.copy_(encode_index_k_scale(scale_values))
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
        # The benchmark times the tuned cores, which reduce the slots as
        # selected.
        "slot_order": "selection",
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


# ===-----------------------------------------------------------------------===#
# Indexer DSA (Kimi-K3.1)
# ===-----------------------------------------------------------------------===#
#
# Kimi-K3.1's lightning indexer scores every history token (no KPool pooling)
# from an mxfp4 Index-K plane and keeps its top-k for the MLA core. Decode
# without KV parallelism scores the whole history (``dsa_decode_topk``) and
# attends it through the projected-value core. Under KV parallelism each rank
# scores its owned pages (``dsa_index_candidates``), selects its share of the
# group's top-k from every rank's gathered scores
# (``dsa_select_owned_candidates``), and attends that share for every rank's
# heads (``dsa_decode_dcp_records``). The cases time one rank (rank 0).

_INDEXER_MODEL_PROFILES = frozenset({"kimi_k31_tp8"})
_INDEXER_DTYPES = {"bfloat16": torch.bfloat16}
_INDEXER_KV_CACHE_DTYPES = {"float8_e4m3fn": torch.float8_e4m3fn}
_INDEXER_INDEX_K_FORMATS = frozenset({"mxfp4"})
_MXFP4_GROUP_SIZE = 32
# UE8M0 exponents [low, high) of the generated key scales (2^-3 to 2^0), so
# generated scores stay finite; 255 would encode NaN.
_MXFP4_SCALE_EXPONENTS = (124, 128)
# The server's --dsa-slot-order default.
_SLOT_ORDER = "selection"


@dataclass(frozen=True)
class _IndexerConfig:
    """One DSA layer whose indexer scores every history token."""

    model_profile: str
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    index_heads: int
    index_head_dim: int
    index_k_format: str
    local_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_page_size: int
    topk_tokens: int
    max_context: int

    @property
    def qk_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def index_row_bytes(self) -> int:
        """Bytes of one mxfp4 Index-K row: E2M1 code pairs, then UE8M0 scales."""
        return self.index_head_dim // 2 + self.index_head_dim // _MXFP4_GROUP_SIZE

    @property
    def index_score_scale(self) -> float:
        """The positive score scale the indexer's top-k multiplies by."""
        return self.index_head_dim**-0.5 * self.index_heads**-0.5

    @property
    def attention_softmax_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

    @property
    def table_columns(self) -> int:
        """Decode page-table width: decode tables are sized by the serving
        context capacity, not by the live history."""
        return math.ceil(self.max_context / self.kv_page_size)


def _resolve_indexer_config(request: BenchmarkRequest) -> _IndexerConfig:
    if request.parameters.get("validation") is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "DSA benchmark correctness validation is not implemented yet",
        )
    if request.registration is not None:
        # The operations select their own leaves and take no exact override.
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Indexer DSA benchmarks take a solution, not an exact registration",
        )
    parameters = request.parameters
    config = _IndexerConfig(
        model_profile=_implemented_value(
            "model_profile", parameters["model_profile"], _INDEXER_MODEL_PROFILES
        ),
        dtype=_parse_dtype("dtype", parameters["dtype"], _INDEXER_DTYPES),
        kv_cache_dtype=_parse_dtype(
            "kv_cache_dtype", parameters["kv_cache_dtype"], _INDEXER_KV_CACHE_DTYPES
        ),
        index_heads=parameters["index_heads"],
        index_head_dim=parameters["index_head_dim"],
        index_k_format=_implemented_value(
            "index_k_format", parameters["index_k_format"], _INDEXER_INDEX_K_FORMATS
        ),
        local_attention_heads=parameters["local_attention_heads"],
        kv_lora_rank=parameters["kv_lora_rank"],
        qk_nope_head_dim=parameters["qk_nope_head_dim"],
        qk_rope_head_dim=parameters["qk_rope_head_dim"],
        v_head_dim=parameters["v_head_dim"],
        kv_page_size=parameters["kv_page_size"],
        topk_tokens=parameters["topk_tokens"],
        max_context=parameters["max_context"],
    )
    if config.index_head_dim % _MXFP4_GROUP_SIZE:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"mxfp4 index_head_dim must be a multiple of {_MXFP4_GROUP_SIZE}",
        )
    return config


def _indexer_common_parameters(config: _IndexerConfig) -> dict[str, object]:
    return {
        "model_profile": config.model_profile,
        "dtype": str(config.dtype).removeprefix("torch."),
        "kv_cache_dtype": str(config.kv_cache_dtype).removeprefix("torch."),
        "index_heads": config.index_heads,
        "index_head_dim": config.index_head_dim,
        "index_k_format": config.index_k_format,
        "attention_heads": config.local_attention_heads,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "kv_page_size": config.kv_page_size,
        "topk_tokens": config.topk_tokens,
        "max_context": config.max_context,
    }


def _verify_shape(
    request: BenchmarkRequest, config: _IndexerConfig
) -> tuple[int, int, int]:
    """``(batch, q_len_per_req, sequence_length)`` of a decode verify step:
    ``batch`` requests of ``q_len_per_req`` rows each, every request holding
    ``sequence_length`` tokens including its verify window."""
    parameters = request.parameters
    batch, q_len_per_req, sequence_length = (
        parameters[name] for name in ("batch", "q_len_per_req", "sequence_length")
    )
    if any(
        type(value) is not int or value < 1
        for value in (batch, q_len_per_req, sequence_length)
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Indexer DSA batch, q_len_per_req and sequence_length must be "
            "positive integers",
        )
    if not q_len_per_req <= config.topk_tokens <= sequence_length:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Indexer DSA decode needs q_len_per_req <= topk_tokens <= "
            "sequence_length, so every row selects a full top-k",
        )
    if sequence_length > config.max_context:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Indexer DSA sequence_length exceeds max_context",
        )
    return batch, q_len_per_req, sequence_length


def _kvp_size(request: BenchmarkRequest) -> int:
    kvp_size = request.parameters["kvp_size"]
    if type(kvp_size) is not int or kvp_size < 2:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Indexer DSA kvp_size must be an integer of at least 2",
        )
    return kvp_size


def _verify_rows(
    batch: int,
    q_len_per_req: int,
    sequence_length: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every verify row's request and causal length: row ``j`` of a
    request's window sees ``sequence_length - (q_len_per_req - 1) + j``
    tokens."""
    offsets = torch.arange(1 - q_len_per_req, 1, dtype=torch.int32, device=device)
    causal_lens = offsets.add(sequence_length).repeat(batch).contiguous()
    requests = torch.arange(batch, dtype=torch.int32, device=device)
    return requests.repeat_interleave(q_len_per_req), causal_lens


def _scheduler_page_table(
    batch: int,
    sequence_length: int,
    *,
    config: _IndexerConfig,
    device: torch.device | str,
) -> torch.Tensor:
    """Every request's scheduler (virtual) page table at the decode width.

    Block 0 is the null block. Request ``r``'s live pages are blocks
    ``1 + r * live_pages + column``; columns past them hold 0.
    """
    live_pages = math.ceil(sequence_length / config.kv_page_size)
    column = torch.arange(config.table_columns, dtype=torch.int32, device=device)
    first = 1 + torch.arange(batch, dtype=torch.int32, device=device) * live_pages
    return torch.where(column < live_pages, first.unsqueeze(1) + column, 0).contiguous()


def _owned_page_table(
    scheduler_table: torch.Tensor, *, kvp_size: int, kvp_rank: int
) -> torch.Tensor:
    """One KVP rank's local page table of ``scheduler_table``.

    As the runtime's cache placement (``virtual_slots_to_local``), rank
    ``r`` owns block ``b > 0`` when ``(b - 1) % kvp_size == r``, as local
    page ``(b - 1) // kvp_size + 1``; other columns are -1.
    """
    before = scheduler_table - 1
    owned = (scheduler_table > 0) & (before % kvp_size == kvp_rank)
    return torch.where(owned, before // kvp_size + 1, -1).to(torch.int32).contiguous()


def _spread_slots(
    causal_lens: torch.Tensor,
    requests: torch.Tensor,
    *,
    topk: int,
    region_slots: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A stand-in for the indexer's selection: ``topk`` slots per row spread
    over its causal history, in request ``r``'s region of ``region_slots``
    cache slots starting at ``r * region_slots``."""
    columns = torch.arange(topk, dtype=torch.int64, device=device)
    phases = torch.arange(causal_lens.numel(), dtype=torch.int64, device=device)
    positions = (
        columns.unsqueeze(0) * _POOL_PERMUTATION_STRIDE
        + phases.unsqueeze(1) * _TOKEN_PERMUTATION_STRIDE
    ).remainder_(causal_lens.to(torch.int64).unsqueeze(1))
    slots = positions + requests.to(torch.int64).unsqueeze(1) * region_slots
    lens = causal_lens.clamp(max=topk)
    slots.masked_fill_(columns.unsqueeze(0) >= lens.unsqueeze(1), -1)
    return slots.to(torch.int32).contiguous(), lens.to(torch.int32).contiguous()


def _mxfp4_index_plane(
    pages: int, *, config: _IndexerConfig, generator: torch.Generator
) -> torch.Tensor:
    """Random mxfp4 Index-K pages as the runtime plane: uint8 ``[pages *
    page_size, row_bytes]``, each page its rows' E2M1 code bytes, then their
    UE8M0 scale bytes (page planar)."""
    code_bytes = config.kv_page_size * config.index_head_dim // 2
    scale_bytes = config.kv_page_size * config.index_head_dim // _MXFP4_GROUP_SIZE
    plane = torch.empty(
        (pages, code_bytes + scale_bytes), dtype=torch.uint8, device="cuda"
    )
    plane[:, :code_bytes] = torch.randint(
        0,
        256,
        (pages, code_bytes),
        generator=generator,
        dtype=torch.uint8,
        device="cuda",
    )
    plane[:, code_bytes:] = torch.randint(
        *_MXFP4_SCALE_EXPONENTS,
        (pages, scale_bytes),
        generator=generator,
        dtype=torch.uint8,
        device="cuda",
    )
    return plane.view(pages * config.kv_page_size, config.index_row_bytes)


def _index_queries(
    rows: int, *, config: _IndexerConfig, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mxfp4 index query codes and scale words, and the per-head score
    weights as a strided view of the indexer's ``[wk; weights_proj]``
    projection rows."""
    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    q = _randn(
        (rows, config.index_heads, config.index_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    codes, scales = dsa_ops.dsa_quantize_index_query_mxfp4(q)
    key_weights = _randn(
        (rows, config.index_head_dim + config.index_heads),
        generator=generator,
        dtype=config.dtype,
    )
    return codes, scales, key_weights[:, config.index_head_dim :]


def _select_indexer_registration(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    mode: str,
    signature: object,
    traits: dict[str, object],
) -> KernelSpec:
    try:
        selected = select_kernel(
            request.family,
            mode,
            signature,
            platform=platform,
            traits=traits,
            solution=request.solution,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec


def _index_scorer_selection(
    config: _IndexerConfig,
) -> tuple[object, dict[str, object]]:
    """The scorers' signature and the traits they share (mxfp4 queries over
    the packed ``[slots, row_bytes]`` plane)."""
    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    signature = format_signature(
        q=dsa_ops.MXFP4_INDEX_QUERY_FORMAT, weights=dense_tensor_format(config.dtype)
    )
    return signature, {
        "index_heads": config.index_heads,
        "head_dim": config.index_head_dim,
        "page_size": config.kv_page_size,
        "index_k_format": config.index_k_format,
        "index_k_layout": "packed",
    }


def _owned_candidates(
    request: BenchmarkRequest,
    config: _IndexerConfig,
    batch: int,
    q_len_per_req: int,
    sequence_length: int,
    kvp_size: int,
    generator: torch.Generator,
) -> dict[str, object]:
    """Rank 0's inputs to ``dsa_index_candidates`` for every verify row:
    owned Index-K pages, queries and the page plan the runtime builds once
    per forward for every indexer layer."""
    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    requests, causal_lens = _verify_rows(
        batch, q_len_per_req, sequence_length, device="cuda"
    )
    local_table = _owned_page_table(
        _scheduler_page_table(batch, sequence_length, config=config, device="cuda"),
        kvp_size=kvp_size,
        kvp_rank=0,
    )
    local_pages = int(local_table.max()) + 1
    plane = _mxfp4_index_plane(local_pages, config=config, generator=generator)
    q, q_scales, weights = _index_queries(
        requests.numel(), config=config, generator=generator
    )
    pages = dsa_ops.plan_dsa_index_candidates(
        q,
        weights,
        plane,
        local_table,
        requests,
        causal_lens,
        page_size=config.kv_page_size,
        solution=request.solution,
        index_k_format=config.index_k_format,
    )
    return {
        "q": q,
        "q_scales": q_scales,
        "weights": weights,
        "plane": plane,
        "local_table": local_table,
        "local_pages": local_pages,
        "requests": requests,
        "causal_lens": causal_lens,
        "pages": pages,
    }


def _score_owned_candidates(
    request: BenchmarkRequest,
    config: _IndexerConfig,
    inputs: dict[str, object],
    q_len_per_req: int,
    scores_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    return dsa_ops.dsa_index_candidates(
        inputs["q"],
        inputs["weights"],
        inputs["plane"],
        inputs["local_table"],
        inputs["requests"],
        inputs["causal_lens"],
        page_size=config.kv_page_size,
        topk=config.topk_tokens,
        softmax_scale=config.index_score_scale,
        initial_tokens=0,
        local_tokens=0,
        solution=request.solution,
        pages=inputs["pages"],
        scores_out=scores_out,
        q_scales=inputs["q_scales"],
        query_group_size=q_len_per_req,
        index_k_format=config.index_k_format,
    )


def _gathered_scores_and_query(
    config: _IndexerConfig,
    rows: int,
    kvp_size: int,
    rank_scores: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The owned top-k's all-gather as it lands: per rank, ``[rows, topk]``
    FP32 candidate scores, then its decode query rows riding along.

    Rank 0's plane holds ``rank_scores``; every other rank's holds them
    permuted along the candidates and slightly perturbed, so the ranks'
    shares of the selection stay balanced as on a uniformly spread
    history. Returns the rank-major ``[ranks, rows, topk]`` score view and
    the rank-major FP8 ``[ranks, rows, heads, qk_head_dim]`` query view.
    """
    score_bytes = rows * config.topk_tokens * torch.float32.itemsize
    query_row_bytes = (
        config.local_attention_heads
        * config.qk_head_dim
        * config.kv_cache_dtype.itemsize
    )
    gathered = torch.empty(
        (kvp_size, score_bytes + rows * query_row_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    scores = (
        gathered[:, :score_bytes]
        .view(torch.float32)
        .view(kvp_size, rows, config.topk_tokens)
    )
    scores[0].copy_(rank_scores)
    for rank in range(1, kvp_size):
        permutation = torch.randperm(
            config.topk_tokens, generator=generator, device="cuda"
        )
        noise = torch.randn(rank_scores.shape, generator=generator, device="cuda").mul_(
            1e-3
        )
        scores[rank].copy_(rank_scores[:, permutation] * (1 + noise))
    query = gathered[:, score_bytes:].view(kvp_size, rows, query_row_bytes)
    query.view(config.kv_cache_dtype).copy_(
        _randn(
            (kvp_size, rows, query_row_bytes // config.kv_cache_dtype.itemsize),
            generator=generator,
            dtype=config.dtype,
        ).to(config.kv_cache_dtype)
    )
    return scores, query.view(config.kv_cache_dtype).view(
        kvp_size, rows, config.local_attention_heads, config.qk_head_dim
    )


def _verify_parameters(
    config: _IndexerConfig, batch: int, q_len_per_req: int, sequence_length: int
) -> dict[str, object]:
    return {
        **_indexer_common_parameters(config),
        "batch": batch,
        "q_len_per_req": q_len_per_req,
        "sequence_length": sequence_length,
        "table_columns": config.table_columns,
    }


def prepare_dsa_decode_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one indexer top-k over the whole history of each request."""

    config = _resolve_indexer_config(request)
    batch, q_len_per_req, sequence_length = _verify_shape(request, config)
    load_builtin_kernels()
    signature, traits = _index_scorer_selection(config)
    spec = _select_indexer_registration(
        request,
        platform,
        "dsa_decode_topk",
        signature,
        {**traits, "q_len": q_len_per_req, "topk": config.topk_tokens},
    )

    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    generator = _generator(request.seed)
    block_table = _scheduler_page_table(
        batch, sequence_length, config=config, device="cuda"
    )
    plane = _mxfp4_index_plane(
        int(block_table.max()) + 1, config=config, generator=generator
    )
    rows = batch * q_len_per_req
    q, q_scales, weights = _index_queries(rows, config=config, generator=generator)
    seq_lens = torch.full((batch,), sequence_length, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, config.topk_tokens), dtype=torch.int32, device="cuda")
    lens_out = torch.empty(rows, dtype=torch.int32, device="cuda")

    def invoke() -> object:
        return dsa_ops.dsa_decode_topk(
            q,
            weights,
            seq_lens,
            block_table,
            page_size=config.kv_page_size,
            topk=config.topk_tokens,
            softmax_scale=config.index_score_scale,
            batch_invariant=False,
            q_len_per_req=q_len_per_req,
            index_k_cache=plane,
            q_scales=q_scales,
            out=out,
            lens_out=lens_out,
            slot_order=_SLOT_ORDER,
            solution=request.solution,
            index_k_format=config.index_k_format,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters=_verify_parameters(config, batch, q_len_per_req, sequence_length),
        validation=None,
    )


def prepare_dsa_decode_projected_value(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one sparse decode of the attention-TP heads with their gated
    value projection."""

    config = _resolve_indexer_config(request)
    batch, q_len_per_req, sequence_length = _verify_shape(request, config)
    rows = batch * q_len_per_req
    load_builtin_kernels()
    spec = _select_indexer_registration(
        request,
        platform,
        "dsa_decode_projected_value",
        format_signature(q=dense_tensor_format(config.kv_cache_dtype)),
        {
            "q_len": q_len_per_req,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "topk": config.topk_tokens,
            "logit_cap": False,
            "topk_layout": "global_slots",
            "gate_kind": "sigmoid",
            "tokens": rows,
            "heads": config.local_attention_heads,
            "q_dtype": config.kv_cache_dtype,
            "kv_dtype": config.kv_cache_dtype,
            "value_head_dim": config.v_head_dim,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "page_size": config.kv_page_size,
            "slot_order": _SLOT_ORDER,
        },
    )

    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    generator = _generator(request.seed)
    requests, causal_lens = _verify_rows(
        batch, q_len_per_req, sequence_length, device="cuda"
    )
    topk_slots, topk_lens = _spread_slots(
        causal_lens,
        requests,
        topk=config.topk_tokens,
        region_slots=sequence_length,
        device="cuda",
    )
    # The core reads an FP8 query over an FP8 cache.
    q = _randn(
        (rows, config.local_attention_heads, config.qk_head_dim),
        generator=generator,
        dtype=config.dtype,
    ).to(config.kv_cache_dtype)
    kv_cache = _randn(
        (batch * sequence_length, 1, config.qk_head_dim),
        generator=generator,
        dtype=config.dtype,
    ).to(config.kv_cache_dtype)
    # The absorbed value projection (w_vc) is K-major: [heads, kv_lora_rank,
    # v_head_dim] viewed over [heads, v_head_dim, kv_lora_rank] storage.
    value_weight = _randn(
        (config.local_attention_heads, config.v_head_dim, config.kv_lora_rank),
        generator=generator,
        dtype=config.dtype,
    ).transpose(1, 2)
    gate = _randn(
        (rows, config.local_attention_heads * config.v_head_dim),
        generator=generator,
        dtype=config.dtype,
    )
    out = torch.empty_like(gate)

    def invoke() -> object:
        return dsa_ops.dsa_decode_projected_value(
            q,
            kv_cache,
            topk_slots,
            topk_lens,
            config.max_context,
            config.qk_nope_head_dim,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            config.attention_softmax_scale,
            config.kv_page_size,
            value_weight,
            gate=gate,
            out=out,
            q_len_per_req=q_len_per_req,
            solution=request.solution,
            slot_order=_SLOT_ORDER,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters=_verify_parameters(config, batch, q_len_per_req, sequence_length),
        validation=None,
    )


def prepare_dsa_index_candidates(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one KVP rank's scoring of its owned pages for every verify
    row, writing its candidates' scores where the all-gather reads them."""

    config = _resolve_indexer_config(request)
    batch, q_len_per_req, sequence_length = _verify_shape(request, config)
    kvp_size = _kvp_size(request)
    load_builtin_kernels()
    signature, traits = _index_scorer_selection(config)
    spec = _select_indexer_registration(
        request, platform, "dsa_index_candidates", signature, traits
    )

    generator = _generator(request.seed)
    inputs = _owned_candidates(
        request, config, batch, q_len_per_req, sequence_length, kvp_size, generator
    )
    rows = batch * q_len_per_req
    scores_out = torch.empty(
        (rows, config.topk_tokens), dtype=torch.float32, device="cuda"
    )

    def invoke() -> object:
        return _score_owned_candidates(
            request, config, inputs, q_len_per_req, scores_out
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_verify_parameters(config, batch, q_len_per_req, sequence_length),
            "kvp_size": kvp_size,
            "owned_index_pages": inputs["local_pages"] - 1,
        },
        validation=None,
    )


def prepare_dsa_select_owned_candidates(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one KVP rank's selection of its share of the group's top-k
    from every rank's gathered candidate scores."""

    config = _resolve_indexer_config(request)
    batch, q_len_per_req, sequence_length = _verify_shape(request, config)
    kvp_size = _kvp_size(request)
    rows = batch * q_len_per_req
    query_row_bytes = (
        config.local_attention_heads
        * config.qk_head_dim
        * config.kv_cache_dtype.itemsize
    )
    load_builtin_kernels()
    spec = _select_indexer_registration(
        request,
        platform,
        "dsa_select_owned_candidates",
        format_signature(scores=dense_tensor_format(torch.float32)),
        {
            "ranks": kvp_size,
            "rows": rows,
            "topk": config.topk_tokens,
            # Each rank's scores are followed by its gathered query rows.
            "rank_stride": rows * config.topk_tokens
            + rows * query_row_bytes // torch.float32.itemsize,
        },
    )

    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    generator = _generator(request.seed)
    inputs = _owned_candidates(
        request, config, batch, q_len_per_req, sequence_length, kvp_size, generator
    )
    rank_scores = torch.empty(
        (rows, config.topk_tokens), dtype=torch.float32, device="cuda"
    )
    positions, _ = _score_owned_candidates(
        request, config, inputs, q_len_per_req, rank_scores
    )
    scores, _ = _gathered_scores_and_query(
        config, rows, kvp_size, rank_scores, generator
    )
    out = torch.empty((rows, config.topk_tokens), dtype=torch.int32, device="cuda")
    lens_out = torch.empty(rows, dtype=torch.int32, device="cuda")

    def invoke() -> object:
        return dsa_ops.dsa_select_owned_candidates(
            scores,
            positions,
            inputs["local_table"],
            inputs["requests"],
            rank=0,
            topk=config.topk_tokens,
            page_size=config.kv_page_size,
            solution=request.solution,
            out=out,
            lens_out=lens_out,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_verify_parameters(config, batch, q_len_per_req, sequence_length),
            "kvp_size": kvp_size,
            "rank_stride": scores.stride(0),
        },
        validation=None,
    )


def prepare_dsa_decode_dcp_records(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one KVP rank's sparse decode of every rank's heads over its
    share of the selection, written as the records the group combines."""

    config = _resolve_indexer_config(request)
    batch, q_len_per_req, sequence_length = _verify_shape(request, config)
    kvp_size = _kvp_size(request)
    rows = batch * q_len_per_req
    load_builtin_kernels()
    spec = _select_indexer_registration(
        request,
        platform,
        "dsa_decode_dcp_records",
        format_signature(q=dense_tensor_format(config.kv_cache_dtype)),
        {
            "q_len": q_len_per_req,
            "tokens": rows,
            "heads": kvp_size * config.local_attention_heads,
            "topk": config.topk_tokens,
            "q_dtype": config.kv_cache_dtype,
            "kv_dtype": config.kv_cache_dtype,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "page_size": config.kv_page_size,
            "logit_cap": False,
            "topk_layout": "global_slots",
            "slot_order": _SLOT_ORDER,
        },
    )

    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    generator = _generator(request.seed)
    inputs = _owned_candidates(
        request, config, batch, q_len_per_req, sequence_length, kvp_size, generator
    )
    rank_scores = torch.empty(
        (rows, config.topk_tokens), dtype=torch.float32, device="cuda"
    )
    positions, _ = _score_owned_candidates(
        request, config, inputs, q_len_per_req, rank_scores
    )
    scores, q = _gathered_scores_and_query(
        config, rows, kvp_size, rank_scores, generator
    )
    # This rank's share of the group's selection, as the runtime selects it.
    topk_slots, topk_lens = dsa_ops.dsa_select_owned_candidates(
        scores,
        positions,
        inputs["local_table"],
        inputs["requests"],
        rank=0,
        topk=config.topk_tokens,
        page_size=config.kv_page_size,
        solution=request.solution,
    )
    kv_cache = _randn(
        (inputs["local_pages"] * config.kv_page_size, 1, config.qk_head_dim),
        generator=generator,
        dtype=config.dtype,
    ).to(config.kv_cache_dtype)
    workspace_bytes = dsa_ops.dsa_decode_dcp_records_workspace_bytes(
        tokens=rows,
        heads=kvp_size * config.local_attention_heads,
        topk=config.topk_tokens,
        q_dtype=config.kv_cache_dtype,
        kv_dtype=config.kv_cache_dtype,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
    )
    workspace = torch.empty(
        math.ceil(workspace_bytes / torch.int32.itemsize),
        dtype=torch.int32,
        device="cuda",
    )

    def invoke() -> object:
        return dsa_ops.dsa_decode_dcp_records(
            q=q,
            kv_cache=kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            qk_nope_head_dim=config.qk_nope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            softmax_scale=config.attention_softmax_scale,
            page_size=config.kv_page_size,
            q_len_per_req=q_len_per_req,
            solution=request.solution,
            slot_order=_SLOT_ORDER,
            workspace=workspace,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_verify_parameters(config, batch, q_len_per_req, sequence_length),
            "kvp_size": kvp_size,
        },
        validation=None,
    )
