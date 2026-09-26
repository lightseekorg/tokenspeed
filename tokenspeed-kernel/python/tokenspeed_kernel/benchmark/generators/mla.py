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

"""Benchmark generators for MLA attention operations."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass

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
    "prepare_mla_decode",
    "prepare_mla_decode_projected_value",
    "prepare_mla_normalize_project_query",
    "prepare_mla_prefill",
]


_IMPLEMENTED_MODEL_PROFILES = frozenset({"kimi_k3_tp8"})
_IMPLEMENTED_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
# Keep random FP8 operands well inside the E4M3 range, like unit-scale KV.
_FP8_SOURCE_SCALE = 0.5


@dataclass(frozen=True)
class _MLAConfig:
    model_profile: str
    local_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_page_size: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def gate_width(self) -> int:
        return self.local_heads * self.v_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5


def _implemented_value(
    name: str,
    value: object,
    implemented: Collection[str],
) -> str:
    if value not in implemented:
        accepted = ", ".join(sorted(implemented))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Implemented MLA {name} values: {accepted}",
        )
    return value


def _parse_dtype(name: str, value: object) -> torch.dtype:
    return _IMPLEMENTED_DTYPES[_implemented_value(name, value, _IMPLEMENTED_DTYPES)]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _positive(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MLA {name} must be a positive integer",
        )
    return value


def _resolve_config(request: BenchmarkRequest) -> _MLAConfig:
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA benchmark correctness validation is not implemented yet",
        )
    return _MLAConfig(
        model_profile=_implemented_value(
            "model_profile",
            parameters["model_profile"],
            _IMPLEMENTED_MODEL_PROFILES,
        ),
        local_heads=_positive("local_heads", parameters["local_heads"]),
        q_lora_rank=_positive("q_lora_rank", parameters["q_lora_rank"]),
        kv_lora_rank=_positive("kv_lora_rank", parameters["kv_lora_rank"]),
        qk_nope_head_dim=_positive("qk_nope_head_dim", parameters["qk_nope_head_dim"]),
        qk_rope_head_dim=_positive("qk_rope_head_dim", parameters["qk_rope_head_dim"]),
        v_head_dim=_positive("v_head_dim", parameters["v_head_dim"]),
        kv_page_size=_positive("kv_page_size", parameters["kv_page_size"]),
    )


def _common_parameters(config: _MLAConfig) -> dict[str, object]:
    return {
        "model_profile": config.model_profile,
        "local_heads": config.local_heads,
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "kv_page_size": config.kv_page_size,
    }


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dtype.is_floating_point and dtype.itemsize == 1:
        source = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        return source.mul_(_FP8_SOURCE_SCALE).to(dtype)
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


def _packed_qkv_gate(
    tokens: int,
    *,
    config: _MLAConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one fused ``[q_a | kv_a + rope | gate]`` projection row.

    The model's decode-sized QKV/gate projection writes one packed row per
    token, so the query latent, KV latent, and output gate are strided views.
    """

    widths = (config.q_lora_rank, config.latent_cache_dim, config.gate_width)
    packed = _randn((tokens, sum(widths)), generator=generator, dtype=torch.bfloat16)
    query, latent_cache, gate = packed.split(widths, dim=-1)
    return query, latent_cache[:, : config.kv_lora_rank], gate


def _decode_page_table(
    requests: int,
    rows_per_request: int,
    cache_length: int,
    max_context_len: int,
    *,
    config: _MLAConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the flattened decode page table and visible cache lengths.

    Each request owns distinct pages; its query rows share them. Multi-row
    requests follow target verification, where row ``i`` of ``n`` sees
    ``cache_length - (n - 1 - i)`` tokens.
    """

    pages_per_request = math.ceil(cache_length / config.kv_page_size)
    table_columns = math.ceil(max_context_len / config.kv_page_size)
    table = torch.zeros(
        (requests, table_columns),
        dtype=torch.int32,
        device=device,
    )
    table[:, :pages_per_request] = torch.arange(
        requests * pages_per_request,
        dtype=torch.int32,
        device=device,
    ).view(requests, pages_per_request)
    offsets = torch.arange(
        1 - rows_per_request,
        1,
        dtype=torch.int32,
        device=device,
    ).repeat(requests)
    cache_seqlens = (
        torch.full(
            (requests * rows_per_request,),
            cache_length,
            dtype=torch.int32,
            device=device,
        )
        + offsets
    )
    page_table = table.repeat_interleave(rows_per_request, dim=0)
    return page_table, cache_seqlens, requests * pages_per_request


def _resolve_decode_shape(
    request: BenchmarkRequest,
    config: _MLAConfig,
) -> dict[str, object]:
    parameters = request.parameters
    requests = _positive("requests", parameters["requests"])
    rows_per_request = _positive("rows_per_request", parameters["rows_per_request"])
    cache_length = _positive("cache_length", parameters["cache_length"])
    max_context_len = _positive("max_context_len", parameters["max_context_len"])
    if cache_length < rows_per_request:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA cache_length must cover every query row of a request",
        )
    if cache_length > max_context_len:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA cache_length must not exceed max_context_len",
        )
    q_dtype = _parse_dtype("q_dtype", parameters["q_dtype"])
    kv_cache_dtype = _parse_dtype("kv_cache_dtype", parameters["kv_cache_dtype"])
    return {
        "requests": requests,
        "rows_per_request": rows_per_request,
        "rows": requests * rows_per_request,
        "cache_length": cache_length,
        "max_context_len": max_context_len,
        "q_dtype": q_dtype,
        "kv_cache_dtype": kv_cache_dtype,
    }


def _decode_traits(rows: int, config: _MLAConfig) -> dict[str, object]:
    # Target verification flattens its rows onto the batch axis, so every call
    # here is a one-row causal query per batch entry.
    return {
        "batch_size": rows,
        "q_len": 1,
        "num_q_heads": config.local_heads,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "page_size": config.kv_page_size,
        "noncausal_block_size": 1,
        "block_on_query_axis": True,
        "logit_cap": False,
        "return_lse": False,
        "sliding_window": False,
    }


def _decode_inputs(
    shape: dict[str, object],
    *,
    config: _MLAConfig,
    generator: torch.Generator,
) -> dict[str, object]:
    rows = shape["rows"]
    page_table, cache_seqlens, pages = _decode_page_table(
        shape["requests"],
        shape["rows_per_request"],
        shape["cache_length"],
        shape["max_context_len"],
        config=config,
        device="cuda",
    )
    query = _randn(
        (rows, config.local_heads, config.latent_cache_dim),
        generator=generator,
        dtype=shape["q_dtype"],
    ).view(rows, 1, config.local_heads, config.latent_cache_dim)
    kv_cache = _randn(
        (pages, config.kv_page_size, 1, config.latent_cache_dim),
        generator=generator,
        dtype=shape["kv_cache_dtype"],
    )
    return {
        "query": query,
        "kv_cache": kv_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "pages": pages,
    }


def _decode_parameters(
    shape: dict[str, object],
    config: _MLAConfig,
    pages: int,
) -> dict[str, object]:
    return {
        **_common_parameters(config),
        "requests": shape["requests"],
        "rows_per_request": shape["rows_per_request"],
        "rows": shape["rows"],
        "cache_length": shape["cache_length"],
        "max_context_len": shape["max_context_len"],
        "q_dtype": _dtype_name(shape["q_dtype"]),
        "kv_cache_dtype": _dtype_name(shape["kv_cache_dtype"]),
        "kv_pages": pages,
    }


def prepare_mla_normalize_project_query(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one fused MLA query/KV RMSNorm and query projection call."""

    config = _resolve_config(request)
    parameters = request.parameters
    tokens = _positive("tokens", parameters["tokens"])
    dtype = _parse_dtype("dtype", parameters["dtype"])
    prepare_absorbed_query = parameters["prepare_absorbed_query"]
    if not isinstance(prepare_absorbed_query, bool):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA prepare_absorbed_query must be a boolean",
        )
    eps = float(parameters["eps"])
    output_width = config.local_heads * config.qk_head_dim

    generator = _generator(request.seed)
    query, kv, _ = _packed_qkv_gate(tokens, config=config, generator=generator)
    query_norm_weight = _randn((config.q_lora_rank,), generator=generator, dtype=dtype)
    kv_norm_weight = _randn((config.kv_lora_rank,), generator=generator, dtype=dtype)
    projection_weight = _randn(
        (output_width, config.q_lora_rank),
        generator=generator,
        dtype=dtype,
    )
    kv_snapshot = kv.clone()

    signature_roles = {
        "query": dtype,
        "kv": dtype,
        "projection_weight": dtype,
        "out": dtype,
    }
    if prepare_absorbed_query:
        signature_roles["tail_out"] = dtype
        prefix_width = config.qk_nope_head_dim
        tail_width = config.qk_rope_head_dim
    else:
        prefix_width = output_width
        tail_width = 0
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles=signature_roles,
        traits={
            "num_tokens": tokens,
            "query_width": config.q_lora_rank,
            "kv_width": config.kv_lora_rank,
            "output_width": output_width,
            "output_prefix_width": prefix_width,
            "output_tail_width": tail_width,
            "inputs_contiguous": all(
                tensor.is_contiguous()
                for tensor in (
                    query,
                    kv,
                    query_norm_weight,
                    kv_norm_weight,
                    projection_weight,
                )
            ),
            "outputs_inner_contiguous": True,
            "split_output": prepare_absorbed_query,
        },
    )

    from tokenspeed_kernel.ops.attention import mla as mla_ops

    def reset() -> None:
        # The operation normalizes the KV latent in place.
        kv.copy_(kv_snapshot)

    def invoke() -> object:
        return mla_ops.mla_normalize_project_query(
            query,
            kv,
            query_norm_weight,
            kv_norm_weight,
            projection_weight,
            eps=eps,
            prepare_absorbed_query=prepare_absorbed_query,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            **_common_parameters(config),
            "tokens": tokens,
            "dtype": _dtype_name(dtype),
            "prepare_absorbed_query": prepare_absorbed_query,
            "eps": eps,
            "input_row_stride": query.stride(0),
        },
        validation=None,
    )


def prepare_mla_decode(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one absorbed MLA decode call over a paged latent cache."""

    config = _resolve_config(request)
    shape = _resolve_decode_shape(request, config)
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={
            "q": shape["q_dtype"],
            "kv_cache": shape["kv_cache_dtype"],
        },
        traits=_decode_traits(shape["rows"], config),
    )

    generator = _generator(request.seed)
    inputs = _decode_inputs(shape, config=config, generator=generator)

    from tokenspeed_kernel.ops.attention import mla as mla_ops

    def invoke() -> object:
        return mla_ops.mla_decode_with_kvcache(
            q=inputs["query"],
            kv_cache=inputs["kv_cache"],
            page_table=inputs["page_table"],
            cache_seqlens=inputs["cache_seqlens"],
            max_seqlen_k=shape["max_context_len"],
            qk_nope_head_dim=config.qk_nope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            softmax_scale=config.softmax_scale,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters=_decode_parameters(shape, config, inputs["pages"]),
        validation=None,
    )


def prepare_mla_decode_projected_value(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one absorbed MLA decode call with fused value projection."""

    config = _resolve_config(request)
    shape = _resolve_decode_shape(request, config)
    output_gate = request.parameters["output_gate"]
    if not isinstance(output_gate, bool):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA output_gate must be a boolean",
        )
    value_dtype = torch.bfloat16
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={
            "q": shape["q_dtype"],
            "kv_cache": shape["kv_cache_dtype"],
            "value_weight": value_dtype,
            "out": value_dtype,
        },
        traits={
            **_decode_traits(shape["rows"], config),
            "value_head_dim": config.v_head_dim,
            "gate_kind": "sigmoid" if output_gate else "none",
        },
    )

    generator = _generator(request.seed)
    inputs = _decode_inputs(shape, config=config, generator=generator)
    value_weight = _randn(
        (config.local_heads, config.kv_lora_rank, config.v_head_dim),
        generator=generator,
        dtype=value_dtype,
    )
    gate = None
    if output_gate:
        _, _, gate = _packed_qkv_gate(shape["rows"], config=config, generator=generator)
    out = torch.empty(
        (shape["rows"], config.gate_width),
        dtype=value_dtype,
        device="cuda",
    )

    from tokenspeed_kernel.ops.attention import mla as mla_ops

    def invoke() -> object:
        return mla_ops.mla_decode_with_kvcache(
            q=inputs["query"],
            kv_cache=inputs["kv_cache"],
            page_table=inputs["page_table"],
            cache_seqlens=inputs["cache_seqlens"],
            max_seqlen_k=shape["max_context_len"],
            qk_nope_head_dim=config.qk_nope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            softmax_scale=config.softmax_scale,
            out=out,
            override=request.registration,
            solution=request.solution,
            value_weight=value_weight,
            gate=gate,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_decode_parameters(shape, config, inputs["pages"]),
            "output_gate": output_gate,
            "gate_row_stride": None if gate is None else gate.stride(0),
        },
        validation=None,
    )


def prepare_mla_prefill(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one non-absorbed MLA prefill call over materialized Q/K/V."""

    config = _resolve_config(request)
    parameters = request.parameters
    batch = _positive("batch", parameters["batch"])
    query_tokens = _positive(
        "query_tokens_per_sequence", parameters["query_tokens_per_sequence"]
    )
    kv_tokens = _positive(
        "kv_tokens_per_sequence", parameters["kv_tokens_per_sequence"]
    )
    is_causal = parameters["is_causal"]
    if not isinstance(is_causal, bool):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA is_causal must be a boolean",
        )
    if is_causal and query_tokens != kv_tokens:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Causal MLA prefill covers only the new tokens, so query and KV "
            "lengths must match",
        )
    return_lse = parameters["return_lse"]
    if not isinstance(return_lse, bool):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MLA return_lse must be a boolean",
        )
    dtype = _parse_dtype("dtype", parameters["dtype"])

    total_q = batch * query_tokens
    total_kv = batch * kv_tokens

    from tokenspeed_kernel.ops.attention import mla as mla_ops

    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        signature_roles={"q": dtype, "k": dtype, "v": dtype},
        traits=mla_ops.mla_prefill_traits(
            batch_size=batch,
            total_q=total_q,
            total_kv=total_kv,
            num_q_heads=config.local_heads,
            head_dim=config.qk_head_dim,
            value_head_dim=config.v_head_dim,
            is_causal=is_causal,
            logit_cap=0.0,
            return_lse=return_lse,
        ),
    )

    generator = _generator(request.seed)
    q = _randn(
        (total_q, config.local_heads, config.qk_head_dim),
        generator=generator,
        dtype=dtype,
    )
    k = _randn(
        (total_kv, config.local_heads, config.qk_head_dim),
        generator=generator,
        dtype=dtype,
    )
    v = _randn(
        (total_kv, config.local_heads, config.v_head_dim),
        generator=generator,
        dtype=dtype,
    )
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda").mul_(
        query_tokens
    )
    cu_seqlens_kv = torch.arange(batch + 1, dtype=torch.int32, device="cuda").mul_(
        kv_tokens
    )
    seq_lens_kv = torch.full((batch,), kv_tokens, dtype=torch.int32, device="cuda")

    def invoke() -> object:
        return mla_ops.mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=query_tokens,
            max_seqlen_kv=kv_tokens,
            softmax_scale=config.softmax_scale,
            seq_lens_kv=seq_lens_kv,
            is_causal=is_causal,
            return_lse=return_lse,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_common_parameters(config),
            "batch": batch,
            "query_tokens_per_sequence": query_tokens,
            "kv_tokens_per_sequence": kv_tokens,
            "is_causal": is_causal,
            "return_lse": return_lse,
            "dtype": _dtype_name(dtype),
        },
        validation=None,
    )
