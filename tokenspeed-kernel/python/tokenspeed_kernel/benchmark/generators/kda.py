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

__all__ = ["prepare_kda_paged_decode", "prepare_kda_paged_prefill"]


_DTYPE_NAMES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
_MODEL_PROFILE = "glm53_flash_tp4"
_DEFAULT_HEADS = 16
_DEFAULT_DIM = 128
_DEFAULT_LOWER_BOUND = -5.0
_DEFAULT_RECURRENT_LAYOUT = "v_major"
_DEFAULT_STATE_PAGES = 841
_DEFAULT_STATE_PAGE_STRIDE = 294912
_DISTINCT_WRITE_PAGE_OFFSET = 16
_STATE_PAGE_RELATIONS = {"in_place", "distinct"}


def _positive_dimension(parameters: dict[str, Any], name: str) -> int:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA parameter {name!r} must be a positive integer",
        )
    return value


def _optional_positive_dimension(
    parameters: dict[str, Any], name: str, fallback: int
) -> int:
    value = parameters.get(name, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA parameter {name!r} must be a positive integer",
        )
    return value


def _parse_dtype(value: object) -> torch.dtype:
    if isinstance(value, str):
        dtype = _DTYPE_NAMES.get(value.lower())
    else:
        dtype = value if isinstance(value, torch.dtype) else None
    if dtype is not torch.bfloat16:
        supported = ", ".join(sorted(_DTYPE_NAMES))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA benchmarks currently support dtype names: {supported}",
        )
    return dtype


def _parse_model_profile(parameters: dict[str, Any]) -> str:
    value = parameters.get("model_profile", _MODEL_PROFILE)
    if value != _MODEL_PROFILE:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA model_profile must be {_MODEL_PROFILE!r}",
        )
    return value


def _parse_lower_bound(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA lower_bound must be a finite number or null",
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA lower_bound must be finite",
        )
    return converted


def _parse_recurrent_layout(parameters: dict[str, Any]) -> str:
    value = parameters.get("recurrent_layout", _DEFAULT_RECURRENT_LAYOUT)
    if value not in {"k_major", "v_major"}:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA recurrent_layout must be 'k_major' or 'v_major'",
        )
    return value


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _randn(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device="cuda", generator=generator)


def _select_registration(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    traits: dict[str, object] | None,
) -> KernelSpec:
    signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k=dense_tensor_format(torch.bfloat16),
        v=dense_tensor_format(torch.bfloat16),
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
    except NoKernelFoundError as exc:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(exc)) from exc

    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec


def _cu_seqlens(
    batch: int,
    tokens_per_sequence: int,
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [index * tokens_per_sequence for index in range(batch + 1)]
    device_boundaries = torch.tensor(values, dtype=torch.int64, device=device)
    host_boundaries = torch.tensor(values, dtype=torch.int64)
    return device_boundaries, host_boundaries


def _packed_beta_logits(
    tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    qkv_width = heads * (2 * key_dim + value_dim)
    projection_width = qkv_width + heads + 2 * key_dim
    projection = _randn(
        (tokens, projection_width),
        dtype=dtype,
        generator=generator,
    )
    return projection[:, qkv_width : qkv_width + heads].view(1, tokens, heads)


def _packed_decode_qkv(
    batch: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    widths = (heads * key_dim, heads * key_dim, heads * value_dim)
    packed = _randn(
        (batch, sum(widths)),
        dtype=dtype,
        generator=generator,
    )
    q_flat, k_flat, v_flat = packed.split(widths, dim=-1)
    return (
        q_flat.view(1, batch, heads, key_dim),
        k_flat.view(1, batch, heads, key_dim),
        v_flat.view(1, batch, heads, value_dim),
    )


def _packed_prefill_inputs(
    tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _randn((1, tokens, heads, key_dim), dtype=dtype, generator=generator)
    k = _randn((1, tokens, heads, key_dim), dtype=dtype, generator=generator)
    v = _randn((1, tokens, heads, value_dim), dtype=dtype, generator=generator)
    g_raw = _randn((1, tokens, heads, key_dim), dtype=dtype, generator=generator)
    beta_logits = _packed_beta_logits(
        tokens,
        heads,
        key_dim,
        value_dim,
        dtype=dtype,
        generator=generator,
    )
    return q, k, v, g_raw, beta_logits


def _decay_parameters(
    heads: int,
    key_dim: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    a_log = torch.randn(
        (heads,),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    dt_bias = torch.randn(
        (heads * key_dim,),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return a_log, dt_bias


def _strided_state_pool(
    state_pages: int,
    heads: int,
    value_dim: int,
    key_dim: int,
    page_stride: int,
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    payload = heads * value_dim * key_dim
    storage_size = (state_pages - 1) * page_stride + payload
    storage = torch.empty(storage_size, dtype=torch.float32, device=device)
    return torch.as_strided(
        storage,
        (state_pages, heads, value_dim, key_dim),
        (page_stride, value_dim * key_dim, key_dim, 1),
    )


def _state_page_indices(
    batch: int,
    relation: str,
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    read_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    write_indices = (
        read_indices.clone()
        if relation == "in_place"
        else read_indices + _DISTINCT_WRITE_PAGE_OFFSET
    )
    return read_indices, write_indices


def _parse_state_page_relation(parameters: dict[str, Any]) -> str:
    relation = parameters.get("state_page_relation", "in_place")
    if relation not in _STATE_PAGE_RELATIONS:
        accepted = ", ".join(sorted(_STATE_PAGE_RELATIONS))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA state_page_relation must be one of: {accepted}",
        )
    return relation


def _normalize_common_parameters(
    request: BenchmarkRequest,
    *,
    allowed: set[str],
) -> tuple[str, int, int, int, torch.dtype, float | None, str]:
    unknown = sorted(set(request.parameters) - allowed)
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown KDA parameters: {', '.join(unknown)}",
        )
    if request.parameters.get("validation") is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA benchmark correctness validation is not implemented yet",
        )
    model_profile = _parse_model_profile(request.parameters)
    heads = _optional_positive_dimension(request.parameters, "heads", _DEFAULT_HEADS)
    key_dim = _optional_positive_dimension(request.parameters, "key_dim", _DEFAULT_DIM)
    value_dim = _optional_positive_dimension(
        request.parameters, "value_dim", _DEFAULT_DIM
    )
    dtype = _parse_dtype(request.parameters.get("dtype", "bfloat16"))
    lower_bound = _parse_lower_bound(
        request.parameters.get("lower_bound", _DEFAULT_LOWER_BOUND)
    )
    recurrent_layout = _parse_recurrent_layout(request.parameters)
    return (
        model_profile,
        heads,
        key_dim,
        value_dim,
        dtype,
        lower_bound,
        recurrent_layout,
    )


def _common_normalized(
    *,
    model_profile: str,
    heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
    recurrent_layout: str,
) -> dict[str, object]:
    return {
        "model_profile": model_profile,
        "heads": heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "dtype": "bfloat16" if dtype is torch.bfloat16 else str(dtype),
        "lower_bound": lower_bound,
        "recurrent_layout": recurrent_layout,
    }


def prepare_kda_paged_prefill(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a GLM-5.3-Flash KDA prefill benchmark."""

    allowed = {
        "model_profile",
        "batch",
        "tokens_per_sequence",
        "heads",
        "key_dim",
        "value_dim",
        "dtype",
        "lower_bound",
        "recurrent_layout",
        "validation",
    }
    (
        model_profile,
        heads,
        key_dim,
        value_dim,
        dtype,
        lower_bound,
        recurrent_layout,
    ) = _normalize_common_parameters(request, allowed=allowed)
    batch = _positive_dimension(request.parameters, "batch")
    tokens_per_sequence = _positive_dimension(request.parameters, "tokens_per_sequence")
    total_tokens = batch * tokens_per_sequence

    load_builtin_kernels()
    spec = _select_registration(request, platform, traits=None)

    generator = _generator(request.seed)
    q, k, v, g_raw, beta_logits = _packed_prefill_inputs(
        total_tokens,
        heads,
        key_dim,
        value_dim,
        dtype=dtype,
        generator=generator,
    )
    a_log, dt_bias = _decay_parameters(heads, key_dim, generator=generator)
    initial_state = torch.randn(
        (batch, heads, value_dim, key_dim),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    cu_seqlens, cu_seqlens_cpu = _cu_seqlens(batch, tokens_per_sequence)

    from tokenspeed_kernel.ops.attention import kda as kda_ops
    from tokenspeed_kernel.ops.attention.gdn.triton import set_total_chunks_hint_uniform

    def invoke() -> object:
        # Preserve the model's device conversion inside the captured work. The
        # hint must bind to the converted tensor to avoid a host read in capture.
        kernel_cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
        set_total_chunks_hint_uniform(
            batch,
            tokens_per_sequence,
            kernel_cu_seqlens,
            (64,),
        )
        return kda_ops.kda_paged_prefill(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            a_log,
            dt_bias,
            initial_state=initial_state,
            cu_seqlens=kernel_cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            capacity=None,
            inputs_packed=False,
            lower_bound=lower_bound,
            override=request.registration,
            solution=request.solution,
            recurrent_layout=recurrent_layout,
        )

    normalized = {
        **_common_normalized(
            model_profile=model_profile,
            heads=heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            lower_bound=lower_bound,
            recurrent_layout=recurrent_layout,
        ),
        "batch": batch,
        "tokens_per_sequence": tokens_per_sequence,
        "total_tokens": total_tokens,
        "beta_token_stride": beta_logits.stride(1),
        "source_sequence_boundaries_dtype": "int64",
        "kernel_sequence_boundaries_dtype": "int32",
    }
    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(
            invoke=invoke,
        ),
        parameters=normalized,
        validation=None,
    )


def prepare_kda_paged_decode(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a GLM-5.3-Flash single-token KDA decode benchmark."""

    allowed = {
        "model_profile",
        "batch",
        "heads",
        "key_dim",
        "value_dim",
        "dtype",
        "lower_bound",
        "recurrent_layout",
        "state_pages",
        "state_page_stride",
        "state_page_relation",
        "validation",
    }
    (
        model_profile,
        heads,
        key_dim,
        value_dim,
        dtype,
        lower_bound,
        recurrent_layout,
    ) = _normalize_common_parameters(request, allowed=allowed)
    batch = _positive_dimension(request.parameters, "batch")
    state_page_relation = _parse_state_page_relation(request.parameters)
    minimum_state_pages = (
        _DISTINCT_WRITE_PAGE_OFFSET + batch + 1
        if state_page_relation == "distinct"
        else batch + 1
    )
    state_pages = _optional_positive_dimension(
        request.parameters,
        "state_pages",
        _DEFAULT_STATE_PAGES,
    )
    if state_pages < minimum_state_pages:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA decode state_pages must be at least {minimum_state_pages}",
        )
    state_page_stride = _optional_positive_dimension(
        request.parameters,
        "state_page_stride",
        _DEFAULT_STATE_PAGE_STRIDE,
    )
    state_payload = heads * value_dim * key_dim
    if state_page_stride < state_payload:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA state_page_stride must be at least {state_payload}",
        )
    traits: dict[str, object] = {
        "indexed_state": True,
        "single_token": True,
        "recurrent_layout": recurrent_layout,
    }

    load_builtin_kernels()
    spec = _select_registration(request, platform, traits)

    generator = _generator(request.seed)
    q, k, v = _packed_decode_qkv(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=dtype,
        generator=generator,
    )
    g_raw = _randn((1, batch, heads, key_dim), dtype=dtype, generator=generator)
    beta_logits = _packed_beta_logits(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=dtype,
        generator=generator,
    )
    a_log, dt_bias = _decay_parameters(heads, key_dim, generator=generator)
    state_pool = _strided_state_pool(
        state_pages,
        heads,
        value_dim,
        key_dim,
        state_page_stride,
    )
    read_indices, write_indices = _state_page_indices(
        batch,
        state_page_relation,
    )
    touched_indices = torch.unique(torch.cat((read_indices, write_indices))).to(
        torch.int64
    )
    state_pool_snapshot = _randn(
        (touched_indices.numel(), heads, value_dim, key_dim),
        dtype=torch.float32,
        generator=generator,
    )
    state_pool.index_copy_(0, touched_indices, state_pool_snapshot)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda")

    from tokenspeed_kernel.ops.attention import kda as kda_ops

    def reset() -> None:
        state_pool.index_copy_(0, touched_indices, state_pool_snapshot)

    def invoke() -> object:
        return kda_ops.kda_paged_decode(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            a_log,
            dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
            override=request.registration,
            solution=request.solution,
            recurrent_layout=recurrent_layout,
        )

    normalized = {
        **_common_normalized(
            model_profile=model_profile,
            heads=heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
            lower_bound=lower_bound,
            recurrent_layout=recurrent_layout,
        ),
        "batch": batch,
        "tokens_per_sequence": 1,
        "state_pages": state_pages,
        "state_page_stride": state_page_stride,
        "state_page_relation": state_page_relation,
        "qkv_token_stride": q.stride(1),
        "beta_token_stride": beta_logits.stride(1),
        "read_write_alias": state_page_relation == "in_place",
    }
    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(
            invoke=invoke,
            reset=reset,
        ),
        parameters=normalized,
        validation=None,
    )
