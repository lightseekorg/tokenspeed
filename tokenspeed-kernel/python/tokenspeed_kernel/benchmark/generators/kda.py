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

from collections.abc import Collection
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
    "prepare_kda_fused_paged_decode",
    "prepare_kda_fused_paged_verify",
    "prepare_kda_paged_decode",
    "prepare_kda_paged_prefill",
    "prepare_kda_replay_commit",
]


_IMPLEMENTED_DTYPES = {
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class _KdaModelProfile:
    """Per-rank KDA geometry and the packed input projection it slices from.

    Models project every KDA input in one GEMM and hand the kernels strided
    views of its rows, so the row width and view offsets are part of the
    benchmarked layout.
    """

    heads: int
    head_dim: int
    projection_width: int
    beta_offset: int
    # Output-gate and decay-gate (f_a) columns, set when the projection feeds
    # the fused pre-convolution kernels.
    gate_offset: int | None = None
    f_a_offset: int | None = None


_MODEL_PROFILES = {
    # Rows are [q | k | v | beta | f_a | f_b].
    "glm53_flash_tp4": _KdaModelProfile(
        heads=16, head_dim=128, projection_width=6416, beta_offset=6144
    ),
    # KimiKDAMergedProj rows are [q | k | v | g | f_a | beta | pad].
    "kimi_k3_tp8": _KdaModelProfile(
        heads=12,
        head_dim=128,
        projection_width=6288,
        beta_offset=6272,
        gate_offset=4608,
        f_a_offset=6144,
    ),
}
_IMPLEMENTED_RECURRENT_LAYOUTS = frozenset({"k_major", "v_major"})
_IMPLEMENTED_STATE_PAGE_RELATIONS = frozenset({"in_place", "distinct"})
_DISTINCT_WRITE_PAGE_OFFSET = 16


def _implemented_value(
    name: str,
    value: object,
    implemented: Collection[str],
) -> str:
    if value not in implemented:
        accepted = ", ".join(sorted(implemented))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Implemented KDA {name} values: {accepted}",
        )
    return value


def _parse_dtype(value: object) -> torch.dtype:
    name = _implemented_value("dtype", value, _IMPLEMENTED_DTYPES)
    return _IMPLEMENTED_DTYPES[name]


def _parse_model_profile(parameters: dict[str, Any]) -> str:
    return _implemented_value(
        "model_profile",
        parameters["model_profile"],
        _MODEL_PROFILES,
    )


def _parse_recurrent_layout(parameters: dict[str, Any]) -> str:
    return _implemented_value(
        "recurrent_layout",
        parameters["recurrent_layout"],
        _IMPLEMENTED_RECURRENT_LAYOUTS,
    )


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
    profile: _KdaModelProfile,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    projection = _randn(
        (tokens, profile.projection_width),
        dtype=dtype,
        generator=generator,
    )
    beta = projection[:, profile.beta_offset : profile.beta_offset + profile.heads]
    return beta.view(1, tokens, profile.heads)


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
    profile: _KdaModelProfile,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (1, tokens, profile.heads, profile.head_dim)
    q = _randn(shape, dtype=dtype, generator=generator)
    k = _randn(shape, dtype=dtype, generator=generator)
    v = _randn(shape, dtype=dtype, generator=generator)
    g_raw = _randn(shape, dtype=dtype, generator=generator)
    beta_logits = _packed_beta_logits(
        tokens,
        profile,
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
    return _implemented_value(
        "state_page_relation",
        parameters["state_page_relation"],
        _IMPLEMENTED_STATE_PAGE_RELATIONS,
    )


def _normalize_common_parameters(
    request: BenchmarkRequest,
) -> tuple[str, int, int, int, torch.dtype, float | None, str]:
    if request.parameters.get("validation") is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA benchmark correctness validation is not implemented yet",
        )
    model_profile = _parse_model_profile(request.parameters)
    heads = request.parameters["heads"]
    key_dim = request.parameters["key_dim"]
    value_dim = request.parameters["value_dim"]
    profile = _MODEL_PROFILES[model_profile]
    if (heads, key_dim, value_dim) != (
        profile.heads,
        profile.head_dim,
        profile.head_dim,
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA {model_profile} uses {profile.heads} heads of width "
            f"{profile.head_dim}",
        )
    dtype = _parse_dtype(request.parameters["dtype"])
    lower_bound = request.parameters["lower_bound"]
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
    """Prepare a KDA prefill benchmark."""

    (
        model_profile,
        heads,
        key_dim,
        value_dim,
        dtype,
        lower_bound,
        recurrent_layout,
    ) = _normalize_common_parameters(request)
    batch = request.parameters["batch"]
    tokens_per_sequence = request.parameters["tokens_per_sequence"]
    total_tokens = batch * tokens_per_sequence

    load_builtin_kernels()
    spec = _select_registration(request, platform, traits=None)

    generator = _generator(request.seed)
    q, k, v, g_raw, beta_logits = _packed_prefill_inputs(
        total_tokens,
        _MODEL_PROFILES[model_profile],
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
    """Prepare a single-token KDA decode benchmark."""

    (
        model_profile,
        heads,
        key_dim,
        value_dim,
        dtype,
        lower_bound,
        recurrent_layout,
    ) = _normalize_common_parameters(request)
    batch = request.parameters["batch"]
    state_page_relation = _parse_state_page_relation(request.parameters)
    minimum_state_pages = (
        _DISTINCT_WRITE_PAGE_OFFSET + batch + 1
        if state_page_relation == "distinct"
        else batch + 1
    )
    state_pages = request.parameters["state_pages"]
    if state_pages < minimum_state_pages:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA decode state_pages must be at least {minimum_state_pages}",
        )
    state_page_stride = request.parameters["state_page_stride"]
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
        _MODEL_PROFILES[model_profile],
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


def _positive_parameter(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA parameter {name!r} must be a positive integer",
        )
    return value


def _bool_parameter(parameters: dict[str, Any], name: str) -> bool:
    value = parameters[name]
    if not isinstance(value, bool):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA parameter {name!r} must be a boolean",
        )
    return value


def _fused_profile(model_profile: str) -> _KdaModelProfile:
    profile = _MODEL_PROFILES[model_profile]
    if profile.gate_offset is None or profile.f_a_offset is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA {model_profile} does not project inputs for the fused kernels",
        )
    return profile


@dataclass(frozen=True)
class _FusedProjection:
    """Strided views of one packed KDA input projection."""

    mixed_qkv: torch.Tensor
    output_gate: torch.Tensor
    f_a_out: torch.Tensor
    beta_logits: torch.Tensor


def _fused_projection(
    rows: int,
    profile: _KdaModelProfile,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> _FusedProjection:
    projection = _randn(
        (rows, profile.projection_width),
        dtype=dtype,
        generator=generator,
    )
    width = profile.heads * profile.head_dim

    def columns(offset: int, count: int) -> torch.Tensor:
        return projection[:, offset : offset + count]

    return _FusedProjection(
        mixed_qkv=columns(0, 3 * width),
        output_gate=columns(profile.gate_offset, width),
        f_a_out=columns(profile.f_a_offset, profile.head_dim),
        beta_logits=columns(profile.beta_offset, profile.heads),
    )


@dataclass(frozen=True)
class _FusedLayerWeights:
    conv_weights: torch.Tensor
    f_b_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


def _fused_layer_weights(
    profile: _KdaModelProfile,
    conv_kernel_size: int,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> _FusedLayerWeights:
    width = profile.heads * profile.head_dim
    a_log, dt_bias = _decay_parameters(
        profile.heads, profile.head_dim, generator=generator
    )
    return _FusedLayerWeights(
        conv_weights=_randn(
            (3 * width, conv_kernel_size), dtype=dtype, generator=generator
        ),
        f_b_weight=_randn((width, profile.head_dim), dtype=dtype, generator=generator),
        a_log=a_log,
        dt_bias=dt_bias,
    )


def _state_arena(
    state_pages: int,
    state_page_bytes: int,
    profile: _KdaModelProfile,
    conv_kernel_size: int,
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(conv_pool, state_pool)`` views over one paged byte arena.

    Each page holds one FP32 V-major recurrent state followed by the BF16
    convolution history, and both pools step by the same page size, matching
    how the cache arena packs KDA state into its pages.
    """
    heads, dim = profile.heads, profile.head_dim
    channels = 3 * heads * dim
    history = conv_kernel_size - 1
    state_bytes = heads * dim * dim * 4
    conv_bytes = channels * history * 2
    if state_page_bytes % 4 or state_page_bytes < state_bytes + conv_bytes:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA state_page_bytes must be a multiple of 4 covering "
            f"{state_bytes + conv_bytes} bytes of state",
        )
    arena = torch.zeros(
        state_pages * state_page_bytes, dtype=torch.uint8, device=device
    )
    state_pool = torch.as_strided(
        arena.view(torch.float32),
        (state_pages, heads, dim, dim),
        (state_page_bytes // 4, dim * dim, dim, 1),
    )
    conv_pool = torch.as_strided(
        arena.view(torch.bfloat16),
        (state_pages, channels, history),
        (state_page_bytes // 2, history, 1),
        storage_offset=state_bytes // 2,
    )
    return conv_pool, state_pool


def _fill_state_pages(
    conv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    pages: torch.Tensor,
    *,
    generator: torch.Generator,
) -> None:
    conv_pool[pages] = _randn(
        (pages.numel(), *conv_pool.shape[1:]),
        dtype=conv_pool.dtype,
        generator=generator,
    )
    state_pool[pages] = _randn(
        (pages.numel(), *state_pool.shape[1:]),
        dtype=state_pool.dtype,
        generator=generator,
    )


def _fused_common(request: BenchmarkRequest) -> dict[str, Any]:
    (
        model_profile,
        heads,
        key_dim,
        value_dim,
        dtype,
        lower_bound,
        recurrent_layout,
    ) = _normalize_common_parameters(request)
    parameters = request.parameters
    return {
        "model_profile": model_profile,
        "profile": _fused_profile(model_profile),
        "dtype": dtype,
        "lower_bound": lower_bound,
        "recurrent_layout": recurrent_layout,
        "batch": _positive_parameter(parameters, "batch"),
        "conv_kernel_size": _positive_parameter(parameters, "conv_kernel_size"),
        "state_pages": _positive_parameter(parameters, "state_pages"),
        "state_page_bytes": _positive_parameter(parameters, "state_page_bytes"),
        "normalized": {
            **_common_normalized(
                model_profile=model_profile,
                heads=heads,
                key_dim=key_dim,
                value_dim=value_dim,
                dtype=dtype,
                lower_bound=lower_bound,
                recurrent_layout=recurrent_layout,
            ),
            "batch": parameters["batch"],
            "conv_kernel_size": parameters["conv_kernel_size"],
            "state_pages": parameters["state_pages"],
            "state_page_bytes": parameters["state_page_bytes"],
        },
    }


def _require_state_pages(state_pages: int, required: int) -> None:
    if state_pages < required:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA state_pages must be at least {required}",
        )


def prepare_kda_fused_paged_decode(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare fused single-token KDA decode from the packed input projection.

    One call runs the short convolution, decay-gate projection, recurrent
    update and gated output RMSNorm against paged state.
    """

    common = _fused_common(request)
    profile = common["profile"]
    batch = common["batch"]
    dtype = common["dtype"]
    state_page_relation = _parse_state_page_relation(request.parameters)
    norm_eps = float(request.parameters["norm_eps"])
    write_page_offset = (
        _DISTINCT_WRITE_PAGE_OFFSET if state_page_relation == "distinct" else 0
    )
    _require_state_pages(common["state_pages"], batch + 1 + write_page_offset)
    traits = {
        "num_heads": profile.heads,
        "head_dim": profile.head_dim,
        "conv_kernel_size": common["conv_kernel_size"],
        "fused_output_norm": True,
        "paged_state": True,
        "recurrent_layout": common["recurrent_layout"],
    }

    load_builtin_kernels()
    spec = _select_registration(request, platform, traits)

    generator = _generator(request.seed)
    # Decode hands the kernel strided views of the projection rows.
    projection = _fused_projection(batch, profile, dtype=dtype, generator=generator)
    weights = _fused_layer_weights(
        profile, common["conv_kernel_size"], dtype=dtype, generator=generator
    )
    norm_weight = _randn((profile.head_dim,), dtype=dtype, generator=generator)
    conv_pool, state_pool = _state_arena(
        common["state_pages"],
        common["state_page_bytes"],
        profile,
        common["conv_kernel_size"],
    )
    read_indices, write_indices = _state_page_indices(batch, state_page_relation)
    touched = torch.unique(torch.cat((read_indices, write_indices))).to(torch.int64)
    _fill_state_pages(conv_pool, state_pool, touched, generator=generator)
    conv_snapshot = conv_pool[touched].clone()
    state_snapshot = state_pool[touched].clone()
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda")

    from tokenspeed_kernel.ops.attention import kda as kda_ops

    def reset() -> None:
        conv_pool[touched] = conv_snapshot
        state_pool[touched] = state_snapshot

    def invoke() -> object:
        return kda_ops.try_kda_fused_paged_decode(
            projection.mixed_qkv,
            weights.conv_weights,
            conv_pool,
            projection.f_a_out,
            weights.f_b_weight,
            projection.beta_logits,
            weights.a_log,
            weights.dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=profile.heads,
            head_dim=profile.head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=common["lower_bound"],
            output_gate=projection.output_gate,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            recurrent_layout=common["recurrent_layout"],
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            **common["normalized"],
            "state_page_relation": state_page_relation,
            "norm_eps": norm_eps,
            "projection_row_stride": projection.mixed_qkv.stride(0),
        },
        validation=None,
    )


def prepare_kda_fused_paged_verify(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare fused KDA target verify over ``draft_token_num`` rows per request.

    With ``capture_replay`` the kernel also writes the Q/K/V, raw decay-gate
    and beta rows that a later replay commit consumes.
    """

    common = _fused_common(request)
    profile = common["profile"]
    batch = common["batch"]
    dtype = common["dtype"]
    draft_token_num = _positive_parameter(request.parameters, "draft_token_num")
    capture_replay = _bool_parameter(request.parameters, "capture_replay")
    store_states = _bool_parameter(request.parameters, "store_states")
    _require_state_pages(common["state_pages"], batch + 1)
    traits = {
        "num_heads": profile.heads,
        "head_dim": profile.head_dim,
        "paged_state": True,
        "recurrent_layout": common["recurrent_layout"],
        "split_producers": False,
        "store_states": store_states,
    }

    load_builtin_kernels()
    spec = _select_registration(request, platform, traits)

    rows = batch * draft_token_num
    width = profile.heads * profile.head_dim
    generator = _generator(request.seed)
    projection = _fused_projection(rows, profile, dtype=dtype, generator=generator)
    # Verify is not a decode forward, so the model compacts Q/K/V first.
    mixed_qkv = projection.mixed_qkv.contiguous()
    weights = _fused_layer_weights(
        profile, common["conv_kernel_size"], dtype=dtype, generator=generator
    )
    conv_pool, state_pool = _state_arena(
        common["state_pages"],
        common["state_page_bytes"],
        profile,
        common["conv_kernel_size"],
    )
    read_indices, write_indices = _state_page_indices(batch, "in_place")
    _fill_state_pages(
        conv_pool, state_pool, read_indices.to(torch.int64), generator=generator
    )
    replay_payload = (
        {
            "replay_mixed_qkv": torch.empty_like(mixed_qkv),
            "replay_gate": torch.empty((rows, width), dtype=dtype, device="cuda"),
            "replay_beta": torch.empty(
                (rows, profile.heads), dtype=dtype, device="cuda"
            ),
        }
        if capture_replay
        else {}
    )

    from tokenspeed_kernel.ops.attention import kda as kda_ops

    def invoke() -> object:
        return kda_ops.try_kda_fused_paged_verify(
            mixed_qkv,
            weights.conv_weights,
            conv_pool,
            conv_pool,
            projection.f_a_out,
            weights.f_b_weight,
            projection.beta_logits,
            weights.a_log,
            weights.dt_bias,
            state_pool=state_pool,
            state_scratch=None,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=profile.heads,
            head_dim=profile.head_dim,
            draft_token_num=draft_token_num,
            lower_bound=common["lower_bound"],
            recurrent_layout=common["recurrent_layout"],
            override=request.registration,
            solution=request.solution,
            store_states=store_states,
            **replay_payload,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **common["normalized"],
            "draft_token_num": draft_token_num,
            "capture_replay": capture_replay,
            "store_states": store_states,
            "projection_row_stride": projection.f_a_out.stride(0),
        },
        validation=None,
    )


def prepare_kda_replay_commit(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare the batched KDA replay commit across every KDA layer.

    One launch replays each request's accepted verify rows from the captured
    payload and commits the resulting convolution and recurrent state for all
    layers. Commits write separate pages so every replay reads the same
    inputs.
    """

    common = _fused_common(request)
    profile = common["profile"]
    batch = common["batch"]
    dtype = common["dtype"]
    parameters = request.parameters
    draft_token_num = _positive_parameter(parameters, "draft_token_num")
    accepted_length = _positive_parameter(parameters, "accepted_length")
    layers = _positive_parameter(parameters, "layers")
    state_groups = _positive_parameter(parameters, "state_groups")
    if accepted_length > draft_token_num:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA accepted_length cannot exceed draft_token_num",
        )
    if layers % state_groups:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "KDA layers must divide evenly into state_groups",
        )
    if common["lower_bound"] is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Batched KDA replay requires a lower_bound",
        )
    _require_state_pages(common["state_pages"], 2 * batch + 1)
    traits = {
        "batched_layers": True,
        "flat_state": True,
        "num_heads": profile.heads,
        "head_dim": profile.head_dim,
    }

    load_builtin_kernels()
    spec = _select_registration(request, platform, traits)
    kernel = KernelRegistry.get().get_impl(spec.name)

    rows = batch * draft_token_num
    width = profile.heads * profile.head_dim
    generator = _generator(request.seed)
    # Replay payloads stack one [rows, width] slab per layer.
    payload_qkv, payload_f_a, payload_beta, payload_gate = (
        _randn((layers, rows, columns), dtype=dtype, generator=generator)
        for columns in (3 * width, profile.head_dim, profile.heads, width)
    )
    read_pages = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    layer_tensors = []
    descriptor_rows = []
    for layer in range(layers):
        weights = _fused_layer_weights(
            profile, common["conv_kernel_size"], dtype=dtype, generator=generator
        )
        conv_pool, state_pool = _state_arena(
            common["state_pages"],
            common["state_page_bytes"],
            profile,
            common["conv_kernel_size"],
        )
        _fill_state_pages(
            conv_pool, state_pool, read_pages.to(torch.int64), generator=generator
        )
        layer_tensors.append((weights, conv_pool, state_pool))
        descriptor_rows.append(
            [
                payload_qkv[layer].data_ptr(),
                weights.conv_weights.data_ptr(),
                conv_pool.data_ptr(),
                payload_f_a[layer].data_ptr(),
                weights.f_b_weight.data_ptr(),
                payload_beta[layer].data_ptr(),
                weights.a_log.data_ptr(),
                weights.dt_bias.data_ptr(),
                state_pool.data_ptr(),
                payload_gate[layer].data_ptr(),
            ]
        )
    descriptors = torch.tensor(descriptor_rows, dtype=torch.uint64, device="cuda")
    layers_per_group = layers // state_groups
    group_indices = torch.arange(layers, dtype=torch.int32, device="cuda").div_(
        layers_per_group, rounding_mode="floor"
    )
    read_indices = read_pages.expand(state_groups, batch).contiguous()
    write_indices = (read_indices + batch).contiguous()
    accepted = torch.full((batch,), accepted_length, dtype=torch.int32, device="cuda")
    _, first_conv, first_state = layer_tensors[0]

    def invoke() -> object:
        # Hold every layer's tensors: the descriptors only record addresses.
        _ = layer_tensors
        return kernel(
            descriptors=descriptors,
            group_indices=group_indices,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted,
            draft_token_num=draft_token_num,
            num_heads=profile.heads,
            head_dim=profile.head_dim,
            f_a_dim=profile.head_dim,
            qkv_stride=payload_qkv.stride(1),
            conv_stride=first_conv.stride(0),
            f_a_stride=payload_f_a.stride(1),
            beta_stride=payload_beta.stride(1),
            state_stride=first_state.stride(0),
            gate_stride=payload_gate.stride(1),
            conv_width=common["conv_kernel_size"],
            lower_bound=common["lower_bound"],
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **common["normalized"],
            "draft_token_num": draft_token_num,
            "accepted_length": accepted_length,
            "layers": layers,
            "state_groups": state_groups,
            "write_page_relation": "distinct",
        },
        validation=None,
    )
