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
from dataclasses import dataclass, replace
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
    "prepare_kda_fused_paged_verify",
    "prepare_kda_paged_prefill",
    "prepare_kda_retained_commit",
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
    # Decay-gate (f_a) column, set when the projection feeds the fused
    # pre-convolution kernels.
    f_a_offset: int | None = None


_MODEL_PROFILES = {
    # Rows are [q | k | v | beta | f_a | g_a].
    "glm53_flash_tp4": _KdaModelProfile(
        heads=16,
        head_dim=128,
        projection_width=6416,
        beta_offset=6144,
        f_a_offset=6160,
    ),
    # KimiKDAMergedProj rows are [q | k | v | g | f_a | beta | pad].
    "kimi_k3_tp8": _KdaModelProfile(
        heads=12,
        head_dim=128,
        projection_width=6288,
        beta_offset=6272,
        f_a_offset=6144,
    ),
}
_IMPLEMENTED_RECURRENT_LAYOUTS = frozenset({"k_major", "v_major"})


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
        set_total_chunks_hint_uniform(batch, tokens_per_sequence, kernel_cu_seqlens)
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


def _positive_parameter(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA parameter {name!r} must be a positive integer",
        )
    return value


def _fused_profile(model_profile: str) -> _KdaModelProfile:
    profile = _MODEL_PROFILES[model_profile]
    if profile.f_a_offset is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA {model_profile} does not project inputs for the fused kernels",
        )
    return profile


@dataclass(frozen=True)
class _FusedProjection:
    """Strided views of one packed KDA input projection."""

    mixed_qkv: torch.Tensor
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


def _history_capacity(window: int) -> int:
    # The runtime's default history capacity for a verify window.
    from tokenspeed_kernel.ops.attention.kda import KDA_DEFAULT_RETAINED_RECORDS

    return window + KDA_DEFAULT_RETAINED_RECORDS


def _retained_benchmark_layer(
    common, window, generator, reads, live, metadata, stamps, count, verified
):
    """One layer's verify inputs over the request slots' shared counts.

    ``count`` is the INT32 [slots, 1] accepted count every layer shares and
    ``verified`` the count its verify reads: the same tensor when no commit
    follows, else a copy of the round's starting count, which the timed
    commits leave as it is.
    """
    from tokenspeed_kernel.ops.attention import kda as kda_ops

    profile = common["profile"]
    rows = reads.numel() * window
    projection = _fused_projection(
        rows, profile, dtype=common["dtype"], generator=generator
    )
    weights = _fused_layer_weights(
        profile, common["conv_kernel_size"], dtype=common["dtype"], generator=generator
    )
    conv, state = _state_arena(
        common["state_pages"],
        common["state_page_bytes"],
        profile,
        common["conv_kernel_size"],
    )
    _fill_state_pages(conv, state, reads.long(), generator=generator)
    capacity = _history_capacity(window)
    shape = (reads.numel(), profile.heads, capacity, profile.head_dim)
    history = kda_ops.KdaRetainedHistory(
        torch.zeros(shape, device="cuda"),
        torch.zeros(shape, device="cuda"),
        torch.zeros(shape, device="cuda"),
        verified,
        live,
        metadata[:, 0],
    )
    records = kda_ops.KdaRetainedRecords(
        torch.empty_like(projection.mixed_qkv),
        history,
        metadata,
        stamps,
        torch.arange(reads.numel() + 1, device="cuda", dtype=torch.int32) * window,
    )
    return dict(
        projection=projection,
        weights=weights,
        conv=conv,
        state=state,
        records=records,
        count=count,
        reads=reads,
        window=window,
    )


def _invoke_retained_layer(layer, common, *, override, solution):
    from tokenspeed_kernel.ops.attention import kda as kda_ops

    profile = common["profile"]
    projection, weights, records = (
        layer["projection"],
        layer["weights"],
        layer["records"],
    )
    split = not kda_ops.kda_fused_paged_verify_fuses_producers(
        common["dtype"],
        recurrent_layout=common["recurrent_layout"],
        num_heads=profile.heads,
        head_dim=profile.head_dim,
        draft_token_num=layer["window"],
        batch_size=layer["reads"].numel(),
    )
    producers = {}
    if split:
        producers["g_raw"] = torch.nn.functional.linear(
            projection.f_a_out, weights.f_b_weight
        )
        # As in serving, the convolution keeps the pre-convolution rows, so
        # the split verify has none to copy.
        producers["conv_qkv"] = kda_ops.kda_verify_conv_update(
            projection.mixed_qkv,
            weights.conv_weights,
            layer["conv"],
            layer["reads"],
            num_heads=profile.heads,
            head_dim=profile.head_dim,
            draft_token_num=layer["window"],
            query_start_loc=records.query_start_loc,
            original_qkv=records.original_qkv,
            recurrent_layout=common["recurrent_layout"],
        )
        records = replace(records, original_qkv=None)
    return kda_ops.kda_fused_paged_verify(
        projection.mixed_qkv,
        weights.conv_weights,
        layer["conv"],
        projection.f_a_out,
        weights.f_b_weight,
        projection.beta_logits,
        weights.a_log,
        weights.dt_bias,
        state_pool=layer["state"],
        read_indices=layer["reads"],
        num_heads=profile.heads,
        head_dim=profile.head_dim,
        draft_token_num=layer["window"],
        lower_bound=common["lower_bound"],
        recurrent_layout=common["recurrent_layout"],
        retained_records=records,
        output_norm=None,
        override=override,
        solution=solution,
        **producers,
    )


def prepare_kda_fused_paged_verify(
    request: BenchmarkRequest, platform: PlatformInfo
) -> PreparedBenchmark:
    """Benchmark a candidate window at a fixed accepted history length."""
    common = _fused_common(request)
    profile, batch = common["profile"], common["batch"]
    window = _positive_parameter(request.parameters, "draft_token_num")
    capacity = _history_capacity(window)
    history_length = request.parameters.get("history_length", 0)
    if type(history_length) is not int or not 0 <= history_length <= capacity - window:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"KDA history_length must be an integer from 0 to {capacity - window}",
        )
    _require_state_pages(common["state_pages"], batch + 1)
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        {
            "num_heads": profile.heads,
            "head_dim": profile.head_dim,
            "paged_state": True,
            "retained_history": True,
            "recurrent_layout": common["recurrent_layout"],
            "spec_query_len": window,
            "batch_size": batch,
        },
    )
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 2] = history_length
    stamps = torch.full((batch, window, 2), -1, device="cuda", dtype=torch.int64)
    generator = _generator(request.seed)
    # No commit follows, so the verify reads the count itself.
    count = torch.zeros(batch, 1, device="cuda", dtype=torch.int32)
    layer = _retained_benchmark_layer(
        common, window, generator, reads, reads.clone(), metadata, stamps, count, count
    )
    initial_conv, initial_state = layer["conv"].clone(), layer["state"].clone()
    history = layer["records"].history
    accepted = slice(0, history_length)
    history.length.fill_(history_length)
    if history_length:
        # Every request slot has an accepted prefix. Reset the candidate
        # suffix between timings and keep those immutable records in place.
        history.correction[:, :, accepted].normal_(std=0.02, generator=generator)
        keys = history.key[:, :, accepted]
        keys.normal_(generator=generator)
        keys.copy_(torch.nn.functional.normalize(keys, dim=-1))
        history.decay[:, :, accepted].uniform_(0.7, 0.95, generator=generator)

    def reset():
        layer["conv"].copy_(initial_conv)
        layer["state"].copy_(initial_state)
        history.length.fill_(history_length)
        for records in (history.correction, history.key, history.decay):
            records[:, :, history_length : history_length + window].zero_()
        stamps.fill_(-1)

    def invoke():
        return _invoke_retained_layer(layer, common, override=spec.name, solution=None)

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            **common["normalized"],
            "draft_token_num": window,
            "history_capacity": capacity,
            "history_length": history_length,
            "projection_row_stride": layer["projection"].f_a_out.stride(0),
        },
        validation=None,
    )


def prepare_kda_retained_commit(
    request: BenchmarkRequest, platform: PlatformInfo
) -> PreparedBenchmark:
    """Time all-layer retained commit from fresh verify-derived records each replay."""
    common = _fused_common(request)
    profile, batch = common["profile"], common["batch"]
    window = _positive_parameter(request.parameters, "draft_token_num")
    take = _positive_parameter(request.parameters, "accepted_length")
    layers = _positive_parameter(request.parameters, "layers")
    groups = _positive_parameter(request.parameters, "state_groups")
    if take > window or layers % groups:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Accepted length must fit the window and layers must divide evenly into state groups",
        )
    _require_state_pages(common["state_pages"], 2 * batch + 1)
    load_builtin_kernels()
    spec = _select_registration(
        request,
        platform,
        {
            "batched_layers": True,
            "retained_history": True,
            "num_heads": profile.heads,
            "head_dim": profile.head_dim,
            "recurrent_layout": common["recurrent_layout"],
        },
    )
    kernel = KernelRegistry.get().get_impl(spec.name)
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    live = reads + batch
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    stamps = torch.full(
        (layers, batch, window, 2), -1, device="cuda", dtype=torch.int64
    )
    generator = _generator(request.seed)
    # Every layer shares a request's count; its verify read the round's start.
    replay_length = torch.zeros(batch, 1, device="cuda", dtype=torch.int32)
    verified = replay_length.clone()
    layer_inputs, descriptors, before = [], [], []
    for layer_idx in range(layers):
        layer = _retained_benchmark_layer(
            common,
            window,
            generator,
            reads,
            live,
            metadata,
            stamps[layer_idx],
            replay_length,
            verified,
        )
        _invoke_retained_layer(layer, common, override=None, solution=None)
        conv = layer["conv"]
        qkv, state = layer["records"].original_qkv, layer["state"]
        descriptors.append(
            [
                qkv.data_ptr(),
                conv.data_ptr(),
                conv.shape[0],
                qkv.stride(0),
                conv.stride(0),
                conv.stride(1),
                conv.stride(2),
                state.data_ptr(),
                *state.stride(),
            ]
        )
        layer_inputs.append(layer)
        before.append(conv.clone())
    descriptors = torch.tensor(descriptors, device="cuda", dtype=torch.uint64)
    initial_count = replay_length.clone()
    group_indices = torch.arange(layers, device="cuda", dtype=torch.int32) // (
        layers // groups
    )
    read_indices = reads.expand(groups, batch).contiguous()
    write_indices = live.expand(groups, batch).contiguous()
    accepted = torch.full((batch,), take, device="cuda", dtype=torch.int32)
    # The timed commit is a step in which no request ends, so it reads no
    # record; the first layer's records stand in for every layer's.
    first = layer_inputs[0]["records"].history
    records = [
        field[:, None].expand(-1, layers, -1, -1, -1)
        for field in (first.correction, first.key, first.decay)
    ]
    request_stopped = torch.zeros(batch, device="cuda", dtype=torch.bool)
    # No request ends in the measured rounds, so no row's report changes.
    request_ends = torch.zeros(batch, device="cuda", dtype=torch.int32)

    def reset():
        for layer, conv in zip(layer_inputs, before):
            layer["conv"].copy_(conv)
        replay_length.copy_(initial_count)

    def invoke():
        # Keep the layers' tensors alive while descriptors contain their addresses.
        _ = layer_inputs
        return kernel(
            descriptors=descriptors,
            group_indices=group_indices,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted,
            query_start_loc=layer_inputs[0]["records"].query_start_loc,
            record_slots=metadata[:, 0],
            record_slot_count=batch,
            record_metadata=metadata,
            record_stamps=stamps,
            retained_correction=records[0],
            retained_key=records[1],
            retained_decay=records[2],
            replay_length=replay_length,
            request_stopped=request_stopped,
            request_ends=request_ends,
            draft_token_num=window,
            history_capacity=_history_capacity(window),
            num_heads=profile.heads,
            head_dim=profile.head_dim,
            conv_width=common["conv_kernel_size"],
            activation_dtype=common["dtype"],
            conv_state_dtype=layer_inputs[0]["conv"].dtype,
            recurrent_state_dtype=layer_inputs[0]["state"].dtype,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            **common["normalized"],
            "draft_token_num": window,
            "accepted_length": take,
            "layers": layers,
            "state_groups": groups,
            "history_capacity": _history_capacity(window),
            "write_page_relation": "distinct",
        },
        validation=None,
    )
