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

"""Benchmark generators for residual-stream operations."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
    PreparedValidation,
    ValidationInvocation,
)
from tokenspeed_kernel.benchmark.validation import OutputValidationSpec
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "prepare_attn_res_fwd",
    "prepare_hyperconnection_mix",
    "prepare_mhc_apply_pre",
    "prepare_mhc_mixes",
    "prepare_mhc_post",
    "prepare_mhc_pre",
    "prepare_post_norm_residual_pre_norm",
]


_IMPLEMENTED_MODEL_PROFILES = frozenset({"kimi_k3_tp8"})
_IMPLEMENTED_MHC_MODEL_PROFILES = frozenset({"dsv41_flash_tp4"})
_IMPLEMENTED_GLM_MHC_MODEL_PROFILES = frozenset({"glm53_flash_tp4"})
_IMPLEMENTED_HC_MIX_MODEL_PROFILES = frozenset({"qwen38_flash_tp2"})
_IMPLEMENTED_GATED_BOUNDARY_MODEL_PROFILES = frozenset({"muse_spark_tp8"})
_GATED_BOUNDARY_OUTPUTS = frozenset({"fp32_and_bf16", "bf16"})


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"Residual {name} must be a positive integer")
    return value


def _flag(parameters: dict[str, Any], name: str) -> bool:
    value = parameters[name]
    if not isinstance(value, bool):
        raise _invalid(f"Residual {name} must be a boolean")
    return value


def _random_tensors(
    seed: int,
    *shapes: tuple[int, ...],
) -> list[torch.Tensor]:
    """Return seeded standard-normal BF16 device tensors, one per shape."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for shape in shapes
    ]


def prepare_attn_res_fwd(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one AttnRes mix with its fused output RMSNorm.

    The call mixes ``num_valid_blocks`` residual snapshots and the current
    residual, then applies the following RMSNorm, as Kimi K3 does before
    attention and before the MLP. Calls that append a snapshot pass the whole
    ``block_slots``-row storage, others pass only the valid snapshots.
    ``has_delta`` adds the reduced attention output into the residual first,
    in place.
    """

    _ = platform
    if request.registration is not None or request.solution is not None:
        raise _invalid(
            "attn_res_fwd selects its own kernel; AttnRes cases use normal selection"
        )
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid("AttnRes benchmark correctness validation is not implemented")
    model_profile = parameters["model_profile"]
    if model_profile not in _IMPLEMENTED_MODEL_PROFILES:
        raise _invalid(
            f"Implemented AttnRes model_profile values: "
            f"{', '.join(sorted(_IMPLEMENTED_MODEL_PROFILES))}"
        )
    if parameters["dtype"] != "bfloat16":
        raise _invalid("AttnRes dtype must be bfloat16")
    tokens = _positive(parameters, "tokens")
    hidden_size = _positive(parameters, "hidden_size")
    block_slots = _positive(parameters, "block_slots")
    num_valid_blocks = parameters["num_valid_blocks"]
    if not isinstance(num_valid_blocks, int) or isinstance(num_valid_blocks, bool):
        raise _invalid("AttnRes num_valid_blocks must be an integer")
    block_write = _flag(parameters, "block_write")
    has_delta = _flag(parameters, "has_delta")
    eps = parameters["eps"]
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise _invalid("AttnRes eps must be a positive float")
    if not 0 <= num_valid_blocks <= block_slots - int(block_write):
        raise _invalid("AttnRes snapshots must fit block_slots")
    if num_valid_blocks == 0 and not block_write and not has_delta:
        raise _invalid("AttnRes without snapshots, a write or a delta is a plain norm")

    layer_residual, storage, res_weight, rms_weight, out_norm_weight, delta = (
        _random_tensors(
            request.seed,
            (tokens, hidden_size),
            (block_slots, tokens, hidden_size),
            (hidden_size,),
            (hidden_size,),
            (hidden_size,),
            (tokens, hidden_size),
        )
    )
    blocks = storage if block_write else storage[:num_valid_blocks]
    if not has_delta:
        delta = None
    block_write_idx = num_valid_blocks if block_write else -1

    from tokenspeed_kernel.ops import residual as residual_ops

    load_builtin_kernels()
    try:
        kernel, _ = residual_ops.select_attn_res_fwd_kernel(
            layer_residual,
            blocks,
            res_weight,
            rms_weight,
            out_norm_weight=out_norm_weight,
            output_eps=eps,
            eps=eps,
            delta=delta,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {kernel.name!r} is not available",
        )

    # A delta is folded into the residual in place; restore it between calls.
    residual_snapshot = layer_residual.clone() if has_delta else None

    def reset() -> None:
        if residual_snapshot is not None:
            layer_residual.copy_(residual_snapshot)

    def invoke() -> object:
        return residual_ops.attn_res_fwd(
            layer_residual,
            blocks,
            res_weight,
            rms_weight,
            eps,
            out_norm_weight,
            eps,
            delta=delta,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hidden_size": hidden_size,
            "block_slots": block_slots,
            "num_valid_blocks": num_valid_blocks,
            "block_write": block_write,
            "has_delta": has_delta,
            "eps": eps,
            "dtype": "bfloat16",
            "block_storage_rows": blocks.shape[0],
        },
        validation=None,
    )


def _mhc_common(request: BenchmarkRequest, mode: str) -> tuple[str, int, int, int]:
    """Validate the shared mHC request fields; returns profile, T, HC and H."""
    if request.registration is not None or request.solution is not None:
        raise _invalid(f"{mode} cases use normal kernel selection")
    parameters = request.parameters
    if parameters.get("validation") is not None and mode not in {"mhc_pre", "mhc_post"}:
        raise _invalid("mHC benchmark correctness validation is not implemented")
    model_profile = parameters["model_profile"]
    profiles = _IMPLEMENTED_MHC_MODEL_PROFILES
    if mode == "mhc_pre":
        profiles = _IMPLEMENTED_GLM_MHC_MODEL_PROFILES
    elif mode == "mhc_post":
        profiles = profiles | _IMPLEMENTED_GLM_MHC_MODEL_PROFILES
    if model_profile not in profiles:
        raise _invalid(
            f"Implemented mHC model_profile values: " f"{', '.join(sorted(profiles))}"
        )
    if parameters["dtype"] != "bfloat16":
        raise _invalid("mHC dtype must be bfloat16")
    return (
        model_profile,
        _positive(parameters, "tokens"),
        _positive(parameters, "hc_mult"),
        _positive(parameters, "hidden_size"),
    )


def _registered_spec(name: str):
    spec = KernelRegistry.get().get_by_name(name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {name!r} is not available",
        )
    return spec


def _mhc_validation_config(value: object) -> dict[str, float | int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _invalid("mHC validation must be an object")
    unknown = sorted(set(value) - {"runs", "atol", "rtol"})
    if unknown:
        raise _invalid(f"Unknown mHC validation parameters: {', '.join(unknown)}")
    config = {"runs": value.get("runs", 1)}
    config["runs"] = _positive(config, "runs")
    # Match the existing complete-mHC numerical tests. Persistent inputs use
    # exact comparison separately, so these tolerances cannot hide mutation.
    for name in ("atol", "rtol"):
        tolerance = value.get(name, 2e-2)
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise _invalid(f"mHC validation {name} must be finite and nonnegative")
        config[name] = float(tolerance)
    return config


def _prepare_mhc_validation(
    config: dict[str, float | int] | None,
    seed: int,
    generate_inputs: Callable[[int], tuple[torch.Tensor, ...]],
    invoke: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    reference: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    *,
    output_count: int,
    input_count: int,
) -> PreparedValidation | None:
    if config is None:
        return None

    def prepare_run(run_index: int) -> ValidationInvocation:
        inputs = generate_inputs(seed + run_index)
        snapshots = tuple(tensor.clone() for tensor in inputs)
        return ValidationInvocation(
            candidate=lambda: (*invoke(inputs), *inputs),
            reference=lambda: (*reference(snapshots), *snapshots),
        )

    return PreparedValidation(
        output_specs=(
            *(
                OutputValidationSpec(
                    "close", {"atol": config["atol"], "rtol": config["rtol"]}
                )
                for _ in range(output_count)
            ),
            *(
                OutputValidationSpec("close", {"atol": 0.0, "rtol": 0.0})
                for _ in range(input_count)
            ),
        ),
        runs=config["runs"],
        prepare_run=prepare_run,
    )


def prepare_mhc_pre(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare GLM's complete pre-operation, including its prenorm projection.

    The public operation computes all mixing coefficients and collapses the
    four residual streams. GLM applies output RMSNorm separately, so
    ``fused_norm`` must explicitly be false. Correctness calls and snapshots
    run outside the timed invocation.
    """

    model_profile, tokens, hc_mult, hidden_size = _mhc_common(request, "mhc_pre")
    parameters = request.parameters
    if hc_mult != 4 or hidden_size != 4096:
        raise _invalid("GLM mhc_pre requires hc_mult 4 and hidden_size 4096")
    fused_norm = _flag(parameters, "fused_norm")
    if fused_norm:
        raise _invalid("GLM mhc_pre requires separate output normalization")
    sinkhorn_iters = _positive(parameters, "sinkhorn_iters")
    epsilons = {}
    for name in ("rms_eps", "hc_eps"):
        value = parameters[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise _invalid(f"mHC {name} must be finite and positive")
        epsilons[name] = float(value)
    rms_eps, hc_eps = epsilons["rms_eps"], epsilons["hc_eps"]
    validation_config = _mhc_validation_config(parameters.get("validation"))

    from tokenspeed_kernel.numerics.torch.residual import torch_mhc_pre
    from tokenspeed_kernel.ops import residual as residual_ops

    load_builtin_kernels()
    try:
        kernel = select_kernel(
            "residual",
            "mhc_pre",
            format_signature(
                residual=dense_tensor_format(torch.bfloat16),
                fn=dense_tensor_format(torch.float32),
                hc_scale=dense_tensor_format(torch.float32),
                hc_base=dense_tensor_format(torch.float32),
            ),
            platform=platform,
            traits={
                "num_tokens": tokens,
                "buffer_offsets_fit_int32": tokens * hc_mult * hidden_size < 2**31,
                "hc_mult": hc_mult,
                "hidden_size": hidden_size,
                "sinkhorn_iters": sinkhorn_iters,
            },
            override=None,
            solution=None,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error

    def generate_inputs(seed: int) -> tuple[torch.Tensor, ...]:
        (residual,) = _random_tensors(seed, (tokens, hc_mult, hidden_size))
        generator = torch.Generator(device="cuda").manual_seed(seed + 1)
        mix_width = 2 * hc_mult + hc_mult * hc_mult
        projection = torch.randn(
            (mix_width, hc_mult * hidden_size),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.01)
        scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
        bias = torch.randn(
            mix_width, device="cuda", dtype=torch.float32, generator=generator
        ).mul_(0.01)
        return residual, projection, scale, bias

    def invoke_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return residual_ops.mhc_pre(
            *inputs,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            override=None,
            solution=None,
            norm_weight=None,
            norm_eps=None,
        )

    def reference(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return torch_mhc_pre(*inputs, rms_eps, hc_eps, sinkhorn_iters, None, None)

    inputs = generate_inputs(request.seed)
    normalized_parameters = {
        "model_profile": model_profile,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
        "sinkhorn_iters": sinkhorn_iters,
        "rms_eps": rms_eps,
        "hc_eps": hc_eps,
        "dtype": "bfloat16",
        "fused_norm": fused_norm,
    }
    if validation_config is not None:
        normalized_parameters["validation"] = validation_config
    return PreparedBenchmark(
        registration=_registered_spec(kernel.name),
        # The operation allocates outputs and must not mutate persistent inputs.
        # No copies or reference work belong in its graph or timed interval.
        invocation=PreparedInvocation(invoke=lambda: invoke_inputs(inputs)),
        parameters=normalized_parameters,
        validation=_prepare_mhc_validation(
            validation_config,
            request.seed,
            generate_inputs,
            invoke_inputs,
            reference,
            output_count=3,
            input_count=4,
        ),
    )


def prepare_mhc_mixes(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one mHC coefficient computation over the full residual stream.

    DeepSeek V4.1 runs this before attention and before the FFN of every layer:
    an RMS-normalized ``[T, HC*H] x [HC*H, 2*HC + HC*HC]`` projection followed
    by the sigmoid pre/post mixes and a Sinkhorn-normalized combine matrix.
    """

    model_profile, tokens, hc_mult, hidden_size = _mhc_common(request, "mhc_mixes")
    parameters = request.parameters
    if hc_mult != 4:
        raise _invalid("mhc_mixes requires hc_mult 4")
    sinkhorn_iters = _positive(parameters, "sinkhorn_iters")
    rms_eps = float(parameters["rms_eps"])
    hc_eps = float(parameters["hc_eps"])
    mix_width = 2 * hc_mult + hc_mult * hc_mult

    from tokenspeed_kernel.ops import residual as residual_ops

    load_builtin_kernels()
    try:
        kernel = select_kernel(
            "residual",
            "mhc_mixes",
            format_signature(residual=dense_tensor_format(torch.bfloat16)),
            platform=platform,
            traits=None,
            override=None,
            solution=None,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error

    (residual,) = _random_tensors(request.seed, (tokens, hc_mult, hidden_size))
    generator = torch.Generator(device="cuda").manual_seed(request.seed + 1)
    weight = torch.randn(
        (mix_width, hc_mult * hidden_size),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.01)
    scale = torch.ones(3, device="cuda", dtype=torch.float32)
    base = torch.zeros(mix_width, device="cuda", dtype=torch.float32)

    def invoke() -> object:
        return residual_ops.mhc_mixes(
            residual, weight, scale, base, rms_eps, hc_eps, sinkhorn_iters
        )

    return PreparedBenchmark(
        registration=_registered_spec(kernel.name),
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hc_mult": hc_mult,
            "hidden_size": hidden_size,
            "sinkhorn_iters": sinkhorn_iters,
            "rms_eps": rms_eps,
            "hc_eps": hc_eps,
            "dtype": "bfloat16",
        },
        validation=None,
    )


def prepare_mhc_apply_pre(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one mHC collapse of the residual streams into a layer input.

    DeepSeek V4.1 collapses the streams with the previous sublayer's pre
    coefficients before attention and before the FFN of every layer, fusing
    the following RMSNorm (``fused_norm``); only the final norm uses the plain
    collapse. ``rms_eps`` is given exactly when ``fused_norm`` is set.
    """

    model_profile, tokens, hc_mult, hidden_size = _mhc_common(request, "mhc_apply_pre")
    parameters = request.parameters
    fused_norm = parameters["fused_norm"]
    if not isinstance(fused_norm, bool):
        raise _invalid("mhc_apply_pre fused_norm must be a boolean")
    rms_eps = parameters.get("rms_eps")
    if fused_norm != (rms_eps is not None):
        raise _invalid("mhc_apply_pre takes rms_eps exactly when fused_norm is set")
    if rms_eps is not None:
        rms_eps = float(rms_eps)
    from tokenspeed_kernel.ops import residual as residual_ops

    load_builtin_kernels()
    try:
        kernel = select_kernel(
            "residual",
            "mhc_apply_pre",
            format_signature(
                residual=dense_tensor_format(torch.bfloat16),
                pre=dense_tensor_format(torch.float32),
            ),
            platform=platform,
            traits={
                "buffer_offsets_fit_int32": tokens * hc_mult * hidden_size < 2**31,
                "fused_norm": fused_norm,
                "hc_mult": hc_mult,
            },
            override=None,
            solution=None,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error

    residual, norm_weight = _random_tensors(
        request.seed,
        (tokens, hc_mult, hidden_size),
        (hidden_size,),
    )
    if not fused_norm:
        norm_weight = None
    generator = torch.Generator(device="cuda").manual_seed(request.seed + 1)
    # Pre coefficients are sigmoid outputs.
    pre = torch.rand(
        (tokens, hc_mult), device="cuda", dtype=torch.float32, generator=generator
    )

    def invoke() -> object:
        return residual_ops.mhc_apply_pre(
            residual, pre, norm_weight=norm_weight, norm_eps=rms_eps
        )

    return PreparedBenchmark(
        registration=_registered_spec(kernel.name),
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hc_mult": hc_mult,
            "hidden_size": hidden_size,
            "fused_norm": fused_norm,
            "rms_eps": rms_eps,
            "dtype": "bfloat16",
        },
        validation=None,
    )


def prepare_mhc_post(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one mHC post-mapping: scatter a sublayer output into the streams."""

    model_profile, tokens, hc_mult, hidden_size = _mhc_common(request, "mhc_post")
    if model_profile in _IMPLEMENTED_GLM_MHC_MODEL_PROFILES and (
        hc_mult != 4 or hidden_size != 4096
    ):
        raise _invalid("GLM mhc_post requires hc_mult 4 and hidden_size 4096")
    validation_config = _mhc_validation_config(request.parameters.get("validation"))

    from tokenspeed_kernel.numerics.torch.residual import torch_mhc_post
    from tokenspeed_kernel.ops import residual as residual_ops

    load_builtin_kernels()
    try:
        kernel = select_kernel(
            "residual",
            "mhc_post",
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                residual=dense_tensor_format(torch.bfloat16),
                post=dense_tensor_format(torch.float32),
                comb=dense_tensor_format(torch.float32),
            ),
            platform=platform,
            traits={
                "num_tokens": tokens,
                "buffer_offsets_fit_int32": tokens * hc_mult * hidden_size < 2**31,
                "hc_mult": hc_mult,
                "hidden_size": hidden_size,
            },
            override=None,
            solution=None,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error

    def generate_inputs(seed: int) -> tuple[torch.Tensor, ...]:
        hidden_states, residual = _random_tensors(
            seed,
            (tokens, hidden_size),
            (tokens, hc_mult, hidden_size),
        )
        generator = torch.Generator(device="cuda").manual_seed(seed + 1)
        post = 2.0 * torch.rand(
            (tokens, hc_mult, 1),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        # Sinkhorn output is doubly stochastic; any row-normalized matrix has
        # the same cost. Preserve the existing DSV post-input distribution.
        comb = torch.rand(
            (tokens, hc_mult, hc_mult),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        comb = comb / comb.sum(-1, keepdim=True)
        return hidden_states, residual, post, comb

    inputs = generate_inputs(request.seed)

    def invoke() -> object:
        return residual_ops.mhc_post(*inputs, override=None, solution=None)

    normalized_parameters = {
        "model_profile": model_profile,
        "tokens": tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
        "dtype": "bfloat16",
    }
    if validation_config is not None:
        normalized_parameters["validation"] = validation_config

    return PreparedBenchmark(
        registration=_registered_spec(kernel.name),
        invocation=PreparedInvocation(invoke=invoke),
        parameters=normalized_parameters,
        validation=_prepare_mhc_validation(
            validation_config,
            request.seed,
            generate_inputs,
            lambda values: (
                residual_ops.mhc_post(*values, override=None, solution=None),
            ),
            lambda values: (torch_mhc_post(*values),),
            output_count=1,
            input_count=4,
        ),
    )


def prepare_hyperconnection_mix(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one Qwen4-style hyper-connection mix.

    The call projects the normalized ``hc_count`` residual streams down to the
    ``lowrank`` mix gate (plus ``hc_count`` inject logits when ``inject`` is
    set, as for the per-layer mixers; the final mixer has none), applies the
    gate and mixes the streams back to one ``hidden_size`` input, as
    ``GatedResidualSimple.mix`` does. Projection storage is padded once during
    preparation, matching model loading, and invocation uses its logical width.
    """

    _ = platform
    if request.registration is not None or request.solution is not None:
        raise _invalid("hyperconnection_mix cases use normal kernel selection")
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid("hyperconnection_mix correctness validation is not implemented")
    model_profile = parameters["model_profile"]
    if model_profile not in _IMPLEMENTED_HC_MIX_MODEL_PROFILES:
        raise _invalid(
            "Implemented hyperconnection_mix model_profile values: "
            f"{', '.join(sorted(_IMPLEMENTED_HC_MIX_MODEL_PROFILES))}"
        )
    if parameters["dtype"] != "bfloat16":
        raise _invalid("hyperconnection_mix dtype must be bfloat16")
    tokens = _positive(parameters, "tokens")
    hc_count = _positive(parameters, "hc_count")
    hidden_size = _positive(parameters, "hidden_size")
    lowrank = _positive(parameters, "lowrank")
    inject = _flag(parameters, "inject")
    wide = hc_count * hidden_size
    projection_rows = lowrank + (hc_count if inject else 0)

    normalized, projection_weight, up_weight = _random_tensors(
        request.seed,
        (tokens, wide),
        (projection_rows, wide),
        (wide, lowrank),
    )

    from tokenspeed_kernel.ops import residual as residual_ops

    projection_weight = residual_ops.pad_gated_residual_projection_weight(
        projection_weight, projection_rows=projection_rows
    )
    load_builtin_kernels()
    try:
        kernel, _ = residual_ops.select_gated_residual_mix_kernel(
            normalized,
            projection_weight,
            up_weight,
            hc_count,
            hidden_size,
            lowrank,
            weights_independent=True,
            override=None,
            solution=None,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {kernel.name!r} is not available",
        )

    def invoke() -> object:
        return residual_ops.gated_residual_mix(
            normalized,
            projection_weight,
            up_weight,
            hc_count,
            hidden_size,
            lowrank,
            projection_rows=projection_rows,
            weights_independent=True,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hc_count": hc_count,
            "hidden_size": hidden_size,
            "lowrank": lowrank,
            "inject": inject,
            "dtype": "bfloat16",
        },
        validation=None,
    )


def prepare_post_norm_residual_pre_norm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one fused decoder-layer boundary of a gated FP32 residual stream.

    The call normalizes the BF16 branch (``post_weighted`` selects an
    input-scale weight, else a weightless norm), adds it to the FP32 stream
    through the norm-preserving residual gate and, with ``pre_weighted``,
    applies the next sublayer's affine RMSNorm. ``outputs="bf16"`` runs the
    entry that returns only the BF16 next input, as Muse Spark's FFN boundary
    does on CDNA4; ``"fp32_and_bf16"`` also returns the FP32 rows its router
    reads after attention.
    """

    _ = platform
    if request.registration is not None or request.solution is not None:
        raise _invalid("post_norm_residual_pre_norm cases use normal kernel selection")
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid(
            "post_norm_residual_pre_norm correctness validation is not implemented"
        )
    model_profile = parameters["model_profile"]
    if model_profile not in _IMPLEMENTED_GATED_BOUNDARY_MODEL_PROFILES:
        raise _invalid(
            "Implemented post_norm_residual_pre_norm model_profile values: "
            f"{', '.join(sorted(_IMPLEMENTED_GATED_BOUNDARY_MODEL_PROFILES))}"
        )
    if parameters["dtype"] != "bfloat16":
        raise _invalid("post_norm_residual_pre_norm branch dtype must be bfloat16")
    tokens = _positive(parameters, "tokens")
    hidden_size = _positive(parameters, "hidden_size")
    post_weighted = _flag(parameters, "post_weighted")
    pre_weighted = _flag(parameters, "pre_weighted")
    outputs = parameters["outputs"]
    if outputs not in _GATED_BOUNDARY_OUTPUTS:
        raise _invalid(
            "post_norm_residual_pre_norm outputs must be one of: "
            f"{', '.join(sorted(_GATED_BOUNDARY_OUTPUTS))}"
        )
    scalars = {}
    for name in ("post_eps", "pre_eps", "gate_eps", "gate_temperature"):
        value = parameters[name]
        if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
            raise _invalid(f"post_norm_residual_pre_norm {name} must be positive")
        scalars[name] = value
    weight_offset = parameters["weight_offset"]
    if not isinstance(weight_offset, float):
        raise _invalid("post_norm_residual_pre_norm weight_offset must be a float")

    branch, post_weight, pre_weight, gate = _random_tensors(
        request.seed,
        (tokens, hidden_size),
        (hidden_size,),
        (hidden_size,),
        (hidden_size,),
    )
    stream = branch.float().mul_(4.0)
    post_weight = post_weight.mul_(0.1) if post_weighted else None
    pre_weight = pre_weight.mul_(0.1) if pre_weighted else None

    from tokenspeed_kernel.ops import residual as residual_ops

    # Mirror the operation's selection traits for these operands.
    from tokenspeed_kernel.ops.residual._cute_dsl.gated_boundary import (
        cluster_boundary_supported,
    )

    load_builtin_kernels()
    alpha, beta = residual_ops.residual_gate_coefficients(
        gate, scalars["gate_temperature"], scalars["gate_eps"]
    )
    try:
        kernel = select_kernel(
            "residual",
            "post_norm_residual_pre_norm",
            format_signature(
                stream=dense_tensor_format(stream.dtype),
                branch=dense_tensor_format(branch.dtype),
            ),
            platform=platform,
            traits={
                "cluster_supported": cluster_boundary_supported(
                    stream, branch, post_weight, alpha, beta, pre_weight, None, 1
                )
            },
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    entry = (
        residual_ops.post_norm_residual_pre_norm_bf16
        if outputs == "bf16"
        else residual_ops.post_norm_residual_pre_norm
    )

    def invoke() -> object:
        return entry(
            stream,
            branch,
            post_weight,
            scalars["post_eps"],
            weight_offset,
            alpha,
            beta,
            pre_weight,
            scalars["pre_eps"],
            weight_offset,
            False,
            None,
            1,
        )

    return PreparedBenchmark(
        registration=_registered_spec(kernel.name),
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hidden_size": hidden_size,
            "post_weighted": post_weighted,
            "pre_weighted": pre_weighted,
            "outputs": outputs,
            **scalars,
            "weight_offset": weight_offset,
            "dtype": "bfloat16",
            "stream_dtype": "float32",
        },
        validation=None,
    )
