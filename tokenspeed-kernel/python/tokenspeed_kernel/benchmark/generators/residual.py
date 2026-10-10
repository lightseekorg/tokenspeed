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
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["prepare_attn_res_fwd", "prepare_mhc_mixes", "prepare_mhc_post"]


_IMPLEMENTED_MODEL_PROFILES = frozenset({"kimi_k3_tp8"})
_IMPLEMENTED_MHC_MODEL_PROFILES = frozenset({"dsv41_flash_tp4"})


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"AttnRes {name} must be a positive integer")
    return value


def _flag(parameters: dict[str, Any], name: str) -> bool:
    value = parameters[name]
    if not isinstance(value, bool):
        raise _invalid(f"AttnRes {name} must be a boolean")
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
    if parameters.get("validation") is not None:
        raise _invalid("mHC benchmark correctness validation is not implemented")
    model_profile = parameters["model_profile"]
    if model_profile not in _IMPLEMENTED_MHC_MODEL_PROFILES:
        raise _invalid(
            f"Implemented mHC model_profile values: "
            f"{', '.join(sorted(_IMPLEMENTED_MHC_MODEL_PROFILES))}"
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


def prepare_mhc_post(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one mHC post-mapping: scatter a sublayer output into the streams."""

    model_profile, tokens, hc_mult, hidden_size = _mhc_common(request, "mhc_post")
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
                "hc_mult": hc_mult,
                "hidden_size": hidden_size,
            },
            override=None,
            solution=None,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error

    hidden_states, residual = _random_tensors(
        request.seed,
        (tokens, hidden_size),
        (tokens, hc_mult, hidden_size),
    )
    generator = torch.Generator(device="cuda").manual_seed(request.seed + 1)
    post = 2.0 * torch.rand(
        (tokens, hc_mult, 1), device="cuda", dtype=torch.float32, generator=generator
    )
    # Sinkhorn output is doubly stochastic; any row-normalized matrix has the
    # same cost.
    comb = torch.rand(
        (tokens, hc_mult, hc_mult),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    comb = comb / comb.sum(-1, keepdim=True)

    def invoke() -> object:
        return residual_ops.mhc_post(
            hidden_states, residual, post, comb, override=None, solution=None
        )

    return PreparedBenchmark(
        registration=_registered_spec(kernel.name),
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hc_mult": hc_mult,
            "hidden_size": hidden_size,
            "dtype": "bfloat16",
        },
        validation=None,
    )
