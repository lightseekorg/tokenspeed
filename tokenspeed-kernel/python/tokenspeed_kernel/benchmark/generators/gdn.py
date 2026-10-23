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

"""Benchmark generators for Gated Delta Net (GDN) linear attention."""

from __future__ import annotations

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
    "prepare_gdn_chunk_prefill",
    "prepare_gdn_decode_mtp",
    "prepare_gdn_decode_step",
]


@dataclass(frozen=True)
class _GdnModelProfile:
    num_q_heads: int
    num_v_heads: int
    head_dim: int
    value_head_dim: int


# Per-rank GDN geometry: Qwen3.8-Flash-Next has 16 key / 48 value heads of
# 128 dims, split over TP2.
_MODEL_PROFILES = {
    "qwen38_flash_tp2": _GdnModelProfile(
        num_q_heads=8, num_v_heads=24, head_dim=128, value_head_dim=128
    ),
}
_IMPLEMENTED_STATE_WRITES = frozenset({"none", "per_token"})


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"GDN {name} must be a positive integer")
    return value


def _profile(request: BenchmarkRequest) -> tuple[str, _GdnModelProfile]:
    if request.parameters.get("validation") is not None:
        raise _invalid("GDN benchmark correctness validation is not implemented")
    if request.parameters["dtype"] != "bfloat16":
        raise _invalid("GDN dtype must be bfloat16")
    name = request.parameters["model_profile"]
    if name not in _MODEL_PROFILES:
        raise _invalid(
            f"Implemented GDN model_profile values: {', '.join(sorted(_MODEL_PROFILES))}"
        )
    return name, _MODEL_PROFILES[name]


def _select_registration(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    traits: dict[str, object],
) -> KernelSpec:
    load_builtin_kernels()
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
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec


def _decode_inputs(
    seed: int, batch: int, tokens: int, profile: _GdnModelProfile
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    return {
        "q": randn(batch, tokens, profile.num_q_heads, profile.head_dim),
        "k": randn(batch, tokens, profile.num_q_heads, profile.head_dim),
        "v": randn(batch, tokens, profile.num_v_heads, profile.value_head_dim),
        "A_log": randn(profile.num_v_heads, dtype=torch.float32),
        "a": randn(batch, tokens, profile.num_v_heads),
        "dt_bias": randn(profile.num_v_heads, dtype=torch.float32),
        "b": randn(batch, tokens, profile.num_v_heads),
    }


def _state_pool(rows: int, profile: _GdnModelProfile) -> torch.Tensor:
    # FP32 K-last recurrent states, as the runtime's SSM pool stores them.
    return torch.zeros(
        (rows, profile.num_v_heads, profile.value_head_dim, profile.head_dim),
        device="cuda",
        dtype=torch.float32,
    )


def prepare_gdn_decode_step(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one single-token GDN decode step over ``batch`` requests.

    Each request reads and writes back its own row of an FP32 state pool.
    """

    model_profile, profile = _profile(request)
    batch = _positive(request.parameters, "batch")
    spec = _select_registration(request, platform, {"head_dim": profile.head_dim})
    inputs = _decode_inputs(request.seed, batch, 1, profile)
    state = _state_pool(batch, profile)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)

    from tokenspeed_kernel.ops.attention.gdn import gdn_decode_step

    def invoke() -> object:
        return gdn_decode_step(
            **inputs,
            initial_state=state,
            initial_state_indices=indices,
            use_qk_l2norm=True,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "batch": batch,
            "dtype": "bfloat16",
            "state_dtype": "float32",
        },
        validation=None,
    )


def prepare_gdn_decode_mtp(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one chain MTP verify step: ``batch`` requests of ``tokens`` each.

    ``state_writes="per_token"`` writes every position's post-update state to
    its own pool row, as the staged verify does; ``"none"`` leaves the pool
    untouched, as the ReplaySSM verify does.
    """

    model_profile, profile = _profile(request)
    batch = _positive(request.parameters, "batch")
    tokens = _positive(request.parameters, "tokens")
    state_writes = request.parameters["state_writes"]
    if state_writes not in _IMPLEMENTED_STATE_WRITES:
        raise _invalid(
            "GDN state_writes must be one of: "
            f"{', '.join(sorted(_IMPLEMENTED_STATE_WRITES))}"
        )
    spec = _select_registration(request, platform, {"head_dim": profile.head_dim})
    inputs = _decode_inputs(request.seed, batch, tokens, profile)
    per_token = state_writes == "per_token"
    # Committed rows first, then one scratch row per verified position.
    state = _state_pool(batch * (1 + tokens) if per_token else batch, profile)
    initial_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    output_indices = (
        (batch + torch.arange(batch * tokens, device="cuda", dtype=torch.int32)).view(
            batch, tokens
        )
        if per_token
        else None
    )

    from tokenspeed_kernel.ops.attention.gdn import gdn_decode_mtp

    def invoke() -> object:
        return gdn_decode_mtp(
            **inputs,
            initial_state=state,
            initial_state_indices=initial_indices,
            disable_state_update=not per_token,
            use_qk_l2norm=True,
            output_state_indices=output_indices,
            tree_ancestors=None,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "batch": batch,
            "tokens": tokens,
            "state_writes": state_writes,
            "dtype": "bfloat16",
            "state_dtype": "float32",
        },
        validation=None,
    )


def prepare_gdn_chunk_prefill(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one chunked GDN prefill of ``sequences`` equal-length prompts.

    Inputs mirror the runtime's prefill scan: FP32 log gates, BF16 betas,
    in-kernel Q/K L2 normalization and an FP32 initial and final state.
    This case does not request intermediate state checkpoints.
    """

    model_profile, profile = _profile(request)
    sequences = _positive(request.parameters, "sequences")
    tokens_per_sequence = _positive(request.parameters, "tokens_per_sequence")
    total = sequences * tokens_per_sequence
    traits = {
        "head_dim": profile.head_dim,
        "value_head_dim": profile.value_head_dim,
        "num_v_gte_num_q": profile.num_v_heads >= profile.num_q_heads,
        "output_h": False,
        "qk_l2norm": True,
        "state_checkpoints": False,
    }
    spec = _select_registration(request, platform, traits)
    generator = torch.Generator(device="cuda").manual_seed(request.seed)

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    q = randn(1, total, profile.num_q_heads, profile.head_dim)
    k = randn(1, total, profile.num_q_heads, profile.head_dim)
    v = randn(1, total, profile.num_v_heads, profile.value_head_dim)
    # Log-space forget gates are non-positive.
    g = -torch.rand(
        (1, total, profile.num_v_heads),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    beta = randn(1, total, profile.num_v_heads).sigmoid()
    state = _state_pool(sequences, profile)
    cu_seqlens = torch.arange(
        0, total + 1, tokens_per_sequence, device="cuda", dtype=torch.int32
    )

    from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill

    def invoke() -> object:
        return gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=profile.head_dim**-0.5,
            initial_state=state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "sequences": sequences,
            "tokens_per_sequence": tokens_per_sequence,
            "dtype": "bfloat16",
            "state_dtype": "float32",
        },
        validation=None,
    )
