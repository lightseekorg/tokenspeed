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

"""Optional FlashInfer Kimi-K3 kernels with their source payloads."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


@functools.cache
def latent_front_available() -> bool:
    try:
        from flashinfer.experimental.kimi_k3_latent_moe import cake_jit
        from flashinfer.experimental.kimi_k3_latent_moe.cake_backend import (
            QUALIFIED_SM_COUNTS,
        )
    except ImportError:
        return False
    root = Path(cake_jit.__file__).parent / "csrc"
    return (
        152 in QUALIFIED_SM_COUNTS
        and bool(cake_jit.MODULES)
        and all(
            (root / source).is_file()
            for record in cake_jit.MODULES.values()
            for source in record["sources"]
        )
    )


def latent_front(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.kimi_k3_latent_moe import kimi_k3_latent_moe_front

    rows = hidden_states.shape[0]
    logits = torch.empty(
        (rows, router_weight.shape[0]), device=hidden_states.device, dtype=torch.float32
    )
    routed = hidden_states.new_empty((rows, routed_weight.shape[0]))
    shared = hidden_states.new_empty((rows, shared_gate_up_weight.shape[0] // 2))
    return kimi_k3_latent_moe_front(
        hidden_states,
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        logits,
        routed,
        shared,
        backend="cake",
    )


@functools.cache
def fused_router_available() -> bool:
    """Whether the installed wheel includes the fused-router source payload."""
    try:
        from flashinfer.experimental.kimi_k3_fused_router import cake_jit
    except ImportError:
        return False
    root = Path(cake_jit.__file__).parent / "csrc"
    return bool(cake_jit.MODULES) and all(
        (root / source).is_file()
        for record in cake_jit.MODULES.values()
        for source in record["sources"]
    )


def fused_router(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized FP32 top-16 weights and INT32 expert IDs.

    ``logits`` and ``correction_bias`` use the Kimi-K3 FP32/896-expert
    contract. ``out`` supplies the two output buffers or requests allocation
    when None. This function preserves caller-owned buffers.
    """
    from flashinfer.fused_moe.kimi_k3_fused_router import (
        allocate_kimi_k3_route_plan,
        prepare_kimi_k3_fused_router,
    )

    plan = allocate_kimi_k3_route_plan(logits.shape[0], 8, logits.device)
    if out is not None:
        plan = plan._replace(topk_weights=out[0], topk_ids=out[1])
    prepare_kimi_k3_fused_router(
        logits, correction_bias, block_m=8, plan=plan, backend="cake"
    )()
    return plan.topk_weights, plan.topk_ids
