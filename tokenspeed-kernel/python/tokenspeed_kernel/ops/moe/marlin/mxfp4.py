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

"""Marlin W4A16 MXFP4 MoE apply for SM90 (Hopper).

Weight-only 4-bit experts: MXFP4 (E2M1 packed, E8M0 group-32 scales) weights
are dequantized inside the Marlin grouped GEMM; activations stay bf16, so no
FP4 tensor cores are required. This is the Hopper path for Kimi-K3, whose
routed experts use the SiTU gated activation (applied as a fused Triton
epilogue between the two GEMMs). Routing is precomputed upstream (the K3 gate
the fused kernels cannot reproduce).

The kernel and its weight repack use the vendored Marlin MoE; the grouped GEMM
lives in ``thirdparty/cuda/csrc/marlin_moe`` and is pre-compiled.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.triton import situ_and_mul
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    marlin_make_workspace,
    marlin_permute_scales,
    moe_align_block_size,
    moe_wna16_marlin_gemm,
    mxfp4_marlin_process_scales,
)

MXFP4_BLOCK = 32


def _block_size_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    """Pick the token block granularity Marlin schedules over.

    Grow the block until each expert holds under ~0.9 blocks of routed tokens,
    keeping small batches on tight blocks and large batches on wide ones.
    """
    block = 8
    for candidate in (8, 16, 32, 48, 64):
        block = candidate
        if num_tokens * top_k / num_experts / candidate < 0.9:
            break
    return block


def marlin_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack loader-format MXFP4 experts into the Marlin layout, once.

    Input (from the mxfp4 weight loader): ``w13_weight``/``w2_weight`` packed
    E2M1 as uint8 ``[E, N, K//2]`` and ``w13_weight_scale``/``w2_weight_scale``
    raw E8M0 as uint8 ``[E, N, K//32]``. Output: Marlin-repacked int32 weights
    and permuted float8_e8m0fnu scales, written back onto the module. K3's
    shapes are already aligned (hidden 7168%256, ispp 3072%128), so no padding.
    """
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    if any(not hasattr(w, name) for name in names):
        raise ValueError("MXFP4 MoE weights are incomplete for Marlin repack")
    if getattr(w, "_marlin_repacked", False):
        return

    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "situ", "swiglu"}:
        raise ValueError(f"Marlin MXFP4 MoE does not support activation {activation!r}")

    w13 = w.w13_weight.data
    w2 = w.w2_weight.data
    w13_scale = w.w13_weight_scale.data
    w2_scale = w.w2_weight_scale.data
    num_experts = w13.shape[0]
    # w13 is [E, 2*ispp, hidden//2] packed; w2 is [E, hidden, ispp//2] packed.
    two_ispp = w13.shape[1]
    hidden = w13.shape[2] * 2
    ispp = w2.shape[2] * 2
    if two_ispp != 2 * ispp:
        raise ValueError(
            f"w13/w2 intermediate mismatch: w13 {two_ispp} vs 2*ispp {2 * ispp}"
        )
    if hidden % 256 != 0 or ispp % MXFP4_BLOCK != 0:
        raise ValueError(
            f"Marlin MXFP4 needs hidden%256==0 and ispp%32==0, got "
            f"hidden={hidden}, ispp={ispp}"
        )

    device = w13.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack(weight: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        # gptq_marlin_repack wants int32 [size_k/pack, size_n]; the packed
        # uint8 [E, size_n, size_k/2] view transposes to that per expert.
        out = []
        for e in range(num_experts):
            qw = weight[e].view(torch.int32).T.contiguous()
            out.append(
                gptq_marlin_repack(qw, perm, size_k=size_k, size_n=size_n, num_bits=4)
            )
        return torch.stack(out)

    def _permute(scale: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        out = []
        for e in range(num_experts):
            s = scale[e].T.contiguous()
            permuted = marlin_permute_scales(
                s, size_k=size_k, size_n=size_n, group_size=MXFP4_BLOCK
            )
            out.append(mxfp4_marlin_process_scales(permuted))
        return torch.stack(out)

    w13_marlin = _repack(w13, size_n=2 * ispp, size_k=hidden)
    w2_marlin = _repack(w2, size_n=hidden, size_k=ispp)
    w13_scale_marlin = _permute(w13_scale, size_n=2 * ispp, size_k=hidden)
    w2_scale_marlin = _permute(w2_scale, size_n=hidden, size_k=ispp)

    w.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)
    w._marlin_hidden_size = hidden
    w._marlin_ispp = ispp
    w._marlin_repacked = True


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp4_precomputed_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp4_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({MXFP4_BLOCK}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def marlin_mxfp4_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply a Marlin W4A16 MXFP4 MoE with precomputed routing.

    Args:
        plan: MoE plan; ``activation`` selects the GEMM1 epilogue.
        x: bf16 hidden states ``[tokens, hidden]``.
        w: Module holding Marlin-repacked ``w13_weight``/``w2_weight`` (int32)
            and permuted ``w13_weight_scale``/``w2_weight_scale``
            (float8_e8m0fnu), plus ``num_local_experts``/``ep_rank`` for EP.
        router_logits: Unused; routing is precomputed.
        topk_weights: Route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``. In EP they are remapped
            to local ids; non-local ids become -1 and contribute zero.
        num_tokens_global: Unused; distributed EP dispatch is not owned here.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (no deferred finalize).
        enable_pdl: Unused launch hint.

    Returns:
        Finalized hidden states ``[tokens, hidden]`` in the dtype of ``x``.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Marlin MXFP4 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP4 MoE requires precomputed top-k")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP4 MoE requires bf16 activations, got {x.dtype}")

    activation = plan.get("activation") or getattr(w, "activation", "silu")
    hidden = int(getattr(w, "_marlin_hidden_size", x.shape[1]))
    ispp = int(getattr(w, "_marlin_ispp", w.w2_weight.shape[1] * 16))
    num_local_experts = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
    ep_size = int(getattr(w, "ep_size", 1))
    ep_rank = int(getattr(w, "ep_rank", 0))
    global_num_experts = num_local_experts * ep_size

    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    is_ep = ep_size > 1
    if is_ep:
        # Global -> local id remap: [expert_start, +num_local) -> [0, num_local),
        # everything else -> -1 (the align kernel parks these in the extra lane
        # and the EP GEMM skips those blocks). Inputs may already carry -1 in
        # masked slots (DeepEP recv layout); clamp before the gather so the
        # negative index cannot wrap to the mapping's last entry, then restore.
        expert_start = ep_rank * num_local_experts
        mapping = torch.full(
            (global_num_experts,), -1, dtype=torch.int32, device=topk_ids.device
        )
        mapping[expert_start : expert_start + num_local_experts] = torch.arange(
            num_local_experts, dtype=torch.int32, device=topk_ids.device
        )
        local_topk_ids = torch.where(
            topk_ids < 0, topk_ids, mapping[topk_ids.long().clamp_(min=0)]
        )
        align_num_experts = num_local_experts
    else:
        local_topk_ids = topk_ids
        align_num_experts = num_local_experts

    num_tokens, top_k = topk_ids.shape
    block_m = _block_size_m(num_tokens, top_k, align_num_experts)
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        local_topk_ids, block_m, align_num_experts
    )

    workspace = marlin_make_workspace(x.device)

    gemm1_n = 2 * ispp
    intermediate1 = moe_wna16_marlin_gemm(
        x,
        None,
        w.w13_weight,
        w.w13_weight_scale,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_m,
        top_k=top_k,
        mul_topk_weights=False,
        is_ep=is_ep,
        size_m=num_tokens,
        size_n=gemm1_n,
        size_k=hidden,
    ).view(-1, gemm1_n)

    beta = float(getattr(w, "activation_situ_beta", 1.0))
    linear_beta = getattr(w, "activation_situ_linear_beta", None)
    if activation == "situ":
        intermediate2 = situ_and_mul(
            intermediate1,
            beta=beta,
            linear_beta=None if linear_beta is None else float(linear_beta),
        )
    else:
        from tokenspeed_kernel.ops.activation.triton import silu_and_mul

        intermediate2 = silu_and_mul(intermediate1)

    # GEMM2: fold the route weights in (mul_topk_weights) so finalize is a
    # plain sum over top_k. EP-masked routes wrote nothing, so zero-init c.
    intermediate3 = moe_wna16_marlin_gemm(
        intermediate2,
        torch.zeros(num_tokens * top_k, hidden, dtype=x.dtype, device=x.device),
        w.w2_weight,
        w.w2_weight_scale,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=is_ep,
        size_m=num_tokens * top_k,
        size_n=hidden,
        size_k=ispp,
    ).view(num_tokens, top_k, hidden)

    return intermediate3.sum(dim=1)
