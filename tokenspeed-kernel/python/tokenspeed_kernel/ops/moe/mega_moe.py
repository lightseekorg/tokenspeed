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

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def trtllm_mega_moe_process_weights(plan: dict, w: torch.nn.Module) -> None:
    from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.blocked_scale import to_blocked

    del plan
    if w.tp_size != 1:
        raise ValueError("NVFP4 MegaMoE requires MoE TP=1")
    gate_scale, up_scale = w.w13_weight_scale_2.unbind(dim=1)
    if not torch.equal(gate_scale, up_scale):
        raise ValueError("NVFP4 SiTU MegaMoE requires equal gate/up weight_scale_2")
    experts = w.num_local_experts
    intermediate = w.intermediate_size
    for name in ("w13_weight", "w13_weight_scale"):
        tensor = getattr(w, name).data.view(torch.uint8)
        gate, up = tensor.split(intermediate, dim=1)
        packed = torch.stack(
            (
                gate.reshape(experts, intermediate // 16, 16, -1),
                up.reshape(experts, intermediate // 16, 16, -1),
            ),
            dim=2,
        )
        getattr(w, name).data = packed.reshape_as(tensor).contiguous()
    for name in ("w13_weight_scale", "w2_weight_scale"):
        tensor = getattr(w, name).data.view(torch.float8_e4m3fn)
        getattr(w, name).data = torch.stack(
            [to_blocked(sf).view(torch.uint8) for sf in tensor]
        )
    w.w13_input_scale_quant = torch.nn.Parameter(
        w.w13_input_scale.reciprocal(), requires_grad=False
    )
    w.mega_fc1_alpha = torch.nn.Parameter(
        (w.w13_input_scale * gate_scale).float(), requires_grad=False
    )
    w.mega_fc2_alpha = torch.nn.Parameter(
        (w.w2_input_scale * w.w2_weight_scale_2).float(), requires_grad=False
    )
    w.mega_fc1_norm = torch.nn.Parameter(
        w.w2_input_scale.reciprocal().expand(experts).contiguous(), requires_grad=False
    )
    for name in (
        "w13_input_scale",
        "w2_input_scale",
        "w13_weight_scale_2",
        "w2_weight_scale_2",
    ):
        delattr(w, name)


@register_kernel(
    "moe",
    "apply",
    name="trtllm_nvfp4_mega_moe_apply",
    solution="mega_moe",
    weight_preprocessor=trtllm_mega_moe_process_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.uint8}),
    traits={
        "weight_dtype": frozenset({"nvfp4"}),
        "activation": frozenset({"situ"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({16}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE - 1,
)
def trtllm_nvfp4_mega_moe_apply(
    plan: dict,
    x: tuple[torch.Tensor, torch.Tensor],
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_tokens_global: int,
    max_num_tokens_per_gpu: int,
    do_finalize: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Dispatch NVFP4 tokens, run SiTU experts, and combine on the source ranks.

    Uses the upstream heuristic by default. Set MEGAMOE_TACTIC_AUTOTUNE=1
    identically on every EP rank before startup to participate in FlashInfer's
    autotuning context and reuse tuned tactics across compatible layers.
    Graph timing includes input staging. One buffer set sized by the configured
    autotuning maximum backs all token views and tactics, with fixed scratch
    offsets so captured calls can alternate sizes without host-side resets.
    Outside tuning, cache misses use the heuristic tactic. Tune and warm up
    before CUDA graph capture; all EP ranks must enter with identical caches.

    Args:
        plan: Selected NVFP4 MoE plan containing the EP process group.
        x: Packed E2M1 activations and linear E4M3 block scales.
        w: Local expert module with MegaMoE-prepared weights.
        router_logits: Unused; routes are supplied explicitly.
        topk_weights: Final routing weights, including the model routing scale.
        topk_ids: Global expert indices for local tokens.
        num_tokens_global: Global token capacity for this collective step.
        max_num_tokens_per_gpu: Rank-identical local token bound for this call.
        do_finalize: Must be true; this operation owns the complete routed result.
        enable_pdl: Reserved; the vendored kernel controls its own launch ordering.

    Returns:
        Local BF16 routed output, valid until the next use of the shared workspace.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.runner import get_runner

    del router_logits, num_tokens_global, enable_pdl
    if not do_finalize or not isinstance(x, tuple):
        raise ValueError(
            "MegaMoE requires prequantized inputs and complete finalization"
        )
    runner = get_runner(
        plan["process_group"],
        w.num_experts,
        w.hidden_size,
        w.intermediate_size,
        w.top_k,
        w.activation_situ_beta,
        w.activation_situ_linear_beta,
    )
    weights = (
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        w.mega_fc1_alpha,
        w.mega_fc2_alpha,
        w.mega_fc1_norm,
    )
    return runner.run(x, topk_ids, topk_weights, weights, max_num_tokens_per_gpu)
