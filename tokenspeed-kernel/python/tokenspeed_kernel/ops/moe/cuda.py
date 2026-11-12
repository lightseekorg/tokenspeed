"""CUDA MoE kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cuda import kimi3_latent_front
from tokenspeed_kernel.thirdparty.cuda.kimi3_router import sigmoid_bias_topk
from tokenspeed_kernel.thirdparty.cuda.moe import moe_finalize_fuse_shared
from tokenspeed_kernel.thirdparty.cuda.routing import (
    hash_softplus_sqrt_topk_flash,
    routing_available,
    softplus_sqrt_topk_flash,
)


def cuda_sqrt_softplus_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused CUDA sqrt-softplus expert selector."""
    logits_f32 = router_logits.float().contiguous()
    topk_weights = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=weights_dtype,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.int32,
        device=router_logits.device,
    )
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        hash_softplus_sqrt_topk_flash(
            logits_f32,
            input_ids.reshape(-1).to(device=router_logits.device).contiguous(),
            hash_indices_table.to(
                device=router_logits.device,
                dtype=torch.int32,
            ).contiguous(),
            topk_ids,
            topk_weights,
            routed_scaling_factor,
            renormalize,
            enable_pdl=pdl_enabled(),
        )
    elif correction_bias is not None:
        softplus_sqrt_topk_flash(
            logits_f32,
            correction_bias.to(
                device=router_logits.device,
                dtype=torch.float32,
            ).contiguous(),
            topk_ids,
            topk_weights,
            routed_scaling_factor,
            renormalize,
            enable_pdl=pdl_enabled(),
        )
    else:
        raise ValueError("fused DeepSeek V4 selection requires bias or hash routing")
    scores = (
        torch.sqrt(F.softplus(router_logits.float())) if need_scores else router_logits
    )
    return topk_weights, topk_ids, scores


if routing_available():
    cuda_sqrt_softplus_topk = register_kernel(
        "moe",
        "topk",
        name="cuda_sqrt_softplus_topk",
        solution="cuda",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=frozenset(
            format_signature(router_logits=dense_tensor_format(dtype))
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ),
        traits={
            "experts": frozenset({256, 384}),
            "top_k": frozenset({6}),
            "renormalize": frozenset({True}),
            "routing_kind": frozenset({"bias", "hash"}),
            "score_function": frozenset({"sqrt_softplus"}),
        },
        priority=Priority.SPECIALIZED,
    )(cuda_sqrt_softplus_topk)


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="cuda_kimi3_sigmoid_bias_topk",
    solution="cuda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {format_signature(router_logits=dense_tensor_format(torch.float32))}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "tokens_min": frozenset({2560}),
        "experts": frozenset({896}),
        "topk": frozenset({16}),
        "bias_dtype": frozenset({torch.float32}),
        "inputs_contiguous": frozenset({True}),
        "normalize_topk_weights": frozenset({True}),
        "routed_scaling_factor": frozenset({1.0}),
        "weights_dtype": frozenset({torch.float32}),
        "ids_dtype": frozenset({torch.int32}),
        "supports_output_buffers": frozenset({True}),
    },
)
def cuda_kimi3_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 normalized top-16 weights and INT32 IDs; see sigmoid_bias_topk."""
    if (
        topk != 16
        or routed_scaling_factor != 1.0
        or not normalize_topk_weights
        or (out[0].dtype if out is not None else weights_dtype) != torch.float32
        or (out is not None and out[1].dtype != torch.int32)
    ):
        raise ValueError(
            "Kimi-K3 CUDA routing requires normalized top-16 weights "
            "with scale 1, FP32 weights, and INT32 ids"
        )
    return sigmoid_bias_topk(router_logits, correction_bias, out=out)


__all__ = [
    "cuda_sqrt_softplus_topk",
    "cuda_kimi3_sigmoid_bias_topk",
    "moe_finalize_fuse_shared",
]

# The fused front is validated on 152-SM GB300 parts only; the vendored programs also
# compile for SM100, which stays on the separate projections until measured.
_FRONT_PLATFORM = current_platform()
if (
    _FRONT_PLATFORM.is_nvidia
    and _FRONT_PLATFORM.arch_version == ArchVersion(10, 3)
    and _FRONT_PLATFORM.sm_count == 152
    and kimi3_latent_front.available()
):

    @register_kernel(
        "moe",
        "latent_input",
        name="cuda_latent_input",
        solution="cuda",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 3),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    router_weight=dense_tensor_format(torch.bfloat16),
                    routed_weight=dense_tensor_format(torch.bfloat16),
                    shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            # The fused front pays off from 2048 rows; smaller batches keep the separate
            # projections (see ops/moe/README.md, "Kimi-K3 front projections"). There is
            # no upper bound: the plan scales with the row count.
            "tokens_min": frozenset({2048}),
            "gate_clamp": frozenset({4.0}),
            "hidden_size": frozenset({kimi3_latent_front.HIDDEN}),
            "inputs_contiguous": frozenset({True}),
            "latent_size": frozenset({kimi3_latent_front.LATENT}),
            "num_experts": frozenset({kimi3_latent_front.NUM_EXPERTS}),
            "shared_size": frozenset({kimi3_latent_front.SHARED_INTERMEDIATE}),
            "up_clamp": frozenset({25.0}),
        },
    )
    def cuda_latent_input(
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        routed_weight: torch.Tensor,
        shared_gate_up_weight: torch.Tensor,
        *,
        gate_clamp: float,
        up_clamp: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused Kimi-K3 router / latent / shared-input projections; see kimi3_latent_front."""
        if gate_clamp != 4.0 or up_clamp != 25.0:
            raise ValueError("the Kimi-K3 CUDA front requires SiTU clamps 4 and 25")
        rows = hidden_states.shape[0]
        logits = torch.empty(
            (rows, router_weight.shape[0]),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        latent = hidden_states.new_empty((rows, routed_weight.shape[0]))
        shared_act = hidden_states.new_empty(
            (rows, shared_gate_up_weight.shape[0] // 2)
        )
        return kimi3_latent_front.front(
            hidden_states,
            router_weight,
            routed_weight,
            shared_gate_up_weight,
            logits,
            latent,
            shared_act,
        )

    __all__.append("cuda_latent_input")
