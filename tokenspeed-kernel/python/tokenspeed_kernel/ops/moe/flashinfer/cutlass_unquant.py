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

import functools
import warnings

import torch
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe
    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

    def flashinfer_cutlass_unquant_moe_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half
        return None

    @functools.cache
    def _warn_pdl_forced_off() -> None:
        warnings.warn(
            "flashinfer_cutlass_unquant_moe_apply forces enable_pdl off: PDL "
            "races inside this fused-MoE chain on SM90 (transient NaN GEMM "
            "rows at decode-sized batches).",
            stacklevel=3,
        )

    # One persistent scratch buffer per device, handed to every call as
    # ``workspace_buffer``. Left to itself the runner allocates its scratch
    # uninitialized on every call, and on SM90 the chain then reads bytes it
    # never wrote: for some (EP rank, routing) combinations the rank's whole
    # routed output came back NaN for finite inputs, reproducibly for that
    # call, while the identical call with any caller-provided buffer -- or
    # with fresh allocator state -- matched the fp32 reference. The buffer
    # is zero-filled when (re)allocated and only ever holds this chain's own
    # writes afterwards. Consecutive MoE layers are stream-ordered through
    # their activations, so one buffer per device is never in use twice.
    _workspaces: dict[torch.device, torch.Tensor] = {}

    def cutlass_unquant_moe_workspace(
        *,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts_total: int,
        top_k: int,
        x_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        device: torch.device,
    ) -> torch.Tensor:
        """The device's persistent workspace, grown (and re-zeroed) on demand.

        Args:
            num_tokens: Rows of this call; the buffer is sized for the largest
                row count seen so far on the device.
            hidden_size: Model hidden size (``w2.shape[1]``).
            intermediate_size: Per-partition intermediate size (``w2.shape[2]``).
            num_experts_total: Experts across the EP group.
            top_k: Routed experts per token.
            x_dtype: Activation dtype.
            weight_dtype: Expert weight dtype.
            tp_size: MoE tensor-parallel width.
            tp_rank: This rank's MoE tensor-parallel coordinate.
            ep_size: Expert-parallel width.
            ep_rank: This rank's expert-parallel coordinate.
            device: CUDA device the call runs on.

        Returns:
            A 1-D ``uint8`` CUDA tensor of at least the bytes
            :func:`flashinfer.fused_moe.cutlass_fused_moe_workspace_size`
            asks for ``num_tokens``.
        """
        needed = cutlass_fused_moe_workspace_size(
            max(int(num_tokens), 1),
            hidden_size,
            intermediate_size,
            num_experts_total,
            top_k,
            x_dtype=x_dtype,
            weight_dtype=weight_dtype,
            output_dtype=x_dtype,
            activation_type=ActivationType.Swiglu,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            device=device,
        )
        workspace = _workspaces.get(device)
        if workspace is None or workspace.numel() < needed:
            workspace = torch.zeros(needed, dtype=torch.uint8, device=device)
            _workspaces[device] = workspace
        return workspace

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_unquant_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_unquant_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(8, 9),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_unquant_moe_apply(
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
    ):
        if enable_pdl:
            # Not silently: a caller enabling PDL globally would otherwise
            # misread its perf measurements with nothing in the logs.
            _warn_pdl_forced_off()
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        ep_size = getattr(w, "ep_size", 1)
        ep_rank = getattr(w, "ep_rank", 0)
        tp_size = getattr(w, "tp_size", 1)
        tp_rank = getattr(w, "tp_rank", 0)
        workspace = cutlass_unquant_moe_workspace(
            num_tokens=x.shape[0],
            hidden_size=w.w2_weight.shape[1],
            intermediate_size=w.w2_weight.shape[2],
            num_experts_total=w.w13_weight.shape[0] * ep_size,
            top_k=topk_ids.shape[1],
            x_dtype=x.dtype,
            weight_dtype=w.w13_weight.dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            device=x.device,
        )
        return cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=x.dtype,
            quant_scales=None,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            activation_type=ActivationType.Swiglu,
            # PDL races inside this fused-MoE chain on SM90 at decode-sized
            # batches: a routed GEMM row transiently reads NaN (rerunning the
            # identical call is clean). Keep the chain fully serialized until
            # the flashinfer kernels are fixed.
            enable_pdl=False,
            workspace_buffer=workspace,
        )[0]
