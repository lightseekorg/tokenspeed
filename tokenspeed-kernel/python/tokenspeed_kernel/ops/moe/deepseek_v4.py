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

"""DeepSeek V4 expert selection."""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _assert_indices_in_range(
    indices: torch.Tensor,
    upper_bound: int,
    name: str,
) -> None:
    valid = ((indices >= 0) & (indices < upper_bound)).all()
    message = f"{name} entries must be in [0, {upper_bound})"
    if indices.device.type == "cpu":
        if not bool(valid.item()):
            raise ValueError(message)
    else:
        torch._assert_async(valid, message)


def _routing_kind(
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> str:
    if hash_indices_table is not None:
        return "hash"
    if correction_bias is not None:
        return "bias"
    return "plain"


def deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    need_scores: bool = True,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select DeepSeek V4 experts from sqrt-softplus router scores.

    Correction bias affects selection only; returned weights are gathered from
    the unbiased scores. Hash routing uses the checkpoint table for expert ids.

    Args:
        router_logits: Router logits shaped [tokens, experts].
        top_k: Number of experts selected for each token.
        renormalize: Normalize selected weights to sum to one when true.
        correction_bias: Optional selection-only bias shaped [experts].
        hash_indices_table: Optional token-id to expert-id table.
        input_ids: Token ids used with hash_indices_table.
        need_scores: Whether callers consume the full score tensor. Specialized
            kernels avoid materializing it when false.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        FP32 weights, INT32 expert ids, and a tensor shaped [tokens, experts].
        The first two tensors have shape [tokens, top_k]. When need_scores is
        false, a specialized kernel may return router_logits as the ignored
        third value instead of materializing scores.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must be a floating-point tensor")
    tokens, experts = router_logits.shape
    if not 0 < top_k <= experts:
        raise ValueError(f"top_k must be in [1, {experts}], got {top_k}")
    if correction_bias is not None and correction_bias.shape != (experts,):
        raise ValueError(f"correction_bias must have shape [{experts}]")
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        if (
            hash_indices_table.ndim != 2
            or hash_indices_table.shape[0] == 0
            or hash_indices_table.shape[1] != top_k
        ):
            raise ValueError("hash_indices_table must have shape [vocabulary, top_k]")
        if hash_indices_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("hash_indices_table must have dtype int32 or int64")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must have dtype int32 or int64")
        if input_ids.numel() != tokens:
            raise ValueError(f"input_ids must contain {tokens} token ids")
        if hash_indices_table.device != router_logits.device:
            raise ValueError(
                "hash_indices_table must be on the same device as router_logits"
            )
        if input_ids.device != router_logits.device:
            raise ValueError("input_ids must be on the same device as router_logits")
        _assert_indices_in_range(input_ids, hash_indices_table.shape[0], "input_ids")
        safe_input_ids = input_ids.clamp(0, hash_indices_table.shape[0] - 1)
        selected_experts = hash_indices_table[safe_input_ids.reshape(-1).long()]
        _assert_indices_in_range(selected_experts, experts, "hash_indices_table")
        input_ids = safe_input_ids

    routing_kind = _routing_kind(correction_bias, hash_indices_table)
    traits = {
        "tokens": int(tokens),
        "experts": experts,
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
    }
    signature = format_signature(router_logits=dense_tensor_format(router_logits.dtype))
    kernel = select_kernel(
        "moe",
        "deepseek_v4_select_experts",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    shape_params = {
        "tokens": int(tokens),
        "experts": int(experts),
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
        "need_scores": bool(need_scores),
    }
    ShapeCapture.get().record(
        "moe",
        "deepseek_v4_select_experts",
        kernel.name,
        router_logits.dtype,
        shape_params,
    )
    try:
        with kernel_scope(
            "moe",
            "deepseek_v4_select_experts",
            router_logits.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            return kernel(
                router_logits,
                top_k,
                renormalize,
                correction_bias,
                hash_indices_table,
                input_ids,
                need_scores,
            )
    except (AttributeError, RuntimeError):
        spec = KernelRegistry.get().get_by_name(kernel.name)
        if override is not None or solution is not None or spec is None:
            raise
        if spec.solution == "torch":
            raise
        fallback = select_kernel(
            "moe",
            "deepseek_v4_select_experts",
            signature,
            traits=traits,
            solution="torch",
        )
        return fallback(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            need_scores,
        )


__all__ = ["deepseek_v4_select_experts"]
