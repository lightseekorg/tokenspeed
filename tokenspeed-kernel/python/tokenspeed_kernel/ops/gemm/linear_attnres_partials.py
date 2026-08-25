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

"""Linear projection with independent AttnRes block partials."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm.kimi3 import KIMI3_HIDDEN_SIZE
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _validate_inputs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    out: torch.Tensor | None,
) -> tuple[int, int, tuple[int, int]]:
    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError("hidden_states must have shape [tokens, input_size]")
    tokens, input_size = hidden_states.shape
    if weight.ndim != 2 or weight.shape[1] != input_size:
        raise ValueError("weight must have shape [output_size, input_size]")
    if (
        blocks.ndim != 3
        or blocks.shape[0] < 1
        or tuple(blocks.shape[1:]) != (tokens, input_size)
    ):
        raise ValueError("blocks must have shape [blocks, tokens, input_size]")
    tensors = (weight, blocks, score_weight_a, score_weight_b)
    if any(
        tensor.dtype != hidden_states.dtype or tensor.device != hidden_states.device
        for tensor in tensors
    ):
        raise ValueError("projection and AttnRes inputs must match dtype and device")
    if score_weight_a.shape != (input_size,) or score_weight_b.shape != (input_size,):
        raise ValueError("score weights must match input_size")

    expected_scratch_shapes = ((tokens,), (tokens,), (tokens, input_size))
    for scratch in (scratch_a, scratch_b):
        if len(scratch) != 3:
            raise ValueError("scratch must be an (max, sum, accumulator) tuple")
        for tensor, shape in zip(scratch, expected_scratch_shapes, strict=True):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or tensor.device != hidden_states.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"scratch tensors must be contiguous FP32 {expected_scratch_shapes}"
                )
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    output_shape = (tokens, weight.shape[0])
    if out is not None and (
        tuple(out.shape) != output_shape
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous with shape {output_shape}")
    return tokens, input_size, output_shape


def _select_registered_kernel(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    out: torch.Tensor | None,
    *,
    override: str | None,
    solution: str | None,
):
    tokens, input_size = hidden_states.shape
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        weight=dense_tensor_format(weight.dtype),
        blocks=dense_tensor_format(blocks.dtype),
        score_weight_a=dense_tensor_format(score_weight_a.dtype),
        score_weight_b=dense_tensor_format(score_weight_b.dtype),
        out=dense_tensor_format(hidden_states.dtype if out is None else out.dtype),
    )
    traits = {
        "tokens": tokens,
        "input_size": input_size,
        "output_size": weight.shape[0],
        "num_blocks": blocks.shape[0],
        "inputs_contiguous": all(
            tensor.is_contiguous()
            for tensor in (
                hidden_states,
                weight,
                blocks,
                score_weight_a,
                score_weight_b,
                *scratch_a,
                *scratch_b,
            )
        )
        and (out is None or out.is_contiguous()),
    }
    if not hidden_states.is_cuda:
        return None, traits
    try:
        kernel = select_kernel(
            "gemm",
            "linear_attnres_partials",
            signature,
            traits=traits,
            override=override,
            solution=solution,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None
    return kernel, traits


def linear_attnres_partials_available(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    eps: float,
) -> bool:
    """Return whether a registered fused implementation accepts these inputs.

    Args:
        hidden_states: Input shaped ``[tokens, input_size]``.
        weight: Projection weight shaped ``[output_size, input_size]``.
        blocks: Residual candidates shaped ``[blocks, tokens, input_size]``.
        score_weight_a: First AttnRes score weight shaped ``[input_size]``.
        score_weight_b: Second AttnRes score weight shaped ``[input_size]``.
        scratch_a: First ``(max, sum, accumulator)`` FP32 output tuple.
        scratch_b: Second ``(max, sum, accumulator)`` FP32 output tuple.
        eps: Positive RMSNorm epsilon used by both reductions.

    Returns:
        ``True`` when registry dispatch can run the fused operation.
    """
    try:
        _validate_inputs(
            hidden_states,
            weight,
            blocks,
            score_weight_a,
            score_weight_b,
            scratch_a,
            scratch_b,
            eps,
            None,
        )
    except ValueError:
        return False
    kernel, _ = _select_registered_kernel(
        hidden_states,
        weight,
        blocks,
        score_weight_a,
        score_weight_b,
        scratch_a,
        scratch_b,
        None,
        override=None,
        solution=None,
    )
    return kernel is not None


def linear_attnres_partials(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    eps: float,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run a linear projection and two independent AttnRes reductions.

    Args:
        hidden_states: Input shaped ``[tokens, input_size]``.
        weight: Projection weight shaped ``[output_size, input_size]``.
        blocks: Residual candidates shaped ``[blocks, tokens, input_size]``.
        score_weight_a: First AttnRes score weight shaped ``[input_size]``.
        score_weight_b: Second AttnRes score weight shaped ``[input_size]``.
        scratch_a: First ``(max, sum, accumulator)`` FP32 output tuple.
        scratch_b: Second ``(max, sum, accumulator)`` FP32 output tuple.
        eps: Positive RMSNorm epsilon used by both reductions.
        out: Optional contiguous projection output.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Projection output shaped ``[tokens, output_size]``. Both scratch tuples
        are overwritten with the corresponding AttnRes partials.
    """
    out_was_provided = out is not None
    tokens, _, output_shape = _validate_inputs(
        hidden_states,
        weight,
        blocks,
        score_weight_a,
        score_weight_b,
        scratch_a,
        scratch_b,
        eps,
        out,
    )
    if not out_was_provided:
        out = hidden_states.new_empty(output_shape)
    kernel, traits = _select_registered_kernel(
        hidden_states,
        weight,
        blocks,
        score_weight_a,
        score_weight_b,
        scratch_a,
        scratch_b,
        out,
        override=override,
        solution=solution,
    )

    if kernel is not None:
        ShapeCapture.get().record(
            "gemm",
            "linear_attnres_partials",
            kernel.name,
            hidden_states.dtype,
            traits,
        )
        with kernel_scope(
            "gemm",
            "linear_attnres_partials",
            hidden_states.dtype,
            kernel_name=kernel.name,
            **traits,
        ):
            return kernel(
                hidden_states=hidden_states,
                weight=weight,
                blocks=blocks,
                score_weight_a=score_weight_a,
                score_weight_b=score_weight_b,
                scratch_a=scratch_a,
                scratch_b=scratch_b,
                eps=eps,
                out=out,
            )

    # decode_gemv falls back to torch.mm for unlisted shapes, so widening this
    # gate past M == 1 only adds the measured kernels.
    if tokens <= 8 and hidden_states.is_contiguous() and weight.is_contiguous():
        from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

        projected = decode_gemv(hidden_states, weight)
        if out_was_provided:
            out.copy_(projected)
        else:
            out = projected
    else:
        torch.mm(hidden_states, weight.T, out=out)
    triton_partial_eligible = (
        blocks.is_cuda
        and blocks.dtype == torch.bfloat16
        and blocks.shape[2] == KIMI3_HIDDEN_SIZE
        and blocks.stride(2) == 1
        and score_weight_a.is_contiguous()
        and score_weight_b.is_contiguous()
    )
    if triton_partial_eligible:
        from tokenspeed_kernel.ops.activation.triton import attnres_partial_dual

        attnres_partial_dual(
            blocks,
            score_weight_a,
            score_weight_b,
            eps,
            scratch_a,
            scratch_b,
        )
    else:
        values = blocks.float()
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + eps)
        for score_weight, scratch in (
            (score_weight_a, scratch_a),
            (score_weight_b, scratch_b),
        ):
            logits = torch.einsum("bth,h->bt", values, score_weight.float())
            logits *= inverse_rms
            maxima = logits.max(dim=0).values
            unnormalized = torch.exp(logits - maxima)
            scratch[0].copy_(maxima)
            scratch[1].copy_(unnormalized.sum(dim=0))
            scratch[2].copy_(torch.einsum("bt,bth->th", unnormalized, values))
    return out


__all__ = ["linear_attnres_partials", "linear_attnres_partials_available"]
