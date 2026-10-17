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

"""Norm-preserving gated residual entry points."""

import torch
from tokenspeed_kernel.ops.residual._cute_dsl import gated_boundary as _cute_dsl
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_format_signature

# A producer asks this before publishing: only the cluster boundary gathers a mailbox.
boundary_gathers_mailbox = _cute_dsl.mailbox_boundary_supported


def residual_gate_coefficients(
    gate: torch.Tensor, temperature: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel ``(alpha, beta)`` of the norm-preserving residual gate.

    The gate is a parameter, so its sigmoid and square root are evaluated once
    here and the fused boundary reads the FP32 results back.

    Args:
        gate: Gate logits ``[width]``.
        temperature: Divisor applied to the logits before the sigmoid.
        eps: Lower bound for the squared stream coefficient.

    Returns:
        ``(alpha, beta)``: FP32 ``[width]`` coefficients of the stream and of
        the branch.
    """
    kernel = select_kernel(
        "residual",
        "residual_gate_coefficients",
        dense_format_signature(("gate",), (gate.dtype,)),
    )
    return kernel(gate, temperature, eps)


def post_norm_residual_pre_norm(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float,
    post_weight_offset: float,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor | None,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
    branch_sentinel: int | None,
    branch_ranks: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fuse a decoder-layer boundary: post-norm, gated residual, then pre-norm.

    Batches of up to ``cute_dsl.CLUSTER_BOUNDARY_MAX_ROWS`` rows run on the
    CuTe DSL cluster kernel where it is available; everything else, and every
    other platform, runs the Triton kernel, both under this contract. The
    two sum each row's variance in a different order, so a batch inside the ceiling and one past it agree to
    the last bit of the variance rather than bitwise; each is deterministic.

    Only the cluster kernel gathers a mailbox branch: it polls the armed words
    itself and re-arms them, so a batch it cannot serve is refused rather
    than handed a mailbox the Triton kernel would read as it stands. A mailbox
    row may hold one partial per rank, side by side; the kernel then sums them
    in BF16 in rank order, the bits the one-shot all-reduce kernel would have
    produced, so the all-reduce launch is not needed.

    Args:
        stream: FP32 residual stream ``[rows, width]``.
        branch: BF16 branch output with ``stream``'s shape, or with a
            sentinel the armed mailbox ``[capacity, width * branch_ranks]``
            holding it, one partial per rank in each row.
        post_weight: Post-norm input-scale weight ``[width]``, scaling the
            branch before its RMS, or None for a weightless norm.
        post_eps: Post-norm epsilon.
        post_weight_offset: Center added to the post-norm checkpoint weight.
        alpha: FP32 stream coefficients ``[width]``.
        beta: FP32 branch coefficients ``[width]``.
        pre_weight: Affine pre-norm weight ``[width]``, or None to stop after
            the residual.
        pre_eps: Pre-norm epsilon.
        pre_weight_offset: Center added to the pre-norm checkpoint weight.
        enable_pdl: Enable NVIDIA programmatic dependent launch.
        branch_sentinel: The word ``branch`` is armed with when it is a
            mailbox, or None for an ordinary branch.
        branch_ranks: Partials per mailbox row to sum; 1 for an ordinary
            branch or a mailbox read as it stands.

    Returns:
        ``(new_stream, pre_fp32, pre_bf16)``: the FP32 stream after the
        residual, and the pre-normed rows in FP32 and BF16, or ``None`` for
        both when ``pre_weight`` is None.
    """
    kernel = select_kernel(
        "residual",
        "post_norm_residual_pre_norm",
        dense_format_signature(("stream", "branch"), (stream.dtype, branch.dtype)),
        traits={
            "cluster_supported": _cute_dsl.cluster_boundary_supported(
                stream,
                branch,
                post_weight,
                alpha,
                beta,
                pre_weight,
                branch_sentinel,
                branch_ranks,
            )
        },
    )
    return kernel(
        stream,
        branch,
        post_weight,
        post_eps,
        post_weight_offset,
        alpha,
        beta,
        pre_weight,
        pre_eps,
        pre_weight_offset,
        enable_pdl,
        branch_sentinel,
        branch_ranks,
    )


def post_norm_residual_pre_norm_bf16(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float,
    post_weight_offset: float,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor | None,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
    branch_sentinel: int | None,
    branch_ranks: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the residual stream and BF16 next input, using the same selection.

    Arguments match ``post_norm_residual_pre_norm``. Triton skips the unused
    FP32 normalized output; other selected implementations may still compute it.
    """
    kernel = select_kernel(
        "residual",
        "post_norm_residual_pre_norm",
        dense_format_signature(("stream", "branch"), (stream.dtype, branch.dtype)),
        traits={
            "cluster_supported": _cute_dsl.cluster_boundary_supported(
                stream,
                branch,
                post_weight,
                alpha,
                beta,
                pre_weight,
                branch_sentinel,
                branch_ranks,
            )
        },
    )
    if kernel.name == "triton_post_norm_residual_pre_norm":
        from tokenspeed_kernel.ops.residual._triton.gated_boundary import (
            _launch_post_norm_residual_pre_norm,
        )

        stream_out, _, next_input = _launch_post_norm_residual_pre_norm(
            stream,
            branch,
            post_weight,
            post_eps,
            post_weight_offset,
            alpha,
            beta,
            pre_weight,
            pre_eps,
            pre_weight_offset,
            enable_pdl,
            branch_sentinel,
            branch_ranks,
            return_pre_fp32=False,
        )
    else:
        stream_out, _, next_input = kernel(
            stream,
            branch,
            post_weight,
            post_eps,
            post_weight_offset,
            alpha,
            beta,
            pre_weight,
            pre_eps,
            pre_weight_offset,
            enable_pdl,
            branch_sentinel,
            branch_ranks,
        )
    return stream_out, next_input


__all__ = [
    "boundary_gathers_mailbox",
    "post_norm_residual_pre_norm",
    "post_norm_residual_pre_norm_bf16",
    "residual_gate_coefficients",
]
