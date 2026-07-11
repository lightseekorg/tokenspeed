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

"""Manifold-constrained hyper-connection kernel entry points."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["mhc_plan", "mhc_pre", "mhc_post"]


def _require_residual_shape(residual: torch.Tensor) -> tuple[tuple[int, ...], int, int]:
    if residual.ndim < 3:
        raise ValueError("residual must have shape [..., HC, H]")
    outer_shape = tuple(residual.shape[:-2])
    return outer_shape, residual.shape[-2], residual.shape[-1]


def mhc_plan(
    *,
    hc_mult: int,
    residual_dtype: torch.dtype = torch.bfloat16,
    fn_dtype: torch.dtype = torch.float32,
    mix_dtype: torch.dtype = torch.float32,
    solution: str | None = None,
) -> dict[str, object]:
    """Select matching mHC pre/post kernels for model setup.

    Args:
        hc_mult: Number of residual streams.
        residual_dtype: Storage dtype of residual and layer tensors.
        fn_dtype: Storage dtype of the pre-map projection weight.
        mix_dtype: Storage dtype of scale, base, post, and combine tensors.
        solution: Optional solution to force through normal selection.

    Returns:
        A plan containing the selected pre/post kernel names and their common
        solution. Execution can pin either kernel through its recorded name.
    """
    pre = select_kernel(
        "mhc",
        "pre",
        format_signature(
            residual=dense_tensor_format(residual_dtype),
            fn=dense_tensor_format(fn_dtype),
            mhc_scale=dense_tensor_format(mix_dtype),
            mhc_base=dense_tensor_format(mix_dtype),
        ),
        traits={"hc_mult": hc_mult},
        solution=solution,
    )
    post = select_kernel(
        "mhc",
        "post",
        format_signature(
            hidden_states=dense_tensor_format(residual_dtype),
            residual=dense_tensor_format(residual_dtype),
            post=dense_tensor_format(mix_dtype),
            comb=dense_tensor_format(mix_dtype),
        ),
        traits={"hc_mult": hc_mult},
        solution=solution,
    )
    registry = KernelRegistry.get()
    pre_spec = registry.get_by_name(pre.name)
    post_spec = registry.get_by_name(post.name)
    if pre_spec is None or post_spec is None:
        raise RuntimeError("Selected mHC kernel is missing its registry spec")
    if pre_spec.solution != post_spec.solution:
        raise RuntimeError(
            "mHC pre/post selection must use one solution, got "
            f"{pre_spec.solution!r} and {post_spec.solution!r}"
        )
    return {
        "hc_mult": hc_mult,
        "pre_kernel_name": pre.name,
        "post_kernel_name": post.name,
        "solution": pre_spec.solution,
    }


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    *,
    solution: str | None = None,
    override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the mHC pre-map through a registered ``mhc.pre`` kernel.

    Args:
        residual: Multi-stream residual tensor shaped ``[..., HC, H]``.
        fn: Projection weight shaped ``[(2 + HC) * HC, HC * H]``.
        mhc_scale: Three FP32 scale values for pre, post, and combine mixes.
        mhc_base: FP32 projection bias shaped ``[(2 + HC) * HC]``.
        rms_eps: RMS normalization epsilon.
        hc_eps: Numerical epsilon used by pre-map and Sinkhorn normalization.
        sinkhorn_iters: Number of alternating Sinkhorn normalization iterations.
        solution: Optional registered solution to select.
        override: Optional exact kernel-name or solution override.

    Returns:
        ``(layer_input, post_mix, comb_mix)`` with shapes ``[..., H]``,
        ``[..., HC, 1]``, and ``[..., HC, HC]`` respectively.
    """
    outer_shape, hc_mult, hidden_size = _require_residual_shape(residual)
    num_tokens = math.prod(outer_shape)
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape,
                hc_mult,
                1,
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual.device,
            ),
        )

    signature = format_signature(
        residual=dense_tensor_format(residual.dtype),
        fn=dense_tensor_format(fn.dtype),
        mhc_scale=dense_tensor_format(mhc_scale.dtype),
        mhc_base=dense_tensor_format(mhc_base.dtype),
    )
    kernel = select_kernel(
        "mhc",
        "pre",
        signature,
        traits={"hc_mult": hc_mult},
        solution=solution,
        override=override,
    )
    shape_params = {
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
    }
    ShapeCapture.get().record(
        "mhc",
        "pre",
        kernel.name,
        residual.dtype,
        shape_params,
    )
    with kernel_scope(
        "mhc",
        "pre",
        residual.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            residual=residual,
            fn=fn,
            mhc_scale=mhc_scale,
            mhc_base=mhc_base,
            rms_eps=rms_eps,
            mhc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
        )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    solution: str | None = None,
    override: str | None = None,
) -> torch.Tensor:
    """Apply the mHC post-map through a registered ``mhc.post`` kernel.

    Args:
        hidden_states: Current layer output shaped ``[..., H]``.
        residual: Multi-stream residual tensor shaped ``[..., HC, H]``.
        post: Per-stream post mix shaped ``[..., HC, 1]``.
        comb: Residual combination matrix shaped ``[..., HC, HC]``.
        solution: Optional registered solution to select.
        override: Optional exact kernel-name or solution override.

    Returns:
        Updated multi-stream residual with the same shape and dtype as
        ``residual``.
    """
    outer_shape, hc_mult, hidden_size = _require_residual_shape(residual)
    num_tokens = math.prod(outer_shape)
    if num_tokens == 0:
        return torch.empty_like(residual)

    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        residual=dense_tensor_format(residual.dtype),
        post=dense_tensor_format(post.dtype),
        comb=dense_tensor_format(comb.dtype),
    )
    kernel = select_kernel(
        "mhc",
        "post",
        signature,
        traits={"hc_mult": hc_mult},
        solution=solution,
        override=override,
    )
    shape_params = {
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
    }
    ShapeCapture.get().record(
        "mhc",
        "post",
        kernel.name,
        residual.dtype,
        shape_params,
    )
    with kernel_scope(
        "mhc",
        "post",
        residual.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            hidden_states=hidden_states,
            residual=residual,
            post=post,
            comb=comb,
        )


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.mhc.deep_gemm  # noqa: E402,F401
