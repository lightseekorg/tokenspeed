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

from contextlib import contextmanager

import torch
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Trigger the redirect that aliases ``triton`` -> ``tokenspeed_triton`` for
# upstream ``triton_kernels`` imports.
with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.matmul_details  # noqa: F401
    import triton_kernels.matmul_details.opt_flags  # noqa: F401
    import triton_kernels.numerics  # noqa: F401
    import triton_kernels.swiglu  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401
    import triton_kernels.topk  # noqa: F401

import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
    matmul,
)
from triton_kernels.matmul_details.opt_flags import (
    scoped_opt_flags_constraints,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import (
    FP4,
    RaggedTensorMetadata,
    convert_layout,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details import layout
from triton_kernels.topk import topk


def _is_bf16_mxfp4(x, w, precision_config):
    if precision_config is None:
        return False
    if getattr(precision_config, "b_mx_scale", None) is None:
        return False
    x_dtype = getattr(x, "dtype", None)
    if x_dtype not in (torch.float16, torch.bfloat16):
        return False
    w_bw = getattr(getattr(w, "dtype", None), "bitwidth", None)
    return w_bw == 4


def _lds_guard_should_apply(x, w, precision_config):
    if scoped_opt_flags_constraints is None:
        return False
    if not current_platform().is_cdna4:
        return False
    return _is_bf16_mxfp4(x, w, precision_config)


@contextmanager
def _maybe_lds_guard(x, w, precision_config):
    if not _lds_guard_should_apply(x, w, precision_config):
        yield
        return
    with scoped_opt_flags_constraints({"block_m": 64, "block_n": 128, "block_k": 256}):
        yield


def _matmul(
    x,
    w,
    bias=None,
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    precision_config=None,
    fused_activation=None,
    epilogue=None,
    betas=None,
    gammas=None,
    out_alpha=None,
    y=None,
    n_tokens=None,
    n_expts_act=None,
    **_ignored,
):
    with _maybe_lds_guard(x, w, precision_config):
        out = matmul(
            x,
            w,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
            fused_activation=fused_activation,
            epilogue=epilogue,
            betas=betas,
            gammas=gammas,
            out_alpha=out_alpha,
            c=y,
        )
    if scatter_indx is not None and n_expts_act is not None and n_expts_act > 1:
        assert (
            n_tokens is not None
        ), "n_tokens required when n_expts_act > 1 for top-k reduction"
        return out.view(n_tokens, n_expts_act, out.shape[-1]).sum(dim=1)
    return out


_matmul_common = dict(
    solution="triton",
    signatures=format_signatures(
        "x", "dense", {torch.float16, torch.bfloat16, torch.uint8}
    ),
    priority=Priority.PERFORMANT + 2,
    tags={"portability"},
)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_dispatch_gemm",
    features={"ragged_metadata", "dispatch_gemm"},
    **_matmul_common,
)(_matmul)

register_kernel(
    "moe",
    "experts",
    name="triton_kernels_gemm_combine",
    features={"ragged_metadata", "gemm_combine"},
    **_matmul_common,
)(_matmul)


@register_kernel(
    "moe",
    "route",
    name="triton_kernels_routing",
    solution="triton",
    signatures=format_signatures(
        "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    traits={"output_type": frozenset({"ragged_metadata"})},
    priority=Priority.PERFORMANT + 2,
    tags={"portability"},
)
def triton_kernels_routing(
    logits: torch.Tensor,
    n_expts_act: int,
    sm_first: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = logits.dtype

    assert logits.ndim == 2, "router_logits must be (n_tokens, n_expts_tot)"
    n_tokens, _ = logits.shape

    assert sm_first is False, "sm_first=True not supported for triton_kernels_routing"

    sparse = topk(logits, n_expts_act, apply_softmax=not sm_first)
    mask_metadata = sparse.mask_metadata

    col_sorted = mask_metadata.col_sorted_indx
    gather_indx = col_sorted // n_expts_act
    scatter_indx = col_sorted

    vals_flat = sparse.vals.reshape(-1)
    if dtype is not None and vals_flat.dtype != dtype:
        vals_flat = vals_flat.to(dtype)
    gate_scal = vals_flat[scatter_indx]

    n_total_rows = n_tokens * n_expts_act
    ragged_metadata = make_ragged_tensor_metadata(mask_metadata.col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal


__all__ = [
    "FP4",
    "FlexCtx",
    "FnSpecs",
    "FusedActivation",
    "InFlexData",
    "PrecisionConfig",
    "convert_layout",
    "layout",
    "opt_flags",
    "swiglu_fn",
    "wrap_torch_tensor",
]
