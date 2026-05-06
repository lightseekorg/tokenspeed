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

# Trigger the redirect that aliases ``triton`` -> ``tokenspeed_triton`` for
# upstream ``triton_kernels`` imports.
import tokenspeed_kernel.thirdparty.triton_kernels  # noqa: F401
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel

try:
    import triton_kernels.matmul_details.opt_flags as opt_flags
except ImportError:
    opt_flags = None

try:
    from triton_kernels.matmul_details.opt_flags import (
        reset_opt_flags_constraints,
        update_opt_flags_constraints,
    )
except ImportError:
    update_opt_flags_constraints = None
    reset_opt_flags_constraints = None

try:
    from tokenspeed_kernel.thirdparty.triton_kernels.routing import (
        routing as _routing_impl,
    )
    from triton_kernels.matmul import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul,
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
except ImportError:
    FlexCtx = None
    FnSpecs = None
    FusedActivation = None
    PrecisionConfig = None
    RaggedTensorMetadata = None
    make_ragged_tensor_metadata = None
    matmul = None
    InFlexData = None
    _routing_impl = None
    swiglu_fn = None
    FP4 = None
    convert_layout = None
    wrap_torch_tensor = None
    layout = None


if _routing_impl is not None:

    def _triton_kernels_routing(logits, n_expts_act, sm_first=False, dtype=None):
        if dtype is None:
            dtype = logits.dtype
        return _routing_impl(logits, n_expts_act, sm_first=sm_first, dtype=dtype)

    register_kernel(
        "moe",
        "route",
        name="triton_kernels_routing",
        solution="triton",
        dtypes={torch.float16, torch.bfloat16, torch.float32},
        traits={"output_type": frozenset({"ragged_metadata"})},
        priority=Priority.PERFORMANT + 2,
        tags={"portability"},
    )(_triton_kernels_routing)


# Hot fix to avoid exceeding LDS budget on MI355 for large GEMMs.
# TODO(kylewng): Remove this once fixed in upstream triton_kernels.
def _lds_guard_should_apply(M, n, n_experts):
    if current_platform().is_nvidia:
        return False
    if update_opt_flags_constraints is None or reset_opt_flags_constraints is None:
        return False
    if n is None or n < 2048:
        return False
    tokens_per_expt = max(1, M // max(1, n_experts))
    return tokens_per_expt >= 512


@contextmanager
def _maybe_lds_guard(M, n, n_experts):
    if not _lds_guard_should_apply(M, n, n_experts):
        yield
        return
    update_opt_flags_constraints({"block_m": 128})
    try:
        yield
    finally:
        reset_opt_flags_constraints()
        update_opt_flags_constraints({"block_k": 256})


def _ragged_n_experts(a_ragged_metadata):
    if a_ragged_metadata is None:
        return 1
    return getattr(a_ragged_metadata, "n_slices", 1) or 1


def _w_out_dim(w):
    try:
        return w.shape[-1]
    except AttributeError:
        return None


if matmul is not None:

    def _matmul_dispatch_gemm(
        x,
        w,
        bias=None,
        a_ragged_metadata=None,
        gather_indx=None,
        precision_config=None,
        fused_activation=None,
        epilogue=None,
        betas=None,
        gammas=None,
        out_alpha=None,
        y=None,
    ):
        if gather_indx is not None and hasattr(gather_indx, "shape"):
            M = gather_indx.shape[0]
        else:
            M = x.shape[-2]
        n_experts = _ragged_n_experts(a_ragged_metadata)
        with _maybe_lds_guard(M, _w_out_dim(w), n_experts):
            return matmul(
                x,
                w,
                bias,
                a_ragged_metadata=a_ragged_metadata,
                gather_indx=gather_indx,
                precision_config=precision_config,
                fused_activation=fused_activation,
                epilogue=epilogue,
                betas=betas,
                gammas=gammas,
                out_alpha=out_alpha,
                c=y,
            )

    def _matmul_gemm_combine(
        x,
        w,
        bias=None,
        a_ragged_metadata=None,
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
    ):
        if scatter_indx is not None and hasattr(scatter_indx, "shape"):
            M = scatter_indx.shape[0]
        else:
            M = x.shape[-2]
        n_experts = _ragged_n_experts(a_ragged_metadata)
        with _maybe_lds_guard(M, _w_out_dim(w), n_experts):
            out = matmul(
                x,
                w,
                bias,
                a_ragged_metadata=a_ragged_metadata,
                scatter_indx=scatter_indx,
                precision_config=precision_config,
                fused_activation=fused_activation,
                epilogue=epilogue,
                betas=betas,
                gammas=gammas,
                out_alpha=out_alpha,
                c=y,
            )
        if n_expts_act is not None and n_expts_act > 1:
            assert (
                n_tokens is not None
            ), "n_tokens required when n_expts_act > 1 for top-k reduction"
            return out.view(n_tokens, n_expts_act, out.shape[-1]).sum(dim=1)
        return out

    _matmul_common = dict(
        solution="triton",
        dtypes={torch.float16, torch.bfloat16, torch.uint8},
        priority=Priority.PERFORMANT + 2,
        tags={"portability"},
    )

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_dispatch_gemm",
        features={"ragged_metadata", "dispatch_gemm"},
        **_matmul_common,
    )(_matmul_dispatch_gemm)

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_gemm_combine",
        features={"ragged_metadata", "gemm_combine"},
        **_matmul_common,
    )(_matmul_gemm_combine)
