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
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
except ImportError:
    opt_flags = None

try:
    from triton_kernels.matmul_ogs_details.opt_flags import (
        update_opt_flags_constraints,
        reset_opt_flags_constraints,
    )
except ImportError:
    update_opt_flags_constraints = None
    reset_opt_flags_constraints = None

try:
    from tokenspeed_kernel.thirdparty.triton_kernels.routing import (
        routing as _routing_impl,
    )
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul_ogs as matmul,
    )
    from triton_kernels.numerics import InFlexData
    from triton_kernels.swiglu import swiglu_fn
    from triton_kernels.tensor import (
        FP4,
        convert_layout,
        wrap_torch_tensor,
    )
    from triton_kernels.tensor_details import layout
except ImportError:
    FlexCtx = None
    FnSpecs = None
    FusedActivation = None
    PrecisionConfig = None
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


_AMD_BF16_MXFP4_TILE = {"block_m": 64, "block_n": 128, "block_k": 256}


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
    if update_opt_flags_constraints is None:
        return False
    if not current_platform().is_cdna4:
        return False
    return _is_bf16_mxfp4(x, w, precision_config)


@contextmanager
def _maybe_lds_guard(x, w, precision_config):
    if not _lds_guard_should_apply(x, w, precision_config):
        yield
        return
    update_opt_flags_constraints(_AMD_BF16_MXFP4_TILE)
    try:
        yield
    finally:
        reset_opt_flags_constraints()


if matmul is not None:

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
        dtypes={torch.float16, torch.bfloat16, torch.uint8},
        priority=Priority.PERFORMANT + 2,
        tags={"portability"},
    )

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_matmul_ogs",
        features={"ragged_metadata"},
        **_matmul_common,
    )(_matmul)

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
