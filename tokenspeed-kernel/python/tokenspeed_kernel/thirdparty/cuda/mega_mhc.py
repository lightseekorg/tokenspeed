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

"""Optional DeepGEMM shifted mHC adapter, with caller-owned BF16 outputs."""

from functools import lru_cache

import torch
from tokenspeed_kernel.platform import pdl_enabled, prepare_cuda_toolkit_env


@lru_cache(maxsize=1)
def _api():
    prepare_cuda_toolkit_env()
    import deep_gemm

    if not callable(getattr(deep_gemm, "mega_mhc", None)):
        raise ImportError("Installed DeepGEMM does not expose mega_mhc")
    return deep_gemm


@lru_cache(maxsize=1)
def is_mega_mhc_available() -> bool:
    try:
        _api()
    except (ImportError, OSError):
        return False
    return True


def shifted_post_pre_norm(
    x,
    residual,
    pre,
    post,
    comb,
    weight,
    scale,
    base,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    norm_weight,
    norm_eps,
):
    """Return new residual, normalized input and next pre/post/comb coefficients.

    Inputs follow ``try_mhc_shifted_post_pre_norm``. Each execution stream must
    be warmed before graph capture: DeepGEMM caches split barriers per stream.
    No input is mutated, and no output is retained outside this invocation.
    """
    api = _api()
    if api.get_pdl() != pdl_enabled():
        api.set_pdl(pdl_enabled())
    new_residual, y = torch.empty_like(residual), torch.empty_like(x)
    new_pre, new_post, new_comb = (
        torch.empty_like(tensor) for tensor in (pre, post, comb)
    )
    api.mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=pre.unsqueeze(-1),
        post_mix=post.unsqueeze(-1),
        comb_res_mix=comb,
        fn=weight,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=4,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=hc_eps,
        num_sinkhorn_iters=sinkhorn_iters,
        rmsnorm_weight=norm_weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=new_residual,
        new_prev_mix=new_pre.unsqueeze(-1),
        new_post_mix=new_post.unsqueeze(-1),
        new_comb_res_mix=new_comb,
        y_bf16=y,
    )
    return new_residual, y, new_pre, new_post, new_comb
