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


def _launch(
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
    quantize_fp8,
):
    api = _api()
    if api.get_pdl() != pdl_enabled():
        api.set_pdl(pdl_enabled())
    new_residual, y = torch.empty_like(residual), torch.empty_like(x)
    new_pre, new_post, new_comb = (
        torch.empty_like(tensor) for tensor in (pre, post, comb)
    )
    tokens, hidden = x.shape
    y_fp8 = y_sf = None
    if quantize_fp8:
        # Packed group-32 scales, MN-major with the token count padded to four
        # so every column starts 16-byte aligned for TMA.
        y_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        y_sf = torch.empty(
            hidden // 128, (tokens + 3) // 4 * 4, dtype=torch.int32, device=x.device
        ).mT[:tokens]
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
        y_fp8=y_fp8,
        y_gemm_sf=y_sf,
    )
    return new_residual, y, new_pre, new_post, new_comb, y_fp8, y_sf


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

    Inputs follow ``try_mhc_shifted_post_pre_norm``. Warm up the eventual
    capture stream first: DeepGEMM initializes its per-stream split barriers
    outside capture and advances their epoch after each launch.
    No input is mutated, and no output is retained outside this invocation.
    """
    return _launch(
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
        False,
    )[:5]


def shifted_post_pre_norm_quantize_fp8(
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
    """``shifted_post_pre_norm`` that also returns the input's FP8 codes and scales.

    The extra outputs are E4M3 ``[T,H]`` codes and int32 ``[T,H/128]`` packed
    group-32 UE8M0 scales, strided ``(1, T padded to 4)``: the layout
    ``fp8_projection`` reads. The BF16 input is returned as well.
    """
    new_residual, y, new_pre, new_post, new_comb, y_fp8, y_sf = _launch(
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
        True,
    )
    return new_residual, y, new_pre, new_post, new_comb, y_fp8, y_sf
