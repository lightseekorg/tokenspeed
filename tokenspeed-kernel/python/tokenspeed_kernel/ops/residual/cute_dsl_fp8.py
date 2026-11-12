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

"""FP8 weight-only and FP8 activation HC mixing through the ordinary dispatcher."""

from __future__ import annotations

import threading

import torch
from tokenspeed_kernel.ops.quantization.triton import (
    fp8_quantize_by_multiplier,
    fp8_quantize_per_token,
    fp8_token_multiplier,
    fp8_token_scale,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

try:
    from cuda.bindings.driver import CUstream
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
        Fp8GatedResidualBranchUp,
        Fp8GatedResidualGemm,
        Fp8GatedResidualInputQuantize,
        Fp8GatedResidualM32WeightStationaryUp,
        Fp8GatedResidualM64H32IndependentAUp,
        Fp8GatedResidualM64H56Up,
        Fp8GatedResidualM64H64CachedS5Up,
        Fp8GatedResidualM64H64Up,
        Fp8GatedResidualPairedHiddenUp,
        Fp8GatedResidualProjectionEpilogue,
        Fp8GatedResidualTokenReuseUp,
        W8GatedResidualBranchUp,
        W8GatedResidualCachedScaleTokenReuse,
        W8GatedResidualCachedScaleUp,
        W8GatedResidualDecodedRingUp,
        W8GatedResidualDecodedTokenReuse,
        W8GatedResidualIndependentUp,
        W8GatedResidualRotatedProjection,
        W8GatedResidualTmemProjection,
        W8GatedResidualTokenReuse,
    )

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False

_PLANS = {}
_PLAN_LOCK = threading.Lock()
_WARMED = set()


_EPI_PLANS = {}


def _projection_epilogue(
    partial,
    input_scale,
    weight_scale,
    activation,
    activation_scale,
    inject,
    *,
    r,
    p,
    g,
    projection_scale,
    input_fp8,
    activation_fp8,
):
    threads = 512 if r >= 1024 else 256
    splits = partial.shape[0]
    # Near-full legacy Up tiles are faster when activation quantization finishes first.
    early_up_signal = (
        pdl_enabled()
        and activation_fp8
        and not (r == 320 and 112 < partial.shape[1] <= 128)
    )
    parallel_inject = (
        pdl_enabled()
        and activation_fp8
        and p > r
        and r == 1024
        and partial.shape[1] > 32
    )
    key = (
        splits,
        partial.device.index,
        activation.dtype,
        inject.dtype,
        r,
        p,
        g,
        projection_scale,
        input_fp8,
        activation_fp8,
        weight_scale.numel() > 1,
        pdl_enabled(),
        threads,
        early_up_signal,
        parallel_inject,
    )
    operands = tuple(
        from_dlpack(
            t.detach(), assumed_align=4 if t.ndim == 1 else 16
        ).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (
            partial,
            input_scale,
            weight_scale,
            activation,
            activation_scale,
            inject,
        )
    )
    stream = CUstream(int(torch.cuda.current_stream(partial.device).cuda_stream))
    compiled = _EPI_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _EPI_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm FP8 projection epilogue before graph capture"
                    )
                kernel = Fp8GatedResidualProjectionEpilogue(
                    lowrank=r,
                    projection_rows=p,
                    hc_count=g,
                    splits=splits,
                    projection_scale=projection_scale,
                    input_fp8=input_fp8,
                    activation_fp8=activation_fp8,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                    use_pdl=pdl_enabled(),
                    block_threads=threads,
                    early_up_signal=early_up_signal,
                    parallel_inject=parallel_inject,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _EPI_PLANS[key] = compiled
    compiled(*operands, stream)


_BRANCH_PLANS = {}


def _branch_up(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    out,
    *,
    h,
    r,
    weights_independent,
):
    n = 16 if x.shape[0] <= 16 else (32 if x.shape[0] <= 32 else 64)
    k, stages = 128, (2 if n == 64 else 3)
    key = (
        x.device.index,
        x.dtype,
        h,
        r,
        n,
        k,
        stages,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
    )
    operands = tuple(
        from_dlpack(
            t.detach(), assumed_align=4 if t.ndim == 1 else 16
        ).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (x, weight, original, weight_scale, activation_scale, out)
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _BRANCH_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _BRANCH_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm W8 branch Up before graph capture")
                kernel = W8GatedResidualBranchUp(
                    hidden_size=h,
                    lowrank=r,
                    token_tile=n,
                    k_tile=k,
                    stages=stages,
                    use_pdl=pdl_enabled(),
                    weights_independent=weights_independent,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _BRANCH_PLANS[key] = compiled
    compiled(*operands, stream)


_REUSE_PLANS = {}


def _reuse_up(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    out,
    *,
    h,
    r,
    weights_independent,
):
    vectorized = weight_scale.numel() > 1 and weight_scale.data_ptr() % 16 == 0
    key = (
        x.device.index,
        x.dtype,
        h,
        r,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
        vectorized,
    )
    operands = tuple(
        (
            from_dlpack(
                t.detach(), assumed_align=4 if t.ndim == 1 else 16
            ).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (x, weight, original, weight_scale, activation_scale, out)
        )
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _REUSE_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _REUSE_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm token-reuse Up before graph capture")
                kernel = W8GatedResidualCachedScaleTokenReuse(
                    hidden_size=h,
                    lowrank=r,
                    raw_stages=4,
                    decoded_stages=3,
                    input_stages=2,
                    use_pdl=pdl_enabled(),
                    weights_independent=weights_independent,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                    weight_scale_vectorized=vectorized,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _REUSE_PLANS[key] = compiled
    compiled(*operands, stream)


_INDEPENDENT_PLANS = {}


def _independent_up(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    out,
    *,
    h,
    r,
    weights_independent,
):
    vectorized = weight_scale.numel() > 1 and weight_scale.data_ptr() % 16 == 0
    up_token_rows = 64 if x.shape[0] <= 64 else 128
    raw_stages = 6 if up_token_rows == 64 else 5
    input_stages = 4 if up_token_rows == 64 else 3
    decoded_stages = 3 if up_token_rows == 64 else 3
    key = (
        x.device.index,
        x.dtype,
        h,
        r,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
        vectorized,
        up_token_rows,
        raw_stages,
        decoded_stages,
        input_stages,
    )
    operands = tuple(
        (
            from_dlpack(
                t.detach(), assumed_align=4 if t.ndim == 1 else 16
            ).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (x, weight, original, weight_scale, activation_scale, out)
        )
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _INDEPENDENT_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _INDEPENDENT_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm token-reuse Up before graph capture")
                kernel = (
                    W8GatedResidualCachedScaleUp
                    if up_token_rows == 128
                    else W8GatedResidualDecodedRingUp
                )(
                    hidden_size=h,
                    lowrank=r,
                    use_pdl=pdl_enabled(),
                    weights_independent=weights_independent,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                    weight_scale_vectorized=vectorized,
                    up_token_rows=up_token_rows,
                    raw_stages=raw_stages,
                    decoded_stages=decoded_stages,
                    input_stages=input_stages,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _INDEPENDENT_PLANS[key] = compiled
    compiled(*operands, stream)


def _gemm(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    partials,
    out,
    *,
    role,
    g,
    h,
    r,
    p,
    weights_independent,
):
    if (
        role == "mix"
        and x.dtype != torch.float8_e4m3fn
        and (r == 1024)
        and (x.shape[0] > 16)
        and (not 128 < x.shape[0] <= 256)
    ):
        return _independent_up(
            x,
            weight,
            original,
            weight_scale,
            activation_scale,
            out,
            h=h,
            r=r,
            weights_independent=weights_independent,
        )
    if (
        role == "mix"
        and x.dtype != torch.float8_e4m3fn
        and (r == 1024)
        and (128 < x.shape[0] <= 256)
    ):
        return _reuse_up(
            x,
            weight,
            original,
            weight_scale,
            activation_scale,
            out,
            h=h,
            r=r,
            weights_independent=weights_independent,
        )
    if (
        role == "mix"
        and x.dtype != torch.float8_e4m3fn
        and (r == 1024)
        and (x.shape[0] <= 32)
    ):
        return _branch_up(
            x,
            weight,
            original,
            weight_scale,
            activation_scale,
            out,
            h=h,
            r=r,
            weights_independent=weights_independent,
        )
    k = 128
    original_n = 16 if x.shape[0] <= 16 else 32 if x.shape[0] <= 32 else 64
    n = 128 if x.shape[0] > 64 else original_n
    m = (64 if r <= 384 else 128) if role == "projection" else 128
    split = 16 if role == "projection" else 1
    stages = (
        3
        if x.dtype == torch.float8_e4m3fn
        else 6 if n == 16 else 5 if n == 32 else 4 if n == 64 else 3
    )
    if x.dtype != torch.float8_e4m3fn and r == 1024 and (n == 16):
        stages = 3
    if x.dtype == torch.float8_e4m3fn:
        return _native_gemm(
            x,
            weight,
            original,
            weight_scale,
            activation_scale,
            partials,
            out,
            role=role,
            g=g,
            h=h,
            r=r,
            p=p,
            weights_independent=weights_independent,
        )
    if role == "projection" and x.shape[0] > 128:
        n = 256
        stages = 2
    token_major_up = role == "mix" and n > 64
    if role == "mix" and x.dtype != torch.float8_e4m3fn and (r == 320) and (n > 64):
        stages = 2
    up_hidden_tile = (
        64
        if role == "mix"
        and x.dtype != torch.float8_e4m3fn
        and (r == 1024)
        and (n >= 64)
        else 32
    )
    if up_hidden_tile == 64:
        stages = 2
    separate_operand_ready = (
        role == "projection"
        and x.dtype != torch.float8_e4m3fn
        and (r == 1024)
        and (x.shape[0] > 64)
    )
    up_token_rows = (
        64
        if role == "mix"
        and x.dtype != torch.float8_e4m3fn
        and (r == 1024)
        and (n == 64)
        else 128
    )
    if up_token_rows == 64:
        token_major_up = True
    if (
        role == "mix"
        and x.dtype != torch.float8_e4m3fn
        and (r == 320)
        and (x.shape[0] > 64)
    ):
        k, stages = (192, 2)
    weight_scale_vectorized = (
        role == "mix"
        and weight_scale.numel() > 1
        and (weight_scale.data_ptr() % 16 == 0)
    )
    use_tmem_weights = (
        role == "projection" and x.dtype != torch.float8_e4m3fn and (r == 1024)
    )
    if use_tmem_weights:
        stages = 7 if n <= 64 else 6 if n == 128 else 4
        separate_operand_ready = True
    key = (
        x.device.index,
        x.dtype,
        original.dtype,
        role,
        g,
        h,
        r,
        p,
        n,
        m,
        split,
        stages,
        k,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
        token_major_up,
        weight_scale_vectorized,
        up_hidden_tile,
        up_token_rows,
        separate_operand_ready,
        use_tmem_weights,
    )
    values = (x, weight, original, weight_scale, activation_scale, partials, out)
    operands = tuple(
        (
            from_dlpack(
                value.detach(), assumed_align=4 if value.ndim == 1 else 16
            ).mark_layout_dynamic(leading_dim=value.ndim - 1)
            for value in values
        )
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm the FP8 mixer tile before CUDA graph capture"
                    )
                if use_tmem_weights:
                    kernel = (
                        W8GatedResidualRotatedProjection
                        if n > 64
                        else W8GatedResidualTmemProjection
                    )(
                        role=role,
                        hc_count=g,
                        hidden_size=h,
                        lowrank=r,
                        projection_rows=p,
                        token_tile=n,
                        projection_tile=m,
                        k_tile=k,
                        stages=stages,
                        split_k=split,
                        use_pdl=pdl_enabled(),
                        weights_independent=weights_independent,
                        weight_scale_per_channel=weight_scale.numel() > 1,
                        token_major_up=token_major_up,
                        up_hidden_tile=up_hidden_tile,
                        up_token_rows=up_token_rows,
                        separate_operand_ready=separate_operand_ready,
                        epilogue_threads=256 if n == 16 else 512,
                        shared_up_scales=False,
                        interleave_tma=False,
                        weight_scale_vectorized=weight_scale_vectorized,
                        raw_stages=(
                            10 if n == 16 else 9 if n == 32 else 8 if n <= 128 else 6
                        ),
                        input_stages=(
                            16
                            if n == 16
                            else (
                                10
                                if n == 32
                                else 6 if n == 64 else 3 if n == 128 else 2
                            )
                        ),
                    )
                else:
                    kernel = Fp8GatedResidualGemm(
                        role=role,
                        hc_count=g,
                        hidden_size=h,
                        lowrank=r,
                        projection_rows=p,
                        token_tile=n,
                        projection_tile=m,
                        k_tile=k,
                        stages=stages,
                        split_k=split,
                        use_pdl=pdl_enabled(),
                        weights_independent=weights_independent,
                        weight_scale_per_channel=weight_scale.numel() > 1,
                        token_major_up=token_major_up,
                        up_hidden_tile=up_hidden_tile,
                        up_token_rows=up_token_rows,
                        separate_operand_ready=separate_operand_ready,
                        epilogue_threads=256 if n == 16 else 512,
                        shared_up_scales=False,
                        interleave_tma=False,
                        weight_scale_vectorized=weight_scale_vectorized,
                    )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _PLANS[key] = compiled
    compiled(*operands, stream)


_NATIVE_BRANCH_PLANS = {}
_PAIRED_PLANS = {}


def _native_branch_up(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    out,
    *,
    h,
    r,
    weights_independent,
):
    n = 16 if x.shape[0] <= 16 else 32
    k = 256
    stages = 3
    key = (
        x.device.index,
        original.dtype,
        h,
        r,
        n,
        k,
        stages,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
    )
    operands = tuple(
        from_dlpack(
            t.detach(), assumed_align=4 if t.ndim == 1 else 16
        ).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (x, weight, original, weight_scale, activation_scale, out)
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _NATIVE_BRANCH_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _NATIVE_BRANCH_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm branch Up before graph capture")
                kernel = Fp8GatedResidualBranchUp(
                    hidden_size=h,
                    lowrank=r,
                    token_tile=n,
                    k_tile=k,
                    stages=stages,
                    use_pdl=pdl_enabled(),
                    weights_independent=weights_independent,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _NATIVE_BRANCH_PLANS[key] = compiled
    compiled(*operands, stream)


def _paired_up(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    out,
    *,
    h,
    r,
    weights_independent,
):
    n = 32 if x.shape[0] <= 32 else 64
    k, stages = (256, 3) if n == 32 else (128, 3)
    key = (
        x.device.index,
        original.dtype,
        h,
        r,
        n,
        k,
        stages,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
    )
    operands = tuple(
        from_dlpack(
            t.detach(), assumed_align=4 if t.ndim == 1 else 16
        ).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (x, weight, original, weight_scale, activation_scale, out)
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _PAIRED_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _PAIRED_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm paired hidden Up before graph capture")
                kernel = Fp8GatedResidualPairedHiddenUp(
                    hidden_size=h,
                    lowrank=r,
                    token_tile=n,
                    k_tile=k,
                    stages=stages,
                    use_pdl=pdl_enabled(),
                    weights_independent=weights_independent,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _PAIRED_PLANS[key] = compiled
    compiled(*operands, stream)


def _native_gemm(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    partials,
    out,
    *,
    role,
    g,
    h,
    r,
    p,
    weights_independent,
):
    middle_fp8 = (
        x.dtype == torch.float8_e4m3fn and r == 1024 and (16 < x.shape[0] <= 64)
    )
    original_n = (
        16
        if x.shape[0] <= 16
        else 32 if x.dtype != torch.float8_e4m3fn or x.shape[0] <= 32 else 64
    )
    n = 128 if x.shape[0] > 64 else original_n
    m = (64 if r <= 384 else 128) if role == "projection" else 128
    split = (
        8
        if role == "projection"
        and x.dtype == torch.float8_e4m3fn
        and (r == 1024)
        and (n > 64)
        else 16 if role == "projection" else 1
    )
    if (
        role == "projection"
        and x.dtype == torch.float8_e4m3fn
        and (r == 1024)
        and (n > 64)
    ):
        m = 64
    stages = 3 if x.dtype == torch.float8_e4m3fn else 2
    token_major_up = role == "mix" and n > 64
    up_hidden_tile = (
        64 if token_major_up and x.dtype == torch.float8_e4m3fn and (r == 1024) else 32
    )
    k = (
        256
        if role == "projection"
        and x.dtype == torch.float8_e4m3fn
        and (r == 1024)
        and (n > 64)
        else 128
    )
    if k == 256:
        stages = 4
    if x.dtype == torch.float8_e4m3fn and r == 1024 and (n <= 64):
        k = 512 if role == "projection" else 1024
        stages = (3 if n == 16 else 2) if role == "projection" else 1
    up_m64 = (
        role == "mix" and x.dtype == torch.float8_e4m3fn and (r == 1024) and (n == 64)
    )
    up_token_rows = 64 if up_m64 else 128
    epilogue_threads = 256 if up_m64 or n == 16 else 512
    if up_m64:
        token_major_up = True
        up_hidden_tile = 32
        k = 1024
        stages = 1
    if role == "projection" and original.shape[0] > 128:
        n = 256
        m = 128 if r == 1024 else 64
        split = 16
        k = 256 if x.dtype == torch.float8_e4m3fn and r == 1024 else 128
        stages = 4 if x.dtype == torch.float8_e4m3fn and r == 320 else 2
    if (
        role == "projection"
        and x.dtype == torch.float8_e4m3fn
        and (r == 1024)
        and (n == 16)
    ):
        m = 64
        split = 8
        k = 512
        stages = 4
        epilogue_threads = 256
    shared_up_scales = (
        token_major_up and up_hidden_tile == 64 and (x.dtype == torch.float8_e4m3fn)
    )
    if (
        role == "projection"
        and x.dtype == torch.float8_e4m3fn
        and (r == 1024)
        and (original.shape[0] > 128)
    ):
        n, m, split, k, stages = (
            128,
            128,
            8,
            128,
            6 if original.shape[0] <= 192 else 7,
        )
    interleave_tma = (
        role == "projection"
        and x.dtype == torch.float8_e4m3fn
        and (r == 1024)
        and (x.shape[0] > 16)
    )
    if role == "projection" and middle_fp8:
        m, split, epilogue_threads = (64, 8, 512)
        k, stages = (512, 4) if n == 32 else (256, 7)
    token_reuse = (
        role == "mix"
        and r == 1024
        and (x.dtype == torch.float8_e4m3fn)
        and (x.shape[0] > 128)
    )
    if up_m64:
        k, stages, up_hidden_tile, epilogue_threads = (128, 4, 32, 512)
        shared_up_scales = True
    up_m32 = (
        role == "mix" and x.dtype == torch.float8_e4m3fn and (r == 1024) and (n <= 32)
    )
    if up_m32:
        k, stages, up_hidden_tile, epilogue_threads = (128, 4, 32, 128)
        up_token_rows = 32
        token_major_up = True
        shared_up_scales = True
    weight_scale_vectorized = (
        role == "mix"
        and weight_scale.numel() > 1
        and (weight_scale.data_ptr() % 16 == 0)
    )
    key = (
        x.device.index,
        x.dtype,
        original.dtype,
        role,
        g,
        h,
        r,
        p,
        n,
        m,
        split,
        stages,
        k,
        pdl_enabled(),
        weights_independent,
        weight_scale.numel() > 1,
        token_major_up,
        up_token_rows,
        epilogue_threads,
        shared_up_scales,
        interleave_tma,
        up_hidden_tile,
        weight_scale_vectorized,
        token_reuse,
    )
    values = (x, weight, original, weight_scale, activation_scale, partials, out)
    operands = tuple(
        (
            from_dlpack(
                value.detach(), assumed_align=4 if value.ndim == 1 else 16
            ).mark_layout_dynamic(leading_dim=value.ndim - 1)
            for value in values
        )
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm the FP8 mixer tile before CUDA graph capture"
                    )
                kernel_type = (
                    Fp8GatedResidualM32WeightStationaryUp
                    if up_m32
                    else (
                        Fp8GatedResidualM64H32IndependentAUp
                        if up_m64
                        else (
                            Fp8GatedResidualTokenReuseUp
                            if token_reuse
                            else Fp8GatedResidualGemm
                        )
                    )
                )
                kernel = kernel_type(
                    role=role,
                    hc_count=g,
                    hidden_size=h,
                    lowrank=r,
                    projection_rows=p,
                    token_tile=n,
                    projection_tile=m,
                    k_tile=k,
                    stages=stages,
                    split_k=split,
                    use_pdl=pdl_enabled(),
                    weights_independent=weights_independent,
                    weight_scale_per_channel=weight_scale.numel() > 1,
                    token_major_up=token_major_up,
                    up_token_rows=up_token_rows,
                    epilogue_threads=epilogue_threads,
                    shared_up_scales=shared_up_scales,
                    interleave_tma=interleave_tma,
                    separate_operand_ready=False,
                    up_hidden_tile=up_hidden_tile,
                    weight_scale_vectorized=weight_scale_vectorized,
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _PLANS[key] = compiled
    compiled(*operands, stream)


_INPUT_QUANT_PLANS = {}


def _pdl_cuda_result(result):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"PDL kernel configuration failed with {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _set_pdl_carveout(compiled, device_index, *, use_pdl):
    if not use_pdl:
        return
    from cuda.bindings import driver

    compiled.to(device_index)
    library = driver.CUlibrary(int(compiled.jit_module.cuda_library[0]))
    handle = _pdl_cuda_result(driver.cuLibraryEnumerateKernels(1, library))[0]
    function = _pdl_cuda_result(driver.cuKernelGetFunction(handle))
    _pdl_cuda_result(
        driver.cuFuncSetAttribute(
            function,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            100,
        )
    )


def _quantize_input(x):
    codes = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    threads = 1024 if x.shape[1] == 32768 and pdl_enabled() else 256
    key = (x.device.index, x.dtype, x.shape[1], threads, pdl_enabled())
    operands = tuple(
        from_dlpack(
            t.detach(), assumed_align=4 if t.ndim == 1 else 16
        ).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (x, codes, scales)
    )
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    compiled = _INPUT_QUANT_PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _INPUT_QUANT_PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm input quantization before graph capture")
                kernel = Fp8GatedResidualInputQuantize(
                    columns=x.shape[1], block_threads=threads, use_pdl=pdl_enabled()
                )
                compiled = cute_ext.compile(kernel, *operands, stream)
                _set_pdl_carveout(compiled, x.device.index, use_pdl=pdl_enabled())
                _INPUT_QUANT_PLANS[key] = compiled
    compiled(*operands, stream)
    return codes, scales


def _execute(
    normalized,
    projection_weight,
    up_weight,
    g,
    h,
    r,
    p,
    projection_scale,
    weights_independent,
    projection_weight_scale,
    up_weight_scale,
    activation_quantization,
):
    rows = normalized.shape[0]
    if activation_quantization == "fp8":
        # Input and post-SiLU quantization share the same rounded scale and
        # multiplier, including the minimum scale for zero and tiny rows.
        if r == 1024 and 0 < rows <= 256 and (rows > 128 or pdl_enabled()):
            x, xs = _quantize_input(normalized)
        else:
            x, xs = fp8_quantize_per_token(normalized, enable_pdl=pdl_enabled())
        xs = xs.reshape(-1)
    else:
        x, xs = normalized, projection_weight_scale[:1]
    splits = 8 if activation_quantization == "fp8" and r == 1024 else 16
    partials = torch.empty(
        (splits, rows, p), device=normalized.device, dtype=torch.float32
    )
    out = torch.empty((rows, h), device=normalized.device, dtype=normalized.dtype)
    _gemm(
        x,
        projection_weight,
        normalized,
        projection_weight_scale,
        xs,
        partials,
        out,
        role="projection",
        g=g,
        h=h,
        r=r,
        p=p,
        weights_independent=weights_independent,
    )
    activated = torch.empty(
        (rows, r),
        device=normalized.device,
        dtype=(
            torch.float8_e4m3fn
            if activation_quantization == "fp8"
            else normalized.dtype
        ),
    )
    scales = torch.empty((rows,), device=normalized.device, dtype=torch.float32)
    inject = (
        torch.empty((rows, g), device=normalized.device, dtype=normalized.dtype)
        if p > r
        else None
    )
    _projection_epilogue(
        partials,
        xs,
        projection_weight_scale,
        activated,
        scales,
        out if inject is None else inject,
        r=r,
        p=p,
        g=g,
        projection_scale=projection_scale,
        input_fp8=x.dtype == torch.float8_e4m3fn,
        activation_fp8=activation_quantization == "fp8",
    )
    _gemm(
        activated,
        up_weight,
        normalized,
        up_weight_scale,
        scales,
        partials,
        out,
        role="mix",
        g=g,
        h=h,
        r=r,
        p=p,
        weights_independent=weights_independent,
    )
    return out, inject


if _AVAILABLE:

    @register_kernel(
        "residual",
        "hyperconnection_mix",
        name="cute_dsl_fp8_hyperconnection_mix",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 9),
        ),
        signatures={
            format_signature(
                normalized=dense_tensor_format(dtype),
                projection_weight=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32, granularity=down_scale
                    ),
                ),
                up_weight=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32, granularity=up_scale
                    ),
                ),
            )
            for dtype in (torch.bfloat16, torch.float16)
            for down_scale in ("tensor", "channel")
            for up_scale in ("tensor", "channel")
        },
        priority=Priority.SPECIALIZED,
        traits={"fused_shape_supported": frozenset({True})},
    )
    def cute_dsl_fp8_hyperconnection_mix(
        normalized,
        projection_weight,
        up_weight,
        hc_count,
        hidden_size,
        lowrank,
        projection_rows,
        projection_scale,
        weights_independent,
        *,
        projection_weight_scale,
        up_weight_scale,
        activation_quantization,
        packed_up_weight=None,
    ):
        """Run FP8-resident Down, scaled SiLU/inject, and fused Up/mix.

        The scale tensors multiply the FP8 codes, per tensor or output channel.
        Activation quantization is explicit: none retains the input's 16-bit
        arithmetic, and fp8 dynamically quantizes each input/SiLU token.
        Returns mixed values and optional inject logits in the input dtype.
        """
        if packed_up_weight is not None:
            raise ValueError("FP8 mixer consumes canonical, unpacked FP8 weights")
        from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc_shape

        if not supports_fused_hc_shape(hc_count, hidden_size, lowrank):
            raise ValueError("FP8 CuTe mixer requires validated model dimensions")
        pscale, uscale = projection_weight_scale.reshape(-1), up_weight_scale.reshape(
            -1
        )
        from tokenspeed_kernel.ops.tuning import is_warming_up

        if is_warming_up():
            key = (
                normalized.device.index,
                normalized.dtype,
                hc_count,
                hidden_size,
                lowrank,
                projection_rows,
                projection_scale,
                weights_independent,
                activation_quantization,
                pscale.numel(),
                uscale.numel(),
                uscale.numel() > 1 and uscale.data_ptr() % 16 == 0,
                pdl_enabled(),
            )
            if key not in _WARMED:
                sample = normalized.new_zeros((193, hc_count * hidden_size))
                warm_rows = (1, 17, 33, 65, 129, 193)
                if (
                    activation_quantization == "fp8"
                    and lowrank == 320
                    and pdl_enabled()
                ):
                    warm_rows = (1, 17, 33, 65, 113, 129, 193)
                for rows in warm_rows:
                    _execute(
                        sample[:rows],
                        projection_weight,
                        up_weight,
                        hc_count,
                        hidden_size,
                        lowrank,
                        projection_rows,
                        projection_scale,
                        weights_independent,
                        pscale,
                        uscale,
                        activation_quantization,
                    )
                _WARMED.add(key)
        return _execute(
            normalized,
            projection_weight,
            up_weight,
            hc_count,
            hidden_size,
            lowrank,
            projection_rows,
            projection_scale,
            weights_independent,
            pscale,
            uscale,
            activation_quantization,
        )
