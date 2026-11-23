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
    fp8_quantize_per_token,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import _fp8_mixer_signatures

try:
    from cuda.bindings.driver import CUstream
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack
    from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
        _has_full_up_tiles,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
        Fp8GatedResidualGemm,
        Fp8GatedResidualInputQuantize,
        Fp8GatedResidualM32WeightStationaryUp,
        Fp8GatedResidualM64H32IndependentAUp,
        Fp8GatedResidualProjectionEpilogue,
        Fp8GatedResidualTokenReuseUp,
        W8GatedResidualBranchUp,
        W8GatedResidualDecodedRingUp,
        W8GatedResidualTmemProjection,
    )

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False

_PLANS = {}
_PLAN_LOCK = threading.Lock()
_WARMED = set()


def _launch_fp8_kernel(kernel_type, tensors, options):
    device = tensors[0].device
    key = (
        kernel_type,
        device.index,
        *(tensor.dtype for tensor in tensors),
        *options.values(),
    )
    operands = tuple(
        from_dlpack(
            tensor.detach(), assumed_align=4 if tensor.ndim == 1 else 16
        ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)
        for tensor in tensors
    )
    stream = CUstream(int(torch.cuda.current_stream(device).cuda_stream))
    compiled = _PLANS.get(key)
    if compiled is None:
        with _PLAN_LOCK:
            compiled = _PLANS.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        f"warm {kernel_type.__name__} before CUDA graph capture"
                    )
                compiled = cute_ext.compile(kernel_type(**options), *operands, stream)
                if kernel_type is Fp8GatedResidualInputQuantize:
                    _set_pdl_carveout(
                        compiled, device.index, use_pdl=options["use_pdl"]
                    )
                _PLANS[key] = compiled
    compiled(*operands, stream)


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
    full_up_tiles = _has_full_up_tiles(r)
    pdl = pdl_enabled()
    # Near-full narrow Up tiles prefer completed activation quantization.
    early_up_signal = (
        pdl
        and activation_fp8
        and not (not full_up_tiles and 112 < partial.shape[1] <= 128)
    )
    _launch_fp8_kernel(
        Fp8GatedResidualProjectionEpilogue,
        (partial, input_scale, weight_scale, activation, activation_scale, inject),
        dict(
            lowrank=r,
            projection_rows=p,
            hc_count=g,
            splits=partial.shape[0],
            projection_scale=projection_scale,
            input_fp8=input_fp8,
            activation_fp8=activation_fp8,
            weight_scale_per_channel=weight_scale.numel() > 1,
            use_pdl=pdl,
            block_threads=512 if full_up_tiles else 256,
            early_up_signal=early_up_signal,
            parallel_inject=pdl
            and activation_fp8
            and p > r
            and full_up_tiles
            and partial.shape[1] > 32,
        ),
    )


def _wide_w8_up(
    x,
    weight,
    original,
    weight_scale,
    activation_scale,
    out,
    *,
    g,
    h,
    r,
    weights_independent,
):
    options = dict(
        hc_count=g,
        hidden_size=h,
        lowrank=r,
        use_pdl=pdl_enabled(),
        weights_independent=weights_independent,
        weight_scale_per_channel=weight_scale.numel() > 1,
    )
    rows = x.shape[0]
    if rows <= 16:
        kernel_type = W8GatedResidualBranchUp
        options.update(token_tile=16, k_tile=128, stages=3)
    else:
        kernel_type = W8GatedResidualDecodedRingUp
        small_tile = rows <= 64
        paired = 128 < rows <= 256
        options.update(
            up_token_rows=64 if small_tile else 128,
            token_parts=2 if paired else 1,
            raw_stages=4 if paired else (6 if small_tile else 5),
            decoded_stages=3,
            input_stages=2 if paired else (4 if small_tile else 3),
        )
    _launch_fp8_kernel(
        kernel_type, (x, weight, original, weight_scale, activation_scale, out), options
    )


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
    native = x.dtype == torch.float8_e4m3fn
    projection = role == "projection"
    full_up_tiles = _has_full_up_tiles(r)
    if not native and not projection and full_up_tiles:
        return _wide_w8_up(
            x,
            weight,
            original,
            weight_scale,
            activation_scale,
            out,
            g=g,
            h=h,
            r=r,
            weights_independent=weights_independent,
        )

    rows = x.shape[0]
    n = 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128
    m = 64 if projection and not full_up_tiles else 128
    split = (8 if native and full_up_tiles else 16) if projection else 1
    token_major_up = not projection and n > 64
    up_hidden_tile = 64 if native and token_major_up and full_up_tiles else 32
    up_token_rows = 128
    shared_up_scales = native and token_major_up and full_up_tiles
    epilogue_threads = 256 if n == 16 else 512
    use_tmem_weights = not native and projection and full_up_tiles
    kernel_type = Fp8GatedResidualGemm
    extra_options = {}
    if native:
        k, stages = 128, 3
        if projection and full_up_tiles:
            if rows > 128:
                stages = 6 if rows <= 192 else 7
            elif n <= 32:
                m, k, stages = 64, 512, 4
            else:
                m, k, stages = 64, 256, 7 if n == 64 else 4
        elif projection and rows > 128:
            n, stages = 256, 4
        elif not projection and full_up_tiles:
            if n <= 64:
                token_major_up = shared_up_scales = True
                stages = 4
                if n <= 32:
                    kernel_type = Fp8GatedResidualM32WeightStationaryUp
                    up_token_rows, epilogue_threads = 32, 128
                else:
                    kernel_type = Fp8GatedResidualM64H32IndependentAUp
                    up_token_rows = 64
            elif rows > 128:
                kernel_type = Fp8GatedResidualTokenReuseUp
    else:
        k = 192 if token_major_up else 128
        stages = 2 if token_major_up else {16: 6, 32: 5, 64: 4, 128: 3}[n]
        if projection and rows > 128:
            n, stages = 256, 2
        if use_tmem_weights:
            kernel_type = W8GatedResidualTmemProjection
            stages = 7 if n <= 64 else 6 if n == 128 else 4
            extra_options = dict(
                rotate_k_tiles=n > 64,
                raw_stages=10 if n == 16 else 9 if n == 32 else 8 if n <= 128 else 6,
                input_stages={16: 16, 32: 10, 64: 6, 128: 3, 256: 2}[n],
            )

    per_channel = weight_scale.numel() > 1
    _launch_fp8_kernel(
        kernel_type,
        (x, weight, original, weight_scale, activation_scale, partials, out),
        dict(
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
            weight_scale_per_channel=per_channel,
            token_major_up=token_major_up,
            up_token_rows=up_token_rows,
            up_hidden_tile=up_hidden_tile,
            epilogue_threads=epilogue_threads,
            shared_up_scales=shared_up_scales,
            interleave_tma=native and projection and full_up_tiles and rows > 16,
            weight_scale_vectorized=token_major_up
            and not shared_up_scales
            and per_channel
            and weight_scale.data_ptr() % 16 == 0,
            **extra_options,
        ),
    )


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


def _quantize_input(x, *, block_threads: int):
    codes = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    _launch_fp8_kernel(
        Fp8GatedResidualInputQuantize,
        (x, codes, scales),
        dict(columns=x.shape[1], block_threads=block_threads, use_pdl=pdl_enabled()),
    )
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
        if _has_full_up_tiles(r) and 0 < rows <= 256 and (rows > 128 or pdl_enabled()):
            x, xs = _quantize_input(
                normalized, block_threads=1024 if pdl_enabled() else 256
            )
        else:
            x, xs = fp8_quantize_per_token(normalized, enable_pdl=pdl_enabled())
        xs = xs.reshape(-1)
    else:
        x, xs = normalized, projection_weight_scale[:1]
    splits = 8 if activation_quantization == "fp8" and _has_full_up_tiles(r) else 16
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
        signatures=_fp8_mixer_signatures(prequantized=False),
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
            full_up_tiles = _has_full_up_tiles(lowrank)
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
                not full_up_tiles
                and uscale.numel() > 1
                and uscale.data_ptr() % 16 == 0,
                pdl_enabled(),
            )
            if key not in _WARMED:
                sample = normalized.new_zeros((193, hc_count * hidden_size))
                warm_rows = (1, 17, 33, 65, 129, 193)
                if (
                    activation_quantization == "fp8"
                    and not full_up_tiles
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
