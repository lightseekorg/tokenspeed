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

"""Register native single-launch HC mixing with prequantized FP8 operands."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format

try:
    import cutlass
    from cuda.bindings import driver
    from cuda.bindings.driver import CUstream
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8_fused import (
        Fp8FusedGatedResidualKernel,
    )

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False
from tokenspeed_kernel.platform import pdl_enabled


@dataclass
class Workspace:
    partials: torch.Tensor
    activation: torch.Tensor
    alpha: torch.Tensor
    epochs: torch.Tensor


@dataclass
class Plan:
    compiled: object
    grid: int
    kernel: object
    resources: dict


_PLANS = {}
_WORKSPACES = {}
_LOCK = threading.RLock()


def _checked(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA driver error {error}")
    return values[0] if len(values) == 1 else values


def _view(value):
    return from_dlpack(
        value.detach(), assumed_align=4 if value.ndim == 1 else 16
    ).mark_layout_dynamic(leading_dim=value.ndim - 1)


def _execute(
    x,
    xs,
    down,
    ds,
    up,
    us,
    *,
    g,
    h,
    r,
    p,
    output_dtype,
    projection_scale,
    weights_independent,
    token_bucket,
):
    rows = x.shape[0]
    out = torch.empty((rows, h), device=x.device, dtype=output_dtype)
    inject = (
        torch.empty((rows, g), device=x.device, dtype=output_dtype) if p > r else None
    )
    if rows == 0:
        return out, inject
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    key = (
        x.device.index,
        g,
        h,
        r,
        p,
        output_dtype,
        projection_scale,
        weights_independent,
        ds.numel() > 1,
        us.numel() > 1,
        pdl_enabled(),
        token_bucket,
    )
    stream_key = key + (int(stream),)
    with _LOCK:
        plan = _PLANS.get(key)
        workspace = _WORKSPACES.get(stream_key)
        if plan is None or workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm the prequantized FP8 bucket on this stream before capture"
                )
            kernel = Fp8FusedGatedResidualKernel(
                g=g,
                h=h,
                r=r,
                p=p,
                token_bucket=token_bucket,
                projection_scale=projection_scale,
                use_pdl=pdl_enabled(),
                weights_independent=weights_independent,
                down_per_channel=ds.numel() > 1,
                up_per_channel=us.numel() > 1,
            )
            if plan is None:
                # Compile with one safe cooperative block, then query real capacity.
                scratch = Workspace(
                    torch.empty(
                        (kernel.splits, token_bucket, p),
                        device=x.device,
                        dtype=torch.float32,
                    ),
                    torch.empty(
                        (token_bucket, r), device=x.device, dtype=torch.float8_e4m3fn
                    ),
                    torch.empty((token_bucket,), device=x.device, dtype=torch.float32),
                    torch.zeros((2,), device=x.device, dtype=torch.int64),
                )
                values = (
                    x,
                    xs.reshape(-1),
                    down,
                    ds.reshape(-1),
                    up,
                    us.reshape(-1),
                    scratch.partials,
                    scratch.activation,
                    scratch.alpha,
                    scratch.epochs,
                    out,
                    out if inject is None else inject,
                )
                operands = tuple(_view(v) for v in values)
                compiled = cute_ext.compile(kernel, *operands, cutlass.Int32(1), stream)
                compiled.to(device=x.device.index)
                library = driver.CUlibrary(int(compiled.jit_module.cuda_library[0]))
                handle = _checked(driver.cuLibraryEnumerateKernels(1, library))[0]
                function = _checked(driver.cuKernelGetFunction(handle))
                blocks = _checked(
                    driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        function, kernel.threads, kernel.shared_bytes
                    )
                )
                properties = torch.cuda.get_device_properties(x.device)
                capacity = blocks * properties.multi_processor_count
                if capacity <= 0:
                    raise RuntimeError(
                        "prequantized FP8 cooperative kernel has no resident capacity"
                    )
                grid = min(capacity, max(kernel.down_tasks, kernel.up_tasks))
                resources = {
                    "registers": _checked(
                        driver.cuFuncGetAttribute(
                            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            function,
                        )
                    ),
                    "local_bytes": _checked(
                        driver.cuFuncGetAttribute(
                            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            function,
                        )
                    ),
                    "query_shared_bytes": kernel.shared_bytes,
                    "blocks_per_sm": blocks,
                    "sm_count": properties.multi_processor_count,
                    "grid": grid,
                    "threads": kernel.threads,
                }
                plan = Plan(compiled, grid, kernel, resources)
                _PLANS[key] = plan
            with torch.inference_mode(False):
                workspace = Workspace(
                    torch.empty(
                        (plan.kernel.splits, token_bucket, p),
                        device=x.device,
                        dtype=torch.float32,
                    ),
                    torch.empty(
                        (token_bucket, r), device=x.device, dtype=torch.float8_e4m3fn
                    ),
                    torch.empty((token_bucket,), device=x.device, dtype=torch.float32),
                    torch.zeros((plan.grid + 1,), device=x.device, dtype=torch.int64),
                )
            _WORKSPACES[stream_key] = workspace
    values = (
        x,
        xs.reshape(-1),
        down,
        ds.reshape(-1),
        up,
        us.reshape(-1),
        workspace.partials,
        workspace.activation,
        workspace.alpha,
        workspace.epochs,
        out,
        out if inject is None else inject,
    )
    operands = tuple(_view(v) for v in values)
    plan.compiled(*operands, cutlass.Int32(plan.grid), stream)
    return out, inject


def _warmup(
    x,
    xs,
    down,
    ds,
    up,
    us,
    *,
    g,
    h,
    r,
    p,
    output_dtype,
    projection_scale,
    weights_independent,
):
    for bucket in (16, 32, 64, 128, 256):
        _execute(
            x[:1],
            xs.reshape(-1)[:1],
            down,
            ds,
            up,
            us,
            g=g,
            h=h,
            r=r,
            p=p,
            output_dtype=output_dtype,
            projection_scale=projection_scale,
            weights_independent=weights_independent,
            token_bucket=bucket,
        )


if _AVAILABLE:

    @register_kernel(
        "residual",
        "hyperconnection_mix",
        name="cute_dsl_fp8_fused_hyperconnection_mix",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 9),
        ),
        signatures={
            format_signature(
                normalized=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(storage_dtype=torch.float32, granularity=xscale),
                ),
                projection_weight=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(storage_dtype=torch.float32, granularity=dscale),
                ),
                up_weight=tensor_format(
                    "scaled-fp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(storage_dtype=torch.float32, granularity=uscale),
                ),
            )
            for xscale in ("tensor", "token")
            for dscale in ("tensor", "channel")
            for uscale in ("tensor", "channel")
        },
        priority=Priority.SPECIALIZED,
        traits={
            "fused_shape_supported": frozenset({True}),
            "contiguous": frozenset({True}),
            "activation_quantization": frozenset({"fp8"}),
        },
    )
    def cute_dsl_fp8_fused_hyperconnection_mix(
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
        normalized_scale,
        output_dtype,
        projection_weight_scale,
        up_weight_scale,
        activation_quantization,
        packed_up_weight=None,
    ):
        """Run Down/SiLU/FP8 quantization/Up/mix in one cooperative launch.

        Inputs are canonical E4M3 codes and FP32 scales. ``normalized_scale``
        contains one value or one per input row; weight scales are tensor or
        output-channel scales. ``output_dtype`` explicitly selects BF16 or FP16
        output and the post-SiLU rounding edge. Mix uses FP8 input codes times
        their input scale, without a separate 16-bit residual. Returns mixed
        values and optional injection logits. Workspace initializes once during
        warmup and is reused on its execution stream; calls require ordering
        when they share that workspace.
        """
        if activation_quantization != "fp8" or packed_up_weight is not None:
            raise ValueError(
                "prequantized FP8 fusion requires fp8 arithmetic and unpacked weights"
            )
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "prequantized FP8 fusion requires BF16 or FP16 output_dtype"
            )
        values = (
            normalized,
            normalized_scale.reshape(-1),
            projection_weight,
            projection_weight_scale.reshape(-1),
            up_weight,
            up_weight_scale.reshape(-1),
        )
        kwargs = dict(
            g=hc_count,
            h=hidden_size,
            r=lowrank,
            p=projection_rows,
            output_dtype=output_dtype,
            projection_scale=projection_scale,
            weights_independent=weights_independent,
        )
        from tokenspeed_kernel.ops.tuning import is_warming_up

        if is_warming_up():
            _warmup(*values, **kwargs)
        rows = normalized.shape[0]
        bucket = min(256, max(16, 1 << max(0, rows - 1).bit_length()))
        return _execute(*values, **kwargs, token_bucket=bucket)
