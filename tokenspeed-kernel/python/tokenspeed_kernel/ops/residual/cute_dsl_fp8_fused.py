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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import _fp8_mixer_signatures

try:
    import cutlass
    from cuda.bindings import driver
    from cuda.bindings.driver import CUstream
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack
    from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
        _has_full_up_tiles,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8_fused import (
        Fp8FusedGatedResidualKernel,
    )

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False


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


def _mixer_operands(x, xs, down, ds, up, us, workspace, out, inject):
    return tuple(
        _view(value)
        for value in (
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
    )


def _allocate_workspace(rows, r, p, splits, epoch_slots, device):
    return Workspace(
        torch.empty((splits, rows, p), device=device, dtype=torch.float32),
        torch.empty((rows, r), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((rows,), device=device, dtype=torch.float32),
        torch.zeros((epoch_slots,), device=device, dtype=torch.int64),
    )


def _standard_workspace(kernel, epoch_slots, device):
    return _allocate_workspace(
        kernel.n, kernel.r, kernel.p, kernel.splits, epoch_slots, device
    )


def _compiled_function(compiled, device):
    compiled.to(device=device.index)
    library = driver.CUlibrary(int(compiled.jit_module.cuda_library[0]))
    handle = _checked(driver.cuLibraryEnumerateKernels(1, library))[0]
    return _checked(driver.cuKernelGetFunction(handle))


def _allocate_outputs(x, g, h, r, p, dtype):
    rows = x.shape[0]
    out = torch.empty((rows, h), device=x.device, dtype=dtype)
    inject = torch.empty((rows, g), device=x.device, dtype=dtype) if p > r else None
    return out, inject


def _kernel_options(
    g, h, r, p, token_bucket, projection_scale, weights_independent, ds, us
):
    return dict(
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


def _launch_prequantized(
    inputs,
    outputs,
    *,
    kernel_type,
    kernel_options,
    key,
    cache,
    workspace_factory,
    occupancy_probe,
    launch_args,
    has_quant_groups,
    resource_tags,
):
    """Cache the cooperative grid and each stream's mixer workspace."""
    device = inputs[0].device
    out, inject = outputs
    stream = CUstream(int(torch.cuda.current_stream(device).cuda_stream))
    plans, workspaces, lock = cache
    stream_key = key + (int(stream),)
    with lock:
        plan = plans.get(key)
        workspace = workspaces.get(stream_key)
        if plan is None or workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm the prequantized FP8 bucket on this stream before capture"
                )
            if plan is None:
                # Compile with one safe cooperative block, then query real capacity.
                kernel = kernel_type(**kernel_options)
                scratch = workspace_factory(kernel, 2, device)
                operands = _mixer_operands(*inputs, scratch, out, inject)
                compiled = cute_ext.compile(
                    kernel, *operands, cutlass.Int32(1), *launch_args, stream
                )
                function = _compiled_function(compiled, device)
                if occupancy_probe is None:
                    shared_bytes = kernel.shared_bytes
                    blocks = _checked(
                        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                            function, kernel.threads, shared_bytes
                        )
                    )
                else:
                    shared_bytes, blocks = occupancy_probe(
                        compiled,
                        operands,
                        function,
                        kernel.threads,
                        device,
                        launch_args,
                    )
                sm_count = torch.cuda.get_device_properties(
                    device
                ).multi_processor_count
                capacity = blocks * sm_count
                if capacity <= 0:
                    raise RuntimeError(
                        "prequantized FP8 cooperative kernel has no resident capacity"
                    )
                desired_grid = max(kernel.down_tasks, kernel.up_tasks)
                if has_quant_groups and kernel.dedicated_quant_ctas:
                    desired_grid += kernel.quant_groups
                grid = min(capacity, desired_grid)
                attribute = driver.CUfunction_attribute
                resources = {
                    "registers": _checked(
                        driver.cuFuncGetAttribute(
                            attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function
                        )
                    ),
                    "local_bytes": _checked(
                        driver.cuFuncGetAttribute(
                            attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function
                        )
                    ),
                    "query_shared_bytes": shared_bytes,
                    "blocks_per_sm": blocks,
                    "sm_count": sm_count,
                    "grid": grid,
                    "threads": kernel.threads,
                    **resource_tags,
                }
                if has_quant_groups:
                    resources["resident_capacity"] = capacity
                plan = Plan(compiled, grid, kernel, resources)
                plans[key] = plan
            with torch.inference_mode(False):
                workspace = workspace_factory(plan.kernel, plan.grid + 1, device)
            workspaces[stream_key] = workspace
    operands = _mixer_operands(*inputs, workspace, out, inject)
    plan.compiled(*operands, cutlass.Int32(plan.grid), *launch_args, stream)
    return outputs


def _supports_sm103(device: torch.device) -> bool:
    return (
        _AVAILABLE
        and device.type == "cuda"
        and current_platform().is_nvidia
        and torch.cuda.get_device_capability(device) == (10, 3)
    )


def supports_packed_fp8_up_weight(device: torch.device, g: int, h: int, r: int) -> bool:
    """Check whether the device can use the wide FP8 Up weight layout."""
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc_shape

    return (
        _supports_sm103(device)
        and supports_fused_hc_shape(g, h, r)
        and _has_full_up_tiles(r)
    )


def prepare_fp8_up_weight(
    weight: torch.Tensor, g: int, h: int, r: int
) -> torch.Tensor | None:
    """Build contiguous K128 by 64-hidden weight tiles for the wide FP8 mixer.

    The caller owns the returned tensor and refreshes its contents in place
    whenever the canonical weight changes.
    """
    if not supports_packed_fp8_up_weight(weight.device, g, h, r):
        return None
    if (
        weight.shape != (g * h, r)
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        raise ValueError("Packed Up tiles require contiguous wide E4M3 weights")
    return (
        weight.view(torch.uint8)
        .view(g, h // 64, 64, r // 128, 128)
        .permute(3, 1, 0, 2, 4)
        .contiguous()
        .view_as(weight)
        .view(weight.dtype)
    )


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
    outputs = _allocate_outputs(x, g, h, r, p, output_dtype)
    if x.shape[0] == 0:
        return outputs
    options = _kernel_options(
        g, h, r, p, token_bucket, projection_scale, weights_independent, ds, us
    )
    key = (
        x.device.index,
        g,
        h,
        r,
        p,
        output_dtype,
        projection_scale,
        weights_independent,
        options["down_per_channel"],
        options["up_per_channel"],
        options["use_pdl"],
        token_bucket,
    )
    return _launch_prequantized(
        (x, xs, down, ds, up, us),
        outputs,
        kernel_type=Fp8FusedGatedResidualKernel,
        kernel_options=options,
        key=key,
        cache=(_PLANS, _WORKSPACES, _LOCK),
        workspace_factory=_standard_workspace,
        occupancy_probe=None,
        launch_args=(),
        has_quant_groups=False,
        resource_tags={},
    )


def _warmup(execute, x, xs, down, ds, up, us, **options):
    sample = (x[:1], xs.reshape(-1)[:1], down, ds, up, us)
    for bucket in (16, 32, 64, 128, 256):
        execute(*sample, token_bucket=bucket, **options)


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
        signatures=_fp8_mixer_signatures(prequantized=True),
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
        when they share that workspace. Wide SM103 injection mixers with a
        prepared Up weight use it in the 16-, 32-, 64-, and 128-token buckets.
        """
        if activation_quantization != "fp8":
            raise ValueError("prequantized FP8 fusion requires fp8 arithmetic")
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

        execute = _execute
        from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc_shape

        if not supports_fused_hc_shape(hc_count, hidden_size, lowrank):
            raise ValueError(
                "prequantized FP8 mixer requires validated model dimensions"
            )
        sm103 = _supports_sm103(normalized.device)
        if sm103 and _has_full_up_tiles(lowrank):
            from tokenspeed_kernel.ops.residual._cute_dsl.hc_fp8_wide import (
                _execute as execute_wide,
            )

            execute = execute_wide
            kwargs["packed_up_weight"] = packed_up_weight
        elif packed_up_weight is not None:
            raise ValueError("Packed E4M3 Up weights require the wide SM103 mixer")
        elif sm103:
            from tokenspeed_kernel.ops.residual._cute_dsl.hc_fp8_narrow import (
                _execute as execute_narrow,
            )

            execute = execute_narrow

        if is_warming_up():
            _warmup(execute, *values, **kwargs)
        rows = normalized.shape[0]
        bucket = min(256, max(16, 1 << max(0, rows - 1).bit_length()))
        return execute(*values, **kwargs, token_bucket=bucket)
