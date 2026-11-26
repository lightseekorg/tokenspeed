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

import cutlass
import torch
from cuda.bindings import driver
from tokenspeed_kernel.ops.residual.cute_dsl_fp8_fused import (
    _allocate_outputs,
    _checked,
    _kernel_options,
    _launch_prequantized,
    _standard_workspace,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.sm103 import WideFp8Mixer

_PLANS = {}
_WORKSPACES = {}
_LOCK = threading.RLock()
# Phase ordinals must never reuse scratch initialized with round generations.
_EPOCH_PROTOCOL_VERSION = 2


def _query_actual_resources(compiled, operands, function, threads, device, launch_args):
    """Query occupancy from the exact captured cooperative launch allocation."""
    with torch.cuda.device(device):
        probe_stream = _checked(
            driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
        )
        graph = None
        try:
            _checked(
                driver.cuStreamBeginCapture(
                    probe_stream,
                    driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            )
            try:
                compiled(
                    *operands,
                    cutlass.Int32(1),
                    *launch_args,
                    probe_stream,
                )
            finally:
                graph = _checked(driver.cuStreamEndCapture(probe_stream))
            _, count = _checked(driver.cuGraphGetNodes(graph, 0))
            nodes, _ = _checked(driver.cuGraphGetNodes(graph, count))
            kernels = [
                node
                for node in nodes
                if _checked(driver.cuGraphNodeGetType(node))
                == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
            ]
            if len(kernels) != 1:
                raise RuntimeError("Expected one native cooperative compute launch")
            params = _checked(driver.cuGraphKernelNodeGetParams(kernels[0]))
            if int(params.func) != int(function) or (
                params.blockDimX,
                params.blockDimY,
                params.blockDimZ,
            ) != (threads, 1, 1):
                raise RuntimeError(
                    "Captured native function or threads differ from the plan"
                )
            cooperative = _checked(
                driver.cuGraphKernelNodeGetAttribute(
                    kernels[0],
                    driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                )
            )
            if cooperative.cooperative != 1:
                raise RuntimeError(
                    "Native phase publication requires cooperative residency"
                )
            blocks = _checked(
                driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    params.func, params.blockDimX, params.sharedMemBytes
                )
            )
            return params.sharedMemBytes, blocks
        finally:
            if graph is not None:
                _checked(driver.cuGraphDestroy(graph))
            _checked(driver.cuStreamDestroy(probe_stream))


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
    packed_up_weight,
):
    has_injection = p == r + g
    up_weight_is_packed = (
        packed_up_weight is not None and has_injection and token_bucket <= 128
    )
    if up_weight_is_packed:
        up = packed_up_weight
    outputs = _allocate_outputs(x, g, h, r, p, output_dtype)
    if x.shape[0] == 0:
        return outputs
    options = _kernel_options(
        g, h, r, p, token_bucket, projection_scale, weights_independent, ds, us
    )
    options["up_weight_is_packed"] = up_weight_is_packed
    tma_l2_ids = (cutlass.Int32(0),) * 4
    weight_priority_ids = (cutlass.Int32(1),) * 2
    key = (
        x.device.index,
        output_dtype,
        _EPOCH_PROTOCOL_VERSION,
        *options.values(),
    )
    return _launch_prequantized(
        (x, xs, down, ds, up, us),
        outputs,
        kernel_type=WideFp8Mixer,
        kernel_options=options,
        key=key,
        cache=(_PLANS, _WORKSPACES, _LOCK),
        workspace_factory=_standard_workspace,
        occupancy_probe=_query_actual_resources,
        launch_args=tma_l2_ids + weight_priority_ids,
        has_quant_groups=False,
        resource_tags=dict(
            token_bucket=token_bucket, prepared_up_weight=up_weight_is_packed
        ),
    )
