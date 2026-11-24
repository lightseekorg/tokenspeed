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

import torch
from tokenspeed_kernel.ops.residual.cute_dsl_fp8_fused import (
    Workspace,
    _allocate_outputs,
    _kernel_options,
    _launch_prequantized,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.sm103 import NarrowFp8Mixer

_PLANS = {}
_WORKSPACES = {}
_LOCK = threading.RLock()
# Phase ordinals must never reuse scratch initialized with round generations.
_EPOCH_PROTOCOL_VERSION = 8


def _allocate_partials(kernel, rows, projection_rows, device):
    if kernel.aligned_partial_rows:
        # The 16-token aligned planes use this buffer as raw storage.
        shape = (rows, kernel.splits, projection_rows)
        return torch.empty(shape, device=device, dtype=torch.float32).transpose(0, 1)
    shape = (kernel.splits, rows, projection_rows)
    return torch.empty(shape, device=device, dtype=torch.float32)


def _activation_workspace(rows, width, device, bundle_alpha):
    if bundle_alpha:
        codes_bytes = rows * width
        buffer = torch.empty(codes_bytes + rows * 4, device=device, dtype=torch.uint8)
        activation = buffer[:codes_bytes].view(torch.float8_e4m3fn).view(rows, width)
        alpha = buffer[codes_bytes:].view(torch.float32)
        return activation, alpha
    return (
        torch.empty((rows, width), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((rows,), device=device, dtype=torch.float32),
    )


def _allocate_activation(kernel, device):
    if kernel.stamped_activation:
        records = torch.zeros(
            (kernel.n, (kernel.activation_payload_records // kernel.n + 1) * 16),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        alpha = torch.empty((kernel.n,), dtype=torch.float32, device=device)
        return records, alpha
    return _activation_workspace(
        kernel.n,
        kernel.r,
        device,
        kernel.copied_alpha,
    )


def _allocate_workspace(kernel, epoch_slots, device):
    return Workspace(
        _allocate_partials(kernel, kernel.n, kernel.p, device),
        *_allocate_activation(kernel, device),
        torch.zeros(
            epoch_slots + kernel.extra_epoch_slots, device=device, dtype=torch.int64
        ),
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
    rows = x.shape[0]
    outputs = _allocate_outputs(x, g, h, r, p, output_dtype)
    if rows == 0:
        return outputs
    if token_bucket == 128 and rows >= 1 << 31:
        raise ValueError("Residual TMA coordinates must fit signed int32")
    options = _kernel_options(
        g, h, r, p, token_bucket, projection_scale, weights_independent, ds, us
    )
    key = (
        x.device.index,
        output_dtype,
        _EPOCH_PROTOCOL_VERSION,
        *options.values(),
    )
    return _launch_prequantized(
        (x, xs, down, ds, up, us),
        outputs,
        kernel_type=NarrowFp8Mixer,
        kernel_options=options,
        key=key,
        cache=(_PLANS, _WORKSPACES, _LOCK),
        workspace_factory=_allocate_workspace,
        occupancy_probe=None,
        launch_args=(),
        has_quant_groups=True,
        resource_tags={},
    )
