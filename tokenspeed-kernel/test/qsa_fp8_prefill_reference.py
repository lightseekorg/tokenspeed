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

"""Independent reference for explicit FP8 approximation; caller sets IEEE matmul."""

import math

import torch


def reference(q, k, v, slots, positions, ids, *, attention_scale, k_scale, v_scale):
    kv_heads = k.shape[1]
    queries = q[ids].float().reshape(len(ids), kv_heads, 12, 256)
    selected, pos = slots[ids], positions[ids]
    visible = (pos + 1).clamp_min(0)
    complete = (visible // 4).clamp_max(512) * 4
    length = complete + visible % 4
    logical = torch.arange(2051, device=q.device)[None, :]
    columns = torch.where(
        logical < complete[:, None], logical, 2048 + logical - complete[:, None]
    ).clamp_max(2050)
    physical = selected.gather(1, columns.long()).long()
    valid = (logical < length[:, None]) & (physical > 0)
    physical = torch.where(valid, physical, torch.zeros_like(physical))
    operands = []
    for cache in (k, v):
        x = cache[physical].float().permute(0, 2, 1, 3).contiguous()
        operands.append(torch.where(valid[:, None, :, None], x, torch.zeros_like(x)))
    keys, values = operands
    maximum = queries.abs().amax(-1, keepdim=True)
    # Use correctly rounded FP32 division; Torch scalar division can instead
    # multiply by a rounded reciprocal and change FP8 midpoint decisions.
    scale = torch.where(
        maximum > 0, (maximum.double() / 448.0).float(), torch.ones_like(maximum)
    )
    quantized = (
        (queries.double() / scale.double())
        .float()
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .float()
    )
    scores = (
        (quantized @ keys.transpose(-1, -2))
        * (
            scale
            * (
                float(k_scale)
                * float(
                    torch.tensor(
                        math.log2(math.e) * attention_scale, dtype=torch.float32
                    )
                )
            )
        )
    ).masked_fill(~valid[:, None, None, :], -torch.inf)
    running_max = torch.full(scores.shape[:-1], -torch.inf, device=q.device)
    denominator = torch.zeros_like(running_max)
    numerator = torch.zeros((*running_max.shape, 256), device=q.device)
    for start in range(0, 2051, 128):
        tile = scores[..., start : start + 128]
        updated = torch.maximum(running_max, tile.amax(-1))
        safe = torch.where(torch.isneginf(updated), torch.zeros_like(updated), updated)
        alpha = torch.exp2(running_max - safe)
        probability = torch.exp2(tile - safe[..., None])
        denominator = denominator * alpha + probability.sum(-1)
        rounded = (probability * 448.0).to(torch.float8_e4m3fn).float()
        numerator = (
            numerator * alpha[..., None] + rounded @ values[..., start : start + 128, :]
        )
        running_max = updated
    return (
        numerator * (v_scale / (denominator.clamp_min(1e-30)[..., None] * 448.0))
    ).reshape(len(ids), kv_heads * 12, 256)
