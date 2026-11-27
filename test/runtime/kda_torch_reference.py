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

"""Layered convolution, retained recurrence and prefill ownership references."""

import torch


def kda_convolution_reference(raw, conv, *, conv_weights, bias, prefill):
    """Compute convolution/windows independently with the producer's precision.

    Prefill multiplies input-dtype taps/weights before adding their rounded
    products to an FP32 accumulator seeded with bias. Verify promotes both
    operands to FP32, uses FMA, and adds bias after the taps.
    """
    conv = conv.float().clone()
    weight = conv_weights.float()
    outputs, windows = [], []
    for token in range(raw.shape[1]):
        current = raw[:, token].float()
        acc = torch.zeros_like(current)
        if prefill:
            if bias is not None:
                acc = acc + bias.float()
            for tap in range(weight.shape[1] - 1):
                product = (conv[:, :, tap] * weight[:, tap]).to(raw.dtype)
                acc = acc + product.float()
            acc = acc + (current * weight[:, -1]).to(raw.dtype).float()
        else:
            for tap in range(weight.shape[1] - 1):
                acc = torch.addcmul(acc, conv[:, :, tap], weight[:, tap])
            acc = torch.addcmul(acc, current, weight[:, -1])
            if bias is not None:
                acc = acc + bias.float()
        outputs.append(torch.nn.functional.silu(acc).to(raw.dtype))
        conv = torch.cat((conv[:, :, 1:], current[..., None]), dim=-1)
        windows.append(conv.to(raw.dtype))
    return torch.stack(outputs, 1), torch.stack(windows, 1)


def checked_convolution(raw, conv, *, conv_weights, bias, prefill):
    """Check producer rounding independently before testing downstream state.

    SiLU implementations can land on opposite sides of a BF16 midpoint.
    Keep the existing convolution tolerance, then use the actual rounded
    producer values for strict recurrence and cache ownership comparisons.
    """
    from tokenspeed_kernel.ops.attention.kda import kda_verify_conv_update

    batch, width, channels = raw.shape
    pool = torch.cat((torch.zeros_like(conv[:1]), conv)).clone()
    pages = torch.arange(1, batch + 1, device=raw.device, dtype=torch.int32)
    bounds = torch.arange(batch + 1, device=raw.device, dtype=torch.int32) * width
    rows = raw.reshape(-1, channels).clone()
    if prefill:
        from tokenspeed_kernel.ops.attention.gdn.triton import (
            CAUSAL_CONV1D_BLOCK_M,
            build_causal_conv1d_prefill_metadata,
        )
        from tokenspeed_kernel.ops.conv import causal_conv1d_fn

        actual = causal_conv1d_fn(
            rows.T,
            conv_weights,
            bias,
            conv_states=pool,
            query_start_loc=bounds,
            cache_indices=pages,
            has_initial_state=torch.ones(batch, device=raw.device, dtype=torch.bool),
            activation="silu",
            prefill_metadata=build_causal_conv1d_prefill_metadata(
                bounds,
                torch.full((batch,), width, dtype=torch.int32),
                CAUSAL_CONV1D_BLOCK_M,
            ),
        ).T.reshape_as(raw)
    else:
        actual = kda_verify_conv_update(
            rows,
            conv_weights,
            pool,
            pages,
            num_heads=channels // (3 * 128),
            head_dim=128,
            draft_token_num=width,
            recurrent_layout="v_major",
            bias=bias,
            query_start_loc=bounds,
        ).reshape_as(raw)
    expected, windows = kda_convolution_reference(
        raw, conv, conv_weights=conv_weights, bias=bias, prefill=prefill
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-4, rtol=8e-3)
    if prefill:
        torch.testing.assert_close(pool[1:], windows[:, -1], atol=0, rtol=0)
    return actual, windows


def kda_prefill_reference(
    raw,
    conv,
    state,
    beta_raw,
    *,
    conv_weights,
    bias,
    gate,
    A_log,
    dt_bias,
    lower_bound,
    solution,
):
    """Isolate prefill page ownership using the selected unchanged scan.

    Callers gather logical source states independently with PyTorch. The
    convolution and raw windows are independently checked above. This reference
    tests materialization/store/reset and graph routing, not the scan's internal
    BF16 chunk arithmetic; standalone kernel tests cover that numerical contract.
    """
    from tokenspeed_kernel.ops.attention.kda import kda_paged_prefill

    batch, width, _ = raw.shape
    heads, dim = state.shape[1:3]
    convolved, windows = checked_convolution(
        raw, conv, conv_weights=conv_weights, bias=bias, prefill=True
    )
    q, k, v = [
        part.reshape(1, batch * width, heads, dim) for part in convolved.chunk(3, -1)
    ]
    bounds_cpu = torch.arange(batch + 1, dtype=torch.int64) * width
    result = kda_paged_prefill(
        q,
        k,
        v,
        gate.reshape_as(q),
        beta_raw.reshape(1, batch * width, heads),
        A_log,
        dt_bias,
        initial_state=state.clone(),
        cu_seqlens=bounds_cpu.to(raw.device),
        cu_seqlens_cpu=bounds_cpu,
        capacity=None,
        lower_bound=lower_bound,
        solution=solution,
        recurrent_layout="v_major",
    )
    return (
        result.out.reshape(batch, width, heads, dim),
        result.final_state,
        windows[:, -1],
    )


def kda_torch_reference(
    raw,
    conv,
    state,
    beta_raw,
    *,
    conv_weights,
    bias,
    gate,
    A_log,
    dt_bias,
    lower_bound,
    convolved=None,
):
    """Return all outputs, recurrent states and convolution windows.

    Inputs use batch/token/channel rows and FP32 V-major recurrent state.
    The producer rounds activated convolution values to the input dtype.
    With convolved=None every operation is independent PyTorch. Callers may
    supply independently checked producer values to isolate strict recurrence
    from BF16 convolution rounding; no production recurrence is used.
    """
    batch, width, channels = raw.shape
    heads, dim = state.shape[1:3]
    reference_qkv, windows = kda_convolution_reference(
        raw, conv, conv_weights=conv_weights, bias=bias, prefill=False
    )
    qkv_rows = reference_qkv if convolved is None else convolved
    state = state.float().clone()
    gate = gate.float().view(batch, width, heads, dim) + dt_bias.view(heads, dim)
    scale = A_log.exp()[None, None, :, None]
    log_decay = (
        -scale * torch.nn.functional.softplus(gate)
        if lower_bound is None
        else lower_bound * torch.sigmoid(scale * gate)
    )
    decay = log_decay.exp()
    beta = beta_raw.float().sigmoid().view(batch, width, heads)
    outputs, states = [], []
    for token in range(width):
        qkv = qkv_rows[:, token].float()
        query, key, value = (part.view(batch, heads, dim) for part in qkv.chunk(3, -1))
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        state = state * decay[:, token, :, None, :]
        correction = (value - (state * key[:, :, None, :]).sum(-1)) * beta[
            :, token, :, None
        ]
        state = state + correction[..., None] * key[:, :, None, :]
        outputs.append(
            ((state * query[:, :, None, :]).sum(-1) * dim**-0.5).to(raw.dtype)
        )
        states.append(state.clone())
    return torch.stack(outputs, 1), torch.stack(states, 1), windows
