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

"""Independent PyTorch convolution/recurrence oracle for retained KDA tests."""

import torch
from tokenspeed_kernel.ops.attention import kda
from tokenspeed_kernel.platform import current_platform


def make_case(
    batch,
    width,
    *,
    heads=2,
    capacity=16,
    accepted_history=2,
    cross_page=False,
    ragged=False,
    strided=False,
    lower_bound=-5.0,
    dtype=torch.bfloat16,
):
    torch.manual_seed(71 + batch + width)
    dim, channels = 128, 3 * heads * 128
    pages = 2 * batch + 3
    reads = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    live = reads + batch if cross_page else reads.clone()
    sizes = [max(1, width - row % 3) if ragged else width for row in range(batch)]
    if ragged and batch > 1:
        sizes[-1] = 0
        reads[-1] = 0
        live[-1] = 0
    starts = torch.tensor(
        [0] + list(torch.tensor(sizes).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    tokens = sum(sizes)
    pad = 7 if strided else 0
    raw = (
        torch.randn(tokens, channels + pad, device="cuda", dtype=dtype)[:, :channels]
        * 0.2
    )
    conv = (
        torch.randn(pages, channels + pad, 3, device="cuda", dtype=dtype)[:, :channels]
        * 0.1
    )
    state = (
        torch.randn(pages, heads + (1 if strided else 0), dim, dim, device="cuda")[
            :, :heads
        ]
        * 0.05
    )
    # Multiply before slicing so the intended page/record strides are preserved.
    if strided:
        raw = (torch.randn(tokens, channels + pad, device="cuda", dtype=dtype) * 0.2)[
            :, :channels
        ]
        conv = (
            torch.randn(pages, channels + pad, 3, device="cuda", dtype=dtype) * 0.1
        )[:, :channels]
        state = (torch.randn(pages, heads + 1, dim, dim, device="cuda") * 0.05)[
            :, :heads
        ]
    shape = (batch, heads, capacity, dim * (2 if strided else 1))
    step = 2 if strided else 1
    correction = (torch.randn(shape, device="cuda") * 0.02)[..., ::step]
    key = (torch.randn(shape, device="cuda") * 0.02)[..., ::step]
    decay = (torch.rand(shape, device="cuda") * 0.2 + 0.7)[..., ::step]
    # One accepted count per request slot; slot i belongs to row i.
    length = torch.zeros(batch, 5 if strided else 1, device="cuda", dtype=torch.int32)[
        :, :1
    ]
    length[reads > 0] = accepted_history
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    history = kda.KdaRetainedHistory(correction, key, decay, length, live, slots)
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 4
    metadata[:, 2] = 2**32 + 13
    stamps = torch.full((batch, max(width, 7), 2), -1, device="cuda", dtype=torch.int64)
    records = kda.KdaRetainedRecords(
        torch.empty_like(raw), history, metadata, stamps, starts
    )
    return dict(
        raw=raw,
        conv=conv,
        state=state,
        reads=reads,
        records=records,
        weights=torch.randn(channels, 4, device="cuda", dtype=dtype) * 0.1,
        gate=torch.randn(tokens, heads * dim, device="cuda", dtype=dtype) * 0.2,
        beta=torch.randn(tokens, heads, device="cuda", dtype=dtype),
        a=torch.randn(heads, device="cuda") * 0.1,
        bias=torch.randn(heads * dim, device="cuda") * 0.1,
        heads=heads,
        width=width,
        lower_bound=lower_bound,
    )


def endpoint(base, history, row):
    """Apply row's slot records, up to the slot's accepted count, to base."""
    slot = int(history.record_slots[row])
    result = base.float().clone()
    for position in range(int(history.length[slot])):
        result = (
            result * history.decay[slot, :, position, None, :]
            + history.correction[slot, :, position, :, None]
            * history.key[slot, :, position, None, :]
        )
    return result


def check_fused_retained_append(
    x, records, expected, run_verify, *, width, raw_gate_only
):
    """Compare native producers and raw-gate input against the real split path."""
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        vllm_triton_kda_recoverssm_commit,
    )

    heads, dim = x["state_pool"].shape[1:3]
    history, reference_history = records.history, expected.history
    pages = x["state_indices"].long()
    slots = history.record_slots.long()
    accepted = 3
    generator = torch.Generator(device="cuda").manual_seed(707)
    for target, reference_field in zip(
        (history.correction, history.key, history.decay),
        (reference_history.correction, reference_history.key, reference_history.decay),
    ):
        prefix = torch.randn(
            pages.numel(), heads, accepted, dim, device="cuda", generator=generator
        )
        if target is history.correction:
            prefix *= 0.02
        elif target is history.key:
            prefix = torch.nn.functional.normalize(prefix, dim=-1)
        else:
            prefix = prefix.sigmoid() * 0.2 + 0.7
        target[slots, :, :accepted] = prefix
        reference_field.copy_(target)
    history.length[slots] = accepted
    reference_history.length.copy_(history.length)
    initial = [
        value.clone() for value in (x["state_pool"], x["conv_states"], history.length)
    ]
    prefix_before = [
        field[:, :, :accepted].clone()
        for field in (history.correction, history.key, history.decay)
    ]
    raw_gate = torch.nn.functional.linear(x["f_a_out"], x["f_b_weight"])
    convolved = kda.kda_verify_conv_update(
        x["mixed_qkv"],
        x["conv_weights"],
        x["conv_states"],
        x["state_indices"],
        num_heads=heads,
        head_dim=dim,
        draft_token_num=width,
        query_start_loc=records.query_start_loc,
        recurrent_layout="v_major",
        original_qkv=expected.original_qkv,
    )
    reference_output = run_verify(x, expected, None, g_raw=raw_gate, conv_qkv=convolved)
    inputs = x
    producers = {}
    if raw_gate_only:
        inputs = dict(x, f_a_out=None, f_b_weight=None)
        producers["g_raw"] = raw_gate
    name = "step" if width == 1 else "window"
    output = run_verify(
        inputs,
        records,
        None,
        override=f"cute_dsl_kda_fused_paged_verify_{name}",
        **producers,
    )
    torch.testing.assert_close(
        output.float(), reference_output.float(), atol=2e-4, rtol=1e-2
    )
    for field, reference_field, atol, rtol in (
        (history.correction, reference_history.correction, 5e-5, 2e-3),
        (history.key, reference_history.key, 1e-7, 2e-6),
        (history.decay, reference_history.decay, 2e-5, 2e-3),
    ):
        torch.testing.assert_close(field, reference_field, atol=atol, rtol=rtol)
    torch.testing.assert_close(records.original_qkv, x["mixed_qkv"], atol=0, rtol=0)
    for actual, before in zip(
        (x["state_pool"], x["conv_states"], history.length), initial
    ):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)
    for field, before in zip(
        (history.correction, history.key, history.decay), prefix_before
    ):
        torch.testing.assert_close(field[:, :, :accepted], before, atol=0, rtol=0)
    candidates = [
        field.clone() for field in (history.correction, history.key, history.decay)
    ]
    q, key, value = (
        part.view(1, -1, heads, dim) for part in convolved.chunk(3, dim=-1)
    )
    vllm_triton_kda_recoverssm_commit(
        q,
        key,
        value,
        checkpoint_state=x["state_pool"],
        retained_history=history,
        state_indices=x["state_indices"],
        query_start_loc=records.query_start_loc,
        num_accepted_tokens=torch.tensor([0, width], device="cuda", dtype=torch.int32),
        spec_query_len=width,
        record_metadata=records.metadata,
        record_stamps=records.stamps,
    )
    assert history.length[slots, 0].tolist() == [accepted, accepted + width]
    torch.testing.assert_close(x["state_pool"], initial[0], atol=0, rtol=0)
    for field, before in zip(
        (history.correction, history.key, history.decay), candidates
    ):
        torch.testing.assert_close(field, before, atol=0, rtol=0)


def reference(case):
    """Return packed outputs plus per-row states/taps after each prefix."""
    records, heads = case["records"], case["heads"]
    starts = records.query_start_loc.tolist()
    output = torch.zeros(case["raw"].shape[0], heads, 128, device="cuda")
    states, windows = [], []
    for row, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        page = int(case["reads"][row])
        running = endpoint(case["state"][page], records.history, row)
        taps = case["conv"][page].float().clone()
        row_states, row_windows = [running.clone()], [taps.clone()]
        for token in range(begin, end):
            joined = torch.cat((taps, case["raw"][token, :, None].float()), dim=-1)
            values = (
                torch.nn.functional.silu((joined * case["weights"].float()).sum(-1))
                .to(case["raw"].dtype)
                .float()
            )
            taps = joined[:, 1:]
            q, key, value = (part.view(heads, 128) for part in values.chunk(3))
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
            key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            gate = case["gate"][token].float().view(heads, 128) + case["bias"].view(
                heads, 128
            )
            a = case["a"].exp()[:, None]
            log_decay = (
                case["lower_bound"] * torch.sigmoid(a * gate)
                if case["lower_bound"] is not None
                else -a * torch.nn.functional.softplus(gate)
            )
            running = running * log_decay.exp()[:, None, :]
            correction = (value - (running * key[:, None, :]).sum(-1)) * torch.sigmoid(
                case["beta"][token].float()
            )[:, None]
            running = running + correction[:, :, None] * key[:, None, :]
            output[token] = (running * q[:, None, :]).sum(-1)
            row_states.append(running.clone())
            row_windows.append(taps.clone())
        states.append(row_states)
        windows.append(row_windows)
    return output, states, windows


def verify(case):
    producers = {}
    if current_platform().is_nvidia:
        producers["conv_qkv"] = kda.kda_verify_conv_update(
            case["raw"],
            case["weights"],
            case["conv"],
            case["reads"],
            num_heads=case["heads"],
            head_dim=128,
            draft_token_num=case["width"],
            query_start_loc=case["records"].query_start_loc,
            recurrent_layout="v_major",
        )
    return kda.kda_fused_paged_verify(
        case["raw"],
        case["weights"],
        case["conv"],
        None,
        None,
        case["beta"],
        case["a"],
        case["bias"],
        state_pool=case["state"],
        read_indices=case["reads"],
        num_heads=case["heads"],
        head_dim=128,
        draft_token_num=case["width"],
        lower_bound=case["lower_bound"],
        recurrent_layout="v_major",
        retained_records=case["records"],
        output_norm=None,
        g_raw=case["gate"],
        override=None,
        solution=None,
        **producers,
    )


def commit(case, accepted, *, ended_slots):
    """Commit accepted prefixes; requests in ``ended_slots`` ended this forward."""
    records, conv, state = case["records"], case["conv"], case["state"]
    history = records.history
    qkv = records.original_qkv
    descriptors = torch.tensor(
        [
            [
                qkv.data_ptr(),
                conv.data_ptr(),
                history.length.data_ptr(),
                conv.shape[0],
                qkv.stride(0),
                conv.stride(0),
                history.length.stride(0),
                conv.stride(1),
                conv.stride(2),
                state.data_ptr(),
                *state.stride(),
            ]
        ],
        device="cuda",
        dtype=torch.uint64,
    )
    request_stopped = torch.zeros(
        history.length.shape[0], device="cuda", dtype=torch.bool
    )
    request_stopped[list(ended_slots)] = True
    kernel = kda.resolve_kda_batched_retained_commit(
        qkv.dtype, num_heads=case["heads"], head_dim=128, recurrent_layout="v_major"
    )
    kernel(
        descriptors=descriptors,
        group_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        read_indices=case["reads"][None],
        write_indices=history.live_state_indices[None],
        accepted_length=accepted,
        query_start_loc=records.query_start_loc,
        record_slots=history.record_slots,
        record_slot_count=history.length.shape[0],
        record_metadata=records.metadata,
        record_stamps=records.stamps[None],
        # One layer: the records gain a unit layer axis.
        retained_correction=history.correction[:, None],
        retained_key=history.key[:, None],
        retained_decay=history.decay[:, None],
        request_stopped=request_stopped,
        draft_token_num=case["width"],
        history_capacity=history.correction.shape[2],
        num_heads=case["heads"],
        head_dim=128,
        conv_width=4,
        activation_dtype=qkv.dtype,
        conv_state_dtype=conv.dtype,
        recurrent_state_dtype=state.dtype,
    )


def check_prefix(case, accepted, expected):
    output, states, windows = expected
    history = case["records"].history
    for row, page in enumerate(history.live_state_indices.tolist()):
        if int(case["reads"][row]) <= 0 or page <= 0:
            continue
        count = max(0, min(int(accepted[row]), len(states[row]) - 1))
        actual = endpoint(case["state"][page], history, row)
        torch.testing.assert_close(actual, states[row][count], atol=2e-4, rtol=1e-2)
        torch.testing.assert_close(
            case["conv"][page],
            windows[row][count].to(case["conv"].dtype),
            atol=0,
            rtol=0,
        )
