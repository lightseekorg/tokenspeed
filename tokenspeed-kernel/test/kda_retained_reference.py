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

from dataclasses import replace

import torch
from tokenspeed_kernel.ops.attention import kda
from tokenspeed_kernel.ops.metadata import REQUEST_ENDED
from tokenspeed_kernel.platform import current_platform


def replay_length(slots, *, pad=0, device="cuda"):
    """A zeroed slot count: column 0 of a fresh INT32 [slots, 2 + pad] replay state.

    The kernels keep each slot's count and its pre-convolution ring state in
    adjacent columns (``replay_state``).
    """
    return torch.zeros(slots, 2 + pad, device=device, dtype=torch.int32)[:, :1]


def replay_state(history):
    """The INT32 [slots, 2] (count, ring state) replay state history.length heads."""
    length = history.length
    slots = length.shape[0]
    end = length.storage_offset() + (slots - 1) * length.stride(0) + 2
    if (
        length.stride(0) < 2
        or end * length.element_size() > length.untyped_storage().nbytes()
    ):
        raise ValueError("history.length must be the count column of a replay state")
    return length.as_strided((slots, 2), (length.stride(0), 1))


# Per history count and batch: the prepare's (applied count, ring head) rows,
# which the verify and the commit that follow it read.
_SNAPSHOTS: dict[tuple, torch.Tensor] = {}


def snapshot(history, batch):
    """The INT32 [batch, 2] snapshot the prepare of ``history`` writes."""
    key = (history.length.data_ptr(), batch, str(history.length.device))
    if key not in _SNAPSHOTS:
        _SNAPSHOTS[key] = torch.zeros(
            batch, 2, device=history.length.device, dtype=torch.int32
        )
    return _SNAPSHOTS[key]


def preconv_ring(history, rows, channels, *, width, dtype=torch.bfloat16):
    """A zeroed ``width + 3`` row pre-convolution ring per slot, headed by the snapshot."""
    ring = torch.zeros(
        history.length.shape[0],
        width + 3,
        channels,
        device=history.length.device,
        dtype=dtype,
    )
    return kda.KdaPreconvRing(ring, snapshot(history, rows)[:, 1])


def ring_taps(history, preconv, slot):
    """The slot's three ring rows before its published head, as [channels, 3] taps.

    None while the ring holds no rows (ring state 0).
    """
    state = int(replay_state(history)[slot, 1])
    if state == 0:
        return None
    rows = preconv.rows.shape[1]
    head = state - 1
    return preconv.rows[slot, [(head - 3 + i) % rows for i in range(3)]].T


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
    length = replay_length(batch, pad=3 if strided else 0)
    length[reads > 0] = accepted_history
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    history = kda.KdaRetainedHistory(correction, key, decay, length, live, slots)
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 4
    metadata[:, 2] = 2**32 + 13
    stamps = torch.full((batch, max(width, 7), 2), -1, device="cuda", dtype=torch.int64)
    records = kda.KdaRetainedRecords(
        preconv_ring(history, batch, channels, width=width, dtype=conv.dtype),
        history,
        metadata,
        stamps,
        starts,
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


# One layer's (descriptor, group index, scratch) per layout, kept for graph capture.
_PREPARED_LAYERS: dict[tuple, tuple[torch.Tensor, ...]] = {}


def prepare_history(
    state_pool,
    history,
    read_indices,
    query_start_loc,
    *,
    draft_token_num,
    conv=None,
    preconv=None,
):
    """Ready one layer's retained history and ring for its verify, as serving does.

    Serving prepares every layer in one launch before the forward; this runs
    the same batched prepare over a single layer. Rows whose count in
    history.length exceeds the history capacity minus the verify's window
    (draft_token_num) get their source state with accepted records applied
    in their live page, and the verify then applies none; other rows that
    move to a private live page get their source state copied there. A slot
    whose ring state is 0 gets ring rows 0..2 seeded from its source page's
    convolution taps (``conv``). The counts and ring states stay as they are;
    the snapshot (``snapshot``) receives each row's applied count and ring
    head. A verify without convolution passes no ``conv`` or ``preconv``, and
    zero-filled scratch ones stand in.
    """
    _, heads, dim, _ = state_pool.shape
    channels = 3 * heads * dim
    if conv is None:
        conv = _scratch(
            ("conv", state_pool.shape[0], channels, str(state_pool.device)),
            (state_pool.shape[0], channels, 3),
            state_pool.device,
        )
    if preconv is None:
        preconv = kda.KdaPreconvRing(
            _scratch(
                (
                    "ring",
                    history.length.shape[0],
                    draft_token_num,
                    channels,
                    str(state_pool.device),
                ),
                (history.length.shape[0], draft_token_num + 3, channels),
                state_pool.device,
            ),
            snapshot(history, read_indices.numel())[:, 1],
        )
    row = _descriptor(preconv.rows, conv, state_pool)
    # Uploaded on the first, eager call; a CUDA-graph capture reuses it.
    key = (*row, str(state_pool.device))
    if key not in _PREPARED_LAYERS:
        _PREPARED_LAYERS[key] = (
            torch.tensor([row], device=state_pool.device, dtype=torch.uint64),
            torch.zeros(1, device=state_pool.device, dtype=torch.int32),
        )
    descriptors, group_indices = _PREPARED_LAYERS[key]
    prepare = kda.resolve_kda_batched_retained_prepare(
        torch.bfloat16, num_heads=heads, head_dim=dim, recurrent_layout="v_major"
    )
    prepare(
        descriptors=descriptors,
        group_indices=group_indices,
        source_indices=read_indices[None],
        live_indices=history.live_state_indices[None],
        query_start_loc=query_start_loc,
        record_slots=history.record_slots,
        record_slot_count=history.length.shape[0],
        # One layer: the records gain a unit layer axis.
        retained_correction=history.correction[:, None],
        retained_key=history.key[:, None],
        retained_decay=history.decay[:, None],
        replay_state=replay_state(history),
        replay_snapshot=snapshot(history, read_indices.numel()),
        history_capacity=history.correction.shape[2],
        draft_token_num=draft_token_num,
        num_heads=heads,
        head_dim=dim,
        conv_state_dtype=conv.dtype,
        recurrent_state_dtype=state_pool.dtype,
    )


# Zero-filled convolution pools and rings for verifies that do not convolve.
_SCRATCH: dict[tuple, torch.Tensor] = {}


def _scratch(key, shape, device):
    if key not in _SCRATCH:
        _SCRATCH[key] = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    return _SCRATCH[key]


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
    """Compare native producers and raw-gate input against the real split path.

    ``run_verify(x, records, output_norm, *, prepare, **producers)`` runs the
    public verify, after the round's prepare unless ``prepare`` is False.
    """
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
        value.clone()
        for value in (x["state_pool"], x["conv_states"], replay_state(history))
    ]
    prefix_before = [
        field[:, :, :accepted].clone()
        for field in (history.correction, history.key, history.decay)
    ]
    raw_gate = torch.nn.functional.linear(x["f_a_out"], x["f_b_weight"])
    # The split producers' path: prepare (it seeds the ring), convolve from
    # the ring, then verify the convolved rows.
    prepare_history(
        x["state_pool"],
        reference_history,
        x["state_indices"],
        records.query_start_loc,
        draft_token_num=width,
        conv=x["conv_states"],
        preconv=expected.preconv,
    )
    convolved = kda.kda_verify_conv_update(
        x["mixed_qkv"],
        x["conv_weights"],
        x["state_indices"],
        live_indices=reference_history.live_state_indices,
        preconv=expected.preconv,
        record_slots=reference_history.record_slots,
        page_count=x["state_pool"].shape[0],
        num_heads=heads,
        head_dim=dim,
        draft_token_num=width,
        query_start_loc=records.query_start_loc,
        recurrent_layout="v_major",
    )
    reference_output = run_verify(
        x,
        replace(expected, preconv=None),
        None,
        prepare=False,
        g_raw=raw_gate,
        conv_qkv=convolved,
    )
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
    # Both paths retained every candidate's raw row in its slot's ring.
    assert torch.equal(records.preconv.rows, expected.preconv.rows)
    for row, page in enumerate(x["state_indices"].tolist()):
        if page > 0:
            slot = int(history.record_slots[row])
            head = int(records.preconv.heads[row])
            kept = [(head + t) % (width + 3) for t in range(width)]
            begin = int(records.query_start_loc[row])
            assert torch.equal(
                records.preconv.rows[slot, kept],
                x["mixed_qkv"][begin : begin + width].to(records.preconv.rows.dtype),
            )
    for actual, before in zip(
        (x["state_pool"], x["conv_states"], replay_state(history)), initial
    ):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)
    for field, before in zip(
        (history.correction, history.key, history.decay), prefix_before
    ):
        torch.testing.assert_close(field[:, :, :accepted], before, atol=0, rtol=0)
    candidates = [
        field.clone() for field in (history.correction, history.key, history.decay)
    ]
    publish(
        history,
        x["state_pool"],
        x["state_indices"],
        records.query_start_loc,
        torch.tensor([0, width], device="cuda", dtype=torch.int32),
        width=width,
        record_metadata=records.metadata,
        record_stamps=records.stamps,
        preconv=records.preconv,
        conv=x["conv_states"],
    )
    assert history.length[slots, 0].tolist() == [accepted, accepted + width]
    # The first round seeded the ring and started at row 3; the commit moved
    # the published head past the window.
    assert replay_state(history)[slots, 1].tolist() == [
        (3 + take) % (width + 3) + 1 for take in (0, width)
    ]
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
    records = case["records"]
    prepare_history(
        case["state"],
        records.history,
        case["reads"],
        records.query_start_loc,
        draft_token_num=case["width"],
        conv=case["conv"],
        preconv=records.preconv,
    )
    producers = {}
    if current_platform().is_nvidia:
        producers["conv_qkv"] = kda.kda_verify_conv_update(
            case["raw"],
            case["weights"],
            case["reads"],
            live_indices=records.history.live_state_indices,
            preconv=records.preconv,
            record_slots=records.history.record_slots,
            page_count=case["state"].shape[0],
            num_heads=case["heads"],
            head_dim=128,
            draft_token_num=case["width"],
            query_start_loc=case["records"].query_start_loc,
            recurrent_layout="v_major",
        )
        records = replace(records, preconv=None)
    return kda.kda_fused_paged_verify(
        case["raw"],
        case["weights"],
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
        retained_records=records,
        output_norm=None,
        g_raw=case["gate"],
        override=None,
        solution=None,
        **producers,
    )


def _descriptor(ring, conv, state):
    """One layer's 13-column retained commit descriptor."""
    return [
        ring.data_ptr(),
        conv.data_ptr(),
        conv.shape[0],
        ring.stride(1),
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        state.data_ptr(),
        *state.stride(),
        ring.stride(0),
    ]


def _run_commit(
    *,
    descriptors,
    group_indices,
    history,
    state,
    read_indices,
    query_start_loc,
    accepted,
    width,
    record_metadata,
    record_stamps,
    request_stopped,
    request_ends,
    conv_state_dtype,
):
    """Run the production batched retained commit over a single layer.

    It reads the snapshot the round's prepare (``prepare_history``) wrote.
    """
    heads, dim = state.shape[1], state.shape[-1]
    kernel = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=heads, head_dim=dim, recurrent_layout="v_major"
    )
    kernel(
        descriptors=descriptors,
        group_indices=group_indices,
        read_indices=read_indices[None],
        write_indices=history.live_state_indices[None],
        accepted_length=accepted,
        query_start_loc=query_start_loc,
        record_slots=history.record_slots,
        record_slot_count=history.length.shape[0],
        record_metadata=record_metadata,
        record_stamps=record_stamps[None],
        # One layer: the records gain a unit layer axis.
        retained_correction=history.correction[:, None],
        retained_key=history.key[:, None],
        retained_decay=history.decay[:, None],
        replay_state=replay_state(history),
        replay_snapshot=snapshot(history, accepted.numel()),
        request_stopped=request_stopped,
        request_ends=request_ends,
        draft_token_num=width,
        history_capacity=history.correction.shape[2],
        num_heads=heads,
        head_dim=dim,
        conv_width=4,
        conv_state_dtype=conv_state_dtype,
        recurrent_state_dtype=state.dtype,
    )


def commit(case, accepted, *, ended_slots):
    """Commit accepted prefixes; requests in ``ended_slots`` ended this forward.

    Returns each row's request end after the commit (``request_ends``).
    """
    records, conv, state = case["records"], case["conv"], case["state"]
    history = records.history
    request_stopped = torch.zeros(
        history.length.shape[0], device="cuda", dtype=torch.bool
    )
    request_stopped[list(ended_slots)] = True
    # Each row's report as the stop conditions leave it; the commit raises a
    # row whose state it cannot complete.
    slots = history.record_slots.long()
    request_ends = torch.where(
        (slots >= 0) & request_stopped[slots.clamp_min(0)],
        REQUEST_ENDED,
        0,
    ).to(torch.int32)
    _run_commit(
        descriptors=torch.tensor(
            [_descriptor(records.preconv.rows, conv, state)],
            device="cuda",
            dtype=torch.uint64,
        ),
        group_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        history=history,
        state=state,
        read_indices=case["reads"],
        query_start_loc=records.query_start_loc,
        accepted=accepted,
        width=case["width"],
        record_metadata=records.metadata,
        record_stamps=records.stamps,
        request_stopped=request_stopped,
        request_ends=request_ends,
        conv_state_dtype=conv.dtype,
    )
    return request_ends


# Uploads and scratch tensors per published layer layout, kept for graph capture.
_PUBLISHED_LAYERS: dict[tuple, tuple[torch.Tensor, ...]] = {}


def publish(
    history,
    state,
    read_indices,
    query_start_loc,
    accepted,
    *,
    width,
    record_metadata=None,
    record_stamps=None,
    preconv=None,
    conv=None,
):
    """Publish one layer's accepted counts through the production batched commit.

    No request ends here, so the commit writes no dense state. It follows the
    round's ``prepare_history``, whose snapshot it reads. A verify run on
    already-convolved q/k/v leaves no ring or convolution pool; the commit
    then advances the prepare's zero-filled scratch ones over the state
    pool's pages. A verify run without stamps gets each row's window stamped
    as a verify would, so the commit checks acceptance against the row's
    pages, slot, window and the history capacity only. Uploads and scratch
    tensors are made on the first, eager call; a CUDA-graph capture of a
    later call reuses them.
    """
    device = state.device
    heads, dim = state.shape[1], state.shape[-1]
    channels = 3 * heads * dim
    batch = accepted.numel()
    if record_metadata is None:
        record_metadata = torch.zeros(batch, 3, device=device, dtype=torch.int64)
        record_metadata[:, 0] = torch.arange(batch, device=device)
        record_stamps = torch.zeros(batch, width, 2, device=device, dtype=torch.int64)
        record_stamps[..., 0] = torch.arange(width, device=device)
    if conv is None:
        conv = _scratch(
            ("conv", state.shape[0], channels, str(device)),
            (state.shape[0], channels, 3),
            device,
        )
    ring = (
        _scratch(
            ("ring", history.length.shape[0], width, channels, str(device)),
            (history.length.shape[0], width + 3, channels),
            device,
        )
        if preconv is None
        else preconv.rows
    )
    layout = (
        state.data_ptr(),
        tuple(state.shape),
        state.stride(),
        history.length.data_ptr(),
        tuple(history.length.shape),
        history.length.stride(),
        batch,
        width,
        str(device),
        (ring.data_ptr(), ring.stride()),
        (conv.data_ptr(), tuple(conv.shape), conv.stride()),
    )
    if layout not in _PUBLISHED_LAYERS:
        _PUBLISHED_LAYERS[layout] = (
            torch.tensor(
                [_descriptor(ring, conv, state)], device=device, dtype=torch.uint64
            ),
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.zeros(history.length.shape[0], device=device, dtype=torch.bool),
            torch.zeros(batch, device=device, dtype=torch.int32),
        )
    descriptors, group_indices, request_stopped, request_ends = _PUBLISHED_LAYERS[
        layout
    ]
    _run_commit(
        descriptors=descriptors,
        group_indices=group_indices,
        history=history,
        state=state,
        read_indices=read_indices,
        query_start_loc=query_start_loc,
        accepted=accepted,
        width=width,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        request_stopped=request_stopped,
        request_ends=request_ends,
        conv_state_dtype=conv.dtype,
    )


def check_prefix(case, accepted, expected):
    """Check each live row's state and convolution window after its accepted prefix.

    The live page's convolution state and, while the request's ring holds
    rows, the three ring rows before its published head both equal the
    window after the accepted prefix, bit for bit.
    """
    output, states, windows = expected
    records = case["records"]
    history = records.history
    for row, page in enumerate(history.live_state_indices.tolist()):
        if int(case["reads"][row]) <= 0 or page <= 0:
            continue
        count = max(0, min(int(accepted[row]), len(states[row]) - 1))
        actual = endpoint(case["state"][page], history, row)
        torch.testing.assert_close(actual, states[row][count], atol=2e-4, rtol=1e-2)
        window = windows[row][count].to(case["conv"].dtype)
        torch.testing.assert_close(case["conv"][page], window, atol=0, rtol=0)
        taps = ring_taps(history, records.preconv, int(history.record_slots[row]))
        if taps is not None:
            torch.testing.assert_close(taps, window, atol=0, rtol=0)
