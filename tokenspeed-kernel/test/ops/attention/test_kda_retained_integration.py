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

"""Retained serving contract: public verify, gathered endpoints, batched commit."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import kda
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _history(pages=8, heads=2, capacity=8, dim=128):
    # Deliberately interleaved arena views: no page/record axis is contiguous.
    shape = (pages, heads, capacity, dim * 2)
    correction = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.03)[
        ..., ::2
    ]
    key = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.03)[..., ::2]
    decay = (torch.rand(shape, device="cuda", dtype=torch.float32) * 0.2 + 0.7)[
        ..., ::2
    ]
    length = torch.zeros(pages, 7, device="cuda", dtype=torch.int32)[:, 2:3]
    live = torch.tensor([3, 4, 5, 0], device="cuda", dtype=torch.int32)
    return kda.KdaRetainedHistory(correction, key, decay, length, live)


def _endpoint(base, history, page, count):
    state = base.clone()
    for token in range(count):
        state = (
            state * history.decay[page, :, token, None, :]
            + history.correction[page, :, token, :, None]
            * history.key[page, :, token, None, :]
        )
    return state


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_materialize_shared_source_and_reset_private_lengths(layout):
    history = _history()
    history.length[1] = 3
    history.length[2] = 1
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    base = torch.randn(8, 2, 128, 128, device="cuda")
    gathered = base[reads.long()].clone()
    gathered[-1].zero_()
    expected = gathered.clone()
    for row, page, count in [(0, 1, 3), (1, 1, 3), (2, 2, 1)]:
        expected[row] = _endpoint(gathered[row], history, page, count)
    if layout == "k_major":
        gathered = gathered.transpose(-1, -2).contiguous()
        expected = expected.transpose(-1, -2)
    before = [
        x.clone()
        for x in (base, history.correction, history.key, history.decay, history.length)
    ]
    kda.kda_materialize_retained_state(
        gathered, history=history, source_indices=reads, recurrent_layout=layout
    )
    torch.testing.assert_close(gathered, expected, atol=2e-6, rtol=2e-6)
    for value, original in zip(
        (base, history.correction, history.key, history.decay, history.length), before
    ):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    history.length[3:6] = 4
    history.length[0] = 7
    kda.kda_reset_retained_length(
        history.length, write_indices=history.live_state_indices
    )
    assert history.length[:, 0].tolist() == [7, 3, 1, 0, 0, 0, 0, 0]


def _commit_case(window, *, stale=False):
    heads, dim, channels, layers = 2, 128, 768, 3
    reads = torch.tensor(
        [[1, 1, 2, 0], [5, 1, 2, -1]], device="cuda", dtype=torch.int32
    )
    writes = torch.tensor(
        [[3, 4, 2, 0], [3, 4, 2, 0]], device="cuda", dtype=torch.int32
    )
    groups = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    starts = torch.tensor(
        [
            0,
            window,
            window + max(1, window - 1),
            2 * window + max(1, window - 1),
            2 * window + max(1, window - 1),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    accepted = torch.tensor(
        [0, max(1, window - 1), window, 0], device="cuda", dtype=torch.int32
    )
    metadata = torch.tensor(
        [[2, 9, 5], [0, 8, 11], [1, 7, 2], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((layers, 3, 7, 2), -1, device="cuda", dtype=torch.int64)
    for row, (slot, generation, start) in enumerate(metadata[:3].tolist()):
        for token in range(window):
            stamps[:, slot, (start + token) % 7] = torch.tensor(
                [start + token, generation], device="cuda"
            )
    if stale:
        stamps[1, 0, 11 % 7, 1] = -1
    originals, convs, lengths, descriptors, expected_convs, expected_lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for layer in range(layers):
        pages = 7 + layer
        qkv = torch.randn(
            int(starts[-1]), channels + layer * 4, device="cuda", dtype=torch.bfloat16
        )[:, :channels]
        conv = torch.randn(
            pages, channels + layer * 3, 8, device="cuda", dtype=torch.float32
        )[:, :channels, 1:7:2]
        length = torch.full((pages, 5 + layer), 2, device="cuda", dtype=torch.int32)[
            :, 1:2
        ]
        expected_conv, expected_length = conv.clone(), length.clone()
        group = int(groups[layer])
        for row in range(3):
            if stale and layer == 1 and row == 1:
                continue
            source, target = int(reads[group, row]), int(writes[group, row])
            take, start = int(accepted[row]), int(starts[row])
            expected_conv[target] = torch.cat(
                (conv[source], qkv[start : start + take].float().T), dim=-1
            )[:, -3:]
            expected_length[target] += take
        descriptors.append(
            [
                qkv.data_ptr(),
                conv.data_ptr(),
                length.data_ptr(),
                pages,
                qkv.stride(0),
                conv.stride(0),
                length.stride(0),
                conv.stride(1),
                conv.stride(2),
            ]
        )
        originals.append(qkv)
        convs.append(conv)
        lengths.append(length)
        expected_convs.append(expected_conv)
        expected_lengths.append(expected_length)
    kwargs = dict(
        descriptors=torch.tensor(descriptors, device="cuda", dtype=torch.uint64),
        group_indices=groups,
        read_indices=reads,
        write_indices=writes,
        accepted_length=accepted,
        query_start_loc=starts,
        record_metadata=metadata,
        record_stamps=stamps,
        draft_token_num=window,
        history_capacity=8,
        num_heads=heads,
        head_dim=dim,
        conv_width=4,
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.float32,
    )
    return kwargs, originals, convs, lengths, expected_convs, expected_lengths


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("stale", [False, True])
def test_batched_retained_commit_packed_groups_and_stamps(window, stale):
    args, keepalive, convs, lengths, expected_convs, expected_lengths = _commit_case(
        window, stale=stale
    )
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    commit(**args)
    for conv, expected in zip(convs, expected_convs):
        torch.testing.assert_close(conv, expected, atol=0, rtol=0)
    for length, expected in zip(lengths, expected_lengths):
        torch.testing.assert_close(length, expected, atol=0, rtol=0)


@pytest.mark.parametrize("window", [1, 4])
def test_public_retained_verify_keeps_shared_source_and_materializes_endpoint(window):
    torch.manual_seed(42)
    history = _history()
    history.length[1] = 2
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    total = 2 * window + max(1, window - 1)
    starts = torch.tensor(
        [0, window, window + max(1, window - 1), total, total],
        device="cuda",
        dtype=torch.int32,
    )
    metadata = torch.tensor(
        [[0, 3, 5], [1, 4, 7], [2, 5, 9], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((4, 7, 2), -1, device="cuda", dtype=torch.int64)
    raw = torch.randn(total, 768, device="cuda", dtype=torch.bfloat16) * 0.2
    original = torch.empty_like(raw)
    conv = torch.randn(8, 768, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(total, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(total, 2, device="cuda", dtype=torch.bfloat16)
    a = torch.zeros(2, device="cuda")
    bias = torch.zeros(256, device="cuda")
    state = torch.randn(8, 2, 128, 128, device="cuda") * 0.1
    saved = state[1].clone()
    length_before = history.length[1].clone()
    records = kda.KdaRetainedRecords(
        history=history,
        original_qkv=original,
        metadata=metadata,
        stamps=stamps,
        query_start_loc=starts,
    )
    producer = {}
    # The split producer consumes the same packed boundaries as verification.
    if current_platform().is_nvidia:
        producer["conv_qkv"] = kda.kda_verify_conv_update(
            raw,
            weights,
            conv,
            reads,
            num_heads=2,
            head_dim=128,
            query_start_loc=starts,
            draft_token_num=window,
            original_qkv=None,
            bias=None,
            recurrent_layout="v_major",
        )
    expected_out = torch.empty(total, 2, 128, device="cuda", dtype=torch.float32)
    expected_final = []
    expected_u, expected_key, expected_decay = (
        torch.empty_like(expected_out) for _ in range(3)
    )
    for row, (begin, end) in enumerate(zip(starts[:3].tolist(), starts[1:4].tolist())):
        page = int(reads[row])
        running = _endpoint(state[page], history, page, int(history.length[page]))
        taps = conv[page].float()
        for token in range(begin, end):
            window_taps = torch.cat([taps, raw[token, :, None].float()], dim=-1)
            convolved = (
                torch.nn.functional.silu((window_taps * weights.float()).sum(-1))
                .to(raw.dtype)
                .float()
            )
            taps = window_taps[:, 1:]
            q, key, value = (x.reshape(2, 128) for x in convolved.chunk(3))
            q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / (128**0.5)
            key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            decay = (-5.0 * torch.sigmoid(gate[token].float().view(2, 128))).exp()
            running = running * decay[:, None, :]
            correction = (value - (running * key[:, None, :]).sum(-1)) * torch.sigmoid(
                beta[token].float()
            )[:, None]
            running = running + correction[:, :, None] * key[:, None, :]
            expected_u[token], expected_key[token], expected_decay[token] = (
                correction,
                key,
                decay,
            )
            expected_out[token] = (running * q[:, None, :]).sum(-1)
        expected_final.append(running)
    out = kda.kda_fused_paged_verify(
        raw,
        weights,
        conv,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        beta,
        a,
        bias,
        state_pool=state,
        read_indices=reads,
        num_heads=2,
        head_dim=128,
        draft_token_num=window,
        lower_bound=-5.0,
        recurrent_layout="v_major",
        retained_records=records,
        output_norm=None,
        g_raw=gate,
        **producer,
        override=None,
        solution=None
    )
    assert out is not None
    torch.testing.assert_close(original, raw, atol=0, rtol=0)
    torch.testing.assert_close(state[1], saved, atol=0, rtol=0)
    torch.testing.assert_close(history.length[1], length_before, atol=0, rtol=0)
    # The private base must be the complete logical source endpoint, not the old base.
    expected_base = _endpoint(saved, history, 1, 2)
    torch.testing.assert_close(state[3], expected_base, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(state[4], expected_base, atol=2e-6, rtol=2e-6)
    assert history.length[3:6, 0].tolist() == [0, 0, 0]
    torch.testing.assert_close(
        out.reshape(total, 2, 128).float(), expected_out, atol=2e-4, rtol=1e-2
    )
    for row in range(3):
        target = int(history.live_state_indices[row])
        begin, end = int(starts[row]), int(starts[row + 1])
        torch.testing.assert_close(
            history.correction[target, :, : end - begin],
            expected_u[begin:end].transpose(0, 1),
            atol=2e-4,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            history.key[target, :, : end - begin],
            expected_key[begin:end].transpose(0, 1),
            atol=1e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            history.decay[target, :, : end - begin],
            expected_decay[begin:end].transpose(0, 1),
            atol=1e-6,
            rtol=1e-6,
        )
        history.length[target] = end - begin
    endpoint = state[history.live_state_indices[:3].long()].clone()
    kda.kda_materialize_retained_state(
        endpoint,
        history=history,
        source_indices=history.live_state_indices[:3],
        recurrent_layout="v_major",
    )
    torch.testing.assert_close(
        endpoint, torch.stack(expected_final), atol=2e-4, rtol=1e-2
    )


def test_packed_verify_convolution_uses_request_boundaries():
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA split producer; AMD verifies packed convolution in fusion")
    torch.manual_seed(7)
    raw = torch.randn(7, 768, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    conv = torch.randn(6, 768, 3, device="cuda", dtype=torch.bfloat16)
    reads = torch.tensor([1, 2, 3, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4, 5, 7, 7], device="cuda", dtype=torch.int32)
    expected = torch.empty_like(raw)
    for row, (begin, end) in enumerate(zip(starts[:-1].tolist(), starts[1:].tolist())):
        history = conv[int(reads[row])].float()
        for token in range(begin, end):
            taps = torch.cat([history, raw[token, :, None].float()], dim=-1)
            expected[token] = torch.nn.functional.silu((taps * weights.float()).sum(-1))
            history = taps[:, 1:]
    output = kda.kda_verify_conv_update(
        raw,
        weights,
        conv,
        reads,
        num_heads=2,
        head_dim=128,
        draft_token_num=4,
        query_start_loc=starts,
        recurrent_layout="v_major",
        bias=None,
        original_qkv=None,
    )
    torch.testing.assert_close(output, expected, atol=0.003, rtol=0.005)


def test_retained_commit_graph_refresh_and_layer_subset():
    args, keepalive, convs, lengths, expected_convs, expected_lengths = _commit_case(4)
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    # Noncontiguous stamp layer views must retain their parent stride.
    subset = dict(
        args,
        descriptors=args["descriptors"][::2].contiguous(),
        group_indices=args["group_indices"][::2].contiguous(),
        record_stamps=args["record_stamps"][::2],
    )
    before_conv = [x.clone() for x in convs]
    before_length = [x.clone() for x in lengths]
    commit(**subset)
    for actual, initial in zip(convs, before_conv):
        actual.copy_(initial)
    for actual, initial in zip(lengths, before_length):
        actual.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit(**subset)
    for actual, initial in zip(convs, before_conv):
        actual.copy_(initial)
    for actual, initial in zip(lengths, before_length):
        actual.copy_(initial)
    subset["accepted_length"].zero_()
    graph.replay()
    for layer in (0, 2):
        for row in range(3):
            group = int(subset["group_indices"][layer // 2])
            source = int(args["read_indices"][group, row])
            target = int(args["write_indices"][group, row])
            torch.testing.assert_close(
                convs[layer][target], before_conv[layer][source], atol=0, rtol=0
            )
        torch.testing.assert_close(lengths[layer], before_length[layer], atol=0, rtol=0)
    torch.testing.assert_close(convs[1], before_conv[1], atol=0, rtol=0)
    torch.testing.assert_close(lengths[1], before_length[1], atol=0, rtol=0)


def test_retained_commit_does_not_specialize_dynamic_metadata():
    from tokenspeed_kernel.ops.attention.kda._retained import _kernels

    args, keepalive, convs, lengths, expected_convs, expected_lengths = _commit_case(4)
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    args["accepted_length"].zero_()
    commit(**args)
    kernels, _ = _kernels()
    shorter = dict(
        args,
        read_indices=args["read_indices"][:, :2],
        write_indices=args["write_indices"][:, :2],
        accepted_length=args["accepted_length"][:2],
        query_start_loc=args["query_start_loc"][:3],
        record_metadata=args["record_metadata"][:2],
        record_stamps=torch.full((3, 5, 11, 2), -1, device="cuda", dtype=torch.int64),
    )
    with assert_no_triton_compile(
        kernels.retained_batched_conv_kernel, kernels.retained_batched_publish_kernel
    ):
        commit(**shorter)


def test_retained_capability_requires_both_explicit_contracts():
    assert kda.kda_retained_supported(
        torch.bfloat16,
        recurrent_layout="v_major",
        num_heads=2,
        head_dim=128,
        conv_kernel_size=4,
    )
    for dtype, layout, heads, dim, width in [
        (torch.float32, "v_major", 2, 128, 4),
        (torch.bfloat16, "k_major", 2, 128, 4),
        (torch.bfloat16, "v_major", 2, 128, 3),
        (torch.bfloat16, "v_major", 0, 128, 4),
    ]:
        assert not kda.kda_retained_supported(
            dtype,
            recurrent_layout=layout,
            num_heads=heads,
            head_dim=dim,
            conv_kernel_size=width,
        )
    assert (
        kda.kda_fused_paged_verify_uses_split_producers(
            torch.bfloat16,
            recurrent_layout="v_major",
            num_heads=2,
            head_dim=128,
        )
        == current_platform().is_nvidia
    )


def test_invalid_checkpoint_verify_raises_without_mutation():
    """A page-count mismatch must fail before touching any page or record."""
    history = _history(heads=1, dim=128)
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 1, 2, 3, 3], device="cuda", dtype=torch.int32)
    raw = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    conv = torch.randn(8, 384, 3, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(7, 1, 128, 128, device="cuda")
    original = torch.full_like(raw, 7)
    metadata = torch.tensor(
        [[0, 1, 0], [1, 1, 0], [2, 1, 0], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((3, 8, 2), -1, device="cuda", dtype=torch.int64)
    records = kda.KdaRetainedRecords(original, history, metadata, stamps, starts)
    mutable = (
        raw,
        conv,
        state,
        original,
        history.correction,
        history.key,
        history.decay,
        history.length,
        stamps,
    )
    before = [value.clone() for value in mutable]
    with pytest.raises(ValueError):
        kda.kda_fused_paged_verify(
            raw,
            torch.ones(384, 4, device="cuda", dtype=torch.bfloat16),
            conv,
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            torch.zeros(3, 1, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, device="cuda"),
            torch.zeros(128, device="cuda"),
            state_pool=state,
            read_indices=reads,
            num_heads=1,
            head_dim=128,
            draft_token_num=1,
            recurrent_layout="v_major",
            retained_records=records,
            output_norm=None,
            g_raw=torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16),
            conv_qkv=raw if current_platform().is_nvidia else None,
            lower_bound=-5.0,
            override=None,
            solution=None,
        )
    for actual, expected in zip(mutable, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
