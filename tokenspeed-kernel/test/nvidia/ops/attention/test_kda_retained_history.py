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

"""Retained KDA history against an independent eager recurrence."""

from __future__ import annotations

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)

import tokenspeed_kernel.ops.attention.kda as kda_ops
from kda_retained_reference import prepare_history, publish
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    vllm_triton_kda_recoverssm_verify,
)


@pytest.mark.parametrize(
    "field,shape,dtype",
    [
        ("decay", (3, 2, 7, 64), torch.float32),
        ("key", (3, 2, 8, 64), torch.bfloat16),
        ("length", (3, 2), torch.int32),
        # One count per request slot, not per cache page.
        ("length", (4, 1), torch.int32),
        ("record_slots", (1,), torch.int32),
        ("record_slots", (2,), torch.float32),
    ],
)
def test_retained_history_rejects_invalid_storage_on_construction(field, shape, dtype):
    fields = dict(
        correction=torch.empty(3, 2, 8, 32),
        key=torch.empty(3, 2, 8, 64),
        decay=torch.empty(3, 2, 8, 64),
        length=torch.empty(3, 1, dtype=torch.int32),
        live_state_indices=torch.zeros(2, dtype=torch.int32),
        record_slots=torch.zeros(2, dtype=torch.int64),
    )
    fields[field] = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError, match="retained"):
        kda_ops.KdaRetainedHistory(**fields)


def _eager(k, v, gate, beta, a, bias, initial, lower_bound):
    # The kernel adds epsilon inside the squared norm.
    key = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    g = gate.float() + bias
    log_decay = (
        lower_bound * torch.sigmoid(a.exp()[:, None] * g)
        if lower_bound is not None
        else -a.exp()[:, None] * torch.nn.functional.softplus(g)
    )
    current = initial.clone()
    states, corrections = [], []
    for t in range(k.shape[0]):
        current = current * log_decay[t].exp()[:, None, :]
        u = v[t].float() - torch.einsum("hvk,hk->hv", current, key[t])
        u = u * beta[t].float().sigmoid()[:, None]
        current = current + u[:, :, None] * key[t, :, None, :]
        states.append(current.clone())
        corrections.append(u)
    return states, torch.stack(corrections), key, log_decay.exp()


def _reconstruct(base, correction, key, decay, length):
    result = base.float().clone()
    for t in range(length):
        result = result * decay[:, t, None, :]
        result = result + correction[:, t, :, None] * key[:, t, None, :]
    return result


@pytest.mark.parametrize("implementation", ["vllm_triton", "cute_dsl"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize(
    "capacity,width,initial_length,accepted_rounds",
    [(8, 4, 2, (2, 4)), (16, 1, 5, (0, 1)), (32, 4, 28, (4, 0)), (8, 4, 6, (2, 4))],
)
def test_retained_history_keeps_accepted_prefix_and_publishes_only_length(
    implementation,
    state_dtype,
    capacity,
    width,
    initial_length,
    accepted_rounds,
    lower_bound,
    strided,
):
    """Catch ignored accepted history, premature publication, and copied records."""
    history_type = kda_ops.KdaRetainedHistory
    verify_kernel = vllm_triton_kda_recoverssm_verify
    if implementation == "cute_dsl":
        from tokenspeed_kernel.platform import ArchVersion, current_platform

        if (
            not ArchVersion(10, 0)
            <= current_platform().arch_version
            <= ArchVersion(10, 3)
        ):
            pytest.skip("SM100/103 required")
        pytest.importorskip("cutlass")
        pytest.importorskip("quack")
        from tokenspeed_kernel.ops.attention.kda.cute_dsl import (
            cute_dsl_kda_recoverssm_verify,
        )

        verify_kernel = cute_dsl_kda_recoverssm_verify
    torch.manual_seed(871)
    heads, dim = 2, 64
    state = (torch.randn(3, heads, dim, dim, device="cuda") * 0.02).to(state_dtype)
    if strided:
        state_view = torch.empty(
            6, heads * 2, dim * 2, dim * 2, device="cuda", dtype=state_dtype
        )[::2, ::2, ::2, ::2]
        state_view.copy_(state)
        state = state_view
    base_before = state.clone()
    u = torch.randn(3, heads, capacity, dim, device="cuda") * 0.02
    key = torch.nn.functional.normalize(torch.randn_like(u), dim=-1)
    decay = torch.rand_like(key) * 0.2 + 0.7
    length = torch.full((3, 1), initial_length, device="cuda", dtype=torch.int32)
    if strided:
        fields = []
        for value in (u, key, decay):
            view = torch.empty(6, heads * 2, capacity * 2, dim * 2, device="cuda")[
                ::2, ::2, ::2, ::2
            ]
            view.copy_(value)
            fields.append(view)
        u, key, decay = fields
        length_view = torch.empty(6, 3, device="cuda", dtype=torch.int32)[::2, 1:2]
        length_view.copy_(length)
        length = length_view
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    history = history_type(u, key, decay, length, indices, torch.zeros_like(indices))
    current = _reconstruct(state[1], u[0], key[0], decay[0], initial_length)
    starts = torch.tensor([0, width], device="cuda", dtype=torch.int32)
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    for accepted in accepted_rounds:
        q, k, v, gate = [
            torch.randn(1, width, heads, dim, device="cuda").bfloat16()
            for _ in range(4)
        ]
        beta = torch.randn(1, width, heads, device="cuda").bfloat16()
        out = torch.empty_like(v)
        old_length = int(length[0, 0])
        consolidated = old_length + width > capacity
        if consolidated:
            materialized = _reconstruct(
                base_before[1], u[0], key[0], decay[0], old_length
            )
            old_length = 0
        # Serving readies the history before every verify round.
        prepare_history(state, history, indices, starts, draft_token_num=width)
        verify_kernel(
            q,
            k,
            v,
            gate,
            beta,
            a,
            bias,
            checkpoint_state=state,
            retained_history=history,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=lower_bound,
            out=out,
            record_metadata=None,
            record_stamps=None,
        )
        assert int(length[0, 0]) == old_length
        if consolidated:
            if state_dtype == torch.bfloat16:
                # Round the FP32 error bounds to BF16. Adjacent BF16 values
                # are valid only if those bounds span their midpoint.
                epsilon = 2e-6 + 2e-5 * materialized.abs()
                lower = (materialized - epsilon).bfloat16().float()
                upper = (materialized + epsilon).bfloat16().float()
                actual = state[1].float()
                assert bool(((actual >= lower) & (actual <= upper)).all())
            else:
                torch.testing.assert_close(state[1], materialized, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(
                state[[0, 2]], base_before[[0, 2]], atol=0, rtol=0
            )
            current = state[1].float().clone()
        else:
            torch.testing.assert_close(state, base_before, atol=0, rtol=0)
        base_before.copy_(state)
        expected_states, expected_u, expected_k, expected_decay = _eager(
            k[0], v[0], gate[0], beta[0], a, bias, current, lower_bound
        )
        normalized_q = (
            q[0].float()
            * torch.rsqrt(q[0].float().square().sum(-1, keepdim=True) + 1e-6)
            * dim**-0.5
        )
        expected_out = torch.stack(
            [
                torch.einsum("hvk,hk->hv", s, normalized_q[t])
                for t, s in enumerate(expected_states)
            ]
        )
        torch.testing.assert_close(out[0].float(), expected_out, atol=2e-3, rtol=5e-3)
        torch.testing.assert_close(
            u[0, :, old_length : old_length + width],
            expected_u.transpose(0, 1),
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            key[0, :, old_length : old_length + width],
            expected_k.transpose(0, 1),
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            decay[0, :, old_length : old_length + width],
            expected_decay.transpose(0, 1),
            atol=2e-6,
            rtol=2e-5,
        )
        records_before = (u.clone(), key.clone(), decay.clone())
        publish(
            history,
            state,
            indices,
            starts,
            torch.tensor([accepted], device="cuda", dtype=torch.int32),
            width=width,
        )
        assert int(length[0, 0]) == old_length + accepted
        torch.testing.assert_close(state, base_before, atol=0, rtol=0)
        for actual, before in zip((u, key, decay), records_before):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)
        current = expected_states[accepted - 1] if accepted else current
        torch.testing.assert_close(
            _reconstruct(state[1], u[0], key[0], decay[0], old_length + accepted),
            current,
            atol=2e-6,
            rtol=2e-5,
        )


@pytest.mark.parametrize("captured", [False, True])
def test_shared_source_ragged_live_pages_and_gpu_metadata_refresh(captured):
    """Catch source mutation, writes by empty/null rows, and stale GPU metadata."""
    from tokenspeed_kernel.ops.attention.kda._triton.retained import (
        retained_batched_conv_kernel,
        retained_batched_prepare_kernel,
        retained_batched_publish_kernel,
    )
    from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
        _kda_recoverssm_verify_kernel,
    )
    from utils import assert_no_triton_compile

    torch.manual_seed(1927)
    pages, heads, dim, capacity, width = 6, 2, 64, 8, 4
    state = torch.randn(pages, heads, dim, dim, device="cuda") * 0.02
    u = torch.randn(pages, heads, capacity, dim, device="cuda") * 0.02
    key = torch.nn.functional.normalize(torch.randn_like(u), dim=-1)
    decay = torch.rand_like(key) * 0.2 + 0.7
    # Counts belong to request slots: rows 0 and 1 start from the shared dense
    # source; the empty row 2, null row 3 and unused slots keep stale counts.
    length = torch.tensor(
        [[0], [0], [6], [5], [4], [2]], device="cuda", dtype=torch.int32
    )
    sources = torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.int32)
    live = torch.tensor([2, 3, 4, 5], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4, 5, 5, 6], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([2, 1, 0, 1], device="cuda", dtype=torch.int32)
    # Stamp capacity and absolute positions are independent of history slots.
    metadata = torch.tensor(
        [[0, 7, 2**32 + 6], [1, 9, 2**32 + 3], [2, 8, 0], [3, 4, 1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((4, 5, 2), -1, device="cuda", dtype=torch.int64)
    history = kda_ops.KdaRetainedHistory(u, key, decay, length, live, metadata[:, 0])
    original = tuple(x.clone() for x in (state, u, key, decay, length))
    shared_endpoint = state[1]
    current = [shared_endpoint.clone(), shared_endpoint.clone()]
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    q, k, v, gate = [
        torch.randn(1, 6, heads, dim, device="cuda").bfloat16() for _ in range(4)
    ]
    beta = torch.randn(1, 6, heads, device="cuda").bfloat16()
    out = torch.full_like(v, float("nan"))

    def run():
        prepare_history(state, history, sources, starts, draft_token_num=width)
        vllm_triton_kda_recoverssm_verify(
            q,
            k,
            v,
            gate,
            beta,
            a,
            bias,
            checkpoint_state=state,
            retained_history=history,
            query_start_loc=starts,
            state_indices=sources,
            spec_query_len=width,
            lower_bound=None,
            out=out,
            record_metadata=metadata,
            record_stamps=stamps,
        )
        publish(
            history,
            state,
            sources,
            starts,
            accepted,
            width=width,
            record_metadata=metadata,
            record_stamps=stamps,
        )

    def restore():
        for value, before in zip((state, u, key, decay, length), original):
            value.copy_(before)
        stamps.fill_(-1)

    for round_index, (boundaries, counts) in enumerate(
        (([0, 4, 5, 5, 6], [2, 1]), ([0, 1, 5, 5, 6], [1, 0]))
    ):
        starts.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        if round_index:
            sources.copy_(torch.tensor([2, 3, 1, 0], device="cuda", dtype=torch.int32))
            accepted.copy_(torch.tensor([1, 0, 0, 1], device="cuda", dtype=torch.int32))
            metadata[:, 1].add_(1)
            metadata[:, 2].add_(9)
        expected_states = []
        for row in range(2):
            begin, end = boundaries[row : row + 2]
            states, _, _, _ = _eager(
                k[0, begin:end],
                v[0, begin:end],
                gate[0, begin:end],
                beta[0, begin:end],
                a,
                bias,
                current[row],
                None,
            )
            q32 = q[0, begin:end].float()
            q32 *= torch.rsqrt(q32.square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
            expected_states.append(
                (
                    states,
                    torch.stack(
                        [
                            torch.einsum("hvk,hk->hv", s, q32[t])
                            for t, s in enumerate(states)
                        ]
                    ),
                )
            )
        if round_index == 0:
            run()
            if captured:
                restore()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                restore()
                graph.replay()
        elif captured:
            graph.replay()
        else:
            with assert_no_triton_compile(
                retained_batched_prepare_kernel,
                retained_batched_conv_kernel,
                retained_batched_publish_kernel,
                _kda_recoverssm_verify_kernel.fn,
            ):
                run()
        for row in range(2):
            begin, end = boundaries[row : row + 2]
            states, expected_out = expected_states[row]
            torch.testing.assert_close(
                out[0, begin:end].float(), expected_out, atol=2e-3, rtol=5e-3
            )
            if counts[row]:
                current[row] = states[counts[row] - 1]
            page = row + 2
            assert int(length[row, 0]) == (
                counts[row] if not round_index else (2, 1)[row] + counts[row]
            )
            torch.testing.assert_close(
                _reconstruct(
                    state[page], u[row], key[row], decay[row], int(length[row, 0])
                ),
                current[row],
                atol=2e-6,
                rtol=2e-5,
            )
            slot, generation, position = metadata[row].tolist()
            positions = position + torch.arange(
                end - begin, device="cuda", dtype=torch.int64
            )
            torch.testing.assert_close(
                stamps[slot, positions % 5, 0], positions, atol=0, rtol=0
            )
            assert bool((stamps[slot, positions % 5, 1] == generation).all())
        torch.testing.assert_close(
            out[0, 5], torch.zeros_like(out[0, 5]), atol=0, rtol=0
        )
        # Shared source, empty destination, and null destination stay immutable,
        # and so do the counts of the empty and null rows' slots.
        torch.testing.assert_close(
            state[[0, 1, 4, 5]], original[0][[0, 1, 4, 5]], atol=0, rtol=0
        )
        torch.testing.assert_close(length[2:], original[4][2:], atol=0, rtol=0)
        for value, before in zip((u, key, decay), original[1:4]):
            torch.testing.assert_close(value[2:], before[2:], atol=0, rtol=0)


def test_retained_history_large_value_tile_and_acceptance_clamping():
    """Exercise retained K128/BV16 scheduling and clamp requests to live windows."""
    torch.manual_seed(20291)
    batch, heads, dim, capacity, width = 16, 2, 128, 8, 4
    query_lengths = [4, 1, 0, 3] * 4
    offsets = [0]
    for count in query_lengths:
        offsets.append(offsets[-1] + count)
    requested = [-2, width + 3, 2, 2] * 4
    accepted = [
        min(max(count, 0), tokens, width)
        for count, tokens in zip(requested, query_lengths)
    ]
    q, k, v, gate = [
        torch.randn(1, offsets[-1], heads, dim, device="cuda").bfloat16()
        for _ in range(4)
    ]
    beta = torch.randn(1, offsets[-1], heads, device="cuda").bfloat16()
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    u = torch.randn(batch + 1, heads, capacity, dim, device="cuda") * 0.02
    key = torch.nn.functional.normalize(torch.randn_like(u), dim=-1)
    decay = torch.rand_like(key) * 0.2 + 0.7
    length = torch.full((batch + 1, 1), 2, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    starts = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    counts = torch.tensor(requested, device="cuda", dtype=torch.int32)
    history = kda_ops.KdaRetainedHistory(u, key, decay, length, indices, indices)
    base_before = state.clone()
    expected_out = torch.empty_like(v, dtype=torch.float32)
    expected_endpoints = []
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    for row, count in enumerate(query_lengths):
        current = _reconstruct(
            state[row + 1], u[row + 1], key[row + 1], decay[row + 1], 2
        )
        if count:
            begin, end = offsets[row : row + 2]
            states, _, _, _ = _eager(
                k[0, begin:end],
                v[0, begin:end],
                gate[0, begin:end],
                beta[0, begin:end],
                a,
                bias,
                current,
                -5.0,
            )
            q32 = q[0, begin:end].float()
            q32 *= torch.rsqrt(q32.square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
            expected_out[0, begin:end] = torch.stack(
                [torch.einsum("hvk,hk->hv", s, q32[t]) for t, s in enumerate(states)]
            )
            if accepted[row]:
                current = states[accepted[row] - 1]
        expected_endpoints.append(current)
    out = torch.empty_like(v)
    prepare_history(state, history, indices, starts, draft_token_num=width)
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        gate,
        beta,
        a,
        bias,
        checkpoint_state=state,
        retained_history=history,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        record_metadata=None,
        record_stamps=None,
    )
    torch.testing.assert_close(out.float(), expected_out, atol=2e-3, rtol=5e-3)
    torch.testing.assert_close(state, base_before, atol=0, rtol=0)
    assert bool((length == 2).all())
    publish(history, state, indices, starts, counts, width=width)
    torch.testing.assert_close(state, base_before, atol=0, rtol=0)
    torch.testing.assert_close(
        length[1:, 0],
        torch.tensor(
            [2 + count for count in accepted], device="cuda", dtype=torch.int32
        ),
        atol=0,
        rtol=0,
    )
    for row in range(batch):
        torch.testing.assert_close(
            _reconstruct(
                state[row + 1],
                u[row + 1],
                key[row + 1],
                decay[row + 1],
                2 + accepted[row],
            ),
            expected_endpoints[row],
            atol=2e-6,
            rtol=2e-5,
        )


def test_rows_outside_the_page_pool_are_skipped_by_verify_and_commit():
    """A source or live page past the checkpoint pool names no page: verify
    zeroes the row's output and leaves the pool alone, and commit keeps the
    row's slot count, as for a null row."""
    torch.manual_seed(1931)
    pages, heads, dim, capacity, width = 4, 2, 64, 8, 2
    state = torch.randn(pages, heads, dim, dim, device="cuda") * 0.02
    u = torch.randn(2, heads, capacity, dim, device="cuda") * 0.02
    key = torch.nn.functional.normalize(torch.randn_like(u), dim=-1)
    decay = torch.rand_like(key) * 0.2 + 0.7
    length = torch.tensor([[3], [3]], device="cuda", dtype=torch.int32)
    # Row 0's source and row 1's live page are one past the pool.
    sources = torch.tensor([pages, 1], device="cuda", dtype=torch.int32)
    live = torch.tensor([2, pages], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    history = kda_ops.KdaRetainedHistory(u, key, decay, length, live, slots)
    starts = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([2, 2], device="cuda", dtype=torch.int32)
    q, k, v, gate = [
        torch.randn(1, 4, heads, dim, device="cuda").bfloat16() for _ in range(4)
    ]
    beta = torch.randn(1, 4, heads, device="cuda").bfloat16()
    out = torch.full_like(v, float("nan"))
    saved = state.clone()
    prepare_history(state, history, sources, starts, draft_token_num=width)
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        gate,
        beta,
        torch.randn(heads, device="cuda") * 0.1,
        torch.randn(heads, dim, device="cuda") * 0.1,
        checkpoint_state=state,
        retained_history=history,
        query_start_loc=starts,
        state_indices=sources,
        spec_query_len=width,
        lower_bound=None,
        out=out,
        record_metadata=None,
        record_stamps=None,
    )
    publish(history, state, sources, starts, accepted, width=width)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    torch.testing.assert_close(state, saved, atol=0, rtol=0)
    assert length.flatten().tolist() == [3, 3]
