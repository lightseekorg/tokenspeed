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

"""Accuracy tests for the vendored vLLM KDA RecoverSSM kernels."""

from __future__ import annotations

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    vllm_triton_kda_recoverssm_commit,
    vllm_triton_kda_recoverssm_verify,
)


def _device_addresses(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        device=tensors[0].device,
        dtype=torch.int64,
    )


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    q32 = q.float()
    k32 = k.float()
    q32 *= torch.rsqrt(q32.square().sum(dim=-1, keepdim=True) + 1e-06)
    q32 *= q.shape[-1] ** (-0.5)
    k32 *= torch.rsqrt(k32.square().sum(dim=-1, keepdim=True) + 1e-06)
    gate_input = raw_g.float() + dt_bias[None, None, :, :]
    scale = A_log.exp()[None, None, :, None]
    gate = (
        lower_bound * torch.sigmoid(scale * gate_input)
        if lower_bound is not None
        else -scale * torch.nn.functional.softplus(gate_input)
    )
    beta = raw_beta.float().sigmoid()
    current = state.float().clone()
    output = torch.empty_like(v, dtype=torch.float32)
    states = []
    for token in range(q.shape[1]):
        current *= gate[:, token].exp()[:, :, None, :]
        correction = v[:, token].float() - torch.einsum(
            "bhvk,bhk->bhv", current, k32[:, token]
        )
        correction *= beta[:, token, :, None]
        current += torch.einsum("bhv,bhk->bhvk", correction, k32[:, token])
        output[:, token] = torch.einsum("bhvk,bhk->bhv", current, q32[:, token])
        states.append(current.clone())
    return (output, states)


@pytest.mark.parametrize(
    "width,accepted",
    [
        (width, accepted)
        for width in (1, 2, 3, 4, 8, 16)
        for accepted in range(width + 1)
    ],
)
@pytest.mark.parametrize("dim", [64, 128])
def test_vllm_recoverssm_verify_and_commit(width: int, accepted: int, dim: int) -> None:
    torch.manual_seed(7331)
    device = "cuda"
    batch, heads = (8 if width == 16 else 4, 12)
    q = torch.randn(batch, width, heads, dim, device=device).bfloat16()
    k = torch.randn_like(q)
    v = torch.randn_like(q).mul_(0.2)
    raw_g = torch.randn_like(q)
    raw_beta = torch.randn(batch, width, heads, device=device).bfloat16()
    A_log = torch.randn(heads, device=device).mul_(0.1)
    dt_bias = torch.randn(heads, dim, device=device).mul_(0.1)
    initial_state = torch.randn(batch, heads, dim, dim, device=device).mul_(0.02)
    expected_out, expected_states = _reference(
        q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, -5.0
    )
    state_pool = torch.zeros(
        batch + 1, heads, dim, dim, device=device, dtype=torch.float32
    )
    state_pool[1:].copy_(initial_state)
    correction_cache = torch.empty(
        batch + 1, heads, width, dim, device=device, dtype=torch.float32
    )
    kg_cache = torch.empty(
        batch + 1, heads, width, 2 * dim, device=device, dtype=torch.bfloat16
    )
    state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    query_start_loc = torch.arange(
        0, (batch + 1) * width, width, device=device, dtype=torch.int32
    )
    packed_q = q.reshape(1, batch * width, heads, dim)
    packed_k = k.reshape_as(packed_q)
    packed_v = v.reshape_as(packed_q)
    packed_g = raw_g.reshape_as(packed_q)
    packed_beta = raw_beta.reshape(1, batch * width, heads)
    out = torch.empty_like(packed_v)
    vllm_triton_kda_recoverssm_verify(
        packed_q,
        packed_k,
        packed_v,
        packed_g,
        packed_beta,
        A_log,
        dt_bias,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        record_metadata=None,
        record_stamps=None,
    )
    num_accepted = torch.full((batch,), accepted, device=device, dtype=torch.int32)
    plan = tuple(
        (torch.empty(batch, device=device, dtype=torch.int32) for _ in range(4))
    )
    vllm_triton_kda_recoverssm_commit(
        packed_q,
        packed_k,
        packed_v,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state_pool,)),
        correction_cache_base_addrs=_device_addresses((correction_cache,)),
        kg_cache_base_addrs=_device_addresses((kg_cache,)),
        spec_query_len=width,
        record_metadata=None,
        record_stamps=None,
        gate_parameters=(A_log, dt_bias, -5.0),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float().reshape_as(expected_out), expected_out, atol=0.02, rtol=0.02
    )
    expected_state = initial_state if accepted == 0 else expected_states[accepted - 1]
    torch.testing.assert_close(state_pool[1:], expected_state, atol=0.03, rtol=0.02)
    torch.testing.assert_close(
        kg_cache[:batch, ..., :dim], k.transpose(1, 2), atol=0, rtol=0
    )
    torch.testing.assert_close(
        kg_cache[:batch, ..., dim:], raw_g.transpose(1, 2), atol=0, rtol=0
    )
    torch.testing.assert_close(state_pool[1:], expected_state, atol=2e-06, rtol=2e-05)


@pytest.mark.parametrize(
    "batch,width,capacity",
    [(4, width, capacity) for width in (1, 4) for capacity in (8, 16, 32)]
    + [(batch, 4, 8) for batch in (1, 2, 3, 8, 16, 32)],
)
def test_recoverssm_raw_records_over_rounds(
    batch: int, width: int, capacity: int
) -> None:
    """Check raw records across wraps, slot reuse and retained prefixes."""
    _check_raw_ring_rounds(batch, width, capacity, capacity - 1)


@pytest.mark.parametrize("batch,width", [(4, 2), (8, 3), (16, 4), (32, 4)])
@pytest.mark.parametrize("start_position", [2**31 - 2, 2**32 - 2])
def test_recoverssm_records_with_large_positions(
    batch: int, width: int, start_position: int
) -> None:
    """Recover accepted prefixes across INT32 boundaries and ring wraps."""
    _check_raw_ring_rounds(batch, width, 7, start_position)


def test_record_chain_uses_verify_inputs() -> None:
    """Perturb verify inputs and check the shared recovery records change too."""
    torch.manual_seed(173)
    batch, width, heads, dim = (4, 4, 12, 128)
    shape = (1, batch * width, heads, dim)
    q, k, v, raw_gate = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, batch * width, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.01
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    results = []
    for perturb in (False, True):
        u = torch.empty(batch, heads, width, dim, device="cuda")
        kd = torch.empty(
            batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty_like(v)
        vllm_triton_kda_recoverssm_verify(
            q + 1 if perturb else q,
            k * 2 if perturb else k,
            v - 1 if perturb else v,
            raw_gate + 1 if perturb else raw_gate,
            beta,
            A,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kd,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            record_metadata=None,
            record_stamps=None,
        )
        results.append((out, u, kd))
    assert not torch.equal(results[0][0], results[1][0])
    assert not torch.equal(results[0][1], results[1][1])
    assert not torch.equal(results[0][2], results[1][2])


@pytest.mark.parametrize("batch", [4, 6, 8, 9, 16, 17, 32])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("record_start", [0, 2**32 - 2])
def test_recoverssm_ragged_prefix_and_null_rows(
    batch: int, captured: bool, record_start: int
) -> None:
    """Verify record values, metadata-selected destinations, and masked rows."""
    torch.manual_seed(1209)
    lengths = ([4, 2, 0, 1, 3, 4] * ((batch + 5) // 6))[:batch]
    lengths[-1] = 4
    full_lengths = lengths.copy()
    heads, dim, width = (12, 128, 4)
    starts_cpu = [0]
    for length in lengths:
        starts_cpu.append(starts_cpu[-1] + length)
    rows = starts_cpu[-1]
    shape = (1, rows, heads, dim)
    q, k, v, g = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, rows, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    initial = state.clone()
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    indices[-1] = 0
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    slots = batch + 2
    u = torch.full((slots, heads, width, dim), float("nan"), device="cuda")
    kd = torch.full(
        (slots, heads, width, 2 * dim),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = torch.full_like(v, float("nan"))
    records = [(request, 1, 0) for request in range(batch)]
    if record_start:
        records = [
            (batch - 1 - request, 7 + request, record_start + request)
            for request in range(batch)
        ]
    metadata = torch.tensor(records, dtype=torch.int64, device="cuda")
    stamps = torch.full((slots, width, 2), -1, dtype=torch.int64, device="cuda")
    record_k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-06)
    record_v = v.float()
    record_d = torch.exp(
        -5.0 * torch.sigmoid(A.exp()[None, None, :, None] * (g.float() + bias))
    )

    def verify() -> None:
        vllm_triton_kda_recoverssm_verify(
            q,
            k,
            v,
            g,
            beta,
            A,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kd,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            record_metadata=metadata,
            record_stamps=stamps,
        )

    buffers = (u, kd, metadata, stamps, out, starts, indices)
    addresses = [(tensor.data_ptr(), tensor.nbytes) for tensor in buffers]
    initial_records = records
    graph = None
    if captured:
        verify()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify()
    for round_index, live in enumerate((batch, max(1, batch // 2), batch)):
        # Padded graphs retain their capacity while the live batch shrinks and grows.
        lengths = [
            length if request < live else 0
            for request, length in enumerate(full_lengths)
        ]
        starts_cpu = [0]
        for length in lengths:
            starts_cpu.append(starts_cpu[-1] + length)
        starts.copy_(torch.tensor(starts_cpu, dtype=torch.int32, device="cuda"))
        records = [
            (slot, generation + round_index, position + round_index * width)
            for slot, generation, position in initial_records
        ]
        metadata.copy_(torch.tensor(records, dtype=torch.int64, device="cuda"))
        out.fill_(float("nan"))
        u.fill_(float("nan"))
        kd.fill_(float("nan"))
        stamps.fill_(-1)
        if graph is None:
            verify()
        else:
            graph.replay()
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        for request, length in enumerate(lengths):
            begin, end = starts_cpu[request : request + 2]
            slot, generation, position = records[request]
            if request == batch - 1:
                torch.testing.assert_close(
                    out[:, begin:end],
                    torch.zeros_like(out[:, begin:end]),
                    atol=0,
                    rtol=0,
                )
                valid = 0
            else:
                valid = length
                if length:
                    expected_out, _ = _reference(
                        q[:, begin:end],
                        k[:, begin:end],
                        v[:, begin:end],
                        g[:, begin:end],
                        beta[:, begin:end],
                        A,
                        bias,
                        initial[request + 1 : request + 2],
                        -5.0,
                    )
                    torch.testing.assert_close(
                        out[:, begin:end].float(), expected_out, atol=2e-3, rtol=8e-3
                    )
                    current = initial[request + 1].clone()
                    for offset in range(length):
                        token = begin + offset
                        ring = (position + offset) % width
                        key = record_k[0, token]
                        decay = record_d[0, token]
                        current = current * decay[:, None, :]
                        correction = (
                            record_v[0, token] - (current * key[:, None, :]).sum(-1)
                        ) * beta[0, token].float().sigmoid()[:, None]
                        torch.testing.assert_close(
                            u[slot, :, ring], correction, atol=2e-6, rtol=2e-5
                        )
                        torch.testing.assert_close(
                            kd[slot, :, ring, :dim], k[0, token], atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            kd[slot, :, ring, dim:], g[0, token], atol=0, rtol=0
                        )
                        current = current + correction[:, :, None] * key[:, None, :]
            unwritten = (position + torch.arange(valid, width, device="cuda")) % width
            assert torch.isnan(u[slot, :, unwritten]).all()
            assert torch.isnan(kd[slot, :, unwritten]).all()
            assert (stamps[slot, unwritten] == -1).all()
            if valid:
                positions = position + torch.arange(
                    valid, device="cuda", dtype=torch.int64
                )
                written = positions % width
                torch.testing.assert_close(
                    stamps[slot, written, 0], positions, atol=0, rtol=0
                )
                assert (stamps[slot, written, 1] == generation).all()
        assert torch.isnan(out[:, starts_cpu[-1] :]).all()
        assert torch.isnan(u[batch:]).all()
        assert torch.isnan(kd[batch:]).all()
        assert (stamps[batch:] == -1).all()
        assert [(tensor.data_ptr(), tensor.nbytes) for tensor in buffers] == addresses


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize(
    "width,capacity", [(4, 4), (4, 32), (16, 16), (16, 19), (16, 32), (32, 32)]
)
@pytest.mark.parametrize("heads", [2, 8])
def test_batched_recoverssm_commit_groups_alias_and_padding(
    captured: bool, width: int, capacity: int, heads: int
) -> None:
    """Replay both recovery variants with changing lengths and wrapped records."""
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    torch.manual_seed(982)
    layers, batch, dim, pages = (3, 6, 128, 12)
    channels = heads * dim
    groups_cpu = [1, 0, 1]
    read_cpu = [[1, 2, 3, 4, 5, -1], [5, 4, 3, 2, 1, -1]]
    write_cpu = [[6, 2, 7, 4, 8, -1], [6, 4, 7, 2, 8, -1]]
    accepted_cpu = [0, 1, 2, width - 1, width, 0]
    groups = torch.tensor(groups_cpu, device="cuda", dtype=torch.int32)
    reads = torch.tensor(read_cpu, device="cuda", dtype=torch.int32)
    writes = torch.tensor(write_cpu, device="cuda", dtype=torch.int32)
    accepted = torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32)
    states = torch.randn(layers, pages, heads, dim, dim, device="cuda") * 0.02
    conv = torch.randn(
        layers, pages, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    raw = torch.randn(
        layers, batch * width, 3 * channels + 16, device="cuda", dtype=torch.bfloat16
    )
    corrections = (
        torch.randn(layers, batch * capacity, channels + 16, device="cuda") * 0.01
    )
    kd = torch.randn(layers, batch * capacity, 2 * channels + 16, device="cuda") * 0.02
    kd[:, :, : 2 * channels].view(layers, batch * capacity, heads, 2, dim)[
        :, :, :, 1
    ].sigmoid_()
    a_log = torch.randn(layers, heads, device="cuda") * 0.1
    dt_bias = torch.randn(layers, heads, dim, device="cuda") * 0.1
    kd = kd.bfloat16()
    record_start = 2**32 + capacity - 2
    metadata = torch.tensor(
        [[request, 7, record_start] for request in range(batch)],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.empty(layers, batch, capacity, 2, device="cuda", dtype=torch.int64)
    positions = record_start + torch.arange(capacity, device="cuda", dtype=torch.int64)
    stamps[:, :, positions % capacity, 0] = positions
    stamps[..., 1] = 7
    stamps[:, -1].fill_(-1)
    seed_state, seed_conv = (states.clone(), conv.clone())
    descriptors = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    for column, storage in (
        (0, raw),
        (2, conv),
        (3, corrections),
        (5, kd),
        (8, states),
        (6, a_log),
        (7, dt_bias),
    ):
        descriptors[:, column] = _device_addresses(tuple(storage.unbind()))

    def launch() -> None:
        batched_recoverssm_commit(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            activation_dtype=kd.dtype,
            conv_state_dtype=conv.dtype,
            draft_token_num=width,
            num_heads=heads,
            head_dim=dim,
            qkv_stride=raw.stride(1),
            conv_stride=conv.stride(1),
            correction_stride=corrections.stride(1),
            key_gate_stride=kd.stride(1),
            state_stride=states.stride(1),
            conv_width=4,
            record_metadata=metadata,
            record_stamps=stamps,
            lower_bound=-5.0,
        )

    launch()
    graph = None
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
    for accepted_cpu in (
        accepted_cpu,
        [width, 0, width - 1, 1, 2, 0],
        [1, 3, 0, width, 4, 0],
    ):
        accepted.copy_(torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32))
        expected_state, expected_conv = (seed_state.clone(), seed_conv.clone())
        for layer, group in enumerate(groups_cpu):
            for request, count in enumerate(accepted_cpu[:-1]):
                source, dest = (read_cpu[group][request], write_cpu[group][request])
                state = seed_state[layer, source].clone()
                history = seed_conv[layer, source].clone()
                for token in range(count):
                    row = request * capacity + (record_start + token) % capacity
                    key, decay = (
                        kd[layer, row, : 2 * channels].view(heads, 2, dim).unbind(1)
                    )
                    correction = corrections[layer, row, :channels].view(heads, dim)
                    key = key.float()
                    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-06)
                    decay = torch.exp(
                        -5.0
                        * torch.sigmoid(
                            a_log[layer].exp()[:, None]
                            * (decay.float() + dt_bias[layer])
                        )
                    )
                    state = state * decay[:, None, :]
                    state = state + correction[:, :, None] * key[:, None, :]
                    history = torch.cat(
                        (
                            history[:, 1:],
                            raw[layer, request * width + token, : 3 * channels, None],
                        ),
                        dim=1,
                    )
                expected_state[layer, dest] = state
                expected_conv[layer, dest] = history
        states.copy_(seed_state)
        conv.copy_(seed_conv)
        if graph is None:
            launch()
        else:
            graph.replay()
        torch.testing.assert_close(states, expected_state, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        states = torch.empty_like(states)
        conv = torch.empty_like(conv)
        descriptors[:, 8] = _device_addresses(tuple(states.unbind()))
        descriptors[:, 2] = _device_addresses(tuple(conv.unbind()))


@pytest.mark.parametrize("batch", [1, 7, 33])
def test_commit_state_groups_live_slots_strides_and_graph(batch: int) -> None:
    from tokenspeed_kernel.ops.attention.kda.triton import commit_state_groups

    tables = [
        torch.arange(batch * (columns + 3), device="cuda", dtype=torch.int32).view(
            batch, columns + 3
        )[:, :columns]
        + group * 1000
        for group, columns in enumerate((2, 3, 5))
    ]
    tables = [torch.nn.functional.pad(table, (0, 2))[:, :-2] for table in tables]
    sources = [
        torch.arange(batch, device="cuda", dtype=torch.int32) + group * 100
        for group in range(3)
    ]
    descriptors = torch.tensor(
        [
            (
                (table.data_ptr() - tables[0].data_ptr()) // tables[0].element_size(),
                table.stride(0),
                table.shape[1],
                source.data_ptr(),
            )
            for table, source in zip(tables, sources, strict=True)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % 6
    pages = torch.empty((3, batch), device="cuda", dtype=torch.int32)
    reads = torch.empty_like(pages)
    steps = torch.empty(batch, device="cuda", dtype=torch.int32)

    def launch() -> None:
        commit_state_groups(
            accepted,
            tables[0],
            descriptors,
            batch_size=batch,
            draft_tokens=4,
            pages_out=pages,
            reads_out=reads,
            steps_out=steps,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for iteration in range(2):
        if iteration:
            accepted.add_(1)
            for source in sources:
                source.add_(7)
            tables[1].add_(23)
        graph.replay()
        torch.testing.assert_close(reads, torch.stack(sources), atol=0, rtol=0)
        expected_steps = accepted.clamp(1, 4)
        torch.testing.assert_close(steps, expected_steps, atol=0, rtol=0)
        for group, table in enumerate(tables):
            expected = table[:, 1]
            torch.testing.assert_close(pages[group], expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [4, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("layers", [1, 69])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("seed", [0, 1573])
def test_recoverssm_gate_recovery_across_layers(
    rows: int, layers: int, lower_bound: float | None, seed: int
) -> None:
    """Check raw gate recovery across batch, layer and gate configurations."""
    _check_gate_recovery(rows, layers, lower_bound, seed)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("project_gate", [False, True])
@pytest.mark.parametrize("with_conv_bias", [False, True])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("entry", ["split", "standalone"])
def test_verify_derived_records_with_serving_producers(
    batch: int,
    width: int,
    project_gate: bool,
    with_conv_bias: bool,
    lower_bound: float | None,
    entry: str,
) -> None:
    """Check producer records and each supported commit against an FP32 recurrence."""
    from tokenspeed_kernel.ops.attention.kda import (
        KdaReplayRecords,
        kda_replay_payload_specs,
        kda_verify_conv_update,
        try_kda_fused_paged_verify,
    )
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    torch.manual_seed(814)
    heads, dim = 12, 128
    rows, channels = batch * width, heads * dim
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    raw = torch.randn(rows, 3 * channels, **kwargs)
    conv_weights = torch.randn(3 * channels, 4, **kwargs) * 0.3
    conv_state = torch.randn(
        batch + 1, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    initial_conv = conv_state.clone()
    fa = torch.randn(rows, dim, **kwargs)
    fb = torch.randn(channels, dim, **kwargs) * 0.05
    beta = torch.randn(rows, heads, **kwargs)
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    conv_bias = torch.randn(3 * channels, **kwargs) * 0.05 if with_conv_bias else None
    gate = (
        torch.nn.functional.linear(fa, fb)
        if project_gate
        else torch.randn(rows, channels, **kwargs)
    )
    if not project_gate:
        # Exercise both sides of the softplus threshold and saturated gates.
        gate.view(rows, heads, dim)[..., :8] = torch.tensor(
            [-100.0, -20.0, -1.0, 0.0, 19.875, 20.125, 80.0, 100.0], **kwargs
        )
    payloads = tuple(
        torch.empty((rows, channels), device="cuda", dtype=payload_dtype)
        for channels, payload_dtype in kda_replay_payload_specs(
            torch.bfloat16,
            conv_channels=3 * channels,
            num_heads=heads,
            head_dim=dim,
        )
    )
    retained = torch.full((rows + 1, 3 * channels + 7), float("nan"), **kwargs)
    payloads = (retained[:rows, : 3 * channels], *payloads[1:])
    qkv = kda_verify_conv_update(
        raw,
        conv_weights,
        conv_state,
        indices,
        num_heads=heads,
        head_dim=dim,
        draft_token_num=width,
        recurrent_layout="v_major",
        bias=conv_bias,
        original_qkv=payloads[0],
    )
    torch.testing.assert_close(payloads[0], raw, atol=0, rtol=0)
    assert torch.isnan(retained[rows:]).all()
    assert torch.isnan(retained[:rows, 3 * channels :]).all()
    if (
        entry == "split"
        and not project_gate
        and not with_conv_bias
        and lower_bound is not None
    ):
        from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
            fused_kda_verify_conv_update_kernel,
        )
        from utils import assert_no_triton_compile

        # A different destination stride must reuse the kernel; replay must
        # regenerate both original and convolved QKV at the captured addresses.
        graph_storage = torch.full(
            (rows + 1, 3 * channels + 16), float("nan"), **kwargs
        )
        graph_retained = graph_storage[:rows, : 3 * channels]
        with assert_no_triton_compile(fused_kda_verify_conv_update_kernel):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_qkv = kda_verify_conv_update(
                    raw,
                    conv_weights,
                    conv_state,
                    indices,
                    num_heads=heads,
                    head_dim=dim,
                    draft_token_num=width,
                    recurrent_layout="v_major",
                    bias=None,
                    original_qkv=graph_retained,
                )
            graph.replay()
            torch.testing.assert_close(graph_qkv, qkv, atol=0, rtol=0)
            graph_retained.fill_(float("nan"))
            graph.replay()
        torch.testing.assert_close(graph_retained, raw, atol=0, rtol=0)
        assert torch.isnan(graph_storage[rows:]).all()
        assert torch.isnan(graph_storage[:rows, 3 * channels :]).all()
    torch.testing.assert_close(conv_state, initial_conv, atol=0, rtol=0)
    history = initial_conv[indices.long()].float().transpose(1, 2)
    sequence = torch.cat((history, raw.float().view(batch, width, -1)), dim=1)
    expected_conv = []
    for token in range(width):
        window = sequence[:, token : token + 4]
        value = (window * conv_weights.float().T).sum(dim=1)
        if conv_bias is not None:
            value += conv_bias.float()
        expected_conv.append(torch.nn.functional.silu(value))
    expected_conv = torch.stack(expected_conv, dim=1).reshape_as(qkv).to(torch.bfloat16)
    torch.testing.assert_close(qkv, expected_conv, atol=2e-5, rtol=8e-3)

    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    initial_state = state.clone()
    correction = payloads[1].view(batch, width, heads, dim).transpose(1, 2)
    key_gate = payloads[2].view(batch, width, heads, 2 * dim).transpose(1, 2)
    slots = torch.arange(batch, device="cuda", dtype=torch.int64)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    positions = slots + width - 1
    metadata = torch.stack((slots, torch.full_like(positions, 7), positions), dim=1)
    stamps = torch.full((batch, width, 2), -1, device="cuda", dtype=torch.int64)
    q, k, v = (part.reshape(1, rows, heads, dim) for part in qkv.chunk(3, dim=-1))
    if entry == "split":
        out = try_kda_fused_paged_verify(
            payloads[0],
            conv_weights,
            conv_state,
            torch.empty_like(conv_state),
            fa,
            fb,
            beta,
            a,
            bias,
            state_pool=state,
            state_scratch=None,
            read_indices=indices,
            write_indices=indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            lower_bound=lower_bound,
            recurrent_layout="v_major",
            store_states=False,
            replay_records=KdaReplayRecords(
                original_qkv=None,
                correction=correction,
                key_gate=key_gate,
                metadata=metadata,
                stamps=stamps,
                query_start_loc=starts,
            ),
            g_raw=gate,
            conv_qkv=qkv,
        )
    else:
        kernel = select_kernel(
            "attention",
            "kda_recoverssm_verify",
            format_signature(
                **{
                    role: dense_tensor_format(torch.bfloat16)
                    for role in ("q", "k", "v")
                }
            ),
            traits={"paged_state": True, "recurrent_layout": "v_major"},
        )
        out = kernel(
            q,
            k,
            v,
            gate.view_as(q),
            beta.view(1, rows, heads),
            a,
            bias,
            checkpoint_state=state,
            correction_cache=correction,
            kg_cache=key_gate,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=lower_bound,
            out=torch.empty_like(v),
            record_metadata=metadata,
            record_stamps=stamps,
        )

    shape = (batch, width, heads, dim)
    expected_out, expected_states = _reference(
        q.reshape(shape),
        k.reshape(shape),
        v.reshape(shape),
        gate.view(shape),
        beta.view(batch, width, heads),
        a,
        bias,
        initial_state[1:],
        lower_bound,
    )
    torch.testing.assert_close(
        out.reshape(shape), expected_out.to(torch.bfloat16), atol=2e-6, rtol=8e-3
    )
    normalized_k = k.float().reshape(shape)
    normalized_k = normalized_k * torch.rsqrt(
        normalized_k.square().sum(-1, keepdim=True) + 1e-6
    )
    gate_input = gate.float().view(shape) + bias
    scale = a.exp()[None, None, :, None]
    log_gate = (
        lower_bound * torch.sigmoid(scale * gate_input)
        if lower_bound is not None
        else -scale * torch.nn.functional.softplus(gate_input)
    )
    decay = log_gate.exp()
    recovered = initial_state[1:].clone()
    for token in range(width):
        ring = (positions + token) % width
        u = correction[slots, :, ring]
        saved = key_gate[slots, :, ring]
        torch.testing.assert_close(
            saved[..., :dim], k.reshape(shape)[:, token], atol=0, rtol=0
        )
        torch.testing.assert_close(
            saved[..., dim:], gate.view(shape)[:, token], atol=0, rtol=0
        )
        recovered = recovered * decay[:, token, :, None, :]
        recovered += u[..., None] * normalized_k[:, token, :, None, :]
        torch.testing.assert_close(
            recovered, expected_states[token], atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            stamps[slots, ring],
            torch.stack((positions + token, torch.full_like(positions, 7)), dim=1),
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(conv_state, initial_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, initial_state, atol=0, rtol=0)

    plan = tuple(torch.empty_like(indices) for _ in range(4))
    for accepted_count in range(1, width + 1):
        state.copy_(initial_state)
        accepted = torch.full_like(indices, accepted_count)
        vllm_triton_kda_recoverssm_commit(
            q,
            k,
            v,
            checkpoint_state=state,
            correction_cache=correction,
            kg_cache=key_gate,
            state_indices=indices,
            query_start_loc=starts,
            num_accepted_tokens=accepted,
            commit_lens=plan[0],
            final_state_indices=plan[1],
            boundary_state_indices=plan[2],
            boundary_recovery_lens=plan[3],
            state_base_addrs=_device_addresses((state,)),
            correction_cache_base_addrs=_device_addresses((correction,)),
            kg_cache_base_addrs=_device_addresses((key_gate,)),
            spec_query_len=width,
            record_metadata=metadata,
            record_stamps=stamps,
            gate_parameters=(a, bias, lower_bound),
        )
        torch.testing.assert_close(
            state[1:], expected_states[accepted_count - 1], atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(state[0], initial_state[0], atol=0, rtol=0)


def _check_raw_ring_rounds(
    batch: int, width: int, capacity: int, start_position: int
) -> None:
    torch.manual_seed(781)
    heads, dim = 2, 128
    device = "cuda"
    initial = torch.randn(batch + 1, heads, dim, dim, device=device) * 0.02
    state, checkpoint = initial.clone(), initial.clone()
    reference = initial[1:].clone()
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    u = torch.full((batch, heads, capacity, dim), float("nan"), device=device)
    kg = torch.full(
        (batch, heads, capacity, 2 * dim),
        float("nan"),
        device=device,
        dtype=torch.bfloat16,
    )
    stamps = torch.full((batch, capacity, 2), -1, device=device, dtype=torch.int64)
    slots = [(i + 1) % batch for i in range(batch)]
    positions = [start_position for _ in range(batch)]
    checkpoint_positions = positions.copy()
    generations = [1] * batch
    a = torch.randn(heads, device=device) * 0.1
    bias = torch.randn(heads, dim, device=device) * 0.1
    for step in range(128):
        if step and step % 43 == 0:
            slots = [(slot + 1) % batch for slot in slots]
            state.normal_(std=0.02)
            reference = state[1:].clone()
            checkpoint.copy_(state)
            checkpoint_positions = positions.copy()
            generations = [g + 1 for g in generations]
        for request in range(batch):
            if positions[request] - checkpoint_positions[request] + width > capacity:
                checkpoint[request + 1].copy_(state[request + 1])
                checkpoint_positions[request] = positions[request]
                generations[request] += 1
        q, k, v, g = [
            torch.randn(batch, width, heads, dim, device=device, dtype=torch.bfloat16)
            * 0.2
            for _ in range(4)
        ]
        beta = torch.randn(batch, width, heads, device=device, dtype=torch.bfloat16)
        expected_out, expected_states = _reference(
            q, k, v, g, beta, a, bias, reference, -5.0
        )
        counts = [(step + i) % (width + 1) for i in range(batch)]
        meta = torch.tensor(
            list(zip(slots, generations, positions)), device=device, dtype=torch.int64
        )
        packed = [x.reshape(1, batch * width, heads, dim) for x in (q, k, v, g)]
        out = torch.empty_like(packed[2])
        vllm_triton_kda_recoverssm_verify(
            *packed,
            beta.reshape(1, batch * width, heads),
            a,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kg,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            record_metadata=meta,
            record_stamps=stamps,
        )
        torch.testing.assert_close(
            out.reshape_as(expected_out).float(), expected_out, atol=2e-3, rtol=8e-3
        )
        for i, count in enumerate(counts):
            if count:
                reference[i].copy_(expected_states[count - 1][i])
            for token in range(count, width):
                row = (positions[i] + token) % capacity
                u[slots[i], :, row].fill_(float("nan"))
                kg[slots[i], :, row].fill_(float("nan"))
            positions[i] += count
        retained = torch.tensor(
            [p - c for p, c in zip(positions, checkpoint_positions)],
            device=device,
            dtype=torch.int32,
        )
        recovery_meta = torch.tensor(
            list(zip(slots, generations, checkpoint_positions)),
            device=device,
            dtype=torch.int64,
        )
        state.copy_(checkpoint)
        plans = [torch.empty_like(indices) for _ in range(4)]
        # Recovery spans retained accepted tokens, not merely the latest window.
        recovery_starts = (
            torch.arange(batch + 1, device=device, dtype=torch.int32) * capacity
        )
        vllm_triton_kda_recoverssm_commit(
            *packed[:3],
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kg,
            state_indices=indices,
            query_start_loc=recovery_starts,
            num_accepted_tokens=retained,
            commit_lens=plans[0],
            final_state_indices=plans[1],
            boundary_state_indices=plans[2],
            boundary_recovery_lens=plans[3],
            state_base_addrs=_device_addresses((state,)),
            correction_cache_base_addrs=_device_addresses((u,)),
            kg_cache_base_addrs=_device_addresses((kg,)),
            spec_query_len=capacity,
            record_metadata=recovery_meta,
            record_stamps=stamps,
            gate_parameters=(a, bias, -5.0),
        )
        torch.testing.assert_close(state[1:], reference, atol=2e-6, rtol=2e-5)


def _check_gate_recovery(
    rows: int, layers: int, lower_bound: float | None, seed: int
) -> None:
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    torch.manual_seed(seed)
    batch, width, heads, dim = rows // 4, 4, 2, 128
    channels = heads * dim
    state = torch.randn(layers, batch + 1, heads, dim, dim, device="cuda") * 0.02
    expected = state.clone()
    conv = torch.zeros(
        layers, batch + 1, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    raw = torch.zeros(layers, rows, 3 * channels, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(layers, rows, channels, device="cuda") * 0.02
    kg = (
        torch.randn(layers, rows, heads, 2, dim, device="cuda", dtype=torch.bfloat16)
        * 0.2
    )
    a = torch.randn(layers, heads, device="cuda") * 0.1
    bias = torch.randn(layers, heads, dim, device="cuda") * 0.1
    descriptors = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    for col, tensor in (
        (0, raw),
        (2, conv),
        (3, u),
        (5, kg),
        (6, a),
        (7, bias),
        (8, state),
    ):
        descriptors[:, col] = _device_addresses(tuple(tensor.unbind()))
    groups = torch.zeros(layers, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)[None]
    counts = torch.arange(batch, device="cuda", dtype=torch.int32) % width + 1
    for token in range(width):
        key = kg[:, token::width, :, 0].float()
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        gate_input = kg[:, token::width, :, 1].float() + bias[:, None]
        log_gate = (
            lower_bound * torch.sigmoid(a.exp()[:, None, :, None] * gate_input)
            if lower_bound is not None
            else -a.exp()[:, None, :, None] * torch.nn.functional.softplus(gate_input)
        )
        correction = u[:, token::width].reshape(layers, batch, heads, dim)
        updated = (
            expected[:, 1:] * log_gate.exp()[..., None, :]
            + correction[..., None] * key[..., None, :]
        )
        expected[:, 1:] = torch.where(
            (counts > token)[None, :, None, None, None], updated, expected[:, 1:]
        )
    batched_recoverssm_commit(
        descriptors,
        groups,
        indices,
        indices,
        counts,
        activation_dtype=kg.dtype,
        conv_state_dtype=conv.dtype,
        draft_token_num=width,
        num_heads=heads,
        head_dim=dim,
        qkv_stride=raw.stride(1),
        conv_stride=conv.stride(1),
        correction_stride=u.stride(1),
        key_gate_stride=kg.stride(1),
        state_stride=state.stride(1),
        conv_width=4,
        record_metadata=None,
        record_stamps=None,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("width", [4, 16, 32])
@pytest.mark.parametrize("raw_gate", [-20.0, 20.0])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("captured", [False, True])
def test_recoverssm_parallel_recovery_extreme_decays(
    width: int, raw_gate: float, lower_bound: float | None, captured: bool
) -> None:
    """Recover every prefix with extreme decays, eagerly or in a CUDA graph."""
    torch.manual_seed(1893)
    batch, heads, dim = width + 1, 8, 128
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    initial_state = state.clone()
    expected = initial_state.clone()
    u = torch.randn(batch, heads, width, dim, device="cuda") * 0.2
    kg = torch.empty(batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16)
    kg[..., :dim].normal_()
    kg[..., dim:].fill_(raw_gate)
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    counts = torch.arange(batch, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    gate_input = torch.tensor(raw_gate, device="cuda")
    gate = (
        -torch.nn.functional.softplus(gate_input)
        if lower_bound is None
        else lower_bound * torch.sigmoid(gate_input)
    )
    decay = gate.exp()
    for token in range(width):
        key = kg[:, :, token, :dim].float()
        key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        update = u[:, :, token, :, None] * key[:, :, None, :]
        expected[1:] = torch.where(
            (counts > token)[:, None, None, None],
            expected[1:] * decay + update,
            expected[1:],
        )
    # Unaccepted records may contain stale or undefined values.
    invalid = (
        torch.arange(width, device="cuda")[None, None, :, None]
        >= counts[:, None, None, None]
    )
    u.masked_fill_(invalid, float("nan"))
    kg.masked_fill_(invalid, float("nan"))
    placeholder = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    plan = [torch.empty_like(indices) for _ in range(4)]
    state_addresses = _device_addresses((state,))
    correction_addresses = _device_addresses((u,))
    key_gate_addresses = _device_addresses((kg,))

    def commit():
        state.copy_(initial_state)
        vllm_triton_kda_recoverssm_commit(
            placeholder,
            placeholder,
            placeholder,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kg,
            state_indices=indices,
            query_start_loc=starts,
            num_accepted_tokens=counts,
            commit_lens=plan[0],
            final_state_indices=plan[1],
            boundary_state_indices=plan[2],
            boundary_recovery_lens=plan[3],
            state_base_addrs=state_addresses,
            correction_cache_base_addrs=correction_addresses,
            kg_cache_base_addrs=key_gate_addresses,
            spec_query_len=width,
            record_metadata=None,
            record_stamps=None,
            gate_parameters=(a, bias, lower_bound),
        )

    commit()
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit()
        graph.replay()
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_recoverssm_aligned_checkpoint_states(
    width: int, use_tensor_cores: bool, captured: bool
) -> None:
    """Aligned recovery preserves boundary, final, and masked states."""
    from tokenspeed_kernel._triton import triton
    from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
        _commit_kda_state_kernel,
    )

    torch.manual_seed(2423)
    batch, heads, dim = 5, 2, 128
    initial = torch.randn(1 + 3 * batch, heads, dim, dim, device="cuda") * 0.02
    state, expected = initial.clone(), initial.clone()
    corrections = torch.randn(batch, heads, width, dim, device="cuda") * 0.2
    key_gate = torch.randn(
        batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16
    )
    a_log = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    sources = [1, 2, 3, 4, 0]
    destinations = [6, 7, 3, 0, 10]
    boundaries = [11, 12, -1, 14, 15]
    counts = [width, width // 2, width, 0, width]
    boundary_lengths = [width // 2, width // 4, width // 2, 0, width // 2]
    for row, count in enumerate(counts):
        if sources[row] <= 0 or destinations[row] <= 0:
            corrections[row].fill_(float("nan"))
            key_gate[row].fill_(float("nan"))
            continue
        current = initial[sources[row]].clone()
        for token in range(count):
            key = key_gate[row, :, token, :dim].float()
            key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            raw_gate = key_gate[row, :, token, dim:].float()
            decay = (
                -5.0 * torch.sigmoid(a_log.exp()[:, None] * (raw_gate + bias))
            ).exp()
            current = current * decay[:, None, :] + (
                corrections[row, :, token, :, None] * key[:, None, :]
            )
            if token + 1 == boundary_lengths[row] and boundaries[row] > 0:
                expected[boundaries[row]].copy_(current)
        expected[destinations[row]].copy_(current)
        corrections[row, :, count:].fill_(float("nan"))
        key_gate[row, :, count:].fill_(float("nan"))
    indices, final_indices, boundary_indices, commit_lengths, boundary_steps = (
        torch.tensor(values, device="cuda", dtype=torch.int32)
        for values in (sources, destinations, boundaries, counts, boundary_lengths)
    )
    state_addresses = _device_addresses((state,))
    correction_addresses = _device_addresses((corrections,))
    key_gate_addresses = _device_addresses((key_gate,))
    a_addresses = _device_addresses((a_log,))
    bias_addresses = _device_addresses((bias,))
    grid = lambda meta: (triton.cdiv(dim, meta["BV"]), batch, heads)

    def commit():
        state.copy_(initial)
        _commit_kda_state_kernel[grid](
            state_ref_ptr=state,
            state_base_addrs_ptr=state_addresses,
            state_block_stride=state.stride(0),
            correction_cache_ref_ptr=corrections,
            correction_cache_base_addrs_ptr=correction_addresses,
            correction_cache_block_stride=corrections.stride(0),
            kg_cache_ref_ptr=key_gate,
            kg_cache_base_addrs_ptr=key_gate_addresses,
            kg_cache_block_stride=key_gate.stride(0),
            state_indices_ptr=indices,
            commit_lens_ptr=commit_lengths,
            final_state_indices_ptr=final_indices,
            boundary_state_indices_ptr=boundary_indices,
            boundary_recovery_lens_ptr=boundary_steps,
            null_block_id=0,
            stride_state_head=state.stride(1),
            stride_state_v=state.stride(2),
            stride_state_k=state.stride(3),
            stride_correction_cache_head=corrections.stride(1),
            stride_correction_cache_pos=corrections.stride(2),
            stride_correction_cache_dim=corrections.stride(3),
            stride_kg_cache_head=key_gate.stride(1),
            stride_kg_cache_pos=key_gate.stride(2),
            stride_kg_cache_dim=key_gate.stride(3),
            stride_state_indices=indices.stride(0),
            group_indices_ptr=None,
            stride_state_group=0,
            stride_pointer_layer=1,
            record_metadata_ptr=None,
            record_stamps_ptr=None,
            stamp_layer_stride=0,
            RECORD_CAPACITY=width,
            RECORD_SLOTS=batch,
            K=dim,
            V=dim,
            BK=dim,
            BT=width,
            USE_TENSOR_CORES=use_tensor_cores,
            NUM_HEADS=heads,
            ALIGN_MODE=True,
            BATCH=batch,
            LAUNCH_DEPENDENT_KERNELS=False,
            a_ptr=a_addresses,
            bias_ptr=bias_addresses,
            USE_GATE_BASE_ADDRESSES=True,
            lower_bound=-5.0,
            USE_LOWER_BOUND=True,
            debug=True,
            sanitize_overflow=False,
            num_stages=2,
        )

    commit()
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit()
        graph.replay()
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-5)
