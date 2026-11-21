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
from kda_retained_reference import endpoint
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    prepare_recoverssm_metadata,
    vllm_triton_kda_recoverssm_commit,
    vllm_triton_kda_recoverssm_verify,
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


@pytest.mark.parametrize("batch", [3, 16, 33])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_vllm_retained_verify_and_publish(batch, width, dim, lower_bound):
    from tokenspeed_kernel.ops.attention.kda import KdaRetainedHistory

    torch.manual_seed(7331)
    heads, pages = 2, 2 * batch + 2
    q = torch.randn(batch, width, heads, dim, device="cuda").bfloat16()
    k, v, raw_g = torch.randn_like(q), torch.randn_like(q) * 0.2, torch.randn_like(q)
    raw_beta = torch.randn(batch, width, heads, device="cuda").bfloat16()
    a, bias = (
        torch.randn(heads, device="cuda") * 0.1,
        torch.randn(heads, dim, device="cuda") * 0.1,
    )
    state = torch.randn(pages, heads, dim, dim, device="cuda") * 0.02
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    live = reads + batch
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    capacity = max(16, width)
    history = KdaRetainedHistory(
        torch.zeros(batch, heads, capacity, dim, device="cuda"),
        torch.zeros(batch, heads, capacity, dim, device="cuda"),
        torch.ones(batch, heads, capacity, dim, device="cuda"),
        torch.zeros(batch, 1, device="cuda", dtype=torch.int32),
        live,
        torch.arange(batch, device="cuda", dtype=torch.int32),
    )
    metadata = torch.tensor(
        [[row, 19, 2**32 + 7] for row in range(batch)], device="cuda", dtype=torch.int64
    )
    stamps = torch.full((batch, capacity + 3, 2), -1, device="cuda", dtype=torch.int64)
    expected_out, expected_states = _reference(
        q, k, v, raw_g, raw_beta, a, bias, state[reads.long()], lower_bound
    )
    source_before = state[reads.long()].clone()
    packed = [x.reshape(1, batch * width, heads, dim) for x in (q, k, v, raw_g)]
    output = vllm_triton_kda_recoverssm_verify(
        *packed,
        raw_beta.reshape(1, batch * width, heads),
        a,
        bias,
        checkpoint_state=state,
        retained_history=history,
        query_start_loc=starts,
        state_indices=reads,
        spec_query_len=width,
        lower_bound=lower_bound,
        out=torch.empty_like(packed[2]),
        record_metadata=metadata,
        record_stamps=stamps,
    )
    torch.testing.assert_close(
        output.reshape_as(expected_out).float(), expected_out, atol=2e-3, rtol=1e-2
    )
    assert history.length.sum().item() == 0
    for accepted in range(width + 1):
        # Each prefix starts from the unpublished candidate records above.
        history.length.zero_()
        vllm_triton_kda_recoverssm_commit(
            *packed[:3],
            checkpoint_state=state,
            retained_history=history,
            state_indices=reads,
            query_start_loc=starts,
            num_accepted_tokens=torch.full(
                (batch,), accepted, device="cuda", dtype=torch.int32
            ),
            spec_query_len=width,
            record_metadata=metadata,
            record_stamps=stamps,
        )
        assert history.length[:, 0].tolist() == [accepted] * batch
        actual = torch.stack(
            [
                endpoint(state[page], history, row)
                for row, page in enumerate(live.tolist())
            ]
        )
        expected = source_before if accepted == 0 else expected_states[accepted - 1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(state[reads.long()], source_before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_record_metadata_takes_the_checkpoint_before_the_verify_window(dtype) -> None:
    """Live rows get (slot, generation, cache length - window); others keep theirs."""
    seq_lens = torch.tensor([9, 40, 8, 1 << 20], dtype=dtype, device="cuda")
    metadata = torch.full((6, 3), -5, dtype=torch.int64, device="cuda")
    prepare_recoverssm_metadata(
        seq_lens, metadata, batch=3, generation=7, draft_token_num=8
    )
    assert metadata[:3].tolist() == [[0, 7, 1], [1, 7, 32], [2, 7, 0]]
    assert (metadata[3:] == -5).all()
    with pytest.raises(ValueError, match="INT32 or INT64"):
        prepare_recoverssm_metadata(
            seq_lens.float(), metadata, batch=3, generation=7, draft_token_num=8
        )
