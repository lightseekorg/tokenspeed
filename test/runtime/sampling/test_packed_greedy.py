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

"""Packed verification must retain the target bonus without overcommitting KV."""

import pytest
import torch

from tokenspeed.runtime.sampling.backends.greedy import verify_packed_chain_greedy


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return request.param


def test_packed_greedy_caps_matches_and_preserves_bonus(device):
    candidates = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32,
        device=device,
    )
    # First request computed only its anchor: 11 is a bonus, not permission
    # to accept input 11 as well. Second rejects at its second prediction.
    target = torch.tensor([11, 21, 99, 23, 31, 32, 33, 34, -9], device=device)
    starts = torch.tensor([0, 1, 4, 8], device=device)
    output = torch.empty(12, dtype=torch.int32, device=device)
    indices = torch.empty((3, 4), dtype=torch.int32, device=device)
    lengths = torch.empty(3, dtype=torch.int32, device=device)
    verify_packed_chain_greedy(output, indices, lengths, candidates, target, starts)
    assert lengths.tolist() == [1, 2, 4]
    expected = [[11], [21, 99], [31, 32, 33, 34]]
    for row, tokens in enumerate(expected):
        assert output.view(3, 4)[row, : len(tokens)].tolist() == tokens
        assert indices[row, : len(tokens)].tolist() == list(
            range(row * 4, row * 4 + len(tokens))
        )
        assert (indices[row, len(tokens) :] == -1).all()


def test_packed_greedy_empty_requests_and_changed_graph_lengths(device):
    candidates = torch.tensor([[1, 2, 3], [5, 6, 7]], dtype=torch.int32, device=device)
    target = torch.tensor([2, 3, 4, 6, 7, 8], device=device)
    starts = torch.tensor([0, 3, 6], device=device)
    output = torch.empty(6, dtype=torch.int32, device=device)
    indices = torch.empty((2, 3), dtype=torch.int32, device=device)
    lengths = torch.empty(2, dtype=torch.int32, device=device)

    def run():
        verify_packed_chain_greedy(output, indices, lengths, candidates, target, starts)

    run()
    graph = None
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    starts.copy_(torch.tensor([0, 1, 1], device=device))
    run() if graph is None else graph.replay()
    assert lengths.tolist() == [1, 0]
    assert output[0].item() == 2
    assert indices.tolist() == [[0, -1, -1], [-1, -1, -1]]
    if graph is not None:
        graph.reset()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("through_executor", [False, True])
def test_backend_verification_and_nan_guard_keep_request_ownership(
    device, packed, through_executor
):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.nan_guard import NanGuard
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    backend = GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=3,
            max_draft_tokens_per_req=4,
            max_req_pool_size=4,
            vocab_size=128,
            device=device,
        )
    )
    candidates = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32,
        device=device,
    )
    targets = torch.tensor(
        [11, 12, 13, 14, 21, 99, 23, 24, 31, 32, 33, 34], device=device
    )
    selected = torch.tensor([0, 4, 5, 6, 8, 9, 10, 11], device=device)
    if packed:
        targets = targets[selected]
    logits = torch.full((len(targets), 128), -1.0, device=device)
    logits.scatter_(1, targets[:, None], 1.0)
    output = LogitsProcessorOutput(next_token_logits=logits)
    starts = torch.tensor([0, 1, 4, 8], device=device) if packed else None
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=3,
        num_extends=0,
        input_num_tokens=len(targets),
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 3, 4),
        decode_query_indptr=starts,
    )
    guard = NanGuard.create(True, 3, device)
    guard.reset(3)
    info = SamplingBatchInfo(vocab_size=128, device=device)
    if through_executor:
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        # Only the model-loading environment is omitted. Exercise the real
        # executor's verification dispatch and acceptance finalization.
        executor = ModelExecutor.__new__(ModelExecutor)
        executor.sampling_backend = backend
        executor.tree_spec = None
        executor._simulated_accept_length = None
        tokens, lengths = executor._sample_and_verify(output, info, ctx, candidates)
    else:
        tokens, lengths = backend.verify(
            output, info, candidates, tree=None, query_start_loc=starts
        )
    assert lengths.tolist() == ([1, 2, 4] if packed else [4, 2, 4])
    assert tokens.view(3, 4)[0, 0].item() == 11
    assert tokens.view(3, 4)[1, :2].tolist() == [21, 99]
    guard.merge_oov(tokens, ctx, 128)
    assert guard.flags.tolist() == [0, 0, 0]
    # A NaN belongs to the third request regardless of its packed row offset.
    logits[4 if packed else 8, 3] = float("nan")
    guard.audit_logits(output, ctx)
    assert guard.flags.tolist() == [0, 0, 1]
    assert torch.isfinite(logits).all()


def test_packed_synthetic_acceptance_respects_computed_rows_and_graph(device):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    backend = GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=4.0,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=3,
            max_draft_tokens_per_req=6,
            max_req_pool_size=4,
            vocab_size=128,
            device=device,
        )
    )
    candidates = torch.tensor(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]],
        dtype=torch.int32,
        device=device,
    )
    target = torch.arange(50, 60, device=device)
    logits = torch.full((10, 128), -1.0, device=device)
    logits.scatter_(1, target[:, None], 1.0)
    output = LogitsProcessorOutput(next_token_logits=logits)
    info = SamplingBatchInfo(vocab_size=128, device=device)
    starts = torch.tensor([0, 4, 9, 10], device=device)

    def run():
        return backend.verify(
            output, info, candidates, tree=None, query_start_loc=starts
        )

    tokens, lengths = run()
    assert lengths.tolist() == [4, 4, 1]
    assert tokens.view(3, 6)[0, :4].tolist() == [11, 12, 13, 53]
    assert tokens.view(3, 6)[1, :4].tolist() == [21, 22, 23, 57]
    assert tokens.view(3, 6)[2, 0].item() == 59
    graph = None
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens, lengths = run()
    starts.copy_(torch.tensor([0, 4, 4, 10], device=device))
    if graph is None:
        tokens, lengths = run()
    else:
        graph.replay()
    assert lengths.tolist() == [4, 0, 4]
    assert backend._accept_index_buf[:18].view(3, 6)[1].tolist() == [-1] * 6
    assert tokens.view(3, 6)[2, :4].tolist() == [31, 32, 33, 57]
    if graph is not None:
        graph.reset()
