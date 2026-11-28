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

"""Packed CSR verification contract tests for ``SonicSamplingBackend``."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=90,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="sonic's kernels need NVIDIA's libdevice (fast_logf).",
)

from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    get_apply_vocab_mask_func,
)
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.sonic import (  # noqa: E402
    SonicSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.utils.server_args import prepare_server_args  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

POOL = 8
MAX_BS = 4
MAX_N = 4


def _kernel_test_utils():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(os.path.dirname(root), "tokenspeed-kernel", "test", "utils.py")
    spec = importlib.util.spec_from_file_location("sonic_packed_kernel_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


def _sp(rid: str, vocab: int, **overrides) -> SamplingParams:
    defaults = dict(temperature=1.0, top_k=-1, top_p=1.0)
    defaults.update(overrides)
    sp = SamplingParams(**defaults)
    sp.verify(vocab)
    sp.resolve_seed(rid)
    sp.normalize(None)
    return sp


def _make_backend(
    vocab: int,
    *,
    max_bs: int = MAX_BS,
    max_n: int = MAX_N,
    enable_packed_verify: bool = True,
    enable_output_logprobs: bool = False,
    synthetic_acceptance_length: float | None = None,
) -> SonicSamplingBackend:
    config = SamplingBackendConfig(
        synthetic_acceptance_length=synthetic_acceptance_length,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        enable_output_logprobs=enable_output_logprobs,
        enable_packed_verify=enable_packed_verify,
        max_bs=max_bs,
        max_draft_tokens_per_req=max_n,
        max_req_pool_size=POOL,
        vocab_size=vocab,
        device="cuda",
    )
    return SonicSamplingBackend(config)


def _allow_tokens(allowed: torch.Tensor, vocab: int) -> torch.Tensor:
    words = (vocab + 31) // 32
    mask = torch.zeros(
        (allowed.shape[0], words), dtype=torch.int32, device=allowed.device
    )
    rows = torch.arange(allowed.shape[0], device=allowed.device)
    for choice in range(allowed.shape[1]):
        tokens = allowed[:, choice].long()
        bits = (torch.ones_like(tokens) << (tokens % 32)).to(torch.int32)
        mask[rows, tokens // 32] |= bits
    return mask


def _masked_logprobs(
    logits: torch.Tensor,
    vocab_mask: torch.Tensor | None,
    predict: torch.Tensor,
    accept_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
) -> torch.Tensor:
    starts = query_start_loc.tolist()
    counts = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    rows = []
    for req, length in enumerate(accept_lengths.tolist()):
        cnt = counts[req]
        for col in range(width):
            if cnt == 0:
                rows.append(0)
            else:
                offset = col if col < length and col < cnt else 0
                rows.append(starts[req] + offset)
    idx = torch.tensor(rows, dtype=torch.int64, device=logits.device)
    selected_logits = logits.index_select(0, idx).clone()
    if vocab_mask is not None:
        bits = torch.arange(32, device=logits.device)
        allowed = (
            ((vocab_mask.index_select(0, idx).long().unsqueeze(-1) >> bits) & 1)
            .flatten(1)
            .bool()[:, : logits.shape[1]]
        )
        selected_logits.masked_fill_(~allowed, float("-inf"))
    logp = torch.log_softmax(selected_logits, dim=-1, dtype=torch.float32)
    expected = (
        logp.gather(1, predict.long().unsqueeze(1)).squeeze(1).view(len(counts), width)
    )
    for req, cnt in enumerate(counts):
        if cnt == 0:
            expected[req].zero_()
    return expected.view(-1)


def _prepare(
    backend, sps: list[SamplingParams], n: int, slots: list[int]
) -> torch.Tensor:
    backend.prepare_step(
        admission_ids=[sp.seed * MAX_BS + i + 1 for i, sp in enumerate(sps)],
        request_pool_indices=slots,
        sampling_params_list=sps,
        num_tokens_per_req=n,
    )
    return torch.tensor(slots, dtype=torch.int64, device="cuda")


def test_sonic_supports_packed_verify_and_config_from_server_args():
    assert SonicSamplingBackend.supports_packed_verify is True
    base = [
        "--model",
        "unused",
        "--speculative-algorithm",
        "DSPARK",
        "--sampling-backend",
        "sonic",
        "--moe-backend",
        "mega_moe",
        "--enable-dp-local-cuda-graph",
    ]
    fixed = prepare_server_args(base)
    assert (
        SamplingBackendConfig.from_server_args(
            fixed,
            vocab_size=1024,
            max_bs=4,
            max_draft_tokens_per_req=4,
            max_req_pool_size=8,
            device="cuda",
        ).enable_packed_verify
        is False
    )

    adaptive = prepare_server_args(
        base + ["--enable-dsv41-adaptive-dspark", "--disable-prefill-graph"]
    )
    assert (
        SamplingBackendConfig.from_server_args(
            adaptive,
            vocab_size=1024,
            max_bs=4,
            max_draft_tokens_per_req=4,
            max_req_pool_size=8,
            device="cuda",
        ).enable_packed_verify
        is True
    )


@pytest.mark.parametrize("vocab", [1024, 151936])
def test_sonic_packed_verify_shortened_empty_grammar_and_logprobs(vocab):
    torch.manual_seed(17)
    backend = _make_backend(vocab, enable_output_logprobs=True)
    sps = [
        _sp("p0", vocab, temperature=0.0),
        _sp("p1", vocab, temperature=0.8, top_k=16),
        _sp("p2", vocab, temperature=0.9, top_p=0.9),
        _sp("p3", vocab, temperature=0.0),
    ]
    slots = _prepare(backend, sps, MAX_N, [1, 3, 5, 7])
    offsets = torch.arange(100, 100 + POOL + 1, dtype=torch.int32, device="cuda")

    starts = torch.tensor([0, 4, 6, 6, 7], dtype=torch.int64, device="cuda")
    padded_rows = 10
    logits = torch.randn((padded_rows, vocab), dtype=torch.bfloat16, device="cuda")
    # Ensure req 0 matches its first 2 drafts and stops cleanly, req 1 picks allowed grammar tokens.
    logits[0, 11] = 12.0
    logits[1, 12] = 12.0
    logits[2, 13] = 12.0
    logits[3, 14] = 12.0
    logits[4, 21] = 12.0
    logits[5, 22] = 12.0
    logits[6, 31] = 12.0

    allowed = torch.stack(
        [
            torch.tensor([11, 12, 13, 14, 21, 22, 31, 0, 0, 0], device="cuda"),
            torch.tensor([101, 102, 103, 104, 121, 122, 131, 1, 1, 1], device="cuda"),
            torch.full((padded_rows,), vocab - 1, device="cuda"),
        ],
        dim=1,
    )
    vocab_mask = _allow_tokens(allowed, vocab)
    candidates = torch.tensor(
        [
            [9, 11, 12, 99],
            [9, 21, 99, 99],
            [9, 31, 32, 33],
            [9, 41, 42, 43],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=offsets,
        vocab_size=vocab,
        vocab_mask=vocab_mask,
        apply_vocab_mask=get_apply_vocab_mask_func("xgrammar"),
        device="cuda",
    )
    predict, accept_lengths = backend.verify(
        output, info, candidates, tree=None, query_start_loc=starts
    )
    assert predict.shape == (MAX_BS * MAX_N,)
    assert accept_lengths.tolist() == [3, 2, 0, 1]
    assert predict.view(MAX_BS, MAX_N)[0, :3].tolist() == [11, 12, 13]
    assert predict.view(MAX_BS, MAX_N)[0, 3].item() == 0
    assert predict.view(MAX_BS, MAX_N)[1, 2:].tolist() == [0, 0]
    assert predict.view(MAX_BS, MAX_N)[2].tolist() == [0, 0, 0, 0]
    assert predict.view(MAX_BS, MAX_N)[3, 1:].tolist() == [0, 0, 0]
    expected_logp = _masked_logprobs(
        logits, vocab_mask, predict, accept_lengths, starts, MAX_N
    )
    torch.testing.assert_close(
        output.next_token_logprobs, expected_logp, atol=1e-5, rtol=1e-5
    )

    # All-empty batch with zero logit rows and empty batch size.
    zero_starts = torch.zeros(MAX_BS + 1, dtype=torch.int64, device="cuda")
    zero_out = LogitsProcessorOutput(next_token_logits=logits[:0])
    zero_info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=offsets,
        vocab_size=vocab,
        vocab_mask=vocab_mask[:0],
        apply_vocab_mask=get_apply_vocab_mask_func("xgrammar"),
        device="cuda",
    )
    zp, za = backend.verify(
        zero_out, zero_info, candidates, tree=None, query_start_loc=zero_starts
    )
    assert zp.count_nonzero().item() == 0
    assert za.tolist() == [0] * MAX_BS
    assert zero_out.next_token_logprobs.count_nonzero().item() == 0

    ep, ea = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits[:0]),
        SamplingBatchInfo(
            req_pool_indices=slots[:0],
            valid_cache_lengths=offsets,
            vocab_size=vocab,
            device="cuda",
        ),
        candidates[:0],
        tree=None,
        query_start_loc=zero_starts[:1],
    )
    assert ep.numel() == 0 and ea.numel() == 0


def test_sonic_packed_verify_n1_handles_empty_and_padded_rows():
    vocab = 1024
    backend = _make_backend(vocab, max_n=1, enable_output_logprobs=True)
    sps = [
        _sp("n0", vocab, temperature=0.0),
        _sp("n1", vocab, temperature=0.8, top_k=8),
        _sp("n2", vocab, temperature=0.0),
    ]
    slots = _prepare(backend, sps, 1, [2, 4, 6])
    offsets = torch.full((POOL + 1,), 15, dtype=torch.int32, device="cuda")
    logits = torch.randn((4, vocab), dtype=torch.bfloat16, device="cuda")
    logits[0, 55] = 15.0
    logits[1, 77] = 15.0
    starts = torch.tensor([0, 1, 1, 2], dtype=torch.int32, device="cuda")
    vocab_mask = _allow_tokens(
        torch.tensor([[55, 56], [77, 78], [99, 100], [1, 2]], device="cuda"), vocab
    )
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=offsets,
        vocab_size=vocab,
        vocab_mask=vocab_mask,
        apply_vocab_mask=get_apply_vocab_mask_func("xgrammar"),
        device="cuda",
    )
    out = LogitsProcessorOutput(next_token_logits=logits.clone())
    cands = torch.zeros((3, 1), dtype=torch.int32, device="cuda")
    pred, lengths = backend.verify(out, info, cands, tree=None, query_start_loc=starts)
    assert pred.tolist() == [55, 0, 77]
    assert lengths.tolist() == [1, 0, 1]
    expected_logp = _masked_logprobs(logits, vocab_mask, pred, lengths, starts, 1)
    torch.testing.assert_close(
        out.next_token_logprobs, expected_logp, atol=1e-5, rtol=1e-5
    )

    # Zero-row N == 1 packed verify.
    zout = LogitsProcessorOutput(next_token_logits=logits[:0])
    zpred, zlen = backend.verify(
        zout,
        SamplingBatchInfo(
            req_pool_indices=slots,
            valid_cache_lengths=offsets,
            vocab_size=vocab,
            device="cuda",
        ),
        cands,
        tree=None,
        query_start_loc=torch.zeros(4, dtype=torch.int32, device="cuda"),
    )
    assert zpred.tolist() == [0, 0, 0]
    assert zlen.tolist() == [0, 0, 0]
    assert zout.next_token_logprobs.tolist() == [0.0, 0.0, 0.0]


def test_sonic_packed_verify_rejects_unsupported_modes():
    from tokenspeed.runtime.sampling.tree_verify import TreeVerifyBatch

    vocab = 1024
    tree_cfg = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        draft_tree=True,
        enable_packed_verify=True,
        max_bs=2,
        max_draft_tokens_per_req=MAX_N,
        max_req_pool_size=POOL,
        vocab_size=vocab,
        device="cuda",
    )
    tree_backend = SonicSamplingBackend(tree_cfg)
    tree = TreeVerifyBatch(
        parents=torch.tensor(
            [[-1, 0, 0, 1], [-1, 0, 0, 1]], device="cuda", dtype=torch.int32
        ),
        depths=torch.tensor(
            [[0, 1, 1, 2], [0, 1, 1, 2]], device="cuda", dtype=torch.int32
        ),
        draft_q=torch.empty(0, device="cuda"),
    )
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([1, 2], device="cuda", dtype=torch.int64),
        valid_cache_lengths=torch.zeros(POOL + 1, device="cuda", dtype=torch.int32),
        vocab_size=vocab,
        device="cuda",
    )
    logits = torch.zeros((2 * MAX_N, vocab), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="tree verification"):
        tree_backend.verify(
            LogitsProcessorOutput(next_token_logits=logits),
            info,
            torch.zeros((2, MAX_N), dtype=torch.int32, device="cuda"),
            tree=tree,
            query_start_loc=torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32),
        )

    synth_backend = _make_backend(vocab, max_bs=2, synthetic_acceptance_length=2.0)
    with pytest.raises(ValueError, match="synthetic acceptance"):
        synth_backend.verify(
            LogitsProcessorOutput(next_token_logits=logits),
            info,
            torch.zeros((2, MAX_N), dtype=torch.int32, device="cuda"),
            tree=None,
            query_start_loc=torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32),
        )


def test_sonic_packed_verify_cuda_graph_replay_and_no_compile():
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import (
        multistep,
        prologue,
    )
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import verify as verify_ops

    vocab = 1024
    utils = _kernel_test_utils()
    backend = _make_backend(vocab, enable_packed_verify=True)
    sps = [
        _sp("g0", vocab, temperature=0.0),
        _sp("g1", vocab, top_k=16, temperature=0.8),
        _sp("g2", vocab, temperature=0.0),
        _sp("g3", vocab, top_p=0.9, temperature=0.9),
    ]
    slots_ref = _prepare(backend, sps, MAX_N, [1, 2, 3, 4])

    cap_rows = 10
    backend.prepare_capture(bs=MAX_BS, num_tokens_per_req=MAX_N)
    logits_buf = torch.zeros((cap_rows, vocab), dtype=torch.bfloat16, device="cuda")
    cand_buf = torch.zeros((MAX_BS, MAX_N), dtype=torch.int32, device="cuda")
    req_buf = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    qsl_buf = torch.tensor([0, 4, 6, 7, 10], dtype=torch.int64, device="cuda")
    offsets_buf = torch.zeros((POOL + 1,), dtype=torch.int32, device="cuda")
    cap_info = SamplingBatchInfo(
        req_pool_indices=req_buf,
        valid_cache_lengths=offsets_buf,
        vocab_size=vocab,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.verify(
                LogitsProcessorOutput(next_token_logits=logits_buf),
                cap_info,
                cand_buf,
                tree=None,
                query_start_loc=qsl_buf,
            )
    torch.cuda.current_stream().wait_stream(stream)
    backend.reset_capture_state()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g_predict, g_accept = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits_buf),
            cap_info,
            cand_buf,
            tree=None,
            query_start_loc=qsl_buf,
        )

    with utils.assert_no_triton_compile(
        prologue.bitpacked_reduction_kernel,
        prologue.whole_reduction_kernel,
        multistep.cumulative_unpack_kernel,
        verify_ops.chain_speculative_verification_kernel,
    ):
        for step, starts in enumerate(
            (
                [0, 4, 6, 7, 10],
                [0, 1, 2, 3, 4],
                [0, 0, 3, 5, 8],
                [0, 2, 4, 6, 8],
            )
        ):
            torch.manual_seed(100 + step)
            live_logits = torch.randn(
                (cap_rows, vocab), dtype=torch.bfloat16, device="cuda"
            )
            candidates = torch.randint(
                0, vocab, (MAX_BS, MAX_N), dtype=torch.int32, device="cuda"
            )
            offsets = torch.full(
                (POOL + 1,), step * MAX_N, dtype=torch.int32, device="cuda"
            )
            qsl = torch.tensor(starts, dtype=torch.int64, device="cuda")

            # Eager baseline on matching state and inputs.
            _prepare(backend, sps, MAX_N, [1, 2, 3, 4])
            eager_counts = backend.state.decode_counts.clone()
            e_predict, e_accept = backend.verify(
                LogitsProcessorOutput(next_token_logits=live_logits.clone()),
                SamplingBatchInfo(
                    req_pool_indices=slots_ref,
                    valid_cache_lengths=offsets,
                    vocab_size=vocab,
                    device="cuda",
                ),
                candidates,
                tree=None,
                query_start_loc=qsl,
            )
            e_predict, e_accept = e_predict.clone(), e_accept.clone()
            after_eager_counts = backend.state.decode_counts.clone()

            # Graph replay with updated CSR starts and live inputs.
            backend.state.decode_counts.copy_(eager_counts)
            logits_buf.copy_(live_logits)
            cand_buf.copy_(candidates)
            offsets_buf.copy_(offsets)
            qsl_buf.copy_(qsl)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(g_predict, e_predict), starts
            assert torch.equal(g_accept, e_accept), starts
            assert torch.equal(backend.state.decode_counts, after_eager_counts), starts
    graph.reset()
