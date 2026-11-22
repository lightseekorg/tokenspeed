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

"""Output logprobs from vocabulary shards, over a real NCCL all-gather.

Each rank hands the triton sampler its raw FP32 head shard of a padded head
(the vocabulary ends inside the last shard), capped as ``tanh_softcap``
caps the gathered rows. Tokens, accept lengths and output logprobs must be
bitwise those of the same sampler on the whole rows the logits processor
produces (gather, cut to the vocabulary, cap, then the in-place mask),
eagerly and replayed from a CUDA graph.

Usage:
    python -m pytest test/runtime/distributed/test_vocab_shard_logprobs_nccl.py -v
"""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-2gpu")

SHARD = 3000
CAP = 30.0
BS, N, POOL = 3, 8, 8


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _bitmask(allowed: torch.Tensor) -> torch.Tensor:
    rows, vocab = allowed.shape
    padded = torch.zeros(rows, -(-vocab // 32) * 32, dtype=torch.bool)
    padded[:, :vocab] = allowed
    bits = padded.view(rows, -1, 32).to(torch.int64) << torch.arange(32)
    bits = bits.sum(-1)
    return torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)


class _Case:
    """The same raw head output and mask on every rank."""

    def __init__(self, world_size: int, rows: int, masked: bool, device):
        from tokenspeed_kernel.ops.activation.triton import tanh_softcap

        # The vocabulary ends 700 columns into the last shard.
        self.vocab = (world_size - 1) * SHARD + 700
        gen = torch.Generator().manual_seed(rows + masked)
        raw = torch.randn(rows, world_size * SHARD, generator=gen) * 4
        peaks = torch.randint(0, self.vocab, (rows,), generator=gen)
        raw[torch.arange(rows), peaks] += 60
        self.raw = raw.to(device)
        allowed = torch.ones(rows, self.vocab, dtype=torch.bool)
        self.mask = None
        if masked:
            allowed = torch.rand(rows, self.vocab, generator=gen) < 0.5
            allowed[torch.arange(rows), peaks] = True
            self.mask = _bitmask(allowed).to(device)
        # Inkling2's whole rows: gathered, cut to the vocabulary, capped.
        self.whole = tanh_softcap(self.raw[:, : self.vocab], CAP)

    def shard(self, rank: int, world_size: int, rows: int):
        from tokenspeed.runtime.layers.sampled_vocab_shard import SampledVocabShard

        start = rank * SHARD
        return SampledVocabShard(
            logits=self.raw[:rows, start : start + SHARD],
            columns=min(SHARD, self.vocab - start),
            vocab_start=start,
            logit_scale=None,
            softcap=CAP,
            softcap_form="tanh",
            tp_size=world_size,
            tp_group=tuple(range(world_size)),
            gather_full=lambda logits: self.whole[:rows].clone(),
        )


def _backend(world_size: int, vocab: int, device):
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend

    return TritonSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="batch",
            logprob_order="torch",
            enable_output_logprobs=True,
            max_bs=BS,
            max_draft_tokens_per_req=N,
            max_req_pool_size=POOL,
            vocab_size=vocab,
            device=device,
            tp_group=tuple(range(world_size)),
            enable_tp_sync=False,
        )
    )


def _prepare(backend, num_tokens_per_req, device):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    rids = [f"r{i}" for i in range(BS)]
    params = []
    for rid in rids:
        sp = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)
    pool_indices = [2, 5, 7]
    backend.prepare_step(
        admission_ids=list(range(1, len(rids) + 1)),
        request_pool_indices=pool_indices,
        sampling_params_list=params,
        num_tokens_per_req=num_tokens_per_req,
    )
    return torch.tensor(pool_indices, dtype=torch.int64, device=device)


def _info(req_pool_indices, mask, vocab, device):
    from tokenspeed.runtime.grammar.base_grammar_backend import (
        get_apply_vocab_mask_func,
    )
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=torch.arange(
            1000, 1000 + POOL + 1, dtype=torch.int32, device=device
        ),
        draft_q=None,
        vocab_mask=mask,
        apply_vocab_mask=get_apply_vocab_mask_func("xgrammar"),
        vocab_size=vocab,
        device=device,
    )


def _sample(backend, case, logits_output, rows, device):
    req = _prepare(backend, 1, device)
    mask = None if case.mask is None else case.mask[:rows]
    sampled, _ = backend.sample(logits_output, _info(req, mask, case.vocab, device))
    return sampled.clone(), logits_output.next_token_logprobs.clone()


def _verify(backend, case, logits_output, device):
    req = _prepare(backend, N, device)
    # Drafts: each row's best token, wrong from the fifth row of request 1.
    candidates = case.whole.argmax(dim=-1).view(BS, N).roll(1, dims=1).int()
    candidates[1, 4:] = 0
    predict, accept_length = backend.verify(
        logits_output,
        _info(req, case.mask, case.vocab, device),
        candidates,
        tree=None,
    )
    return (
        predict.clone(),
        accept_length.clone(),
        logits_output.next_token_logprobs.clone(),
    )


def _check(rank, world_size, device, group, **_):
    from tokenspeed.runtime.layers.logits_processor_output import (
        LogitsProcessorOutput,
    )

    for masked in (False, True):
        case = _Case(world_size, BS * N, masked, device)
        for step in ("sample", "verify"):
            results = []
            for sharded in (False, True):
                backend = _backend(world_size, case.vocab, device)
                rows = BS if step == "sample" else BS * N
                output = LogitsProcessorOutput(
                    next_token_logits=None if sharded else case.whole[:rows].clone(),
                    vocab_shard=case.shard(rank, world_size, rows) if sharded else None,
                )
                if step == "sample":
                    results.append(_sample(backend, case, output, rows, device))
                else:
                    results.append(_verify(backend, case, output, device))
            for whole, shard in zip(*results):
                assert torch.equal(shard, whole), (step, masked)


def _check_graph(rank, world_size, device, group, **_):
    """The shard logprobs replay from a CUDA graph as they ran eagerly."""
    from tokenspeed.runtime.layers.logits_processor_output import (
        LogitsProcessorOutput,
    )

    case = _Case(world_size, BS * N, True, device)
    backend = _backend(world_size, case.vocab, device)
    shard = case.shard(rank, world_size, BS * N)
    eager = _verify(
        backend, case, LogitsProcessorOutput(None, vocab_shard=shard), device
    )
    req = _prepare(backend, N, device)
    info = _info(req, case.mask, case.vocab, device)
    candidates = case.whole.argmax(dim=-1).view(BS, N).roll(1, dims=1).int()
    candidates[1, 4:] = 0
    output = LogitsProcessorOutput(next_token_logits=None, vocab_shard=shard)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        predict, accept_length = backend.verify(output, info, candidates, tree=None)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(predict, eager[0])
    assert torch.equal(accept_length, eager[1])
    assert torch.equal(output.next_token_logprobs, eager[2])


def _worker(rank, world_size, port, test_fn, error_dict):
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        group = tuple(range(world_size))
        pg_manager.init_process_group(group)
        test_fn(rank=rank, world_size=world_size, device=device, group=group)
        dist.destroy_process_group()
    except Exception:
        import traceback

        error_dict[rank] = traceback.format_exc()


def _run(world_size, test_fn):
    if not torch.cuda.is_available() or world_size > torch.cuda.device_count():
        pytest.skip(f"need {world_size} GPUs")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, _get_open_port(), test_fn, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in error_dict.items()))


@pytest.mark.parametrize("world_size", [2, 4])
def test_shard_logprobs_and_tokens_match_whole_rows(world_size):
    _run(world_size, _check)


def test_shard_logprobs_replay_from_a_cuda_graph():
    _run(2, _check_graph)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
