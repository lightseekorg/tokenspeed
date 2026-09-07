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


"""Shard-local candidate selection, over a real NCCL all-gather.

The single-process test stands a hand-computed peer in for the other rank.
This one runs the actual collective on two GPUs, so the packing, the global-id
offset and the rank-major to row-major fold are checked against the layout
NCCL really delivers rather than against the one the mock reproduces.

Usage:
    python -m pytest test/runtime/distributed/test_vocab_parallel_topk_nccl.py -v
"""

import os
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-2gpu")

VOCAB, HIDDEN, ROWS, TOP_K = 64, 16, 5, 4
SEED = 3


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _inputs(device):
    """The same weight and activations on every rank."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator)
    hidden = torch.randn(ROWS, HIDDEN, generator=generator)
    return weight.to(device), hidden.to(device)


def _quantized_head(shard, device, logits):
    """A head the production predicate accepts as genuinely quantized.

    Only ``apply`` is stood in for -- with a real packed weight the collective
    would still be exercised, but the reference would have to model NVFP4
    rounding. The point here is that the quantized branch survives the real
    all-gather, not that NVFP4 multiplies correctly.
    """
    from torch import nn

    from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4W4A16LinearMethod

    head = nn.Module()
    head.register_parameter(
        "weight",
        nn.Parameter(
            torch.empty((shard, HIDDEN // 2), dtype=torch.uint8, device=device),
            requires_grad=False,
        ),
    )
    head.register_parameter(
        "weight_scale",
        nn.Parameter(
            torch.ones((1,), dtype=torch.float32, device=device), requires_grad=False
        ),
    )
    head.alpha = torch.ones((1,), dtype=torch.float32, device=device)
    head.input_size_per_partition = HIDDEN
    head.output_size_per_partition = shard
    head.quant_method = Nvfp4W4A16LinearMethod(SimpleNamespace(group_size=16))
    head.quant_method.apply = lambda layer, x, bias: logits
    return head


def _selector(rank, world_size, device, group, quantized=False):
    """The production selector, planned against this rank's shard."""
    from tokenspeed.runtime.layers.vocab_parallel_topk import VocabParallelTopK

    weight, _ = _inputs(device)
    shard = VOCAB // world_size
    if quantized:
        _, hidden = _inputs(device)
        rank_logits = torch.matmul(hidden, weight[rank * shard : (rank + 1) * shard].T)
        head = _quantized_head(shard, device, rank_logits)
    else:
        head = SimpleNamespace(
            weight=weight[rank * shard : (rank + 1) * shard], quant_method=None
        )
    head.shard_indices = SimpleNamespace(
        num_org_elements=shard,
        num_org_elements_padded=shard,
        num_added_elements=0,
        org_vocab_start_index=rank * shard,
    )
    selector = VocabParallelTopK(
        head,
        tp_size=world_size,
        tp_rank=rank,
        tp_group=group,
        vocab_size=VOCAB,
        top_k=TOP_K,
        max_rows=ROWS,
        logit_scale=None,
        softcapping=None,
        skip_all_gather=False,
        dp_sampling_enabled=False,
    )
    assert selector.enabled, "this shard geometry must reach the packed gather"
    return selector


def _check_candidates(rank, world_size, device, group, quantized=False, **_):
    """Every rank must agree with one top-k over the whole vocabulary."""
    weight, hidden = _inputs(device)
    selector = _selector(rank, world_size, device, group, quantized=quantized)

    candidate_ids, unary_logits = selector(hidden)

    reference = torch.matmul(hidden, weight.T).float()
    want = torch.topk(reference, TOP_K, dim=-1).values
    got = torch.gather(reference, 1, candidate_ids)
    assert sorted(got.flatten().tolist()) == sorted(want.flatten().tolist())
    torch.testing.assert_close(unary_logits, got, atol=1e-4, rtol=0)


def _check_capture_and_replay(rank, world_size, device, group, **_):
    """The staging and landing buffers must survive graph capture and replay."""
    weight, hidden = _inputs(device)
    selector = _selector(rank, world_size, device, group)

    # Warm: allocates the resident buffers and runs NCCL eagerly once, which
    # capture requires.
    selector(hidden)
    resident = selector._gather_buffers

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_ids, captured_logits = selector(hidden)
    assert selector._gather_buffers is resident, "capture reallocated"

    graph.replay()
    torch.cuda.synchronize()

    reference = torch.matmul(hidden, weight.T).float()
    want = torch.topk(reference, TOP_K, dim=-1).values
    got = torch.gather(reference, 1, captured_ids)
    assert sorted(got.flatten().tolist()) == sorted(want.flatten().tolist())
    torch.testing.assert_close(captured_logits, got, atol=1e-4, rtol=0)


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


def _check_quantized_candidates(rank, world_size, device, group, **kwargs):
    _check_candidates(rank, world_size, device, group, quantized=True, **kwargs)


def test_two_ranks_pick_what_a_whole_vocabulary_topk_would():
    _run(2, _check_candidates)


def test_a_quantized_head_survives_the_real_collective():
    _run(2, _check_quantized_candidates)


def test_two_ranks_keep_their_candidates_through_a_cuda_graph():
    _run(2, _check_capture_and_replay)
