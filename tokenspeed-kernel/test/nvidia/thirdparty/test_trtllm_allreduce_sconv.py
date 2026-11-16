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

"""``allreduce_sconv`` releases the decode ring convolution before its
all-reduce ends; the convolution must still see the finished sum: the output
and the ring it updates equal the workspace all-reduce followed by
``inkling_ring_sconv`` bit for bit, eagerly and under CUDA-graph replay,
sharing the plain all-reduce's workspace.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=4 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, TAPS, RING_ROWS, SLOTS = 4608, 4, 16, 6


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {2, 4, 8, 16},
    reason="launch with torchrun world size 2, 4, 8 or 16",
)


def _setup():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    from tokenspeed_kernel.ops.communication.trtllm import (
        ensure_workspace_initialized,
    )

    try:
        armed = ensure_workspace_initialized(
            rank=dist.get_rank(), group=dist.group.WORLD, max_token_num=64, hidden_dim=H
        )
    except RuntimeError as exc:
        pytest.skip(f"trtllm fusion workspace unavailable: {exc}")
    if not armed:
        pytest.skip("trtllm fusion workspace unavailable")
    return dist.get_rank(), torch.device("cuda", local)


def _batch(chunk, prefixes, slots, has_state, device):
    """A decode batch of uniform ``chunk``-row requests after ``prefixes``."""
    requests = len(prefixes)
    cu_seqlens = torch.arange(0, (requests + 1) * chunk, chunk, dtype=torch.int32)
    return dict(
        cu_seqlens=cu_seqlens.to(device),
        seq_idx=torch.arange(requests, dtype=torch.int32)
        .repeat_interleave(chunk)
        .to(device),
        cache_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        has_initial_state=torch.tensor(has_state, dtype=torch.bool, device=device),
        seq_lens=torch.tensor(
            [p + chunk for p in prefixes], dtype=torch.int32, device=device
        ),
    )


def _inputs(rank, tokens, device, tiny_rows):
    # The taps and the ring are replicated: every rank draws the same ones.
    shared = torch.Generator(device=device).manual_seed(7)
    weight = (torch.randn(H, TAPS, generator=shared, device=device) * 0.5).to(
        torch.bfloat16
    )
    ring = torch.randn(SLOTS, RING_ROWS, H, generator=shared, device=device).to(
        torch.bfloat16
    )
    own = torch.Generator(device=device).manual_seed(100 + rank)
    x = torch.randn(tokens, H, generator=own, device=device)
    # Rows near the bottom of the FP32 range exercise denormal products.
    x[tiny_rows] *= 1e-37
    x[0, :64] = -0.0
    return weight, ring, x.to(torch.bfloat16)


def _unused_checkpoint(batch, device):
    # A decode convolution without a live publish reads neither.
    requests = batch["cache_indices"].numel()
    return (
        torch.zeros(requests, 2, dtype=torch.int32, device=device),
        torch.zeros(1, TAPS - 1, H, dtype=torch.bfloat16, device=device),
    )


def _reference(x, weight, ring, batch, fp32_acc):
    from tokenspeed_kernel.ops.communication.trtllm import trtllm_workspace_allreduce
    from tokenspeed_kernel.ops.conv import inkling_ring_sconv

    summed = trtllm_workspace_allreduce(x, dist.group.WORLD, fp32_acc=fp32_acc)
    return inkling_ring_sconv(
        summed,
        weight,
        ring,
        batch["cu_seqlens"],
        batch["seq_idx"],
        batch["cache_indices"],
        batch["has_initial_state"],
        batch["seq_lens"],
        *_unused_checkpoint(batch, x.device),
        None,
        num_extends=0,
        publish_live=False,
        dynamic_weights=None,
        output_scale=TAPS**-0.5 * 1.25,
        token_range=None,
    )


def _fused(x, weight, ring, batch, fp32_acc):
    from tokenspeed_kernel.ops.communication.trtllm import allreduce_sconv

    return allreduce_sconv(
        x,
        weight,
        ring,
        batch["cu_seqlens"],
        batch["seq_idx"],
        batch["cache_indices"],
        batch["has_initial_state"],
        batch["seq_lens"],
        *_unused_checkpoint(batch, x.device),
        output_scale=TAPS**-0.5 * 1.25,
        fp32_acc=fp32_acc,
        group=dist.group.WORLD,
    )


# (chunk, prefixes, slots, has_initial_state): one long chunk; several
# requests with a padded one; a fresh request whose taps start at position 0.
BATCHES = [
    (8, [37], [2], [True]),
    (4, [5, 90, 0], [0, -1, 4], [True, True, False]),
    (2, [1, 2, 300, 7], [1, 3, 5, 0], [True, False, True, True]),
]


@pytest.mark.parametrize("fp32_acc", [False, True])
@pytest.mark.parametrize("case", range(len(BATCHES)))
def test_fused_convolution_equals_the_all_reduce_then_the_convolution(case, fp32_acc):
    from tokenspeed_kernel.ops.communication.trtllm import supports_allreduce_sconv

    rank, device = _setup()
    chunk, prefixes, slots, has_state = BATCHES[case]
    batch = _batch(chunk, prefixes, slots, has_state, device)
    tokens = chunk * len(prefixes)
    weight, ring, x = _inputs(rank, tokens, device, tiny_rows=[tokens - 1])
    assert supports_allreduce_sconv(x, weight, dist.group.WORLD)
    ring_ref, ring_fused = ring.clone(), ring.clone()
    expected = _reference(x, weight, ring_ref, batch, fp32_acc)
    actual = _fused(x, weight, ring_fused, batch, fp32_acc)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(ring_fused.view(torch.int16), ring_ref.view(torch.int16))


def test_graph_replays_match_and_share_the_workspace_rotation():
    """Captured fused calls between plain all-reduces replay with fresh
    inputs; each replay equals its eager reference."""
    from tokenspeed_kernel.ops.communication.trtllm import trtllm_workspace_allreduce

    rank, device = _setup()
    chunk, prefixes, slots, has_state = BATCHES[1]
    batch = _batch(chunk, prefixes, slots, has_state, device)
    tokens = chunk * len(prefixes)
    weight, ring, x = _inputs(rank, tokens, device, tiny_rows=[1])
    static_x, static_ring = x.clone(), ring.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _fused(static_x, weight, static_ring, batch, False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plain = trtllm_workspace_allreduce(static_x, dist.group.WORLD, fp32_acc=False)
        out = _fused(static_x, weight, static_ring, batch, False)
    for step in range(3):
        fresh = torch.Generator(device=device).manual_seed(1000 * step + rank)
        static_x.copy_(torch.randn(tokens, H, generator=fresh, device=device))
        static_ring.copy_(ring)
        graph.replay()
        ring_ref = ring.clone()
        expected = _reference(static_x, weight, ring_ref, batch, False)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(static_ring.view(torch.int16), ring_ref.view(torch.int16))
        reduced = trtllm_workspace_allreduce(static_x, dist.group.WORLD, fp32_acc=False)
        assert torch.equal(plain.view(torch.int16), reduced.view(torch.int16))
