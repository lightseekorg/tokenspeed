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


"""A Lamport mailbox carries a decode all-reduce on its producer: the ring
convolution that consumes it equals ``inkling_ring_sconv`` of the FP32
rank-order sum of the published rows bit for bit (its output and the ring),
re-arms every slot it read, and ``publish_projection`` publishes each rank's
FP32-accumulated product rounded once; eager calls and CUDA-graph replays of
the producer + consumer chain agree.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=4 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, K, TAPS, RING_ROWS, SLOTS = 4608, 1536, 4, 16, 6


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {2, 4, 8},
    reason="launch with torchrun world size 2, 4 or 8",
)

_MAILBOX = None


def _setup():
    global _MAILBOX
    from tokenspeed_kernel.ops.communication.mailbox import Mailbox, mailbox_supported

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    if not mailbox_supported(device):
        pytest.skip("mailboxes need Blackwell, the CuTe DSL and symmetric memory")
    if _MAILBOX is None:
        try:
            _MAILBOX = Mailbox(dist.group.WORLD, width=H, device=device)
        except RuntimeError as exc:
            pytest.skip(f"no multicast memory: {exc}")
    return dist.get_rank(), device, _MAILBOX


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


def _conv_inputs(device):
    # The taps and the ring are replicated: every rank draws the same ones.
    shared = torch.Generator(device=device).manual_seed(7)
    weight = (torch.randn(H, TAPS, generator=shared, device=device) * 0.5).to(
        torch.bfloat16
    )
    ring = torch.randn(SLOTS, RING_ROWS, H, generator=shared, device=device).to(
        torch.bfloat16
    )
    return weight, ring


def _checkpoint(batch, device):
    # A decode convolution without a live publish reads neither.
    requests = batch["cache_indices"].numel()
    return (
        torch.zeros(requests, 2, dtype=torch.int32, device=device),
        torch.zeros(1, TAPS - 1, H, dtype=torch.bfloat16, device=device),
    )


def _plain_conv(x, weight, ring, batch):
    from tokenspeed_kernel.ops.conv import inkling_ring_sconv

    return inkling_ring_sconv(
        x,
        weight,
        ring,
        batch["cu_seqlens"],
        batch["seq_idx"],
        batch["cache_indices"],
        batch["has_initial_state"],
        batch["seq_lens"],
        *_checkpoint(batch, x.device),
        None,
        num_extends=0,
        publish_live=False,
        dynamic_weights=None,
        output_scale=TAPS**-0.5 * 1.25,
        token_range=None,
    )


def _mailbox_conv(mailbox, parity, tokens, weight, ring, batch):
    from tokenspeed_kernel.ops.conv import inkling_ring_sconv_mailbox

    return inkling_ring_sconv_mailbox(
        mailbox.slots(parity),
        tokens,
        weight,
        ring,
        batch["cu_seqlens"],
        batch["seq_idx"],
        batch["cache_indices"],
        batch["has_initial_state"],
        batch["seq_lens"],
        *_checkpoint(batch, ring.device),
        output_scale=TAPS**-0.5 * 1.25,
    )


def _rank_order_sum(rows):
    """FP32 sum of ``[tp, T, H]`` BF16 rows in rank order from +0.0, rounded once."""
    total = torch.zeros(rows.shape[1:], dtype=torch.float32, device=rows.device)
    for row in rows:
        total += row.float()
    return total.to(torch.bfloat16)


def _armed(slots, tokens):
    return bool((slots[:, :tokens].view(torch.int16) == -32768).all())


# (chunk, prefixes, slots, has_initial_state): one verify chunk; several
# requests with a padded one; a fresh request whose taps start at position 0.
BATCHES = [
    (8, [37], [2], [True]),
    (4, [5, 90, 0], [0, -1, 4], [True, True, False]),
    (2, [1, 2, 300, 7], [1, 3, 5, 0], [True, False, True, True]),
]


@pytest.mark.parametrize("case", range(len(BATCHES)))
def test_mailbox_convolution_equals_the_rank_order_sum_then_the_convolution(case):
    rank, device, mailbox = _setup()
    chunk, prefixes, slots, has_state = BATCHES[case]
    batch = _batch(chunk, prefixes, slots, has_state, device)
    tokens = chunk * len(prefixes)
    weight, ring = _conv_inputs(device)
    own = torch.Generator(device=device).manual_seed(100 + rank)
    partial = torch.randn(tokens, H, generator=own, device=device)
    # Rows near the bottom of the FP32 range exercise denormal products.
    partial[tokens - 1] *= 1e-37
    partial = partial.to(torch.bfloat16)
    # Producers publish +0.0 for -0.0 (the armed value).
    partial[partial == 0] = 0.0
    for peer in range(mailbox.tp):
        mailbox.peer_slots(peer, 0)[rank, :tokens].copy_(partial)
    torch.cuda.synchronize()
    dist.barrier()
    published = mailbox.slots(0)[:, :tokens].clone()
    ring_ref, ring_mailbox = ring.clone(), ring.clone()
    expected = _plain_conv(_rank_order_sum(published), weight, ring_ref, batch)
    actual = _mailbox_conv(mailbox, 0, tokens, weight, ring_mailbox, batch)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(ring_mailbox.view(torch.int16), ring_ref.view(torch.int16))
    assert _armed(mailbox.slots(0), tokens)
    dist.barrier()


def test_published_projection_is_the_fp32_product_rounded_once():
    from tokenspeed_kernel.ops.communication.mailbox import publish_projection

    rank, device, mailbox = _setup()
    batch = _batch(*BATCHES[0], device)
    tokens = 8
    weight, ring = _conv_inputs(device)
    own = torch.Generator(device=device).manual_seed(200 + rank)
    x = (torch.randn(tokens, K, generator=own, device=device) * 0.5).to(torch.bfloat16)
    w = (torch.randn(H, K, generator=own, device=device) * 0.02).to(torch.bfloat16)
    publish_projection(x, w, mailbox, 1)
    torch.cuda.synchronize()
    dist.barrier()
    published = mailbox.slots(1)[:, :tokens].clone()
    # Within one BF16 step of the FP32 product (the accumulation order differs).
    product = (x.float() @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(
        published[rank].float(), product.float(), rtol=2**-7, atol=1e-6
    )
    assert not bool((published.view(torch.int16) == -32768).any())
    ring_ref, ring_mailbox = ring.clone(), ring.clone()
    expected = _plain_conv(_rank_order_sum(published), weight, ring_ref, batch)
    actual = _mailbox_conv(mailbox, 1, tokens, weight, ring_mailbox, batch)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert _armed(mailbox.slots(1), tokens)
    dist.barrier()


def test_graph_replays_of_the_chain_match_eager_calls():
    """Two chained producer + consumer pairs (one per parity) captured once
    replay with fresh inputs and equal the same chain run eagerly."""
    from tokenspeed_kernel.ops.communication.mailbox import publish_projection

    rank, device, mailbox = _setup()
    chunk, prefixes, slots, has_state = BATCHES[0]
    batch = _batch(chunk, prefixes, slots, has_state, device)
    tokens = chunk
    weight, ring = _conv_inputs(device)
    own = torch.Generator(device=device).manual_seed(300 + rank)
    w = [
        (torch.randn(H, K, generator=own, device=device) * 0.02).to(torch.bfloat16)
        for _ in range(2)
    ]
    w_in = (torch.randn(K, H, generator=own, device=device) * 0.02).to(torch.bfloat16)
    static_x = torch.empty(tokens, K, dtype=torch.bfloat16, device=device)
    static_ring = ring.clone()

    def chain():
        publish_projection(static_x, w[0], mailbox, 0)
        y = _mailbox_conv(mailbox, 0, tokens, weight, static_ring, batch)
        x2 = (y @ w_in.T).contiguous()
        publish_projection(x2, w[1], mailbox, 1)
        return _mailbox_conv(mailbox, 1, tokens, weight, static_ring, batch)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        static_x.normal_()
        chain()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = chain()
    for step in range(3):
        fresh = torch.Generator(device=device).manual_seed(1000 * step + rank)
        x = (torch.randn(tokens, K, generator=fresh, device=device) * 0.5).to(
            torch.bfloat16
        )
        static_x.copy_(x)
        static_ring.copy_(ring)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_ring = out.clone(), static_ring.clone()
        dist.barrier()
        static_x.copy_(x)
        static_ring.copy_(ring)
        eager = chain()
        torch.cuda.synchronize()
        assert torch.equal(replayed.view(torch.int16), eager.view(torch.int16))
        assert torch.equal(
            replayed_ring.view(torch.int16), static_ring.view(torch.int16)
        )
        assert _armed(mailbox.slots(0), tokens) and _armed(mailbox.slots(1), tokens)
        dist.barrier()


def test_published_route_fold_lands_in_every_rank_slot():
    """A route fold published into the mailbox puts each rank's local fold
    (``-0.0`` as ``+0.0``) in that rank's slot everywhere; the mailbox
    convolution then completes the sum."""
    from tokenspeed_kernel.ops.communication.mailbox import MailboxTarget
    from tokenspeed_kernel.ops.moe._triton.block_scaled import _fold_routes

    rank, device, mailbox = _setup()
    batch = _batch(*BATCHES[1], device)
    tokens, top_k, local_experts = 12, 8, 4
    weight, ring = _conv_inputs(device)
    own = torch.Generator(device=device).manual_seed(400 + rank)
    rows = torch.randn(tokens * top_k, H, generator=own, device=device).to(
        torch.bfloat16
    )
    rows[:top_k, :32] = -0.0
    ids = torch.randint(
        0, 16, (tokens, top_k), generator=own, device=device, dtype=torch.int32
    )
    expert_start = rank * local_experts
    local = _fold_routes(
        rows,
        None,
        ids,
        expert_start=expert_start,
        num_experts=local_experts,
        fold_into=None,
    )
    assert (
        _fold_routes(
            rows,
            None,
            ids,
            expert_start=expert_start,
            num_experts=local_experts,
            fold_into=MailboxTarget(mailbox, 0),
        )
        is None
    )
    torch.cuda.synchronize()
    dist.barrier()
    published = mailbox.slots(0)[:, :tokens].clone()
    expected = local[:, 0].clone()
    expected[expected == 0] = 0.0
    assert torch.equal(published[rank].view(torch.int16), expected.view(torch.int16))
    ring_ref, ring_mailbox = ring.clone(), ring.clone()
    reference = _plain_conv(_rank_order_sum(published), weight, ring_ref, batch)
    actual = _mailbox_conv(mailbox, 0, tokens, weight, ring_mailbox, batch)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    assert _armed(mailbox.slots(0), tokens)
    dist.barrier()
