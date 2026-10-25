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

"""The data plane's side of the EPD prefill receive: the receive pools'
geometry, publishing landed rows onto their items (with the row-shard
rebuild), reporting copied-out slots, and the invariant that every
handshaked item a forward scatters was received. The host's receiver
(``ts-pd``'s ``epd::receiver``) is tested in Rust; the rebuild broadcasts
are recorded instead of run, as no process group exists here.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CPU tests scheduled in runtime-1gpu because they import the full runtime;
# the copy-event test needs the GPU.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.epd import prefill_receive, recv_pool  # noqa: E402
from tokenspeed.runtime.epd.prefill_receive import (  # noqa: E402
    EmbeddingPublisher,
    ReceivePools,
    ShardGroup,
    assert_embeddings_received,
    build_epd_receive,
    shard_rows,
)
from tokenspeed.runtime.multimodal.inputs import (  # noqa: E402
    Modality,
    MultimodalDataItem,
)

HIDDEN = 8


def _item(rows: int) -> MultimodalDataItem:
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, rows - 1)])
    item.encode_handshake = {
        "bootstrap_host": "h",
        "bootstrap_port": 1,
        "bootstrap_room": 2,
    }
    return item


def _publisher(pools, *, num_deepstack=0, shards=None, device="cpu"):
    return EmbeddingPublisher(
        pools,
        hidden=HIDDEN,
        num_deepstack=num_deepstack,
        dtype=torch.float32,
        device=device,
        shards=shards,
    )


def _land(pools, slot, rows, *, num_deepstack=0, base=0.0):
    """Write ``rows`` distinct rows into ``slot`` as an encode node would, and
    return them (and the deepstack rows)."""
    main = torch.arange(rows * HIDDEN, dtype=torch.float32).view(rows, HIDDEN) + base
    pools.rows(slot, rows, HIDDEN, torch.float32, deepstack=False).copy_(main)
    deep = None
    if num_deepstack:
        width = HIDDEN * num_deepstack
        deep = -torch.arange(rows * width, dtype=torch.float32).view(rows, width) - base
        pools.rows(slot, rows, width, torch.float32, deepstack=True).copy_(deep)
    return main, deep


def _record_broadcasts(monkeypatch):
    calls = []

    def broadcast(tensor, src=None, group=None):
        calls.append((tensor, src, group))

    monkeypatch.setattr(prefill_receive.dist, "broadcast", broadcast)
    return calls


# --- the shard geometry ---


@pytest.mark.parametrize("span", [1, 2, 3, 7, 8, 128, 129])
@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_shards_tile_the_span_in_rank_order(span, size):
    cursor = 0
    for rank in range(size):
        start, count = shard_rows(span, rank, size)
        assert start == cursor
        assert count >= 0
        cursor += count
    assert cursor == span
    counts = [shard_rows(span, rank, size)[1] for rank in range(size)]
    assert max(counts) - min(counts) <= 1


# --- the pools ---


@pytest.mark.parametrize("num_deepstack", [0, 1, 3])
@pytest.mark.parametrize("slot_bytes", [1 << 20, 1000, 4096 + 7])
def test_a_slot_holds_an_item_and_its_deepstack_rows_within_its_bytes(
    num_deepstack, slot_bytes
):
    main, deep = recv_pool.split_slot(slot_bytes, num_deepstack)
    assert main + deep <= slot_bytes
    assert deep == main * num_deepstack
    # Slot starts stay aligned for any element type.
    assert main % 256 == 0
    pools = ReceivePools(
        slots=3, slot_bytes=slot_bytes, num_deepstack=num_deepstack, device="cpu"
    )
    assert pools.buffer.numel() <= 3 * slot_bytes
    assert (pools.slot_bytes, pools.deepstack_slot_bytes) == (main, deep)
    if num_deepstack:
        assert pools.deepstack_base == pools.base + 3 * main
    else:
        assert pools.deepstack_base == 0


def test_slots_address_disjoint_bytes_of_their_pools():
    pools = ReceivePools(slots=2, slot_bytes=8192, num_deepstack=1, device="cpu")
    width = HIDDEN
    rows = pools.slot_bytes // (width * 4)
    views = [
        pools.rows(slot, rows, width, torch.float32, deepstack=deepstack)
        for deepstack in (False, True)
        for slot in range(2)
    ]
    starts = [view.data_ptr() - pools.base for view in views]
    assert starts == [0, pools.slot_bytes, 2 * pools.slot_bytes, 3 * pools.slot_bytes]
    for view, value in zip(views, range(4)):
        view.fill_(value)
    for view, value in zip(views, range(4)):
        assert torch.all(view == value)


# --- publishing ---


def test_publish_copies_each_landed_item_out_of_its_slot():
    pools = ReceivePools(slots=2, slot_bytes=4096, num_deepstack=0, device="cpu")
    publisher = _publisher(pools)
    first, second = _item(5), _item(3)
    rows_a, _ = _land(pools, 1, 5)
    rows_b, _ = _land(pools, 0, 3, base=1000.0)

    publisher.publish(
        [([first, second], [(0, 1, 0, 5, 5, False), (1, 0, 0, 3, 3, False)])]
    )

    assert torch.equal(first.encoded, rows_a)
    assert torch.equal(second.encoded, rows_b)
    assert first.encoded_deepstack is None
    # A copy, not a view: the slot takes the next item's rows.
    pools.buffer.zero_()
    assert torch.equal(first.encoded, rows_a)
    # Copies on the host finish at once: both slots return, once.
    assert sorted(publisher.released()) == [0, 1]
    assert publisher.released() == []


def test_publish_takes_the_deepstack_rows_from_the_same_slot():
    pools = ReceivePools(slots=2, slot_bytes=8192, num_deepstack=3, device="cpu")
    publisher = _publisher(pools, num_deepstack=3)
    item = _item(4)
    _land(pools, 0, 4, num_deepstack=3, base=500.0)
    rows, deep = _land(pools, 1, 4, num_deepstack=3)

    publisher.publish([([item], [(0, 1, 0, 4, 4, False)])])

    assert torch.equal(item.encoded, rows)
    assert item.encoded_deepstack.shape == (4, HIDDEN * 3)
    assert torch.equal(item.encoded_deepstack, deep)


def test_a_shard_lands_at_its_rows_and_the_ranks_rebuild_the_rest(monkeypatch):
    pools = ReceivePools(slots=1, slot_bytes=8192, num_deepstack=1, device="cpu")
    shards = ShardGroup(nccl_group="nccl", group_ranks=(7, 9))
    publisher = _publisher(pools, num_deepstack=1, shards=shards)
    calls = _record_broadcasts(monkeypatch)
    item = _item(5)
    # Rank 1 of 2 receives rows [3, 5) of the image's 5, packed at the slot's
    # start.
    start, count = shard_rows(5, 1, 2)
    assert (start, count) == (3, 2)
    rows, deep = _land(pools, 0, count, num_deepstack=1)

    publisher.publish([([item], [(0, 0, start, count, 5, True)])])

    assert item.encoded.shape == (5, HIDDEN)
    assert torch.equal(item.encoded[3:5], rows)
    assert torch.equal(item.encoded_deepstack[3:5], deep)
    # Each rank's shard comes from that rank, the rows' then the deepstack
    # rows', into the published tensors.
    placed = [
        (
            tensor.data_ptr() - owner.data_ptr(),
            tensor.shape[0],
            src,
            group,
        )
        for (tensor, src, group), owner in zip(
            calls,
            [item.encoded] * 2 + [item.encoded_deepstack] * 2,
        )
    ]
    # One deepstack level: its rows are as wide as the item's.
    row = HIDDEN * 4
    assert placed == [
        (0, 3, 7, "nccl"),
        (3 * row, 2, 9, "nccl"),
        (0, 3, 7, "nccl"),
        (3 * row, 2, 9, "nccl"),
    ]
    assert publisher.released() == [0]


def test_an_empty_shard_takes_no_slot_but_joins_the_rebuild(monkeypatch):
    pools = ReceivePools(slots=1, slot_bytes=4096, num_deepstack=0, device="cpu")
    shards = ShardGroup(nccl_group="nccl", group_ranks=(0, 1, 2, 3))
    publisher = _publisher(pools, shards=shards)
    calls = _record_broadcasts(monkeypatch)
    item = _item(2)
    # Rank 3 of 4 receives nothing of a 2-row image.
    assert shard_rows(2, 3, 4) == (2, 0)

    publisher.publish([([item], [(0, None, 2, 0, 2, True)])])

    assert item.encoded.shape == (2, HIDDEN)
    assert [(tensor.shape[0], src) for tensor, src, _ in calls] == [(1, 0), (1, 1)]
    assert publisher.released() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_slot_returns_only_once_its_copy_finished():
    pools = ReceivePools(slots=1, slot_bytes=1 << 20, num_deepstack=0, device="cuda")
    publisher = _publisher(pools, device="cuda")
    item = _item(8)
    rows, _ = _land(pools, 0, 8)
    # Hold the stream behind a long kernel so the copy is still queued.
    torch.cuda.synchronize()
    torch.cuda._sleep(1 << 30)
    publisher.publish([([item], [(0, 0, 0, 8, 8, False)])])
    assert publisher.released() == []
    torch.cuda.synchronize()
    assert publisher.released() == [0]
    assert torch.equal(item.encoded.cpu(), rows)


# --- building the receive ---


def _server_args(mode, *, first_stage=True):
    return SimpleNamespace(
        disaggregation_mode=mode,
        mapping=SimpleNamespace(is_first_pp_rank=first_stage),
    )


@pytest.mark.parametrize(
    "mode, active, first_stage",
    [
        ("decode", True, True),
        ("null", True, True),
        ("encode", True, True),
        ("prefill", False, True),
        # A later pipeline stage follows the first's decisions.
        ("prefill", True, False),
    ],
)
def test_only_a_multimodal_prefill_first_stage_receives(
    monkeypatch, mode, active, first_stage
):
    """The facts callable must not fire elsewhere: reading the vision tower's
    dtype raises on a text-only model, and every text-only PD node builds
    its rank through here."""
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOTS", "2")
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOT_MB", "1")

    def facts():  # pragma: no cover — reaching this is the failure
        raise AssertionError("encoder facts read on a node that does not receive")

    server_args = _server_args(mode, first_stage=first_stage)
    receive = build_epd_receive(
        server_args,
        model_config=SimpleNamespace(is_multimodal_active=active),
        encoder_model_facts=facts,
        mapping=server_args.mapping,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pg_manager=None,
    )

    assert receive is None
    assert recv_pool.recv_pool_bytes(server_args, active) == 0


@pytest.mark.parametrize("slots, slot_mb", [(2, 1), (0, 1), (2, 0)])
@pytest.mark.parametrize("num_deepstack", [0, 3])
def test_the_receive_builds_the_pool_the_budget_left_out(
    monkeypatch, slots, slot_mb, num_deepstack
):
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOTS", str(slots))
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOT_MB", str(slot_mb))
    monkeypatch.setenv("TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT", "30")
    server_args = _server_args("prefill")
    facts = SimpleNamespace(
        device="cpu", hidden=16, num_deepstack=num_deepstack, dtype=torch.bfloat16
    )

    receive = build_epd_receive(
        server_args,
        model_config=SimpleNamespace(is_multimodal_active=True),
        encoder_model_facts=lambda: facts,
        mapping=server_args.mapping,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pg_manager=None,
    )

    settings = receive.settings
    budget = recv_pool.recv_pool_bytes(server_args, True)
    pool = settings.slots * (settings.slot_bytes + settings.deepstack_slot_bytes)
    assert pool <= budget
    assert (pool > 0) == (slots > 0 and slot_mb > 0)
    if pool:
        assert settings.pool_base != 0
        assert (settings.deepstack_base != 0) == (num_deepstack > 0)
    assert settings.hidden == 16
    assert settings.num_deepstack == num_deepstack
    assert (settings.dtype, settings.element_size) == ("torch.bfloat16", 2)
    assert (settings.tp_size, settings.tp_rank, settings.shard) == (1, 0, False)
    assert settings.timeout_s == 30.0


# --- the forward's check ---


def _context(*mm_items, prefix: int = 0, seq: int = 8):
    return SimpleNamespace(
        has_extend_inputs=lambda: True,
        mm_inputs=[SimpleNamespace(mm_items=list(mm_items))],
        extend_prefix_lens=[prefix],
        extend_seq_lens=[seq],
    )


def _publisher_stub():
    return _publisher(None)


def test_assert_embeddings_received_raises_on_unreceived_handshaked_item():
    with pytest.raises(RuntimeError, match="un-received"):
        assert_embeddings_received(_publisher_stub(), _context(_item(4)))


def test_assert_embeddings_received_accepts_filled_items_and_other_ranks():
    filled = _item(4)
    filled.encoded = torch.zeros(4, HIDDEN)
    plain = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, 3)])
    assert_embeddings_received(_publisher_stub(), _context(filled, plain))
    # A rank that does not receive: a no-op even with an un-received item.
    assert_embeddings_received(None, _context(_item(4)))


def test_assert_embeddings_received_ignores_items_an_earlier_chunk_prefilled():
    """A chunked prompt: the image's tokens [0, 3] were all prefilled by the
    first chunk, which dropped the encoding; the second chunk [8, 16)
    scatters nothing of it. An item still overlapping the chunk must hold its
    embedding."""
    assert_embeddings_received(_publisher_stub(), _context(_item(4), prefix=8, seq=8))
    straddling = _item(4)
    straddling.offsets = [(0, 3), (10, 12)]
    with pytest.raises(RuntimeError, match="un-received"):
        assert_embeddings_received(
            _publisher_stub(), _context(straddling, prefix=8, seq=8)
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
