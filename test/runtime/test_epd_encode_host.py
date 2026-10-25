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

"""An EPD encode rank's data plane: admitting requests (their cache lookups
and shared-memory pixels, opened on every rank before any consumes them),
embedding batches and copying each item's rows into its send slot, and
reporting the finished copies. The host's loop and sender are tested in
Rust (``ts-control``'s ``encode``, ``ts-pd``'s ``epd::sender``); the vision
tower here is a stand-in that returns the items' pixels.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta
from multiprocessing import shared_memory

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# CPU tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.cache.embedding_cache import (  # noqa: E402
    EmbeddingCache,
    TieredEmbeddingCache,
)
from tokenspeed.runtime.engine.io_struct import MsgpackEncoder  # noqa: E402
from tokenspeed.runtime.epd.encode_host import (  # noqa: E402
    EncodeDataPlane,
    SendRing,
    assign_encoded_embeddings,
    embedding_cache_bytes,
    make_embedding_cache,
)
from tokenspeed.runtime.epd.encode_worker import EncodeRequest  # noqa: E402
from tokenspeed.runtime.multimodal.embedder import EncoderSpec  # noqa: E402
from tokenspeed.runtime.multimodal.inputs import (  # noqa: E402
    Modality,
    MultimodalDataItem,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle  # noqa: E402
from tokenspeed.runtime.utils.env import envs  # noqa: E402

HIDDEN = 4


class _Tower:
    """A vision tower whose output rows are the items' pixels."""

    num_deepstack_embeddings = 0

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[list[int]] = []
        self._fail = fail

    def image_encoder(self, items):
        self.calls.append([item.hash for item in items])
        if self._fail is not None:
            raise self._fail
        return torch.cat([item.feature for item in items])

    def get_multimodal_encoder_specs(self):
        return {Modality.IMAGE: EncoderSpec(self.image_encoder)}


def _item(image: int, rows: int) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=image,
        offsets=[(0, rows - 1)],
        feature=torch.full((rows, HIDDEN), float(image)),
    )


def _frames(*items: MultimodalDataItem, room: int = 1) -> list[bytes]:
    request = EncodeRequest(
        request_id=f"r{room}",
        bootstrap_host="h",
        bootstrap_port=1,
        bootstrap_room=room,
        items=list(items),
    )
    return [bytes(frame) for frame in MsgpackEncoder().encode(request)]


def _plane(tower, *, slots=4, slot_bytes=1024, cache_bytes=1 << 20, deepstack=False):
    ring = SendRing(
        slots=slots, slot_bytes=slot_bytes, deepstack=deepstack, device="cpu"
    )
    plane = EncodeDataPlane(
        model=tower,
        cache=EmbeddingCache(cache_bytes),
        ring=ring,
        device="cpu",
        tp_cpu_group=None,
    )
    return plane, ring


def _slot(ring: SendRing, slot: int, rows: int) -> torch.Tensor:
    start = slot * ring.slot_bytes
    return (
        ring.buffer[start : start + rows * HIDDEN * 4]
        .view(torch.float32)
        .view(rows, HIDDEN)
    )


# --- the data plane ---


def test_a_batch_runs_the_tower_once_and_lands_in_its_slots():
    tower = _Tower()
    plane, ring = _plane(tower)
    assert plane.admit(
        [(1, _frames(_item(7, 3), room=1)), (2, _frames(_item(8, 2), room=2))]
    ) == [
        False,
        False,
    ]
    plane.encode([(1, 2), (2, 0)])
    assert tower.calls == [[7, 8]]
    # Copies on the host finish at once.
    assert plane.encoded() == [
        (1, None, 3, HIDDEN, "torch.float32", 3 * HIDDEN * 4, 0, 0),
        (2, None, 2, HIDDEN, "torch.float32", 2 * HIDDEN * 4, 0, 0),
    ]
    assert plane.encoded() == []
    assert torch.equal(_slot(ring, 2, 3), torch.full((3, HIDDEN), 7.0))
    assert torch.equal(_slot(ring, 0, 2), torch.full((2, HIDDEN), 8.0))


def test_a_cached_image_skips_the_tower_but_still_lands():
    tower = _Tower()
    plane, ring = _plane(tower)
    plane.admit([(1, _frames(_item(7, 3), room=1))])
    plane.encode([(1, 0)])
    plane.encoded()
    # The same image again: the cache answers at admission.
    assert plane.admit([(2, _frames(_item(7, 3), room=2))]) == [True]
    plane.encode([(2, 1)])
    assert tower.calls == [[7]]
    assert plane.encoded()[0][:3] == (2, None, 3)
    assert torch.equal(_slot(ring, 1, 3), torch.full((3, HIDDEN), 7.0))


def test_a_cached_image_keeps_its_deepstack_rows():
    class DeepTower(_Tower):
        num_deepstack_embeddings = 1

        def image_encoder(self, items):
            rows = super().image_encoder(items)
            return torch.cat([rows, -rows], dim=1)

        @staticmethod
        def separate_deepstack_embeds(embedding):
            return embedding[:, :HIDDEN], embedding[:, HIDDEN:]

    plane, ring = _plane(DeepTower(), deepstack=True)
    plane.admit([(1, _frames(_item(5, 2), room=1))])
    plane.encode([(1, 0)])
    assert plane.encoded()[0][6:] == (HIDDEN, 2 * HIDDEN * 4)
    assert plane.admit([(2, _frames(_item(5, 2), room=2))]) == [True]
    plane.encode([(2, 1)])
    # Both halves go out on a hit.
    assert plane.encoded()[0][6:] == (HIDDEN, 2 * HIDDEN * 4)
    deep = ring.deepstack_buffer[ring.slot_bytes : ring.slot_bytes + 2 * HIDDEN * 4]
    assert torch.equal(
        deep.view(torch.float32).view(2, HIDDEN), torch.full((2, HIDDEN), -5.0)
    )


def test_a_tower_failure_fails_only_its_batch():
    plane, _ = _plane(_Tower(fail=RuntimeError("bad grid")))
    plane.admit([(1, _frames(_item(7, 3), room=1))])
    plane.encode([(1, 0)])
    ((admission, failure, *_),) = plane.encoded()
    assert admission == 1
    assert "bad grid" in failure


def test_an_item_larger_than_its_slot_fails_alone():
    tower = _Tower()
    # 64 bytes a slot: four rows of four floats.
    plane, ring = _plane(tower, slot_bytes=64)
    plane.admit([(1, _frames(_item(7, 5), room=1)), (2, _frames(_item(8, 4), room=2))])
    plane.encode([(1, 0), (2, 1)])
    big, fits = plane.encoded()
    assert big[0] == 1 and "TOKENSPEED_EPD_ENCODE_RING_SLOT_MB" in big[1]
    assert fits[:2] == (2, None)
    assert torch.equal(_slot(ring, 1, 4), torch.full((4, HIDDEN), 8.0))


def test_a_request_with_two_items_is_taken_but_not_kept():
    plane, _ = _plane(_Tower())
    # The host refuses it and releases it; the cache is not consulted.
    assert plane.admit([(1, _frames(_item(7, 1), _item(8, 1), room=1))]) == [False]
    plane.release([1])
    with pytest.raises(KeyError):
        plane.encode([(1, 0)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_copy_is_reported_only_once_it_finished():
    tower = _Tower()
    ring = SendRing(slots=1, slot_bytes=1024, deepstack=False, device="cuda")
    plane = EncodeDataPlane(
        model=tower,
        cache=EmbeddingCache(1 << 20),
        ring=ring,
        device="cuda",
        tp_cpu_group=None,
    )
    item = _item(7, 3)
    item.feature = item.feature.cuda()
    plane._items[1] = item
    torch.cuda.synchronize()
    # Hold the stream behind a long kernel so the copy is still queued.
    torch.cuda._sleep(1 << 30)
    plane.encode([(1, 0)])
    assert plane.encoded() == []
    torch.cuda.synchronize()
    assert plane.encoded()[0][:3] == (1, None, 3)


# --- shared-memory pixels across the TP ranks ---


def _admit_on_rank(rank, rendezvous, metadata, cached):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        cache = EmbeddingCache(1 << 20)
        if cached:
            cache.put(0, (torch.zeros(3, HIDDEN), None), 48)
        plane = EncodeDataPlane(
            model=_Tower(),
            cache=cache,
            ring=SendRing(slots=1, slot_bytes=1024, deepstack=False, device="cpu"),
            device="cpu",
            tp_cpu_group=dist.group.WORLD,
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=0,
            offsets=[(0, 2)],
            feature_shm=ShmTensorHandle(*metadata),
        )
        # Without the rendezvous rank 0 would unlink the segment before rank 1
        # opened it.
        if rank == 1:
            time.sleep(0.5)
        assert plane.admit([(1, _frames(item))]) == [cached]
        if not cached:
            assert torch.equal(plane._items[1].feature, torch.arange(12).reshape(3, 4))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cached", [False, True], ids=["cache-miss", "cache-hit"])
def test_the_ranks_open_every_segment_before_any_consumes_one(tmp_path, cached):
    handle = ShmTensorHandle.publish(torch.arange(12).reshape(3, 4))
    try:
        mp.spawn(
            _admit_on_rank,
            args=(
                f"file://{tmp_path / 'rendezvous'}",
                (handle.shm_name, handle.shape, handle.dtype),
                cached,
            ),
            nprocs=2,
        )
        # Consumed on a hit too.
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)
    finally:
        handle.release()


# --- the tower's output and the cache's size ---


def test_the_tower_output_splits_into_items_and_deepstack_columns():
    class Model:
        num_deepstack_embeddings = 3

        @staticmethod
        def separate_deepstack_embeds(embedding):
            return embedding[:, :4], embedding[:, 4:]

    items = [
        MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, 1)]),
        MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, 2)]),
    ]
    output = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 16)
    assign_encoded_embeddings(items, output, Model())
    assert torch.equal(items[0].encoded, output[0:2, :4])
    assert torch.equal(items[0].encoded_deepstack, output[0:2, 4:])
    assert torch.equal(items[1].encoded, output[2:5, :4])
    assert items[1].encoded_deepstack.is_contiguous()
    with pytest.raises(ValueError, match="token-count"):
        assign_encoded_embeddings(items[:1], output, Model())


def test_the_cache_capacity_knobs(monkeypatch):
    field = envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB
    monkeypatch.setenv(field.name, "8")
    assert embedding_cache_bytes(field) == 8 << 20
    monkeypatch.setenv(field.name, "-5")
    with pytest.raises(ValueError, match=field.name):
        embedding_cache_bytes(field)
    tiered = make_embedding_cache(4 << 30, 8 << 30, "cuda:0")
    assert type(tiered) is TieredEmbeddingCache
    assert (tiered.l1.capacity_bytes, tiered.l2.capacity_bytes) == (4 << 30, 8 << 30)
    assert type(make_embedding_cache(4 << 30, 0, "cuda:0")) is EmbeddingCache


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
