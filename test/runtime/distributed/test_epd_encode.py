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

"""EPD encode-side compute and scheduling: the encode worker (tower dedup,
ring-slot fault isolation), the executor row/column scatter geometry, the
batch scheduler packing, and the encode-loop config selectors."""

from __future__ import annotations

import threading

import pytest
import torch

import tokenspeed.runtime.epd.encode_loop as encode_loop
from tokenspeed.runtime.cache.embedding_cache import (
    EmbeddingCache,
    TieredEmbeddingCache,
)
from tokenspeed.runtime.epd.encode_executor import (
    DisaggEncodeExecutor,
    assign_encoded_embeddings,
)
from tokenspeed.runtime.epd.encode_loop import (
    _embedding_cache_bytes,
    _make_embedding_cache,
    _maybe_install_encoder_cudagraph,
)
from tokenspeed.runtime.epd.encode_scheduler import (
    EncodeScheduler,
    PendingEncodeItem,
)
from tokenspeed.runtime.epd.encode_worker import (
    EncodeRequest,
    EncodeWorker,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
)
from tokenspeed.runtime.pd.base.status import TransferPoll


class _FakeExecutor:
    def __init__(self):
        self.registered = []
        self.executed_batches = []  # list of [item.hash, ...] the tower ran
        self.sent_direct = []  # cache-hit hashes shipped without the tower

    def register(self, rid, host, port, room):
        self.registered.append((rid, room))

    def execute(self, request_items):
        self.executed_batches.append([item.hash for _, item in request_items])
        for _, item in request_items:
            item.encoded = torch.zeros(2, 4)  # simulate tower output (shipped inside)

    def send_item(self, rid, item):
        self.sent_direct.append(item.hash)

    def reap_concluded_senders(self, _pending_request_ids):
        pass

    def drain_deferred(self):
        pass

    def has_deferred(self):
        return False


def _item(h, tokens=2):
    return MultimodalDataItem(
        modality=Modality.IMAGE, hash=h, offsets=[(0, tokens - 1)]
    )


def _worker(max_tokens=10_000, max_items=99, cap_bytes=10**6):
    ex = _FakeExecutor()
    return ex, EncodeWorker(
        ex, EncodeScheduler(max_tokens, max_items), EmbeddingCache(cap_bytes)
    )


def _req(rid, items, room=0):
    return EncodeRequest(rid, "h", 1, room, items)


def test_miss_runs_tower_and_caches():
    ex, w = _worker()
    w.submit(_req("r0", [_item(111)]))
    assert w.has_pending()
    assert w.step() == 1
    assert ex.executed_batches == [[111]]
    assert 111 in w.cache
    assert not w.has_pending()
    assert w.step() == 0


def test_cache_hit_skips_tower_but_still_ships():
    ex, w = _worker()
    w.submit(_req("r0", [_item(111)], room=0))
    w.step()  # tower runs, 111 cached
    w.submit(_req("r1", [_item(111)], room=1))  # same image
    # cache hit: shipped directly, not queued for the tower
    assert ex.sent_direct == [111]
    assert not w.has_pending()
    assert ex.executed_batches == [[111]]  # tower ran exactly once


# --- The encode worker drives the cache through get()/put() only, so it works
# with either the single-tier EmbeddingCache or the two-tier TieredEmbeddingCache.
# These exercise the real consumer path against the tiered cache (with cpu copies)
# and the production device<->host copy helpers the unit tests stub out. ---


# --- A per-item contract violation in the tower step (a bad grid/token-count ->
# ValueError, or an embedding larger than a ring slot -> RuntimeError) must fail
# only the rooms in that batch, NOT raise out of the encode loop into the
# engine's SIGUSR1 handler (which kills the whole worker and loses every other
# request's in-flight image, since the gateway round-robins images across
# workers). ---


class _FakeSender:
    def __init__(self, room):
        self.bootstrap_room = room


class _FakeFailManager:
    def __init__(self):
        self.request_status = {}
        self.failed = []  # (room, reason) for each fail_room call

    def fail_room(self, room, reason):
        self.request_status[room] = TransferPoll.Failed
        self.failed.append((room, reason))


class _RaisingExecutor:
    """Tower step raises a per-item contract violation."""

    def __init__(self, exc):
        self._exc = exc
        self.manager = _FakeFailManager()
        self.senders = {}
        self.executed = 0

    def register(self, rid, host, port, room):
        self.senders[rid] = _FakeSender(room)

    def execute(self, request_items):
        self.executed += 1
        raise self._exc

    def send_item(self, rid, item):
        pass

    def reap_concluded_senders(self, _pending_request_ids):
        pass

    def drain_deferred(self):
        pass

    def has_deferred(self):
        return False

    def fail_rooms(self, request_ids, exc):
        rooms = set()
        for rid in request_ids:
            s = self.senders.get(rid)
            if s is not None:
                rooms.add(s.bootstrap_room)
        for room in rooms:
            self.manager.fail_room(room, str(exc))
        return len(rooms)


def _raising_worker(exc):
    ex = _RaisingExecutor(exc)
    return ex, EncodeWorker(ex, EncodeScheduler(10_000, 99), EmbeddingCache(10**6))


def test_tower_valueerror_concludes_room_failed_not_crash():
    ex, w = _raising_worker(ValueError("rows != tokens"))
    w.submit(_req("r0", [_item(111)], room=7))
    # the raise is swallowed: step returns 0, the worker survives
    assert w.step() == 0
    assert ex.executed == 1
    # the room is concluded Failed (the receiver learns via the rank-synced abort)
    assert ex.manager.failed == [(7, "rows != tokens")]
    # the batch is drained from pending so the loop never re-runs the bad item
    assert not w.has_pending()


# --- DisaggEncodeExecutor ring buffers: every transferred buffer must be a
# registered, non-overlapping memory region. Registering each fresh per-request
# ``item.encoded`` address fails on RDMA (the torch caching allocator packs
# freed-but-still-registered tensors together -> a later grown region straddles
# others -> "overlapped memory region" -> the one-sided write fails -> the prefill
# scheduler dies). The executor instead collapses every send through a fixed RING
# of pre-registered bounce buffers, each registered exactly once at a fixed size;
# ``item.encoded`` is COPIED into the next slot, so the registered set never grows,
# shrinks, or overlaps regardless of how the allocator reuses addresses. ---


class _RecordingEngine:
    def __init__(self):
        self.calls = []  # ("reg", ptr, nbytes) | ("dereg", ptr, None)

    def register(self, ptr, length):
        self.calls.append(("reg", ptr, length))

    def deregister(self, ptr):
        self.calls.append(("dereg", ptr, None))


class _FakeManager:
    def __init__(self, engine):
        self.engine = engine


def _encode_executor(ring_slots=3, ring_bytes=256):
    # Tiny ring so the registration + staging path is exercised on CPU without a
    # multi-GiB allocation (production defaults to 64 x 256 MiB).
    eng = _RecordingEngine()
    ex = DisaggEncodeExecutor(
        _FakeManager(eng),
        multimodal_model=None,
        device="cpu",
        ring_slots=ring_slots,
        ring_bytes=ring_bytes,
    )
    return eng, ex


def test_copy_into_rejects_oversized_embedding():
    # An embedding larger than a slot must fail loud rather than silently truncate.
    eng, ex = _encode_executor(ring_slots=2, ring_bytes=8)
    ex._ensure_rings()
    too_big = torch.zeros(64, dtype=torch.float32)  # 256 B >> 8 B slot
    with pytest.raises(RuntimeError):
        ex._copy_into(ex._main_ring, 0, too_big)


# --- Slot lease: a wrapped-around slot must not be overwritten while its last
# send can still read the pointer (in-flight, or parked for re-send). ---


class _LeaseManager:
    """Fake manager exposing the status/parking surface the lease probes."""

    def __init__(self, engine):
        self.engine = engine
        self.request_status = {}
        self._pending = {}
        self._pending_lock = threading.Lock()
        # Deadline already in the past: a blocked lease raises immediately
        # instead of stalling the test for the real 130s parking window.
        self.bootstrap_time_out = -11.0

    def room_status(self, room):
        return self.request_status.get(room)

    def is_parked(self, room):
        with self._pending_lock:
            return room in self._pending


# --- Staging errors on the UNGUARDED paths must fail the room, not the worker.
# EncodeWorker.step only wraps execute(); the cache-hit send_item() and the
# deferred drain_deferred() reach _stage_and_send unguarded, so an oversized
# embedding there used to escape into the SIGUSR1 handler and kill the worker. ---


class _StageFailManager:
    """Fake manager exposing the lease surface AND the public ``fail_room`` seam,
    so a staging error inside _stage_and_send can be concluded per-room on CPU."""

    def __init__(self, engine):
        self.engine = engine
        self.request_status = {}
        self._pending = {}
        self._pending_lock = threading.Lock()
        self.bootstrap_time_out = -11.0
        self.failed = []  # (room, reason)

    def room_status(self, room):
        return self.request_status.get(room)

    def is_parked(self, room):
        with self._pending_lock:
            return room in self._pending

    def fail_room(self, room, reason):
        self.request_status[room] = TransferPoll.Failed
        self.failed.append((room, reason))


def _stagefail_executor(ring_slots=2, ring_bytes=8):
    ex = DisaggEncodeExecutor(
        _StageFailManager(_RecordingEngine()),
        multimodal_model=None,
        device="cpu",
        ring_slots=ring_slots,
        ring_bytes=ring_bytes,
    )
    ex._ensure_rings()
    return ex


def _oversized_item(ring_bytes):
    # An embedding strictly larger than a ring slot -> _copy_into RuntimeError.
    n = ring_bytes  # float32 => 4 B/elt, so ring_bytes elts = 4x a slot
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        encoded=torch.zeros(n, 1, dtype=torch.float32),
    )


def test_oversized_item_does_not_poison_a_healthy_sibling():
    # A bad item fails only its own room; a well-sized sibling in the same
    # _stage_and_send batch still ships.
    ex = _stagefail_executor(ring_slots=2, ring_bytes=4096)
    sent = []

    class _Sender:
        def __init__(self, room):
            self.bootstrap_room = room

        def send(self, **kw):
            sent.append(self.bootstrap_room)

    ex.senders["bad"] = _Sender(1)
    ex.senders["ok"] = _Sender(2)
    big = _oversized_item(ring_bytes=4096)
    ok = MultimodalDataItem(
        modality=Modality.IMAGE,
        encoded=torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    ex._stage_and_send([("bad", big), ("ok", ok)])  # must NOT raise
    assert ex.manager.request_status[1] == TransferPoll.Failed
    assert sent == [2]  # the healthy sibling still shipped


class _DeepstackModel:
    """ndeep=3: encoded width is hidden*4, split into main[:hidden] + deep[hidden:]."""

    num_deepstack_embeddings = 3

    def separate_deepstack_embeds(self, emb):
        hidden = emb.shape[-1] // (1 + self.num_deepstack_embeddings)
        return emb[:, :hidden], emb[:, hidden:]


class _PlainModel:
    num_deepstack_embeddings = 0


def _exec_item(*offset_pairs):
    return MultimodalDataItem(modality=Modality.IMAGE, offsets=list(offset_pairs))


def test_split_assigns_per_item_rows_and_deepstack():
    # item0 = 2 tokens, item1 = 3 tokens; hidden=4 -> width 16
    items = [_exec_item((0, 1)), _exec_item((0, 2))]
    width = 16
    output = torch.arange(5 * width, dtype=torch.float32).reshape(5, width)
    assign_encoded_embeddings(items, output, _DeepstackModel())

    assert items[0].encoded.shape == (2, 4)
    assert items[0].encoded_deepstack.shape == (2, 12)
    assert items[1].encoded.shape == (3, 4)
    assert items[1].encoded_deepstack.shape == (3, 12)
    # value alignment: rows are contiguous in order, columns split at hidden=4
    assert torch.equal(items[0].encoded, output[0:2, :4])
    assert torch.equal(items[0].encoded_deepstack, output[0:2, 4:])
    assert torch.equal(items[1].encoded, output[2:5, :4])
    assert torch.equal(items[1].encoded_deepstack, output[2:5, 4:])
    assert items[0].encoded.is_contiguous()
    assert items[1].encoded_deepstack.is_contiguous()


def test_token_count_mismatch_raises():
    item = _exec_item((0, 1))  # 2 tokens
    output = torch.randn(5, 8)  # 5 rows != 2
    with pytest.raises(ValueError):
        assign_encoded_embeddings([item], output, _PlainModel())


class _FeatureFnModel:
    """Surfaces the three encode entry points _feature_fn dispatches between."""

    def __init__(self):
        # In a real model image_encoder defaults to get_image_feature and is
        # swapped to the cudagraph wrapper by _maybe_install_encoder_cudagraph;
        # use a distinct sentinel so the test proves IMAGE routes via the seam.
        self.image_encoder = lambda items: "via-image_encoder-seam"
        self.get_image_feature = lambda items: "eager-get_image_feature"
        self.get_video_feature = lambda items: "video"


def test_feature_fn_image_routes_through_image_encoder_seam():
    from tokenspeed.runtime.epd.encode_executor import (
        DisaggEncodeExecutor,
    )

    model = _FeatureFnModel()
    exe = DisaggEncodeExecutor(object(), model, "cpu")
    # IMAGE must dispatch through image_encoder (the cudagraph seam), NOT
    # get_image_feature directly -- else the captured graph would be bypassed.
    assert exe._feature_fn(Modality.IMAGE) is model.image_encoder
    assert exe._feature_fn(Modality.IMAGE) is not model.get_image_feature
    # VIDEO has no captured graph and stays on the eager entry point.
    assert exe._feature_fn(Modality.VIDEO) is model.get_video_feature


def _sched_item(rid: str, idx: int, cost: int) -> PendingEncodeItem:
    return PendingEncodeItem(
        request_id=rid,
        item_index=idx,
        cost=cost,
    )


def test_scheduler_packs_until_token_budget():
    s = EncodeScheduler(max_tokens_per_batch=100, max_items_per_batch=99)
    s.add(_sched_item("r0", 0, 40))
    s.add(_sched_item("r0", 1, 40))
    s.add(_sched_item("r1", 0, 40))  # 120 > 100 -> stays for next batch
    b = s.next_batch()
    assert [(i.request_id, i.item_index) for i in b] == [("r0", 0), ("r0", 1)]
    assert s.pending_size() == 1
    b2 = s.next_batch()
    assert [(i.request_id, i.item_index) for i in b2] == [("r1", 0)]
    assert s.pending_size() == 0
    assert s.next_batch() == []


def test_scheduler_respects_max_items():
    s = EncodeScheduler(max_tokens_per_batch=10_000, max_items_per_batch=2)
    for i in range(5):
        s.add(_sched_item("r0", i, 1))
    assert len(s.next_batch()) == 2
    assert len(s.next_batch()) == 2
    assert len(s.next_batch()) == 1


def test_scheduler_oversized_single_item_returned_alone():
    s = EncodeScheduler(max_tokens_per_batch=50, max_items_per_batch=99)
    s.add(_sched_item("r0", 0, 500))  # cost > budget: must still make progress
    s.add(_sched_item("r0", 1, 10))
    b = s.next_batch()
    assert [(i.request_id, i.item_index) for i in b] == [("r0", 0)]
    b2 = s.next_batch()
    assert [(i.request_id, i.item_index) for i in b2] == [("r0", 1)]


def test_scheduler_rejects_bad_budgets():
    with pytest.raises(ValueError):
        EncodeScheduler(max_tokens_per_batch=0, max_items_per_batch=1)
    with pytest.raises(ValueError):
        EncodeScheduler(max_tokens_per_batch=1, max_items_per_batch=0)


# --------------------------------------------------------------------------- #
# _embedding_cache_bytes
# --------------------------------------------------------------------------- #
def test_bytes_override(monkeypatch):
    env_field = encode_loop.envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB
    monkeypatch.setenv(env_field.name, "8")
    assert _embedding_cache_bytes(env_field) == 8 * 1024 * 1024


def test_bytes_negative_raises_with_env_name(monkeypatch):
    env_field = encode_loop.envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB
    monkeypatch.setenv(env_field.name, "-5")
    with pytest.raises(ValueError) as exc:
        _embedding_cache_bytes(env_field)
    assert env_field.name in str(exc.value)


# --------------------------------------------------------------------------- #
# _make_embedding_cache (cache-type selection: the "L2 default off" property)
# --------------------------------------------------------------------------- #
def test_make_cache_l2_enabled_is_tiered_with_caps_and_device():
    cache = _make_embedding_cache(4 << 30, 8 << 30, "cuda:0")
    assert type(cache) is TieredEmbeddingCache
    assert cache.l1.capacity_bytes == (4 << 30)
    assert cache.l2.capacity_bytes == (8 << 30)
    assert cache._device == "cuda:0"


# --------------------------------------------------------------------------- #
# _maybe_install_encoder_cudagraph (gate parity with the aggregated ModelExecutor
# install; the actual capture is GPU-only and validated at e2e)
# --------------------------------------------------------------------------- #
_WRAPPER = object()  # stands in for the EncoderCudaGraphWrapper


class _FakeModel:
    """Minimal multimodal model surface the install gate touches."""

    def __init__(self, *, multimodal=True):
        self.is_multimodal_active = multimodal
        self.mapping = object()
        # The model leaves image_encoder == get_image_feature by default; the
        # wrapper install overrides it.
        self.image_encoder = self.get_image_feature
        self.built_with = None

    def get_image_feature(self, items):
        return "eager"

    def make_encoder_cudagraph_wrapper(self, mapping):
        self.built_with = mapping
        return _WRAPPER


class _FakeServerArgs:
    def __init__(self, backend="trtllm_ragged"):
        self.mm_attention_backend = backend


def _set_graph_flag(monkeypatch, value):
    monkeypatch.setattr(
        encode_loop.envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH,
        "get",
        lambda: value,
    )


def test_encoder_cudagraph_installed_when_enabled(monkeypatch):
    _set_graph_flag(monkeypatch, True)
    m = _FakeModel()
    assert _maybe_install_encoder_cudagraph(m, _FakeServerArgs()) is True
    assert m.image_encoder is _WRAPPER
    assert m.built_with is m.mapping
