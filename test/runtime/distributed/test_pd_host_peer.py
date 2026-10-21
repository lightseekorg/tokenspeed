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

"""CPU-only tests for the data plane's half of PD under the Rust host.

The host peers (``pd/host_peer.py``) plan what the Rust transfer peer moves.
A transfer they plan must move exactly the bytes the Python sender moves for
the same registration and manifests; the outbox stands in for the host, and
the Python sender's Mooncake engine for RDMA.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.cache_pd_test_utils import block_manifest as make_block_manifest
from test.runtime.cache_pd_test_utils import group as make_group  # noqa: E402
from test.runtime.cache_pd_test_utils import layout as make_layout
from test.runtime.cache_pd_test_utils import operation as make_operation
from test.runtime.cache_pd_test_utils import segment as make_segment

from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    CachePDBlockManifest,
    build_cache_block_manifest,
)
from tokenspeed.runtime.pd.host_peer import (  # noqa: E402
    DecodeSettings,
    HostDecodePeer,
    HostPrefillPeer,
    PeerSettings,
    PrefillSettings,
)
from tokenspeed.runtime.pd.mooncake.entities import (  # noqa: E402
    KVArgs,
    KVArgsRegisterInfo,
)
from tokenspeed.runtime.pd.mooncake.pack import (  # noqa: E402
    PackedCopy,
    PageFieldCopies,
    expand_packed_copy,
)
from tokenspeed.runtime.pd.mooncake.prefill import (  # noqa: E402
    cache_transfer_blocks,
    prepare_decode_registration,
)
from tokenspeed.runtime.pd.topology import PDParallelTopology  # noqa: E402

_SOURCE = 0x1000
_DESTINATION = 0x9000


def _layout():
    return make_layout(
        make_group(
            "history",
            make_segment(
                "layer.0.kv", dtype="bfloat16", shape=(8,), offset=0, stride=32
            ),
        ),
        make_group(
            "state",
            make_segment(
                "layer.1.state", dtype="bfloat16", shape=(8,), offset=16, stride=32
            ),
            attn_kind="state",
        ),
        capacity=16,
        page_bytes=32,
    )


def _kv_args(layout) -> KVArgs:
    return KVArgs(
        engine_rank=0,
        kv_data_ptr=_SOURCE,
        ib_device="",
        gpu_id=0,
        cache_layout=layout,
        cache_fields_by_stage=(tuple(field.field_id for field in layout.plan.fields),),
    )


def _topology() -> PDParallelTopology:
    return PDParallelTopology(
        tp_size=1, tp_rank=0, dp_size=1, dp_rank=0, world_size=1, global_rank=0
    )


def _settings() -> PeerSettings:
    return PeerSettings(
        library="libtokenspeed_transfer_engine.so",
        local_ip="127.0.0.1",
        ib_device="",
        engine_rank=0,
        regions=((_SOURCE, 512),),
    )


def _op(*, request_id: str = "request-0"):
    # The remote-decode (P) or remote-prefill (D) op of one prompt of 5
    # tokens, 2 of them cached: history blocks 1-3, the prompt-end state in
    # block 6.
    return make_operation(
        {
            "history": np.asarray([[1, 2, 3]], dtype=np.int32),
            "state": np.asarray([[0, 6, -1]], dtype=np.int32),
        },
        request_ids=[request_id],
        request_pool_indices=[7],
        extend_prefix_lens=[2],
        prefill_lengths=[5],
        num_extends=lambda: 1,
        decode_input_ids=[42],
        spec_candidate_ids=[[42, 43]],
        state_prefix_cached=[False],
    )


def _destination_manifest() -> CachePDBlockManifest:
    return make_block_manifest(
        ("history", (10, 11)), ("state", (12,)), prefix=2, prompt=5
    )


def _registration_frames(layout) -> list[bytes]:
    return [
        b"None",
        b"127.0.0.1",
        b"9000",
        b"decode:7000",
        np.asarray([_DESTINATION], dtype=np.uint64).tobytes(),
        layout.to_wire_bytes(),
        b"1",
        b"0",
    ]


class _Outbox:
    """Stands in for the Rust host: records what the peer hands it."""

    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.unplanned: list[tuple] = []
        self.received: list[tuple] = []

    def send(self, rid, room, bootstrap_token, spec_candidate_ids) -> None:
        self.sent.append((rid, room, bootstrap_token, spec_candidate_ids))

    def send_unplanned(self, rid, room) -> None:
        self.unplanned.append((rid, room))

    def receive(self, rid, room, request_pool_index, manifest) -> None:
        self.received.append((rid, room, request_pool_index, manifest))


def _prefill_peer(layout, outbox) -> HostPrefillPeer:
    return HostPrefillPeer(
        _kv_args(layout),
        _topology(),
        outbox,
        _settings(),
        PrefillSettings(
            status_rank=0,
            bootstrap_url="http://127.0.0.1:8998/route",
            bootstrap_payload="{}",
            bootstrap_timeout_s=120.0,
            failed_session_ttl_s=30.0,
            transfer_workers=4,
        ),
    )


def _spans(items) -> list[tuple[int, int, int]]:
    """Every (source, destination, length) span a host plan writes, in order,
    as the host expands it."""
    spans = []
    for kind, a, b, c in items:
        if kind == "pages":
            src_pages = np.frombuffer(a, dtype="<i8")
            dst_pages = np.frombuffer(b, dtype="<i8")
            fields = np.frombuffer(c, dtype="<i8").reshape(-1, 5)
            for src_base, src_stride, dst_base, dst_stride, length in fields:
                for page, peer in zip(src_pages, dst_pages, strict=True):
                    spans.append(
                        (
                            int(src_base + page * src_stride),
                            int(dst_base + peer * dst_stride),
                            int(length),
                        )
                    )
        else:
            assert kind == "spans"
            spans.extend(
                zip(
                    np.frombuffer(a, dtype="<u8").tolist(),
                    np.frombuffer(b, dtype="<u8").tolist(),
                    np.frombuffer(c, dtype="<u8").tolist(),
                    strict=True,
                )
            )
    return spans


def _planner_spans(layout, frames, source_manifest, destination_manifest):
    """The copies the transfer planner (``cache_transfer_blocks``) gives for
    the same transfer, expanded to spans."""
    kv_args = _kv_args(layout)
    registration = prepare_decode_registration(
        kv_args, _topology(), KVArgsRegisterInfo.from_zmq(frames)
    )
    (items,) = cache_transfer_blocks(
        kv_args,
        dst_ptr=registration.dst_kv_ptr,
        dst_tp_rank=registration.decode_tp_rank,
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        transfer_fragments=registration.transfer_fragments,
        owner_filters=registration.transfer_owner_filters,
        dst_cache_layout=registration.peer_cache_layout,
        step_field_ids=(None,),
    )
    spans = []
    for item in items:
        if isinstance(item, PageFieldCopies):
            for (
                src_base,
                src_stride,
                dst_base,
                dst_stride,
                length,
            ) in item.fields.tolist():
                for page, peer in zip(
                    item.src_pages.tolist(), item.dst_pages.tolist(), strict=True
                ):
                    spans.append(
                        (
                            src_base + page * src_stride,
                            dst_base + peer * dst_stride,
                            length,
                        )
                    )
        elif isinstance(item, PackedCopy):
            spans.extend(expand_packed_copy(item))
        else:
            spans.append(tuple(item))
    return spans


def test_a_planned_send_hands_the_host_the_planners_copies():
    layout = _layout()
    outbox = _Outbox()
    peer = _prefill_peer(layout, outbox)
    frames = _registration_frames(layout)

    assert peer.prepare_registration(frames) == (
        "127.0.0.1",
        9000,
        "decode:7000",
        1,
        0,
        False,
        [0],
    )
    preallocation = [b"9", b"decode:7000", _destination_manifest().to_wire_bytes()]
    assert peer.accept_preallocation(preallocation) is False
    with pytest.raises(ValueError, match="duplicate"):
        peer.accept_preallocation(preallocation)

    # The remote decode hands the request over with its bootstrap payload.
    peer.track("request-0", 9)
    peer.execute(_op())
    assert outbox.sent == [("request-0", 9, 42, [42, 43])]

    planned = _spans(peer.plan_send(9, "decode:7000"))
    source_manifest = build_cache_block_manifest(
        _op(),
        layout=layout,
        request_row=0,
        prefix_len=2,
        prompt_len=5,
        state_prefix_cached=False,
    )
    expected = _planner_spans(layout, frames, source_manifest, _destination_manifest())
    assert planned and planned == expected
    # Every span reads this rank's arena and lands in the peer's.
    assert all(
        source >= _SOURCE and target >= _DESTINATION for source, target, _ in planned
    )

    # An ended request takes its room's manifests with it.
    peer.release("request-0")
    with pytest.raises(KeyError):
        peer.plan_send(9, "decode:7000")


def test_a_remote_decode_without_destinations_is_handed_over_unplanned():
    outbox = _Outbox()
    peer = _prefill_peer(_layout(), outbox)
    peer.track("request-0", 9)
    # Untracked requests are not this peer's to hand over.
    peer.execute(_op(request_id="other"))
    peer.execute(_op())
    assert outbox.unplanned == [("request-0", 9)]
    assert outbox.sent == []


def test_a_conflicting_registration_is_refused():
    layout = _layout()
    peer = _prefill_peer(layout, _Outbox())
    frames = _registration_frames(layout)
    peer.prepare_registration(frames)
    moved = list(frames)
    moved[4] = np.asarray([_DESTINATION + 0x100], dtype=np.uint64).tobytes()
    with pytest.raises(ValueError, match="conflicting"):
        peer.prepare_registration(moved)
    # Once the host rejected the session, a new registration is planned.
    peer.reject_registration("decode:7000")
    assert peer.prepare_registration(moved)[2] == "decode:7000"


def test_a_decode_peer_routes_and_publishes_each_admitted_manifest():
    layout = _layout()
    outbox = _Outbox()
    peer = HostDecodePeer(
        _kv_args(layout),
        _topology(),
        outbox,
        _settings(),
        DecodeSettings(
            arena_base=_SOURCE,
            layout=layout.to_wire_bytes(),
            tp_size=1,
            tp_rank=0,
            waiting_timeout_s=300.0,
            heartbeat_interval_s=5.0,
            heartbeat_max_failures=2,
        ),
    )
    prefill_info = json.dumps(
        {
            "prefill_tp_size": 1,
            "prefill_dp_size": 1,
            "prefill_pp_size": 1,
            "cache_layout": _kv_args(layout)
            .wire_layout.to_wire_bytes()
            .decode("ascii"),
            "cache_fields_by_stage": [list(_kv_args(layout).cache_fields_by_stage[0])],
        }
    )
    assert peer.plan_route(prefill_info) == ([0], [], [0], 1)
    with pytest.raises(RuntimeError, match="non-cache-transfer"):
        peer.plan_route(json.dumps({"prefill_tp_size": 1, "prefill_dp_size": 1}))

    peer.track("request-0", 9)
    peer.execute(_op())
    ((rid, room, slot, manifest),) = outbox.received
    assert (rid, room, slot) == ("request-0", 9, 7)
    manifest = CachePDBlockManifest.from_wire_bytes(manifest)
    assert [group.block_ids for group in manifest.groups] == [(2, 3), (6,)]
    assert (manifest.prefix_len, manifest.prompt_len) == (2, 5)
