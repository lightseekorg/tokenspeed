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

"""CPU-only tests for P-side layerwise KV streaming: which blocks each
prefill chunk sends, the producer steps it waits for (``HostPrefillPeer``,
whose transfers the Rust peer runs), and the attention backends' step
publication."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

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
from test.runtime.cache_pd_test_utils import producer_schedule as make_producer_schedule
from test.runtime.cache_pd_test_utils import segment as make_segment

from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    CachePDLayerwiseBlockSelection,
    CachePDLayerwiseGroupSelection,
    CacheProducerSchedule,
    CacheTransferContract,
    build_cache_layerwise_block_selection,
)
from tokenspeed.runtime.pd.host_peer import (  # noqa: E402
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
from tokenspeed.runtime.pd.transfer_plan import CacheTransferPlanner  # noqa: E402

_SOURCE = 0x1000
_DESTINATION = 0x9000


def _segment(
    field_id: str,
    *,
    producer_step: int,
    page_zero_offset: int,
    page_stride_bytes: int = 32,
    shape: tuple[int, ...] = (2, 2),
    partition_axis: int | None = None,
    partition_global_extent: int | None = None,
):
    return make_segment(
        field_id,
        dtype="bfloat16",
        shape=shape,
        offset=page_zero_offset,
        stride=page_stride_bytes,
        axis=partition_axis,
        extent=partition_global_extent,
        producer_step=producer_step,
    )


def _group(group_id, attn_kind, *segments, sliding_window_tokens: int | None = None):
    return make_group(
        group_id, *segments, attn_kind=attn_kind, window=sliding_window_tokens
    )


def _grouped_groups():
    return (
        _group(
            "history",
            "full_history",
            _segment("layer.0.kv", producer_step=0, page_zero_offset=0),
        ),
        _group(
            "state-a",
            "state",
            _segment("layer.2.state", producer_step=2, page_zero_offset=64),
        ),
        _group(
            "state-b",
            "state",
            _segment("layer.4.state", producer_step=4, page_zero_offset=128),
        ),
    )


def _grouped_layout() -> CacheTransferContract:
    return make_layout(
        *_grouped_groups(),
        block_size=2,
        capacity=32,
        page_bytes=256,
    )


def test_group_aware_two_chunk_history_and_state_selection() -> None:
    layout = _grouped_layout()
    tables = {
        "history": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
        "state-a": np.asarray([[5, 8, -1, -1]], dtype=np.int32),
        "state-b": np.asarray([[0, 12, -1, -1]], dtype=np.int32),
    }
    operation = SimpleNamespace(block_tables_arrays=lambda: tables)
    first = build_cache_layerwise_block_selection(
        operation,
        layout=layout,
        request_row=0,
        prefix_len=2,
        prompt_len=7,
        chunk_start=0,
        chunk_end=4,
        state_prefix_cached=False,
    )
    assert first.groups == (
        CachePDLayerwiseGroupSelection((2,), (0,)),
        CachePDLayerwiseGroupSelection((), ()),
        CachePDLayerwiseGroupSelection((), ()),
    )
    final = build_cache_layerwise_block_selection(
        operation,
        layout=layout,
        request_row=0,
        prefix_len=2,
        prompt_len=7,
        chunk_start=4,
        chunk_end=7,
        state_prefix_cached=False,
    )
    # The final chunk sends each state group's live slot: the prompt-end state.
    assert final.groups == (
        CachePDLayerwiseGroupSelection((3, 4), (1, 2)),
        CachePDLayerwiseGroupSelection((8,), (0,)),
        CachePDLayerwiseGroupSelection((12,), (0,)),
    )
    cached = build_cache_layerwise_block_selection(
        operation,
        layout=layout,
        request_row=0,
        prefix_len=2,
        prompt_len=7,
        chunk_start=4,
        chunk_end=7,
        state_prefix_cached=True,
    )
    assert cached.groups == (
        CachePDLayerwiseGroupSelection((3, 4), (1, 2)),
        CachePDLayerwiseGroupSelection((), ()),
        CachePDLayerwiseGroupSelection((), ()),
    )


class _Counter:
    """The producer step counter as the attention backends tick it."""

    def __init__(self) -> None:
        self.reserved = 0

    def current_step(self) -> tuple[int, int]:
        return self.reserved, 0

    def advance_step(self, *, delta_cache_step: int, delta_aux_step: int) -> None:
        assert delta_aux_step == 0
        self.reserved += delta_cache_step


class _Outbox:
    """Stands in for the Rust host: records the layerwise chunks it is handed."""

    def __init__(self) -> None:
        self.layerwise: list[tuple] = []

    def send_layerwise(self, rid, room, chunk_id, target_steps, is_last) -> None:
        self.layerwise.append((rid, room, chunk_id, list(target_steps), is_last))


def _kv_args(layout: CacheTransferContract, schedule=None) -> KVArgs:
    return KVArgs(
        engine_rank=0,
        kv_data_ptr=_SOURCE,
        ib_device="",
        gpu_id=0,
        cache_layout=layout,
        cache_fields_by_stage=(tuple(field.field_id for field in layout.plan.fields),),
        cache_producer_schedule=schedule,
    )


def _registration_frames(layout: CacheTransferContract) -> list[bytes]:
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


def _layerwise_peer(layout, schedule, interval: int, rooms: dict[str, tuple]):
    """A Prefill peer streaming layerwise, with a Decode destination for each
    ``rid -> (room, destination manifest)``."""
    outbox = _Outbox()
    topology = PDParallelTopology(
        tp_size=1, tp_rank=0, dp_size=1, dp_rank=0, world_size=1, global_rank=0
    )
    peer = HostPrefillPeer(
        _kv_args(layout, schedule),
        topology,
        outbox,
        PeerSettings(
            library="libtokenspeed_transfer_engine.so",
            local_ip="127.0.0.1",
            ib_device="",
            engine_rank=0,
            regions=((_SOURCE, 4096),),
        ),
        PrefillSettings(
            status_rank=0,
            bootstrap_url="http://127.0.0.1:8998/route",
            bootstrap_payload="{}",
            bootstrap_timeout_s=120.0,
            failed_session_ttl_s=30.0,
            transfer_workers=1,
        ),
    )
    counter = _Counter()
    peer.register_layerwise_step_counter(counter, interval)
    peer.prepare_registration(_registration_frames(layout))
    for rid, (room, manifest) in rooms.items():
        peer.accept_preallocation(
            [str(room).encode(), b"decode:7000", manifest.to_wire_bytes()]
        )
        peer.track(rid, room)
    return peer, outbox, counter


def test_a_prefill_forward_checks_every_row_then_reserves_its_steps_once() -> None:
    layout = _grouped_layout()
    schedule = make_producer_schedule(*_grouped_groups(), steps=5)
    manifest = make_block_manifest(
        ("history", (20, 21)), ("state-a", (22,)), ("state-b", (23,)), prompt=4
    )
    peer, outbox, counter = _layerwise_peer(
        layout,
        schedule,
        interval=2,
        rooms={"request-0": (9, manifest), "request-1": (10, manifest)},
    )
    tables = {
        "history": np.asarray([[1, 2], [3, 4]], dtype=np.int32),
        "state-a": np.asarray([[5, 6], [7, 8]], dtype=np.int32),
        "state-b": np.asarray([[9, 10], [11, 12]], dtype=np.int32),
    }
    operation = make_operation(
        tables,
        request_ids=["request-0", "request-1"],
        request_pool_indices=[3, 4],
        extend_prefix_lens=[0, 0],
        input_lengths=[4, 4],
        num_extends=lambda: 2,
    )

    peer.prepare_prefill(operation)

    # One producer range for the forward; each interval of two steps ends
    # at its last step.
    assert counter.reserved == 5
    assert [entry[:2] for entry in outbox.layerwise] == [
        ("request-0", 9),
        ("request-1", 10),
    ]
    assert all(entry[3] == [1, 3, 4] and entry[4] for entry in outbox.layerwise)

    # A later bad row rejects the whole batch before any step is reserved or
    # an earlier valid row is armed.
    outbox.layerwise.clear()
    tables["history"][1, 0] = 0
    with pytest.raises(ValueError, match="invalid block ID"):
        peer.prepare_prefill(operation)
    assert counter.reserved == 5
    assert outbox.layerwise == []


def _host_spans(items) -> list[tuple[int, int, int]]:
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


def _planner_spans(layout, schedule, manifest, selection) -> list[list[tuple]]:
    """Per producer step, the spans the transfer planner gives for one
    chunk's ``selection``."""
    kv_args = _kv_args(layout, schedule)
    topology = PDParallelTopology(
        tp_size=1, tp_rank=0, dp_size=1, dp_rank=0, world_size=1, global_rank=0
    )
    registration = prepare_decode_registration(
        kv_args, topology, KVArgsRegisterInfo.from_zmq(_registration_frames(layout))
    )
    intervals = cache_transfer_blocks(
        kv_args,
        dst_ptr=registration.dst_kv_ptr,
        dst_tp_rank=registration.decode_tp_rank,
        src_block_manifest=None,
        dst_block_manifest=manifest,
        transfer_fragments=registration.transfer_fragments,
        owner_filters=registration.transfer_owner_filters,
        dst_cache_layout=registration.peer_cache_layout,
        block_selection=selection,
        step_field_ids=[
            schedule.fields_in_range(step, step + 1)
            for step in range(schedule.step_count)
        ],
    )
    planned = []
    for items in intervals:
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
        planned.append(spans)
    return planned


def test_the_first_layerwise_chunk_carries_a_deeper_prefill_cache_hit() -> None:
    layout = _grouped_layout()
    schedule = make_producer_schedule(*_grouped_groups(), steps=5)
    manifest = make_block_manifest(
        ("history", (20, 21, 22, 23)),
        ("state-a", (24,)),
        ("state-b", (25,)),
        prefix=2,
        prompt=9,
    )
    peer, outbox, _ = _layerwise_peer(
        layout, schedule, interval=1, rooms={"request-0": (9, manifest)}
    )
    tables = {
        "history": np.asarray([[1, 2, 3, 4, 5]], dtype=np.int32),
        "state-a": np.asarray([[6, 7, 8, 9, 10]], dtype=np.int32),
        "state-b": np.asarray([[11, 12, 13, 14, 15]], dtype=np.int32),
    }

    def chunk(begin: int, length: int):
        return make_operation(
            tables,
            request_ids=["request-0"],
            request_pool_indices=[3],
            extend_prefix_lens=[begin],
            input_lengths=[length],
            num_extends=lambda: 1,
        )

    # Decode holds 2 tokens and this rank's cache 4: the first chunk, [4, 7),
    # also sends the history blocks of [2, 4) Decode lacks; the last sends
    # from its own boundary on.
    peer.prepare_prefill(chunk(4, 3))
    peer.prepare_prefill(chunk(7, 2))
    first, last = outbox.layerwise
    assert (first[4], last[4]) == (False, True)
    empty = CachePDLayerwiseGroupSelection((), ())
    expected = CachePDLayerwiseBlockSelection(
        groups=(CachePDLayerwiseGroupSelection((2, 3), (0, 1)), empty, empty)
    )
    (planned,) = peer.plan_layerwise(first[2], ["decode:7000"])
    assert [_host_spans(items) for items in planned] == _planner_spans(
        layout, schedule, manifest, expected
    )
    # The last chunk's first producer step moves its own history blocks.
    (planned,) = peer.plan_layerwise(last[2], ["decode:7000"])
    expected_history = _planner_spans(
        layout,
        schedule,
        manifest,
        CachePDLayerwiseBlockSelection(
            groups=(CachePDLayerwiseGroupSelection((4, 5), (2, 3)), empty, empty)
        ),
    )
    assert _host_spans(planned[0]) == expected_history[0]


def _draft_final_group():
    return _group(
        "history",
        "full_history",
        _segment(
            "layer.0.kv",
            producer_step=0,
            page_zero_offset=0,
            page_stride_bytes=128,
        ),
        _segment(
            "layer.1.kv",
            producer_step=1,
            page_zero_offset=32,
            page_stride_bytes=128,
        ),
        _segment(
            "layer.2.kv",
            producer_step=2,
            page_zero_offset=64,
            page_stride_bytes=128,
        ),
        _segment(
            "layer.3.kv",
            producer_step=2,
            page_zero_offset=96,
            page_stride_bytes=128,
        ),
    )


def _draft_final_layout() -> CacheTransferContract:
    return make_layout(
        _draft_final_group(),
        capacity=8,
        page_bytes=256,
    )


def _stage_producer_schedule(
    stage: int, target_cache_windows: tuple[tuple[int, int], ...]
) -> CacheProducerSchedule:
    """One stage's schedule over ``_draft_final_group``: two target cache
    layers (``layer.0``, ``layer.1``) and two draft layers (``layer.2``,
    ``layer.3``) that the final stage finalizes in one trailing step."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
        pipeline_cache_ownership,
    )

    owner = pipeline_cache_ownership(
        num_target_cache_layers=2,
        num_draft_cache_layers=2,
        target_cache_windows=target_cache_windows,
    )[stage]
    return CacheProducerSchedule(
        tuple(
            tuple(f"layer.{layer}.kv" for layer in step)
            for step in owner.producer_cache_layers
        )
    )


@pytest.mark.parametrize(
    "stage,target_cache_windows,target_steps,draft_final",
    [
        # One stage owns everything: two target steps and the draft-final step.
        (0, ((0, 2),), 2, True),
        # A pipeline's first stage owns one target layer and no draft fields;
        # its readiness needs no draft-final event.
        (0, ((0, 1), (1, 2)), 1, False),
        # The last stage owns its target layer plus the trailing draft step.
        (1, ((0, 1), (1, 2)), 1, True),
    ],
)
def test_each_forward_waits_for_its_own_steps_including_the_draft_final_one(
    stage, target_cache_windows, target_steps, draft_final
) -> None:
    schedule = _stage_producer_schedule(stage, target_cache_windows)
    if stage == 0 and len(target_cache_windows) == 1:
        assert schedule == make_producer_schedule(_draft_final_group(), steps=3)
    steps_per_forward = target_steps + int(draft_final)
    assert schedule.step_count == steps_per_forward
    manifest = make_block_manifest(("history", (5,)), prompt=2)
    peer, outbox, counter = _layerwise_peer(
        _draft_final_layout(), schedule, interval=1, rooms={"request-0": (9, manifest)}
    )
    tables = {"history": np.asarray([[1]], dtype=np.int32)}
    for _ in range(2):
        peer.prepare_prefill(
            make_operation(
                tables,
                request_ids=["request-0"],
                request_pool_indices=[3],
                extend_prefix_lens=[0],
                input_lengths=[2],
                num_extends=lambda: 1,
            )
        )
    # Every forward reserves its whole range, the draft-final step included,
    # so the next forward's steps start where its own end.
    assert [entry[3] for entry in outbox.layerwise] == [
        list(range(begin, begin + steps_per_forward))
        for begin in (0, steps_per_forward)
    ]
    assert counter.reserved == 2 * steps_per_forward


def _heterogeneous_groups(*, destination: bool):
    state_shape = (4, 2) if destination else (2, 2)
    return (
        _group(
            "history",
            "full_history",
            _segment("layer.0.latent", producer_step=0, page_zero_offset=0),
        ),
        _group(
            "state",
            "state",
            _segment(
                "layer.2.state",
                producer_step=2,
                page_zero_offset=128,
                shape=state_shape,
                partition_axis=0,
                partition_global_extent=4,
            ),
        ),
    )


def _heterogeneous_layout(*, destination: bool) -> CacheTransferContract:
    return make_layout(
        *_heterogeneous_groups(destination=destination),
        capacity=8,
        page_bytes=256,
    )


def test_heterogeneous_zero_edge_interval_does_not_fall_back_to_identity() -> None:
    source_layout = _heterogeneous_layout(destination=False)
    destination_layout = _heterogeneous_layout(destination=True)
    planner = CacheTransferPlanner(
        prefill_tp_size=2,
        decode_tp_size=1,
        prefill_layout=source_layout,
        decode_layout=destination_layout,
        prefill_field_ids=None,
    )
    rank_one_fragments = planner.plan_for_decode_rank(0).fragments_by_prefill_rank[1]
    assert tuple(
        (fragment.group_id, fragment.field_id) for fragment in rank_one_fragments
    ) == (("state", "layer.2.state"),)

    selection = CachePDLayerwiseBlockSelection(
        groups=(
            CachePDLayerwiseGroupSelection((1,), (0,)),
            CachePDLayerwiseGroupSelection((2,), (0,)),
        ),
    )
    destination_block_manifest = make_block_manifest(("history", (3,)), ("state", (4,)))
    kv_args = _kv_args(source_layout)

    # Rank 1 has no edge for the replicated latent field in layer 0. An empty
    # fragment set for this interval is a no-op, never a raw-page identity copy.
    schedule = make_producer_schedule(
        *_heterogeneous_groups(destination=False), steps=3
    )

    def blocks(begin, end):
        return next(
            cache_transfer_blocks(
                kv_args,
                dst_tp_rank=0,
                dst_ptr=0x20000,
                src_block_manifest=None,
                dst_block_manifest=destination_block_manifest,
                transfer_fragments=rank_one_fragments,
                owner_filters={},
                dst_cache_layout=destination_layout,
                block_selection=selection,
                step_field_ids=(schedule.fields_in_range(begin, end),),
            )
        )

    assert blocks(0, 1) == []
    assert len(blocks(2, 3)) == 1


def test_dsa_sparse_prefill_publishes_one_cache_step_after_cache_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_backend

    events = []
    backend = object.__new__(dsa_backend.DSABackend)
    backend.index_topk = 2
    backend.kvp_group = (0,)
    backend.v_head_dim = 1
    backend.data_type = torch.bfloat16
    backend.qk_nope_head_dim = 1
    backend.kv_lora_rank = 1
    backend.qk_rope_head_dim = 0
    backend.kernel_page_size = 64
    backend.kernel_solution = None
    backend.prefill_kernel_plan = None
    backend.decode_kernel_plan = None
    backend.slot_order = "selection"
    backend.num_local_heads = 1
    backend.num_attention_heads = 1
    backend.query_shard_metadata = None
    backend.step_counter = SimpleNamespace(record_cache=lambda: events.append("ready"))

    def fake_dsa_prefill(**_kwargs):
        events.append("attention")
        return torch.ones((1, 1, 1), dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_backend, "dsa_prefill", fake_dsa_prefill)
    output = backend.forward_sparse_prefill(
        q=torch.zeros((1, 1), dtype=torch.bfloat16),
        layer=SimpleNamespace(
            logit_cap=0,
            tp_q_head_num=1,
            v_head_dim=1,
            head_dim=1,
            layer_id=0,
            scaling=1.0,
        ),
        token_to_kv_pool=SimpleNamespace(
            quant_method=None,
            get_key_buffer=lambda _layer_id: torch.zeros(1),
        ),
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        topk_slots=torch.zeros((1, 2), dtype=torch.int64),
        topk_lens=torch.ones(1, dtype=torch.int32),
        max_seq_len=1,
    )

    assert output.shape == (1, 1)
    assert events == ["attention", "ready"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
