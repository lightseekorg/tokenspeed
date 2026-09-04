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

from __future__ import annotations

import os
import sys
from collections import defaultdict
from contextlib import nullcontext
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

from runtime.cache_pd_test_utils import block_manifest as make_block_manifest
from runtime.cache_pd_test_utils import group as make_group  # noqa: E402
from runtime.cache_pd_test_utils import layout as make_layout
from runtime.cache_pd_test_utils import producer_schedule as make_producer_schedule
from runtime.cache_pd_test_utils import segment as make_segment

from tokenspeed.runtime.pd.base.status import TransferPoll  # noqa: E402
from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    CachePDLayerwiseBlockSelection,
    CachePDLayerwiseGroupSelection,
    CacheTransferContract,
    build_cache_layerwise_block_selection,
)
from tokenspeed.runtime.pd.mooncake.entities import (  # noqa: E402
    TransferInfo,
    TransferKVChunk,
)
from tokenspeed.runtime.pd.transfer_plan import (  # noqa: E402
    CacheTransferPlanner,
)
from tokenspeed.runtime.pd.utils import StepCounter  # noqa: E402


def test_step_counter_readiness_is_strict_and_wrap_safe() -> None:
    limit = StepCounter.COUNT_NUM_MAX
    assert not StepCounter.is_step_ready(0, 0)
    assert not StepCounter.is_step_ready(limit - 1, 0)
    assert StepCounter.is_step_ready(1, 0)
    assert StepCounter.is_step_ready(0, limit - 1)


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


def _group(
    group_id,
    family,
    *segments,
    retention="full_history",
    sliding_window_tokens: int | None = None,
):
    return make_group(
        group_id,
        *segments,
        family=family,
        retention=retention,
        window=sliding_window_tokens,
    )


def _grouped_groups():
    return (
        _group(
            "history",
            "history",
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
        "state-a": np.asarray([[5, 6, 7, 8]], dtype=np.int32),
        "state-b": np.asarray([[9, 10, 11, 12]], dtype=np.int32),
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
    )
    assert final.groups == (
        CachePDLayerwiseGroupSelection((3, 4), (1, 2)),
        CachePDLayerwiseGroupSelection((8,), (0,)),
        CachePDLayerwiseGroupSelection((12,), (0,)),
    )


def test_prepare_cache_prefill_preflights_batch_and_reserves_once() -> None:
    from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor

    layout = _grouped_layout()
    destination_block_manifest = make_block_manifest(
        ("history", (20, 21)),
        ("state-a", (22,)),
        ("state-b", (23,)),
        prompt=4,
    )
    destination = SimpleNamespace(
        is_dummy=False,
        block_manifest=destination_block_manifest,
    )
    calls = []

    class _Sender:
        def __init__(self, room: int):
            self.bootstrap_room = room
            self.started = False

        def layerwise_chunk_submitted(self):
            return self.started

        def send_layerwise(self, *args, **kwargs):
            calls.append((self.bootstrap_room, args, kwargs))
            self.started = True

    reserves = []
    manager = SimpleNamespace(
        transfer_infos={
            9: {"decode-0": destination},
            10: {"decode-0": destination},
        },
        reserve_layerwise_cache_steps=lambda: reserves.append(True) or 77,
        get_decode_registration=lambda _destination: SimpleNamespace(
            peer_cache_layout=layout
        ),
    )
    tables = {
        "history": np.asarray([[1, 2], [3, 4]], dtype=np.int32),
        "state-a": np.asarray([[5, 6], [7, 8]], dtype=np.int32),
        "state-b": np.asarray([[9, 10], [11, 12]], dtype=np.int32),
    }
    operation = SimpleNamespace(
        request_ids=["request-0", "request-1"],
        request_pool_indices=[3, 4],
        extend_prefix_lens=[0, 0],
        input_lengths=[4, 4],
        num_extends=lambda: 2,
        block_tables_arrays=lambda: tables,
    )
    executor = object.__new__(DisaggPrefillExecutor)
    executor.cache_layout = layout
    executor.senders = {"request-0": _Sender(9), "request-1": _Sender(10)}
    executor.kv_manager = manager
    executor._layerwise_interval = 2

    executor._prepare_cache_prefill(operation)

    assert reserves == [True]
    assert [call[2]["begin_cache_step"] for call in calls] == [77, 77]
    assert [call[2]["layerwise_interval"] for call in calls] == [2, 2]
    assert all(call[1] == (True,) for call in calls)

    # A later bad row rejects the whole batch before any step is reserved or
    # an earlier valid row is enqueued.
    calls.clear()
    reserves.clear()
    tables["history"][1, 0] = 0
    with pytest.raises(ValueError, match="invalid block ID"):
        executor._prepare_cache_prefill(operation)
    assert reserves == []
    assert calls == []


def test_first_layerwise_chunk_includes_deeper_prefill_cache_hit() -> None:
    from tokenspeed.runtime.pd.mooncake.sender import MooncakeKVSender
    from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor

    layout = _grouped_layout()
    destination = SimpleNamespace(
        is_dummy=False,
        block_manifest=make_block_manifest(
            ("history", (20, 21, 22, 23)),
            ("state-a", (24,)),
            ("state-b", (25,)),
            prefix=2,
            prompt=9,
        ),
    )
    calls = []

    manager = SimpleNamespace(
        begin_room=lambda _room: None,
        add_transfer_request=lambda *args, **kwargs: calls.append((args, kwargs)),
        transfer_infos={9: {"decode-0": destination}},
        reserve_layerwise_cache_steps=lambda: 77,
    )
    sender = MooncakeKVSender(manager, "127.0.0.1:9000", 9)
    tables = {
        "history": np.asarray([[1, 2, 3, 4, 5]], dtype=np.int32),
        "state-a": np.asarray([[6, 7, 8, 9, 10]], dtype=np.int32),
        "state-b": np.asarray([[11, 12, 13, 14, 15]], dtype=np.int32),
    }
    executor = object.__new__(DisaggPrefillExecutor)
    executor.cache_layout = layout
    executor.senders = {"request-0": sender}
    executor.kv_manager = manager
    executor._layerwise_interval = 1

    def operation(chunk_begin, chunk_length):
        return SimpleNamespace(
            request_ids=["request-0"],
            extend_prefix_lens=[chunk_begin],
            input_lengths=[chunk_length],
            num_extends=lambda: 1,
            block_tables_arrays=lambda: tables,
        )

    assert not sender.layerwise_chunk_submitted()
    executor._prepare_cache_prefill(operation(4, 3))
    assert sender.layerwise_chunk_submitted()
    assert not sender.layerwise_final_chunk_submitted()
    executor._prepare_cache_prefill(operation(7, 2))
    assert sender.layerwise_final_chunk_submitted()

    first_history = calls[0][1]["cache_block_selection"].groups[0]
    assert first_history.source_block_ids == (2, 3)
    assert first_history.destination_positions == (0, 1)
    second_history = calls[1][1]["cache_block_selection"].groups[0]
    assert second_history.source_block_ids == (4, 5)
    assert second_history.destination_positions == (2, 3)


def _ordinary_layerwise_group():
    segments = (
        _segment(
            "layer.0.k", producer_step=0, page_zero_offset=0, page_stride_bytes=128
        ),
        _segment(
            "layer.0.v", producer_step=0, page_zero_offset=32, page_stride_bytes=128
        ),
        _segment(
            "layer.2.k", producer_step=2, page_zero_offset=64, page_stride_bytes=128
        ),
        _segment(
            "layer.4.v", producer_step=4, page_zero_offset=96, page_stride_bytes=128
        ),
    )
    return _group("history", "history", *segments)


def _ordinary_layerwise_layout() -> CacheTransferContract:
    return make_layout(
        _ordinary_layerwise_group(),
        capacity=8,
        page_bytes=256,
    )


def _draft_final_group():
    return _group(
        "history",
        "history",
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


def test_draft_final_reservation_stays_aligned_across_forwards() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    class _Counter:
        def __init__(self):
            self.reserved = 0
            self.ready = 0

        def current_step(self):
            return self.reserved, 0

        def advance_step(self, *, delta_cache_step, delta_aux_step):
            assert delta_aux_step == 0
            self.reserved += delta_cache_step

    counter = _Counter()
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(
        cache_layout=_draft_final_layout(),
        cache_producer_schedule=make_producer_schedule(_draft_final_group(), steps=3),
    )
    manager.register_layerwise_step_counter(counter, interval=1)

    for expected_begin in (0, 3):
        begin = manager.reserve_layerwise_cache_steps()
        assert begin == expected_begin
        counter.ready += 2  # target layers
        final_target = begin + manager.producer_step_count - 1
        assert not StepCounter.is_step_ready(counter.ready, final_target)
        counter.ready += 1  # one synthetic draft-final event
        assert StepCounter.is_step_ready(counter.ready, final_target)

    assert counter.reserved == counter.ready == 6


def _single_history_selection() -> CachePDLayerwiseBlockSelection:
    return CachePDLayerwiseBlockSelection(
        groups=(CachePDLayerwiseGroupSelection((2,), (0,)),),
    )


def _heterogeneous_groups(*, destination: bool):
    state_shape = (4, 2) if destination else (2, 2)
    return (
        _group(
            "history",
            "history",
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
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    source_layout = _heterogeneous_layout(destination=False)
    destination_layout = _heterogeneous_layout(destination=True)
    planner = CacheTransferPlanner(
        prefill_tp_size=2,
        decode_tp_size=1,
        prefill_layout=source_layout,
        decode_layout=destination_layout,
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
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(
        cache_layout=source_layout,
        kv_data_ptr=0x10000,
    )
    manager.topology = SimpleNamespace(tp_rank=1)

    # Rank 1 has no edge for the replicated latent field in layer 0. An empty
    # fragment set for this interval is a no-op, never a raw-page identity copy.
    schedule = make_producer_schedule(
        *_heterogeneous_groups(destination=False), steps=3
    )

    def blocks(begin, end):
        return list(
            manager._cache_transfer_blocks(
                dst_ptr=0x20000,
                src_block_manifest=None,
                dst_block_manifest=destination_block_manifest,
                transfer_fragments=rank_one_fragments,
                dst_cache_layout=destination_layout,
                block_selection=selection,
                field_ids=schedule.fields_in_range(begin, end),
            )
        )

    assert blocks(0, 1) == []
    assert len(blocks(2, 3)) == 1


def _layerwise_fanout_context():
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    layout = _ordinary_layerwise_layout()
    block_manifest = make_block_manifest(("history", (3,)))
    requests = tuple(
        TransferInfo(
            room=9,
            mooncake_session_id=f"session-{rank}",
            block_manifest=block_manifest,
        )
        for rank in range(2)
    )
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.producer_schedule = make_producer_schedule(
        _ordinary_layerwise_group(), steps=5
    )
    manager.kv_args = SimpleNamespace(cache_layout=layout)
    manager.producer_step_count = manager.producer_schedule.step_count
    manager.session_lock = nullcontext()
    manager._is_session_failed = lambda _session: False
    manager.decode_kv_args_table = {
        request.mooncake_session_id: SimpleNamespace(
            dst_kv_ptr=0x20000 + rank * 0x1000,
            endpoint=f"decode-{rank}",
            dst_port=9000 + rank,
            peer_cache_layout=layout,
            transfer_fragments=(),
        )
        for rank, request in enumerate(requests)
    }
    manager.layerwise_debug = False
    manager.kv_transfer_metrics = None
    manager.session_failures = defaultdict(int)
    return manager, requests


def test_layerwise_fanout_is_layer_major_and_completes_once() -> None:
    manager, requests = _layerwise_fanout_context()
    chunk = TransferKVChunk(
        room=9,
        is_last=True,
        bootstrap_token=42,
        begin_cache_step=100,
        layerwise_interval=2,
        wait_for_bootstrap_token=True,
        cache_block_selection=_single_history_selection(),
    )
    events = []
    status_updates = []
    notifications = []
    manager._wait_until_cache_step = (
        lambda step, *, room, session_ids=(): events.append(("wait", step, room))
    )

    def cache_transfer_blocks(**kwargs):
        destination = kwargs["dst_ptr"]
        field_ids = kwargs["field_ids"]
        events.append(("prepare", destination, field_ids))
        yield (len(field_ids), destination, len(field_ids))

    manager._cache_transfer_blocks = cache_transfer_blocks

    def transfer_data(session, blocks, packer=None):
        events.append(("send", session, tuple(blocks)))
        return 0

    manager._transfer_data = transfer_data
    manager._wait_prefill_metadata = lambda *_args: (42, None)
    manager.check_status = lambda _room: TransferPoll.WaitingForInput
    manager.update_status = lambda room, status: status_updates.append((room, status))
    manager.sync_status_to_decode_endpoint = (
        lambda endpoint, port, room, status, rank, **kwargs: notifications.append(
            (endpoint, port, room, status, rank, kwargs)
        )
    )
    manager.topology = SimpleNamespace(tp_rank=0)
    manager.abort_room = lambda *_args: pytest.fail("successful fanout must not abort")

    assert manager._send_cache_layerwise_fanout(chunk, requests)

    assert [event[:2] for event in events] == [
        ("wait", 101),
        ("prepare", 0x20000),
        ("prepare", 0x21000),
        ("send", "session-0"),
        ("send", "session-1"),
        ("wait", 103),
        ("prepare", 0x20000),
        ("prepare", 0x21000),
        ("send", "session-0"),
        ("send", "session-1"),
        ("wait", 104),
        ("prepare", 0x20000),
        ("prepare", 0x21000),
        ("send", "session-0"),
        ("send", "session-1"),
    ]
    assert [event[2] for event in events if event[0] == "prepare"] == [
        frozenset(("layer.0.k", "layer.0.v")),
        frozenset(("layer.0.k", "layer.0.v")),
        frozenset(("layer.2.k",)),
        frozenset(("layer.2.k",)),
        frozenset(("layer.4.v",)),
        frozenset(("layer.4.v",)),
    ]
    assert status_updates == [(9, TransferPoll.Success)]
    assert [notification[:4] for notification in notifications] == [
        ("decode-0", 9000, 9, TransferPoll.Success),
        ("decode-1", 9001, 9, TransferPoll.Success),
    ]
    assert all(
        notification[5]["bootstrap_token"] == 42 for notification in notifications
    )


def test_layerwise_fanout_failure_aborts_before_later_intervals() -> None:
    manager, requests = _layerwise_fanout_context()
    chunk = TransferKVChunk(
        room=9,
        is_last=True,
        begin_cache_step=100,
        layerwise_interval=2,
        cache_block_selection=_single_history_selection(),
    )
    events = []
    aborts = []
    manager._wait_until_cache_step = lambda step, **_kwargs: events.append(
        ("wait", step)
    )
    manager._cache_transfer_blocks = lambda **_kwargs: iter(((1, 2, 3),))

    def transfer(session, _blocks, packer=None):
        events.append(("send", session))
        return -1 if session == "session-1" else 0

    manager._transfer_data = transfer
    manager._mark_session_failed = lambda session, reason: events.append(
        ("failed", session, reason)
    )
    manager.abort_room = lambda room, reason: aborts.append((room, reason))

    assert not manager._send_cache_layerwise_fanout(chunk, requests)
    assert events[:3] == [
        ("wait", 101),
        ("send", "session-0"),
        ("send", "session-1"),
    ]
    assert not any(event == ("wait", 103) for event in events)
    assert len(aborts) == 1


def test_dsa_sparse_prefill_publishes_one_cache_step_after_cache_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_backend

    events = []
    backend = object.__new__(dsa_backend.DSABackend)
    backend.index_topk = 2
    backend.data_type = torch.bfloat16
    backend.qk_nope_head_dim = 1
    backend.kv_lora_rank = 1
    backend.qk_rope_head_dim = 0
    backend.kernel_page_size = 64
    backend.step_counter = SimpleNamespace(record_cache=lambda: events.append("ready"))

    monkeypatch.setattr(
        dsa_backend,
        "workspace_topk_to_global_slots",
        lambda **_kwargs: torch.zeros((1, 2), dtype=torch.int64),
    )

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
            k_scale_float=None,
            scaling=1.0,
        ),
        token_to_kv_pool=SimpleNamespace(
            quant_method=None,
            get_key_buffer=lambda _layer_id: torch.zeros(1),
        ),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        workspace_indices=torch.zeros((1, 2), dtype=torch.int64),
        topk_lens=torch.ones(1, dtype=torch.int32),
        kv_workspace_slots=torch.zeros(1, dtype=torch.int64),
        max_seq_len=1,
    )

    assert output.shape == (1, 1)
    assert events == ["attention", "ready"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
