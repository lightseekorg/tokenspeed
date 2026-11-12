from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_scheduler")
torch = pytest.importorskip("torch")


def _scheduler(*, mixed: bool = False):
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 1
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.max_resident_requests = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.enable_mixed_prefill_decode = mixed
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=2,
            total_pages=32,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="swa",
            block_granularity=2,
            total_pages=32,
            attn_kind=ts.AttnKind.SlidingWindow,
            sliding_window_tokens=4,
        ),
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=32,
            attn_kind=ts.AttnKind.State,
        ),
    ]
    return ts.Scheduler(cfg)


def _submit(scheduler, request_id: str, tokens: list[int]) -> None:
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    scheduler.submit_requests([spec])


def _advance(scheduler, request_id: str, tokens: list[int]) -> None:
    event = ts.ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = tokens
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _forward_op(plan):
    assert plan.forward
    return plan.forward[0]


def _decode_op(*, stop_token: int, include_prefill: bool = False):
    scheduler = _scheduler(mixed=True)
    _submit(scheduler, "decode", list(range(1, 9)))
    _forward_op(scheduler.next_execution_plan())
    for token in range(9, stop_token + 1):
        _advance(scheduler, "decode", [token])
        if include_prefill and token == stop_token:
            _submit(scheduler, "new-prefill", [17, 18, 19, 20])
        plan = scheduler.next_execution_plan()
    return _forward_op(plan)


def _assert_packed_matches_absolute(op, result):
    legacy = op.block_tables_arrays()
    assert list(result.tables) == list(legacy)
    assert list(result.tables_cpu) == list(legacy)
    for group_id, expected in legacy.items():
        assert result.tables[group_id].dtype == torch.int32
        assert result.tables[group_id].cpu().tolist() == expected.tolist()
        assert result.tables_cpu[group_id].tolist() == expected.tolist()


def _assert_windows_reconstruct_absolute(op):
    windows = op.block_table_windows()
    spans = op.block_table_spans()
    legacy = op.block_tables_arrays()
    for group_id, expected in legacy.items():
        window, logical_columns = windows[group_id]
        assert window.dtype == np.int32
        assert not window.flags.writeable
        assert logical_columns == expected.shape[1]
        row_spans = spans[group_id].reshape(-1, 2)
        reconstructed = np.full(expected.shape, -1, dtype=np.int32)
        width = window.shape[1]
        for row, (_, pages) in enumerate(row_spans):
            first = max(int(pages) - width, 0)
            reconstructed[row, :first] = 0
            reconstructed[row, first : first + width] = window[row]
        np.testing.assert_array_equal(reconstructed, expected)


def _window_op(window: np.ndarray, columns: int, spans: np.ndarray):
    return SimpleNamespace(
        block_table_windows=lambda: {"full": (window, columns)},
        block_table_spans=lambda: {"full": spans},
    )


def _multi_window_op(groups):
    return SimpleNamespace(
        block_table_windows=lambda: {
            group_id: (window, columns)
            for group_id, (window, columns, _) in groups.items()
        },
        block_table_spans=lambda: {
            group_id: spans for group_id, (_, _, spans) in groups.items()
        },
    )


def test_cpu_packing_expands_real_scheduler_windows_for_full_swa_and_state():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _decode_op(stop_token=16, include_prefill=True)
    _assert_windows_reconstruct_absolute(op)
    assert op.request_ids == ["new-prefill", "decode"]
    assert op.num_extends() == 1
    windows = op.block_table_windows()
    assert set(windows) == {"full", "swa", "state"}
    swa_window, swa_columns = windows["swa"]
    assert swa_window.shape == (2, 3)
    assert swa_columns == 8
    assert op.block_table_spans()["swa"].tolist() == [[0, 3], [6, 8]]

    result = packed_block_tables_from_forward_op(
        op, "cpu", host_tables=True, num_reqs=len(op.request_ids)
    )
    _assert_packed_matches_absolute(op, result)


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_device_packing_keeps_absolute_rows_and_host_mirror(device_type):
    if device_type == "npu":
        pytest.importorskip("torch_npu")
    if not getattr(torch, device_type).is_available():
        pytest.skip(f"{device_type} device unavailable")

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _decode_op(stop_token=16, include_prefill=True)
    _assert_windows_reconstruct_absolute(op)
    assert op.request_ids == ["new-prefill", "decode"]
    assert op.num_extends() == 1
    swa_window, swa_columns = op.block_table_windows()["swa"]
    assert swa_window.shape == (2, 3)
    assert swa_columns == 8
    assert op.block_table_spans()["swa"].tolist() == [[0, 3], [6, 8]]
    result = packed_block_tables_from_forward_op(
        op, device_type, host_tables=True, num_reqs=len(op.request_ids)
    )
    _assert_packed_matches_absolute(op, result)


def test_decode_window_compacts_long_reclaimed_prefix():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _decode_op(stop_token=10)
    assert op.request_ids == ["decode"]
    assert op.num_extends() == 0
    window, columns = op.block_table_windows()["swa"]
    assert window.shape == (1, 2)
    assert columns == 5
    assert op.block_table_spans()["swa"].tolist() == [[3, 5]]
    legacy = op.block_tables_arrays()["swa"]
    expected = legacy.copy()
    assert not legacy.flags.writeable
    assert legacy.ctypes.data == op.block_tables_arrays()["swa"].ctypes.data
    result = packed_block_tables_from_forward_op(
        op, "cpu", host_tables=True, num_reqs=1
    )
    _assert_packed_matches_absolute(op, result)
    del op
    gc.collect()
    assert np.array_equal(legacy, expected)


def test_packed_views_keep_cpu_mirror_alive_after_plan_release():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    scheduler = _scheduler()
    _submit(scheduler, "prefill", [1, 2, 3, 4])
    plan = scheduler.next_execution_plan()
    op = _forward_op(plan)
    expected = {key: value.copy() for key, value in op.block_tables_arrays().items()}
    packed = packed_block_tables_from_forward_op(
        op, "cpu", host_tables=True, num_reqs=1
    )
    del plan, op
    gc.collect()
    for group_id, table in packed.tables_cpu.items():
        assert table.tolist() == expected[group_id].tolist()


def test_fully_reclaimed_nonzero_width_row_accepts_zero_width_window():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.empty((1, 0), dtype=np.int32),
        4,
        np.array([[4, 4]], dtype=np.int32),
    )
    result = packed_block_tables_from_forward_op(
        op,
        "cpu",
        host_tables=True,
        num_reqs=1,
        expected_group_ids=("full",),
    )
    assert result.tables["full"].tolist() == [[0, 0, 0, 0]]
    assert result.tables_cpu["full"].tolist() == [[0, 0, 0, 0]]


def test_page_bound_rejects_implicit_zero_in_fully_reclaimed_row():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.empty((1, 0), dtype=np.int32),
        4,
        np.array([[4, 4]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="outside -1..-1"):
        packed_block_tables_from_forward_op(
            op,
            "cpu",
            host_tables=True,
            num_reqs=1,
            expected_group_ids=("full",),
            max_page_id=-1,
        )


def test_strict_validation_rejects_negative_page_id():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.array([[-2]], dtype=np.int32),
        1,
        np.array([[0, 1]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="outside -1..8"):
        packed_block_tables_from_forward_op(
            op,
            "cpu",
            host_tables=True,
            num_reqs=1,
            expected_group_ids=("full",),
            max_page_id=8,
        )


def test_strict_validation_rejects_entries_outside_row_span():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.array([[3]], dtype=np.int32),
        1,
        np.array([[0, 0]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="outside its row spans"):
        packed_block_tables_from_forward_op(
            op,
            "cpu",
            host_tables=True,
            num_reqs=1,
            expected_group_ids=("full",),
            max_page_id=8,
        )


def test_invalid_logical_columns_are_rejected():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.array([[3, 4]], dtype=np.int32),
        1,
        np.array([[0, 1]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="invalid absolute columns"):
        packed_block_tables_from_forward_op(
            op,
            "cpu",
            host_tables=True,
            num_reqs=1,
            expected_group_ids=("full",),
        )


def test_logical_columns_must_fit_int32_before_allocation():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.empty((1, 0), dtype=np.int32),
        1 << 31,
        np.array([[0, 0]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r"2\*\*31"):
        packed_block_tables_from_forward_op(
            op,
            "cpu",
            host_tables=True,
            num_reqs=1,
            expected_group_ids=("full",),
        )


def test_total_table_elements_must_fit_int32_before_allocation():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    groups = {
        group_id: (
            np.empty((1, 0), dtype=np.int32),
            1 << 30,
            np.array([[0, 0]], dtype=np.int32),
        )
        for group_id in ("first", "second")
    }
    op = _multi_window_op(groups)
    with pytest.raises(ValueError, match=r"2\*\*31"):
        packed_block_tables_from_forward_op(
            op,
            "cpu",
            host_tables=True,
            num_reqs=1,
            expected_group_ids=("first", "second"),
        )


def test_ragged_windows_rebuild_absolute_padding():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.array([[4, 5], [6, -1]], dtype=np.int32),
        3,
        np.array([[1, 3], [0, 1]], dtype=np.int32),
    )
    result = packed_block_tables_from_forward_op(
        op,
        "cpu",
        host_tables=True,
        num_reqs=2,
        expected_group_ids=("full",),
        max_page_id=8,
    )
    expected = [[0, 4, 5], [6, -1, -1]]
    assert result.tables["full"].tolist() == expected
    assert result.tables_cpu["full"].tolist() == expected


def test_window_cannot_omit_a_live_slot():
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _window_op(
        np.array([[4, 5]], dtype=np.int32),
        3,
        np.array([[0, 3]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="omits live entries"):
        packed_block_tables_from_forward_op(op, "cpu", host_tables=False)


def test_legacy_and_window_views_are_read_only_and_outlive_the_batch():
    op = _decode_op(stop_token=16, include_prefill=True)
    absolute = op.block_tables_arrays()["swa"]
    window, columns = op.block_table_windows()["swa"]
    assert window.shape[1] < columns == absolute.shape[1]
    absolute_before = absolute.copy()
    window_before = window.copy()
    for view in (absolute, window):
        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[0, 0] = 0

    del op
    gc.collect()
    np.testing.assert_array_equal(absolute, absolute_before)
    np.testing.assert_array_equal(window, window_before)
