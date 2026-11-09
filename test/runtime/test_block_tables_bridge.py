from __future__ import annotations

import os
import sys
from test.runtime.conftest import whole_row_spans, whole_row_windows
from types import SimpleNamespace

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _exact_spans(arrays):
    """Per-row (released, pages): the leading run of 0s and the position
    where the trailing run of -1s starts -- exact for any int32 table,
    regardless of what sits between them (including an embedded 0 or -1)."""
    import numpy as np

    spans = {}
    for group_id, arr in arrays.items():
        rows, cols = arr.shape
        released = np.zeros(rows, dtype=np.int32)
        pages = np.full(rows, cols, dtype=np.int32)
        for r in range(rows):
            row = arr[r]
            rel = 0
            while rel < cols and row[rel] == 0:
                rel += 1
            released[r] = rel
            pg = cols
            while pg > rel and row[pg - 1] == -1:
                pg -= 1
            pages[r] = pg
        spans[group_id] = np.stack([released, pages], axis=1).astype(np.int32)
    return spans


def _import_bridge():
    """Import the bridge; skip if torch / tokenspeed_scheduler ext absent."""
    from tokenspeed.runtime.engine.scheduler_utils import (
        block_tables_from_forward_op,
    )

    return block_tables_from_forward_op


@pytest.fixture
def scheduler_forward_op():
    import tokenspeed_scheduler as ts

    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 1
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=group_id,
            block_granularity=2,
            total_pages=32,
            attn_kind=ts.AttnKind.Full,
        )
        for group_id in ("full", "swa")
    ]
    scheduler = ts.Scheduler(cfg)
    spec = ts.RequestSpec()
    spec.request_id = "page-id-validation"
    spec.tokens = [1, 2, 3, 4]
    scheduler.submit_requests([spec])
    plan = scheduler.next_execution_plan()
    assert len(plan.forward) == 1
    yield plan.forward[0]


class TestBlockTablesBridgeTest:
    def setup_method(self):
        try:
            self.bridge = _import_bridge()
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(
                f"cache bridge unavailable (needs torch + tokenspeed_scheduler "
                f"extension): {exc}"
            )
        import torch

        self.torch = torch

    def _make_op(self, block_tables, *, spans_kind="exact"):
        from types import SimpleNamespace

        import numpy as np

        def rect(v):
            a = np.asarray(v, dtype=np.int32)
            return a if a.ndim == 2 else a.reshape(len(v), 0)

        arrays = {k: rect(v) for k, v in block_tables.items()}
        spans = (
            whole_row_spans(arrays) if spans_kind == "whole" else _exact_spans(arrays)
        )
        return SimpleNamespace(
            block_tables=block_tables,
            block_table_windows=lambda: whole_row_windows(arrays),
            block_table_spans=lambda: spans,
        )

    def test_two_groups_shape_and_null_hole_preserved(self):
        op = self._make_op(
            {
                "full": [[11, 12], [13, 0]],
                "swa": [[21], [0]],
            }
        )
        out = self.bridge(op, device="cpu", num_reqs=2)
        assert (set(out.keys())) == ({"full", "swa"})

        full = out["full"]
        assert (tuple(full.shape)) == ((2, 2))
        assert (full.dtype) == (self.torch.int32)
        assert (full.tolist()) == ([[11, 12], [13, 0]])

        swa = out["swa"]
        assert (tuple(swa.shape)) == ((2, 1))
        assert (swa.tolist()) == ([[21], [0]])

    @pytest.mark.parametrize("spans_kind", ["exact", "whole"])
    def test_two_groups_shape_and_null_hole_preserved_both_span_kinds(self, spans_kind):
        op = self._make_op(
            {
                "full": [[11, 12], [13, 0]],
                "swa": [[21], [0]],
            },
            spans_kind=spans_kind,
        )
        out = self.bridge(op, device="cpu", num_reqs=2)
        assert out["full"].tolist() == [[11, 12], [13, 0]]
        assert out["swa"].tolist() == [[21], [0]]

    def test_window_export_is_consumed(self):
        op = self._make_op({"full": [[1]]})
        out = self.bridge(op, device="cpu", num_reqs=1)
        assert (out["full"].tolist()) == ([[1]])

    def test_row_count_mismatch_raises(self):
        op = self._make_op({"full": [[1, 2]]})
        with pytest.raises((ValueError)):
            self.bridge(op, device="cpu", num_reqs=2)

    def test_empty_rows_group_on_live_batch_raises(self):
        # An empty row list may not silently vanish on a live op: downstream
        # replay would see a per-group hole over stale pages.
        op = self._make_op({"full": [[1, 2], [3, 4]], "swa": []})
        with pytest.raises((ValueError), match=(r"swa.*0 rows")):
            self.bridge(op, device="cpu", num_reqs=2)

    def test_empty_rows_group_on_zero_req_op_dropped(self):
        # bs==0 replay/idle paths treat the resulting {} as "no tables".
        op = self._make_op({"full": [], "swa": []})
        assert (self.bridge(op, device="cpu", num_reqs=0)) == ({})
        assert (self.bridge(op, device="cpu")) == ({})

    def test_strict_contract_rejects_missing_extra_and_duplicate_normalized_ids(self):
        op = self._make_op({"full": [[1]]})
        with pytest.raises((ValueError), match=("missing=.*swa")):
            self.bridge(
                op,
                device="cpu",
                num_reqs=1,
                expected_group_ids=("full", "swa"),
            )

        op = self._make_op({"full": [[1]], "swa": [[2]], "extra": [[3]]})
        with pytest.raises((ValueError), match=("extra=.*extra")):
            self.bridge(
                op,
                device="cpu",
                num_reqs=1,
                expected_group_ids=("full", "swa"),
            )

        import numpy as np

        collision = SimpleNamespace(
            block_table_windows=lambda: {
                1: (np.array([[1]], dtype=np.int32), 1),
                "1": (np.array([[2]], dtype=np.int32), 1),
            },
            # Never reached: the "collide" ValueError below raises before any
            # group's spans are read. block_table_spans must still exist and
            # be callable, since the bridge calls it unconditionally.
            block_table_spans=lambda: {},
        )
        with pytest.raises((ValueError), match=("collide")):
            self.bridge(collision, device="cpu", num_reqs=1, max_page_id=8)

    @pytest.mark.parametrize("page_id", [-2, 99])
    def test_strict_contract_rejects_malformed_and_out_of_range_tables(self, page_id):
        import numpy as np

        wrong_dtype = SimpleNamespace(
            block_table_windows=lambda: {"full": (np.array([[1]], dtype=np.int64), 1)},
            # Same note as above: not reached before the dtype check raises.
            block_table_spans=lambda: {},
        )
        with pytest.raises((ValueError), match=("int32")):
            self.bridge(wrong_dtype, device="cpu", num_reqs=1, max_page_id=8)

        valid = self._make_op({"full": [[-1, 0, 8]]})
        valid_tables = self.bridge(valid, "cpu", num_reqs=1, max_page_id=8)
        assert valid_tables["full"].tolist() == [[-1, 0, 8]]
        out_of_range = self._make_op({"full": [[1, page_id]]})
        with pytest.raises((ValueError), match=("outside -1..8")):
            self.bridge(
                out_of_range,
                device="cpu",
                num_reqs=1,
                expected_group_ids=("full",),
                max_page_ids={"full": 8},
            )

    @pytest.mark.parametrize("per_group", [False, True])
    def test_page_ids_are_scanned_only_against_supplied_bounds(
        self, scheduler_forward_op, per_group
    ):
        bounds = (
            {"max_page_ids": {"full": 0, "swa": 0}} if per_group else {"max_page_id": 0}
        )
        arrays = scheduler_forward_op.block_tables_arrays()
        assert any((array > 0).any() for array in arrays.values())
        tables = self.bridge(
            scheduler_forward_op,
            "cpu",
            num_reqs=1,
            expected_group_ids=("swa", "full"),
        )
        assert list(tables) == ["swa", "full"]
        for group_id, array in arrays.items():
            assert tables[group_id].tolist() == array.tolist()
        assert tables["full"].untyped_storage().data_ptr() == (
            tables["swa"].untyped_storage().data_ptr()
        )

        with pytest.raises(ValueError, match="outside -1..0"):
            self.bridge(scheduler_forward_op, "cpu", num_reqs=1, **bounds)

    def test_page_id_bounds_must_cover_every_group(self, scheduler_forward_op):
        with pytest.raises(ValueError, match="missing cache group 'swa'"):
            self.bridge(
                scheduler_forward_op,
                "cpu",
                num_reqs=1,
                max_page_ids={"full": 31},
            )
        with pytest.raises(ValueError, match="not both"):
            self.bridge(
                scheduler_forward_op,
                "cpu",
                num_reqs=1,
                max_page_id=31,
                max_page_ids={"full": 31, "swa": 31},
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "npu"])
def test_packed_tables_keep_independent_host_staging(device_type):
    import numpy as np
    import torch

    if device_type == "npu":
        pytest.importorskip("torch_npu")
    if device_type != "cpu" and not getattr(torch, device_type).is_available():
        pytest.skip(f"{device_type} device unavailable")
    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    class ForwardTables:
        def __init__(self, value):
            self.arrays = {
                "state": np.array([[value, -1], [0, value + 1]], dtype=np.int32)
            }

        def block_table_windows(self):
            return whole_row_windows(self.arrays)

        def block_table_spans(self):
            return whole_row_spans(self.arrays)

    first = packed_block_tables_from_forward_op(
        ForwardTables(7), device_type, host_tables=True, num_reqs=2
    )
    second = packed_block_tables_from_forward_op(
        ForwardTables(19), device_type, host_tables=True, num_reqs=2
    )
    for result, value in ((first, 7), (second, 19)):
        assert result.tables_cpu["state"].is_pinned() == (device_type != "cpu")
        assert result.tables["state"].cpu().tolist() == [[value, -1], [0, value + 1]]
        assert result.tables_cpu["state"].tolist() == [[value, -1], [0, value + 1]]
    assert first.tables_cpu["state"].data_ptr() != second.tables_cpu["state"].data_ptr()


def test_packed_tables_host_tables_false_copies_nothing():
    """Without host tables the device tables are still exact and nothing
    is copied to the host."""
    import numpy as np

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    arrays = {"full": np.array([[1, 2], [3, 4]], dtype=np.int32)}
    op = SimpleNamespace(
        block_table_windows=lambda: whole_row_windows(arrays),
        block_table_spans=lambda: whole_row_spans(arrays),
    )
    result = packed_block_tables_from_forward_op(
        op, "cpu", host_tables=False, num_reqs=2
    )
    assert result.tables["full"].tolist() == [[1, 2], [3, 4]]
    assert result.tables_cpu == {}


def _minimal_cache_runtime_contract():
    """One full-history group's contract, built by hand like
    test_kvp_cache_contract.py's ``_contract`` -- no pool/recipe machinery
    needed for a single CacheBatchMetadata.from_forward_op call."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
        CacheRuntimeContract,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        CacheGroupSpec,
    )

    spec = CacheGroupSpec(
        group_id="full",
        attn_kind="full_history",
        rows_per_page=64,
        entry_stride_tokens=4,
        sliding_window_tokens=None,
        shard_count=1,
        replayable=False,
    )
    return CacheRuntimeContract(
        prefix_granularity=256,
        num_lcm_blocks=4,
        token_capacity=4 * 1 * 256,
        group_specs=(spec,),
        group_page_counts={"full": 1 + 4 * 1},
        group_packing={"full": 1},
    )


def test_cache_batch_metadata_tables_cpu_raises_without_host_tables():
    """Metadata built without host tables refuses ``tables_cpu`` instead of
    handing out empty or stale ones."""
    import numpy as np

    from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
        CacheBatchMetadata,
    )

    contract = _minimal_cache_runtime_contract()
    arrays = {"full": np.array([[1, 2]], dtype=np.int32)}
    spans = {"full": np.array([[0, 2]], dtype=np.int32)}
    forward_op = SimpleNamespace(
        block_table_windows=lambda: whole_row_windows(arrays),
        block_table_spans=lambda: spans,
    )
    metadata = CacheBatchMetadata.from_forward_op(
        forward_op,
        device="cpu",
        contract=contract,
        num_requests=1,
        validate_page_ids=True,
        host_tables=False,
    )
    assert metadata.tables(active_forward_op=forward_op)["full"].tolist() == [[1, 2]]
    with pytest.raises(RuntimeError, match="without host tables"):
        metadata.tables_cpu(active_forward_op=forward_op)


def test_cache_batch_metadata_tables_cpu_available_with_host_tables():
    import numpy as np

    from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
        CacheBatchMetadata,
    )

    contract = _minimal_cache_runtime_contract()
    arrays = {"full": np.array([[1, 2]], dtype=np.int32)}
    spans = {"full": np.array([[0, 2]], dtype=np.int32)}
    forward_op = SimpleNamespace(
        block_table_windows=lambda: whole_row_windows(arrays),
        block_table_spans=lambda: spans,
    )
    metadata = CacheBatchMetadata.from_forward_op(
        forward_op,
        device="cpu",
        contract=contract,
        num_requests=1,
        validate_page_ids=True,
        host_tables=True,
    )
    assert metadata.tables_cpu(active_forward_op=forward_op)["full"].tolist() == [
        [1, 2]
    ]


def _malformed_fuzz_arrays(seed: int):
    """One round's per-group int32 tables with deliberately malformed rows:
    holes and padding anywhere, pages before holes, holes inside the live
    span, all-padding rows, all-hole rows, a width-1 group, and extreme int32
    values. ``packed_block_tables_from_forward_op``/``expand_block_table_rows``
    must handle these exactly through either span kind -- the real scheduler
    never produces them, so this is the only way to cover them (see AGENTS.md
    on mocks)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    num_groups = int(rng.integers(1, 4))
    groups = {}
    for g in range(num_groups):
        rows = int(rng.integers(1, 6))
        cols = int(rng.integers(1, 12))
        arr = rng.integers(-1, 50, size=(rows, cols), dtype=np.int32)
        pattern = int(rng.integers(0, 6))
        if pattern == 0:
            pass  # fully random row-level malformation
        elif pattern == 1:
            arr[0, :] = 0  # all-hole row
        elif pattern == 2:
            arr[0, :] = -1  # all-padding row
        elif pattern == 3 and cols >= 4:
            mid = cols // 2
            arr[0, :mid] = 0
            arr[0, mid] = 7  # a live page
            arr[0, mid + 1] = 0  # a hole INSIDE the live span
            arr[0, mid + 2 :] = -1
        elif pattern == 4:
            arr[0, 0] = np.iinfo(np.int32).min
            arr[0, -1] = np.iinfo(np.int32).max
        else:
            arr[0, -1] = 50  # a page right after another row's padding start
            arr[-1, 0] = -1  # padding at a row's very first column
        groups[f"group{g}"] = arr
    return groups


class _MalformedForwardOp:
    """Exposes block_table_windows()/block_table_spans() like a scheduler
    forward op, but with rows the real scheduler cannot produce (see module
    docstring above)."""

    def __init__(self, arrays, spans):
        self._arrays = arrays
        self._spans = spans

    def block_table_windows(self):
        import numpy as np

        windows = {}
        for group, table in self._arrays.items():
            released, pages = self._spans[group].T
            width = int((pages - released).max())
            first = np.maximum(pages - width, 0)
            window = table[
                np.arange(table.shape[0])[:, None], first[:, None] + np.arange(width)
            ]
            windows[group] = (window, table.shape[1])
        return windows

    def block_table_spans(self):
        return self._spans


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("spans_kind", ["exact", "whole"])
@pytest.mark.parametrize("seed", range(12))
def test_packed_tables_fuzz_malformed_rows(device_type, spans_kind, seed):
    """Any int32 table -- not only the scheduler's holes, pages, padding
    layout -- comes back exactly through exact or whole-row spans, every
    group in one device storage."""
    import torch

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda device unavailable")
    groups = _malformed_fuzz_arrays(seed)
    spans = (whole_row_spans if spans_kind == "whole" else _exact_spans)(groups)
    result = packed_block_tables_from_forward_op(
        _MalformedForwardOp(groups, spans), device_type, host_tables=True
    )
    for key, arr in groups.items():
        assert result.tables[key].cpu().numpy().tolist() == arr.tolist(), key
        assert result.tables_cpu[key].numpy().tolist() == arr.tolist(), key
    storages = {
        result.tables[key].untyped_storage().data_ptr()
        for key, arr in groups.items()
        if arr.size
    }
    assert len(storages) <= 1


@pytest.mark.parametrize("spans_kind", ["exact", "whole"])
def test_packed_tables_malformed_rows_fixed_cases_cpu(spans_kind):
    """A deterministic companion to the seeded fuzz above, so a reviewer who
    disables the fuzz still exercises every listed malformation once, under
    both span kinds: width-1 group, all-hole row, all-padding row, a hole
    inside the live span, and INT32_MIN/MAX page IDs."""
    import numpy as np
    import torch

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    groups = {
        "width_one": np.array([[0], [-1], [7]], dtype=np.int32),
        "all_hole": np.array([[0, 0, 0]], dtype=np.int32),
        "all_padding": np.array([[-1, -1, -1]], dtype=np.int32),
        "hole_inside_live_span": np.array([[0, 0, 5, 0, 7, -1, -1]], dtype=np.int32),
        "int32_extremes": np.array(
            [[np.iinfo(np.int32).min, 3, np.iinfo(np.int32).max]], dtype=np.int32
        ),
    }
    spans = (whole_row_spans if spans_kind == "whole" else _exact_spans)(groups)
    result = packed_block_tables_from_forward_op(
        _MalformedForwardOp(groups, spans), torch.device("cpu"), host_tables=True
    )
    for key, arr in groups.items():
        assert result.tables[key].tolist() == arr.tolist(), key
        assert result.tables_cpu[key].tolist() == arr.tolist(), key


def _span_op(spans):
    """A one-group op over the row [1, 2, -1] with the given span export."""
    import numpy as np

    arr = np.array([[1, 2, -1]], dtype=np.int32)
    return SimpleNamespace(
        block_table_windows=lambda: {"g": (arr, arr.shape[1])},
        block_table_spans=lambda: spans,
    )


@pytest.mark.parametrize(
    ("spans", "match"),
    [
        ({}, r"block_table_spans\[g\]"),
        ({"g": [[0, 2]], "dtype": "int64"}, r"block_table_spans\[g\]"),
        ({"g": [[0, 2, 0]]}, r"block_table_spans\[g\]"),
        ({"g": [[0, 2], [0, 2]]}, r"block_table_spans\[g\]"),
        ({"g": [0, 2]}, r"block_table_spans\[g\]"),
        ({"g": [[2, 1]]}, "exceeds its table"),
        ({"g": [[0, 4]]}, "exceeds its table"),
        ({"g": [[-1, 2]]}, "exceeds its table"),
    ],
)
def test_malformed_spans_are_rejected_without_bounds(spans, match):
    """A missing, mis-shaped or out-of-row span would make the device rebuild
    read past its upload; it is rejected on every batch, not only under the
    page-id scan."""
    import numpy as np

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    dtype = np.int64 if "dtype" in spans else np.int32
    op = _span_op(
        {
            gid: np.array(rows, dtype=dtype)
            for gid, rows in spans.items()
            if gid != "dtype"
        }
    )
    with pytest.raises(ValueError, match=match):
        packed_block_tables_from_forward_op(op, "cpu", host_tables=False)


@pytest.mark.parametrize("rows", [[[1, 3]], [[0, 1]]])
def test_spans_disagreeing_with_their_rows_fail_the_page_id_scan(rows):
    """A span whose holes cover a page (released 1 over entry 1) or whose
    padding covers one (pages 1 before entry 2) is caught when page-id bounds
    ask the packer to scan every row."""
    import numpy as np

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    op = _span_op({"g": np.array(rows, dtype=np.int32)})
    with pytest.raises(ValueError, match="holds entries outside its row spans"):
        packed_block_tables_from_forward_op(op, "cpu", host_tables=False, max_page_id=8)


def _real_scheduler_module():
    """Import the C++ scheduler extension; skip the caller's test if it is
    not built in this environment."""
    import pytest as _pytest

    ts = _pytest.importorskip("tokenspeed_scheduler")
    return ts


def _scheduler_bridge_config(ts, *, max_scheduled_tokens: int, num_usable_pages: int):
    """One SlidingWindow group (replayable, so a prefix hit re-feeds its
    window) and one Full group, Fused role, prefix caching on -- the shape
    ``packed_block_tables_from_forward_op`` is built to pack."""
    PAGE = 4
    WINDOW_TOKENS = 16
    cfg = ts.SchedulerConfig()
    cfg.num_device_pages = num_usable_pages + 1  # page 0 is the null sentinel
    cfg.num_host_pages = 0
    cfg.disable_l2_cache = True
    cfg.enable_l3_storage = False
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = 8
    cfg.num_snapshot_pages = cfg.num_device_pages
    cfg.max_retracted_requests = cfg.max_batch_size
    cfg.prefix_granularity = PAGE
    cfg.decode_input_tokens = 1
    cfg.overlap_schedule_depth = 0
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="swa",
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            attn_kind=ts.AttnKind.SlidingWindow,
            sliding_window_tokens=WINDOW_TOKENS,
            replayable=True,
        ),
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            attn_kind=ts.AttnKind.Full,
        ),
    ]
    cfg.host_group_packing = [1, 1]
    return cfg, PAGE, WINDOW_TOKENS


def _land_cache_ops(ts, scheduler, plan) -> int:
    """Acknowledge every snapshot/restore/write-back of the plan as if its
    copy finished at once. Returns how many requests the plan retracted."""
    events = ts.ExecutionEvent()
    num_events = 0
    retracted = 0
    for op in plan.cache:
        if isinstance(op, ts.Cache.SnapshotOp):
            retracted += len(op.request_ids)
            done_type = ts.Cache.SnapshotDoneEvent
        elif isinstance(op, ts.Cache.RestoreOp):
            done_type = ts.Cache.RestoreDoneEvent
        elif isinstance(op, ts.Cache.WriteBackOp):
            done_type = ts.Cache.WriteBackDoneEvent
        else:
            continue
        for op_id in op.op_ids:
            done = done_type()
            done.op_id = int(op_id)
            events.add_event(done)
            num_events += 1
    if num_events:
        scheduler.advance(events)
    return retracted


def _assert_spans_agree_with_arrays(arrays, spans):
    """``row[:released]`` is all 0 and ``row[pages:]`` is all -1, for every
    group and row of one forward op -- the span/array agreement the bridge
    assumes unconditionally (not only when bounds are supplied)."""
    for key, arr in arrays.items():
        released, pages = spans[key][:, 0], spans[key][:, 1]
        for r in range(arr.shape[0]):
            assert (arr[r, : released[r]] == 0).all(), (key, r, "released")
            assert (arr[r, pages[r] :] == -1).all(), (key, r, "pages")


def test_forward_tables_match_through_chunking_prefix_hit_and_sliding_holes():
    """Through chunked prefill, a prefix hit that replays the sliding window
    and decode steps that slide it into null holes, every forward's device
    and host tables equal the scheduler's, its spans agree with its rows,
    and a pure decode's sliding-window span stays within the window."""
    import math

    import torch

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    ts = _real_scheduler_module()
    # The scheduler requires max_scheduled_tokens >= window + max(window,
    # prefix_granularity) for a replayable sliding-window group (32 here);
    # a 48-token prompt then needs more than one chunk to admit.
    cfg, page, window_tokens = _scheduler_bridge_config(
        ts, max_scheduled_tokens=32, num_usable_pages=80
    )
    scheduler = ts.Scheduler(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_window_pages = math.ceil(window_tokens / page) + 2

    def spec(request_id, tokens, max_new_tokens):
        s = ts.RequestSpec()
        s.request_id = request_id
        s.tokens = list(tokens)
        s.max_new_tokens = max_new_tokens
        return s

    shared = list(range(100, 100 + 40))  # 40-token shared prefix
    r0 = spec("r0", shared + list(range(1, 9)), max_new_tokens=40)
    siblings = [
        spec(
            f"r{k}", shared + list(range(900 + 10 * k, 908 + 10 * k)), max_new_tokens=40
        )
        for k in (1, 2)
    ]
    scheduler.submit_requests([r0])

    generated = {req.request_id: 0 for req in [r0, *siblings]}
    finished: set[str] = set()
    saw_chunking = False
    saw_prefix_hit = False
    saw_sliding_hole = False
    siblings_submitted = False

    for _ in range(400):
        plan = scheduler.next_execution_plan()
        _land_cache_ops(ts, scheduler, plan)
        batch = next((op for op in plan.forward if list(op.request_ids)), None)
        if batch is None:
            if finished == set(generated):
                break
            continue

        arrays = batch.block_tables_arrays()
        spans = batch.block_table_spans()
        _assert_spans_agree_with_arrays(arrays, spans)
        packed = packed_block_tables_from_forward_op(batch, device, host_tables=True)
        for key, arr in arrays.items():
            assert packed.tables[key].cpu().numpy().tolist() == arr.tolist(), (
                key,
                list(batch.request_ids),
            )
            assert packed.tables_cpu[key].numpy().tolist() == arr.tolist(), key

        num_extends = batch.num_extends()
        prefill_lengths = list(batch.prefill_lengths)
        extend_prefix_lens = list(batch.extend_prefix_lens)
        input_lengths = list(batch.input_lengths)
        for row in range(num_extends):
            if extend_prefix_lens[row] + input_lengths[row] < prefill_lengths[row]:
                saw_chunking = True
            if extend_prefix_lens[row] > 0:
                saw_prefix_hit = True
        if num_extends == 0:
            swa_released, swa_pages = spans["swa"][:, 0], spans["swa"][:, 1]
            width = int((swa_pages - swa_released).max())
            assert width <= max_window_pages, (width, max_window_pages)
            if (arrays["swa"] == 0).any():
                saw_sliding_hole = True

        # Siblings arrive once r0 has produced at least one decode token, so
        # their shared prefix is published and hittable.
        if not siblings_submitted and generated["r0"] >= 1:
            scheduler.submit_requests(siblings)
            siblings_submitted = True

        events = ts.ExecutionEvent()
        num_events = 0
        for row, rid in enumerate(batch.request_ids):
            if rid in finished:
                continue
            if row < num_extends and extend_prefix_lens[row] + input_lengths[row] < (
                prefill_lengths[row]
            ):
                continue  # an intermediate prefill chunk produces no token
            result = ts.ForwardEvent.ExtendResult()
            result.request_id = rid
            generated[rid] += 1
            result.tokens = [9000 + generated[rid]]
            events.add_event(result)
            num_events += 1
            if generated[rid] >= 40:
                finished.add(rid)
                finish = ts.ForwardEvent.Finish()
                finish.request_id = rid
                events.add_event(finish)
            elif row >= num_extends:
                # Only decode rows update the reserve, as the liveness
                # driver does: the row that completes the prompt already
                # carries the decode slot's reservation.
                reserve = ts.ForwardEvent.UpdateReserveNumTokens()
                reserve.request_id = rid
                reserve.reserve_num_tokens_in_next_schedule_event = 1
                events.add_event(reserve)
        if num_events:
            scheduler.advance(events)

    assert finished == set(generated), (finished, generated)
    assert siblings_submitted, "the shared-prefix siblings were never admitted"
    assert saw_chunking, "the small token budget never produced an intermediate chunk"
    assert saw_prefix_hit, "no sibling request hit the published shared prefix"
    assert saw_sliding_hole, "decode never slid the sliding-window group's pages"


def test_forward_tables_match_through_retraction():
    """Same table-equality and span-agreement contract as above, under a
    device pool small enough relative to concurrent demand that the victim
    policy retracts a request (image stored, pages freed) and later
    restores it. If retraction never fires in this environment, that is
    reported, not silently dropped: see the final assertion."""
    import torch

    from tokenspeed.runtime.engine.scheduler_utils import (
        packed_block_tables_from_forward_op,
    )

    ts = _real_scheduler_module()
    # Mirrors tokenspeed-scheduler/python/tests/test_retraction_image.py's
    # proven shape: admit everyone with no growth headroom (one request with
    # 3x the pages of the other three), then decode until every request's
    # growth is blocked, so the policy must retract the one with the most to
    # release.
    cfg, _page, _window_tokens = _scheduler_bridge_config(
        ts, max_scheduled_tokens=64, num_usable_pages=20
    )
    scheduler = ts.Scheduler(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def spec(request_id, tokens, max_new_tokens):
        s = ts.RequestSpec()
        s.request_id = request_id
        s.tokens = list(tokens)
        s.max_new_tokens = max_new_tokens
        return s

    request_ids = ("a", "b", "c", "d")
    specs = [spec("a", list(range(1, 13)), max_new_tokens=20)] + [
        spec(rid, list(range(100 * i + 1, 100 * i + 5)), max_new_tokens=20)
        for i, rid in enumerate(request_ids[1:], 1)
    ]
    scheduler.submit_requests(specs)

    generated = {s.request_id: 0 for s in specs}
    finished: set[str] = set()
    retracted = 0

    for _ in range(600):
        plan = scheduler.next_execution_plan()
        retracted += _land_cache_ops(ts, scheduler, plan)
        batch = next((op for op in plan.forward if list(op.request_ids)), None)
        if batch is None:
            if finished == set(generated):
                break
            continue

        arrays = batch.block_tables_arrays()
        spans = batch.block_table_spans()
        _assert_spans_agree_with_arrays(arrays, spans)
        packed = packed_block_tables_from_forward_op(batch, device, host_tables=True)
        for key, arr in arrays.items():
            assert packed.tables[key].cpu().numpy().tolist() == arr.tolist(), key
            assert packed.tables_cpu[key].numpy().tolist() == arr.tolist(), key

        num_extends = batch.num_extends()
        prefill_lengths = list(batch.prefill_lengths)
        extend_prefix_lens = list(batch.extend_prefix_lens)
        input_lengths = list(batch.input_lengths)

        events = ts.ExecutionEvent()
        num_events = 0
        for row, rid in enumerate(batch.request_ids):
            if rid in finished:
                continue
            if row < num_extends and extend_prefix_lens[row] + input_lengths[row] < (
                prefill_lengths[row]
            ):
                continue
            result = ts.ForwardEvent.ExtendResult()
            result.request_id = rid
            generated[rid] += 1
            result.tokens = [9000 + generated[rid]]
            events.add_event(result)
            num_events += 1
            if generated[rid] >= 20:
                finished.add(rid)
                finish = ts.ForwardEvent.Finish()
                finish.request_id = rid
                events.add_event(finish)
            elif row >= num_extends:
                reserve = ts.ForwardEvent.UpdateReserveNumTokens()
                reserve.request_id = rid
                reserve.reserve_num_tokens_in_next_schedule_event = 1
                events.add_event(reserve)
        if num_events:
            scheduler.advance(events)

    assert finished == set(generated), (finished, generated)
    assert retracted > 0, (
        "this configuration never triggered a retraction; the table-equality "
        "assertions above still ran on every forward op, but the retraction "
        "+ restore path specifically is uncovered by this test"
    )
