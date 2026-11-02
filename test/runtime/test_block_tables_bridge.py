from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


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

    def _make_op(self, block_tables):
        from types import SimpleNamespace

        import numpy as np

        def rect(v):
            a = np.asarray(v, dtype=np.int32)
            return a if a.ndim == 2 else a.reshape(len(v), 0)

        arrays = {k: rect(v) for k, v in block_tables.items()}
        return SimpleNamespace(
            block_tables=block_tables,
            block_tables_arrays=lambda: arrays,
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

    def test_array_export_is_consumed(self):
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
            block_tables_arrays=lambda: {
                1: np.array([[1]], dtype=np.int32),
                "1": np.array([[2]], dtype=np.int32),
            }
        )
        with pytest.raises((ValueError), match=("collide")):
            self.bridge(collision, device="cpu", num_reqs=1, max_page_id=8)

    @pytest.mark.parametrize("page_id", [-2, 99])
    def test_strict_contract_rejects_malformed_and_out_of_range_tables(
        self, monkeypatch, page_id
    ):
        import numpy as np

        monkeypatch.delenv("TOKENSPEED_VALIDATE_PAGE_IDS", raising=False)
        wrong_dtype = SimpleNamespace(
            block_tables_arrays=lambda: {"full": np.array([[1]], dtype=np.int64)}
        )
        with pytest.raises((ValueError), match=("int32")):
            self.bridge(wrong_dtype, device="cpu", num_reqs=1, max_page_id=8)

        monkeypatch.setenv("TOKENSPEED_VALIDATE_PAGE_IDS", "1")
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
    def test_page_id_scan_is_opt_in(self, monkeypatch, scheduler_forward_op, per_group):
        monkeypatch.delenv("TOKENSPEED_VALIDATE_PAGE_IDS", raising=False)
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
            **bounds,
        )
        assert list(tables) == ["swa", "full"]
        for group_id, array in arrays.items():
            assert tables[group_id].tolist() == array.tolist()
        assert tables["full"].untyped_storage().data_ptr() == (
            tables["swa"].untyped_storage().data_ptr()
        )

        monkeypatch.setenv("TOKENSPEED_VALIDATE_PAGE_IDS", "1")
        with pytest.raises(ValueError, match="outside -1..0"):
            self.bridge(scheduler_forward_op, "cpu", num_reqs=1, **bounds)

    @pytest.mark.parametrize("validate_page_ids", ["0", "1"])
    def test_page_id_bound_contract_is_always_checked(
        self, monkeypatch, scheduler_forward_op, validate_page_ids
    ):
        monkeypatch.setenv("TOKENSPEED_VALIDATE_PAGE_IDS", validate_page_ids)
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

        def block_tables_arrays(self):
            return self.arrays

    first = packed_block_tables_from_forward_op(
        ForwardTables(7), device_type, num_reqs=2
    )
    second = packed_block_tables_from_forward_op(
        ForwardTables(19), device_type, num_reqs=2
    )
    for result, value in ((first, 7), (second, 19)):
        assert result.tables_cpu["state"].is_pinned() == (device_type != "cpu")
        assert result.tables["state"].cpu().tolist() == [[value, -1], [0, value + 1]]
        assert result.tables_cpu["state"].tolist() == [[value, -1], [0, value + 1]]
    assert first.tables_cpu["state"].data_ptr() != second.tables_cpu["state"].data_ptr()
