from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("kimi_k3_dspark_width_sweep.py")
SPEC = importlib.util.spec_from_file_location("kimi_k3_dspark_width_sweep", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def _write_result(root: Path, arm: str, *, tpot: float, output: float) -> None:
    directory = root / "tokenspeed" / arm
    directory.mkdir(parents=True)
    metadata = {
        "engine": "tokenspeed",
        "group": "output",
        "name": "1k-512-c1",
        "arm": arm,
        "repeat": 1,
        "isl": 1024,
        "osl": 512,
        "concurrency": 1,
    }
    if arm == "dspark":
        metadata.update(
            {"accept_length": 3.0, "accept_rate": 0.4, "speculative_width": 4}
        )
    payload = {
        "ab_metadata": metadata,
        "failed": 0,
        "median_tpot_ms": tpot,
        "output_throughput": output,
        "total_token_throughput": output * 3,
    }
    (directory / "result.json").write_text(json.dumps(payload))


def test_aggregate_widths_reports_break_even_economics(tmp_path) -> None:
    _write_result(tmp_path / "baseline", "no-spec", tpot=100, output=80)
    _write_result(tmp_path / "w4", "dspark", tpot=40, output=120)

    rows = sweep.aggregate_widths(tmp_path, [4])
    assert len(rows) == 1
    row = rows[0]
    assert row["verify_width"] == 4
    assert row["draft_count"] == 3
    assert row["output_speedup"] == pytest.approx(1.5)
    assert row["spec_iteration_ms"] == pytest.approx(120)
    assert row["break_even_accept"] == pytest.approx(1.2)
    assert row["accept_margin"] == pytest.approx(1.8)
