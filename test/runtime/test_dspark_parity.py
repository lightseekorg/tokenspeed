from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from tokenspeed.runtime.execution.dspark_parity import DSparkParityRecorder

SCRIPT = (
    Path(__file__).parents[1]
    / "random_benchmark"
    / "tokenspeed"
    / "kimi_k3_dspark_parity.py"
)
SPEC = importlib.util.spec_from_file_location("kimi_k3_dspark_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


def test_recorder_writes_manifest_and_honors_stage_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    recorder = DSparkParityRecorder(tmp_path, max_records_per_stage=2, ranks={0})
    assert recorder.record("target/tap:2", torch.ones(2, 3)) is not None
    assert recorder.record("target/tap:2", torch.zeros(2, 3)) is not None
    assert recorder.record("target/tap:2", torch.full((2, 3), 2.0)) is None

    manifest = parity.load_manifest(tmp_path)
    assert set(manifest) == {("target_tap_2", 0, 0), ("target_tap_2", 1, 0)}


def test_compare_dumps_checks_float_and_token_stages(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    actual = tmp_path / "actual"
    reference = tmp_path / "reference"
    actual_recorder = DSparkParityRecorder(actual, ranks={0})
    reference_recorder = DSparkParityRecorder(reference, ranks={0})

    actual_recorder.record("target_tap_2", torch.tensor([[1.0, 2.0]]))
    reference_recorder.record("target_tap_2", torch.tensor([[1.001, 1.999]]))
    tokens = torch.tensor([[11, 12, 13]], dtype=torch.int32)
    actual_recorder.record("proposed_tokens", tokens)
    reference_recorder.record("proposed_tokens", tokens.clone())

    report = parity.compare_dumps(actual, reference)
    assert report["passed"] is True
    assert {row["verdict"] for row in report["records"]} == {"pass"}


def test_compare_dumps_reports_discrete_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    actual = tmp_path / "actual"
    reference = tmp_path / "reference"
    DSparkParityRecorder(actual, ranks={0}).record(
        "accept_length", torch.tensor([2], dtype=torch.int32)
    )
    DSparkParityRecorder(reference, ranks={0}).record(
        "accept_length", torch.tensor([4], dtype=torch.int32)
    )

    report = parity.compare_dumps(actual, reference)
    assert report["passed"] is False
    assert report["records"][0]["verdict"] == "token_mismatch"
