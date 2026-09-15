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

"""Device-independent release checks, not GPU integration or numerical tests.

Execute only the actual pure policy/context definitions selected from source;
do not import the device-detecting package or substitute a fake GPU backend.
The regular integration and distributed tests remain separate requirements.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import statistics
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMM = ROOT / "python/tokenspeed/runtime/models/kimi_k3_comm.py"


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            declared = {node.name}
        elif isinstance(node, ast.Assign):
            declared = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
        else:
            continue
        if declared & names:
            selected.append(node)
            found.update(declared & names)
    assert found == names
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _policy():
    names = {
        "K3MoETailTier",
        "MULTIMEM_AR_MIN_TOKENS",
        "MULTIMEM_AR_MAX_TOKENS",
        "MNNVL_BT_QUALIFIED_TOKENS",
        "MNNVL_HT_QUALIFIED_TOKENS",
        "MNNVL_DEFERRED_ENABLE_ENV",
        "FUSED_RS_UP_AG_ENABLE_ENV",
        "FUSED_RS_UP_AG_TOKENS",
        "_fused_rs_up_ag_enabled",
        "_mnnvl_deferred_enabled",
        "_mnnvl_tp8_layout",
        "select_k3_moe_tail_tier",
    }
    return _definitions(COMM, names, {"IntEnum": IntEnum, "os": os})


def test_opt_in_is_disabled_unless_explicitly_one(monkeypatch):
    policy = _policy()
    name = policy["MNNVL_DEFERRED_ENABLE_ENV"]
    monkeypatch.delenv(policy["FUSED_RS_UP_AG_ENABLE_ENV"], raising=False)
    monkeypatch.delenv(name, raising=False)
    assert not policy["_mnnvl_deferred_enabled"]()
    for value in ("0", "true", "", "yes"):
        monkeypatch.setenv(name, value)
        assert not policy["_mnnvl_deferred_enabled"]()
    monkeypatch.setenv(name, "1")
    assert policy["_mnnvl_deferred_enabled"]()


@pytest.mark.parametrize(
    "tokens", [256, 384, 512, 768, 1024, 1280, 2048, 4096, 6144, 8192]
)
@pytest.mark.parametrize(
    "prefill,decode", [(False, False), (True, False), (True, True)]
)
def test_pure_selector_keeps_deferred_protocols_prefill_only(tokens, prefill, decode):
    policy = _policy()
    bt = tokens in policy["MNNVL_BT_QUALIFIED_TOKENS"]
    ht = tokens in policy["MNNVL_HT_QUALIFIED_TOKENS"]
    selected = policy["select_k3_moe_tail_tier"](
        num_tokens=tokens,
        graph_phase=False,
        tail_fusion_max_tokens=32,
        fused_moe_ar=True,
        multimem_ok=True,
        is_decode=decode,
        join_moe_reduce=False,
        mnnvl_bt_deferred_ok=bt,
        mnnvl_ht_deferred_ok=ht,
        fused_rs_up_ag_ok=False,
        prefill_graph_phase=prefill,
    )
    tiers = policy["K3MoETailTier"]
    if prefill and not decode:
        assert selected is (tiers.MNNVL_BT_DEFERRED if bt else tiers.MNNVL_HT_DEFERRED)
    else:
        assert selected is (tiers.FUSED_LANE_AR if decode else tiers.MULTIMEM_AR)


@pytest.mark.parametrize(
    "tp,ep,expected",
    [(8, 1, True), (1, 8, False), (2, 4, False), (4, 2, False), (16, 1, False)],
)
def test_tp8_is_not_interchangeable_with_eight_expert_parallel_ranks(tp, ep, expected):
    mapping = SimpleNamespace(moe=SimpleNamespace(tp_size=tp, ep_size=ep))
    assert _policy()["_mnnvl_tp8_layout"](mapping) is expected


def test_complete_fusion_opt_in_also_arms_required_routed_finalize(monkeypatch):
    policy = _policy()
    monkeypatch.delenv(policy["MNNVL_DEFERRED_ENABLE_ENV"], raising=False)
    name = policy["FUSED_RS_UP_AG_ENABLE_ENV"]
    for value in ("0", "true", "", "yes"):
        monkeypatch.setenv(name, value)
        assert not policy["_fused_rs_up_ag_enabled"]()
        assert not policy["_mnnvl_deferred_enabled"]()
    monkeypatch.setenv(name, "1")
    assert policy["_fused_rs_up_ag_enabled"]()
    assert policy["_mnnvl_deferred_enabled"]()


@pytest.mark.parametrize(
    "tokens", [0, 32, 256, 1024, 2048, 4095, 4096, 4097, 6144, 8192]
)
@pytest.mark.parametrize(
    "prefill,decode", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("armed", [False, True])
def test_complete_fusion_only_preempts_qualified_prefill(
    tokens, prefill, decode, armed
):
    policy = _policy()
    selected = policy["select_k3_moe_tail_tier"](
        num_tokens=tokens,
        graph_phase=False,
        tail_fusion_max_tokens=32,
        fused_moe_ar=True,
        multimem_ok=True,
        is_decode=decode,
        join_moe_reduce=False,
        mnnvl_bt_deferred_ok=False,
        mnnvl_ht_deferred_ok=True,
        fused_rs_up_ag_ok=armed,
        prefill_graph_phase=prefill,
    )
    eligible = armed and prefill and not decode and tokens in (4096, 8192)
    assert (selected is policy["K3MoETailTier"].FUSED_RS_UP_AG) is eligible


def test_real_prefill_context_restores_state_without_enabling_decode_capture():
    names = {
        "_is_capture_mode",
        "_is_cuda_graph_phase",
        "_is_prefill_graph_phase",
        "get_is_capture_mode",
        "get_is_cuda_graph_phase",
        "get_is_prefill_graph_phase",
        "prefill_graph_phase",
    }
    state = _definitions(
        ROOT / "python/tokenspeed/runtime/execution/forward_step.py",
        names,
        {"contextmanager": contextmanager},
    )
    with pytest.raises(RuntimeError, match="sentinel"):
        with state["prefill_graph_phase"]():
            with state["prefill_graph_phase"]():
                assert state["get_is_prefill_graph_phase"]()
                assert not state["get_is_capture_mode"]()
                assert not state["get_is_cuda_graph_phase"]()
            assert state["get_is_prefill_graph_phase"]()
            raise RuntimeError("sentinel")
    assert not state["get_is_prefill_graph_phase"]()


@pytest.mark.parametrize("filename", ["device_kernel.py", "primitives.py"])
def test_native_kernel_provenance_matches_documented_bytes(filename):
    directory = (
        ROOT
        / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/mnnvl_k3_ht"
    )
    digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    assert digest in (directory / "README.md").read_text(encoding="utf-8")
    assert "Apache License, Version 2.0" in (directory / filename).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "name,suffix,panel",
    [
        ("ht-complete-tail-samples.json", "tail", 0),
        ("ht-situ-tail-samples.json", "layer", 1),
    ],
)
def test_public_performance_summary_is_backed_by_rank_max_samples(name, suffix, panel):
    directory = ROOT / "docs/public/data"
    raw = json.loads((directory / name).read_text(encoding="utf-8"))
    summary = json.loads(
        (directory / "k3-mnnvl-prefill-performance.json").read_text(encoding="utf-8")
    )
    assert raw["metadata"]["latency_rank_reduction"] == "max"
    by_m = {row["tokens"]: row for row in raw["rows"]}
    for summary_row in summary["boundaries"][panel]["rows"]:
        row = by_m[summary_row["m"]]
        for prefix, label in (("old", "baseline"), ("new", "candidate")):
            samples = row[f"{prefix}_samples_us_per_{suffix}"]
            assert len(samples) == raw["metadata"]["rounds"] == 15
            p50 = statistics.median(samples)
            assert p50 == pytest.approx(row[f"{prefix}_p50_us_per_{suffix}"])
            p90 = sorted(samples)[math.ceil(0.9 * len(samples)) - 1]
            assert p90 == pytest.approx(row[f"{prefix}_p90_us_per_{suffix}"])
            assert round(p50, 2) == summary_row[f"{label}_p50"]
            assert (
                round(row[f"{prefix}_p90_us_per_{suffix}"], 2)
                == summary_row[f"{label}_p90"]
            )
        assert round(row["p50_speedup"], 4) == summary_row["p50_speedup"]
        assert round(row["p90_speedup"], 4) == summary_row["p90_speedup"]


def test_optional_backend_imports_stay_below_the_kernel_boundary():
    files = [
        COMM,
        ROOT
        / "tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/mnnvl_cutedsl_finalize.py",
        ROOT
        / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mnnvl_cutedsl_finalize.py",
    ]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert not any(
            name == "flashinfer" or name.startswith("flashinfer.") for name in modules
        )
        assert not any(name.startswith("cutlass") for name in modules)
