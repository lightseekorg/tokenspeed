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

"""Dependency-boundary tests for K3 MNNVL deferred finalize."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tokenspeed_kernel.ops.communication import mnnvl_cutedsl_finalize
from tokenspeed_kernel.platform import ArchVersion
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.thirdparty.flashinfer import mnnvl_cutedsl_finalize as adapter

_KERNEL_ROOT = Path(__file__).resolve().parents[3]
_OPS_SOURCE = (
    _KERNEL_ROOT
    / "python/tokenspeed_kernel/ops/communication/mnnvl_cutedsl_finalize.py"
)
_ADAPTER_SOURCE = (
    _KERNEL_ROOT
    / "python/tokenspeed_kernel/thirdparty/flashinfer/mnnvl_cutedsl_finalize.py"
)


def _top_level_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return tuple(names)


def test_ops_wrapper_does_not_import_third_party_protocols_directly():
    imports = _top_level_imports(_OPS_SOURCE)

    assert not any(
        name == "flashinfer" or name.startswith("flashinfer.") for name in imports
    )
    assert not any("thirdparty.cute_dsl" in name for name in imports)

    source = _OPS_SOURCE.read_text(encoding="utf-8")
    assert "BTProtocol(" not in source
    assert "K3H3584HTProtocol(" not in source
    assert "BTFinalizeTuning(" not in source
    assert "HTFinalizeTuning(" not in source


def test_flashinfer_adapter_keeps_optional_imports_lazy():
    imports = _top_level_imports(_ADAPTER_SOURCE)

    assert not any(
        name == "flashinfer" or name.startswith("flashinfer.") for name in imports
    )
    assert not any("thirdparty.cute_dsl" in name for name in imports)


def test_deferred_finalize_execution_is_registered():
    name = "flashinfer_cutedsl_mnnvl_deferred_finalize_allreduce_rmsnorm"
    registry = KernelRegistry.get()
    spec = registry.get_by_name(name)

    assert spec is not None
    assert spec.family == "communication"
    assert spec.mode == "deferred_finalize_allreduce_rmsnorm"
    assert spec.solution == "flashinfer_cutedsl"
    assert spec.capability.min_arch_version == ArchVersion(10, 0)
    assert spec.capability.max_arch_version == ArchVersion(10, 3)
    assert registry.get_impl(name) is (
        mnnvl_cutedsl_finalize.mnnvl_cutedsl_deferred_finalize_allreduce_rmsnorm
    )


@pytest.mark.parametrize("capability", [(10, 0), (10, 1), (10, 2), (10, 3)])
def test_adapter_accepts_only_the_validated_blackwell_family(capability):
    assert adapter._is_supported_blackwell_capability(capability)


@pytest.mark.parametrize("capability", [(9, 0), (9, 9), (11, 0), (12, 0)])
def test_adapter_rejects_unqualified_architectures(capability):
    assert not adapter._is_supported_blackwell_capability(capability)


def test_public_workspace_selects_registered_execution(monkeypatch):
    selected = object()
    call = {}

    def fake_select_kernel(family, mode, signature, **kwargs):
        call.update(
            family=family,
            mode=mode,
            signature=signature,
            kwargs=kwargs,
        )
        return selected

    monkeypatch.setattr(mnnvl_cutedsl_finalize, "select_kernel", fake_select_kernel)

    assert mnnvl_cutedsl_finalize._select_deferred_finalize_kernel("ht") is selected
    assert call["family"] == "communication"
    assert call["mode"] == "deferred_finalize_allreduce_rmsnorm"
    assert call["kwargs"]["traits"]["protocol"] == "ht"
    assert call["kwargs"]["solution"] == "flashinfer_cutedsl"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stages": 1}, "stages must be at least 2"),
        ({"reduction_warps": 3}, "reduction_warps must be 1, 2, 4, or 8"),
        ({"rms_token_groups": 3}, "rms_token_groups must be 1, 2, or 4"),
        ({"rms_pipeline_stages": 4}, "rms_pipeline_stages must be 1, 2, or 3"),
        (
            {"consumer_threads": 960, "reduction_warps": 8},
            "warp roles exceed the CUDA block limit",
        ),
        (
            {"consumer_threads": 64, "rms_pipeline_stages": 3},
            "finalize storage cannot hold the RMSNorm pipeline",
        ),
        ({"rms_shard_major": True}, "integer number of reduction shards"),
    ],
)
def test_invalid_ht_tuning_is_rejected_before_backend_build(overrides, message):
    values = {
        "max_tokens": 1,
        "persistent_ctas": None,
        "consumer_threads": 448,
        "vectors_per_thread": 1,
        "stages": 7,
        "reduction_warps": 2,
        "reduction_cta_groups": None,
        "rms_token_groups": 2,
        "rms_pipeline_stages": 3,
        "rms_shard_major": False,
        "enable_pdl": True,
    }
    values.update(overrides)
    route = mnnvl_cutedsl_finalize.MNNVLCuteDSLHTFinalizeTuning(**values)

    with pytest.raises(ValueError, match=message):
        mnnvl_cutedsl_finalize._validate_ht_tuning_routes(
            routes=(route,),
            candidate_min_tokens=1,
            candidate_max_tokens=1,
        )
