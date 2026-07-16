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

"""Static-schema and execution-evidence tests for flat KV completion."""

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]

with mock.patch.dict(sys.modules):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = object
        sys.modules["torch"] = fake_torch

    types_path = _ROOT / "python/tokenspeed/runtime/execution/types.py"
    types_spec = importlib.util.spec_from_file_location(
        "flat_kv_progress_execution_types_test",
        types_path,
    )
    assert types_spec is not None and types_spec.loader is not None
    execution_types = importlib.util.module_from_spec(types_spec)
    sys.modules[types_spec.name] = execution_types
    types_spec.loader.exec_module(execution_types)

    contract_path = _ROOT / "python/tokenspeed/runtime/configs/flat_kv_contract.py"
    contract_spec = importlib.util.spec_from_file_location(
        "tokenspeed.runtime.configs.flat_kv_contract",
        contract_path,
    )
    assert contract_spec is not None and contract_spec.loader is not None
    contract = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_spec.name] = contract
    contract_spec.loader.exec_module(contract)

    progress_path = _ROOT / "python/tokenspeed/runtime/execution/flat_kv_progress.py"
    progress_spec = importlib.util.spec_from_file_location(
        "flat_kv_progress_test_module",
        progress_path,
    )
    assert progress_spec is not None and progress_spec.loader is not None
    progress = importlib.util.module_from_spec(progress_spec)
    sys.modules[progress_spec.name] = progress
    progress_spec.loader.exec_module(progress)

FlatKVCompletionInput = execution_types.FlatKVCompletionInput
FlatKVExecutionTracker = progress.FlatKVExecutionTracker
FlatKVProgressSchema = progress.FlatKVProgressSchema
V4_PRODUCER_TARGET_MAIN = progress.V4_PRODUCER_TARGET_MAIN
V4_PRODUCER_TARGET_INDEXER = progress.V4_PRODUCER_TARGET_INDEXER
V4_PRODUCER_DRAFT_MAIN = progress.V4_PRODUCER_DRAFT_MAIN
V4_PRODUCER_DRAFT_INDEXER = progress.V4_PRODUCER_DRAFT_INDEXER


@pytest.fixture(autouse=True)
def _install_execution_types_for_lazy_import(monkeypatch):
    """Satisfy the helper's lazy types import in dependency-light CI."""
    for name in ("tokenspeed", "tokenspeed.runtime", "tokenspeed.runtime.execution"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.execution.types",
        execution_types,
    )


def _spec(group_id: str, stride: int, mask: int):
    return SimpleNamespace(
        group_id=group_id,
        entry_stride_tokens=stride,
        required_producer_domain_mask=mask,
    )


def _component(owner: str, group_id: str, layer: int, component: str):
    return SimpleNamespace(
        owner=owner,
        group_id=group_id,
        layer=layer,
        component=component,
    )


def _v4_plan(*, draft_layers: int = 1):
    all_domains = (
        V4_PRODUCER_TARGET_MAIN
        | V4_PRODUCER_TARGET_INDEXER
        | V4_PRODUCER_DRAFT_MAIN
        | V4_PRODUCER_DRAFT_INDEXER
    )
    specs = (
        _spec("history", 4, all_domains),
        _spec(
            "state",
            1,
            V4_PRODUCER_TARGET_MAIN | V4_PRODUCER_DRAFT_MAIN,
        ),
        _spec(
            "index_state",
            1,
            V4_PRODUCER_TARGET_INDEXER | V4_PRODUCER_DRAFT_INDEXER,
        ),
    )
    components = [
        _component("target", "history", 0, "compressed_kv"),
        _component("target", "history", 0, "indexer_kv"),
        _component("target", "state", 0, "compressor_state"),
        _component("target", "index_state", 0, "indexer_state"),
    ]
    for layer in range(draft_layers):
        components.extend(
            (
                _component("draft", "history", layer, "compressed_kv"),
                _component("draft", "history", layer, "indexer_kv"),
                _component("draft", "state", layer, "compressor_state"),
                _component("draft", "index_state", layer, "indexer_state"),
            )
        )
    return SimpleNamespace(
        scheduler_group_specs=specs,
        pools=(SimpleNamespace(tensors=tuple(components)),),
    )


def _input(*, start: int = 5, end: int = 7, protected: int = 11):
    return FlatKVCompletionInput(
        request_id="r",
        table_generation=3,
        dispatch_seq=9,
        dispatch_raw_start=start,
        dispatch_raw_end=end,
        protected_raw_end=protected,
    )


def test_v4_target_draft_and_indexer_domains_pack_in_bit_order():
    schema = FlatKVProgressSchema.from_v4_plan(_v4_plan())
    tracker = FlatKVExecutionTracker(schema)
    completion_input = _input()
    tracker.begin_dispatch([completion_input], request_ids=["r"])

    evidence = tracker.finish_dispatch(
        target_forward_enqueued=True,
        draft_continuous_prefix_enqueued=True,
    )
    assert evidence is not None
    assert evidence.completed_domain_mask == (
        V4_PRODUCER_TARGET_MAIN
        | V4_PRODUCER_TARGET_INDEXER
        | V4_PRODUCER_DRAFT_MAIN
        | V4_PRODUCER_DRAFT_INDEXER
    )
    assert type(evidence.completed_domain_mask) is int
    groups = evidence.materialize_group_completions(
        accepted_raw_ends=[6],
    )[0]

    by_id = {group.group_id: group for group in groups}
    # Bits are packed in ascending order: target main, target indexer, draft
    # main, draft indexer.  The compressed boundary can precede dispatch start.
    assert by_id["history"].domain_valid_ends == (4, 4, 4, 4)
    assert by_id["state"].domain_valid_ends == (7, 6)
    assert by_id["index_state"].domain_valid_ends == (7, 6)


def test_multi_layer_draft_reports_accepted_end_after_dense_catchup():
    schema = FlatKVProgressSchema.from_v4_plan(_v4_plan(draft_layers=2))
    tracker = FlatKVExecutionTracker(schema)
    completion_input = _input(start=8, end=12, protected=16)
    tracker.begin_dispatch([completion_input], request_ids=["r"])
    evidence = tracker.finish_dispatch(
        target_forward_enqueued=True,
        draft_continuous_prefix_enqueued=True,
    )
    assert evidence is not None

    groups = evidence.materialize_group_completions(
        accepted_raw_ends=[10],
    )[0]
    by_id = {group.group_id: group for group in groups}
    # Dense catch-up covers every planned draft layer/domain, but only through
    # the accepted prefix. History ends are rounded down to stride four.
    assert by_id["history"].domain_valid_ends == (12, 12, 8, 8)
    assert by_id["state"].domain_valid_ends == (12, 10)
    assert by_id["index_state"].domain_valid_ends == (12, 10)


@pytest.mark.parametrize(
    ("target_enqueued", "draft_enqueued"),
    ((True, False), (False, True)),
)
def test_missing_required_v4_producer_domain_fails_closed(
    target_enqueued: bool, draft_enqueued: bool
):
    schema = FlatKVProgressSchema.from_v4_plan(_v4_plan(draft_layers=2))
    tracker = FlatKVExecutionTracker(schema)
    tracker.begin_dispatch(
        [_input(start=8, end=12, protected=16)],
        request_ids=["r"],
    )
    evidence = tracker.finish_dispatch(
        target_forward_enqueued=target_enqueued,
        draft_continuous_prefix_enqueued=draft_enqueued,
    )
    assert evidence is not None

    with pytest.raises(RuntimeError, match="missing continuous producer domains"):
        evidence.materialize_group_completions(accepted_raw_ends=[10])


def test_tracker_rejects_non_host_enqueue_evidence():
    schema = FlatKVProgressSchema.from_v4_plan(_v4_plan(draft_layers=2))
    tracker = FlatKVExecutionTracker(schema)
    tracker.begin_dispatch([_input()], request_ids=["r"])

    with pytest.raises(TypeError, match="must be a host Python bool"):
        tracker.finish_dispatch(
            target_forward_enqueued=True,
            draft_continuous_prefix_enqueued=1,
        )


def test_zero_width_dispatch_returns_seed_safe_completion_without_forward():
    schema = FlatKVProgressSchema.from_v4_plan(_v4_plan(draft_layers=2))
    tracker = FlatKVExecutionTracker(schema)
    completion_input = _input(start=10, end=10, protected=16)
    tracker.begin_dispatch([completion_input], request_ids=["r"])
    evidence = tracker.finish_dispatch(
        target_forward_enqueued=False,
        draft_continuous_prefix_enqueued=False,
    )
    assert evidence is not None

    groups = evidence.materialize_group_completions(
        accepted_raw_ends=[10],
    )[0]
    by_id = {group.group_id: group for group in groups}
    assert by_id["history"].domain_valid_ends == (8, 8, 8, 8)
    assert by_id["state"].domain_valid_ends == (10, 10)
    assert by_id["index_state"].domain_valid_ends == (10, 10)


@pytest.mark.parametrize(
    ("owner", "group_id", "component_name"),
    (
        ("target", "history", "indexer_kv"),
        ("draft", "state", "compressor_state"),
    ),
)
def test_v4_schema_rejects_missing_required_owner_domain(
    owner: str,
    group_id: str,
    component_name: str,
):
    plan = _v4_plan()
    plan.pools[0].tensors = tuple(
        component
        for component in plan.pools[0].tensors
        if not (
            component.owner == owner
            and component.group_id == group_id
            and component.component == component_name
        )
    )

    with pytest.raises(ValueError, match="missing required producer planes"):
        FlatKVProgressSchema.from_v4_plan(plan)
