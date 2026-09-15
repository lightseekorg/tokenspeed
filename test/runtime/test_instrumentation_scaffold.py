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

import argparse
import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.metrics.dsv4_vision_instrumentation import (
    DSV4_VISION_DISPATCH_LOG_ENV,
    DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION,
    DispatchRecord,
    DSV4VisionInstrumentationRecorder,
    begin_current_stream_timing,
    build_request_media_bindings,
    configure_dsv4_vision_instrumentation,
    finish_current_stream_timing,
    read_dispatch_log,
    read_internal_states,
    record_encoder_call,
)


def _record_dispatch(
    recorder: DSV4VisionInstrumentationRecorder,
    request_ids=("request-1",),
    extend_prefix_lens=(8,),
):
    return recorder.record_dispatch(
        path="eager",
        num_tokens=12,
        batch_size=len(request_ids),
        request_ids=request_ids,
        num_extends=len(extend_prefix_lens),
        extend_prefix_lens=extend_prefix_lens,
        intersects_span=True,
        wall_ns=101,
    )


def _load_class_methods(source, class_name, method_names, globals_dict):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"), filename=str(source))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = []
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name in method_names
        ):
            node.decorator_list = []
            methods.append(node)
    assert {node.name for node in methods} == set(method_names)
    namespace = dict(globals_dict)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    result = type(class_name, (), {})
    for method_name in method_names:
        setattr(result, method_name, namespace[method_name])
    return result


def test_recorder_round_trip_watermark_request_key_item_key(tmp_path):
    dispatch_path = tmp_path / "dispatch.jsonl"
    recorder = DSV4VisionInstrumentationRecorder(
        enabled=True,
        global_rank=0,
        dispatch_log_path=str(dispatch_path),
    )
    recorder.record_scheduler_config(
        disable_prefix_cache=True,
        max_scheduled_tokens=8192,
        prefix_granularity=256,
    )
    recorder.record_model_parameters(["z.weight", "a.weight"])
    recorder.record_logits_dtype("torch.float32")
    recorder.record_prefill_index_buffer((4, 512), 8192)
    recorder.record_prefill_index_buffer((8, 1024), 32768)
    recorder.begin_dispatch()
    recorder.record_encoder_call(
        "image", [("hash-a", 114), ("hash-b", 110)], wall_ns=37
    )
    first = _record_dispatch(recorder)
    second = _record_dispatch(recorder, extend_prefix_lens=(99,))

    assert first is not None and second is not None
    records = read_dispatch_log(dispatch_path)
    assert records == [first, second]
    assert records[0].to_dict() == {
        "schema_version": DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION,
        "forward_index": 1,
        "path": "eager",
        "num_tokens": 12,
        "batch_size": 1,
        "request_ids": ["request-1"],
        "num_extends": 1,
        "extend_prefix_lens": [8],
        "intersects_span": True,
        "media_bindings": [],
        "encoder_wall_ns": 37,
        "wall_ns": 101,
    }
    assert read_dispatch_log(dispatch_path, after_forward_index=1) == [second]
    assert read_dispatch_log(dispatch_path, through_forward_index=1) == [first]

    state = recorder.internal_state()
    snapshot = read_internal_states(
        [state], item_keys=["image:hash-a"], request_ids=["request-1"]
    )
    assert snapshot["totals"] == {
        "mm_encoder_calls": 1,
        "mm_encoded_items": 2,
        "mm_encoded_rows": 224,
    }
    assert snapshot["by_item"]["image:hash-a"] == {
        "encoder_calls": 1,
        "encoded_rows": 114,
    }
    # Accepted-prefix ownership belongs to the first extend only.
    assert snapshot["by_request"]["request-1"] == {
        "accepted_prefix_tokens": 8,
        "first_extend_forward_index": 1,
        "extend_forwards": 2,
    }
    assert snapshot["scheduler_config"] == {
        "disable_prefix_cache": True,
        "max_scheduled_tokens": 8192,
        "prefix_granularity": 256,
    }
    assert snapshot["model"] == {
        "param_count": 2,
        "param_names_sha256": hashlib.sha256(b"a.weight\nz.weight").hexdigest(),
        "logits_dtype": "torch.float32",
        "prefill_index_buffer_shape": [4, 512],
        "prefill_index_buffer_bytes": 8192,
    }
    recorder.close()


def test_encoder_call_adapter_normalizes_modality_and_tensor_rows():
    import torch

    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    recorder = configure_dsv4_vision_instrumentation(enabled=True, global_rank=0)
    items = [
        MultimodalDataItem(modality=Modality.IMAGE, hash=17),
        MultimodalDataItem(modality=Modality.IMAGE, hash=23),
    ]
    outputs = [torch.empty((7, 4)), torch.empty((11, 4))]

    try:
        record_encoder_call(Modality.IMAGE, items, outputs, wall_ns=19)
        snapshot = recorder.internal_state()["dsv4_instrumentation"]
        assert snapshot["totals"] == {
            "mm_encoder_calls": 1,
            "mm_encoded_items": 2,
            "mm_encoded_rows": 18,
        }
        assert snapshot["by_item"] == {
            "image:17": {"encoder_calls": 1, "encoded_rows": 7},
            "image:23": {"encoder_calls": 1, "encoded_rows": 11},
        }
    finally:
        configure_dsv4_vision_instrumentation(enabled=False, global_rank=0)


def test_request_media_binding_records_order_placeholder_pad_and_encoder_rows():
    import torch

    context = SimpleNamespace(
        request_ids=["request-e2e"],
        mm_inputs=[
            SimpleNamespace(
                im_token_id=129264,
                mm_items=[
                    SimpleNamespace(
                        modality=SimpleNamespace(name="IMAGE"),
                        hash=17,
                        pad_value=1_000_017,
                        offsets=[(6, 12)],
                        model_specific_data={
                            "dsv4_compress_pad": torch.tensor([3], dtype=torch.uint32)
                        },
                        encoded=torch.empty((7, 4)),
                    )
                ],
            )
        ],
    )

    records = build_request_media_bindings(context)

    assert [record.to_dict() for record in records] == [
        {
            "request_id": "request-e2e",
            "item_index": 0,
            "modality": "image",
            "content_hash_u64": 17,
            "placeholder_token_id": 129264,
            "pad_value": 1_000_017,
            "offsets": [[6, 12]],
            "compress_pad": 3,
            "encoded_rows": 7,
        }
    ]


def test_schema_rejection_and_absent_correlation_keys():
    value = {
        "schema_version": DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION + 1,
        "forward_index": 1,
        "path": "eager",
        "num_tokens": 1,
        "batch_size": 1,
        "request_ids": ["request-1"],
        "num_extends": 1,
        "extend_prefix_lens": [0],
        "intersects_span": False,
        "media_bindings": [],
        "encoder_wall_ns": 0,
        "wall_ns": 1,
    }
    with pytest.raises(ValueError, match="Unsupported.*schema version"):
        DispatchRecord.from_dict(value)
    value["schema_version"] = DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION
    value["unknown"] = 1
    with pytest.raises(ValueError, match="fields do not match schema"):
        DispatchRecord.from_dict(value)

    recorder = DSV4VisionInstrumentationRecorder(enabled=True)
    with pytest.raises(KeyError, match="correlation key is absent"):
        read_internal_states(
            [recorder.internal_state()],
            item_keys=["image:missing"],
            request_ids=["missing"],
        )


def test_fifo_retention_and_reset_keep_monotone_watermarks():
    recorder = DSV4VisionInstrumentationRecorder(enabled=True, max_keyed_entries=2)
    recorder.record_scheduler_config(
        disable_prefix_cache=False,
        max_scheduled_tokens=4096,
        prefix_granularity=64,
    )
    recorder.record_model_parameters(["weight"])
    for index in range(3):
        recorder.record_encoder_call("image", [(f"hash-{index}", index + 1)])
        _record_dispatch(
            recorder,
            request_ids=(f"request-{index}",),
            extend_prefix_lens=(index,),
        )

    before = recorder.internal_state()["dsv4_instrumentation"]
    assert list(before["by_item"]) == ["image:hash-1", "image:hash-2"]
    assert list(before["by_request"]) == ["request-1", "request-2"]
    assert before["by_item_evicted"] == 1
    assert before["by_request_evicted"] == 1
    snapshot_id = before["snapshot_id"]
    forward_index = before["dispatch_forward_index"]

    recorder.reset()
    reset = recorder.internal_state()["dsv4_instrumentation"]
    assert reset["snapshot_id"] > snapshot_id
    assert reset["dispatch_forward_index"] == forward_index
    assert reset["totals"]["mm_encoder_calls"] == 0
    assert reset["by_item"] == {}
    assert reset["by_request"] == {}
    assert reset["scheduler_config"] == before["scheduler_config"]
    assert reset["model"] == before["model"]

    record = _record_dispatch(recorder)
    assert record is not None
    assert record.forward_index == forward_index + 1


def test_default_retention_evicts_oldest_entry_after_4096_keys():
    recorder = DSV4VisionInstrumentationRecorder(enabled=True)
    for index in range(4097):
        recorder.record_encoder_call("image", [(f"hash-{index}", 1)])
        _record_dispatch(
            recorder,
            request_ids=(f"request-{index}",),
            extend_prefix_lens=(0,),
        )

    snapshot = recorder.internal_state()["dsv4_instrumentation"]
    assert len(snapshot["by_item"]) == 4096
    assert len(snapshot["by_request"]) == 4096
    assert "image:hash-0" not in snapshot["by_item"]
    assert "request-0" not in snapshot["by_request"]
    assert snapshot["by_item_evicted"] == 1
    assert snapshot["by_request_evicted"] == 1


def test_default_off_and_non_rank_zero_never_open_a_sink(monkeypatch, tmp_path):
    dispatch_path = tmp_path / "must-not-exist.jsonl"
    monkeypatch.setenv(DSV4_VISION_DISPATCH_LOG_ENV, str(dispatch_path))
    open_calls = []

    def fail_open(*args, **kwargs):
        open_calls.append((args, kwargs))
        raise AssertionError("default-off recorder opened a sink")

    disabled = DSV4VisionInstrumentationRecorder(enabled=False, opener=fail_open)
    disabled.record_encoder_call("image", [("hash", 1)])
    assert _record_dispatch(disabled) is None
    assert disabled.internal_state() == {}

    non_rank_zero = DSV4VisionInstrumentationRecorder(
        enabled=True,
        global_rank=1,
        opener=fail_open,
    )
    assert _record_dispatch(non_rank_zero) is not None
    assert non_rank_zero.internal_state() == {}
    assert open_calls == []
    assert not dispatch_path.exists()


def test_rank_zero_is_the_only_single_stream_writer(tmp_path):
    dispatch_path = tmp_path / "dispatch.jsonl"
    rank_zero = DSV4VisionInstrumentationRecorder(
        enabled=True,
        global_rank=0,
        dispatch_log_path=str(dispatch_path),
    )
    rank_one = DSV4VisionInstrumentationRecorder(
        enabled=True,
        global_rank=1,
        dispatch_log_path=str(dispatch_path),
    )
    _record_dispatch(rank_zero, request_ids=("rank-zero",))
    _record_dispatch(rank_one, request_ids=("rank-one",))

    assert rank_zero.has_dispatch_stream is True
    assert rank_one.has_dispatch_stream is False
    assert [record.request_ids for record in read_dispatch_log(dispatch_path)] == [
        ("rank-zero",)
    ]
    rank_zero.close()


def test_server_flag_defaults_off_and_parses_explicit_enablement():
    from tokenspeed.runtime.utils.server_args import ServerArgs

    assert ServerArgs(model="model").enable_dsv4_vision_instrumentation is False
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model", "model", "--enable-dsv4-vision-instrumentation"]
    )
    assert parsed.enable_dsv4_vision_instrumentation is True


def test_model_executor_flag_off_does_not_synchronize():
    class Sliceable:
        def __getitem__(self, _key):
            return self

    class DeviceModule:
        def synchronize(self):
            raise AssertionError("flag-off target forward synchronized the device")

    expected = object()
    model_executor_path = (
        Path(__file__).parents[2]
        / "python/tokenspeed/runtime/execution/model_executor.py"
    )
    model_executor_class = _load_class_methods(
        model_executor_path,
        "ModelExecutor",
        {"_run_target_forward"},
        {
            "begin_current_stream_timing": begin_current_stream_timing,
            "finish_current_stream_timing": finish_current_stream_timing,
            "get_is_cuda_graph_phase": lambda: False,
        },
    )
    executor = object.__new__(model_executor_class)
    executor.config = SimpleNamespace(
        enable_dsv4_vision_instrumentation=False,
        model_is_mrope=False,
        pp_size=1,
    )
    executor._active_positions_override = None
    executor._active_multimodal_context = None
    executor.device_module = DeviceModule()
    executor.input_buffers = SimpleNamespace(
        positions_buf=Sliceable(),
        input_ids_buf=Sliceable(),
        out_cache_loc_buf=Sliceable(),
        seq_lens_buf=Sliceable(),
        extend_prefix_lens_buf=Sliceable(),
        ngram_model_kwargs=lambda _count: {},
    )
    executor.prefill_graph = SimpleNamespace(
        can_run=lambda *_args: (_ for _ in ()).throw(
            AssertionError("prefill graph should not be consulted for mode=None")
        )
    )
    executor.model_runner = SimpleNamespace(forward=lambda *_args, **_kwargs: expected)
    ctx = SimpleNamespace(
        bs=1,
        input_num_tokens=2,
        num_extends=1,
        forward_mode=None,
    )

    assert executor._run_target_forward(ctx) is expected


def test_instrumented_target_forward_model_invocations_match_default_path():
    model_executor_path = (
        Path(__file__).parents[2]
        / "python/tokenspeed/runtime/execution/model_executor.py"
    )
    tree = ast.parse(model_executor_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelExecutor"
    )
    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_run_target_forward", "_run_target_forward_instrumented"}
    }

    model_calls = {
        "self.model_runner.forward",
        "self.prefill_graph.can_run",
        "self.prefill_graph.replay",
    }

    def call_name(call):
        return ast.unparse(call.func)

    def contains_model_call(node):
        return any(
            isinstance(candidate, ast.Call) and call_name(candidate) in model_calls
            for candidate in ast.walk(node)
        )

    class StripInstrumentation(ast.NodeTransformer):
        """Remove observation-only statements before comparing control flow."""

        instrumentation_names = {
            "record_instrumentation",
            "instrumentation_timing",
            "path",
            "wall_ns",
            "request_ids",
            "num_extends",
            "extend_prefix_lens",
        }

        def visit_Assign(self, node):
            assigned_names = {
                name.id
                for target in node.targets
                for name in ast.walk(target)
                if isinstance(name, ast.Name)
            }
            if assigned_names & self.instrumentation_names:
                assert not contains_model_call(node)
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            test_names = {
                name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)
            }
            if "record_instrumentation" in test_names:
                assert not contains_model_call(node)
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func)
                == "get_dsv4_vision_instrumentation().record_dispatch"
            ):
                return None
            return self.generic_visit(node)

    def invocation_structure(method):
        stripped = StripInstrumentation().visit(method)
        ast.fix_missing_locations(stripped)
        invocations = []

        def walk_block(statements, incoming_paths):
            paths = list(incoming_paths)
            for statement in statements:
                if not paths:
                    break
                if isinstance(statement, ast.If):
                    if not contains_model_call(statement):
                        continue
                    test = ast.unparse(statement.test)
                    for call in ast.walk(statement.test):
                        if (
                            isinstance(call, ast.Call)
                            and call_name(call) in model_calls
                        ):
                            for path in paths:
                                invocations.append(
                                    (path, call_name(call), ast.unparse(call))
                                )
                    true_paths = [path + ((test, True),) for path in paths]
                    false_paths = [path + ((test, False),) for path in paths]
                    body_fallthrough = walk_block(statement.body, true_paths)
                    if statement.orelse:
                        else_fallthrough = walk_block(statement.orelse, false_paths)
                    else:
                        else_fallthrough = false_paths
                    paths = body_fallthrough + else_fallthrough
                    continue

                for call in ast.walk(statement):
                    if isinstance(call, ast.Call) and call_name(call) in model_calls:
                        for path in paths:
                            invocations.append(
                                (path, call_name(call), ast.unparse(call))
                            )
                if isinstance(statement, ast.Return):
                    paths = []
            return paths

        walk_block(stripped.body, [()])
        return invocations

    default_structure = invocation_structure(methods["_run_target_forward"])
    instrumented_structure = invocation_structure(
        methods["_run_target_forward_instrumented"]
    )

    assert instrumented_structure == default_structure
    assert [entry[1] for entry in default_structure] == [
        "self.model_runner.forward",
        "self.prefill_graph.can_run",
        "self.prefill_graph.replay",
        "self.model_runner.forward",
    ]


def test_model_executor_source_initializes_device_module_after_startup_forwards():
    model_executor_path = (
        Path(__file__).parents[2]
        / "python/tokenspeed/runtime/execution/model_executor.py"
    )
    tree = ast.parse(model_executor_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelExecutor"
    )
    init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    device_module_lines = [
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "device_module"
            for target in node.targets
        )
    ]
    startup_forward_lines = [
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"prewarm_comm_states", "capture"}
    ]

    assert len(device_module_lines) == 1
    assert startup_forward_lines
    assert device_module_lines[0] > max(startup_forward_lines)


@pytest.mark.parametrize(
    ("enforce_eager", "expected_startup_hook"),
    [(True, "prewarm"), (False, "capture")],
)
def test_model_executor_init_and_deferred_capture_keep_instrumentation_safe(
    monkeypatch, enforce_eager, expected_startup_hook
):
    from tokenspeed.runtime.execution import model_executor as model_executor_module

    startup_hooks = []

    class DeviceModule:
        def Stream(self):
            return object()

        def default_stream(self, device):
            return object()

    device_module = DeviceModule()

    def assert_deferred(forward_func, hook):
        owner = forward_func.__self__
        if hook == "prewarm":
            assert not hasattr(owner, "device_module")
        else:
            assert owner.device_module is device_module
        startup_hooks.append(hook)

    class FakeForwardStepRunner:
        def __init__(self, *, forward_func, config, **_kwargs):
            self.forward_func = forward_func
            self.disable = config.enforce_eager

        def prewarm_comm_states(self, *, batch_sizes):
            assert batch_sizes == (1,)
            assert_deferred(self.forward_func, "prewarm")

        def capture(self):
            assert_deferred(self.forward_func, "capture")

    class FakePrefillGraph:
        disable = True

        def __init__(self, **_kwargs):
            pass

    class FakeWorkspacePool:
        def freeze(self):
            pass

    class FakeMultimodalRuntime:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(
        model_executor_module.torch,
        "get_device_module",
        lambda device: device_module,
    )
    monkeypatch.setattr(
        model_executor_module.torch, "tensor", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        model_executor_module, "validate_scheduler_config", lambda **_k: None
    )
    monkeypatch.setattr(
        model_executor_module,
        "InputBuffers",
        lambda **_k: SimpleNamespace(init_ngram_buffers=lambda _n: None),
    )
    monkeypatch.setattr(
        model_executor_module,
        "RuntimeStates",
        lambda **_k: SimpleNamespace(init_ngram_state=lambda _n: None),
    )
    monkeypatch.setattr(
        model_executor_module,
        "NanGuard",
        SimpleNamespace(create=lambda *_a, **_k: object()),
    )
    monkeypatch.setattr(
        model_executor_module, "create_grammar_runtime", lambda **_k: object()
    )
    monkeypatch.setattr(
        model_executor_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    monkeypatch.setattr(model_executor_module, "bind_cache_groups", lambda *_a: None)
    monkeypatch.setattr(
        model_executor_module, "setup_dp_sampling", lambda **_k: object()
    )
    monkeypatch.setattr(
        model_executor_module, "ForwardStepRunner", FakeForwardStepRunner
    )
    monkeypatch.setattr(model_executor_module, "PrefillGraph", FakePrefillGraph)
    monkeypatch.setattr(
        model_executor_module.ModelExecutor, "_autotune", lambda _self: None
    )
    monkeypatch.setattr(
        model_executor_module, "workspace_pool", lambda _device: FakeWorkspacePool()
    )
    monkeypatch.setattr(
        model_executor_module, "ForwardThread", lambda _device: object()
    )
    monkeypatch.setattr(
        model_executor_module, "MultimodalRuntime", FakeMultimodalRuntime
    )
    monkeypatch.setattr(
        model_executor_module,
        "resolve_cuda_graph_support",
        lambda *_a: SimpleNamespace(decode_graph=True, prefill_graph=True),
    )

    runtime_contract = SimpleNamespace(group_specs=())
    arena = SimpleNamespace(runtime_contract=runtime_contract, cache_group_specs=())
    token_pool = SimpleNamespace(arena=arena)
    attn_backend = SimpleNamespace(
        cache_group_tables_replace_draft_page_table=False,
        configure_runtime=lambda **_kwargs: None,
    )
    model_runner = SimpleNamespace(
        model=object(),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        sliding_window_size=None,
        encoder_graph_wrappers={},
    )
    config = SimpleNamespace(
        device="cuda",
        prefix_granularity=64,
        physical_context_len=128,
        max_num_seqs=1,
        data_parallel_size=1,
        spec_num_tokens=1,
        spec_algo=None,
        chunked_prefill_size=128,
        max_req_pool_size=2,
        vocab_size=8,
        output_length=1,
        enable_nan_detection=False,
        grammar_backend="none",
        disable_capturable_grammar=False,
        dp_sampling=False,
        dp_sampling_min_bs=None,
        enforce_eager=enforce_eager,
        model_is_mrope=False,
        enable_dsv4_vision_instrumentation=True,
        enable_dsv4_numerical_attribution=False,
        global_rank=0,
    )

    executor = model_executor_module.ModelExecutor(
        config=config,
        model_runner=model_runner,
        attn_backend=attn_backend,
        token_to_kv_pool=token_pool,
        sampling_backend=object(),
    )

    if not enforce_eager:
        assert startup_hooks == []
        executor.capture_graphs()
    assert startup_hooks == [expected_startup_hook]
    assert executor.device_module is device_module
    assert (
        executor._run_target_forward.__func__
        is model_executor_module.ModelExecutor._run_target_forward_instrumented
    )


def test_default_off_hot_path_helpers_do_not_create_events():
    class DeviceModule:
        def current_stream(self):
            raise AssertionError("default-off timing hook requested a stream")

        def Event(self, **_kwargs):
            raise AssertionError("default-off timing hook created an event")

        def synchronize(self):
            raise AssertionError("default-off timing hook synchronized the device")

    device_module = DeviceModule()
    timing = begin_current_stream_timing(False, device_module)
    assert timing is None
    assert finish_current_stream_timing(timing) is None


def test_current_stream_timing_skips_graph_capture_without_events():
    class DeviceModule:
        @staticmethod
        def is_current_stream_capturing():
            return True

        @staticmethod
        def current_stream():
            raise AssertionError("capture-safe timing requested a stream")

        @staticmethod
        def Event(**_kwargs):
            raise AssertionError("capture-safe timing created an event")

    assert begin_current_stream_timing(True, DeviceModule()) is None


def test_default_off_hot_paths_run_in_a_subprocess():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-q",
            __file__,
            "-k",
            "default_off_and_non_rank_zero or "
            "default_off_hot_path_helpers_do_not_create_events",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_executor_flag_on_uses_current_stream_events_and_records_dispatch(
    tmp_path,
):
    class Sliceable:
        def __getitem__(self, _key):
            return self

    class Event:
        def __init__(self, elapsed_ms):
            self.elapsed_ms = elapsed_ms
            self.recorded_streams = []
            self.synchronize_calls = 0

        def record(self, stream):
            self.recorded_streams.append(stream)

        def synchronize(self):
            self.synchronize_calls += 1

        def elapsed_time(self, end_event):
            assert end_event is not self
            return self.elapsed_ms

    class DeviceModule:
        def __init__(self):
            self.stream = object()
            self.events = []

        def current_stream(self):
            return self.stream

        def Event(self, *, enable_timing):
            assert enable_timing is True
            event = Event(1.25)
            self.events.append(event)
            return event

        def synchronize(self):
            raise AssertionError("instrumentation synchronized the whole device")

    dispatch_path = tmp_path / "current-stream.jsonl"
    recorder = configure_dsv4_vision_instrumentation(
        enabled=True,
        global_rank=0,
        dispatch_log_path=str(dispatch_path),
    )
    expected_forward_output = object()
    expected_execution_result = object()

    def execute_forward_op(executor, *_args, **_kwargs):
        output = executor._run_target_forward_instrumented(ctx)
        assert output is expected_forward_output
        return expected_execution_result

    model_executor_path = (
        Path(__file__).parents[2]
        / "python/tokenspeed/runtime/execution/model_executor.py"
    )
    model_executor_class = _load_class_methods(
        model_executor_path,
        "ModelExecutor",
        {
            "_execute_forward_op_instrumented",
            "_run_target_forward_instrumented",
        },
        {
            "begin_current_stream_timing": begin_current_stream_timing,
            "finish_current_stream_timing": finish_current_stream_timing,
            "get_is_cuda_graph_phase": lambda: False,
            "get_dsv4_vision_instrumentation": lambda: recorder,
            "build_request_media_bindings": build_request_media_bindings,
            "ModelExecutor": SimpleNamespace(execute_forward_op=execute_forward_op),
        },
    )
    executor = object.__new__(model_executor_class)
    executor.config = SimpleNamespace(
        enable_dsv4_vision_instrumentation=True,
        model_is_mrope=False,
        pp_size=1,
    )
    executor._dsv4_instrumentation_forward = (("request-1",), 1, (7,))
    executor._active_positions_override = None
    executor._active_multimodal_context = object()
    executor.device_module = DeviceModule()
    executor.input_buffers = SimpleNamespace(
        positions_buf=Sliceable(),
        input_ids_buf=Sliceable(),
        out_cache_loc_buf=Sliceable(),
        seq_lens_buf=Sliceable(),
        extend_prefix_lens_buf=Sliceable(),
        ngram_model_kwargs=lambda _count: {},
    )
    executor.prefill_graph = SimpleNamespace(can_run=lambda *_args: False)
    executor.model_runner = SimpleNamespace(
        forward=lambda *_args, **_kwargs: expected_forward_output
    )
    ctx = SimpleNamespace(
        bs=1,
        input_num_tokens=2,
        num_extends=1,
        forward_mode=None,
        dsv4_vision=SimpleNamespace(intersects_span=True),
    )
    forward_op = SimpleNamespace(
        num_extends=lambda: 1,
        request_ids=("request-1",),
        extend_prefix_lens=(7,),
    )

    result = executor._execute_forward_op_instrumented(
        forward_op, [], ngram_inputs=None
    )
    assert result is expected_execution_result
    assert len(executor.device_module.events) == 2
    start_event, end_event = executor.device_module.events
    assert start_event.recorded_streams == [executor.device_module.stream]
    assert end_event.recorded_streams == [executor.device_module.stream]
    assert start_event.synchronize_calls == 0
    assert end_event.synchronize_calls == 1
    snapshot = recorder.internal_state()["dsv4_instrumentation"]
    assert snapshot["dispatch_forward_index"] == 1
    assert snapshot["by_request"]["request-1"]["accepted_prefix_tokens"] == 7
    record = read_dispatch_log(dispatch_path)[0]
    assert record.path == "eager"
    assert record.intersects_span is True
    assert record.encoder_wall_ns == 0
    assert record.wall_ns == 1_250_000
    configure_dsv4_vision_instrumentation(enabled=False, global_rank=0)


def test_model_executor_capture_phase_never_synchronizes_or_records():
    class Sliceable:
        def __getitem__(self, _key):
            return self

    class DeviceModule:
        def current_stream(self):
            raise AssertionError("CUDA graph phase requested an event stream")

        def Event(self, **_kwargs):
            raise AssertionError("CUDA graph phase created a timing event")

        def synchronize(self):
            raise AssertionError("CUDA graph phase synchronized the device")

    recorder = configure_dsv4_vision_instrumentation(enabled=True, global_rank=0)
    model_executor_path = (
        Path(__file__).parents[2]
        / "python/tokenspeed/runtime/execution/model_executor.py"
    )
    model_executor_class = _load_class_methods(
        model_executor_path,
        "ModelExecutor",
        {"_run_target_forward_instrumented"},
        {
            "begin_current_stream_timing": begin_current_stream_timing,
            "finish_current_stream_timing": finish_current_stream_timing,
            "get_is_cuda_graph_phase": lambda: True,
            "get_dsv4_vision_instrumentation": lambda: recorder,
        },
    )
    executor = object.__new__(model_executor_class)
    executor.config = SimpleNamespace(
        enable_dsv4_vision_instrumentation=True,
        model_is_mrope=False,
        pp_size=1,
    )
    executor._dsv4_instrumentation_forward = (("capture-request",), 1, (0,))
    executor._active_positions_override = None
    executor._active_multimodal_context = None
    # Capture must not even read device_module.  ModelExecutor initialization
    # assigns it after the startup forwards, and a future capture can likewise
    # enter before a runtime device handle is available.
    executor.input_buffers = SimpleNamespace(
        positions_buf=Sliceable(),
        input_ids_buf=Sliceable(),
        out_cache_loc_buf=Sliceable(),
        seq_lens_buf=Sliceable(),
        extend_prefix_lens_buf=Sliceable(),
        ngram_model_kwargs=lambda _count: {},
    )
    executor.prefill_graph = SimpleNamespace(
        can_run=lambda *_args: (_ for _ in ()).throw(
            AssertionError("prefill graph should not be consulted for mode=None")
        )
    )
    executor.model_runner = SimpleNamespace(forward=lambda *_args, **_kwargs: object())
    ctx = SimpleNamespace(bs=1, input_num_tokens=2, num_extends=1, forward_mode=None)

    executor._run_target_forward_instrumented(ctx)
    state = recorder.internal_state()["dsv4_instrumentation"]
    assert state["dispatch_forward_index"] == 0
    assert state["by_request"] == {}
    configure_dsv4_vision_instrumentation(enabled=False, global_rank=0)


def test_prefill_executor_rejection_releases_registered_transfer_state():
    class Sender:
        bootstrap_room = "room-1"

        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    class KVManager:
        def __init__(self):
            self.aborted = []
            self.discarded = []

        def abort_room(self, room, reason):
            self.aborted.append((room, reason))

        def discard_room(self, room):
            self.discarded.append(room)

    prefill_executor_path = (
        Path(__file__).parents[2] / "python/tokenspeed/runtime/pd/prefill_executor.py"
    )
    prefill_executor_class = _load_class_methods(
        prefill_executor_path,
        "DisaggPrefillExecutor",
        {"_drop_request_state", "reject_at_admission"},
        {"BootstrapInfo": object},
    )
    executor = object.__new__(prefill_executor_class)
    sender = Sender()
    executor.senders = {"request-1": sender}
    executor._local_states = {"request-1": object()}
    executor.kv_manager = KVManager()
    bootstrap = SimpleNamespace(bootstrap_room="room-1")

    executor.reject_at_admission("request-1", bootstrap, "invalid request")

    assert executor.kv_manager.aborted == [("room-1", "invalid request")]
    assert executor.kv_manager.discarded == ["room-1"]
    assert sender.cleared is True
    assert executor.senders == {}
    assert executor._local_states == {}


def test_decode_executor_rejection_releases_registered_transfer_state():
    class Receiver:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    decode_executor_path = (
        Path(__file__).parents[2] / "python/tokenspeed/runtime/pd/decode_executor.py"
    )
    decode_executor_class = _load_class_methods(
        decode_executor_path,
        "DisaggDecodeExecutor",
        {"_drop_request_state", "reject_at_admission"},
        {"BootstrapInfo": object},
    )
    executor = object.__new__(decode_executor_class)
    receiver = Receiver()
    executor.receivers = {"request-1": receiver}
    executor._local_states = {"request-1": object()}
    executor._request_pool_indices = {"request-1": 17}
    executor._remote_cache_slots = {"request-1": 18}
    executor._remote_spec_candidate_ids = {"request-1": (18, [1, 2])}

    executor.reject_at_admission("request-1", object(), "invalid request")

    assert receiver.cleared is True
    assert executor.receivers == {}
    assert executor._local_states == {}
    assert executor._request_pool_indices == {}
    assert executor._remote_cache_slots == {}
    assert executor._remote_spec_candidate_ids == {}


@pytest.mark.parametrize(
    "validation_error",
    [
        "Scheduler: request tokens must be non-empty",
        "Scheduler: duplicate request id 'invalid'",
        "Scheduler: max_new_tokens must be non-negative",
        "Scheduler: request token limit exceeds int32 range",
        "Scheduler: request token limit exceeds cache capacity",
    ],
)
def test_admission_scheduler_validation_exception_isolated_and_loop_remains_usable(
    validation_error,
):
    class FakeScheduler:
        def __init__(self, error):
            self.accepted = []
            self.error = error

        def submit_requests(self, specs):
            if any(spec.invalid for spec in specs):
                raise ValueError(self.error)
            self.accepted.extend(spec.request_id for spec in specs)

    class FakeState:
        def __init__(self):
            self.error = None

        def set_finish_with_abort(self, message):
            self.error = message

    class FakeOutputProcessor:
        def __init__(self):
            self.finished = []
            self.rid_to_state = {}

        def publish_finished_at_admission(self, request_id, state):
            self.rid_to_state[request_id] = state
            self.finished.append((request_id, state.error))
            self.rid_to_state.pop(request_id)

    logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    event_loop_path = (
        Path(__file__).parents[2] / "python/tokenspeed/runtime/engine/event_loop.py"
    )
    event_loop_class = _load_class_methods(
        event_loop_path,
        "EventLoop",
        {"_reject_at_admission", "_submit_admitted_requests"},
        {"logger": logger},
    )
    loop = object.__new__(event_loop_class)
    loop.scheduler = FakeScheduler(validation_error)
    loop.output_processor = FakeOutputProcessor()
    cleaned = []
    loop.kv_transfer = SimpleNamespace(
        reject_at_admission=lambda request_id, bootstrap, reason: cleaned.append(
            (request_id, bootstrap, reason)
        )
    )
    valid_1 = SimpleNamespace(request_id="valid-1", invalid=False)
    invalid = SimpleNamespace(request_id="invalid", invalid=True)
    valid_2 = SimpleNamespace(request_id="valid-2", invalid=False)
    invalid_state = FakeState()
    previous_state = FakeState() if "duplicate request id" in validation_error else None
    if previous_state is not None:
        loop.output_processor.rid_to_state["invalid"] = invalid_state

    loop._submit_admitted_requests(
        [
            (valid_1, FakeState()),
            (invalid, invalid_state, previous_state, "bootstrap-invalid"),
            (valid_2, FakeState()),
        ]
    )
    assert loop.scheduler.accepted == ["valid-1", "valid-2"]
    assert loop.output_processor.finished == [
        ("invalid", f"Scheduler rejected request invalid: {validation_error}")
    ]
    if previous_state is not None:
        assert loop.output_processor.rid_to_state["invalid"] is previous_state
    assert cleaned == [
        (
            "invalid",
            "bootstrap-invalid",
            f"Scheduler rejected request invalid: {validation_error}",
        )
    ]

    # The same event-loop object remains operational after the bad admission.
    valid_3 = SimpleNamespace(request_id="valid-3", invalid=False)
    loop._submit_admitted_requests([(valid_3, FakeState())])
    assert loop.scheduler.accepted[-1] == "valid-3"
