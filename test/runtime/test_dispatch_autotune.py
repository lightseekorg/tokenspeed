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

"""CPU tests of startup/dispatch policy, without importing GPU backends.

Execute the production functions with fake kernel boundaries and real tensors.
These check branch coverage and output ownership, not GPU tactic performance.
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "python/tokenspeed/runtime"
KERNEL = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"


def _functions(path, owner, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    # Use injected adapter modules for imports now placed at the file header.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in sys.modules:
            module = sys.modules[node.module]
            for name in node.names:
                if hasattr(module, name.name):
                    namespace.setdefault(
                        name.asname or name.name, getattr(module, name.name)
                    )
    scope = (
        tree
        if owner is None
        else next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    )
    body = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in body} == set(names)
    # These tests exercise function bodies, not registry initialization.
    for function in body:
        function.decorator_list = [
            decorator
            for decorator in function.decorator_list
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "register_kernel"
            )
        ]
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(path), "exec", dont_inherit=False), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("chunk_size", [-1, 16, 17])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("prefill_only", [False, True])
def test_startup_uses_native_buckets_without_reading_capture_sizes(
    chunk_size, speculative, disabled, failure, prefill_only
):
    events = []
    metadata = []
    ngram_history = torch.ones(32, 3, dtype=torch.int64)
    ngram_mask = torch.ones(32, dtype=torch.bool)

    def scrub(*, batch_size, total_tokens):
        assert (batch_size, total_tokens) == (8, 32)
        ngram_history.zero_()
        ngram_mask.zero_()
        events.append("scrub")

    def forward(**kwargs):
        n = kwargs["input_ids"].numel()
        assert torch.equal(kwargs["engram_previous_tokens"], ngram_history[:n])
        assert torch.equal(kwargs["engram_token_mask"], ngram_mask[:n])
        assert not ngram_history.any() and not ngram_mask.any()
        assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        events.append(("target", kwargs["input_ids"].numel()))
        metadata.append(torch.zeros(1))
        if failure:
            raise RuntimeError("profiling failed")

    def draft(*, batch_sizes, graph_phase):
        assert batch_sizes == (1,)
        assert graph_phase is False
        events.append("draft")

    def prefill_batch(num_tokens, batch_size):
        assert batch_size == -(-num_tokens // 4)
        assert 0 < batch_size <= 8
        return object()

    policy = Mock(return_value=nullcontext())
    namespace = dict(
        torch=torch,
        time=time,
        logger=logging.getLogger(__name__),
        autotune=policy,
        active_forward=lambda ctx: nullcontext(),
        set_autotune_max_num_tokens=Mock(),
        set_autotune_process_group=lambda group: events.append(("group", group)),
        autotune_cache_path=lambda key: "cache.json",
        load_autotune_cache=lambda *args: events.append("load"),
        save_autotune_cache=lambda *args: events.append("save"),
    )
    executor = SimpleNamespace(
        config=SimpleNamespace(
            max_num_seqs=8,
            data_parallel_size=1,
            chunked_prefill_size=chunk_size,
            context_len=4,
            world_size=1,
            world_group=(0,),
            global_rank=0,
            autotune_cache_key={},
            pp_size=1,
            disable_autotune=disabled,
            model_is_mrope=False,
            prefill_only=prefill_only,
        ),
        model_runner=SimpleNamespace(forward=forward),
        input_buffers=SimpleNamespace(
            input_ids_buf=torch.ones(32),
            positions_buf=torch.arange(32),
            max_bs=8,
            max_num_tokens=32,
            fill_dummy_decode_buffers=scrub,
            ngram_model_kwargs=lambda n: {
                "engram_previous_tokens": ngram_history[:n],
                "engram_token_mask": ngram_mask[:n],
            },
        ),
        prefill_graph=SimpleNamespace(make_dummy_batch=prefill_batch),
        # Deliberately no capture_bs or graph-enabled flag: neither controls tuning.
        forward_step=SimpleNamespace(warmup_decode_path=draft),
        drafter=object() if speculative else None,
        _autotune_draft_experts=lambda n: events.append(("draft_experts", n)),
        device="cpu",
    )
    method = _functions(
        RUNTIME / "execution/model_executor.py",
        "ModelExecutor",
        ("_autotune",),
        namespace,
    )
    if failure and not disabled:
        with pytest.raises(RuntimeError, match="profiling failed"):
            method._autotune(executor)
        assert events == [
            "load",
            ("group", None),
            "scrub",
            ("target", (32 if chunk_size < 0 else chunk_size)),
        ]
        return
    method._autotune(executor)
    if disabled:
        assert events == ["load"]
        policy.assert_not_called()
    else:
        assert events == [
            "load",
            ("group", None),
            "scrub",
            ("target", (32 if chunk_size < 0 else chunk_size)),
            *(
                [("draft_experts", (32 if chunk_size < 0 else chunk_size))]
                if speculative
                else []
            ),
            *(["draft"] if speculative and not prefill_only else []),
            ("group", None),
            "save",
        ]
        policy.assert_called_once_with(
            tune_mode=True, tuning_buckets=None, round_up=None
        )
    for tensor in metadata:
        tensor.fill_(1)  # Capture must be able to mutate warmup metadata later.


def _moe_api(impl, routing_modes, deferred):
    spec = SimpleNamespace(
        name="test_moe",
        solution="test",
        weight_preprocessor=None,
        traits={
            "routing_mode": frozenset(routing_modes),
            "supports_deferred_finalize": frozenset({deferred}),
        },
    )
    impl.name = spec.name
    impl.impl = impl
    namespace = dict(
        torch=torch,
        _normalize_weight_dtype=lambda dtype: dtype,
        _validate_a2a_backend=Mock(),
        _validate_routing_mode=Mock(),
        _validate_deepep_mode=Mock(),
        _validate_selected_deepep_mode=Mock(),
        _build_traits=Mock(return_value={}),
        select_kernel=lambda *args, **kwargs: impl,
        format_signature=lambda **kwargs: kwargs,
        dense_tensor_format=lambda dtype: dtype,
        KernelRegistry=SimpleNamespace(
            get=lambda: SimpleNamespace(get_by_name=lambda name: spec)
        ),
        _uses_all_to_all_ep=lambda backend: False,
        pdl_enabled=lambda: False,
    )
    return _functions(
        KERNEL / "ops/moe/__init__.py", None, ("moe_plan", "moe_apply"), namespace
    )


def _make_moe_plan(api, routing_mode):
    return api.moe_plan(
        weight_dtype="mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        requires_deferred_finalize=False,
        routing_mode=routing_mode,
        a2a_backend=None,
        ep_size=1,
        ispp=128,
        hidden=6,
        fp8_scale_block_shape=None,
        internal_activation_dtype="fp8",
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        with_bias=False,
        process_group=None,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution=None,
    )


def _situ_backend(calls, fail, tuning):
    prepared = {}

    def launch(
        w,
        router_logits,
        topk_weights,
        topk_ids,
        x,
        hidden_states_scale,
        output,
        enable_pdl,
        do_finalize,
    ):
        assert x is prepared["x"]
        assert hidden_states_scale.data_ptr() == prepared["scales"].data_ptr()
        mode = "kernel_routing" if topk_ids is None else "precomputed_topk"
        calls.append((mode, do_finalize, output))
        if fail:
            raise RuntimeError("profiling failed")
        # Only the ordinary apply call may write the model's persistent buffer.
        if output is not None:
            output.fill_(10 if mode == "kernel_routing" else 20)
            return output
        if topk_ids is not None:
            assert topk_ids.shape == topk_weights.shape == (x.shape[0], w.top_k)
            assert topk_ids.min() >= 0 and topk_ids.max() < w.num_experts
        return (
            torch.full(x.shape, 30, dtype=torch.bfloat16),
            torch.ones(x.shape[0], w.top_k, dtype=torch.bfloat16),
            torch.arange(x.shape[0], dtype=torch.int32),
        )

    def quantize(x, is_sf_swizzled_layout, *, enable_pdl, alignment, backend):
        assert x.shape[-1] == alignment == 8
        assert torch.count_nonzero(x[:, 6:]) == 0
        prepared["x"] = x.clone()
        prepared["scales"] = torch.ones(x.shape[0], 1, dtype=torch.uint8)
        return prepared["x"], prepared["scales"]

    quantizer = Mock(side_effect=quantize)
    namespace = dict(
        torch=torch,
        mxfp8_quantize=quantizer,
        is_autotuning=lambda: tuning,
        _call_mxfp4_situ_moe=launch,
        _register_mxfp4_situ_kernel=lambda fn: fn,
    )
    backend = _functions(
        KERNEL / "ops/moe/flashinfer/trtllm_mxfp4.py",
        None,
        (
            "_autotune_mxfp4_situ_moe",
            "flashinfer_trtllm_mxfp4_situ_moe_apply",
        ),
        namespace,
    )
    return backend, quantizer


@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("do_finalize", [False, True])
@pytest.mark.parametrize("routing", ["kernel_routing", "precomputed_topk", "topk_only"])
def test_moe_profiles_on_scratch_and_runs_normal_apply_once(
    monkeypatch, tuning, do_finalize, routing
):
    calls = []
    backend, quantizer = _situ_backend(calls, False, tuning)
    impl = backend.flashinfer_trtllm_mxfp4_situ_moe_apply
    api = _moe_api(impl, ("kernel_routing", "precomputed_topk"), True)
    plan = _make_moe_plan(api, "precomputed_topk" if routing == "topk_only" else None)
    x = torch.randn(3, 6, dtype=torch.bfloat16)
    logits = None if routing == "topk_only" else torch.randn(3, 4)
    ids = (
        None
        if routing == "kernel_routing"
        else torch.tensor([[0, 1]] * 3, dtype=torch.int32)
    )
    weights = None if ids is None else torch.full((3, 2), 0.5, dtype=torch.bfloat16)
    inputs = [t for t in (x, logits, ids, weights) if t is not None]
    before = [t.clone() for t in inputs]
    out = torch.full((3, 8), -1, dtype=torch.bfloat16)
    w = SimpleNamespace(
        hidden_size_padded=8,
        hidden_size_original=6,
        w2_weight_scale=torch.empty(4, 8),
        top_k=2,
        num_experts=4,
        _situ_output_buffer=out,
    )
    result = api.moe_apply(
        plan=plan,
        x=x,
        w=w,
        router_logits=logits,
        topk_weights=weights,
        topk_ids=ids,
        num_tokens_global=3,
        max_num_tokens_per_gpu=3,
        do_finalize=do_finalize,
        low_latency=False,
        overlap_fn=None,
        shared_input=None,
        shared_weight=None,
        shared_out=None,
    )
    actual = ("kernel_routing" if ids is None else "precomputed_topk", do_finalize)
    modes = (
        ("precomputed_topk",)
        if logits is None
        else ("kernel_routing", "precomputed_topk")
    )
    variants = {(r, f) for r in modes for f in (False, True)}
    assert {(r, f) for r, f, _ in calls} == (variants if tuning else {actual})
    assert len(calls) == (len(variants) if tuning else 1)
    assert calls[-1][:2] == actual
    assert quantizer.call_count == 1
    for _, _, scratch in calls[:-1]:
        assert scratch is None or scratch.data_ptr() != out.data_ptr()
    if do_finalize:
        assert calls[-1][2] is out
        torch.testing.assert_close(result, out[:, :6])
    else:
        assert isinstance(result, tuple) and len(result) == 3
        assert torch.all(result[0] == 30)
        assert torch.all(out == -1)
    for tensor, original in zip(inputs, before):
        torch.testing.assert_close(tensor, original)


@pytest.mark.parametrize("case", ["empty", "no_router", "missing_weights"])
def test_situ_profiling_validates_inputs_before_launch(case):
    calls = []
    backend, quantizer = _situ_backend(calls, False, True)
    plan = {}
    context = nullcontext() if case == "empty" else pytest.raises(ValueError)
    with context:
        backend.flashinfer_trtllm_mxfp4_situ_moe_apply(
            plan=plan,
            x=torch.empty(0 if case == "empty" else 3, 6, dtype=torch.bfloat16),
            w=SimpleNamespace(w2_weight_scale=torch.empty(4, 8)),
            router_logits=None if case == "no_router" else torch.ones(3, 4),
            topk_weights=None,
            topk_ids=(
                torch.zeros(3, 2, dtype=torch.int32)
                if case == "missing_weights"
                else None
            ),
            num_tokens_global=0 if case == "empty" else 3,
            max_num_tokens_per_gpu=3,
            do_finalize=True,
            enable_pdl=False,
        )
    assert calls == []
    quantizer.assert_not_called()


def test_moe_profiling_failure_leaves_model_output_untouched(monkeypatch):
    calls = []
    backend, _ = _situ_backend(calls, True, True)
    impl = backend.flashinfer_trtllm_mxfp4_situ_moe_apply
    api = _moe_api(impl, ("kernel_routing", "precomputed_topk"), True)
    plan = _make_moe_plan(api, None)
    out = torch.full((3, 8), -1, dtype=torch.bfloat16)
    w = SimpleNamespace(
        hidden_size_padded=8,
        w2_weight_scale=torch.empty(4, 8),
        top_k=2,
        num_experts=4,
        _situ_output_buffer=out,
    )
    with pytest.raises(RuntimeError, match="profiling failed"):
        api.moe_apply(
            plan=plan,
            x=torch.ones(3, 6, dtype=torch.bfloat16),
            w=w,
            router_logits=torch.ones(3, 4),
            topk_weights=None,
            topk_ids=None,
            num_tokens_global=3,
            max_num_tokens_per_gpu=3,
            do_finalize=True,
            low_latency=False,
            overlap_fn=None,
            shared_input=None,
            shared_weight=None,
            shared_out=None,
        )
    assert len(calls) == 1
    assert torch.all(out == -1)


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_situ_shared_launcher_preserves_ep_routing_and_return_contract(
    precomputed, finalize
):
    output = torch.empty(3, 8, dtype=torch.bfloat16) if finalize else None
    expert_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)
    # Kernel routing writes a packed BF16 prefix into FP32 storage.
    packed = torch.empty(3, 2, dtype=torch.float32)
    packed.view(torch.bfloat16).flatten()[:6].copy_(expert_weights.flatten())
    deferred = (
        torch.zeros(6, 8, dtype=torch.bfloat16),
        expert_weights if precomputed else packed,
        torch.arange(6, dtype=torch.int32),
    )
    fi = Mock(return_value=output if finalize else deferred)
    backend = _functions(
        KERNEL / "ops/moe/flashinfer/trtllm_mxfp4.py",
        None,
        (
            "_local_expert_range",
            "_routing_value",
            "_call_mxfp4_routed_moe",
            "_normalize_kernel_routing_deferred_result",
            "_call_mxfp4_situ_moe",
        ),
        dict(
            torch=torch,
            trtllm_fp4_block_scale_routed_moe=fi,
            trtllm_fp4_block_scale_moe=fi,
            ActivationType=SimpleNamespace(Situ=42),
            get_autotune_max_num_tokens=lambda: 8192,
        ),
    )
    w = SimpleNamespace(
        num_experts=8,
        num_local_experts=4,
        ep_rank=1,
        top_k=2,
        intermediate_size_per_partition=128,
        w13_weight=torch.empty(4, 16, 8),
        w2_weight=torch.empty(4, 8, 16),
        w13_weight_scale=torch.zeros(4, 16, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(4, 8, dtype=torch.uint8),
        gemm1_alpha=torch.ones(4),
        gemm1_beta=torch.ones(4),
        routing_config={"routed_scaling_factor": 1.25, "n_group": 2, "topk_group": 1},
    )
    logits = torch.randn(3, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 5], [2, 7], [3, 6]], dtype=torch.int64)
    result = backend._call_mxfp4_situ_moe(
        w=w,
        router_logits=logits,
        topk_weights=expert_weights if precomputed else None,
        topk_ids=ids if precomputed else None,
        x=torch.ones(3, 8),
        hidden_states_scale=torch.ones(3, 1),
        output=output,
        enable_pdl=True,
        do_finalize=finalize,
    )
    fi.assert_called_once()
    args = fi.call_args.kwargs
    assert args["num_experts"] == 8
    assert args["local_num_experts"] == 4 and args["local_expert_offset"] == 4
    assert args["enable_pdl"] is True and args["do_finalize"] is finalize
    assert args["tune_max_num_tokens"] == 8192
    assert args["output"] is output
    if precomputed:
        assert args["topk_ids"][0].dtype == torch.int32
        torch.testing.assert_close(args["topk_ids"][0].long(), ids)
    else:
        torch.testing.assert_close(args["routing_logits"], logits.float())
        assert args["routed_scaling_factor"] == 1.25
        assert args["n_group"] == 2 and args["topk_group"] == 1
    if finalize:
        assert result is output
    else:
        assert result[0] is deferred[0] and result[2] is deferred[2]
        torch.testing.assert_close(result[1], expert_weights)


@pytest.mark.parametrize("decode_disabled", [False, True])
@pytest.mark.parametrize("prefill_disabled", [False, True])
@pytest.mark.parametrize("has_drafter", [False, True])
def test_capture_lifecycle_preserves_main_order(
    decode_disabled, prefill_disabled, has_drafter
):
    events = []
    step = SimpleNamespace(
        disable=decode_disabled,
        stream=object(),
        capture=lambda: events.append("decode"),
    )

    def capture_prefill(runner):
        assert runner is step
        events.append("prefill")

    def capture_draft(stream):
        assert stream is step.stream
        events.append("draft")

    executor = SimpleNamespace(
        _autotune=lambda: events.append("tune"),
        device="cpu",
        forward_step=step,
        prefill_graph=SimpleNamespace(
            disable=prefill_disabled, capture=capture_prefill
        ),
        drafter=(
            SimpleNamespace(capture_prefill_graph=capture_draft)
            if has_drafter
            else None
        ),
    )
    methods = _functions(
        RUNTIME / "execution/model_executor.py",
        "ModelExecutor",
        ("capture_graphs",),
        {
            "workspace_pool": lambda device: SimpleNamespace(
                freeze=lambda: events.append("freeze")
            )
        },
    )
    methods.capture_graphs(executor)
    assert events == [
        "tune",
        "freeze",
        *([] if decode_disabled else ["decode"]),
        *([] if prefill_disabled else ["prefill"]),
        *(["draft"] if not prefill_disabled and has_drafter else []),
    ]


def test_executor_construction_does_not_capture_or_tune():
    tree = ast.parse((RUNTIME / "execution/model_executor.py").read_text())
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelExecutor"
    )
    init = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    forbidden = {"_autotune", "capture", "capture_graphs", "capture_prefill_graph"}
    assert not [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in forbidden
    ]


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("provided_out", [False, True])
@pytest.mark.parametrize(
    "n,k,expected",
    [
        (32, 128, ["tgv", "cute-dsl"]),
        (31, 256, ["tgv", "cute-dsl"]),
        (32, 160, ["tgv"]),
        (32, 2112, ["tgv"]),
    ],
)
def test_bf16_joint_adapter_uses_one_fi_dispatch(pdl, provided_out, n, k, expected):
    calls = []
    workspace = object()

    def dispatch(**kwargs):
        calls.append(kwargs)
        kwargs["out"].copy_(kwargs["a"] @ kwargs["b"])

    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("flashinfer_bf16_gemm", "_bf16_gemm_runner_names"),
        dict(
            torch=torch,
            BF16_GEMM_MAX_M=32,
            pdl_enabled=lambda: pdl,
            flashinfer_joint_bf16_supported=lambda *args: True,
            _fi_gemm=SimpleNamespace(
                DEFAULT_WORKSPACE_SIZE=4096,
                _get_cache_buf=Mock(return_value=workspace),
                bf16_gemm_sm100=dispatch,
            ),
        ),
    )
    x = torch.randn(3, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    out = torch.empty(3, n, dtype=torch.bfloat16) if provided_out else None
    result = api.flashinfer_bf16_gemm(x, w, out)
    torch.testing.assert_close(result, x @ w.T)
    assert not provided_out or result is out
    assert len(calls) == 1
    assert calls[0]["runner_names"] == expected
    assert calls[0]["bias"] is None  # No artificial zero bias excluding direct.
    assert calls[0]["workspace_buffer"] is workspace
    assert calls[0]["pdl"] is pdl


@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_bf16_discovery_uses_one_native_joint_call(tuning, supported):
    probe = Mock()
    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("autotune_bf16_gemm",),
        dict(
            torch=torch,
            is_autotuning=lambda: tuning,
            BF16_GEMM_MAX_M=32,
            flashinfer_joint_bf16_supported=lambda *args: supported,
            flashinfer_bf16_gemm=probe,
        ),
    )
    x = torch.ones(128, 128, dtype=torch.bfloat16)
    w = torch.ones(32, 128, dtype=torch.bfloat16)
    api.autotune_bf16_gemm(x, w)
    assert torch.all(x == 1) and torch.all(w == 1)
    if tuning and supported:
        probe.assert_called_once()
        sample, weight, out = probe.call_args.args
        assert sample.shape == (32, 128) and sample.data_ptr() != x.data_ptr()
        assert weight is w and out is None
    else:
        probe.assert_not_called()


@pytest.mark.parametrize(
    "invalid", [None, "cpu", "dtype", "strided", "unaligned", "n", "k", "out"]
)
def test_joint_bf16_support_contract(invalid):
    x = SimpleNamespace(
        is_cuda=invalid != "cpu",
        device="cuda:0",
        ndim=2,
        dtype=torch.float32 if invalid == "dtype" else torch.bfloat16,
        shape=(128, 128),
        is_contiguous=lambda: invalid != "strided",
        data_ptr=lambda: 4 if invalid == "unaligned" else 32,
    )
    w = SimpleNamespace(
        device="cuda:0",
        ndim=2,
        dtype=torch.bfloat16,
        shape=(0 if invalid == "n" else 32, 127 if invalid == "k" else 128),
        is_contiguous=lambda: True,
        data_ptr=lambda: 64,
    )
    out = (
        SimpleNamespace(shape=(128, 32), dtype=torch.float32)
        if invalid == "out"
        else None
    )
    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("flashinfer_joint_bf16_supported", "_bf16_gemm_runner_names"),
        dict(torch=torch, _fi_gemm=object(), has_flashinfer_cute_dsl_bf16=lambda: True),
    )
    assert api.flashinfer_joint_bf16_supported(x, w, out) == (invalid is None)


@pytest.mark.parametrize("rows", [4, 32, 33, 64, 128])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("override", [None, "explicit", "context", "environment"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_mm_joint_dispatch_respects_overrides_and_contract(
    rows, bias, override, out_dtype
):
    probe = Mock()
    joint = Mock(side_effect=lambda a, b, out: torch.mm(a, b.T, out=out))
    record = Mock()
    scope = Mock(return_value=nullcontext())

    def generic(a, b, a_scales, b_scales, out_dtype, *, alpha, block_size, out):
        return out.copy_(torch.mm(a, b.T).to(out_dtype))

    generic.name = "test_mm"
    ns = dict(
        torch=torch,
        autotune_bf16_gemm=probe,
        flashinfer_bf16_gemm=joint,
        flashinfer_joint_bf16_supported=lambda *args: True,
        BF16_GEMM_MAX_M=32,
        resolve_kernel_override=lambda family, mode, explicit: override,
        pdl_enabled=lambda: False,
        Platform=SimpleNamespace(get=lambda: SimpleNamespace(is_blackwell_plus=True)),
        _gemm_format_signature=lambda *args: SimpleNamespace(
            storage_dtype_for=lambda name: torch.bfloat16
        ),
        select_kernel=lambda *args, **kwargs: generic,
        _KERNELS_WITH_FUSED_BIAS=set(),
        _KERNELS_WITH_PDL=set(),
        ShapeCapture=SimpleNamespace(get=lambda: SimpleNamespace(record=record)),
        kernel_scope=scope,
    )
    api = _functions(
        KERNEL / "ops/gemm/__init__.py", None, ("mm", "_validate_gemm_out"), ns
    )
    x = torch.randn(rows, 4, dtype=torch.bfloat16)
    w = torch.randn(8, 4, dtype=torch.bfloat16)
    out = torch.empty(rows, 8, dtype=out_dtype)
    b = torch.ones(8, dtype=torch.bfloat16) if bias else None
    got = api.mm(
        x,
        w,
        A_scales=None,
        B_scales=None,
        bias=b,
        out=out,
        out_dtype=out_dtype,
        alpha=None,
        block_size=None,
        quant=None,
        override=None,
        prepacked_scales=False,
    )
    assert got is out
    torch.testing.assert_close(
        got, (x @ w.T).to(out_dtype) + (b if b is not None else 0)
    )
    # Rank-local bias must not make ranks skip collective tactic profiling.
    assert probe.call_count == int(override is None and out_dtype == torch.bfloat16)
    assert joint.call_count == int(
        override is None and not bias and out_dtype == torch.bfloat16 and rows <= 32
    )
    kernel_name = "flashinfer_bf16_gemm" if joint.called else "test_mm"
    record.assert_called_once_with(
        "gemm", "mm", kernel_name, x.dtype, {"M": rows, "N": 8, "K": 4}
    )
    scope.assert_called_once_with(
        "gemm",
        "mm",
        x.dtype,
        kernel_name=kernel_name,
        M=rows,
        N=8,
        K=4,
        has_out=True,
    )


def test_joint_adapter_propagates_fi_failure():
    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("flashinfer_bf16_gemm", "_bf16_gemm_runner_names"),
        dict(
            torch=torch,
            BF16_GEMM_MAX_M=32,
            pdl_enabled=lambda: False,
            flashinfer_joint_bf16_supported=lambda *args: True,
            _fi_gemm=SimpleNamespace(
                DEFAULT_WORKSPACE_SIZE=4096,
                _get_cache_buf=Mock(return_value=object()),
                bf16_gemm_sm100=Mock(side_effect=RuntimeError("FI failed")),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="FI failed"):
        api.flashinfer_bf16_gemm(torch.empty(1, 128), torch.empty(32, 128), None)


@pytest.mark.parametrize(
    "name",
    [
        "kimi3_qkvfab_projection",
        "kimi3_latent_projection",
        "kimi3_mla_qkv_gate_projection",
    ],
)
def test_projection_discovery_precedes_auto_dispatch(name):
    tree = ast.parse((KERNEL / "ops/gemm/kimi3.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    auto = next(
        n
        for n in fn.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "solution == 'auto'"
    )
    assert isinstance(auto.body[0], ast.Expr)
    assert auto.body[0].value.func.id == "autotune_bf16_gemm"


@pytest.mark.parametrize("m", [33, 48, 64, 128])
def test_joint_bf16_adapter_declines_large_m(m):
    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("flashinfer_bf16_gemm", "_bf16_gemm_runner_names"),
        dict(
            torch=torch,
            BF16_GEMM_MAX_M=32,
            flashinfer_joint_bf16_supported=lambda *args: True,
        ),
    )
    with pytest.raises(ValueError, match="Unsupported input"):
        api.flashinfer_bf16_gemm(torch.empty(m, 128), torch.empty(32, 128), None)


@pytest.mark.parametrize(
    "k,expected",
    [
        (0, []),
        (127, []),
        (128, ["tgv", "cute-dsl"]),
        (160, ["tgv"]),
        (320, ["tgv"]),
        (2112, ["tgv"]),
        (2560, ["tgv", "cute-dsl"]),
    ],
)
def test_bf16_backend_eligibility_is_independent(k, expected):
    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py", None, ("_bf16_gemm_runner_names",), {}
    )
    assert api._bf16_gemm_runner_names(k) == expected


@pytest.mark.parametrize("n,k", [(2560, 160), (2560, 320), (7168, 2112), (4120, 2560)])
def test_joint_bf16_restores_previously_routed_shapes(n, k):
    def tensor(shape):
        return SimpleNamespace(
            is_cuda=True,
            device="cuda:0",
            ndim=2,
            dtype=torch.bfloat16,
            shape=shape,
            is_contiguous=lambda: True,
            data_ptr=lambda: 32,
        )

    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("flashinfer_joint_bf16_supported", "_bf16_gemm_runner_names"),
        dict(torch=torch, _fi_gemm=object(), has_flashinfer_cute_dsl_bf16=lambda: True),
    )
    # Discovery must not reject the N/K just because the actual warmup is large.
    for m in (1, 3, 31, 32, 128):
        assert api.flashinfer_joint_bf16_supported(
            tensor((m, k)), tensor((n, k)), tensor((m, n))
        )


def test_flashinfer_probe_reads_the_declared_backends() -> None:
    """0.6.18 declares the backend; earlier wheels name every other one."""

    def upstreamed(backend: Literal["cudnn", "cute-dsl"] = "cudnn") -> None: ...

    def earlier(backend: Literal["cudnn", "tgv", "tinygemm"] = "cudnn") -> None: ...

    api = _functions(
        KERNEL / "ops/gemm/flashinfer.py",
        None,
        ("_declares_cute_dsl_backend",),
        dict(inspect=inspect, get_args=get_args, _CUTE_DSL_BACKEND="cute-dsl"),
    )
    assert api._declares_cute_dsl_backend(upstreamed)
    assert not api._declares_cute_dsl_backend(earlier)
    assert not api._declares_cute_dsl_backend(lambda: None)


@pytest.mark.parametrize(
    "fi,m,cdna5,k,registered,expected",
    [
        (True, 1, False, 128, False, True),
        (True, 32, False, 128, False, True),
        (True, 33, False, 128, False, False),
        (False, 16, True, 1536, True, True),
        (False, 32, True, 1536, True, True),
        (False, 16, True, 1536, False, False),
        (False, 1, True, 128, True, False),
        (False, 16, False, 1536, True, False),
    ],
)
def test_decode_gemv_eligibility_preserves_fi_and_cdna5(
    fi, m, cdna5, k, registered, expected
):
    fallback = Mock()
    select = Mock(return_value=Mock() if registered else fallback)
    tensor = lambda shape: SimpleNamespace(
        shape=shape,
        ndim=2,
        dtype=torch.bfloat16,
        is_cuda=True,
        is_contiguous=lambda: True,
    )
    api = _functions(
        KERNEL / "ops/gemm/triton_gemv.py",
        None,
        ("use_decode_gemv",),
        dict(
            torch=torch,
            BF16_GEMM_MAX_M=32,
            flashinfer_joint_bf16_supported=lambda *args: fi,
            current_platform=lambda: SimpleNamespace(is_cdna5=cdna5),
            _select=select,
            torch_decode_gemv=fallback,
        ),
    )
    assert api.use_decode_gemv(tensor((m, k)), tensor((7168, k))) is expected
    if fi and m <= 32:
        select.assert_not_called()


@pytest.mark.parametrize(
    "m,n,k,on_cuda,expected",
    [
        (1, 32, 256, True, True),
        (1, 48, 256, True, True),
        (2, 32, 256, True, False),
        (1, 31, 256, True, False),
        (1, 32, 128, True, False),
        (1, 32, 256, False, False),
    ],
)
def test_decode_gemv_selection_obeys_shape_traits(m, n, k, on_cuda, expected):
    from tokenspeed_kernel.selection import (
        spec_matches_shape_traits,
        spec_matches_traits,
    )

    platform = object()
    impl, fallback = Mock(), Mock()
    spec = SimpleNamespace(
        name="specialized",
        traits={
            "m": frozenset({1}),
            "n_align": frozenset({16}),
            "k_min": frozenset({256}),
        },
    )
    registry = SimpleNamespace(
        get_for_operator=Mock(return_value=[spec]),
        get_impl=Mock(return_value=impl),
    )
    api = _functions(
        KERNEL / "ops/gemm/triton_gemv.py",
        None,
        ("_select",),
        dict(
            functools=functools,
            KernelRegistry=SimpleNamespace(get=lambda: registry),
            current_platform=lambda: platform,
            spec_matches_traits=spec_matches_traits,
            spec_matches_shape_traits=spec_matches_shape_traits,
            torch_decode_gemv=fallback,
        ),
    )
    assert api._select(m, n, k, on_cuda) is (impl if expected else fallback)
    if on_cuda:
        registry.get_for_operator.assert_called_once_with(
            "gemm", "decode_gemv", platform=platform
        )
    else:
        registry.get_for_operator.assert_not_called()


@pytest.mark.parametrize(
    "capturing,warmed", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("failure", [False, True])
def test_skinny_add3_preserves_capture_trust(capturing, warmed, failure):
    key = (0, 1, 4, 8)
    known = {key} if warmed else set()
    kernel = Mock(side_effect=RuntimeError("compile failed") if failure else None)
    kernel.supports.return_value = True
    fallback = Mock()
    api = _functions(
        KERNEL / "ops/gemm/kimi3.py",
        None,
        ("_skinny_gemv_add3",),
        dict(
            torch=SimpleNamespace(
                bfloat16=torch.bfloat16,
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing),
            ),
            _SKINNY_ADD3_CONFIGS={(1, 4, 8): (64, 4, 2)},
            _skinny_add3_arch_supported=lambda dev: True,
            _skinny_add3_warmed=known,
            _skinny_add3_warmed_lock=threading.Lock(),
            _skinny_add3_fallback=fallback,
            SkinnyGemmConfig=lambda *args: args,
            shape_dynamic_skinny_gemm=kernel,
        ),
    )
    x, w = torch.empty(1, 8, dtype=torch.bfloat16), torch.empty(
        4, 8, dtype=torch.bfloat16
    )
    a, c = torch.empty(1, 4, dtype=torch.bfloat16), torch.empty(
        1, 4, dtype=torch.bfloat16
    )
    if capturing and not warmed:
        assert api._skinny_gemv_add3(x, w, a, c, None) is fallback.return_value
        kernel.assert_not_called()
        assert not known
    elif failure:
        with pytest.raises(RuntimeError, match="compile failed"):
            api._skinny_gemv_add3(x, w, a, c, None)
        assert known == ({key} if warmed else set())
    else:
        assert api._skinny_gemv_add3(x, w, a, c, None) is kernel.return_value
        kernel.assert_called_once()
        fallback.assert_not_called()
        assert key in known
