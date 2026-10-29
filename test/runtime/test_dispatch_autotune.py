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

"""Startup-tuning and GEMM/MoE dispatch policy tests, against the real code.

The functions under test are imported and executed; doubles stand in only at
the boundaries a unit-test process genuinely lacks: GPU kernel launches and
CUDA tensors, torch.distributed process groups, and the heavyweight
ModelExecutor boot (model weights, cache pools, backends). Assertions check
outcomes -- computed values, buffer ownership, dispatch decisions and the
order of the startup steps -- not the shape of the production source.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal, get_args

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# CPU-side policy tests; scheduled in runtime-1gpu because importing the real
# runtime and kernel modules requires a detected device platform.
register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed_kernel.ops import gemm as gemm_module  # noqa: E402
from tokenspeed_kernel.ops import moe as moe_module  # noqa: E402
from tokenspeed_kernel.ops import tuning as tuning_module  # noqa: E402
from tokenspeed_kernel.ops.gemm import flashinfer as fi_gemm_module  # noqa: E402
from tokenspeed_kernel.ops.gemm import triton as gemm_triton_module  # noqa: E402
from tokenspeed_kernel.ops.moe import flashinfer as mxfp4_module  # noqa: E402
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec  # noqa: E402
from tokenspeed_kernel.selection import (  # noqa: E402
    SelectedKernel,
    kernel_override,
)

from tokenspeed.runtime.execution import model_executor as executor_module  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.query_shard import QueryShardPlan  # noqa: E402
from tokenspeed.runtime.execution.workspace import (  # noqa: E402
    reset_workspace_pools,
    workspace_pool,
)
from tokenspeed.runtime.layers.dense import unquant as unquant_module  # noqa: E402
from tokenspeed.runtime.layers.dense.unquant import (  # noqa: E402
    UnquantizedLinearMethod,
)
from tokenspeed.runtime.utils import env as env_module  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _tap(monkeypatch, module, name, record):
    """Spy on ``module.name``: record the call, then run the real function."""
    real = getattr(module, name)

    def tapped(*args, **kwargs):
        record(*args, **kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, tapped)
    return real


# ---------------------------------------------------------------------------
# ModelExecutor.autotune doubles
# ---------------------------------------------------------------------------


@dataclass
class _AutotuneConfig:
    """The ModelExecutorConfig fields ModelExecutor.autotune reads."""

    max_num_seqs: int = 8
    data_parallel_size: int = 1
    chunked_prefill_size: int = -1
    context_len: int = 4
    world_size: int = 1
    world_group: tuple[int, ...] | None = (0,)
    global_rank: int = 0
    autotune_cache_key: dict | None = None
    pp_size: int = 1
    disable_autotune: bool = False
    model_is_mrope: bool = False
    prefill_only: bool = False
    decode_only_attention: bool = False


class _ModelRunnerDouble:
    def __init__(self, forward):
        self.forward = forward


class _InputBuffersDouble:
    def __init__(
        self, *, input_ids_buf, positions_buf, max_bs, max_num_tokens, on_fill_dummy
    ):
        self.input_ids_buf = input_ids_buf
        self.positions_buf = positions_buf
        self.mrope_positions_buf = None
        self.max_bs = max_bs
        self.max_num_tokens = max_num_tokens
        self._on_fill_dummy = on_fill_dummy

    def fill_dummy_decode_buffers(self, *, batch_size, total_tokens):
        self._on_fill_dummy(batch_size=batch_size, total_tokens=total_tokens)


class _DummyBatch:
    def __init__(self, *, bs, query_shard):
        self.bs = bs
        self.query_shard = query_shard


class _PrefillGraphDouble:
    def __init__(self, make_dummy_batch):
        self.make_dummy_batch = make_dummy_batch


class _ForwardStepDouble:
    def __init__(
        self, *, warmup_decode_path=None, max_decode_bs=None, max_tokens_per_req=None
    ):
        self.warmup_decode_path = warmup_decode_path
        self.max_decode_bs = max_decode_bs
        self.max_tokens_per_req = max_tokens_per_req


class _AutotuneExecutor:
    """The ModelExecutor state autotune() reads, without the GPU boot."""

    def __init__(
        self,
        *,
        config,
        forward,
        input_buffers,
        prefill_graph,
        forward_step,
        drafter,
        model_input_kwargs,
        events,
    ):
        self.config = config
        self.model_runner = _ModelRunnerDouble(forward)
        self.input_buffers = input_buffers
        self.prefill_graph = prefill_graph
        self.forward_step = forward_step
        self.drafter = drafter
        self._model_input_kwargs_hook = model_input_kwargs
        self._events = events
        self.device = "cpu"

    def _model_input_kwargs(self, num_tokens, bs, rows):
        return self._model_input_kwargs_hook(num_tokens, bs, rows)

    def _autotune_draft_experts(self, num_tokens):
        assert tuning_module.is_warming_up()
        self._events.append(("draft_experts", num_tokens))


class _ProcessGroupManagerDouble:
    """Hands out a stand-in CPU group; a real gloo group needs torch.distributed."""

    def __init__(self, group):
        self._group = group
        self.requests = []

    def get_process_group(self, backend, world_group):
        self.requests.append((backend, world_group))
        return self._group


@pytest.mark.parametrize("chunk_size", [-1, 16, 17])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("prefill_only", [False, True])
def test_startup_uses_native_buckets_without_reading_capture_sizes(
    monkeypatch, tmp_path, chunk_size, speculative, disabled, failure, prefill_only
):
    events = []
    metadata = []
    ngram_history = torch.ones(32, 2, 6, dtype=torch.int64)
    ngram_mask = torch.ones(32, dtype=torch.bool)

    def scrub(*, batch_size, total_tokens):
        assert (batch_size, total_tokens) == (8, 32)
        ngram_history.zero_()
        ngram_mask.zero_()
        events.append("scrub")

    def forward(**kwargs):
        n = kwargs["input_ids"].numel()
        assert torch.equal(kwargs["engram_hash_ids"], ngram_history[:n])
        assert torch.equal(kwargs["engram_token_mask"], ngram_mask[:n])
        assert not ngram_history.any() and not ngram_mask.any()
        assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        assert tuning_module.is_warming_up()
        assert tuning_module.is_autotuning() is (
            not disabled and tuning_module._autotuner is not None
        )
        events.append(("target", kwargs["input_ids"].numel()))
        metadata.append(torch.zeros(1))
        if failure:
            raise RuntimeError("profiling failed")

    def draft(*, batch_sizes, graph_phase):
        assert batch_sizes == (1,)
        assert graph_phase is False
        assert tuning_module.is_warming_up()
        events.append("draft")

    def prefill_batch(num_tokens, batch_size):
        assert batch_size == -(-num_tokens // 4)
        assert 0 < batch_size <= 8
        return _DummyBatch(bs=batch_size, query_shard=None)

    executor = _AutotuneExecutor(
        config=_AutotuneConfig(
            chunked_prefill_size=chunk_size,
            autotune_cache_key={},
            disable_autotune=disabled,
            prefill_only=prefill_only,
        ),
        forward=forward,
        input_buffers=_InputBuffersDouble(
            input_ids_buf=torch.ones(32),
            positions_buf=torch.arange(32),
            max_bs=8,
            max_num_tokens=32,
            on_fill_dummy=scrub,
        ),
        prefill_graph=_PrefillGraphDouble(make_dummy_batch=prefill_batch),
        # Deliberately no capture_bs or graph-enabled flag: neither controls tuning.
        forward_step=_ForwardStepDouble(warmup_decode_path=draft),
        drafter=object() if speculative else None,
        model_input_kwargs=lambda n, bs, rows: {
            "engram_hash_ids": ngram_history[rows],
            "engram_token_mask": ngram_mask[rows],
        },
        events=events,
    )
    # The cache round-trips through a tmp file; every other tuning helper runs
    # for real behind a recording tap.
    monkeypatch.setattr(
        executor_module, "autotune_cache_path", lambda key: str(tmp_path / "cache.json")
    )
    _tap(
        monkeypatch,
        executor_module,
        "load_autotune_cache",
        lambda *args: events.append("load"),
    )
    _tap(
        monkeypatch,
        executor_module,
        "save_autotune_cache",
        lambda *args: events.append("save"),
    )
    _tap(
        monkeypatch,
        executor_module,
        "set_autotune_process_group",
        lambda group: events.append(("group", group)),
    )
    if failure:
        with pytest.raises(RuntimeError, match="profiling failed"):
            ModelExecutor.autotune(executor)
        assert events == [
            "load",
            ("group", None),
            "scrub",
            ("target", (32 if chunk_size < 0 else chunk_size)),
            ("group", None),
        ]
    else:
        ModelExecutor.autotune(executor)
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
            *(["save"] if not disabled else []),
        ]
    assert not tuning_module.is_warming_up()
    assert not tuning_module.is_autotuning()
    for tensor in metadata:
        tensor.fill_(1)  # Capture must be able to mutate warmup metadata later.


def test_a_query_sharding_engine_tunes_on_the_shards_rows(monkeypatch):
    """The dummy extend carries the engine's shard plan (make_dummy_batch),
    and the target forward sees this rank's rows of the span -- ids,
    positions and the per-row model inputs -- as a real sharded extend does."""
    plan = QueryShardPlan.from_forward(
        total_tokens=16, input_lengths=[4, 4, 4, 4], size=4, rank=2, device="cpu"
    )
    # Every request is four rows, one per zigzag segment of the first four:
    # rank 2 computes the third row of each.
    assert plan.local_row_ids.tolist() == [2, 6, 10, 14]
    seen = {}
    events = []
    group = object()  # Stands in for the gloo CPU group.
    pg_manager = _ProcessGroupManagerDouble(group)

    def forward(**kwargs):
        seen.update(kwargs)

    executor = _AutotuneExecutor(
        config=_AutotuneConfig(
            chunked_prefill_size=16,
            world_size=4,
            world_group=(0, 1, 2, 3),
            global_rank=2,
            prefill_only=True,
        ),
        forward=forward,
        input_buffers=_InputBuffersDouble(
            input_ids_buf=torch.arange(32),
            positions_buf=torch.arange(32) * 10,
            max_bs=8,
            max_num_tokens=32,
            on_fill_dummy=lambda **kwargs: None,
        ),
        prefill_graph=_PrefillGraphDouble(
            make_dummy_batch=lambda n, bs: _DummyBatch(bs=bs, query_shard=plan)
        ),
        forward_step=_ForwardStepDouble(),
        drafter=None,
        model_input_kwargs=lambda n, bs, rows: {"rows": rows, "span": n},
        events=events,
    )
    monkeypatch.setattr(executor_module, "pg_manager", pg_manager)
    # The group is a stand-in, so the tuning collectives it would drive are
    # recorded rather than run; the tuning window itself stays real.
    monkeypatch.setattr(
        executor_module,
        "set_autotune_process_group",
        lambda group: events.append(("group", group)),
    )
    monkeypatch.setattr(
        executor_module,
        "load_autotune_cache",
        lambda path, group, owner: events.append(("load", path, group, owner)),
    )
    monkeypatch.setattr(
        executor_module,
        "save_autotune_cache",
        lambda path, group, owner: events.append(("save", path, group, owner)),
    )
    _tap(monkeypatch, executor_module, "autotune", lambda *args, **kwargs: None)
    ModelExecutor.autotune(executor)
    assert seen["input_ids"].tolist() == [2, 6, 10, 14]
    assert seen["positions"].tolist() == [20, 60, 100, 140]
    assert seen["rows"].tolist() == [2, 6, 10, 14] and seen["span"] == 16
    assert pg_manager.requests == [("gloo", (0, 1, 2, 3))]
    assert events == [
        ("load", None, group, 0),
        ("group", group),
        ("group", None),
        ("save", None, group, 0),
    ]


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_decode_only_attention_tunes_on_a_decode_step(
    monkeypatch, tmp_path, speculative, disabled, failure
):
    """Head TP serves decode rows only: no extend-shaped dummy is built; the
    traversal is one decode step at the largest batch, draft experts included."""
    events = []

    def forward(**kwargs):
        raise AssertionError("no extend forward on a decode-only layout")

    def warmup(*, batch_sizes, graph_phase):
        assert batch_sizes == (8,)
        assert graph_phase is False
        assert tuning_module.is_warming_up()
        assert tuning_module.is_autotuning() is (
            not disabled and tuning_module._autotuner is not None
        )
        events.append("decode")
        if failure:
            raise RuntimeError("decode warmup failed")

    def prefill_batch(num_tokens, batch_size):
        raise AssertionError("no extend dummy batch on a decode-only layout")

    def no_extend_inputs(num_tokens, bs, rows):
        raise AssertionError("no extend inputs on a decode-only layout")

    executor = _AutotuneExecutor(
        config=_AutotuneConfig(
            autotune_cache_key={},
            decode_only_attention=True,
            disable_autotune=disabled,
        ),
        forward=forward,
        input_buffers=_InputBuffersDouble(
            input_ids_buf=torch.ones(32),
            positions_buf=torch.arange(32),
            max_bs=8,
            max_num_tokens=32,
            on_fill_dummy=lambda **kwargs: events.append("scrub"),
        ),
        prefill_graph=_PrefillGraphDouble(make_dummy_batch=prefill_batch),
        forward_step=_ForwardStepDouble(
            warmup_decode_path=warmup, max_decode_bs=8, max_tokens_per_req=3
        ),
        drafter=object() if speculative else None,
        model_input_kwargs=no_extend_inputs,
        events=events,
    )
    monkeypatch.setattr(
        executor_module, "autotune_cache_path", lambda key: str(tmp_path / "cache.json")
    )
    max_tokens = []
    _tap(
        monkeypatch,
        executor_module,
        "set_autotune_max_num_tokens",
        lambda num_tokens: max_tokens.append(num_tokens),
    )
    _tap(
        monkeypatch,
        executor_module,
        "set_autotune_process_group",
        lambda group: events.append(("group", group)),
    )
    _tap(
        monkeypatch,
        executor_module,
        "load_autotune_cache",
        lambda *args: events.append("load"),
    )
    _tap(
        monkeypatch,
        executor_module,
        "save_autotune_cache",
        lambda *args: events.append("save"),
    )
    if failure:
        with pytest.raises(RuntimeError, match="decode warmup failed"):
            ModelExecutor.autotune(executor)
    else:
        ModelExecutor.autotune(executor)
    assert max_tokens == [8 * 3]
    assert events == [
        "load",
        ("group", None),
        "scrub",
        *([("draft_experts", 24)] if speculative else []),
        "decode",
        ("group", None),
        *(["save"] if not disabled and not failure else []),
    ]
    assert not tuning_module.is_warming_up()
    assert not tuning_module.is_autotuning()


@pytest.mark.parametrize("disabled", [False, True])
def test_pipeline_parallelism_does_not_run_unsupported_dummy_forward(
    disabled,
):
    def unexpected_forward(**kwargs):
        raise AssertionError("pipeline stages need transfers before a forward")

    def unexpected_scrub(**kwargs):
        raise AssertionError("pipeline stages must not prepare a dummy forward")

    executor = _AutotuneExecutor(
        config=_AutotuneConfig(pp_size=2, disable_autotune=disabled),
        forward=unexpected_forward,
        input_buffers=_InputBuffersDouble(
            input_ids_buf=torch.ones(32),
            positions_buf=torch.arange(32),
            max_bs=8,
            max_num_tokens=32,
            on_fill_dummy=unexpected_scrub,
        ),
        prefill_graph=_PrefillGraphDouble(make_dummy_batch=unexpected_forward),
        forward_step=_ForwardStepDouble(),
        drafter=None,
        model_input_kwargs=unexpected_forward,
        events=[],
    )
    ModelExecutor.autotune(executor)
    assert not tuning_module.is_warming_up()


# ---------------------------------------------------------------------------
# ModelExecutor.capture_graphs doubles
# ---------------------------------------------------------------------------


class _DecodeCaptureDouble:
    def __init__(self, *, disable, stream, on_capture):
        self.disable = disable
        self.stream = stream
        self._on_capture = on_capture

    def capture(self, *, entries, observer):
        self._on_capture()


class _PrefillCaptureDouble:
    def __init__(self, *, disable, on_capture):
        self.disable = disable
        self._on_capture = on_capture

    def capture(self, runner, *, entries, observer):
        self._on_capture(runner)


class _DrafterCaptureDouble:
    def __init__(self, on_capture):
        self._on_capture = on_capture

    def capture_prefill_graph(self, stream, observer):
        self._on_capture(stream)


class _ObserverDouble:
    def measure(self, name):
        return nullcontext()


class _CaptureExecutor:
    """The ModelExecutor state capture_graphs() reads, without the GPU boot."""

    def __init__(
        self, *, captures_drafter_prefill_graph, forward_step, prefill_graph, drafter
    ):
        self.captures_drafter_prefill_graph = captures_drafter_prefill_graph
        self.device = "cpu"
        self.forward_step = forward_step
        self.prefill_graph = prefill_graph
        self.drafter = drafter


@pytest.mark.parametrize("decode_disabled", [False, True])
@pytest.mark.parametrize("prefill_disabled", [False, True])
@pytest.mark.parametrize("has_drafter", [False, True])
def test_capture_lifecycle_preserves_main_order(
    decode_disabled, prefill_disabled, has_drafter
):
    events = []
    reset_workspace_pools()
    pool = workspace_pool("cpu")
    assert not pool._frozen

    def capture_decode():
        assert pool._frozen, "decode capture ran before the workspace freeze"
        events.append("decode")

    step = _DecodeCaptureDouble(
        disable=decode_disabled, stream=object(), on_capture=capture_decode
    )

    def capture_prefill(runner):
        assert runner is step
        assert pool._frozen, "prefill capture ran before the workspace freeze"
        events.append("prefill")

    def capture_draft(stream):
        assert stream is step.stream
        assert pool._frozen, "draft capture ran before the workspace freeze"
        events.append("draft")

    executor = _CaptureExecutor(
        captures_drafter_prefill_graph=not prefill_disabled and has_drafter,
        forward_step=step,
        prefill_graph=_PrefillCaptureDouble(
            disable=prefill_disabled, on_capture=capture_prefill
        ),
        drafter=(
            _DrafterCaptureDouble(on_capture=capture_draft) if has_drafter else None
        ),
    )
    ModelExecutor.capture_graphs(
        executor,
        entries=None,
        observer=_ObserverDouble(),
    )
    assert pool._frozen
    assert events == [
        *([] if decode_disabled else ["decode"]),
        *([] if prefill_disabled else ["prefill"]),
        *(["draft"] if not prefill_disabled and has_drafter else []),
    ]
    reset_workspace_pools()


# ---------------------------------------------------------------------------
# _autotune_cache_key
# ---------------------------------------------------------------------------


@dataclass
class _HfConfigDouble:
    architectures: list[str]


@dataclass
class _ModelConfigDouble:
    """The ModelConfig fields _autotune_cache_key reads.

    Stays a double: a real ModelConfig resolves the HF config of its model
    path (filesystem/network probing), so it cannot be built around a
    synthetic model in a unit test.
    """

    model_path: str
    revision: str | None
    hf_config: _HfConfigDouble
    quantization: str | None


def test_pipeline_stages_share_shape_keyed_cache_identity():
    # Real args: attn_tp_size=2 infers world_size=2 and resolves dense/moe/
    # linear-attention widths to the same 2 with moe ep/dp and pp at 1.
    args = ServerArgs(model="x", dtype="bfloat16", attn_tp_size=2)
    args.mapping.rank = 0
    model = _ModelConfigDouble(
        model_path="model",
        revision=None,
        hf_config=_HfConfigDouble(architectures=["Model"]),
        quantization=None,
    )
    cache_key = executor_module._autotune_cache_key
    full_model = cache_key(args, model)
    args.mapping.pp_size = 2
    assert cache_key(args, model) == full_model
    args.disaggregation_mode = "prefill"
    prefill = cache_key(args, model)
    args.disaggregation_mode = "decode"
    assert cache_key(args, model) != prefill
    args.disaggregation_mode = "null"
    args.mapping.moe.ep_size = 2
    assert cache_key(args, model) != full_model
    args.numerics = "rl-bitwise"
    assert cache_key(args, model) is None


# ---------------------------------------------------------------------------
# FlashInfer MXFP4 SiTU MoE doubles (registered on NVIDIA only)
# ---------------------------------------------------------------------------

_SITU_APPLY_NAME = "flashinfer_trtllm_mxfp4_situ_moe_apply"
_situ_apply = getattr(mxfp4_module, _SITU_APPLY_NAME, None)
requires_situ_apply = pytest.mark.skipif(
    _situ_apply is None,
    reason="FlashInfer MXFP4 SiTU kernels register on NVIDIA only",
)


def _pin_situ_selection(monkeypatch):
    """Route moe plan/apply selection to the real SiTU apply kernel."""
    selected = SelectedKernel(name=_SITU_APPLY_NAME, impl=_situ_apply)
    monkeypatch.setattr(moe_module, "select_kernel", lambda *args, **kwargs: selected)


class _SituLaunchDouble:
    """Stands in for the TRT-LLM MXFP4 SiTU CUDA launch; records each variant."""

    def __init__(self, calls, fail):
        self.calls = calls
        self.fail = fail
        self.prepared_x = None
        self.prepared_scales = None

    def __call__(
        self,
        *,
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
        assert x is self.prepared_x
        assert hidden_states_scale.data_ptr() == self.prepared_scales.data_ptr()
        mode = "kernel_routing" if topk_ids is None else "precomputed_topk"
        self.calls.append((mode, do_finalize, output))
        if self.fail:
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


class _Mxfp8QuantizeDouble:
    """Stands in for FlashInfer's mxfp8_quantize; keeps real tensors."""

    def __init__(self, launch):
        self.calls = 0
        self._launch = launch

    def __call__(self, x, is_sf_swizzled_layout, *, enable_pdl, alignment, backend):
        assert x.shape[-1] == alignment == 8
        assert torch.count_nonzero(x[:, 6:]) == 0
        self.calls += 1
        self._launch.prepared_x = x.clone()
        self._launch.prepared_scales = torch.ones(x.shape[0], 1, dtype=torch.uint8)
        return self._launch.prepared_x, self._launch.prepared_scales


def _situ_backend(monkeypatch, calls, fail, tuning):
    launch = _SituLaunchDouble(calls, fail)
    quantizer = _Mxfp8QuantizeDouble(launch)
    monkeypatch.setattr(mxfp4_module, "_call_mxfp4_situ_moe", launch)
    monkeypatch.setattr(mxfp4_module, "mxfp8_quantize", quantizer)
    monkeypatch.setattr(mxfp4_module, "is_autotuning", lambda: tuning)
    return quantizer


class _SituWeightsDouble:
    """The weight-module attributes the SiTU apply path reads."""

    def __init__(
        self,
        *,
        hidden_size_padded,
        hidden_size_original,
        top_k=2,
        num_experts=4,
        situ_output_buffer=None,
    ):
        self.hidden_size_padded = hidden_size_padded
        self.hidden_size_original = hidden_size_original
        self.w2_weight_scale = torch.empty(num_experts, hidden_size_padded)
        self.top_k = top_k
        self.num_experts = num_experts
        self._situ_output_buffer = situ_output_buffer


def _situ_plan(routing_mode):
    return moe_module.moe_plan(
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
        combine_order="rank",
        features=frozenset(),
        with_bias=False,
        process_group=None,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution=None,
    )


@requires_situ_apply
@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("do_finalize", [False, True])
@pytest.mark.parametrize("routing", ["kernel_routing", "precomputed_topk", "topk_only"])
def test_moe_profiles_on_scratch_and_runs_normal_apply_once(
    monkeypatch, tuning, do_finalize, routing
):
    calls = []
    quantizer = _situ_backend(monkeypatch, calls, False, tuning)
    _pin_situ_selection(monkeypatch)
    plan = _situ_plan("precomputed_topk" if routing == "topk_only" else None)
    assert plan["apply_kernel_name"] == _SITU_APPLY_NAME
    assert plan["support_routing"] and plan["supports_precomputed_topk"]
    assert plan["supports_deferred_finalize"]
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
    w = _SituWeightsDouble(
        hidden_size_padded=8, hidden_size_original=6, situ_output_buffer=out
    )
    result = moe_module.moe_apply(
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
    assert quantizer.calls == 1
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


@requires_situ_apply
@pytest.mark.parametrize("case", ["empty", "no_router", "missing_weights"])
def test_situ_profiling_validates_inputs_before_launch(monkeypatch, case):
    calls = []
    quantizer = _situ_backend(monkeypatch, calls, False, True)
    plan = {}
    context = nullcontext() if case == "empty" else pytest.raises(ValueError)
    with context:
        _situ_apply(
            plan=plan,
            x=torch.empty(0 if case == "empty" else 3, 6, dtype=torch.bfloat16),
            w=_SituWeightsDouble(hidden_size_padded=8, hidden_size_original=8),
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
    assert quantizer.calls == 0


@requires_situ_apply
def test_moe_profiling_failure_leaves_model_output_untouched(monkeypatch):
    calls = []
    _situ_backend(monkeypatch, calls, True, True)
    _pin_situ_selection(monkeypatch)
    plan = _situ_plan(None)
    out = torch.full((3, 8), -1, dtype=torch.bfloat16)
    w = _SituWeightsDouble(
        hidden_size_padded=8, hidden_size_original=8, situ_output_buffer=out
    )
    with pytest.raises(RuntimeError, match="profiling failed"):
        moe_module.moe_apply(
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


class _RoutedWeightsDouble:
    """The weight-module attributes the shared SiTU launcher reads."""

    def __init__(self):
        self.num_experts = 8
        self.num_local_experts = 4
        self.ep_rank = 1
        self.top_k = 2
        self.intermediate_size_per_partition = 128
        self.w13_weight = torch.empty(4, 16, 8)
        self.w2_weight = torch.empty(4, 8, 16)
        self.w13_weight_scale = torch.zeros(4, 16, dtype=torch.uint8)
        self.w2_weight_scale = torch.zeros(4, 8, dtype=torch.uint8)
        self.gemm1_alpha = torch.ones(4)
        self.gemm1_beta = torch.ones(4)
        self.routing_config = {
            "routed_scaling_factor": 1.25,
            "n_group": 2,
            "topk_group": 1,
        }


class _FiMoeDouble:
    """Records one TRT-LLM FP4 block-scale MoE launch and returns the preset."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@requires_situ_apply
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_situ_shared_launcher_preserves_ep_routing_and_return_contract(
    monkeypatch, precomputed, finalize
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
    fi = _FiMoeDouble(output if finalize else deferred)
    monkeypatch.setattr(mxfp4_module, "trtllm_fp4_block_scale_moe", fi)
    monkeypatch.setattr(mxfp4_module, "trtllm_fp4_block_scale_routed_moe", fi)
    monkeypatch.setattr(tuning_module, "_autotune_max_num_tokens", 9216)
    w = _RoutedWeightsDouble()
    logits = torch.randn(3, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 5], [2, 7], [3, 6]], dtype=torch.int64)
    result = mxfp4_module._call_mxfp4_situ_moe(
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
    assert len(fi.calls) == 1
    args = fi.calls[0]
    assert args["num_experts"] == 8
    assert args["local_num_experts"] == 4 and args["local_expert_offset"] == 4
    assert args["enable_pdl"] is True and args["do_finalize"] is finalize
    assert args["tune_max_num_tokens"] == 9216
    assert args["activation_type"] is mxfp4_module.ActivationType.Situ
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


# ---------------------------------------------------------------------------
# FlashInfer joint BF16 GEMM doubles
# ---------------------------------------------------------------------------


class _CudaTensorDouble:
    """CUDA-tensor metadata double; the support contract reads no tensor data."""

    def __init__(
        self,
        shape,
        *,
        dtype=torch.bfloat16,
        device="cuda:0",
        ndim=2,
        contiguous=True,
        data_ptr=32,
        is_cuda=True,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.ndim = ndim
        self.is_cuda = is_cuda
        self._contiguous = contiguous
        self._data_ptr = data_ptr

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._data_ptr


class _FiGemmDouble:
    """Stands in for flashinfer.gemm.gemm_base; records the joint dispatch."""

    DEFAULT_WORKSPACE_SIZE = 4096

    def __init__(self, dispatch):
        self.workspace = object()
        self.cache_buf_requests = []
        self.dispatches = []
        self._dispatch = dispatch

    def _get_cache_buf(self, name, size, device):
        self.cache_buf_requests.append((name, size, device))
        return self.workspace

    def bf16_gemm_sm100(self, **kwargs):
        self.dispatches.append(kwargs)
        self._dispatch(kwargs)


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
def test_bf16_joint_adapter_uses_one_fi_dispatch(
    monkeypatch, pdl, provided_out, n, k, expected
):
    def dispatch(kwargs):
        kwargs["out"].copy_(kwargs["a"] @ kwargs["b"])

    fi_gemm = _FiGemmDouble(dispatch)
    monkeypatch.setattr(fi_gemm_module, "_fi_gemm", fi_gemm)
    monkeypatch.setattr(
        fi_gemm_module, "flashinfer_joint_bf16_supported", lambda *args: True
    )
    monkeypatch.setattr(fi_gemm_module, "pdl_enabled", lambda: pdl)
    x = torch.randn(3, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    out = torch.empty(3, n, dtype=torch.bfloat16) if provided_out else None
    result = fi_gemm_module.flashinfer_bf16_gemm(x, w, out)
    torch.testing.assert_close(result, x @ w.T)
    assert not provided_out or result is out
    assert len(fi_gemm.dispatches) == 1
    call = fi_gemm.dispatches[0]
    assert call["runner_names"] == expected
    assert call["bias"] is None  # No artificial zero bias excluding direct.
    assert call["workspace_buffer"] is fi_gemm.workspace
    assert call["pdl"] is pdl


@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_bf16_discovery_uses_one_native_joint_call(monkeypatch, tuning, supported):
    probe = []
    monkeypatch.setattr(fi_gemm_module, "is_autotuning", lambda: tuning)
    monkeypatch.setattr(
        fi_gemm_module, "flashinfer_joint_bf16_supported", lambda *args: supported
    )
    monkeypatch.setattr(
        fi_gemm_module, "flashinfer_bf16_gemm", lambda *args: probe.append(args)
    )
    x = torch.ones(128, 128, dtype=torch.bfloat16)
    w = torch.ones(32, 128, dtype=torch.bfloat16)
    fi_gemm_module.autotune_bf16_gemm(x, w)
    assert torch.all(x == 1) and torch.all(w == 1)
    if tuning and supported:
        assert len(probe) == 1
        sample, weight, out = probe[0]
        assert sample.shape == (32, 128) and sample.data_ptr() != x.data_ptr()
        assert weight is w and out is None
    else:
        assert probe == []


@pytest.mark.parametrize(
    "invalid", [None, "cpu", "dtype", "strided", "unaligned", "n", "k", "out"]
)
def test_joint_bf16_support_contract(monkeypatch, invalid):
    x = _CudaTensorDouble(
        (128, 128),
        is_cuda=invalid != "cpu",
        dtype=torch.float32 if invalid == "dtype" else torch.bfloat16,
        contiguous=invalid != "strided",
        data_ptr=4 if invalid == "unaligned" else 32,
    )
    w = _CudaTensorDouble(
        (0 if invalid == "n" else 32, 127 if invalid == "k" else 128),
        data_ptr=64,
    )
    out = (
        _CudaTensorDouble((128, 32), dtype=torch.float32) if invalid == "out" else None
    )
    monkeypatch.setattr(fi_gemm_module, "_fi_gemm", object())
    monkeypatch.setattr(fi_gemm_module, "has_flashinfer_cute_dsl_bf16", lambda: True)
    assert fi_gemm_module.flashinfer_joint_bf16_supported(x, w, out) == (
        invalid is None
    )
    if invalid is None:
        # Discovery may use large M and accepts a valid caller-provided out.
        assert fi_gemm_module.flashinfer_joint_bf16_supported(
            x, w, _CudaTensorDouble((128, 32), data_ptr=64)
        )


@pytest.mark.parametrize("m", [33, 48, 64, 128])
def test_joint_bf16_adapter_declines_large_m(monkeypatch, m):
    monkeypatch.setattr(
        fi_gemm_module, "flashinfer_joint_bf16_supported", lambda *args: True
    )
    with pytest.raises(ValueError, match="Unsupported input"):
        fi_gemm_module.flashinfer_bf16_gemm(
            torch.empty(m, 128), torch.empty(32, 128), None
        )


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
    assert fi_gemm_module._bf16_gemm_runner_names(k) == expected


def test_flashinfer_probe_reads_the_declared_backends() -> None:
    """0.6.18 declares the backend; earlier wheels name every other one."""

    def upstreamed(backend: Literal["cudnn", "cute-dsl"] = "cudnn") -> None: ...

    def earlier(backend: Literal["cudnn", "tgv", "tinygemm"] = "cudnn") -> None: ...

    declares = fi_gemm_module._declares_cute_dsl_backend
    assert declares(upstreamed)
    assert not declares(earlier)
    assert not declares(lambda: None)


@pytest.mark.parametrize(
    "name",
    [
        "kimi3_qkvfab_projection",
        "kimi3_latent_projection",
        "kimi3_mla_qkv_gate_projection",
    ],
)
def test_projection_discovery_precedes_auto_dispatch(monkeypatch, name):
    """An auto-dispatched projection exposes its own N/K to BF16 discovery
    exactly once -- ahead of the shape-based dispatch shortcuts -- and the
    fallback still computes the plain projection."""
    events = []
    monkeypatch.setattr(
        gemm_module,
        "autotune_bf16_gemm",
        lambda x, weight: events.append(("discover", x, weight)),
    )
    _tap(
        monkeypatch,
        gemm_module,
        "_use_mv",
        lambda *args: events.append(("gemv",)),
    )
    # mv shares the patched module and runs its own discovery; keep only the
    # projection's.
    monkeypatch.setattr(gemm_module, "mv", lambda x, weight, out=None: x @ weight.T)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randn(24, 16, dtype=torch.bfloat16)
    if name == "kimi3_mla_qkv_gate_projection":
        result = gemm_module.kimi3_mla_qkv_gate_projection(
            x, weight, 16, solution="auto"
        )
        torch.testing.assert_close(result.qkv, x @ weight[:16].T)
        torch.testing.assert_close(result.gate, x @ weight[16:].T)
        assert [event for event, *_ in events] == ["discover"]
    else:
        project = getattr(gemm_module, name)
        result = project(x, weight, solution="auto")
        torch.testing.assert_close(result, x @ weight.T)
        assert [event for event, *_ in events] == ["discover", "gemv"]
    assert events[0][1] is x and events[0][2] is weight


# ---------------------------------------------------------------------------
# Decode GEMV dispatch doubles
# ---------------------------------------------------------------------------


class _PlatformDouble:
    def __init__(self, *, vendor="nvidia", is_cdna4=False, is_cdna5=False):
        self.vendor = vendor
        self.is_cdna4 = is_cdna4
        self.is_cdna5 = is_cdna5


class _GetAccess:
    """Injects one instance through a class-level ``get()`` (registry/platform)."""

    def __init__(self, instance):
        self._instance = instance

    def get(self):
        return self._instance


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
    monkeypatch, fi, m, cdna5, k, registered, expected
):
    fallback = object()
    selected = object() if registered else fallback
    select_calls = []
    monkeypatch.setattr(
        gemm_module, "flashinfer_joint_bf16_supported", lambda *args: fi
    )
    monkeypatch.setattr(gemm_module, "is_serving", lambda: False)
    monkeypatch.setattr(
        gemm_module, "current_platform", lambda: _PlatformDouble(is_cdna5=cdna5)
    )
    monkeypatch.setattr(
        gemm_module,
        "_select_mv",
        lambda *args: select_calls.append(args) or selected,
    )
    monkeypatch.setattr(gemm_module, "torch_decode_gemv", fallback)
    x = _CudaTensorDouble((m, k))
    weight = _CudaTensorDouble((7168, k))
    assert gemm_module._use_mv(x, weight) is expected
    if fi and m <= 32:
        assert select_calls == []


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
def test_decode_gemv_selection_obeys_shape_traits(
    monkeypatch, m, n, k, on_cuda, expected
):
    def specialized_gemv(x, weight, out=None):
        raise AssertionError("selection only; never launched")

    # An isolated real registry: registration, format filtering and lookup
    # are ordinary Python, so the production path runs as-is. The spec carries
    # the real BF16 format signature; an empty one would be filtered out.
    registry = KernelRegistry()
    registry.register(
        KernelSpec(
            name="specialized",
            family="gemm",
            mode="mv",
            format_signatures=frozenset(
                {gemm_module._MV_SIGNATURES[(torch.bfloat16, torch.bfloat16)]}
            ),
            traits={
                "m": frozenset({1}),
                "n_align": frozenset({16}),
                "k_min": frozenset({256}),
            },
        ),
        specialized_gemv,
    )
    monkeypatch.setattr(gemm_module, "KernelRegistry", _GetAccess(registry))
    gemm_module._select_mv.cache_clear()
    fallback = gemm_module.torch_decode_gemv
    selected = gemm_module._select_mv(
        m, n, k, on_cuda, torch.bfloat16, torch.bfloat16, False
    )
    # A full trait match selects the registered kernel. Anything less keeps
    # the portable fallback -- including the non-CUDA caller, whose early
    # return wins despite the registered match (m=1, n=32, k=256 matches).
    assert selected is (specialized_gemv if expected else fallback)
    # A dtype pair with no registry signature keeps the fallback as well.
    assert (
        gemm_module._select_mv(m, n, k, on_cuda, torch.float16, torch.float16, False)
        is fallback
    )
    gemm_module._select_mv.cache_clear()


@pytest.mark.parametrize(
    "m,k,routed",
    [
        (1, 7168, True),
        (1, 2048, True),
        (1, 16384, True),
        (1, 65537, False),
        (8, 7168, True),
        (8, 8192, False),
        (12, 4096, True),
        (16, 16384, False),
    ],
)
def test_fp32_rowcta_routing_bounds_k(m, k, routed):
    # A direct call takes any width, but the registry sends a second 64K block
    # for one row, and more than 15 blocks of 512 for more rows, to Torch.
    assert gemm_triton_module._fp32_rowcta_k_fits(m, 256, k) is routed


@pytest.mark.parametrize("serving", [False, True])
def test_serving_sends_joint_bf16_rows_to_a_gemm_that_never_compiles(
    monkeypatch, serving
):
    """Once serving, FI's rows skip FI and the registry's possibly cold Triton GEMV."""
    joint_calls, fallback_calls, select_calls = [], [], []
    monkeypatch.setattr(
        gemm_module, "flashinfer_joint_bf16_supported", lambda *args: True
    )
    monkeypatch.setattr(
        gemm_module,
        "flashinfer_bf16_gemm",
        lambda *args: joint_calls.append(args) or torch.mm(args[0], args[1].T),
    )
    monkeypatch.setattr(gemm_module, "is_serving", lambda: serving)
    monkeypatch.setattr(
        gemm_module,
        "torch_decode_gemv",
        lambda *args: fallback_calls.append(args) or torch.mm(args[0], args[1].T),
    )
    monkeypatch.setattr(
        gemm_module, "_select_mv", lambda *args: select_calls.append(args)
    )
    x = torch.ones(1, 128, dtype=torch.bfloat16)
    w = torch.ones(32, 128, dtype=torch.bfloat16)
    assert gemm_module._use_mv(x, w) is not serving
    result = gemm_module.mv(x, w)
    torch.testing.assert_close(result, x @ w.T)
    used, idle = (
        (fallback_calls, joint_calls) if serving else (joint_calls, fallback_calls)
    )
    (call,) = used
    assert call[0] is x and call[1] is w and call[2] is None
    assert idle == []
    assert select_calls == []


# ---------------------------------------------------------------------------
# Skinny add3 (K3 latent-up) doubles
# ---------------------------------------------------------------------------


class _SkinnyGemmDouble:
    """Stands in for the cute-dsl skinny GEMM; records each launch."""

    def __init__(self, calls, fail):
        self.calls = calls
        self.fail = fail

    def supports(self, config, m, n, k):
        return True

    def __call__(self, x, weight, config, *, residual, residual2, out):
        self.calls.append((x, weight, config, residual, residual2, out))
        if self.fail:
            raise RuntimeError("compile failed")
        return torch.addmm(residual2, x, weight.t()) + residual


@pytest.mark.parametrize(
    "capturing,warmed", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("failure", [False, True])
def test_skinny_add3_preserves_capture_trust(monkeypatch, capturing, warmed, failure):
    key = (0, 1, 4, 8)
    known = {key} if warmed else set()
    kernel_calls, fallback_calls = [], []
    monkeypatch.setattr(gemm_module, "_SKINNY_ADD3_CONFIGS", {(1, 4, 8): (64, 4, 2)})
    monkeypatch.setattr(gemm_module, "_skinny_add3_arch_supported", lambda dev: True)
    monkeypatch.setattr(gemm_module, "_skinny_add3_warmed", known)
    monkeypatch.setattr(
        gemm_module,
        "shape_dynamic_skinny_gemm",
        _SkinnyGemmDouble(kernel_calls, failure),
    )
    monkeypatch.setattr(
        gemm_module,
        "_skinny_add3_fallback",
        lambda *args: fallback_calls.append(args) or "fallback",
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    rng = torch.Generator().manual_seed(20261013)
    x = torch.randn(1, 8, dtype=torch.bfloat16, generator=rng)
    w = torch.randn(4, 8, dtype=torch.bfloat16, generator=rng)
    a = torch.randn(1, 4, dtype=torch.bfloat16, generator=rng)
    c = torch.randn(1, 4, dtype=torch.bfloat16, generator=rng)
    if capturing and not warmed:
        assert gemm_module._skinny_gemv_add3(x, w, a, c, None) == "fallback"
        assert kernel_calls == []
        assert not known
    elif failure:
        with pytest.raises(RuntimeError, match="compile failed"):
            gemm_module._skinny_gemv_add3(x, w, a, c, None)
        assert known == ({key} if warmed else set())
    else:
        result = gemm_module._skinny_gemv_add3(x, w, a, c, None)
        torch.testing.assert_close(result, torch.addmm(c, x, w.t()) + a)
        assert len(kernel_calls) == 1
        assert fallback_calls == []
        assert key in known


@pytest.mark.parametrize(
    "capability,expected", [((10, 0), False), ((10, 3), True), ((10, 7), False)]
)
def test_skinny_add3_uses_only_measured_architecture(monkeypatch, capability, expected):
    monkeypatch.setattr(
        gemm_module,
        "Platform",
        _GetAccess(_PlatformDouble(vendor="nvidia")),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    gemm_module._skinny_add3_arch_supported.cache_clear()
    assert gemm_module._skinny_add3_arch_supported(0) is expected
    gemm_module._skinny_add3_arch_supported.cache_clear()


# ---------------------------------------------------------------------------
# mm dispatch doubles
# ---------------------------------------------------------------------------


class _SelectKernelDouble:
    """Pins selection to one kernel; records the resolved override."""

    def __init__(self, kernel):
        self._kernel = kernel
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._kernel


def _generic_mm(calls):
    def generic(a, b, a_scales, b_scales, out_dtype, *, alpha, block_size, out):
        calls.append((a, b, out))
        return out.copy_(torch.mm(a, b.T).to(out_dtype))

    generic.name = "test_mm"
    return generic


@pytest.mark.parametrize("rows", [4, 32, 33, 64, 128])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("override", [None, "explicit", "context", "environment"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_mm_joint_dispatch_respects_overrides_and_contract(
    monkeypatch, rows, bias, override, out_dtype
):
    probe_calls, joint_calls, generic_calls = [], [], []
    monkeypatch.setattr(
        gemm_module,
        "autotune_bf16_gemm",
        lambda *args: probe_calls.append(args),
    )
    monkeypatch.setattr(
        gemm_module,
        "flashinfer_bf16_gemm",
        lambda a, b, out: joint_calls.append((a, b, out)) or torch.mm(a, b.T, out=out),
    )
    monkeypatch.setattr(
        gemm_module, "flashinfer_joint_bf16_supported", lambda *args: True
    )
    monkeypatch.setattr(gemm_module, "is_serving", lambda: False)
    monkeypatch.setattr(gemm_module, "pdl_enabled", lambda: False)
    select = _SelectKernelDouble(_generic_mm(generic_calls))
    monkeypatch.setattr(gemm_module, "select_kernel", select)
    if override == "environment":
        monkeypatch.setenv("TOKENSPEED_KERNEL_OVERRIDE_GEMM_MM", "forced")
    x = torch.randn(rows, 4, dtype=torch.bfloat16)
    w = torch.randn(8, 4, dtype=torch.bfloat16)
    out = torch.empty(rows, 8, dtype=out_dtype)
    b = torch.ones(8, dtype=torch.bfloat16) if bias else None
    context = (
        kernel_override("gemm", "mm", "forced")
        if override == "context"
        else nullcontext()
    )
    with context:
        got = gemm_module.mm(
            x,
            w,
            bias=b,
            out=out,
            out_dtype=out_dtype,
            override="forced" if override == "explicit" else None,
        )
    assert got is out
    torch.testing.assert_close(
        got, (x @ w.T).to(out_dtype) + (b if b is not None else 0)
    )
    # Rank-local bias must not make ranks skip collective tactic profiling.
    assert len(probe_calls) == int(override is None and out_dtype == torch.bfloat16)
    assert len(joint_calls) == int(
        override is None and not bias and out_dtype == torch.bfloat16 and rows <= 32
    )
    if override is None:
        return
    # The resolved override reaches selection; the joint shortcut stands down.
    assert len(generic_calls) == 1
    assert select.calls[0]["override"] == "forced"


@pytest.mark.parametrize("serving", [False, True])
def test_mm_runs_the_joint_bf16_gemm_only_before_serving(monkeypatch, serving):
    joint_calls = []
    monkeypatch.setattr(gemm_module, "autotune_bf16_gemm", lambda *args: None)
    monkeypatch.setattr(
        gemm_module,
        "flashinfer_bf16_gemm",
        lambda a, b, out: joint_calls.append((a, b, out)) or torch.mm(a, b.T, out=out),
    )
    monkeypatch.setattr(
        gemm_module, "flashinfer_joint_bf16_supported", lambda *args: True
    )
    monkeypatch.setattr(gemm_module, "is_serving", lambda: serving)
    monkeypatch.setattr(gemm_module, "pdl_enabled", lambda: False)
    generic_calls = []
    monkeypatch.setattr(
        gemm_module, "select_kernel", _SelectKernelDouble(_generic_mm(generic_calls))
    )
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    w = torch.randn(8, 4, dtype=torch.bfloat16)
    out = torch.empty(3, 8, dtype=torch.bfloat16)
    assert gemm_module.mm(x, w, out=out) is out
    torch.testing.assert_close(out, x @ w.T)
    assert len(joint_calls) == int(not serving)
    assert len(generic_calls) == int(serving)


# ---------------------------------------------------------------------------
# Dense unquantized apply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["environment", "context"])
@pytest.mark.parametrize("numerics", ["auto"])
def test_dense_mm_override_precedes_decode_shortcut(monkeypatch, kind, numerics):
    monkeypatch.setitem(env_module.global_server_args_dict, "numerics", numerics)
    if kind == "environment":
        monkeypatch.setenv("TOKENSPEED_KERNEL_OVERRIDE_GEMM_MM", "forced")
    mm_result = object()
    mm_calls = []

    def kernel_mm(*args, **kwargs):
        mm_calls.append((args, kwargs))
        return mm_result

    monkeypatch.setattr(unquant_module, "kernel_mm", kernel_mm)
    method = UnquantizedLinearMethod()
    layer = torch.nn.Linear(4, 8, bias=False)
    x = torch.randn(3, 4)
    context = (
        kernel_override("gemm", "mm", "forced") if kind == "context" else nullcontext()
    )
    with context:
        assert method.apply(layer, x) is mm_result
    ((args, kwargs),) = mm_calls
    assert args[0] is x and args[1] is layer.weight
    expected = {"bias": None}
    assert kwargs == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
