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

"""Dots3 startup hooks; TP transport is stubbed, CUDA sync-debug is real."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

import tokenspeed.runtime.layers.logits_processor as logits_module
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.models.dots3_note import Dot3NoteForCausalLM
from tokenspeed.runtime.models.dots3_note_nextn import Dot3NoteForCausalLMNextN
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeLimits,
    setup_dp_sampling,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class _Head(nn.Linear):
    def __init__(self, *, device: str):
        super().__init__(2, 4096, bias=False, dtype=torch.bfloat16, device=device)
        self.quant_method = None


def _logits_model(*, device: str, do_argmax: bool):
    model_cls = Dot3NoteForCausalLMNextN if do_argmax else Dot3NoteForCausalLM
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.lm_head = _Head(device=device)
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 2, dtype=torch.bfloat16, device=device)
    model.logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=8192),
        do_argmax=do_argmax,
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        dp_lm_head_tp=False,
    )
    return model


def _runner(model, *, is_draft_worker):
    runner = ModelRunner.__new__(ModelRunner)
    runner.is_draft_worker = is_draft_worker
    runner.model = model
    runner.model_config = SimpleNamespace(
        dtype=torch.bfloat16, is_multimodal_active=False
    )
    return runner


@pytest.fixture
def communication(monkeypatch):
    runtime = SimpleNamespace(
        available=True,
        argmax=object(),
        gather=object(),
        serving=False,
        calls=[],
    )

    def rendezvous(name, result):
        assert not runtime.serving, f"serving attempted {name} initialization"
        runtime.calls.append(name)
        return result

    def no_forward(*args, **kwargs):
        pytest.fail("communication startup must not run a dummy model forward")

    monkeypatch.setattr(Dot3NoteForCausalLM, "forward", no_forward)
    monkeypatch.setattr(Dot3NoteForCausalLMNextN, "forward", no_forward)
    monkeypatch.setattr(LogitsProcessor, "_LOGITS_AG_STATES", {})
    monkeypatch.setattr(LogitsProcessor, "_LOGITS_DIST_ARGMAX_STATES", {})
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    monkeypatch.setattr(
        logits_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True, is_amd=False),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(logits_module.pg_manager, "get_process_group", lambda *a: None)
    # Leave _agree_across_tp intact: CUDA tensor construction and .item() must
    # finish before the serving guard, even when the fast path is unavailable.
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda *a, **k: rendezvous("agree", None)
    )
    monkeypatch.setattr(
        logits_module, "dist_argmax_available", lambda: runtime.available
    )
    monkeypatch.setattr(
        logits_module,
        "try_create_dist_argmax_state",
        lambda **k: rendezvous("argmax", runtime.argmax),
    )
    monkeypatch.setattr(
        logits_module, "create_state", lambda **k: rendezvous("gather", runtime.gather)
    )
    monkeypatch.setattr(
        logits_module,
        "all_gather_inner",
        lambda state, logits, **k: logits.repeat(1, 2),
    )
    monkeypatch.setattr(
        logits_module,
        "all_gather_single",
        lambda out, logits, group: out.copy_(logits.repeat(2, 1)),
    )
    monkeypatch.setattr(
        logits_module,
        "distributed_argmax",
        lambda state, logits: (None, logits.argmax(-1)),
    )
    monkeypatch.setattr(
        logits_module, "sampling_argmax", lambda logits: logits.argmax(-1)
    )
    return runtime


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("outcome", ["ready", "unavailable", "failed", "plain_gather"])
def test_graph_disabled_prefill_startup_never_rendezvouses_in_serving(
    monkeypatch, communication, device, outcome
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for the sync-debug regression")
    from tokenspeed.runtime.execution import factory
    from tokenspeed.runtime.execution.device import build_device_side
    from tokenspeed.runtime.layers.attention import registry

    if outcome == "unavailable":
        communication.available = False
    if outcome in ("failed", "plain_gather"):
        communication.argmax = None
    if outcome == "plain_gather":
        communication.gather = None
    # Exercise both sides of the row-capacity fallback without large tensors.
    monkeypatch.setattr(LogitsProcessor, "_LOGITS_DIST_ARGMAX_MAX_TOKENS", 2)
    target_model = _logits_model(device=device, do_argmax=False)
    draft_model = _logits_model(device=device, do_argmax=True)
    target = _runner(target_model, is_draft_worker=False)
    draft = _runner(draft_model, is_draft_worker=True)
    target.mapping = SimpleNamespace(has_pp=False)
    embed = draft_model.model.embed_tokens.weight

    def create_runners(*args):
        # Production factory shares the loaded head before returning either runner.
        factory._share_target_embed_and_head(target, draft_model)
        assert draft_model.lm_head.weight is target_model.lm_head.weight
        assert draft_model.model.embed_tokens.weight is embed
        assert communication.calls == []
        return target, draft

    monkeypatch.setattr(factory, "create_model_runner", create_runners)
    monkeypatch.setattr(ModelRunner, "prepare_multimodal_runtime", lambda self: None)

    class CachePlanningReached(Exception):
        pass

    def cache_planning(*args, **kwargs):
        for model in (target_model, draft_model):
            processor = model.logits_processor
            assert processor._all_gather_state is communication.gather
            assert (
                processor._dist_argmax_state
                is not processor._LOGITS_DIST_ARGMAX_UNINITIALIZED
            )
        raise CachePlanningReached

    monkeypatch.setattr(registry, "create_attn_components", cache_planning)
    with pytest.raises(CachePlanningReached):
        build_device_side(
            server_args=SimpleNamespace(
                disaggregation_mode="prefill",
                enforce_eager=True,
                disable_prefill_graph=True,
                disable_cudagraph_memory_reserve=False,
                attention_backend=None,
                drafter_attention_backend=None,
                chunked_prefill_size=16,
                enable_memory_saver=False,
            ),
            model_config=target.model_config,
            draft_model_config=draft.model_config,
            gpu_id=0,
            global_rank=0,
            attn_tp_rank=0,
            min_per_gpu_mem=0,
            overlap_schedule_depth=1,
            decode_input_tokens=1,
            max_batch_size=1,
        )

    assert communication.calls.count("gather") == 1
    assert communication.calls.count("argmax") == (outcome != "unavailable")
    assert communication.calls.count("agree") == (1 if outcome == "unavailable" else 2)

    # The builder hook precedes DP sampling setup. Prefill still uses the
    # ordinary gather even after decode's batch-DP layout is configured.
    monkeypatch.setattr(logits_module, "LogitsLayoutExecutor", lambda **k: object())
    runtime = setup_dp_sampling(
        model=target_model,
        sampling_backend=SimpleNamespace(
            _SUPPORTS_DP_VERIFY=True, configure_dp_sampling=Mock()
        ),
        requested=True,
        drafter_available=True,
        limits=DpSamplingRuntimeLimits(
            runtime_vocab_size=8192,
            max_num_seqs=4,
            data_parallel_size=1,
            num_tokens_per_req=2,
            configured_min_bs=2,
            device=device,
        ),
    )
    assert runtime.enabled
    metadata = {
        rows: LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            gather_ids=torch.arange(rows, device=device),
            query_shard=None,
        )
        for rows in (1, 3)
    }
    hidden = torch.ones((3, 2), dtype=torch.bfloat16, device=device)
    communication.serving = True
    old_sync_debug = torch.cuda.get_sync_debug_mode() if device == "cuda" else None
    try:
        if device == "cuda":
            torch.cuda.set_sync_debug_mode("error")
        # Repeated preparation must also reuse cached negative verdicts.
        target.prepare_communication_runtime(16)
        draft.prepare_communication_runtime(16)
        outputs = [
            model.logits_processor(None, hidden[:rows], model.lm_head, metadata[rows])
            for model in (target_model, draft_model)
            for rows in (1, 3)
        ]
    finally:
        if device == "cuda":
            torch.cuda.set_sync_debug_mode(old_sync_debug)
    for model, model_outputs in zip(
        (target_model, draft_model), (outputs[:2], outputs[2:])
    ):
        for rows, output in zip((1, 3), model_outputs):
            local_logits = hidden[:rows] @ model.lm_head.weight.T
            sharded = model is draft_model and outcome == "ready" and rows == 1
            expected = local_logits if sharded else local_logits.repeat(1, 2)
            torch.testing.assert_close(
                output.next_token_logits, expected, rtol=0, atol=0
            )
            if model is draft_model:
                torch.testing.assert_close(
                    output.next_token_ids, local_logits.argmax(-1), rtol=0, atol=0
                )


@pytest.mark.parametrize(
    "tp_size,skip_gather,do_argmax,dtype,vocab_size,expected",
    [
        (1, False, True, torch.bfloat16, 4096, []),
        (2, True, True, torch.bfloat16, 8192, []),
        (2, False, False, torch.bfloat16, 8192, ["gather"]),
        (2, False, True, torch.bfloat16, 8192, ["agree", "argmax", "agree", "gather"]),
        (2, False, True, torch.float16, 8192, ["agree", "argmax", "agree"]),
        (2, False, True, torch.float32, 8192, ["agree", "argmax", "agree"]),
        (2, False, True, torch.bfloat16, 8191, ["gather"]),
        (33, False, True, torch.bfloat16, 4096 * 33, ["gather"]),
    ],
)
def test_startup_respects_tp_vocab_and_dense_logits_dtype(
    communication, tp_size, skip_gather, do_argmax, dtype, vocab_size, expected
):
    model = _logits_model(device="cpu", do_argmax=do_argmax)
    model.logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=vocab_size),
        tp_rank=0,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
        dp_lm_head_tp=False,
        skip_all_gather=skip_gather,
        do_argmax=do_argmax,
    )
    model.lm_head.to(dtype=dtype)
    for _ in range(2):
        assert model.prepare_communication_runtime(16) == bool(expected)
    assert communication.calls == expected


@pytest.mark.parametrize("gate", ["deterministic", "dp_sampling"])
def test_startup_preserves_collective_gates(monkeypatch, communication, gate):
    model = _logits_model(device="cpu", do_argmax=True)
    if gate == "deterministic":
        monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    else:
        model.logits_processor.dp_sampling_enabled = True
    _runner(model, is_draft_worker=False).prepare_communication_runtime(16)
    assert communication.calls == ([] if gate == "deterministic" else ["gather"])


@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantized_head_prepares_gather_for_activation_dtype(
    communication, hidden_dtype
):
    model = _logits_model(device="cpu", do_argmax=False)
    model.lm_head.weight = nn.Parameter(
        torch.ones((4096, 2), dtype=torch.uint8), requires_grad=False
    )
    model.lm_head.quant_method = SimpleNamespace(
        apply=lambda head, hidden, bias: hidden @ head.weight.to(hidden.dtype).T
    )
    model.model.embed_tokens.to(dtype=hidden_dtype)
    model.prepare_communication_runtime(16)
    communication.serving = True
    hidden = torch.ones((1, 2), dtype=hidden_dtype)
    logits = model.logits_processor._get_logits(
        hidden,
        model.lm_head,
        LogitsMetadata(ForwardMode.EXTEND, query_shard=None),
        require_full_vocab=False,
    )
    assert logits.dtype == hidden_dtype
    assert logits.shape == (1, 8192)
    assert communication.calls == (["gather"] if hidden_dtype == torch.bfloat16 else [])


def test_startup_keeps_softcap_and_large_batch_gather_fallback(
    monkeypatch, communication
):
    model = _logits_model(device="cpu", do_argmax=True)
    processor = model.logits_processor
    processor.final_logit_softcapping = 1.0
    monkeypatch.setattr(LogitsProcessor, "_LOGITS_AG_MAX_TOKENS", 2)
    monkeypatch.setattr(
        logits_module,
        "fused_softcap_generic",
        lambda logits, cap: logits.copy_(torch.tanh(logits / cap) * cap),
    )
    gather = Mock(wraps=logits_module.all_gather_single)
    monkeypatch.setattr(logits_module, "all_gather_single", gather)
    argmax = Mock(side_effect=AssertionError("softcapping requires full logits"))
    monkeypatch.setattr(logits_module, "distributed_argmax", argmax)
    assert model.prepare_communication_runtime(16)
    communication.serving = True
    hidden = torch.ones((3, 2), dtype=torch.bfloat16)
    output = processor(
        None,
        hidden,
        model.lm_head,
        LogitsMetadata(
            ForwardMode.EXTEND, gather_ids=torch.arange(3), query_shard=None
        ),
    )
    expected = torch.tanh(hidden @ model.lm_head.weight.T).repeat(1, 2)
    torch.testing.assert_close(output.next_token_logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.next_token_ids, expected.argmax(-1), rtol=0, atol=0
    )
    gather.assert_called_once()
    argmax.assert_not_called()


def test_runner_does_not_scan_logits_or_nested_model_hooks(communication):
    head = _logits_model(device="cpu", do_argmax=True)
    root = nn.ModuleList([head])
    root.lm_head = head.lm_head
    root.logits_processor = head.logits_processor
    assert not _runner(root, is_draft_worker=True).prepare_communication_runtime(16)
    assert communication.calls == []
    assert (
        head.logits_processor._dist_argmax_state
        is head.logits_processor._LOGITS_DIST_ARGMAX_UNINITIALIZED
    )
