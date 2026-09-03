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

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.model_executor import ModelExecutor


class _RuntimeStates:
    def __init__(self):
        self.valid_cache_lengths = torch.arange(20, dtype=torch.int32)

    def reset_states(self, req_pool_indices, prefix_lens):
        self.valid_cache_lengths[req_pool_indices] = prefix_lens


class _ExecutionStream:
    def wait_stream(self, _):
        return None


def test_mixed_batch_resets_only_prefill_lengths(monkeypatch):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cpu"
    executor.device_module = torch.cuda
    executor.execution_stream = _ExecutionStream()
    executor.runtime_states = _RuntimeStates()

    forward_op = SimpleNamespace(
        request_pool_indices=[2, 3, 4],
        extend_prefix_lens=[10],
        num_extends=lambda: 1,
    )

    torch_tensor = torch.tensor

    def tensor_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor_without_pinning)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    executor._reset_valid_cache_length(forward_op)

    assert executor.runtime_states.valid_cache_lengths[2].item() == 10
    assert executor.runtime_states.valid_cache_lengths[3].item() == 3
    assert executor.runtime_states.valid_cache_lengths[4].item() == 4


def test_remote_prefill_seeds_the_complete_prompt_length(monkeypatch):
    """A PD decode destination never runs the prompt, so no forward of its own
    can establish these lengths: they come from the prefill node's complete
    prompt, not from the local extend prefix."""
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cpu"
    executor.device_module = torch.cuda
    executor.execution_stream = _ExecutionStream()
    executor.runtime_states = _RuntimeStates()

    forward_op = SimpleNamespace(
        request_pool_indices=[7, 11, 13],
        prefill_lengths=[15, 17, 19],
        extend_prefix_lens=[0, 0],
        num_extends=lambda: 2,
    )

    torch_tensor = torch.tensor

    def tensor_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor_without_pinning)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    executor.reset_remote_prefill_cache_lengths(forward_op)

    assert executor.runtime_states.valid_cache_lengths[7].item() == 15
    assert executor.runtime_states.valid_cache_lengths[11].item() == 17
    # Only the extend rows are seeded; the third row is not part of this op.
    assert executor.runtime_states.valid_cache_lengths[13].item() == 13


def test_draft_final_step_follows_the_complete_drafter_run():
    events = []

    class _Drafter:
        supports_pd_layerwise_finalization = True

        def prepare_target_forward(self, ctx):
            events.append("prepare-target")

        def run(self, **_kwargs):
            events.extend(("draft-write-0", "draft-write-1", "draft-return"))
            return torch.tensor([7], dtype=torch.int32)

    class _FutureInputMap:
        def __setitem__(self, _key, _value):
            events.append("future-input")

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([0]),
        state_write_req_pool_indices_buf=torch.tensor([0]),
    )
    executor.grammar_runtime = None
    executor.drafter = _Drafter()
    executor.config = SimpleNamespace(spec_algo="EAGLE3", pp_size=1, output_length=4)
    executor.runtime_states = SimpleNamespace(
        future_input_map=_FutureInputMap(),
        vocab_size=32,
    )
    executor.nan_guard = SimpleNamespace(
        audit_logits=lambda *_args: None,
        merge_oov=lambda *_args: None,
    )
    executor._run_target_forward = lambda *_args: SimpleNamespace(
        next_token_logprobs=None
    )
    executor._run_sampling = lambda *_args: (
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    executor._draft_final_step_counter = SimpleNamespace(
        record_cache=lambda: events.append("draft-final")
    )
    ctx = SimpleNamespace(bs=1, num_extends=1, input_num_tokens=1)

    executor._forward_step(bs=1, ctx=ctx, sampling_info=object())

    assert events == [
        "prepare-target",
        "draft-write-0",
        "draft-write-1",
        "draft-return",
        "future-input",
        "draft-final",
    ]


def test_cudagraph_gc_flag_reaches_the_capture_context():
    """The operator flag must survive ServerArgs -> config -> capture.

    Freezing the collector for the duration of capture is the default; the
    flag is the escape hatch. It previously never arrived -- the wrapper read
    it off a config that never carried it -- so capture never froze and the
    flag moved nothing in either direction. Pin the whole path, not the
    default: read the value the capture context would actually see.
    """
    import dataclasses
    import gc

    from tokenspeed.runtime.execution.forward_step import (
        ForwardStepRunner,
        freeze_gc,
    )
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig

    fields = {f.name for f in dataclasses.fields(ModelExecutorConfig)}
    assert "enable_cudagraph_gc" in fields, "config must carry the flag"

    for flag in (False, True):
        config = SimpleNamespace(enable_cudagraph_gc=flag)
        wrapper = ForwardStepRunner.__new__(ForwardStepRunner)
        # Only the flag plumbing is under test; __init__ needs a live model.
        wrapper.enable_cudagraph_gc = config.enable_cudagraph_gc
        assert wrapper.enable_cudagraph_gc is flag

        # get_freeze_count() is process-global and other code freezes too, so
        # compare against the count on entry rather than against zero.
        before = gc.get_freeze_count()
        canary = [object() for _ in range(64)]
        with freeze_gc(wrapper.enable_cudagraph_gc):
            during = gc.get_freeze_count()
        after = gc.get_freeze_count()
        assert (during > before) is (not flag), (flag, before, during)
        assert after <= before, (before, after)
        assert len(canary) == 64


def test_non_spec_decode_routes_through_verify():
    """The unified sampling rule: decode rows verify even without a drafter.

    Non-speculative decode is verify's N == 1 case — _decode_candidates
    yields the one-column window (this step's input token per request) and
    _run_sampling must call verify(), never the sample() fast path.
    """
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    calls = []

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.drafter = None
    executor.config = SimpleNamespace(output_length=1)
    executor.input_buffers = SimpleNamespace(
        input_ids_buf=torch.arange(8, dtype=torch.int32),
        force_single_token_verify_buf=torch.zeros(8, dtype=torch.bool),
    )
    executor.sampling_backend = SimpleNamespace(
        sample=lambda *_a, **_k: calls.append("sample") or (None, None),
        verify=lambda _lo, _si, cand: calls.append(("verify", tuple(cand.shape)))
        or (
            torch.zeros(cand.shape[0], dtype=torch.int32),
            torch.ones(cand.shape[0], dtype=torch.int32),
        ),
    )

    # Pure decode, bs=3, N=1: candidates are the tail 3 ids as [3, 1].
    ctx = SimpleNamespace(
        bs=3, num_extends=0, input_num_tokens=3, decode_input_ids=None
    )
    candidates = executor._decode_candidates(ctx)
    assert candidates.shape == (3, 1)
    assert candidates.view(-1).tolist() == [0, 1, 2]

    executor._run_sampling(object(), object(), ctx, candidates)
    assert calls == [("verify", (3, 1))]

    # Pure prefill still samples.
    calls.clear()
    ctx2 = SimpleNamespace(
        bs=2, num_extends=2, input_num_tokens=6, decode_input_ids=None
    )
    assert executor._decode_candidates(ctx2) is None
    executor._run_sampling(object(), object(), ctx2, None)
    assert calls == ["sample"]
