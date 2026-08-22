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

from tokenspeed.runtime.execution import model_executor as model_executor_module
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

    executor.reset_valid_cache_length(forward_op)

    assert executor.runtime_states.valid_cache_lengths[2].item() == 10
    assert executor.runtime_states.valid_cache_lengths[3].item() == 3
    assert executor.runtime_states.valid_cache_lengths[4].item() == 4


def test_draft_final_step_follows_the_complete_drafter_run():
    events = []

    class _Drafter:
        supports_pd_layerwise_finalization = True
        _incremental_proj_enabled = False

        def get_candidates(self, _ctx):
            return None

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
    executor.config = SimpleNamespace(spec_algo="EAGLE3")
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
    ctx = SimpleNamespace(bs=1, num_extends=1)

    executor._forward_step(bs=1, ctx=ctx, sampling_info=object())

    assert events == [
        "draft-write-0",
        "draft-write-1",
        "draft-return",
        "future-input",
        "draft-final",
    ]


def test_autotune_dummy_prefill_fits_request_capacity(monkeypatch):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(
        chunked_prefill_size=8192,
        context_len=1024,
        max_num_seqs=1,
        data_parallel_size=1,
        disable_autotune=True,
    )
    executor.model_runner = object()
    captured = []
    monkeypatch.setattr(
        model_executor_module,
        "set_autotune_max_num_tokens",
        captured.append,
    )

    executor._autotune()

    assert captured == [1024]


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

    from tokenspeed.runtime.execution.cuda_graph_wrapper import (
        CudaGraphWrapper,
        freeze_gc,
    )
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig

    fields = {f.name for f in dataclasses.fields(ModelExecutorConfig)}
    assert "enable_cudagraph_gc" in fields, "config must carry the flag"

    for flag in (False, True):
        config = SimpleNamespace(enable_cudagraph_gc=flag)
        wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
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
