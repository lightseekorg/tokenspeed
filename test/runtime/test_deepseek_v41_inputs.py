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

"""Bounded Engram inputs across physical-token request lifecycles.

CPU checks stub only CUDA metadata preparation/pinning; CUDA checks exercise the
real InputBuffers fill kernels. No checkpoint-sized model or history cache is
allocated. Run with CUDA_VISIBLE_DEVICES selecting an available GPU.
"""

from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.engine.scheduler_utils import (
    engram_context_len,
    ngram_inputs_for_forward,
)
from tokenspeed.runtime.execution import input_buffer, model_executor, weight_loader
from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    NGramInputs,
    PlannedForward,
)
from tokenspeed.runtime.execution.weight_loader import WeightLoader
from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41ForCausalLM

VOCAB_SIZE = 128


def _state(prompt, output):
    return SimpleNamespace(
        prompt_input_ids=list(prompt),
        prompt_input_ids_unpadded=[99],
        output_ids=list(output),
    )


def _op(states, rids, slots, lengths, prefixes, overrides):
    ids = []
    for rid, length, prefix in zip(rids, lengths, prefixes):
        state = states[rid]
        ids.extend(
            (state.prompt_input_ids + state.output_ids)[prefix : prefix + length]
        )
    return SimpleNamespace(
        request_ids=rids,
        request_pool_indices=slots,
        input_lengths=lengths,
        prefill_lengths=[
            len(states[r].prompt_input_ids) + len(states[r].output_ids) for r in rids
        ],
        input_ids=ids,
        shifted_input_ids=ids[1:] + [-1] if ids else [],
        extend_prefix_lens=prefixes,
        decode_input_ids=overrides,
        num_extends=lambda: len(prefixes),
    )


@pytest.fixture(params=["cpu", "cuda"])
def buffers(request, monkeypatch):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "cpu":

        def bulk(self, *specs):
            return [torch.empty(n, dtype=dtype) for n, dtype in specs]

        def positions(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum, out):
            values = [
                torch.arange(int(p), int(p + n))
                for p, n in zip(extend_prefix_lens, extend_seq_lens)
            ]
            out.copy_(torch.cat(values) if values else torch.empty(0))
            return out, None

        def decode_positions(
            positions_ptr,
            seq_lens_out_ptr,
            req_pool_indices,
            valid_cache_lengths,
            uniform_input_length,
        ):
            starts = valid_cache_lengths[req_pool_indices]
            positions_ptr.copy_(
                (starts[:, None] + torch.arange(uniform_input_length)).flatten()
            )
            seq_lens_out_ptr.copy_(starts + uniform_input_length)

        monkeypatch.setattr(InputBuffers, "_bulk_pinned", bulk)
        monkeypatch.setattr(input_buffer, "compute_position_triton", positions)
        monkeypatch.setattr(input_buffer, "fused_decode_input_prep", decode_positions)
        tensor = torch.tensor

        def unpinned_tensor(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return tensor(*args, **kwargs)

        monkeypatch.setattr(torch, "tensor", unpinned_tensor)
    ib = InputBuffers(
        max_bs=4, max_num_tokens=32, state_write_padding_pool_index=5, device=device
    )
    ib.init_ngram_buffers(3)
    runtime = RuntimeStates(
        req_pool_size=5, vocab_size=VOCAB_SIZE, output_length=1, device=device
    )
    return ib, runtime


def _fill(ib, runtime, op, snapshot):
    n = op.num_extends()
    runtime.reset_states(
        torch.tensor(op.request_pool_indices[:n], dtype=torch.int64, device=ib.device),
        torch.tensor(op.extend_prefix_lens, dtype=torch.int32, device=ib.device),
    )
    ib.fill_input_buffers(op, runtime, sum(op.input_lengths), ngram_inputs=snapshot)


def _expected(tokens, positions):
    return [
        [
            tokens[p - d] if p >= d and 0 <= tokens[p - d] < VOCAB_SIZE else -1
            for d in (1, 2, 3)
        ]
        for p in positions
    ]


def _assert_rows(ib, expected, mask):
    count = len(expected)
    kwargs = ib.ngram_model_kwargs(count)
    assert kwargs["engram_previous_tokens"].dtype == torch.int64
    assert kwargs["engram_previous_tokens"].cpu().tolist() == expected
    assert kwargs["engram_token_mask"].dtype == torch.bool
    assert kwargs["engram_token_mask"].cpu().tolist() == mask
    assert (ib.ngram_previous_tokens_buf[count:] == -1).all()
    assert not ib.ngram_token_mask_buf[count:].any()


def _sample(ib, runtime, ids, num_extends):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.drafter = None
    executor.config = SimpleNamespace(output_length=1)
    executor.runtime_states = runtime
    bs = len(ids)
    ModelExecutor._update_runtime_state(
        executor,
        ib.req_pool_indices_buf[:bs],
        torch.tensor(ids, dtype=torch.int32, device=ib.device),
        torch.ones(bs, dtype=torch.int32, device=ib.device),
        ib.input_lengths_buf[:bs],
        num_extends,
    )


@pytest.mark.parametrize("overlap", [False, True])
def test_chunked_prefill_and_pending_overlap_samples(buffers, overlap):
    ib, runtime = buffers
    states = {"a": _state([10, 11, 12, 13, 14], [])}
    for prefix, length in [(0, 2), (2, 3)]:
        op = _op(states, ["a"], [2], [length], [prefix], [])
        _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
        _assert_rows(
            ib,
            _expected(states["a"].prompt_input_ids, range(prefix, prefix + length)),
            [True] * length,
        )
        _sample(ib, runtime, [15], 1)

    pointer = ib.ngram_previous_tokens_buf.data_ptr()
    for current in [15, 16, 17, 18]:
        if not overlap:
            states["a"].output_ids.append(current)
        op = _op(states, ["a"], [2], [1], [], [-1])
        snapshot = ngram_inputs_for_forward(op, states, 3)
        if overlap:
            # Commit can mutate request state after dispatch but before the
            # forward thread consumes its frozen snapshot. The current ID is
            # still resolved from future_input_map, never from host input_ids.
            states["a"].output_ids.append(current)
        _fill(ib, runtime, op, snapshot)
        full = states["a"].prompt_input_ids + states["a"].output_ids
        assert ib.input_ids_buf[0].item() == current
        _assert_rows(ib, _expected(full, [len(full) - 1]), [True])
        assert ib.ngram_previous_tokens_buf.data_ptr() == pointer
        _sample(ib, runtime, [current + 1], 0)
    assert set(vars(runtime)) == {
        "device",
        "vocab_size",
        "valid_cache_lengths",
        "future_input_map",
        "remote_spec_candidate_ready",
    }


@pytest.mark.parametrize("barrier", [-7, VOCAB_SIZE + 50])
def test_prefix_hit_mixed_reordered_requests_and_raw_barriers(buffers, barrier):
    ib, runtime = buffers
    states = {
        "prefill": _state([1, 2, 3, barrier, 5, 6, 7], []),
        "decode": _state([20, 21, 22, 23], [24]),
    }
    runtime.valid_cache_lengths[3] = 4
    runtime.future_input_map[3, 0] = 24
    op = _op(states, ["prefill", "decode"], [1, 3], [4, 1], [3], [-1])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    expected = _expected(states["prefill"].prompt_input_ids, range(3, 7)) + [
        [23, 22, 21]
    ]
    _assert_rows(ib, expected, [False, True, True, True, True])
    assert ib.input_ids_buf[0].item() == min(max(barrier, 0), VOCAB_SIZE - 1)
    assert ib.positions_buf[:5].tolist() == [3, 4, 5, 6, 4]
    _sample(ib, runtime, [8, 25], 1)
    states["prefill"].output_ids.append(8)
    states["decode"].output_ids.append(25)
    op = _op(states, ["decode", "prefill"], [3, 1], [1, 1], [], [-1, -1])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    _assert_rows(ib, [[24, 23, 22], [7, 6, 5]], [True, True])


def test_retraction_readmission_pd_bootstrap_and_slot_reuse(buffers):
    ib, runtime = buffers
    states = {"a": _state([10, 11, 12], [13, 14, 15])}
    # Retraction turns accepted output back into a prefill suffix. Neither
    # prefix matching nor a pool-slot change changes physical token identity.
    op = _op(states, ["a"], [4], [3], [3], [])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    _assert_rows(ib, [[12, 11, 10], [13, 12, 11], [14, 13, 12]], [True] * 3)
    del states["a"]
    states["b"] = _state([30], [])
    op = _op(states, ["b"], [4], [1], [0], [])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    _assert_rows(ib, [[-1, -1, -1]], [True])

    # A PD destination need not have executed any local prefill. Its bootstrap
    # token and complete physical prompt suffice, with the existing override.
    states["pd"] = _state([40, 41, 42], [43])
    runtime.valid_cache_lengths[1] = 3
    op = _op(states, ["pd"], [1], [1], [], [43])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    _assert_rows(ib, [[42, 41, 40]], [True])
    assert ib.input_ids_buf[0].item() == 43


def test_empty_prefill_padding_and_idle_scrub(buffers):
    ib, runtime = buffers
    states = {"a": _state([1, 2, 3], [])}
    op = _op(states, ["a"], [1], [3], [0], [])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    pointer = ib.ngram_previous_tokens_buf.data_ptr()
    ib.fill_dummy_decode_buffers(batch_size=4, total_tokens=4)
    _assert_rows(ib, [[-1] * 3] * 4, [False] * 4)
    op = _op(states, ["a"], [1], [0], [3], [])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    assert ib.ngram_model_kwargs(0)["engram_previous_tokens"].shape == (0, 3)
    assert (ib.ngram_previous_tokens_buf == -1).all()
    assert not ib.ngram_token_mask_buf.any()
    assert ib.ngram_previous_tokens_buf.data_ptr() == pointer


def test_snapshot_validation_and_no_long_history_copy(buffers):
    ib, _ = buffers

    class NoIteration(list):
        def __iter__(self):
            raise AssertionError("must not copy full request history")

        def __add__(self, other):
            raise AssertionError("must not concatenate full request history")

    state = _state([], [])
    state.prompt_input_ids = NoIteration(range(100_000))
    op = SimpleNamespace(request_ids=["a"], input_lengths=[1], num_extends=lambda: 0)
    snapshot = ngram_inputs_for_forward(op, {"a": state}, 3)
    assert snapshot.tokens == ((99999, 99998, 99997, 99996),)
    assert snapshot.positions == (99999,)
    with pytest.raises(FrozenInstanceError):
        snapshot.positions = (0,)
    with pytest.raises(ValueError, match="one history snapshot"):
        ib.fill_ngram_inputs(None, 1, VOCAB_SIZE)
    with pytest.raises(ValueError, match="history width"):
        ib.fill_ngram_inputs(
            NGramInputs(tokens=((1, 2),), positions=(0,)), 1, VOCAB_SIZE
        )
    op.input_lengths = [2]
    with pytest.raises(NotImplementedError, match="speculation"):
        ngram_inputs_for_forward(op, {"a": state}, 3)
    assert ngram_inputs_for_forward(None, {}, 0) is None
    assert (
        engram_context_len(SimpleNamespace(ple_layer_ids=[1], ngram_context_len=2)) == 0
    )
    assert (
        engram_context_len(SimpleNamespace(engram_layer_ids=[1], ngram_context_len=3))
        == 3
    )
    with pytest.raises(ValueError, match="ngram_context_len = 3"):
        engram_context_len(SimpleNamespace(engram_layer_ids=[1]))


@pytest.mark.parametrize(
    "mode",
    [ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.MIXED, ForwardMode.IDLE],
)
def test_target_runner_passes_model_kwargs_not_context_tensors(buffers, mode):
    ib, _ = buffers
    num_tokens = 0 if mode == ForwardMode.IDLE else 2

    class Model:
        def forward(self, ctx, input_ids, positions, **kwargs):
            return DeepseekV41ForCausalLM.prepare_model_kwargs(
                self, ctx, input_ids, kwargs
            )

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = Model()
    runner.is_generation = True
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.model_runner = runner
    executor.input_buffers = ib
    executor.config = SimpleNamespace(model_is_mrope=False, pp_size=1)
    executor._active_positions_override = None
    executor._active_multimodal_context = None
    executor.prefill_graph = SimpleNamespace(can_run=lambda ctx, mm: False)
    ctx = SimpleNamespace(input_num_tokens=num_tokens, forward_mode=mode)
    original = vars(ctx).copy()
    result = executor._run_target_forward(ctx)
    assert result["engram_previous_tokens"].shape == (num_tokens, 3)
    assert result["engram_token_mask"].shape == (num_tokens,)
    if num_tokens:
        assert (
            result["engram_previous_tokens"].data_ptr()
            == ib.ngram_previous_tokens_buf.data_ptr()
        )
    assert vars(ctx) == original


@pytest.mark.parametrize("context_len", [3, 4])
def test_autotune_passes_engram_views_and_resets_dummy_inputs(
    buffers, monkeypatch, context_len
):
    """Run the startup forward, not just its serving-path counterpart."""
    ib, _ = buffers
    num_tokens = min(7, context_len * 2)
    lengths = [context_len, num_tokens - context_len]
    events = []
    metadata = []
    tuning = False
    views = ib.ngram_model_kwargs(num_tokens)
    for value in ib.ngram_model_kwargs(ib.max_num_tokens).values():
        value.fill_(1)

    @contextmanager
    def tuner():
        nonlocal tuning
        events.append("tuner-enter")
        tuning = True
        yield
        tuning = False
        events.append("tuner-exit")

    def init_metadata(**kwargs):
        assert tuning
        assert kwargs["bs"] == kwargs["num_extends"] == 2
        assert kwargs["forward_mode"] == ForwardMode.EXTEND
        assert kwargs["seq_lens"].tolist() == lengths
        assert kwargs["extend_prefix_lens"].tolist() == [0, 0]
        assert not kwargs["extend_with_prefix"]
        # An unbound fake pool exercises dummy setup without allocating KV.
        assert "block_tables" not in kwargs
        metadata.append(kwargs)
        events.append("metadata")

    def forward(ctx, input_ids, positions, **kwargs):
        assert tuning
        model_kwargs = DeepseekV41ForCausalLM.prepare_model_kwargs(
            runner.model, ctx, input_ids, kwargs
        )
        for key, view in views.items():
            actual = model_kwargs[key]
            assert actual.shape == view.shape
            assert actual.dtype == view.dtype
            assert actual.data_ptr() == view.data_ptr()
        assert (model_kwargs["engram_previous_tokens"] == -1).all()
        assert not model_kwargs["engram_token_mask"].any()
        assert input_ids.shape == (num_tokens,)
        assert input_ids.data_ptr() == ib.input_ids_buf.data_ptr()
        assert positions.data_ptr() == ib.positions_buf.data_ptr()
        assert positions.tolist() == list(range(lengths[0])) + list(range(lengths[1]))
        assert ctx.attn_backend is pg.attn_backend
        assert ctx.input_num_tokens == num_tokens
        assert ctx.bs == ctx.num_extends == 2
        assert ctx.forward_mode == ForwardMode.EXTEND
        assert "engram_previous_tokens" not in vars(ctx)
        assert "engram_token_mask" not in vars(ctx)
        events.append("forward")

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(forward=forward)
    runner.is_generation = True
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = ib.device
    executor.config = SimpleNamespace(
        max_num_seqs=2,
        data_parallel_size=1,
        chunked_prefill_size=7,
        context_len=context_len,
        physical_context_len=context_len,
        pp_size=1,
        world_size=1,
        disable_autotune=False,
        model_is_mrope=False,
        device=ib.device,
    )
    executor.input_buffers = ib
    executor.model_runner = runner
    pg = PrefillGraph.__new__(PrefillGraph)
    pg.config = executor.config
    pg.input_buffers = ib
    pg.attn_backend = SimpleNamespace(init_forward_metadata=init_metadata)
    pg.token_to_kv_pool = SimpleNamespace(arena=SimpleNamespace(cache_group_specs=()))
    pg.dp_size = 1
    pg.drafter = None
    executor.prefill_graph = pg
    monkeypatch.setattr(model_executor, "autotune", tuner)
    monkeypatch.setattr(
        model_executor,
        "set_autotune_max_num_tokens",
        lambda count: events.append(("max_tokens", count)),
    )
    monkeypatch.setattr(
        model_executor,
        "set_autotune_process_group",
        lambda group: events.append(("group", group)),
    )
    monkeypatch.setattr(
        model_executor.dist, "barrier", lambda: events.append("barrier")
    )

    executor._autotune()

    assert (ib.ngram_previous_tokens_buf == -1).all()
    assert not ib.ngram_token_mask_buf.any()
    assert len(metadata) == 1
    assert events == [
        ("max_tokens", num_tokens),
        ("group", None),
        "tuner-enter",
        "metadata",
        "forward",
        "tuner-exit",
        ("group", None),
        "barrier",
    ]


def test_execute_idle_forward_passes_empty_engram_views(buffers):
    ib, runtime = buffers
    seen = []

    def forward(ctx, input_ids, positions, **kwargs):
        assert ctx.forward_mode == ForwardMode.IDLE
        assert ctx.bs == ctx.input_num_tokens == 0
        assert input_ids.shape == positions.shape == (0,)
        for key, view in ib.ngram_model_kwargs(0).items():
            assert kwargs[key].shape == view.shape
            assert (
                kwargs[key].untyped_storage().data_ptr()
                == view.untyped_storage().data_ptr()
            )
        seen.append(ctx)

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = ib.device
    executor.input_buffers = ib
    executor.runtime_states = runtime
    executor.attn_backend = SimpleNamespace()
    executor.token_to_kv_pool = SimpleNamespace()
    executor.model_runner = SimpleNamespace(forward=forward)
    executor.forward_step = SimpleNamespace(can_run=lambda bs, ctx: False)
    executor.drafter = None
    executor.execute_idle_forward(
        DpForwardMetadata(
            global_num_tokens=[0],
            global_batch_size=[0],
            global_forward_mode=[ForwardMode.IDLE],
            all_decode_or_idle=True,
            all_extend=False,
            need_idle_forward=True,
        )
    )
    assert len(seen) == 1


def test_history_views_replay_after_batch_shrink_and_idle(buffers):
    """Check the input contract, not V4.1 backend graph support (still disabled)."""
    ib, runtime = buffers
    if ib.device != "cuda":
        pytest.skip("CUDA graph buffer contract")
    states = {"a": _state([10, 11, 12], [])}
    op = _op(states, ["a"], [1], [3], [0], [])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        history = ib.ngram_model_kwargs(4)["engram_previous_tokens"].clone()
        mask = ib.ngram_model_kwargs(4)["engram_token_mask"].clone()
    states["b"] = _state([20, 21], [])
    op = _op(states, ["b"], [1], [1], [1], [])
    _fill(ib, runtime, op, ngram_inputs_for_forward(op, states, 3))
    graph.replay()
    assert history.tolist() == [[20, -1, -1]] + [[-1] * 3] * 3
    assert mask.tolist() == [True, False, False, False]
    ib.fill_dummy_decode_buffers(batch_size=4, total_tokens=4)
    graph.replay()
    assert history.tolist() == [[-1] * 3] * 4
    assert not mask.any()


def test_dispatch_owns_snapshot_until_forward_thread_consumes_it():
    states = {"a": _state([1, 2, 3], [])}
    op = _op(states, ["a"], [1], [1], [], [-1])
    snapshot = ngram_inputs_for_forward(op, states, 3)
    submitted, consumed = [], []

    def submit(fn):
        future = Future()
        submitted.append((future, fn))
        return future

    def execute(forward_op, sampling_params_list, **kwargs):
        consumed.append(kwargs["ngram_inputs"])
        return SimpleNamespace(sync=lambda: None)

    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=SimpleNamespace(submit=submit), execute_forward_op=execute
        ),
        l2_cache_executor=None,
        kv_transfer=None,
    )
    planned = PlannedForward(
        forward_op=op,
        sampling_params_list=[],
        dp_metadata=None,
        grammar_inputs=None,
        multimodal_context=None,
        ngram_inputs=snapshot,
    )
    pending = handle._submit_forward(planned, capture_next_input_ids=False)
    states["a"].prompt_input_ids.clear()
    states.clear()
    assert not consumed
    future, fn = submitted.pop()
    future.set_result(fn())
    pending.result()
    assert consumed == [NGramInputs(tokens=((3, 2, 1, -1),), positions=(2,))]


@pytest.mark.parametrize("has_engram", [False, True])
def test_weight_loader_initializes_engram_once_in_weight_region(
    monkeypatch, has_engram
):
    events = []
    tokenizer = object()

    def initialize(value):
        assert value is tokenizer
        events.append("initialize")

    model = SimpleNamespace(initialize_engram=initialize) if has_engram else object()

    @contextmanager
    def region(tag, enable_cpu_backup):
        assert (tag, enable_cpu_backup) == ("weights", True)
        events.append("enter")
        yield
        events.append("exit")

    def load(**kwargs):
        events.append("load")
        return model

    def get_tokenizer(path, **kwargs):
        assert path == "configured-tokenizer"
        assert kwargs == dict(
            tokenizer_mode="auto",
            trust_remote_code=False,
            revision="revision",
            architectures=["DeepseekV41ForCausalLM"],
        )
        events.append("tokenizer")
        return tokenizer

    monkeypatch.setattr(weight_loader, "get_model", load)
    monkeypatch.setattr(weight_loader, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(weight_loader, "get_available_gpu_memory", lambda *args: 1.0)
    monkeypatch.setattr(weight_loader, "set_cuda_arch", lambda: None)
    monkeypatch.setattr(weight_loader, "LoadConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(weight_loader, "DeviceConfig", lambda value: value)
    args = SimpleNamespace(
        load_format="dummy",
        download_dir=None,
        ext_yaml=None,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=1,
        kv_cache_dtype="bfloat16",
        tokenizer="configured-tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
        revision="revision",
    )
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]),
    )
    result = WeightLoader.load_model(
        model_config=config,
        server_args=args,
        device="cpu",
        gpu_id=0,
        memory_saver_adapter=SimpleNamespace(region=region),
    )
    assert result is model
    assert events == (
        ["enter", "load", "tokenizer", "initialize", "exit"]
        if has_engram
        else ["enter", "load", "exit"]
    )
