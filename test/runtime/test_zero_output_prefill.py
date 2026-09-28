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

"""CPU orchestration contracts without importing CUDA-only model dependencies.

Load the production methods themselves, with explicit stand-ins only for
GPU collaborators. These tests do not validate model numerics or graph replay.
"""

from __future__ import annotations

import ast
import dataclasses
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = "python/tokenspeed/runtime/"


def load_symbol(path, name, *, owner=None, **bindings):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if owner is not None:
        body = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == owner
        ).body
    node = next(node for node in body if getattr(node, "name", None) == name)
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    scope = dict(__name__=__name__, torch=torch, dataclasses=dataclasses, **bindings)
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return scope[name]


@dataclasses.dataclass
class Output:
    next_token_logits: torch.Tensor
    next_token_ids: torch.Tensor | None = None
    next_token_logprobs: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    logits_layout_plan: object = None


class SharedBufferSampler:
    def __init__(self):
        self.tokens = torch.empty(32, dtype=torch.int32)
        self.lengths = torch.empty(8, dtype=torch.int32)
        self.logprobs = torch.empty(32)
        self.calls = []

    def sample(self, output, info):
        self.calls.append(("sample", info))
        n = output.next_token_logits.shape[0]
        self.tokens[:n] = output.next_token_logits.argmax(-1)
        self.lengths[:n] = 1
        self.logprobs[:n] = -torch.arange(1, n + 1)
        output.next_token_logprobs = self.logprobs[:n]
        return self.tokens[:n], self.lengths[:n]

    def verify(self, output, info, candidates):
        self.calls.append(("verify", info))
        n, width = candidates.shape
        self.tokens[: n * width] = output.next_token_logits.argmax(-1)
        self.lengths[:n] = width
        self.logprobs[: n * width] = -10 - torch.arange(n * width)
        output.next_token_logprobs = self.logprobs[: n * width]
        return self.tokens[: n * width], self.lengths[:n]


@pytest.mark.parametrize(
    "p,e,d,width",
    [(1, 2, 1, 1), (1, 2, 2, 3), (0, 1, 1, 3), (0, 1, 0, 3), (1, 2, 0, 1)],
)
def test_sampling_preserves_request_parameters_and_shared_outputs(p, e, d, width):
    info_cls = load_symbol(
        RUNTIME + "sampling/sampling_batch_info.py", "SamplingBatchInfo"
    )
    sample = load_symbol(
        RUNTIME + "execution/model_executor.py",
        "_run_sampling",
        owner="ModelExecutor",
        LogitsProcessorOutput=Output,
    )
    backend = SharedBufferSampler()
    executor = SimpleNamespace(
        sampling_backend=backend,
        _apply_force_single_token_verify=lambda lengths, *args: lengths,
    )
    ctx = SimpleNamespace(
        bs=e + d,
        num_extends=e,
        decode_input_ids=None,
        output_layout=ForwardOutputLayout(e, p, d, width),
    )
    info = info_cls(
        req_pool_indices=torch.arange(e + d) + 10,
        vocab_mask=torch.arange((e + d) * width)[:, None],
    )
    logits = torch.full((p + d * width, 32), -10.0)
    if logits.shape[0]:
        logits[torch.arange(logits.shape[0]), torch.arange(logits.shape[0]) + 1] = 10
    output = Output(logits)
    tokens, lengths = sample(
        executor, output, info, ctx, torch.zeros(d, width, dtype=torch.int32)
    )
    assert tokens.tolist() == list(range(1, p + d * width + 1))
    assert lengths.tolist() == [1] * p + [0] * (e - p) + [width] * d
    assert [kind for kind, _ in backend.calls] == (["sample"] if p else []) + (
        ["verify"] if d else []
    )
    for kind, subset in backend.calls:
        if kind == "sample":
            assert subset.req_pool_indices.tolist() == list(range(10, 10 + p))
            assert subset.vocab_mask[:, 0].tolist() == list(range(0, p * width, width))
        else:
            assert subset.req_pool_indices.tolist() == list(range(10 + e, 10 + e + d))
            assert subset.batch_row_offset == e
            assert subset.vocab_mask[:, 0].tolist() == list(
                range(e * width, (e + d) * width)
            )
    if p:
        assert output.next_token_logprobs[:p].tolist() == [-1.0] * p


def test_nan_and_oov_map_to_original_decode_rows():
    ctx = SimpleNamespace(
        bs=3, num_extends=2, output_layout=ForwardOutputLayout(2, 1, 1, 2)
    )
    guard = NanGuard(3, "cpu")
    logits = torch.zeros(3, 16)
    logits[2, 3] = float("nan")
    guard.audit_logits(Output(logits), ctx)
    assert guard.flags.tolist() == [0, 0, 1]
    guard.reset(3)
    guard.merge_oov(torch.tensor([19, 2, 3]), ctx, 16)
    assert guard.flags.tolist() == [1, 0, 0]


class Matcher:
    finished = False

    def __init__(self):
        self.tokens = []

    def is_terminated(self):
        return False

    def accept_token(self, token):
        self.tokens.append(token)


@pytest.mark.parametrize("hostfunc", [True, False])
def test_grammar_consumers_share_compact_token_offsets(hostfunc):
    grammars = [Matcher(), Matcher(), Matcher()]
    completion = SimpleNamespace(
        grammars=grammars,
        bs=3,
        tokens_per_req=2,
        advance_mask=[True, False, True],
        output_layout=ForwardOutputLayout(2, 1, 1, 2),
        lock=threading.Lock(),
        event=threading.Event(),
    )
    tokens = torch.tensor([11, 21, 22])
    lengths = torch.tensor([1, 0, 2])
    if hostfunc:
        method = load_symbol(
            RUNTIME + "grammar/capturable_grammar.py",
            "_advance_prev",
            owner="CapturableGrammarExecutor",
        )
        method(
            SimpleNamespace(output_tokens_host=tokens, accept_lengths_host=lengths),
            dict(
                completion=completion,
                grammars=grammars,
                bs=3,
                tokens_per_req=2,
                advance_mask=completion.advance_mask,
            ),
        )
        assert completion.event.is_set()
    else:
        method = load_symbol(
            RUNTIME + "engine/generation_output_processor.py",
            "_host_advance_matcher",
            owner="OutputProcesser",
        )
        method(
            None,
            completion,
            SimpleNamespace(output_tokens=tokens, output_lengths=lengths),
        )
    assert [g.tokens for g in grammars] == [[11], [], [21, 22]]


def test_dspark_anchors_do_not_read_or_write_incomplete_prefill():
    anchors = load_symbol(
        "tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/__init__.py",
        "dspark_anchors",
    )
    next_tokens = torch.full((3, 3), -7, dtype=torch.int32)
    starts = torch.empty(1, dtype=torch.int64)
    anchors(
        torch.tensor([11, 21, 22, 23]),
        torch.tensor([1, 0, 2]),
        torch.tensor([50, 51, 52]),
        2,
        3,
        next_tokens,
        starts,
        num_prefill_outputs=1,
    )
    assert next_tokens.tolist() == [[11] * 3, [-7] * 3, [22] * 3]
    assert starts.tolist() == [51]


def test_zero_decoder_rows_do_not_select_a_padded_graph():
    import bisect

    bucket = load_symbol(
        RUNTIME + "execution/prefill_graph.py",
        "_decoder_bucket",
        owner="PrefillGraph",
        bisect=bisect,
    )
    assert bucket(SimpleNamespace(decoder_buckets=[16, 32, 64]), 0) is None


def test_cache_progress_is_independent_of_generated_tokens():
    advance = load_symbol(
        "tokenspeed-kernel/python/tokenspeed_kernel/ops/metadata/accepted_frontier.py",
        "_advance_accepted_frontier_torch",
    )
    update = load_symbol(
        RUNTIME + "execution/model_executor.py",
        "_update_runtime_state",
        owner="ModelExecutor",
        advance_accepted_frontier=advance,
    )
    states = SimpleNamespace(
        future_input_map=torch.full((6, 1), 77, dtype=torch.int32),
        valid_cache_lengths=torch.zeros(6, dtype=torch.int32),
        ngram_accepted_tokens=None,
    )
    buffers = SimpleNamespace(
        state_write_padding_pool_index=0,
        ngram_previous_tokens_buf=None,
        ngram_token_mask_buf=None,
    )
    executor = SimpleNamespace(
        drafter=None, runtime_states=states, input_buffers=buffers
    )
    update(
        executor,
        torch.tensor([4, 2, 5]),
        torch.tensor([11, 21]),
        torch.tensor([1, 0, 1]),
        torch.tensor([128, 128, 1]),
        2,
        output_layout=ForwardOutputLayout(2, 1, 1, 1),
    )
    assert states.valid_cache_lengths.tolist() == [0, 0, 128, 0, 128, 1]
    assert states.future_input_map[:, 0].tolist() == [77, 77, 77, 77, 11, 21]


def test_zero_rows_return_before_lm_head_and_retain_empty_taps():
    forward = load_symbol(
        RUNTIME + "layers/logits_processor.py",
        "forward",
        owner="LogitsProcessor",
        LogitsProcessorOutput=Output,
    )
    metadata = SimpleNamespace(
        logits_rows_selected=True,
        extend_return_logprob=False,
        capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
    )
    # No LM-head method exists on this object; reaching it fails the test.
    output = forward(
        SimpleNamespace(config=SimpleNamespace(vocab_size=32)),
        torch.arange(4),
        torch.empty(0, 8),
        None,
        metadata,
        [torch.empty(0, 8)] * 3,
    )
    assert output.next_token_logits.shape == (0, 32)
    assert output.hidden_states.shape == (0, 24)


def test_selected_logits_do_not_gather_original_input_indices():
    forward = load_symbol(
        RUNTIME + "layers/logits_processor.py",
        "forward",
        owner="LogitsProcessor",
        LogitsProcessorOutput=Output,
    )
    metadata = SimpleNamespace(
        logits_rows_selected=True,
        extend_return_logprob=False,
        gather_ids=torch.tensor([127, 255, 256]),
        capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
    )
    hidden = torch.arange(16).view(2, 8).float()
    processor = SimpleNamespace(
        _resolve_logits_layout_plan=lambda *args: None,
        _get_logits=lambda h, *args, **kwargs: h,
        do_argmax=False,
    )
    output = forward(processor, torch.arange(257), hidden, None, metadata)
    assert output.next_token_logits is hidden


def test_zero_decoder_attention_still_runs_the_global_producer():
    positions = torch.arange(4)
    requests = torch.zeros(4, dtype=torch.int64)
    full = SimpleNamespace(positions=positions, request_indices=requests)
    empty = SimpleNamespace(positions=positions[:0], request_indices=requests[:0])
    plan = SimpleNamespace(
        source=full, query=empty, keep_rows=torch.empty(0, dtype=torch.int64)
    )
    forward = load_symbol(
        RUNTIME + "models/deepseek_v41.py",
        "forward",
        owner="DeepseekV41Attention",
        _row_plan=lambda *args: plan,
        current_forward_ctx=lambda: None,
    )
    writes = []
    attention = SimpleNamespace(
        layer_id=20,
        ced_decoder_start=20,
        _write_global_kv=lambda hidden, *args: writes.append(hidden.clone()),
    )
    hidden = torch.randn(4, 8)
    output = forward(
        attention,
        positions,
        hidden,
        SimpleNamespace(attn_backend=None, forward_mode=object()),
    )
    assert output.shape == (0, 8)
    torch.testing.assert_close(writes[0], hidden)
    assert len(writes) == 1


def test_zero_decoder_skips_layers_and_normalization():
    forward = load_symbol(
        RUNTIME + "models/deepseek_v41.py", "decoder_forward", owner="DeepseekV41Model"
    )
    model = SimpleNamespace(ced_decoder_start=20, dspark_capture_layers=(19, 20, 39))
    state = SimpleNamespace(
        rows=0,
        hidden=torch.empty(0, 4, 8),
        pre_mix=torch.empty(0, 4),
        captured=[torch.empty(0, 8)] * 2,
    )
    hidden, captures = forward(model, state, None)
    assert hidden.shape == (0, 8)
    assert [t.shape for t in captures] == [(0, 8)] * 3


@pytest.mark.parametrize("decode", [False, True])
def test_identity_sampling_keeps_backend_buffer_aliases(decode):
    info_cls = load_symbol(
        RUNTIME + "sampling/sampling_batch_info.py", "SamplingBatchInfo"
    )
    sample = load_symbol(
        RUNTIME + "execution/model_executor.py",
        "_run_sampling",
        owner="ModelExecutor",
        LogitsProcessorOutput=Output,
    )
    backend = SharedBufferSampler()
    executor = SimpleNamespace(
        sampling_backend=backend,
        _apply_force_single_token_verify=lambda lengths, *args: lengths,
    )
    info = info_cls(req_pool_indices=torch.tensor([3, 7]))
    ctx = SimpleNamespace(
        bs=2, num_extends=0 if decode else 2, decode_input_ids=None, output_layout=None
    )
    tokens, lengths = sample(
        executor, Output(torch.eye(2)), info, ctx, torch.zeros(2, 1, dtype=torch.int32)
    )
    assert tokens.data_ptr() == backend.tokens.data_ptr()
    assert lengths.data_ptr() == backend.lengths.data_ptr()
    assert backend.calls[0][1] is info
    assert tokens.tolist() == [0, 1]


def test_idle_graph_grammar_enqueues_an_identity_completion():
    import queue
    from contextlib import nullcontext

    add = load_symbol(
        RUNTIME + "grammar/capturable_grammar.py",
        "add_batch",
        owner="CapturableGrammarExecutor",
        GrammarStepCompletion=SimpleNamespace,
    )
    grammar = SimpleNamespace(queue=queue.Queue())
    grammar.add_batch = lambda **kwargs: add(grammar, **kwargs)
    idle = load_symbol(
        RUNTIME + "execution/model_executor.py",
        "execute_idle_forward",
        owner="ModelExecutor",
        ForwardMode=SimpleNamespace(DECODE=0),
        ForwardContext=SimpleNamespace,
        SamplingBatchInfo=SimpleNamespace,
        nvtx_range=lambda *args, **kwargs: nullcontext(),
    )

    class Step:
        def __init__(self): self.called = False
        def can_run(self, **kwargs):
            return True

        def padded_bs(self, **kwargs):
            return 1

        def __call__(self, **kwargs):
            self.called = True

    step = Step()
    buffers = SimpleNamespace(
        req_pool_indices_buf=torch.zeros(1),
        fill_dummy_decode_buffers=lambda **kwargs: None,
    )
    for name in (
        "extend_prefix_lens_buf",
        "extend_prefix_lens_cpu",
        "extend_seq_lens_buf",
        "extend_seq_lens_cpu",
        "extend_replay_lens_cpu",
        "extend_prompt_lens_cpu",
    ):
        setattr(buffers, name, torch.zeros(1))
    executor = SimpleNamespace(
        attn_backend=None,
        token_to_kv_pool=None,
        input_buffers=buffers,
        runtime_states=SimpleNamespace(
            valid_cache_lengths=torch.zeros(1), vocab_size=32
        ),
        device="cpu",
        forward_step=step,
        config=SimpleNamespace(output_length=1),
        capturable_grammar=grammar,
    )
    idle(
        executor,
        SimpleNamespace(
            global_num_tokens=[0, 1], global_batch_size=[0, 1], all_decode_or_idle=True
        ),
    )
    assert step.called
    completion = grammar.queue.get_nowait()["completion"]
    assert completion.output_layout is None
    assert grammar.queue.empty()


@pytest.mark.parametrize("capturable", [True,False])
def test_grammar_mask_producers_walk_only_original_decode_candidates(capturable,monkeypatch):
    class MaskMatcher(Matcher):
        def __init__(self,tag):
            super().__init__()
            self.tag=tag
        def fill_vocab_mask(self,mask,row): mask[row,0]=self.tag+sum(self.tokens)
        def try_accept_token(self,token):
            self.tokens.append(token)
            return True
        def rollback(self,count): del self.tokens[-count:]
    grammars=[MaskMatcher(100),MaskMatcher(200),MaskMatcher(300)]
    layout=ForwardOutputLayout(2,1,1,3)
    masks=torch.empty(9,1,dtype=torch.int32)
    candidates=torch.full((3,3),-99,dtype=torch.int32)
    if capturable:
        candidates[2]=torch.tensor([20,21,22])
        fill=load_symbol(RUNTIME+"grammar/capturable_grammar.py","_fill_current",owner="CapturableGrammarExecutor")
        executor=SimpleNamespace(max_tokens_per_req=3,bitmask_host=masks,candidates_host=candidates)
        fill(executor,dict(grammars=grammars,bs=3,has_candidates=True,completion=SimpleNamespace(output_layout=layout)))
    else:
        fill=load_symbol(RUNTIME+"grammar/capturable_grammar.py","_fill_eager_bitmask")
        monkeypatch.setattr(torch.cuda,"Event",lambda:SimpleNamespace(record=lambda:None,synchronize=lambda:None))
        buffers=SimpleNamespace(candidates_cpu_buf=candidates,vocab_mask_spec_cpu_buf=masks,
            vocab_mask_spec_buf=torch.empty_like(masks))
        fill(grammars,3,buffers,3,True,torch.tensor([1,2,3,4,5,20,21,22]),layout)
        torch.testing.assert_close(buffers.vocab_mask_spec_buf,masks)
        assert candidates[2].tolist()==[20,21,22]
    assert masks[:,0].tolist()==[100,-1,-1,-1,-1,-1,300,321,343]
    assert [g.tokens for g in grammars]==[[],[],[]]
