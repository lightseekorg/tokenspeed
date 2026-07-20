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

"""Tests for the ``return_token_ids`` request flag.

Covers the two halves of the engine-side plumbing that lets RL trainers
reuse the rollout's exact token sequence instead of re-tokenizing the
returned text (avoiding retokenization drift):

1. Request contract (``GenerateReqInput``): the flag defaults off, is
   accepted through ``normalize_batch_and_arguments``, and propagates to
   the per-sample sub-requests produced by ``__getitem__`` (batch / n>1
   fan-out).
2. Output plumbing (``OutputProcessor.handle_batch_output``): when the
   flag is set, ``meta_info`` gains ``prompt_token_ids`` (the tokenizer-
   valid, unpadded prompt ids captured at send time) and
   ``output_token_ids`` (the full cumulative generated sequence). When it
   is off, neither key appears.

A ``_StubTokenizerManager`` bypasses ZMQ / ModelConfig / HF bring-up so
the output-plumbing assertions run the exact production code path without
a GPU or network.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from typing import Any, Dict, List, Optional

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from transformers import AutoTokenizer  # noqa: E402

from tokenspeed.runtime.engine.collector import (  # noqa: E402
    RequestOutputCollector,
)
from tokenspeed.runtime.engine.detokenizer import (  # noqa: E402
    IncrementalDetokenizer,
)
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    BatchTokenIDOut,
    GenerateReqInput,
)
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)

_GPT2_TOKENIZER = "gpt2"


# ---------------------------------------------------------------------------
# Request-contract tests (no tokenizer / GPU needed).
# ---------------------------------------------------------------------------


class TestGenerateReqInputContract(unittest.TestCase):
    def test_flag_defaults_off(self):
        obj = GenerateReqInput(text="hello")
        self.assertFalse(obj.return_token_ids)

    def test_flag_survives_normalize_single(self):
        obj = GenerateReqInput(text="hello", return_token_ids=True)
        obj.normalize_batch_and_arguments()
        self.assertTrue(obj.is_single)
        self.assertTrue(obj.return_token_ids)

    def test_flag_propagates_to_batch_subrequests(self):
        obj = GenerateReqInput(
            text=["a", "b", "c"],
            return_token_ids=True,
        )
        obj.normalize_batch_and_arguments()
        self.assertFalse(obj.is_single)
        for i in range(3):
            sub = obj[i]
            self.assertTrue(
                sub.return_token_ids,
                msg=f"sub-request {i} lost return_token_ids",
            )


# ---------------------------------------------------------------------------
# Output-plumbing stubs (mirror test_inline_detokenizer_receiver.py).
# ---------------------------------------------------------------------------


class _StubTokenizerManager:
    """Minimal AsyncLLM stand-in for ``OutputProcessor.handle_batch_output``."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.processor = None
        self.rid_to_state: Dict[str, ReqState] = {}
        self.enable_metrics = False
        self.dump_requests_folder = False
        self.log_requests = False
        self.server_args = types.SimpleNamespace(
            enable_inline_detokenizer=True,
            stream_output=True,
            speculative_algorithm=None,
            skip_tokenizer_init=False,
        )
        self.output_processor = OutputProcessor(self)


class _StubReqObj:
    """Minimal GenerateReqInput stand-in read by handle_batch_output."""

    def __init__(
        self,
        *,
        stream: bool = False,
        return_token_ids: bool = False,
        rid: str = "r1",
    ) -> None:
        self.stream = stream
        self.return_logprob = False
        self.return_token_ids = return_token_ids
        self.rid = rid
        self.log_metrics = False
        self.top_logprobs_num = []
        self.token_ids_logprob = []
        self.return_text_in_logprobs = False


def _batch_token_id_out(
    rids: List[str],
    *,
    decode_ids: List[List[int]],
    finished_reasons: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> BatchTokenIDOut:
    n = len(rids)
    return BatchTokenIDOut(
        rids=rids,
        finished_reasons=(
            finished_reasons if finished_reasons is not None else [None] * n
        ),
        decoded_texts=[""] * n,
        decode_ids=decode_ids,
        read_offsets=[0] * n,
        skip_special_tokens=[True] * n,
        spaces_between_special_tokens=[True] * n,
        no_stop_trim=[False] * n,
        output_ids=None,
        output_multi_ids=None,
        prompt_tokens=[0] * n,
        completion_tokens=[0] * n,
        cached_tokens=[0] * n,
        spec_verify_ct=[0] * n,
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[[] for _ in range(n)],
        batch_accept_draft_tokens=[],
        output_extra_infos=[],
        generated_time=0,
    )


def _mk_state(
    *,
    return_token_ids: bool,
    prompt_token_ids: Optional[List[int]] = None,
    rid: str = "r1",
) -> ReqState:
    state = ReqState(
        RequestOutputCollector(),
        False,
        asyncio.Event(),
        _StubReqObj(stream=False, return_token_ids=return_token_ids, rid=rid),
        created_time=0.0,
    )
    if prompt_token_ids is not None:
        state.prompt_token_ids = prompt_token_ids
    return state


class TestReturnTokenIdsPlumbing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = AutoTokenizer.from_pretrained(_GPT2_TOKENIZER)

    def test_flag_on_adds_prompt_and_output_token_ids(self):
        mgr = _StubTokenizerManager(self.tok)
        prompt_ids = self.tok.encode("What is the capital of France?")
        state = _mk_state(return_token_ids=True, prompt_token_ids=prompt_ids)
        mgr.rid_to_state[state.obj.rid] = state

        gen_ids = self.tok.encode(" Paris.")
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[gen_ids],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        out = state.collector.take()
        meta = out["meta_info"]

        self.assertEqual(meta["prompt_token_ids"], prompt_ids)
        # output_token_ids is the full cumulative generated sequence.
        self.assertEqual(meta["output_token_ids"], gen_ids)
        # And it matches the top-level output_ids for a single finished frame.
        self.assertEqual(meta["output_token_ids"], out["output_ids"])
        # prompt_token_ids must be a copy, not an alias of the state list.
        self.assertIsNot(meta["prompt_token_ids"], state.prompt_token_ids)

    def test_flag_off_omits_token_id_keys(self):
        mgr = _StubTokenizerManager(self.tok)
        state = _mk_state(return_token_ids=False, prompt_token_ids=[1, 2, 3])
        mgr.rid_to_state[state.obj.rid] = state

        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[self.tok.encode("hi")],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        meta = state.collector.take()["meta_info"]
        self.assertNotIn("prompt_token_ids", meta)
        self.assertNotIn("output_token_ids", meta)

    def test_output_token_ids_accumulate_across_frames(self):
        mgr = _StubTokenizerManager(self.tok)
        prompt_ids = self.tok.encode("prefix")
        state = _mk_state(return_token_ids=True, prompt_token_ids=prompt_ids)
        mgr.rid_to_state[state.obj.rid] = state

        ids_a = self.tok.encode("foo ")
        ids_b = self.tok.encode("bar")
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(["r1"], decode_ids=[ids_a])
        )
        first = state.collector.take()["meta_info"]
        # Non-stream: cumulative after first frame is just ids_a.
        self.assertEqual(first["output_token_ids"], ids_a)

        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[ids_b],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        final = state.collector.take()["meta_info"]
        self.assertEqual(final["prompt_token_ids"], prompt_ids)
        self.assertEqual(final["output_token_ids"], ids_a + ids_b)

    def test_inline_detokenizer_still_runs_with_flag(self):
        # The flag must not disturb the normal text detokenization path.
        mgr = _StubTokenizerManager(self.tok)
        state = _mk_state(return_token_ids=True, prompt_token_ids=[1])
        mgr.rid_to_state[state.obj.rid] = state
        source = "The quick brown fox"
        ids = self.tok.encode(source)
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        out = state.collector.take()
        self.assertEqual(out["text"], source)
        self.assertIsInstance(state.inline_detokenizer, IncrementalDetokenizer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
