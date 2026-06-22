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

"""Inline detokenization receiver tests.

Drive the ``AsyncLLM._inline_detokenize_one`` helper and the
``BatchTokenIDOut`` dispatch branch. They verify:

1. Flag-off regression — ``BatchTokenIDOut`` still flows through the
   raw-token path and produces an out_dict with an empty ``text``.
2. Flag-on inline emit — out_dict gains a ``text`` key populated by
   the per-request ``IncrementalDetokenizer`` and matches the shape
   the ``BatchStrOut`` branch produces byte-for-byte.
3. Per-request lifecycle — the inline detokenizer is lazily created
   per rid, persists across frames for the same rid, and is
   independent between rids.
4. Subprocess-vs-inline text parity — for a given sequence of
   frames, the cumulative ``state.text`` accumulated through the
   inline path equals what ``incremental_decode_batch`` would emit
   character-for-character.
5. Stream vs non-stream ``output_ids`` shape, stop trimming
   pass-through, and finish-reason propagation.

A ``_StubTokenizerManager`` bypasses ZMQ / ModelConfig / HF-tokenizer
bring-up so the tests can exercise the exact production code path
without GPU or network.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from typing import Any, Dict, List, Optional

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from transformers import AutoTokenizer  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import (  # noqa: E402
    RequestOutputCollector,
)
from tokenspeed.runtime.engine.detokenizer import (  # noqa: E402
    DecodeStatus,
    IncrementalDetokenizer,
    incremental_decode_batch,
)
from tokenspeed.runtime.engine.io_struct import BatchTokenIDOut  # noqa: E402
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)

_GPT2_TOKENIZER = "gpt2"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTokenizerManager(AsyncLLM):
    """Bypass ZMQ + ModelConfig + HF bring-up for unit tests.

    We only need the pieces touched by
    ``OutputProcessor.handle_batch_output`` and
    ``_inline_detokenize_one``: ``server_args``, ``tokenizer``, the
    ``rid_to_state`` map, and a handful of flags. Everything else
    that the real ``__init__`` populates (metrics, sockets, model
    config) is untouched because these tests never reach those
    paths.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        enable_inline_detokenizer: bool = True,
        stream_output: bool = True,
        speculative_algorithm: Any = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = None
        self.rid_to_state: Dict[str, ReqState] = {}
        self.enable_metrics = False
        self.dump_requests_folder = False
        self.log_requests = False
        # Build a tiny ServerArgs-shaped object so the branch conditions in
        # ``handle_batch_output`` keep working without loading the real
        # ServerArgs dataclass (which pulls torch through ModelConfig).
        self.server_args = types.SimpleNamespace(
            enable_inline_detokenizer=enable_inline_detokenizer,
            stream_output=stream_output,
            speculative_algorithm=speculative_algorithm,
            skip_tokenizer_init=False,
        )
        # OutputProcessor holds a back-reference to this stub via
        # ``engine``.
        self.output_processor = OutputProcessor(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpt2_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(_GPT2_TOKENIZER)


def _batch_token_id_out(
    rids: List[str],
    *,
    decode_ids: List[List[int]],
    decoded_texts: Optional[List[str]] = None,
    read_offsets: Optional[List[int]] = None,
    finished_reasons: Optional[List[Optional[Dict[str, Any]]]] = None,
    no_stop_trim: Optional[List[bool]] = None,
    skip_special_tokens: Optional[List[bool]] = None,
    spaces_between_special_tokens: Optional[List[bool]] = None,
    **overrides: Any,
) -> BatchTokenIDOut:
    """Build a ``BatchTokenIDOut`` with safe defaults."""
    n = len(rids)
    defaults: Dict[str, Any] = {
        "output_ids": None,
        "output_multi_ids": None,
        "prompt_tokens": [0] * n,
        "completion_tokens": [0] * n,
        "cached_tokens": [0] * n,
        "spec_verify_ct": [0] * n,
        "input_token_logprobs_val": [],
        "input_token_logprobs_idx": [],
        "output_token_logprobs_val": [],
        "output_token_logprobs_idx": [],
        "input_top_logprobs_val": [],
        "input_top_logprobs_idx": [],
        "output_top_logprobs_val": [],
        "output_top_logprobs_idx": [],
        "input_token_ids_logprobs_val": [],
        "input_token_ids_logprobs_idx": [],
        "output_token_ids_logprobs_val": [],
        "output_token_ids_logprobs_idx": [],
        "output_hidden_states": [[] for _ in range(n)],
        "batch_accept_draft_tokens": [],
        "output_extra_infos": [],
        "generated_time": 0,
    }
    defaults.update(overrides)
    return BatchTokenIDOut(
        rids=rids,
        finished_reasons=(
            finished_reasons if finished_reasons is not None else [None] * n
        ),
        decoded_texts=decoded_texts if decoded_texts is not None else [""] * n,
        decode_ids=decode_ids,
        read_offsets=read_offsets if read_offsets is not None else [0] * n,
        skip_special_tokens=(
            skip_special_tokens if skip_special_tokens is not None else [True] * n
        ),
        spaces_between_special_tokens=(
            spaces_between_special_tokens
            if spaces_between_special_tokens is not None
            else [True] * n
        ),
        no_stop_trim=no_stop_trim if no_stop_trim is not None else [False] * n,
        **defaults,
    )


class _StubReqObj:
    """Minimal stand-in for GenerateReqInput used by ``_handle_batch_output``."""

    def __init__(
        self,
        *,
        stream: bool = True,
        return_logprob: bool = False,
        rid: str = "r1",
        log_metrics: bool = False,
    ) -> None:
        self.stream = stream
        self.return_logprob = return_logprob
        self.rid = rid
        self.log_metrics = log_metrics
        # Fields normally consumed only when return_logprob=True — provide
        # benign defaults so the attribute access never raises.
        self.top_logprobs_num = []
        self.token_ids_logprob = []
        self.return_text_in_logprobs = False


def _mk_state(*, stream: bool = True, rid: str = "r1") -> ReqState:
    return ReqState(
        RequestOutputCollector(),
        False,
        __import__("asyncio").Event(),
        _StubReqObj(stream=stream, rid=rid),
        created_time=0.0,
    )


def _register(manager: _StubTokenizerManager, state: ReqState) -> None:
    manager.rid_to_state[state.obj.rid] = state


# ---------------------------------------------------------------------------
# Flag-off regression: BatchTokenIDOut still takes the raw-token path.
# ---------------------------------------------------------------------------


class TestFlagOffRegression(unittest.TestCase):
    """Flag off → inline path stays dormant. We verify this at the receiver
    level: when a ``BatchTokenIDOut`` reaches ``_handle_batch_output`` with
    the flag off, the inline helper is never invoked and no
    ``inline_detokenizer`` is lazily created on the request state.

    (The pre-existing raw-token path for ``--skip-tokenizer-init`` requires
    ``recv_obj.output_ids`` to be populated by the scheduler; we don't
    exercise that path here — it isn't changed by this PR.)
    """

    def test_flag_off_receiver_does_not_take_inline_branch(self):
        tok = _gpt2_tokenizer()
        mgr = _StubTokenizerManager(tok, enable_inline_detokenizer=False)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        # Populate output_ids so the raw-token fallback path doesn't crash;
        # we only care that the inline branch is NOT taken.
        tokens = tok.encode("hello world")
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[tokens],
            output_ids=[tokens],
            batch_accept_draft_tokens=[1.5],
        )
        mgr.output_processor.handle_batch_output(recv)

        out = state.collector.take()
        self.assertIsNotNone(out)
        # The inline detokenizer does NOT run on this path (the assertion
        # this test exists for). What's emitted is the raw-token out_dict,
        # which since the D.1-regression hotfix carries an empty ``text``
        # key — matching the pre-D.1 BatchStrOut shape that subprocess
        # conversion used to guarantee. The state machine that would have
        # populated ``state.text`` never ran, so the value is "".
        self.assertEqual(out["text"], "")
        self.assertEqual(out["meta_info"]["accept_draft_tokens"], 1.5)
        self.assertIsNone(state.inline_detokenizer)
        self.assertEqual(state.text, "")


# ---------------------------------------------------------------------------
# Inline path basics.
# ---------------------------------------------------------------------------


class TestInlineBasicEmit(unittest.TestCase):
    """Flag-on path produces a BatchStrOut-shape out_dict with ``text``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def test_single_frame_populates_text_and_output_ids(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        source = "The quick brown fox"
        ids = self.tok.encode(source)
        # Emit all tokens as one finished frame so we can assert the
        # final text without partial-UTF-8 deferral buffering.
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[ids],
            finished_reasons=[{"type": "stop", "matched": None}],
        )
        mgr.output_processor.handle_batch_output(recv)

        out = state.collector.take()
        self.assertIn("text", out)
        self.assertEqual(out["text"], source)
        self.assertEqual(out["output_ids"], ids)
        self.assertIs(out["meta_info"]["id"], "r1")
        # Inline detokenizer instantiated and still reachable on state.
        self.assertIsInstance(state.inline_detokenizer, IncrementalDetokenizer)

    def test_second_frame_reuses_per_request_detokenizer(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        first_ids = self.tok.encode("Hello ")
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(["r1"], decode_ids=[first_ids])
        )
        det_after_first = state.inline_detokenizer
        self.assertIsNotNone(det_after_first)

        second_ids = self.tok.encode("world")
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[second_ids],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        self.assertIs(state.inline_detokenizer, det_after_first)

        # Final state.text must carry the full cumulative decoded string.
        self.assertEqual(state.text, self.tok.decode(first_ids + second_ids))

    def test_meta_info_includes_finish_and_prompt_tokens(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        ids = self.tok.encode("end.")
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[ids],
            finished_reasons=[{"type": "stop", "matched": None}],
            prompt_tokens=[7],
            completion_tokens=[len(ids)],
            cached_tokens=[0],
        )
        mgr.output_processor.handle_batch_output(recv)
        out = state.collector.take()
        self.assertEqual(out["meta_info"]["prompt_tokens"], 7)
        self.assertEqual(out["meta_info"]["completion_tokens"], len(ids))
        self.assertEqual(
            out["meta_info"]["finish_reason"], {"type": "stop", "matched": None}
        )
        self.assertTrue(state.finished)


# ---------------------------------------------------------------------------
# output_ids stream-vs-non-stream shape parity.
# ---------------------------------------------------------------------------


class TestOutputIdsShape(unittest.TestCase):
    """Inline path must match BatchStrOut branch's output_ids contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def test_stream_mode_emits_delta_output_ids(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        ids_a = self.tok.encode("foo ")
        ids_b = self.tok.encode("bar")

        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(["r1"], decode_ids=[ids_a])
        )
        out1 = state.collector.take()
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[ids_b],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        out2 = state.collector.take()

        # Deltas: first frame is ids_a, second is ids_b.
        self.assertEqual(out1["output_ids"], ids_a)
        self.assertEqual(out2["output_ids"], ids_b)
        # Cumulative state carries the full list.
        self.assertEqual(state.output_ids, ids_a + ids_b)

    def test_non_stream_mode_emits_full_cumulative_output_ids(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=False, rid="r1")
        _register(mgr, state)

        ids_a = self.tok.encode("foo ")
        ids_b = self.tok.encode("bar")
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(["r1"], decode_ids=[ids_a])
        )
        out1 = state.collector.take()
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[ids_b],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        out2 = state.collector.take()

        # Full copy every frame in non-stream mode.
        self.assertEqual(out1["output_ids"], ids_a)
        self.assertEqual(out2["output_ids"], ids_a + ids_b)


# ---------------------------------------------------------------------------
# Stop trimming passes through unchanged.
# ---------------------------------------------------------------------------


class TestStopTrimmingPassThrough(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def test_matched_string_is_trimmed_in_text(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        source = "hello STOP world"
        ids = self.tok.encode(source)
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[ids],
            finished_reasons=[{"type": "stop", "matched": "STOP"}],
        )
        mgr.output_processor.handle_batch_output(recv)
        out = state.collector.take()
        # Matched stop string and everything after must be trimmed.
        self.assertEqual(out["text"], "hello ")

    def test_no_stop_trim_flag_preserves_matched_content(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        source = "hello STOP world"
        ids = self.tok.encode(source)
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[ids],
            finished_reasons=[{"type": "stop", "matched": "STOP"}],
            no_stop_trim=[True],
        )
        mgr.output_processor.handle_batch_output(recv)
        out = state.collector.take()
        self.assertEqual(out["text"], source)

    def test_matched_int_with_no_stop_trim_preserves_last_token(self):
        # Gap fill for ``matched=int`` (stop-token) case. The two places
        # ``trim_matched_stop`` fires inside the inline branch both take a
        # different code path for ``matched=int`` than for ``matched=str``:
        #   1. ``read_ids = trim_matched_stop(s.decode_ids[surr:], finish,
        #      no_stop_trim)`` — on the id-list side, matched=int drops the
        #      last token from ``read_ids`` before ``batch_decode``, which
        #      shortens the resulting text by whatever that token contributes.
        #   2. ``trim_matched_stop(s.decoded_text + new_text, ...)`` — on
        #      the string side, matched=int falls through and returns the
        #      output unchanged.
        # When ``no_stop_trim=True``, both short-circuit to "return output
        # unchanged", so the decoded text must include every token in
        # ``decode_ids``. This test locks that interaction on the inline
        # path so a later refactor cannot accidentally collapse the
        # matched=int case into the matched=str case.
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        source = "Hello world"
        ids = self.tok.encode(source)
        self.assertGreaterEqual(len(ids), 2)

        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[ids],
            finished_reasons=[{"type": "stop", "matched": ids[-1]}],
            no_stop_trim=[True],
        )
        mgr.output_processor.handle_batch_output(recv)
        out = state.collector.take()
        # Full source preserved — matched=int + no_stop_trim means no
        # token is dropped from read_ids and no string trimming occurs.
        self.assertEqual(out["text"], self.tok.decode(ids))
        self.assertEqual(state.text, self.tok.decode(ids))


# ---------------------------------------------------------------------------
# Per-request independence.
# ---------------------------------------------------------------------------


class TestPerRequestIndependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def test_two_rids_share_one_manager_without_cross_talk(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        s_a = _mk_state(stream=True, rid="a")
        s_b = _mk_state(stream=True, rid="b")
        _register(mgr, s_a)
        _register(mgr, s_b)

        text_a = "apple pie"
        text_b = "blueberry muffin"
        ids_a = self.tok.encode(text_a)
        ids_b = self.tok.encode(text_b)

        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["a", "b"],
                decode_ids=[ids_a, ids_b],
                finished_reasons=[
                    {"type": "stop", "matched": None},
                    {"type": "stop", "matched": None},
                ],
            )
        )
        out_a = s_a.collector.take()
        out_b = s_b.collector.take()

        self.assertEqual(out_a["text"], text_a)
        self.assertEqual(out_b["text"], text_b)
        self.assertIsNot(s_a.inline_detokenizer, s_b.inline_detokenizer)


# ---------------------------------------------------------------------------
# Parity: inline text matches subprocess incremental_decode_batch output
# character-for-character for the same frame sequence.
# ---------------------------------------------------------------------------


def _run_subprocess_path(
    tokenizer: Any,
    rid: str,
    frames: List[Dict[str, Any]],
) -> List[str]:
    """Drive ``incremental_decode_batch`` with a single-request sequence.

    Returns the per-frame output_strs list.

    ``incremental_decode_batch`` aliases ``recv_obj.decode_ids[i]`` as the
    per-request ``DecodeStatus.decode_ids`` list and extends it in place on
    subsequent frames. Deep-copy each frame's ``decode_ids`` so the caller's
    frame fixture isn't corrupted across runs.
    """
    safe_frames = [
        {**frame, "decode_ids": list(frame["decode_ids"])} for frame in frames
    ]
    status: Dict[str, DecodeStatus] = {}
    pieces: List[str] = []
    for frame in safe_frames:
        recv = _batch_token_id_out(
            [rid],
            decode_ids=[frame["decode_ids"]],
            decoded_texts=[frame.get("decoded_text", "")],
            read_offsets=[frame.get("read_offset", 0)],
            finished_reasons=[frame.get("finished_reason")],
            no_stop_trim=[frame.get("no_stop_trim", False)],
        )
        out_strs = incremental_decode_batch(tokenizer, status, recv)
        pieces.append(out_strs[0])
    return pieces


def _run_inline_path(
    tokenizer: Any,
    rid: str,
    frames: List[Dict[str, Any]],
) -> List[str]:
    """Drive the inline receiver for a single request and collect the
    incremental text fragments that would have appeared in the stream.

    We read ``state.text`` before and after each frame and use the delta.
    This matches what an OpenAI streaming client would observe.

    Same deep-copy dance as ``_run_subprocess_path``: the state machine
    aliases the frame's ``decode_ids`` list on the first frame and extends
    it in place afterward, so shield the caller's fixture from mutation.
    """
    safe_frames = [
        {**frame, "decode_ids": list(frame["decode_ids"])} for frame in frames
    ]
    mgr = _StubTokenizerManager(tokenizer, enable_inline_detokenizer=True)
    state = _mk_state(stream=True, rid=rid)
    _register(mgr, state)

    pieces: List[str] = []
    prev_text = ""
    for idx, frame in enumerate(safe_frames):
        seed_text = frame.get("decoded_text", "") if idx == 0 else ""
        seed_offset = frame.get("read_offset", 0) if idx == 0 else 0
        recv = _batch_token_id_out(
            [rid],
            decode_ids=[frame["decode_ids"]],
            decoded_texts=[seed_text],
            read_offsets=[seed_offset],
            finished_reasons=[frame.get("finished_reason")],
            no_stop_trim=[frame.get("no_stop_trim", False)],
        )
        mgr.output_processor.handle_batch_output(recv)
        state.collector.take()  # drain
        pieces.append(state.text[len(prev_text) :])
        prev_text = state.text
    return pieces


class TestSubprocessVsInlineParity(unittest.TestCase):
    """For identical frame sequences the two paths must emit identical text.

    The per-frame emits are compared byte-for-byte; the cumulative text is
    compared too. Every drift between inline and subprocess behavior would
    surface here.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def _assert_parity(self, frames: List[Dict[str, Any]]) -> None:
        subp = _run_subprocess_path(self.tok, "r1", frames)
        inline = _run_inline_path(self.tok, "r1", frames)
        self.assertEqual(
            inline,
            subp,
            msg=f"inline={inline!r} subprocess={subp!r}",
        )

    def test_parity_single_frame_ascii(self):
        self._assert_parity(
            [
                {
                    "decode_ids": self.tok.encode("Hello, world!"),
                    "finished_reason": {"type": "stop", "matched": None},
                }
            ]
        )

    def test_parity_per_token_ascii(self):
        source = "streaming tokens one at a time"
        ids = self.tok.encode(source)
        frames = [{"decode_ids": [tid]} for tid in ids[:-1]]
        frames.append(
            {
                "decode_ids": [ids[-1]],
                "finished_reason": {"type": "stop", "matched": None},
            }
        )
        self._assert_parity(frames)

    def test_parity_two_frame_split(self):
        source = "hello there friend"
        ids = self.tok.encode(source)
        mid = len(ids) // 2
        self._assert_parity(
            [
                {"decode_ids": ids[:mid]},
                {
                    "decode_ids": ids[mid:],
                    "finished_reason": {"type": "stop", "matched": None},
                },
            ]
        )

    def test_parity_cjk_per_token(self):
        # CJK exercises partial-UTF-8 deferral + find_printable_text.
        source = "你好世界"
        ids = self.tok.encode(source)
        frames = [{"decode_ids": [tid]} for tid in ids[:-1]]
        frames.append(
            {
                "decode_ids": [ids[-1]],
                "finished_reason": {"type": "stop", "matched": None},
            }
        )
        self._assert_parity(frames)

    def test_parity_finish_with_matched_stop_string(self):
        source = "keep this STOP drop this"
        ids = self.tok.encode(source)
        self._assert_parity(
            [
                {
                    "decode_ids": ids,
                    "finished_reason": {"type": "stop", "matched": "STOP"},
                }
            ]
        )

    def test_parity_unfinished_streaming_does_not_trim(self):
        # Streaming (no finish yet) must NOT apply stop trimming.
        source = "prefix STOP more"
        ids = self.tok.encode(source)
        self._assert_parity(
            [
                {"decode_ids": ids[:3]},
                {"decode_ids": ids[3:]},  # no finish — still streaming
            ]
        )

    def test_parity_emoji_per_token(self):
        # 4-byte UTF-8 emoji per-token streaming. Exercises a different
        # partial-byte shape than CJK: the emoji codepoint is split
        # across ~4 byte-level BPE tokens (one per UTF-8 byte) and
        # ``find_printable_text`` has to defer every intermediate frame
        # until the full codepoint arrives. Any drift between the inline
        # path's offset bookkeeping and the subprocess path's surfaces
        # here because the defer-then-commit timing has to match
        # byte-for-byte.
        source = "a🌟b"
        ids = self.tok.encode(source)
        frames: List[Dict[str, Any]] = [{"decode_ids": [tid]} for tid in ids[:-1]]
        frames.append(
            {
                "decode_ids": [ids[-1]],
                "finished_reason": {"type": "stop", "matched": None},
            }
        )
        self._assert_parity(frames)

    def test_parity_finish_on_later_frame_with_prior_unfinished_commits(self):
        # The highest-risk state-machine branch: finish arrives on a
        # non-first frame after one or more unfinished commits have
        # already landed. On the finished frame the commit block is
        # skipped and ``trim_matched_stop`` runs on
        # ``s.decoded_text + new_text`` using the offsets accumulated
        # from prior commits. This mirrors the batch detokenizer coverage in
        # ``test_finish_arrives_on_later_frame_applies_trim_at_finish_only``
        # against the inline receiver path so any inline-specific drift shows up.
        source = "Hello world the long sentence STOP tail text"
        ids = self.tok.encode(source)
        self.assertGreaterEqual(len(ids), 6)

        # Pick the largest split whose decoded prefix is still
        # STOP-free so frame 1 commits a clean prefix.
        split: Optional[int] = None
        for candidate in range(len(ids) - 1, 0, -1):
            if "STOP" not in self.tok.decode(ids[:candidate]):
                split = candidate
                break
        self.assertIsNotNone(split, "need a STOP-free prefix split")

        self._assert_parity(
            [
                {"decode_ids": ids[:split]},  # unfinished commit
                {
                    "decode_ids": ids[split:],
                    "finished_reason": {"type": "stop", "matched": "STOP"},
                },
            ]
        )


# ---------------------------------------------------------------------------
# Seed handling from BatchTokenIDOut.decoded_texts / read_offsets.
# ---------------------------------------------------------------------------


class TestSeedHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def test_first_frame_honors_decoded_text_seed(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        seed = "<resume> "
        ids = self.tok.encode("continuation")
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[ids],
            decoded_texts=[seed],
            finished_reasons=[{"type": "stop", "matched": None}],
        )
        mgr.output_processor.handle_batch_output(recv)
        out = state.collector.take()
        self.assertEqual(out["text"], seed + self.tok.decode(ids))

    def test_only_first_frame_seeds_detokenizer(self):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_state(stream=True, rid="r1")
        _register(mgr, state)

        ids_first = self.tok.encode("A")
        ids_second = self.tok.encode("B")

        # First frame with seed.
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[ids_first],
                decoded_texts=["seed-"],
                read_offsets=[0],
            )
        )
        state.collector.take()

        # Second frame passes a misleading decoded_texts which MUST be ignored
        # because the detokenizer is already initialized.
        mgr.output_processor.handle_batch_output(
            _batch_token_id_out(
                ["r1"],
                decode_ids=[ids_second],
                decoded_texts=["ignored-"],
                read_offsets=[999],
                finished_reasons=[{"type": "stop", "matched": None}],
            )
        )
        state.collector.take()
        self.assertEqual(state.text, "seed-" + self.tok.decode(ids_first + ids_second))


# ---------------------------------------------------------------------------
# Logprob / meta field pass-through through the inline branch.
# ---------------------------------------------------------------------------


class _LogprobReqObj(_StubReqObj):
    """Request object that asks for output logprobs in a given dialect so
    ``_handle_batch_output`` invokes ``convert_logprob_style`` on the recv_obj.

    ``fmt="vllm"`` sets ``sampling_params["logprobs"]=0`` (the output processor
    renders the vLLM ``logprobs`` dict). ``fmt="sglang"`` leaves sampling_params
    without ``logprobs`` so the SGLang ``return_logprob`` flag drives the
    tuple-list rendering.
    """

    def __init__(
        self,
        *,
        rid: str = "r1",
        fmt: str = "vllm",
    ) -> None:
        super().__init__(stream=True, return_logprob=True, rid=rid)
        self.sampling_params = {"logprobs": 0} if fmt == "vllm" else {}
        self.logprob_format = None  # auto: match the request dialect


def _mk_logprob_state(*, rid: str = "r1", fmt: str = "vllm") -> ReqState:
    return ReqState(
        RequestOutputCollector(),
        False,
        __import__("asyncio").Event(),
        _LogprobReqObj(rid=rid, fmt=fmt),
        created_time=0.0,
    )


class TestInlineLogprobPassThrough(unittest.TestCase):
    """Verify the sampled-token output logprob arrays on a ``BatchTokenIDOut``
    flow through the inline branch of ``_handle_batch_output`` into ``meta_info``
    — in BOTH dialects, selected per request, from the same wire arrays.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _gpt2_tokenizer()

    def _run(self, fmt):
        mgr = _StubTokenizerManager(self.tok, enable_inline_detokenizer=True)
        state = _mk_logprob_state(rid="r1", fmt=fmt)
        _register(mgr, state)
        recv = _batch_token_id_out(
            ["r1"],
            decode_ids=[self.tok.encode("Hello")],
            finished_reasons=[{"type": "stop", "matched": None}],
            input_token_logprobs_val=[[]],
            input_token_logprobs_idx=[[]],
            output_token_logprobs_val=[[-0.5, -0.6]],
            output_token_logprobs_idx=[[1, 2]],
        )
        mgr.output_processor.handle_batch_output(recv)
        return state.collector.take()["meta_info"]

    def test_vllm_format(self):
        meta = self._run("vllm")
        # vLLM shape: "logprobs" is a list[dict[int, Logprob]], one per token.
        self.assertIn("logprobs", meta)
        self.assertNotIn("output_token_logprobs", meta)
        self.assertEqual(len(meta["logprobs"]), 2)
        self.assertEqual(meta["logprobs"][0][1].logprob, -0.5)
        self.assertEqual(meta["logprobs"][0][1].rank, 0)
        self.assertEqual(meta["logprobs"][1][2].logprob, -0.6)
        self.assertAlmostEqual(meta["cumulative_logprob"], -1.1)

    def test_sglang_format(self):
        meta = self._run("sglang")
        # SGLang shape: "output_token_logprobs" is a list of (val, idx, text)
        # tuples (text None when not decoding); no vLLM "logprobs" key.
        self.assertIn("output_token_logprobs", meta)
        self.assertNotIn("logprobs", meta)
        self.assertEqual([e[0] for e in meta["output_token_logprobs"]], [-0.5, -0.6])
        self.assertEqual([e[1] for e in meta["output_token_logprobs"]], [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
