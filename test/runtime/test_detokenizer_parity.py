"""Parity tests locking the detokenizer state machine.

These tests exercise the ``incremental_decode_batch`` state machine
against real HuggingFace tokenizers so that a future inline
``IncrementalDetokenizer`` in ``runtime/engine/`` can be cross-checked
against the same fixtures. Coverage spans the parity gates that are
testable at the detokenizer level plus every hot-path branch in
``handle_batch_token_id_out``:

* Gate 1 — same streamed text as the prior path.
* Gate 2 — ``output_ids`` semantics.
* Gate 4 — ``no_stop_trim`` behavior (matched string and matched token,
  matched=int + no_stop_trim=True, multi-stop strings, missing stop).
* Gate 5 — partial-UTF-8 deferral via ``find_printable_text`` covering
  CJK, 4-byte emoji, ZWJ emoji sequences, and NFD combining characters.
* Gate 6 — prompt/output logprob and meta scalar pass-through.

The tokenizer-sensitive tests (gates 1/5/6 on streaming behavior) are
defined on shared helpers and run against two concrete tokenizers — ``gpt2``
(byte-level BPE, OpenAI vocab) and ``Qwen/Qwen2.5-0.5B`` (Tiktoken-style
BPE with explicit CJK merges) — so that tokenizer-specific
detokenization regressions are caught.

Tokenizer-independent edge cases (``decode_grouped_batch``, eviction
errors, ``is_dummy`` flag, ``decoded_texts`` seed re-emission, stop-trim
corner cases, ``output_multi_ids`` pass-through) each live in their own
``unittest.TestCase`` subclass and use ``gpt2`` as the reference
tokenizer.

Gates 3, 7, 8, and 9 live above the detokenizer layer (raw-token mode
routing, ``stream_interval`` scheduling, abort wiring, shared-socket
dispatch) and are out of scope for this file.

The tests do not need GPU execution — only the full tokenspeed import
graph (transformers, torch, triton, etc.) — so they run on the
``runtime-1gpu`` suite for scheduling convenience. ``est_time`` is
set to 90s to account for the dual-tokenizer matrix and the added edge
case classes.
"""

import os
import sys
import unicodedata
import unittest
from typing import Any, Dict, List, Optional

# CI registration (parsed via AST, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=90, suite="runtime-1gpu")

from transformers import AutoTokenizer  # noqa: E402

from tokenspeed.runtime.engine.detokenizer import (  # noqa: E402
    DETOKENIZER_MAX_STATES,
    DecodeStatus,
    IncrementalDetokenizer,
    LimitedCapacityDict,
    incremental_decode_batch,
)
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
)

_GPT2_TOKENIZER = "gpt2"
_QWEN_TOKENIZER = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


class _StubDetokenizerManager:
    """In-test harness that drives the batch-detokenize state machine.

    There is no ``DetokenizerManager`` class in the runtime anymore
    (the subprocess wrapper was removed when the inline detokenizer
    became the only path). These parity tests still need a thin
    object that owns ``tokenizer`` + ``decode_status`` and exposes
    the three ``handle_*`` methods that the old subprocess event
    loop used to dispatch against. The methods below are verbatim
    re-creations of the former ``DetokenizerManager`` methods,
    calling the same
    ``incremental_decode_batch`` leaf function.

    The ``is_dummy`` attribute preserves the former class's unused-attribute
    contract (set by ``load_format == "dummy"``) so the lifecycle test that
    asserts it does not alter decoding still passes unchanged.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        capacity: int = DETOKENIZER_MAX_STATES,
        is_dummy: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.decode_status = LimitedCapacityDict(capacity=capacity)
        self.is_dummy = is_dummy

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOut):
        return recv_obj

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        output_strs = incremental_decode_batch(
            self.tokenizer, self.decode_status, recv_obj
        )
        return BatchStrOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.decode_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_hidden_states=recv_obj.output_hidden_states,
            batch_accept_draft_tokens=recv_obj.batch_accept_draft_tokens,
            output_extra_infos=recv_obj.output_extra_infos,
            generated_time=recv_obj.generated_time,
        )


def _batch(
    rids: List[str],
    *,
    decode_ids: List[List[int]],
    decoded_texts: Optional[List[str]] = None,
    read_offsets: Optional[List[int]] = None,
    finished_reasons: Optional[List[Optional[Dict[str, Any]]]] = None,
    no_stop_trim: Optional[List[bool]] = None,
    skip_special_tokens: Optional[List[bool]] = None,
    spaces_between_special_tokens: Optional[List[bool]] = None,
    output_multi_ids: Any = None,
    **overrides: Any,
) -> BatchTokenIDOut:
    """Build a ``BatchTokenIDOut`` with safe defaults so tests only fill
    in the fields they care about. Every pass-through field defaults to
    a neutral value and can be overridden via kwargs.
    """
    n = len(rids)
    defaults: Dict[str, Any] = {
        "output_ids": None,
        "output_multi_ids": output_multi_ids,
        "prompt_tokens": [0] * n,
        "completion_tokens": [0] * n,
        "cached_tokens": [0] * n,
        "spec_verify_ct": [0] * n,
        "input_token_logprobs_val": [0.0] * n,
        "input_token_logprobs_idx": [0] * n,
        "output_token_logprobs_val": [0.0] * n,
        "output_token_logprobs_idx": [0] * n,
        "input_top_logprobs_val": [[] for _ in range(n)],
        "input_top_logprobs_idx": [[] for _ in range(n)],
        "output_top_logprobs_val": [[] for _ in range(n)],
        "output_top_logprobs_idx": [[] for _ in range(n)],
        "input_token_ids_logprobs_val": [[] for _ in range(n)],
        "input_token_ids_logprobs_idx": [[] for _ in range(n)],
        "output_token_ids_logprobs_val": [[] for _ in range(n)],
        "output_token_ids_logprobs_idx": [[] for _ in range(n)],
        "output_hidden_states": [[] for _ in range(n)],
        "batch_accept_draft_tokens": [0.0] * n,
        "output_extra_infos": [{} for _ in range(n)],
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


def _stream_per_token(
    manager: _StubDetokenizerManager, rid: str, ids: List[int]
) -> List[str]:
    """Stream a token sequence one id at a time and collect the
    incremental output strings emitted per frame.
    """
    pieces: List[str] = []
    for tid in ids:
        out = manager.handle_batch_token_id_out(_batch([rid], decode_ids=[[tid]]))
        pieces.append(out.output_strs[0])
    return pieces


def _run_batch_path(
    tokenizer: Any,
    rid: str,
    frames: List[Dict[str, Any]],
) -> List[str]:
    """Drive ``incremental_decode_batch`` with a single-request sequence of
    frames and return the list of per-frame incremental output strings.

    Each frame is a dict with keys: ``decode_ids`` (required, list of ints),
    ``decoded_text`` (optional str for first-frame seed), ``read_offset``
    (optional int for first-frame seed), ``finished_reason`` (optional dict),
    ``no_stop_trim`` (optional bool).

    Aliasing note: ``incremental_decode_batch`` binds
    ``s.decode_ids = recv_obj.decode_ids[i]`` on the new-request branch
    and then calls ``.extend()`` on subsequent frames, which mutates
    whatever list the caller passed in. That is fine in production
    (the scheduler never reuses ``recv_obj``) but it leaks across
    helper invocations in tests where the same ``frames`` list is
    handed to both ``_run_batch_path`` and ``_run_per_request_path``.
    We deep-copy each frame's ``decode_ids`` on entry so the caller's
    ``frames`` argument is untouched when this helper returns.
    """
    isolated_frames = [
        {**frame, "decode_ids": list(frame["decode_ids"])} for frame in frames
    ]
    decode_status: Dict[str, DecodeStatus] = {}
    emits: List[str] = []
    for frame in isolated_frames:
        batch = _batch(
            [rid],
            decode_ids=[frame["decode_ids"]],
            decoded_texts=(
                [frame["decoded_text"]] if "decoded_text" in frame else None
            ),
            read_offsets=([frame["read_offset"]] if "read_offset" in frame else None),
            finished_reasons=[frame.get("finished_reason")],
            no_stop_trim=[frame.get("no_stop_trim", False)],
        )
        output_strs = incremental_decode_batch(tokenizer, decode_status, batch)
        emits.append(output_strs[0])
    return emits


def _run_per_request_path(tokenizer: Any, frames: List[Dict[str, Any]]) -> List[str]:
    """Drive ``IncrementalDetokenizer`` with the same frame sequence and
    return the list of per-frame incremental output strings.

    The first frame's optional ``decoded_text`` and ``read_offset`` keys
    are consumed at class construction time; subsequent frames must not
    supply them (the per-request class initializes seed state exactly
    once).

    Defensive deep-copy of ``decode_ids`` gives the helpers symmetric
    isolation. The per-request class's ``process`` method itself does not
    mutate the caller's list
    (``s.decode_ids.extend(new_decode_ids)`` reads from ``new_decode_ids``
    without writing back), but copying here keeps the helpers
    interchangeable regardless of invocation order.
    """
    isolated_frames = [
        {**frame, "decode_ids": list(frame["decode_ids"])} for frame in frames
    ]
    first = isolated_frames[0]
    det = IncrementalDetokenizer(
        decoded_text=first.get("decoded_text", ""),
        read_offset=first.get("read_offset", 0),
    )
    emits: List[str] = []
    for frame in isolated_frames:
        emit = det.process(
            tokenizer,
            new_decode_ids=frame["decode_ids"],
            finished_reason=frame.get("finished_reason"),
            no_stop_trim=frame.get("no_stop_trim", False),
        )
        emits.append(emit)
    return emits


# ---------------------------------------------------------------------------
# Tokenizer-sensitive tests: run against every tokenizer in the matrix
# ---------------------------------------------------------------------------


class _DetokenizerParityBase:
    """Shared tests that must pass for every tokenizer in the matrix.

    Concrete subclasses set ``tokenizer_name`` and inherit
    ``unittest.TestCase``; ``setUpClass`` loads the tokenizer exactly
    once per class so subsequent tests reuse the cached HF download.
    """

    tokenizer_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:  # type: ignore[override]
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.tokenizer_name)

    def setUp(self) -> None:
        self.manager = _StubDetokenizerManager(self.tokenizer)

    # ---- gate 1: streamed text ------------------------------------------

    def test_single_frame_roundtrips_source_text(self):
        source = "Hello world"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids])
        )
        self.assertEqual(out.output_strs, [source])

    def test_two_frame_split_reconstructs_source_text(self):
        source = "Hello world, how are you today?"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 4, "test requires at least 4 tokens")
        mid = len(ids) // 2

        out1 = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids[:mid]])
        )
        out2 = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids[mid:]])
        )

        self.assertEqual(out1.output_strs[0] + out2.output_strs[0], source)
        status = self.manager.decode_status["req-1"]
        self.assertEqual(status.decode_ids, ids)
        self.assertEqual(status.decoded_text, source)

    def test_per_token_streaming_reconstructs_source_text(self):
        source = "The quick brown fox jumps over the lazy dog."
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        pieces = _stream_per_token(self.manager, "req-1", ids)

        self.assertEqual("".join(pieces), source)

    # ---- gate 5: partial-UTF-8 deferral (CJK, emoji, combining) ---------

    def test_cjk_per_token_streaming_never_leaks_replacement_chars(self):
        # Tiktoken-family tokenizers with explicit CJK merges may tokenize
        # a short CJK phrase into fewer tokens than characters. We do not
        # assert a minimum token count — the invariant under test is that
        # whatever token stream the tokenizer produces, the detokenizer's
        # per-frame emit concatenates back to the source and no single
        # frame leaks a standalone replacement character. For tokenizers
        # that fold CJK into single tokens, the stream degenerates to a
        # full-text emit on the first frame and still satisfies both
        # invariants.
        source = "你好世界"
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        pieces = _stream_per_token(self.manager, "req-1", ids)

        self.assertEqual("".join(pieces), source)
        for i, piece in enumerate(pieces):
            self.assertNotIn(
                "\ufffd", piece, f"frame {i} leaked replacement char: {piece!r}"
            )

    def test_cjk_two_frame_split_reconstructs_source(self):
        source = "你好世界"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        mid = len(ids) // 2

        out1 = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids[:mid]])
        )
        out2 = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids[mid:]])
        )

        self.assertEqual(out1.output_strs[0] + out2.output_strs[0], source)

    def test_emoji_per_token_streaming_never_leaks_partial_bytes(self):
        # "🌟" is U+1F31F (4 bytes in UTF-8). Byte-level BPE tokenizers
        # typically split it across multiple tokens and decode the
        # partials as U+FFFD. find_printable_text must defer the
        # partial bytes until the codepoint is complete.
        source = "a🌟b"
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        pieces = _stream_per_token(self.manager, "req-1", ids)

        self.assertEqual("".join(pieces), source)
        for i, piece in enumerate(pieces):
            self.assertNotIn(
                "\ufffd",
                piece,
                f"frame {i} leaked replacement char: {piece!r}",
            )

    def test_zwj_emoji_sequence_round_trips_through_streaming(self):
        # A ZWJ emoji sequence ("family": man + ZWJ + woman + ZWJ + girl)
        # is multiple codepoints joined by U+200D. Each codepoint is
        # 4 bytes, ZWJ is 3 bytes. The concatenation of per-frame
        # emits must reconstruct the full sequence exactly.
        source = "👨\u200d👩\u200d👧"
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        pieces = _stream_per_token(self.manager, "req-1", ids)

        self.assertEqual("".join(pieces), source)

    def test_nfd_combining_character_round_trip(self):
        # "á" in NFD form is two codepoints: U+0061 (a) + U+0301
        # (combining acute). Byte-level BPE tokenizers split U+0301
        # into its two-byte UTF-8 encoding, exercising a different
        # partial-byte path than a full standalone codepoint. Some
        # tokenizers (notably Qwen / Tiktoken family) apply Unicode
        # normalization during encode/decode and return the NFC form
        # "á" (U+00E1) even when given NFD input. Compare normalized
        # forms so the assertion locks "detokenizer faithfully
        # reproduces what the tokenizer produces modulo normalization"
        # without forcing a specific normalization choice.
        source = "a\u0301"
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        pieces = _stream_per_token(self.manager, "req-1", ids)

        self.assertEqual(
            unicodedata.normalize("NFC", "".join(pieces)),
            unicodedata.normalize("NFC", source),
        )

    def test_mixed_ascii_cjk_emoji_stream_reconstructs_source(self):
        source = "Hello 你好 🌟 World"
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        pieces = _stream_per_token(self.manager, "req-1", ids)

        self.assertEqual("".join(pieces), source)
        for i, piece in enumerate(pieces):
            self.assertNotIn(
                "\ufffd",
                piece,
                f"frame {i} leaked replacement char: {piece!r}",
            )

    # ---- gate 2: output_ids pass-through --------------------------------

    def test_output_ids_field_reflects_recv_obj_decode_ids(self):
        ids = self.tokenizer.encode("Hello", add_special_tokens=False)
        batch = _batch(["req-1"], decode_ids=[ids])
        out = self.manager.handle_batch_token_id_out(batch)

        self.assertIs(out.output_ids, batch.decode_ids)
        self.assertEqual(out.output_ids, [ids])

    # ---- gate 4: stop trimming (matched string, matched token) ----------

    def test_finished_with_matched_stop_string_trims_output(self):
        source = "answer STOP trailing"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": "STOP"}],
            )
        )

        self.assertEqual(out.output_strs, ["answer "])

    def test_finished_with_matched_stop_token_drops_last_id_text(self):
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 2)

        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": ids[-1]}],
            )
        )

        expected = self.tokenizer.decode(ids[:-1])
        self.assertEqual(out.output_strs[0], expected)

    def test_no_stop_trim_true_preserves_matched_string_content(self):
        source = "answer STOP trailing"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": "STOP"}],
                no_stop_trim=[True],
            )
        )

        self.assertEqual(out.output_strs, [source])

    def test_unfinished_state_does_not_apply_stop_trim(self):
        source = "answer STOP"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[None],
            )
        )

        self.assertEqual(out.output_strs, [source])

    # ---- gate 6: logprob / meta pass-through ----------------------------

    def test_all_logprob_fields_flow_through_unchanged(self):
        ids = self.tokenizer.encode("x", add_special_tokens=False)
        batch = _batch(
            ["req-1"],
            decode_ids=[ids],
            input_token_logprobs_val=[-1.5],
            input_token_logprobs_idx=[42],
            output_token_logprobs_val=[-0.25],
            output_token_logprobs_idx=[1],
            input_top_logprobs_val=[[[-1.0, -2.0]]],
            input_top_logprobs_idx=[[[10, 20]]],
            output_top_logprobs_val=[[[-0.1, -0.2]]],
            output_top_logprobs_idx=[[[1, 2]]],
            input_token_ids_logprobs_val=[[[-3.0]]],
            input_token_ids_logprobs_idx=[[[7]]],
            output_token_ids_logprobs_val=[[[-0.3]]],
            output_token_ids_logprobs_idx=[[[1]]],
        )

        out = self.manager.handle_batch_token_id_out(batch)

        self.assertEqual(out.input_token_logprobs_val, [-1.5])
        self.assertEqual(out.input_token_logprobs_idx, [42])
        self.assertEqual(out.output_token_logprobs_val, [-0.25])
        self.assertEqual(out.output_token_logprobs_idx, [1])
        self.assertEqual(out.input_top_logprobs_val, [[[-1.0, -2.0]]])
        self.assertEqual(out.input_top_logprobs_idx, [[[10, 20]]])
        self.assertEqual(out.output_top_logprobs_val, [[[-0.1, -0.2]]])
        self.assertEqual(out.output_top_logprobs_idx, [[[1, 2]]])
        self.assertEqual(out.input_token_ids_logprobs_val, [[[-3.0]]])
        self.assertEqual(out.input_token_ids_logprobs_idx, [[[7]]])
        self.assertEqual(out.output_token_ids_logprobs_val, [[[-0.3]]])
        self.assertEqual(out.output_token_ids_logprobs_idx, [[[1]]])

    def test_meta_scalar_fields_and_extras_flow_through_unchanged(self):
        ids = self.tokenizer.encode("x", add_special_tokens=False)
        batch = _batch(
            ["req-1"],
            decode_ids=[ids],
            prompt_tokens=[11],
            completion_tokens=[7],
            cached_tokens=[3],
            spec_verify_ct=[2],
            output_hidden_states=[[0.5, 0.25]],
            batch_accept_draft_tokens=[0.75],
            output_extra_infos=[{"frame": 1}],
            generated_time=123456,
            finished_reasons=[{"type": "stop", "matched": None}],
        )

        out = self.manager.handle_batch_token_id_out(batch)

        self.assertEqual(out.rids, ["req-1"])
        self.assertEqual(out.prompt_tokens, [11])
        self.assertEqual(out.completion_tokens, [7])
        self.assertEqual(out.cached_tokens, [3])
        self.assertEqual(out.spec_verify_ct, [2])
        self.assertEqual(out.output_hidden_states, [[0.5, 0.25]])
        self.assertEqual(out.batch_accept_draft_tokens, [0.75])
        self.assertEqual(out.output_extra_infos, [{"frame": 1}])
        self.assertEqual(out.generated_time, 123456)
        self.assertEqual(out.finished_reasons, [{"type": "stop", "matched": None}])

    # ---- multi-request and lifecycle ------------------------------------

    def test_batched_requests_produce_independent_per_request_outputs(self):
        ids_1 = self.tokenizer.encode("Hello", add_special_tokens=False)
        ids_2 = self.tokenizer.encode("world", add_special_tokens=False)

        out = self.manager.handle_batch_token_id_out(
            _batch(["req-1", "req-2"], decode_ids=[ids_1, ids_2])
        )

        self.assertEqual(out.output_strs[0], "Hello")
        self.assertEqual(out.output_strs[1], "world")
        self.assertEqual(self.manager.decode_status["req-1"].decoded_text, "Hello")
        self.assertEqual(self.manager.decode_status["req-2"].decoded_text, "world")

    def test_second_frame_on_one_request_does_not_disturb_the_other(self):
        ids_a = self.tokenizer.encode("Hello", add_special_tokens=False)
        ids_b = self.tokenizer.encode("world", add_special_tokens=False)
        self.manager.handle_batch_token_id_out(
            _batch(["req-1", "req-2"], decode_ids=[ids_a, ids_b])
        )

        ids_a_more = self.tokenizer.encode(" there", add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids_a_more])
        )

        self.assertEqual(
            self.manager.decode_status["req-1"].decoded_text, "Hello there"
        )
        self.assertEqual(
            out.output_strs[0],
            self.manager.decode_status["req-1"].decoded_text[len("Hello") :],
        )
        self.assertEqual(self.manager.decode_status["req-2"].decoded_text, "world")

    def test_embedding_batch_passes_through_unchanged(self):
        recv = BatchEmbeddingOut(
            rids=["req-1", "req-2"],
            finished_reasons=[None, None],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            prompt_tokens=[5, 6],
        )

        out = self.manager.handle_batch_embedding_out(recv)

        self.assertIs(out, recv)

    # ---- multi-frame state machine branches (gap closure #1/#2/#3) -----

    def test_finish_arrives_on_later_frame_applies_trim_at_finish_only(self):
        # Gap 1. Earlier stop-trim tests put the finished request
        # into a single-frame batch. Production flow streams N unfinished
        # frames then one finished frame — a different code branch in
        # handle_batch_token_id_out that skips the commit block and
        # runs trim_matched_stop on ``s.decoded_text + new_text`` with
        # accumulated offsets. Lock that branch in here.
        source = "Hello world the long sentence STOP tail text"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 6)

        # Find the largest split index whose decoded prefix is still
        # free of "STOP". This guarantees frame 1 commits a clean
        # prefix and the stop trim is applied entirely inside the
        # finished frame's new_text contribution.
        split = None
        for candidate in range(len(ids) - 1, 0, -1):
            prefix_text = self.tokenizer.decode(ids[:candidate])
            if "STOP" not in prefix_text:
                split = candidate
                break
        self.assertIsNotNone(split, "need a clean STOP-free prefix split")

        out1 = self.manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids[:split]])
        )
        out2 = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids[split:]],
                finished_reasons=[{"type": "stop", "matched": "STOP"}],
            )
        )

        total_emitted = out1.output_strs[0] + out2.output_strs[0]
        full_decoded = self.tokenizer.decode(ids)
        expected = full_decoded[: full_decoded.find("STOP")]
        self.assertEqual(total_emitted, expected)

        # The finished frame does NOT commit to s.decoded_text. Lock
        # that semantic: the status reflects only the unfinished commit.
        status = self.manager.decode_status["req-1"]
        self.assertEqual(status.decoded_text, out1.output_strs[0])

    def test_nonzero_read_offset_seed_changes_surr_slice_on_first_frame(self):
        # Gap 2. When the scheduler sends a non-zero read_offset on a
        # new rid (resumption / reattach scenario), the new
        # DecodeStatus initializes ``s.read_offset`` from it directly
        # while ``s.surr_offset`` stays 0. That changes the first
        # frame's ``surr_ids = decode_ids[surr_offset:read_offset]``
        # slice from the usual empty list to a non-empty prefix, so
        # ``new_text = read_texts - surr_texts`` excludes the
        # already-read portion.
        source = "Hello world goodbye universe"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 3)

        initial_read_offset = 2
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                read_offsets=[initial_read_offset],
            )
        )

        surr_text = self.tokenizer.decode(ids[:initial_read_offset])
        full_text = self.tokenizer.decode(ids)
        expected_emit = full_text[len(surr_text) :]
        self.assertEqual(out.output_strs, [expected_emit])

        status = self.manager.decode_status["req-1"]
        self.assertEqual(status.decode_ids, ids)
        self.assertEqual(status.decoded_text, expected_emit)
        # After commit, surr_offset bumps to the OLD read_offset and
        # read_offset bumps to len(decode_ids).
        self.assertEqual(status.surr_offset, initial_read_offset)
        self.assertEqual(status.read_offset, len(ids))
        self.assertEqual(status.sent_offset, len(expected_emit))

    def test_offset_bookkeeping_monotonic_across_partial_utf8_stream(self):
        # Gap 3. Across per-token CJK streaming, (read_offset,
        # sent_offset) must advance monotonically and the committed
        # decoded_text must always be a prefix of the final source.
        # This catches state machine drifts that still produce the
        # right final concatenation but move offsets incorrectly on
        # intermediate frames.
        source = "你好"
        ids = self.tokenizer.encode(source, add_special_tokens=False)

        prev_read_offset = 0
        prev_sent_offset = 0
        for tid in ids:
            self.manager.handle_batch_token_id_out(
                _batch(["req-1"], decode_ids=[[tid]])
            )
            s = self.manager.decode_status["req-1"]

            self.assertGreaterEqual(s.read_offset, prev_read_offset)
            self.assertGreaterEqual(s.sent_offset, prev_sent_offset)
            self.assertTrue(
                source.startswith(s.decoded_text),
                f"decoded_text={s.decoded_text!r} not a prefix of "
                f"source={source!r}",
            )

            prev_read_offset = s.read_offset
            prev_sent_offset = s.sent_offset

        final = self.manager.decode_status["req-1"]
        self.assertEqual(final.decoded_text, source)
        self.assertEqual(final.read_offset, len(ids))

    def test_empty_decode_ids_frame_is_noop(self):
        # Gap 4. A frame with decode_ids=[[]] (empty delta) should be
        # a complete no-op: no offset movement, no text emitted, no
        # state mutation. Characterize this so the IncrementalDetokenizer
        # port cannot silently regress to e.g. raising on empty input.
        source = "Hello"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        self.manager.handle_batch_token_id_out(_batch(["req-1"], decode_ids=[ids]))
        s_before = self.manager.decode_status["req-1"]
        before = (
            s_before.decoded_text,
            s_before.surr_offset,
            s_before.read_offset,
            s_before.sent_offset,
            list(s_before.decode_ids),
        )

        out = self.manager.handle_batch_token_id_out(_batch(["req-1"], decode_ids=[[]]))

        self.assertEqual(out.output_strs, [""])
        s_after = self.manager.decode_status["req-1"]
        after = (
            s_after.decoded_text,
            s_after.surr_offset,
            s_after.read_offset,
            s_after.sent_offset,
            list(s_after.decode_ids),
        )
        self.assertEqual(before, after)

    # ---- per-request class vs batch function ---------------------------
    #
    # These tests drive the same frame sequence through
    # `IncrementalDetokenizer.process` and `incremental_decode_batch`
    # (single-request batch) and assert byte-equal per-frame emits.
    # They lock the contract that the per-request class and the batch
    # function produce identical streams.

    def test_per_request_matches_batch_single_frame_ascii(self):
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)
        frames = [{"decode_ids": ids}]

        batch_emits = _run_batch_path(self.tokenizer, "req-1", frames)
        per_req_emits = _run_per_request_path(self.tokenizer, frames)

        self.assertEqual(batch_emits, per_req_emits)

    def test_per_request_matches_batch_two_frame_split(self):
        source = "Hello world, how are you today?"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 4)
        mid = len(ids) // 2
        frames = [
            {"decode_ids": ids[:mid]},
            {"decode_ids": ids[mid:]},
        ]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_per_token_ascii(self):
        source = "The quick brown fox jumps over the lazy dog."
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        frames = [{"decode_ids": [tid]} for tid in ids]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_cjk_per_token(self):
        source = "你好世界"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        frames = [{"decode_ids": [tid]} for tid in ids]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_emoji_per_token(self):
        source = "a🌟b"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        frames = [{"decode_ids": [tid]} for tid in ids]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_finish_with_matched_stop_string(self):
        source = "answer STOP trailing"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        frames = [
            {
                "decode_ids": ids,
                "finished_reason": {"type": "stop", "matched": "STOP"},
            }
        ]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_finish_on_later_frame(self):
        source = "Hello world the long sentence STOP tail text"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 6)

        # Pick the largest split where the decoded prefix still lacks
        # "STOP" so frame 1 commits a clean prefix.
        split = None
        for candidate in range(len(ids) - 1, 0, -1):
            if "STOP" not in self.tokenizer.decode(ids[:candidate]):
                split = candidate
                break
        self.assertIsNotNone(split)

        frames = [
            {"decode_ids": ids[:split]},
            {
                "decode_ids": ids[split:],
                "finished_reason": {"type": "stop", "matched": "STOP"},
            },
        ]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_no_stop_trim(self):
        source = "answer STOP trailing"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        frames = [
            {
                "decode_ids": ids,
                "finished_reason": {"type": "stop", "matched": "STOP"},
                "no_stop_trim": True,
            }
        ]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_decoded_text_seed(self):
        ids = self.tokenizer.encode(" world", add_special_tokens=False)
        frames = [
            {
                "decode_ids": ids,
                "decoded_text": "Hello,",
            }
        ]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )

    def test_per_request_matches_batch_nonzero_read_offset_seed(self):
        ids = self.tokenizer.encode(
            "Hello world goodbye universe", add_special_tokens=False
        )
        self.assertGreaterEqual(len(ids), 3)
        frames = [
            {
                "decode_ids": ids,
                "read_offset": 2,
            }
        ]

        self.assertEqual(
            _run_batch_path(self.tokenizer, "req-1", frames),
            _run_per_request_path(self.tokenizer, frames),
        )


# Concrete tokenizer matrix.


class TestGpt2DetokenizerParity(_DetokenizerParityBase, unittest.TestCase):
    tokenizer_name = _GPT2_TOKENIZER


class TestQwen2DetokenizerParity(_DetokenizerParityBase, unittest.TestCase):
    tokenizer_name = _QWEN_TOKENIZER


# ---------------------------------------------------------------------------
# Gap 2: decode_grouped_batch path
# ---------------------------------------------------------------------------


class TestDetokenizerGroupedBatch(unittest.TestCase):
    """Force the ``all_same=False`` branch that routes through
    ``decode_grouped_batch`` instead of ``tokenizer.batch_decode`` on
    the full batch. This branch activates whenever requests in one
    batch disagree on ``skip_special_tokens`` or
    ``spaces_between_special_tokens``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(_GPT2_TOKENIZER)

    def setUp(self) -> None:
        self.manager = _StubDetokenizerManager(self.tokenizer)

    def test_mixed_skip_special_tokens_activates_grouped_path(self):
        ids_a = self.tokenizer.encode("Hello", add_special_tokens=False)
        ids_b = self.tokenizer.encode("world", add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1", "req-2"],
                decode_ids=[ids_a, ids_b],
                skip_special_tokens=[True, False],
            )
        )

        self.assertEqual(out.output_strs, ["Hello", "world"])
        self.assertEqual(self.manager.decode_status["req-1"].decoded_text, "Hello")
        self.assertEqual(self.manager.decode_status["req-2"].decoded_text, "world")

    def test_mixed_spaces_between_special_tokens_activates_grouped_path(self):
        ids_a = self.tokenizer.encode("Hello", add_special_tokens=False)
        ids_b = self.tokenizer.encode("world", add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1", "req-2"],
                decode_ids=[ids_a, ids_b],
                spaces_between_special_tokens=[True, False],
            )
        )

        self.assertEqual(out.output_strs, ["Hello", "world"])

    def test_grouped_path_preserves_per_request_ordering(self):
        # Three requests with alternating skip_special_tokens settings.
        # Grouped-decode partitions by (skip, spaces) and must put each
        # result back in its original position so output_strs[i] still
        # matches decode_ids[i].
        ids_1 = self.tokenizer.encode("alpha", add_special_tokens=False)
        ids_2 = self.tokenizer.encode("beta", add_special_tokens=False)
        ids_3 = self.tokenizer.encode("gamma", add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1", "req-2", "req-3"],
                decode_ids=[ids_1, ids_2, ids_3],
                skip_special_tokens=[True, False, True],
            )
        )

        self.assertEqual(out.output_strs, ["alpha", "beta", "gamma"])


# ---------------------------------------------------------------------------
# Gaps 3 and 8: stop trimming corner cases
# ---------------------------------------------------------------------------


class TestDetokenizerStopEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(_GPT2_TOKENIZER)

    def setUp(self) -> None:
        self.manager = _StubDetokenizerManager(self.tokenizer)

    def test_no_stop_trim_true_with_matched_int_preserves_last_token(self):
        # Gap 3. The production code's isinstance check distinguishes
        # matched=str (string trim) from matched=int (last-id drop).
        # no_stop_trim=True must short-circuit BOTH branches and keep
        # the full sequence untouched.
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)
        self.assertGreaterEqual(len(ids), 2)

        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": ids[-1]}],
                no_stop_trim=[True],
            )
        )

        expected = self.tokenizer.decode(ids)
        self.assertEqual(out.output_strs, [expected])

    def test_multiple_stop_strings_in_text_only_trims_the_matched_one(self):
        # Gap 8. trim_matched_stop has a literal
        # "Current limitation: handle the case where multiple stop strs are
        # hit" — lock in the current single-stop behavior so this gap
        # cannot be silently regressed. Only the matched stop string is
        # trimmed; every other stop-looking substring is preserved.
        source = "answer STOP1 middle STOP2 tail"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": "STOP2"}],
            )
        )

        # Output cuts at the first occurrence of "STOP2" only; the
        # earlier "STOP1" survives untouched.
        self.assertEqual(out.output_strs, ["answer STOP1 middle "])

    def test_matched_string_not_present_in_decoded_text_is_noop(self):
        source = "pure answer"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "stop", "matched": "STOP"}],
            )
        )

        # trim_matched_stop returns output unchanged when
        # output.find(matched) == -1.
        self.assertEqual(out.output_strs, [source])

    def test_finished_reason_without_matched_key_is_noop(self):
        source = "pure answer"
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                finished_reasons=[{"type": "length"}],
            )
        )

        self.assertEqual(out.output_strs, [source])


# ---------------------------------------------------------------------------
# Gap 4: raw-token path / output_multi_ids pass-through
# ---------------------------------------------------------------------------


class TestDetokenizerRawTokenPath(unittest.TestCase):
    """BatchTokenIDOut.output_multi_ids is populated in the raw-token
    path (scheduler → tokenizer_manager direct, bypassing detokenizer).
    The detokenizer must still accept a BatchTokenIDOut that has this
    field set without crashing — even though BatchStrOut itself has no
    ``output_multi_ids`` field, so there is nowhere for the value to be
    surfaced from this layer. The real raw-token contract is tested
    above this layer (TokenizerManager / AsyncLLM).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(_GPT2_TOKENIZER)

    def setUp(self) -> None:
        self.manager = _StubDetokenizerManager(self.tokenizer)

    def test_output_multi_ids_on_input_does_not_affect_detokenized_text(self):
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)
        out = self.manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                output_multi_ids=[[101, 102], [103, 104]],
            )
        )

        # Detokenized text is unaffected by the raw-token payload.
        self.assertEqual(out.output_strs, ["Hello world"])
        # BatchStrOut does not expose output_multi_ids by design.
        self.assertFalse(hasattr(out, "output_multi_ids"))


# ---------------------------------------------------------------------------
# Gaps 5, 7, 9: lifecycle / contract edge cases
# ---------------------------------------------------------------------------


class TestDetokenizerLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(_GPT2_TOKENIZER)

    def test_missing_decode_status_raises_descriptive_runtime_error_on_eviction(
        self,
    ):
        # Gap 5. LimitedCapacityDict with capacity=2 + a 3-request batch
        # forces the first rid to be evicted while the third is being
        # assigned in the first loop. The second loop's direct dict
        # lookup then KeyErrors and handle_batch_token_id_out must
        # surface a RuntimeError pointing operators at
        # TOKENSPEED_DETOKENIZER_MAX_STATES.
        manager = _StubDetokenizerManager(self.tokenizer, capacity=2)
        ids_1 = self.tokenizer.encode("alpha", add_special_tokens=False)
        ids_2 = self.tokenizer.encode("beta", add_special_tokens=False)
        ids_3 = self.tokenizer.encode("gamma", add_special_tokens=False)

        with self.assertRaises(RuntimeError) as cm:
            manager.handle_batch_token_id_out(
                _batch(
                    ["req-1", "req-2", "req-3"],
                    decode_ids=[ids_1, ids_2, ids_3],
                )
            )

        message = str(cm.exception)
        self.assertIn("TOKENSPEED_DETOKENIZER_MAX_STATES", message)
        self.assertIn("req-1", message)

    def test_is_dummy_flag_does_not_alter_decoding_output(self):
        # Gap 7. The production code sets self.is_dummy from
        # server_args.load_format but handle_batch_token_id_out never
        # reads it. Lock in that contract so a future refactor does not
        # accidentally make is_dummy a behavioral switch at this layer.
        dummy_manager = _StubDetokenizerManager(self.tokenizer, is_dummy=True)
        live_manager = _StubDetokenizerManager(self.tokenizer, is_dummy=False)
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)

        dummy_out = dummy_manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids])
        )
        live_out = live_manager.handle_batch_token_id_out(
            _batch(["req-1"], decode_ids=[ids])
        )

        self.assertEqual(dummy_out.output_strs, live_out.output_strs)
        self.assertEqual(dummy_out.output_strs, ["Hello world"])

    def test_decoded_texts_seed_is_reemited_on_first_frame(self):
        # Gap 9. When recv_obj.decoded_texts[i] is non-empty on the
        # first frame for a new rid, DecodeStatus is initialized with
        # that seed but sent_offset stays at 0 — the default. The first
        # frame's incremental emit therefore contains the seed prefix
        # plus whatever the new tokens decode to. This test
        # characterizes that behavior so that any future change to
        # the contract has to explicitly update this test.
        manager = _StubDetokenizerManager(self.tokenizer)
        ids = self.tokenizer.encode(" world", add_special_tokens=False)
        seed = "Hello,"

        out = manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
                decoded_texts=[seed],
            )
        )

        status = manager.decode_status["req-1"]
        self.assertEqual(status.decoded_text, seed + " world")
        # Current behavior: the full committed text (seed + new) is
        # emitted on the first frame because sent_offset starts at 0.
        self.assertEqual(out.output_strs, [seed + " world"])

    def test_empty_logprob_arrays_pass_through_unchanged(self):
        # Gap 5. Empty logprob arrays (which the scheduler can emit
        # once the request has stopped producing new logprob values)
        # must pass through the detokenizer without being coerced or
        # dropped. BatchStrOut keeps the same empty lists so the
        # consumer's merge policy can rely on their identity.
        manager = _StubDetokenizerManager(self.tokenizer)
        ids = manager.tokenizer.encode("Hello", add_special_tokens=False)

        out = manager.handle_batch_token_id_out(
            _batch(
                ["req-1"],
                decode_ids=[ids],
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
            )
        )

        self.assertEqual(out.input_token_logprobs_val, [])
        self.assertEqual(out.input_token_logprobs_idx, [])
        self.assertEqual(out.output_token_logprobs_val, [])
        self.assertEqual(out.output_token_logprobs_idx, [])
        self.assertEqual(out.input_top_logprobs_val, [])
        self.assertEqual(out.input_top_logprobs_idx, [])
        self.assertEqual(out.output_top_logprobs_val, [])
        self.assertEqual(out.output_top_logprobs_idx, [])
        self.assertEqual(out.input_token_ids_logprobs_val, [])
        self.assertEqual(out.input_token_ids_logprobs_idx, [])
        self.assertEqual(out.output_token_ids_logprobs_val, [])
        self.assertEqual(out.output_token_ids_logprobs_idx, [])

    def test_limited_capacity_dict_update_does_not_evict(self):
        # Regression for the update-at-capacity eviction bug caught by
        # code review. LimitedCapacityDict must only evict when inserting
        # a *new* key at capacity; updating an existing
        # key is size-preserving and must never drop the oldest entry.
        # Production detokenizer code writes via
        # `self.decode_status[rid] = s` only on the new-request branch,
        # so this path is dormant in practice, but the contract is
        # defensive against any future caller that uses the dict as a
        # pure KV store with overwrites.
        d = LimitedCapacityDict(capacity=2)
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(len(d), 2)
        self.assertEqual(list(d.keys()), ["a", "b"])

        # Update existing key at capacity: no eviction.
        d["b"] = 20
        self.assertEqual(len(d), 2)
        self.assertEqual(list(d.keys()), ["a", "b"])
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 20)

        # Update the oldest key: still no eviction.
        d["a"] = 10
        self.assertEqual(len(d), 2)
        self.assertEqual(d["a"], 10)
        self.assertEqual(d["b"], 20)

        # Inserting a genuinely new key at capacity evicts the oldest —
        # this is the add-new-key path that the production flow uses,
        # and it is unchanged by the fix.
        d["c"] = 3
        self.assertEqual(len(d), 2)
        self.assertIn("c", d)
        self.assertEqual(d["c"], 3)
        # "a" was the oldest (after the update above moved "b" to end?
        # No — update of an existing key in OrderedDict does NOT move
        # it to the end. Insertion order is ["a", "b"] throughout, so
        # "a" is still the oldest and gets evicted when "c" is added).
        self.assertNotIn("a", d)
        self.assertIn("b", d)

    def test_tokenizer_decode_error_propagates_unchanged(self):
        # Gap 7. If the underlying tokenizer's batch_decode raises,
        # handle_batch_token_id_out must let the exception propagate
        # without catching, wrapping, or suppressing it. Silently
        # swallowing a tokenizer failure would mask production bugs.
        class _RaisingTokenizer:
            def batch_decode(self, *args: Any, **kwargs: Any) -> List[str]:
                raise RuntimeError("tokenizer exploded")

        manager = _StubDetokenizerManager(_RaisingTokenizer())

        with self.assertRaises(RuntimeError) as cm:
            manager.handle_batch_token_id_out(_batch(["req-1"], decode_ids=[[1, 2, 3]]))
        self.assertIn("tokenizer exploded", str(cm.exception))


# ---------------------------------------------------------------------------
# Per-request IncrementalDetokenizer construction
# ---------------------------------------------------------------------------


class TestIncrementalDetokenizerConstruction(unittest.TestCase):
    """Tokenizer-independent tests for the per-request class's init
    and state accessor. The matrix-based cross-check tests on
    ``_DetokenizerParityBase`` already validate end-to-end semantics;
    these tests lock the class's own API shape.
    """

    def test_default_init_starts_with_empty_state(self):
        det = IncrementalDetokenizer()
        s = det.status

        self.assertEqual(s.decoded_text, "")
        self.assertEqual(s.decode_ids, [])
        self.assertEqual(s.surr_offset, 0)
        self.assertEqual(s.read_offset, 0)
        self.assertEqual(s.sent_offset, 0)

    def test_init_accepts_decoded_text_seed(self):
        det = IncrementalDetokenizer(decoded_text="Hello,")
        s = det.status

        self.assertEqual(s.decoded_text, "Hello,")
        self.assertEqual(s.decode_ids, [])
        self.assertEqual(s.surr_offset, 0)
        self.assertEqual(s.read_offset, 0)
        self.assertEqual(s.sent_offset, 0)

    def test_init_accepts_nonzero_read_offset_seed(self):
        det = IncrementalDetokenizer(read_offset=7)
        s = det.status

        self.assertEqual(s.decoded_text, "")
        self.assertEqual(s.decode_ids, [])
        self.assertEqual(s.surr_offset, 0)
        self.assertEqual(s.read_offset, 7)
        self.assertEqual(s.sent_offset, 0)

    def test_status_property_exposes_live_state_not_copy(self):
        # The status property must return the underlying DecodeStatus
        # so callers can inspect state without copying. Mutations
        # through process() must be visible on subsequent .status
        # reads.
        det = IncrementalDetokenizer()
        first_ref = det.status
        second_ref = det.status
        self.assertIs(first_ref, second_ref)

    def test_each_instance_owns_independent_state(self):
        det_a = IncrementalDetokenizer(decoded_text="A")
        det_b = IncrementalDetokenizer(decoded_text="B")

        self.assertEqual(det_a.status.decoded_text, "A")
        self.assertEqual(det_b.status.decoded_text, "B")
        self.assertIsNot(det_a.status, det_b.status)
        self.assertIsNot(det_a.status.decode_ids, det_b.status.decode_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
