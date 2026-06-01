"""Unit tests for compute_log_probs pure helpers (CPU, no GPU required)."""

from __future__ import annotations

import math
import os

import pytest

from tokenspeed.runtime.engine.compute_log_probs import (
    InvalidSequenceError,
    build_score_kwargs,
    compute_log_probs_core,
    extract_completion_logprobs,
    validate_sequence,
)

# ---------------------------------------------------------------------------
# build_score_kwargs tests
# ---------------------------------------------------------------------------


def test_build_score_kwargs_shape():
    kw = build_score_kwargs([1, 2, 3, 4], [5, 6, 7], temperature=1.0)
    assert kw["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert kw["sampling_params"] == {"max_new_tokens": 0, "temperature": 1.0}
    assert kw["return_logprob"] is True
    assert kw["logprob_start_len"] == 3  # len(prompt) - 1 (score from preceding token)


def test_build_score_kwargs_rejects_empty_prompt():
    with pytest.raises(InvalidSequenceError):
        build_score_kwargs([], [5, 6], temperature=1.0)


def test_build_score_kwargs_rejects_empty_completion():
    with pytest.raises(InvalidSequenceError):
        build_score_kwargs([1, 2], [], temperature=1.0)


def test_build_score_kwargs_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        build_score_kwargs([1, 2], [3], temperature=0.0)


def test_build_score_kwargs_rejects_negative_temperature():
    with pytest.raises(ValueError):
        build_score_kwargs([1, 2], [3], temperature=-0.5)


# ---------------------------------------------------------------------------
# validate_sequence direct tests
# ---------------------------------------------------------------------------


def test_validate_sequence_rejects_empty_prompt():
    with pytest.raises(InvalidSequenceError):
        validate_sequence([], [1])


def test_validate_sequence_rejects_empty_completion():
    with pytest.raises(InvalidSequenceError):
        validate_sequence([1], [])


def test_validate_sequence_accepts_nonempty():
    # Should not raise.
    validate_sequence([1], [2])


# ---------------------------------------------------------------------------
# extract_completion_logprobs tests
# ---------------------------------------------------------------------------


def _meta(entries):
    # entries: list of (logprob, token_id, text-or-None) as produced by
    # LogprobsProcessor.detokenize_logprob_tokens (engine/logprobs.py:148).
    return {"input_token_logprobs": entries}


def test_extract_completion_logprobs_happy_path():
    meta = _meta([(-0.12, 5, None), (-0.47, 6, None), (-0.31, 7, None)])
    log_probs, tokens = extract_completion_logprobs(meta, num_completion=3)
    assert log_probs == [-0.12, -0.47, -0.31]
    assert tokens == [5, 6, 7]


def test_extract_completion_logprobs_length_mismatch_raises():
    meta = _meta([(-0.12, 5, None), (-0.47, 6, None)])
    with pytest.raises(ValueError):
        extract_completion_logprobs(meta, num_completion=3)


def test_extract_completion_logprobs_missing_key_raises():
    with pytest.raises(ValueError):
        extract_completion_logprobs({}, num_completion=3)


# ---------------------------------------------------------------------------
# compute_log_probs_core tests
# ---------------------------------------------------------------------------


def _fake_generate_factory():
    """Returns a generate_fn that echoes deterministic logprobs per sequence,
    so we can assert ordering and slicing without a GPU."""

    def generate_fn(*, input_ids, sampling_params, return_logprob, logprob_start_len):
        # Mirror the engine (verified on B200): input_token_logprobs[k] is the
        # logprob of the token at position logprob_start_len+1+k (one-position
        # shift), followed by one trailing sampled-position entry (target -1).
        targets = input_ids[logprob_start_len + 1 :] + [-1]
        entries = [(-0.1 * (i + 1), tok, None) for i, tok in enumerate(targets)]
        return {"meta_info": {"input_token_logprobs": entries}}

    return generate_fn


def test_core_multi_sequence_ordering_and_slicing():
    seqs = [
        {"prompt_token_ids": [1, 2, 3], "completion_token_ids": [4, 5]},
        {"prompt_token_ids": [9], "completion_token_ids": [8, 7, 6]},
    ]
    out = compute_log_probs_core(seqs, _fake_generate_factory(), temperature=1.0)
    assert out["tokens"] == [[4, 5], [8, 7, 6]]
    assert out["log_probs"][0] == [-0.1, -0.2]
    assert out["log_probs"][1] == pytest.approx([-0.1, -0.2, -0.3])


def test_core_empty_sequences_returns_empty():
    out = compute_log_probs_core([], _fake_generate_factory(), temperature=1.0)
    assert out == {"log_probs": [], "tokens": []}


def test_core_rejects_non_unit_temperature():
    seqs = [{"prompt_token_ids": [1], "completion_token_ids": [2]}]
    with pytest.raises(NotImplementedError):
        compute_log_probs_core(seqs, _fake_generate_factory(), temperature=2.0)


# ---------------------------------------------------------------------------
# GPU integration (deferred lane). Skipped unless TOKENSPEED_RUN_GPU_TESTS=1 on a
# GPU box. Verifies the end-to-end Engine.compute_log_probs path: shape, ordering,
# and that returned values are valid log-probabilities.
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_RUN_GPU_TESTS") != "1",
    reason="set TOKENSPEED_RUN_GPU_TESTS=1 on a GPU box to run",
)

# Engine config for deterministic prefill-only scoring. The default `mha`
# attention backend cannot serve a mixed (prefill+decode) batch eagerly, and the
# captured-CUDA-graph decode path it falls back to rejects multi-token query
# shapes — both of which the scoring path (and chunked prefill in particular)
# hit. flashinfer + eager + no-overlap keeps scoring on a pure-extend path that
# every backend handles. Validated on B200 (Qwen2-1.5B-Instruct). Overridable via
# TOKENSPEED_TEST_ATTN_BACKEND for other GPUs/models.
_SCORING_ENGINE_KWARGS = {
    "attention_backend": os.environ.get("TOKENSPEED_TEST_ATTN_BACKEND", "flashinfer"),
    "enforce_eager": True,
    "disable_overlap_schedule": True,
    "log_level": "error",
}


@requires_gpu
def test_compute_log_probs_end_to_end():
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=os.environ.get("TOKENSPEED_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        **_SCORING_ENGINE_KWARGS,
    )
    try:
        seqs = [
            {"prompt_token_ids": [1, 2, 3, 4], "completion_token_ids": [5, 6, 7]},
            {"prompt_token_ids": [10, 11], "completion_token_ids": [12]},
        ]
        out = engine.compute_log_probs(seqs, temperature=1.0)
        assert out["tokens"] == [[5, 6, 7], [12]]
        assert len(out["log_probs"]) == 2
        assert len(out["log_probs"][0]) == 3 and len(out["log_probs"][1]) == 1
        for row in out["log_probs"]:
            for lp in row:
                assert lp <= 0.0 and math.isfinite(lp)  # valid log-probabilities
    finally:
        engine.shutdown()


@requires_gpu
def test_compute_log_probs_long_sequence_chunked_matches_single_chunk():
    """Regression: scoring a sequence whose logprob window spans >1 prefill chunk.

    ``logprob_start_len = len(prompt) - 1``, so the scored window covers the
    last prompt token plus every completion token. When a long prompt+completion
    is split across prefill chunks, that window straddles a chunk boundary. The
    original wiring dropped every chunk after the first in-window one — the C++
    scheduler set ``extend_logprob_start_len = -1`` for ``rel < 0`` and the
    output processor discarded non-final prefill chunks — so long sequences
    returned too few logprobs (ValueError) instead of scoring.

    We pin the prompt length to a multiple of a small ``chunked_prefill_size`` so
    the completion begins exactly on a chunk boundary (guaranteeing the window
    crosses it), then assert the chunked result matches the single-chunk path
    (the configuration validated on B200) within bf16 reduction-order tolerance.
    """
    from tokenspeed.runtime.entrypoints.engine import Engine

    model = os.environ.get("TOKENSPEED_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    chunk = 128
    # 512 = 4 * chunk -> completion starts on a chunk boundary; window crosses it.
    prompt = [10 + (i % 1000) for i in range(4 * chunk)]
    completion = [200 + i for i in range(40)]
    seqs = [{"prompt_token_ids": prompt, "completion_token_ids": completion}]

    # Reference: large chunk size -> the whole sequence prefills in one chunk.
    ref_engine = Engine(
        model=model, chunked_prefill_size=8192, **_SCORING_ENGINE_KWARGS
    )
    try:
        ref = ref_engine.compute_log_probs(seqs, temperature=1.0)
    finally:
        ref_engine.shutdown()

    # Chunked: small chunk size -> the window spans multiple prefill chunks.
    chunked_engine = Engine(
        model=model, chunked_prefill_size=chunk, **_SCORING_ENGINE_KWARGS
    )
    try:
        got = chunked_engine.compute_log_probs(seqs, temperature=1.0)
    finally:
        chunked_engine.shutdown()

    # Correct count (the original bug raised here) and valid values.
    assert got["tokens"] == [completion]
    assert len(got["log_probs"][0]) == len(completion)
    for lp in got["log_probs"][0]:
        assert lp <= 0.0 and math.isfinite(lp)

    # Chunking the prefill must not change the scores.
    assert len(ref["log_probs"][0]) == len(completion)
    max_abs_diff = max(
        abs(a - b) for a, b in zip(ref["log_probs"][0], got["log_probs"][0])
    )
    assert max_abs_diff < 0.05, f"chunked vs single-chunk diverged: {max_abs_diff}"
