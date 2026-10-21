"""End-to-end tests for --disable-prefix-caching.

Validates that the ``--disable-prefix-caching`` flag actually disables
prefix caching at runtime, by checking the ``cached_tokens`` field in
``Engine.generate()`` response ``meta_info``.

Runs against two models: ``openai/gpt-oss-20b`` and
``txn545/Qwen3.5-35B-A3B-NVFP4``.  Override via the ``ONLY_RUN`` environment
variable to test a single model, e.g.::

    ONLY_RUN=openai/gpt-oss-20b python3 -m pytest test_prefix_cache_e2e.py -v

Usage:
    cd test/runtime
    python3 -m pytest test_prefix_cache_e2e.py -v

Environment (all optional):
    ONLY_RUN  Only run tests for this model id (substring match).
"""

import dataclasses
import os
import sys

import pytest
import torch

# Repository root on sys.path so ``test.runners`` resolves.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runners import get_dtype_str  # noqa: E402

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=300, suite="runtime-prefix-cache-e2e")

from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: E402

_TORCH_DTYPE = torch.bfloat16

_ONLY_RUN = os.environ.get("ONLY_RUN", "")


@dataclasses.dataclass
class ModelCase:
    model_path: str
    extra_kwargs: dict = dataclasses.field(default_factory=dict)
    # Whether the model is a thinking model (e.g. Qwen3.5) that needs
    # chat-template rendering with enable_thinking=False to suppress
    # reasoning output.
    is_thinking_model: bool = False
    # Whether the model carries recurrent-state cache groups, whose prefix
    # hits exist only at published checkpoints and resume within tolerance.
    has_state_groups: bool = False


_MODEL_CASES = [
    ModelCase(
        "openai/gpt-oss-20b",
        extra_kwargs={
            "moe_backend": "flashinfer_trtllm",
            "disable_prefill_graph": True,
        },
    ),
    ModelCase(
        "txn545/Qwen3.5-35B-A3B-NVFP4",
        extra_kwargs={
            "attention_backend": "trtllm",
            "moe_backend": "flashinfer_trtllm",
            "quantization": "nvfp4",
        },
        is_thinking_model=True,
        has_state_groups=True,
    ),
]

_ACTIVE_MODELS = [
    mc for mc in _MODEL_CASES if not _ONLY_RUN or _ONLY_RUN in mc.model_path
]

# A long system prompt ensures a substantial prefix that the cache can
# reuse.  Two different user questions follow the same system prefix so
# the second request should hit the cached prefix when caching is on.
_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible. "
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something incorrect. "
    "If you don't know the answer to a question, please don't share false information. "
    "Please think step by step and be thorough in your reasoning. "
    "Before answering, identify the user's actual intent, check every relevant fact, "
    "and distinguish facts from assumptions. Prefer clear explanations with concrete "
    "examples, preserve important constraints from earlier instructions, and avoid "
    "inventing details that were not provided. When several interpretations are "
    "possible, choose the most useful reasonable interpretation and state any assumption "
    "that materially affects the answer. Keep the final response focused, accurate, "
    "self-contained, and easy to verify."
)


def _render_prompt(case: ModelCase, user_msg: str) -> str:
    """Render a prompt string for the model.

    For thinking models (e.g. Qwen3.5), use the tokenizer's chat template
    with ``enable_thinking=False`` so the model answers directly without a
    reasoning section.  For other models, use a plain text format.
    """
    if case.is_thinking_model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(case.model_path, trust_remote_code=True)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
    return f"{_SYSTEM_PROMPT}\n\nUser: {user_msg}\nAssistant:"


def _render_simple_prompt(case: ModelCase, user_msg: str) -> str:
    """Render a simple (no system prompt) prompt for quality checks."""
    if case.is_thinking_model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(case.model_path, trust_remote_code=True)
        messages = [{"role": "user", "content": user_msg}]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
    return user_msg


def _make_engine(case: ModelCase, enable_prefix_caching: bool, **overrides) -> Engine:
    kwargs = {
        "model": case.model_path,
        "dtype": get_dtype_str(_TORCH_DTYPE),
        "seed": 42,
        "enable_prefix_caching": enable_prefix_caching,
        "max_model_len": 8192,
        "max_num_seqs": 4,
        "max_prefill_tokens": 1024,
        "chunked_prefill_size": 1024,
        "gpu_memory_utilization": 0.7,
        "disable_kvstore": True,
        "disable_autotune": True,
    }
    kwargs.update(case.extra_kwargs)
    kwargs.update(overrides)
    return Engine(**kwargs)


class TestPrefixCacheDisabled:
    """When prefix caching is disabled, no tokens should be served from cache."""

    def test_prefix_cache_disabled_no_cached_tokens(self, subtests):
        for case in _ACTIVE_MODELS:
            with subtests.test(model=case.model_path):
                engine = _make_engine(case, enable_prefix_caching=False)
                try:
                    sampling = {"max_new_tokens": 8, "temperature": 0}

                    # First request: primes the system prompt in KV cache.
                    engine.generate(
                        prompt=_render_prompt(
                            case, "What is 1+1? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        stream=False,
                    )

                    # Second request: shares the same system prefix.
                    resp = engine.generate(
                        prompt=_render_prompt(
                            case, "What is 2+2? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        stream=False,
                    )
                    cached = resp["meta_info"].get("cached_tokens", 0)
                    _pytest_first_1 = cached
                    _pytest_second_2 = 0
                    _pytest_msg_3 = (
                        f"[{case.model_path}] cached_tokens should be 0 when "
                        f"prefix caching is disabled, got {cached}"
                    )
                    assert _pytest_first_1 == _pytest_second_2, _pytest_msg_3
                finally:
                    engine.shutdown()


class TestPrefixCacheEnabled:
    """When prefix caching is enabled, a shared prefix becomes reusable."""

    def test_prefix_cache_enabled_has_cached_tokens(self, subtests):
        for case in _ACTIVE_MODELS:
            with subtests.test(model=case.model_path):
                engine = _make_engine(case, enable_prefix_caching=True)
                try:
                    sampling = {"max_new_tokens": 8, "temperature": 0}

                    # The first request publishes its prompt end and finish.
                    # The second request shares the system prefix below that,
                    # hits its hash-complete pages and, under the junction
                    # cut, ends its first chunk at the shared boundary so the
                    # third request hits the converged cross-group prefix.
                    engine.generate(
                        prompt=_render_prompt(
                            case, "What is 1+1? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        stream=False,
                    )
                    engine.generate(
                        prompt=_render_prompt(
                            case, "What is 2+2? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        stream=False,
                    )
                    resp = engine.generate(
                        prompt=_render_prompt(
                            case, "What is 3+3? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        stream=False,
                    )
                    cached = resp["meta_info"].get("cached_tokens", 0)
                    _pytest_a_4 = cached
                    _pytest_b_5 = 0
                    _pytest_msg_6 = (
                        f"[{case.model_path}] cached_tokens should be > 0 when "
                        f"prefix caching is enabled, got {cached}"
                    )
                    assert _pytest_a_4 > _pytest_b_5, _pytest_msg_6
                finally:
                    engine.shutdown()


def _prompt_ids(case: ModelCase, user_msg: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(case.model_path, trust_remote_code=True)
    return tok(_render_prompt(case, user_msg), add_special_tokens=False)["input_ids"]


def _greedy_ids(
    engine: Engine, input_ids: list[int], max_new_tokens: int, *, bypass_cache: bool
):
    """Greedy continuation and the prefix-cache hit the request got.

    ``bypass_cache`` asks for prompt logprobs from position 0: every scored
    position must be recomputed, so the admission probe is bounded at 0 and
    the request computes its whole prompt whatever the cache holds (the only
    per-request cache bound the API exposes). Its forwards are the same
    layer computations; only the LM head additionally scores the prompt rows.
    """
    resp = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        return_logprob=bypass_cache,
        logprob_start_len=0 if bypass_cache else None,
        stream=False,
    )
    return list(resp["output_ids"]), int(resp["meta_info"].get("cached_tokens", 0))


class TestTokenLevelPrefixHits:
    """A hit ends at the token where a published prefix ended, not at a page.

    The oracle is shape-matched: under ``--numerics auto`` a position's
    logits depend on the shape of the forward that computed them (a 1-row
    extend and a 276-row prefill round differently), so the reference must
    run the same forwards as the warm path. With ``chunked_prefill_size =
    |X| - 1`` the reference ``X`` is prefilled as ``[0, |X|-1)`` then the
    1-row chunk ``[|X|-1, |X|)``; the warm path is ``X[:-1]`` prefilled whole
    (the same ``[0, |X|-1)`` forward, run first so nothing it could hit is
    published yet) followed by a repeat of ``X`` that resumes at its
    published prompt end, a 1-row extend from the same boundary with the same
    state and KV rows. The reference bypasses the cache so it computes the
    same forwards itself; the warm repeat must then reproduce its continuation
    token for token, for KV-only and recurrent-state models alike. ``X + Y``
    hits the reference's prompt end exactly; ``X[:-37]`` ends inside the last
    page before any publication point and finds no state below it; a next
    turn hits the reference's finish, which settled after its last forward
    landed and covers every token the reference sampled but the one it ended
    on.
    """

    STEPS = 24

    def test_hits_at_natural_boundaries_reproduce_the_cold_continuation(self, subtests):
        for case in _ACTIVE_MODELS:
            with subtests.test(model=case.model_path):
                hybrid = case.has_state_groups
                prefix = _prompt_ids(
                    case,
                    "Please summarize, in order, the forty-two rules of the "
                    "house style guide, then explain why rule seventeen "
                    "exists and how it interacts with rules three and nine. "
                    "For each rule, name the section it belongs to, the "
                    "year it was introduced, the most common way it is "
                    "broken in practice, and the reviewer who usually "
                    "catches that mistake. Where two rules appear to "
                    "conflict, say which one wins and why the committee "
                    "decided so. Reply with the summary only, one rule per "
                    "line, without any preamble or closing remarks.",
                )
                # The cuts below must leave at least one hash-complete page
                # under the largest prefix granularity a case may use.
                assert (len(prefix)) > (128 + 37)
                engine = _make_engine(
                    case,
                    enable_prefix_caching=True,
                    chunked_prefill_size=len(prefix) - 1,
                    # The reference asks for prompt logprobs (its cache bypass).
                    enable_output_logprobs=True,
                )
                try:
                    # X[:-1] first: one cold forward over [0, |X|-1), the
                    # shape the reference's first chunk has. Its prompt end
                    # publishes the tail and the state at |X| - 1.
                    shorter = prefix[:-1]
                    warm, cached = _greedy_ids(engine, shorter, 1, bypass_cache=False)
                    assert (cached) == (0)
                    assert (len(warm)) == (1)

                    # The reference: X computed whole, chunked [0, |X|-1) and
                    # [|X|-1, |X|), with the cache bypassed.
                    cold, cached = _greedy_ids(
                        engine, prefix, self.STEPS, bypass_cache=True
                    )
                    assert (cached) == (0)
                    assert (len(cold)) == (self.STEPS)

                    # X + Y: the reference's prompt-end checkpoint is hit
                    # exactly, whatever the page geometry, by KV and state
                    # groups alike.
                    suffix = _prompt_ids(case, "Now do the same for the other guide.")
                    _, cached = _greedy_ids(
                        engine, prefix + suffix, 4, bypass_cache=False
                    )
                    assert (cached) == (len(prefix))

                    # X[:-37] ends inside the last page, before any publication
                    # point: a hybrid finds no state below it; a KV-only model
                    # may resume at its hash-complete pages when every group
                    # holds them (a sliding-window group only where its
                    # lookback was published).
                    deeper = prefix[:-37]
                    warm, cached = _greedy_ids(engine, deeper, 1, bypass_cache=False)
                    assert (cached) < (len(deeper))
                    if hybrid:
                        assert (cached) == (0)
                    assert (len(warm)) == (1)

                    # X again: resumes at |X| - 1 from the state and KV rows
                    # X[:-1] published -- the same forward the reference's
                    # second chunk ran -- so the continuation is the
                    # reference's, token for token.
                    warm, cached = _greedy_ids(
                        engine, prefix, self.STEPS, bypass_cache=False
                    )
                    assert (cached) == (len(prefix) - 1)
                    assert (warm) == (cold)

                    # A next turn -- X, the reference's whole continuation and
                    # more -- hits its finish checkpoint: the state and KV
                    # before the token the request ended on (its last, which
                    # reached the token budget), whatever the schedule, for KV
                    # and state groups alike.
                    _, cached = _greedy_ids(
                        engine, prefix + cold + suffix, 1, bypass_cache=False
                    )
                    assert (cached) == (len(prefix) + len(cold) - 1)
                finally:
                    engine.shutdown()


class TestPrefixCacheDisabledOutputQuality:
    """Disabling prefix caching should not break output quality."""

    def test_prefix_cache_disabled_correct_output(self, subtests):
        for case in _ACTIVE_MODELS:
            with subtests.test(model=case.model_path):
                engine = _make_engine(case, enable_prefix_caching=False)
                try:
                    resp = engine.generate(
                        prompt=_render_simple_prompt(
                            case, "What is 2+2? Reply with just the number."
                        ),
                        sampling_params={"max_new_tokens": 32, "temperature": 0},
                        stream=False,
                    )
                    text = resp["text"].strip()
                    _pytest_member_7 = "4"
                    _pytest_container_8 = text
                    _pytest_msg_9 = (
                        f"[{case.model_path}] Expected '4' in output when "
                        f"prefix caching is disabled, got {text!r}"
                    )
                    assert _pytest_member_7 in _pytest_container_8, _pytest_msg_9
                finally:
                    engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
