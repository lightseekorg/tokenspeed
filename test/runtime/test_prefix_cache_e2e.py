"""End-to-end tests for --no-enable-prefix-caching.

Validates that the ``--no-enable-prefix-caching`` flag actually disables
prefix caching at runtime, by checking the ``cached_tokens`` field in
``Engine.generate()`` response ``meta_info``.

Runs against two models: ``openai/gpt-oss-20b`` and
``Qwen/Qwen3.5-35B-A3B``.  Override via the ``ONLY_RUN`` environment
variable to test a single model, e.g.::

    ONLY_RUN=openai/gpt-oss-20b python3 -m unittest test_prefix_cache_e2e -v

Usage:
    cd test/runtime
    python3 -m unittest test_prefix_cache_e2e -v

Environment (all optional):
    ONLY_RUN  Only run tests for this model id (substring match).
"""

import dataclasses
import os
import sys
import unittest

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


_MODEL_CASES = [
    ModelCase(
        "openai/gpt-oss-20b",
        extra_kwargs={
            "moe_backend": "flashinfer_mxfp4",
            "disable_prefill_graph": True,
        },
    ),
    ModelCase(
        "Qwen/Qwen3.5-35B-A3B",
        is_thinking_model=True,
        extra_kwargs={
            "reasoning_parser": "qwen3-thinking",
        },
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
    "Please think step by step and be thorough in your reasoning."
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


def _make_engine(case: ModelCase, enable_prefix_caching: bool) -> Engine:
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
    }
    # KVStore requires prefix caching; prevent auto-enabling when prefix
    # caching is off (resolve_cache sets enable_kvstore=True unless
    # disable_kvstore=True).
    if not enable_prefix_caching:
        kwargs["disable_kvstore"] = True
    kwargs.update(case.extra_kwargs)
    return Engine(**kwargs)


class TestPrefixCacheDisabled(unittest.TestCase):
    """When prefix caching is disabled, no tokens should be served from cache."""

    def test_prefix_cache_disabled_no_cached_tokens(self):
        for case in _ACTIVE_MODELS:
            with self.subTest(model=case.model_path):
                engine = _make_engine(case, enable_prefix_caching=False)
                try:
                    sampling = {"max_new_tokens": 8, "temperature": 0}

                    # First request: primes the system prompt in KV cache.
                    engine.generate(
                        prompt=_render_prompt(
                            case, "What is 1+1? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        return_logprob=False,
                        stream=False,
                    )

                    # Second request: shares the same system prefix.
                    resp = engine.generate(
                        prompt=_render_prompt(
                            case, "What is 2+2? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        return_logprob=False,
                        stream=False,
                    )
                    cached = resp["meta_info"].get("cached_tokens", 0)
                    self.assertEqual(
                        cached,
                        0,
                        f"[{case.model_path}] cached_tokens should be 0 when "
                        f"prefix caching is disabled, got {cached}",
                    )
                finally:
                    engine.shutdown()


class TestPrefixCacheEnabled(unittest.TestCase):
    """When prefix caching is enabled (default), the shared prefix should be cached."""

    def test_prefix_cache_enabled_has_cached_tokens(self):
        for case in _ACTIVE_MODELS:
            with self.subTest(model=case.model_path):
                engine = _make_engine(case, enable_prefix_caching=True)
                try:
                    sampling = {"max_new_tokens": 8, "temperature": 0}

                    # First request: primes the system prompt.
                    engine.generate(
                        prompt=_render_prompt(
                            case, "What is 1+1? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        return_logprob=False,
                        stream=False,
                    )

                    # Second request: shares the same system prefix — should hit cache.
                    resp = engine.generate(
                        prompt=_render_prompt(
                            case, "What is 2+2? Reply with just the number."
                        ),
                        sampling_params=sampling,
                        return_logprob=False,
                        stream=False,
                    )
                    cached = resp["meta_info"].get("cached_tokens", 0)
                    self.assertGreater(
                        cached,
                        0,
                        f"[{case.model_path}] cached_tokens should be > 0 when "
                        f"prefix caching is enabled, got {cached}",
                    )
                finally:
                    engine.shutdown()


class TestPrefixCacheDisabledOutputQuality(unittest.TestCase):
    """Disabling prefix caching should not break output quality."""

    def test_prefix_cache_disabled_correct_output(self):
        for case in _ACTIVE_MODELS:
            with self.subTest(model=case.model_path):
                engine = _make_engine(case, enable_prefix_caching=False)
                try:
                    resp = engine.generate(
                        prompt=_render_simple_prompt(
                            case, "What is 2+2? Reply with just the number."
                        ),
                        sampling_params={"max_new_tokens": 32, "temperature": 0},
                        return_logprob=False,
                        stream=False,
                    )
                    text = resp["text"].strip()
                    self.assertIn(
                        "4",
                        text,
                        f"[{case.model_path}] Expected '4' in output when "
                        f"prefix caching is disabled, got {text!r}",
                    )
                finally:
                    engine.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
