from tokenspeed.runtime.utils.hf_transformers_utils import _wrap_deepseek_v4_tokenizer


class _FakeTokenizer:
    vocab_size = 10

    def get_added_vocab(self):
        return {"<fake>": 10}

    def encode(self, prompt, add_special_tokens=True, **kwargs):
        self.last_encode = (prompt, add_special_tokens, kwargs)
        return [1, 2, 3]


def _wrapped_tokenizer():
    calls = []

    def fake_encode_messages(messages, **kwargs):
        calls.append(kwargs)
        return "rendered prompt"

    return _wrap_deepseek_v4_tokenizer(_FakeTokenizer(), fake_encode_messages), calls


def test_deepseek_v4_tokenizer_defaults_thinking_effort_to_max() -> None:
    tokenizer, calls = _wrapped_tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        enable_thinking=True,
    )

    assert prompt == "rendered prompt"
    assert calls[-1]["thinking_mode"] == "thinking"
    assert calls[-1]["reasoning_effort"] == "max"


def test_deepseek_v4_tokenizer_preserves_explicit_high_effort() -> None:
    tokenizer, calls = _wrapped_tokenizer()

    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="high",
    )

    assert calls[-1]["thinking_mode"] == "thinking"
    assert calls[-1]["reasoning_effort"] == "high"


def test_deepseek_v4_tokenizer_reasoning_none_disables_thinking() -> None:
    tokenizer, calls = _wrapped_tokenizer()

    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert calls[-1]["thinking_mode"] == "chat"
    assert calls[-1]["reasoning_effort"] is None


def test_deepseek_v4_tokenizer_chat_mode_does_not_force_max() -> None:
    tokenizer, calls = _wrapped_tokenizer()

    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
    )

    assert calls[-1]["thinking_mode"] == "chat"
    assert calls[-1]["reasoning_effort"] is None
