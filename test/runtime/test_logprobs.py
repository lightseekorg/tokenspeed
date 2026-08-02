"""Tests for detailed logprob response rendering."""

from types import SimpleNamespace

from tokenspeed.runtime.engine.logprobs import LogprobsProcessor


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def batch_decode(self, token_ids):
        self.calls.append(token_ids)
        return [f"token-{row[0]}" for row in token_ids]


def test_detokenization_preserves_prompt_placeholder_alignment():
    tokenizer = _Tokenizer()
    processor = LogprobsProcessor(SimpleNamespace(tokenizer=tokenizer))

    sampled = processor.detokenize_logprob_tokens(
        [None, -0.25],
        [10, 11],
        decode_to_text=True,
    )
    top = processor.detokenize_top_logprobs_tokens(
        [None, [-0.1, -1.0]],
        [None, [7, 9]],
        decode_to_text=True,
    )

    assert sampled == [
        (None, 10, "token-10"),
        (-0.25, 11, "token-11"),
    ]
    assert top == [
        None,
        [(-0.1, 7, "token-7"), (-1.0, 9, "token-9")],
    ]
    assert tokenizer.calls == [[[10], [11]], [[7], [9]]]
