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

"""Retarget the shared CI conversations to the tested model's token budgets."""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path


def prompt_tokens(messages: list[dict], tokenizer) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def trim_turn(
    messages: list[dict],
    *,
    prefix: list[dict],
    prefix_tokens: int,
    tokenizer,
    max_prompt_tokens: int,
) -> tuple[list[dict], int]:
    """Keep a turn's text within the templated prompt budget, including prefix."""
    remaining = max_prompt_tokens - prefix_tokens
    delta = []
    for message in messages:
        if message["role"] != "user" or not isinstance(message["content"], str):
            raise ValueError("Source conversations must contain text user messages")
        if remaining <= 0:
            break
        ids = tokenizer.encode(message["content"], add_special_tokens=False)
        item = dict(message)
        if len(ids) > remaining:
            item["content"] = tokenizer.decode(
                ids[:remaining], skip_special_tokens=False
            )
        delta.append(item)
        remaining -= min(len(ids), remaining)

    # Bare text does not include chat separators or the generation prompt.
    # Trim their cost too so the first request actually fits the Qwen budget.
    while delta:
        actual = prompt_tokens([*prefix, *delta], tokenizer)
        excess = actual - max_prompt_tokens
        if excess <= 0:
            return delta, actual
        ids = tokenizer.encode(delta[-1]["content"], add_special_tokens=False)
        if len(ids) <= excess:
            delta.pop()
        else:
            content = delta[-1]["content"]
            shortened = tokenizer.decode(ids[:-excess], skip_special_tokens=False)
            # A cut through a multibyte character can decode to a replacement
            # character. Always make progress if re-encoding keeps its size.
            delta[-1]["content"] = (
                shortened if len(shortened) < len(content) else content[:-1]
            )
    raise ValueError("Token budget is too small for a nonempty user turn")


def build_dataset(
    source: dict,
    *,
    tokenizer,
    model_path: str,
    tokenizer_revision: str,
    source_sha256: str,
    source_revision: str,
    first_turn_length: int,
    subsequent_turn_length: int,
) -> dict:
    """Preserve conversation order and turn counts while retokenizing input text."""
    conversations = []
    for conversation in source["conversations"]:
        prefix = []
        turns = []
        previous_tokens = 0
        for turn_index, turn in enumerate(conversation):
            target = (
                first_turn_length
                if turn_index == 0
                else previous_tokens + subsequent_turn_length
            )
            delta, actual = trim_turn(
                turn["messages"],
                prefix=prefix,
                prefix_tokens=previous_tokens,
                tokenizer=tokenizer,
                max_prompt_tokens=target,
            )
            prefix.extend(delta)
            turns.append({"messages": delta, "prompt_tokens": actual})
            previous_tokens = actual
        conversations.append(turns)
    first_lengths = [c[0]["prompt_tokens"] for c in conversations]
    return {
        "metadata": {
            "model_path": model_path,
            "tokenizer_revision": tokenizer_revision,
            "source_revision": source_revision,
            "source_sha256": source_sha256,
            "construction": "trim_each_turn_with_model_tokenizer_v1",
            "first_turn_length": first_turn_length,
            "subsequent_turn_length": subsequent_turn_length,
            "num_conversations": len(conversations),
            "first_prompt_tokens_min": min(first_lengths),
            "first_prompt_tokens_max": max(first_lengths),
            "chat_template_included": True,
        },
        "conversations": conversations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--first-turn-length", type=int, required=True)
    parser.add_argument("--subsequent-turn-length", type=int, required=True)
    args = parser.parse_args()
    if args.first_turn_length <= 0 or args.subsequent_turn_length <= 0:
        parser.error("Token lengths must be positive")
    source_bytes = args.source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    fingerprint = {
        "model_path": args.model_path,
        "tokenizer_revision": args.tokenizer_revision,
        "source_revision": args.source_revision,
        "source_sha256": source_sha256,
        "construction": "trim_each_turn_with_model_tokenizer_v1",
        "first_turn_length": args.first_turn_length,
        "subsequent_turn_length": args.subsequent_turn_length,
    }
    if args.output_path.exists():
        metadata = json.loads(args.output_path.read_text())["metadata"]
        if all(metadata.get(key) == value for key, value in fingerprint.items()):
            print(f"Using cached Qwen dataset: {args.output_path}")
            return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, revision=args.tokenizer_revision, trust_remote_code=True
    )
    data = build_dataset(
        json.loads(source_bytes),
        tokenizer=tokenizer,
        model_path=args.model_path,
        tokenizer_revision=args.tokenizer_revision,
        source_sha256=source_sha256,
        source_revision=args.source_revision,
        first_turn_length=args.first_turn_length,
        subsequent_turn_length=args.subsequent_turn_length,
    )
    temporary = args.output_path.with_name(f"{args.output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output_path)
    print(json.dumps(data["metadata"], indent=2))


if __name__ == "__main__":
    main()
