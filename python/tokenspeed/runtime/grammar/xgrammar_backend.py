# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""Constrained decoding with xgrammar backend."""

import json
from collections.abc import Iterable

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTagItem,
    TokenizerInfo,
    allocate_token_bitmask,
)

from tokenspeed.runtime.grammar.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


MAX_ROLLBACK_TOKENS = 200

# The matcher methods the capturable grammar's build hostfunc calls. tvm-ffi
# releases the GIL around every call by default, and on the CUDA callback
# thread each release lets the scheduler thread take the GIL for up to a
# switch interval (5 ms). Each call takes microseconds, so they keep it.
GIL_HOLDING_MATCHER_METHODS = frozenset(
    (
        "accept_token",
        "fill_next_token_bitmask",
        "traverse_draft_tree",
        "rollback",
        "is_terminated",
    )
)


def keep_gil_in_matcher_calls() -> None:
    """Make GIL_HOLDING_MATCHER_METHODS keep the GIL, for the whole process."""
    from xgrammar.base import _core

    for method in _core.GrammarMatcher.__tvm_ffi_type_info__.methods:
        if method.name in GIL_HOLDING_MATCHER_METHODS:
            method.func.release_gil = False


class XGrammarGrammar(BaseGrammarObject):
    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: list[int] | int | None,
    ) -> None:

        self.matcher = matcher
        self.vocab_size = vocab_size

        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens

        # Accepting one of these ends the grammar.
        self.stop_token_ids = frozenset(matcher.stop_token_ids)

        self.finished = False
        self.accepted_tokens: list[int] = []

    def is_terminated(self):

        return self.matcher.is_terminated()

    def accept_token(self, token: int):

        if not self.is_terminated():
            if not self.matcher.accept_token(token):
                raise ValueError(
                    f"Tokens not accepted: {token}\n"
                    f"Accepted tokens: {self.accepted_tokens}\n"
                    f"Terminated: {self.matcher.is_terminated()}\n"
                )

        else:
            self.accepted_tokens.append(token)

    def try_accept_token(self, token: int) -> bool:
        """Non-raising accept used by the spec-verify mask fill.

        Returns True iff the matcher accepts the token; on rejection the
        matcher state is unchanged. Used to walk draft chains where some
        positions may diverge from the grammar.
        """
        if self.is_terminated():
            return False

        else:
            return self.matcher.accept_token(token)

    def rollback(self, k: int):

        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:

        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:

        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    def fill_vocab_masks_along_drafts(
        self,
        vocab_masks: torch.Tensor,
        draft_tokens: torch.Tensor,
        next_node: torch.Tensor,
        next_sibling: torch.Tensor,
    ) -> None:
        """Fill ``vocab_masks[k]`` with the mask after ``draft_tokens[1..k]``,
        in one matcher call; ``draft_tokens[0]`` is already accepted.

        ``next_node`` / ``next_sibling`` give the draft tree (int64, one entry
        per node, -1 for none). The walk stops at the first draft token the
        grammar rejects or that ends it: that node's row is zeroed and the
        rows below it are not written. The matcher state is unchanged.
        """
        self.matcher.traverse_draft_tree(
            next_node, next_sibling, draft_tokens, vocab_masks
        )

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:

        return vocab_mask.to(device, non_blocking=True)

    def copy(self):

        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )

        return XGrammarGrammar(
            matcher, self.vocab_size, self.ctx, self.override_stop_tokens
        )

    @property
    def memory_size_bytes(self) -> int:

        # The compiled grammar (xgrammar's estimate) and this object's
        # matcher, mostly a token bitset of the vocabulary size.
        return self.ctx.memory_size_bytes + self.vocab_size // 8


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        disable_any_whitespace: bool = False,
        stop_token_ids: Iterable[int] | None = None,
    ) -> None:

        super().__init__()

        keep_gil_in_matcher_calls()

        # The matcher ends a grammar on the tokens the engine stops on, the
        # model's end-of-sequence ids with the tokenizer's own EOS and extra
        # stop ids, rather than on the tokenizer's one EOS only: a grammar that
        # may end there then takes any of them as its end.
        if stop_token_ids:
            stop_token_ids = set(stop_token_ids)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                stop_token_ids.add(eos_token_id)
            extra = getattr(tokenizer, "additional_stop_token_ids", None)
            stop_token_ids.update(extra or ())
            stop_token_ids = sorted(stop_token_ids)
        else:
            stop_token_ids = None
        tokenizer_info = TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size, stop_token_ids=stop_token_ids
        )

        # BaseGrammarBackend keeps the compiled grammars that requests reuse,
        # within RETAINED_GRAMMAR_LIMIT_BYTES. xgrammar's own cache is off:
        # its heap keeps growing past cache_limit_bytes as distinct schemas
        # arrive, and clear_cache() does not free all of it.
        self.grammar_compiler = GrammarCompiler(
            tokenizer_info=tokenizer_info, cache_enabled=False
        )

        self.vocab_size = vocab_size
        self.override_stop_tokens = None
        self.disable_any_whitespace = disable_any_whitespace

    def init_value_impl(
        self, key: tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:

        key_type, key_string = key
        any_whitespace = not self.disable_any_whitespace
        try:
            if key_type == "json":
                if key_string == "$$ANY$$":
                    ctx = self.grammar_compiler.compile_json_schema(
                        '{"type": "object"}', any_whitespace=any_whitespace
                    )
                else:
                    ctx = self.grammar_compiler.compile_json_schema(
                        schema=key_string, any_whitespace=any_whitespace
                    )

            elif key_type == "ebnf":
                ctx = self.grammar_compiler.compile_grammar(key_string)

            elif key_type == "regex":
                ctx = self.grammar_compiler.compile_regex(key_string)

            elif key_type == "structural_tag":
                structural_tag = json.loads(key_string)

                # Built-in structural-tag payloads include a ``format`` field
                # and can be compiled directly. Explicit structures/triggers
                # payloads are expanded into xgrammar tag items below.
                if "format" in structural_tag:
                    ctx = self.grammar_compiler.compile_structural_tag(structural_tag)
                else:
                    tags = [
                        StructuralTagItem(
                            begin=structure["begin"],
                            schema=json.dumps(structure["schema"]),
                            end=structure["end"],
                        )
                        for structure in structural_tag["structures"]
                    ]
                    ctx = self.grammar_compiler.compile_structural_tag(
                        tags, structural_tag["triggers"]
                    )

            else:
                raise ValueError(f"Invalid key_type: {key_type}")

        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                f"Failed to compile {key_type!s} grammar: key_string={key_string!r}, e="
                f"{exc!r}",
            )
            return InvalidGrammarObject(f"{type(exc).__name__}: {exc}")

        matcher = GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, ctx, self.override_stop_tokens)

    def reset(self):

        super().reset()
        self.grammar_compiler and self.grammar_compiler.clear_cache()
