# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        disable_any_whitespace: bool = False,
    ) -> None:

        super().__init__()

        tokenizer_info = TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)

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

        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to compile %s grammar: key_string=%r, e=%r",
                key_type,
                key_string,
                e,
            )
            return InvalidGrammarObject(f"{type(e).__name__}: {e}")

        matcher = GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, ctx, self.override_stop_tokens)

    def reset(self):

        self.grammar_compiler and self.grammar_compiler.clear_cache()
