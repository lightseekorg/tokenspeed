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

"""Grammar matchers the Rust engine host owns, as the data plane drives them.

Under the Rust host (``tokenspeed-engine/crates/ts-host``) the control plane
compiles each request's grammar and owns its matcher (``ts-grammar``). The
data plane still fills the token bitmasks, and on the capturable path
advances the matchers inside the forward; it reaches them through the C
entries the host hands over at startup (``set_grammar_entries``), which a
``HostGrammar`` wraps in the grammar interface the data plane takes
(``BaseGrammarObject``). The capturable path's host node calls the same
entries from C++ (``grammar_host``).
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch

from tokenspeed.runtime.grammar.base_grammar_backend import BaseGrammarObject
from tokenspeed.runtime.grammar.grammar_host import load_grammar_host_ext


class HostGrammarEntries:
    """The host's matcher entries, callable from Python.

    Args:
        addresses: Entry name -> its address (``ts_grammar::table::entries``).
    """

    def __init__(self, addresses: dict[str, int]) -> None:
        # The calls keep the GIL: each takes microseconds, and releasing the
        # GIL per call would hand it to another thread for up to a switch
        # interval (5 ms).
        def entry(name: str, restype, *argtypes):
            return ctypes.PYFUNCTYPE(restype, *argtypes)(addresses[name])

        u64, i32, i64, usize = (
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_size_t,
        )
        self.is_terminated = entry("is_terminated", i32, u64)
        self.accept_token = entry("accept_token", i32, u64, i32)
        self.rollback = entry("rollback", i32, u64, i32)
        self.fill_bitmask = entry("fill_bitmask", i32, u64, usize, i64, i64, i32)
        self.traverse_draft_tree = entry(
            "traverse_draft_tree", i32, u64, usize, usize, usize, i64, usize, i64
        )
        self.free = entry("free", None, u64)
        self._addresses = addresses
        # The entries for the capturable grammar's host node, made on first use.
        self._native: Any | None = None

    def native(self) -> Any:
        """The entries as the capturable grammar's host node calls them,
        without the GIL (``grammar_host.Entries``)."""
        if self._native is None:
            self._native = load_grammar_host_ext().Entries(self._addresses)
        return self._native


class HostGrammar(BaseGrammarObject):
    """One request's matcher in the host, by handle.

    Args:
        entries: The host's matcher entries.
        handle: The matcher's handle.
        stop_token_ids: The tokens that end the grammar.
    """

    def __init__(
        self, entries: HostGrammarEntries, handle: int, stop_token_ids: frozenset[int]
    ) -> None:
        self._entries = entries
        self.handle = handle
        self.stop_token_ids = stop_token_ids
        self.finished = False
        self.accepted_tokens: list[int] = []
        # The capturable grammar's C++ reference to the matcher, made on first use.
        self._native_matcher: Any | None = None

    def native_matcher(self) -> Any:
        """The matcher for the capturable grammar's host node, which calls
        it without the GIL. It holds the handle, and each forward's batch
        holds it until the node is done with that forward."""
        if self._native_matcher is None:
            self._native_matcher = load_grammar_host_ext().Matcher(
                self._entries.native(), self.handle, sorted(self.stop_token_ids)
            )
        return self._native_matcher

    def _check(self, status: int, what: str) -> int:
        if status < 0:
            raise RuntimeError(f"grammar matcher {self.handle}: {what} failed")
        return status

    def is_terminated(self) -> bool:
        return bool(
            self._check(self._entries.is_terminated(self.handle), "a state read")
        )

    def accept_token(self, token: int) -> None:
        if self.is_terminated():
            self.accepted_tokens.append(token)
            return
        if not self._check(self._entries.accept_token(self.handle, token), "accept"):
            raise ValueError(
                f"Tokens not accepted: {token}\n"
                f"Accepted tokens: {self.accepted_tokens}\n"
                f"Terminated: {self.is_terminated()}\n"
            )

    def try_accept_token(self, token: int) -> bool:
        """Non-raising accept for the spec-verify mask fill: whether the
        grammar takes ``token``; on rejection the matcher is unchanged."""
        if self.is_terminated():
            return False
        return bool(
            self._check(self._entries.accept_token(self.handle, token), "accept")
        )

    def rollback(self, k: int) -> None:
        self._check(self._entries.rollback(self.handle, k), "rollback")
        self.accepted_tokens = self.accepted_tokens[:-k]

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        rows, width = vocab_mask.shape
        self._check(
            self._entries.fill_bitmask(
                self.handle, vocab_mask.data_ptr(), rows, width, idx
            ),
            "a bitmask fill",
        )

    def fill_vocab_masks_along_drafts(
        self,
        vocab_masks: torch.Tensor,
        draft_tokens: torch.Tensor,
        next_node: torch.Tensor,
        next_sibling: torch.Tensor,
    ) -> None:
        """Fill ``vocab_masks[k]`` with the mask after ``draft_tokens[1..k]``
        along the draft tree; the matcher is left as it was."""
        columns = [
            column.to(torch.int64).contiguous()
            for column in (next_node, next_sibling, draft_tokens)
        ]
        self._check(
            self._entries.traverse_draft_tree(
                self.handle,
                columns[0].data_ptr(),
                columns[1].data_ptr(),
                columns[2].data_ptr(),
                columns[2].numel(),
                vocab_masks.data_ptr(),
                vocab_masks.shape[-1],
            ),
            "a draft-tree fill",
        )

    def release(self) -> None:
        """Drop the request's holds on the matcher; queued forwards keep
        theirs (``native_matcher``)."""
        self._native_matcher = None
        self._entries.free(self.handle)
