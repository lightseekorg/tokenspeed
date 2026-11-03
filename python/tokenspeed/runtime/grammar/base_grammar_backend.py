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

"""The grammar interface the data plane drives, and the bitmask apply."""


class BaseGrammarObject:
    """A request's grammar matcher as the data plane drives it: the bitmask
    fills, and on the capturable path the accepts and rollbacks
    (``HostGrammar``, over the matcher the Rust host owns)."""


def get_apply_vocab_mask_func(grammar_backend: str):
    """Return the backend-specific in-place vocab-mask-apply function.

    The function's signature is ``(logits, vocab_mask) -> None``. It is
    stored on ``SamplingBatchInfo.apply_vocab_mask`` so the captured
    sampler can call it without branching on backend. Only xgrammar is
    wired up today; add branches here when new backends are added.
    """
    if grammar_backend == "xgrammar":

        from xgrammar import apply_token_bitmask_inplace

        # xgrammar's native signature is (logits, bitmask, *, ...);
        # adapt to the canonical (logits, vocab_mask) kwargs the
        # sampler uses.
        def _apply(logits, vocab_mask):
            apply_token_bitmask_inplace(logits, vocab_mask)

        return _apply

    raise ValueError(f"Unsupported grammar backend: {grammar_backend}")
