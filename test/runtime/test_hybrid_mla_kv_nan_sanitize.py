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

"""The KDA cache sanitizes its latent writes; the plain MLA cache does not.

Context (the Kimi-K3 CUDA-graph "!!!" bug): under the prefill breakable graph,
the dummy-batch capture (out_cache_loc == the reserved ``dummy_kv_slot``)
writes NaN K/V. The paged MLA decode kernel then reads that shared dummy slot
through the zero-padded block-table entries and computes ``q.k`` *before* the
causal mask, so ``NaN + -inf = NaN`` survives the mask and poisons a live row's
softmax -> all-NaN logits -> ``argmax`` picks token 0 ("!"). Eager prefill
leaves the dummy slot finite (``q.0`` masks cleanly), so the bug only appears
with the prefill graph on.

Sanitization is a fact of *this cache*, not of a wrapper around it: no call
site passes ``sanitize=`` at all, so the default the pool class declares IS
the contract -- and it holds for every view of the arena, target or draft.
"""

from __future__ import annotations

import inspect
import pathlib

from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool


def _sanitize_default(cls) -> bool:
    return inspect.signature(cls.set_mla_kv_buffer).parameters["sanitize"].default


def test_kda_cache_sanitizes_latent_writes_by_default() -> None:
    assert _sanitize_default(HybridKDATokenToKVPool) is True


def test_plain_mla_cache_does_not_sanitize_by_default() -> None:
    """The bug needs the KDA arena's aliased state pages; a pure MLA pool
    keeps the cheaper write, so the two defaults must stay distinct."""
    assert _sanitize_default(MLATokenToKVPool) is False


def test_the_default_is_the_whole_contract() -> None:
    """No caller may pass ``sanitize=``: the pools' own defaults decide.

    A call site that opts in per write would put the graph-padding contract
    back in the callers' hands, which is how it came to live in a wrapper.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "python"
    allowed = {"hybrid_kda.py", "mla.py"}  # the two definitions themselves
    offenders = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name not in allowed and "sanitize=" in path.read_text()
    )
    assert offenders == []
