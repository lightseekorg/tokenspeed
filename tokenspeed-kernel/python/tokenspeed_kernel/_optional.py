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

"""Presence probes for the kernel packages' optional third-party dependencies."""

from __future__ import annotations

import importlib.util


def has_optional_package(*names: str) -> bool:
    """Whether every named optional package can be imported, without importing it.

    Args:
        *names: Import names to probe. A dotted name (``flash_attn.cute``)
            imports its parent package while probing, so list the parent
            first: ``all`` short-circuits, and the parent probe then answers
            before the submodule probe runs.

    Returns:
        True when every probe finds an importable spec, False at the first
        miss. A name blocked with ``sys.modules[name] = None`` counts as
        missing.
    """
    return all(importlib.util.find_spec(name) is not None for name in names)


def has_fa4_cute() -> bool:
    """Whether FlashAttention 4's CuTe DSL interface is present.

    Returns:
        True when the tokenspeed-fa4 ``flash_attn`` package, ``quack`` and
        the ``flash_attn.cute`` submodule are all importable.
    """
    return has_optional_package("flash_attn", "quack", "flash_attn.cute")


def has_trtllm_kernel() -> bool:
    """Whether TRT-LLM's kernel package is present with its DeepEP partner.

    DeepEP must initialize before TRT-LLM's static CUDA runtime, so the two
    are probed as one stack.

    Returns:
        True when both ``trtllm_kernel`` and ``deep_ep`` are importable.
    """
    return has_optional_package("trtllm_kernel", "deep_ep")
