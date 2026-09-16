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

"""FlashInfer token-bucket utilities.

Serving never opens a live tuning window. It establishes the bucket mapper and
optionally loads a generated warmup bundle before CUDA graph capture; uncovered
keys use the library-selected tactic without profiling in the serving process.
"""

from __future__ import annotations

__all__ = ["get_autotune_max_num_tokens", "set_autotune_max_num_tokens"]

_autotune_max_num_tokens = 8192


def set_autotune_max_num_tokens(num_tokens: int) -> None:
    """Set the maximum token count used by FlashInfer's bucket mapper.

    Args:
        num_tokens: Largest token count a single forward can carry. Values
            below 8192 retain FlashInfer's existing 8192-token floor.
    """
    global _autotune_max_num_tokens
    _autotune_max_num_tokens = max(int(num_tokens), 8192)


def get_autotune_max_num_tokens() -> int:
    """Return the process-wide FlashInfer token-bucket ceiling."""
    return _autotune_max_num_tokens
