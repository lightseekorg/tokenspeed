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

"""Execution-side lifecycle for algorithm-specific verification state.

Instances are assembled once at startup and handed to the forward runner.
They allocate transient workspace against an already bound cache plan and
consume acceptance after the complete forward, on the execution stream.
"""

from typing import Protocol

import torch


class SpeculativeState(Protocol):
    """A fixed execution participant with its own verification workspace."""

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate scratch for the full decode capacity; return workspace bytes."""
        ...

    def commit_after_verify(
        self, accepted_lengths: torch.Tensor, *, num_extends: int
    ) -> None:
        """Commit live accepted rows, excluding the leading extend requests."""
        ...
