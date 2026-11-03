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

"""Token capacity the trtllm all-reduce workspace is armed for."""

from tokenspeed_kernel.ops.communication.trtllm import MNNVL_TWOSHOT_MAX_TOKEN

from tokenspeed.runtime.execution.distributed_initializer import (
    workspace_token_capacity,
)

HIDDEN = 4608
ONESHOT_TOKENS = (2 * 1024 * 1024) // (HIDDEN * 2)


def test_default_capacity_is_the_oneshot_window():
    assert (
        workspace_token_capacity(HIDDEN, 2048, batch_invariant_collectives=False)
        == ONESHOT_TOKENS
    )


def test_batch_invariant_capacity_holds_a_chunk_through_the_twoshot_range():
    # A prefill chunk inside the two-shot range folds in the workspace.
    chunk = min(1024, MNNVL_TWOSHOT_MAX_TOKEN)
    assert workspace_token_capacity(
        HIDDEN, chunk, batch_invariant_collectives=True
    ) == max(ONESHOT_TOKENS, chunk)
    # Larger forwards stop at the two-shot range.
    assert workspace_token_capacity(
        HIDDEN, 1 << 20, batch_invariant_collectives=True
    ) == max(ONESHOT_TOKENS, MNNVL_TWOSHOT_MAX_TOKEN)
    # Never below the one-shot window decode relies on.
    assert (
        workspace_token_capacity(HIDDEN, 8, batch_invariant_collectives=True)
        == ONESHOT_TOKENS
    )
