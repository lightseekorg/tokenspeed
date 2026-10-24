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

"""LongCat policy for the shared distributed argmax operator."""


def try_create_argmax_state(group, rank_in_group, max_M, dtype, device, skip_ping_pong):
    """Create shared argmax state with LongCat's boundary and publication policy.

    Args:
        group: Tensor-parallel process group.
        rank_in_group: Rank within the group.
        max_M: Maximum logits row count.
        dtype: Logits value dtype.
        device: Device owning the symmetric workspace.
        skip_ping_pong: Require external synchronization between calls.

    Returns:
        Shared argmax state, or ``None`` if any rank lacks multicast support.
    """
    from tokenspeed_kernel.ops.sampling.cute_dsl import _build_dist_argmax_state
    from tokenspeed_kernel.thirdparty.cute_dsl.longcat.argmax_policy import (
        LongcatArgmaxPolicy,
    )

    return _build_dist_argmax_state(
        group,
        rank_in_group,
        max_M,
        dtype,
        device,
        skip_ping_pong,
        strict=False,
        policy=LongcatArgmaxPolicy(),
    )
