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

"""Startup capacity for the persistent DeepEP low-latency receive buffers."""

from math import gcd


def resolve_deepep_token_capacity(
    *,
    requested_capacity: int | None,
    max_num_seqs: int,
    attn_dp_size: int,
    attn_tp_size: int,
    tokens_per_request: int,
    token_sliced: bool,
    low_latency_enabled: bool,
    max_extend_tokens: int,
) -> int:
    """Resolve the maximum number of token rows sent by one EP rank.

    Args:
        requested_capacity: Explicit CLI capacity, or None for automatic sizing.
        max_num_seqs: Global scheduler request capacity across attention DP.
        attn_dp_size: Number of independent attention request batches.
        attn_tp_size: Number of TP ranks sharing each attention batch.
        tokens_per_request: Target verify width, including the anchor token.
        token_sliced: Whether TP ranks dispatch disjoint token slices.
        low_latency_enabled: Whether low-latency buffers will be allocated.
        max_extend_tokens: Scheduled extend/recovery token bound when every
            forward is pinned to low latency; zero when normal legs exist.

    Returns:
        Per-rank send capacity, fixed before the first forward. Token-sliced
        plans cover all admissible decode batches, including eager batches above
        the capture ladder. Other plans retain the legacy automatic value 256.

    Raises:
        ValueError: A capacity is nonpositive or cannot hold the configured
            token-sliced decode workload.
    """
    if requested_capacity is not None and requested_capacity <= 0:
        raise ValueError("--low-latency-max-num-tokens-per-gpu must be positive")
    if not token_sliced or not low_latency_enabled:
        return 256 if requested_capacity is None else requested_capacity
    if attn_dp_size <= 0 or attn_tp_size <= 0 or tokens_per_request <= 0:
        raise ValueError("DeepEP parallel sizes and verify width must be positive")
    if max_extend_tokens < 0:
        raise ValueError("DeepEP extend capacity must be nonnegative")
    if max_num_seqs < attn_dp_size:
        raise ValueError("DeepEP requires at least one request slot per attention DP")

    # This is the same per-DP request bound as ForwardStepRunner. Capture
    # buckets never exceed it; a smaller ladder must not reduce send capacity.
    # The D scheduler budgets recovery prompt tokens separately from decode
    # rows. A mixed round can therefore contain both bounds simultaneously.
    max_rows = (max_num_seqs // attn_dp_size) * tokens_per_request + max_extend_tokens
    required_capacity = (max_rows + attn_tp_size - 1) // attn_tp_size
    # Legacy DeepEP requires EP_size * capacity to be divisible by four,
    # including BF16 dispatch. K3's EP group is attention TP * DP.
    alignment = 4 // gcd(attn_tp_size * attn_dp_size, 4)
    required_capacity = (required_capacity + alignment - 1) // alignment * alignment
    if requested_capacity is None:
        return required_capacity
    if requested_capacity < required_capacity:
        raise ValueError(
            f"--low-latency-max-num-tokens-per-gpu={requested_capacity} cannot "
            f"hold the configured workload: need at least "
            f"{required_capacity} rows per rank after TP{attn_tp_size} slicing "
            f"({max_num_seqs // attn_dp_size} requests per DP rank, "
            f"verify width {tokens_per_request}, extend bound {max_extend_tokens}). "
            "Increase the capacity, reduce the batch/chunk limits, or use "
            "--deepep-mode auto to send extends through normal dispatch."
        )
    if requested_capacity % alignment:
        raise ValueError(
            f"--low-latency-max-num-tokens-per-gpu must be a multiple of "
            f"{alignment} for this DeepEP group"
        )
    return requested_capacity
