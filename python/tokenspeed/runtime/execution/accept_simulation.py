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

"""Simulated speculative acceptance for runs whose outputs carry no signal.

Under dummy weights or an emulated rank the target accepts whatever drafts its
outputs happen to match. ``TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN`` replaces each
chain verify step's accepted widths with the average a real run measured, and
its tokens with the drafts those widths accept.
"""

from __future__ import annotations

import torch

# Fixed-point denominator of the simulated average.
ACCEPT_LENGTH_SCALE = 1000


def parse_simulated_accept_length(
    value: str, *, spec_algorithm: str | None, verify_width: int, draft_tree: bool
) -> int | None:
    """Validate the simulated tokens per verify step.

    Args:
        value: Raw ``TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN``; empty disables it.
        spec_algorithm: Speculative algorithm, or None without drafting.
        verify_width: Most tokens one verify step keeps per request.
        draft_tree: Whether the drafts form a tree rather than a chain.

    Returns:
        The average scaled by ``ACCEPT_LENGTH_SCALE``, or None when disabled.

    Raises:
        ValueError: Drafting is off or drafts a tree, or the value is not a
            number in ``[1, verify_width]``.
    """
    if not value:
        return None
    try:
        length = float(value)
    except ValueError:
        raise ValueError(
            f"TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN must be a number, got {value!r}"
        ) from None
    if spec_algorithm is None:
        raise ValueError(
            "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN requires speculative decoding"
        )
    if draft_tree:
        # Tree verify keeps the accepted path for compaction, which a
        # simulated width would run past.
        raise ValueError(
            "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN supports chain drafts only, "
            "not --speculative-eagle-topk > 1"
        )
    if not 1 <= length <= verify_width:
        raise ValueError(
            "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN must lie between 1 and the "
            f"verify width {verify_width:d}, got {value!r}"
        )
    return round(length * ACCEPT_LENGTH_SCALE)


def simulated_accept_lengths(
    cache_lengths: torch.Tensor, scaled_length: int
) -> torch.Tensor:
    """Widths that keep every request on the simulated average.

    A request keeps the tokens that take its cache length to the next value of
    ``floor(k * average)``. From then on its widths alternate between the
    whole numbers either side of the average and match it over any run of
    steps. Only the cache length is read, which every TP rank holds alike and
    graph replay reads live.

    Args:
        cache_lengths: ``[rows]`` committed cache length of each request.
        scaled_length: Average scaled by ``ACCEPT_LENGTH_SCALE``.

    Returns:
        ``[rows]`` int64 widths in ``[1, ceil(average)]``.
    """
    cached = cache_lengths.to(torch.int64)
    steps = ((cached + 1) * ACCEPT_LENGTH_SCALE + scaled_length - 1) // scaled_length
    return steps * scaled_length // ACCEPT_LENGTH_SCALE - cached


def simulated_output_tokens(
    tokens: torch.Tensor,
    candidates: torch.Tensor,
    verified_lengths: torch.Tensor,
    kept_lengths: torch.Tensor,
) -> torch.Tensor:
    """A chain verify's tokens, made to match the widths each row keeps.

    Verify writes a row's tokens only through the width it accepted; past
    that the sampler's buffer holds earlier steps' tokens, possibly another
    request's. Each kept position before the last emits the draft after it,
    as accepting that draft would, and so does the last one when verify did
    not write it.

    Args:
        tokens: ``[rows, N]`` tokens verify wrote.
        candidates: ``[rows, N]`` verify window: the last verified token,
            then the drafts.
        verified_lengths: ``[rows]`` widths verify accepted.
        kept_lengths: ``[rows]`` widths the step keeps, each in ``[1, N]``.

    Returns:
        ``[rows, N]`` tokens; each row emits its first ``kept_lengths``.
    """
    width = candidates.shape[1]
    following = torch.cat((candidates[:, 1:], candidates[:, -1:]), dim=1)
    position = torch.arange(width, device=tokens.device)
    drafted = (position < kept_lengths.unsqueeze(1) - 1) | (
        position >= verified_lengths.unsqueeze(1)
    )
    return torch.where(drafted, following.to(tokens.dtype), tokens)
