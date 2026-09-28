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

"""Shared budget packing for eager and CUDA-graph multimodal encoders."""

from collections.abc import Sequence


def pack_encoder_batches(
    token_counts: Sequence[int],
    metadata_sequences: Sequence[int],
    *,
    max_tokens: int,
    max_items: int | None,
    max_metadata_sequences: int | None,
) -> list[list[int]]:
    """Return stable smallest-first groups of original item indices.

    Token counts use the execution stage's units, supplied by the caller.
    Optional limits of None impose no extra constraint. An indivisible item
    exceeding a budget is returned alone; the caller owns eager fallback or
    admission rejection. Callers must restore original item order afterward.
    All ranks in an encoder TP group must supply identical counts and limits.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive or None")
    if max_metadata_sequences is not None and max_metadata_sequences <= 0:
        raise ValueError("max_metadata_sequences must be positive or None")
    if len(token_counts) != len(metadata_sequences):
        raise ValueError("Token and metadata counts must have the same length")
    if any(n < 0 for n in token_counts) or any(n < 0 for n in metadata_sequences):
        raise ValueError("Encoder counts must be nonnegative")
    groups: list[list[int]] = []
    group: list[int] = []
    tokens = sequences = 0
    for index in sorted(range(len(token_counts)), key=token_counts.__getitem__):
        if group and (
            tokens + token_counts[index] > max_tokens
            or (max_items is not None and len(group) >= max_items)
            or (
                max_metadata_sequences is not None
                and sequences + metadata_sequences[index] > max_metadata_sequences
            )
        ):
            groups.append(group)
            group = []
            tokens = sequences = 0
        group.append(index)
        tokens += token_counts[index]
        sequences += metadata_sequences[index]
    if group:
        groups.append(group)
    return groups
