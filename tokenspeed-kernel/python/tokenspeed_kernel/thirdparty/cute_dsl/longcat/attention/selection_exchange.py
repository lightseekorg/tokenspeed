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

"""Selection transport layout and tag ABI for the fused A0 engines.

The indexer writes Lamport-tagged indices and lengths to the peers' current
payload slots. Workers poll those words in the sparse MLA routine. The fused
indexer retains the final system fence and generation-flag publication.
"""

NUM_TOKENS = 8
TOPK = 2048
INDICES_WORDS = NUM_TOKENS * TOPK  # 16384
LENS_WORDS = NUM_TOKENS
NUM_RANKS = 8
DOUBLE_SLOTS = 2

# One index-K row contains 128 FP8 bytes and its FP32 scale.
INDEX_K_WORDS = 33

# Workspace int32-word layout per rank region:
#   [0, PAYLOAD_WORDS): slot 0 payload
#   [PAYLOAD_WORDS, 2*PAYLOAD_WORDS): slot 1 payload
#   [2*PAYLOAD_WORDS]: flag (generation published by rank 0)
#
# Lamport tagging of the indices and lens words (engine P4 publish only):
# every 32-bit word carries ``generation & TAG_MASK`` in its top byte and
# the value in the low 24 bits (slots sign-extended from 24 bits, so ``-1``
# padding survives; physical slot ids must stay below 2**23).  A consumer
# that polls its own words for the step's tag sees a row as soon as the
# bytes land, without waiting for the publisher's system-scope fence and
# flag store; the flag is still written for consumers that do not decode.
TAG_SHIFT = 24
TAG_MASK = 0xFF
VALUE_MASK = (1 << TAG_SHIFT) - 1
MAX_TAGGED_SLOT = 1 << (TAG_SHIFT - 1)


def payload_words(index_k_words: int) -> int:
    """Int32 words of one slot's payload for the given index-K row width."""
    return INDICES_WORDS + LENS_WORDS + NUM_TOKENS * index_k_words


def region_words(index_k_words: int) -> int:
    """Int32 words of one rank's symmetric region (two slots + flag)."""
    return DOUBLE_SLOTS * payload_words(index_k_words) + 1


__all__ = [
    "DOUBLE_SLOTS",
    "INDEX_K_WORDS",
    "INDICES_WORDS",
    "LENS_WORDS",
    "MAX_TAGGED_SLOT",
    "NUM_RANKS",
    "NUM_TOKENS",
    "TAG_MASK",
    "TAG_SHIFT",
    "TOPK",
    "VALUE_MASK",
    "payload_words",
    "region_words",
]
