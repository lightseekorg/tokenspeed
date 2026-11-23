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


"""The scale word of ``fp8_scaled`` index-K rows.

An ``fp8_scaled`` row stores FP8 E4M3 keys and one four-byte scale word per
128 elements (the DSA README, "Index-K plane formats"). The word's encoding
follows the GPU that scores the plane:

- sm90 stores the fp32 scale itself.
- Blackwell stores the scale rounded up to a power of two as four equal
  UE8M0 exponent bytes. DeepGEMM 2.8.2's sm100 MQA logits read that word as
  the MX scales of the row's four 32-element groups, and every other
  Blackwell reader decodes it the same way. Keys are quantized with the
  rounded scale, so a plane is self-consistent whichever leaf scores it.

Writers call :func:`index_k_scale_word` on the scale they would have stored
and quantize with the scale it returns; readers call
:func:`index_k_scale_value` on the loaded word. Both take the int32 view of
the word, so one code path serves either encoding.
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform

# Four equal UE8M0 bytes: one biased exponent replicated into every group.
_UE8M0_X4 = 0x01010101
# The four UE8M0 bytes of 1.0, the MX scales of an unscaled row or query.
UNIT_UE8M0_X4 = 0x7F7F7F7F


@functools.cache
def index_k_scale_ue8m0() -> bool:
    """Whether this GPU's ``fp8_scaled`` planes store UE8M0 scale words."""
    return current_platform().is_blackwell


def index_k_scale_dtype() -> torch.dtype:
    """The dtype a plane's scale words are viewed as: int32 UE8M0 or float32."""
    return torch.int32 if index_k_scale_ue8m0() else torch.float32


def round_index_k_scale(scale: torch.Tensor) -> torch.Tensor:
    """The fp32 scale a writer quantizes with: ``scale``, or on Blackwell the
    power of two at or above it."""
    if not index_k_scale_ue8m0():
        return scale
    bits = scale.float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent << 23).view(torch.float32)


def encode_index_k_scale(scale: torch.Tensor) -> torch.Tensor:
    """The stored scale words of fp32 scales already rounded by
    :func:`round_index_k_scale`: the scales themselves, or int32 UE8M0 words."""
    if not index_k_scale_ue8m0():
        return scale
    exponent = (scale.float().view(torch.int32) >> 23) & 0xFF
    return exponent * _UE8M0_X4


def decode_index_k_scale(words: torch.Tensor) -> torch.Tensor:
    """The fp32 scales of stored words (either encoding, given as its own
    dtype or as raw int32)."""
    if not index_k_scale_ue8m0():
        return words.view(torch.float32)
    return ((words.view(torch.int32) & 0xFF) << 23).view(torch.float32)


@triton.jit
def index_k_scale_word(scale, UE8M0: tl.constexpr):
    """``(scale to quantize with, int32 word to store)`` for fp32 ``scale``.

    With ``UE8M0`` the scale rounds up to a power of two and the word packs
    its biased exponent four times; otherwise the word is the scale's bits.
    """
    bits = scale.to(tl.int32, bitcast=True)
    if UE8M0:
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        return (exponent << 23).to(tl.float32, bitcast=True), exponent * 0x01010101
    else:
        return scale, bits


@triton.jit
def index_k_scale_value(word, UE8M0: tl.constexpr):
    """The fp32 scale of an int32 scale ``word`` of either encoding."""
    word = word.to(tl.int32, bitcast=True)
    if UE8M0:
        return ((word & 0xFF) << 23).to(tl.float32, bitcast=True)
    else:
        return word.to(tl.float32, bitcast=True)
