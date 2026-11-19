# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2026 SGLang contributors
#
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


"""Byte layouts and build settings of the Kimi-K3 MoE layer tail.

The tail runs at Kimi-K3.1's TP8 sizes: hidden size 7168, latent size 3584,
at most eight rows. Its stages hand off through two byte-addressed buffers:

- a per-rank uncached scratch (:func:`scratch_layout`), for handoffs between
  the workgroups of one rank;
- an uncached peer buffer on every rank (:func:`peer_alloc_bytes`), which
  every rank writes into: two parity halves of tagged pairs
  (:func:`peer_layout`), the stop flags (:func:`flags_offset`) and the
  four-byte exchange words of the reductions across ranks
  (:func:`sent_layout`).

This module imports no GPU libraries at import time.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/layer.py``, ``kimi_k3/gfx950/stages/oproj.py``,
``kimi_k3/gfx950/stages/moe.py`` and ``common/guard.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

HIDDEN = 7168
LATENT = 3584
TP = 8
MAX_ROWS = 8
EXPERTS = 896
# This rank's widths at TP8: the attention output (96 heads of 128), the
# routed experts' intermediate (3072) and the shared expert's (6144).
ATTENTION = 96 * 128 // TP
ROUTED = 3072 // TP
SHARED = 6144 // TP
# The routed experts' MXFP4 weight arrangement the tail reads (the N16
# layout of the regular MoE kernels).
EXPERT_LAYOUT = "n16"
# The tagged post-attention-norm rows start at this scratch offset; the next
# layer's mix rows and their ready words follow them. It lies past every other
# scratch region of every build (checked in scratch_layout), so every build
# shares one scratch layout and one allocation size.
TAG_OFFSET = 0xC1000


@dataclass(frozen=True)
class KimiK3MoeTailBuild:
    """The compile-time settings of one tail build.

    ``tokens`` is the row count. It is a compile-time constant: a server serves
    one decode or verify width, so it compiles one value. ``nblocks`` is the
    number of completed attention-residual blocks the layer mixes, besides the
    running prefix. A ``reset`` layer starts a new block: its prefix restarts
    from this layer's attention output instead of adding it to the incoming
    stream. ``eps``, ``out_eps`` and ``ln_eps`` are the epsilons of the MLP
    attention-residual norm, the post-attention norm and the latent norm.

    With ``next_mix`` the tail also writes the next layer's attention input:
    its output folded into the next layer's FP32 attention-residual block
    partial, then RMS-normalized (as ``attnres_combine`` does). ``mix_eps`` is
    that mix's epsilon, None without it.
    """

    tokens: int
    nblocks: int
    reset: bool
    eps: float
    out_eps: float
    ln_eps: float
    next_mix: bool
    mix_eps: float | None

    def __post_init__(self):
        if not 1 <= self.tokens <= MAX_ROWS or not 1 <= self.nblocks <= 8:
            raise ValueError("The MoE tail takes 1-8 rows and 1-8 residual blocks")
        if (self.mix_eps is not None) != self.next_mix:
            raise ValueError("mix_eps is the next layer's mix's, given exactly with it")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _pairs(items) -> dict[str, tuple[int, int]]:
    # Regions of eight-byte tagged pairs: name -> (byte offset, byte size).
    result, offset = {}, 0
    for name, count in items:
        offset = _align(offset, 8)
        result[name] = (offset, count * 8)
        offset += count * 8
    return result


def _oproj_scratch(s: int, nb: int) -> dict[str, tuple[int, int]]:
    # The attention-residual mix: per-slice block partials, per-row block
    # weights, per-slice sums of squares and the mixed rows' ready flags.
    return _pairs(
        (
            ("part", 56 * s * (nb + 1) * 2),
            ("wgt", s * (nb + 2)),
            ("msq", 56 * s),
            ("xrdy", 56),
        )
    )


def _expert_scratch(s: int) -> dict[str, tuple[int, int]]:
    # The shared expert's gate/up outputs and activations, the latent
    # projection's partials, each row's experts and weights, the routed
    # experts' FP4 intermediates and scales, the latent norm's partials and
    # ready flags, and the shared down projection's ready flags; then the
    # normalized latent rows and this rank's shared-expert output columns
    # (BF16). ``rpart`` is not read; it is kept for the same reason as the
    # unread peer regions (see peer_layout).
    result = _pairs(
        (
            ("sgu", s * 768),
            ("h", s * 768 // 2),
            ("rpart", 7 * 4 * s * 16),
            ("lpart", 14 * 4 * s * 32),
            ("route", s * 32),
            ("inter", s * 16 * 48),
            ("isc", s * 16 * 12),
            ("lmsq", 28 * s),
            ("lnrdy", 28),
            ("shrdy", 56),
        )
    )
    end = max(offset + size for offset, size in result.values())
    result["ln"] = (_align(end, 16), s * LATENT * 2)
    end = sum(result["ln"])
    result["shp"] = (_align(end, 16), s * 896 * 2)
    return result


def scratch_layout(key: KimiK3MoeTailBuild) -> dict[str, tuple[int, int]]:
    """Return the scratch regions as name -> (byte offset, byte size)."""
    result = _oproj_scratch(key.tokens, key.nblocks)
    base = _align(max(offset + size for offset, size in result.values()), 16)
    for name, (offset, size) in _expert_scratch(key.tokens).items():
        result[name] = (base + offset, size)
    end = max(offset + size for offset, size in result.values())
    # The post-attention-norm rows (BF16).
    result["xrow"] = (_align(end, 16), key.tokens * HIDDEN * 2)
    assert sum(result["xrow"]) <= TAG_OFFSET, "scratch regions overlap the tagged rows"
    if key.next_mix:
        result.update(mix_scratch_layout())
    return result


def peer_layout(s: int) -> dict[str, tuple[int, int]]:
    """Return one parity half of the peer buffer's tagged regions.

    The tail reads ``rlogp``, the router logits' partials from every rank and
    quarter, and ``latq``, the quantized latent rows (FP4 payload and E8M0
    scales). The other regions are not read. They are kept because the
    offsets that follow them, of ``latq``, ``rlogp`` and the stop flags, are
    compile-time constants of the kernel: removing a region changes the
    generated code, and this layout is the one the kernel was validated with.
    """
    return _pairs(
        (
            ("ar", TP * s * HIDDEN // 2),
            ("rlog", TP * s * 112),
            ("lat", s * LATENT // 2),
            ("latq", s * (LATENT // 8 + LATENT // 32)),
            ("latr", TP * s * LATENT // 2),
            ("final", TP * s * HIDDEN // 2),
            ("rlogp", TP * 4 * s * 112),
        )
    )


def peer_bytes(s: int) -> int:
    """Return the bytes of both parity halves of the tagged regions."""
    return 2 * max(offset + size for offset, size in peer_layout(s).values())


def flags_offset() -> int:
    """Return the offset of the eight per-rank stop flags in the peer buffer."""
    return peer_bytes(MAX_ROWS)


def sent_layout() -> dict[str, tuple[int, int]]:
    """Return one parity half of the four-byte exchange words.

    Each is two BF16 values a rank sends to every peer: ``ars`` the attention
    output projection's slices, ``latrs`` the routed experts' latent outputs,
    ``finals`` the layer's output slices. 0xFFFFFFFF marks a word not sent
    yet; its consumer re-arms it after reading.
    """
    result, offset = {}, 0
    for name, width in (("ars", HIDDEN), ("latrs", LATENT), ("finals", HIDDEN)):
        size = TP * MAX_ROWS * width // 2 * 4
        result[name] = (offset, size)
        offset += _align(size, 256)
    return result


def sent_half_bytes() -> int:
    return max(offset + size for offset, size in sent_layout().values())


def sent_offset() -> int:
    """Return the offset of the exchange words in the peer buffer."""
    return _align(flags_offset() + 256, 4096)


def peer_alloc_bytes() -> int:
    """Return the bytes of each rank's peer buffer."""
    return sent_offset() + 2 * sent_half_bytes()


def mix_scratch_layout() -> dict[str, tuple[int, int]]:
    """Return the next-layer mix's scratch regions, after the tagged rows.

    ``mixrow`` holds the layer's output rows and ``mixrdy`` their ready
    words. Only this rank reads them, after the layer's own launch wrote
    them, so launches in stream order reuse them.
    """
    base = TAG_OFFSET + MAX_ROWS * HIDDEN // 2 * 8
    row_bytes = MAX_ROWS * HIDDEN * 2
    return {"mixrow": (base, row_bytes), "mixrdy": (base + row_bytes, 512)}


def scratch_alloc_bytes() -> int:
    """Return the bytes of each rank's scratch, for every build."""
    return max(offset + size for offset, size in mix_scratch_layout().values())


def init_peer_bytes(peer_bytes: torch.Tensor) -> None:
    """Put a rank's peer buffer in the start state the tail requires.

    ``peer_bytes`` is the rank's whole peer buffer as a uint8 tensor of at
    least :func:`peer_alloc_bytes` bytes (``PeerBuffer.bytes``). The tagged
    halves and the stop flags start at zero; every four-byte exchange word
    starts at 0xFFFFFFFF, "not sent yet", and the kernel re-arms each word
    after reading it. The scratch must start zeroed. Every rank initializes
    its buffers before any rank's first launch, which writes into its peers'
    exchange words without waiting.
    """
    import torch

    if peer_bytes.dtype != torch.uint8 or peer_bytes.dim() != 1:
        raise ValueError("the peer buffer must be a one-dimensional uint8 tensor")
    if peer_bytes.numel() < peer_alloc_bytes():
        raise ValueError(
            f"the peer buffer holds {peer_bytes.numel()} bytes, not at least "
            f"{peer_alloc_bytes()}"
        )
    peer_bytes.zero_()
    peer_bytes[sent_offset() : peer_alloc_bytes()].fill_(0xFF)
