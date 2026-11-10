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

"""The gfx950 Gluon pieces of the Muse Spark monokernel inputs: expert
weights in the Gluon MXFP4 layout the kernels read, and the stand-in for
absent ranks that lets a bench or test launch the TP8 kernels on fewer GPUs.

The kernels have no registry entry to dispatch through: they are this
solution's own, so the imports below name it directly.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.benchmark.muse_spark_monokernel import WORLD, raw_experts


class _MoEWeights(torch.nn.Module):
    """The attributes the regular path's MoE weight preprocessing reads."""

    class _QuantConfig:
        use_dynamic_mxfp4_activations = True

    def __init__(self, w13, s13, w2, s2) -> None:
        super().__init__()
        self.w13_input_layout = "interleaved"
        self.quant_config = self._QuantConfig()
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)


def preprocessed_experts(rank: int, experts: int, device: torch.device):
    """:func:`raw_experts` through the regular path's MoE weight
    preprocessing (the gfx950 Gluon MXFP4 layout the monokernel reads).

    Returns:
        ``(w13, s13, w2, s2)`` on ``device``.
    """
    # The gfx950 Gluon MoE preprocessing has no registry entry: it is the
    # layout this AMD-only monokernel bakes in.
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )

    module = _MoEWeights(*(t.to(device) for t in raw_experts(rank, experts)))
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return (
        _extract_gluon_raw_w(module.w13_weight_triton_tensor),
        _extract_gluon_raw_s(module.w13_precision_config.b_mx_scale),
        _extract_gluon_raw_w(module.w2_weight_triton_tensor),
        _extract_gluon_raw_s(module.w2_precision_config.b_mx_scale),
    )


# A launch with fewer GPUs than ranks (the layer bench's --gpus 1, 2 or 4):
# before each launch the host writes the absent ranks' segment of every peer
# region with the launch's tag, as if those ranks had already put.


@triton.jit(do_not_specialize=["lay", "n"])
def _fill_absent_ranks(
    scratch, idx, bits, step, lay, n, SLOTS: tl.constexpr, BLOCK: tl.constexpr
):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tag = ((tl.load(step).to(tl.int64) * SLOTS + lay + 1) & 0xFFFFFFFF) << 32
    tl.store(
        scratch + tl.load(idx + o, mask=m), tl.load(bits + o, mask=m) | tag, mask=m
    )


def fill_absent_ranks(scratch, index, bits, step, layer: int) -> None:
    """Write ``bits | tag << 32`` at pairs ``index`` of ``scratch``, the tag
    the launch of ``layer`` at device step ``step`` puts.

    Args:
        scratch: int64 mailbox pairs.
        index: int64 pair indices (:func:`absent_index`).
        bits: int64 low words, one per index (:func:`ghost_payloads`).
        step: int32 ``[1]`` device step counter.
        layer: The launch's layer.
    """
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import LAYER_SLOTS

    n = index.numel()
    _fill_absent_ranks[(triton.cdiv(n, 1024),)](
        scratch, index, bits, step, layer, n, LAYER_SLOTS, 1024
    )


def _peer_slices(plan, parity: int):
    """(offset, segment) of copy ``parity`` of each rank-ordered block of
    each peer region (:func:`moe_tail.rank_blocks`), in every row: rank
    ``r`` writes segment ``r`` of a block."""
    return [(off, seg) for _, off, seg in _named_peer_slices(plan, parity)]


def _named_peer_slices(plan, parity: int):
    """:func:`_peer_slices` with each slice's region name first."""
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import moe_tail

    sizes = dict(plan.regions)
    out = []
    for r in plan.peers:
        blocks = moe_tail.rank_blocks(r, sizes[r])
        assert sum(n for _, n in blocks) == sizes[r], (r, blocks, sizes[r])
        for row in range(plan.rows):
            base = plan.offset(r) + parity * plan.peer_copy(r) + row * plan.row_pairs
            for start, n in blocks:
                assert n % WORLD == 0, (r, start, n)
                out.append((r, base + start, n // WORLD))
    return out


def absent_index(plan, parity: int, gpus: int) -> torch.Tensor:
    """Pair indices of the absent ranks' segments in copy ``parity``."""
    return torch.cat(
        [
            torch.arange(off + gpus * seg, off + WORLD * seg)
            for off, seg in _peer_slices(plan, parity)
        ]
    )


def absent_flags(plan, parity: int, gpus: int) -> torch.Tensor:
    """Which of :func:`absent_index`'s pairs are flags (``Plan.flags``)."""
    return torch.cat(
        [
            torch.full(((WORLD - gpus) * seg,), plan.is_flag(r))
            for r, _, seg in _named_peer_slices(plan, parity)
        ]
    )


def ghost_payloads(plan, parity: int, gpus: int, payloads=None) -> torch.Tensor:
    """The low words the fill writes into :func:`absent_index`'s pairs:
    ``payloads`` (zeros when None), with every flag pair 1, the payload of
    a raised flag (``Mailbox.poll_flags`` waits for exactly that)."""
    flags = absent_flags(plan, parity, gpus)
    out = torch.zeros(len(flags), dtype=torch.int64) if payloads is None else payloads
    out = out.clone()
    out[flags.to(out.device)] = 1
    return out


def rank0_payloads(plan, parity: int, gpus: int, scratch) -> torch.Tensor:
    """Payloads (low 32 bits) of rank 0's segments, once per absent rank."""
    return (
        torch.cat(
            [
                scratch[off : off + seg].repeat(WORLD - gpus)
                for off, seg in _peer_slices(plan, parity)
            ]
        )
        & 0xFFFFFFFF
    )
