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

The kernels are registered (``ops/model/muse_spark/monokernel/gluon.py``),
but the registry hands out only their launches: these helpers build the
inputs in the layouts of this one AMD solution, so they import it directly.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

# OWNED is re-exported for the draft-layer bench (gluon_inputs.OWNED).
from tokenspeed_kernel.benchmark.muse_spark_monokernel import (  # noqa: F401
    DRAFT_HEAD,
    HIDDEN,
    OWNED,
    WORLD,
    draft_attn_rank,
    draft_attn_shared,
    draft_mlp_rank,
    draft_mlp_shared,
    draft_rope_table,
    raw_experts,
)

_PAGE = 64  # KV cache tokens a page


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


def preprocess_experts(raw):
    """Raw MXFP4 expert shards (``(w13, s13, w2, s2)``, as :func:`raw_experts`
    makes them) through the regular path's MoE weight preprocessing: the
    gfx950 Gluon MXFP4 layout the monokernel reads, on the shards' device.

    Returns:
        ``(w13, s13, w2, s2)``.
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

    module = _MoEWeights(*raw)
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return (
        _extract_gluon_raw_w(module.w13_weight_triton_tensor),
        _extract_gluon_raw_s(module.w13_precision_config.b_mx_scale),
        _extract_gluon_raw_w(module.w2_weight_triton_tensor),
        _extract_gluon_raw_s(module.w2_precision_config.b_mx_scale),
    )


def preprocessed_experts(rank: int, experts: int, device: torch.device):
    """:func:`raw_experts` of ``rank`` on ``device`` through
    :func:`preprocess_experts`."""
    return preprocess_experts(tuple(t.to(device) for t in raw_experts(rank, experts)))


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


def draft_layer_inputs(prefix: int, rank: int, rows: int, device):
    """Every input of one draft layer launch on ``rank`` at ``prefix``
    context tokens: the host tensors (the attention and MLP halves' inputs,
    the context, positions, slots and rotary table), the packed
    ``DraftLayerWeights`` on ``device``, and the launch's runtime tensors.

    Returns:
        ``(host, weights, runtime)``.
    """
    parts, residual, in_gain, a_coef, a_base, q_norm, k_norm = draft_attn_shared(
        prefix, rows
    )
    qkv_w, o_w, ctx_k, ctx_v = draft_attn_rank(prefix, rank, rows)
    _, _, post_gain, m_coef, m_base = draft_mlp_shared(prefix, rows)
    gu_w, down_w = draft_mlp_rank(prefix, rank)
    end = prefix + rows
    pages = (end + _PAGE - 1) // _PAGE
    g = torch.Generator().manual_seed(prefix)
    table = torch.randperm(4 * pages + 8, generator=g)[: pages + 1].to(torch.int32)
    k_cache = torch.zeros((4 * pages + 8) * _PAGE, 1, DRAFT_HEAD, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    tok = torch.arange(prefix)
    cslot = table[tok // _PAGE].long() * _PAGE + tok % _PAGE
    k_cache[cslot, 0] = ctx_k[:prefix]
    v_cache[cslot, 0] = ctx_v[:prefix]
    btok = prefix + torch.arange(rows)
    slots = table[btok // _PAGE].long() * _PAGE + btok % _PAGE
    host = dict(
        parts=parts,
        residual=residual,
        in_gain=in_gain,
        post_gain=post_gain,
        a_coef=a_coef,
        a_base=a_base,
        m_coef=m_coef,
        m_base=m_base,
        q_norm=q_norm,
        k_norm=k_norm,
        qkv_w=qkv_w,
        o_w=o_w,
        gu_w=gu_w,
        down_w=down_w,
        ctx_k=ctx_k,
        ctx_v=ctx_v,
        positions=btok.to(torch.int32),
        slots=slots,
        cos_sin=draft_rope_table(end + 8),
        end=end,
    )
    weights = draft_layer_weights(host, rank, device)
    runtime = dict(
        x=parts[rank].to(device),
        residual=residual.to(device),
        residual_out=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        out=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        final_gain=None,
        normed=None,
        seq_len=torch.full((1,), end, dtype=torch.int32, device=device),
        positions=host["positions"].to(device),
        slots=slots.to(device),
        page_table=table.to(device)[None],  # one request's row
        k_cache=k_cache.to(device),
        v_cache=v_cache.to(device),
        cos_sin=host["cos_sin"].to(device),
    )
    return host, weights, runtime


def draft_layer_weights(host: dict, rank: int, device):
    """``rank``'s packed ``DraftLayerWeights`` on ``device`` of the host
    tensors of :func:`draft_layer_inputs` (``host``)."""
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_layer

    packed = draft_layer.pack_draft_layer(
        {
            "input_norm": host["in_gain"],
            "post_norm": host["post_gain"],
            "attn_coef": host["a_coef"],
            "mlp_coef": host["m_coef"],
            "attn_base": host["a_base"],
            "mlp_base": host["m_base"],
            "qkv": host["qkv_w"],
            "q_norm": host["q_norm"],
            "k_norm": host["k_norm"],
            "o_proj": host["o_w"],
            "gate_up": host["gu_w"],
            "down": host["down_w"],
        },
        rank,
    )
    return draft_layer.DraftLayerWeights(
        rank=rank,
        **{
            f: getattr(packed, f).to(device)
            for f in packed.__dataclass_fields__
            if f != "rank"
        },
    )
