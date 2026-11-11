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

"""TP8 contracts for the Kimi-K3 decode chain (MoE layers, AttnRes entries and
o-projection exits) against dense references.

Run on eight GPUs, e.g. two four-GPU nodes:
``torchrun --nnodes 2 --nproc-per-node 4 ... -m pytest <this file>``.
"""

import os
import random
import types

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode_support import (
    BATCHES,
    EXPERTS,
    HIDDEN,
    LATENT,
)

TOP_K, TP = 16, 8
I_RANK, SHARED_RANK = 3072 // TP, 6144 // TP
BETA, LINEAR_BETA, SCALE, EPS = 4.0, 25.0, 2.5, 1e-5
# FP32 logits agree to ~1e-7; tokens whose 16th and 17th choices are closer may differ.
TIE_MARGIN = 1e-5

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != TP, reason="requires TP8"
)

FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), torch.device("cuda", local_rank)


def _require_multicast():
    """Rank-agreed multicast probe; a one-sided skip hangs the rendezvous."""
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
    from tokenspeed_kernel.ops.moe._cute_dsl.latent_tail import (
        multicast_backend_available,
    )

    gather_fabric_map()
    ok = multicast_backend_available(dist.group.WORLD)
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        pytest.skip("platform has no rank-agreed multicast support")


def _dequant_fp4(packed, scales):
    """[rows, k/2] E2M1 pairs and [rows, k/32] E8M0 scales to FP32 [rows, k]."""
    table = torch.tensor(FP4, device=packed.device)
    values = torch.stack((table[(packed & 15).long()], table[(packed >> 4).long()]), -1)
    values = values.flatten(-2)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _mxfp8(x, *, floor):
    """Per-32 MXFP8 E4M3 round trip; ``floor`` picks the floor(log2 amax) - 8 scale."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=2.0**-126)
    if floor:
        exponent = torch.floor(torch.log2(amax)) - 8
    else:
        exponent = torch.ceil(torch.log2(amax / 448))
    scale = torch.exp2(exponent)
    quantized = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (quantized * scale).flatten(-2)


def _situ(gate, up):
    gate = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    return gate * LINEAR_BETA * torch.tanh(up / LINEAR_BETA)


class _Weights:
    """Every rank's shards of one layer from ``seed``, so each rank can build the
    reference; ``experts`` reuses another layer's (large) expert weights."""

    def __init__(self, device, seed, experts=None):
        g = torch.Generator(device=device).manual_seed(seed)

        def normal(*shape, std):
            return (torch.randn(shape, device=device, generator=g) * std).bfloat16()

        def byte(*shape, lo=0, hi=256):
            return torch.randint(
                lo, hi, shape, dtype=torch.uint8, device=device, generator=g
            )

        self.router = normal(EXPERTS, HIDDEN, std=0.02)
        self.bias = torch.randn(EXPERTS, device=device, generator=g) * 0.05
        self.down = normal(LATENT, HIDDEN, std=0.02)
        self.up = normal(HIDDEN, LATENT, std=0.02)
        self.norm = (
            1 + torch.randn(LATENT, device=device, generator=g) * 0.1
        ).bfloat16()
        self.shared_gate_up = [
            normal(2 * SHARED_RANK, HIDDEN, std=0.02) for _ in range(TP)
        ]
        self.shared_down = [normal(HIDDEN, SHARED_RANK, std=0.02) for _ in range(TP)]
        if experts is not None:
            self.w13, self.w13_scale = experts.w13, experts.w13_scale
            self.w2, self.w2_scale = experts.w2, experts.w2_scale
            return
        # MXFP4 experts with scales 2^-8 .. 2^-4, intermediate split by rank.
        self.w13 = [byte(EXPERTS, 2 * I_RANK, LATENT // 2) for _ in range(TP)]
        self.w13_scale = [
            byte(EXPERTS, 2 * I_RANK, LATENT // 32, lo=119, hi=124) for _ in range(TP)
        ]
        self.w2 = [byte(EXPERTS, LATENT, I_RANK // 2) for _ in range(TP)]
        self.w2_scale = [
            byte(EXPERTS, LATENT, I_RANK // 32, lo=119, hi=124) for _ in range(TP)
        ]


def _serving_experts(weights, rank):
    """This rank's experts in the TRT-LLM-gen serving layout the chain reads."""
    from tokenspeed_kernel.ops.moe.flashinfer import (
        flashinfer_trtllm_mxfp4_situ_moe_weights,
    )

    layer = types.SimpleNamespace(
        w13_weight=weights.w13[rank].clone(),
        w13_weight_scale=weights.w13_scale[rank].clone(),
        w2_weight=weights.w2[rank].clone(),
        w2_weight_scale=weights.w2_scale[rank].clone(),
        activation="situ",
        activation_situ_beta=BETA,
        activation_situ_linear_beta=LINEAR_BETA,
        w13_input_layout="concatenated",
        swiglu_arg=None,
        swiglu_beta=None,
    )
    flashinfer_trtllm_mxfp4_situ_moe_weights({}, layer)
    return layer


def _reference(weights, x):
    """FP32 MoE output for ``x`` and a mask of tokens whose top-16 has no near tie."""
    x32 = x.float()
    scores = torch.sigmoid(x32 @ weights.router.float().T)
    choice = scores + weights.bias
    ranked = choice.sort(dim=-1, descending=True).values
    clear = (ranked[:, TOP_K - 1] - ranked[:, TOP_K]) > TIE_MARGIN
    ids = choice.topk(TOP_K, dim=-1).indices
    routed = scores.gather(1, ids)
    routed = routed / (routed.sum(-1, keepdim=True) + 1e-20) * SCALE

    latent = _mxfp8((x32 @ weights.down.float().T).bfloat16(), floor=False)
    combined = torch.zeros(x.shape[0], LATENT, device=x.device)
    for t in range(x.shape[0]):
        chosen = ids[t]
        for r in range(TP):
            w13 = _dequant_fp4(weights.w13[r][chosen], weights.w13_scale[r][chosen])
            w2 = _dequant_fp4(weights.w2[r][chosen], weights.w2_scale[r][chosen])
            gate_up = w13 @ latent[t]
            h = _mxfp8(_situ(gate_up[:, :I_RANK], gate_up[:, I_RANK:]), floor=True)
            out = (w2 @ h.unsqueeze(-1)).squeeze(-1)
            combined[t] += (routed[t].unsqueeze(-1) * out).sum(0)
    rms = torch.rsqrt(combined.pow(2).mean(-1, keepdim=True) + EPS)
    normed = (combined * rms * weights.norm.float()).bfloat16().float()
    output = normed @ weights.up.float().T
    for r in range(TP):
        gate_up = x32 @ weights.shared_gate_up[r].float().T
        h = _situ(gate_up[:, :SHARED_RANK], gate_up[:, SHARED_RANK:]).bfloat16()
        output += h.float() @ weights.shared_down[r].float().T
    return output, clear


# Five MoE layers rotate through all three tail parities and, like the model's
# 92, wrap with 5 % 3 == 2; each has its own non-expert weights.
MOE_LAYERS = 5


@pytest.fixture(scope="module")
def moe_weights():
    """Every MoE layer's weights and this rank's experts in the serving layout."""
    rank, device = _setup()
    _require_multicast()
    first = _Weights(device, 1234)
    weights = (first,) + tuple(
        _Weights(device, 1234 + ordinal, experts=first)
        for ordinal in range(1, MOE_LAYERS)
    )
    return weights, _serving_experts(first, rank)


def _moe_chain(moe_weights, verify_width, decoders):
    """The owner and every MoE layer's chain, for every row count.

    MoE layer i follows decoder layer i and prefetches decoder layer i + 1's
    weights.
    """
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode import KimiK3DecodeMoE

    weights, experts = moe_weights
    following = [(d.in_proj, d.self_attn.o_proj.weight) for d in decoders[0][1:]]
    rank = dist.get_rank()
    owner = KimiK3DecodeMoE(
        group=dist.group.WORLD,
        batches=BATCHES,
        verify_width=verify_width,
        moe_layers=MOE_LAYERS,
        beta=BETA,
        linear_beta=LINEAR_BETA,
    )
    rows = slice(rank * (LATENT // TP), (rank + 1) * (LATENT // TP))
    up_rows = slice(rank * (HIDDEN // TP), (rank + 1) * (HIDDEN // TP))
    layers = [
        owner.layer(
            ordinal=ordinal,
            layer_idx=ordinal,
            router=w.router,
            correction_bias=w.bias,
            routed_scaling_factor=SCALE,
            renormalize=True,
            down_shard=w.down[rows].contiguous(),
            shared_gate_up=w.shared_gate_up[rank],
            shared_down=w.shared_down[rank],
            w13=experts.w13_weight,
            w13_scale=experts.w13_weight_scale,
            w2=experts.w2_weight,
            w2_scale=experts.w2_weight_scale,
            norm_weight=w.norm,
            norm_eps=EPS,
            up_shard=w.up[up_rows].contiguous(),
            next_attention=following[ordinal] if ordinal < len(following) else (),
        )
        for ordinal, w in enumerate(weights)
    ]
    return owner, layers


@pytest.fixture(scope="module", params=(1, 8), ids=("decode", "verify-8"))
def chain(request, moe_weights, decoders):
    """The chain for every row count, built for plain decode or a verify width of 8.

    A verify width of 8 picks the shared expert's verify splits.
    """
    owner, layers = _moe_chain(moe_weights, request.param, decoders)
    yield moe_weights[0], layers, owner.device, owner
    dist.barrier()


def _revert(policy, patch):
    """Set one scheduling policy that only moves work back to its old tables.

    late-release: the B5-8 routing kernel releases the experts at exit, and the
    shared expert joins the main stream before the tail.
    no-prefetch: no L2 prefetch, neither on the side stream nor in the experts.
    """
    from tokenspeed_kernel.ops.moe._cute_dsl import k3_decode_prefetch
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe import front

    if policy == "late-release":
        patch.setitem(front.EARLY_ROUTE, TP, "")
    else:
        patch.setattr(k3_decode_prefetch, "OPROJ_DOWN", "")
        patch.setattr(k3_decode_prefetch, "NEXT_ATTENTION", "")
        patch.setattr(k3_decode_prefetch, "EXPERTS_AHEAD", "")
        patch.setitem(k3_decode_prefetch.ATTENTION_MOE, "kda", "")
        patch.setitem(k3_decode_prefetch.ATTENTION_MOE, "mla", "")


@pytest.fixture(scope="module", params=("late-release", "no-prefetch"))
def reverted(request):
    return request.param


@pytest.fixture(scope="module")
def reverted_chain(reverted, chain, moe_weights, decoders):
    """``chain`` (same weights and verify width) with one policy reverted.

    It must match ``chain`` bit for bit.
    """
    with pytest.MonkeyPatch.context() as patch:
        _revert(reverted, patch)
        owner, layers = _moe_chain(moe_weights, chain[3].verify_width, decoders)
    yield moe_weights[0], layers, owner.device, owner
    dist.barrier()


def _assert_arms_differ(reverted, chain, reverted_chain):
    """The default chain takes the policy where the reverted one does not."""
    (_, layers, _, owner), (_, other, _, other_owner) = chain, reverted_chain
    if reverted == "late-release":
        gated = [b for b in BATCHES if layers[0].chain.front.gated(b)]
        assert gated and not any(other[0].chain.front.gated(b) for b in BATCHES)
    else:
        assert owner.prefetch.launches and owner.prefetch.tables
        assert not other_owner.prefetch.launches and not other_owner.prefetch.tables


def _inputs(batch, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, HIDDEN, device=device, generator=g).bfloat16()
    residual = torch.randn(batch, HIDDEN, device=device, generator=g).bfloat16()
    return x, residual


def _assert_close(weights, x, residual, out):
    expected, clear = _reference(weights, x)
    assert clear.any(), "every token tied at the 16th expert"
    delta = out.float() - residual.float()
    error = (delta - expected).norm(dim=-1) / expected.norm(dim=-1)
    worst = error[clear].max().item()
    assert worst < 0.02, f"row relative error {worst:.4f} (median {error.median():.4f})"


def _step(layers, x, residual):
    """Every MoE layer in model order, each accumulating onto the previous output.

    The step ends by rejoining the chain's side streams, as the model's forward
    does.
    """
    outputs = []
    for layer in layers:
        residual = layer.run(x, residual)
        outputs.append(residual)
    layers[0].owner.join()
    return outputs


def _check_step(weights, x, residual, outputs):
    for w, before, out in zip(weights, (residual, *outputs[:-1]), outputs):
        _assert_close(w, x, before, out)


@pytest.mark.parametrize("batch", BATCHES)
def test_chain_matches_the_reference_eagerly(chain, batch):
    weights, layers, device, _ = chain
    x, residual = _inputs(batch, device, 100 * batch)
    outputs = _step(layers, x, residual)
    torch.cuda.synchronize()
    _check_step(weights, x, residual, outputs)


# Row counts of 40 consecutive steps: every count, crossing the experts' B<=8 /
# B>8 modes in both directions.
SEQUENCE = random.Random(7).choices(BATCHES, k=40)
assert set(SEQUENCE) == set(BATCHES)
assert any(a <= 8 < b for a, b in zip(SEQUENCE, SEQUENCE[1:]))
assert any(a > 8 >= b for a, b in zip(SEQUENCE, SEQUENCE[1:]))


def test_chain_follows_changing_row_counts_eagerly(chain):
    weights, layers, device, _ = chain
    for step, batch in enumerate(SEQUENCE):
        x, residual = _inputs(batch, device, 2000 + step)
        outputs = _step(layers, x, residual)
        torch.cuda.synchronize()
        _check_step(weights, x, residual, outputs)


def _captured_steps(layers, device):
    """(graph, x, residual, outputs) of one captured ``_step`` per row count.

    Each is warmed up eagerly on a side stream first.
    """
    graphs = {}
    for batch in BATCHES:
        x, residual = _inputs(batch, device, batch)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _step(layers, x, residual)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = _step(layers, x, residual)
        graphs[batch] = graph, x, residual, outputs
    return graphs


def test_chain_follows_changing_row_counts_in_cuda_graphs(chain):
    """One graph per row count, replayed in the same mixed order with fresh inputs.

    Every layer re-arms what it read, and the first one also what the last layer
    may have published at the largest row count.
    """
    weights, layers, device, _ = chain
    graphs = _captured_steps(layers, device)
    for step, batch in enumerate(SEQUENCE):
        graph, x, residual, outputs = graphs[batch]
        fresh_x, fresh_residual = _inputs(batch, device, 3000 + step)
        x.copy_(fresh_x)
        residual.copy_(fresh_residual)
        graph.replay()
        torch.cuda.synchronize()
        _check_step(weights, x, residual, outputs)


def _assert_equal(actual, expected, what):
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(a, e), f"{what}: layer {i}"


def test_reverted_scheduling_matches_bit_for_bit(reverted, chain, reverted_chain):
    """A scheduling policy only moves work.

    The chain eagerly and in CUDA graphs equals the chain with the policy
    reverted, run eagerly, at every row count.
    """
    _assert_arms_differ(reverted, chain, reverted_chain)
    _, layers, device, _ = chain
    other = reverted_chain[1]
    for batch in BATCHES:
        x, residual = _inputs(batch, device, 5000 + batch)
        expected = _step(other, x, residual)
        actual = _step(layers, x, residual)
        torch.cuda.synchronize()
        _assert_equal(actual, expected, f"eager, {batch} rows")
    graphs = _captured_steps(layers, device)
    for step, batch in enumerate(SEQUENCE):
        graph, x, residual, outputs = graphs[batch]
        fresh_x, fresh_residual = _inputs(batch, device, 6000 + step)
        x.copy_(fresh_x)
        residual.copy_(fresh_residual)
        graph.replay()
        expected = _step(other, x, residual)
        torch.cuda.synchronize()
        _assert_equal(outputs, expected, f"step {step} ({batch} rows)")


# Per-rank o-projection input width (12 heads of 128) and the input projections;
# a Kimi-K3.1 DSA layer's adds its indexer's 160 key and head-weight rows to MLA's.
K_ATTN, MLA_ROWS, DSA_ROWS, KDA_ROWS = 1536, 3648, 3808, 6288


def _decoder(device, seed, *, layer_idx, prev_valid_blocks, write, rows):
    """A decoder layer's AttnRes and projection weights, every rank's o-proj shard."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, std):
        return torch.randn(shape, device=device, generator=g) * std

    def norm():
        weight = (1 + randn(HIDDEN, std=0.1)).bfloat16()
        return types.SimpleNamespace(weight=weight, variance_epsilon=1e-6)

    def proj():
        return types.SimpleNamespace(weight=randn(1, HIDDEN, std=0.05).bfloat16())

    o_proj = [randn(HIDDEN, K_ATTN, std=0.02).bfloat16() for _ in range(TP)]
    return types.SimpleNamespace(
        self_attention_res_norm=norm(),
        self_attention_res_proj=proj(),
        input_layernorm=norm(),
        mlp_res_norm=norm(),
        mlp_res_proj=proj(),
        post_attention_layernorm=norm(),
        self_attn=types.SimpleNamespace(
            o_proj=types.SimpleNamespace(weight=o_proj[dist.get_rank()])
        ),
        o_proj_shards=o_proj,
        in_proj=randn(rows, HIDDEN, std=0.02).bfloat16(),
        layer_idx=layer_idx,
        prev_valid_blocks=prev_valid_blocks,
        is_block_write_layer=write,
        block_write_idx=prev_valid_blocks if write else -1,
    )


def _attn_res(prefix, blocks, nb, norm, proj, out_norm, *, delta=None, write=-1):
    """Reference AttnRes on token-major ``blocks``; returns (out, prefix, blocks)."""
    from tokenspeed_kernel.ops.residual.torch import torch_attn_res_fwd

    prefix, block_major = prefix.clone(), blocks.transpose(0, 1).contiguous()
    out = torch_attn_res_fwd(
        layer_residual=prefix,
        block_residual=block_major,
        res_weight=proj.weight.reshape(-1),
        rms_weight=norm.weight,
        eps=norm.variance_epsilon,
        out_norm_weight=out_norm.weight,
        out_norm_eps=out_norm.variance_epsilon,
        delta=delta,
        num_valid_blocks=nb,
        block_write_idx=write,
    )
    return out, prefix, block_major.transpose(0, 1)


def _locals(batch, device, seed):
    """Every rank's o-projection input for one layer (the attention core's output)."""
    g = torch.Generator(device=device).manual_seed(seed)
    return [
        (torch.randn(batch, K_ATTN, device=device, generator=g) * 0.5).bfloat16()
        for _ in range(TP)
    ]


def _close(actual, expected, rows, what, tol=0.02):
    error = (actual.float() - expected.float()).norm(dim=-1)
    error = error / expected.float().norm(dim=-1).clamp(min=1e-6)
    worst = error[rows].max().item()
    assert worst < tol, f"{what}: row relative error {worst:.4f}"


# (prev_valid_blocks, writes a snapshot, input projection rows) per decoder layer:
# the first writes block 0, the fourth block 1; KDA and MLA-family (MLA, then
# DSA) projections alternate.
DECODERS = (
    (0, True, KDA_ROWS),
    (1, False, MLA_ROWS),
    (1, False, KDA_ROWS),
    (1, True, DSA_ROWS),
    (2, False, KDA_ROWS),
)
NUM_BLOCKS = 2


@pytest.fixture(scope="module")
def decoders():
    """One decoder layer before each MoE layer, and the closing AttnRes weights.

    The first decoder layer block-writes; the rest are ring-fed.
    """
    _, device = _setup()
    layers = tuple(
        _decoder(
            device,
            21 + i,
            layer_idx=i,
            prev_valid_blocks=nb,
            write=write,
            rows=rows,
        )
        for i, (nb, write, rows) in enumerate(DECODERS)
    )
    final = _decoder(device, 29, layer_idx=5, prev_valid_blocks=2, write=False, rows=8)
    return layers, final


def _layer_chain(moe_chain, decoders):
    """Every decoder layer's chain over ``moe_chain`` and the closing AttnRes."""
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode import KimiK3DecodeAttention

    weights, moe_layers, device, owner = moe_chain
    decoders, final = decoders
    attention = KimiK3DecodeAttention(
        group=dist.group.WORLD, batches=BATCHES, moe=owner
    )
    layers = [
        attention.layer(
            decoder,
            layer_idx=decoder.layer_idx,
            mla=decoder.in_proj.shape[0] in (MLA_ROWS, DSA_ROWS),
            first=decoder.layer_idx == 0,
            projection_weight=decoder.in_proj,
            moe=moe,
        )
        for decoder, moe in zip(decoders, moe_layers, strict=True)
    ]
    closing = attention.final(
        res_norm=final.self_attention_res_norm,
        res_proj=final.self_attention_res_proj,
        out_norm=final.input_layernorm,
        num_blocks=NUM_BLOCKS,
    )
    return weights, moe_layers, device, decoders, layers, (final, closing)


@pytest.fixture(scope="module")
def layer_chain(chain, decoders):
    return _layer_chain(chain, decoders)


@pytest.fixture(scope="module")
def reverted_layer_chain(reverted, reverted_chain, decoders):
    with pytest.MonkeyPatch.context() as patch:
        _revert(reverted, patch)
        return _layer_chain(reverted_chain, decoders)


def _layer_step(layer_chain, embedding, step):
    """One decode step through every layer; checks every stage against the
    reference fed with the chain's own previous outputs and returns the closing
    output and every layer's input projection."""
    weights, moe_layers, device, decoders, layers, (final, closing) = layer_chain
    batch = embedding.shape[0]
    prefix = torch.empty_like(embedding)
    blocks = embedding.new_empty(batch, NUM_BLOCKS, HIDDEN)
    entry, feed = torch.empty_like(embedding), torch.empty_like(embedding)
    rows = torch.ones(batch, dtype=torch.bool, device=device)
    delta, ring_expected, projections = embedding, None, []
    for i, (decoder, layer, moe) in enumerate(zip(decoders, layers, moe_layers)):
        nb, write = decoder.prev_valid_blocks, decoder.block_write_idx
        prefix_before, blocks_before = prefix.clone(), blocks.clone()
        if i == 0:
            h = layer.enter(delta, prefix, blocks, entry, ring=False)
            expected = _attn_res(
                delta,
                blocks_before,
                nb,
                decoder.self_attention_res_norm,
                decoder.self_attention_res_proj,
                decoder.input_layernorm,
                write=write,
            )
        else:
            h = layer.enter(None, prefix, blocks, entry, ring=True)
            torch.cuda.synchronize()
            # The ring row carries the MoE chain's own error; mix what was summed.
            summed = prefix_before.float() + ring_expected
            _close(prefix, summed, rows, f"layer {i} ring sum")
            expected = _attn_res(
                prefix,
                blocks_before,
                nb,
                decoder.self_attention_res_norm,
                decoder.self_attention_res_proj,
                decoder.input_layernorm,
                write=write,
            )
        torch.cuda.synchronize()
        _close(h, expected[0], rows, f"layer {i} entry")
        _close(prefix, expected[1], rows, f"layer {i} prefix")
        if write >= 0:
            _close(blocks[:, write], expected[2][:, write], rows, f"layer {i} snapshot")

        projected = layer.project(h)
        _close(projected, h.float() @ decoder.in_proj.float().T, rows, "projection")
        projections.append(projected)

        shards = _locals(batch, device, 1000 * step + i)
        local = shards[dist.get_rank()]
        reduced = sum(
            x.float() @ w.float().T for x, w in zip(shards, decoder.o_proj_shards)
        )
        prefix_before, blocks_before = prefix.clone(), blocks.clone()
        x = layer.leave(local, prefix, blocks, feed)
        torch.cuda.synchronize()
        start = torch.zeros_like(prefix_before) if write >= 0 else prefix_before
        expected = _attn_res(
            start,
            blocks_before,
            nb + decoder.is_block_write_layer,
            decoder.mlp_res_norm,
            decoder.mlp_res_proj,
            decoder.post_attention_layernorm,
            delta=reduced.bfloat16(),
        )
        _close(x, expected[0], rows, f"layer {i} exit")
        _close(prefix, expected[1], rows, f"layer {i} reduced prefix")

        moe.publish(x)
        ring_expected, clear = _reference(weights[i], x)
        rows = rows & clear
    out = torch.empty_like(embedding)
    prefix_before = prefix.clone()
    closing.run_ring(prefix, blocks, out)
    torch.cuda.synchronize()
    assert rows.any(), "every token tied at the 16th expert"
    _close(prefix, prefix_before.float() + ring_expected, rows, "closing ring sum")
    expected = _attn_res(
        prefix,
        blocks,
        NUM_BLOCKS,
        final.self_attention_res_norm,
        final.self_attention_res_proj,
        final.input_layernorm,
    )
    _close(out, expected[0], rows, "closing AttnRes")
    moe_layers[0].owner.join()
    return out, projections


@pytest.mark.parametrize("batch", BATCHES)
def test_layer_chain_matches_the_reference_eagerly(layer_chain, batch):
    embedding, _ = _inputs(batch, layer_chain[2], 50 * batch)
    _layer_step(layer_chain, embedding, 0)


def _run(layer_chain, embedding, shards):
    """One decode step through every layer with this rank's o-projection inputs;
    returns the closing output and every layer's input projection."""
    _, moe_layers, _, _, layers, (_, closing) = layer_chain
    prefix = torch.empty_like(embedding)
    blocks = embedding.new_empty(embedding.shape[0], NUM_BLOCKS, HIDDEN)
    entry, feed = torch.empty_like(embedding), torch.empty_like(embedding)
    projections = []
    for i, (layer, moe) in enumerate(zip(layers, moe_layers)):
        h = layer.enter(embedding, prefix, blocks, entry, ring=i > 0)
        projections.append(layer.project(h))
        moe.publish(layer.leave(shards[i], prefix, blocks, feed))
    out = closing.run_ring(prefix, blocks, torch.empty_like(embedding))
    moe_layers[0].owner.join()
    return out, projections


def _shards(batch, device, step):
    """This rank's o-projection input of every layer, as ``_layer_step`` draws them."""
    rank = dist.get_rank()
    return [_locals(batch, device, 1000 * step + i)[rank] for i in range(len(DECODERS))]


# Row counts of consecutive layer-chain steps: every count, crossing the
# experts' B<=8 / B>8 modes and the o-projection's two-shot (B>32) both ways.
LAYER_SEQUENCE = (64, 1, 32, 64, 8, 16, 2, 3, 64, 4, 5, 6, 7, 32, 16, 64, 1)
assert set(LAYER_SEQUENCE) == set(BATCHES)
for limit in (8, 32):
    pairs = list(zip(LAYER_SEQUENCE, LAYER_SEQUENCE[1:]))
    assert any(a <= limit < b for a, b in pairs)
    assert any(a > limit >= b for a, b in pairs)


def _captured_runs(layer_chain):
    """One captured ``_run`` per row count: (graph, embedding, shards, outputs).

    The outputs are (out, projections); each run is warmed up eagerly on a side
    stream first.
    """
    device, graphs = layer_chain[2], {}
    for batch in BATCHES:
        embedding, shards = _inputs(batch, device, batch)[0], _shards(batch, device, 0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run(layer_chain, embedding, shards)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = _run(layer_chain, embedding, shards)
        graphs[batch] = graph, embedding, shards, outputs
    return graphs


def _replay(graphs, step, batch, device):
    """Replay the graph of ``batch`` rows on step ``step``'s inputs."""
    graph, embedding, shards, outputs = graphs[batch]
    embedding.copy_(_inputs(batch, device, 4000 + step)[0])
    for shard, fresh in zip(shards, _shards(batch, device, step)):
        shard.copy_(fresh)
    graph.replay()
    return embedding, shards, outputs


def _assert_same_run(actual, expected, what):
    (out, projections), (expected_out, expected_projections) = actual, expected
    assert torch.equal(out, expected_out), f"{what}: output"
    for i, (p, q) in enumerate(zip(projections, expected_projections, strict=True)):
        assert torch.equal(p, q), f"{what}: projection {i}"


def test_layer_chain_follows_changing_row_counts_in_cuda_graphs(layer_chain):
    """The eager chain over a mixed row-count sequence, checked stage by stage.

    Then one graph per row count, replayed in the same order with the same
    inputs, matches every eager step bit for bit.
    """
    device = layer_chain[2]
    eager = [
        _layer_step(layer_chain, _inputs(batch, device, 4000 + step)[0], step)
        for step, batch in enumerate(LAYER_SEQUENCE)
    ]
    graphs = _captured_runs(layer_chain)
    for step, batch in enumerate(LAYER_SEQUENCE):
        _, _, outputs = _replay(graphs, step, batch, device)
        torch.cuda.synchronize()
        _assert_same_run(outputs, eager[step], f"step {step} ({batch} rows)")


def test_reverted_scheduling_matches_bit_for_bit_in_the_layer_chain(
    reverted, layer_chain, reverted_layer_chain
):
    """The layer chain's graphs equal the chain with a scheduling policy reverted.

    The graphs replay the mixed row-count sequence; the reverted chain runs
    eagerly on the same inputs.
    """
    if reverted == "no-prefetch":  # the attention's own prefetch points
        owners = (layer_chain[1][0].owner, reverted_layer_chain[1][0].owner)
        points = [{key[0] for key in owner.prefetch.launches} for owner in owners]
        assert {"oproj_down", "attention_moe"} <= points[0] and not points[1]
    device = layer_chain[2]
    graphs = _captured_runs(layer_chain)
    for step, batch in enumerate(LAYER_SEQUENCE):
        embedding, shards, outputs = _replay(graphs, step, batch, device)
        expected = _run(reverted_layer_chain, embedding, shards)
        torch.cuda.synchronize()
        _assert_same_run(outputs, expected, f"step {step} ({batch} rows)")
