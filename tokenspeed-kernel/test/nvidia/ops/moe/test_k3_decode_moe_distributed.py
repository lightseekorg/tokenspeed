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

"""TP8 contracts for the Kimi-K3 decode MoE chain against a dense reference.

Run on eight GPUs, e.g. two four-GPU nodes:
``torchrun --nnodes 2 --nproc-per-node 4 ... -m pytest <this file>``.
"""

import os
import types

import pytest
import torch
import torch.distributed as dist

H, L, E, K = 7168, 3584, 896, 16
TP = 8
I_RANK, SHARED_RANK = 3072 // TP, 6144 // TP
BETA, LINEAR_BETA, SCALE, EPS = 4.0, 25.0, 2.5, 1e-5
BATCHES = (*range(1, 9), 16, 32, 64)
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
        normal = lambda *s, std: (
            torch.randn(s, device=device, generator=g) * std
        ).bfloat16()
        byte = lambda *s, lo=0, hi=256: torch.randint(
            lo, hi, s, dtype=torch.uint8, device=device, generator=g
        )
        self.router = normal(E, H, std=0.02)
        self.bias = torch.randn(E, device=device, generator=g) * 0.05
        self.down = normal(L, H, std=0.02)
        self.up = normal(H, L, std=0.02)
        self.norm = (1 + torch.randn(L, device=device, generator=g) * 0.1).bfloat16()
        self.shared_gate_up = [normal(2 * SHARED_RANK, H, std=0.02) for _ in range(TP)]
        self.shared_down = [normal(H, SHARED_RANK, std=0.02) for _ in range(TP)]
        if experts is not None:
            self.w13, self.w13_scale = experts.w13, experts.w13_scale
            self.w2, self.w2_scale = experts.w2, experts.w2_scale
            return
        # MXFP4 experts with scales 2^-8 .. 2^-4, intermediate split by rank.
        self.w13 = [byte(E, 2 * I_RANK, L // 2) for _ in range(TP)]
        self.w13_scale = [
            byte(E, 2 * I_RANK, L // 32, lo=119, hi=124) for _ in range(TP)
        ]
        self.w2 = [byte(E, L, I_RANK // 2) for _ in range(TP)]
        self.w2_scale = [byte(E, L, I_RANK // 32, lo=119, hi=124) for _ in range(TP)]


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
    clear = (ranked[:, K - 1] - ranked[:, K]) > TIE_MARGIN
    ids = choice.topk(K, dim=-1).indices
    routed = scores.gather(1, ids)
    routed = routed / (routed.sum(-1, keepdim=True) + 1e-20) * SCALE

    latent = _mxfp8((x32 @ weights.down.float().T).bfloat16(), floor=False)
    combined = torch.zeros(x.shape[0], L, device=x.device)
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


@pytest.fixture(scope="module")
def chain():
    rank, device = _setup()
    _require_multicast()
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode import KimiK3DecodeMoE

    first = _Weights(device, 1234)
    weights = (first, _Weights(device, 5678, experts=first))
    experts = _serving_experts(first, rank)
    owner = KimiK3DecodeMoE(
        group=dist.group.WORLD,
        batches=BATCHES,
        moe_layers=2,
        beta=BETA,
        linear_beta=LINEAR_BETA,
    )
    rows = slice(rank * (L // TP), (rank + 1) * (L // TP))
    up_rows = slice(rank * (H // TP), (rank + 1) * (H // TP))
    layers = [
        owner.layer(
            ordinal=ordinal,
            layer_idx=layer_idx,
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
        )
        for ordinal, (w, layer_idx) in enumerate(zip(weights, (1, 4)))
    ]
    yield weights, layers, device
    dist.barrier()


def _inputs(batch, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, H, device=device, generator=g).bfloat16()
    residual = torch.randn(batch, H, device=device, generator=g).bfloat16()
    return x, residual


def _assert_close(weights, x, residual, out):
    expected, clear = _reference(weights, x)
    assert clear.any(), "every token tied at the 16th expert"
    delta = out.float() - residual.float()
    error = (delta - expected).norm(dim=-1) / expected.norm(dim=-1)
    worst = error[clear].max().item()
    assert worst < 0.02, f"row relative error {worst:.4f} (median {error.median():.4f})"


@pytest.mark.parametrize("batch", BATCHES)
def test_chain_matches_the_reference_eagerly(chain, batch):
    weights, layers, device = chain
    for step, (w, layer) in enumerate(zip(weights, layers)):
        x, residual = _inputs(batch, device, 100 * batch + step)
        out = layer.run(x, residual)
        torch.cuda.synchronize()
        _assert_close(w, x, residual, out)


@pytest.mark.parametrize("batch", (1, 16))
def test_chain_replays_in_a_cuda_graph_with_fresh_inputs(chain, batch):
    """Two layers per graph replay; each replay re-arms the mailboxes it read."""
    weights, layers, device = chain
    x, residual = _inputs(batch, device, 7)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layers[1].run(layers[0].run(x, residual), residual)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first = layers[0].run(x, residual)
        second = layers[1].run(x, first)
    for seed in (11, 12, 13):
        fresh_x, fresh_residual = _inputs(batch, device, seed)
        x.copy_(fresh_x)
        residual.copy_(fresh_residual)
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(weights[0], x, residual, first)
        _assert_close(weights[1], x, first, second)
