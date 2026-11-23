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


from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from utils import make_mxfp4_moe_weights


@dataclass
class _SwiGLU:
    alpha: float = 1.0
    limit: float = 10.0


class Weights(torch.nn.Module):
    def __init__(self, raw, ep_rank, ep_size):
        super().__init__()
        for dest, src in (
            ("w13_weight", "w13_weight"),
            ("w13_weight_scale", "w13_scale"),
            ("w2_weight", "w2_weight"),
            ("w2_weight_scale", "w2_scale"),
        ):
            self.register_buffer(dest, raw[src].clone())
        self.num_local_experts = raw["w13_weight"].shape[0]
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_experts = self.num_local_experts * ep_size
        self.activation = "swiglu"
        self.w13_input_layout = "concatenated"
        self.swiglu_arg = _SwiGLU()
        self.swiglu_beta = 0.0
        self._moe_backend_state: object | None = None


def roundtrip(x):
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


def reference(x, w, ids, weights, quantize):
    w13 = dequantize_mxfp4(w.w13_weight, w.w13_weight_scale)
    w2 = dequantize_mxfp4(w.w2_weight, w.w2_weight_scale)
    xx = roundtrip(x) if quantize else x.float()
    contributions = []
    for expert in range(w.num_local_experts):
        gu = (xx @ w13[expert].T).to(torch.bfloat16).float()
        gate, up = gu.chunk(2, -1)
        act = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).to(
            torch.bfloat16
        )
        aa = roundtrip(act) if quantize else act.float()
        down = (aa @ w2[expert].T).to(torch.bfloat16).float()
        route_weight = (
            (ids == expert + w.ep_rank * w.num_local_experts) * weights
        ).sum(-1)
        contributions.append((down * route_weight[:, None]).to(torch.bfloat16))
    return torch.stack(contributions).sum(0)


def require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")


@pytest.mark.parametrize("tokens", [1, 32, 192, 513])
def test_sm12x_ep_graph(tokens):
    require_sm120()
    gen = torch.Generator(device="cuda").manual_seed(45)
    raw = make_mxfp4_moe_weights(2, 5120, 2304, gen)
    canonical = Weights(raw, 1, 2)
    w = Weights(raw, 1, 2)
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=2,
        ispp=2304,
        hidden=5120,
        swiglu_form="standard",
        activation_clamped=True,
        expert_id_repeats=False,
        num_experts=4,
        top_k=2,
        internal_activation_dtype="fp8",
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="flashinfer",
    )
    moe_process_weights(plan, w)
    x = torch.randn(tokens, 5120, device="cuda", generator=gen).to(torch.bfloat16) * 0.2
    if tokens == 1:
        x = x.as_strided(x.shape, (0, 1))
    ids = torch.rand(tokens, 4, device="cuda", generator=gen).topk(2, -1).indices
    weights = torch.rand(tokens, 2, device="cuda", generator=gen)
    weights /= weights.sum(-1, keepdim=True)

    def run():
        return moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for route in ([2, 3], [0, 1], [0, 3], [2, -1]):
        ids.copy_(torch.tensor(route, device="cuda").expand_as(ids))
        graph.replay()
        eager = run()
        torch.testing.assert_close(actual, eager, atol=0.01, rtol=0.03)
        expected = reference(x, canonical, ids, weights, True)
        assert torch.isfinite(actual).all()
        if route == [0, 1]:
            assert torch.count_nonzero(actual) == 0
        else:
            # The reference uses group-32 activation scales and BF16
            # intermediates; native fused W4A8 uses group-128 scales.
            error = (actual.float() - expected.float()).norm()
            assert error / expected.float().norm() < 0.05
