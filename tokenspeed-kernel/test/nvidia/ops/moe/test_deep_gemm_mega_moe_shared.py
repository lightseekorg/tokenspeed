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

"""DeepGEMM MegaMoE running an MXFP8 shared expert inside its launch."""

import importlib.util
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn

if not (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and importlib.util.find_spec("deep_gemm") is not None
):
    pytest.skip("DeepGEMM MegaMoE needs an SM100-family GPU", allow_module_level=True)

from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.moe import deep_gemm as mega_moe

EXPERTS, HIDDEN, INTER, TOP_K, LIMIT = 16, 2048, 512, 4, 10.0


class _Experts(nn.Module):
    """The routed MXFP4 weights and fields the MegaMoE kernel functions read."""

    def __init__(self):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(1)

        def tensor(shape, low, high):
            return nn.Parameter(
                torch.randint(
                    low, high, shape, dtype=torch.uint8, device="cuda", generator=g
                ),
                requires_grad=False,
            )

        e, h, i = EXPERTS, HIDDEN, INTER
        self.w13_weight = tensor((e, 2 * i, h // 2), 0, 256)
        self.w13_weight_scale = tensor((e, 2 * i, h // 32), 118, 122)
        self.w2_weight = tensor((e, h, i // 2), 0, 256)
        self.w2_weight_scale = tensor((e, h, i // 32), 118, 122)
        self.tp_size = 1
        self.num_local_experts = self.num_experts = e
        self.hidden_size, self.intermediate_size, self.top_k = h, i, TOP_K
        self.swiglu_arg = type("SwigluArg", (), {"limit": LIMIT})()
        self._moe_backend_state = None


def _fp8_round_trip(x):
    """Group-32 UE8M0 E4M3 quantize and dequantize, as the input staging rounds."""
    tokens, size = x.shape
    groups = x.float().view(tokens, size // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(1e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)[..., None]
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).view(tokens, size)


def _dequantize(codes, scales):
    scale = (scales.int() << 23).view(torch.float32)
    return (codes.float().view(codes.shape[0], -1, 32) * scale[..., None]).view(
        codes.shape
    )


def _shared_weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (torch.randn(2 * INTER, HIDDEN, device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    down = (torch.randn(HIDDEN, INTER, device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )

    def scales(rows, columns):
        return torch.randint(
            117, 121, (rows, columns), dtype=torch.uint8, device="cuda", generator=g
        )

    return gate_up, scales(2 * INTER, HIDDEN // 32), down, scales(HIDDEN, INTER // 32)


@pytest.fixture
def process_group():
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{tmp}/rendezvous",
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=10),
            device_id=torch.cuda.current_device(),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def _apply(plan, w, x, weights, ids):
    return mega_moe.deep_gemm_mxfp4_mega_moe_apply(
        plan, x, w, None, weights, ids, None, None, True, False
    )


@pytest.mark.usefixtures("process_group")
def test_fused_shared_expert_adds_its_output():
    plan = {
        "persistent_max_num_tokens_per_gpu": 256,
        "process_group": None,
        "fast_math": True,
    }
    w = _Experts()
    mega_moe.deep_gemm_mxfp4_mega_moe_process_weights(plan, w)
    gate_up, gate_up_scale, down, down_scale = _shared_weights(2)
    # Reference: E4M3 inputs, BF16 gate/up as the kernel clamps them, then the
    # FP8 round trip of the activations before the down projection.
    w1, w2 = _dequantize(gate_up, gate_up_scale), _dequantize(down, down_scale)
    g = torch.Generator(device="cuda").manual_seed(3)
    cases = []
    for tokens in (1, 64, 200):
        x = torch.randn(
            tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g
        )
        ids = torch.stack(
            [
                torch.randperm(EXPERTS, device="cuda", generator=g)[:TOP_K]
                for _ in range(tokens)
            ]
        ).int()
        weights = torch.rand(tokens, TOP_K, device="cuda", generator=g)
        routed = _apply(plan, w, x, weights, ids).clone()
        gate_up_out = (_fp8_round_trip(x) @ w1.T).bfloat16()
        activation = torch.empty(tokens, INTER, device="cuda", dtype=torch.bfloat16)
        silu_and_mul(gate_up_out, activation, limit=LIMIT)
        shared = _fp8_round_trip(activation) @ w2.T
        # Zero routing weights isolate the shared expert from the BF16 rounding
        # of a much larger routed sum.
        cases.append((x, ids, weights, torch.zeros_like(weights), routed, shared))

    assert mega_moe.fuse_deep_gemm_mxfp4_mega_moe_shared_expert(
        plan, w, gate_up, gate_up_scale, down, down_scale
    )
    for x, ids, weights, no_routes, routed, shared in cases:
        got = _apply(plan, w, x, no_routes, ids).float()
        # The kernel re-quantizes the SwiGLU output from FP32, the reference
        # from BF16; an E4M3 code flip moves an element by up to 1/16.
        assert (got - shared).norm() / shared.norm() < 0.03
        assert (
            torch.nn.functional.cosine_similarity(
                got.flatten(), shared.flatten(), dim=0
            )
            > 0.999
        )
        both = _apply(plan, w, x, weights, ids).float()
        expected = routed.float() + shared
        assert (both - expected).norm() / expected.norm() < 0.01


@pytest.mark.usefixtures("process_group")
def test_fused_shared_expert_takes_an_empty_batch():
    # A DP rank with no local tokens still joins the MegaMoE launch.
    plan = {
        "persistent_max_num_tokens_per_gpu": 64,
        "process_group": None,
        "fast_math": True,
    }
    w = _Experts()
    mega_moe.deep_gemm_mxfp4_mega_moe_process_weights(plan, w)
    assert mega_moe.fuse_deep_gemm_mxfp4_mega_moe_shared_expert(
        plan, w, *_shared_weights(5)
    )
    x = torch.empty(0, HIDDEN, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(0, TOP_K, device="cuda")
    ids = torch.empty(0, TOP_K, device="cuda", dtype=torch.int32)
    out = _apply(plan, w, x, weights, ids)
    torch.cuda.synchronize()
    assert out.shape == (0, HIDDEN)


@pytest.mark.usefixtures("process_group")
def test_mismatched_shared_expert_stays_unfused():
    plan = {
        "persistent_max_num_tokens_per_gpu": 64,
        "process_group": None,
        "fast_math": True,
    }
    w = _Experts()
    mega_moe.deep_gemm_mxfp4_mega_moe_process_weights(plan, w)
    gate_up, gate_up_scale, down, down_scale = _shared_weights(4)
    # Twice the routed intermediate size: not one routed expert's shape.
    wide = gate_up.repeat(2, 1), gate_up_scale.repeat(2, 1)
    before = [t.clone() for t in (*wide, down, down_scale)]
    assert not mega_moe.fuse_deep_gemm_mxfp4_mega_moe_shared_expert(
        plan, w, *wide, down, down_scale
    )
    for tensor, original in zip((*wide, down, down_scale), before):
        assert torch.equal(tensor.view(torch.uint8), original.view(torch.uint8))
    assert w._moe_backend_state.num_shared_experts == 0
