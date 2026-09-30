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

"""Eight-rank numerical checks for the registered Petit

Run with torchrun --standalone --nproc_per_node=8 -m pytest --import-mode=importlib
<path-to-this-file>. Optional PETIT_OUTPUT_DIR saves outputs for an exact
before/after comparison. Local fixtures are independent of benchmark timing
scope. Normal single-process pytest skips the GPU test.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import tokenspeed_kernel
import torch
import torch.distributed as dist
from utils import is_cdna4


@dataclass(frozen=True)
class _BenchmarkProfile:
    name: str
    experts: int
    top_k: int
    hidden: int
    logical_intermediate: int
    compute_intermediate: int
    activation: str
    has_bias: bool


_PROFILES = {
    "gpt_oss_120b": _BenchmarkProfile(
        name="gpt_oss_120b",
        experts=128,
        top_k=4,
        hidden=2880,
        logical_intermediate=2880,
        compute_intermediate=3072,
        activation="swiglu",
        has_bias=True,
    ),
    "dsv4": _BenchmarkProfile(
        name="dsv4",
        experts=384,
        top_k=6,
        hidden=7168,
        logical_intermediate=3072,
        compute_intermediate=3072,
        activation="swiglu",
        has_bias=False,
    ),
    "kimi_k3": _BenchmarkProfile(
        name="kimi_k3",
        experts=896,
        top_k=16,
        hidden=3584,
        logical_intermediate=3072,
        compute_intermediate=3072,
        activation="situ",
        has_bias=False,
    ),
}


def _parameter(shape: tuple[int, ...], device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.uint8, device=device),
        requires_grad=False,
    )


class _TestMoeLayer(torch.nn.Module):
    def __init__(self, profile: _BenchmarkProfile, device: torch.device) -> None:
        super().__init__()
        local_experts = profile.experts // 8
        self.num_experts = profile.experts
        self.top_k = profile.top_k
        self.hidden_size = profile.hidden
        self.intermediate_size = profile.logical_intermediate
        self.num_local_experts = local_experts
        self.ep_size = 8
        self.tp_size = 1
        self.activation = profile.activation
        self.activation_situ_beta = 4.0 if profile.activation == "situ" else None
        self.activation_situ_linear_beta = (
            25.0 if profile.activation == "situ" else None
        )
        self.swiglu_beta = 1.0 if profile.has_bias else None
        self.swiglu_arg = (
            argparse.Namespace(alpha=1.702, limit=7.0) if profile.has_bias else None
        )
        self.w13_input_layout = "interleaved" if profile.has_bias else "concatenated"
        self.register_parameter(
            "w13_weight",
            _parameter(
                (
                    local_experts,
                    2 * profile.logical_intermediate,
                    profile.hidden // 2,
                ),
                device,
            ),
        )
        self.register_parameter(
            "w13_weight_scale",
            _parameter(
                (
                    local_experts,
                    2 * profile.logical_intermediate,
                    profile.hidden // 32,
                ),
                device,
            ),
        )
        self.register_parameter(
            "w2_weight",
            _parameter(
                (
                    local_experts,
                    profile.hidden,
                    profile.logical_intermediate // 2,
                ),
                device,
            ),
        )
        self.register_parameter(
            "w2_weight_scale",
            _parameter(
                (
                    local_experts,
                    profile.hidden,
                    profile.logical_intermediate // 32,
                ),
                device,
            ),
        )
        if profile.has_bias:
            self.register_parameter(
                "w13_weight_bias",
                torch.nn.Parameter(
                    torch.zeros(
                        (local_experts, 2 * profile.logical_intermediate),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "w2_weight_bias",
                torch.nn.Parameter(
                    torch.zeros(
                        (local_experts, profile.hidden),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("w13_weight_bias", None)
            self.register_parameter("w2_weight_bias", None)


def _make_plan(profile: _BenchmarkProfile) -> dict:
    return tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation=profile.activation,
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="gluon_petit",
        ep_size=8,
        ispp=profile.logical_intermediate,
        fp8_scale_block_shape=None,
        internal_activation_dtype="mxfp4",
        with_bias=profile.has_bias,
        process_group=None,
        hidden=profile.hidden,
        swiglu_form=(
            "generalized"
            if profile.has_bias
            else "standard" if profile.activation == "swiglu" else None
        ),
        activation_clamped=profile.has_bias,
        expert_id_repeats=False,
        fast_math=True,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution="gluon",
    )


def _routing(
    tokens: int,
    rank: int,
    profile: _BenchmarkProfile,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
    route = torch.arange(profile.top_k, dtype=torch.int32, device=device)[None, :]
    local_experts = profile.experts // 8
    destination = (rank + token + token // 16 + route) % 8
    local = (token * profile.top_k + route) % local_experts
    ids = (destination * local_experts + local).to(torch.int32).contiguous()
    raw_weights = 1.0 + ((token + 2 * route) % 7).to(torch.float32)
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()
    return ids, weights


def _quantized_reference(
    value: torch.Tensor, *, input_quantization: bool
) -> torch.Tensor:
    """Reference input/intermediate scales and nearest-even E2M1 values."""
    shape = value.shape
    blocks = value.float().reshape(-1, 32)
    maximum = blocks.abs().amax(dim=1)
    if input_quantization:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    else:
        bits = maximum.contiguous().view(torch.int32)
        exponent = (((bits + 0x00400000) & 0x7F800000) >> 23).clamp(min=2) - 2
    scale = (exponent << 23).view(torch.float32)[:, None]
    # A zero scale denotes an all-zero block; avoid division by zero in the oracle.
    scale = torch.where(scale == 0, 1.0, scale)
    normalized = blocks.abs() / scale
    # Tie priority is even E2M1 code, matching the hardware RNE conversion.
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0], device=value.device)
    closest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[closest] * blocks.sign() * scale).reshape(shape)


def _weights(layer, profile, rank: int) -> None:
    """Sparse nonzero weights give an independent, inexpensive exact matmul oracle."""
    device = layer.w13_weight.device
    inter = profile.logical_intermediate
    hidden = profile.hidden
    rows = torch.arange(2 * inter, device=device)
    gate = rows % 2 == 0 if profile.has_bias else rows < inter
    channel = rows // 2 if profile.has_bias else rows % inter
    cols = (channel * 7 + torch.where(gate, 0, 3)) % hidden
    for local in range(profile.experts // 8):
        expert = rank * (profile.experts // 8) + local
        codes = torch.where(gate, 2 + expert % 3, 2 + (expert + 1) % 3).to(torch.uint8)
        layer.w13_weight[local, rows, cols // 2] = (codes << ((cols % 2) * 4)).to(
            torch.uint8
        )
        out_rows = torch.arange(hidden, device=device)
        out_cols = (out_rows * 11 + expert) % inter
        layer.w2_weight[local, out_rows, out_cols // 2] = (
            (2 + expert % 3) << ((out_cols % 2) * 4)
        ).to(torch.uint8)
        if profile.has_bias:
            layer.w13_weight_bias[local].copy_(
                torch.where(gate, 0.03125 * (expert % 3 + 1), -0.03125)
            )
            layer.w2_weight_bias[local].fill_(0.015625 * (expert % 5 - 2))
    layer.w13_weight_scale.data.fill_(123)
    layer.w2_weight_scale.data.fill_(123)


def _reference(x, ids, weights, profile):
    xq = _quantized_reference(x, input_quantization=True)
    inter = profile.logical_intermediate
    channel = torch.arange(inter, device=x.device)
    output_cols = torch.arange(profile.hidden, device=x.device)
    output = torch.zeros_like(x, dtype=torch.float32)
    # The sparse weights above select one input per matrix row. Both matrix
    # products are therefore exact FP32 multiplies, with no reduction ambiguity.
    levels = (1.0, 1.5, 2.0)
    for expert in range(profile.experts):
        token, slot = torch.where(ids == expert)
        if token.numel() == 0:
            continue
        gate = xq[token[:, None], (channel * 7) % profile.hidden] * (
            levels[expert % 3] / 16
        )
        up = xq[token[:, None], (channel * 7 + 3) % profile.hidden] * (
            levels[(expert + 1) % 3] / 16
        )
        if profile.has_bias:
            gate += 0.03125 * (expert % 3 + 1)
            up -= 0.03125
            gate = gate.clamp(max=7)
            hidden = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)
        elif profile.activation == "situ":
            hidden = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
                25.0 * torch.tanh(up / 25.0)
            )
        else:
            hidden = gate * torch.sigmoid(gate) * up
        hidden = _quantized_reference(hidden, input_quantization=False)
        result = hidden[:, (output_cols * 11 + expert) % inter] * (
            levels[expert % 3] / 16
        )
        if profile.has_bias:
            result += 0.015625 * (expert % 5 - 2)
        # The kernel rounds each weighted route to BF16 before FP32 combine.
        routed = (result * weights[token, slot, None]).to(torch.bfloat16).float()
        output.index_add_(0, token, routed)
    return output.to(torch.bfloat16)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8 or not is_cdna4(),
    reason="requires torchrun with eight gfx950 GPUs",
)
@pytest.mark.parametrize("profile_name", ("gpt_oss_120b", "dsv4", "kimi_k3"))
def test_distributed_petit_reference(profile_name):
    expected_root = os.environ.get("PETIT_EXPECTED_RUNTIME_ROOT")
    if expected_root is not None:
        assert (
            Path(tokenspeed_kernel.__file__)
            .resolve()
            .is_relative_to(Path(expected_root).resolve())
        ), "pytest imported a different runtime checkout"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    profile = _PROFILES[profile_name]
    layer = _TestMoeLayer(profile, device)
    _weights(layer, profile, rank)
    plan = _make_plan(profile)
    tokenspeed_kernel.moe_process_weights(plan, layer)
    output_dir = os.environ.get("PETIT_OUTPUT_DIR")
    # Exercise empty source ranks, padding/tile boundaries and capacity, then
    # refresh both hidden states and destinations within the same captured graph.
    for case, counts in (
        ("balanced", [16] * 8),
        ("uneven", [0, 1, 15, 16, 31, 32, 63, 64]),
        ("partial_scales", [256] * 8),
        ("capacity", [1024] + [1] * 7),
    ):
        count = counts[rank]
        x = torch.empty((count, profile.hidden), device=device, dtype=torch.bfloat16)
        ids, weights = _routing(count, rank, profile, device)

        def refresh(step):
            values = torch.arange(x.numel(), device=device).reshape(x.shape)
            input_scale = 4.0 if profile.activation == "situ" else 1.0 / 32
            x.copy_(
                (((values * 13 + rank * 7 + step * 17) % 127) - 63).float()
                * input_scale
            )
            new_ids, new_weights = _routing(count, rank + step, profile, device)
            if step:
                # Concentrate routes on one destination without duplicate experts.
                new_ids = (new_ids % (profile.experts // 8)) + (step % 8) * (
                    profile.experts // 8
                )
            ids.copy_(new_ids)
            weights.copy_(new_weights)

        def run():
            return tokenspeed_kernel.moe_apply(
                plan,
                x,
                layer,
                x.new_empty((count, 0)),
                topk_weights=weights,
                topk_ids=ids,
                num_tokens_global=sum(counts),
                max_num_tokens_per_gpu=max(counts),
                do_finalize=True,
                low_latency=None,
                overlap_fn=None,
                shared_input=None,
                shared_weight=None,
                shared_out=None,
            )

        refresh(0)
        eager = run().clone()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for step in (0, 1, 2):
            refresh(step)
            graph.replay()
            actual = captured.clone()
            expected = _reference(x, ids, weights, profile)
            # Sparse inputs keep the quantization-aware reference tight; BF16
            # route rounding permits one final BF16 rounding difference.
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=8e-3)
            torch.testing.assert_close(
                actual, eager if step == 0 else run(), atol=0, rtol=0
            )
            assert torch.isfinite(actual).all()
            if output_dir:
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(
                    actual.cpu(),
                    directory / f"{profile_name}-{case}-{step}-rank{rank}.pt",
                )
            dist.barrier()
