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

"""MXFP4 MegaMoE numerics, uneven/empty EP owners and CUDA graph replay.

Run with pytest for EP1, or torchrun --standalone --nproc-per-node=2 -m pytest.
"""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe import (
    deep_gemm,
    moe_apply,
    moe_plan,
    moe_process_weights,
    moe_warmup,
)
from torch import nn

pytestmark = pytest.mark.skipif(
    not deep_gemm._HAS_DEEP_GEMM, reason="Blackwell and tokenspeed-deepgemm required"
)


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
    else:
        path = tmp_path_factory.mktemp("deepgemm") / "rendezvous"
        dist.init_process_group(
            "nccl", init_method=f"file://{path}", rank=0, world_size=1
        )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield dist.group.WORLD
    torch.backends.cuda.matmul.allow_tf32 = previous
    dist.destroy_process_group()


def _dequant(packed, scales):
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    sf = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return levels[codes] * sf


def _fp8(x):
    if not x.numel():
        return x
    groups = x.reshape(x.shape[0], -1, 32)
    scale = torch.exp2(
        torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    return (
        (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
        * scale.unsqueeze(-1)
    ).reshape_as(x)


class Experts(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.tp_size = 1
        self.num_experts = 8
        self.num_local_experts = 8 // group.size()
        self.hidden_size, self.intermediate_size = 512, 256
        self.top_k = 2
        self.swiglu_arg = None
        self._moe_backend_state = None
        torch.manual_seed(731)
        self.reference_weights = (
            torch.randint(0, 256, (8, 512, 256), dtype=torch.uint8, device="cuda"),
            torch.randint(120, 123, (8, 512, 16), dtype=torch.uint8, device="cuda"),
            torch.randint(0, 256, (8, 512, 128), dtype=torch.uint8, device="cuda"),
            torch.randint(120, 123, (8, 512, 8), dtype=torch.uint8, device="cuda"),
        )
        start = group.rank() * self.num_local_experts
        for name, value in zip(
            ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
            self.reference_weights,
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    value[start : start + self.num_local_experts].clone(),
                    requires_grad=False,
                ),
            )

    def reference(self, x, ids, scores):
        w13, sf13, w2, sf2 = self.reference_weights
        result = torch.zeros_like(x, dtype=torch.float32)
        xq = _fp8(x.float())
        for expert in range(8):
            rows, slots = torch.where(ids == expert)
            if not rows.numel():
                continue
            gate, up = (
                (xq[rows] @ _dequant(w13[expert], sf13[expert]).T)
                .to(torch.bfloat16)
                .float()
                .chunk(2, -1)
            )
            # DeepGEMM rounds FC1 to BF16 and applies routing weights before
            # quantizing the intermediate activation for FC2.
            inter = _fp8(F.silu(gate) * up * scores[rows, slots, None])
            value = inter @ _dequant(w2[expert], sf2[expert]).T
            result.index_add_(0, rows, value)
        return result


def test_uneven_empty_and_graph_replay(group):
    experts = Experts(group)
    plan = moe_plan(
        "mxfp4",
        activation="silu",
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=group.size(),
        ispp=256,
        hidden=512,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        num_experts=8,
        top_k=2,
        process_group=group,
        persistent_max_num_tokens_per_gpu=64,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="mega_moe",
    )
    moe_process_weights(plan, experts)
    assert experts._moe_backend_state.num_shared_experts == 0
    with pytest.raises(ValueError, match=r"warmup max_num_tokens must be in \[1, 64\]"):
        moe_warmup(plan, experts, max_num_tokens=65)
    moe_warmup(plan, experts, max_num_tokens=32)
    for rows in (7, 0, 1, 31, 3):
        n = rows if group.rank() == 0 else rows + 3
        x = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16) * 0.25
        ids = torch.stack(
            (
                torch.arange(n, device="cuda") % 8,
                (torch.arange(n, device="cuda") + 3) % 8,
            ),
            -1,
        ).int()
        scores = torch.tensor([0.3, 0.7], device="cuda").expand(n, 2).contiguous()
        expected = experts.reference(x, ids, scores)
        run = partial(
            moe_apply,
            plan,
            x,
            experts,
            None,
            scores,
            ids,
            max_num_tokens_per_gpu=rows + 3,
            do_finalize=True,
        )
        actual = run()
        torch.cuda.synchronize()
        assert actual.shape == x.shape
        if n:
            torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
            assert (actual.float() - expected).norm() / expected.norm() < 0.01
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
