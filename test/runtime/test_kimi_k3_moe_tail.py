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

"""Which Kimi-K3 deployments bind the fused MoE layer tail, and which forwards
run it.

CPU-only: the tail's numerics are covered by the kernel package's 8-GPU test.
Every rank must decide alike, or the ranks' in-kernel exchange waits forever.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models import kimi_k3
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=5, suite="runtime-1gpu")

# Kimi-K3.1's sizes, as the tail's shape takes them.
_CONFIG = SimpleNamespace(
    hidden_size=7168,
    num_attention_heads=96,
    v_head_dim=128,
    num_experts=896,
    num_experts_per_token=16,
    routed_expert_hidden_size=3584,
    moe_intermediate_size=3072,
    num_shared_experts=2,
    num_expert_group=1,
    topk_group=1,
    moe_router_activation_func="sigmoid",
    routed_scaling_factor=1.0,
    moe_renormalize=True,
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
)


def _mapping(dp=1, pp=1, ep=1, shared_group=True):
    group = object()
    return SimpleNamespace(
        pp_size=pp,
        attn=SimpleNamespace(dp_size=dp, tp_size=8, tp_group=group),
        moe=SimpleNamespace(
            ep_size=ep, tp_ep_group=group if shared_group else object()
        ),
    )


@pytest.mark.parametrize(
    "mapping,role,deterministic,rows",
    [
        ({}, "null", False, 8),
        ({"dp": 2}, "null", False, None),
        ({"pp": 2}, "null", False, None),
        ({"ep": 8}, "null", False, None),
        ({"shared_group": False}, "null", False, None),
        ({}, "prefill", False, None),
        ({}, "null", True, None),
    ],
    ids=[
        "tp8",
        "attn-dp",
        "pp",
        "ep",
        "split-groups",
        "prefill-role",
        "batch-invariant",
    ],
)
def test_one_tp_group_for_attention_and_experts_binds_the_verify_width(
    monkeypatch, mapping, role, deterministic, rows
):
    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", role)
    monkeypatch.setitem(
        global_server_args_dict, "batch_invariant_collectives", deterministic
    )
    monkeypatch.setitem(global_server_args_dict, "speculative_algorithm", "DSPARK")
    monkeypatch.setitem(global_server_args_dict, "speculative_num_draft_tokens", 8)
    shape = kimi_k3._moe_tail_shape(_CONFIG, _mapping(**mapping), "mxfp4")
    assert (shape and shape.rows) == rows


@pytest.mark.parametrize(
    "mode,rows,serves",
    [
        (ForwardMode.DECODE, 8, True),
        (ForwardMode.DECODE, 16, False),
        (ForwardMode.EXTEND, 8, False),
    ],
    ids=["verify", "two-requests", "prefill"],
)
def test_only_decode_steps_of_the_tail_rows_run_it(mode, rows, serves):
    tail = SimpleNamespace(shape=SimpleNamespace(rows=8))
    binding = kimi_k3._MoeTailBinding(
        layer=SimpleNamespace(tail=tail), projection_weight=None, next_layer=None
    )
    ctx = SimpleNamespace(forward_mode=mode)
    assert binding.serves(rows, ctx) is serves


def _layer(*, attention=torch.bfloat16, gated=True, dense=torch.bfloat16):
    # Only the dtypes and the split/gate state matter here; the weights' sizes
    # do not, and real layers need checkpoint weights and process groups.
    def linear(dtype, **extra):
        return SimpleNamespace(weight=torch.empty(0, dtype=dtype), **extra)

    def decode_projection_parts():
        # An MLA layer without the output gate may lack Q-LoRA, and then has
        # no fused projection to read.
        if not gated:
            raise AttributeError("no fused projection")
        return (torch.empty(0, dtype=attention),)

    attn = SimpleNamespace(
        o_proj=linear(attention),
        decode_projection_parts=decode_projection_parts,
        qkv_parallel=None,
        output_parallel=None,
        use_output_gate=gated,
    )
    moe = SimpleNamespace(
        gate=linear(dense),
        routed_expert_down_proj=linear(dense, narrowed=False),
        routed_expert_up_proj=linear(dense),
        routed_expert_norm=object(),
        shared_experts=SimpleNamespace(
            gate_up_proj=linear(dense), down_proj=linear(dense)
        ),
    )
    return SimpleNamespace(self_attn=attn, block_sparse_moe=moe)


@pytest.mark.parametrize(
    "change,reads",
    [
        ({}, True),
        ({"attention": torch.float8_e4m3fn}, False),
        ({"gated": False}, False),
        ({"dense": torch.float8_e4m3fn}, False),
    ],
    ids=["bf16", "fp8-attention", "ungated", "fp8-router-latent-shared"],
)
def test_layers_the_tail_cannot_read_keep_the_regular_path(change, reads):
    # A checkpoint whose layers the tail cannot read must start on the regular
    # path rather than fail while binding.
    assert kimi_k3._moe_tail_reads_layer(_layer(**change)) is reads
