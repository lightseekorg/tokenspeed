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

"""Which K3 MoE calls take the TP8 decode chain, and which models bind it.

CPU-only: the chain's numerics are covered by the kernel package's TP8 test.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

import tokenspeed.runtime.models.kimi_k3 as kimi_k3
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.kimi_k3 import KimiLinearForCausalLM, KimiLinearMoE

register_cuda_ci(est_time=5, suite="runtime-1gpu")

HIDDEN = 16


class _FellThrough(Exception):
    """The call left the chain for the ordinary MoE."""


def _moe(batches=(1, 8)):
    group = object()
    return SimpleNamespace(
        mapping=SimpleNamespace(
            attn=SimpleNamespace(dp_size=1, tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
        ),
        native_latent_moe=None,
        execution_plan=SimpleNamespace(use_native=False),
        decode_moe=SimpleNamespace(
            owner=SimpleNamespace(batches=batches),
            run=mock.Mock(return_value="chain"),
        ),
        _routing_output_format=mock.Mock(side_effect=_FellThrough),
    )


def _forward(moe, mode, rows, *, sharded=False, ctx=True):
    hidden = torch.zeros(rows, HIDDEN)
    prefix = torch.zeros(rows // 8 if sharded else rows, HIDDEN)
    return KimiLinearMoE.forward(
        moe,
        hidden,
        prefix,
        rows,
        rows,
        SimpleNamespace(forward_mode=mode) if ctx else None,
        prefix_is_sharded=sharded,
    )


def test_decode_rows_of_a_compiled_batch_take_the_chain():
    moe = _moe()
    assert _forward(moe, ForwardMode.DECODE, 8) == "chain"
    hidden, prefix = moe.decode_moe.run.call_args.args
    assert hidden.shape == prefix.shape == (8, HIDDEN)


@pytest.mark.parametrize(
    "mode,rows,sharded,ctx",
    [
        (ForwardMode.EXTEND, 8, False, True),
        (ForwardMode.MIXED, 8, False, True),
        (ForwardMode.DECODE, 4, False, True),
        (ForwardMode.DECODE, 8, True, True),
        (ForwardMode.DECODE, 8, False, False),
    ],
    ids=["extend", "mixed", "uncompiled-batch", "sharded-residual", "no-context"],
)
def test_other_calls_keep_the_ordinary_moe(mode, rows, sharded, ctx):
    moe = _moe()
    with pytest.raises(_FellThrough):
        _forward(moe, mode, rows, sharded=sharded, ctx=ctx)
    moe.decode_moe.run.assert_not_called()


def test_an_unbound_layer_keeps_the_ordinary_moe():
    moe = _moe()
    moe.decode_moe = None
    with pytest.raises(_FellThrough):
        _forward(moe, ForwardMode.DECODE, 8)


def _mapping(**overrides):
    group = object()
    sizes = dict(dp=1, pp=1, attn_tp=8, moe_tp=8, ep=1, shared_group=True)
    sizes.update(overrides)
    return SimpleNamespace(
        pp_size=sizes["pp"],
        attn=SimpleNamespace(
            dp_size=sizes["dp"], tp_size=sizes["attn_tp"], tp_group=group
        ),
        moe=SimpleNamespace(
            tp_size=sizes["moe_tp"],
            ep_size=sizes["ep"],
            tp_ep_group=group if sizes["shared_group"] else object(),
        ),
    )


@pytest.mark.parametrize(
    "blackwell,overrides",
    [
        (False, {}),
        (True, {"dp": 2}),
        (True, {"pp": 2}),
        (True, {"attn_tp": 4}),
        (True, {"moe_tp": 4}),
        (True, {"ep": 2}),
        (True, {"shared_group": False}),
    ],
    ids=["hopper", "attn-dp", "pp", "attn-tp4", "moe-tp4", "ep", "split-groups"],
)
def test_only_single_group_tp8_blackwell_binds_the_chain(
    monkeypatch, blackwell, overrides
):
    monkeypatch.setattr(
        kimi_k3,
        "current_platform",
        lambda: SimpleNamespace(is_blackwell=blackwell),
    )
    # Gating precedes any layer access, so a model without layers suffices.
    model = SimpleNamespace(mapping=_mapping(**overrides), model=None)
    assert KimiLinearForCausalLM._prepare_decode_moe(model) is False


@pytest.mark.parametrize(
    "algorithm,width,captured,disable_padding,rows",
    [
        (None, None, [1, 2, 4], False, {1, 2, 4}),
        ("DSPARK", 8, [1, 2, 4], False, {8, 16, 32}),
        ("EAGLE3", 3, [1, 2], False, {3, 6}),
        # No explicit ladder: the executor's default one at 16 sequences (it adds
        # max_num_seqs - 1 and max_num_seqs below its larger default entries).
        (None, None, None, False, {1, 2, 4, 8, 15, 16}),
        ("DSPARK", 8, None, True, {8 * b for b in range(1, 17)}),
    ],
    ids=["decode", "dspark-verify", "eagle3-verify", "default", "default-unpadded"],
)
def test_decode_graph_rows_follow_the_executor_ladder(
    monkeypatch, algorithm, width, captured, disable_padding, rows
):
    args = kimi_k3.global_server_args_dict
    monkeypatch.setitem(args, "speculative_algorithm", algorithm)
    monkeypatch.setitem(args, "speculative_num_draft_tokens", width)
    monkeypatch.setitem(args, "cudagraph_capture_sizes", captured)
    monkeypatch.setitem(args, "max_num_seqs", 16)
    monkeypatch.setitem(args, "max_cudagraph_capture_size", 16)
    monkeypatch.setitem(args, "disable_cuda_graph_padding", disable_padding)
    monkeypatch.setitem(args, "mapping", _mapping())
    assert kimi_k3._decode_graph_rows() == rows


class _Owner:
    """Records the chain owner's arguments and each layer it binds."""

    built: list[_Owner] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs, self.layers = kwargs, []
        _Owner.built.append(self)

    def layer(self, **kwargs):
        self.layers.append(kwargs)
        return SimpleNamespace(bound=kwargs)


def _weight(tag: str) -> SimpleNamespace:
    return SimpleNamespace(weight=f"{tag}.weight")


def _decoder(layer_id: int, *, moe: bool):
    layer = object.__new__(kimi_k3.KimiLinearDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_id, layer.is_moe_layer = layer_id, moe
    if not moe:
        return layer
    block = object.__new__(KimiLinearMoE)
    torch.nn.Module.__init__(block)
    tag = f"layer{layer_id}"
    block.execution_plan = SimpleNamespace(use_trtllm=True)
    block.experts = SimpleNamespace(
        plan={"weight_dtype": "mxfp4"},
        w13_weight=f"{tag}.w13",
        w13_weight_scale=f"{tag}.w13_scale",
        w2_weight=f"{tag}.w2",
        w2_weight_scale=f"{tag}.w2_scale",
    )
    block.routed_expert_down_proj = SimpleNamespace(narrowed=True, weight=f"{tag}.down")
    block.routed_expert_up_proj = SimpleNamespace(narrowed=True, weight=f"{tag}.up")
    block.routed_expert_norm = SimpleNamespace(
        weight=f"{tag}.norm", variance_epsilon=1e-5
    )
    block.shared_experts = SimpleNamespace(
        shared_parallel=None,
        gate_up_proj=_weight(f"{tag}.shared_gate_up"),
        down_proj=_weight(f"{tag}.shared_down"),
    )
    block.gate = SimpleNamespace(
        weight=f"{tag}.router", e_score_correction_bias=f"{tag}.bias"
    )
    block.routed_scaling_factor = 2.5
    block.decode_moe = None
    layer.block_sparse_moe = block
    return layer


def _bindable(monkeypatch, layers, *, multicast=True):
    _Owner.built.clear()
    monkeypatch.setattr(
        kimi_k3, "current_platform", lambda: SimpleNamespace(is_blackwell=True)
    )
    monkeypatch.setattr(kimi_k3, "multicast_backend_available", lambda g: multicast)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.moe._cute_dsl.k3_decode",
        SimpleNamespace(BATCHES=(*range(1, 9), 16, 32, 64), KimiK3DecodeMoE=_Owner),
    )
    manager = importlib.import_module(
        "tokenspeed.runtime.distributed.process_group_manager"
    ).process_group_manager
    monkeypatch.setattr(
        manager,
        "get_device_process_group",
        lambda group: ("device", group),
    )
    args = kimi_k3.global_server_args_dict
    monkeypatch.setitem(args, "speculative_algorithm", None)
    monkeypatch.setitem(args, "speculative_num_draft_tokens", None)
    monkeypatch.setitem(args, "cudagraph_capture_sizes", [1, 8, 12])
    monkeypatch.setitem(args, "max_num_seqs", 64)
    monkeypatch.setitem(args, "max_cudagraph_capture_size", 64)
    monkeypatch.setitem(args, "disable_cuda_graph_padding", False)
    monkeypatch.setitem(args, "mapping", _mapping())
    return SimpleNamespace(
        mapping=_mapping(),
        model=SimpleNamespace(layers=layers),
        config=SimpleNamespace(
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            moe_renormalize=True,
        ),
    )


def test_binding_hands_each_moe_layer_its_own_weights(monkeypatch):
    layers = [_decoder(0, moe=False), _decoder(1, moe=True), _decoder(3, moe=True)]
    model = _bindable(monkeypatch, layers)

    assert KimiLinearForCausalLM._prepare_decode_moe(model) is True

    (owner,) = _Owner.built
    assert owner.kwargs == dict(
        group=("device", model.mapping.moe.tp_ep_group),
        batches=(1, 8),
        moe_layers=2,
        beta=4.0,
        linear_beta=25.0,
    )
    for ordinal, layer in enumerate(layers[1:]):
        tag = f"layer{layer.layer_id}"
        assert layer.block_sparse_moe.decode_moe.bound == dict(
            ordinal=ordinal,
            layer_idx=layer.layer_id,
            router=f"{tag}.router",
            correction_bias=f"{tag}.bias",
            routed_scaling_factor=2.5,
            renormalize=True,
            down_shard=f"{tag}.down",
            shared_gate_up=f"{tag}.shared_gate_up.weight",
            shared_down=f"{tag}.shared_down.weight",
            w13=f"{tag}.w13",
            w13_scale=f"{tag}.w13_scale",
            w2=f"{tag}.w2",
            w2_scale=f"{tag}.w2_scale",
            norm_weight=f"{tag}.norm",
            norm_eps=1e-5,
            up_shard=f"{tag}.up",
        )


@pytest.mark.parametrize("moe_layers", [1, 4])
def test_layer_counts_whose_tail_parities_collide_are_not_bound(
    monkeypatch, moe_layers
):
    layers = [_decoder(i, moe=True) for i in range(moe_layers)]
    model = _bindable(monkeypatch, layers)
    assert KimiLinearForCausalLM._prepare_decode_moe(model) is False
    assert not _Owner.built


def test_a_group_without_multicast_is_not_bound(monkeypatch):
    layers = [_decoder(i, moe=True) for i in range(2)]
    model = _bindable(monkeypatch, layers, multicast=False)
    assert KimiLinearForCausalLM._prepare_decode_moe(model) is False
    assert not _Owner.built
