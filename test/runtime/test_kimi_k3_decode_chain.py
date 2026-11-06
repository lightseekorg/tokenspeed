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

"""Which K3 forwards take the TP8 decode chain, how it runs the layers, and which
models bind it.

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

import tokenspeed.runtime.layers.moe.latent as latent
import tokenspeed.runtime.models.kimi_k3 as kimi_k3
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.moe.utils import MoeBackend
from tokenspeed.runtime.models.kimi_k3 import (
    KimiLinearForCausalLM,
    KimiLinearModel,
    KimiLinearMoE,
)

register_cuda_ci(est_time=5, suite="runtime-1gpu")

HIDDEN = 16
# The model width the chain's kernels serve (``k3_decode_support.HIDDEN``).
HIDDEN_SIZE = 7168


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
    "blackwell,backend,overrides,allowed",
    [
        (True, MoeBackend.FLASHINFER_TRTLLM, {}, True),
        (True, MoeBackend.AUTO, {}, True),
        (False, MoeBackend.FLASHINFER_TRTLLM, {}, False),
        (True, MoeBackend.MARLIN, {}, False),
        (True, MoeBackend.FLASHINFER_TRTLLM, {"dp": 2}, False),
        (True, MoeBackend.FLASHINFER_TRTLLM, {"pp": 2}, False),
        (True, MoeBackend.FLASHINFER_TRTLLM, {"moe_tp": 4}, False),
        (True, MoeBackend.FLASHINFER_TRTLLM, {"ep": 2}, False),
        (True, MoeBackend.FLASHINFER_TRTLLM, {"shared_group": False}, False),
    ],
    ids=[
        "trtllm",
        "auto",
        "hopper",
        "marlin",
        "attn-dp",
        "pp",
        "moe-tp-differs",
        "ep",
        "split-groups",
    ],
)
def test_the_plan_allows_the_chain_for_one_shared_tp_group_on_blackwell_trtllm(
    monkeypatch, blackwell, backend, overrides, allowed
):
    monkeypatch.setattr(
        latent, "current_platform", lambda: SimpleNamespace(is_blackwell=blackwell)
    )
    monkeypatch.setattr(latent, "native_latent_moe_available", lambda: False)
    monkeypatch.setattr(latent, "_marlin_moe_available", lambda: False)
    plan = latent.Kimi3MoEExecutionPlan.build(_mapping(**overrides), backend, None)
    assert plan.decode_chain is allowed


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
    monkeypatch.setitem(args, "request_state_seqs", 16)
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


class _Attention:
    """Records the attention chain's arguments, its layers and its closing AttnRes."""

    built: list[_Attention] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs, self.layers, self.closing = kwargs, [], None
        _Attention.built.append(self)

    def layer(self, decoder, **kwargs):
        self.layers.append((decoder, kwargs))
        return SimpleNamespace(bound=kwargs)

    def final(self, **kwargs):
        self.closing = kwargs
        return "closing"


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
    block.execution_plan = SimpleNamespace(use_trtllm=True, decode_chain=True)
    block.mapping, block.routed_hidden = _mapping(), 3584
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
        weight=_Weight(f"{tag}.router"), e_score_correction_bias=f"{tag}.bias"
    )
    block.routed_scaling_factor = 2.5
    block.decode_moe = None
    layer.block_sparse_moe = block
    return layer


class _Weight(str):
    """A named stand-in weight that reports the BF16 dtype the binding checks."""

    dtype = torch.bfloat16


def _config(**overrides):
    config = SimpleNamespace(
        hidden_size=7168,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        moe_renormalize=True,
        num_experts=896,
        num_experts_per_token=16,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _bindable(
    monkeypatch, layers, *, multicast=True, pdl=True, attention=False, **arg_overrides
):
    _Owner.built.clear()
    _Attention.built.clear()
    monkeypatch.setattr(kimi_k3, "pdl_enabled", lambda: pdl)
    monkeypatch.setattr(kimi_k3, "multicast_backend_available", lambda g: multicast)
    # The real row counts and envelope (``k3_decode_support``); the kernel module,
    # which needs cutlass, is replaced by one that records construction.
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.moe._cute_dsl.k3_decode",
        SimpleNamespace(KimiK3DecodeMoE=_Owner, KimiK3DecodeAttention=_Attention),
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
    monkeypatch.setitem(args, "request_state_seqs", 64)
    monkeypatch.setitem(args, "max_cudagraph_capture_size", 64)
    monkeypatch.setitem(args, "disable_cuda_graph_padding", False)
    monkeypatch.setitem(args, "mapping", _mapping())
    monkeypatch.setitem(args, "disaggregation_mode", "null")
    monkeypatch.setitem(args, "force_deterministic_rsag", False)
    monkeypatch.setitem(args, "batch_invariant_collectives", False)
    for name, value in arg_overrides.items():
        monkeypatch.setitem(args, name, value)
    return SimpleNamespace(
        mapping=_mapping(),
        model=SimpleNamespace(
            layers=layers,
            output_attn_res_norm="final.res_norm",
            output_attn_res_proj="final.res_proj",
            norm="final.norm",
            decode_final=None,
            decode_batches=(),
        ),
        # Only the attention-binding tests give every layer its attention.
        config=_config(
            num_hidden_layers=len(layers) if attention else 93,
            attn_res_block_size=2,
        ),
    )


def test_binding_hands_each_moe_layer_its_own_weights(monkeypatch):
    layers = [_decoder(0, moe=False), _decoder(1, moe=True), _decoder(3, moe=True)]
    model = _bindable(monkeypatch, layers)

    assert KimiLinearForCausalLM._prepare_decode_chain(model) is True

    (owner,) = _Owner.built
    assert owner.kwargs == dict(
        group=("device", model.mapping.moe.tp_ep_group),
        batches=(1, 8),
        verify_width=1,
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
    assert KimiLinearForCausalLM._prepare_decode_chain(model) is False
    assert not _Owner.built


def test_a_group_without_multicast_is_not_bound(monkeypatch):
    layers = [_decoder(i, moe=True) for i in range(2)]
    model = _bindable(monkeypatch, layers, multicast=False)
    assert KimiLinearForCausalLM._prepare_decode_chain(model) is False
    assert not _Owner.built


@pytest.mark.parametrize(
    "bind_overrides,config_overrides",
    [
        ({"disaggregation_mode": "prefill"}, {}),
        ({"pdl": False}, {}),
        ({"force_deterministic_rsag": True}, {}),
        ({"batch_invariant_collectives": True}, {}),
        ({}, {"activation_situ_linear_beta": None}),
        ({}, {"num_experts": 384}),
        ({}, {"num_experts_per_token": 8}),
        ({}, {"num_expert_group": 8}),
        ({}, {"moe_router_activation_func": "softmax"}),
    ],
    ids=[
        "prefill-role",
        "pdl-off",
        "deterministic-rsag",
        "batch-invariant",
        "no-linear-beta",
        "384-experts",
        "top-8",
        "grouped-routing",
        "softmax-router",
    ],
)
def test_modes_and_routing_the_kernels_do_not_serve_are_not_bound(
    monkeypatch, bind_overrides, config_overrides
):
    layers = [_decoder(i, moe=True) for i in range(2)]
    model = _bindable(monkeypatch, layers, **bind_overrides)
    for name, value in config_overrides.items():
        setattr(model.config, name, value)
    assert KimiLinearForCausalLM._prepare_decode_chain(model) is False
    assert not _Owner.built


def test_a_non_bf16_router_is_not_bound(monkeypatch):
    layers = [_decoder(i, moe=True) for i in range(2)]
    layers[1].block_sparse_moe.gate.weight = SimpleNamespace(dtype=torch.float32)
    model = _bindable(monkeypatch, layers)
    assert KimiLinearForCausalLM._prepare_decode_chain(model) is False
    assert not _Owner.built


def test_speculative_decoding_hands_the_chain_its_verify_width(monkeypatch):
    layers = [_decoder(i, moe=True) for i in range(2)]
    model = _bindable(
        monkeypatch,
        layers,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=8,
    )
    assert KimiLinearForCausalLM._prepare_decode_chain(model) is True
    (owner,) = _Owner.built
    # Verify graphs of 1 and 8 requests run 8 and 64 rows.
    assert owner.kwargs["verify_width"] == 8
    assert owner.kwargs["batches"] == (8, 64)


def test_a_layer_whose_plan_excludes_the_chain_keeps_every_layer_unbound(
    monkeypatch,
):
    layers = [_decoder(i, moe=True) for i in range(2)]
    layers[1].block_sparse_moe.execution_plan = SimpleNamespace(
        use_trtllm=True, decode_chain=False
    )
    model = _bindable(monkeypatch, layers)
    assert KimiLinearForCausalLM._prepare_decode_chain(model) is False
    assert not _Owner.built


class _NoConfig:
    """A model config that marks the ordinary forward by failing on first use."""

    def __getattr__(self, name):
        raise _FellThrough(name)


def _model(**overrides):
    model = SimpleNamespace(
        embed_tokens=lambda ids: torch.zeros(ids.shape[0], HIDDEN),
        decode_final=object(),
        decode_batches=(1, 8),
        layers_to_capture=[],
        eagle3_layers_to_capture=(),
        config=_NoConfig(),
        _forward_decode_chain=mock.Mock(return_value=("chain", "aux")),
    )
    for name, value in overrides.items():
        setattr(model, name, value)
    return model


def _model_forward(model, mode=ForwardMode.DECODE, rows=8, **ctx_fields):
    ctx = SimpleNamespace(forward_mode=mode, dspark_context_producer=None)
    for name, value in ctx_fields.items():
        setattr(ctx, name, value)
    return KimiLinearModel.forward(
        model, torch.zeros(rows, dtype=torch.long), torch.zeros(rows), ctx
    )


@pytest.mark.parametrize(
    "overrides,ctx_fields",
    [
        ({}, {}),
        ({"layers_to_capture": [3]}, {}),
        ({"eagle3_layers_to_capture": (3,)}, {}),
        ({}, {"dspark_context_producer": object()}),
    ],
    ids=["plain", "dflash", "eagle3", "dspark"],
)
def test_decode_rows_of_a_compiled_batch_run_the_layer_chain(overrides, ctx_fields):
    model = _model(**overrides)
    assert _model_forward(model, **ctx_fields) == ("chain", "aux")


@pytest.mark.parametrize(
    "overrides,mode,rows",
    [
        ({}, ForwardMode.EXTEND, 8),
        ({}, ForwardMode.DECODE, 4),
        ({"decode_final": None}, ForwardMode.DECODE, 8),
    ],
    ids=["extend", "uncompiled-rows", "unbound"],
)
def test_other_forwards_keep_the_ordinary_layers(overrides, mode, rows):
    model = _model(**overrides)
    with pytest.raises(_FellThrough):
        _model_forward(model, mode, rows)
    model._forward_decode_chain.assert_not_called()


class _Chain:
    """Records one layer's chain calls into a shared event log; its entry leaves
    ``index`` in the prefix (the previous layer's output) and ``100 + index`` in
    the first snapshot."""

    def __init__(self, name, log, index):
        self.name, self.log, self.index = name, log, index

    def project(self, x):
        return x

    def enter(self, delta, prefix, blocks, out, *, ring):
        self.log.append((self.name, "enter", "ring" if ring else delta))
        prefix.fill_(self.index)
        blocks[:, 0].fill_(100 + self.index)
        return out

    def leave(self, local, prefix, blocks, out):
        self.log.append((self.name, "leave", local))
        return f"{self.name}.x"


def _chain_model(log, **overrides):
    """Three layers (dense, MoE, MoE) whose chain calls go to ``log``."""

    def layer(name, moe, index):
        attention = SimpleNamespace(
            o_proj_input=lambda positions, h, ctx, comm, *, projection: f"{name}.attn"
        )
        decoder = SimpleNamespace(
            decode_attention=_Chain(name, log, index),
            self_attn=attention,
            comm_manager=None,
            is_moe_layer=moe,
        )
        if moe:
            decoder.block_sparse_moe = SimpleNamespace(
                decode_moe=SimpleNamespace(
                    publish=lambda x: log.append((name, "publish", x))
                )
            )
        else:
            decoder.mlp = lambda x: f"{name}.mlp"
        return decoder

    def closing(prefix, out, event):
        log.append(event)
        prefix.fill_(99)
        return out

    final = SimpleNamespace(
        run_ring=lambda prefix, blocks, out: closing(prefix, out, ("final", "ring")),
        run=lambda delta, prefix, blocks, out: closing(prefix, out, ("final", delta)),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=3, attn_res_block_size=2),
        layers=[
            layer("dense", False, 0),
            layer("moe0", True, 1),
            layer("moe1", True, 2),
        ],
        pp_start_layer=0,
        pp_end_layer=3,
        decode_final=final,
        layers_to_capture=[],
        eagle3_layers_to_capture=(),
    )
    for name, value in overrides.items():
        setattr(model, name, value)
    return model


def test_layer_chain_feeds_each_layer_from_the_previous_feed_forward():
    log = []
    model = _chain_model(log)
    hidden = torch.zeros(2, HIDDEN)
    ctx = SimpleNamespace(dspark_context_producer=None)
    out, aux = KimiLinearModel._forward_decode_chain(model, None, hidden, ctx)
    assert out.shape == hidden.shape and aux is None
    assert log[0][:2] == ("dense", "enter") and log[0][2] is hidden
    assert log[1:] == [
        ("dense", "leave", "dense.attn"),
        ("moe0", "enter", "dense.mlp"),
        ("moe0", "leave", "moe0.attn"),
        ("moe0", "publish", "moe0.x"),
        ("moe1", "enter", "ring"),
        ("moe1", "leave", "moe1.attn"),
        ("moe1", "publish", "moe1.x"),
        ("final", "ring"),
    ]


def _attention(
    layer,
    *,
    mla: bool,
    fp8_o_proj: bool = False,
    o_proj_width: int = 1536,
    projection_dtype: torch.dtype = torch.bfloat16,
):
    cls = kimi_k3.KimiLinearMLAAttention if mla else kimi_k3.KimiLinearKDA
    attn = object.__new__(cls)
    torch.nn.Module.__init__(attn)
    attn.qkv_parallel = attn.output_parallel = None
    attn.o_proj = SimpleNamespace(
        supports_fp8_group128_input=fp8_o_proj,
        weight=SimpleNamespace(dtype=torch.bfloat16, shape=(HIDDEN_SIZE, o_proj_width)),
    )
    rows = 3648 if mla else 6288
    projection = SimpleNamespace(
        weight=SimpleNamespace(dtype=projection_dtype, shape=(rows, HIDDEN_SIZE))
    )
    if mla:
        attn.use_output_gate = True
        attn.fused_qkv_a_proj_with_mqa = projection
    else:
        attn.qkvgb_proj = projection
    layer.self_attn = attn
    return layer


def _attention_layers(*, odd_layer=None, **odd):
    """Three decoder layers (dense KDA, MoE KDA, MoE MLA); ``odd`` overrides the
    attention of layer ``odd_layer``."""
    layers = []
    for i in range(3):
        layer = _decoder(i, moe=i > 0)
        layers.append(_attention(layer, mla=i == 2, **(odd if i == odd_layer else {})))
    return layers


def test_binding_gives_every_layer_its_entry_and_exit(monkeypatch):
    layers = _attention_layers()
    model = _bindable(monkeypatch, layers, attention=True)

    assert KimiLinearForCausalLM._prepare_decode_chain(model) is True

    (attention,) = _Attention.built
    assert attention.kwargs == dict(
        group=("device", model.mapping.attn.tp_group),
        batches=(1, 8),
        moe=_Owner.built[0],
    )
    for i, (decoder, kwargs) in enumerate(attention.layers):
        mla = i == 2
        projection = (
            decoder.self_attn.fused_qkv_a_proj_with_mqa
            if mla
            else decoder.self_attn.qkvgb_proj
        )
        assert decoder is layers[i]
        assert kwargs == dict(
            layer_idx=i, mla=mla, first=i == 0, projection_weight=projection.weight
        )
        assert decoder.decode_attention.bound == kwargs
    assert attention.closing == dict(
        res_norm="final.res_norm",
        res_proj="final.res_proj",
        out_norm="final.norm",
        num_blocks=2,
    )
    assert model.model.decode_final == "closing"
    assert model.model.decode_batches == (1, 8)


@pytest.mark.parametrize(
    "odd_layer,odd",
    [
        (1, {"fp8_o_proj": True}),
        (2, {"o_proj_width": 1024}),
        (0, {"projection_dtype": torch.float32}),
    ],
    ids=["fp8-o-proj-input", "o-proj-width", "fp32-projection"],
)
def test_an_unsupported_attention_keeps_only_the_moe_chain(monkeypatch, odd_layer, odd):
    layers = _attention_layers(odd_layer=odd_layer, **odd)
    model = _bindable(monkeypatch, layers, attention=True)

    assert KimiLinearForCausalLM._prepare_decode_chain(model) is True

    assert _Owner.built and not _Attention.built
    assert model.model.decode_final is None
    assert all(layer.block_sparse_moe.decode_moe for layer in layers[1:])


class _Producer:
    """A DSpark context producer that keeps what the target hands it."""

    def __init__(self):
        self.captures, self.written = {}, None

    def begin_stage(self, hidden, inbound):
        return "projected"

    def add_capture(self, projected, index, captured):
        self.captures[index] = (projected, captured.clone())

    def write_context(self, projected, positions, locations):
        self.written = (projected, positions, locations)


def _capturing_model(log, stream):
    model = _chain_model(log, layers_to_capture=[0, 2])
    model.dflash_aux_stream = stream
    model._dflash_capture_idx_map = {0: 0, 2: 1}
    for name in ("_capture_tap", "_dspark_capture_stream", "_write_dspark_context"):
        setattr(model, name, getattr(KimiLinearModel, name).__get__(model))
    return model


def _dspark_ctx(rows):
    return SimpleNamespace(
        dspark_context_producer=_Producer(),
        target_capture_sink=None,
        num_extends=0,
        bs=rows,
        input_num_tokens=rows,
        attn_backend=SimpleNamespace(
            decode_window_locations=lambda: torch.arange(rows + 5)
        ),
    )


def test_layer_chain_captures_each_output_once_the_next_entry_formed_it():
    """A layer's output is the prefix only after the next entry (or the closing
    AttnRes) added its feed-forward row, so the prefix stream is captured there."""
    log = []
    model = _capturing_model(log, "prefix")
    ctx = _dspark_ctx(3)
    _, aux = KimiLinearModel._forward_decode_chain(
        model, "positions", torch.zeros(3, HIDDEN), ctx
    )
    producer = ctx.dspark_context_producer
    # Layer 0's output is what layer 1's entry left; the last one, the closing's.
    assert {i: (p, c.unique().tolist()) for i, (p, c) in producer.captures.items()} == {
        0: ("projected", [1.0]),
        1: ("projected", [99.0]),
    }
    projected, positions, locations = producer.written
    assert (projected, positions) == ("projected", "positions")
    assert locations.tolist() == [0, 1, 2]
    assert aux is None


def test_layer_chain_hands_attnres_captures_block_major_snapshots(monkeypatch):
    log, seen = [], []
    model = _capturing_model(log, "attn_res")

    def mix(prefix, blocks, *weights):
        """Record the prefix and first snapshot an AttnRes capture reads."""
        seen.append(
            (prefix.unique().tolist(), blocks[0].unique().tolist(), blocks.shape)
        )
        return prefix.clone()

    model.layers[1].capture_attnres = mix
    model.output_attn_res_proj = model.output_attn_res_norm = None
    monkeypatch.setattr(kimi_k3, "_apply_attn_res", mix)
    ctx = _dspark_ctx(3)
    KimiLinearModel._forward_decode_chain(
        model, "positions", torch.zeros(3, HIDDEN), ctx
    )
    # Layer 1's entry wrote the first snapshot; the closing left it in place.
    assert seen == [([1.0], [101.0], (2, 3, HIDDEN)), ([99.0], [102.0], (2, 3, HIDDEN))]
    assert sorted(ctx.dspark_context_producer.captures) == [0, 1]


def test_layer_chain_collects_eagle3_outputs_after_the_next_entry():
    """Layer 0's output is what layer 1's entry left, layer 2's what the closing
    left; each is a copy the later prefix updates do not change."""
    log = []
    model = _chain_model(log, eagle3_layers_to_capture=(1, 3))
    ctx = SimpleNamespace(dspark_context_producer=None)
    hidden = torch.zeros(2, HIDDEN)
    _, aux = KimiLinearModel._forward_decode_chain(model, None, hidden, ctx)
    assert [a.shape for a in aux] == [hidden.shape] * 2
    assert [a.unique().tolist() for a in aux] == [[1.0], [99.0]]
