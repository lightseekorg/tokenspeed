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

"""Aux hidden-state capture under a deferred (fused) all-reduce."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# CPU-only test scheduled in runtime-1gpu because it imports the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.base.decoder_layer import (  # noqa: E402
    capture_layer_input,
)
from tokenspeed.runtime.models.deepseek_v3 import (  # noqa: E402
    DeepseekV3DecoderLayer,
    DeepseekV3Model,
)
from tokenspeed.runtime.models.glm5 import GlmMoeDsaDecoderLayer  # noqa: E402
from tokenspeed.runtime.models.longcat_flash import (  # noqa: E402
    _RuntimeLongcatDecoderLayer,
    _RuntimeLongcatModel,
)
from tokenspeed.runtime.models.qwen3_5 import (  # noqa: E402
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5LinearDecoderLayer,
)

TP = 2


class _DeferredReduceLayer:
    """A TP rank whose MLP output leaves the layer un-reduced: the next layer's
    opening add reduces it, as the fused all-reduce + norm does."""

    def __init__(self):
        comm = SimpleNamespace(
            gather_residual=lambda rows, ctx: rows,
            final_norm=lambda hidden, residual, ctx, norm: (hidden, None),
        )
        self.comm_manager = comm
        self.branch_comm = [comm, comm]
        self.final_norm_comm = comm
        self.input_stream: torch.Tensor | None = None

    def __call__(self, *args, aux_hidden_states=None, **kwargs):
        # Qwen3.5 passes keywords; DeepSeek-V3 and LongCat pass (positions, hidden, ctx, residual).
        if args:
            _, hidden_states, _, residual = args
        else:
            hidden_states, residual = kwargs["hidden_states"], kwargs["residual"]
        stream = hidden_states * TP + (0 if residual is None else residual)
        self.input_stream = stream.clone()
        capture_layer_input(self.comm_manager, stream, None, aux_hidden_states)
        partial = stream * 0.5
        # A later norm may update the residual in place; the capture must not see it.
        stream.mul_(0.25)
        return partial, stream


@pytest.mark.parametrize(
    "model_cls", [Qwen3_5ForCausalLM, DeepseekV3Model, _RuntimeLongcatModel]
)
def test_aux_capture_is_the_reduced_layer_input(model_cls):
    layers = [_DeferredReduceLayer() for _ in range(4)]
    model = SimpleNamespace(
        layers=layers,
        layers_to_capture={1, 3},
        _dflash_capture_idx_map={},
        norm=None,
    )
    x = torch.randn(3, 8)
    ctx = SimpleNamespace(
        target_capture_sink=None,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )

    _, aux = model_cls.forward(
        model,
        torch.zeros(3, dtype=torch.long),
        None,
        ctx,
        input_embeds=x,
    )

    assert len(aux) == 2
    for got, layer in zip(aux, (layers[1], layers[3])):
        torch.testing.assert_close(got, layer.input_stream, rtol=0, atol=0)


def test_capture_layer_input_gathers_and_detaches():
    gathered_from = []

    def gather_residual(rows, ctx):
        gathered_from.append(rows)
        return rows

    comm = SimpleNamespace(gather_residual=gather_residual)
    residual = torch.randn(3, 8)
    expected = residual.clone()
    aux: list[torch.Tensor] = []

    capture_layer_input(comm, residual, None, None)
    assert gathered_from == []

    capture_layer_input(comm, residual, None, aux)
    residual.mul_(0.25)

    assert gathered_from == [residual]
    torch.testing.assert_close(aux[0], expected, rtol=0, atol=0)


class _Comm:
    """A communication policy whose opening reduce + add returns a distinct
    residual, and whose later norms update that residual in place."""

    def __init__(self):
        self.mapping = SimpleNamespace(
            has_attn_tp=True, dense=SimpleNamespace(has_tp=False)
        )
        self.layer_boundary_norm = "fused"
        self.reduced: torch.Tensor | None = None
        self.gathered: list[torch.Tensor] = []

    def get_num_tokens(self, ctx):
        return 0, 0

    def input_reduce_norm(self, hidden_states, residual):
        self.reduced = hidden_states * TP + (0 if residual is None else residual)
        return hidden_states * 0.5, self.reduced

    def gather_residual(self, rows, ctx):
        self.gathered.append(rows)
        return rows

    def post_attn_reduce_norm(self, hidden_states, residual, ctx):
        residual.mul_(0.25)
        return hidden_states, residual

    def intra_layer_add_norm(self, hidden_states, residual):
        residual.mul_(0.25)
        return hidden_states, residual

    def pre_attn_comm(self, hidden_states, ctx):
        return hidden_states

    def pre_mlp_comm(self, hidden_states, ctx):
        return hidden_states

    def post_mlp_fused(self, hidden_states, residual, ctx):
        return hidden_states, residual


class _LinearAttn:
    def input_fp8_scale(self):
        return None

    def __call__(self, hidden_states, hidden_fp8, ctx):
        return hidden_states


def _attn(**kwargs):
    return kwargs["hidden_states"] if kwargs else None


def _stub_layer(cls):
    """A real decoder layer class whose submodules are identity stubs."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    comm = _Comm()
    mlp = SimpleNamespace(
        nvfp4_global_scale=lambda: None,
        forward_prequantized=lambda hidden_states, fp4: hidden_states,
    )
    if cls is _RuntimeLongcatDecoderLayer:
        layer.moe_comm = comm
        layer.branch_comm = [comm, comm]
        layer.self_attn = [_attn, _attn]
        layer._forward_moe = lambda hidden_states, residual, ctx, *_: (
            hidden_states,
            None,
        )
        layer._forward_dense_mlp = lambda i, hidden_states, residual, ctx: (
            hidden_states,
            residual,
        )
        layer._forward_idle = lambda positions, hidden_states, residual, ctx, *_: (
            hidden_states,
            residual,
        )
        return layer, comm
    layer.comm_manager = comm
    layer.input_layernorm = None
    layer.post_attention_layernorm = None
    layer.mlp = mlp if cls.__name__.startswith("Qwen3_5") else (lambda h, *_: h)
    layer.is_moe_layer = False
    layer.linear_attn = _LinearAttn()
    layer.qkv_proj = SimpleNamespace(
        quant_method=SimpleNamespace(static_fp8_input_scale=lambda proj: None)
    )
    layer.self_attention = _attn
    layer.self_attn = lambda **kwargs: kwargs["hidden_states"]
    return layer, comm


def _call(layer, hidden_states, residual, ctx, aux):
    if isinstance(layer, (Qwen3_5LinearDecoderLayer, Qwen3_5AttentionDecoderLayer)):
        return layer(
            positions=None,
            hidden_states=hidden_states,
            residual=residual,
            ctx=ctx,
            aux_hidden_states=aux,
        )
    return layer(None, hidden_states, ctx, residual, aux_hidden_states=aux)


LAYER_CLASSES = [
    Qwen3_5LinearDecoderLayer,
    Qwen3_5AttentionDecoderLayer,
    DeepseekV3DecoderLayer,
    GlmMoeDsaDecoderLayer,
    _RuntimeLongcatDecoderLayer,
]


@pytest.mark.parametrize("layer_cls", LAYER_CLASSES)
def test_layer_captures_its_reduced_input(layer_cls):
    layer, comm = _stub_layer(layer_cls)
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False), draft_narrowing=None
    )
    hidden_states, residual = torch.randn(3, 8), torch.randn(3, 8)
    expected = hidden_states * TP + residual
    aux: list[torch.Tensor] = []

    _call(layer, hidden_states, residual, ctx, aux)

    # The tap is the gathered residual right after the reduce + add, detached
    # from the in-place updates that follow it.
    assert len(comm.gathered) == 1 and comm.gathered[0] is comm.reduced
    assert len(aux) == 1
    torch.testing.assert_close(aux[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("layer_cls", LAYER_CLASSES)
def test_layer_skips_capture_when_not_requested(layer_cls):
    layer, comm = _stub_layer(layer_cls)
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False), draft_narrowing=None
    )

    _call(layer, torch.randn(3, 8), torch.randn(3, 8), ctx, None)

    assert comm.gathered == []


@pytest.mark.parametrize("layer_cls", LAYER_CLASSES)
def test_idle_layer_appends_one_empty_gathered_tap(layer_cls):
    layer, comm = _stub_layer(layer_cls)
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: True), draft_narrowing=None
    )
    empty = torch.empty(0, 8)
    aux: list[torch.Tensor] = []

    _call(layer, empty, None, ctx, aux)

    assert len(comm.gathered) == 1 and comm.gathered[0] is empty
    assert len(aux) == 1 and aux[0].shape == (0, 8)
