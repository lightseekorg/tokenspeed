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

"""CPU-only tests of the model dispatch seam, without accelerator bootstrap."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/tokenspeed/runtime/models/flash_local_attention.py"
)


@pytest.fixture
def dispatch():
    # Execute the actual methods while excluding unrelated platform imports.
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SeparateProjectionKimiLinearMLAAttention"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_mla_prolog_inputs", "_project_q_with_mla_prolog", "forward")
    ]

    class Base:
        def forward(self, *args):
            return args

    available = Mock(return_value=True)
    namespace = dict(
        torch=torch,
        Any=object,
        KimiLinearMLAAttention=Base,
        mla_prolog_available=available,
        mla_prolog=Mock(),
        sigmoid_mul=lambda x, gate: x * torch.sigmoid(gate),
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    layer = namespace[cls.name]()
    for name, value in dict(
        hidden_size=3072,
        num_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_output_gate=True,
        attention_backend="mla",
        _MLA_KERNEL_BACKENDS=("mla",),
        w_kc=object(),
        component_mapping=SimpleNamespace(tp_size=1),
        attn_mqa=SimpleNamespace(layer_id=3),
    ).items():
        setattr(layer, name, value)
    for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj"):
        setattr(layer, name, SimpleNamespace(_weight_nz_transposed=True))
    device = SimpleNamespace(type="npu")
    hidden = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        ndim=2,
        shape=(32, 3072),
        new_empty=torch.empty,
    )
    cache = Mock(spec=torch.Tensor)
    cache.dtype, cache.device = torch.bfloat16, device
    cache.is_contiguous.return_value = True
    cache.numel.return_value = 64 * 576
    locations = torch.arange(32)
    backend = SimpleNamespace(
        spec_num_tokens=1, write_locations=Mock(return_value=locations)
    )
    ctx = SimpleNamespace(
        num_extends=0,
        input_num_tokens=32,
        bs=32,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(
            quant_method=None,
            get_key_buffer=Mock(return_value=cache),
            arena=SimpleNamespace(kv_page_size=64),
        ),
    )
    return layer, hidden, ctx, cache, locations, available


def test_prolog_uses_backend_owned_write_locations(dispatch):
    layer, hidden, ctx, cache, locations, _ = dispatch
    result = layer._mla_prolog_inputs(hidden, ctx, None, None, None)
    assert result == (cache.view.return_value, locations)
    ctx.attn_backend.write_locations.assert_called_once_with(
        layer.attn_mqa, ctx.forward_mode
    )
    cache.view.assert_called_once_with(-1, 64, 1, 576)


@pytest.mark.parametrize("needs_gather", [False, True])
def test_prolog_preserves_pre_attention_communication(dispatch, needs_gather):
    layer, hidden, ctx, _, _, _ = dispatch
    comm = SimpleNamespace(
        layer_id=3,
        attn_mapping=SimpleNamespace(has_tp=True),
        prev_is_moe=True,
        use_all_reduce=Mock(return_value=not needs_gather),
    )
    result = layer._mla_prolog_inputs(hidden, ctx, comm, None, None)
    assert (result is None) == needs_gather
    if needs_gather:
        ctx.attn_backend.write_locations.assert_not_called()


@pytest.mark.parametrize(
    "exclusion", ["speculative", "missing", "layout", "locations", "prefill"]
)
def test_prolog_retains_fallback_for_exclusions(dispatch, exclusion):
    layer, hidden, ctx, cache, _, available = dispatch
    if exclusion == "speculative":
        ctx.attn_backend.spec_num_tokens = 2
    elif exclusion == "missing":
        available.return_value = False
    elif exclusion == "layout":
        cache.is_contiguous.return_value = False
    elif exclusion == "locations":
        ctx.attn_backend.write_locations.return_value = torch.arange(31)
    else:
        ctx.num_extends = 1
    assert layer._mla_prolog_inputs(hidden, ctx, None, None, None) is None


def test_forward_keeps_current_comm_manager_signature(dispatch):
    layer, hidden, ctx, _, _, available = dispatch
    available.return_value = False
    comm = SimpleNamespace(layer_id=0)
    result = layer.forward(None, hidden, ctx, comm, None, None)
    assert result == (None, hidden, ctx, comm, None, None)


@pytest.mark.parametrize("fuse_value_gate", [False, True])
def test_forward_passes_selected_cache_to_packaged_path(dispatch, fuse_value_gate):
    layer, hidden, ctx, cache, locations, _ = dispatch
    events = []
    query = torch.ones(32, 2)
    gate = torch.zeros(32, layer.num_heads * layer.v_head_dim)
    layer._project_q_with_mla_prolog = Mock(return_value=query)
    ctx.attn_backend.supports_mla_projected_value_decode = fuse_value_gate

    def project_gate(x):
        assert x is hidden
        events.append("gate")
        return (gate,)

    def attention(q, kv, context, slots, output, *, record_kv_cache, output_gate):
        assert q is query and kv is None and context is ctx
        assert slots is locations and record_kv_cache is None
        assert output_gate is (gate if fuse_value_gate else None)
        events.append("attention")
        output.fill_(1 if fuse_value_gate else 2)

    def project_output(x):
        torch.testing.assert_close(x, torch.ones_like(gate))
        events.append("out_projection")
        return ("fused",)

    layer.g_proj = project_gate
    layer.forward_absorb_attn_v_proj = attention
    layer.o_proj = project_output
    assert layer.forward(None, hidden, ctx, None, None, None) == "fused"
    layer._project_q_with_mla_prolog.assert_called_once_with(
        hidden, cache.view.return_value, locations
    )
    assert events == ["gate", "attention", "out_projection"]


def test_projection_helper_only_runs_prolog_and_query_concat(dispatch):
    layer, hidden, _, cache, locations, _ = dispatch
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"):
        getattr(layer, name).weight = object()
    for name in ("q_a_layernorm", "kv_a_layernorm"):
        setattr(layer, name, SimpleNamespace(weight=object(), variance_epsilon=1e-5))
    layer.g_proj = Mock(side_effect=AssertionError("gate is not prolog"))
    layer.forward_absorb_attn_v_proj = Mock(
        side_effect=AssertionError("attention is not prolog")
    )
    layer.o_proj = Mock(side_effect=AssertionError("output projection is not prolog"))
    op = layer._project_q_with_mla_prolog.__func__.__globals__["mla_prolog"]
    nope, tail = torch.ones(32, 2), torch.zeros(32, 1)
    op.return_value = (nope, tail)
    result = layer._project_q_with_mla_prolog(hidden, cache, locations)
    torch.testing.assert_close(result, torch.cat((nope, tail), dim=-1))
    assert op.call_args.args[7] is cache
    assert op.call_args.args[8] is locations
    op.return_value = None
    assert layer._project_q_with_mla_prolog(hidden, cache, locations) is None
    op.side_effect = RuntimeError("operator failed after admission")
    with pytest.raises(RuntimeError, match="after admission"):
        layer._project_q_with_mla_prolog(hidden, cache, locations)


def test_operator_admission_rejection_keeps_primitive_forward(dispatch):
    layer, hidden, ctx, _, _, _ = dispatch
    layer._project_q_with_mla_prolog = Mock(return_value=None)
    result = layer.forward(None, hidden, ctx, None, None, None)
    assert result == (None, hidden, ctx, None, None, None)
