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

from __future__ import annotations

import ast
import importlib
import os
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_deepseek_v4_for_routing_tests() -> ModuleType:
    if torch.cuda.is_available():
        return importlib.import_module("tokenspeed.runtime.models.deepseek_v4")

    source_path = REPO_ROOT / "python/tokenspeed/runtime/models/deepseek_v4.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    selected_names = {
        "dsv4_select_experts",
        "dsv4_linear_fp32",
        "DeepseekV4MoEGate",
        "DeepseekV4MoE",
        "DeepseekV4DecoderLayer",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in selected_names
    ]
    assert {node.name for node in selected} == selected_names
    module = ModuleType("_dsv4_routing_cpu_extract")

    class NoKernelFoundError(RuntimeError):
        pass

    def unavailable_kernel(*args, **kwargs):
        del args, kwargs
        raise NoKernelFoundError

    module.__dict__.update(
        {
            "torch": torch,
            "F": F,
            "nn": nn,
            "NoKernelFoundError": NoKernelFoundError,
            "_kernel_dsv4_select_experts": unavailable_kernel,
            "_kernel_dsv4_linear_fp32": unavailable_kernel,
            "nvtx_range": lambda name: nullcontext(),
            "get_is_capture_mode": lambda: False,
            "pg_manager": SimpleNamespace(get_device_process_group=lambda group: group),
            "mhc_pre": None,
            "mhc_fused_hc": None,
            "slice_to_real_tokens": lambda count, *tensors: tuple(
                tensor[:count] if tensor is not None else None for tensor in tensors
            ),
        }
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(source_path), "exec"), module.__dict__)
    return module


deepseek_v4_model = _load_deepseek_v4_for_routing_tests()

from tokenspeed.runtime.models.deepseek_v4_vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEWLINE,
    IMAGE_PAD,
    IMAGE_START,
    build_dsv4_vision_forward,
    build_image_block,
)

DeepseekV4DecoderLayer = deepseek_v4_model.DeepseekV4DecoderLayer
DeepseekV4MoE = deepseek_v4_model.DeepseekV4MoE
DeepseekV4MoEGate = deepseek_v4_model.DeepseekV4MoEGate
dsv4_select_experts = deepseek_v4_model.dsv4_select_experts


class _ImmediateStreamFork:
    def scope(self, *, enable):
        del enable
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    @staticmethod
    def branch():
        return nullcontext()


TOP_K = 6


def _scores(router_logits: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.nn.functional.softplus(router_logits.float()))


def _oracle(
    router_logits: torch.Tensor,
    *,
    image_mask: torch.Tensor | None,
    bias: torch.Tensor | None,
    bias_vl: torch.Tensor,
    table: torch.Tensor | None,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = _scores(router_logits)
    if table is not None:
        if image_mask is None:
            ids = table[input_ids.long()]
        else:
            ids = torch.empty(
                (scores.shape[0], TOP_K), device=scores.device, dtype=table.dtype
            )
            ids[~image_mask] = table[input_ids[~image_mask].long()]
            ids[image_mask] = torch.topk(
                scores[image_mask] + bias_vl, TOP_K, sorted=True
            ).indices.to(table.dtype)
    else:
        if image_mask is None:
            selection_bias = bias
        else:
            selection_bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
        ids = torch.topk(scores + selection_bias, TOP_K, sorted=True).indices
    weights = scores.gather(1, ids.long())
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), ids.int(), scores


def _router_fixture(
    *,
    device: torch.device | str = "cpu",
    tokens: int = 5,
    experts: int = 16,
):
    logits = (
        torch.arange(tokens * experts, device=device, dtype=torch.float32)
        .reshape(tokens, experts)
        .mul_(0.017)
        .sub_(1.3)
    )
    bias = torch.linspace(-0.4, 0.3, experts, device=device)
    bias_vl = torch.linspace(0.5, -0.2, experts, device=device)
    table = torch.stack(
        [
            (torch.arange(TOP_K, device=device) * 2 + row + 1) % experts
            for row in range(12)
        ]
    ).to(torch.int32)
    return logits, bias, bias_vl, table


def test_image_mask_none_calls_registered_selector_argument_for_argument(monkeypatch):
    logits, bias, bias_vl, table = _router_fixture()
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    expected = (torch.ones(1), torch.ones(1, dtype=torch.int32), torch.ones(1))
    calls = []

    def registered(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(deepseek_v4_model, "_kernel_dsv4_select_experts", registered)
    actual = dsv4_select_experts(
        logits,
        TOP_K,
        True,
        bias,
        table,
        input_ids,
        False,
        image_mask=None,
        bias_vl=bias_vl,
    )

    assert actual is expected
    assert calls == [(logits, TOP_K, True, bias, table, input_ids, False)]


@pytest.mark.parametrize("layer_index", [0, 1, 2])
def test_hash_text_rows_keep_exact_stored_order_in_image_eager_path(
    monkeypatch, layer_index
):
    del layer_index
    logits, _, bias_vl, table = _router_fixture()
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    monkeypatch.setattr(
        deepseek_v4_model,
        "_kernel_dsv4_select_experts",
        lambda *args: (_ for _ in ()).throw(AssertionError("fused path called")),
    )

    weights, ids, scores = dsv4_select_experts(
        logits,
        TOP_K,
        True,
        hash_indices_table=table,
        input_ids=input_ids,
        image_mask=mask,
        bias_vl=bias_vl,
    )
    expected_weights, expected_ids, expected_scores = _oracle(
        logits,
        image_mask=mask,
        bias=None,
        bias_vl=bias_vl,
        table=table,
        input_ids=input_ids,
    )
    assert torch.equal(ids, table[input_ids])
    assert torch.equal(ids, expected_ids)
    assert torch.equal(scores, expected_scores)
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([False, True, False, True, False]),
        torch.ones(5, dtype=torch.bool),
    ],
    ids=["mixed", "all_image"],
)
def test_hash_image_rows_with_oov_ids_never_index_the_table(monkeypatch, mask):
    logits, _, bias_vl, table = _router_fixture()
    table[0].fill_(15)
    input_ids = torch.tensor([2, 50_001, 4, 50_003, 6])
    input_ids[mask] = torch.arange(mask.sum(), dtype=torch.int64) + 50_000
    if bool((~mask).any()):
        input_ids[~mask] = torch.arange((~mask).sum(), dtype=torch.int64) + 1
    monkeypatch.setattr(
        deepseek_v4_model,
        "_kernel_dsv4_select_experts",
        lambda *args: (_ for _ in ()).throw(AssertionError("fused path called")),
    )

    weights, ids, _ = dsv4_select_experts(
        logits,
        TOP_K,
        True,
        hash_indices_table=table,
        input_ids=input_ids,
        image_mask=mask,
        bias_vl=bias_vl,
    )
    expected_weights, expected_ids, _ = _oracle(
        logits,
        image_mask=mask,
        bias=None,
        bias_vl=bias_vl,
        table=table,
        input_ids=input_ids,
    )
    assert torch.equal(ids, expected_ids)
    assert not torch.any(ids[mask] == table[0, 0])
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.parametrize("layer_index", [3, 42])
def test_non_hash_image_routing_matches_bias_oracle(monkeypatch, layer_index):
    del layer_index
    logits, bias, bias_vl, _ = _router_fixture()
    mask = torch.tensor([False, True, False, True, False])
    input_ids = torch.arange(mask.numel())
    monkeypatch.setattr(
        deepseek_v4_model,
        "_kernel_dsv4_select_experts",
        lambda *args: (_ for _ in ()).throw(AssertionError("fused path called")),
    )

    weights, ids, scores = dsv4_select_experts(
        logits,
        TOP_K,
        True,
        correction_bias=bias,
        input_ids=input_ids,
        image_mask=mask,
        bias_vl=bias_vl,
    )
    expected_weights, expected_ids, expected_scores = _oracle(
        logits,
        image_mask=mask,
        bias=bias,
        bias_vl=bias_vl,
        table=None,
        input_ids=input_ids,
    )
    shifted_weights = (scores + torch.where(mask[:, None], bias_vl, bias)).gather(
        1, ids.long()
    )
    shifted_weights /= shifted_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(scores, expected_scores)
    torch.testing.assert_close(weights, expected_weights)
    assert not torch.allclose(weights, shifted_weights)


def test_image_routing_validates_mask_bias_and_unmasked_hash_ids():
    logits, _, bias_vl, table = _router_fixture()
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="rank-1 bool"):
        dsv4_select_experts(
            logits,
            TOP_K,
            True,
            input_ids=input_ids,
            image_mask=torch.zeros((1, 5), dtype=torch.bool),
            bias_vl=bias_vl,
        )
    with pytest.raises(ValueError, match="bias_vl must be loaded"):
        dsv4_select_experts(
            logits,
            TOP_K,
            True,
            input_ids=input_ids,
            image_mask=torch.zeros(5, dtype=torch.bool),
        )
    with pytest.raises(IndexError, match="outside the hash table"):
        dsv4_select_experts(
            logits,
            TOP_K,
            True,
            hash_indices_table=table,
            input_ids=torch.tensor([1, 2, 99, 4, 5]),
            image_mask=torch.zeros(5, dtype=torch.bool),
            bias_vl=bias_vl,
        )


def test_route_scale_is_owned_once_by_normal_and_mega_downstream_seams():
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    input_ids = torch.arange(4)
    image_mask = torch.tensor([False, True, True, False])
    selected = torch.full((4, 2), 0.5)
    selected_ids = torch.tensor([[0, 1]] * 4, dtype=torch.int32)
    normal_seen = {}

    def select(states, ids, mask):
        assert states.shape[0] == ids.shape[0] == mask.shape[0]
        return selected, selected_ids, torch.empty(0)

    def make_topk(states, weights, ids, scores):
        del states, ids, scores
        normal_seen["weights"] = weights.clone()
        return SimpleNamespace(format=SimpleNamespace(is_bypassed=lambda: False))

    def normal_experts(**kwargs):
        return torch.ones_like(kwargs["hidden_states"])

    normal = SimpleNamespace(
        _select_experts=select,
        _make_topk_output=make_topk,
        experts=normal_experts,
        stream_fork=_ImmediateStreamFork(),
        routed_scaling_factor=1.5,
        shared_experts=None,
    )
    normal._forward_shared_experts = MethodType(
        DeepseekV4MoE._forward_shared_experts, normal
    )
    normal_output = DeepseekV4MoE.forward_normal(
        normal, hidden, input_ids, 4, 4, image_mask
    )
    torch.testing.assert_close(normal_seen["weights"], selected)
    torch.testing.assert_close(normal_output, torch.full_like(hidden, 1.5))

    mega_seen = {}

    def mega_experts(states, weights, ids):
        del ids
        mega_seen["rows"] = states.shape[0]
        mega_seen["weights"] = weights.clone()
        return states

    mega = SimpleNamespace(
        _select_experts=select,
        experts=mega_experts,
        stream_fork=_ImmediateStreamFork(),
        routed_scaling_factor=1.5,
        shared_experts=None,
        config=SimpleNamespace(num_experts_per_tok=2),
    )
    mega._forward_shared_experts = MethodType(
        DeepseekV4MoE._forward_shared_experts, mega
    )
    padded_hidden = torch.cat([hidden, torch.full((2, 3), 999.0)])
    padded_ids = torch.cat([input_ids, torch.tensor([99, 100])])
    mega_output = DeepseekV4MoE.forward_mega_moe(
        mega,
        padded_hidden,
        padded_ids,
        ctx=object(),
        comm_manager=object(),
        image_mask=image_mask,
    )
    assert mega_seen["rows"] == image_mask.numel()
    torch.testing.assert_close(mega_seen["weights"], selected * 1.5)
    torch.testing.assert_close(mega_output, hidden)


def test_build_image_block_marks_every_atomic_sentinel_and_image_position():
    semantic_types = {IMAGE_START, IMAGE_END, IMAGE_NEWLINE, IMAGE_PAD, IMAGE}
    for residue in range(4):
        start = 4 + residue
        types, _ = build_image_block(3, 2, start)
        assert semantic_types.issubset(set(types.tolist()))
        end = start + types.numel() - 1
        item = SimpleNamespace(
            modality=SimpleNamespace(name="IMAGE"), offsets=[(start, end)]
        )
        context = SimpleNamespace(
            mm_inputs=[SimpleNamespace(mm_items=[item])],
            extend_prefix_lens=[0],
            extend_seq_lens=[end + 2],
        )
        payload = build_dsv4_vision_forward(
            context,
            num_tokens=end + 2,
            device="cpu",
            max_image_tokens=384,
        )
        assert payload is not None
        assert payload.image_mask[start : end + 1].all()
        assert not payload.image_mask[start - 1]
        assert not payload.image_mask[end + 1]


def _gate_config():
    return SimpleNamespace(
        n_routed_experts=16,
        hidden_size=8,
        num_hash_layers=3,
        vocab_size=32,
        num_experts_per_tok=TOP_K,
        topk_method="noaux_tc",
        num_hidden_layers=43,
        vision_n_layers=32,
        _tokenspeed_dsv4_vision_active=True,
    )


def test_bias_vl_exists_on_all_43_base_layers_and_not_drafts():
    config = _gate_config()
    gates = [DeepseekV4MoEGate(config, layer) for layer in range(46)]
    assert all(gate.bias_vl is not None for gate in gates[:43])
    assert all(gate.is_base_layer for gate in gates[:43])
    assert all(gate.bias_vl is None for gate in gates[43:])
    assert not any(gate.is_base_layer for gate in gates[43:])


def test_normal_mask_sidecar_uses_int32_false_padding_trim_and_rank_order(
    monkeypatch,
):
    calls = []

    class Comm:
        @staticmethod
        def use_all_reduce(*, is_moe):
            assert is_moe
            return False

        @staticmethod
        def moe_tp_ep_group_scattered_num_tokens(ctx):
            del ctx
            return [2, 1]

    layer = SimpleNamespace(
        mapping=SimpleNamespace(
            moe=SimpleNamespace(has_tp_ep=True, tp_ep_group=(0, 1))
        ),
        comm_manager=Comm(),
    )
    layer._pre_mlp_sidecar_comm = MethodType(
        DeepseekV4DecoderLayer._pre_mlp_sidecar_comm, layer
    )
    layer._pre_mlp_input_ids_comm = MethodType(
        DeepseekV4DecoderLayer._pre_mlp_input_ids_comm, layer
    )

    monkeypatch.setattr(
        deepseek_v4_model.pg_manager,
        "get_device_process_group",
        lambda group: group,
    )

    def all_gather(outputs, padded, group):
        calls.append((padded.clone(), group))
        if padded.dtype == torch.int32:
            assert padded.tolist() == [0, 0]
            outputs[0].copy_(torch.tensor([0, 0], dtype=torch.int32))
            outputs[1].copy_(torch.tensor([1, 0], dtype=torch.int32))
        else:
            assert padded.tolist() == [7, 8]
            outputs[0].copy_(torch.tensor([7, 8]))
            outputs[1].copy_(torch.tensor([9, 0]))

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    actual = DeepseekV4DecoderLayer._pre_mlp_image_mask_comm(
        layer, torch.tensor([False, False]), object()
    )
    assert actual.dtype == torch.bool
    assert actual.tolist() == [False, False, True]
    gathered_ids = layer._pre_mlp_input_ids_comm(torch.tensor([7, 8]), object())
    assert gathered_ids.tolist() == [7, 8, 9]
    assert [call[0].shape for call in calls] == [torch.Size([2]), torch.Size([2])]


def test_moe_group_activity_reduces_only_group_rank_indices():
    layer = SimpleNamespace(
        mapping=SimpleNamespace(
            attn=SimpleNamespace(has_dp=True),
            moe=SimpleNamespace(tp_ep_group=(2, 3)),
        )
    )
    ctx = SimpleNamespace(
        dsv4_vision=object(),
        global_dsv4_image_span_intersections=[True, False, False, False],
    )

    assert not DeepseekV4DecoderLayer._dsv4_moe_group_has_image_span(layer, ctx)
    ctx.global_dsv4_image_span_intersections[3] = True
    assert DeepseekV4DecoderLayer._dsv4_moe_group_has_image_span(layer, ctx)

    ctx.global_dsv4_image_span_intersections = None
    ctx.dsv4_vision = None
    assert not DeepseekV4DecoderLayer._dsv4_moe_group_has_image_span(layer, ctx)

    ctx.dsv4_vision = object()
    with pytest.raises(RuntimeError, match="rank-symmetric"):
        DeepseekV4DecoderLayer._dsv4_moe_group_has_image_span(layer, ctx)


@pytest.mark.parametrize("local_rows", [2, 0], ids=["text_peer", "idle_peer"])
def test_rsag_mask_gather_is_rank_symmetric_and_cached_once(monkeypatch, local_rows):
    gathered_masks = []
    ffn_masks = []

    class FFN:
        use_mega_moe = False
        gate = SimpleNamespace(is_base_layer=True, is_hash_moe=False, bias_vl=object())

        def __call__(self, hidden_states, input_ids, *args, **kwargs):
            del input_ids, args
            ffn_masks.append(kwargs["image_mask"])
            return hidden_states

    class Comm:
        @staticmethod
        def use_all_reduce(*, is_moe):
            assert is_moe
            return False

        @staticmethod
        def moe_tp_ep_group_scattered_num_tokens(ctx):
            del ctx
            return [local_rows, 1]

        @staticmethod
        def pre_mlp_comm(hidden_states, ctx):
            del ctx
            peer = torch.zeros((1, *hidden_states.shape[1:]), dtype=hidden_states.dtype)
            return torch.cat([hidden_states, peer])

        @staticmethod
        def get_num_tokens(ctx):
            del ctx
            return local_rows + 1, max(local_rows, 1)

        @staticmethod
        def post_mlp_comm(hidden_states, residual, ctx):
            del residual, ctx
            return hidden_states[:local_rows], None

    layer = SimpleNamespace(
        ffn=FFN(),
        comm_manager=Comm(),
        mapping=SimpleNamespace(
            attn=SimpleNamespace(has_dp=True),
            moe=SimpleNamespace(has_tp_ep=True, tp_ep_group=(0, 1), tp_ep_rank=0),
        ),
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        ffn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        attn=lambda *args: args[1],
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=1,
    )
    for method_name in (
        "_dsv4_moe_group_has_image_span",
        "_validate_dsv4_rsag_local_rows",
        "_cached_dsv4_rsag_image_mask",
    ):
        setattr(
            layer,
            method_name,
            MethodType(getattr(DeepseekV4DecoderLayer, method_name), layer),
        )

    def gather_mask(self, image_mask, ctx):
        del self, ctx
        gathered_masks.append(image_mask.clone())
        return torch.cat([image_mask, torch.ones(1, dtype=torch.bool)])

    layer._pre_mlp_image_mask_comm = MethodType(gather_mask, layer)
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_pre",
        lambda hidden, *args, **kwargs: (hidden, None, None),
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_fused_hc",
        lambda hidden, residual, *args, **kwargs: (residual, hidden, None, None),
    )
    ctx = SimpleNamespace(
        dsv4_vision=None,
        global_dsv4_image_span_intersections=[False, True],
        dsv4_moe_image_mask=None,
        dsv4_moe_image_mask_cached=False,
    )
    hidden = torch.zeros((local_rows, 2, 4))
    input_ids = torch.arange(local_rows)

    for _ in range(2):
        DeepseekV4DecoderLayer.forward(
            layer,
            torch.arange(local_rows),
            hidden,
            ctx,
            input_ids,
        )

    assert len(gathered_masks) == 1
    assert gathered_masks[0].shape == torch.Size([local_rows])
    assert not gathered_masks[0].any()
    assert len(ffn_masks) == 2
    assert all(mask.tolist() == [False] * local_rows + [True] for mask in ffn_masks)


def test_genuinely_all_false_gather_collapses_to_cached_text_rollback():
    calls = []
    layer = SimpleNamespace()

    def gather(image_mask, ctx):
        del ctx
        calls.append(image_mask.clone())
        return torch.zeros(3, dtype=torch.bool)

    layer._pre_mlp_image_mask_comm = gather
    ctx = SimpleNamespace(
        dsv4_moe_image_mask=None,
        dsv4_moe_image_mask_cached=False,
    )
    input_ids = torch.arange(2)

    first = DeepseekV4DecoderLayer._cached_dsv4_rsag_image_mask(
        layer, None, 2, input_ids, ctx
    )
    second = DeepseekV4DecoderLayer._cached_dsv4_rsag_image_mask(
        layer, torch.ones(2, dtype=torch.bool), 2, input_ids, ctx
    )

    assert first is None
    assert second is None
    assert ctx.dsv4_moe_image_mask_cached
    assert len(calls) == 1
    assert not calls[0].any()


@pytest.mark.parametrize(
    ("bias_vl", "intersections"),
    [
        (object(), [False, False]),
        (None, None),
    ],
    ids=["vision_active", "vision_disabled"],
)
def test_all_text_group_adds_no_image_mask_collective(
    monkeypatch, bias_vl, intersections
):
    class FFN:
        use_mega_moe = False
        gate = SimpleNamespace(is_base_layer=True, is_hash_moe=False, bias_vl=bias_vl)

        def __call__(self, hidden_states, input_ids, *args, **kwargs):
            del input_ids, args
            assert kwargs["image_mask"] is None
            return hidden_states

    class Comm:
        @staticmethod
        def use_all_reduce(*, is_moe):
            assert is_moe
            return False

        @staticmethod
        def pre_mlp_comm(hidden_states, ctx):
            del ctx
            return hidden_states

        @staticmethod
        def get_num_tokens(ctx):
            del ctx
            return 2, 2

        @staticmethod
        def post_mlp_comm(hidden_states, residual, ctx):
            del ctx
            return hidden_states, residual

    layer = SimpleNamespace(
        ffn=FFN(),
        comm_manager=Comm(),
        mapping=SimpleNamespace(
            attn=SimpleNamespace(has_dp=True),
            moe=SimpleNamespace(has_tp_ep=True, tp_ep_group=(0, 1)),
        ),
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        ffn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        attn=lambda *args: args[1],
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=1,
    )
    layer._dsv4_moe_group_has_image_span = MethodType(
        DeepseekV4DecoderLayer._dsv4_moe_group_has_image_span, layer
    )
    layer._pre_mlp_image_mask_comm = lambda *args: (_ for _ in ()).throw(
        AssertionError("image-mask collective called")
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_pre",
        lambda hidden, *args, **kwargs: (hidden, None, None),
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_fused_hc",
        lambda hidden, residual, *args, **kwargs: (residual, hidden, None, None),
    )
    ctx = SimpleNamespace(
        dsv4_vision=None,
        global_dsv4_image_span_intersections=intersections,
    )

    DeepseekV4DecoderLayer.forward(
        layer,
        torch.arange(2),
        torch.zeros((2, 2, 4)),
        ctx,
        torch.arange(2),
    )


@pytest.mark.parametrize("base_layer", [True, False])
def test_decoder_consumes_only_base_gate_semantic_mask(monkeypatch, base_layer):
    mask = torch.tensor([False, True, True])
    seen = []

    class FFN:
        use_mega_moe = False
        gate = SimpleNamespace(is_base_layer=base_layer, is_hash_moe=False)

        def __call__(self, hidden_states, input_ids, *args, **kwargs):
            del input_ids, args
            seen.append(kwargs["image_mask"])
            return hidden_states

    class Comm:
        @staticmethod
        def pre_mlp_comm(hidden_states, ctx):
            del ctx
            return hidden_states

        @staticmethod
        def get_num_tokens(ctx):
            del ctx
            return 3, 3

        @staticmethod
        def post_mlp_comm(hidden_states, residual, ctx):
            del ctx
            return hidden_states, residual

    layer = SimpleNamespace(
        ffn=FFN(),
        comm_manager=Comm(),
        mapping=SimpleNamespace(moe=SimpleNamespace(has_tp_ep=False)),
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        ffn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        attn=lambda *args: args[1],
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=1,
    )
    layer._pre_mlp_sidecar_comm = MethodType(
        DeepseekV4DecoderLayer._pre_mlp_sidecar_comm, layer
    )
    layer._pre_mlp_image_mask_comm = MethodType(
        DeepseekV4DecoderLayer._pre_mlp_image_mask_comm, layer
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_pre",
        lambda hidden, *args, **kwargs: (hidden, None, None),
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_fused_hc",
        lambda hidden, residual, *args, **kwargs: (residual, hidden, None, None),
    )
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    ctx = SimpleNamespace(dsv4_vision=SimpleNamespace(image_mask=mask))
    DeepseekV4DecoderLayer.forward(
        layer,
        torch.arange(3),
        hidden,
        ctx,
        torch.tensor([7, 100_001, 100_002]),
    )
    assert len(seen) == 1
    if base_layer:
        assert torch.equal(seen[0], mask)
    else:
        assert seen[0] is None


@pytest.mark.skipif(
    os.environ.get("DSV4_VISION_RUN_T1") != "1",
    reason="set DSV4_VISION_RUN_T1=1 for the one-B200 routing gate",
)
def test_t1_gpu_fused_text_and_eager_image_oracles(monkeypatch):
    assert torch.cuda.device_count() == 1
    gpu_name = torch.cuda.get_device_name(0)
    assert "B200" in gpu_name
    print(f"WP-05 T1 GPU count=1 name={gpu_name}")

    device = torch.device("cuda:0")
    tokens = 7
    experts = 256
    logits, bias, bias_vl, _ = _router_fixture(
        device=device, tokens=tokens, experts=experts
    )
    table = torch.stack(
        [
            (torch.arange(TOP_K, device=device) * 17 + row * 7 + 3) % experts
            for row in range(32)
        ]
    ).to(torch.int32)
    text_ids = torch.arange(tokens, device=device, dtype=torch.int64) + 1
    mixed_ids = text_ids.clone()
    image_mask = torch.tensor(
        [False, True, False, True, False, True, False], device=device
    )
    mixed_ids[image_mask] = 129_280 + torch.arange(image_mask.sum(), device=device)

    original = deepseek_v4_model._kernel_dsv4_select_experts
    fused_calls = []

    def tracked_fused(*args):
        result = original(*args)
        fused_calls.append(args)
        return result

    monkeypatch.setattr(deepseek_v4_model, "_kernel_dsv4_select_experts", tracked_fused)
    for layer_index in (0, 1, 2, 3, 42):
        hash_layer = layer_index < 3
        correction = None if hash_layer else bias
        hash_table = table if hash_layer else None
        text_weights, text_experts, _ = dsv4_select_experts(
            logits,
            TOP_K,
            True,
            correction_bias=correction,
            hash_indices_table=hash_table,
            input_ids=text_ids,
            image_mask=None,
            bias_vl=bias_vl,
        )
        expected_text = _oracle(
            logits,
            image_mask=None,
            bias=correction,
            bias_vl=bias_vl,
            table=hash_table,
            input_ids=text_ids,
        )
        assert torch.equal(text_experts, expected_text[1])
        torch.testing.assert_close(text_weights, expected_text[0], rtol=1e-6, atol=1e-7)

        image_weights, image_experts, _ = dsv4_select_experts(
            logits,
            TOP_K,
            True,
            correction_bias=correction,
            hash_indices_table=hash_table,
            input_ids=mixed_ids,
            image_mask=image_mask,
            bias_vl=bias_vl,
        )
        expected_image = _oracle(
            logits,
            image_mask=image_mask,
            bias=correction,
            bias_vl=bias_vl,
            table=hash_table,
            input_ids=mixed_ids,
        )
        assert torch.equal(image_experts, expected_image[1])
        torch.testing.assert_close(
            image_weights, expected_image[0], rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            image_weights * 1.5,
            expected_image[0] * 1.5,
            rtol=1e-6,
            atol=1e-7,
        )

    assert len(fused_calls) == 5
    assert all(call[5] is text_ids for call in fused_calls)
