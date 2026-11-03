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

"""Beam's MoE, expert parallelism and quantized checkpoints, on CPU.

The MoE class each backend selects, the routed scale, the ranks' agreement,
expert parallelism, and the compressed-tensors FP8 dynamic W8A8 and ModelOpt
NVFP4 checkpoints: what each admits, plans, loads and refuses. Expert, FP8 and
FP4 kernels need a GPU: the tests record their calls or run their statements
instead.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform
from tokenspeed_kernel.thirdparty.flashinfer import trtllm_bf16_moe

from tokenspeed.runtime.configs import model_config as model_config_module
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.moe.loader import MoECheckpointLoader
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as w8a8_scheme,
)
from tokenspeed.runtime.layers.quantization.nvfp4 import (
    Nvfp4Config,
    Nvfp4W4A16Config,
)
from tokenspeed.runtime.models import beam
from tokenspeed.runtime.models.beam_quant import beam_checkpoint_format
from tokenspeed.runtime.utils.env import global_server_args_dict

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.beam_fixture import (
    tiny_config,
    tiny_weights,
    write_tiny_checkpoint,
)
from test.runtime.models.conftest import model_config as _model_config
from test.runtime.models.conftest import server_args as _server_args

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="runtime-1gpu", disabled_on_runners=["amd-*", "h100-*"]
)

KERNEL_ROUTING_APPLY = "flashinfer_trtllm_unquant_moe_apply"
ROUTED_APPLY = "flashinfer_trtllm_unquant_routed_moe_apply"
FP8_ROUTED = "flashinfer_trtllm_fp8_channel_routed_moe_apply"
NVFP4_ROUTED = "flashinfer_trtllm_nvfp4_routed_moe_apply"
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def nvidia_platform():
    """These vendor policy and launch tests exercise the NVIDIA path on CPU."""
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor="nvidia", arch_version=ArchVersion(10, 3))
    )
    yield
    Platform.override(original)


def _moe(mapping=None, config=None, cls=None):
    cls = cls or beam.beam_moe_class()
    mapping = mapping or Mapping(rank=0, world_size=1)
    return cls(config or tiny_config(), mapping, 1, "model.layers.1.mlp")


def _model(quant_config=None, mapping=None, config=None):
    return beam.BeamForCausalLM(
        config or tiny_config(), mapping or Mapping(rank=0, world_size=1), quant_config
    )


def _config(**overrides):
    config = tiny_config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------- MoE policy
@pytest.mark.parametrize(
    "backend,expected",
    [
        ("auto", beam.BeamMoE),
        ("triton", beam.BeamMoE),
        ("flashinfer_trtllm", beam.BeamVendorMoE),
    ],
)
def test_the_backend_selects_the_moe_class(server, backend, expected):
    server(backend)
    assert beam.beam_moe_class() is expected


def test_vendor_routes_take_the_routed_scale_once(server):
    """On Beam's routes the vendor layer's route epilogue folds the routed
    scale into the FP32 weights, once (TopK applies none): the Triton layer's
    unscaled weights times 2.5 rounded to FP32, the same int32 ids."""
    server("auto")
    default = _moe()
    server("flashinfer_trtllm")
    vendor = _moe()
    bias = torch.linspace(-0.1, 0.1, 8, dtype=torch.float32)
    default.gate.e_score_correction_bias.data.copy_(bias)
    vendor.gate.e_score_correction_bias.data.copy_(bias)
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(5, 8, generator=generator, dtype=torch.float32)
    hidden = torch.zeros(5, 128, dtype=torch.bfloat16)
    unscaled = default.topk(hidden, logits)
    scaled = vendor.topk(hidden, logits)
    assert scaled.topk_ids.dtype == torch.int32
    assert torch.equal(scaled.topk_ids, unscaled.topk_ids)
    assert scaled.topk_weights.dtype == torch.float32
    expected = unscaled.topk_weights * torch.tensor(2.5, dtype=torch.float32)
    assert torch.equal(
        scaled.topk_weights.view(torch.int32), expected.view(torch.int32)
    )


def test_kernel_routed_layer_refuses_a_non_finite_checkpoint_bias(server):
    server("flashinfer_trtllm", kernel_routing=True)
    layer = _moe()
    bias = layer.gate.e_score_correction_bias
    finite = torch.linspace(-1, 1, bias.numel(), dtype=torch.float32)
    bias.weight_loader(bias, finite)
    assert torch.equal(bias.data, finite)
    loaded = finite.clone()
    loaded[3] = float("nan")
    with pytest.raises(ValueError, match="finite e_score_correction_bias"):
        bias.weight_loader(bias, loaded)
    assert torch.equal(bias.data, finite)


def test_vendor_layer_needs_normalized_routes(server):
    server("flashinfer_trtllm", kernel_routing=True)
    config = tiny_config()
    config.norm_topk_prob = False
    with pytest.raises(NotImplementedError, match="normalized routes"):
        beam.BeamVendorMoE(
            config, Mapping(rank=0, world_size=1), 1, "model.layers.1.mlp"
        )


# ---------------------------------------------------------------- rank agreement
def _two_ranks(monkeypatch, peer):
    """This process is rank 0 of two; rank 1 reports ``peer``."""
    gathers = []

    def all_gather_object(decisions, decision, group):
        gathers.append((decision, group))
        decisions[:] = [decision, peer]

    monkeypatch.setattr(
        beam,
        "dist",
        SimpleNamespace(
            is_initialized=lambda: True, all_gather_object=all_gather_object
        ),
    )
    monkeypatch.setattr(
        beam,
        "pg_manager",
        SimpleNamespace(
            init_process_group=lambda group, backend: None,
            get_process_group=lambda backend, group: f"{backend}{group}",
        ),
    )
    return gathers


WORLD = SimpleNamespace(world_size=2, world_group=(0, 1))


@pytest.mark.parametrize(
    "peer", [("BeamVendorMoE", False), ("BeamMoE", False)], ids=["routing", "class"]
)
def test_ranks_that_disagree_fail_startup(monkeypatch, peer):
    # Rank 0 routes in the kernel; rank 1 could not build that routing, or kept
    # the Triton MoE.
    _two_ranks(monkeypatch, peer)
    with pytest.raises(RuntimeError, match="Beam's MoE differs across ranks"):
        beam.beam_moe_agreement(WORLD, ("BeamVendorMoE", True))


# ---------------------------------------------------------------- expert parallelism
EP_GRID = dict(
    num_experts=256,
    n_group=8,
    topk_group=4,
    num_experts_per_tok=8,
    num_key_value_heads=8,
)


def _ep_mapping(rank, ep=8, **sizes):
    return Mapping(rank=rank, world_size=ep, moe_ep_size=ep, **sizes)


def _count_expert_loads(monkeypatch):
    names = []
    real_load = MoECheckpointLoader.load

    def load(self, name, weight):
        names.append(name)
        return real_load(self, name, weight)

    monkeypatch.setattr(MoECheckpointLoader, "load", load)
    return names


def _expert_slots(model, layer=1):
    """{checkpoint name: runtime view} of one MoE layer's local expert slots."""
    experts = model.model.layers[layer].mlp.experts
    width = experts.intermediate_size // experts.tp_size
    half = experts.w13_weight.shape[1] // 2
    first = experts.ep_rank * experts.num_local_experts
    views = {}
    for slot in range(experts.num_local_experts):
        prefix = f"model.layers.{layer}.mlp.experts.{first + slot}"
        views[f"{prefix}.gate_proj.weight"] = experts.w13_weight[slot, :width]
        views[f"{prefix}.up_proj.weight"] = experts.w13_weight[slot, half:][:width]
        views[f"{prefix}.down_proj.weight"] = experts.w2_weight[slot, :, :width]
    return views


def test_the_config_admits_ep_over_whole_experts_only():
    _config(**EP_GRID).validate_parallelism(tensor_parallel_size=8, ep_size=8)
    with pytest.raises(ValueError, match="ep_size must be a positive integer"):
        _config(**EP_GRID).validate_parallelism(tensor_parallel_size=8, ep_size=3)


@pytest.mark.parametrize(
    "backend, mapping, redundant, match",
    [
        ("flashinfer_trtllm", _ep_mapping(0), 8, "redundant experts"),
        ("auto", _ep_mapping(0), 0, r"--ep-size needs --moe-backend flashinfer_trtllm"),
        # attention TP8 but MoE TP x EP 4: the MoE output would need a
        # reduce-scatter / all-gather pair instead of the one all-reduce.
        (
            "flashinfer_trtllm",
            Mapping(rank=0, world_size=8, moe_ep_size=4, moe_dp_size=2),
            0,
            "same TP size for attention, dense and MoE",
        ),
    ],
    ids=["redundant", "triton-moe", "moe-tp-ep"],
)
def test_the_model_refuses_an_expert_layout_it_cannot_serve(
    server, monkeypatch, backend, mapping, redundant, match
):
    server(backend)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", redundant)
    with pytest.raises(ValueError, match=match):
        _model(mapping=mapping, config=_config(**EP_GRID))


def test_ep8_rank_hands_global_routes_and_its_expert_offset_to_the_kernel(
    server, monkeypatch
):
    server("flashinfer_trtllm")
    layer = _moe(_ep_mapping(5), _config(**EP_GRID)).experts
    out = torch.full((3, 128), 2.0, dtype=torch.bfloat16)
    launcher = mock.Mock(return_value=[out])
    monkeypatch.setattr(trtllm_bf16_moe, "trtllm_bf16_routed_moe", launcher)
    ids = torch.tensor([[0, 160, 191, 255, 3, 170, 7, 8]] * 3, dtype=torch.int32)
    weights = torch.rand(3, 8, dtype=torch.float32)
    hidden = torch.zeros(3, 128, dtype=torch.bfloat16)
    routes = StandardTopKOutput(weights, ids, torch.zeros(3, 256))
    assert layer(hidden, routes, num_global_tokens=3, max_num_tokens_per_gpu=3) is out
    kwargs = launcher.call_args.kwargs
    # Global expert ids; the kernel keeps [160, 192) and skips the rest.
    assert (kwargs["local_expert_offset"], kwargs["local_num_experts"]) == (160, 32)
    assert (kwargs["num_experts"], kwargs["top_k"]) == (256, 8)
    assert kwargs["intermediate_size"] == 64 and kwargs["do_finalize"] is True
    topk_ids, topk_weights = kwargs["topk_ids"]
    assert torch.equal(topk_ids, ids)
    assert torch.equal(topk_weights, weights.to(torch.bfloat16))


def test_ep8_ranks_load_only_their_experts_and_check_the_whole_checkpoint(
    server, monkeypatch
):
    server("flashinfer_trtllm")
    config = _config(**EP_GRID)
    weights = tiny_weights(config)
    loads = _count_expert_loads(monkeypatch)
    owners = {}
    for rank in range(8):
        model = _model(mapping=_ep_mapping(rank), config=config)
        loads.clear()
        assert model.load_weights(weights.items()) == weights.keys()
        views = _expert_slots(model)
        # Each rank calls the expert loader only for its own 32 x 3 tensors.
        assert sorted(loads) == sorted(views) and len(views) == 32 * 3
        for name, view in views.items():
            assert torch.equal(view, weights[name]), name
            owners.setdefault(name, []).append(rank)
    expert_names = {name for name in weights if ".mlp.experts." in name}
    assert set(owners) == expert_names and len(expert_names) == 256 * 3
    assert all(len(ranks) == 1 for ranks in owners.values())
    assert owners["model.layers.1.mlp.experts.37.down_proj.weight"] == [1]


@pytest.mark.parametrize(
    "defect,error",
    [
        ("missing-remote", "Missing Beam checkpoint tensors"),
        ("unexpected-expert", "Unexpected Beam checkpoint tensor"),
    ],
)
def test_ep8_skip_keeps_the_checkpoint_checks(server, defect, error):
    server("flashinfer_trtllm")
    config = _config(**EP_GRID)
    weights = tiny_weights(config)
    model = _model(mapping=_ep_mapping(0), config=config)
    remote = "model.layers.1.mlp.experts.255.down_proj.weight"
    items = list(weights.items())
    if defect == "missing-remote":
        items = [(name, w) for name, w in items if name != remote]
    else:
        items.append((remote.replace(".255.", ".256."), weights[remote]))
    with pytest.raises((KeyError, ValueError), match=error):
        model.load_weights(items)


# ---------------------------------------------------------------- FP8 checkpoints
# llm-compressor's FP8_DYNAMIC config, as Beam's FP8 checkpoint carries it.
_ARGS = {"actorder": None, "block_structure": None, "group_size": None}
RAW = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "ignore": ["lm_head", "model.embed_tokens", "re:.*mlp\\.gate$"],
    "kv_cache_scheme": None,
    "config_groups": {
        "FP8_DYNAMIC": {
            "format": "float-quantized",
            "targets": ["Linear"],
            "output_activations": None,
            "weights": {
                **_ARGS,
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "dynamic": False,
                "strategy": "channel",
                "scale_dtype": None,
            },
            "input_activations": {
                **_ARGS,
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "dynamic": True,
                "strategy": "token",
            },
        }
    },
}

GROUP = ("config_groups", "FP8_DYNAMIC")


def ct(raw=RAW):
    return CompressedTensorsConfig.from_config(copy.deepcopy(raw))


def edited(path, value):
    raw = copy.deepcopy(RAW)
    *keys, last = path
    node = raw
    for key in keys:
        node = node[key]
    node[last] = value
    return raw


def quantize(weight, scale_dtype=torch.bfloat16):
    """FP8 codes and ``[N, 1]`` scales of one weight: amax / 448 per output
    channel in the model dtype, codes saturated at 448 and rounded to nearest."""
    amax = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 448.0).to(scale_dtype)
    codes = (weight.float() / scale.float()).clamp(-448.0, 448.0).to(FP8)
    return codes, scale


def bf16_once(exact):
    """The FP64 values rounded once to BF16, to nearest even, on their bits."""
    bits = exact.double().view(torch.int64)
    lsb = (bits >> 45) & 1
    bits = ((bits + (1 << 44) - 1 + lsb) >> 45) << 45
    return bits.view(torch.float64).float().to(torch.bfloat16)


def fp8_checkpoint(scale_dtype=torch.bfloat16, seed=0):
    """(checkpoint tensors, BF16 reference) of the tiny Beam in the FP8 layout:
    every linear but the router and lm_head quantized, as RAW's ignore says."""
    torch.manual_seed(seed)
    tensors, reference = {}, {}
    for name, weight in tiny_weights(tiny_config()).items():
        module = name.removesuffix(".weight")
        quantized = (
            name.endswith("proj.weight")
            and weight.ndim == 2
            and ".mlp.gate." not in name
        )
        if not quantized:
            tensors[name] = reference[name] = weight
            continue
        codes, scale = quantize(weight, scale_dtype)
        tensors[name], tensors[f"{module}.weight_scale"] = codes, scale
        reference[name] = bf16_once(codes.double() * scale.double())
    return tensors, reference


def test_the_provider_layouts_are_admitted():
    assert beam_checkpoint_format(ct()) == "fp8_channel"
    naive = edited(("format",), "naive-quantized")
    assert beam_checkpoint_format(ct(naive)) == "fp8_channel"
    assert beam_checkpoint_format(NVFP4) == "nvfp4"
    assert beam.beam_quant_policy(NVFP4).fmt == "nvfp4"


W, A = (*GROUP, "weights"), (*GROUP, "input_activations")


@pytest.mark.parametrize(
    "quant_config, reason",
    [
        (lambda: ct(edited(("format",), "pack-quantized")), "format 'pack-quantized'"),
        (
            lambda: ct(edited((*W, "strategy"), "tensor")),
            "weights of Linear strategy 'tensor'",
        ),
        (lambda: Nvfp4Config(group_size=32), "ModelOpt NVFP4 with group_size 32"),
        (lambda: Nvfp4W4A16Config(), "quantization nvfp4 (Nvfp4W4A16Config)"),
    ],
)
def test_other_quantized_variants_are_refused(server, quant_config, reason):
    server("flashinfer_trtllm")
    with pytest.raises(ValueError, match="Beam refuses") as error:
        _model(quant_config())
    assert reason in str(error.value) and "admitted:" in str(error.value)


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_quantized_experts_need_the_vendor_moe(server, backend):
    server(backend)
    for quant_config in (ct(), NVFP4):
        with pytest.raises(ValueError, match="needs --moe-backend flashinfer_trtllm"):
            _model(quant_config)


@pytest.mark.parametrize("backend", ["auto", "triton", "gluon", "flashinfer_trtllm"])
def test_nvidia_mxfp4_experts_remain_unsupported(server, backend):
    from test.runtime.models.test_beam_mxfp4 import mxfp4_config

    server(backend)
    with pytest.raises(
        ValueError, match="requires? AMD gfx950|needs --moe-backend flashinfer_trtllm"
    ):
        _model(mxfp4_config())


def _stacked(tensors, layer, names):
    return torch.cat([tensors[f"{layer}.{name}"] for name in names])


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_the_fp8_load_keeps_codes_and_scales(server, scale_dtype):
    """Every FP8 weight keeps its codes, bit for bit, and its channel scale,
    exactly as FP32 (Q, K and V stacked, gate and up stacked, the experts'
    gate and up rows then down); every other tensor is the checkpoint's."""
    server("flashinfer_trtllm")
    tensors, _ = fp8_checkpoint(scale_dtype)
    model = _model(ct())
    assert model.load_weights(list(tensors.items())) == set(tensors)
    actual = dict(model.named_parameters())
    for layer in ("model.layers.0", "model.layers.1"):
        attn = f"{layer}.self_attn"
        qkv = ("q_proj", "k_proj", "v_proj")
        for suffix in ("weight", "weight_scale"):
            names = [f"{p}.{suffix}" for p in qkv]
            expected = _stacked(tensors, attn, names)
            got = actual[f"{attn}.qkv_proj.{suffix}"]
            if suffix == "weight":
                assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
            else:
                assert torch.equal(got, expected.float())
        for proj in ("o_proj", "gate_proj"):
            got = actual[f"{attn}.{proj}.weight"]
            assert torch.equal(
                got.view(torch.uint8),
                tensors[f"{attn}.{proj}.weight"].view(torch.uint8),
            )
            assert torch.equal(
                actual[f"{attn}.{proj}.weight_scale"],
                tensors[f"{attn}.{proj}.weight_scale"].float(),
            )
    dense = "model.layers.0.mlp"
    assert torch.equal(
        actual[f"{dense}.gate_up_proj.weight_scale"],
        _stacked(
            tensors, dense, ["gate_proj.weight_scale", "up_proj.weight_scale"]
        ).float(),
    )
    experts = "model.layers.1.mlp.experts"
    for e in range(8):
        w13 = torch.cat(
            [tensors[f"{experts}.{e}.{p}.weight"] for p in ("gate_proj", "up_proj")]
        )
        s13 = torch.cat(
            [
                tensors[f"{experts}.{e}.{p}.weight_scale"]
                for p in ("gate_proj", "up_proj")
            ]
        )
        assert torch.equal(
            actual[f"{experts}.w13_weight"][e].view(torch.uint8), w13.view(torch.uint8)
        )
        assert torch.equal(
            actual[f"{experts}.w13_weight_scale"][e], s13.float().reshape(-1)
        )
        assert torch.equal(
            actual[f"{experts}.w2_weight_scale"][e],
            tensors[f"{experts}.{e}.down_proj.weight_scale"].float().reshape(-1),
        )
    for name in ("model.layers.1.mlp.gate.weight", "model.embed_tokens.weight"):
        assert torch.equal(actual[name], tensors[name])


def _post_load(module):
    """The loader's post-load pass over ``module``: each module in
    ``named_modules`` order (a module before its children), its quant method's
    processing, then its own."""
    for _, child in module.named_modules():
        quant_method = getattr(child, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(child)
        process = getattr(child, "process_weights_after_loading", None)
        if process is not None:
            process(child)


def _fp8_linear_stand_in(calls):
    """fp8_dynamic_linear on CPU: per-token codes and scales, FP64 product."""

    def linear(x, weight, weight_scale):
        calls.append(weight)
        x_scale = x.double().abs().amax(-1, keepdim=True) / 448.0
        codes = (x.double() / x_scale).clamp(-448, 448).float().to(FP8)
        out = (codes.double() * x_scale) @ (
            weight.double() * weight_scale.double().reshape(1, -1)
        )
        return out.to(x.dtype)

    return linear


def _fp8_gemm_stand_in(calls):
    """kernel_mm on CPU: the codes and scales it is given, FP64 product."""

    def gemm(a, b, *, A_scales, B_scales, out_dtype, quant):
        calls.append(a)
        out = (a.double() * A_scales.double()) @ (
            b.double() * B_scales.double().reshape(1, -1)
        )
        return out.to(out_dtype)

    return gemm


def test_attention_runs_qkv_and_gate_as_one_fp8_gemm(server, monkeypatch):
    """After loading, the Q/K/V and output gate FP8 codes and channel scales
    form one GEMM (Q/K/V rows, then the gate rows) that gives both outputs
    exactly as the two projections do."""
    server("flashinfer_trtllm")
    tensors, _ = fp8_checkpoint()
    model = _model(ct())
    model.load_weights(list(tensors.items()))
    for layer in model.model.layers:
        attn = layer.self_attn
        projections = (attn.qkv_proj, attn.gate_proj)
        codes = torch.cat([p.weight.data.view(torch.uint8) for p in projections])
        scales = torch.cat([p.weight_scale.data for p in projections])
        _post_load(attn)
        assert torch.equal(attn.qkv_gate_weight.t().view(torch.uint8), codes)
        assert torch.equal(attn.qkv_gate_scale, scales)

    calls = []
    linear = _fp8_linear_stand_in(calls)
    monkeypatch.setattr(beam, "fp8_dynamic_linear", linear)
    monkeypatch.setattr(w8a8_scheme, "fp8_dynamic_linear", linear)
    attn = model.model.layers[0].self_attn
    x = torch.randn(3, attn.qkv_proj.weight.shape[0]).to(torch.bfloat16)
    qkv, gate = attn.project_qkv_gate(x)
    assert len(calls) == 1
    separate_qkv, _ = attn.qkv_proj(x)
    separate_gate, _ = attn.gate_proj(x)
    assert torch.equal(qkv, separate_qkv) and torch.equal(gate, separate_gate)


def test_the_offered_token_fp8_is_consumed_once(server, monkeypatch):
    """The input norm's per-token FP8 of the rows it returned serves exactly
    the one projection of those rows: the fused GEMM runs on the offered codes
    and scales, then the offer is gone — a second call, even of the same rows,
    quantizes them again itself."""
    server("flashinfer_trtllm")
    tensors, _ = fp8_checkpoint()
    model = _model(ct())
    model.load_weights(list(tensors.items()))
    attn = model.model.layers[0].self_attn
    _post_load(attn)
    x = torch.randn(3, attn.qkv_proj.weight.shape[0]).to(torch.bfloat16)
    # What the input norm's rmsnorm_quantize_fp8 launch offers for x's rows:
    # the per-token quantization _fp8_linear_stand_in computes again below.
    x_scale = x.double().abs().amax(-1, keepdim=True) / 448.0
    codes = (x.double() / x_scale).clamp(-448, 448).float().to(FP8)
    gemm_calls, linear_calls = [], []
    monkeypatch.setattr(beam, "kernel_mm", _fp8_gemm_stand_in(gemm_calls))
    linear = _fp8_linear_stand_in(linear_calls)
    monkeypatch.setattr(beam, "fp8_dynamic_linear", linear)

    attn.offer_token_fp8(x, codes, x_scale)
    offered_qkv, offered_gate = attn.project_qkv_gate(x)
    assert len(gemm_calls) == 1 and gemm_calls[0] is codes
    assert linear_calls == []
    assert attn._token_fp8 is None

    again_qkv, again_gate = attn.project_qkv_gate(x)
    assert len(gemm_calls) == 1 and len(linear_calls) == 1
    assert torch.equal(again_qkv, offered_qkv)
    assert torch.equal(again_gate, offered_gate)

    # Rows the norm did not offer get no stale codes either.
    attn.offer_token_fp8(x, codes, x_scale)
    attn.project_qkv_gate(x.clone())
    assert len(gemm_calls) == 1 and len(linear_calls) == 2
    assert attn._token_fp8 is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 kernels need CUDA")
@pytest.mark.parametrize("sandwich", [False, True])
@pytest.mark.parametrize("consumer_kind", ["dense", "attention"])
def test_norm_fp8_offer_matches_separate_quantization(server, sandwich, consumer_kind):
    """Real norm and GEMM kernels agree with separately quantizing the BF16 rows."""
    from tokenspeed.runtime.models.beam_sandwich_fusion import BeamSandwichNorm

    server("flashinfer_trtllm")
    tensors, _ = fp8_checkpoint()
    model = _model(ct())
    model.load_weights(list(tensors.items()))
    model.cuda()
    layer = model.model.layers[0]
    consumer = layer.mlp if consumer_kind == "dense" else layer.self_attn
    _post_load(consumer)
    norm = (
        layer.post_attention_layernorm
        if consumer_kind == "dense"
        else layer.input_layernorm
    )
    project = consumer if consumer_kind == "dense" else consumer.project_qkv_gate

    def equal_outputs(left, right):
        if isinstance(left, tuple):
            return all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
        return torch.equal(left, right)

    if sandwich:
        norm.__class__ = BeamSandwichNorm
        object.__setattr__(norm, "_sandwich_producer", layer.attn_output_layernorm)
        object.__setattr__(norm, "_sandwich_mapping", model.mapping)
    for tokens in (1, 8, 64):
        x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
        rows, _ = norm(x, x, x_scale=0.5, residual_scale=0.75)
        offered_rows, codes, scales = consumer._token_fp8
        assert offered_rows is rows
        expected_codes, expected_scales = beam.quantize_fp8(
            rows, granularity="token", enable_pdl=False
        )
        assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
        assert torch.equal(scales, expected_scales)
        offered_output = project(rows)
        assert consumer._token_fp8 is None
        separate_output = project(rows)
        assert equal_outputs(offered_output, separate_output)
        # An offer belongs to exactly one tensor and cannot survive a different input.
        consumer.offer_token_fp8(rows, codes, scales)
        assert equal_outputs(project(rows.clone()), separate_output)
        assert consumer._token_fp8 is None
    assert not consumer.takes_token_fp8(
        torch.zeros(65, 128, device="cuda", dtype=torch.bfloat16)
    )
    bf16 = _model().cuda().model.layers[0].mlp
    _post_load(bf16)
    assert not bf16.takes_token_fp8(
        torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    )


def test_dense_fp8_offer_is_owned_by_one_call(server):
    """Row identity and single consumption hold even before kernels are launched."""
    server("flashinfer_trtllm")
    mlp = _model(ct()).model.layers[0].mlp
    rows = torch.zeros(1, 128, dtype=torch.bfloat16)
    codes = torch.zeros(1, 128).to(FP8)
    scales = torch.ones(1, 1)
    mlp.offer_token_fp8(rows, codes, scales)
    offered = mlp._input_fp8(rows)
    assert offered[0] is codes and offered[1] is scales
    assert mlp._input_fp8(rows) is None
    mlp.offer_token_fp8(rows, codes, scales)
    assert mlp._input_fp8(rows.clone()) is None
    assert mlp._input_fp8(rows) is None


Q = "model.layers.0.self_attn.q_proj"
ROUTER = "model.layers.1.mlp.gate"


@pytest.mark.parametrize(
    "name, edit, error, match",
    [
        (
            "router",
            lambda t: t.__setitem__(f"{ROUTER}.weight", torch.zeros(8, 128, dtype=FP8)),
            TypeError,
            "router",
        ),
        (
            "scale",
            lambda t: t.__setitem__(f"{Q}.weight_scale", t[f"{Q}.weight_scale"] * 0),
            ValueError,
            "finite and positive",
        ),
        (
            "pair",
            lambda t: t.pop(f"{Q}.weight_scale"),
            ValueError,
            "without their weight or scale",
        ),
        ("twice", lambda t: None, ValueError, f"Duplicate quantized tensor: {Q}"),
    ],
    ids=["router", "scale", "pair", "twice"],
)
def test_fp8_loader_refusals(server, name, edit, error, match):
    """One case per rule: an FP8 router, a scale that is not finite and
    positive, a weight without its scale, and a quantized tensor twice."""
    server("flashinfer_trtllm")
    tensors, _ = fp8_checkpoint()
    edit(tensors)
    items = list(tensors.items())
    if name == "twice":
        items.insert(0, (f"{Q}.weight", tensors[f"{Q}.weight"]))
    with pytest.raises(error, match=match):
        _model(ct()).load_weights(items)


@pytest.fixture(scope="module")
def fp8_checkpoints(tmp_path_factory):
    """The tiny FP8 checkpoint on disk, and one in a refused variant."""
    root = tmp_path_factory.mktemp("fp8")
    paths = {}
    for label, raw in (("fp8", RAW), ("tensor", edited((*W, "strategy"), "tensor"))):
        path = write_tiny_checkpoint(root / label)
        config = json.loads((path / "config.json").read_text())
        config["quantization_config"] = raw
        (path / "config.json").write_text(json.dumps(config))
        save_file(fp8_checkpoint()[0], str(path / "model.safetensors"))
        paths[label] = path
    return paths


BEAM_GEOMETRY = dict(
    hidden_size=5120,
    moe_intermediate_size=2560,
    num_experts=256,
    num_experts_per_tok=8,
    n_group=8,
    topk_group=4,
)
# layout: (mapping, ispp planned and resident, local experts, EP size)
LAYOUTS = {
    "tp8": (dict(world_size=8), 320, 256, 1),
    "tp4": (dict(world_size=4), 640, 256, 1),
    "ep8": (dict(world_size=8, moe_ep_size=8), 2560, 32, 8),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("fmt", ["fp8_channel", "nvfp4"])
def test_plan_at_the_beam_geometry(server, fmt, layout):
    """Rank 0's selected kernel: FP8 per-channel or NVFP4 SiLU experts on
    precomputed top-k at TP8 (ispp 320), TP4 (640) and EP8 (32 whole experts of
    2,560), none padded."""
    server("flashinfer_trtllm")
    mapping, ispp, local, _ = LAYOUTS[layout]
    quant_config, policy, routed = {
        "fp8_channel": (ct(), beam.BEAM_MOE_FP8_POLICY, FP8_ROUTED),
        "nvfp4": (NVFP4, beam.BEAM_MOE_NVFP4_POLICY, NVFP4_ROUTED),
    }[fmt]
    moe = beam.BeamVendorMoE(
        _config(**BEAM_GEOMETRY),
        Mapping(rank=0, **mapping),
        1,
        "model.layers.1.mlp",
        quant_config,
    )
    assert moe.experts.plan["apply_kernel_name"] == routed
    assert moe.experts.plan["precision_policy"] == policy
    assert moe.experts.num_local_experts == local
    assert moe.experts.intermediate_size // moe.experts.tp_size == ispp


@pytest.mark.parametrize("rank", [0, 1])
def test_ep_ranks_load_their_own_fp8_experts(server, rank):
    """Under EP each rank holds its whole experts' codes, bit for bit, and their
    channel scales exactly as FP32; the other rank's experts are skipped."""
    server("flashinfer_trtllm")
    tensors, _ = fp8_checkpoint(torch.bfloat16)
    model = _model(ct(), mapping=Mapping(rank=rank, world_size=2, moe_ep_size=2))
    experts = model.model.layers[1].mlp.experts
    assert experts.num_local_experts == 4
    model.load_weights(list(tensors.items()))
    prefix = "model.layers.1.mlp.experts"
    for local in range(4):
        e = rank * 4 + local
        w13 = torch.cat(
            [tensors[f"{prefix}.{e}.{p}.weight"] for p in ("gate_proj", "up_proj")]
        )
        s13 = torch.cat(
            [
                tensors[f"{prefix}.{e}.{p}.weight_scale"]
                for p in ("gate_proj", "up_proj")
            ]
        )
        assert torch.equal(
            experts.w13_weight[local].view(torch.uint8), w13.view(torch.uint8)
        )
        assert torch.equal(experts.w13_weight_scale[local], s13.float().reshape(-1))
        assert torch.equal(
            experts.w2_weight[local].view(torch.uint8),
            tensors[f"{prefix}.{e}.down_proj.weight"].view(torch.uint8),
        )
        assert torch.equal(
            experts.w2_weight_scale[local],
            tensors[f"{prefix}.{e}.down_proj.weight_scale"].float().reshape(-1),
        )


# ---------------------------------------------------------------- NVFP4 checkpoints
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# The provider layout on the tiny Beam: experts only, as ModelOpt globs it.
EXCLUDED = [
    "lm_head",
    "model.embed_tokens",
    "model.layers.0*",
    "model.layers.1.mlp.gate",
    "model.layers.1.self_attn*",
]

HF_QUANT_CONFIG = {
    "producer": {"name": "modelopt", "version": "0.47.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "exclude_modules": EXCLUDED,
    },
}

NVFP4 = Nvfp4Config.from_config(HF_QUANT_CONFIG)


def f32(value):
    return torch.tensor(value, dtype=torch.float32)


def quantize_nvfp4(weight):
    rows, columns = weight.shape
    global_scale = weight.float().abs().max() / (448.0 * 6.0)
    blocks = weight.float().reshape(rows, columns // 16, 16)
    block_scale = (blocks.abs().amax(-1) / 6.0 / global_scale).to(FP8)
    unit = (block_scale.float() * global_scale).unsqueeze(-1)
    values = (blocks / unit).reshape(rows, columns)
    index = (values.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    codes = (index | ((values < 0).long() << 3)).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, block_scale, global_scale.reshape(())


def nvfp4_checkpoint():
    """The tiny Beam's tensors with ModelOpt NVFP4 experts, the rest BF16 and
    the router FP32; every expert's input scale 0.5."""
    tensors = {}
    for name, weight in tiny_weights(tiny_config()).items():
        if ".mlp.experts." not in name:
            tensors[name] = weight
            continue
        module = name.removesuffix(".weight")
        packed, block_scale, global_scale = quantize_nvfp4(weight)
        tensors[name] = packed
        tensors[f"{module}.weight_scale"] = block_scale
        tensors[f"{module}.weight_scale_2"] = global_scale
        tensors[f"{module}.input_scale"] = f32(0.5)
    return tensors


def test_excluded_experts_are_refused(server):
    server("flashinfer_trtllm")
    config = Nvfp4Config(exclude_modules=[*EXCLUDED, "model.layers.1.mlp.experts*"])
    with pytest.raises(ValueError, match="requires NVFP4 SiLU experts"):
        _model(config)


def _assert_experts_hold(experts, tensors):
    """Gate and up merge into w13 with their own global scales; down fills w2."""
    for expert in range(8):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        gate, up, down = (
            f"{prefix}.{p}" for p in ("gate_proj", "up_proj", "down_proj")
        )
        assert torch.equal(experts.w13_weight[expert, :64], tensors[f"{gate}.weight"])
        assert torch.equal(experts.w13_weight[expert, 64:], tensors[f"{up}.weight"])
        for rows, name in ((slice(0, 64), gate), (slice(64, 128), up)):
            assert torch.equal(
                experts.w13_weight_scale[expert, rows].view(torch.uint8),
                tensors[f"{name}.weight_scale"].view(torch.uint8),
            )
        assert torch.equal(
            experts.w13_weight_scale_2[expert],
            torch.stack(
                (tensors[f"{gate}.weight_scale_2"], tensors[f"{up}.weight_scale_2"])
            ),
        )
        assert torch.equal(experts.w2_weight[expert], tensors[f"{down}.weight"])
        assert torch.equal(
            experts.w2_weight_scale_2[expert], tensors[f"{down}.weight_scale_2"]
        )


def test_nvfp4_expert_tensors_load_unchanged(server):
    server("flashinfer_trtllm")
    tensors = nvfp4_checkpoint()
    model, reference = _model(NVFP4), _model()
    assert model.load_weights(list(tensors.items())) == tensors.keys()
    experts = model.model.layers[1].mlp.experts
    _assert_experts_hold(experts, tensors)
    assert experts.w13_input_scale.item() == 0.5
    # w2's input scale is stored per expert; scalar-per-expert loads accumulate
    # the running layer maximum into every slot.
    assert torch.equal(
        experts.w2_input_scale, torch.full_like(experts.w2_input_scale, 0.5)
    )
    # Every other tensor is the BF16 checkpoint's.
    reference.load_weights([(n, t) for n, t in tiny_weights(tiny_config()).items()])
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if ".experts." not in name:
            assert torch.equal(parameter, expected[name]), name


SCALE = "model.layers.1.mlp.experts.5.down_proj"


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda t: t.__setitem__(f"{SCALE}.weight_scale_2", f32(0.0)), "positive"),
        (
            lambda t: t.update(
                {
                    name: f32(0.0)
                    for name in t
                    if name.endswith(".input_scale")
                    and name.split(".")[-2] in ("gate_proj", "up_proj")
                }
            ),
            r"0 in every expert: \['model.layers.1.mlp.experts.\*.\{gate,up\}_proj",
        ),
    ],
    ids=["scale", "input-scales"],
)
def test_nvfp4_loader_refusals(server, edit, match):
    """One case per rule: a scale that is not positive, and an input scale of
    0 in every expert."""
    server("flashinfer_trtllm")
    tensors = nvfp4_checkpoint()
    edit(tensors)
    with pytest.raises(ValueError, match=match):
        _model(NVFP4).load_weights(list(tensors.items()))


def test_quantized_attention_is_refused_where_the_config_quantizes_it(server):
    server("flashinfer_trtllm")
    config = Nvfp4Config(exclude_modules=["lm_head", "model.layers.*.mlp.gate"])
    tensors = nvfp4_checkpoint()
    q = "model.layers.0.self_attn.q_proj"
    packed, block, global_scale = quantize_nvfp4(tensors[f"{q}.weight"])
    tensors.update({f"{q}.weight": packed, f"{q}.weight_scale_2": global_scale})
    with pytest.raises(ValueError, match="NVFP4 on its experts only; its attention"):
        _model(config).load_weights(list(tensors.items()))


def test_an_input_scale_of_zero_loads_beside_others(server):
    """ModelOpt writes 0 for an expert no calibration token reached: the MoE
    keeps each layer's largest input scale of w13 and of w2."""
    server("flashinfer_trtllm")
    tensors = nvfp4_checkpoint()
    tensors["model.layers.1.mlp.experts.2.gate_proj.input_scale"] = f32(0.0)
    tensors["model.layers.1.mlp.experts.6.down_proj.input_scale"] = f32(0.0)
    tensors["model.layers.1.mlp.experts.4.up_proj.input_scale"] = f32(0.75)
    model = _model(NVFP4)
    model.load_weights(list(tensors.items()))
    experts = model.model.layers[1].mlp.experts
    assert experts.w13_input_scale.item() == 0.75
    assert torch.equal(
        experts.w2_input_scale, torch.full_like(experts.w2_input_scale, 0.5)
    )


def test_a_quantized_launch_detects_the_format_and_keeps_the_moe_default(
    fp8_checkpoints, tmp_path, caplog, monkeypatch
):
    """Beam's vendor MoE default applies to an admitted FP8 or NVFP4 checkpoint
    (its experts run on the vendor MoE), also at TP8 (ispp 320) where the BF16
    kernels need 128 (the FP8 and NVFP4 kernels take it), and stays off for a
    refused one."""
    import tokenspeed_kernel.ops.moe.flashinfer as moe_flashinfer

    monkeypatch.setattr(moe_flashinfer, "TRTLLM_UNQUANT_ISPP_ALIGNMENT", 128)
    caplog.set_level(logging.INFO, logger=model_config_module.__name__)
    nvfp4 = write_tiny_checkpoint(tmp_path / "nvfp4")
    (nvfp4 / "hf_quant_config.json").write_text(json.dumps(HF_QUANT_CONFIG))
    save_file(nvfp4_checkpoint(), str(nvfp4 / "model.safetensors"))
    beam_width = json.dumps({"moe_intermediate_size": 2560})
    for path, quantization in (
        (fp8_checkpoints["fp8"], "compressed-tensors"),
        (nvfp4, "nvfp4"),
    ):
        args = _server_args(path, world_size=8)
        assert _model_config(args, beam_width).quantization == quantization
        assert args.moe_backend == "flashinfer_trtllm"
    refused = _server_args(fp8_checkpoints["tensor"])
    _model_config(refused)
    assert refused.moe_backend == "auto"
    assert any(
        "not applied, left auto: quantization compressed-tensors" in r.getMessage()
        for r in caplog.records
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
