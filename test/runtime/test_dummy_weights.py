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

"""Dummy loading gives integer parameters valid values before post-processing
reads them, tells an EAGLE3 draft its embedding is initialized, and derives a
wrapped text model's weights."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k25_config import KimiK25Config  # noqa: E402
from tokenspeed.runtime.configs.load_config import LoadConfig  # noqa: E402
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.model_loader import loader  # noqa: E402
from tokenspeed.runtime.model_loader.weight_utils import (  # noqa: E402
    initialize_dummy_integer_weights,
)
from tokenspeed.runtime.models import kimi_k25  # noqa: E402


class _QuantizedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 0xFF stands in for uninitialized memory; as an E8M0 scale it is NaN.
        self.weight = nn.Parameter(
            torch.full((4, 4), 0xFF, dtype=torch.uint8), requires_grad=False
        )
        self.bias = nn.Parameter(torch.ones(4))
        self.register_buffer("order", torch.arange(4))


def test_integer_parameters_are_zeroed_and_everything_else_kept():
    layer = _QuantizedLinear()
    initialize_dummy_integer_weights(layer)
    assert torch.all(layer.weight == 0)
    assert torch.all(layer.bias == 1)
    assert torch.equal(layer.order, torch.arange(4))


def test_the_dummy_loader_fills_integers_before_post_processing(monkeypatch):
    seen = []

    class _Model(_QuantizedLinear):
        def post_load_weights(self) -> None:
            seen.append(self.weight.clone())

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: _Model())
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert len(seen) == 1 and torch.all(seen[0] == 0)


def test_the_dummy_loader_marks_the_draft_embedding_initialized(monkeypatch):
    marked = []

    class _Draft(_QuantizedLinear):
        def mark_embedding_initialized(self) -> None:
            marked.append(True)

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: _Draft())
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert marked == [True]


def test_the_dummy_loader_derives_the_kimi_k25_text_model(monkeypatch):
    derived = []

    # The DeepSeek-V3 text model needs process groups; record its derivation.
    class _TextModel(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def post_load_weights(self) -> None:
            derived.append(True)

    monkeypatch.setattr(kimi_k25, "DeepseekV3ForCausalLM", _TextModel)
    model = kimi_k25.KimiK25ForConditionalGeneration(
        KimiK25Config(),
        mapping=Mapping(rank=0, world_size=1),
        is_multimodal_active=False,
    )
    monkeypatch.setattr(loader, "_initialize_model", lambda *args: model)
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert derived == [True]


def _check_mxfp4_dummy(weight, scale, fan_in):
    """Decode synthetic packed weights with a CPU E2M1 reference."""
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 7).long()] * torch.where(codes < 8, 1, -1)
    scales = torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)
    decoded = values * scales
    assert torch.isfinite(decoded).all()
    assert (decoded > 0).any() and (decoded < 0).any()
    # A full projection has bounded variance, including TP partitions.
    assert 0.02 < decoded.square().mean().item() * fan_in < 1.2


def test_mxfp4_dummy_dense_and_moe_weights_have_finite_nonzero_values():
    from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LinearMethod
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.mxfp4 import create_mxfp4_weight_pair
    from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config

    dense = nn.Module()
    Mxfp4LinearMethod(Mxfp4Config()).create_weights(
        dense,
        input_size_per_partition=128,
        output_partition_sizes=[64, 64],
        input_size=256,
        output_size=128,
        params_dtype=torch.bfloat16,
    )
    moe = nn.Module()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=256,
        activation="silu",
        tp_rank=0,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
    )
    create_mxfp4_weight_pair(spec, moe, with_bias=False, solution=None)
    initialize_dummy_integer_weights(dense)
    initialize_dummy_integer_weights(moe)
    _check_mxfp4_dummy(dense.weight, dense.weight_scale, 256)
    _check_mxfp4_dummy(moe.w13_weight, moe.w13_weight_scale, 128)
    _check_mxfp4_dummy(moe.w2_weight, moe.w2_weight_scale, 256)
    previous = dense.weight.clone()
    initialize_dummy_integer_weights(dense)
    assert torch.equal(previous, dense.weight)


def test_floating_dummy_initialization_preserves_custom_channel_scales():
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.fp8_channel import (
        create_fp8_channel_weight_pair,
    )
    from tokenspeed.runtime.model_loader.weight_utils import initialize_dummy_weights

    layer = nn.Module()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=256,
        activation="silu",
        tp_rank=0,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
    )
    create_fp8_channel_weight_pair(spec, layer)
    initialize_dummy_integer_weights(layer)
    initialize_dummy_weights(layer)
    assert torch.all(layer.w13_weight_scale == 1)
    assert torch.all(layer.w2_weight_scale == 1)
    assert torch.isfinite(layer.w13_weight.float()).all()


@pytest.mark.parametrize("tp_rank", [0, 7])
def test_dummy_output_column_copy_matches_replicated_input_shards(monkeypatch, tp_rank):
    from tokenspeed.runtime.layers.moe.down_columns import MoEDownColumns
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.mxfp4 import create_mxfp4_weight_pair

    experts = nn.Module()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=512,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=8,
        ep_rank=0,
        ep_size=1,
    )
    create_mxfp4_weight_pair(spec, experts, with_bias=False, solution=None)
    columns = MoEDownColumns(4, 128, 512, tp_rank=tp_rank, tp_size=8)
    initialize_dummy_integer_weights(experts)
    initialize_dummy_integer_weights(columns)
    from tokenspeed.runtime.models.muse_spark import MuseSparkForConditionalGeneration
    from tokenspeed.runtime.utils.env import global_server_args_dict

    class MinimalMuse(MuseSparkForConditionalGeneration):
        def __init__(self):
            nn.Module.__init__(self)
            self._weight_update_active = False
            self.model = nn.Module()
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = experts
            layer.mlp.down_columns = columns
            self.model.layers = nn.ModuleList([layer])

    model = MinimalMuse()
    monkeypatch.setitem(global_server_args_dict, "simulate", True)
    model.post_load_weights()
    for source, copy in (
        (experts.w2_weight, columns.weight),
        (experts.w2_weight_scale, columns.weight_scale),
    ):
        full = torch.cat([source] * 8, dim=-1)
        assert torch.equal(copy, full[:, tp_rank * 16 : (tp_rank + 1) * 16])
    _check_mxfp4_dummy(columns.weight, columns.weight_scale, 512)


def test_dummy_loader_derives_cached_weights_from_final_initialized_values(monkeypatch):
    class _Derived(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.full((32, 32), float("nan")))
            self.register_buffer("absorbed", torch.zeros(32, 32))

        def post_load_weights(self):
            self.absorbed.copy_(self.weight.square())

    model = _Derived()
    monkeypatch.setattr(loader, "_initialize_model", lambda *args: model)
    with torch.no_grad():
        loader.DummyModelLoader(LoadConfig()).load_model(
            model_config=SimpleNamespace(dtype=torch.float32),
            device_config=SimpleNamespace(device="cpu"),
        )
    assert torch.isfinite(model.absorbed).all()
    assert torch.equal(model.absorbed, model.weight.square())


@pytest.mark.parametrize("mxfp8", [False, True])
def test_fp8_block_dummy_scales_decode_to_one(mxfp8):
    from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights import create_layer_weights
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config
    from tokenspeed.runtime.model_loader.weight_utils import initialize_dummy_weights

    config = (Mxfp8Config if mxfp8 else Fp8Config)(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
    )
    layer = nn.Module()
    Fp8LinearMethod(config).create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[64],
        input_size=128,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args: None,
    )
    initialize_dummy_integer_weights(layer)
    initialize_dummy_weights(layer)
    if layer.weight_scale_inv.dtype == torch.uint8:
        assert layer.weight_scale_inv.eq(127).all()
    else:
        assert layer.weight_scale_inv.eq(1).all()

    class _ExpertWeights(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_config = config

    moe = _ExpertWeights()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=256,
        activation="silu",
        tp_rank=0,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
    )
    create_layer_weights(spec, moe, "fp8", config, with_bias=False, solution=None)
    initialize_dummy_integer_weights(moe)
    initialize_dummy_weights(moe)
    for scale in (moe.w13_weight_scale_inv, moe.w2_weight_scale_inv):
        decoded = (
            torch.exp2(scale.float() - 127) if scale.dtype == torch.uint8 else scale
        )
        assert decoded.eq(1).all()
    assert torch.isfinite(moe.w13_weight.float()).all()
    assert torch.isfinite(moe.w2_weight.float()).all()


@pytest.mark.parametrize("shape", [(), (0,), (7, 9, 23)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_low_precision_dummy_fills_strided_views_reproducibly(shape, device):
    from tokenspeed.runtime.model_loader.weight_utils import _fill_low_precision_dummy

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    backing = torch.full((*shape, 2), 7, dtype=torch.float8_e4m3fn, device=device)
    view = backing[..., 0]
    generator = torch.Generator(device=device).manual_seed(9)
    _fill_low_precision_dummy(view, generator, low=-1, high=1, chunk_elements=64)
    values = view.float()
    assert torch.isfinite(values).all()
    assert values.abs().le(1).all()
    assert backing[..., 1].float().eq(7).all()
    if view.numel() > 1:
        assert (values < 0).any() and (values > 0).any()
    generator.manual_seed(9)
    _fill_low_precision_dummy(view, generator, low=-1, high=1, chunk_elements=64)
    assert torch.equal(values, view.float())


def test_low_precision_dummy_scratch_does_not_scale_with_the_weight():
    from tokenspeed.runtime.model_loader.weight_utils import _fill_low_precision_dummy

    if not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    weight = torch.empty((8192, 8192), dtype=torch.float8_e4m3fn, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(12)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _fill_low_precision_dummy(
        weight, generator, low=-1, high=1, chunk_elements=1024 * 1024
    )
    torch.cuda.synchronize()
    # FP16 scratch is 2 MiB here, independent of this 64 MiB weight. Allow
    # allocator rounding/workspace, but not a weight-sized FP16 conversion.
    assert torch.cuda.max_memory_allocated() - before < 8 * 1024**2
