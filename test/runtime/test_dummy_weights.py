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

"""Dummy loading gives every parameter its value before post-processing reads
it, tells an EAGLE3 draft its embedding is initialized, and derives a wrapped
text model's weights."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

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


class _DerivedCopy(nn.Module):
    """A model whose post-processing derives a copy of its weight.

    The copy stands in for a prepared GEMV weight.
    """

    def __init__(self) -> None:
        super().__init__()
        # NaN stands in for uninitialized memory.
        self.weight = nn.Parameter(
            torch.full((4, 4), float("nan")), requires_grad=False
        )
        self.prepared: torch.Tensor | None = None

    def process_weights_after_loading(self, module: nn.Module) -> None:
        self.prepared = self.weight.detach().t().contiguous()


def test_the_dummy_loader_derives_copies_from_the_final_weights(monkeypatch):
    model = _DerivedCopy()
    monkeypatch.setattr(loader, "_initialize_model", lambda *args: model)
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert not torch.isnan(model.weight).any()
    assert torch.equal(model.prepared, model.weight.t())


def test_a_dummy_initializer_sets_its_parameter_before_post_processing(
    monkeypatch,
):
    seen = []

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(
                torch.full((4,), float("nan")), requires_grad=False
            )
            self.scale.dummy_initializer = lambda param, generator: param.data.fill_(
                1.0
            )

        def post_load_weights(self) -> None:
            seen.append(self.scale.clone())

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: _Model())
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert len(seen) == 1 and torch.all(seen[0] == 1.0)


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


def _dflash2_test_config(attention_mode: str):
    from transformers import PretrainedConfig

    return PretrainedConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=1024,
        decode_monokernel=False,
        dflash_config={
            "attention_mode": attention_mode,
            "block_size": 16,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": [0],
        },
    )


def test_the_dummy_loader_absorbs_dflash2_mla_kv_b_proj(monkeypatch):
    from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

    config = _dflash2_test_config("mla")
    with torch.device("cpu"):
        model = DFlash2DraftModel(config, Mapping(rank=0, world_size=1))
    for layer in model.layers:
        assert layer.self_attn.w_kc is None
        assert layer.self_attn.w_vc is None

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: model)
    loaded = loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device_config=SimpleNamespace(device="cpu"),
    )

    initial_ptrs: list[tuple[int, int]] = []
    for layer in loaded.layers:
        attn = layer.self_attn
        assert attn.w_kc is not None
        assert attn.w_vc is not None
        kc, vc = attn.kv_b_proj.weight.unflatten(0, (2, 64)).split([32, 32], dim=1)
        torch.testing.assert_close(attn.w_kc, kc)
        torch.testing.assert_close(attn.w_vc, vc.transpose(1, 2))
        initial_ptrs.append((attn.w_kc.data_ptr(), attn.w_vc.data_ptr()))

    # Re-running post_load_weights (e.g. after a live weight update or real
    # load) is idempotent and updates absorbed storage in place.
    for layer in loaded.layers:
        layer.self_attn.kv_b_proj.weight.data.mul_(2.0)
    loaded.post_load_weights()
    for layer, (kc_ptr, vc_ptr) in zip(loaded.layers, initial_ptrs, strict=True):
        attn = layer.self_attn
        assert attn.w_kc.data_ptr() == kc_ptr
        assert attn.w_vc.data_ptr() == vc_ptr
        kc_updated, vc_updated = attn.kv_b_proj.weight.unflatten(0, (2, 64)).split(
            [32, 32], dim=1
        )
        torch.testing.assert_close(attn.w_kc, kc_updated)
        torch.testing.assert_close(attn.w_vc, vc_updated.transpose(1, 2))


def test_the_dummy_loader_skips_mla_absorption_for_gqa_dflash2(monkeypatch):
    from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

    config = _dflash2_test_config("gqa")
    with torch.device("cpu"):
        model = DFlash2DraftModel(config, Mapping(rank=0, world_size=1))

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: model)
    loaded = loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert len(loaded.layers) == 2
