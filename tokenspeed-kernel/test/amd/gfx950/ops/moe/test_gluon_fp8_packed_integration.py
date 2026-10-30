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

"""CPU planning, load lifecycle, and call-boundary checks for packed FP8 MoE."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.moe.gluon as fp8
import torch

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess",
    reason="tokenspeed-kernel-amd is required for packed FP8 MoE integration tests",
)

from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (  # noqa: E402
    _copy_packed_fp8_shard,
    pack_fp8_weight_n16k32,
)


def _plan(**changes: object) -> dict:
    from tokenspeed_kernel.ops.moe import moe_plan

    arguments = {
        "weight_dtype": "fp8",
        "input_dtype": torch.bfloat16,
        "activation": "swiglu",
        "routing_mode": "precomputed_topk",
        "a2a_backend": "none",
        "ep_size": 4,
        "ispp": 2048,
        "hidden": 4096,
        "swiglu_form": "standard",
        "activation_clamped": True,
        "expert_id_repeats": False,
        "fp8_scale_block_shape": (128, 128),
        "internal_activation_dtype": "input",
        "fast_math": True,
        "combine_order": "rank",
        "features": frozenset(),
    }
    arguments.update(changes)
    return moe_plan(**arguments)


@pytest.mark.parametrize("ep_size,ispp", [(1, 512), (4, 2048)])
def test_packed_fp8_plan_selects_packed_preprocessor(ep_size: int, ispp: int) -> None:
    plan = _plan(ep_size=ep_size, ispp=ispp)
    assert plan["apply_kernel_name"] == (
        f"gluon_fp8_packed_ep{ep_size}_precomputed_moe_apply"
    )
    assert plan["weight_preprocessor"] is fp8.gluon_fp8_packed_moe_weights
    assert plan["weight_preprocessor"].__module__ == (
        "tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess"
    )
    assert plan["supports_precomputed_topk"]
    assert not plan["supports_deferred_finalize"]


@pytest.mark.parametrize(
    "change",
    [
        {"hidden": 512},
        {"ispp": 1024},
        {"ep_size": 4, "ispp": 512},
        {"ep_size": 1, "ispp": 2048},
        {"activation_clamped": False},
        {"swiglu_form": "generalized"},
        {"activation": "silu", "swiglu_form": None},
        {"expert_id_repeats": True},
    ],
)
def test_packed_fp8_plan_preserves_other_gluon_fp8_cases(change: dict) -> None:
    assert _plan(**change)["apply_kernel_name"] == (
        "gluon_fp8_block_precomputed_moe_apply"
    )


def _small_fp8_weight(modulus: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.arange(2 * 32 * 32, dtype=torch.int32)
        .remainder(modulus)
        .to(torch.uint8)
        .reshape(2, 32, 32)
        .view(torch.float8_e4m3fn),
        requires_grad=False,
    )


def _small_layer() -> torch.nn.Module:
    from tokenspeed.runtime.layers.moe.expert import MoELayer

    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.w13_weight = _small_fp8_weight(128)
    layer.w2_weight = _small_fp8_weight(64)
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones(2, 2, 1), requires_grad=False
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones(2, 1, 1), requires_grad=False
    )

    layer._weights_processed = False
    layer._quant_kind = "fp8"
    layer._moe_backend_state = None
    layer.plan = _plan()
    return layer


def _small_glm_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.nn.Module, torch.nn.Module, str]:
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.loaders import (
        make_group_scale_loader,
        make_weight_loader,
    )
    from tokenspeed.runtime.models import glm53_flash

    layer = _small_layer()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=2,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="swiglu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        prefix="model.layers.0.mlp",
        a2a_backend="none",
    )
    layer.w13_weight.weight_loader = make_weight_loader(spec)
    layer.w2_weight.weight_loader = make_weight_loader(spec)
    layer.w13_weight_scale_inv.weight_loader = make_group_scale_loader(spec)
    layer.w2_weight_scale_inv.weight_loader = make_group_scale_loader(spec)

    class TinyMoE(torch.nn.Module):
        def __init__(self, experts: torch.nn.Module) -> None:
            super().__init__()
            self.experts = experts

    monkeypatch.setattr(glm53_flash, "Glm53FlashMoE", TinyMoE)
    block = torch.nn.Module()
    block.self_attn = SimpleNamespace(fuse_conv_weights=lambda: None)
    block.mlp = TinyMoE(layer)
    language = glm53_flash.Glm53FlashForCausalLM.__new__(
        glm53_flash.Glm53FlashForCausalLM
    )
    torch.nn.Module.__init__(language)
    language.config = SimpleNamespace(n_routed_experts=2, num_hidden_layers=1)
    language.mapping = SimpleNamespace(moe=SimpleNamespace(ep_rank=0, ep_size=1))
    language.model = torch.nn.Module()
    language.model.layers = torch.nn.ModuleList([block])
    wrapper = glm53_flash.Glm53FlashForConditionalGeneration.__new__(
        glm53_flash.Glm53FlashForConditionalGeneration
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = language
    wrapper.vision = None
    name = "model.language_model.model.layers.0.mlp.experts"
    return wrapper, layer, name


def _shard(value: int, shape: tuple[int, int]) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float8_e4m3fn)


def _checkpoint(name: str, base: int) -> list[tuple[str, torch.Tensor]]:
    projections = (
        ("gate_proj", (16, 32)),
        ("up_proj", (16, 32)),
        ("down_proj", (32, 32)),
    )
    return [
        (
            f"{name}.{expert}.{projection}.weight",
            _shard(base + expert * 3 + index, shape),
        )
        for expert in range(2)
        for index, (projection, shape) in enumerate(projections)
    ] + [
        (
            f"{name}.{expert}.{projection}.weight_scale_inv",
            torch.full((1, 1), float(base + 10 + expert * 3 + index)),
        )
        for expert in range(2)
        for index, (projection, _) in enumerate(projections)
    ]


def _expected_weights(base: int) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up = torch.stack(
        [
            torch.cat([_shard(base + expert * 3 + index, (16, 32)) for index in (0, 1)])
            for expert in range(2)
        ]
    )
    down = torch.stack([_shard(base + expert * 3 + 2, (32, 32)) for expert in range(2)])
    return gate_up, down


def _assert_packed_weights(
    layer: torch.nn.Module,
    addresses: tuple[int, int],
    expected13: torch.Tensor,
    expected2: torch.Tensor,
) -> None:
    assert layer._moe_backend_state is None
    assert (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr()) == addresses
    for actual, expected in (
        (layer.w13_weight, expected13),
        (layer.w2_weight, expected2),
    ):
        assert torch.equal(
            actual.view(torch.uint8), pack_fp8_weight_n16k32(expected).view(torch.uint8)
        )


def test_packed_fp8_glm53_wrapper_stays_packed_after_each_update_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.models.base.weight_update import weight_update_session

    wrapper, layer, name = _small_glm_wrapper(monkeypatch)
    wrapper.load_weights(_checkpoint(name, 1))
    addresses = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    expected13, expected2 = _expected_weights(1)
    _assert_packed_weights(layer, addresses, expected13, expected2)
    # Repeated post-load preparation must not interpret packed bytes as raw FP8.
    layer.process_weights_after_loading(layer)
    _assert_packed_weights(layer, addresses, expected13, expected2)

    # The exported multimodal wrapper does not enter the inner LM's session:
    # each partial load must return with packed layout ready for inference.
    with weight_update_session([wrapper], require_complete=False):
        gate = _shard(13, (16, 32))
        wrapper.load_weights([(f"{name}.1.gate_proj.weight", gate)])
        expected13[1, :16].copy_(gate)
        _assert_packed_weights(layer, addresses, expected13, expected2)

        down = _shard(17, (32, 32))
        wrapper.load_weights([(f"{name}.0.down_proj.weight", down)])
        expected2[0].copy_(down)
        _assert_packed_weights(layer, addresses, expected13, expected2)

        wrapper.load_weights(
            [(f"{name}.1.gate_proj.weight_scale_inv", torch.full((1, 1), 7.0))]
        )
        _assert_packed_weights(layer, addresses, expected13, expected2)
        assert layer.w13_weight_scale_inv[1].tolist() == [[7.0], [15.0]]


def test_packed_fp8_reload_rejects_tile_aligned_partial_input_columns() -> None:
    linear = torch.arange(32 * 64).remainder(127).to(torch.uint8).reshape(1, 32, 64)
    packed = pack_fp8_weight_n16k32(linear.view(torch.float8_e4m3fn))
    original = packed.view(torch.uint8).clone()
    source = torch.ones(32, 32, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="complete local FP8 rows"):
        _copy_packed_fp8_shard(packed[0, :, :32], source)
    assert torch.equal(packed.view(torch.uint8), original)


def test_packed_fp8_preprocess_packs_plain_tensor_weights() -> None:
    class PlainTensorModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w13_weight = _small_fp8_weight(128).detach()
            self.w2_weight = _small_fp8_weight(64).detach()
            self.w13_weight_scale_inv = torch.ones(2, 2, 1)
            self.w2_weight_scale_inv = torch.ones(2, 1, 1)
            # Synthetic weights are already loaded, like the benchmark inputs.
            self.w13_weight.weight_loader = lambda *args, **kwargs: None
            self.w2_weight.weight_loader = lambda *args, **kwargs: None

    module = PlainTensorModule()
    original13, original2 = module.w13_weight.clone(), module.w2_weight.clone()
    fp8.gluon_fp8_packed_moe_weights({}, module)
    assert torch.equal(
        module.w13_weight.view(torch.uint8),
        pack_fp8_weight_n16k32(original13).view(torch.uint8),
    )
    assert torch.equal(
        module.w2_weight.view(torch.uint8),
        pack_fp8_weight_n16k32(original2).view(torch.uint8),
    )
    for weight in (module.w13_weight, module.w2_weight):
        packed = weight.view(torch.uint8).clone()
        weight.weight_loader(weight, torch.empty(0))
        assert torch.equal(weight.view(torch.uint8), packed)


def test_linear_fp8_preprocessor_receives_original_checkpoint_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, layer, _ = _small_glm_wrapper(monkeypatch)
    original13, original2 = layer.w13_weight.clone(), layer.w2_weight.clone()

    observed: list[torch.Tensor] = []

    def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        observed.append(weight.clone())
        return weight.float()

    monkeypatch.setattr(fp8, "gluon_fp8_block_dequantize", dequantize)
    fp8.gluon_fp8_moe_weights(layer.plan, layer)
    assert torch.equal(observed[0].view(torch.uint8), original13.view(torch.uint8))
    assert torch.equal(observed[1].view(torch.uint8), original2.view(torch.uint8))
    torch.testing.assert_close(
        layer.w13_weight_prefill_bf16,
        original13.float(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        layer.w2_weight_prefill_bf16, original2.float(), rtol=0, atol=0, equal_nan=True
    )


def test_packed_fp8_reload_slices_tensor_parallel_shards_before_packing() -> None:
    from functools import partial

    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.loaders import make_weight_loader

    spec = MoELayerSpec(8, 2, 2, 64, 64, "swiglu", 1, 2, 0, 1)
    loader = partial(make_weight_loader(spec), weight_copy=_copy_packed_fp8_shard)
    gate_up = torch.nn.Parameter(
        torch.zeros(2, 64, 64, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    down = torch.nn.Parameter(
        torch.zeros(2, 64, 32, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    addresses = (gate_up.data_ptr(), down.data_ptr())
    source = torch.arange(64 * 64).remainder(127).to(torch.uint8).reshape(64, 64)
    source = source.view(torch.float8_e4m3fn)
    expected = torch.zeros_like(gate_up)
    for shard_id, offset in [("w1", 0), ("w3", 32)]:
        loader(gate_up, source, shard_id=shard_id, local_expert_id=1)
        expected[1, offset : offset + 32].copy_(source[32:])
        assert torch.equal(
            gate_up.view(torch.uint8),
            pack_fp8_weight_n16k32(expected).view(torch.uint8),
        )
    loader(down, source, shard_id="w2", local_expert_id=1)
    expected_down = torch.zeros_like(down)
    expected_down[1].copy_(source[:, 32:])
    assert torch.equal(
        down.view(torch.uint8), pack_fp8_weight_n16k32(expected_down).view(torch.uint8)
    )
    assert addresses == (gate_up.data_ptr(), down.data_ptr())


def test_packed_fp8_reload_packs_after_online_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.layers.moe.weights import loaders

    quantized = _small_fp8_weight(127)[0].detach()
    destination = torch.empty_like(quantized)
    scales = torch.empty(1, 1)
    addresses = (destination.data_ptr(), scales.data_ptr())
    source = torch.ones_like(quantized, dtype=torch.bfloat16)

    def quantize(
        tensor: torch.Tensor, *, granularity: str, block_size: tuple[int, int]
    ):
        assert tensor is source and granularity == "block" and block_size == (32, 32)
        return quantized, torch.full((1, 1), 7.0)

    monkeypatch.setattr(loaders, "quantize_fp8", quantize)
    loaders.copy_expert_shard(
        destination, source, scales, (32, 32), weight_copy=_copy_packed_fp8_shard
    )
    assert torch.equal(
        destination.view(torch.uint8),
        pack_fp8_weight_n16k32(quantized.unsqueeze(0))[0].view(torch.uint8),
    )
    assert scales.item() == 7.0
    assert addresses == (destination.data_ptr(), scales.data_ptr())
