"""Fp8LinearMethod online (bf16 checkpoint) block-scaled quantization path."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _load_bf16_weight(
    method: Fp8LinearMethod, n: int, k: int, device: str = "cuda"
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Run create_weights, fill the weight as a loader would, then finalize."""
    layer = torch.nn.Module()
    method.create_weights(layer, k, [n], k, n, torch.bfloat16, weight_loader=None)
    layer = layer.to(device)

    torch.manual_seed(0)
    reference = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
    layer.weight.data = reference.clone()
    method.process_weights_after_loading(layer)
    return layer, reference


def test_online_quantization_defaults_to_block_scales() -> None:
    config = Fp8Config()
    assert not config.is_checkpoint_fp8_serialized
    assert config.weight_block_size == [128, 128]


def test_serialized_checkpoint_keeps_per_tensor_default() -> None:
    """A per-tensor FP8 checkpoint must not be reinterpreted as block-scaled."""
    config = Fp8Config(is_checkpoint_fp8_serialized=True)
    assert config.weight_block_size is None


def test_block_quant_rejects_static_activation() -> None:
    with pytest.raises(ValueError, match="dynamic activation scheme"):
        Fp8Config(activation_scheme="static")


def test_process_weights_quantizes_bf16_checkpoint() -> None:
    n, k = 512, 2048
    method = Fp8LinearMethod(Fp8Config())
    layer, reference = _load_bf16_weight(method, n, k)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (n, k)
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer.weight_scale_inv.shape == (n // 128, k // 128)
    assert layer.input_scale is None
    del reference


@pytest.mark.parametrize("n, k", [(512, 2048), (2048, 512)])
def test_apply_matches_bf16_reference(n: int, k: int) -> None:
    method = Fp8LinearMethod(Fp8Config())
    layer, reference = _load_bf16_weight(method, n, k)

    x = torch.randn(32, k, device="cuda", dtype=torch.bfloat16) * 0.1
    out = method.apply(layer, x)

    expected = x.float() @ reference.float().t()
    assert out.shape == (32, n)
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.99


def test_online_partitions_need_no_block_alignment() -> None:
    """Online quantization blocks each shard after sharding, so a partition
    that is not a multiple of the block shape must still be accepted."""
    method = Fp8LinearMethod(Fp8Config())
    layer = torch.nn.Module()
    # 200 is not a multiple of block_n=128, and 300 not of block_k=128.
    method.create_weights(
        layer, 300, [200], 600, 400, torch.bfloat16, weight_loader=None
    )
    layer = layer.to("cuda")
    layer.weight.data = (
        torch.randn(200, 300, device="cuda", dtype=torch.bfloat16) * 0.05
    )
    method.process_weights_after_loading(layer)

    assert layer.weight_scale_inv.shape == (2, 3)


def test_online_quantization_ignores_linear_attention_layers() -> None:
    """Online fp8 must exclude GDN/mamba projections (delta-rule recurrence
    amplifies quantization error; official pre-quantized checkpoints exclude
    linear_attn as well). Serialized checkpoints keep their own list."""
    from tokenspeed.runtime.layers.quantization.utils import (
        check_equal_or_regex_match,
    )

    online = Fp8Config()
    for name in (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.3.linear_attn.conv1d",
        "model.layers.7.linear_attn.out_proj",
    ):
        assert check_equal_or_regex_match(name, online.ignored_layers)
    assert not check_equal_or_regex_match(
        "model.layers.0.self_attn.qkv_proj", online.ignored_layers
    )

    serialized = Fp8Config(is_checkpoint_fp8_serialized=True)
    assert not serialized.ignored_layers


def test_online_quantization_ignores_grouped_wo_a() -> None:
    """Online fp8 must exclude DeepSeek-V4 ``wo_a``.

    Its grouped projection plan requires serialized weights and scales for
    backend-owned preprocessing. Online quantization does not provide that
    checkpoint contract.
    """
    from tokenspeed.runtime.layers.quantization.utils import (
        check_equal_or_regex_match,
    )

    online = Fp8Config()
    assert check_equal_or_regex_match("model.layers.0.attn.wo_a", online.ignored_layers)
    for name in ("model.layers.0.attn.wo_b", "model.layers.0.attn.q_proj"):
        assert not check_equal_or_regex_match(name, online.ignored_layers)

    serialized = Fp8Config(is_checkpoint_fp8_serialized=True)
    assert not check_equal_or_regex_match(
        "model.layers.0.attn.wo_a", serialized.ignored_layers
    )
