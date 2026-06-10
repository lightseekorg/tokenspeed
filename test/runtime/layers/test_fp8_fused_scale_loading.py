import torch

from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config


def _fp8_config() -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="static",
    )


def _modelopt_mixed_config() -> Nvfp4Config:
    return Nvfp4Config(
        fp4_modules=["model.layers.0.mixer.experts"],
        fp8_modules=["model.layers.0.mixer.in_proj"],
    )


def test_merged_column_fp8_scalar_scale_broadcasts_to_all_logical_slots():
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[8, 4, 2],
        bias=False,
        quant_config=_fp8_config(),
    )

    layer.weight_scale.data.fill_(torch.finfo(torch.float32).min)
    layer.input_scale.data.fill_(torch.finfo(torch.float32).min)

    layer.weight_loader_v2(layer.weight_scale, torch.tensor(0.25))
    layer.weight_loader_v2(layer.input_scale, torch.tensor(0.5))

    torch.testing.assert_close(layer.weight_scale, torch.full((3,), 0.25))
    torch.testing.assert_close(layer.input_scale, torch.full((3,), 0.5))


def test_qkv_fp8_scalar_scale_broadcasts_to_all_logical_slots():
    layer = QKVParallelLinear(
        hidden_size=8,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        quant_config=_fp8_config(),
        tp_rank=0,
        tp_size=1,
    )

    layer.weight_scale.data.fill_(torch.finfo(torch.float32).min)
    layer.input_scale.data.fill_(torch.finfo(torch.float32).min)

    layer.weight_loader_v2(layer.weight_scale, torch.tensor(0.125))
    layer.weight_loader_v2(layer.input_scale, torch.tensor(0.75))

    torch.testing.assert_close(layer.weight_scale, torch.full((3,), 0.125))
    torch.testing.assert_close(layer.input_scale, torch.full((3,), 0.75))


def test_merged_column_fp8_requantizes_to_scalar_max_weight_scale(monkeypatch):
    from tokenspeed.runtime.layers.dense import fp8 as dense_fp8

    monkeypatch.setattr(
        dense_fp8,
        "_use_cutlass_fp8_channelwise_scales",
        lambda: False,
    )
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[2, 2],
        bias=False,
        quant_config=_fp8_config(),
    )
    layer.weight.data[:2].fill_(1.0)
    layer.weight.data[2:].fill_(1.0)
    layer.weight_scale.data.copy_(torch.tensor([0.5, 1.0]))
    layer.input_scale.data.copy_(torch.tensor([0.25, 0.25]))

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 4)
    torch.testing.assert_close(layer.weight_scale, torch.tensor(1.0))
    torch.testing.assert_close(layer.input_scale, torch.tensor(0.25))
    torch.testing.assert_close(
        layer.weight.t()[:2].float(),
        torch.full((2, 4), 0.5),
    )
    torch.testing.assert_close(
        layer.weight.t()[2:].float(),
        torch.ones((2, 4)),
    )


def test_merged_column_fp8_uses_sglang_channelwise_scale_layout(monkeypatch):
    from tokenspeed.runtime.layers.dense import fp8 as dense_fp8

    monkeypatch.setattr(
        dense_fp8,
        "_use_cutlass_fp8_channelwise_scales",
        lambda: True,
    )
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[2, 2],
        bias=False,
        quant_config=_fp8_config(),
    )
    layer.weight.data[:2].fill_(1.0)
    layer.weight.data[2:].fill_(1.0)
    layer.weight_scale.data.copy_(torch.tensor([0.5, 1.0]))
    layer.input_scale.data.copy_(torch.tensor([0.25, 0.25]))

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 4)
    assert layer.weight_scale.shape == (4, 1)
    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[0.5], [0.5], [1.0], [1.0]]),
    )
    torch.testing.assert_close(layer.input_scale, torch.tensor(0.25))
    torch.testing.assert_close(
        layer.weight.t()[:2].float(),
        torch.ones((2, 4)),
    )
    torch.testing.assert_close(
        layer.weight.t()[2:].float(),
        torch.ones((2, 4)),
    )


def test_modelopt_fp8_requantizes_before_channelwise_scale(monkeypatch):
    from tokenspeed.runtime.layers.dense import fp8 as dense_fp8

    monkeypatch.setattr(
        dense_fp8,
        "_use_cutlass_fp8_channelwise_scales",
        lambda: True,
    )
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[2, 2],
        bias=False,
        quant_config=_modelopt_mixed_config(),
        prefix="model.layers.0.mixer.in_proj",
    )
    assert type(layer.quant_method).__name__ == "ModelOptFp8LinearMethod"

    layer.weight.data[:2].fill_(1.0)
    layer.weight.data[2:].fill_(1.0)
    layer.weight_scale.data.copy_(torch.tensor([0.5, 1.0]))
    layer.input_scale.data.copy_(torch.tensor([0.25, 0.25]))

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 4)
    assert layer.weight_scale.shape == (4, 1)
    torch.testing.assert_close(layer.weight_scale, torch.ones((4, 1)))
    torch.testing.assert_close(layer.input_scale, torch.tensor(0.25))
    torch.testing.assert_close(
        layer.weight.t()[:2].float(),
        torch.full((2, 4), 0.5),
    )
    torch.testing.assert_close(
        layer.weight.t()[2:].float(),
        torch.ones((2, 4)),
    )


def test_static_activation_scale_repeats_on_cutlass_fp8(monkeypatch):
    from tokenspeed.runtime.layers.dense import fp8 as dense_fp8

    repeat_flags = []

    def fake_static_quant_fp8(input_, input_scale, *, repeat_scale=False):
        repeat_flags.append(repeat_scale)
        return input_.to(torch.float8_e4m3fn), torch.ones(
            1, 1, dtype=torch.float32, device=input_.device
        )

    def fake_mm(input_, weight, **kwargs):
        del kwargs
        return torch.empty(
            input_.shape[0], weight.shape[1], dtype=torch.bfloat16, device=input_.device
        )

    monkeypatch.setattr(
        dense_fp8,
        "_use_cutlass_fp8_channelwise_scales",
        lambda: True,
    )
    monkeypatch.setattr(dense_fp8, "static_quant_fp8", fake_static_quant_fp8)
    monkeypatch.setattr(dense_fp8.tokenspeed_kernel, "mm", fake_mm)

    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[2, 2],
        bias=False,
        quant_config=_fp8_config(),
    )
    layer.weight = torch.nn.Parameter(
        torch.empty(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)

    layer.quant_method.apply(layer, torch.ones(1, 4, dtype=torch.bfloat16))

    assert repeat_flags == [True]
