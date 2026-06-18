"""CPU-only coverage for Kimi MXFP4 dense/shared MLP linear weights."""

from __future__ import annotations

import importlib.util
import sys
from math import prod
from pathlib import Path

import pytest
import torch
from torch import nn


def _load_mxfp4_dense_module():
    module_path = (
        Path(__file__).parents[2]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "layers"
        / "dense"
        / "mxfp4.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_tokenspeed_dense_mxfp4",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MXFP4_DENSE = _load_mxfp4_dense_module()
Mxfp4LinearMethod = _MXFP4_DENSE.Mxfp4LinearMethod


def _copy_loader(param, loaded_weight, *args, **kwargs) -> None:
    del args, kwargs
    param.data.copy_(loaded_weight)


def _uint8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    return ((values + offset) % 251).to(torch.uint8)


def _maybe_e8m0(raw_bytes: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        return raw_bytes
    return raw_bytes.view(e8m0_dtype)


def _create_weights(
    *,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    params_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    layer = nn.Module()
    method = Mxfp4LinearMethod(quant_config=None)
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size_per_partition,
        output_partition_sizes=output_partition_sizes,
        input_size=input_size_per_partition,
        output_size=sum(output_partition_sizes),
        params_dtype=params_dtype,
        weight_loader=_copy_loader,
    )
    layer.quant_method = method
    return layer


def test_mxfp4_linear_method_allocates_kimi_shared_and_dense_shapes() -> None:
    shared_gate_up = _create_weights(
        input_size_per_partition=7168,
        output_partition_sizes=[512, 512],
    )
    assert tuple(shared_gate_up.weight.shape) == (1024, 3584)
    assert tuple(shared_gate_up.weight_scale.shape) == (1024, 224)
    assert shared_gate_up.weight.output_dim == 0
    assert shared_gate_up.weight.input_dim == 1
    assert shared_gate_up.weight_scale.output_dim == 0
    assert shared_gate_up.weight_scale.input_dim == 1

    shared_down = _create_weights(
        input_size_per_partition=512,
        output_partition_sizes=[7168],
    )
    assert tuple(shared_down.weight.shape) == (7168, 256)
    assert tuple(shared_down.weight_scale.shape) == (7168, 16)
    assert shared_down.weight.input_dim == 1
    assert shared_down.weight_scale.input_dim == 1

    dense_gate_up = _create_weights(
        input_size_per_partition=7168,
        output_partition_sizes=[4608, 4608],
    )
    assert tuple(dense_gate_up.weight.shape) == (9216, 3584)
    assert tuple(dense_gate_up.weight_scale.shape) == (9216, 224)

    dense_down = _create_weights(
        input_size_per_partition=4608,
        output_partition_sizes=[7168],
    )
    assert tuple(dense_down.weight.shape) == (7168, 2304)
    assert tuple(dense_down.weight_scale.shape) == (7168, 144)


def test_mxfp4_linear_method_preserves_e8m0_bytes_and_keeps_packed_weights() -> None:
    layer = _create_weights(
        input_size_per_partition=64,
        output_partition_sizes=[4],
        params_dtype=torch.float32,
    )
    packed = _uint8_pattern((4, 32), offset=5)
    raw_scales = torch.full((4, 2), 127, dtype=torch.uint8)

    layer.weight.weight_loader(layer.weight, packed)
    layer.weight_scale.weight_loader(layer.weight_scale, _maybe_e8m0(raw_scales))
    assert torch.equal(layer.weight_scale, raw_scales)

    layer.quant_method.process_weights_after_loading(layer)

    assert layer._mxfp4_dense_processed is True
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.dtype == torch.uint8
    assert torch.equal(layer.weight, packed)
    assert torch.equal(layer.weight_scale, raw_scales)
    assert layer.weight_triton_tensor.data_ptr() == layer.weight.data.data_ptr()
    assert (
        layer.weight_scale_triton_tensor.data_ptr()
        == layer.weight_scale.data.data_ptr()
    )


def test_mxfp4_linear_method_rejects_non_block_input_partition() -> None:
    method = Mxfp4LinearMethod(quant_config=None)
    with pytest.raises(ValueError, match="divisible by 32"):
        method.create_weights(
            layer=nn.Module(),
            input_size_per_partition=48,
            output_partition_sizes=[4],
            input_size=48,
            output_size=4,
            params_dtype=torch.bfloat16,
            weight_loader=_copy_loader,
        )
