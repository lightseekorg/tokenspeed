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

"""Fp8BlockWeightOnlyLinearMethod (ModelOpt FP8_PB_WO): load + W8A16 apply.

Covers the ModelOpt 4-D scale layout normalization, TP narrowing of weight and
block-scale grids, the unloaded-scale sentinel, the weight-only guard against
pre-quantized activations, and numerics against a bf16-dequant reference GEMM.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.quantization.fp8 import Fp8BlockWeightOnlyConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the triton W8A16 GEMM"
)

_BLOCK_N, _BLOCK_K = 128, 128
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _method():
    from tokenspeed.runtime.layers.dense.fp8_wo import Fp8BlockWeightOnlyLinearMethod

    return Fp8BlockWeightOnlyLinearMethod(
        Fp8BlockWeightOnlyConfig(weight_block_size=[_BLOCK_N, _BLOCK_K])
    )


def _quantize_per_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    nb = (n + _BLOCK_N - 1) // _BLOCK_N
    kb = (k + _BLOCK_K - 1) // _BLOCK_K
    padded = torch.zeros(
        nb * _BLOCK_N, kb * _BLOCK_K, dtype=torch.float32, device=w.device
    )
    padded[:n, :k] = w.float()
    blocks = padded.view(nb, _BLOCK_N, kb, _BLOCK_K)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scales = (amax / _FP8_MAX).contiguous()
    q = (blocks / scales[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    q = q.view(nb * _BLOCK_N, kb * _BLOCK_K)[:n, :k].to(torch.float8_e4m3fn)
    return q.contiguous(), scales


def _dequant_bf16(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    s_full = scales.repeat_interleave(_BLOCK_N, dim=0)[:n].repeat_interleave(
        _BLOCK_K, dim=1
    )[:, :k]
    return (q.float() * s_full).to(torch.bfloat16)


def _replicated_loader(param, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def _column_loader(tp_rank: int):
    def _loader(param, loaded_weight: torch.Tensor) -> None:
        param.load_column_parallel_weight(loaded_weight, tp_rank=tp_rank)

    return _loader


def _create_layer(n: int, k: int, weight_loader, n_full: int | None = None):
    """create_weights for an [n, k] partition of an [n_full, k] weight."""
    method = _method()
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n_full if n_full is not None else n,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
    return method, layer


def test_load_modelopt_4d_scale_and_apply_matches_bf16_dequant() -> None:
    torch.manual_seed(0)
    n, k, m = 384, 640, 9
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    w *= torch.logspace(-2, 1, steps=n, device="cuda")[:, None]
    q, scales = _quantize_per_block(w)

    method, layer = _create_layer(n, k, _replicated_loader)
    layer.weight.weight_loader(layer.weight, q)
    # ModelOpt exports the scale as [nb, 1, kb, 1].
    scale_4d = scales.reshape(scales.shape[0], 1, scales.shape[1], 1)
    layer.weight_scale.weight_loader(layer.weight_scale, scale_4d)
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn  # fp8-resident, no dequant
    assert layer.weight_scale.shape == (scales.shape[0], scales.shape[1])

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x)
    torch.cuda.synchronize()

    w_dq = _dequant_bf16(q, scales)
    ref32 = x.float() @ w_dq.float().t()
    ref_bf16 = x @ w_dq.t()
    scale_ref = ref32.abs().amax().clamp(min=1.0)
    err_kernel = (out.float() - ref32).abs().amax() / scale_ref
    err_torch = (ref_bf16.float() - ref32).abs().amax() / scale_ref
    # Weight-only: the only deviation allowed is bf16-GEMM rounding itself.
    assert err_kernel <= 4 * err_torch + 1e-5, f"{err_kernel=:.3e} {err_torch=:.3e}"


def test_tp_column_shards_narrow_weight_and_scale_together() -> None:
    torch.manual_seed(1)
    tp_size, n_full, k = 2, 512, 256
    n_local = n_full // tp_size
    w = torch.randn(n_full, k, device="cuda", dtype=torch.float32)
    q, scales = _quantize_per_block(w)
    scale_4d = scales.reshape(scales.shape[0], 1, scales.shape[1], 1)
    x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)

    full_ref = x @ _dequant_bf16(q, scales).t()
    outs = []
    for rank in range(tp_size):
        method, layer = _create_layer(n_local, k, _column_loader(rank), n_full=n_full)
        layer.weight.weight_loader(layer.weight, q)
        layer.weight_scale.weight_loader(layer.weight_scale, scale_4d)
        # Shards must hold the matching rows of both grids.
        assert torch.equal(layer.weight.data, q[rank * n_local : (rank + 1) * n_local])
        nb_local = n_local // _BLOCK_N
        assert torch.equal(
            layer.weight_scale.data,
            scales[rank * nb_local : (rank + 1) * nb_local],
        )
        method.process_weights_after_loading(layer)
        outs.append(method.apply(layer, x))
    torch.cuda.synchronize()
    out = torch.cat(outs, dim=-1)
    scale_ref = full_ref.float().abs().amax().clamp(min=1.0)
    assert (out.float() - full_ref.float()).abs().amax() / scale_ref < 1e-2


def test_unloaded_scale_sentinel_raises() -> None:
    n, k = 128, 128
    method, layer = _create_layer(n, k, _replicated_loader)
    layer.weight.weight_loader(
        layer.weight, torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn)
    )
    with pytest.raises(RuntimeError, match="weight_scale was never"):
        method.process_weights_after_loading(layer)


def test_rejects_prequantized_activation_block_scale() -> None:
    torch.manual_seed(2)
    n, k = 128, 128
    q, scales = _quantize_per_block(torch.randn(n, k, device="cuda"))
    method, layer = _create_layer(n, k, _replicated_loader)
    layer.weight.weight_loader(layer.weight, q)
    layer.weight_scale.weight_loader(layer.weight_scale, scales)
    method.process_weights_after_loading(layer)
    x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="weight-only"):
        method.apply(layer, x, None, torch.ones(2, 1, device="cuda"), torch.bfloat16)


def test_sharded_partitions_must_align_to_blocks() -> None:
    method = _method()
    layer = torch.nn.Module()
    with pytest.raises(ValueError, match="block_k"):
        method.create_weights(
            layer=layer,
            input_size_per_partition=96,  # 96 % 128 != 0 while row-sharded
            output_partition_sizes=[128],
            input_size=192,
            output_size=128,
            params_dtype=torch.bfloat16,
            weight_loader=_replicated_loader,
        )
    with pytest.raises(ValueError, match="block_n"):
        method.create_weights(
            layer=layer,
            input_size_per_partition=128,
            output_partition_sizes=[96],  # 96 % 128 != 0 while column-sharded
            input_size=128,
            output_size=192,
            params_dtype=torch.bfloat16,
            weight_loader=_replicated_loader,
        )


def test_modelopt_mixed_dispatches_fp8_pb_wo() -> None:
    """FP8_PB_WO aliases to FP8_BLOCK_SCALES (TRT-LLM precedent).

    Modules whose weight flows through quant_method.apply run the DeepSeek
    w8a8 blockwise Fp8LinearMethod; raw-consumed modules are dequantized at
    load and dispatch UnquantizedLinearMethod. The standalone W8A16 method in
    this file is NOT a dispatch target.
    """
    from tokenspeed.runtime.layers.dense import (
        Fp8LinearMethod,
        UnquantizedLinearMethod,
    )
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        ModelOptMixedConfig,
    )

    config = ModelOptMixedConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.self_attn.o_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.0.self_attn.kv_b_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.0.self_attn.q_b_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.0.self_attn.f_b_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.0.mlp.experts.0.w1": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            },
            "exclude_modules": ["lm_head"],
        }
    )
    layer = torch.nn.Module()
    for leaf in ("o_proj", "kv_b_proj"):
        method = config.get_quant_method(layer, f"model.layers.0.self_attn.{leaf}")
        assert isinstance(method, Fp8LinearMethod)
        assert method.block_quant
        assert method.quant_config.weight_block_size == [128, 128]
        assert method.quant_config.scale_fmt is None
    for leaf in ("q_b_proj", "f_b_proj"):
        method = config.get_quant_method(layer, f"model.layers.0.self_attn.{leaf}")
        assert isinstance(method, UnquantizedLinearMethod)
