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

"""ModelOpt FP8_PB_WO loading: w8a8 dispatch path + load-time dequant (GPU).

FP8_PB_WO aliases to FP8_BLOCK_SCALES (TRT-LLM precedent): plain-Linear
modules keep FP8 weights and run the DeepSeek-style w8a8 blockwise
Fp8LinearMethod (dynamic per-token-128-group activation quantization), with
the ModelOpt ``weight_scale`` renamed/squeezed to ``weight_scale_inv`` by
``preprocess_fp8_pb_wo_weights``. Raw-consumed modules arrive from the same
preprocessor block-dequantized to bf16. Also covers the fused_qkv_a verbatim
assembly (``_assemble_fp8_fused_qkv_a``): bitwise segment/scale stacking in
the private [gate | q_a | kv_a | pad] order and the 128-padded rows keeping
the flashinfer blockscale GEMM selected.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
    ModelOptMixedConfig,
    preprocess_fp8_pb_wo_weights,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the w8a8 GEMM path"
)

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _config() -> ModelOptMixedConfig:
    return ModelOptMixedConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.self_attn.o_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.0.self_attn.kv_b_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.0.self_attn.f_b_proj": {"quant_algo": "FP8_PB_WO"},
            },
            "exclude_modules": ["lm_head"],
        }
    )


def _quantize_per_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    nb, kb = (n + 127) // 128, (k + 127) // 128
    padded = torch.zeros(nb * 128, kb * 128, dtype=torch.float32, device=w.device)
    padded[:n, :k] = w.float()
    blocks = padded.view(nb, 128, kb, 128)
    scales = (blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / _FP8_MAX).contiguous()
    q = (blocks / scales[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return (
        q.view(nb * 128, kb * 128)[:n, :k].to(torch.float8_e4m3fn).contiguous(),
        scales,
    )


def _dequant_f32(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    s_full = scales.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return q.float() * s_full


def test_w8a8_path_end_to_end_matches_fp32_reference() -> None:
    """dispatch -> create_weights -> preprocessed load -> process -> apply."""
    torch.manual_seed(0)
    config = _config()
    n, k, m = 512, 640, 33
    w = torch.randn(n, k, device="cuda") * 0.05
    q, scales = _quantize_per_block(w)
    scale_4d = scales.reshape(scales.shape[0], 1, scales.shape[1], 1)

    method = config.get_quant_method(
        torch.nn.Module(), "model.layers.0.self_attn.o_proj"
    )
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=torch.bfloat16,
            weight_loader=lambda param, lw: param.data.copy_(lw),
        )
    assert hasattr(layer, "weight_scale_inv")  # DeepSeek-style block scale

    stream = [
        ("model.layers.0.self_attn.o_proj.weight", q),
        ("model.layers.0.self_attn.o_proj.weight_scale", scale_4d),
    ]
    for name, tensor in preprocess_fp8_pb_wo_weights(iter(stream), config):
        suffix = name.rsplit(".", 1)[-1]
        param = getattr(layer, suffix)
        param.weight_loader(param, tensor)
    assert layer.weight.dtype == torch.float8_e4m3fn  # FP8-resident
    assert torch.equal(layer.weight_scale_inv.data, scales)

    method.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x)
    torch.cuda.synchronize()

    ref32 = x.float() @ _dequant_f32(q, scales).t()
    rel_err = ((out.float() - ref32).abs().amax() / ref32.abs().amax()).item()
    # w8a8: fp8 dynamic activation quantization dominates; same error band as
    # the DeepSeek R1-style block-FP8 layers this path serves.
    assert rel_err < 5e-2, f"{rel_err=:.3e}"


def test_w8a8_scale_shards_with_column_parallel_narrow() -> None:
    torch.manual_seed(1)
    config = _config()
    tp_size, n_full, k = 2, 512, 256
    n_local = n_full // tp_size
    w = torch.randn(n_full, k, device="cuda")
    q, scales = _quantize_per_block(w)
    scale_4d = scales.reshape(scales.shape[0], 1, scales.shape[1], 1)

    for rank in range(tp_size):
        method = config.get_quant_method(
            torch.nn.Module(), "model.layers.0.self_attn.kv_b_proj"
        )
        layer = torch.nn.Module()

        def _col_loader(param, lw, rank=rank):
            param.load_column_parallel_weight(lw, tp_rank=rank)

        with torch.device("cuda"):
            method.create_weights(
                layer=layer,
                input_size_per_partition=k,
                output_partition_sizes=[n_local],
                input_size=k,
                output_size=n_full,
                params_dtype=torch.bfloat16,
                weight_loader=_col_loader,
            )
        stream = [
            ("model.layers.0.self_attn.kv_b_proj.weight", q),
            ("model.layers.0.self_attn.kv_b_proj.weight_scale", scale_4d),
        ]
        for name, tensor in preprocess_fp8_pb_wo_weights(iter(stream), config):
            suffix = name.rsplit(".", 1)[-1]
            param = getattr(layer, suffix)
            param.weight_loader(param, tensor)
        # Weight rows and scale rows must narrow together.
        assert torch.equal(layer.weight.data, q[rank * n_local : (rank + 1) * n_local])
        nb_local = n_local // 128
        assert torch.equal(
            layer.weight_scale_inv.data,
            scales[rank * nb_local : (rank + 1) * nb_local],
        )


def test_fp8_fused_assembly_verbatim_and_pad() -> None:
    """Reordered fused assembly copies codes/scales bit-identically.

    The private [gate | q_a | kv_a | pad] order puts every boundary on the
    128 grid, so segment codes and scale rows stack verbatim (zero
    requantization); the ragged last segment's trailing scale block is
    completed by the zero pad rows, which dequantize to exact zeros under it
    (pad lemma).
    """
    from tokenspeed.runtime.models.kimi_k3 import _assemble_fp8_fused_qkv_a

    torch.manual_seed(4)
    k = 256
    seg_rows = [256, 256, 192]  # [gate | q_a | kv_a]-style, ragged last
    segments = [
        _quantize_per_block(torch.randn(rows, k, device="cuda")) for rows in seg_rows
    ]
    fused_w, fused_s = _assemble_fp8_fused_qkv_a(segments, total_rows=768)
    assert fused_w.dtype == torch.float8_e4m3fn
    assert fused_w.shape == (768, k) and fused_s.shape == (6, k // 128)

    row = block = 0
    for codes, scales in segments:
        rows, nblocks = codes.shape[0], scales.shape[0]
        assert torch.equal(
            fused_w[row : row + rows].view(torch.uint8), codes.view(torch.uint8)
        )
        assert torch.equal(fused_s[block : block + nblocks], scales)
        row += rows
        block += nblocks
    # Tail pad rows are exact-zero codes; the last (ragged) segment's final
    # scale row covers them, so their dequant is exactly zero.
    assert torch.all(fused_w[row:].view(torch.uint8) == 0)
    assert block == fused_s.shape[0]

    # Interior segments must be 128-row multiples (only the tail may be
    # ragged) -- the layout invariant that makes verbatim stacking valid.
    with pytest.raises(ValueError, match="only the last segment"):
        _assemble_fp8_fused_qkv_a(
            [segments[2], segments[0], segments[1]], total_rows=768
        )


def test_padded_fused_assembly_selects_flashinfer_blockscale() -> None:
    """The 128-padded fused_qkv_a assembly must keep the flashinfer GEMM.

    Kimi-K3's fused_qkv_a has 2880 real rows per rank at tp16; without tail
    padding (N % 128 == 64) Fp8LinearMethod falls back to the portable
    blockscale GEMM. Assert the padded verbatim assembly prepares a native
    kernel plan and that the pad rows produce exact-zero GEMM outputs.
    """
    from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale

    from tokenspeed.runtime.models.kimi_k3 import _assemble_fp8_fused_qkv_a

    if has_flashinfer_fp8_blockscale is None or not has_flashinfer_fp8_blockscale():
        pytest.skip("requires the flashinfer fp8 blockscale GEMM")

    torch.manual_seed(3)
    config = ModelOptMixedConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa": {
                    "quant_algo": "FP8_PB_WO"
                },
                "model.layers.0.mlp.experts.0.w1": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            },
            "exclude_modules": ["lm_head"],
        }
    )
    k = 256
    seg_rows = [256, 256, 192]  # 704 real rows: % 128 == 64, like K3's 2880
    segments = [
        _quantize_per_block(torch.randn(rows, k, device="cuda")) for rows in seg_rows
    ]
    n_real = sum(seg_rows)

    def _build(n_out: int, fused_w: torch.Tensor, fused_s: torch.Tensor):
        method = config.get_quant_method(
            torch.nn.Module(), "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"
        )
        layer = torch.nn.Module()
        with torch.device("cuda"):
            method.create_weights(
                layer=layer,
                input_size_per_partition=k,
                output_partition_sizes=[n_out],
                input_size=k,
                output_size=n_out,
                params_dtype=torch.bfloat16,
                weight_loader=lambda param, lw: param.data.copy_(lw),
            )
        layer.weight.weight_loader(layer.weight, fused_w)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, fused_s)
        method.process_weights_after_loading(layer)
        return method, layer

    # Unpadded control: 704 % 128 != 0 lands on the Triton fallback. (Single
    # ragged tensor: the verbatim assembly itself only emits padded rows.)
    ctrl_w = torch.cat([w for w, _ in segments])
    ctrl_s = torch.cat([segments[0][1], segments[1][1], segments[2][1]])
    _, layer_ctrl = _build(n_real, ctrl_w, ctrl_s)
    assert layer_ctrl.weight_scale_inv.dim() == 2

    # Padded assembly: 768 % 128 == 0 selects the flashinfer blockscale GEMM.
    fused_w, fused_s = _assemble_fp8_fused_qkv_a(segments, total_rows=768)
    method, layer = _build(768, fused_w, fused_s)
    assert method.prepared_linear_plan(layer) is not None

    x = torch.randn(9, k, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x)
    torch.cuda.synchronize()
    assert out.shape == (9, 768)
    # Zero pad rows contribute exact-zero outputs.
    assert torch.all(out[:, n_real:] == 0)
    # Real rows track the fp32 dequant reference within the w8a8 band.
    w_dq = _dequant_f32(layer.weight[:n_real], layer.weight_scale_inv[: 704 // 128 + 1])
    ref32 = x.float() @ w_dq.t()
    rel_err = (
        (out[:, :n_real].float() - ref32).abs().amax() / ref32.abs().amax()
    ).item()
    assert rel_err < 5e-2, f"{rel_err=:.3e}"


def test_dequant_route_bitwise_on_gpu() -> None:
    torch.manual_seed(2)
    config = _config()
    n, k = 96, 200  # ragged both axes (b_proj-like)
    q, scales = _quantize_per_block(torch.randn(n, k, device="cuda"))
    scale_4d = scales.reshape(scales.shape[0], 1, scales.shape[1], 1)
    stream = [
        ("model.layers.0.self_attn.f_b_proj.weight", q),
        ("model.layers.0.self_attn.f_b_proj.weight_scale", scale_4d),
    ]
    out = dict(preprocess_fp8_pb_wo_weights(iter(stream), config))
    got = out["model.layers.0.self_attn.f_b_proj.weight"]
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, _dequant_f32(q, scales).to(torch.bfloat16))
    assert "model.layers.0.self_attn.f_b_proj.weight_scale" not in out
