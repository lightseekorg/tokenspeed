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

"""Portable dense dispatch remains independent of Gluon attention and experts."""

import argparse
import sys
from pathlib import Path

import pytest
import tokenspeed_kernel.ops.gemm as gemm
import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format

from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as channel,
)
from tokenspeed.runtime.models.beam import BeamAttention
from tokenspeed.runtime.utils.server_args import ServerArgs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

FAMILIES = [(1792, 5120), (5120, 768), (5120, 5120), (5120, 2560)]


@pytest.mark.parametrize("n,k", FAMILIES)
@pytest.mark.parametrize("m", [1, 16, 32, 64, 128, 256, 8192])
def test_channel_registry_selects_original_portable_implementation(m, n, k):
    scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    operand = tensor_format("scaled-fp8", torch.float8_e4m3fn, scale=scale)
    signature = format_signature(a=operand, b=operand)
    traits = {"m": m, "n": n, "k": k, "b_layout": "KN", "out_dtype": torch.bfloat16}
    selected = select_kernel("gemm", "mm", signature, traits=traits)
    assert selected.name == "triton_mm_fp8_scaled"
    with pytest.raises(NoKernelFoundError):
        select_kernel("gemm", "mm", signature, traits=traits, solution="gluon")


@pytest.mark.parametrize("n,k", FAMILIES)
def test_dynamic_linear_preserves_public_quantization_without_override(
    monkeypatch, n, k
):
    x = torch.empty((16, k), device="meta", dtype=torch.bfloat16)
    weight = torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn).T
    scales = torch.empty((n, 1), device="meta", dtype=torch.float32)

    def unavailable_device_gemm(a, b, **kwargs):
        assert a is x and b is weight
        assert kwargs == {"B_scales": scales, "out_dtype": x.dtype, "quant": "fp8"}
        return torch.empty((16, n), device="meta", dtype=x.dtype)

    monkeypatch.setattr(gemm, "mm", unavailable_device_gemm)
    assert channel.fp8_dynamic_linear(x, weight, scales).shape == (16, n)


def test_beam_offered_fp8_preserves_original_codes_and_scales(monkeypatch):
    import tokenspeed.runtime.models.beam as beam

    hidden = torch.empty((32, 5120), device="meta", dtype=torch.bfloat16)
    codes = hidden.to(torch.float8_e4m3fn)
    token_scales = torch.empty((32, 1), device="meta", dtype=torch.float32)
    weight = torch.empty((1792, 5120), device="meta", dtype=codes.dtype).T
    scales = torch.empty((1792, 1), device="meta", dtype=torch.float32)
    attention = BeamAttention.__new__(BeamAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_gate_weight, attention.qkv_gate_scale = weight, scales
    attention.q_size, attention.kv_size = 768, 128
    attention._token_fp8 = (hidden, codes, token_scales)

    def unavailable_device_gemm(a, b, **kwargs):
        assert a is codes and b is weight
        assert kwargs == {
            "A_scales": token_scales,
            "B_scales": scales,
            "out_dtype": hidden.dtype,
            "quant": "fp8",
        }
        return torch.empty((32, 1792), device="meta", dtype=hidden.dtype)

    monkeypatch.setattr(beam, "kernel_mm", unavailable_device_gemm)
    qkv, gate = attention.project_qkv_gate(hidden)
    assert qkv.shape == (32, 1024) and gate.shape == (32, 768)
    assert attention._token_fp8 is None


def test_removed_dense_option_does_not_remove_other_gluon_options():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    options = parser.parse_args(
        [
            "--model",
            "x",
            "--attention-backend",
            "gluon_mha",
            "--moe-backend",
            "gluon",
            "--enable-mixed-batch",
        ]
    )
    assert options.dense_gemm_backend == "auto"
    assert options.attention_backend == "gluon_mha"
    assert options.moe_backend == "gluon" and options.enable_mixed_batch
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "x", "--dense-gemm-backend", "gluon"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
