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

"""Fused model input feeds real QKV/QKVZ GEMMs and BF16 BA on its stream."""

import pytest
import torch
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5GatedDeltaNet,
    _input_norm,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100/103",
)


@pytest.mark.parametrize("backend", ["deepgemm", "auto", "trtllm_cutedsl"])
@pytest.mark.parametrize("kind", ["qkv", "qkvz"])
@pytest.mark.parametrize("rows", [4, 32])
@pytest.mark.parametrize("pdl", [False, True])
@torch.no_grad()
def test_model_projection_graph(monkeypatch, backend, kind, rows, pdl):
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", backend)
    monkeypatch.setitem(global_server_args_dict, "enable_fp8_attn_proj", True)
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "fused")
    monkeypatch.setenv("TOKENSPEED_FUSE_FP8_INPUT_QUANT", "1")
    width = 2560
    config = Qwen3_5TextConfig(
        hidden_size=width,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention", "full_attention"],
    )
    config.dtype = torch.bfloat16
    mapping = Mapping(rank=0, world_size=1)
    quant = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        weight_block_size=[128, 128],
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        if kind == "qkvz":
            node = Qwen3_5GatedDeltaNet(
                config,
                mapping,
                1,
                quant_config=quant,
                prefix="model.layers.1.linear_attn",
                alt_stream=torch.cuda.Stream(),
            ).cuda()
            projection = node.input_projection()
            node.in_proj_ba.weight.normal_(std=0.02)
            assert node._ba_fork.aux_stream is not None
        else:
            node = Qwen3_5AttentionDecoderLayer(
                config,
                mapping,
                1,
                quant_config=quant,
                prefix="model.layers.1.self_attn",
            ).cuda()
            projection = node.qkv_proj
    finally:
        torch.set_default_dtype(previous_dtype)
    projection.weight.normal_(std=0.02)
    projection.quant_method.process_weights_after_loading(projection)
    norm = GemmaRMSNorm(width).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.normal_(std=0.3)
    manager = CommManager(
        mapping, 1, False, False, False, input_layernorm=norm, query_sharded=False
    )
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) * 3

    def run():
        bf16, copy, updated = _input_norm(
            manager, norm, x * 1.0, residual * 1.0, None, projection=projection
        )
        if kind == "qkvz":
            outputs = node._forward_input_proj(bf16, copy)
        else:
            outputs = node._project_qkv(bf16 if copy is None else copy)
        return bf16, updated, outputs

    previous_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for magnitude in (0.01, 1.0, 16.0):
            x.normal_().mul_(magnitude)
            residual.normal_().mul_(magnitude * 3)
            graph.replay()
            monkeypatch.setenv("TOKENSPEED_FUSE_FP8_INPUT_QUANT", "0")
            expected = run()
            monkeypatch.setenv("TOKENSPEED_FUSE_FP8_INPUT_QUANT", "1")
            for a, e in zip(actual[:2], expected[:2]):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            for a, e in zip(actual[2], expected[2]):
                if a is not None:
                    assert a.dtype == torch.bfloat16
                    torch.testing.assert_close(a, e, rtol=0, atol=0)
    finally:
        pdl_enabled(previous_pdl)
