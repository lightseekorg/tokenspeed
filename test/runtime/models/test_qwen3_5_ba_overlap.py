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

"""GDN BA overlap: input/output dependencies, PDL and graph replay."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn._triton.qkv_split import pack_gdn_projection
from tokenspeed_kernel.ops.tuning import autotune
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5GatedDeltaNet
from tokenspeed.runtime.utils.env import global_server_args_dict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM10x FP8 projection kernels",
)


def make_gdn(*, hidden_size, overlap):
    config = Qwen3_5TextConfig(
        hidden_size=hidden_size,
        linear_num_key_heads=2,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config.dtype = torch.bfloat16
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        gdn = Qwen3_5GatedDeltaNet(
            config,
            Mapping(rank=0, world_size=1),
            0,
            prefix="model.layers.0.linear_attn",
            alt_stream=torch.cuda.Stream() if overlap else None,
        ).cuda()
    finally:
        torch.set_default_dtype(dtype)
    torch.manual_seed(123)
    for proj in [gdn.in_proj_qkvz, gdn.in_proj_ba]:
        proj.weight.data.normal_(0, 0.02)
        proj.quant_method.process_weights_after_loading(proj)
    return gdn


def project_and_pack(gdn, norm, x):
    hidden = norm(x)
    qkvz, ba = gdn._forward_input_proj(hidden, None)
    qkv = qkvz[:, : gdn.key_dim * 2 + gdn.value_dim]
    b, a = ba.chunk(2, dim=-1)
    packed = pack_gdn_projection(qkv, a, b)
    return (*packed, qkvz[:, -gdn.value_dim :])


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("backend", ["auto", "deepgemm"])
@pytest.mark.parametrize(
    "rows,hidden_size",
    [(1, 512), (8, 512), (32, 512), (129, 512), (8, 8192), (4096, 8192)],
)
def test_ba_overlap_graph_replay_matches_serial(
    monkeypatch, pdl, backend, rows, hidden_size
):
    monkeypatch.setitem(global_server_args_dict, "enable_fp8_attn_proj", True)
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", backend)
    previous_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        serial = make_gdn(hidden_size=hidden_size, overlap=False)
        parallel = make_gdn(hidden_size=hidden_size, overlap=True)
        norm = GemmaRMSNorm(hidden_size).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
        with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
            for gdn in [serial, parallel]:
                project_and_pack(gdn, norm, x)
        torch.cuda.synchronize()
        # Capture on a non-default stream, as the serving graph runner does.
        capture = torch.cuda.Stream()
        capture.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture):
            actual = project_and_pack(parallel, norm, x)
        torch.cuda.current_stream().wait_stream(capture)
        for i in range(8):
            x.normal_().mul_(i + 1)
            expected = project_and_pack(serial, norm, x)
            eager = project_and_pack(parallel, norm, x)
            graph.replay()
            for a, e, v in zip(actual, expected, eager):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
                torch.testing.assert_close(v, e, atol=0, rtol=0)
    finally:
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("pdl", [False, True])
def test_ba_overlap_reuses_fork_across_captured_batches(monkeypatch, pdl):
    monkeypatch.setitem(global_server_args_dict, "enable_fp8_attn_proj", True)
    previous_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        serial = make_gdn(hidden_size=512, overlap=False)
        parallel = make_gdn(hidden_size=512, overlap=True)
        norm = GemmaRMSNorm(512).to(device="cuda", dtype=torch.bfloat16)
        captures = []
        for rows in [1, 8]:
            x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
            project_and_pack(parallel, norm, x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = project_and_pack(parallel, norm, x)
            captures.append((x, graph, output))
        for _ in range(8):
            for x, graph, output in reversed(captures):
                x.normal_()
                graph.replay()
                reference = project_and_pack(serial, norm, x)
                for a, b in zip(output, reference):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
    finally:
        pdl_enabled(previous_pdl)
