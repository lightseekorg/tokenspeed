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

"""Real FlashInfer planning from the MLA leaf's CPU length mirrors."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="FlashMLA prefill uses NVIDIA FlashInfer wrappers.",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_flashmla_prefill_plans_without_device_downloads(
    monkeypatch, with_prefix, mixed
):
    from test.runtime.conftest import make_kimi_pool

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.paged import flashmla
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "mla_disable_ragged", False)
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 256)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    monkeypatch.setattr(flashmla, "get_flashinfer_workspace_size", lambda: 64 << 20)
    spec = MLAConfig(
        backend_name="flashmla",
        num_attention_heads=16,
        num_kv_heads=1,
        head_dim=192,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=128,
        kernel_page_size=64,
        context_len=512,
        max_bs=4,
        kv_cache_quant_method="",
        components=(spec,),
    )
    leaf = flashmla.FlashMLABackend(config, spec, kernel_page_size=64)
    leaf.set_cache_pool(make_kimi_pool("cuda", usable_pages=6))
    leaf.init_cuda_graph_state(max_bs=4)
    table = torch.tensor(
        [[2, 5, 1, 0, 0, 0, 0, 0], [4, 3, 1, 0, 0, 0, 0, 0], [1] * 8],
        dtype=torch.int32,
        device="cuda",
    )
    prefix_cpu = torch.tensor([65, 17] if with_prefix else [0, 0], dtype=torch.int32)
    extends_cpu = torch.tensor([4, 3], dtype=torch.int32)
    prefix = prefix_cpu.to("cuda")
    extends = extends_cpu.to("cuda")
    lengths = torch.cat((prefix + extends, torch.tensor([8], device="cuda")))
    lengths = lengths.to(torch.int32)
    torch.manual_seed(43)
    if with_prefix:
        q_nope = torch.randn(7, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        q_pe = torch.randn(7, 16, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        ckv = torch.randn(6 * 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        kpe = torch.randn(6 * 64, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    else:
        query = torch.randn(7, 16, 192, device="cuda", dtype=torch.bfloat16) * 0.1
        keys = torch.randn_like(query) * 0.1
        values = torch.randn(7, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.1

    previous_sync_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        leaf.init_forward_metadata(
            3 if mixed else 2,
            2,
            lengths,
            table,
            ForwardMode.MIXED if mixed else ForwardMode.EXTEND,
            extend_seq_lens=extends,
            extend_seq_lens_cpu=extends_cpu,
            extend_prefix_lens=prefix,
            extend_prefix_lens_cpu=prefix_cpu,
            extend_with_prefix=with_prefix,
            query_shard=None,
            page_table_cpu=None,
        )
        if with_prefix:
            output = leaf.prefill_wrapper_paged.run(
                q_nope, q_pe, ckv.view(-1, 1, 512), kpe.view(-1, 1, 64)
            )
        else:
            output = leaf.prefill_wrapper_ragged.forward(
                query, keys, values, causal=True, sm_scale=spec.scaling
            )
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_mode)

    expected = []
    start = 0
    for row, count in enumerate(extends_cpu.tolist()):
        end = start + count
        if with_prefix:
            length = int(prefix_cpu[row]) + count
            positions = torch.arange(length, device="cuda")
            slots = table[row, positions // 64].long() * 64 + positions % 64
            q = torch.cat((q_nope[start:end], q_pe[start:end]), dim=-1)
            k = torch.cat((ckv[slots], kpe[slots]), dim=-1)[:, None, :]
            v = ckv[slots, None, :]
            causal_mask = positions[None, :] <= (
                int(prefix_cpu[row]) + torch.arange(count, device="cuda")[:, None]
            )
        else:
            q, k, v = query[start:end], keys[start:end], values[start:end]
            causal_mask = torch.ones(
                count, count, dtype=torch.bool, device="cuda"
            ).tril()
        expected.append(
            F.scaled_dot_product_attention(
                q.transpose(0, 1).float(),
                k.transpose(0, 1).float(),
                v.transpose(0, 1).float(),
                attn_mask=causal_mask,
                scale=spec.scaling,
            ).transpose(0, 1)
        )
        start = end
    torch.testing.assert_close(
        output.float(), torch.cat(expected), atol=2e-3, rtol=2e-3
    )
