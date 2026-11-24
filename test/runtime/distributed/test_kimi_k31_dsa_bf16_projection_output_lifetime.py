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

"""Exercise Kimi 3.1 DSA's actual QKV-A, latent KV, and gate split under CUDA Graph."""

import gc
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.linear import (
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )
    from tokenspeed.runtime.models.kimi_k3_projection import (
        validate_projection_settings,
    )
    from tokenspeed.runtime.models.kimi_k31 import KimiK31SparseMLAAttention
    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    world = 4
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    group = tuple(range(world))
    process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    torch.set_default_dtype(torch.bfloat16)
    mapping = Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        attn_dp_size=world,
        linear_attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=world,
    )
    global_server_args_dict.update(
        mapping=mapping,
        dsa_index_k_format="fp8_scaled",
        chunked_prefill_size=512,
        dense_gemm_backend="auto",
    )
    parallel, _ = validate_projection_settings(mapping, world, world)
    m, k, n = 512, 7168, 14400
    config = KimiLinearConfig(
        hidden_size=k,
        num_hidden_layers=2,
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
        dsa_config={
            "dsa_layers": [2],
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_num_heads": 32,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
    )
    with torch.device(device):
        attention = torch.nn.ModuleList(
            [
                KimiK31SparseMLAAttention(
                    config=config,
                    mapping=mapping,
                    hidden_size=k,
                    num_heads=96,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    q_lora_rank=1536,
                    kv_lora_rank=512,
                    max_position_embeddings=4096,
                    quant_config=None,
                    layer_id=1,
                    prefix=f"attn{i}",
                    reduce_attn_results=False,
                    alt_stream=None,
                    qkv_parallel=parallel,
                    output_parallel=None,
                )
                for i in range(2)
            ]
        )
    modules = [layer.fused_qkv_a_proj_with_mqa for layer in attention]
    logical_indices = torch.arange(n, device=device)
    local_indices = torch.arange(modules[0].weight.shape[0], device=device)
    global_rows = rank * modules[0].weight.shape[0] + local_indices
    valid = global_rows < n
    weights = (7, 11)
    for mod, stride in zip(modules, weights):
        mod.weight.data.zero_()
        mod.weight.data[local_indices[valid], (global_rows[valid] * stride) % k] = 1
        mod.quant_method.process_weights_after_loading(mod)
    backend = AutoBackend()
    backend.configure(use_pynccl=False)
    prepare_dp_linear_communication(attention, m, torch.bfloat16, backend)
    native = modules[0].projection_workspace.fused_column
    assert native is modules[1].projection_workspace.fused_column
    assert native.input_multicast_ptr != 0
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=m,
        num_extends=0,
        input_num_tokens=m,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, m, 1),
        global_num_tokens=[m] * world,
        collective_global_num_tokens=[m] * world,
    )
    torch.manual_seed(17 + rank)
    x = torch.randint(-511, 512, (m, k), device=device).to(torch.bfloat16) / 128
    other = -x
    first_columns = (logical_indices * weights[0]) % k
    second_columns = (logical_indices * weights[1]) % k

    def project(layer, values, context):
        return layer._project_qkv_gate_a(
            values,
            context,
            comm_manager=None,
            block_scale=None,
            attnres_partial_args=None,
        )

    # A normal projection output remains valid after other invocations.
    retained, _ = modules[0](x, ctx=ctx)
    expected_retained = x[:, first_columns]
    borrowed = project(attention[0], x, ctx)
    torch.testing.assert_close(
        torch.cat(borrowed, dim=1), expected_retained, rtol=0, atol=0
    )
    assert borrowed[0].data_ptr() == native.receive.data_ptr()
    assert retained.data_ptr() != borrowed[0].data_ptr()

    short_rows = 384
    short_ctx = replace(
        ctx,
        bs=short_rows,
        input_num_tokens=short_rows,
        output_layout=ForwardOutputLayout(0, 0, short_rows, 1),
        global_num_tokens=[short_rows] * world,
        collective_global_num_tokens=[short_rows] * world,
    )
    short_output = project(attention[0], x[:short_rows], short_ctx)
    assert short_output[0].data_ptr() != native.receive.data_ptr()
    torch.testing.assert_close(
        torch.cat(short_output, dim=1), expected_retained[:short_rows], rtol=0, atol=0
    )

    saved = torch.empty_like(retained)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, stream=stream):
            torch.cuda._sleep(25000 * rank)
            first = project(attention[0], x, ctx)
            torch.cuda._sleep(75000 * ((rank + 2) % world))
            torch.cat(first, dim=1, out=saved)
            short, _ = modules[0](x[:short_rows], ctx=short_ctx)
            assert short.data_ptr() != native.receive.data_ptr()
            second = project(attention[1], other, ctx)
    finally:
        if gc_enabled:
            gc.enable()
    for replay in range(3):
        x.neg_()
        other.copy_(-x)
        torch.cuda._sleep(25000 * ((rank + replay) % world))
        graph.replay()
        torch.testing.assert_close(saved, x[:, first_columns], rtol=0, atol=0)
        torch.testing.assert_close(short, x[:short_rows, first_columns], rtol=0, atol=0)
        torch.testing.assert_close(
            torch.cat(second, dim=1), other[:, second_columns], rtol=0, atol=0
        )
        torch.testing.assert_close(retained, expected_retained, rtol=0, atol=0)
    del graph, borrowed, first, second
    release_dp_linear_communication(attention)
    dist.destroy_process_group()


def test_native_k31_dsa_qkv_intermediate_output_lifetime(tmp_path):
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires four SM100 or SM103 GPUs on one node")
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=4,
        join=True,
    )
