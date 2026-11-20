# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Real K3.1 DSA with projection TP, including owners with no local tokens."""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_TEST_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TEST_DIR))
sys.path.insert(0, str(_TEST_DIR.parent))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="runtime-2gpu")


def _forward(attention, comm, router, pool, hidden, prefix, counts, rank, decode):
    from test.runtime.test_kimi_k31 import _block_table

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    rows = counts[rank]
    mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
    blocks = _block_table([1])
    lengths = torch.tensor([prefix + rows], dtype=torch.int32, device="cuda")
    slot = torch.tensor([0], dtype=torch.int32, device="cuda")
    if rows and decode:
        router.refresh_decode_metadata(
            1, 1, slot, lengths, forward_mode=mode, block_tables=blocks
        )
    elif rows:
        extend = torch.tensor([rows], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([prefix], dtype=torch.int32, device="cuda")
        router.init_forward_metadata(
            1,
            1,
            slot,
            lengths,
            mode,
            block_tables=blocks,
            extend_seq_lens=extend,
            extend_seq_lens_cpu=extend.cpu(),
            extend_prefix_lens=prefix_lens,
            extend_prefix_lens_cpu=prefix_lens.cpu(),
            extend_replay_lens_cpu=torch.tensor([0], dtype=torch.int32),
            extend_prompt_lens_cpu=lengths.cpu(),
            extend_with_prefix=prefix > 0,
            query_shard=None,
            block_tables_cpu={name: value.cpu() for name, value in blocks.items()},
        )
    ctx = ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=int(rows > 0),
        num_extends=int(rows > 0 and not decode),
        input_num_tokens=rows,
        forward_mode=mode if rows else ForwardMode.IDLE,
        output_layout=(
            ForwardOutputLayout(0, 0, int(rows > 0), 1)
            if decode
            else ForwardOutputLayout(int(rows > 0), int(rows > 0), 0, 1)
        ),
        global_num_tokens=counts,
    )
    return attention(
        torch.arange(prefix, prefix + rows, device="cuda"),
        hidden[prefix : prefix + rows],
        ctx,
        comm,
        projection_out=None,
    )


@torch.no_grad()
def _worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from test.runtime.test_kimi_k31 import (
        _DSA_CONFIG,
        _assert_close_to_reference,
        _dense_gated_mla,
        _k31_attention_stack,
    )

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.linear import (
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )
    from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
    from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights
    from tokenspeed.runtime.models.kimi_k3_projection import (
        validate_projection_settings,
    )
    from tokenspeed.runtime.models.kimi_k31 import KimiK31SparseMLAAttention
    from tokenspeed.runtime.utils.env import global_server_args_dict

    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=120),
    )
    mapping = Mapping(
        rank=rank,
        world_size=size,
        attn_tp_size=1,
        attn_dp_size=size,
        linear_attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=size,
    )
    process_group_manager.register_process_group(
        "nccl", mapping.world_group, dist.group.WORLD
    )
    global_server_args_dict.update(
        mapping=mapping, chunked_prefill_size=8192, dense_gemm_backend="auto"
    )
    initialize_comm_backend(use_pynccl=False)
    torch.set_default_dtype(torch.bfloat16)
    heads = 12
    config = KimiLinearConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        q_lora_rank=128,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        dsa_config=dict(_DSA_CONFIG, dsa_layers=[4], index_topk=2048),
    )
    steps = ([5, 0, 2, 1][:size], [1, 1, 0, 1][:size], [1, 0, 1, 1][:size], [0] * size)
    for qkv_size, output_size in ((size, 1), (1, size), (size, size)):
        torch.manual_seed(5)
        whole, comm, router, pool = _k31_attention_stack(
            kv_cache_dtype=torch.bfloat16,
            spec_width=1,
            heads=heads,
            index_topk=2048,
            kvp_size=1,
            kvp_rank=0,
        )
        qkv_parallel, output_parallel = validate_projection_settings(
            mapping, qkv_size, output_size
        )
        with torch.device("cuda"):
            attention = KimiK31SparseMLAAttention(
                config=config,
                mapping=mapping,
                hidden_size=256,
                num_heads=heads,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                q_lora_rank=128,
                kv_lora_rank=512,
                max_position_embeddings=4096,
                quant_config=None,
                layer_id=3,
                prefix="model.layers.3.self_attn",
                reduce_attn_results=False,
                alt_stream=None,
                qkv_parallel=qkv_parallel,
                output_parallel=output_parallel,
            )
        target = dict(attention.named_parameters())
        for name, source in whole.named_parameters():
            if name == "fused_qkv_a_proj_with_mqa.weight" and qkv_parallel is not None:
                target[name].weight_loader(target[name], source)
            elif name == "o_proj.weight" and output_parallel is not None:
                target[name].weight_loader(target[name], source)
            else:
                target[name].copy_(source)
        attention.w_kc, attention.w_vc = _prepare_mla_kv_b_proj_weights(
            attention.kv_b_proj.weight, attention
        )
        bind_cache_groups(attention, pool)
        prepare_dp_linear_communication(attention, 16, torch.bfloat16, AutoBackend())
        hidden = torch.randn(
            sum(counts[rank] for counts in steps),
            256,
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(100 + rank),
        )
        reference = _dense_gated_mla(whole, hidden, heads)
        prefix = 0
        for step, counts in enumerate(steps):
            actual = _forward(
                attention, comm, router, pool, hidden, prefix, counts, rank, step > 0
            )
            expected = reference[prefix : prefix + counts[rank]]
            assert actual.shape == expected.shape
            if counts[rank]:
                _assert_close_to_reference(actual, expected, 2e-2)
            prefix += counts[rank]
        release_dp_linear_communication(attention)
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4], ids=["tp2", "tp4"])
def test_k31_projection_tp_runs_sparse_prefill_decode_and_empty_owners(size, tmp_path):
    if torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} GPUs")
    mp.spawn(
        _worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=size,
        join=True,
    )
