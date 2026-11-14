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

"""A long DSA sparse prefill attends every head on a row shard (2 GPUs).

Each rank holds the same K3.1 latent rows and half of 96 query heads. Past
the row threshold the backend hands each rank half of the rows with every
head (padded to the core's 128), attends them and hands the heads back; each
rank must get the attention the plain path gives its heads. A prefill under
the threshold keeps the plain path.
"""

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
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=90,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the row-shard sparse prefill runs on NVIDIA only",
)

_HEADS = 96
_ROWS = 300


def _leaf(pool, rank: int):
    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    spec = DSAConfig(
        backend_name="dsa",
        num_attention_heads=_HEADS,
        num_kv_heads=1,
        head_dim=192,
        attn_tp_size=2,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_k_format="fp8_scaled",
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kernel_page_size=None,
        context_len=4096,
        max_bs=4,
        kv_cache_quant_method=None,
        speculative_num_draft_tokens=1,
        attn_tp_rank=rank,
        attn_tp_group=(0, 1),
        components=(spec,),
    )
    leaf = DSABackend(
        config,
        spec,
        kernel_page_size=DSABackend.resolve_kernel_page_size(config, 128),
    )
    leaf.set_cache_pool(pool)
    leaf.init_cuda_graph_state(max_bs=4)
    return leaf


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    import test.runtime.test_kimi_k31 as k31

    import tokenspeed.runtime.layers.attention.backends.paged.dsa as dsa
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        set_global_backend,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=90),
    )
    try:
        process_group_manager.register_process_group("nccl", (0, 1), dist.group.WORLD)
        set_global_backend(AutoBackend())
        torch.manual_seed(5)
        pool, _ = k31._k31_pool("cuda", num_lcm_blocks=1)
        local_heads = _HEADS // 2
        layer = k31._mla_layer(local_heads)
        slots = k31._slots([4, 11, 1], _ROWS, "cuda")
        latent = torch.randn(_ROWS, 1, 576, device="cuda", dtype=torch.bfloat16)
        pool.set_mla_kv_buffer(
            layer, slots, latent[..., :512], latent[..., 512:], write_mask=None
        )
        leaf = _leaf(pool, rank)
        visible = torch.arange(1, _ROWS + 1, device="cuda", dtype=torch.int32)
        topk = torch.full((_ROWS, 2048), -1, dtype=torch.int32, device="cuda")
        for row, count in enumerate(visible.tolist()):
            topk[row, :count] = slots[:count].to(torch.int32)
        heads = slice(rank * local_heads, (rank + 1) * local_heads)
        q = torch.randn(_ROWS, _HEADS, 576, device="cuda", dtype=torch.bfloat16)
        q = q[:, heads].contiguous()

        exchanged = []
        exchange = dsa.exchange_query_rows_padded

        def counted(query, *args):
            exchanged.append(query.shape[0])
            return exchange(query, *args)

        dsa.exchange_query_rows_padded = counted

        def attend():
            return (
                leaf.forward_sparse_prefill(
                    q=q,
                    layer=layer,
                    token_to_kv_pool=pool,
                    kv_seq_lens=visible,
                    topk_slots=topk,
                    topk_lens=visible,
                    max_seq_len=_ROWS,
                )
                .view(_ROWS, local_heads, 512)
                .float()
            )

        dsa._ROW_SHARD_MIN_ROWS = _ROWS + 1
        plain = attend()
        assert exchanged == []
        dsa._ROW_SHARD_MIN_ROWS = _ROWS
        sharded = attend()
        assert exchanged == [_ROWS]
        k31._assert_attention_close(sharded, plain)
    finally:
        dist.destroy_process_group()


def test_row_shard_prefill_matches_the_plain_prefill(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=True)
