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

"""Gluon MLA decode/verify against dense FP32 attention with real RCCL.

Run from the repository root with the runtime/kernel packages installed:
PYTHONPATH=. python -m torch.distributed.run --standalone --nproc-per-node=8 \
    test/runtime/distributed/run_gluon_mla_dcp.py --kvp-size 8

This is a correctness check, not a throughput benchmark or a model accuracy test.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from test.runtime.cache_pool_test_utils import (
    make_arena,
    make_mla_memory_plan,
    plan_group_specs,
)

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import normalize_dcp_partials
from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache

from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME, Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.mla import GluonMLAAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils.env import global_server_args_dict


def run_case(
    *, rank, mapping, context, queries, draft, block, query_dtype, coordination
):
    device = torch.device("cuda", rank)
    degree = mapping.kvp_size
    granularity, page, heads, latent, dim = 128, 64, 12, 512, 576
    blocks = (
        ((context + 2 * granularity - 1) // granularity + degree - 1) // degree * degree
    )
    spec = MLAConfig(
        backend_name="gluon",
        num_attention_heads=heads * mapping.tp_size,
        num_kv_heads=1,
        head_dim=dim,
        attn_tp_size=mapping.tp_size,
        kv_lora_rank=latent,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=dim,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_quant_method="none",
        context_len=blocks * granularity,
        max_bs=3,
        speculative_num_draft_tokens=4 if draft else queries,
        is_draft=draft,
        draft_block_decode=block,
        kvp_size=degree,
        kvp_rank=mapping.kvp_rank,
        kvp_group=mapping.kvp_group,
        components=(spec,),
    )
    plan = make_mla_memory_plan(
        size=blocks // degree * granularity,
        prefix_granularity=granularity,
        layer_num=1,
        latent_width=dim,
        dtype=config.kv_cache_dtype,
    )
    arena = make_arena(
        plan,
        device,
        cache_group_specs=tuple(
            replace(group, shard_count=degree) for group in plan_group_specs(plan)
        ),
    )
    pool = HybridKDATokenToKVPool(
        arena=arena,
        layer_types=("full_attention",),
        model_dtype=config.dtype,
        dtype=config.kv_cache_dtype,
        quant_method=None,
        kv_lora_rank=latent,
        qk_rope_head_dim=64,
        layer_num=1,
        rank=rank,
    )
    leaf = GluonMLAAttnBackend(config, spec, kernel_page_size=page)
    leaf.set_cache_pool(pool)
    leaf.configure_runtime(
        block_granularity=granularity,
        virtual_block_count=blocks + 1,
        shard_count=degree,
    )
    leaf.init_cuda_graph_state(3)
    layer = PagedAttention(
        num_heads=heads,
        head_dim=dim,
        scaling=spec.scaling,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=latent,
        rotary_emb=None,
        qk_norm=None,
    )
    torch.manual_seed(173)
    full_cache = (torch.randn(blocks + 1, granularity, dim, device=device) * 0.1).to(
        config.kv_cache_dtype
    )
    full_cache[0].zero_()
    owned = [0] + list(range(mapping.kvp_rank + 1, blocks + 1, degree))
    pool.get_key_buffer(0).copy_(full_cache[owned].reshape_as(pool.get_key_buffer(0)))
    # Allocation IDs deliberately differ from logical page positions.
    order = torch.cat(
        (
            torch.ones(1, device=device, dtype=torch.int32),
            torch.arange(blocks, 1, -1, device=device, dtype=torch.int32),
        )
    )
    pages = (order[:, None] * 2 + torch.arange(2, device=device)).flatten().int()
    table = pages[None].expand(3, -1).clone()
    table[-1].zero_()
    lengths = torch.tensor([context - 1, 8, 0], dtype=torch.int32, device=device)
    all_q = (
        torch.randn(3, queries, heads * mapping.tp_size, dim, device=device) * 0.1
    ).to(query_dtype)
    q = (
        all_q[:, :, mapping.tp_rank * heads : (mapping.tp_rank + 1) * heads]
        .contiguous()
        .flatten(0, 1)
    )

    def refresh():
        leaf.refresh_decode_metadata(3, 2, lengths, table)

    def forward():
        if draft:
            if block:
                leaf.fill_block_decode_seq_lens(3, lengths)
            else:
                leaf.advance_draft_forward_metadata(lengths)
        return leaf.forward_decode(q, None, None, layer, None, pool, 3).view(
            3, queries, heads, latent
        )

    def check(actual):
        expected = torch.zeros_like(actual, dtype=torch.float32)
        # Use the same quantized Q/KV values but independent dense FP32 softmax.
        for request in range(2):
            for query in range(queries):
                end = (
                    int(lengths[request])
                    if block
                    else max(0, int(lengths[request]) - queries + query + 1)
                )
                slots = torch.arange(end, device=device)
                physical = table[request, slots // page].long() * page + slots % page
                history = full_cache.view(-1, dim)[physical].float()
                scores = (
                    q.view(3, queries, heads, dim)[request, query].float()
                    @ history.T
                    * spec.scaling
                )
                expected[request, query] = scores.softmax(-1) @ history[:, :latent]
        offsets = (
            torch.zeros(queries, dtype=torch.int32, device=device)
            if block
            else torch.arange(1 - queries, 1, dtype=torch.int32, device=device)
        )
        visible = (lengths[:, None] + offsets).clamp_min(0).reshape(-1)
        baseline, baseline_lse = mla_decode_with_kvcache(
            q=q.view(-1, 1, heads, dim),
            kv_cache=full_cache.view(-1, page, 1, dim),
            page_table=table.repeat_interleave(queries, dim=0),
            cache_seqlens=visible,
            max_seqlen_k=blocks * granularity,
            qk_nope_head_dim=128,
            kv_lora_rank=latent,
            qk_rope_head_dim=64,
            softmax_scale=spec.scaling,
            return_lse=True,
            solution="gluon",
        )
        baseline, _ = normalize_dcp_partials(
            baseline.view(-1, heads, latent),
            baseline_lse.view(-1, heads),
            visible,
            None,
        )
        baseline = baseline.view_as(actual).float()
        norm = torch.linalg.vector_norm(expected, dim=(-2, -1)).clamp_min(1e-12)
        dense_error = (
            torch.linalg.vector_norm(actual.float() - expected, dim=(-2, -1)) / norm
        )
        baseline_error = (
            torch.linalg.vector_norm(baseline - expected, dim=(-2, -1)) / norm
        )
        exchange_error = (
            torch.linalg.vector_norm(actual.float() - baseline, dim=(-2, -1)) / norm
        )
        # The native FP8 kernel already rounds softmax probabilities. Use its
        # existing 5% relative-L2 contract per query, and independently bound
        # the additional sharding/merge error against the no-DCP kernel.
        if mapping.kvp_rank == 0:
            print(
                f"NUMERICS relative-L2 dense={dense_error.max().item():.6g} TP={baseline_error.max().item():.6g} DCP-vs-TP={exchange_error.max().item():.6g}",
                flush=True,
            )
        assert (
            dense_error.max() < 0.05
        ), f"rank={rank} dense errors={dense_error.tolist()}"
        # Context partitioning changes FP8 probability rounding. Compare each
        # result's error against dense attention, allowing at most one extra
        # percentage point over TP, rather than requiring two independently
        # rounded outputs to agree more closely than their own FP8 accuracy.
        assert (
            dense_error <= baseline_error + 0.01
        ).all(), f"rank={rank} DCP={dense_error.tolist()} TP={baseline_error.tolist()}"
        assert not actual[-1].any()

    refresh()
    for _ in range(3):
        result = forward()
    check(result)
    torch.cuda.synchronize()
    dist.barrier(group=coordination)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = forward()
    try:
        lengths[0] = context + 1
        refresh()
        graph.replay()
        check(replayed)
        if draft:
            lengths[0] = context - 3
            graph.replay()
            check(replayed)
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        graph.reset()
    if mapping.kvp_rank == 0:
        print(
            f"PASS context={context} Q={queries} draft={draft} block={block} query_dtype={query_dtype} group={mapping.kvp_group}",
            flush=True,
        )
    if queries == 1 and not draft:
        prefix = torch.tensor([context - 3, 8, 0], dtype=torch.int32)
        extend = torch.ones(3, dtype=torch.int32)
        leaf.init_forward_metadata(
            3,
            3,
            (prefix + extend).to(device),
            table,
            ForwardMode.EXTEND,
            extend_seq_lens=extend.to(device),
            extend_seq_lens_cpu=extend,
            extend_prefix_lens=prefix.to(device),
            extend_prefix_lens_cpu=prefix,
            extend_with_prefix=True,
            query_shard=None,
            page_table_cpu=table.cpu(),
        )
        metadata = leaf.chunked_prefill_metadata
        assert not metadata.use_absorbed_cached_extend
        history = leaf.read_prefix_history(
            pool, layer, metadata.chunk_kv_indices_list, dst_dtype=torch.bfloat16
        )
        loops = 0
        for slots, (latent_rows, rope_rows) in zip(
            metadata.chunk_kv_indices_list, history, strict=True
        ):
            restored = torch.cat(
                (latent_rows.reshape(-1, latent), rope_rows.reshape(-1, 64)), dim=-1
            )
            reference = full_cache.view(-1, dim)[slots.long()].to(torch.bfloat16)
            torch.testing.assert_close(restored, reference, atol=0, rtol=0)
            loops += 1
        assert loops > 0
        if mapping.kvp_rank == 0:
            print(f"PASS prefix-history context={context} loops={loops}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvp-size", type=int, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 65536])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        mapping = Mapping(
            rank=rank,
            world_size=dist.get_world_size(),
            attn_tp_size=dist.get_world_size(),
            attn_kvp_size=args.kvp_size,
        )
        process_group_manager.register_process_group(
            "nccl", mapping.world_group, dist.group.WORLD
        )
        process_group_manager.init_process_group(mapping.attn.kvp_group, backend="nccl")
        process_group_manager.init_process_group(
            mapping.attn.kvp_group, backend="nccl", name=KVP_HISTORY_GROUP_NAME
        )
        coordination = dist.new_group(backend="gloo")
        global_server_args_dict.update(
            mapping=mapping,
            numerics="auto",
            chunked_prefill_size=8192,
            max_prefill_tokens=8192,
            max_model_len=max(args.contexts) + 256,
            max_num_seqs=3,
            speculative_algorithm=None,
        )
        backend = NcclBackend()
        backend.configure(use_pynccl=False)
        set_global_backend(backend)
        for context in args.contexts:
            for query_dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for queries, draft, block in [
                    (1, False, False),
                    (4, False, False),
                    (1, True, False),
                    (4, True, True),
                ]:
                    run_case(
                        rank=rank,
                        mapping=mapping.attn,
                        context=context,
                        queries=queries,
                        draft=draft,
                        block=block,
                        query_dtype=query_dtype,
                        coordination=coordination,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
