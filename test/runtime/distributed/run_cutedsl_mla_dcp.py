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

"""CuTe MLA DCP attention against unsharded attention, using real collectives.

From the repository root, run with PYTHONPATH=. and
python -m torch.distributed.run --standalone --nproc-per-node=4
test/runtime/distributed/run_cutedsl_mla_dcp.py --dcp-size 2.
Use --dcp-size 4 to cover a single DCP group across all four TP ranks.
Covers BF16/FP8, decode/verify/draft, empty shards, eager/graph replay,
and chunked prefill with owner-masked writes to a real cache pool.
"""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.registry import initialize_comm_backend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.attention.backends.paged import tokenspeed_mla
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.utils.env import global_server_args_dict


def run_case(*, rank, mapping, context, dtype, queries, draft, block):
    degree = mapping.dcp_size
    device = torch.device("cuda", rank)
    heads, latent, dim, page, granularity = 16, 512, 576, 64, 128
    blocks = (context + granularity - 1) // granularity + 1
    blocks = (blocks + degree - 1) // degree * degree
    batch = 3
    num_extends = int(queries > 1 and not draft)
    spec_queries = 4 if draft else queries
    spec = MLAConfig(
        backend_name="tokenspeed_mla",
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
        kv_cache_dtype=dtype,
        kv_cache_quant_method="",
        prefix_granularity=granularity,
        context_len=blocks * granularity,
        max_bs=batch + num_extends,
        speculative_num_draft_tokens=spec_queries,
        is_draft=draft,
        draft_block_decode=block,
        dcp_size=degree,
        dcp_rank=mapping.dcp_rank,
        dcp_group=mapping.dcp_group,
        components=(spec,),
    )
    # Skip unrelated prefill compilation. Decode, metadata kernels and
    # communications below are the actual implementations.
    with patch.object(tokenspeed_mla, "warmup_compile_prefill", lambda **kw: None):
        leaf = tokenspeed_mla.CuteDSLMLABackend(config, spec, kernel_page_size=page)
    leaf.set_cache_pool(object())
    leaf.configure_runtime(
        block_granularity=granularity,
        virtual_block_count=blocks + 1,
        shard_count=degree,
    )
    leaf.init_cuda_graph_state(batch + num_extends)
    torch.manual_seed(173)
    full_cache = torch.randn(
        blocks + 1, granularity, dim, device=device, dtype=torch.bfloat16
    ).to(dtype)
    full_cache[0].zero_()
    local_cache = full_cache[
        [0] + list(range(mapping.dcp_rank + 1, blocks + 1, degree))
    ].contiguous()
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: local_cache)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        head_dim=dim,
        v_head_dim=latent,
        scaling=spec.scaling,
        sliding_window_size=-1,
        k_scale_float=1.0,
    )
    # Nonconsecutive physical blocks: logical page position is not its owner.
    order = torch.cat(
        (
            torch.ones(1, device=device, dtype=torch.int32),
            torch.arange(blocks, 1, -1, device=device, dtype=torch.int32),
        )
    )
    pages = (order[:, None] * 2 + torch.arange(2, device=device)).flatten().int()
    table = pages[None].expand(batch, -1).clone()
    table[-1].zero_()
    ends = torch.tensor([context - 1, 8, 0], device=device, dtype=torch.int32)
    all_q = torch.randn(
        batch,
        queries,
        heads * mapping.tp_size,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = (
        all_q[:, :, mapping.tp_rank * heads : (mapping.tp_rank + 1) * heads]
        .contiguous()
        .flatten(0, 1)
    )

    def refresh():
        # A mixed round places extend rows before the decode slice consumed
        # here; this verifies that DCP uses the same request offset.
        round_ends = torch.cat((ends.new_zeros(num_extends), ends))
        round_table = torch.cat((table.new_zeros((num_extends, table.shape[1])), table))
        leaf.refresh_decode_metadata(
            batch + num_extends,
            batch - 1 + num_extends,
            round_ends,
            round_table,
            num_extends=num_extends,
        )

    def forward():
        return leaf.forward_decode(q, None, None, layer, None, pool, batch).view(
            batch, queries, heads, latent
        )

    def reference():
        offsets = (
            torch.zeros(queries, device=device, dtype=torch.int32)
            if block
            else torch.arange(1 - queries, 1, device=device, dtype=torch.int32)
        )
        visible = (ends[:, None] + offsets).clamp_min(0)
        return tokenspeed_mla.tokenspeed_mla_decode(
            query=q.view(batch, queries, heads, dim).to(dtype),
            kv_cache=full_cache.view(-1, page, dim),
            workspace_buffer=leaf._cutedsl_workspace(queries),
            kv_lora_rank=latent,
            qk_rope_head_dim=64,
            block_tables=table,
            seq_lens=ends,
            max_seq_len=blocks * granularity,
            softmax_scale=spec.scaling,
            causal_mask=not block,
            local_visible_lens=visible,
        )

    refresh()
    # Warm up JIT and collective state before capture.
    for _ in range(3):
        actual = forward()
    expected = reference()
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
    assert not actual[-1].any()
    if mapping.dcp_rank > 0:
        assert (
            leaf.forward_decode_metadata.dcp.local_seq_lens[num_extends + 1].item() == 0
        )
    torch.cuda.synchronize()
    leaf._workspace_pool.freeze()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = forward()
    # A later round crosses a scheduler block boundary; graph pointers persist.
    ends[0] = context + 1
    refresh()
    graph.replay()
    expected = reference()
    torch.testing.assert_close(replay_output, expected, atol=0.003, rtol=0.03)
    assert not replay_output[-1].any()
    # Draft updates change visibility without rebuilding the compact table.
    if draft:
        ends[0] = context - 3
        if block:
            leaf.fill_block_decode_seq_lens(batch, ends)
        else:
            leaf.advance_draft_forward_metadata(ends)
        graph.replay()
        expected = reference()
        torch.testing.assert_close(replay_output, expected, atol=0.003, rtol=0.03)
    leaf._workspace_pool.unfreeze()
    if mapping.dcp_rank == 0:
        print(
            f"PASS group={mapping.dcp_group} dtype={dtype} context={context} Q={queries} draft={draft} "
            f"block={block}",
            flush=True,
        )


def _make_prefill_backend(
    *, spec, rank, mapping, dtype, blocks, granularity, context_len, sharded
):
    """Build a CuTe leaf and physical cache for a sharded or reference run."""
    from dataclasses import replace
    from test.runtime.cache_pool_test_utils import (
        make_arena,
        make_mla_memory_plan,
        plan_group_specs,
    )

    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )

    device = torch.device("cuda", rank)
    shards = mapping.dcp_size if sharded else 1
    plan = make_mla_memory_plan(
        size=blocks // shards * granularity,
        prefix_granularity=granularity,
        layer_num=1,
        latent_width=spec.kv_cache_dim,
        dtype=dtype,
    )
    arena = make_arena(
        plan,
        device,
        cache_group_specs=tuple(
            replace(g, shard_count=shards) for g in plan_group_specs(plan)
        ),
    )
    pool = HybridKDATokenToKVPool(
        arena=arena,
        layer_types=("full_attention",),
        model_dtype=torch.bfloat16,
        dtype=dtype,
        quant_method=None,
        kv_lora_rank=spec.kv_lora_rank,
        qk_rope_head_dim=spec.qk_rope_head_dim,
        layer_num=1,
        rank=rank,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=dtype,
        kv_cache_quant_method="",
        prefix_granularity=granularity,
        context_len=context_len,
        max_bs=3,
        speculative_num_draft_tokens=1,
        is_draft=False,
        draft_block_decode=False,
        dcp_size=shards,
        dcp_rank=mapping.dcp_rank if sharded else 0,
        dcp_group=mapping.dcp_group if sharded else (rank,),
        components=(spec,),
    )
    with patch.object(tokenspeed_mla, "warmup_compile_prefill", lambda **kw: None):
        leaf = tokenspeed_mla.CuteDSLMLABackend(config, spec, kernel_page_size=64)
    leaf.set_cache_pool(pool)
    leaf.configure_runtime(
        block_granularity=granularity,
        virtual_block_count=blocks + 1,
        shard_count=shards,
    )
    return leaf, pool


def run_prefill_case(*, rank, mapping, context, dtype):
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

    device = torch.device("cuda", rank)
    degree = mapping.dcp_size
    granularity, heads, latent, dim = 128, 16, 512, 576
    prefix = torch.tensor([context - 3, 128, 0], dtype=torch.int32)
    extend = torch.tensor([7, 5, 3], dtype=torch.int32)
    per_request = (context + 7 + granularity - 1) // granularity
    blocks = (3 * per_request + degree - 1) // degree * degree
    global_server_args_dict.update(
        chunked_prefill_size=min(4096, context // 2), mla_chunk_multiplier=1
    )
    spec = MLAConfig(
        backend_name="tokenspeed_mla",
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

    leaf, pool = _make_prefill_backend(
        spec=spec,
        rank=rank,
        mapping=mapping,
        dtype=dtype,
        blocks=blocks,
        granularity=granularity,
        context_len=per_request * granularity,
        sharded=True,
    )
    reference_leaf, reference_pool = _make_prefill_backend(
        spec=spec,
        rank=rank,
        mapping=mapping,
        dtype=dtype,
        blocks=blocks,
        granularity=granularity,
        context_len=per_request * granularity,
        sharded=False,
    )
    layer = PagedAttention(
        num_heads=heads,
        head_dim=192,
        scaling=spec.scaling,
        num_kv_heads=heads,
        layer_id=0,
        v_head_dim=128,
        rotary_emb=None,
        qk_norm=None,
    )
    torch.manual_seed(312)
    history = torch.randn(
        blocks * granularity, 1, dim, device=device, dtype=torch.bfloat16
    )
    loc = torch.arange(granularity, (blocks + 1) * granularity, device=device)
    reference_pool.get_key_buffer(0).zero_()
    pool.get_key_buffer(0).zero_()
    reference_pool.set_mla_kv_buffer(
        layer, loc, history[..., :latent], history[..., latent:], write_mask=None
    )
    slots, mask = resolve_cache_slots(loc, leaf.cache_placement(layer))
    pool.set_mla_kv_buffer(
        layer, slots, history[..., :latent], history[..., latent:], write_mask=mask
    )
    order = (
        torch.randperm(blocks, device=device)[: 3 * per_request].view(3, per_request)
        + 1
    )
    pages = (order[..., None] * 2 + torch.arange(2, device=device)).reshape(3, -1).int()
    positions = torch.cat(
        [
            torch.arange(p, p + n, device=device)
            for p, n in zip(prefix.tolist(), extend.tolist())
        ]
    )
    request_ids = torch.repeat_interleave(
        torch.arange(3, device=device), extend.to(device).long()
    )
    write_locs = (
        order[request_ids, positions // granularity] * granularity
        + positions % granularity
    ).long()
    new_latent = torch.randn(
        int(extend.sum()), dim, device=device, dtype=torch.bfloat16
    )
    # TP projections/queries differ by rank; latent KV is shared across DCP.
    torch.manual_seed(713 + rank)
    q = torch.randn(int(extend.sum()), heads * 192, device=device, dtype=torch.bfloat16)
    weight = (
        torch.randn(latent, heads * 256, device=device, dtype=torch.bfloat16)
        / latent**0.5
    )
    model = SimpleNamespace(
        kv_lora_rank=latent,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        qk_head_dim=192,
        num_local_heads=heads,
        v_head_dim=128,
        kv_b_proj=lambda x: (x @ weight,),
        attn_mha=layer,
    )
    for backend in (leaf, reference_leaf):
        backend._init_prefill_metadata(
            (prefix + extend).to(device),
            pages,
            prefix.to(device),
            prefix,
            extend.to(device),
            extend,
        )
    assert leaf.chunked_prefill_metadata.chunked_loop_num > 1
    assert (
        max(t.numel() for t in leaf.chunked_prefill_metadata.chunk_kv_indices_list)
        <= global_server_args_dict["chunked_prefill_size"]
    )

    def run(backend, cache):
        ctx = SimpleNamespace(attn_backend=backend, token_to_kv_pool=cache)
        prepared = DeepseekV3AttentionMLA.forward_normal_chunked_kv_prepare(
            model,
            positions,
            q.clone(),
            new_latent.clone(),
            ctx,
            write_locs,
        )
        output = torch.empty(
            int(extend.sum()), heads * 128, device=device, dtype=torch.bfloat16
        )
        return DeepseekV3AttentionMLA.forward_normal_chunked_kv_core(
            model,
            prepared.query,
            prepared.key,
            prepared.value,
            ctx,
            output,
        )

    expected = run(reference_leaf, reference_pool)
    actual = run(leaf, pool)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
    # Compare every physical row after model-side writes, including untouched
    # reserve slots and null page: foreign writes must not corrupt local KV.
    full = reference_pool.get_key_buffer(0).view(blocks + 1, granularity, 1, dim)
    expected_cache = full[[0] + list(range(mapping.dcp_rank + 1, blocks + 1, degree))]
    torch.testing.assert_close(
        pool.get_key_buffer(0).float(),
        expected_cache.reshape(-1, 1, dim).float(),
        atol=0,
        rtol=0,
    )
    if mapping.dcp_rank == 0:
        print(
            f"PASS prefill group={mapping.dcp_group} dtype={dtype} context={context} chunks={leaf.chunked_prefill_metadata.chunked_loop_num}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcp-size", type=int, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 65536])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world_size = dist.get_world_size()
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=world_size,
        attn_cp_size=1,
        attn_dp_size=1,
        attn_dcp_size=args.dcp_size,
    )
    process_group_manager.register_process_group(
        "nccl", mapping.world_group, dist.group.WORLD
    )
    process_group_manager.init_process_group(mapping.attn.dcp_group, backend="nccl")
    global_server_args_dict.update(
        mapping=mapping,
        chunked_prefill_size=64,
        max_prefill_tokens=64,
        max_model_len=max(args.contexts),
        max_num_seqs=3,
        speculative_algorithm=None,
    )
    initialize_comm_backend(use_pynccl=False)
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for context in args.contexts:
            for queries, draft, block in (
                (1, False, False),
                (4, False, False),
                (1, True, False),
                (4, True, True),
            ):
                run_case(
                    rank=rank,
                    mapping=mapping.attn,
                    context=context,
                    dtype=dtype,
                    queries=queries,
                    draft=draft,
                    block=block,
                )
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for context in args.contexts:
            run_prefill_case(
                rank=rank, mapping=mapping.attn, context=context, dtype=dtype
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
