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

"""Small multi-GPU DSA DCP correctness check; no model weights required.

Single node: torchrun --standalone --nproc-per-node=2 (or 4/8) and this file.
Multiple nodes use the same entry point with torchrun's --nnodes,
--nproc-per-node, --node-rank and rendezvous options. --cases sink selects
only the BF16/packed-FP8 sink checks; --cases all also runs index selection.
All collectives use the production NCCL backend, without substitutions.
"""

from tokenspeed_kernel.ops.quantization import quantize_fp8


def main() -> None:
    import argparse
    import json
    import os
    from datetime import timedelta
    from pathlib import Path

    import torch
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", choices=("all", "indexer", "sink"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", help="Unique identifier shared by this run's ranks")
    args = parser.parse_args()
    if args.output_dir is not None and not args.run_id:
        parser.error(
            "--output-dir requires --run-id and a fresh directory for this run"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank), timeout=timedelta(minutes=5)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8, 16, 32) or world_size % local_world_size:
        raise ValueError("DSA DCP harness requires 2/4/8/16/32 equally placed ranks")
    group = tuple(range(world_size))
    process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.utils.env import global_server_args_dict

    global_server_args_dict.update(
        mapping=Mapping(
            rank=rank,
            world_size=world_size,
            attn_tp_size=world_size,
            attn_kvp_size=world_size,
            nprocs_per_node=local_world_size,
            nnodes=world_size // local_world_size,
        ),
        chunked_prefill_size=16,
        max_prefill_tokens=16,
        max_model_len=1024,
        max_num_seqs=4,
        speculative_algorithm=None,
    )
    # Pin the real vendor collective, so a single-node run does not silently
    # exercise an auto-selected symmetric-memory implementation instead.
    set_global_backend(NcclBackend())
    completed = []
    if args.cases in ("all", "sink"):
        completed.extend(_run_sink_attention(group, rank))
    if args.cases in ("all", "indexer"):
        _run_indexer(group, rank)
        completed.append("distributed_indexer_and_verify")
    # A rank writes success only after every rank reached the same boundary.
    dist.barrier()
    result = {
        "run_id": args.run_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "device": torch.cuda.get_device_name(),
        "collective_backend": "nccl",
        "completed": completed,
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite evidence from another launch. The collector must
        # check this run id, every expected rank, and every torchrun exit code.
        with (args.output_dir / f"rank-{rank:03d}.json").open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


def _run_indexer(group: tuple[int, ...], rank: int) -> None:
    import torch
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter

    from tokenspeed.runtime.layers.attention.dcp.indexer import select_dsa_topk
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    torch.manual_seed(912)
    page_size, topk, pages = 64, 512, 17
    keys = torch.randn(pages * page_size, 128, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    full = torch.zeros(pages * page_size, 132, device="cuda", dtype=torch.uint8)
    index_k_block_split_scatter(
        full,
        values,
        scales,
        torch.arange(keys.shape[0], device="cuda"),
        page_size=page_size,
        head_dim=128,
        group_size=128,
        write_mask=None,
    )
    q = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    table = torch.tensor(
        [[7, 2, 5, 1, 8, 3, 9, 4, 6, 10, 11, 12, 13, 14, 15, 16]],
        device="cuda",
        dtype=torch.int32,
    )
    requests = torch.zeros(3, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 87, 1007], device="cuda", dtype=torch.int32)
    kw = dict(
        page_size=page_size,
        topk=topk,
        softmax_scale=0.1,
        initial_tokens=4,
        local_tokens=8,
        max_logits_bytes=1 << 20,
    )
    reference, ref_lens = select_dsa_topk(
        q,
        w,
        full,
        table,
        requests,
        lengths,
        placement=CachePlacement(64, pages, (rank,), 0),
        **kw,
    )
    local = (
        full.reshape(pages, 64, 132)[[0] + list(range(rank + 1, pages, len(group)))]
        .reshape(-1, 132)
        .contiguous()
    )
    result, lens = select_dsa_topk(
        q,
        w,
        local,
        table,
        requests,
        lengths,
        placement=CachePlacement(64, pages, group, rank),
        **kw,
    )
    # Mandatory candidates have equal scores; compare sets, not tie order.
    torch.testing.assert_close(
        result.sort(dim=-1).values, reference.sort(dim=-1).values
    )
    torch.testing.assert_close(lens, ref_lens)
    # Compare the global selection with the existing unsharded native indexer.
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, dsa_plan

    seq2d = lengths.unsqueeze(1).contiguous()
    native_slots, native_lens = dsa_decode_topk(
        q,
        w,
        lengths,
        table.expand(3, -1).contiguous(),
        page_size=64,
        topk=topk,
        softmax_scale=0.1,
        batch_invariant=False,
        index_k_cache=full,
        seq_lens_2d=seq2d,
        plan=dsa_plan(page_size=64, seq_lens_2d=seq2d),
        solution="deep_gemm",
        slot_order="selection",
    )
    # Main's native indexer has no forced-window arguments. Compare its
    # unforced policy separately; the checks above exercise mandatory windows.
    unforced, unforced_lens = select_dsa_topk(
        q,
        w,
        local,
        table,
        requests,
        lengths,
        placement=CachePlacement(64, pages, group, rank),
        **{**kw, "initial_tokens": 0, "local_tokens": 0},
    )
    expected_slots = torch.where(
        unforced >= 0,
        table[0, unforced.clamp_min(0).long() // 64] * 64 + unforced % 64,
        -1,
    )
    torch.testing.assert_close(unforced_lens, native_lens)
    for row, count in enumerate(unforced_lens.tolist()):
        assert set(expected_slots[row, :count].tolist()) == set(
            native_slots[row, :count].tolist()
        )
    print(
        f"rank {rank}: distributed selection matched native unsharded DeepGEMM",
        flush=True,
    )
    # A second call with changed lengths verifies distributed masking and reuse.
    lengths.copy_(torch.tensor([25, 0, 511], device="cuda", dtype=torch.int32))
    reference, ref_lens = select_dsa_topk(
        q,
        w,
        full,
        table,
        requests,
        lengths,
        placement=CachePlacement(64, pages, (rank,), 0),
        **kw,
    )
    result, lens = select_dsa_topk(
        q,
        w,
        local,
        table,
        requests,
        lengths,
        placement=CachePlacement(64, pages, group, rank),
        **kw,
    )
    # Mandatory candidates have equal scores; compare sets, not tie order.
    torch.testing.assert_close(
        result.sort(dim=-1).values, reference.sort(dim=-1).values
    )
    torch.testing.assert_close(lens, ref_lens)
    print(
        f"rank {rank}: distributed DSA top-k matched unsharded, both batches",
        flush=True,
    )
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode

    from tokenspeed.runtime.layers.attention.dcp.comm import (
        combine_attention_partials,
        gather_query_heads,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    torch.manual_seed(173)
    all_q = torch.randn(3, 2 * len(group), 128, device="cuda", dtype=torch.bfloat16)
    latent = torch.randn(pages * 64, 128, device="cuda", dtype=torch.bfloat16)
    local_latent = (
        latent.reshape(pages, 64, 128)[[0] + list(range(rank + 1, pages, len(group)))]
        .reshape(-1, 128)
        .contiguous()
    )
    virtual_slots = torch.where(
        result >= 0, table[0, result.clamp_min(0).long() // 64] * 64 + result % 64, -1
    )
    local_slots, owned = resolve_cache_slots(
        virtual_slots, CachePlacement(64, pages, group, rank)
    )
    local_slots = torch.where(owned, local_slots, -1)
    kwargs = dict(
        sparse_kv_cache=None,
        topk_lens=lens,
        max_seqlen_k=1024,
        qk_nope_head_dim=128,
        kv_lora_rank=128,
        qk_rope_head_dim=0,
        softmax_scale=0.1,
        page_size=64,
        return_lse=True,
        slot_order="selection",
    )
    ref_out, _ = dsa_decode(
        q=all_q, kv_cache=latent, topk_slots=virtual_slots, **kwargs
    )
    gathered = gather_query_heads(
        all_q[:, rank * 2 : (rank + 1) * 2].contiguous(), group
    )
    partial, lse = dsa_decode(
        q=gathered, kv_cache=local_latent, topk_slots=local_slots, **kwargs
    )
    output = combine_attention_partials(
        partial, lse, group=group, rank=rank, sink=None, keep_all_heads=False
    )
    torch.testing.assert_close(
        output, ref_out[:, rank * 2 : (rank + 1) * 2], rtol=0.01, atol=0.005
    )
    print(f"rank {rank}: distributed attention/LSE merge matched unsharded", flush=True)
    # Verify three consecutive tokens after changing the visible frontier
    # across a page boundary. This exercises sparse forward and collectives;
    # test_dsa_decode_plan.py separately validates the draft metadata lifecycle.
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend

    backend = object.__new__(DSABackend)
    backend.kernel_page_size = 64
    backend.kernel_solution = "triton"
    backend.prefill_kernel_plan = None
    backend.decode_kernel_plan = None
    backend.data_type = torch.bfloat16
    backend.kv_lora_rank = 128
    backend.qk_nope_head_dim = 128
    backend.qk_rope_head_dim = 0
    backend.index_topk = topk
    backend.max_context_len = 1024
    backend.kvp_group = group
    backend.kvp_rank = rank
    backend.kvp_block_granularity = 64
    backend.kvp_virtual_block_count = pages
    # Each rank holds the attention-TP head slice (two heads); the sparse
    # decode gathers them over the KVP group and reduces in selection order.
    backend.num_local_heads = 2
    backend.num_attention_heads = 2 * len(group)
    backend.slot_order = "selection"
    end_length = torch.tensor([129], device="cuda", dtype=torch.int32)
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=end_length,
            max_seq_len_k=1024,
        )
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=128,
        v_head_dim=128,
        scaling=0.1,
        logit_cap=0.0,
        attn_sink=None,
    )
    pool = SimpleNamespace(quant_method=None, get_key_buffer=lambda _: local_latent)
    for end in (129, 132, 128):
        end_length.fill_(end)
        lengths.copy_(
            end_length + torch.arange(-2, 1, device="cuda", dtype=torch.int32)
        )
        reference, ref_lens = select_dsa_topk(
            q,
            w,
            full,
            table,
            requests,
            lengths,
            placement=CachePlacement(64, pages, (rank,), 0),
            **kw,
        )
        result, lens = select_dsa_topk(
            q,
            w,
            local,
            table,
            requests,
            lengths,
            placement=CachePlacement(64, pages, group, rank),
            **kw,
        )
        torch.testing.assert_close(result.sort(-1).values, reference.sort(-1).values)
        torch.testing.assert_close(lens, ref_lens)
        virtual_slots = torch.where(
            result >= 0,
            table[0, result.clamp_min(0).long() // 64] * 64 + result % 64,
            -1,
        )
        ref_out, _ = dsa_decode(
            q=all_q,
            kv_cache=latent,
            topk_slots=virtual_slots,
            **{**kwargs, "topk_lens": lens, "q_len_per_req": 3, "kv_seq_lens": lengths},
        )
        output = backend.forward_sparse_decode(
            q=all_q[:, rank * 2 : (rank + 1) * 2].contiguous(),
            layer=layer,
            token_to_kv_pool=pool,
            bs=1,
            topk_indices=virtual_slots,
            topk_lens=lens,
        ).reshape(3, 2, 128)
        torch.testing.assert_close(
            output, ref_out[:, rank * 2 : (rank + 1) * 2], rtol=0.01, atol=0.005
        )
    print(f"rank {rank}: DSA verify frontier edits matched unsharded", flush=True)


def _sink_reference(query, keys, slots, lengths, sink):
    """Independent FP32 softmax with one zero-value sink per head."""
    import torch

    valid = (slots >= 0) & (
        torch.arange(slots.shape[1], device=slots.device)[None] < lengths[:, None]
    )
    selected = keys.float()[slots.clamp_min(0).long()]
    logits = (
        torch.einsum("thd,tkd->thk", query.float(), selected) * 576**-0.5
    ).masked_fill(~valid[:, None, :], -torch.inf)
    lse = torch.logsumexp(logits, dim=-1)
    if sink is not None:
        lse = torch.logaddexp(lse, sink)
    weights = (logits - lse[..., None]).exp().nan_to_num()
    return torch.einsum("thk,tkd->thd", weights, selected[..., :512])


def _run_sink_attention(group: tuple[int, ...], rank: int) -> list[str]:
    import math

    import torch
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
    from tokenspeed_kernel.ops.kvcache.flashmla import set_flashmla_kv_buffer

    from tokenspeed.runtime.layers.attention.dcp.comm import (
        combine_attention_partials,
        gather_query_heads,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        CachePlacement,
        resolve_cache_slots,
    )

    torch.manual_seed(217)
    degree = len(group)
    page_size, local_heads, tokens, topk = 64, 3, 3, 512
    heads = degree * local_heads
    # Six/twelve/twenty-four actual heads pad to 64 on 2/4/8 ranks;
    # 32 ranks carry 96 actual heads and pad to 128. Every rank has two pages.
    pages = 1 + degree * 2
    keys = torch.randn((pages * page_size, 1, 576), device="cuda", dtype=torch.bfloat16)
    query = torch.randn((tokens, heads, 576), device="cuda", dtype=torch.bfloat16)
    keys[..., :512].add_(2)
    query.mul_(0.5)
    # Match the ordinary row's KV mass so counting the sink once per rank
    # instead of once globally produces a large, detectable error even at TP2.
    global_sink = torch.linspace(
        math.log(4 * degree) - 1,
        math.log(4 * degree) + 1,
        heads,
        device="cuda",
        dtype=torch.float32,
    )
    global_sink[1::local_heads] = -torch.inf
    global_sink[2::local_heads] = torch.inf
    head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
    selected_pages = [0] + list(range(rank + 1, pages, degree))
    placement = CachePlacement(page_size, pages, group, rank)
    completed = []
    for packed in (False, True):
        if packed:
            cache = torch.zeros(
                (pages * page_size, 1, 656), device="cuda", dtype=torch.uint8
            )
            set_flashmla_kv_buffer(
                cache,
                torch.arange(pages * page_size, device="cuda", dtype=torch.int64),
                keys[..., :512],
                keys[..., 512:],
                sanitize=False,
                write_mask=None,
            )
            # Read the public row bytes without using a production KV gather.
            quantized = (
                cache[:, 0, :512].view(torch.float8_e4m3fn).float().reshape(-1, 4, 128)
            )
            scales = cache[:, 0, 512:528].view(torch.float32).bfloat16()
            latent = (quantized * scales[..., None]).bfloat16().reshape(-1, 512)
            reference_keys = torch.cat((latent, keys[:, 0, 512:]), dim=-1)
        else:
            cache, reference_keys = keys, keys[:, 0]
        local_cache = (
            cache.reshape(pages, page_size, 1, -1)[selected_pages]
            .flatten(0, 1)
            .contiguous()
        )
        common = dict(
            max_seqlen_k=pages * page_size,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=576**-0.5,
            page_size=page_size,
            return_lse=True,
            solution="flashmla",
            slot_order="selection",
        )
        # The native BF16 no-sink decode provider is intentionally unchanged;
        # sparse prefill is the installed FlashMLA partial kernel for that form.
        attention = dsa_decode if packed else dsa_prefill
        if packed:
            common["q_len_per_req"] = tokens
        for empty_rank in (False, True):
            slots = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
            active = degree - int(empty_rank)
            chosen = torch.tensor(
                [
                    page * page_size + offset
                    for cycle in range(2)
                    for page in range(1 + cycle * degree, active + 1 + cycle * degree)
                    for offset in (0, 63)
                ],
                device="cuda",
                dtype=torch.int32,
            )
            slots[0, : chosen.numel()] = chosen
            slots[1, :2] = torch.tensor([64, 65], device="cuda")
            # Deliberately valid padding and an all-empty query exercise lengths.
            slots[1, 2] = degree * page_size
            lengths = torch.tensor(
                [chosen.numel(), 2, 0], device="cuda", dtype=torch.int32
            )
            local_slots, owned = resolve_cache_slots(slots, placement)
            local_slots = torch.where(owned, local_slots, -1)
            if empty_rank and rank == degree - 1:
                assert not (
                    (local_slots >= 0)
                    & (torch.arange(topk, device="cuda")[None] < lengths[:, None])
                ).any()
            for keep_all_heads in (False, True):
                q = (
                    query
                    if keep_all_heads
                    else gather_query_heads(query[:, head_slice].contiguous(), group)
                )
                torch.testing.assert_close(q, query, rtol=0, atol=0)
                partial, local_lse = attention(
                    q=q,
                    kv_cache=None if packed else local_cache,
                    sparse_kv_cache=local_cache if packed else None,
                    topk_slots=local_slots,
                    topk_lens=lengths,
                    **common,
                )
                assert not partial[2].any() and torch.isneginf(local_lse[2]).all()
                for with_sink in (False, True):
                    sink = global_sink if with_sink else None
                    returned_sink = (
                        sink
                        if keep_all_heads or sink is None
                        else sink[head_slice].contiguous()
                    )
                    actual = combine_attention_partials(
                        partial,
                        local_lse,
                        group=group,
                        rank=rank,
                        sink=returned_sink,
                        keep_all_heads=keep_all_heads,
                    )
                    expected = _sink_reference(
                        query, reference_keys, slots, lengths, sink
                    )
                    full_attention = dsa_decode if packed or with_sink else dsa_prefill
                    full_common = dict(common)
                    if not packed and with_sink:
                        full_common["q_len_per_req"] = tokens
                    native, _ = full_attention(
                        q=query,
                        kv_cache=None if packed else cache,
                        sparse_kv_cache=cache if packed else None,
                        topk_slots=slots,
                        topk_lens=lengths,
                        attn_sink=sink,
                        **full_common,
                    )
                    torch.testing.assert_close(
                        native.float(), expected, atol=0.02, rtol=0.02
                    )
                    if not keep_all_heads:
                        expected = expected[:, head_slice]
                        native = native[:, head_slice]
                    torch.testing.assert_close(
                        actual.float(), expected, atol=0.02, rtol=0.02
                    )
                    torch.testing.assert_close(actual, native, atol=0.02, rtol=0.02)
                    assert torch.isfinite(actual).all() and not actual[2].any()
                    if with_sink:
                        assert not actual[:, 2::local_heads].any()
                    completed.append(
                        f"{'fp8' if packed else 'bf16'}_empty_rank{int(empty_rank)}_"
                        f"replicated{int(keep_all_heads)}_sink{int(with_sink)}"
                    )
    print(
        f"rank {rank}: {len(completed)} NCCL sink/control cases matched FP32 reference",
        flush=True,
    )
    return completed


if __name__ == "__main__":
    main()
