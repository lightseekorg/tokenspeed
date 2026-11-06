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

Run with torchrun --standalone --nproc-per-node=2 (or 4/8) and this file.
Exercises the production collectives, candidate selection and attention merge.
"""

from tokenspeed_kernel.ops.quantization import quantize_fp8


def check_attention_merge(group) -> None:
    import torch
    import torch.distributed as dist
    from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
        dcp_apply_sink,
        dcp_weight_for_reduce_scatter,
    )

    from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend
    from tokenspeed.runtime.distributed.comm_ops import reduce_scatter
    from tokenspeed.runtime.layers.attention.dcp.comm import combine_attention_partials

    rank = dist.get_rank()
    backend = get_global_backend()
    for tokens in (1, 4, 16):
        torch.manual_seed(981 + rank)
        output = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.bfloat16)
        # Strided FP32 LSE, including empty shards and non-finite entries.
        lse = torch.randn(tokens, 128, device="cuda")[:, ::2]
        lse[0, :4] = torch.tensor(
            [-torch.inf, torch.inf, float("nan"), -0.0], device="cuda"
        )
        sink = torch.randn(64 // len(group), device="cuda")

        def merge():
            return combine_attention_partials(
                output, lse, group=group, rank=rank, sink=sink, keep_all_heads=False
            )

        def reference():
            gathered = backend.nccl.all_gather(
                lse.unsqueeze(0).contiguous(), group, dim=0
            )
            weighted, global_lse = dcp_weight_for_reduce_scatter(output, gathered, rank)
            reduced = reduce_scatter(weighted, group).movedim(0, 1)
            return dcp_apply_sink(reduced, global_lse, sink, dtype=output.dtype)

        expected = reference()
        torch.testing.assert_close(merge(), expected, rtol=0, atol=0, equal_nan=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = merge()
        for _ in range(3):
            lse.add_(0.125 * (rank + 1))
            graph.replay()
            torch.testing.assert_close(
                actual, reference(), rtol=0, atol=0, equal_nan=True
            )
    print(
        f"rank {rank}: DCP merge and graph replay matched FP32 NCCL reference",
        flush=True,
    )


def main() -> None:
    import os

    import torch
    import torch.distributed as dist
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter

    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.indexer import (
        select_dsa_topk,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    group = tuple(range(dist.get_world_size()))
    process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.utils.env import global_server_args_dict

    global_server_args_dict.update(
        mapping=Mapping(rank=rank, world_size=len(group)),
        chunked_prefill_size=16,
        max_prefill_tokens=16,
        max_model_len=1024,
        max_num_seqs=4,
        speculative_algorithm=None,
    )
    initialize_comm_backend(use_pynccl=False)
    check_attention_merge(group)
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
    for row, count in enumerate(lens.tolist()):
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
    # Verify three consecutive tokens, then advance/re-anchor across a page
    # boundary. Use the actual DSA backend and production collectives.
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
    # A three-row verify without causal lengths: decode derives them from seq_lens_k.
    backend.is_draft = False
    backend.spec_num_tokens = 3
    end_length = torch.tensor([129], device="cuda", dtype=torch.int32)
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=end_length,
            max_seq_len_k=1024,
            _dsa_kv_seq_lens=None,
        )
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=128,
        v_head_dim=128,
        scaling=0.1,
        logit_cap=0.0,
    )
    pool = SimpleNamespace(quant_method=None, get_key_buffer=lambda _: local_latent)
    # At 64 the whole window sits in one page, so the other ranks' rows are empty.
    for end in (129, 132, 128, 64):
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
    print(f"rank {rank}: DSA verify and re-anchor matched unsharded", flush=True)
    # K3.1 verify geometry: one-tile, three-row-tile and unsharded metadata must agree.
    from tokenspeed.runtime.layers.attention.dcp.indexer import (
        build_dcp_index_metadata,
    )

    torch.manual_seed(2048)
    blocks, q_len = 40, 8
    k31_keys = torch.randn(blocks * 128, 128, device="cuda", dtype=torch.bfloat16)
    k31_values, k31_scales = quantize_fp8(
        k31_keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )

    def k31_plane(placement):
        local_blocks = -(-(blocks - 1) // len(placement.group)) + 1
        plane = torch.zeros(local_blocks * 128, 132, device="cuda", dtype=torch.uint8)
        slots, owned = resolve_cache_slots(
            torch.arange(blocks * 128, device="cuda"), placement
        )
        index_k_block_split_scatter(
            plane,
            k31_values,
            k31_scales,
            slots,
            page_size=64,
            head_dim=128,
            group_size=128,
            write_mask=owned,
        )
        return plane

    k31_sharded = CachePlacement(128, blocks, group, rank)
    k31_unsharded = CachePlacement(128, blocks, (rank,), 0)
    k31_local, k31_full = k31_plane(k31_sharded), k31_plane(k31_unsharded)
    # Block order unrelated to owners; one request past topk, one short.
    request_blocks = (
        [17, 3, 30, 8, 25, 1, 12, 36, 21, 5, 33, 14, 28, 9, 2, 39, 19, 26, 6, 31],
        [11, 4, 22],
    )
    k31_table = torch.zeros(2, 64, device="cuda", dtype=torch.int32)
    for row, ids in enumerate(request_blocks):
        pages = torch.tensor(ids, device="cuda", dtype=torch.int32) * 2
        k31_table[row, : 2 * len(ids)] = torch.stack((pages, pages + 1), 1).flatten()
    k31_seq = torch.tensor([2523, 300], device="cuda", dtype=torch.int32)
    k31_q = torch.randn(2 * q_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k31_w = torch.randn(2 * q_len, 32, device="cuda", dtype=torch.bfloat16)
    k31_requests = torch.arange(2 * q_len, device="cuda", dtype=torch.int32) // q_len
    k31_causal = (
        k31_seq[k31_requests.long()]
        - (q_len - 1)
        + torch.arange(2 * q_len, device="cuda") % q_len
    )
    k31_kw = dict(
        page_size=64, topk=2048, softmax_scale=0.1, initial_tokens=0, local_tokens=0
    )
    metadata = build_dcp_index_metadata(
        k31_q,
        k31_w,
        k31_local,
        k31_table,
        k31_requests,
        k31_causal,
        placement=k31_sharded,
        page_size=64,
    )
    single, single_lens = select_dsa_topk(
        k31_q,
        k31_w,
        k31_local,
        k31_table,
        metadata.query_requests,
        metadata.causal_lens,
        placement=k31_sharded,
        max_logits_bytes=None,
        metadata=metadata,
        **k31_kw,
    )
    tiled, tiled_lens = select_dsa_topk(
        k31_q,
        k31_w,
        k31_local,
        k31_table,
        k31_requests,
        k31_causal,
        placement=k31_sharded,
        max_logits_bytes=3 * 64 * 64 * 4,
        **k31_kw,
    )
    reference, ref_lens = select_dsa_topk(
        k31_q,
        k31_w,
        k31_full,
        k31_table,
        k31_requests,
        k31_causal,
        placement=k31_unsharded,
        max_logits_bytes=None,
        **k31_kw,
    )
    torch.testing.assert_close(ref_lens, k31_causal.clamp(0, 2048).int())
    for result, lens in ((single, single_lens), (tiled, tiled_lens)):
        torch.testing.assert_close(lens, ref_lens)
        torch.testing.assert_close(
            result.sort(-1).values, reference.sort(-1).values, rtol=0, atol=0
        )
    seq2d = k31_seq.repeat_interleave(q_len).unsqueeze(1).contiguous()
    native_slots, native_lens = dsa_decode_topk(
        k31_q,
        k31_w,
        k31_seq,
        k31_table,
        page_size=64,
        topk=2048,
        softmax_scale=0.1,
        batch_invariant=False,
        q_len_per_req=q_len,
        index_k_cache=k31_full,
        seq_lens_2d=seq2d,
        plan=dsa_plan(page_size=64, seq_lens_2d=seq2d),
        solution="deep_gemm",
        slot_order="selection",
    )
    expected_slots = torch.where(
        single >= 0,
        k31_table[
            k31_requests.long().unsqueeze(1), single.clamp_min(0).long() // 64
        ].long()
        * 64
        + single % 64,
        -1,
    )
    torch.testing.assert_close(native_lens, single_lens)
    for row in range(2 * q_len):
        assert {s for s in expected_slots[row].tolist() if s >= 0} == {
            s for s in native_slots[row].tolist() if s >= 0
        }
    print(f"rank {rank}: K3.1 8-row verify top-k matched unsharded", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
