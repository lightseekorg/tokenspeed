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


def main() -> None:
    import os

    import torch
    import torch.distributed as dist
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.indexer import (
        merge_index_candidates,
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
    torch.manual_seed(912)
    page_size, topk, pages = 64, 512, 17
    keys = torch.randn(pages * page_size, 128, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8_with_scale(
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
        index_k_cache=full,
        seq_lens_2d=seq2d,
        plan=dsa_plan(page_size=64, seq_lens_2d=seq2d),
        solution="deep_gemm",
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
    output = combine_attention_partials(partial, lse, group=group, rank=rank, sink=None)
    torch.testing.assert_close(
        output, ref_out[:, rank * 2 : (rank + 1) * 2], rtol=0.01, atol=0.005
    )
    print(f"rank {rank}: distributed attention/LSE merge matched unsharded", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
