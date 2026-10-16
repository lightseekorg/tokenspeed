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

"""Host chunked-prefill core, owner-masked store and sharded history.

Synthetic BF16 inputs exercise integration without claiming 3B weight acceptance.
Run with torchrun --nproc-per-node=8.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.attention.mla import mla_project_value
from tokenspeed_kernel.ops.attention.prologue import (
    LatentKVCache,
    MLAExpandedKV,
    mla_prologue,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg,
)
from tokenspeed.runtime.layers.attention.backends.paged.ascend_mla import (
    AscendMLABackend,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
from tokenspeed.runtime.utils.env import global_server_args_dict

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world == 8
torch.set_num_threads(1)
torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
pg.init_distributed(NS(world_size=world, rank=rank), backend="hccl", timeout=180)
pg.init_process_group(tuple(range(world)), backend="hccl")

global_server_args_dict.update(chunked_prefill_size=256, mla_chunk_multiplier=1)
results = []


class Pool:
    dtype = torch.bfloat16
    quant_method = "none"
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    get_mla_kv_buffer = MLATokenToKVPool.get_mla_kv_buffer

    def __init__(self, cache):
        self.cache = cache

    def get_key_buffer(self, layer):
        return self.cache


def run(degree):
    torch.manual_seed(793)
    B, S, N, H = 2, 128, 5, 4
    capacity = B * N + 1
    table = (torch.randperm(capacity - 1).reshape(B, N) + 1).int()
    global_cache = torch.zeros(capacity, S, 1, 576, dtype=torch.bfloat16)
    local_blocks = (capacity - 2) // degree + 2
    cache = torch.zeros(local_blocks * S, 1, 576, dtype=torch.bfloat16, device="npu")
    pool = Pool(cache)
    spec = MLAConfig(
        backend_name="tokenspeed_mla",
        num_attention_heads=32,
        num_kv_heads=1,
        head_dim=576,
        attn_tp_size=8,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
    )
    config = NS(
        device="npu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        is_draft=False,
        draft_block_decode=False,
        speculative_num_draft_tokens=1,
        context_len=N * S,
        kvp_size=degree,
        kvp_rank=rank if degree > 1 else 0,
        kvp_group=tuple(range(world)) if degree > 1 else (rank,),
    )
    backend = AscendMLABackend(config, spec, kernel_page_size=S)
    backend.configure_runtime(
        block_granularity=S, virtual_block_count=capacity, shard_count=degree
    )
    backend.init_cuda_graph_state(B)
    layer = NS(
        layer_id=0,
        scaling=192**-0.5,
        logit_cap=0,
        tp_q_head_num=H,
        tp_k_head_num=H,
        tp_v_head_num=H,
        v_head_dim=128,
        qk_head_dim=192,
    )
    weight = (torch.randn(512, 32, 256, dtype=torch.bfloat16) * 0.03125)[
        :, rank * H : (rank + 1) * H
    ].contiguous()
    wn = weight.flatten(1).npu()
    model = NS(
        num_local_heads=H,
        v_head_dim=128,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        kv_b_proj=lambda x: (x @ wn, None),
        attn_mha=layer,
    )
    ctx = NS(attn_backend=backend, token_to_kv_pool=pool)
    for prefix, extend, label in [
        ([0, 0], [129, 257], "prefill"),
        ([129, 257], [5, 3], "cached_prefill"),
        ([0, 0], [17, 9], "page_reuse"),
    ]:
        if label == "page_reuse":
            # Reassign the same physical pages to other requests without clearing.
            table = table.flip(0).contiguous()
        lengths = [p + e for p, e in zip(prefix, extend)]
        rows = sum(extend)
        slots = torch.tensor(
            [
                int(table[b, pos // S]) * S + pos % S
                for b in range(B)
                for pos in range(prefix[b], lengths[b])
            ],
            dtype=torch.int64,
        )
        latent = torch.randn(rows, 576, dtype=torch.bfloat16) * 0.25
        q = (torch.randn(rows, 32, 192, dtype=torch.bfloat16) * 0.25)[
            :, rank * H : (rank + 1) * H
        ].contiguous()
        global_cache.view(-1, 576)[slots] = latent
        mapped, mask = resolve_cache_slots(slots.npu(), backend._placement)
        before_q = q.npu()
        qn = before_q.clone()
        ln = latent.npu()
        kv = (ln[:, :512].contiguous() @ wn).view(rows, H, 256)
        prologue = mla_prologue(
            qn,
            qn[..., 128:],
            ln,
            expanded=MLAExpandedKV(k_nope=kv[..., :128], value=kv[..., 128:]),
            rotary=None,
            cache=LatentKVCache(cache, False, mapped, mask),
            solution=None,
            override=None,
        )
        torch.testing.assert_close(qn, before_q, rtol=0, atol=0)
        backend._init_prefill_metadata(
            torch.tensor(lengths, dtype=torch.int32).npu(),
            torch.tensor(prefix, dtype=torch.int32).npu(),
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(extend, dtype=torch.int32).npu(),
            torch.tensor(extend, dtype=torch.int32),
            table.npu(),
        )
        output = torch.empty(rows, H * 128, dtype=torch.bfloat16, device="npu")
        if max(prefix):
            output = DeepseekV3AttentionMLA.forward_normal_chunked_kv_core(
                model, prologue.query, prologue.key, prologue.value, ctx, output
            )
        else:
            output = backend.forward_extend(
                prologue.query,
                prologue.key,
                prologue.value,
                layer,
                slots.npu(),
                pool,
                B,
            )
        refs = []
        offset = 0
        for b in range(B):
            lat = global_cache[table[b].long()].reshape(-1, 576)[: lengths[b]]
            kv = (
                (lat[:, :512].float() @ weight.flatten(1).float())
                .bfloat16()
                .float()
                .view(-1, H, 256)
            )
            keys = torch.cat(
                (kv[..., :128], lat[:, None, 512:].float().expand(-1, H, -1)), dim=-1
            )
            for j in range(extend[b]):
                end = prefix[b] + j + 1
                scores = (
                    torch.einsum("hd,shd->hs", q[offset + j].float(), keys[:end])
                    * layer.scaling
                )
                refs.append(
                    torch.einsum("hs,shd->hd", scores.softmax(-1), kv[:end, :, 128:])
                )
            offset += extend[b]
        ref = torch.stack(refs)
        actual = output.cpu().float().view_as(ref)
        err = (actual - ref).abs().max().item()
        rel = ((actual - ref).norm() / ref.norm()).item()
        assert torch.isfinite(actual).all() and err < 0.004 and rel < 0.02, (
            rank,
            degree,
            label,
            err,
            rel,
        )
        for block in range(1, capacity):
            if (block - 1) % degree == backend._dcp.rank:
                torch.testing.assert_close(
                    cache.view(-1, S, 1, 576)[(block - 1) // degree + 1].cpu(),
                    global_cache[block],
                    rtol=0,
                    atol=0,
                )
        result = {
            "rank": rank,
            "dcp": degree,
            "stage": label,
            "max_abs": err,
            "relative_l2": rel,
        }
        results.append(result)
        if rank == 0:
            print("LIFECYCLE_PASS", json.dumps(result), flush=True)
    # Continue from the reused pages into absorbed decode. Capture the actual
    # owner-write + attention + value projection/gate, then advance slots/lengths.
    q_expanded = (
        (torch.randn(B, 32, 192, dtype=torch.bfloat16) * 0.25)[
            :, rank * H : (rank + 1) * H
        ]
        .contiguous()
        .npu()
    )
    wkc = weight[:, :, :128].permute(1, 2, 0).contiguous().npu()
    wvc = weight[:, :, 128:].permute(1, 0, 2).contiguous().npu()
    qabs = torch.cat(
        (
            torch.bmm(
                q_expanded[..., :128].transpose(0, 1).contiguous(), wkc
            ).transpose(0, 1),
            q_expanded[..., 128:],
        ),
        dim=-1,
    ).contiguous()
    values = torch.empty(B, 576, dtype=torch.bfloat16, device="npu")
    write_slots = torch.empty(B, dtype=torch.int64, device="npu")
    write_mask = torch.empty(B, dtype=torch.bool, device="npu")
    gate = torch.randn(B, H * 128, dtype=torch.bfloat16).npu()
    projected = torch.empty_like(gate)

    def decode():
        prologue = mla_prologue(
            qabs,
            qabs[..., 512:],
            values,
            expanded=None,
            rotary=None,
            cache=LatentKVCache(cache, False, write_slots, write_mask),
            solution=None,
            override=None,
        )
        absorbed = backend.forward_decode(
            prologue.query, None, None, layer, write_slots, pool, B
        ).view(B, H, 512)
        return mla_project_value(absorbed, wvc, gate=gate, out=projected)

    graph = None
    for step in range(3):
        new_slots = torch.tensor(
            [int(table[b, lengths[b] // S]) * S + lengths[b] % S for b in range(B)],
            dtype=torch.int64,
        )
        lengths = [x + 1 for x in lengths]
        fresh = torch.randn(B, 576, dtype=torch.bfloat16) * 0.25
        global_cache.view(-1, 576)[new_slots] = fresh
        values.copy_(fresh.npu())
        mapped, mask = resolve_cache_slots(new_slots.npu(), backend._placement)
        write_slots.copy_(mapped)
        write_mask.copy_(mask)
        backend.refresh_decode_metadata(
            B, B, torch.tensor(lengths, dtype=torch.int32).npu(), table.npu()
        )
        if step == 0:
            actual = decode()
        elif step == 1:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, auto_dispatch_capture=True):
                actual = decode()
            # Like TS ForwardStep, update auto-dispatched FIA tasks before
            # EVERY replay, including the first execution after capture.
            graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
            graph.replay()
        else:
            graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
            graph.replay()
        torch.npu.synchronize()
        refs = []
        for b in range(B):
            lat = global_cache[table[b].long()].reshape(-1, 576)[: lengths[b]].float()
            scores = qabs[b].cpu().float() @ lat.T * layer.scaling
            absorbed = scores.softmax(-1) @ lat[:, :512]
            refs.append(torch.bmm(absorbed[:, None, :], wvc.cpu().float()).squeeze(1))
        ref = torch.stack(refs).flatten(1) * gate.cpu().float().sigmoid()
        err = (actual.cpu().float() - ref).abs().max().item()
        rel = ((actual.cpu().float() - ref).norm() / ref.norm()).item()
        assert torch.isfinite(actual).all() and err < 0.004 and rel < 0.02, (
            rank,
            degree,
            step,
            err,
            rel,
        )
        result = {
            "rank": rank,
            "dcp": degree,
            "stage": "decode_projection_gate_" + str(step),
            "max_abs": err,
            "relative_l2": rel,
        }
        results.append(result)
        if rank == 0:
            print("LIFECYCLE_PASS", json.dumps(result), flush=True)
    torch.distributed.barrier()


for degree in [1, 8]:
    run(degree)
Path(f"results/mla-lifecycle-rank{rank}.json").write_text(json.dumps(results, indent=2))
torch.distributed.destroy_process_group()
