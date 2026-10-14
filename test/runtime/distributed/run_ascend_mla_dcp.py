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

"""Run with torchrun --nproc-per-node=8; independent dense CPU oracle.

Exercises the actual backend and owner-masked prologue, DCP1/8 and dynamic
NPU graph replay. This is a backend check, not full-engine acceptance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401 -- register the NPU device
from tokenspeed_kernel.ops.attention.prologue import LatentKVCache, mla_prologue

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution import forward_step
from tokenspeed.runtime.layers.attention.backends.paged.ascend_mla import (
    AscendMLABackend,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world == 8
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
pg_manager.init_distributed(
    NS(world_size=world, rank=rank), backend="hccl", timeout=180
)
pg_manager.init_process_group(tuple(range(world)), backend="hccl")

results = []


def run(B, W, L, G, degree, graph_test):
    S = 128
    torch.manual_seed(310 + L + B + W)
    nblocks = (L + W + G - 1) // G
    capacity = B * nblocks + 1
    physical = torch.randperm(capacity - 1).reshape(B, nblocks) + 1
    subpages = G // S
    table = (
        (physical[:, :, None] * subpages + torch.arange(subpages))
        .flatten(1)
        .to(torch.int32)
    )
    maxcontext = nblocks * G
    lengths = torch.tensor([max(W, L - i % 3) for i in range(B)], dtype=torch.int32)
    if B > 1:
        lengths[-1] = 0
        table[-1].zero_()
    global_cache = torch.randn(capacity, G, 1, 576, dtype=torch.bfloat16) * 0.25
    global_cache[0].zero_()
    global_q = torch.randn(B * W, 32, 576, dtype=torch.bfloat16) * 0.25
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
        speculative_num_draft_tokens=W,
        context_len=maxcontext,
        kvp_size=degree,
        kvp_rank=rank if degree > 1 else 0,
        kvp_group=tuple(range(world)) if degree > 1 else (rank,),
    )
    backend = AscendMLABackend(config, spec, kernel_page_size=S)
    backend.configure_runtime(
        block_granularity=G, virtual_block_count=capacity, shard_count=degree
    )
    backend.init_cuda_graph_state(B)
    g = backend._placement
    local_blocks = (capacity - 2) // degree + 2
    local = torch.zeros(local_blocks, G, 1, 576, dtype=torch.bfloat16)
    for b in range(1, capacity):
        if (b - 1) % degree == g.rank:
            local[(b - 1) // degree + 1] = global_cache[b]
    expected_local = local.clone()
    cache = local.npu().view(-1, S, 1, 576)
    q = global_q[:, rank * 4 : (rank + 1) * 4].contiguous().npu()
    t = table.npu()
    lens = lengths.npu()
    pool = NS(get_key_buffer=lambda layer_id: cache)
    layer = NS(layer_id=0, scaling=192**-0.5, logit_cap=0, sliding_window_size=-1)
    # Re-write live candidate rows through the same registered prologue as
    # model execution. Nonowners and null padding must preserve cache bytes.
    slots = []
    for b in range(B):
        for j in range(W):
            pos = int(lengths[b]) - W + j
            slots.append(
                int(table[b, pos // S]) * S + pos % S if pos >= 0 and lengths[b] else 0
            )
    slots = torch.tensor(slots, dtype=torch.int64).npu()
    local_slots, owned = resolve_cache_slots(slots, g)
    values = global_cache.view(-1, 576)[slots.cpu()].npu()
    cache.view(-1, 576)[local_slots[owned].long()] = 0
    target = LatentKVCache(cache.view(-1, 1, 576), False, local_slots, owned)
    query = mla_prologue(
        q,
        q[..., 512:],
        values,
        expanded=None,
        rotary=None,
        cache=target,
        solution=None,
        override=None,
    ).query
    torch.testing.assert_close(
        cache.cpu().view_as(expected_local), expected_local, rtol=0, atol=0
    )

    def refresh():
        backend.refresh_decode_metadata(B, B, lens, t)

    def forward():
        return backend.forward_decode(query, None, None, layer, slots, pool, B)

    def check(out, label):
        out = out.cpu().float().reshape(B * W, 4, 512)
        qc = query.cpu().float()
        cc = global_cache.view(-1, S, 1, 576).float()
        tt = t.cpu().long()
        ll = lens.cpu()
        refs = []
        for b in range(B):
            for j in range(W):
                end = max(0, int(ll[b]) - W + j + 1)
                if not int(tt[b].sum()) or end == 0:
                    refs.append(torch.zeros(4, 512))
                    continue
                kv = cc[tt[b]].reshape(-1, 576)[:end]
                scores = qc[b * W + j] @ kv.T * layer.scaling
                refs.append(scores.softmax(-1) @ kv[:, :512])
        ref = torch.stack(refs)
        err = (out - ref).abs().max().item()
        rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        assert torch.isfinite(out).all() and err < 0.004 and rel < 0.02, (
            rank,
            B,
            W,
            L,
            degree,
            label,
            err,
            rel,
        )
        result = {
            "rank": rank,
            "B": B,
            "W": W,
            "L": L,
            "G": G,
            "dcp": degree,
            "mode": label,
            "max_abs": err,
            "relative_l2": rel,
            "physical_bytes": cache.numel() * cache.element_size(),
        }
        results.append(result)
        if rank == 0:
            print("PASS", json.dumps(result), flush=True)

    refresh()
    for _ in range(2):
        out = forward()
    check(out, "eager")
    if W > 1:
        # Alter only the final candidate after writing all W candidates. Earlier
        # queries must remain bitwise unchanged, even across page/owner boundaries.
        before = out.clone().reshape(B, W, -1)
        final_slots = slots.view(B, W)[:, -1]
        live = final_slots.cpu() > 0
        global_cache.view(-1, 576)[final_slots.cpu()[live]] += 0.5
        changed = global_cache.view(-1, 576)[final_slots.cpu()].npu()
        mapped, mask = resolve_cache_slots(final_slots, g)
        mla_prologue(
            query.view(B, W, 4, 576)[:, -1].contiguous(),
            query.view(B, W, 4, 576)[:, -1, :, 512:].contiguous(),
            changed,
            expanded=None,
            rotary=None,
            cache=LatentKVCache(cache.view(-1, 1, 576), False, mapped, mask),
            solution=None,
            override=None,
        )
        after = forward()
        torch.testing.assert_close(
            after.reshape(B, W, -1)[:, :-1], before[:, :-1], rtol=0, atol=0
        )
        check(after, "future_isolation")

        # Accept one candidate, discard the remaining W-1, then overwrite the
        # next W candidates. Rejected bytes may remain outside visible endpoints.
        lens.copy_(torch.where(lens > 0, lens + 1, lens))
        new_slots = []
        for b in range(B):
            for j in range(W):
                pos = int(lens[b].cpu()) - W + j
                new_slots.append(
                    int(table[b, pos // S]) * S + pos % S if live[b] else 0
                )
        new_slots = torch.tensor(new_slots, dtype=torch.int64)
        new_values = torch.randn(B * W, 576, dtype=torch.bfloat16) * 0.25
        global_cache.view(-1, 576)[new_slots[new_slots > 0]] = new_values[new_slots > 0]
        mapped, mask = resolve_cache_slots(new_slots.npu(), g)
        query = mla_prologue(
            query,
            query[..., 512:],
            new_values.npu(),
            expanded=None,
            rotary=None,
            cache=LatentKVCache(cache.view(-1, 1, 576), False, mapped, mask),
            solution=None,
            override=None,
        ).query
        refresh()
        check(forward(), "accept_one_next_verify")
    if graph_test:
        graph = torch.npu.NPUGraph()
        with patch.object(forward_step, "_is_cuda_graph_phase", True):
            for _ in range(2):
                forward()
            with (
                patch.object(forward_step, "_is_capture_mode", True),
                torch.npu.graph(graph, auto_dispatch_capture=True),
            ):
                out = forward()
        for step in range(2):
            lens.copy_(torch.where(lens > 0, (lens - W - 3).clamp_min(W), lens))
            t.copy_(t.flip(1))
            query.add_(0.015625)
            refresh()
            graph.update(
                cpu_update_input=[{"actual_seq_lengths_kv": lens.cpu().tolist()}]
            )
            graph.replay()
            torch.npu.synchronize()
            check(out, "graph" + str(step))
    torch.distributed.barrier()


for B, W, L, G in [
    (1, 1, 127, 128),
    (1, 1, 128, 128),
    (1, 1, 129, 128),
    (8, 4, 129, 128),
    (16, 8, 2048, 256),
    (8, 8, 8192, 128),
]:
    for degree in [1, 8]:
        run(B, W, L, G, degree, True)
Path("results").mkdir(exist_ok=True)
Path(f"results/mla-dcp-rank{rank}.json").write_text(json.dumps(results, indent=2))
torch.distributed.destroy_process_group()
