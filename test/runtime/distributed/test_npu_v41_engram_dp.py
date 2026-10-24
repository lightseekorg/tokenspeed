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

"""Run with torchrun --standalone --nproc_per_node=16 -m pytest -q this_file.

Uses small tables only; validates arbitrary DP queries, empty replicas, exact
original FP8 table rows and complete Engram blocks under native graph replay.
"""

import os

import pytest


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "0")) != 16,
    reason="requires an isolated sixteen-device NPU world",
)
def test_world_sharded_engram_with_attention_dp():
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.distributed.comm_backend import set_global_backend
    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg,
    )
    from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding
    from tokenspeed.runtime.models.deepseek_v41_parallel import (
        resolve_engram_table_mapping,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict as args

    rank = int(os.environ["RANK"])
    torch.npu.set_device(rank)
    torch.set_num_threads(2)
    mapping = Mapping(
        rank=rank,
        world_size=16,
        attn_tp_size=4,
        attn_dp_size=4,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=16,
        nprocs_per_node=16,
        nnodes=1,
    )
    args.update(
        mapping=mapping,
        layer_boundary_norm="fused",
        moe_combine_order="rank",
        enable_allreduce_fusion=False,
        force_deterministic_rsag=True,
    )
    pg.init_distributed(mapping, backend="hccl", timeout=180)
    set_global_backend(HcclBackend())
    tm, exchange = resolve_engram_table_mapping(
        mapping, 16, device_type="npu", host_table=False, role="decode"
    )
    pg.init_process_group(tm.attn.tp_group)
    pg.init_process_group(exchange.dp_group)
    table = RowShardedEngramEmbedding(389, 256, tm, "npu", False, "gpu", 1)
    codes = ((torch.arange(389 * 256).reshape(389, 256) % 31 - 15) / 4).to(
        torch.float8_e4m3fn
    )
    scales = (torch.arange(389 * 8).reshape(389, 8) % 9 + 122).to(torch.uint8)
    table.weight_loader(table.weight, codes)
    table.weight_loader(table.scale, scales)
    full = (
        codes.float()
        * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
    ).bfloat16()
    checks = 0
    for counts in [(48, 48, 48, 48), (0, 18, 42, 6), (6, 0, 0, 0), (0, 0, 0, 0)]:
        n = counts[rank // 4]
        world = [c for c in counts for _ in range(4)]
        torch.manual_seed(334 + rank // 4)
        ids = torch.randint(-4, 395, (n, 24), device="npu")

        def forward():
            gathered, start = exchange.gather(ids, world)
            return table(gathered)[start : start + n]

        def check(out):
            host = ids.cpu()
            ref = full[host.clamp(0, 388)].masked_fill(
                ((host < 0) | (host >= 389)).unsqueeze(-1), 0
            )
            torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)

        out = forward()
        torch.npu.synchronize()
        check(out)
        checks += 1
        if len(set(counts)) == 1 and n:
            for _ in range(3):
                forward()
            torch.npu.synchronize()
            dist.barrier()
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                out = forward()
            for step in range(4):
                torch.manual_seed(443 + rank // 4 + step * 11)
                ids.copy_(torch.randint(-4, 395, ids.shape, device="npu"))
                g.replay()
                torch.npu.synchronize()
                check(out)
                checks += 1
        dist.barrier()
    # Exercise the actual Engram block's gather / local slice / projection wiring.
    from types import SimpleNamespace as NS

    from tokenspeed.runtime.models.deepseek_v41_engram import DeepseekV41Engram

    config = NS(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[72, 204],
        engram_max_ngram_size=4,
        engram_vocab_size=5,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=9,
        hc_mult=4,
        hidden_size=32,
        rms_norm_eps=1e-20,
    )
    pg.init_process_group(mapping.attn.tp_group)
    args.update(engram_tp_size=16, disaggregation_mode="decode")
    shared = DeepseekV41Engram(config, 1, mapping, None, "probe", "npu", False, "gpu")
    args["engram_tp_size"] = None
    reference = DeepseekV41Engram(
        config, 1, mapping, None, "reference", "npu", False, "gpu"
    )
    torch.manual_seed(751)
    weight = torch.randn(160, 192, dtype=torch.bfloat16) * 0.01
    for module in [shared, reference]:
        module.embed.weight_loader(module.embed.weight, codes[:72, :32].contiguous())
        module.embed.weight_loader(module.embed.scale, scales[:72, :1].contiguous())
        module.wkv.weight.weight_loader(module.wkv.weight, weight)
    for counts in [(8, 8, 8, 8), (0, 3, 7, 1), (0, 0, 0, 0)]:
        n = counts[rank // 4]
        world = [c for c in counts for _ in range(4)]
        torch.manual_seed(863 + rank // 4)
        ids = torch.randint(0, 72, (n, 6), device="npu")
        hidden = torch.randn(n, 4, 32, device="npu", dtype=torch.bfloat16)
        mask = torch.ones(n, device="npu", dtype=torch.bool)
        mask[::3] = False

        def run_block():
            return shared(hidden, ids, mask, global_num_tokens=world)

        out = run_block()
        expected = reference(hidden, ids, mask)
        torch.npu.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        checks += 1
        if len(set(counts)) == 1 and n:
            for _ in range(3):
                run_block()
            torch.npu.synchronize()
            dist.barrier()
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                out = run_block()
            for step in range(3):
                torch.manual_seed(971 + rank // 4 + step * 13)
                ids.copy_(torch.randint(0, 72, ids.shape, device="npu"))
                hidden.copy_(torch.randn_like(hidden))
                g.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(
                    out, reference(hidden, ids, mask), atol=0, rtol=0
                )
                checks += 1
        dist.barrier()
    # Full model idle entry must execute the same Engram collectives as live rows.
    from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41Model, V41RowState

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_id = 1
            self.engram = shared

        def forward(self, hidden, pre_mix, positions, image_mask, ctx):
            return hidden, pre_mix

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.config = config
    model.mapping = mapping
    model.embed_tokens = torch.nn.Embedding(1, 32, dtype=torch.bfloat16, device="npu")
    model.layers = torch.nn.ModuleList([Layer()])
    model.dspark_capture_layers = ()
    model.dp_replicated_rows = False
    model.ced_decoder_start = 2
    for counts in [(8, 8, 8, 8), (0, 3, 7, 1), (0, 0, 1, 0), (0, 0, 0, 0)]:
        n = counts[rank // 4]
        world = [c for c in counts for _ in range(4)]
        ids = torch.randint(0, 72, (n, 6), device="npu")
        hidden = torch.randn(n, 4, 32, device="npu", dtype=torch.bfloat16)
        mask = torch.ones(n, device="npu", dtype=torch.bool)
        positions = torch.arange(n, device="npu")
        mix = torch.zeros(n, 4, device="npu")
        ctx = NS(global_num_tokens=world, collective_global_num_tokens=None)
        if n == 0:
            output, taps = model(
                positions,
                positions,
                ctx,
                None,
                None,
                engram_hash_ids=None,
                engram_token_mask=mask,
                image_mask=None,
            )
            assert output.shape == (0, 32) and taps is None
        else:
            state = V41RowState(hidden, mix, positions, None, ids[:, None], mask, [])
            output, _ = model._run_layer(model.layers[0], hidden, mix, state, ctx, [])
            torch.testing.assert_close(
                output, reference(hidden, ids, mask), atol=0, rtol=0
            )
        torch.npu.synchronize()
        dist.barrier()
        checks += 1
    assert checks == 18
    dist.destroy_process_group()
