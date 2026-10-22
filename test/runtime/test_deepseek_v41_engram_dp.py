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

"""Engram table ownership is independent of attention row ownership."""

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.deepseek_v41_parallel import resolve_engram_table_mapping


@pytest.mark.parametrize("rank", range(16))
def test_world_table_shards_and_replica_counts(rank):
    mapping = Mapping(rank=rank, world_size=16, attn_tp_size=4, attn_dp_size=4)
    table, exchange = resolve_engram_table_mapping(
        mapping, 16, device_type="npu", host_table=False, role="decode"
    )
    assert table.attn.tp_rank == rank and table.attn.tp_group == tuple(range(16))
    assert exchange.dp_group == tuple(range(rank % 4, 16, 4))
    counts = [0, 18, 42, 6]
    rows = [n for n in counts for _ in range(4)]
    assert exchange.counts(rows, counts[rank // 4]) == counts
    assert resolve_engram_table_mapping(
        mapping, None, device_type="cpu", host_table=False, role="null"
    ) == (mapping, None)


@pytest.mark.parametrize(
    "requested,device,host,role",
    [
        (8, "npu", False, "decode"),
        (16, "cuda", False, "decode"),
        (16, "npu", True, "decode"),
        (16, "npu", False, "null"),
        (16, "npu", False, "prefill"),
        (0, "npu", False, "decode"),
    ],
)
def test_reject_unimplemented_sharing(requested, device, host, role):
    mapping = Mapping(rank=0, world_size=16, attn_tp_size=4, attn_dp_size=4)
    with pytest.raises(ValueError, match="Wider Engram TP"):
        resolve_engram_table_mapping(
            mapping, requested, device_type=device, host_table=host, role=role
        )


def test_bad_row_table_rejected_before_collective():
    mapping = Mapping(rank=0, world_size=16, attn_tp_size=4, attn_dp_size=4)
    _, exchange = resolve_engram_table_mapping(
        mapping, 16, device_type="npu", host_table=False, role="decode"
    )
    for rows in (None, [6] * 15, [5] * 16, [6, 5] + [6] * 14, [6] * 15 + [-1]):
        with pytest.raises(ValueError):
            exchange.counts(rows, 6)


def test_server_options_reach_model_construction(monkeypatch):
    from tokenspeed.runtime.utils import env
    from tokenspeed.runtime.utils.server_args import ServerArgs

    args = ServerArgs.__new__(ServerArgs)
    args.engram_tp_size = 16
    args.disaggregation_mode = "decode"
    monkeypatch.setattr(env, "global_server_args_dict", {})
    env.global_server_args_dict_update(args)
    assert env.global_server_args_dict["engram_tp_size"] == 16
    assert env.global_server_args_dict["disaggregation_mode"] == "decode"


def test_idle_replica_participates_before_skipping_local_projection():
    import torch
    from torch import nn

    from tokenspeed.runtime.models.deepseek_v41_engram import DeepseekV41Engram

    events = []

    class Exchange:
        def gather(self, hashes, counts):
            assert hashes.shape == (0, 6) and counts == [0, 3]
            events.append("gather")
            return torch.zeros(3, 6, dtype=torch.int64), 0

    class Embed(nn.Module):
        def forward(self, hashes):
            assert hashes.shape == (3, 6)
            events.append("embedding_reduce")
            return torch.zeros(3, 6, 32, dtype=torch.bfloat16)

    block = DeepseekV41Engram.__new__(DeepseekV41Engram)
    nn.Module.__init__(block)
    block.hc_mult, block.dim, block.n_hash_cols = 4, 32, 6
    block.row_exchange = Exchange()
    block.embed = Embed()
    hidden = torch.empty(0, 4, 32, dtype=torch.bfloat16)
    actual = block(
        hidden,
        torch.empty(0, 6, dtype=torch.int64),
        torch.empty(0, dtype=torch.bool),
        global_num_tokens=[0, 3],
    )
    assert actual is hidden and events == ["gather", "embedding_reduce"]


def test_idle_model_preserves_engram_and_moe_collective_order():
    from types import SimpleNamespace as NS

    import torch
    from torch import nn

    from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41Model

    events = []
    rows = [0] * 4 + [6] * 12

    class Engram(nn.Module):
        n_hash_cols = 6

        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id

        def forward(self, hidden, hashes, mask, *, global_num_tokens):
            assert hashes.shape == (0, 6) and mask.shape == (0,)
            assert global_num_tokens == rows
            events.append((self.layer_id, "engram"))
            return hidden

    class Layer(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.engram = Engram(layer_id)

        def forward(self, hidden, pre_mix, positions, image_mask, ctx):
            events.append((self.layer_id, "moe"))
            return hidden, pre_mix

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.config = NS(hc_mult=4, hidden_size=32)
    model.mapping = Mapping(rank=0, world_size=16, attn_tp_size=4, attn_dp_size=4)
    model.embed_tokens = nn.Embedding(1, 32, dtype=torch.bfloat16)
    model.layers = nn.ModuleList([Layer(0), Layer(1)])
    model.dspark_capture_layers = ()
    model.dp_replicated_rows = True
    model.decoder_cpu_group = None
    model.ced_decoder_start = 1
    ctx = NS(
        input_num_tokens=0,
        global_num_tokens=rows,
        all_decode_or_idle=True,
        collective_global_num_tokens=None,
        collective_num_tokens=None,
    )
    empty = torch.empty(0, dtype=torch.int64)
    hidden, taps = model(
        empty,
        empty,
        ctx,
        None,
        None,
        engram_hash_ids=None,
        engram_token_mask=None,
        image_mask=None,
    )
    assert hidden.shape == (0, 32) and taps is None
    assert events == [(0, "engram"), (0, "moe"), (1, "engram"), (1, "moe")]
    assert ctx.collective_global_num_tokens is None
