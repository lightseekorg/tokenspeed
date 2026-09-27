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

"""The startup device memory summary: weight buckets and the KV cache row."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.engine.scheduler_utils import (
    _classify_param,
    _kv_pool_bytes,
    log_gpu_memory_summary,
)


@pytest.mark.parametrize(
    "name, group",
    [
        ("model.layers.1.engram.embed.weight", "engram_weights"),
        ("model.layers.1.engram.wkv.weight", "engram_weights"),
        ("model.layers.1.self_attn.wo.weight", "attention_weights"),
        ("model.layers.3.mlp.experts.w13_weight", "moe_weights"),
        ("model.layers.3.mlp.shared_experts.down_proj.weight", "dense_mlp_weights"),
        ("model.embed_tokens.weight", "other_weights"),
        ("lm_head.weight", "other_weights"),
    ],
)
def test_classify_param(name, group):
    assert _classify_param(name) == group


def test_kv_pool_bytes_dedupes_views_of_one_arena():
    arena = object()
    target = SimpleNamespace(arena=arena, get_kv_size_bytes=lambda: 1 << 20)
    draft = SimpleNamespace(arena=arena, get_kv_size_bytes=lambda: 1 << 20)
    assert _kv_pool_bytes(target, draft, None) == 1 << 20


class _Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 32, device=device)
        self.layers = nn.ModuleList(
            [
                nn.Module(),
            ]
        )
        self.layers[0].engram = nn.Module()
        self.layers[0].engram.embed = nn.Module()
        # 64 MiB: large enough to survive the table's two-decimal GB rounding.
        self.layers[0].engram.embed.weight = nn.Parameter(
            torch.zeros(1 << 22, 8, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_summary_lists_engram_tables_and_kv_pool_bytes(caplog):
    device = torch.device("cuda", 0)
    model = _Model(device)
    kv = torch.empty(48 << 20, dtype=torch.uint8, device=device)
    pool = SimpleNamespace(
        arena=SimpleNamespace(buffer=kv), get_kv_size_bytes=lambda: kv.nbytes
    )
    logger = logging.getLogger("tokenspeed.test_gpu_memory_summary")
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_gpu_memory_summary(model, 0, 0, logger, kv_pool=pool, device="cuda")
    assert len(caplog.records) == 1, "the summary must be one record"
    table = {
        line.split("|")[1].strip(): float(line.split("|")[2])
        for line in caplog.records[0].getMessage().splitlines()
        if line.startswith("| ") and not line.startswith("| Component")
    }
    gb = 1024**3
    engram_bytes = model.layers[0].engram.embed.weight.nbytes
    assert table["Engram weights (tables/wkv)"] == round(engram_bytes / gb, 2) > 0
    assert table["KV cache"] == round(kv.nbytes / gb, 2) > 0
    # No Engram parameters on the device: the row is omitted, not printed as 0.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_gpu_memory_summary(
            nn.Embedding(4, 4, device=device), 0, 0, logger, device="cuda"
        )
    assert "Engram weights" not in caplog.records[0].getMessage()
