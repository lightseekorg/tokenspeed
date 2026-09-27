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

"""Attention-TP KPool row slicing and collective layout contracts."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention import kpool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.mark.parametrize("counts", [[2, 2, 2], [3, 2, 2], [1, 1, 1]])
@pytest.mark.parametrize("width", [None, 5])
def test_gather_rows_preserves_order(monkeypatch, counts, width):
    rows = sum(counts)
    shape = (rows,) if width is None else (rows, width)
    expected = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.int32
    ).reshape(shape)
    group = tuple(range(len(counts)))
    chunks = expected.split(counts)
    for rank, local in enumerate(chunks):

        def gather(output, padded, actual_group):
            assert actual_group == group
            torch.testing.assert_close(padded[: counts[rank]], local)
            output.fill_(-999)
            for peer, chunk in enumerate(chunks):
                output[peer * max(counts) : peer * max(counts) + len(chunk)].copy_(
                    chunk
                )

        monkeypatch.setattr(kpool, "all_gather_single", gather)
        torch.testing.assert_close(kpool._gather_rows(local, counts, group), expected)


@pytest.mark.parametrize(
    "rows,tp_size,max_pools,split",
    [
        (7, 3, 9, True),
        (6, 3, 9, True),
        (2, 3, 9, False),
        (7, 3, 8, False),
        (7, 1, 9, False),
    ],
)
def test_prefill_tp_slices_all_row_metadata(
    monkeypatch, rows, tp_size, max_pools, split
):
    runtime = kpool.KPoolRuntime(pool_size=4, index_topk=16)
    lengths = torch.tensor([1, rows - 1], dtype=torch.int32)
    prefix = torch.tensor([32, 64], dtype=torch.int32)
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    plan = kpool.build_kpool_prefill_plan(
        prefix_lens_cpu=prefix,
        extend_lens_cpu=lengths,
        index_block_table=table,
        request_slots=torch.tensor([0, 1]),
        kpool=4,
        index_rows_per_page=16,
    )
    plan = replace(plan, max_num_pools=max_pools)
    runtime.prefill_plan = plan
    metadata = SimpleNamespace(
        extend_prefix_lens=prefix,
        extend_seq_lens=lengths,
        extend_prefix_lens_cpu=prefix,
        extend_seq_lens_cpu=lengths,
    )
    backend = SimpleNamespace(
        chunked_prefill_metadata=metadata, kpool_prefill_page_table=lambda count: table
    )
    cache = torch.empty(5, 16, 128)
    ctx = SimpleNamespace(
        num_extends=2,
        token_to_kv_pool=SimpleNamespace(
            get_kpool_buffers=lambda layer: (cache,),
            arena=SimpleNamespace(kv_page_size=64),
        ),
    )
    query = torch.arange(rows * 8).reshape(rows, 2, 4).float()
    weights = query[:, :, 0].contiguous()
    expected = torch.stack([torch.arange(rows), torch.full((rows,), -1)], dim=1).int()
    expected_lens = torch.ones(rows, dtype=torch.int32)
    monkeypatch.setitem(
        kpool.global_server_args_dict, "deepseek_v4_indexer_prefill_max_logits_mb", 1
    )
    group = tuple(range(tp_size))
    for rank in range(tp_size):
        base, extra = divmod(rows, tp_size)
        counts = [base + (peer < extra) for peer in range(tp_size)]
        start = base * rank + min(rank, extra) if split else 0
        end = start + counts[rank] if split else rows

        def select(q, actual_cache, w, positions, boundaries, *tables, **kwargs):
            torch.testing.assert_close(q, query[start:end])
            torch.testing.assert_close(w, weights[start:end])
            torch.testing.assert_close(positions, plan.positions[start:end])
            assert boundaries is plan.query_start_loc
            assert actual_cache is cache
            assert kwargs["pool_workspace_slots"] is plan.pool_workspace_slots
            for name, value in (
                ("req_ids", plan.req_ids),
                ("causal_lens", plan.causal_lens),
                ("row_starts", plan.row_starts),
                ("row_ends", plan.row_ends),
            ):
                torch.testing.assert_close(kwargs[name], value[start:end])
            return expected[start:end].clone(), expected_lens[start:end].clone()

        gathered = []

        def gather(local, actual_counts, actual_group):
            assert split
            assert actual_counts == counts
            assert actual_group == group
            result = expected if local.ndim == 2 else expected_lens
            torch.testing.assert_close(local, result[start:end])
            gathered.append(local)
            return result.clone()

        monkeypatch.setattr(kpool, "kpool_prefill_topk", select)
        monkeypatch.setattr(kpool, "_gather_rows", gather)
        result = runtime.select_prefill(
            query=query,
            weights=weights,
            softmax_scale=0.5,
            ctx=ctx,
            backend=backend,
            layer_id=0,
            num_prefill_tokens=rows,
            tp_group=group,
            tp_rank=rank,
            tp_size=tp_size,
        )
        torch.testing.assert_close(result.kv_workspace_slots, expected.flatten())
        torch.testing.assert_close(result.topk_lens, expected_lens)
        assert len(gathered) == (2 if split else 0)
