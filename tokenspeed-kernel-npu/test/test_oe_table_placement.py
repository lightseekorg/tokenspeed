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

from test.runtime.legacy_host_oe_reference import LegacyHostOEReference

import pytest
import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    append_packed_lookup_,
    register_host_tables_,
)

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("flash_ops")


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("table_placement", ["cpu", "npu"])
def test_append_packed_lookup_accepts_host_or_device_tables(
    table_placement: str,
) -> None:
    spec = OverEmbeddingSpec(
        profile="longcat-lite-test",
        tp_size=1,
        rank=0,
        vocab_size=32,
        branch_count=2,
        branch_width=256,
        hidden_size=512,
        max_ngram_order=3,
        fragments=(
            TableFragmentSpec(0, 2, 11, 0, 256),
            TableFragmentSpec(1, 3, 13, 0, 256),
        ),
        ignored_token_ids=(3,),
        segment_ignored_tokens=True,
    )
    host_tables = (
        torch.arange(1, 12, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 256)
        .contiguous(),
        torch.arange(1, 14, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 256)
        .contiguous(),
    )
    tables = (
        host_tables
        if table_placement == "cpu"
        else tuple(table.npu() for table in host_tables)
    )
    register_host_tables_(
        host_tables,
        device=torch.device("npu", torch.npu.current_device()),
        solution="flash_npu_kernel",
    )
    history = torch.zeros((2, 8), device="npu", dtype=torch.int32)

    output = append_packed_lookup_(
        torch.tensor([5, 3, 7], device="npu", dtype=torch.int32),
        torch.tensor([0, 3], device="npu", dtype=torch.int32),
        torch.tensor([0], device="npu", dtype=torch.int64),
        torch.tensor([True], device="npu"),
        history,
        torch.zeros(2, device="npu", dtype=torch.int32),
        tables,
        spec=spec,
        solution="flash_npu_kernel",
        enable_pdl=False,
    )
    torch.npu.synchronize()

    reference_config = type(
        "ReferenceConfig",
        (),
        {
            "vocab_size": spec.vocab_size,
            "emb_neighbor_num": spec.max_ngram_order,
            "emb_split_num": 1,
            "oe_component_count": spec.branch_count,
            "special_token_ids": spec.ignored_token_ids,
            "ngram_exclude_sp_token": spec.segment_ignored_tokens,
            "ngram_fix_normalize_factor": False,
            "oe_table_rows": lambda _, table_id: spec.fragments[table_id].modulus,
        },
    )()
    reference = LegacyHostOEReference(
        reference_config,
        (),
        torch.empty(0),
    )
    reference_ids, reference_special, _ = reference.ngram_ids(
        torch.tensor([5, 3, 7], dtype=torch.int32),
        torch.zeros((1, spec.history_lookback), dtype=torch.int32),
        [3],
    )
    expected = torch.cat(
        [
            host_tables[table_id][reference_ids[:, table_id]]
            for table_id in range(spec.branch_count)
        ],
        dim=1,
    )
    expected[reference_special] = 0

    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    assert torch.equal(
        output.cpu(),
        append_packed_lookup_(
            torch.tensor([5, 3, 7], device="npu", dtype=torch.int32),
            torch.tensor([0, 3], device="npu", dtype=torch.int32),
            torch.tensor([0], device="npu", dtype=torch.int64),
            torch.tensor([True], device="npu"),
            torch.zeros((2, 8), device="npu", dtype=torch.int32),
            torch.zeros(2, device="npu", dtype=torch.int32),
            host_tables,
            spec=spec,
            solution="flash_npu_kernel",
            enable_pdl=False,
        ).cpu(),
    )
    assert history[0, :3].cpu().tolist() == [5, 3, 7]


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")
def test_host_lookup_supports_npu_graph_replay() -> None:
    spec = OverEmbeddingSpec(
        profile="longcat-lite-graph-test",
        tp_size=1,
        rank=0,
        vocab_size=32,
        branch_count=2,
        branch_width=256,
        hidden_size=512,
        max_ngram_order=3,
        fragments=(
            TableFragmentSpec(0, 2, 11, 0, 256),
            TableFragmentSpec(1, 3, 13, 0, 256),
        ),
        ignored_token_ids=(3,),
        segment_ignored_tokens=True,
    )
    tables = (
        torch.arange(1, 12, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 256)
        .contiguous(),
        torch.arange(1, 14, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 256)
        .contiguous(),
    )
    register_host_tables_(
        tables,
        device=torch.device("npu", torch.npu.current_device()),
        solution="flash_npu_kernel",
    )
    input_ids = torch.tensor([5, 7], device="npu", dtype=torch.int32)
    offsets = torch.tensor([0, 2], device="npu", dtype=torch.int32)
    slots = torch.tensor([0], device="npu", dtype=torch.int64)
    active = torch.tensor([True], device="npu")
    history = torch.zeros((2, 8), device="npu", dtype=torch.int32)
    committed = torch.zeros(2, device="npu", dtype=torch.int32)
    output = torch.empty((2, 512), device="npu", dtype=torch.bfloat16)

    def lookup() -> torch.Tensor:
        return append_packed_lookup_(
            input_ids,
            offsets,
            slots,
            active,
            history,
            committed,
            tables,
            spec=spec,
            out=output,
            solution="flash_npu_kernel",
            enable_pdl=False,
        )

    for _ in range(3):
        history.zero_()
        lookup()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    history.zero_()
    with torch.npu.graph(
        graph,
        stream=torch.npu.Stream(),
        auto_dispatch_capture=True,
    ):
        graph_output = lookup()
    torch.npu.synchronize()
    assert graph_output.data_ptr() == output.data_ptr()

    input_ids.copy_(torch.tensor([8, 9], device="npu", dtype=torch.int32))
    history.zero_()
    graph.replay()
    torch.npu.synchronize()

    expected = torch.cat((tables[0][[8, 1]], tables[1][[8, 5]]), dim=1)
    torch.testing.assert_close(graph_output.cpu(), expected, atol=0, rtol=0)
    assert history[0, :2].cpu().tolist() == [8, 9]
