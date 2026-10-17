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

"""Single-NPU graph regression for compact owned pages and query endpoints."""

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    compact_kvp_pages,
    kvp_local_visible_lengths,
)

torch.npu.set_device(0)
torch.manual_seed(27)
checks = 0
for rows, cols in [(1, 3), (8, 64), (64, 64), (2, 4096)]:
    for degree in (1, 8):
        for with_prefix in (False, True):
            table = torch.randint(-1, 67, (rows, cols), dtype=torch.int32)
            table[0].zero_()
            lengths = torch.randint(0, cols * 64 + 1, (rows,), dtype=torch.int32)
            visible = torch.stack((lengths // 2, lengths), dim=1)
            table_n, lengths_n, visible_n = table.npu(), lengths.npu(), visible.npu()
            out = torch.empty_like(table_n)
            local = torch.empty_like(lengths_n)
            prefix = (
                torch.empty((rows, cols + 1), dtype=torch.int32, device="npu")
                if with_prefix
                else None
            )
            endpoints = torch.empty_like(visible_n)
            kwargs = dict(
                page_size=64,
                block_granularity=128,
                virtual_block_count=32,
                degree=degree,
                rank=degree - 1,
            )

            def run():
                compact_kvp_pages(
                    table_n,
                    lengths_n,
                    out=out,
                    local_lengths=local,
                    page_prefix=prefix,
                    **kwargs,
                )
                if prefix is not None:
                    kvp_local_visible_lengths(
                        prefix,
                        visible_n,
                        page_size=64,
                        out=endpoints,
                        local_lengths=local,
                    )

            stream = torch.npu.Stream()
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                for _ in range(3):
                    run()
                    torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
                run()
            for iteration in range(2):
                if iteration:
                    table = table.roll(1, 1)
                    lengths = (lengths // 2).clamp_min(0)
                    visible = torch.stack((lengths // 2, lengths), dim=1)
                    table_n.copy_(table)
                    lengths_n.copy_(lengths)
                    visible_n.copy_(visible)
                    torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                expected = torch.empty_like(table)
                expected_lengths = torch.empty_like(lengths)
                expected_prefix = (
                    torch.empty((rows, cols + 1), dtype=torch.int32)
                    if with_prefix
                    else None
                )
                compact_kvp_pages(
                    table,
                    lengths,
                    out=expected,
                    local_lengths=expected_lengths,
                    page_prefix=expected_prefix,
                    **kwargs,
                )
                if with_prefix:
                    expected_endpoints = torch.empty_like(visible)
                    kvp_local_visible_lengths(
                        expected_prefix,
                        visible,
                        page_size=64,
                        out=expected_endpoints,
                        local_lengths=expected_lengths,
                    )
                    torch.testing.assert_close(
                        prefix.cpu(), expected_prefix, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        endpoints.cpu(), expected_endpoints, rtol=0, atol=0
                    )
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    local.cpu(), expected_lengths, rtol=0, atol=0
                )
                checks += 1
print(f"ASCEND_DCP_METADATA_GRAPH_PASS checks={checks}", flush=True)
