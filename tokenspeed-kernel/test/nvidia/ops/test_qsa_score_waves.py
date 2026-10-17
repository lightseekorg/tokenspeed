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

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.qsa import triton as qsa
from tokenspeed_kernel.platform import current_platform, max_active_ctas
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="CUDA score occupancy requires NVIDIA"
)


def _score_launch(graph):
    # AMD collects this module too; load CUDA bindings only when the tests run.
    from cuda.bindings import driver as cu

    raw = cu.CUgraph(graph.raw_cuda_graph())
    status, nodes, count = cu.cuGraphGetNodes(raw, 1)
    assert status == cu.CUresult.CUDA_SUCCESS
    assert count == 1
    status, params = cu.cuGraphKernelNodeGetParams(nodes[0])
    assert status == cu.CUresult.CUDA_SUCCESS
    threads = params.blockDimX * params.blockDimY * params.blockDimZ
    status, resident = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        params.func, threads, params.sharedMemBytes
    )
    assert status == cu.CUresult.CUDA_SUCCESS
    return (params.gridDimX, params.gridDimY, params.gridDimZ), resident


@pytest.mark.parametrize("num_stages", [2, 3, 5])
@pytest.mark.parametrize("batch", [32, 64])
def test_score_wave_budget_tracks_compiled_resources(batch: int, num_stages: int):
    torch.manual_seed(359)
    width, heads, dim, page_size, num_blocks = 4, 4, 128, 64, 65536
    rows = batch * width
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(256, 1, dim, device="cuda", dtype=query.dtype)
    table = torch.ones(1, num_blocks // page_size, device="cuda", dtype=torch.int32)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int64)
    complete = torch.full((rows,), 129, device="cuda", dtype=torch.int32)
    logits = torch.full((rows, num_blocks), float("nan"), device="cuda")
    args = (
        query,
        keys,
        table,
        requests,
        complete,
        logits,
        heads,
        dim,
        num_blocks,
        batch,
        page_size,
        *query.stride(),
        keys.stride(0),
        keys.stride(2),
        table.stride(0),
        logits.stride(0),
    )
    constants = {
        "BLOCK_N": 128,
        "BLOCK_H": heads,
        "BLOCK_D": dim,
        "QUERY_GROUP_SIZE": width,
        "LOOP_QUERY_GROUPS": False,
        "ENABLE_PDL": False,
    }
    kernel = qsa._qwen4_exp_qsa_score_blocks_kernel.warmup(
        *args, **constants, grid=(1,), num_warps=4, num_stages=num_stages
    )
    sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
    grid = qsa._score_grid(kernel, batch, triton.cdiv(num_blocks, 128), sm_count)
    kernel[grid](*args, *constants.values())
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        kernel[grid](*args, *constants.values())
    actual_grid, resident = _score_launch(graph)
    assert max_active_ctas(kernel) == resident
    launched = math.prod(actual_grid)
    budget = 3 * sm_count * resident
    assert launched <= budget
    # Capacity is large enough to fill the budget to within one group column.
    assert budget - launched < batch
    expected = torch.relu(query.float() @ keys[64:128, 0].float().T).sum(dim=1)
    expected = expected.repeat(1, 3)[:, :129]
    graph.replay()
    torch.testing.assert_close(logits[:, :129], expected, rtol=1e-5, atol=1e-4)
    assert torch.isnan(logits[:, 129:]).all()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("large_batch", [False, True])
def test_score_launch_bounds_waves_and_refreshes_groups(
    enabled: bool, large_batch: bool
):
    torch.manual_seed(367)
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    # Exceed three waves even at the hardware's maximum block residency.
    rows = 64 * sm_count + 1 if large_batch else 128
    heads, dim, page_size, num_blocks = 3, 32, 64, 1024
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 1, dim, device="cuda", dtype=query.dtype)
    table = torch.ones(1, num_blocks // page_size, device="cuda", dtype=torch.int32)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int64)
    complete = torch.full((rows,), 129, device="cuda", dtype=torch.int32)

    def forward(query_view):
        return qsa.triton_qsa_block_scores(
            query_view,
            keys,
            table,
            requests,
            complete,
            page_size=page_size,
            queries_per_request=None,
            enable_pdl=enabled,
        )

    forward(query)
    with assert_no_triton_compile(qsa._qwen4_exp_qsa_score_blocks_kernel):
        # Runtime row/group counts must reuse the same specialization.
        forward(query[:-1])
        if not large_batch:
            for count in (1, 3, 16, 32):
                result = forward(query[:count])
                expected = torch.relu(
                    query[:count].float() @ keys[64:128, 0].float().T
                ).sum(dim=1)
                expected = expected.repeat(1, 3)[:, :129]
                torch.testing.assert_close(
                    result[:, :129], expected, rtol=1e-5, atol=1e-4
                )
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            actual = forward(query)
    grid, resident = _score_launch(graph)
    assert math.prod(grid) <= 3 * sm_count * resident
    if large_batch:
        assert grid[0] < rows
    for length in (129, 0, 17, 129):
        query.normal_()
        complete.fill_(length)
        graph.replay()
        expected = torch.relu(query.float() @ keys[64:128, 0].float().T).sum(dim=1)
        expected = expected.repeat(1, triton.cdiv(length, page_size))[:, :length]
        torch.testing.assert_close(actual[:, :length], expected, rtol=1e-5, atol=1e-4)
