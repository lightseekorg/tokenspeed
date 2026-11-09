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

"""Real CUDA graph partition and cross-stream result visibility."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="CUDA graph experiment"
)


@pytest.mark.parametrize("scheduling", ["serial", "overlap"])
def test_partition_replays_live_inputs_and_joins_draft(scheduling):
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/nvidia/gdn_commit_overlap/graph_split.py"
    )
    spec = importlib.util.spec_from_file_location("gdn_commit_graph_split", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = module.DraftState(scheduling=scheduling)
    x = torch.randn(64, 64, device="cuda")
    y = torch.randn_like(x)
    z = torch.randn_like(x)
    out = torch.empty_like(x)
    after_join = torch.empty_like(x)
    independent_copy = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            torch.mm(x @ y, z, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        intermediate = x @ y
        state.mark()
        torch.mm(intermediate, z, out=out)
    del intermediate
    # A useful independent branch models the root/side nodes present in the
    # actual serving graph. It must survive partitioning, even though it has
    # no dependency on the MTP marker.
    cu = module.cu
    params = cu.CUDA_MEMCPY3D()
    params.srcMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
    params.srcDevice = cu.CUdeviceptr(x.data_ptr())
    params.srcPitch = 64 * x.element_size()
    params.srcHeight = 64
    params.dstMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
    params.dstDevice = cu.CUdeviceptr(independent_copy.data_ptr())
    params.dstPitch = 64 * x.element_size()
    params.dstHeight = 64
    params.WidthInBytes = 64 * x.element_size()
    params.Height = 64
    params.Depth = 1
    (context,) = module.checked(cu.cuCtxGetCurrent())
    module.checked(
        cu.cuGraphAddMemcpyNode(
            cu.CUgraph(graph.raw_cuda_graph()), [], 0, params, context
        )
    )
    try:
        for _ in range(4):
            x.normal_()
            state.replay(graph, batch=8, graph_batch=8)
            # Independent main-stream work stands in for state commit. The
            # following read must observe the draft even on a different stream.
            after_join.zero_()
            state.join()
            after_join.copy_(out)
            torch.testing.assert_close(after_join, (x @ y) @ z)
            torch.testing.assert_close(independent_copy, x, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        for pair in state.graphs.values():
            pair.close()


@pytest.mark.parametrize("scheduling", ["serial", "overlap"])
def test_partition_preserves_gdn_pdl_state_writes(scheduling):
    from tokenspeed_kernel.platform import pdl_enabled

    test_root = Path(__file__).resolve().parents[1]
    case_path = test_root / "nvidia/ops/test_gdn_replay_pipeline.py"
    case_spec = importlib.util.spec_from_file_location("gdn_partition_case", case_path)
    case_module = importlib.util.module_from_spec(case_spec)
    sys.modules[case_spec.name] = case_module
    case_spec.loader.exec_module(case_module)
    module = sys.modules.get("gdn_commit_graph_split")
    if module is None:
        path = test_root.parent / "benchmarks/nvidia/gdn_commit_overlap/graph_split.py"
        spec = importlib.util.spec_from_file_location("gdn_commit_graph_split", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    case = case_module._case(
        batch=5,
        tokens=4,
        layers=3,
        state_dtype=torch.bfloat16,
        row_padding=0,
        token_capacity=23,
    )
    state = module.DraftState(scheduling=scheduling)
    saved = pdl_enabled()
    pdl_enabled(True)
    x = torch.randn(64, 64, device="cuda")
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            case.launch(buffers=2, max_ctas=3)
            torch.mm(x, x, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            case.launch(buffers=2, max_ctas=3)
            state.mark()
            torch.mm(x, x, out=out)
        for counts in ([0, 1, 2, 3, 4], [4, 2, 1, 0, 4]):
            case.accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            initial = case.storage.clone()
            state.replay(graph, batch=5, graph_batch=5)
            state.join()
            torch.cuda.synchronize()
            case_module._check_result(case, initial)
            torch.testing.assert_close(out, x @ x)
    finally:
        torch.cuda.synchronize()
        pdl_enabled(saved)
        for pair in state.graphs.values():
            pair.close()
