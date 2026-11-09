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

"""GPU correctness and placement of the benchmark green-context stream."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.platform import current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="CUDA SM placement experiment"
)


@pytest.fixture(autouse=True)
def require_green_resource_groups():
    cu = pytest.importorskip("cuda.bindings.driver")
    if cu.cuDriverGetVersion()[1] < 13020:
        pytest.skip("placement experiment requires CUDA driver API 13.2")
    torch.cuda.init()
    result = cu.cuDeviceGetDevResource(
        cu.CUdevice(torch.cuda.current_device()),
        cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
    )
    if result[0] != cu.CUresult.CUDA_SUCCESS or result[1].sm.smCount < 32:
        pytest.skip("requires at least 32 available CUDA SMs")
    params = cu.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
    params.smCount = 16
    params.coscheduledSmCount = 16
    params.preferredCoscheduledSmCount = 16
    params.flags = 0
    if (
        cu.cuDevSmResourceSplit(1, result[1], 0, [params])[0]
        != cu.CUresult.CUDA_SUCCESS
    ):
        pytest.skip("requires a co-scheduled group of 16 SMs")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@gluon.jit
def record_smids(output):
    sm = gl.inline_asm_elementwise(
        "mov.u32 $0, %smid;",
        constraints="=r",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.store(output + gl.program_id(0), sm)


@pytest.mark.parametrize("name", ["green16", "green32", "green32_cta32"])
def test_green_stream_bounds_smids_and_captures_live_inputs(name):
    root = Path(__file__).resolve().parents[2]
    module = load_file(
        "gdn_placement", root / "benchmarks/nvidia/gdn_commit_overlap/placement.py"
    )
    placement = module.CommitPlacement(spec=module.PLACEMENTS[name])
    x = torch.randn(4096, device="cuda")
    y = torch.empty_like(x)
    smids = torch.empty(1024, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    try:
        with placement.ordered():
            record_smids[(1024,)](smids, num_warps=4)
            torch.add(x, 2, out=y)
        torch.testing.assert_close(y, x + 2)
        assert smids.unique().numel() <= placement.spec.sm_count
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with placement.ordered():
                record_smids[(1024,)](smids, num_warps=4)
                torch.mul(x, 3, out=y)
        for _ in range(3):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(y, x * 3)
            assert smids.unique().numel() <= placement.spec.sm_count
    finally:
        torch.cuda.synchronize()
        del graph
        placement.close()


@pytest.mark.parametrize(
    "name", ["unrestricted16", "green16", "green32", "unrestricted32", "green32_cta32"]
)
def test_placed_gdn_preserves_accepted_states(name):
    root = Path(__file__).resolve().parents[2]
    module = load_file(
        "gdn_placement", root / "benchmarks/nvidia/gdn_commit_overlap/placement.py"
    )
    cases = load_file(
        "gdn_placement_cases", root / "test/nvidia/ops/test_gdn_replay_pipeline.py"
    )
    case = cases._case(
        batch=5,
        tokens=4,
        layers=3,
        state_dtype=torch.bfloat16,
        row_padding=3,
        token_capacity=23,
    )
    placement = module.CommitPlacement(spec=module.PLACEMENTS[name])
    saved = pdl_enabled()
    pdl_enabled(True)
    graph = None
    arguments = dict(
        state_addresses=case.addresses,
        state_row_strides=case.strides,
        read_indices=case.read,
        write_indices=case.write,
        accepted_length=case.accepted,
        draft_token_num=4,
        geometry=(2, 6, 128, 128),
        state_dtype=case.pool.dtype,
    )
    try:
        placement.launch(case.payload, case.parameters, **arguments)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            placement.launch(case.payload, case.parameters, **arguments)
        for counts in ([0, 1, 2, 3, 4], [4, 2, 0, 1, 3]):
            case.accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            initial = case.storage.clone()
            graph.replay()
            torch.cuda.synchronize()
            cases._check_result(case, initial)
    finally:
        torch.cuda.synchronize()
        del graph
        pdl_enabled(saved)
        placement.close()


def test_runtime_control_switches_real_gdn_placement(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    directory = root / "benchmarks/nvidia/gdn_commit_overlap"
    monkeypatch.syspath_prepend(str(directory))
    module = load_file("gdn_runtime_placement", directory / "placement_runtime.py")
    cases = load_file(
        "gdn_runtime_cases", root / "test/nvidia/ops/test_gdn_replay_pipeline.py"
    )
    control_file = tmp_path / "placement-control"
    control_file.write_bytes(bytes([0]))
    control = module.PlacementControl(control_path=control_file)
    case = cases._case(
        batch=5,
        tokens=4,
        layers=3,
        state_dtype=torch.bfloat16,
        row_padding=3,
        token_capacity=23,
    )
    try:
        for index in [0, 2, 1, 3, 5, 4]:
            control.control[0] = index
            initial = case.storage.clone()
            control.launch(
                case.payload,
                case.parameters,
                state_addresses=case.addresses,
                state_row_strides=case.strides,
                read_indices=case.read,
                write_indices=case.write,
                accepted_length=case.accepted,
                draft_token_num=4,
                geometry=(2, 6, 128, 128),
                state_dtype=case.pool.dtype,
            )
            torch.cuda.synchronize()
            cases._check_result(case, initial)
    finally:
        torch.cuda.synchronize()
        control.close()
