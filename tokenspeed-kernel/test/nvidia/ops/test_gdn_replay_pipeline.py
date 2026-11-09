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

"""Accepted-prefix behavior of bounded-grid, asynchronously prefetched replay."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.gdn import (
    gdn_gate,
    gdn_l2norm,
    torch_gdn_recurrence,
)
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

if not current_platform().is_nvidia:
    pytest.skip("the pipeline experiment uses NVIDIA cp.async", allow_module_level=True)

from tokenspeed_kernel.ops.attention.gdn.gluon_nvidia import (
    _gdn_replay_commit_persistent_kernel,
    launch_gdn_replay_commit_persistent,
)


@pytest.mark.parametrize("batch", [8, 32])
def test_replay_pipeline_registry_dispatch(batch):
    from tokenspeed_kernel.ops.attention.gdn import gdn_replay_commit

    case = _case(
        batch=batch,
        tokens=4,
        layers=3,
        state_dtype=torch.bfloat16,
        row_padding=3,
        token_capacity=batch * 4 + 3,
    )
    initial = case.storage.clone()
    gdn_replay_commit(
        case.payload,
        case.parameters,
        state_addresses=case.addresses,
        state_row_strides=case.strides,
        read_indices=case.read,
        write_indices=case.write,
        accepted_length=case.accepted,
        draft_token_num=case.tokens,
        geometry=(2, 6, 128, 128),
        state_dtype=case.pool.dtype,
        override="gluon_nvidia_gdn_replay_commit",
    )
    _check_result(case, initial)


@dataclass
class ReplayCase:
    payload: torch.Tensor
    parameters: torch.Tensor
    storage: torch.Tensor
    pool: torch.Tensor
    addresses: torch.Tensor
    strides: torch.Tensor
    read: torch.Tensor
    write: torch.Tensor
    accepted: torch.Tensor
    tokens: int
    geometry: tuple[int, int, int, int]

    def launch(self, *, buffers: int, max_ctas: int):
        return launch_gdn_replay_commit_persistent(
            self.payload,
            self.parameters,
            state_addresses=self.addresses,
            state_row_strides=self.strides,
            read_indices=self.read,
            write_indices=self.write,
            accepted_length=self.accepted,
            draft_token_num=self.tokens,
            geometry=self.geometry,
            state_dtype=self.pool.dtype,
            max_ctas=max_ctas,
            num_buffers=buffers,
            num_warps=8,
            max_registers=168,
        )


def _make_case(
    *,
    batch: int,
    tokens: int,
    layers: int,
    state_dtype: torch.dtype,
    row_padding: int,
    token_capacity: int,
    geometry: tuple[int, int, int, int],
) -> ReplayCase:
    torch.manual_seed(3927)
    h, hv, k, v = geometry
    width = h * k + hv * v + 2 * hv
    dtype = torch.float16 if state_dtype == torch.float16 else torch.bfloat16
    payload = torch.randn(layers, token_capacity, width, device="cuda", dtype=dtype)
    parameters = torch.randn(layers, 2, hv, device="cuda") * 0.1
    storage = (
        torch.randn(
            layers,
            batch * 2 + 3,
            hv * v * k + row_padding,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.02
    )
    pool = storage[..., : hv * v * k].view(layers, batch * 2 + 3, hv, v, k)
    read = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).repeat(
        layers, 1
    )
    for layer in range(layers):
        read[layer].copy_(read[layer].roll(layer))
    write = torch.arange(
        batch + 1, batch * 2 + 1, device="cuda", dtype=torch.int32
    ).repeat(layers, 1)
    read[:, 0] = -1
    if batch > 1:
        write[:, 1] = read[:, 1]
    if batch > 2:
        write[:, -1] = -1
    accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % (tokens + 1)
    return ReplayCase(
        payload,
        parameters,
        storage,
        pool,
        torch.tensor(
            [pool[i].data_ptr() for i in range(layers)],
            device="cuda",
            dtype=torch.uint64,
        ),
        torch.full((layers,), pool.stride(1), device="cuda", dtype=torch.int64),
        read,
        write,
        accepted,
        tokens,
        geometry,
    )


def _case(
    *,
    batch: int,
    tokens: int,
    layers: int,
    state_dtype: torch.dtype,
    row_padding: int,
    token_capacity: int,
) -> ReplayCase:
    return _make_case(
        batch=batch,
        tokens=tokens,
        layers=layers,
        state_dtype=state_dtype,
        row_padding=row_padding,
        token_capacity=token_capacity,
        geometry=(2, 6, 128, 128),
    )


def _check_result(case: ReplayCase, initial_storage: torch.Tensor) -> None:
    layers, batch = case.read.shape
    h, hv, k, v = case.geometry
    expected = initial_storage.clone()
    initial_pool = initial_storage[..., : hv * v * k].view(case.pool.shape)
    expected_pool = expected[..., : hv * v * k].view(case.pool.shape)
    written = torch.zeros_like(expected, dtype=torch.bool)
    accepted = case.accepted.clamp(min=0, max=case.tokens)
    for layer in range(layers):
        records = case.payload[layer, : batch * case.tokens].view(
            batch, case.tokens, -1
        )
        keys = gdn_l2norm(
            records[..., : h * k].view(batch, case.tokens, h, k).float(), 1e-6
        )
        values = records[..., h * k : h * k + hv * v].view(batch, case.tokens, hv, v)
        initial = initial_pool[layer, case.read[layer].clamp_min(0).long()].clone()
        initial[case.read[layer] < 0] = 0
        _, states = torch_gdn_recurrence(
            keys,
            keys,
            values,
            gdn_gate(
                records[..., -2 * hv : -hv].float(),
                case.parameters[layer, 1],
                case.parameters[layer, 0],
            ),
            torch.sigmoid(records[..., -hv:].float()),
            initial,
            scale=1.0,
            parents=None,
            compute_dtype=torch.float32,
        )
        for request, count in enumerate(accepted.tolist()):
            destination = int(case.write[layer, request])
            if destination < 0:
                continue
            final = states[request, count - 1] if count else initial[request]
            expected_pool[layer, destination].copy_(final)
            written[layer, destination, : hv * v * k] = True
    rtol, atol = {
        torch.bfloat16: (2e-2, 1e-3),
        torch.float16: (1e-3, 1e-6),
        torch.float32: (1e-5, 1e-6),
    }[case.pool.dtype]
    torch.testing.assert_close(case.storage, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        case.storage[~written], initial_storage[~written], rtol=0, atol=0
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("buffers", [1, 2])
@pytest.mark.parametrize("row_padding", [0, 3])
def test_replay_pipeline_accepted_prefix(state_dtype, buffers, row_padding):
    case = _case(
        batch=5,
        tokens=4,
        layers=3,
        state_dtype=state_dtype,
        row_padding=row_padding,
        token_capacity=23,
    )
    for lengths in ([0, 1, 7, 4, 2], [4, -1, 1, 2, 4]):
        case.accepted.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        initial = case.storage.clone()
        case.launch(buffers=buffers, max_ctas=7)
        _check_result(case, initial)


@pytest.mark.parametrize("tokens", [1, 8, 16])
def test_replay_pipeline_ignores_rejected_suffix(tokens):
    case = _case(
        batch=5,
        tokens=tokens,
        layers=2,
        state_dtype=torch.bfloat16,
        row_padding=0,
        token_capacity=5 * tokens + 3,
    )
    case.accepted.copy_(
        torch.tensor(
            [tokens, 0, 1, max(0, tokens - 1), tokens], device="cuda", dtype=torch.int32
        )
    )
    packed = case.payload[:, : 5 * tokens].view(2, 5, tokens, -1)
    rejected = torch.arange(tokens, device="cuda")[None, :] >= case.accepted[:, None]
    packed[:, rejected] = float("nan")
    initial = case.storage.clone()
    case.launch(buffers=2, max_ctas=3)
    _check_result(case, initial)


@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4])
def test_replay_pipeline_actual_accepted_length(batch, accepted):
    case = _case(
        batch=batch,
        tokens=4,
        layers=2,
        state_dtype=torch.bfloat16,
        row_padding=0,
        token_capacity=batch * 4 + 3,
    )
    case.accepted.fill_(accepted)
    records = case.payload[:, : batch * 4].view(2, batch, 4, -1)
    records[:, :, accepted:] = float("nan")
    initial = case.storage.clone()
    case.launch(buffers=2, max_ctas=24)
    _check_result(case, initial)


def test_replay_pipeline_graph_refresh():
    case = _case(
        batch=5,
        tokens=4,
        layers=2,
        state_dtype=torch.bfloat16,
        row_padding=0,
        token_capacity=27,
    )
    case.launch(buffers=2, max_ctas=3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.launch(buffers=2, max_ctas=3)
    for offset in (1, 3):
        case.accepted.copy_(
            (torch.arange(5, device="cuda") + offset).to(torch.int32) % 5
        )
        case.read[:, 2:].copy_(case.read[:, 2:].roll(1, dims=1))
        case.payload.normal_()
        initial = case.storage.clone()
        graph.replay()
        _check_result(case, initial)


def test_replay_pipeline_shapes_and_cta_budget_do_not_recompile():
    cases = [
        _case(
            batch=b,
            tokens=t,
            layers=l,
            state_dtype=torch.bfloat16,
            row_padding=0,
            token_capacity=capacity,
        )
        for b, t, l, capacity in [
            (3, 3, 2, 19),
            (5, 4, 3, 31),
            (32, 4, 1, 131),
            (2, 3, 1, 13),
        ]
    ]
    cases[0].launch(buffers=2, max_ctas=3)
    with assert_no_triton_compile(_gdn_replay_commit_persistent_kernel):
        for case, budget in zip(cases, (3, 7, 24, 1), strict=True):
            initial = case.storage.clone()
            case.launch(buffers=2, max_ctas=budget)
            _check_result(case, initial)


@dataclass(frozen=True)
class SavedReplay:
    module: ModuleType
    warps: int
    registers: int
    buffers: int


def _load_saved_replay(
    name: str, *, warps: int, registers: int, buffers: int
) -> SavedReplay:
    path = (
        Path(__file__).resolve().parents[3]
        / f"benchmarks/nvidia/gdn_replay_cta16/{name}.py"
    )
    module_name = f"gdn_replay_cta16_{name}"
    if module_name in sys.modules:
        return SavedReplay(sys.modules[module_name], warps, registers, buffers)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return SavedReplay(module, warps, registers, buffers)


@pytest.fixture(
    scope="module",
    params=[
        ("kernel", 8, 168, 2),
        ("kernel_stream", 16, 96, 2),
        ("kernel_stream", 16, 96, 3),
    ],
    ids=["scaled", "stream-2stage", "stream-3stage"],
)
def cta16_candidate(request):
    name, warps, registers, buffers = request.param
    return _load_saved_replay(name, warps=warps, registers=registers, buffers=buffers)


@pytest.fixture(scope="module")
def counted_stream():
    return _load_saved_replay("kernel_stream", warps=16, registers=96, buffers=2)


def _launch_cta16(candidate: SavedReplay, case: ReplayCase):
    return candidate.module.launch_gdn_replay_commit_persistent(
        case.payload,
        case.parameters,
        state_addresses=case.addresses,
        state_row_strides=case.strides,
        read_indices=case.read,
        write_indices=case.write,
        accepted_length=case.accepted,
        draft_token_num=case.tokens,
        geometry=case.geometry,
        state_dtype=case.pool.dtype,
        max_ctas=16,
        num_buffers=candidate.buffers,
        num_warps=candidate.warps,
        max_registers=candidate.registers,
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 3, 4, 8, 16])
@pytest.mark.parametrize("row_padding", [0, 3])
def test_cta16_candidate_prefix_oracle(
    cta16_candidate, state_dtype, tokens, row_padding
):
    case = _case(
        batch=5,
        tokens=tokens,
        layers=2,
        state_dtype=state_dtype,
        row_padding=row_padding,
        token_capacity=5 * tokens + 3,
    )
    case.accepted.copy_(
        torch.tensor([0, 1, tokens + 3, -1, tokens], device="cuda", dtype=torch.int32)
    )
    if cta16_candidate.buffers == 3 and state_dtype == torch.float32 and tokens > 4:
        with pytest.raises(
            ValueError, match="three buffers support FP32 state windows"
        ):
            _launch_cta16(cta16_candidate, case)
        return
    initial = case.storage.clone()
    _launch_cta16(cta16_candidate, case)
    _check_result(case, initial)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("a_log", [-12.0, 6.0, 20.0])
def test_cta16_candidate_extreme_decay(cta16_candidate, state_dtype, a_log):
    case = _case(
        batch=5,
        tokens=4,
        layers=2,
        state_dtype=state_dtype,
        row_padding=0,
        token_capacity=23,
    )
    case.parameters[:, 0].fill_(a_log)
    case.accepted.copy_(torch.arange(5, device="cuda", dtype=torch.int32))
    initial = case.storage.clone()
    _launch_cta16(cta16_candidate, case)
    _check_result(case, initial)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cta16_candidate_rejected_nan(cta16_candidate, state_dtype):
    case = _case(
        batch=5,
        tokens=4,
        layers=2,
        state_dtype=state_dtype,
        row_padding=3,
        token_capacity=23,
    )
    case.accepted.copy_(torch.tensor([1, 2, 3, 4, 0], device="cuda", dtype=torch.int32))
    packed = case.payload[:, :20].view(2, 5, 4, -1)
    rejected = torch.arange(4, device="cuda")[None, :] >= case.accepted[:, None]
    packed[:, rejected] = float("nan")
    initial = case.storage.clone()
    _launch_cta16(cta16_candidate, case)
    _check_result(case, initial)


def test_cta16_candidate_graph_refresh(cta16_candidate):
    case = _case(
        batch=5,
        tokens=4,
        layers=2,
        state_dtype=torch.bfloat16,
        row_padding=0,
        token_capacity=23,
    )
    _launch_cta16(cta16_candidate, case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_cta16(cta16_candidate, case)
    for offset in [1, 3]:
        case.payload.normal_()
        case.accepted.copy_(
            (torch.arange(5, device="cuda") + offset).to(torch.int32) % 5
        )
        initial = case.storage.clone()
        graph.replay()
        _check_result(case, initial)


def test_cta16_candidate_runtime_scalars_reuse_compilation(cta16_candidate):
    cases = [
        _case(
            batch=batch,
            tokens=tokens,
            layers=layers,
            state_dtype=torch.bfloat16,
            row_padding=0,
            token_capacity=capacity,
        )
        for batch, tokens, layers, capacity in [
            (3, 3, 2, 19),
            (8, 4, 3, 37),
            (32, 4, 1, 131),
            (2, 3, 1, 13),
        ]
    ]
    _launch_cta16(cta16_candidate, cases[0])
    with assert_no_triton_compile(
        cta16_candidate.module._gdn_replay_commit_persistent_kernel
    ):
        for case in cases:
            initial = case.storage.clone()
            _launch_cta16(cta16_candidate, case)
            _check_result(case, initial)


@pytest.mark.parametrize("row_padding", [0, 3])
def test_cta16_candidate_many_ring_wraps(cta16_candidate, row_padding):
    case = _case(
        batch=32,
        tokens=4,
        layers=5,
        state_dtype=torch.bfloat16,
        row_padding=row_padding,
        token_capacity=131,
    )
    # Every CTA processes 60 items, repeatedly reusing the state/input rings.
    lengths = torch.tensor([4, 0, 3, 1, 4, 2, 0, 1], device="cuda", dtype=torch.int32)
    case.accepted.copy_(lengths.repeat(4))
    _launch_cta16(cta16_candidate, case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_cta16(cta16_candidate, case)
    for offset in (1, 3, 0):
        case.payload.normal_()
        case.accepted.copy_((lengths.repeat(4) + offset) % 5)
        initial = case.storage.clone()
        graph.replay()
        _check_result(case, initial)


@pytest.mark.parametrize("row_padding", [0, 3])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cta16_stream_tp4_state_dtype(counted_stream, state_dtype, row_padding):
    candidate = SavedReplay(counted_stream.module, 16, 96, 3)
    case = _make_case(
        batch=8,
        tokens=4,
        layers=3,
        state_dtype=state_dtype,
        row_padding=row_padding,
        token_capacity=35,
        geometry=(4, 12, 128, 128),
    )
    case.accepted.copy_(
        torch.tensor([0, 1, 2, 3, 4, 5, -1, 4], device="cuda", dtype=torch.int32)
    )
    initial = case.storage.clone()
    _launch_cta16(candidate, case)
    _check_result(case, initial)


@pytest.mark.parametrize("row_padding", [0, 3])
@pytest.mark.parametrize("buffers", [1, 2, 3])
@pytest.mark.parametrize("warps", [4, 8, 16])
def test_cta16_stream_tp4_publication(counted_stream, warps, buffers, row_padding):
    candidate = SavedReplay(counted_stream.module, warps, 96, buffers)
    case = _make_case(
        batch=8,
        tokens=4,
        layers=3,
        state_dtype=torch.bfloat16,
        row_padding=row_padding,
        token_capacity=35,
        geometry=(4, 12, 128, 128),
    )
    lengths = torch.tensor([4, 0, 3, 1, 2, 4, 0, 1], device="cuda", dtype=torch.int32)
    case.accepted.copy_(lengths)
    _launch_cta16(candidate, case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_cta16(candidate, case)
    for offset in (1, 3):
        case.payload.normal_()
        case.accepted.copy_((lengths + offset) % 5)
        # Preserve the in-place request while refreshing the other source rows.
        case.read[:, 2:].copy_(case.read[:, 2:].roll(1, dims=1))
        initial = case.storage.clone()
        graph.replay()
        _check_result(case, initial)
