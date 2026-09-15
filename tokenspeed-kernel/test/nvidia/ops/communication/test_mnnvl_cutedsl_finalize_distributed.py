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

"""Strict TP8 deferred-finalize numerical and CUDA Graph contracts.

Launch this test on two four-GPU GB300 Slurm segments with::

    REQUIRE_MNNVL_CUTEDSL=1 torchrun --nnodes=2 --nproc-per-node=4 ... \
      -m pytest -q tokenspeed-kernel/test/ops/communication/\
test_mnnvl_cutedsl_finalize_distributed.py

The oracle deliberately models the two BF16 communication boundaries.  Each
rank accumulates top-k routes in a fixed FP32 order and rounds to BF16; rank
contributions are then accumulated in rank order in FP32 and rounded to BF16
before an FP32 RMSNorm and final BF16 conversion.

Both fixtures call the public TokenSpeed-kernel communication wrappers.  The
HT wrapper reaches the native 3584-wide specialization behind the third-party
boundary, so this covers its 56-pack reduction-shard tail and graph state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

HIDDEN = 3584
TOP_K = 16
TP_SIZE = 8
RMS_EPS = 1e-5
BT_MAX_TOKENS = 257
HT_GRAPH_TOKENS = 257
HT_MAX_TOKENS = 2048


def _required() -> bool:
    return os.environ.get("REQUIRE_MNNVL_CUTEDSL") == "1"


def _unavailable(message: str) -> None:
    if _required():
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


@pytest.fixture(scope="module")
def distributed_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != TP_SIZE:
        _unavailable("deferred-finalize validation requires exactly eight ranks")
    if not torch.cuda.is_available():
        _unavailable("deferred-finalize validation requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 10:
        _unavailable("deferred-finalize validation requires data-center Blackwell")

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        if dist.get_world_size() != TP_SIZE:
            pytest.fail("initialized process group is not TP8", pytrace=False)
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


def _collective_support_vote(local: bool, group: dist.ProcessGroup) -> bool:
    vote = torch.tensor(
        [int(local)],
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    return bool(vote.item())


@pytest.fixture(scope="module")
def bt_workspace(distributed_group):
    from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
        MNNVLCuteDSLBTFinalizeTuning,
        MNNVLCuteDSLFinalizeAllReduceRMSNorm,
        mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported,
    )

    group = distributed_group
    local_support = mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported(
        group=group,
        tp_size=TP_SIZE,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        dtype=torch.bfloat16,
        candidate_min_tokens=1,
        candidate_max_tokens=BT_MAX_TOKENS,
    )
    if not _collective_support_vote(local_support, group):
        _unavailable("rank-agreed BT deferred-finalize support is unavailable")

    return MNNVLCuteDSLFinalizeAllReduceRMSNorm.initialize(
        group=group,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        candidate_min_tokens=1,
        candidate_max_tokens=BT_MAX_TOKENS,
        tuning_routes=(
            MNNVLCuteDSLBTFinalizeTuning(
                max_tokens=BT_MAX_TOKENS,
                elements_per_thread=2,
                threads=256,
                prefetch_group=1,
                reduction_threads=224,
                rms_threads=448,
                enable_pdl=True,
            ),
        ),
    )


@pytest.fixture(scope="module")
def ht_workspace(distributed_group):
    try:
        from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
            MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
            MNNVLCuteDSLHTFinalizeTuning,
            mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported,
        )

        local_support = mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported(
            group=distributed_group,
            tp_size=TP_SIZE,
            hidden_size=HIDDEN,
            top_k=TOP_K,
            dtype=torch.bfloat16,
            candidate_min_tokens=1,
            candidate_max_tokens=HT_MAX_TOKENS,
        )
    except Exception:  # The capability check must agree before any allocation.
        local_support = False
    if not _collective_support_vote(local_support, distributed_group):
        _unavailable("rank-agreed native-H3584 HT support is unavailable")

    return MNNVLCuteDSLHTFinalizeAllReduceRMSNorm.initialize(
        group=distributed_group,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        candidate_min_tokens=1,
        candidate_max_tokens=HT_MAX_TOKENS,
        tuning_routes=(
            MNNVLCuteDSLHTFinalizeTuning(
                max_tokens=HT_MAX_TOKENS,
                persistent_ctas=None,
                consumer_threads=448,
                vectors_per_thread=1,
                stages=7,
                reduction_warps=2,
                reduction_cta_groups=None,
                rms_token_groups=2,
                rms_pipeline_stages=3,
                rms_shard_major=False,
                enable_pdl=True,
            ),
        ),
    )


@dataclass
class _Inputs:
    routed: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    gamma: torch.Tensor


@dataclass
class _CapturedGraphCase:
    label: str
    inputs: _Inputs
    output: torch.Tensor
    alias: torch.Tensor
    graph: torch.cuda.CUDAGraph


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _random_inputs(
    m: int,
    rank: int,
    device: torch.device,
    seed: int,
    *,
    gamma_ones: bool = False,
) -> _Inputs:
    routed = (
        torch.randn(m * TOP_K, HIDDEN, generator=_generator(seed + rank)) * 0.125
    ).to(device=device, dtype=torch.bfloat16)
    weights = (torch.randn(m, TOP_K, generator=_generator(seed + 10_000)) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    indices = torch.arange(m * TOP_K, dtype=torch.int32, device=device).reshape(
        m, TOP_K
    )
    slots = torch.arange(m * TOP_K, device=device).reshape(m, TOP_K)
    indices.masked_fill_((slots + rank + seed) % 11 == 0, -1)
    if gamma_ones:
        gamma = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    else:
        gamma = (0.5 + torch.rand(HIDDEN, generator=_generator(seed + 20_000))).to(
            device=device, dtype=torch.bfloat16
        )
    return _Inputs(
        routed.contiguous(),
        weights.contiguous(),
        indices.contiguous(),
        gamma.contiguous(),
    )


def _empty_rank_inputs(m: int, rank: int, device: torch.device, seed: int) -> _Inputs:
    weights = (0.25 + torch.rand(m, TOP_K, generator=_generator(seed)) * 0.25).to(
        device=device, dtype=torch.bfloat16
    )
    indices = torch.full((m, TOP_K), -1, dtype=torch.int32, device=device)
    flat_slots = torch.arange(m * TOP_K, device=device)
    if rank == 0:
        routed = torch.empty((0, HIDDEN), dtype=torch.bfloat16, device=device)
    else:
        owned = flat_slots % (TP_SIZE - 1) == rank - 1
        local_rows = int(owned.sum().item())
        routed = (
            torch.randn(local_rows, HIDDEN, generator=_generator(seed + rank)) * 0.125
        ).to(device=device, dtype=torch.bfloat16)
        indices.view(-1)[owned] = torch.arange(
            local_rows, dtype=torch.int32, device=device
        )
    return _Inputs(
        routed.contiguous(),
        weights.contiguous(),
        indices.contiguous(),
        torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
    )


def _sanitize_negative_zero(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    bits = value.view(torch.int16)
    bits.masked_fill_(bits == -32768, 0)
    return value


def _local_finalize(inputs: _Inputs) -> torch.Tensor:
    m = inputs.weights.shape[0]
    local = torch.zeros((m, HIDDEN), dtype=torch.float32, device=inputs.routed.device)
    for route in range(TOP_K):
        rows = inputs.indices[:, route].to(torch.int64)
        valid = rows >= 0
        if bool(valid.any()):
            selected = inputs.routed.index_select(0, rows.clamp_min(0)).float()
            selected.masked_fill_(~valid[:, None], 0.0)
            local.add_(selected * inputs.weights[:, route, None].float())
    return _sanitize_negative_zero(local.to(torch.bfloat16))


def _rank_order_reduce(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    peers = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, local, group=group)
    reduced = torch.zeros_like(local, dtype=torch.float32)
    for peer in peers:
        reduced.add_(peer.float())
    return _sanitize_negative_zero(reduced.to(torch.bfloat16))


def _oracle_components(
    inputs: _Inputs, group: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local = _local_finalize(inputs)
    prenorm = _rank_order_reduce(local, group)
    prenorm_f32 = prenorm.float()
    inv_rms = torch.rsqrt(prenorm_f32.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    output = (prenorm_f32 * inv_rms * inputs.gamma.float()).to(torch.bfloat16)
    return local, prenorm, output


def _oracle(inputs: _Inputs, group: dist.ProcessGroup) -> torch.Tensor:
    return _oracle_components(inputs, group)[2]


def _ordered_bf16(bits: torch.Tensor) -> torch.Tensor:
    unsigned = bits.to(torch.int32) & 0xFFFF
    negative = (unsigned & 0x8000) != 0
    return torch.where(negative, 0x8000 - (unsigned & 0x7FFF), 0x8000 + unsigned)


def _max_bf16_ulp(actual: torch.Tensor, reference: torch.Tensor) -> int:
    actual_bits = actual.view(torch.int16)
    reference_bits = reference.to(torch.bfloat16).view(torch.int16)
    ulp = (_ordered_bf16(actual_bits) - _ordered_bf16(reference_bits)).abs()
    both_zero = (actual == 0) & (reference == 0)
    return int(torch.where(both_zero, 0, ulp).max().item())


def _assert_ulp(
    actual: torch.Tensor,
    reference: torch.Tensor,
    limit: int,
    label: str,
    group: dist.ProcessGroup,
) -> None:
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype == torch.bfloat16
    finite = torch.tensor(
        [int(torch.isfinite(actual).all())],
        dtype=torch.int32,
        device=actual.device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    assert bool(finite.item()), f"{label}: non-finite candidate output"
    local = _max_bf16_ulp(actual, reference)
    rank_max = torch.tensor([local], dtype=torch.int32, device=actual.device)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX, group=group)
    assert (
        int(rank_max.item()) <= limit
    ), f"{label}: rank-max BF16 ULP {int(rank_max.item())} exceeds {limit}"


def _call_ht(workspace, inputs: _Inputs, output: torch.Tensor):
    output.copy_(
        workspace(
            inputs.routed,
            inputs.weights,
            inputs.indices,
            inputs.gamma,
        )
    )
    return output


@pytest.mark.parametrize("m", (1, 33, BT_MAX_TOKENS))
@torch.inference_mode()
def test_bt_matches_fixed_order_oracle(distributed_group, bt_workspace, m: int) -> None:
    rank = dist.get_rank(distributed_group)
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = _random_inputs(m, rank, device, seed=1_000 + m, gamma_ones=m == 33)
    expected = _oracle(inputs, distributed_group)
    actual = bt_workspace(inputs.routed, inputs.weights, inputs.indices, inputs.gamma)
    torch.cuda.synchronize()
    _assert_ulp(actual, expected, 2, f"BT M={m}", distributed_group)
    dist.barrier(group=distributed_group)


@torch.inference_mode()
def test_bt_empty_local_rows_and_minus_one_routes(
    distributed_group, bt_workspace
) -> None:
    rank = dist.get_rank(distributed_group)
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = _empty_rank_inputs(8, rank, device, seed=2_000)
    expected = _oracle(inputs, distributed_group)
    actual = bt_workspace(inputs.routed, inputs.weights, inputs.indices, inputs.gamma)
    torch.cuda.synchronize()
    _assert_ulp(actual, expected, 2, "BT empty-local-row", distributed_group)
    dist.barrier(group=distributed_group)


@torch.inference_mode()
def test_bt_sanitizes_negative_zero(distributed_group, bt_workspace) -> None:
    m = 8
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = _Inputs(
        routed=torch.full(
            (m * TOP_K, HIDDEN),
            -0.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        weights=torch.ones((m, TOP_K), dtype=torch.bfloat16, device=device),
        indices=torch.arange(m * TOP_K, dtype=torch.int32, device=device).reshape(
            m, TOP_K
        ),
        gamma=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
    )
    actual = bt_workspace(inputs.routed, inputs.weights, inputs.indices, inputs.gamma)
    torch.cuda.synchronize()
    assert torch.count_nonzero(actual.view(torch.int16)).item() == 0
    dist.barrier(group=distributed_group)


@torch.inference_mode()
def test_native_ht_matches_fixed_order_oracle(distributed_group, ht_workspace) -> None:
    rank = dist.get_rank(distributed_group)
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = _random_inputs(HT_MAX_TOKENS, rank, device, seed=3_000, gamma_ones=True)
    expected = _oracle(inputs, distributed_group)
    output = torch.empty_like(expected)
    actual = _call_ht(ht_workspace, inputs, output)
    torch.cuda.synchronize()
    # HT uses multimem reduction, whose hardware tree need not reproduce the
    # rank-order FP32 sum bit-for-bit.  Three ULP is the end-to-end contract
    # against the deterministic rank-order oracle, including RMSNorm rounding.
    _assert_ulp(actual, expected, 3, "native HT", distributed_group)
    dist.barrier(group=distributed_group)


@torch.inference_mode()
def test_native_ht_sanitizes_negative_zero(distributed_group, ht_workspace) -> None:
    m = 8
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = _Inputs(
        routed=torch.full(
            (m * TOP_K, HIDDEN),
            -0.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        weights=torch.ones((m, TOP_K), dtype=torch.bfloat16, device=device),
        indices=torch.arange(m * TOP_K, dtype=torch.int32, device=device).reshape(
            m, TOP_K
        ),
        gamma=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
    )
    output = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
    actual = _call_ht(ht_workspace, inputs, output)
    torch.cuda.synchronize()
    assert torch.count_nonzero(actual.view(torch.int16)).item() == 0
    dist.barrier(group=distributed_group)


def _copy_random_inputs(target: _Inputs, rank: int, seed: int) -> None:
    replacement = _random_inputs(
        target.weights.shape[0],
        rank,
        target.routed.device,
        seed,
        gamma_ones=False,
    )
    target.routed.copy_(replacement.routed)
    target.weights.copy_(replacement.weights)
    target.indices.copy_(replacement.indices)
    target.gamma.copy_(replacement.gamma)


@torch.inference_mode()
def test_bt_ht_multicall_changed_input_cuda_graph(
    distributed_group, bt_workspace, ht_workspace
) -> None:
    """Alternate BT/HT and two distinct gammas in one replayable graph."""

    rank = dist.get_rank(distributed_group)
    device = torch.device("cuda", torch.cuda.current_device())
    bt_first = _random_inputs(33, rank, device, seed=4_000)
    bt_second = _random_inputs(33, rank, device, seed=4_100)
    ht_first = _random_inputs(HT_GRAPH_TOKENS, rank, device, seed=4_200)
    ht_second = _random_inputs(HT_GRAPH_TOKENS, rank, device, seed=4_300)
    bt_first_out = torch.empty((33, HIDDEN), dtype=torch.bfloat16, device=device)
    bt_second_out = torch.empty_like(bt_first_out)
    ht_first_out = torch.empty(
        (HT_GRAPH_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device
    )
    ht_second_out = torch.empty_like(ht_first_out)

    def sequence() -> None:
        bt_first_out.copy_(
            bt_workspace(
                bt_first.routed,
                bt_first.weights,
                bt_first.indices,
                bt_first.gamma,
            )
        )
        _call_ht(ht_workspace, ht_first, ht_first_out)
        bt_second_out.copy_(
            bt_workspace(
                bt_second.routed,
                bt_second.weights,
                bt_second.indices,
                bt_second.gamma,
            )
        )
        _call_ht(ht_workspace, ht_second, ht_second_out)

    for _ in range(4):
        sequence()
    torch.cuda.synchronize()
    dist.barrier(group=distributed_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sequence()

    for replay in range(3):
        base_seed = 5_000 + replay * 100
        _copy_random_inputs(bt_first, rank, base_seed)
        _copy_random_inputs(bt_second, rank, base_seed + 10)
        _copy_random_inputs(ht_first, rank, base_seed + 20)
        _copy_random_inputs(ht_second, rank, base_seed + 30)
        graph.replay()
        torch.cuda.synchronize()

        bt_first_ref = _oracle(bt_first, distributed_group)
        _assert_ulp(
            bt_first_out,
            bt_first_ref,
            2,
            f"graph BT first replay={replay}",
            distributed_group,
        )
        ht_first_ref = _oracle(ht_first, distributed_group)
        _assert_ulp(
            ht_first_out,
            ht_first_ref,
            3,
            f"graph HT first replay={replay}",
            distributed_group,
        )
        bt_second_ref = _oracle(bt_second, distributed_group)
        _assert_ulp(
            bt_second_out,
            bt_second_ref,
            2,
            f"graph BT second replay={replay}",
            distributed_group,
        )
        ht_second_ref = _oracle(ht_second, distributed_group)
        _assert_ulp(
            ht_second_out,
            ht_second_ref,
            3,
            f"graph HT second replay={replay}",
            distributed_group,
        )
        dist.barrier(group=distributed_group)


def _assert_collectively(condition: bool, label: str, group: dist.ProcessGroup) -> None:
    vote = torch.tensor(
        [int(condition)],
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    assert bool(vote.item()), label


def _assert_exact_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    group: dist.ProcessGroup,
) -> None:
    _assert_collectively(torch.equal(actual, expected), label, group)


def _symmetric_buffer_identity(buffer) -> tuple[int, int, int]:
    peers = buffer.peer_addresses
    return (
        buffer.tensor.data_ptr(),
        0 if peers is None else peers.data_ptr(),
        0 if buffer.multicast_address is None else buffer.multicast_address,
    )


def _capture_graph_case(
    workspace,
    inputs: _Inputs,
    label: str,
    group: dist.ProcessGroup,
) -> _CapturedGraphCase:
    output = torch.empty(
        (inputs.weights.shape[0], HIDDEN),
        dtype=torch.bfloat16,
        device=inputs.routed.device,
    )

    def invoke() -> torch.Tensor:
        alias = workspace(
            inputs.routed,
            inputs.weights,
            inputs.indices,
            inputs.gamma,
        )
        output.copy_(alias)
        return alias

    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        alias = invoke()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return _CapturedGraphCase(label, inputs, output, alias, graph)


def _assert_distinct_graph_cases(
    first: _CapturedGraphCase,
    second: _CapturedGraphCase,
    workspace,
    group: dist.ProcessGroup,
) -> None:
    persistent = workspace._backend._output
    conditions = (
        first.graph is not second.graph
        and first.inputs.weights.shape[0] != second.inputs.weights.shape[0]
        and first.inputs.routed.data_ptr() != second.inputs.routed.data_ptr()
        and first.inputs.weights.data_ptr() != second.inputs.weights.data_ptr()
        and first.inputs.indices.data_ptr() != second.inputs.indices.data_ptr()
        and first.inputs.gamma.data_ptr() != second.inputs.gamma.data_ptr()
        and not torch.equal(first.inputs.gamma, second.inputs.gamma)
        and first.output.data_ptr() != second.output.data_ptr()
        and first.output.data_ptr() != persistent.data_ptr()
        and second.output.data_ptr() != persistent.data_ptr()
        and first.alias.data_ptr() == persistent.data_ptr()
        and second.alias.data_ptr() == persistent.data_ptr()
        and first.alias.storage_offset() == 0
        and second.alias.storage_offset() == 0
        and first.alias.shape == first.output.shape
        and second.alias.shape == second.output.shape
    )
    _assert_collectively(
        conditions,
        f"{first.label}/{second.label}: graph inputs or output aliases overlap",
        group,
    )


def _replay_graph_case(
    workspace,
    current: _CapturedGraphCase,
    other: _CapturedGraphCase,
    ulp_limit: int,
    replay_label: str,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    other_before = other.output.clone()
    persistent = workspace._backend._output
    tail_before = None
    if current.output.shape[0] < other.output.shape[0]:
        tail_before = persistent[
            current.output.shape[0] : other.output.shape[0]
        ].clone()

    current.graph.replay()
    torch.cuda.synchronize()
    _assert_exact_tensor(
        other.output,
        other_before,
        f"{replay_label}: replay overwrote the other graph's saved output",
        group,
    )
    _assert_exact_tensor(
        persistent[: current.output.shape[0]],
        current.output,
        f"{replay_label}: returned view does not alias persistent output",
        group,
    )
    if tail_before is not None:
        _assert_exact_tensor(
            persistent[current.output.shape[0] : other.output.shape[0]],
            tail_before,
            f"{replay_label}: smaller graph overwrote persistent-output tail",
            group,
        )

    local, prenorm, expected = _oracle_components(current.inputs, group)
    _assert_ulp(current.output, expected, ulp_limit, replay_label, group)
    return local, prenorm


def _assert_output_changed(
    actual: torch.Tensor,
    before: torch.Tensor,
    label: str,
    group: dist.ProcessGroup,
) -> None:
    _assert_collectively(
        not torch.equal(actual, before),
        f"{label}: changed graph inputs did not change output",
        group,
    )


def _bt_state_identity(workspace) -> tuple[int, ...]:
    state = workspace._backend._protocol.state
    return (
        *_symmetric_buffer_identity(state.contribution_mailbox),
        *_symmetric_buffer_identity(state.prenorm_mailbox),
        state.stage_state.data_ptr(),
    )


def _assert_bt_protocol_state(
    workspace,
    initial_next_stage: int,
    launches: int,
    mailbox_identity: tuple[int, ...],
    label: str,
    group: dist.ProcessGroup,
) -> None:
    state = workspace._backend._protocol.state
    next_stage = (initial_next_stage + launches) % 3
    expected_stage = torch.tensor(
        [next_stage, (next_stage - 1) % 3],
        dtype=torch.int32,
        device=state.stage_state.device,
    )
    _assert_exact_tensor(
        state.stage_state,
        expected_stage,
        f"{label}: BT stage counter did not advance exactly once",
        group,
    )
    _assert_collectively(
        _bt_state_identity(workspace) == mailbox_identity,
        f"{label}: BT mailbox identity changed",
        group,
    )
    for mailbox_name, mailbox in (
        ("contribution", state.contribution_mailbox),
        ("prenorm", state.prenorm_mailbox),
    ):
        is_reset = bool((mailbox.tensor.view(torch.int16) == -32768).all().item())
        _assert_collectively(
            is_reset,
            f"{label}: BT {mailbox_name} mailbox was not reset to sentinel",
            group,
        )


@dataclass
class _HTCounterSnapshot:
    routed_ready: torch.Tensor
    routed_processed: torch.Tensor
    all_reduce_ready: torch.Tensor
    all_reduce_processed: torch.Tensor


def _ht_counter_snapshot(workspace) -> _HTCounterSnapshot:
    state = workspace._backend._protocol.state
    return _HTCounterSnapshot(
        state.routed_ready_counters.tensor.to(torch.int64).clone(),
        state.routed_processed_counters.to(torch.int64).clone(),
        state.all_reduce_ready_counters.tensor.to(torch.int64).clone(),
        state.all_reduce_processed_counters.to(torch.int64).clone(),
    )


def _ht_state_identity(workspace) -> tuple[int, ...]:
    state = workspace._backend._protocol.state
    return (
        *_symmetric_buffer_identity(state.local_contributions),
        *_symmetric_buffer_identity(state.prenorm_mailbox),
        *_symmetric_buffer_identity(state.routed_ready_counters),
        state.routed_processed_counters.data_ptr(),
        *_symmetric_buffer_identity(state.all_reduce_ready_counters),
        state.all_reduce_processed_counters.data_ptr(),
    )


def _assert_ht_protocol_state(
    workspace,
    baseline: _HTCounterSnapshot,
    replayed_sizes: list[int],
    local_reference: torch.Tensor,
    prenorm_reference: torch.Tensor,
    mailbox_identity: tuple[int, ...],
    label: str,
    group: dist.ProcessGroup,
) -> None:
    state = workspace._backend._protocol.state
    rank = dist.get_rank(group)
    slots = baseline.routed_ready.numel()
    owned_tokens = (
        torch.arange(slots, dtype=torch.int64, device=baseline.routed_ready.device)
        * TP_SIZE
        + rank
    )
    increments = torch.zeros_like(owned_tokens)
    for m in replayed_sizes:
        increments.add_((owned_tokens < m).to(torch.int64) * TP_SIZE)
    expected_ready = baseline.routed_ready + increments
    expected_processed = baseline.routed_processed + increments[:, None]
    _assert_exact_tensor(
        state.routed_ready_counters.tensor.to(torch.int64),
        expected_ready,
        f"{label}: HT routed-ready counters advanced incorrectly",
        group,
    )
    _assert_exact_tensor(
        state.routed_processed_counters.to(torch.int64),
        expected_processed,
        f"{label}: HT routed-processed counters advanced incorrectly",
        group,
    )
    _assert_exact_tensor(
        state.all_reduce_ready_counters.tensor.to(torch.int64),
        baseline.all_reduce_ready,
        f"{label}: finalize graph touched HT all-reduce-ready counters",
        group,
    )
    _assert_exact_tensor(
        state.all_reduce_processed_counters.to(torch.int64),
        baseline.all_reduce_processed,
        f"{label}: finalize graph touched HT all-reduce-processed counters",
        group,
    )
    _assert_collectively(
        _ht_state_identity(workspace) == mailbox_identity,
        f"{label}: HT protocol storage identity changed",
        group,
    )
    m = local_reference.shape[0]
    _assert_ulp(
        state.local_contributions.tensor[:m],
        local_reference,
        1,
        f"{label}: HT local-contribution mailbox",
        group,
    )
    _assert_ulp(
        state.prenorm_mailbox.tensor[:m],
        prenorm_reference,
        2,
        f"{label}: HT prenorm mailbox",
        group,
    )


@torch.inference_mode()
def test_bt_independent_graphs_multi_bucket_lifetime(
    distributed_group, bt_workspace
) -> None:
    """Interleave two BT graph buckets across input and gamma changes."""

    rank = dist.get_rank(distributed_group)
    device = torch.device("cuda", torch.cuda.current_device())
    first = _capture_graph_case(
        bt_workspace,
        _random_inputs(33, rank, device, seed=6_000, gamma_ones=True),
        "BT-A-M33",
        distributed_group,
    )
    second = _capture_graph_case(
        bt_workspace,
        _random_inputs(BT_MAX_TOKENS, rank, device, seed=6_100),
        f"BT-B-M{BT_MAX_TOKENS}",
        distributed_group,
    )
    _assert_distinct_graph_cases(first, second, bt_workspace, distributed_group)
    for case in (first, second):
        _assert_ulp(
            case.output,
            _oracle(case.inputs, distributed_group),
            2,
            f"{case.label} capture",
            distributed_group,
        )

    state = bt_workspace._backend._protocol.state
    initial_next_stage = int(state.stage_state[0].item())
    _assert_collectively(
        int(state.stage_state[1].item()) == (initial_next_stage - 1) % 3,
        "BT capture left inconsistent stage counters",
        distributed_group,
    )
    mailbox_identity = _bt_state_identity(bt_workspace)
    replayed: list[int] = []
    for phase, changed in (("original", False), ("changed", True)):
        if changed:
            first_before = first.output.clone()
            second_before = second.output.clone()
            first_gamma = first.inputs.gamma.clone()
            second_gamma = second.inputs.gamma.clone()
            _copy_random_inputs(first.inputs, rank, seed=6_200)
            _copy_random_inputs(second.inputs, rank, seed=6_300)
            _assert_collectively(
                not torch.equal(first.inputs.gamma, first_gamma)
                and not torch.equal(second.inputs.gamma, second_gamma),
                "BT changed-input replay did not replace both gammas",
                distributed_group,
            )

        for step, (current, other) in enumerate(
            ((first, second), (second, first), (first, second))
        ):
            label = f"BT {phase} A/B/A step={step} {current.label}"
            _replay_graph_case(
                bt_workspace,
                current,
                other,
                2,
                label,
                distributed_group,
            )
            replayed.append(current.output.shape[0])
            _assert_bt_protocol_state(
                bt_workspace,
                initial_next_stage,
                len(replayed),
                mailbox_identity,
                label,
                distributed_group,
            )

        if changed:
            _assert_output_changed(
                first.output,
                first_before,
                first.label,
                distributed_group,
            )
            _assert_output_changed(
                second.output,
                second_before,
                second.label,
                distributed_group,
            )
    dist.barrier(group=distributed_group)


@torch.inference_mode()
def test_native_ht_independent_graphs_multi_bucket_lifetime(
    distributed_group, ht_workspace
) -> None:
    """Interleave two native-HT graph buckets without resetting counters."""

    rank = dist.get_rank(distributed_group)
    device = torch.device("cuda", torch.cuda.current_device())
    first = _capture_graph_case(
        ht_workspace,
        _random_inputs(HT_GRAPH_TOKENS, rank, device, seed=7_000, gamma_ones=True),
        f"HT-A-M{HT_GRAPH_TOKENS}",
        distributed_group,
    )
    second = _capture_graph_case(
        ht_workspace,
        _random_inputs(HT_MAX_TOKENS, rank, device, seed=7_100),
        f"HT-B-M{HT_MAX_TOKENS}",
        distributed_group,
    )
    _assert_distinct_graph_cases(first, second, ht_workspace, distributed_group)
    for case in (first, second):
        _assert_ulp(
            case.output,
            _oracle(case.inputs, distributed_group),
            3,
            f"{case.label} capture",
            distributed_group,
        )

    counters = _ht_counter_snapshot(ht_workspace)
    mailbox_identity = _ht_state_identity(ht_workspace)
    replayed: list[int] = []
    for phase, changed in (("original", False), ("changed", True)):
        if changed:
            first_before = first.output.clone()
            second_before = second.output.clone()
            first_gamma = first.inputs.gamma.clone()
            second_gamma = second.inputs.gamma.clone()
            _copy_random_inputs(first.inputs, rank, seed=7_200)
            _copy_random_inputs(second.inputs, rank, seed=7_300)
            _assert_collectively(
                not torch.equal(first.inputs.gamma, first_gamma)
                and not torch.equal(second.inputs.gamma, second_gamma),
                "HT changed-input replay did not replace both gammas",
                distributed_group,
            )

        for step, (current, other) in enumerate(
            ((first, second), (second, first), (first, second))
        ):
            label = f"HT {phase} A/B/A step={step} {current.label}"
            local, prenorm = _replay_graph_case(
                ht_workspace,
                current,
                other,
                3,
                label,
                distributed_group,
            )
            replayed.append(current.output.shape[0])
            _assert_ht_protocol_state(
                ht_workspace,
                counters,
                replayed,
                local,
                prenorm,
                mailbox_identity,
                label,
                distributed_group,
            )

        if changed:
            _assert_output_changed(
                first.output,
                first_before,
                first.label,
                distributed_group,
            )
            _assert_output_changed(
                second.output,
                second_before,
                second.label,
                distributed_group,
            )
    dist.barrier(group=distributed_group)
