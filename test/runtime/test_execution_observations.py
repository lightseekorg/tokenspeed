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

"""CPU observation contracts; execution itself is covered by GPU integration."""

from copy import deepcopy
from pathlib import Path

import msgspec
import pytest
import torch

from tokenspeed.runtime.engine.io_struct import BatchTokenIDOutSlim
from tokenspeed.runtime.entrypoints.execution_trace import (
    CACHE,
    EXECUTION,
    ExecutionTrace,
)
from tokenspeed.runtime.execution.observations import TargetExecution, validate_scope


class PreObservationBatch(
    msgspec.Struct, array_like=True, tag="BatchTokenIDOutSlim", kw_only=True
):
    rids: list[str]
    output_ids: list[list[int]]
    finished_reasons: list[str]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]
    output_token_logprobs_val: list[list[float]]
    output_token_logprobs_idx: list[list[int]]
    engine_index: int = 0
    num_running: int = 0
    num_waiting: int = 0
    kv_active_pages: int = 0
    kv_total_pages: int = 0
    finish_reason_details: list = []
    spec_verify_ct: list[int] = []
    accept_draft_tokens: list[float | None] = []
    input_token_logprobs_val: list = []
    input_token_logprobs_idx: list = []
    input_top_logprobs_val: list = []
    input_top_logprobs_idx: list = []
    output_top_logprobs_val: list = []
    output_top_logprobs_idx: list = []
    kv_used_pages: int = 0
    output_token_cumulative_probs_val: list = []
    reasoning_tokens: list[int] = []
    num_resident_tokens: int | None = None
    num_waiting_uncached_tokens: int | None = None


def test_old_producer_absence_and_old_decoder_ignoring_new_tail():
    fixtures = (
        Path(__file__).resolve().parents[2]
        / "tokenspeed-engine/crates/ts-wire/tests/fixtures"
    )
    old_bytes = bytes.fromhex(
        (fixtures / "output_pre_execution.hex").read_text().strip()
    )
    new_bytes = bytes.fromhex((fixtures / "output_execution.hex").read_text().strip())
    old = msgspec.msgpack.decode(old_bytes, type=BatchTokenIDOutSlim)
    assert old.spec_verify_ct == [2, 0]
    assert old.execution_observations == []
    legacy = msgspec.msgpack.decode(new_bytes, type=PreObservationBatch)
    assert legacy.output_ids == [[7], [8]]
    assert legacy.spec_verify_ct == [2, 0]
    assert legacy.output_token_cumulative_probs_val == [[0.75], []]
    current = msgspec.msgpack.decode(new_bytes, type=BatchTokenIDOutSlim)
    assert current.execution_observations[0]["admission_id"] == 2**63 + 13
    assert current.execution_observations[1] is None


@pytest.mark.parametrize("width", [1, 8])
def test_width_is_the_consumed_candidate_layout_not_the_kept_prefix(width):
    candidates = torch.zeros((3, width), dtype=torch.int32, device="cpu")
    route = TargetExecution("eager", 0, 3, 3 * width, 0).verified(
        candidates, packed=False
    )
    assert route.verify_width == width
    replay = route.replayed(request_capacity=4, query_capacity=4 * width)
    assert replay.payload((11, 12, 13))[2:5] == (width, 3, 4)
    with pytest.raises(ValueError, match="packed"):
        route.verified(candidates, packed=True)


@pytest.mark.parametrize(
    "choice",
    ["pp_size", "query_shard_size", "algorithm", "tree", "narrowed", "synthetic"],
)
def test_unsupported_observation_routes_are_refused(choice):
    options = dict(
        device="cuda",
        pp_size=1,
        query_shard_size=1,
        algorithm="DFLASH",
        tree=False,
        narrowed=False,
        synthetic=False,
    )
    validate_scope(**options)
    options[choice] = {
        "pp_size": 2,
        "query_shard_size": 2,
        "algorithm": "DSPARK",
        "tree": True,
        "narrowed": True,
        "synthetic": True,
    }[choice]
    with pytest.raises(ValueError, match="execution observations v1"):
        validate_scope(**options)


def trace_fixture():
    records = []
    sequences = {CACHE: 0, EXECUTION: 0}

    def add(protocol, event, **fields):
        base = dict(
            protocol=protocol,
            instance_id="test",
            role="fused",
            dp_rank=0,
            physical_rank=0,
            scope="local_physical" if protocol == EXECUTION else "physical_rank",
            publication_order=len(records),
            record_sequence=sequences[protocol],
            log_dropped_lines=0,
            event=event,
        )
        if event not in ("trace_started", "trace_finished"):
            base.update(request_id="r", admission_id=7)
            if protocol == EXECUTION:
                base["origin"] = "serving"
        base.update(fields)
        records.append(base)
        sequences[protocol] += 1

    add(CACHE, "trace_started")
    add(EXECUTION, "trace_started")
    add(EXECUTION, "request_started")
    for event, slot in (
        ("slot_assigned", 3),
        ("slot_released", 3),
        ("slot_assigned", 5),
    ):
        add(
            CACHE,
            event,
            kind="slot",
            scope="scheduler_replica",
            slot=slot,
            slot_generation=1,
            slot_previous_admission_id=0,
        )
    restore = dict(
        kind="snapshot_restore",
        op_generation=1,
        op_id=3,
        slot=5,
        slot_generation=1,
        restore_sequence=1,
        previous_slot=3,
        previous_slot_generation=1,
        slot_previous_admission_id=0,
        committed_tokens=8,
        computed_tokens=8,
        prompt_tokens=8,
    )
    add(CACHE, "restore_scheduled", scope="scheduler_replica", **restore)
    add(
        CACHE,
        "restore_completed",
        scope="local_physical",
        **restore,
        l2_rows=1,
        l2_bytes=256,
        pool_rows=1,
        pool_bytes=64,
        slot_state_image_bytes=32,
    )
    add(
        EXECUTION,
        "target_round_dispatched",
        round=1,
        phase="decode",
        slot=5,
        input_length=8,
    )
    add(
        EXECUTION,
        "target_round_completed",
        round=1,
        phase="decode",
        slot=5,
        input_length=8,
        path="decode_graph",
        verify_width=8,
        kept_prefix=3,
        committed_tokens=2,
        graph_segments=1,
        live_requests=1,
        request_capacity=4,
        query_capacity=32,
        row_index=0,
        num_extends=0,
        discarded=False,
        numerical_error=False,
        accepted_before_stop=5,
        force_single=False,
    )
    add(
        EXECUTION,
        "request_closed",
        assigned_rounds=1,
        completed_rounds=1,
        missing_rounds=0,
        terminal="length",
        evidence_valid=True,
    )
    add(
        CACHE,
        "slot_released",
        kind="slot",
        scope="scheduler_replica",
        slot=5,
        slot_generation=1,
        slot_previous_admission_id=0,
    )
    add(EXECUTION, "trace_finished", lost_observations=0, evidence_valid=True)
    add(CACHE, "trace_finished", lost_observations=0, evidence_valid=True)
    snapshot = dict(
        version=1,
        instance_id="test",
        role="fused",
        dp_rank=0,
        physical_rank=0,
        request_id="r",
        admission_id=7,
        origin="serving",
        assigned_rounds=1,
        completed_rounds=1,
        through_round=1,
        missing_rounds=0,
        committed_tokens=2,
        evidence_valid=True,
        histograms_omitted=False,
        bins=[
            dict(
                phase="decode",
                path="decode_graph",
                verify_width=8,
                kept_prefix=3,
                count=1,
            )
        ],
        pre_stop_bins=[
            dict(
                path="decode_graph",
                verify_width=8,
                accepted_before_stop=5,
                force_single=False,
                count=1,
            )
        ],
    )
    return records, snapshot


def event(records, name):
    return next(record for record in records if record["event"] == name)


def resequence(records):
    sequences = {}
    orders = {}
    for record in records:
        rank = tuple(
            record[k] for k in ("instance_id", "role", "dp_rank", "physical_rank")
        )
        key = (*rank, record["protocol"])
        record["record_sequence"] = sequences.get(key, 0)
        record["publication_order"] = orders.get(rank, 0)
        sequences[key] = record["record_sequence"] + 1
        orders[rank] = record["publication_order"] + 1


def test_closed_trace_reconciles_snapshot_and_native_restore_order():
    records, snapshot = trace_fixture()
    trace = ExecutionTrace(records)
    trace.require(
        snapshot,
        physical_ranks=[0],
        cells=[("decode", "decode_graph", 8)],
        restored=True,
    )
    for ranks, cells in [([0, 1], []), ([0], [("decode", "decode_graph", 1)])]:
        with pytest.raises(ValueError):
            trace.require(snapshot, physical_ranks=ranks, cells=cells, restored=False)
    other = dict(snapshot, admission_id=8)
    with pytest.raises(ValueError, match="no closed trace"):
        trace.check_snapshot(other)


@pytest.mark.parametrize(
    "live_requests,num_extends", [(1, 0), (4, 0), (5, 0), (4, 1), (5, 1)]
)
def test_eager_overflow_requires_actual_pure_decode_batch(live_requests, num_extends):
    records, snapshot = trace_fixture()
    row = event(records, "target_round_completed")
    row.update(
        path="eager",
        graph_segments=0,
        live_requests=live_requests,
        request_capacity=live_requests,
        query_capacity=live_requests * 8,
        num_extends=num_extends,
        row_index=num_extends,
    )
    for field in ("bins", "pre_stop_bins"):
        snapshot[field][0]["path"] = "eager"
    trace = ExecutionTrace(records)
    # Every fixture has valid eager execution; only the above-ladder pure batch
    # establishes this distinct fallback condition.
    trace.require(
        snapshot, physical_ranks=[0], cells=[("decode", "eager", 8)], restored=False
    )
    if live_requests > 4 and num_extends == 0:
        trace.require_eager_overflow(
            snapshot, physical_ranks=[0], verify_width=8, capture_max_requests=4
        )
    else:
        with pytest.raises(ValueError, match="above the capture maximum"):
            trace.require_eager_overflow(
                snapshot, physical_ranks=[0], verify_width=8, capture_max_requests=4
            )


@pytest.mark.parametrize("missing_rank", [0, 1])
def test_eager_overflow_requires_each_measured_rank_not_an_unrelated_admission(
    missing_rank,
):
    records, snapshot = trace_fixture()
    row = event(records, "target_round_completed")
    row.update(path="eager", graph_segments=0, request_capacity=1, query_capacity=8)
    for field in ("bins", "pre_stop_bins"):
        snapshot[field][0]["path"] = "eager"
    peer = deepcopy(records)
    for record in peer:
        record["physical_rank"] = 1
    event(peer, "target_round_completed").update(
        live_requests=5, request_capacity=5, query_capacity=40
    )
    if missing_rank == 1:
        row.update(live_requests=5, request_capacity=5, query_capacity=40)
        event(peer, "target_round_completed").update(
            live_requests=1, request_capacity=1, query_capacity=8
        )
    unrelated = deepcopy(records)
    for record in unrelated:
        if "request_id" in record:
            record.update(request_id="pressure", admission_id=8)
    event(unrelated, "target_round_completed").update(
        live_requests=5, request_capacity=5, query_capacity=40
    )
    # Use a separate instance so each complete trace keeps its own endpoints.
    for record in unrelated:
        record["instance_id"] = "pressure-instance"
    trace = ExecutionTrace(records + peer + unrelated)
    with pytest.raises(
        ValueError, match=f"above the capture maximum on rank {missing_rank}"
    ):
        trace.require_eager_overflow(
            snapshot, physical_ranks=[0, 1], verify_width=8, capture_max_requests=4
        )


@pytest.mark.parametrize(
    "fault",
    [
        "loss",
        "missing_end",
        "duplicate",
        "missing_round",
        "wrong_admission",
        "graph_without_replay",
    ],
)
def test_incomplete_or_false_execution_evidence_fails(fault):
    records, _ = trace_fixture()
    records = deepcopy(records)
    if fault == "loss":
        records[-1]["lost_observations"] = 1
    elif fault == "missing_end":
        records.pop()
    elif fault == "duplicate":
        records.insert(
            records.index(event(records, "target_round_completed")) + 1,
            deepcopy(event(records, "target_round_completed")),
        )
    elif fault == "missing_round":
        event(records, "request_closed")["completed_rounds"] = 0
    elif fault == "wrong_admission":
        event(records, "target_round_completed")["admission_id"] = 8
    else:
        event(records, "target_round_completed")["graph_segments"] = 0
    with pytest.raises(ValueError):
        ExecutionTrace(records)


def test_a_delayed_pre_restore_completion_is_not_post_restore_execution():
    records, snapshot = trace_fixture()
    restore = records.index(event(records, "restore_completed"))
    dispatch = records.index(event(records, "target_round_dispatched"))
    records[restore], records[dispatch] = records[dispatch], records[restore]
    resequence(records)
    trace = ExecutionTrace(records)
    with pytest.raises(ValueError, match="after restore"):
        trace.require(snapshot, physical_ranks=[0], cells=[], restored=True)


def test_verification_snapshot_preserves_pre_stop_values_and_applied_mask():
    from tokenspeed.runtime.execution.observations import VerificationObservations

    observations = VerificationObservations(max_bs=5, device="cpu")
    lengths = torch.tensor([1, 7, 1, 8], dtype=torch.int32)
    mask = torch.tensor([False, True, False])
    saved = observations.record(lengths, num_extends=1, force_single=mask)
    lengths.clamp_(max=2)
    mask.zero_()
    assert saved.tolist() == [[-1, 0], [7, 0], [1, 1], [8, 0]]
    # The graph refreshes every observed column, including a formerly forced row.
    refreshed = observations.record(lengths, num_extends=1, force_single=None)
    assert refreshed.tolist() == [[-1, 0], [2, 0], [1, 0], [2, 0]]


@pytest.mark.parametrize(
    "fault",
    ["missing_pre_stop", "pre_stop_below_kept", "forced_many", "wrong_histogram"],
)
def test_pre_stop_evidence_cannot_be_replaced_by_kept_prefix(fault):
    records, snapshot = trace_fixture()
    if fault == "missing_pre_stop":
        del snapshot["pre_stop_bins"]
    elif fault == "pre_stop_below_kept":
        event(records, "target_round_completed")["accepted_before_stop"] = 2
    elif fault == "forced_many":
        event(records, "target_round_completed")["force_single"] = True
    else:
        snapshot["pre_stop_bins"][0]["accepted_before_stop"] = 3
    with pytest.raises(ValueError):
        ExecutionTrace(records).check_snapshot(snapshot)


@pytest.mark.parametrize(
    "field", ["op_id", "op_generation", "slot_generation", "restore_sequence"]
)
def test_restore_join_requires_the_exact_native_operation_and_generation(field):
    records, _ = trace_fixture()
    completed = next(
        record for record in records if record["event"] == "restore_completed"
    )
    completed[field] += 1
    with pytest.raises(ValueError, match="matching native operation"):
        ExecutionTrace(records)


@pytest.mark.parametrize(
    "field",
    [
        "previous_slot",
        "previous_slot_generation",
        "slot_previous_admission_id",
        "committed_tokens",
        "computed_tokens",
        "prompt_tokens",
    ],
)
def test_copied_restore_rejects_conflicting_frozen_recovery_metadata(field):
    records, snapshot = trace_fixture()
    event(records, "restore_completed")[field] = 99
    with pytest.raises(ValueError, match="contradicts frozen scheduled metadata"):
        ExecutionTrace(records).require_copied_restore(snapshot, physical_ranks=[0])


@pytest.mark.parametrize(
    "field",
    [
        "previous_slot",
        "previous_slot_generation",
        "slot_previous_admission_id",
        "committed_tokens",
        "computed_tokens",
        "prompt_tokens",
    ],
)
@pytest.mark.parametrize("stage", ["restore_scheduled", "restore_completed"])
@pytest.mark.parametrize("fault", ["missing", None, True, "8", 8.0, -1, "overflow"])
def test_restore_frozen_fields_require_native_types_on_both_records(
    field, stage, fault
):
    records, snapshot = trace_fixture()
    record = event(records, stage)
    if fault == "missing":
        del record[field]
    elif fault == "overflow":
        record[field] = 2 ** (
            64
            if field in ("previous_slot_generation", "slot_previous_admission_id")
            else 31
        )
    else:
        record[field] = fault
    with pytest.raises(ValueError, match="invalid or missing frozen restore metadata"):
        ExecutionTrace(records).require_copied_restore(snapshot, physical_ranks=[0])


def test_restore_frontiers_are_compared_without_inventing_prompt_end_inequalities():
    records, snapshot = trace_fixture()
    for stage in ("restore_scheduled", "restore_completed"):
        event(records, stage)["computed_tokens"] = 4
    ExecutionTrace(records).require_copied_restore(snapshot, physical_ranks=[0])


def test_orphan_restore_and_reused_slot_cannot_establish_recovery():
    records, snapshot = trace_fixture()
    records.remove(event(records, "restore_scheduled"))
    resequence(records)
    with pytest.raises(ValueError, match="matching native operation"):
        ExecutionTrace(records)
    records, snapshot = trace_fixture()
    dispatch = records.index(event(records, "target_round_dispatched"))
    owner = deepcopy(
        next(
            record
            for record in records
            if record["event"] == "slot_assigned" and record["slot"] == 5
        )
    )
    released = dict(owner, event="slot_released")
    replacement = dict(owner, slot_generation=2)
    records[dispatch:dispatch] = [released, replacement]
    resequence(records)
    with pytest.raises(ValueError, match="after restore"):
        ExecutionTrace(records).require(
            snapshot, physical_ranks=[0], cells=[], restored=True
        )


def test_zero_copy_ack_does_not_claim_physical_copy():
    records, snapshot = trace_fixture()
    completed = event(records, "restore_completed")
    for name in ("l2_rows", "l2_bytes", "pool_rows", "pool_bytes"):
        completed[name] = 0
    # An image span is not a claim of copied bytes.
    completed["slot_state_image_bytes"] = 64
    trace = ExecutionTrace(records)
    trace.require(snapshot, physical_ranks=[0], cells=[], restored=True)
    with pytest.raises(ValueError, match="positive local cache copy"):
        trace.require_copied_restore(snapshot, physical_ranks=[0])


def test_required_peer_startup_execution_cannot_qualify_serving_snapshot():
    records, snapshot = trace_fixture()
    peer = deepcopy(records)
    for record in peer:
        record["physical_rank"] = 1
        if "origin" in record:
            record["origin"] = "startup"
    trace = ExecutionTrace(records + peer)
    with pytest.raises(ValueError, match="peer rank supplied startup"):
        trace.require(
            snapshot,
            physical_ranks=[0, 1],
            cells=[("decode", "decode_graph", 8)],
            restored=False,
        )


def test_w8_cannot_be_claimed_for_one_frozen_input_row():
    records, snapshot = trace_fixture()
    event(records, "target_round_dispatched")["input_length"] = 1
    event(records, "target_round_completed")["input_length"] = 1
    with pytest.raises(ValueError, match="width differs from frozen input"):
        ExecutionTrace(records).require(
            snapshot,
            physical_ranks=[0],
            cells=[("decode", "decode_graph", 8)],
            restored=False,
        )


@pytest.mark.parametrize(
    "field",
    ["row_index", "num_extends", "live_requests", "request_capacity", "query_capacity"],
)
def test_real_native_geometry_cannot_be_omitted(field):
    records, _ = trace_fixture()
    del event(records, "target_round_completed")[field]
    with pytest.raises(ValueError, match="missing native row geometry"):
        ExecutionTrace(records)


@pytest.mark.parametrize(
    "field,value",
    [("row_index", 1), ("num_extends", 2), ("num_extends", 1), ("query_capacity", 8)],
)
def test_padding_and_partition_geometry_cannot_be_claimed_as_live_execution(
    field, value
):
    records, _ = trace_fixture()
    event(records, "target_round_completed")[field] = value
    with pytest.raises(ValueError):
        ExecutionTrace(records)


def test_serialized_snapshot_budget_rejects_4096_bins_over_256_kib():
    from tokenspeed.runtime.execution.observations import validate_snapshot

    _, snapshot = trace_fixture()
    snapshot.update(
        assigned_rounds=2048,
        completed_rounds=2048,
        through_round=2048,
        committed_tokens=2048 * 2049 // 2,
    )
    snapshot["bins"] = [
        dict(
            phase="decode",
            path="prefill_breakable",
            verify_width=2048,
            kept_prefix=i,
            count=1,
        )
        for i in range(1, 2049)
    ]
    snapshot["pre_stop_bins"] = [
        dict(
            path="prefill_breakable",
            verify_width=2048,
            accepted_before_stop=i,
            force_single=False,
            count=1,
        )
        for i in range(1, 2049)
    ]
    assert len(msgspec.msgpack.encode(snapshot)) > 256 * 1024
    with pytest.raises(ValueError, match="256 KiB"):
        validate_snapshot(snapshot, None)
    # Locate the adjacent below/above cases using the actual MessagePack budget.
    low, high = 1, 2048
    while low + 1 < high:
        middle = (low + high) // 2
        candidate = dict(
            snapshot,
            bins=snapshot["bins"][:middle],
            pre_stop_bins=snapshot["pre_stop_bins"][:middle],
            assigned_rounds=middle,
            completed_rounds=middle,
            through_round=middle,
            committed_tokens=middle * (middle + 1) // 2,
        )
        if len(msgspec.msgpack.encode(candidate)) <= 256 * 1024:
            low = middle
        else:
            high = middle
    for count, accepted in ((low, True), (high, False)):
        candidate = dict(
            snapshot,
            bins=snapshot["bins"][:count],
            pre_stop_bins=snapshot["pre_stop_bins"][:count],
            assigned_rounds=count,
            completed_rounds=count,
            through_round=count,
            committed_tokens=count * (count + 1) // 2,
        )
        if accepted:
            validate_snapshot(candidate, None)
        else:
            with pytest.raises(ValueError, match="256 KiB"):
                validate_snapshot(candidate, None)
