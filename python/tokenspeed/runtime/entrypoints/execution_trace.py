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

"""Validate closed native execution/cache traces and cumulative wire snapshots.

Run with ``python -m tokenspeed.runtime.entrypoints.execution_trace LOG ...``.
This module reads CPU JSON only and imports neither Torch nor the native host.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tokenspeed.runtime.execution.observations import validate_snapshot

EXECUTION = "tokenspeed-execution-observations-v1"
CACHE = "tokenspeed-cache-events-v1"
RESTORE_FROZEN_FIELDS = (
    "previous_slot",
    "previous_slot_generation",
    "slot_previous_admission_id",
    "committed_tokens",
    "computed_tokens",
    "prompt_tokens",
)


def validate_restore_metadata(record):
    # Match the native integer domains, including the nonnegative i32 recovery
    # fields. Frontiers are frozen values, not an ordering inequality to infer.
    for name, bits in (
        ("admission_id", 64),
        ("op_generation", 64),
        ("op_id", 32),
        ("slot", 31),
        ("slot_generation", 64),
        ("restore_sequence", 64),
        ("previous_slot", 31),
        ("previous_slot_generation", 64),
        ("slot_previous_admission_id", 64),
        ("committed_tokens", 31),
        ("computed_tokens", 31),
        ("prompt_tokens", 31),
    ):
        if (
            name not in record
            or type(record[name]) is not int
            or not 0 <= record[name] < 2**bits
        ):
            raise ValueError(f"invalid or missing frozen restore metadata: {name}")


def rank_key(record):
    return tuple(record[k] for k in ("instance_id", "role", "dp_rank", "physical_rank"))


def request_key(record):
    return (*rank_key(record), record["admission_id"], record["request_id"])


class ExecutionTrace:
    """Closed evidence, with no credit inferred from configuration or time."""

    def __init__(self, records):
        self.rounds = {}
        self.dispatched = {}
        self.closed = {}
        self.started = {}
        self.restores = []
        self.restore_scheduled = {}
        self.restore_completed = set()
        self.slot_events = []
        streams = {}
        orders = {}
        for record in records:
            protocol = record["protocol"]
            if protocol not in (EXECUTION, CACHE):
                raise ValueError("unsupported observation protocol")
            if any(
                name not in record
                or type(record[name]) is not int
                or not 0 <= record[name] < 2**64
                for name in (
                    "dp_rank",
                    "physical_rank",
                    "record_sequence",
                    "publication_order",
                    "log_dropped_lines",
                )
            ):
                raise ValueError("invalid native observation header")
            if record.get("role") not in ("fused", "prefill", "decode"):
                raise ValueError("invalid observation role")
            scope = record.get("scope")
            if scope not in (
                ("local_physical",)
                if protocol == EXECUTION
                else ("local_physical", "scheduler_replica", "physical_rank")
            ):
                raise ValueError("invalid observation scope")
            rank = rank_key(record)
            stream = streams.setdefault((rank, protocol), [])
            if record["record_sequence"] != len(stream):
                raise ValueError("noncontiguous observation sequence")
            if record["publication_order"] <= orders.get(rank, -1):
                raise ValueError("nonmonotonic native publication order")
            orders[rank] = record["publication_order"]
            if record["log_dropped_lines"] or record["event"] == "observation_loss":
                raise ValueError("observation trace lost records")
            if stream and stream[-1]["event"] == "trace_finished":
                raise ValueError("record after trace closure")
            if record["event"] == "trace_started" and stream:
                raise ValueError("duplicate trace start")
            stream.append(record)
            event = record["event"]
            if protocol == CACHE:
                if event in ("slot_assigned", "slot_released"):
                    self.slot_events.append(record)
                elif event in ("restore_scheduled", "restore_completed"):
                    if record["kind"] != "snapshot_restore":
                        raise ValueError("invalid native restore kind")
                    validate_restore_metadata(record)
                    join = (
                        *request_key(record),
                        *(
                            record[k]
                            for k in (
                                "kind",
                                "op_generation",
                                "op_id",
                                "slot",
                                "slot_generation",
                                "restore_sequence",
                            )
                        ),
                    )
                    if event == "restore_scheduled":
                        if join in self.restore_scheduled:
                            raise ValueError("duplicate native restore operation")
                        self.restore_scheduled[join] = record
                    else:
                        if join not in self.restore_scheduled:
                            raise ValueError(
                                "restore completion has no matching native operation"
                            )
                        if join in self.restore_completed:
                            raise ValueError("duplicate native restore completion")
                        scheduled = self.restore_scheduled[join]
                        if any(
                            record[name] != scheduled[name]
                            for name in RESTORE_FROZEN_FIELDS
                        ):
                            raise ValueError(
                                "restore completion contradicts frozen scheduled metadata"
                            )
                        self.restore_completed.add(join)
                        for name in (
                            "l2_rows",
                            "l2_bytes",
                            "pool_rows",
                            "pool_bytes",
                            "slot_state_image_bytes",
                        ):
                            if (
                                name not in record
                                or type(record[name]) is not int
                                or not 0 <= record[name] < 2**64
                            ):
                                raise ValueError(
                                    "invalid or missing local restore copy counts"
                                )
                        if any(
                            request_key(owner) == request_key(record)
                            and owner["event"] == "slot_assigned"
                            and owner["slot"] == record["slot"]
                            and owner["slot_generation"] == record["slot_generation"]
                            for owner in self.slot_events
                        ):
                            self.restores.append(record)
                        else:
                            raise ValueError(
                                "restore completion has no matching native slot ownership"
                            )
                continue
            if event in ("trace_started", "trace_finished"):
                continue
            key = request_key(record)
            if event == "request_started":
                if key in self.started:
                    raise ValueError("duplicate native admission")
                self.started[key] = record
                self.rounds[key] = {}
                self.dispatched[key] = {}
                continue
            if (
                type(record["admission_id"]) is not int
                or not 0 < record["admission_id"] < 2**64
            ):
                raise ValueError("invalid native admission identity")
            if key not in self.started or key in self.closed:
                raise ValueError("round outside its native admission lifetime")
            if record["origin"] != self.started[key]["origin"]:
                raise ValueError("admission origin changed")
            if event == "target_round_dispatched":
                if (
                    type(record["round"]) is not int
                    or type(record["slot"]) is not int
                    or record["slot"] <= 0
                    or record["phase"] not in ("prefill", "decode")
                    or type(record["input_length"]) is not int
                    or record["input_length"] < 0
                ):
                    raise ValueError("invalid frozen dispatch geometry")
                if record["round"] != len(self.dispatched[key]) + 1:
                    raise ValueError("noncontiguous admission dispatch ordinals")
                self.dispatched[key][record["round"]] = record
            elif event == "target_round_completed":
                ordinal = record["round"]
                dispatch = self.dispatched[key].get(ordinal)
                if dispatch is None or ordinal in self.rounds[key]:
                    raise ValueError("missing dispatch or duplicate completion")
                if any(
                    record[k] != dispatch[k] for k in ("slot", "phase", "input_length")
                ):
                    raise ValueError("completed row differs from frozen dispatch")
                geometry = (
                    "row_index",
                    "num_extends",
                    "live_requests",
                    "request_capacity",
                    "query_capacity",
                    "input_length",
                    "verify_width",
                    "kept_prefix",
                    "committed_tokens",
                    "graph_segments",
                )
                if any(
                    name not in record
                    or type(record[name]) is not int
                    or not 0 <= record[name] < 2**64
                    for name in geometry
                ):
                    raise ValueError("invalid or missing native row geometry")
                live, extends, index, capacity = (
                    record[name]
                    for name in (
                        "live_requests",
                        "num_extends",
                        "row_index",
                        "request_capacity",
                    )
                )
                if not (0 <= extends <= live <= capacity and 0 <= index < live):
                    raise ValueError("invalid live/padded row geometry")
                phase = "prefill" if index < extends else "decode"
                if record["phase"] != phase:
                    raise ValueError("row phase differs from frozen native partition")
                path = record["path"]
                if path not in (
                    "none",
                    "eager",
                    "decode_graph",
                    "prefill_breakable",
                    "prefill_inline",
                ):
                    raise ValueError("unknown target route")
                width, kept, committed = (
                    record[k]
                    for k in ("verify_width", "kept_prefix", "committed_tokens")
                )
                if phase == "decode" and (
                    width == 0 or width != record["input_length"]
                ):
                    raise ValueError(
                        "ordinary verify width differs from frozen input length"
                    )
                if phase == "prefill" and width != 0:
                    raise ValueError("prefill has no verify window")
                if record["query_capacity"] < record["input_length"]:
                    raise ValueError("target capacity does not cover the live row")
                if path == "decode_graph" and (
                    extends != 0 or record["query_capacity"] != capacity * width
                ):
                    raise ValueError("invalid ordinary decode graph geometry")
                if path in ("prefill_breakable", "prefill_inline") and extends == 0:
                    raise ValueError("prefill graph has no prefill rows")
                if path in ("eager", "prefill_breakable") and capacity != live:
                    raise ValueError("unexpected request padding for this target route")
                if not 0 <= committed <= kept:
                    raise ValueError("invalid native kept/committed prefix")
                if record["phase"] == "decode" and not 0 <= kept <= width or width < 0:
                    raise ValueError("invalid verification width")
                accepted, forced = (
                    record["accepted_before_stop"],
                    record["force_single"],
                )
                if type(forced) is not bool or type(accepted) is not int:
                    raise ValueError("invalid acceptance-before-stop observation")
                if record["phase"] == "decode":
                    if (
                        not 0 <= kept <= accepted <= width
                        or accepted < 1
                        or (forced and accepted != 1)
                    ):
                        raise ValueError(
                            "invalid pre-stop/kept acceptance relationship"
                        )
                elif accepted != -1 or forced:
                    raise ValueError("prefill cannot report decode acceptance")
                if record["numerical_error"]:
                    raise ValueError("numerical corruption in observed execution")
                if record["path"] == "none":
                    if record["input_length"] != 0:
                        raise ValueError("no-work row had target input")
                elif record["input_length"] <= 0:
                    raise ValueError("execution without live target rows")
                if (
                    record["path"]
                    in ("decode_graph", "prefill_breakable", "prefill_inline")
                    and record["graph_segments"] <= 0
                ):
                    raise ValueError("graph route without a replay")
                if record["discarded"] and committed:
                    raise ValueError("discarded execution committed client tokens")
                self.rounds[key][ordinal] = record
            elif event == "request_closed":
                n = len(self.dispatched[key])
                if (
                    not record["evidence_valid"]
                    or record["missing_rounds"]
                    or record["terminal"] is None
                    or record["assigned_rounds"] != n
                    or record["completed_rounds"] != n
                    or len(self.rounds[key]) != n
                ):
                    raise ValueError("incomplete admission execution evidence")
                self.closed[key] = record
            else:
                raise ValueError(f"unknown execution event {event}")
        if not streams:
            raise ValueError("missing observation trace")
        for (rank, protocol), records in streams.items():
            if (
                records[0]["event"] != "trace_started"
                or records[-1]["event"] != "trace_finished"
                or not records[-1]["evidence_valid"]
                or records[-1]["lost_observations"]
            ):
                raise ValueError("missing or invalid trace closure")
            if (rank, EXECUTION) not in streams or (rank, CACHE) not in streams:
                raise ValueError(
                    "both execution and cache stream endpoints are required"
                )
        if self.started.keys() != self.closed.keys():
            raise ValueError("unclosed native admission")

    @classmethod
    def read(cls, paths):
        records = []
        for path in paths:
            for line in Path(path).read_text().splitlines():
                for marker in ("[execution-event] ", "[cache-event] "):
                    if marker in line:
                        records.append(json.loads(line.split(marker, 1)[1]))
                        break
        return cls(records)

    def check_snapshot(self, snapshot):
        validate_snapshot(snapshot, None)
        if not snapshot["evidence_valid"]:
            raise ValueError("invalid execution snapshot")
        key = request_key(snapshot)
        if key not in self.closed:
            raise ValueError("snapshot admission has no closed trace")
        if self.started[key]["origin"] != snapshot["origin"]:
            raise ValueError("snapshot admission origin differs")
        prefix = [
            r
            for ordinal, r in self.rounds[key].items()
            if ordinal <= snapshot["through_round"]
        ]
        counts = Counter(
            (r["phase"], r["path"], r["verify_width"], r["kept_prefix"]) for r in prefix
        )
        pre_counts = Counter(
            (r["path"], r["verify_width"], r["accepted_before_stop"], r["force_single"])
            for r in prefix
            if r["phase"] == "decode"
        )
        expected_pre = {
            (
                b["path"],
                b["verify_width"],
                b["accepted_before_stop"],
                b["force_single"],
            ): b["count"]
            for b in snapshot["pre_stop_bins"]
        }
        if pre_counts != expected_pre:
            raise ValueError(
                "wire pre-stop histogram does not match native completed rounds"
            )
        expected = {
            (b["phase"], b["path"], b["verify_width"], b["kept_prefix"]): b["count"]
            for b in snapshot["bins"]
        }
        if (
            counts != expected
            or len(prefix) != snapshot["completed_rounds"]
            or sum(r["committed_tokens"] for r in prefix)
            != snapshot["committed_tokens"]
            or snapshot["assigned_rounds"] > len(self.dispatched[key])
        ):
            raise ValueError("wire snapshot does not match native completed rounds")

    def require(self, snapshot, *, physical_ranks, cells, restored):
        """Qualify this explicit measured admission on each requested rank.

        A cell is (phase, path, W); prefill W is zero. Caller separately checks
        successful output and numerical equality. Pressure admissions never
        substitute for the supplied snapshot identity.
        """
        self.check_snapshot(snapshot)
        if not physical_ranks or snapshot["physical_rank"] not in physical_ranks:
            raise ValueError(
                "required physical ranks must include the snapshot's producer"
            )
        if snapshot["origin"] != "serving":
            raise ValueError("startup cannot qualify a measured request")
        for rank in physical_ranks:
            identity = dict(snapshot, physical_rank=rank)
            key = request_key(identity)
            if key not in self.closed:
                raise ValueError("missing physical-rank admission evidence")
            if self.started[key]["origin"] != "serving":
                raise ValueError(
                    "required peer rank supplied startup rather than serving execution"
                )
            rows = self.rounds[key].values()
            actual = {
                (r["phase"], r["path"], r["verify_width"])
                for r in rows
                if r["input_length"] > 0 and not r["discarded"]
            }
            if not set(cells) <= actual:
                raise ValueError(
                    f"measured admission lacks required execution cells on rank {rank}"
                )
            if restored:
                matches = [
                    restore for restore in self.restores if request_key(restore) == key
                ]
                if not any(
                    self._owns_restored_slot(
                        restore, self.dispatched[key][row["round"]]
                    )
                    and restore["publication_order"]
                    < self.dispatched[key][row["round"]]["publication_order"]
                    < row["publication_order"]
                    and row["input_length"] > 0
                    and not row["discarded"]
                    for restore in matches
                    for row in rows
                ):
                    raise ValueError(
                        "no completed target execution dispatched after restore"
                    )

    def require_eager_overflow(
        self, snapshot, *, physical_ranks, verify_width, capture_max_requests
    ):
        """Require a live pure-decode batch above the configured graph ladder."""
        if type(capture_max_requests) is not int or capture_max_requests < 1:
            raise ValueError("capture maximum must be a positive request count")
        self.require(
            snapshot,
            physical_ranks=physical_ranks,
            cells=[("decode", "eager", verify_width)],
            restored=False,
        )
        for rank in physical_ranks:
            key = request_key(dict(snapshot, physical_rank=rank))
            if not any(
                row["phase"] == "decode"
                and row["path"] == "eager"
                and row["verify_width"] == verify_width
                and row["num_extends"] == 0
                and row["live_requests"] > capture_max_requests
                and row["input_length"] > 0
                and not row["discarded"]
                for row in self.rounds[key].values()
            ):
                raise ValueError(
                    f"no measured eager decode above the capture maximum on rank {rank}"
                )

    def require_copied_restore(self, snapshot, *, physical_ranks):
        """Additionally require positive locally owned cache copy, not an empty ACK."""
        self.require(snapshot, physical_ranks=physical_ranks, cells=[], restored=True)
        for rank in physical_ranks:
            key = request_key(dict(snapshot, physical_rank=rank))
            if not any(
                request_key(restore) == key
                and (
                    (restore["l2_rows"] > 0 and restore["l2_bytes"] > 0)
                    or (restore["pool_rows"] > 0 and restore["pool_bytes"] > 0)
                )
                and self._owns_restored_slot(
                    restore, self.dispatched[key][row["round"]]
                )
                and restore["publication_order"]
                < self.dispatched[key][row["round"]]["publication_order"]
                < row["publication_order"]
                and row["input_length"] > 0
                and not row["discarded"]
                for restore in self.restores
                for row in self.rounds[key].values()
            ):
                raise ValueError(
                    "no positive local cache copy followed by measured execution"
                )

    def _owns_restored_slot(self, restore, dispatch):
        if restore["slot"] != dispatch["slot"]:
            return False
        owners = [
            event
            for event in self.slot_events
            if rank_key(event) == rank_key(dispatch)
            and event["slot"] == dispatch["slot"]
            and event["event"] == "slot_assigned"
            and event["publication_order"] < dispatch["publication_order"]
        ]
        if not owners:
            return False
        # A delayed release of an older generation must not erase a newer
        # binding. Generations come from native slot ownership, not time.
        owner = max(owners, key=lambda event: event["slot_generation"])
        if (
            request_key(owner) != request_key(restore)
            or owner["slot_generation"] != restore["slot_generation"]
        ):
            return False
        return not any(
            event["event"] == "slot_released"
            and rank_key(event) == rank_key(owner)
            and event["slot"] == owner["slot"]
            and event["slot_generation"] == owner["slot_generation"]
            and event["publication_order"] < dispatch["publication_order"]
            for event in self.slot_events
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument(
        "--snapshots", required=True, help="JSON list of measured completion snapshots"
    )
    parser.add_argument(
        "--ranks",
        required=True,
        help="comma-separated physical ranks required per admission",
    )
    parser.add_argument(
        "--require", action="append", default=[], metavar="PHASE:PATH:WIDTH"
    )
    parser.add_argument("--restored", action="store_true")
    parser.add_argument("--copied-restore", action="store_true")
    args = parser.parse_args()
    cells = [
        (phase, path, int(width))
        for phase, path, width in (cell.split(":") for cell in args.require)
    ]
    trace = ExecutionTrace.read(args.logs)
    snapshots = json.loads(Path(args.snapshots).read_text())
    if not snapshots:
        raise ValueError("no measured snapshots")
    for snapshot in snapshots:
        trace.require(
            snapshot,
            physical_ranks=[int(rank) for rank in args.ranks.split(",")],
            cells=cells,
            restored=args.restored,
        )
        if args.copied_restore:
            trace.require_copied_restore(
                snapshot, physical_ranks=[int(rank) for rank in args.ranks.split(",")]
            )
    print(
        f"Validated {len(snapshots)} measured admission snapshots and their closed execution traces"
    )


if __name__ == "__main__":
    main()
