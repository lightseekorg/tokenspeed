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

"""Validate actual PTA/native ownership witnesses; no device imports are needed."""

from collections import Counter


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def covered(pointer, size, base, extent):
    return (
        all(type(value) is int for value in (pointer, size, base, extent))
        and pointer >= 0
        and base >= 0
        and size > 0
        and extent > 0
        and base <= pointer < base + extent
        and pointer + size <= base + extent
    )


def validate_witness(row):
    require(row["parameters"], "No real model parameter storage was witnessed")
    pools = {tuple(pool["id"]): pool for pool in row["pools"]}
    require(
        len(pools) == len(row["pools"]), "Different logical pools share a PTA pool ID"
    )
    canonical = [
        pool
        for pool in pools.values()
        if (pool["tag"], pool["policy"], pool["scope"])
        == ("weights", "cpu_backup", "persistent_buffers")
    ]
    require(
        len(canonical) == 1,
        "Canonical persistent CPU-backup pool is missing or duplicated",
    )
    persistent_id = tuple(canonical[0]["id"])
    segments = row["PTA_segments"]
    blocks = row["native"]["blocks"]
    for parameter in row["parameters"]:
        require(
            covered(
                parameter["storage_ptr"],
                parameter["storage_bytes"],
                parameter["storage_ptr"],
                parameter["storage_bytes"],
            ),
            "Invalid parameter range",
        )
    for segment in segments:
        require(
            covered(
                segment["address"],
                segment["total_size"],
                segment["address"],
                segment["total_size"],
            ),
            "Invalid PTA segment range",
        )
    for block in blocks:
        base = int(block["ptr"], 16)
        require(
            covered(base, block["aligned_bytes"], base, block["aligned_bytes"]),
            "Invalid native block range",
        )
    require(
        not row["native"]["callback_fault"],
        "Native callback fault in ownership witness",
    )

    def segment_for(pointer, size):
        matches = [
            segment
            for segment in segments
            if covered(pointer, size, segment["address"], segment["total_size"])
        ]
        require(
            len(matches) == 1,
            "Storage is not covered by exactly one actual PTA segment",
        )
        return matches[0]

    def block_for(pointer, size):
        matches = [
            block
            for block in blocks
            if covered(pointer, size, int(block["ptr"], 16), block["aligned_bytes"])
        ]
        require(len(matches) == 1, "Storage bypassed actual native VMM backing")
        return matches[0]

    def owner_for(segment):
        pool_id = tuple(segment["segment_pool_id"])
        require(pool_id in pools, "Actual PTA segment has no registered owned pool")
        return pools[pool_id]

    for parameter in row["parameters"]:
        segment = segment_for(parameter["storage_ptr"], parameter["storage_bytes"])
        owner = owner_for(segment)
        block = block_for(parameter["storage_ptr"], parameter["storage_bytes"])
        require(
            covered(
                segment["address"],
                segment["total_size"],
                int(block["ptr"], 16),
                block["aligned_bytes"],
            ),
            "The full selected PTA segment is not covered by the parameter native block",
        )
        require(
            owner["tag"] == "weights"
            and owner["policy"] == "discard"
            and tuple(segment["segment_pool_id"]) != persistent_id
            and (block["tag"], block["policy"]) == ("weights", "discard"),
            f"Actual model parameter {parameter['name']} is not in a separate DISCARD weights pool",
        )

    persistent_segments = [
        segment
        for segment in segments
        if tuple(segment["segment_pool_id"]) == persistent_id
    ]
    require(
        persistent_segments, "Canonical persistent pool has no actual PTA allocation"
    )
    for segment in persistent_segments:
        block = block_for(segment["address"], segment["total_size"])
        require(
            (block["tag"], block["policy"]) == ("weights", "backup"),
            "Persistent PTA allocation did not use native CPU backup",
        )

    # Classify every weights-tag block. Merely accepting a mixed policy set is
    # insufficient: an unrelated or unowned backup block must not pass.
    for block in blocks:
        if block["tag"] != "weights":
            continue
        owned_segments = [
            segment
            for segment in segments
            if covered(
                segment["address"],
                segment["total_size"],
                int(block["ptr"], 16),
                block["aligned_bytes"],
            )
        ]
        require(owned_segments, "A weights block has no matching actual PTA ownership")
        for segment in owned_segments:
            pool_id = tuple(segment["segment_pool_id"])
            owner = owner_for(segment)
            if block["policy"] == "backup":
                require(
                    pool_id == persistent_id,
                    "Backup weights block is not the canonical persistent pool",
                )
            else:
                require(
                    block["policy"] == owner["policy"] == "discard"
                    and owner["tag"] == "weights",
                    "Unclassified weights block or policy mismatch",
                )
    return {
        "rank": row["rank"],
        "parameter_storages": len(row["parameters"]),
        "persistent_pool_id": persistent_id,
        "persistent_PTA_segments": len(persistent_segments),
    }


def validate_trace_ownership(events, *, tp_size):
    transitions = [row for row in events if row["event"] in {"pause", "resume"}]
    pids = {row["pid"] for row in transitions}
    witnesses = [row for row in events if row["event"] == "persistent-ownership"]
    starts = [row for row in witnesses if row["operation"] == "validate_startup"]
    require(
        len(pids) == tp_size
        and {row["pid"] for row in starts} == pids
        and len(starts) == tp_size
        and {row["rank"] for row in starts} == set(range(tp_size)),
        "Missing, duplicated, or rank-mismatched real startup ownership witness",
    )
    rank_by_pid = {row["pid"]: row["rank"] for row in starts}
    expected = Counter(
        (row["pid"], row["event"], row["tag"], when)
        for row in transitions
        for when in ("before", "after")
    )
    actual = Counter(
        (row["pid"], row["operation"], row["tag"], row["when"])
        for row in witnesses
        if row["operation"] != "validate_startup"
    )
    require(
        actual == expected,
        "Each actual transition needs exactly one before/after ownership witness",
    )
    for transition in transitions:
        for when in ("before", "after"):
            matching = [
                row
                for row in witnesses
                if row["pid"] == transition["pid"]
                and row["operation"] == transition["event"]
                and row["tag"] == transition["tag"]
                and row["when"] == when
            ]
            (witness,) = matching
            require(
                witness["native"] == transition[when],
                "Ownership witness is not bound to the actual transition's native snapshot",
            )
    summaries = []
    for row in witnesses:
        require(
            row["rank"] == rank_by_pid[row["pid"]],
            "Witness PID/rank association changed",
        )
        summaries.append(validate_witness(row))
    for pid in pids:
        per_pid = [row for row in witnesses if row["pid"] == pid]
        parameter_sets = [
            {
                (item["name"], item["storage_ptr"], item["storage_bytes"])
                for item in row["parameters"]
            }
            for row in per_pid
        ]
        require(
            all(items == parameter_sets[0] for items in parameter_sets),
            "Actual parameter storage identity changed during the transition",
        )
        canonical_ids = {
            item["persistent_pool_id"]
            for item in summaries
            if item["rank"] == rank_by_pid[pid]
        }
        require(len(canonical_ids) == 1, "Persistent PTA pool identity changed")
    return {
        "ranks": sorted(rank_by_pid.values()),
        "ownership_witnesses": len(witnesses),
        "actual_parameters_discard": True,
        "canonical_persistent_backup": True,
        "model_constant_contents_qualified": False,
    }
