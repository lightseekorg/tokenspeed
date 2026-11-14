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

"""Check serving assertions against downloaded controller and rank records."""

import argparse
import json
from pathlib import Path


def nccl_counts(rank_rows, ranks):
    observations = [
        row for row in rank_rows if row["event"] in {"nccl_inventory", "nccl_complete"}
    ]
    assert len(observations) == 4 * len(ranks), "Missing or duplicate NCCL evidence"
    assert all(
        {"rank", "release", "operation", "ns"} <= row.keys()
        and type(row["release"]) is bool
        and type(row["operation"]) is int
        and row["operation"] > 0
        for row in observations
    ), "Invalid NCCL operation evidence"
    assert {row["rank"] for row in observations} == ranks, "Missing NCCL rank"
    counts = {}
    for rank in ranks:
        phases = {}
        for release in (True, False):
            rows = [
                row
                for row in observations
                if row["rank"] == rank and row["release"] == release
            ]
            inventory = [row for row in rows if row["event"] == "nccl_inventory"]
            completed = [row for row in rows if row["event"] == "nccl_complete"]
            assert len(inventory) == len(completed) == 1, (
                "Missing or duplicate NCCL phase",
                rank,
                release,
            )
            inventory, completed = inventory[0], completed[0]
            ids = inventory.get("backend_ids")
            count = completed.get("count")
            assert (
                isinstance(ids, list)
                and ids
                and all(type(value) is int and value > 0 for value in ids)
            ), "Missing or invalid NCCL inventory"
            assert len(set(ids)) == len(ids), "Duplicate NCCL backend"
            assert type(count) is int and count > 0 and count == len(ids), (
                "NCCL count differs from its production inventory",
                rank,
                release,
            )
            assert (
                inventory["operation"] == completed["operation"]
            ), "Mismatched NCCL operation"
            apply = [
                row
                for row in rank_rows
                if row["event"] in {"apply_begin", "apply_complete"}
                and row["rank"] == rank
                and row["release"] == release
                and "nccl" in row["tags"]
            ]
            begins = [row for row in apply if row["event"] == "apply_begin"]
            ends = [row for row in apply if row["event"] == "apply_complete"]
            assert (
                len(begins) == len(ends) == 1
            ), "Missing or duplicate NCCL apply phase"
            stamps = [
                begins[0]["ns"],
                inventory["ns"],
                completed["ns"],
                ends[0]["ns"],
            ]
            assert all(type(stamp) is int and stamp >= 0 for stamp in stamps) and all(
                first < second for first, second in zip(stamps, stamps[1:])
            ), "NCCL evidence lies outside its apply phase"
            phases[release] = (set(ids), inventory["operation"], stamps)
        assert phases[True][0] == phases[False][0], "Unpaired NCCL backend inventory"
        assert phases[True][1] < phases[False][1], "Unpaired NCCL operation order"
        assert phases[True][2][-1] < phases[False][2][0], "Unpaired NCCL phase order"
        counts[str(rank)] = len(phases[True][0])
    return counts


def summarize(directory):
    serving = directory / "serving"
    result = json.loads((serving / "result.json").read_text())
    rank_rows = [
        json.loads(line)
        for path in sorted(serving.glob("node-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    mapping = [row for row in rank_rows if row["event"] == "mapping"]
    negotiated = [row for row in rank_rows if row["event"] == "negotiated"]
    ranks = {row["rank"] for row in mapping}
    assert len(mapping) == 4 and ranks == set(range(4))
    assert {row["rank"] for row in negotiated} == ranks
    nodes_by_rank = {row["rank"]: row["node"] for row in mapping}
    assert all(
        len(row.get("mapped_memory_saver_hooks", [])) == 1 for row in mapping
    ), "Missing or duplicate memory-saver hook evidence; rerun with the current harness"
    assert {row["rank"]: row["protocol"] for row in negotiated} == {
        rank: result["protocols"][node] for rank, node in nodes_by_rank.items()
    }
    assert set(nodes_by_rank.values()) == {0, 1}
    if result["layout"] == "dp2-tp2-ep4":
        for row in mapping:
            rank = row["rank"]
            assert row["tp"] == [2 * row["node"], 2 * row["node"] + 1]
            assert row["dp"] == [rank % 2, rank % 2 + 2]
            assert row["ep"] == [0, 1, 2, 3]
    assert (directory / "serving.exit").read_text().strip() == "0"
    summary = {
        "layout": result["layout"],
        "protocols": result["protocols"],
        "status": result["status"],
        "rank_count": len(ranks),
        "launcher_exits": result["launcher_exits"],
    }
    if directory.name.startswith("serve-flashinfer-"):
        selections = (directory / "kernel-selection.log").read_text().splitlines()
        kernel = "flashinfer_cutlass_unquant_moe_apply"
        for node in (0, 1):
            assert any(
                line.startswith(f"node-{node}:") and f"-> {kernel}" in line
                for line in selections
            ), (directory, node)
        summary["moe_apply_kernel"] = kernel
    if result["status"] == "rejected-before-ready":
        assert result["launcher_exits"] == [1, 1]
        assert result["wire"] == {"HELLO": 0, "READY": 0, "input": 0}
        assert {
            row["rank"] for row in rank_rows if row["event"] == "rendezvous_refused"
        } == ranks
        assert not any(
            row["event"] in ("service_ready", "apply_begin") for row in rank_rows
        )
        return summary | {"wire": result["wire"]}
    assert result["status"] == "passed"
    assert result["launcher_exits"] == [0, 0]
    assert {
        row["rank"] for row in rank_rows if row["event"] == "service_ready"
    } == ranks
    # Keep both phases in the archive: a passing live assertion alone cannot
    # let a reviewer independently check the restored output.
    assert {
        "tokens_per_ingress",
        "tokens_after_resume_per_ingress",
    } <= result.keys(), (
        "Missing before/after token evidence; rerun with the current harness"
    )
    before = result["tokens_per_ingress"]
    after = result["tokens_after_resume_per_ingress"]
    ingresses = range({"ep4": 4, "dp2-tp2-ep4": 2}[result["layout"]])
    expected = {str(ingress) for ingress in ingresses}
    assert isinstance(before, dict) and isinstance(after, dict)
    assert set(before) == set(after) == expected, "Missing or unexpected ingress output"
    assert result["token_count"] == 8
    assert all(
        isinstance(tokens, list)
        and len(tokens) == 8
        and all(type(token) is int and token >= 0 for token in tokens)
        for phase in (before, after)
        for tokens in phase.values()
    ), "Expected eight token IDs for every ingress in both phases"
    assert before == after, "Token output differs after resume"
    tags = result["tags"]
    # CUDA v2 keeps one legacy apply per direction. Only the independent v3
    # validation branch separates NCCL teardown/recreation from memory work.
    split_nccl = result["protocols"] == [3, 3] and "nccl" in tags
    expected_phases = {
        True: [set(tags) - {"nccl"}, {"nccl"}] if split_nccl else [set(tags)],
        False: [set(), set(tags)] if split_nccl else [set(tags)],
    }
    waits = {}
    for operation, release in (("release", True), ("resume", False)):
        finished = [
            row
            for row in rank_rows
            if row["event"] == "apply_complete" and row["release"] == release
        ]
        for rank in ranks:
            phases = sorted(
                (row for row in finished if row["rank"] == rank),
                key=lambda row: row["ns"],
            )
            assert [set(row["tags"]) for row in phases] == expected_phases[release], (
                "Unexpected apply phases",
                operation,
                rank,
                phases,
            )
        # Timing alone cannot prove that every ingress replied.
        acks = result.get(f"{operation}_ack_ns")
        assert isinstance(acks, list) and len(acks) == len(
            expected
        ), f"Expected {len(expected)} {operation} ACKs, one per ingress"
        assert all(
            type(ack) is int and ack >= 0 for ack in acks
        ), f"Expected integer timestamps for {operation} ACKs"
        earliest_ack = min(acks)
        waits[operation] = round(
            (earliest_ack - max(row["ns"] for row in finished)) / 1e6, 3
        )
        if result["protocols"] == [3, 3]:
            assert earliest_ack > max(row["ns"] for row in finished)
    if "nccl" in tags:
        summary["nccl_backend_counts_per_rank"] = nccl_counts(rank_rows, ranks)
    return summary | {
        "tags": tags,
        "tokens_per_ingress": result["tokens_per_ingress"],
        "tokens_after_resume_per_ingress": after,
        "first_ack_after_last_rank_ms": waits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({p.name: summarize(p) for p in args.directories}, indent=2))
