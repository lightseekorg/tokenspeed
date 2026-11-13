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
    assert ranks == set(range(len(mapping)))
    assert {row["rank"] for row in negotiated} == ranks
    nodes_by_rank = {row["rank"]: row["node"] for row in mapping}
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
    assert all(
        len(tokens) == result["token_count"]
        for tokens in result["tokens_per_ingress"].values()
    )
    tags = result["tags"]
    expected_phases = {
        True: [set(tags) - {"nccl"}, {"nccl"}] if "nccl" in tags else [set(tags)],
        False: [set(), set(tags)] if "nccl" in tags else [set(tags)],
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
                operation,
                rank,
                phases,
            )
        earliest_ack = min(result[f"{operation}_ack_ns"])
        waits[operation] = round(
            (earliest_ack - max(row["ns"] for row in finished)) / 1e6, 3
        )
        if result["protocols"] == [3, 3]:
            assert earliest_ack > max(row["ns"] for row in finished)
    return summary | {
        "tags": tags,
        "tokens_per_ingress": result["tokens_per_ingress"],
        "first_ack_after_last_rank_ms": waits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({p.name: summarize(p) for p in args.directories}, indent=2))
