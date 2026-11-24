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

"""Host-only receipts from actual target calls, carried by the result fence.

V1 observes ordinary chains. Optional batch scratch snapshots acceptance
before stop clipping and the force-single mask the verifier actually consumes.
The existing output_lengths supplies the prefix kept after clipping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


def enabled() -> bool:
    value = os.environ.get("TOKENSPEED_EXECUTION_EVENT_LOG", "0")
    if value not in ("0", "1"):
        raise ValueError("TOKENSPEED_EXECUTION_EVENT_LOG must be 0 or 1")
    return value == "1"


def validate_scope(
    *,
    device: str,
    pp_size: int,
    query_shard_size: int,
    algorithm: str | None,
    tree: bool,
    narrowed: bool,
    synthetic: bool,
) -> None:
    """Refuse unsupported observations before capture; never change a route."""
    if (
        device != "cuda"
        or pp_size != 1
        or query_shard_size != 1
        or algorithm not in (None, "DUMMY", "DFLASH")
        or tree
        or narrowed
        or synthetic
    ):
        raise ValueError(
            "execution observations v1 require GPU PP1 ordinary unsharded "
            "chains (no drafter, DUMMY or DFLASH), without narrowing or "
            "synthetic acceptance; packed/tree/harness routes are unsupported"
        )


@dataclass(frozen=True)
class TargetExecution:
    """One successful target invocation; completion still requires its fence."""

    path: str
    verify_width: int
    request_capacity: int
    query_capacity: int
    graph_segments: int

    def verified(self, candidates, *, packed: bool) -> TargetExecution:
        """Attach the layout actually consumed by the ordinary verifier."""
        if packed:
            raise ValueError(
                "execution observations v1 do not support packed verification"
            )
        width = 0 if candidates is None else candidates.shape[1]
        return replace(self, verify_width=width)

    def replayed(
        self, *, request_capacity: int, query_capacity: int
    ) -> TargetExecution:
        """Called only after the supplied target decode graph was replayed."""
        if self.verify_width <= 0:
            raise ValueError("observed decode graph contains no verification")
        if query_capacity != request_capacity * self.verify_width:
            raise ValueError("execution observations v1 do not support packed graphs")
        return replace(
            self,
            path="decode_graph",
            request_capacity=request_capacity,
            query_capacity=query_capacity,
            graph_segments=1,
        )

    def payload(self, admission_ids: tuple[int, ...]) -> tuple:
        return (
            admission_ids,
            self.path,
            self.verify_width,
            len(admission_ids),
            self.request_capacity,
            self.query_capacity,
            self.graph_segments,
        )


class VerificationObservations:
    """Batch scratch, allocated in the executor's CPU-backed named region.

    No request ownership lives here. Each sampling call overwrites every row;
    the result owns a fresh CPU copy under the ordinary completion fence.
    """

    def __init__(self, *, max_bs: int, device: str):
        import torch

        self.buffer = torch.empty((max_bs, 2), dtype=torch.int32, device=device)

    def record(self, accept_lengths, *, num_extends: int, force_single):
        rows = self.buffer[: accept_lengths.shape[0]]
        rows[:, 0].copy_(accept_lengths)
        rows[:num_extends, 0].fill_(-1)
        rows[:, 1].zero_()
        if force_single is not None:
            rows[num_extends:, 1].copy_(force_single)
        return rows


def validate_snapshot(value: dict, previous: dict | None) -> None:
    """Validate an append-only cumulative snapshot before accepting its output."""
    import msgspec

    if len(msgspec.msgpack.encode(value)) > 256 * 1024:
        raise ValueError("execution snapshot exceeds the 256 KiB wire byte budget")
    omitted = value.get("histograms_omitted", False)
    if type(omitted) is not bool or (
        omitted and (value["evidence_valid"] or value["bins"] or value["pre_stop_bins"])
    ):
        raise ValueError("omitted histograms must be explicitly invalid and empty")
    identity = (
        "version",
        "instance_id",
        "role",
        "dp_rank",
        "physical_rank",
        "request_id",
        "admission_id",
        "origin",
    )
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported execution observation version")
    for key in (
        "admission_id",
        "assigned_rounds",
        "completed_rounds",
        "through_round",
        "missing_rounds",
        "committed_tokens",
        "physical_rank",
        "dp_rank",
    ):
        if type(value[key]) is not int or not 0 <= value[key] < 2**64:
            raise ValueError(f"invalid execution observation {key}")
    if value["admission_id"] == 0 or value["origin"] not in ("serving", "startup"):
        raise ValueError("invalid execution observation identity")
    if type(value["evidence_valid"]) is not bool:
        raise ValueError("invalid execution observation validity")
    if (
        value["completed_rounds"] + value["missing_rounds"] > value["assigned_rounds"]
        or value["through_round"] > value["assigned_rounds"]
    ):
        raise ValueError("invalid execution observation counters")
    bins = value["bins"]
    if "pre_stop_bins" not in value:
        raise ValueError("execution observation is missing acceptance before stop")
    if len(bins) + len(value["pre_stop_bins"]) > 4096:
        raise ValueError("execution observation histogram exceeds its bound")
    counts = {}
    for item in bins:
        phase, path = item["phase"], item["path"]
        width, kept, count = (item[k] for k in ("verify_width", "kept_prefix", "count"))
        if (
            phase not in ("prefill", "decode")
            or path
            not in (
                "none",
                "eager",
                "decode_graph",
                "prefill_breakable",
                "prefill_inline",
            )
            or any(
                type(n) is not int or not 0 <= n < 2**64 for n in (width, kept, count)
            )
            or count == 0
            or (phase == "decode" and not 0 <= kept <= width)
            or (phase == "decode" and width == 0)
            or (phase == "prefill" and (width != 0 or path == "decode_graph"))
        ):
            raise ValueError("invalid execution observation bin")
        key = (phase, path, width, kept)
        if key in counts:
            raise ValueError("duplicate execution observation bin")
        counts[key] = count
    if value["evidence_valid"] and sum(counts.values()) != value["completed_rounds"]:
        raise ValueError(
            "execution observation histogram does not cover completed rounds"
        )
    pre_counts = {}
    for item in value["pre_stop_bins"]:
        path, width, accepted, forced, count = (
            item[k]
            for k in (
                "path",
                "verify_width",
                "accepted_before_stop",
                "force_single",
                "count",
            )
        )
        if (
            path not in ("eager", "decode_graph", "prefill_breakable", "prefill_inline")
            or any(
                type(n) is not int or not 0 <= n < 2**64
                for n in (width, accepted, count)
            )
            or not 1 <= accepted <= width
            or count == 0
            or type(forced) is not bool
            or (forced and accepted != 1)
        ):
            raise ValueError("invalid acceptance-before-stop bin")
        key = (path, width, accepted, forced)
        if key in pre_counts:
            raise ValueError("duplicate acceptance-before-stop bin")
        pre_counts[key] = count
    decode_count = sum(
        count for (phase, _, _, _), count in counts.items() if phase == "decode"
    )
    if value["evidence_valid"] and sum(pre_counts.values()) != decode_count:
        raise ValueError(
            "acceptance-before-stop histogram does not cover decode rounds"
        )
    if previous is not None:
        if any(previous[k] != value[k] for k in identity):
            raise ValueError("execution observation identity changed")
        for key in (
            "assigned_rounds",
            "completed_rounds",
            "through_round",
            "missing_rounds",
            "committed_tokens",
        ):
            if value[key] < previous[key]:
                raise ValueError(f"execution observation {key} decreased")
        if not previous["evidence_valid"] and value["evidence_valid"]:
            raise ValueError("execution observation validity cannot recover")
        if previous.get("histograms_omitted", False) and not omitted:
            raise ValueError("omitted execution histograms cannot recover")
        if omitted:
            return
        for item in previous["bins"]:
            key = (
                item["phase"],
                item["path"],
                item["verify_width"],
                item["kept_prefix"],
            )
            if counts.get(key, 0) < item["count"]:
                raise ValueError("execution observation histogram decreased")

        for item in previous["pre_stop_bins"]:
            key = (
                item["path"],
                item["verify_width"],
                item["accepted_before_stop"],
                item["force_single"],
            )
            if pre_counts.get(key, 0) < item["count"]:
                raise ValueError("acceptance-before-stop histogram decreased")
