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

"""Fail-closed oracle shared by the draft-graph regression tools.

Only actual replay rows count as graph evidence. Control (eager/eager) calls
are tracked separately, and full input/output/cache snapshots are compared
in addition to proposals. Kept with the graph PR so its oracle has independent
negative tests before the optional serving tools are merged.
"""

from __future__ import annotations

import math


def compare_proposals(
    graph: list[dict], eager: list[dict], min_match: float, max_gap_delta: float
) -> tuple[list[str], str]:
    """Check matched step-0 traces; no percentage of arbitrary errors is allowed."""
    if min_match != 1.0:
        raise ValueError(
            "min_match must be 1.0; arbitrary proposal errors are not allowed"
        )
    if not math.isfinite(max_gap_delta) or max_gap_delta < 0:
        raise ValueError("max_gap_delta must be finite and non-negative")
    if not graph or len(graph) != len(eager):
        return ["missing or unequal draft trace lengths"], ""
    problems = []
    replay_rows = control_rows = 0
    for number, (g, e) in enumerate(zip(graph, eager, strict=True)):
        try:
            bs = g["bs"]
            if bs != e["bs"] or g["num_extends"] != e["num_extends"]:
                raise ValueError("batch layouts differ")
            first_g, first_e = g["first_step"], e["first_step"]
            if first_g["path"] not in ("graph", "eager") or first_e["path"] != "eager":
                raise ValueError("missing or invalid execution-path evidence")
            if not first_g["input_key"] or first_g["input_key"] != first_e["input_key"]:
                raise ValueError("step-0 inputs/frontiers differ")
            if g["accept"] != e["accept"] or len(g["accept"]) != bs:
                raise ValueError("accepted frontiers differ")
            for call in (g, e):
                if len(call["draft"]) != bs or len(call["margin"]) != bs:
                    raise ValueError("truncated proposal or margin rows")
                for ids, gaps in zip(call["draft"], call["margin"], strict=True):
                    if not ids or len(ids) != len(gaps):
                        raise ValueError("truncated proposal or margin columns")
            if first_g["path"] != "graph":
                control_rows += bs
                continue
            if g["num_extends"] != bs:
                raise ValueError("a mixed/decode call unexpectedly replayed step 0")
            replay_rows += bs
            for row in range(bs):
                ids_g, ids_e = g["draft"][row], e["draft"][row]
                gap_g, gap_e = g["margin"][row], e["margin"][row]
                if len(ids_g) != len(ids_e):
                    raise ValueError("draft-step counts differ")
                if ids_g[0] != ids_e[0]:
                    problems.append(f"call {number} row {row}: replay proposal differs")
                if not all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in (gap_g[0], gap_e[0])
                ):
                    problems.append(
                        f"call {number} row {row}: non-finite/missing step-0 gap"
                    )
                elif abs(gap_g[0] - gap_e[0]) > max_gap_delta:
                    problems.append(
                        f"call {number} row {row}: step-0 gap exceeds tolerance"
                    )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            problems.append(f"call {number}: invalid trace: {error}")
    if replay_rows == 0:
        problems.append("no actual replay rows were validated")
    return (
        problems,
        f"{replay_rows} replay rows; {control_rows} eager control rows (not counted)",
    )


def compare_payloads(
    graph: dict,
    eager: dict,
    *,
    logit_atol: float,
    logit_rtol: float,
    kv_atol: float,
    kv_rtol: float,
) -> list[str]:
    """Compare every input, full logits and written KV/index-K, never a sample.

    Inputs must be identical. If free-running runs diverge, they are not an
    oracle for each other; report that instead of aligning by batch size.
    Outputs permit the caller's explicit BF16 tolerances. Missing, empty or
    non-finite evidence always fails.
    """
    import torch

    if not isinstance(graph, dict) or not isinstance(eager, dict):
        return ["tensor payload must be a mapping"]
    problems = []
    tolerances = (logit_atol, logit_rtol, kv_atol, kv_rtol)
    if not all(math.isfinite(v) and v >= 0 for v in tolerances):
        raise ValueError("tensor tolerances must be finite and non-negative")
    for section, required in (
        ("inputs", {"input_ids", "positions", "hidden", "gather_ids", "frontier"}),
        ("outputs", {"logits", "latent_kv", "index_k"}),
    ):
        if not isinstance(graph.get(section), dict) or not isinstance(
            eager.get(section), dict
        ):
            problems.append(f"missing {section} payload")
            continue
        g, e = graph[section], eager[section]
        if set(g) != required or set(e) != required:
            problems.append(f"missing or unexpected {section} tensors")
            continue
        for name in sorted(required):
            left, right = g[name], e[name]
            label = f"{section}.{name}"
            if not isinstance(left, torch.Tensor) or not isinstance(
                right, torch.Tensor
            ):
                problems.append(f"{label}: missing tensor")
                continue
            if (
                left.shape != right.shape
                or left.dtype != right.dtype
                or left.numel() == 0
            ):
                problems.append(f"{label}: different/empty shape or dtype")
                continue
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                problems.append(f"{label}: non-finite values")
                continue
            if section == "inputs":
                if not torch.equal(left, right):
                    problems.append(f"{label}: inputs are not identical")
                continue
            atol, rtol = (
                (logit_atol, logit_rtol) if name == "logits" else (kv_atol, kv_rtol)
            )
            try:
                torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
            except AssertionError:
                problems.append(f"{label}: full tensor exceeds tolerance")
    return problems


def validate_trace_counts(run: dict) -> list[str]:
    """Require a trace for every counted served step; paired truncation fails."""
    try:
        calls = run["trace"]
        counts = run["counts"]
        paths = [call["first_step"]["path"] for call in calls]
        replayed = paths.count("graph")
        eager_extends = sum(
            path == "eager" and call["num_extends"] > 0
            for path, call in zip(paths, calls, strict=True)
        )
        if any(path not in ("graph", "eager") for path in paths):
            return ["trace contains an unknown execution path"]
        if replayed != counts["replays"] or eager_extends != counts["eager_extends"]:
            return ["served step counts do not match the trace; evidence is incomplete"]
    except (KeyError, TypeError, ValueError) as error:
        return [f"missing or invalid trace-count evidence: {error}"]
    return []
