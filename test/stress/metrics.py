"""Aggregation + post-run summary.

Reads the JSONL produced by events.JsonlSink and prints percentiles for
TTFT, TPOT, end-to-end latency, plus error taxonomy and health-probe stats.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


def _pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


@dataclass
class RequestStats:
    submitted: int = 0
    completed: int = 0
    cancelled: int = 0
    errored: int = 0
    invalid_schema: int = 0
    ttft_s: List[float] = field(default_factory=list)
    e2e_s: List[float] = field(default_factory=list)
    tpot_s: List[float] = field(default_factory=list)  # inter-token mean per request
    prompt_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    error_kinds: Counter = field(default_factory=Counter)
    cancel_stages: Counter = field(default_factory=Counter)
    # One entry per REQUEST_INVALID_SCHEMA event: kind -> count.
    invalid_schema_kinds: Counter = field(default_factory=Counter)
    # Count per workload tag (e.g. "grammar_schema/person_basic"); helps
    # diagnose schema-specific failures in mixed workloads.
    invalid_schema_by_workload: Counter = field(default_factory=Counter)
    # Output-quality audit findings (audit_finding events).
    audit_by_check: Counter = field(default_factory=Counter)  # check -> count
    audit_by_check_severity: Counter = field(default_factory=Counter)
    audit_by_workload: Counter = field(default_factory=Counter)
    audit_examples: Dict[str, str] = field(default_factory=dict)  # check -> detail
    # Streaming stalls (request_stall events), keyed by stage.
    stalls: Counter = field(default_factory=Counter)
    # Spec-decode acceptance length samples (from metrics_probe events).
    accept_len: List[float] = field(default_factory=list)


@dataclass
class HealthStats:
    probes: int = 0
    ok: int = 0
    fail: int = 0
    transitions: List[Tuple[float, str, str]] = field(default_factory=list)
    latencies_s: List[float] = field(default_factory=list)


@dataclass
class RssStats:
    """Aggregate of rss_probe events from RssMonitor.

    Stores only the minimal timeline needed to show growth — we keep
    (ts, total_kb) samples plus the first/last per-pid breakdown so the
    summary can name which process accounts for the growth.
    """

    samples: List[Tuple[float, int]] = field(default_factory=list)
    first_per_pid: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_per_pid: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: int = 0


def load_events(path: str) -> Iterable[dict]:
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip torn lines rather than aborting the whole summary.
                continue


def aggregate(
    path: str,
) -> Tuple[RequestStats, Dict[str, HealthStats], float, RssStats]:
    reqs = RequestStats()
    health: Dict[str, HealthStats] = defaultdict(HealthStats)
    rss = RssStats()

    # Per-request scratchpad so we can compute TPOT across streaming chunks.
    by_rid: Dict[str, dict] = {}
    run_start: Optional[float] = None
    run_end: Optional[float] = None
    last_ts: float = 0.0

    for ev in load_events(path):
        kind = ev.get("kind")
        ts = ev.get("ts", 0.0)
        if isinstance(ts, (int, float)):
            last_ts = max(last_ts, ts)
        data = ev.get("data", {})
        if kind == "run_started":
            run_start = ts
        elif kind == "run_finished":
            run_end = ts
        elif kind == "request_submitted":
            reqs.submitted += 1
            by_rid[data["rid"]] = {"submit_ts": ts, "first_ts": None}
        elif kind == "request_first_token":
            rid = data["rid"]
            if rid in by_rid:
                by_rid[rid]["first_ts"] = ts
        elif kind == "request_completed":
            reqs.completed += 1
            scratch = by_rid.pop(data["rid"], None)
            if scratch:
                if scratch["first_ts"] is not None:
                    reqs.ttft_s.append(scratch["first_ts"] - scratch["submit_ts"])
                    # Average inter-token latency for this request.
                    out_toks = data.get("output_tokens", 0)
                    if out_toks > 1:
                        decode_time = ts - scratch["first_ts"]
                        reqs.tpot_s.append(decode_time / max(1, out_toks - 1))
                reqs.e2e_s.append(ts - scratch["submit_ts"])
            reqs.prompt_tokens.append(data.get("prompt_tokens", 0))
            reqs.output_tokens.append(data.get("output_tokens", 0))
        elif kind == "request_cancelled":
            reqs.cancelled += 1
            reqs.cancel_stages[data.get("stage", "unknown")] += 1
            by_rid.pop(data.get("rid", ""), None)
        elif kind == "request_error":
            reqs.errored += 1
            reqs.error_kinds[data.get("error_kind", "unknown")] += 1
            by_rid.pop(data.get("rid", ""), None)
        elif kind == "request_invalid_schema":
            # Legacy event (pre-auditor harness); fold into the audit taxonomy.
            reqs.invalid_schema += 1
            reqs.invalid_schema_kinds[data.get("error_kind", "unknown")] += 1
            reqs.invalid_schema_by_workload[data.get("workload", "")] += 1
        elif kind == "audit_finding":
            check = data.get("check", "unknown")
            sev = data.get("severity", "warn")
            reqs.audit_by_check[check] += 1
            reqs.audit_by_check_severity[(check, sev)] += 1
            reqs.audit_by_workload[data.get("workload", "")] += 1
            if check not in reqs.audit_examples and data.get("detail"):
                reqs.audit_examples[check] = data["detail"]
        elif kind == "request_stall":
            reqs.stalls[data.get("stage", "unknown")] += 1
        elif kind == "metrics_probe":
            al = data.get("accept_len")
            if isinstance(al, (int, float)):
                reqs.accept_len.append(float(al))
        elif kind == "health_probe":
            ep = data.get("endpoint", "?")
            st = health[ep]
            st.probes += 1
            if data.get("ok"):
                st.ok += 1
            else:
                st.fail += 1
            lat = data.get("latency_s")
            if isinstance(lat, (int, float)):
                st.latencies_s.append(float(lat))
        elif kind == "health_transition":
            ep = data.get("endpoint", "?")
            health[ep].transitions.append(
                (ts, data.get("from", "?"), data.get("to", "?"))
            )
        elif kind == "rss_probe":
            if "error" in data:
                rss.errors += 1
                continue
            total_kb = data.get("total_kb")
            if isinstance(total_kb, int):
                rss.samples.append((ts, total_kb))
            per_pid = data.get("per_pid") or {}
            if per_pid:
                if not rss.first_per_pid:
                    rss.first_per_pid = dict(per_pid)
                rss.last_per_pid = dict(per_pid)

    # When a run is killed mid-flight (the common fault-injection case) the
    # `run_finished` event never lands, so fall back to the last observed
    # event timestamp rather than reporting a nonsensical negative duration.
    if run_start is None:
        duration = 0.0
    else:
        end = run_end if run_end is not None else last_ts
        duration = max(0.0, end - run_start)
    return reqs, dict(health), duration, rss


def format_summary(
    reqs: RequestStats,
    health: Dict[str, HealthStats],
    duration: float,
    rss: Optional[RssStats] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"=== stress summary (duration: {duration:.1f}s) ===")
    lines.append(
        f"requests: submitted={reqs.submitted} completed={reqs.completed} "
        f"cancelled={reqs.cancelled} errored={reqs.errored}"
    )
    if duration > 0:
        lines.append(f"throughput: {reqs.completed / duration:.2f} completed/s")
        tot_out = sum(reqs.output_tokens)
        if tot_out:
            lines.append(f"output tok/s: {tot_out / duration:.1f}")

    def _row(name: str, vals: List[float], unit: str = "s") -> str:
        if not vals:
            return f"  {name}: <none>"
        return (
            f"  {name}: p50={_pct(vals, 0.50):.3f}{unit} "
            f"p95={_pct(vals, 0.95):.3f}{unit} "
            f"p99={_pct(vals, 0.99):.3f}{unit} "
            f"max={max(vals):.3f}{unit}"
        )

    lines.append("latency:")
    lines.append(_row("TTFT", reqs.ttft_s))
    lines.append(_row("TPOT", reqs.tpot_s))
    lines.append(_row("E2E ", reqs.e2e_s))

    if reqs.error_kinds:
        lines.append("errors:")
        for k, v in reqs.error_kinds.most_common():
            lines.append(f"  {k}: {v}")
    if reqs.cancel_stages:
        lines.append("cancels by stage:")
        for k, v in reqs.cancel_stages.most_common():
            lines.append(f"  {k}: {v}")
    if reqs.invalid_schema:
        lines.append(f"invalid schema responses (legacy): {reqs.invalid_schema}")
        for k, v in reqs.invalid_schema_kinds.most_common():
            lines.append(f"  {k}: {v}")
        if reqs.invalid_schema_by_workload:
            lines.append("  by workload:")
            for k, v in reqs.invalid_schema_by_workload.most_common():
                lines.append(f"    {k}: {v}")

    if reqs.stalls:
        lines.append(f"stalls: {sum(reqs.stalls.values())}")
        for stage, v in reqs.stalls.most_common():
            lines.append(f"  {stage}: {v}")

    if reqs.audit_by_check:
        total = sum(reqs.audit_by_check.values())
        lines.append(f"audit findings: {total}")
        for check, v in reqs.audit_by_check.most_common():
            sevs = {
                s: c
                for (chk, s), c in reqs.audit_by_check_severity.items()
                if chk == check
            }
            sev_str = " ".join(f"{s}={c}" for s, c in sorted(sevs.items()))
            lines.append(f"  {check}: {v} ({sev_str})")
            if check in reqs.audit_examples:
                lines.append(f"    e.g. {reqs.audit_examples[check][:120]}")

    if reqs.accept_len:
        lines.append(
            f"spec accept_len: min={min(reqs.accept_len):.3f} "
            f"p50={_pct(reqs.accept_len, 0.50):.3f} "
            f"last={reqs.accept_len[-1]:.3f} (samples={len(reqs.accept_len)})"
        )

    for ep, st in sorted(health.items()):
        lines.append(f"health {ep}: probes={st.probes} ok={st.ok} fail={st.fail}")
        if st.latencies_s:
            lines.append(_row("  latency", st.latencies_s))
        if st.transitions:
            lines.append(f"  transitions: {len(st.transitions)}")
            for ts, frm, to in st.transitions:
                lines.append(f"    {ts:.2f} {frm} -> {to}")

    if rss is not None and rss.samples:
        first_ts, first_kb = rss.samples[0]
        last_ts, last_kb = rss.samples[-1]
        delta_kb = last_kb - first_kb
        span_s = max(1e-6, last_ts - first_ts)
        slope_kb_min = delta_kb / span_s * 60.0
        peak_kb = max(v for _, v in rss.samples)
        lines.append(
            f"rss: samples={len(rss.samples)} "
            f"first={first_kb / 1024:.1f}MB "
            f"last={last_kb / 1024:.1f}MB "
            f"peak={peak_kb / 1024:.1f}MB "
            f"delta={delta_kb / 1024:+.1f}MB "
            f"slope={slope_kb_min / 1024:+.2f}MB/min "
            f"errors={rss.errors}"
        )
        # Show the top 5 pids by growth so we can name the culprit process.
        if rss.first_per_pid and rss.last_per_pid:
            growth: List[Tuple[str, int, str]] = []
            for pid, entry in rss.last_per_pid.items():
                last_k = entry.get("rss_kb", 0)
                first_entry = rss.first_per_pid.get(pid)
                first_k = first_entry.get("rss_kb", 0) if first_entry else 0
                growth.append((pid, last_k - first_k, entry.get("comm", "?")))
            growth.sort(key=lambda x: x[1], reverse=True)
            top = [g for g in growth[:5] if g[1] > 0]
            if top:
                lines.append("  top growth by pid (delta MB):")
                for pid, delta, comm in top:
                    lines.append(
                        f"    pid={pid} comm={comm} delta={delta / 1024:+.1f}MB"
                    )
    return "\n".join(lines)
