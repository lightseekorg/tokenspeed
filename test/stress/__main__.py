"""CLI entrypoint.

Usage:
    python -m test.stress run --workload shared_prefix --duration 60
    python -m test.stress run --workload cancel_mix --arrival poisson --rate 20
    python -m test.stress summarize --events out/run.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
import time
from typing import List

from .audits import AuditConfig
from .audits import known as known_auditors
from .events import JsonlSink
from .launcher import LaunchConfig, ServerProcess
from .metrics import aggregate, format_summary
from .monitors.health import HealthMonitor
from .monitors.metrics import MetricsMonitor
from .monitors.rss import RssMonitor
from .runner import ArrivalSpec, BreakerSpec, run
from .workloads import get as get_workload
from .workloads import known as known_workloads


def _kv_pairs(items: List[str]) -> dict:
    """Parse `key=value` strings into a dict with int/float/bool coercion."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--workload-arg must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        vl = v.lower()
        if vl in {"true", "false"}:
            out[k] = vl == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _make_workload(name: str, args: dict):
    factory = get_workload(name)
    sig = inspect.signature(factory)
    unknown = set(args) - set(sig.parameters)
    if unknown:
        raise SystemExit(
            f"workload {name!r} has no parameters: {sorted(unknown)}; "
            f"accepted: {sorted(sig.parameters)}"
        )
    return factory(**args)


def _resolve_auditors(selected: List[str]) -> tuple:
    """Map --audit selections to registered auditor names (default: all)."""
    if not selected:
        return tuple(known_auditors())
    unknown = set(selected) - set(known_auditors())
    if unknown:
        raise SystemExit(
            f"unknown --audit {sorted(unknown)}; known: {known_auditors()}"
        )
    return tuple(selected)


def _parse_fail_on(items: List[str]) -> dict:
    """Parse `check=threshold` gate specs into {check: max_allowed_count}."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--fail-on-audit must be check=threshold, got: {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            raise SystemExit(f"--fail-on-audit threshold must be an int, got: {v!r}")
    return out


def _evaluate_gate(reqs, fail_on: dict) -> List[str]:
    """Return a list of gate-violation messages (empty => the run passes).

    Keys are auditor names (counted from audit findings) plus the special key
    ``stall`` (total streaming stalls). A check fails when its count exceeds
    the configured threshold.
    """
    violations: List[str] = []
    for check, limit in fail_on.items():
        if check == "stall":
            count = sum(reqs.stalls.values())
        else:
            count = reqs.audit_by_check.get(check, 0)
        if count > limit:
            violations.append(f"{check}: {count} > {limit}")
    return violations


async def _cmd_run(args: argparse.Namespace) -> int:
    out_dir = args.out or f"stress_out/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, "events.jsonl")
    sink = JsonlSink(events_path)

    server: "ServerProcess | None" = None
    if args.launch_cmd:
        server = ServerProcess(
            LaunchConfig(
                cmd=args.launch_cmd,
                base_url=args.base_url,
                log_path=os.path.join(out_dir, "server.log"),
                readiness_timeout_s=args.launch_timeout,
            )
        )

    # Initialized before the try so the finally can always stop them, even if
    # server.start()/wait_ready() or run() raises before they're created.
    monitor = None
    rss_monitor = None
    metrics_monitor = None
    try:
        if server is not None:
            server.start()
            await server.wait_ready()

        if not args.no_health:
            monitor = HealthMonitor(
                args.base_url,
                sink,
                interval_s=args.health_interval,
                request_timeout_s=args.health_timeout,
            )
            monitor.start()

        # RSS monitor: prefer explicit --server-pid, else fall back to the
        # launched server's pid. Skipped if neither is available or --no-rss.
        if not args.no_rss:
            rss_pid: "int | None" = args.server_pid
            if rss_pid is None and server is not None and server.running:
                rss_pid = server.pid
            if rss_pid is not None:
                rss_monitor = RssMonitor(
                    root_pid=rss_pid,
                    sink=sink,
                    interval_s=args.rss_interval,
                )
                rss_monitor.start()
                print(f"[stress] RSS monitor on pid={rss_pid}", flush=True)
            else:
                print(
                    "[stress] RSS monitor disabled: no --server-pid and "
                    "no --launch-cmd (pass --server-pid <N> to enable)",
                    flush=True,
                )

        # Spec-decode acceptance (and other server-wide gauges) via /metrics.
        if not args.no_spec_metrics:
            metrics_monitor = MetricsMonitor(
                args.base_url,
                sink,
                interval_s=args.metrics_interval,
                accept_len_min=args.accept_len_min,
            )
            metrics_monitor.start()

        audit_cfg = AuditConfig(
            enabled=() if args.no_audit else _resolve_auditors(args.audit),
            content_cap=args.audit_content_cap,
            stall_timeout_s=args.stall_timeout,
            global_stall_timeout_s=args.global_stall_timeout,
        )

        workload_args = _kv_pairs(args.workload_arg or [])
        workload = _make_workload(args.workload, workload_args)

        arrival = ArrivalSpec(
            kind=args.arrival,
            rate=args.rate,
            burst_size=args.burst_size,
            burst_period_s=args.burst_period,
            burst_gap_s=args.burst_gap,
            min_concurrency=args.min_concurrency,
            triangle_period_s=args.triangle_period,
            max_concurrency=args.max_concurrency,
            duration_s=args.duration,
            max_requests=args.max_requests,
        )

        breaker = BreakerSpec(
            enabled=not args.no_breaker,
            trip_after_consecutive_errors=args.breaker_threshold,
            open_duration_s=args.breaker_cool_s,
        )

        await run(
            base_url=args.base_url,
            model=args.model,
            workload=workload,
            arrival=arrival,
            sink=sink,
            timeout_s=args.request_timeout,
            breaker=breaker,
            audit_cfg=audit_cfg,
        )

    finally:
        # Stop monitors in finally so a crash/abort in run() can't leak the
        # poller tasks (which would keep emitting into a closed sink).
        if monitor is not None:
            await monitor.stop()
        if rss_monitor is not None:
            await rss_monitor.stop()
        if metrics_monitor is not None:
            await metrics_monitor.stop()
        sink.close()
        if server is not None:
            server.stop()

    reqs, health, duration, rss = aggregate(events_path)
    print(format_summary(reqs, health, duration, rss))
    print(f"\nevents: {events_path}")

    # Fatal conditions (e.g. a global decode wedge) always fail the run,
    # regardless of the opt-in gate.
    if reqs.aborted or reqs.fatal_findings:
        print("\nFATAL: run aborted on a server-wide failure (see above)")
        return 2

    # Opt-in gate: nonzero exit if any --fail-on-audit threshold is exceeded.
    violations = _evaluate_gate(reqs, _parse_fail_on(args.fail_on_audit or []))
    if violations:
        print("\nGATE FAILED:")
        for v in violations:
            print(f"  {v}")
        return 1
    return 0


def _cmd_summarize(args: argparse.Namespace) -> None:
    reqs, health, duration, rss = aggregate(args.events)
    print(format_summary(reqs, health, duration, rss))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test.stress")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a stress workload")
    p_run.add_argument("--base-url", default="http://127.0.0.1:22345")
    p_run.add_argument("--model", default="openai/gpt-oss-120b")
    p_run.add_argument(
        "--workload",
        required=True,
        choices=known_workloads(),
        help="Registered workload name",
    )
    p_run.add_argument(
        "--workload-arg",
        action="append",
        default=[],
        metavar="K=V",
        help="Workload kwarg; repeatable",
    )
    p_run.add_argument(
        "--arrival",
        default="constant",
        choices=["constant", "poisson", "bursty", "sawtooth", "burst"],
    )
    p_run.add_argument("--rate", type=float, default=10.0, help="Poisson rate req/s")
    p_run.add_argument("--burst-size", type=int, default=20)
    p_run.add_argument(
        "--burst-period", type=float, default=5.0, help="bursty: seconds between bursts"
    )
    p_run.add_argument(
        "--burst-gap",
        type=float,
        default=1.0,
        help="burst: sleep after drain-to-zero before the next burst",
    )
    p_run.add_argument(
        "--min-concurrency",
        type=int,
        default=1,
        help="Sawtooth arrival: floor of in-flight cap (default 1)",
    )
    p_run.add_argument(
        "--triangle-period",
        type=float,
        default=20.0,
        help="Sawtooth arrival: full up-then-down period in seconds (default 20)",
    )
    p_run.add_argument("--max-concurrency", type=int, default=64)
    p_run.add_argument(
        "--duration", type=float, default=60.0, help="seconds; 0 = unlimited"
    )
    p_run.add_argument("--max-requests", type=int, default=None)
    p_run.add_argument("--request-timeout", type=float, default=600.0)
    p_run.add_argument("--health-interval", type=float, default=1.0)
    p_run.add_argument("--health-timeout", type=float, default=30.0)
    p_run.add_argument("--no-health", action="store_true")
    p_run.add_argument(
        "--rss-interval",
        type=float,
        default=2.0,
        help="RSS monitor poll interval in seconds (default: 2)",
    )
    p_run.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="PID of the server process tree to sample for RSS. "
        "Ignored when --launch-cmd is used (we grab its pid).",
    )
    p_run.add_argument(
        "--no-rss",
        action="store_true",
        help="Disable the RSS monitor (on by default when a pid is known)",
    )
    p_run.add_argument(
        "--out", default=None, help="output dir (events.jsonl written here)"
    )
    p_run.add_argument(
        "--launch-cmd",
        default=None,
        help="Shell command to start the server (e.g. 'bash run-mm25.sh'). "
        "Harness waits for /health, runs workload, then kills the process group.",
    )
    p_run.add_argument(
        "--launch-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for /health=200 after launch (default: 900)",
    )
    p_run.add_argument(
        "--no-breaker",
        action="store_true",
        help="Disable circuit breaker (not recommended; dead server -> error storm)",
    )
    p_run.add_argument(
        "--breaker-threshold",
        type=int,
        default=32,
        help="Trip breaker after N consecutive request errors (default: 32)",
    )
    p_run.add_argument(
        "--breaker-cool-s",
        type=float,
        default=5.0,
        help="Pause dispatch for this many seconds once tripped (default: 5)",
    )

    # --- Output auditing -------------------------------------------------
    p_run.add_argument(
        "--audit",
        action="append",
        default=[],
        metavar="NAME",
        choices=known_auditors(),
        help="Enable a specific output auditor; repeatable. "
        f"Default: all ({', '.join(known_auditors())}).",
    )
    p_run.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable all per-response output auditors.",
    )
    p_run.add_argument(
        "--audit-content-cap",
        type=int,
        default=16384,
        help="Max chars of response content retained for auditing (default: 16384)",
    )
    p_run.add_argument(
        "--stall-timeout",
        type=float,
        default=20.0,
        help="Per-request (WARN): flag a single streaming request as stalled if "
        "no token arrives for this many seconds (default: 20; 0 disables). One "
        "request the engine may have lost track of.",
    )
    p_run.add_argument(
        "--global-stall-timeout",
        type=float,
        default=20.0,
        help="Server-wide (FATAL): if requests are in decode but NObody yields "
        "a token for this many seconds, the engine has hung — abort the run and "
        "exit nonzero (default: 20; 0 disables). Catches the wedge directly, "
        "well before downstream symptoms (e.g. SMG evicting the worker).",
    )
    p_run.add_argument(
        "--no-spec-metrics",
        action="store_true",
        help="Disable the /metrics monitor (spec-decode acceptance, etc.).",
    )
    p_run.add_argument(
        "--metrics-interval",
        type=float,
        default=5.0,
        help="/metrics scrape interval in seconds (default: 5)",
    )
    p_run.add_argument(
        "--accept-len-min",
        type=float,
        default=1.1,
        help="Raise a FATAL finding when spec-decode accept_len drops below "
        "this for consecutive windows (~1.0 = accepting zero draft tokens). "
        "Only active when spec decode is enabled. (default: 1.1)",
    )
    p_run.add_argument(
        "--fail-on-audit",
        action="append",
        default=[],
        metavar="CHECK=N",
        help="Exit nonzero if a check's finding count exceeds N. CHECK is an "
        "auditor name or 'stall'. Repeatable. e.g. length_consistency=0 stall=0",
    )

    p_sum = sub.add_parser("summarize", help="Summarize an existing events.jsonl")
    p_sum.add_argument("--events", required=True)

    ns = parser.parse_args(argv)
    if ns.cmd == "run":
        if ns.duration == 0:
            ns.duration = None
        return asyncio.run(_cmd_run(ns))
    if ns.cmd == "summarize":
        _cmd_summarize(ns)
        return 0
    parser.error("unknown command")
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
