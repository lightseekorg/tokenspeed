#!/usr/bin/env python3
"""Summarise a pytest --report-log, including for runs that never finished.

pytest prints its tallies, failure tracebacks and short summary only when the
session ends. A rocJITsu abort leaves its xdist worker blocked in a HIP call
that no signal can interrupt, and the controller waits on that worker forever,
so the run is killed by a wall-clock timeout and reaches none of that output.
The report log is written one JSON object per event as the run proceeds, so it
still holds every result the run did produce, plus enough to name the test that
wedged: it is the one that started and never reached teardown.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

TRACEBACK_LINES = 40
INCOMPLETE_LIMIT = 20


def load(path):
    """Group the log's test reports by node, tolerating a truncated tail."""
    phases: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    collect_errors: list[dict] = []
    # A module skipped as a whole never yields test reports, and pytest counts
    # each one as a skip, so they have to come from the collection reports for
    # this tally to agree with pytest's own.
    collect_skips = 0
    finished = False

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A killed run can leave its last line half written.
                continue

            kind = event.get("$report_type")
            if kind == "TestReport":
                phases[event["nodeid"]][event["when"]] = event
            elif kind == "CollectReport":
                if event.get("outcome") == "failed":
                    collect_errors.append(event)
                elif event.get("outcome") == "skipped":
                    collect_skips += 1
            elif kind == "SessionFinish":
                finished = True

    return phases, collect_errors, collect_skips, finished


def outcome_of(node_phases):
    """Reduce one node's phase reports to a single outcome."""
    # Passes, failures and skips all report a teardown; only a test whose
    # process stopped responding is missing one.
    if "teardown" not in node_phases:
        return "incomplete"
    if any(report.get("outcome") == "failed" for report in node_phases.values()):
        return "failed"
    if any(report.get("outcome") == "skipped" for report in node_phases.values()):
        return "skipped"
    return "passed"


def traceback_lines(report):
    """Recover printable failure text from a serialised longrepr."""
    longrepr = report.get("longrepr")
    if not longrepr:
        return []
    if isinstance(longrepr, str):
        return longrepr.splitlines()

    lines: list[str] = []
    for entry in longrepr.get("reprtraceback", {}).get("reprentries", []):
        lines.extend(entry.get("data", {}).get("lines", []) or [])

    crash = longrepr.get("reprcrash") or {}
    if crash:
        message = (crash.get("message") or "").splitlines()
        lines.append(
            f"{crash.get('path', '?')}:{crash.get('lineno', '?')}: "
            f"{message[0] if message else ''}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_log")
    args = parser.parse_args()

    try:
        phases, collect_errors, collect_skips, finished = load(args.report_log)
    except FileNotFoundError:
        print(f"no report log at {args.report_log}: the run produced no results")
        return 1

    by_outcome = collections.Counter()
    failed, incomplete = [], []
    for nodeid, node_phases in phases.items():
        outcome = outcome_of(node_phases)
        by_outcome[outcome] += 1
        if outcome == "failed":
            failed.append((nodeid, node_phases))
        elif outcome == "incomplete":
            incomplete.append(nodeid)

    print("rocJITsu result digest")
    print(
        "  {passed} passed, {failed} failed, {skipped} skipped, "
        "{incomplete} never completed".format(
            passed=by_outcome["passed"],
            failed=by_outcome["failed"],
            skipped=by_outcome["skipped"] + collect_skips,
            incomplete=by_outcome["incomplete"],
        )
    )
    if not finished:
        print("  the session was killed before pytest could print its summary")

    for nodeid, node_phases in sorted(failed):
        print(f"\nFAILED {nodeid}")
        report = next(
            (
                node_phases[when]
                for when in ("call", "setup", "teardown")
                if node_phases.get(when, {}).get("outcome") == "failed"
            ),
            {},
        )
        lines = traceback_lines(report)
        for line in lines[-TRACEBACK_LINES:]:
            print(f"  {line}")
        if len(lines) > TRACEBACK_LINES:
            print(f"  ... {len(lines) - TRACEBACK_LINES} earlier traceback lines")

    for report in collect_errors:
        print(f"\nCOLLECTION ERROR {report.get('nodeid', '?')}")
        for line in traceback_lines(report)[-TRACEBACK_LINES:]:
            print(f"  {line}")

    if incomplete:
        # The emulator's abort message lands on whatever line was being written
        # at the time, so it points at an unrelated test. This is the reliable
        # way to name the test that stopped its worker.
        print(
            f"\n{len(incomplete)} test(s) started and never completed, "
            "which is what a rocJITsu abort or hang looks like:"
        )
        for nodeid in sorted(incomplete)[:INCOMPLETE_LIMIT]:
            print(f"  {nodeid}")
        if len(incomplete) > INCOMPLETE_LIMIT:
            print(f"  ... and {len(incomplete) - INCOMPLETE_LIMIT} more")

    if not phases:
        print("  the run recorded no test results at all")
        return 1
    return 1 if (failed or incomplete or collect_errors or not finished) else 0


if __name__ == "__main__":
    sys.exit(main())
