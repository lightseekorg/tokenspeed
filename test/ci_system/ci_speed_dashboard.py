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

"""Build the data file behind the CI speed dashboard (docs/public/ci/index.html).

Reads GitHub Actions runs and first-attempt jobs over a 14-day window, caches
everything that can no longer change, and writes aggregated metrics to one
JSON file that the static page renders.

usage: python ci_speed_dashboard.py --repo <owner/name> --cache <dir>
    --out <data.json> --max-new-runs <N>
Needs GITHUB_TOKEN with ``actions: read``. Uses only the standard library.
Proxies come from the usual HTTPS_PROXY environment variable.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc


@dataclass(frozen=True)
class Workflow:
    file: str
    key: str
    name: str
    # The workflow starts with a ``scan`` job that may select no tests.
    scan_gated: bool
    # The workflow ends with a ``finish`` job, the required status check.
    required: bool
    # Lower values get their job lists fetched first.
    job_priority: int


WORKFLOWS = (
    Workflow("nvidia-b200-tests.yml", "B200", "NVIDIA B200 Tests", True, True, 0),
    Workflow("amd-mi350-tests.yml", "MI350", "AMD MI350 Tests", True, True, 0),
    Workflow("amd-mi450-tests.yml", "MI450", "AMD MI450 Tests", True, True, 1),
    Workflow("nvidia-gb200-tests.yml", "GB200", "NVIDIA GB200 Tests", True, False, 1),
    Workflow("ascend-npu-tests.yml", "Ascend", "Ascend NPU Tests", True, False, 1),
    Workflow(
        "amd-kernel-benchmark.yml", "KBench", "AMD Kernel Benchmark", True, False, 1
    ),
    Workflow("lint.yml", "Lint", "Lint", False, False, 2),
    Workflow(
        "tokenspeed-engine-test.yml",
        "Rust",
        "TokenSpeed Engine Rust Test",
        False,
        False,
        3,
    ),
    Workflow(
        "mooncake-l3-test.yml", "Mooncake", "Mooncake L3 Master Test", False, False, 3
    ),
)
WF_BY_KEY = {w.key: w for w in WORKFLOWS}
CRITICAL_PATH_WORKFLOWS = ("B200", "MI350")
KEY_LABELS = (
    "amd-mi35x-8gpu-test",
    "amd-mi350-8gpu-bench",
    "amd-mi35x-4gpu-test",
    "amd-mi35x-2gpu-test",
    "amd-mi355-1gpu",
    "b200v2-1gpu",
    "b200v2-2gpu",
    "b200v2-4gpu",
    "b200v2-8gpu",
    "amd-mi45x-cpu-test",
    "tokenspeed-npu-arm64",
)
# A scan-gated run that ends this soon after creation only ran ``scan``.
NOOP_MINUTES = 3.0
# A run still not completed this long after creation counts as stuck.
STUCK_HOURS = 12
# Runs of the last few days can still change (reruns), so their lists refetch.
MUTABLE_DAYS = 2
HOSTED_LABEL = re.compile(r"^(ubuntu|windows|macos)-")
SEGMENTS = ("scan", "barrier", "mtq", "mtx")

RUN_FIELDS = (
    "id",
    "head_sha",
    "head_branch",
    "created_at",
    "updated_at",
    "status",
    "conclusion",
    "run_attempt",
)
JOB_FIELDS = (
    "name",
    "status",
    "conclusion",
    "created_at",
    "started_at",
    "completed_at",
    "labels",
    "runner_name",
)


# ----------------------------------------------------------------------------
# time and statistics helpers


def ts(value: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def minutes(a: dt.datetime, b: dt.datetime) -> float:
    return (b - a).total_seconds() / 60.0


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile; None for an empty list."""
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "med": rnd(percentile(values, 0.5)),
        "p90": rnd(percentile(values, 0.9)),
    }


def rnd(x: float | None) -> float | None:
    return None if x is None else round(x, 2)


def pt_day(t: dt.datetime) -> dt.date:
    return t.astimezone(PT).date()


def job_label(job: dict) -> str | None:
    labels = [x for x in job.get("labels") or [] if x != "self-hosted"]
    return labels[0] if labels else None


def never_picked_up(job: dict) -> bool:
    """A job that never got a runner has no runner name and no steps.

    GitHub sets its started_at to created_at, so it must be treated as queued
    until completed_at, never as running.
    """
    return not job.get("runner_name") and not job.get("nsteps")


def self_hosted_job_ran(jobs: list[dict]) -> bool:
    return any(
        j["conclusion"] != "skipped"
        and not never_picked_up(j)
        and not HOSTED_LABEL.match(job_label(j) or "ubuntu-")
        for j in jobs
    )


def jobs_end(jobs: list[dict]) -> dt.datetime | None:
    ends = [ts(j["completed_at"]) for j in jobs if j.get("completed_at")]
    return max(ends) if ends else None


# ----------------------------------------------------------------------------
# GitHub API and cache


class GitHub:
    """Minimal read-only REST client that tracks the remaining request budget."""

    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token
        self.calls = 0
        self.remaining: int | None = None

    def get(self, path: str) -> dict | None:
        url = f"https://api.github.com/repos/{self.repo}/{path}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tokenspeed-ci-speed-dashboard",
            },
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    self.calls += 1
                    rem = resp.headers.get("X-RateLimit-Remaining")
                    if rem is not None:
                        self.remaining = int(rem)
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                self.calls += 1
                rem = e.headers.get("X-RateLimit-Remaining") if e.headers else None
                if rem is not None:
                    self.remaining = int(rem)
                if e.code == 404:
                    return None
                if e.code in (403, 429) and self.remaining == 0:
                    raise BudgetExhausted() from e
                if attempt == 3 or e.code not in (403, 429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError):
                # Connection errors and read timeouts; HTTPError is handled above.
                if attempt == 3:
                    raise
            time.sleep(5 * 2**attempt)
        return None


class BudgetExhausted(Exception):
    pass


# Errors after which the run still renders from the cache. HTTPError is a
# URLError, and a read timeout raises TimeoutError.
FETCH_ERRORS = (BudgetExhausted, urllib.error.URLError, TimeoutError)


def slim_run(run: dict) -> dict:
    return {k: run.get(k) for k in RUN_FIELDS} | {
        "head_repo": (run.get("head_repository") or {}).get("full_name")
    }


def slim_job(job: dict) -> dict:
    return {k: job.get(k) for k in JOB_FIELDS} | {"nsteps": len(job.get("steps") or [])}


def list_runs(gh: GitHub, wf: Workflow, query: str, lo: str, hi: str) -> list[dict]:
    """Runs created in [lo, hi]; halves the range past the 1000-result cap."""
    out: list[dict] = []
    page = 1
    while True:
        r = gh.get(
            f"actions/workflows/{wf.file}/runs?{query}&created={lo}..{hi}"
            f"&per_page=100&page={page}"
        )
        if r is None:
            return out
        if page == 1 and r["total_count"] > 1000:
            a, b = ts(lo), ts(hi)
            mid = a + (b - a) / 2
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            return list_runs(gh, wf, query, lo, mid.strftime(fmt)) + list_runs(
                gh, wf, query, (mid + dt.timedelta(seconds=1)).strftime(fmt), hi
            )
        out += [slim_run(x) for x in r["workflow_runs"]]
        if len(r["workflow_runs"]) < 100:
            return out
        page += 1


EVENTS = (("pr", "event=pull_request"), ("main", "event=push&branch=main"))


def runs_path(cache: Path, wf: Workflow, ev: str, day: dt.date) -> Path:
    return cache / "runs" / wf.file / ev / f"{day}.json"


def job_path(cache: Path, run_id: int, attempt: int) -> Path:
    if attempt == 1:
        return cache / "jobs" / f"{run_id}.json"
    return cache / "jobs_latest" / f"{run_id}_a{attempt}.json"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    tmp.replace(path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_jobs(cache: Path, run_id: int, attempt: int) -> list[dict] | None:
    """Cached job list, or None when it is missing or not yet complete."""
    path = job_path(cache, run_id, attempt)
    jobs = read_json(path)
    if jobs is None:
        return None
    if any(j.get("status") != "completed" for j in jobs):
        return None
    return jobs


def fetch_run_lists(gh: GitHub, cache: Path, now: dt.datetime, days: int) -> None:
    start = (now - dt.timedelta(days=days)).date()
    for wf in WORKFLOWS:
        for ev, query in EVENTS:
            day = start
            while day <= now.date():
                path = runs_path(cache, wf, ev, day)
                cached = read_json(path)
                frozen = (
                    cached is not None
                    and day < now.date() - dt.timedelta(days=MUTABLE_DAYS)
                    and all(r["status"] == "completed" for r in cached["runs"])
                )
                if not frozen:
                    runs = list_runs(
                        gh, wf, query, f"{day}T00:00:00Z", f"{day}T23:59:59Z"
                    )
                    write_json(path, {"fetched": now.isoformat(), "runs": runs})
                day += dt.timedelta(days=1)


def load_runs(cache: Path, now: dt.datetime, days: int) -> list[dict]:
    lo = now - dt.timedelta(days=days)
    runs: dict[int, dict] = {}
    for wf in WORKFLOWS:
        for ev, _ in EVENTS:
            d = cache / "runs" / wf.file / ev
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.json")):
                for r in (read_json(f) or {}).get("runs", []):
                    if ts(r["created_at"]) >= lo:
                        runs[r["id"]] = r | {
                            "wf": wf.key,
                            "ev": ev,
                            "run_attempt": r.get("run_attempt") or 1,
                        }
    return list(runs.values())


def needs_jobs(run: dict) -> bool:
    """Whether the run's first-attempt job list carries information.

    A scan-gated first attempt that ended within NOOP_MINUTES only ran scan or
    was cancelled before a test job was queued, so it is skipped.
    """
    if run["status"] != "completed" or run["conclusion"] in (
        "action_required",
        "skipped",
    ):
        return False
    quick = minutes(ts(run["created_at"]), ts(run["updated_at"])) < NOOP_MINUTES
    return not (WF_BY_KEY[run["wf"]].scan_gated and quick and run["run_attempt"] == 1)


def fetch_jobs(gh: GitHub, run_id: int, attempt: int) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    while True:
        r = gh.get(
            f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100&page={page}"
        )
        if r is None:
            break
        jobs += [slim_job(j) for j in r["jobs"]]
        if len(jobs) >= r["total_count"] or not r["jobs"]:
            break
        page += 1
    return jobs


def fetch_missing_jobs(
    gh: GitHub, cache: Path, runs: list[dict], max_new: int, reserve: int
) -> int:
    """Fetch up to max_new job lists, GPU workflows first, newest first."""
    todo = []
    for r in runs:
        if not needs_jobs(r):
            continue
        age = -ts(r["created_at"]).timestamp()
        pri = WF_BY_KEY[r["wf"]].job_priority + (1 if r["ev"] == "main" else 0)
        if r["run_attempt"] > 1 and read_jobs(cache, r["id"], r["run_attempt"]) is None:
            todo.append(((pri, age), r["id"], r["run_attempt"]))
        if read_jobs(cache, r["id"], 1) is None:
            todo.append(((pri, age), r["id"], 1))
    todo.sort()
    todo = todo[:max_new]
    done = 0

    # The first failed fetch stops the rest, so an API outage does not retry
    # every queued run before the page renders from the cache.
    stop = threading.Event()

    def work(item):
        _, run_id, attempt = item
        if stop.is_set() or (gh.remaining is not None and gh.remaining < reserve):
            return False
        try:
            jobs = fetch_jobs(gh, run_id, attempt)
        except FETCH_ERRORS:
            stop.set()
            raise
        if jobs and all(j["status"] == "completed" for j in jobs):
            write_json(job_path(cache, run_id, attempt), jobs)
        return True

    ex = concurrent.futures.ThreadPoolExecutor(4)
    try:
        done = sum(ex.map(work, todo))
    except FETCH_ERRORS as e:
        print(
            f"job fetch stopped ({e!r}); rendering with the cached data",
            file=sys.stderr,
        )
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return done


def prune_cache(cache: Path, runs: list[dict], now: dt.datetime, days: int) -> int:
    """Delete run lists older than the fetch window and job lists of runs that
    left it, so each saved cache stays at about one window of data."""
    oldest = (now - dt.timedelta(days=days + 1)).date().isoformat()
    removed = 0
    for f in (cache / "runs").glob("*/*/*.json"):
        if f.stem < oldest:
            f.unlink()
            removed += 1
    keep = {str(r["id"]) for r in runs}
    for f in (cache / "jobs").glob("*.json"):
        if f.stem not in keep:
            f.unlink()
            removed += 1
    for f in (cache / "jobs_latest").glob("*.json"):
        if f.stem.split("_a")[0] not in keep:
            f.unlink()
            removed += 1
    return removed


def fetch_markers(gh: GitHub, numbers: list[int]) -> list[dict]:
    out = []
    for n in numbers:
        p = gh.get(f"pulls/{n}")
        if p:
            out.append(
                {
                    "number": n,
                    "title": p["title"],
                    "state": p["state"],
                    "draft": bool(p.get("draft")),
                    "merged_at": p.get("merged_at"),
                    "url": p["html_url"],
                }
            )
    return out


# ----------------------------------------------------------------------------
# aggregation


def annotate_runs(
    runs: list[dict], jobs1: dict[int, list], jobs_latest: dict[int, list]
) -> None:
    """Add end times, wall time and the scan-only flag to each run.

    A run's updated_at is not a reliable end time: bulk events bump it days
    after the run finished. End times come from job completed_at; updated_at is
    trusted only when it is within NOOP_MINUTES of creation.
    """
    for r in runs:
        r["end1"] = r["end_latest"] = r["wall"] = None
        r["noop"] = False
        r["wall_missing"] = False
        if r["status"] != "completed":
            continue
        created = ts(r["created_at"])
        quick = (
            minutes(created, ts(r["updated_at"])) < NOOP_MINUTES
            and r["run_attempt"] == 1
        )
        js = jobs1.get(r["id"])
        if js is not None:
            r["end1"] = jobs_end(js) or ts(r["updated_at"])
        elif quick:
            r["end1"] = ts(r["updated_at"])
        if r["run_attempt"] == 1:
            r["end_latest"] = r["end1"]
        elif r["id"] in jobs_latest:
            r["end_latest"] = jobs_end(jobs_latest[r["id"]])
        if r["conclusion"] in ("success", "failure"):
            if r["end1"] is not None:
                r["wall"] = minutes(created, r["end1"])
            else:
                r["wall_missing"] = True
        if WF_BY_KEY[r["wf"]].scan_gated:
            r["noop"] = quick or (js is not None and not self_hosted_job_ran(js))


def workflow_stats(runs: list[dict], now: dt.datetime, cut: dt.datetime) -> dict:
    by_branch = collections.defaultdict(list)
    for r in runs:
        if r["ev"] == "pr":
            by_branch[(r["wf"], r["head_repo"], r["head_branch"])].append(
                ts(r["created_at"])
            )
    out = {}
    for wf in WORKFLOWS:
        rs = [r for r in runs if r["wf"] == wf.key]
        stats = {}
        for per in ("cur", "prev"):
            sub = [
                r
                for r in rs
                if r["ev"] == "pr" and (ts(r["created_at"]) >= cut) == (per == "cur")
            ]
            base = [r for r in sub if r["conclusion"] != "action_required"]
            cancelled = [r for r in base if r["conclusion"] == "cancelled"]
            superseded = [
                r
                for r in cancelled
                if any(
                    t > ts(r["created_at"])
                    for t in by_branch[(r["wf"], r["head_repo"], r["head_branch"])]
                )
            ]
            unfinished = [r for r in base if r["status"] != "completed"]
            stuck = [
                r
                for r in unfinished
                if minutes(ts(r["created_at"]), now) > STUCK_HOURS * 60
            ]
            walls = [r["wall"] for r in sub if r["wall"] is not None and not r["noop"]]
            stats[per] = summary(walls) | {
                "runs": len(sub),
                "noop": sum(1 for r in sub if r["noop"]),
                "base": len(base),
                "cancelled": len(cancelled),
                "superseded": len(superseded),
                "stuck": len(stuck),
                "action_required": len(sub) - len(base),
            }
        main = [r for r in rs if r["ev"] == "main"]
        stats["main"] = summary(
            [r["wall"] for r in main if r["wall"] is not None and not r["noop"]]
        ) | {"runs": len(main)}
        out[wf.key] = stats
    return out


def daily_series(runs: list[dict], days: list[dt.date]) -> dict:
    buckets = collections.defaultdict(list)
    for r in runs:
        if r["wall"] is not None and not r["noop"]:
            buckets[(r["wf"], r["ev"], pt_day(ts(r["created_at"])))].append(r["wall"])
    out = {}
    for wf in WORKFLOWS:
        pr, main = [], []
        for d in days:
            v = buckets.get((wf.key, "pr", d), [])
            pr.append(
                [rnd(percentile(v, 0.5)), rnd(percentile(v, 0.9)), len(v)]
                if v
                else None
            )
            m = buckets.get((wf.key, "main", d), [])
            main.append([rnd(percentile(m, 0.5)), len(m)] if m else None)
        if any(pr) or any(main):
            out[wf.key] = {"pr": pr, "main": main}
    return out


def time_to_green(runs: list[dict], cut: dt.datetime) -> list[dict]:
    """One record per PR head commit that ran a required workflow.

    Labels, reopening and ready-for-review can start extra runs on the same
    commit; the newest run of each required workflow decides the check. The
    clock runs from the earliest deciding run's creation until the last one
    finishes (reruns included).
    """
    heads = collections.defaultdict(list)
    branch_heads = collections.defaultdict(list)
    for r in runs:
        if r["ev"] != "pr":
            continue
        branch_heads[(r["head_repo"], r["head_branch"])].append(
            (ts(r["created_at"]), r["head_sha"])
        )
        if WF_BY_KEY[r["wf"]].required and r["conclusion"] != "action_required":
            heads[(r["head_repo"], r["head_sha"])].append(r)
    out = []
    for (repo, sha), rs in heads.items():
        deciding = {}
        for r in rs:
            cur = deciding.get(r["wf"])
            if cur is None or ts(r["created_at"]) > ts(cur["created_at"]):
                deciding[r["wf"]] = r
        dec = list(deciding.values())
        first = min(ts(r["created_at"]) for r in dec)
        green = all(r["conclusion"] == "success" for r in dec)
        unknown = green and any(r["end_latest"] is None for r in dec)
        ttg = None
        if green and not unknown:
            ttg = minutes(first, max(r["end_latest"] for r in dec))
        out.append(
            {
                "per": "cur" if first >= cut else "prev",
                "ttg": ttg,
                "unknown": unknown,
                "pending": any(r["status"] != "completed" for r in dec),
                "substantive": any(
                    not r["noop"] for r in dec if r["status"] == "completed"
                ),
                "superseded": any(
                    t > first and s != sha
                    for t, s in branch_heads[(repo, dec[0]["head_branch"])]
                ),
            }
        )
    return out


def ttg_kpi(records: list[dict]) -> dict:
    """Time-to-green distribution of heads whose required runs did real work.

    Green heads without cached job data have no known end time; they count as
    green for the never-green share but are left out of the distribution.
    """
    out = {}
    for per in ("cur", "prev"):
        rs = [r for r in records if r["per"] == per and r["substantive"]]
        decided = [r for r in rs if not r["pending"]]
        never = [r for r in decided if r["ttg"] is None and not r["unknown"]]
        out[per] = summary([r["ttg"] for r in rs if r["ttg"] is not None]) | {
            "heads": len(decided),
            "never_green": len(never),
            "never_green_superseded": sum(1 for r in never if r["superseded"]),
            "unknown": sum(1 for r in rs if r["unknown"]),
        }
    all_green = [
        r["ttg"] for r in records if r["per"] == "cur" and r["ttg"] is not None
    ]
    out["all_heads_cur"] = summary(all_green)
    return out


def critical_path(
    runs: list[dict], jobs1: dict[int, list], days: list[dt.date], cut: dt.datetime
) -> dict:
    """Split successful B200/MI350 runs into scan, unit-test barrier, model-test
    queue and model-test execution, along the last-finishing model test."""
    segs = collections.defaultdict(list)
    for r in runs:
        if (
            r["wf"] not in CRITICAL_PATH_WORKFLOWS
            or r["ev"] != "pr"
            or r["conclusion"] != "success"
            or r["id"] not in jobs1
        ):
            continue
        js = jobs1[r["id"]]
        scan = [j for j in js if j["name"] == "scan" and j.get("completed_at")]
        ran = [j for j in js if j["conclusion"] != "skipped"]
        ut = [j for j in ran if j["name"].startswith("unit-test")]
        mt = [j for j in ran if j["name"].startswith("model-test")]
        if not scan or not ut or not mt or any(never_picked_up(j) for j in ut + mt):
            continue
        created = ts(r["created_at"])
        scan_end = ts(scan[0]["completed_at"])
        barrier_end = max(ts(j["completed_at"]) for j in ut)
        crit = max(mt, key=lambda j: ts(j["completed_at"]))
        start, end = ts(crit["started_at"]), ts(crit["completed_at"])
        segs[r["wf"]].append(
            {
                "scan": minutes(created, scan_end),
                "barrier": minutes(scan_end, barrier_end),
                "mtq": max(0.0, minutes(barrier_end, start)),
                "mtx": minutes(start, end),
                "total": minutes(created, end),
                "day": pt_day(created),
                "per": "cur" if created >= cut else "prev",
                "holder": max(ut, key=lambda j: ts(j["completed_at"]))["name"],
            }
        )
    out = {}
    for key in CRITICAL_PATH_WORKFLOWS:
        ss = segs.get(key, [])
        daily = []
        for d in days:
            sd = [s for s in ss if s["day"] == d]
            daily.append(
                [rnd(percentile([s[k] for s in sd], 0.5)) for k in SEGMENTS]
                + [len(sd), rnd(percentile([s["total"] for s in sd], 0.5))]
                if sd
                else None
            )
        periods = {}
        for per in ("cur", "prev"):
            sp = [s for s in ss if s["per"] == per]
            periods[per] = {
                k: summary([s[k] for s in sp]) for k in SEGMENTS + ("total",)
            } | {"n": len(sp)}
        holders = collections.Counter(s["holder"] for s in ss if s["per"] == "cur")
        out[key] = {
            "daily": daily,
            "periods": periods,
            "barrier_holders": holders.most_common(3),
        }
    return out


def runner_labels(jobs1: dict[int, list]) -> list[dict]:
    acc = collections.defaultdict(
        lambda: {"q": [], "run": [], "never": 0, "never_wait": [], "iv": []}
    )
    for js in jobs1.values():
        for j in js:
            label = job_label(j)
            if not label or HOSTED_LABEL.match(label) or j["conclusion"] == "skipped":
                continue
            a = acc[label]
            created, started, completed = (
                ts(j["created_at"]),
                ts(j["started_at"]),
                ts(j["completed_at"]),
            )
            if never_picked_up(j):
                a["never"] += 1
                if created and completed:
                    a["never_wait"].append(minutes(created, completed))
                continue
            if created and started:
                a["q"].append(max(0.0, minutes(created, started)))
            if started and completed:
                a["run"].append(minutes(started, completed))
                a["iv"].append((started, completed))
    out = []
    for label, a in acc.items():
        events = sorted(
            [(s, 1) for s, _ in a["iv"]] + [(e, -1) for _, e in a["iv"]],
            key=lambda x: (x[0], x[1]),
        )
        cur = peak = 0
        for _, step in events:
            cur += step
            peak = max(peak, cur)
        out.append(
            {
                "label": label,
                "key": label in KEY_LABELS,
                "jobs": len(a["q"]) + a["never"],
                "q_med": rnd(percentile(a["q"], 0.5)),
                "q_p90": rnd(percentile(a["q"], 0.9)),
                "q_max": rnd(max(a["q"])) if a["q"] else None,
                "run_med": rnd(percentile(a["run"], 0.5)),
                "conc": peak,
                "never": a["never"],
                "never_wait_med": rnd(percentile(a["never_wait"], 0.5)),
            }
        )
    seen = {x["label"] for x in out}
    out += [
        {"label": label, "key": True, "jobs": 0, "never": 0}
        for label in KEY_LABELS
        if label not in seen
    ]
    out.sort(key=lambda x: (not x["key"], -x["jobs"]))
    return out


def slowest_jobs(runs_by_id: dict, jobs1: dict[int, list], top: int) -> list[dict]:
    acc = collections.defaultdict(list)
    for rid, js in jobs1.items():
        r = runs_by_id[rid]
        if r["ev"] != "pr":
            continue
        for j in js:
            if never_picked_up(j) or j["conclusion"] not in ("success", "failure"):
                continue
            if HOSTED_LABEL.match(job_label(j) or ""):
                continue
            acc[(r["wf"], j["name"])].append(
                minutes(ts(j["started_at"]), ts(j["completed_at"]))
            )
    rows = [
        {"wf": wf, "name": name, "max": rnd(max(v))} | summary(v)
        for (wf, name), v in acc.items()
        if len(v) >= 3
    ]
    rows.sort(key=lambda x: -x["med"])
    return rows[:top]


def flakiness(
    runs_by_id: dict,
    jobs1: dict[int, list],
    jobs_latest: dict[int, list],
    top: int,
    min_runs: int,
) -> list[dict]:
    acc = collections.defaultdict(
        lambda: {
            "ok": 0,
            "fail": 0,
            "rerun_pass": 0,
            "rerun_fail": 0,
            "main_ok": 0,
            "main_fail": 0,
        }
    )
    for rid, js in jobs1.items():
        r = runs_by_id[rid]
        latest = None
        if rid in jobs_latest:
            latest = {j["name"]: j["conclusion"] for j in jobs_latest[rid]}
        for j in js:
            if j["conclusion"] not in ("success", "failure") or j["name"] == "finish":
                continue
            a = acc[(r["wf"], j["name"])]
            failed = j["conclusion"] == "failure"
            if r["ev"] == "main":
                a["main_fail" if failed else "main_ok"] += 1
                continue
            a["fail" if failed else "ok"] += 1
            if failed and latest is not None and j["name"] in latest:
                a["rerun_pass" if latest[j["name"]] == "success" else "rerun_fail"] += 1
    rows = [
        {"wf": wf, "name": name} | a
        for (wf, name), a in acc.items()
        if a["fail"] > 0 and a["ok"] + a["fail"] >= min_runs
    ]
    rows.sort(key=lambda x: (-x["fail"], x["name"]))
    return rows[:top]


def aggregate(
    runs: list[dict],
    jobs1: dict[int, list],
    jobs_latest: dict[int, list],
    now: dt.datetime,
    days: int,
) -> dict:
    """All dashboard metrics from runs and cached job lists; no network."""
    lo = now - dt.timedelta(days=days)
    cut = now - dt.timedelta(days=7)
    runs = [r for r in runs if ts(r["created_at"]) >= lo]
    runs_by_id = {r["id"]: r for r in runs}
    jobs1 = {k: v for k, v in jobs1.items() if k in runs_by_id}
    jobs_latest = {k: v for k, v in jobs_latest.items() if k in runs_by_id}
    annotate_runs(runs, jobs1, jobs_latest)
    d0 = pt_day(lo)
    day_list = [d0 + dt.timedelta(days=i) for i in range((pt_day(now) - d0).days + 1)]

    check_total = check_disagree = 0
    for rid, js in jobs1.items():
        r = runs_by_id[rid]
        if (
            WF_BY_KEY[r["wf"]].scan_gated
            and r["run_attempt"] == 1
            and r["wall"] is not None
        ):
            quick = minutes(ts(r["created_at"]), ts(r["updated_at"])) < NOOP_MINUTES
            check_total += 1
            check_disagree += self_hosted_job_ran(js) == quick

    reruns = [
        r
        for r in runs
        if r["ev"] == "pr" and r["run_attempt"] > 1 and r["status"] == "completed"
    ]
    return {
        "schema": 1,
        "generated_at": now.isoformat(),
        "window": {
            "start": lo.isoformat(),
            "end": now.isoformat(),
            "cut": cut.isoformat(),
        },
        "days": [d.isoformat() for d in day_list],
        "params": {"noop_minutes": NOOP_MINUTES, "stuck_hours": STUCK_HOURS},
        "workflows": [
            {"key": w.key, "name": w.name, "file": w.file, "required": w.required}
            for w in WORKFLOWS
        ],
        "counts": {
            "pr_runs": sum(1 for r in runs if r["ev"] == "pr"),
            "main_runs": sum(1 for r in runs if r["ev"] == "main"),
            "wall_missing": sum(1 for r in runs if r["wall_missing"]),
        },
        "ttg": ttg_kpi(time_to_green(runs, cut)),
        "workflow_stats": workflow_stats(runs, now, cut),
        "daily": daily_series(runs, day_list),
        "critical_path": critical_path(runs, jobs1, day_list, cut),
        "labels": runner_labels(jobs1),
        "slowest": slowest_jobs(runs_by_id, jobs1, 15),
        "flaky": flakiness(runs_by_id, jobs1, jobs_latest, top=25, min_runs=5),
        "reruns": {
            "n": len(reruns),
            "final_success": sum(1 for r in reruns if r["conclusion"] == "success"),
            "latest_cached": sum(1 for r in reruns if r["id"] in jobs_latest),
        },
        "coverage": {
            w.key: [
                sum(
                    1
                    for r in runs
                    if r["wf"] == w.key and needs_jobs(r) and r["id"] in jobs1
                ),
                sum(1 for r in runs if r["wf"] == w.key and needs_jobs(r)),
            ]
            for w in WORKFLOWS
        },
        "noop_check": [check_disagree, check_total],
    }


def load_cached_jobs(cache: Path, runs: list[dict]) -> tuple[dict, dict]:
    jobs1, latest = {}, {}
    for r in runs:
        js = read_jobs(cache, r["id"], 1)
        if js is not None:
            jobs1[r["id"]] = js
        if r["run_attempt"] > 1:
            lj = read_jobs(cache, r["id"], r["run_attempt"])
            if lj is not None:
                latest[r["id"]] = lj
    return jobs1, latest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--max-new-runs",
        required=True,
        type=int,
        help="job lists to fetch in this invocation (each costs one request)",
    )
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument(
        "--reserve",
        type=int,
        default=50,
        help="stop fetching job lists when this few requests remain",
    )
    ap.add_argument(
        "--marker-prs",
        default="",
        help="comma-separated PR numbers to mark on the charts when merged",
    )
    ap.add_argument(
        "--offline", action="store_true", help="aggregate the cache without network"
    )
    args = ap.parse_args()

    now = dt.datetime.now(UTC).replace(microsecond=0)
    markers_path = args.cache / "markers.json"
    fetched_new = 0
    if args.offline:
        stamp = read_json(args.cache / "fetched.json")
        if stamp:
            now = dt.datetime.fromisoformat(stamp["at"])
    else:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            sys.exit("GITHUB_TOKEN is not set")
        gh = GitHub(args.repo, token)
        try:
            fetch_run_lists(gh, args.cache, now, args.days + 1)
            numbers = [int(x) for x in args.marker_prs.split(",") if x.strip()]
            write_json(markers_path, fetch_markers(gh, numbers))
        except FETCH_ERRORS as e:
            print(f"listing runs stopped ({e!r})", file=sys.stderr)
        runs = load_runs(args.cache, now, args.days)
        fetched_new = fetch_missing_jobs(
            gh, args.cache, runs, args.max_new_runs, args.reserve
        )
        write_json(args.cache / "fetched.json", {"at": now.isoformat()})
        print(
            f"{gh.calls} API calls, {fetched_new} job lists fetched, "
            f"{gh.remaining} requests left",
            file=sys.stderr,
        )
    runs = load_runs(args.cache, now, args.days)
    if not args.offline:
        removed = prune_cache(args.cache, runs, now, args.days)
        print(f"pruned {removed} cache files outside the window", file=sys.stderr)
    jobs1, latest = load_cached_jobs(args.cache, runs)
    data = aggregate(runs, jobs1, latest, now, args.days)
    data["repo"] = args.repo
    data["markers"] = read_json(markers_path) or []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
