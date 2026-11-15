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

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
from pathlib import Path

import pytest
from ci_speed_dashboard import (
    GitHub,
    aggregate,
    fetch_jobs,
    fetch_missing_jobs,
    flakiness,
    load_cached_jobs,
    load_runs,
    prune_cache,
    write_json,
)

NOW = dt.datetime(2026, 11, 14, 18, 0, tzinfo=dt.timezone.utc)
SHA_A = "a" * 40
SHA_B = "b" * 40


def iso(minutes_before_now: float) -> str:
    t = NOW - dt.timedelta(minutes=minutes_before_now)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def run(run_id, created, updated, conclusion, sha=SHA_A, branch="feat", attempt=1):
    return {
        "id": run_id,
        "head_sha": sha,
        "head_branch": branch,
        "head_repo": "lightseekorg/tokenspeed",
        "created_at": iso(created),
        "updated_at": iso(updated),
        "status": "completed",
        "conclusion": conclusion,
        "run_attempt": attempt,
    }


def job(name, created, started, completed, conclusion="success", label="b200v2-1gpu"):
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "created_at": iso(created),
        "started_at": iso(started),
        "completed_at": iso(completed),
        "labels": [label],
        "runner_name": "runner-1",
        "nsteps": 5,
    }


def never_picked(name, created, completed, label):
    # GitHub reports started_at == created_at for a job that never got a runner.
    return job(name, created, created, completed, "cancelled", label) | {
        "runner_name": "",
        "nsteps": 0,
    }


B200_JOBS = [
    job("scan", 200, 200, 199, label="ubuntu-latest"),
    job("unit-test / ut-a (b200v2-1gpu)", 199, 198, 170),
    job("unit-test / ut-b (b200v2-1gpu)", 199, 198, 180),
    job("model-test / mt-x (b200v2-4gpu)", 170, 160, 140, label="b200v2-4gpu"),
    job("finish", 140, 140, 139, label="ubuntu-latest"),
]


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A cache laid out the way the fetcher writes it."""
    day = NOW.date()
    b200 = [
        # updated_at is bumped days later; the wall time must come from jobs.
        run(1, 200, 0, "success"),
        # A scan-only run: done within a minute.
        run(2, 100, 99.5, "success", sha=SHA_B),
    ]
    mi350 = [
        run(10, 200, 100, "success", attempt=2),
        run(11, 90, 89.7, "cancelled", sha=SHA_B),
    ]
    mi450 = [run(20, 200, 150, "success")]
    write_json(
        tmp_path / "runs/nvidia-b200-tests.yml/pr" / f"{day}.json", {"runs": b200}
    )
    write_json(
        tmp_path / "runs/amd-mi350-tests.yml/pr" / f"{day}.json", {"runs": mi350}
    )
    write_json(
        tmp_path / "runs/amd-mi450-tests.yml/pr" / f"{day}.json", {"runs": mi450}
    )
    write_json(tmp_path / "jobs/1.json", B200_JOBS)
    write_json(
        tmp_path / "jobs/2.json", [job("scan", 100, 100, 99.6, label="ubuntu-latest")]
    )
    write_json(
        tmp_path / "jobs/10.json",
        [
            job("scan", 200, 200, 199, label="ubuntu-latest"),
            job(
                "unit-test / ut-k (amd-mi355-1gpu)",
                199,
                199,
                120,
                "failure",
                "amd-mi355-1gpu",
            ),
            never_picked(
                "model-test / mt-y (amd-mi35x-8gpu-test)",
                120,
                110,
                "amd-mi35x-8gpu-test",
            ),
        ],
    )
    write_json(
        tmp_path / "jobs_latest/10_a2.json",
        [
            job("scan", 100, 100, 99, label="ubuntu-latest"),
            job(
                "unit-test / ut-k (amd-mi355-1gpu)", 99, 99, 60, label="amd-mi355-1gpu"
            ),
        ],
    )
    write_json(
        tmp_path / "jobs/20.json",
        [
            job("scan", 200, 200, 199, label="ubuntu-latest"),
            job(
                "unit-test / ut-z (amd-mi45x-cpu-test)",
                199,
                195,
                160,
                label="amd-mi45x-cpu-test",
            ),
        ],
    )
    return tmp_path


def build(cache: Path) -> dict:
    runs = load_runs(cache, NOW, 14)
    jobs1, latest = load_cached_jobs(cache, runs)
    return json.loads(json.dumps(aggregate(runs, jobs1, latest, NOW, 14)))


def test_wall_time_comes_from_jobs_and_scan_only_runs_are_excluded(cache: Path):
    stats = build(cache)["workflow_stats"]["B200"]["cur"]
    # Run 1 ended with its finish job at 139 min before now, created at 200.
    assert stats["n"] == 1
    assert stats["med"] == pytest.approx(61.0)
    assert stats["noop"] == 1


def test_never_picked_up_job_counts_as_queued_not_run(cache: Path):
    labels = {x["label"]: x for x in build(cache)["labels"]}
    big = labels["amd-mi35x-8gpu-test"]
    assert big["never"] == 1
    assert big["never_wait_med"] == pytest.approx(10.0)
    assert big["run_med"] is None
    assert labels["b200v2-1gpu"]["q_med"] == pytest.approx(1.0)
    assert labels["b200v2-1gpu"]["conc"] == 2


def test_critical_path_segments_add_up_to_the_last_model_test(cache: Path):
    p = build(cache)["critical_path"]["B200"]["periods"]["cur"]
    assert p["n"] == 1
    assert p["scan"]["med"] == pytest.approx(1.0)
    assert p["barrier"]["med"] == pytest.approx(29.0)
    assert p["mtq"]["med"] == pytest.approx(10.0)
    assert p["mtx"]["med"] == pytest.approx(20.0)
    assert p["total"]["med"] == pytest.approx(60.0)


def test_time_to_green_waits_for_the_rerun_of_the_last_required_workflow(cache: Path):
    ttg = build(cache)["ttg"]["cur"]
    # Head A: B200 done at -139, MI450 at -160, MI350 only after its rerun at
    # -60, all created at -200. Head B: B200 scan-only and MI350 cancelled, so
    # it never did real work and is left out.
    assert ttg["heads"] == 1
    assert ttg["never_green"] == 0
    assert ttg["n"] == 1
    assert ttg["med"] == pytest.approx(140.0)


def test_failure_that_passes_on_rerun_is_counted_as_flaky(cache: Path):
    runs = load_runs(cache, NOW, 14)
    jobs1, latest = load_cached_jobs(cache, runs)
    rows = flakiness({r["id"]: r for r in runs}, jobs1, latest, top=10, min_runs=1)
    row = next(x for x in rows if x["name"].startswith("unit-test / ut-k"))
    assert (row["fail"], row["rerun_pass"], row["rerun_fail"]) == (1, 1, 0)
    data = build(cache)
    assert data["reruns"] == {"n": 1, "final_success": 1, "latest_cached": 1}
    stats = data["workflow_stats"]["MI350"]["cur"]
    assert stats["cancelled"] == 1
    assert stats["superseded"] == 0


def test_superseded_cancellation(cache: Path):
    day = NOW.date()
    path = cache / "runs/amd-mi350-tests.yml/pr" / f"{day}.json"
    runs = json.loads(path.read_text())["runs"]
    runs.append(run(12, 30, 10, "success", sha="c" * 40))
    write_json(path, {"runs": runs})
    stats = build(cache)["workflow_stats"]["MI350"]["cur"]
    assert stats["superseded"] == 1


def test_prune_keeps_one_window_of_cache(cache: Path):
    old = NOW.date() - dt.timedelta(days=20)
    write_json(cache / "runs/nvidia-b200-tests.yml/pr" / f"{old}.json", {"runs": []})
    write_json(cache / "jobs/999.json", [])
    write_json(cache / "jobs_latest/999_a2.json", [])
    runs = load_runs(cache, NOW, 14)
    assert prune_cache(cache, runs, NOW, 14) == 3
    assert not (cache / "runs/nvidia-b200-tests.yml/pr" / f"{old}.json").exists()
    assert sorted(f.name for f in (cache / "jobs").iterdir()) == [
        "1.json",
        "10.json",
        "2.json",
        "20.json",
    ]
    assert [f.name for f in (cache / "jobs_latest").iterdir()] == ["10_a2.json"]


def test_a_failing_job_fetch_stops_the_rest_and_the_page_still_renders(
    cache: Path, monkeypatch
):
    # Twenty runs without cached jobs, more than the four workers.
    day_file = cache / "runs/nvidia-b200-tests.yml/pr" / f"{NOW.date()}.json"
    listed = json.loads(day_file.read_text())
    listed["runs"] += [run(100 + i, 200, 0, "success") for i in range(20)]
    write_json(day_file, listed)
    gh = GitHub("o/r", "token")
    calls = []

    def fail(path):
        calls.append(path)
        raise urllib.error.HTTPError(path, 502, "bad gateway", None, None)

    monkeypatch.setattr(gh, "get", fail)
    runs = load_runs(cache, NOW, 14)
    assert fetch_missing_jobs(gh, cache, runs, 10, 0) == 0
    # Queued fetches after the first failure are skipped (up to the 4 workers
    # already running may each try once).
    assert len(calls) <= 4
    assert build(cache)["workflow_stats"]


def test_a_job_list_cut_short_is_not_cached(cache: Path, monkeypatch):
    gh = GitHub("o/r", "token")
    pages = {1: {"total_count": 150, "jobs": [{"name": "a"}] * 100}}
    monkeypatch.setattr(
        gh, "get", lambda path: pages.get(int(path.rsplit("page=", 1)[1]))
    )
    # Page 2 is missing although total_count promises 150 jobs.
    assert fetch_jobs(gh, 1, 1) is None


def test_fetched_count_survives_a_later_failure(cache: Path, monkeypatch):
    day_file = cache / "runs/nvidia-b200-tests.yml/pr" / f"{NOW.date()}.json"
    listed = json.loads(day_file.read_text())
    listed["runs"] += [run(100 + i, 200 - i, 0, "success") for i in range(3)]
    write_json(day_file, listed)
    gh = GitHub("o/r", "token")
    done_job = [job("scan", 10, 10, 9, label="ubuntu-latest")]

    def get(path):
        if "/101/" in path:
            # Fail after the other workers have finished their fetches.
            time.sleep(0.3)
            raise urllib.error.HTTPError(path, 502, "bad gateway", None, None)
        return {"total_count": 1, "jobs": done_job}

    monkeypatch.setattr(gh, "get", get)
    monkeypatch.setattr("ci_speed_dashboard.slim_job", lambda j: j)
    runs = load_runs(cache, NOW, 14)
    # Runs 100 and 102 were fetched before run 101 failed.
    assert fetch_missing_jobs(gh, cache, runs, 10, 0) == 2
