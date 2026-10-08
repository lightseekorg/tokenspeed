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

"""Small, versioned records shared by the plan publisher and PR controller."""

import base64
import json
import re

BOT = "lightseek-bot"
BOT_ID = 243258330
REPO = "lightseekorg/tokenspeed"
COMMAND = re.compile(r"\s*@lightseek-bot[ \t]+(watch|fix)\s*", re.IGNORECASE)
SHA = re.compile(r"[0-9a-f]{40}")
NATIVE_CHECKS = {
    "scheduler-cpp-test.yml": {
        "label": "Scheduler C++",
        "reason": "Cache classification, capacity and admission regressions",
        "step": "Run scheduler C++ tests",
        "job": "test",
    },
    "scheduler-python-test.yml": {
        "label": "Scheduler Python",
        "reason": "Python bindings, cache capacity and admission regressions",
        "step": "Run scheduler Python tests",
        "job": "test",
    },
    "nvidia-kernel-library-tests.yml": {
        "label": "NVIDIA kernel libraries",
        "reason": "Native MLA numerics, CUDA Graph replay and kernel regressions",
        "step": "Run native library unit tests",
        "job": "native-libraries",
        "target": "Native GPU CI",
    },
}
STATUSES = {"passed", "waiting", "failed", "missing", "blocked"}


def marker(kind: str, data: dict) -> str:
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode()
    ).decode()
    return f"\n<!-- pr-ci-{kind}: {encoded} -->\n"


def record(comment: dict, kind: str) -> dict | None:
    if comment["user"]["login"] != BOT or comment["user"]["id"] != BOT_ID:
        return None
    matches = re.findall(
        rf"<!-- pr-ci-{kind}: ([A-Za-z0-9+/=]+) -->", comment["body"] or ""
    )
    if len(matches) != 1 or len(matches[0]) > 40000:
        return None
    try:
        data = json.loads(base64.b64decode(matches[0], validate=True))
    except (ValueError, UnicodeError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or data.get("repository") != REPO
        or type(data.get("pr")) is not int
        or not all(
            isinstance(data.get(k), str) and SHA.fullmatch(data[k])
            for k in ("head", "base")
        )
    ):
        return None
    common = {"version", "repository", "pr", "head", "base"}
    if kind == "plan":
        if (
            set(data) != common | {"run", "tests", "tasks"}
            or type(data.get("run")) is not int
            or not isinstance(data.get("tests"), list)
            or not all(
                isinstance(p, str) and re.fullmatch(r"[A-Za-z0-9_./-]+", p)
                for p in data["tests"]
            )
        ):
            return None
    elif kind == "assist":
        required = common | {
            "command",
            "action",
            "phase",
            "tasks",
            "statuses",
            "run_ids",
            "conflicts",
            "since",
            "submitted",
        }
        if (
            not required.issubset(data)
            or set(data)
            - required
            - {"candidate", "repair_run", "plan_refresh", "native_checks"}
            or type(data.get("command")) is not int
            or data.get("action") not in {"watch", "fix"}
            or data.get("phase")
            not in {
                "watching",
                "waiting-plan",
                "repairing",
                "validating",
                "done",
                "manual",
                "stale",
                "promoted",
            }
        ):
            return None
        if "plan_refresh" in data and (
            type(data["plan_refresh"]) is not int or data["plan_refresh"] < 1
        ):
            return None
        if (
            type(data["conflicts"]) is not bool
            or not isinstance(data["statuses"], list)
            or any(s not in STATUSES for s in data["statuses"])
        ):
            return None
        checks = data.get("native_checks", [])
        if not isinstance(checks, list) or len(checks) > len(NATIVE_CHECKS):
            return None
        for check in checks:
            if (
                not isinstance(check, dict)
                or set(check) != {"workflow", "status", "run"}
                or not isinstance(check["workflow"], str)
                or check["workflow"] not in NATIVE_CHECKS
                or not isinstance(check["status"], str)
                or check["status"] not in STATUSES
                or type(check["run"]) is not int
                or check["run"] < 0
            ):
                return None
        if len({c["workflow"] for c in checks}) != len(checks):
            return None
        if (
            type(data["since"]) is not int
            or not isinstance(data["submitted"], list)
            or not all(isinstance(t, str) for t in data["submitted"])
        ):
            return None
        if not isinstance(data["run_ids"], dict) or any(
            not re.fullmatch(r"[A-Za-z0-9_./@-]+", k) or type(v) is not int or v < 1
            for k, v in data["run_ids"].items()
        ):
            return None
        if "candidate" in data:
            c = data["candidate"]
            if (
                not isinstance(c, dict)
                or set(c) != {"patch", "validation", "tree", "branch"}
                or not all(
                    isinstance(c[k], str) and SHA.fullmatch(c[k])
                    for k in ("patch", "validation", "tree")
                )
                or c["branch"] != f"bot/pr-ci-assist-{data['pr']}-{data['command']}"
            ):
                return None
    else:
        return None
    if not isinstance(data.get("tasks"), list) or len(data["tasks"]) > 8:
        return None
    for task in data["tasks"]:
        if (
            not isinstance(task, dict)
            or set(task) != {"config", "runner", "cluster"}
            or not all(
                isinstance(v, str)
                and len(v) <= 180
                and re.fullmatch(r"[A-Za-z0-9_./-]*", v)
                for v in task.values()
            )
            or not task["config"].startswith("test/ci/")
            or task["cluster"] not in {"", "gb200", "gb300"}
        ):
            return None
    if (
        kind == "assist"
        and "repair_run" in data
        and (type(data["repair_run"]) is not int or data["repair_run"] < 1)
    ):
        return None
    if kind == "assist":
        sources = {data["head"]} | (
            {data["candidate"]["validation"]} if "candidate" in data else set()
        )
        titles = set()
        for source in sources:
            for task in data["tasks"]:
                if task["cluster"]:
                    for cluster in (
                        {"gb200", "gb300"}
                        if task["cluster"] == "gb200"
                        else {task["cluster"]}
                    ):
                        titles.add(
                            f"Slurm {source} | {task['config']} | {task['runner']} | {cluster}"
                        )
                else:
                    titles.add(f"K8s {source} | {task['config']} | {task['runner']}")
        if any(t not in titles for t in data["submitted"]):
            return None
    if kind == "assist" and not set(data["run_ids"]).issubset(
        {f"{t['config']}@{t['runner']}@{t['cluster']}" for t in data["tasks"]}
    ):
        return None
    return data
