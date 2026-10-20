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

"""CPU-only change-selection tests using real Git histories."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from ci.npu_ops_test.changes import load_manifest, select_suite

ROOT = Path(__file__).resolve().parents[2]


def git(repo, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def commit_file(repo, path):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{path}\n")
    git(repo, "add", "--", path)
    git(repo, "commit", "-m", f"Add {path}")
    return git(repo, "rev-parse", "HEAD")


def make_pr(repo, path, base_state):
    git(repo, "init", "--initial-branch=main")
    git(repo, "config", "user.name", "CI Test")
    git(repo, "config", "user.email", "ci-test@example.com")
    base = commit_file(repo, "seed.txt")
    if base_state != "unchanged":
        commit_file(repo, "tokenspeed-kernel-npu/upstream.py")
    topic_base = (
        git(repo, "rev-parse", "HEAD") if base_state == "updated-head" else base
    )
    git(repo, "switch", "-c", "topic", topic_base)
    head = commit_file(repo, path)
    git(repo, "switch", "main")
    git(repo, "merge", "--no-ff", "-m", "PR merge", "topic")
    merge = git(repo, "rev-parse", "HEAD")
    event = {
        "number": 1,
        "pull_request": {
            "base": {"sha": base, "ref": "main"},
            "head": {"sha": head, "ref": "topic"},
        },
    }
    return event, merge


def select_pr(repo, event, merge):
    return select_suite(
        repo,
        load_manifest(ROOT / "ci/npu_ops_test/manifest.json"),
        "pull_request",
        event,
        merge,
        "refs/pull/1/merge",
    )


class ChangeSelectionTest(unittest.TestCase):
    def test_pr_selection_with_base_updates(self):
        for base_state in ("unchanged", "advanced", "updated-head"):
            for path, watched in (
                ("docs/topic.md", False),
                ("tokenspeed-kernel-npu/topic.py", True),
            ):
                with self.subTest(base_state=base_state, path=path):
                    with tempfile.TemporaryDirectory() as directory:
                        repo = Path(directory)
                        event, merge = make_pr(repo, path, base_state)
                        result = select_pr(repo, event, merge)
                        self.assertEqual(
                            result,
                            {
                                "should_run": watched,
                                "reason": (
                                    "watched-path-changed"
                                    if watched
                                    else "no-watched-changes"
                                ),
                                "changed_files": 1,
                                "matched_files": int(watched),
                                "matched_paths_sample": [path] if watched else [],
                            },
                        )

    def test_pr_rejects_wrong_head_or_non_merge_checkout(self):
        for mismatch in ("head", "non-merge"):
            with self.subTest(mismatch=mismatch):
                with tempfile.TemporaryDirectory() as directory:
                    repo = Path(directory)
                    event, merge = make_pr(repo, "docs/topic.md", "advanced")
                    if mismatch == "head":
                        event["pull_request"]["head"]["sha"] = event["pull_request"][
                            "base"
                        ]["sha"]
                    else:
                        merge = event["pull_request"]["head"]["sha"]
                        git(repo, "checkout", "--detach", merge)
                    with self.assertRaisesRegex(ValueError, "event head"):
                        select_pr(repo, event, merge)

    def test_pr_rejects_divergent_base(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            event, merge = make_pr(repo, "docs/topic.md", "advanced")
            git(repo, "switch", "-c", "diverged", event["pull_request"]["base"]["sha"])
            event["pull_request"]["base"]["sha"] = commit_file(repo, "diverged.txt")
            git(repo, "checkout", "--detach", merge)
            with self.assertRaisesRegex(ValueError, "descend from event base"):
                select_pr(repo, event, merge)


if __name__ == "__main__":
    unittest.main()
