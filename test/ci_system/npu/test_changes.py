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

"""CPU-only selection tests with fixtures for the external GitHub API."""

import copy
import subprocess
import unittest
from pathlib import Path

from changes import changed_paths, load_manifest, select_suite

ROOT = Path(__file__).resolve().parents[3]
REPOSITORY = "example/tokenspeed"
HEAD_SHA = "1" * 40
BASE_SHA = "2" * 40


def pr_event():
    return {
        "number": 11,
        "pull_request": {
            "head": {"sha": HEAD_SHA, "ref": "topic"},
            "base": {"sha": BASE_SHA, "ref": "main"},
        },
    }


class GitHubFixture:
    def __init__(self, files):
        self.before = {
            "head": {"sha": HEAD_SHA},
            "base": {"sha": BASE_SHA},
            "changed_files": len(files),
        }
        self.after = copy.deepcopy(self.before)
        self.pages = [files]
        self.comparison = {"files": files}
        self.metadata_reads = 0

    def __call__(self, endpoint, *, paginate):
        if endpoint.endswith("/files?per_page=100"):
            return copy.deepcopy(self.pages)
        if "/compare/" in endpoint:
            return copy.deepcopy(self.comparison)
        if endpoint == f"/repos/{REPOSITORY}/pulls/11":
            response = self.before if self.metadata_reads == 0 else self.after
            self.metadata_reads += 1
            return copy.deepcopy(response)
        raise ValueError(f"Unexpected fixture endpoint: {endpoint}")


def unavailable_api(endpoint, *, paginate):
    raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])


class ChangeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(ROOT / "test/ci/npu/manifest.json")

    def select_pr(self, event, api):
        paths = changed_paths(REPOSITORY, "pull_request", event, api=api)
        return select_suite(self.manifest, "pull_request", paths)

    def test_pr_selection_from_complete_api_files(self):
        for path, watched in (
            ("docs/topic.md", False),
            ("tokenspeed-kernel-npu/topic.py", True),
            (".github/workflows/run-ci-task-matrix.yml", True),
        ):
            with self.subTest(path=path):
                result = self.select_pr(pr_event(), GitHubFixture([{"filename": path}]))
                self.assertEqual(result["should_run"], watched)
                self.assertEqual(result["changed_files"], 1)
                self.assertEqual(result["matched_files"], int(watched))
                self.assertEqual(
                    result["reason"],
                    "watched-path-changed" if watched else "no-watched-changes",
                )

    def test_pr_with_old_test_merge_uses_event_identity_and_api_files(self):
        event = pr_event()
        # A stale test-merge identity must not replace the API's PR file list.
        event["pull_request"]["merge_commit_sha"] = "3" * 40
        result = self.select_pr(
            event, GitHubFixture([{"filename": "tokenspeed-kernel-npu/topic.py"}])
        )
        self.assertTrue(result["should_run"])
        self.assertEqual(
            result["matched_paths_sample"], ["tokenspeed-kernel-npu/topic.py"]
        )

    def test_pr_pagination_and_renames_check_both_paths(self):
        files = [
            {
                "filename": "docs/renamed.md",
                "previous_filename": "tokenspeed-kernel-npu/old.py",
            },
            {
                "filename": "tokenspeed-kernel-npu/new.py",
                "previous_filename": "docs/old.md",
            },
        ]
        api = GitHubFixture(files)
        api.pages = [[files[0]], [files[1]]]
        result = self.select_pr(pr_event(), api)
        self.assertTrue(result["should_run"])
        self.assertEqual(result["changed_files"], 4)
        self.assertEqual(result["matched_files"], 2)

    def test_pr_rejects_event_identity_changes(self):
        for ref in ("head", "base"):
            with self.subTest(ref=ref):
                api = GitHubFixture([])
                api.before[ref]["sha"] = "4" * 40
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    self.select_pr(pr_event(), api)

    def test_pr_rejects_changes_during_pagination(self):
        for ref in ("head", "base", "count"):
            with self.subTest(ref=ref):
                api = GitHubFixture([])
                if ref == "count":
                    api.after["changed_files"] = 1
                else:
                    api.after[ref]["sha"] = "4" * 40
                with self.assertRaisesRegex(ValueError, "changed while reading"):
                    self.select_pr(pr_event(), api)

    def test_pr_rejects_incomplete_or_duplicate_files(self):
        for scenario in ("missing", "duplicate", "invalid-pages"):
            with self.subTest(scenario=scenario):
                api = GitHubFixture([{"filename": "docs/topic.md"}])
                if scenario == "missing":
                    api.pages = [[]]
                elif scenario == "duplicate":
                    api.before["changed_files"] = api.after["changed_files"] = 2
                    api.pages = [[{"filename": "docs/topic.md"}] * 2]
                else:
                    api.pages = [{"filename": "docs/topic.md"}]
                with self.assertRaises(ValueError):
                    self.select_pr(pr_event(), api)

    def test_pr_file_limit(self):
        files = [{"filename": f"docs/{index}.md"} for index in range(3000)]
        api = GitHubFixture(files)
        api.pages = [files[index : index + 100] for index in range(0, len(files), 100)]
        self.assertFalse(self.select_pr(pr_event(), api)["should_run"])
        for count in (3001, -1, True):
            with self.subTest(count=count):
                api = GitHubFixture([])
                api.before["changed_files"] = count
                with self.assertRaisesRegex(ValueError, "completeness limit"):
                    self.select_pr(pr_event(), api)

    def test_pr_rejects_invalid_paths(self):
        for path in ("", "/absolute.py", "../escape.py"):
            for field in ("filename", "previous_filename"):
                with self.subTest(path=path, field=field):
                    item = {"filename": "docs/topic.md", field: path}
                    with self.assertRaisesRegex(ValueError, "Invalid changed path"):
                        self.select_pr(pr_event(), GitHubFixture([item]))

    def test_pr_rejects_invalid_number(self):
        for number in (0, -1, True):
            with self.subTest(number=number):
                event = pr_event()
                event["number"] = number
                with self.assertRaisesRegex(ValueError, "number"):
                    self.select_pr(event, GitHubFixture([]))

    def test_push_comparison_and_renames(self):
        api = GitHubFixture(
            [
                {
                    "filename": "docs/new.md",
                    "previous_filename": "tokenspeed-kernel-npu/old.py",
                }
            ]
        )
        paths = changed_paths(
            REPOSITORY, "push", {"before": BASE_SHA, "after": HEAD_SHA}, api=api
        )
        result = select_suite(self.manifest, "push", paths)
        self.assertTrue(result["should_run"])
        self.assertEqual(
            result["matched_paths_sample"], ["tokenspeed-kernel-npu/old.py"]
        )

    def test_push_comparison_limit(self):
        for count in (299, 300):
            with self.subTest(count=count):
                api = GitHubFixture(
                    [{"filename": f"docs/{i}.md"} for i in range(count)]
                )
                event = {"before": BASE_SHA, "after": HEAD_SHA}
                if count == 300:
                    with self.assertRaisesRegex(ValueError, "truncated"):
                        changed_paths(REPOSITORY, "push", event, api=api)
                else:
                    self.assertEqual(
                        len(changed_paths(REPOSITORY, "push", event, api=api)), count
                    )

    def test_push_rejects_zero_or_invalid_shas(self):
        for ref in ("before", "after"):
            for value in ("0" * 40, "invalid", None):
                with self.subTest(ref=ref, value=value):
                    event = {"before": BASE_SHA, "after": HEAD_SHA, ref: value}
                    with self.assertRaisesRegex(ValueError, "nonzero"):
                        changed_paths(REPOSITORY, "push", event, api=GitHubFixture([]))

    def test_manual_selection_does_not_require_api(self):
        paths = changed_paths(REPOSITORY, "workflow_dispatch", {}, api=unavailable_api)
        result = select_suite(self.manifest, "workflow_dispatch", paths)
        self.assertTrue(result["should_run"])
        self.assertEqual(result["reason"], "manual")

    def test_api_failure_is_not_a_skip(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.select_pr(pr_event(), unavailable_api)

    def test_invalid_repository_or_event(self):
        for repository in ("repo", "owner/repo?query", "owner/repo/extra"):
            with self.subTest(repository=repository):
                with self.assertRaisesRegex(ValueError, "Invalid repository"):
                    changed_paths(
                        repository, "pull_request", pr_event(), api=GitHubFixture([])
                    )
        with self.assertRaisesRegex(ValueError, "Unsupported event"):
            changed_paths(REPOSITORY, "release", {}, api=GitHubFixture([]))


if __name__ == "__main__":
    unittest.main()
