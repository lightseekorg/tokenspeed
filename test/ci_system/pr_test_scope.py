"""Apply a reviewer's request to widen a pull request's tests.

The ``ci-full-tests`` label turns off every changed-path narrowing, and each
``CI-Extra-Changes: <path>`` line in the description counts as a changed path.
The workflows read both when they scan, so re-running a workflow after adding
the label or editing the description picks the request up.

usage: python pr_test_scope.py --repository <owner/name> --pull-request <number>
    --changed-files <file>; appends the extra paths to the file and prints
    ``true`` when the pull request asks for every test, ``false`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FULL_TESTS_LABEL = "ci-full-tests"
EXTRA_CHANGES_PREFIX = "CI-Extra-Changes:"


def extra_changes(description: str) -> list[str]:
    """The paths that ``CI-Extra-Changes:`` lines in a description name."""
    return [
        line.strip()[len(EXTRA_CHANGES_PREFIX) :].strip()
        for line in description.splitlines()
        if line.strip().startswith(EXTRA_CHANGES_PREFIX)
        and line.strip()[len(EXTRA_CHANGES_PREFIX) :].strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pull-request", required=True)
    parser.add_argument("--changed-files", type=Path, required=True)
    args = parser.parse_args()
    pull_request = json.loads(
        subprocess.check_output(
            ["gh", "api", f"/repos/{args.repository}/pulls/{args.pull_request}"]
        )
    )
    paths = extra_changes(pull_request["body"] or "")
    if paths:
        with args.changed_files.open("a", encoding="utf-8") as handle:
            handle.write("".join(f"{path}\n" for path in paths))
    labels = {label["name"] for label in pull_request["labels"]}
    print("true" if FULL_TESTS_LABEL in labels else "false")


if __name__ == "__main__":
    main()
