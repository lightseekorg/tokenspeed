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

"""Record the source actually checked out by a native CI job."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def write_source(work: Path, expected: str, config: str, runner: str) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", actual) or (
        re.fullmatch(r"[0-9a-f]{40}", expected) and expected != actual
    ):
        raise ValueError("CI source differs from selected commit")
    target = work / ".ci-artifacts/source.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(
        json.dumps({"source_sha": actual, "config": config, "runner": runner})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runner", required=True)
    args = parser.parse_args()
    try:
        write_source(args.work_dir, args.expected, args.config, args.runner)
    except (OSError, ValueError, subprocess.CalledProcessError):
        raise SystemExit("CI source verification failed.") from None
