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

"""``tokenspeed cache-budget``: check a deployment's declared Host memory offline.

Exit status: 0 when every node and container fits, 1 when one is
overcommitted, 2 when the manifest cannot be read or is invalid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenspeed.runtime.cache.host_budget import (
    ManifestInvalid,
    ManifestIssue,
    evaluate_manifest,
    invalid_report,
    parse_manifest_json,
)

EXIT_CODES = {"fits": 0, "overcommitted": 1, "invalid": 2}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "manifest",
        type=Path,
        help="The deployment's JSON Host memory manifest "
        "(docs/configuration/host-memory-budget.md).",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=("text", "json"),
        help="'text' prints a breakdown per node and container; 'json' prints "
        "the machine-readable report.",
    )


def build_report(manifest_path: Path) -> dict[str, Any]:
    """The report for the manifest at ``manifest_path``, valid or not."""
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return invalid_report(
            [
                ManifestIssue(
                    path="$", message=f"cannot read manifest {manifest_path}: {error}"
                )
            ]
        )
    try:
        return evaluate_manifest(parse_manifest_json(text))
    except ManifestInvalid as invalid:
        return invalid_report(invalid.issues)


def _row(indent: str, label: str, value: int) -> str:
    return f"{indent}{label}".ljust(44) + f"{value:>20,} B {value / 2**30:>11.2f} GiB"


def _domain_rows(
    lines: list[str],
    domain: dict[str, Any],
    *,
    limit_label: str,
    limit_key: str,
    indent: str,
) -> None:
    lines.append(_row(indent, limit_label, domain[limit_key]))
    if "system_reserved_bytes" in domain:
        lines.append(_row(indent, "system reserved", domain["system_reserved_bytes"]))
    lines.append(_row(indent, "declared allocations", domain["allocated_bytes"]))
    for kind, value in domain["by_kind"].items():
        lines.append(_row(indent, f"  kind {kind}", value))
    for job, value in domain["by_job"].items():
        lines.append(_row(indent, f"  job {job}", value))
    if domain["shared_bytes"]:
        lines.append(_row(indent, "  shared allocations", domain["shared_bytes"]))
    lines.append(_row(indent, "total", domain["total_bytes"]))
    lines.append(
        _row(indent, "free" if domain["fits"] else "OVER BY", abs(domain["free_bytes"]))
    )


def render_text(report: dict[str, Any]) -> str:
    """A human-readable breakdown of ``report``."""
    lines = [
        "TokenSpeed declared Host memory budget estimate",
        report["notice"],
        "",
    ]
    if report["status"] == "invalid":
        lines.append("Manifest is invalid:")
        lines.extend(f"  {e['path']}: {e['message']}" for e in report["errors"])
    containers = {c["id"]: c for c in report["containers"]}
    for node in report["nodes"]:
        verdict = "fits" if node["fits"] else "OVERCOMMITTED"
        lines.append(f"Node {node['id']}: {verdict}")
        _domain_rows(
            lines, node, limit_label="memory", limit_key="memory_bytes", indent="  "
        )
        for container_id in node["containers"]:
            container = containers[container_id]
            verdict = "fits" if container["fits"] else "OVERCOMMITTED"
            lines.append(f"  Container {container_id}: {verdict}")
            _domain_rows(
                lines,
                container,
                limit_label="limit",
                limit_key="memory_limit_bytes",
                indent="    ",
            )
        lines.append("")
    for warning in report["warnings"]:
        lines.append(f"warning: {warning}")
    lines.append(f"Status: {report['status']}")
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    """``report`` as deterministic JSON."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main(args: argparse.Namespace) -> None:
    report = build_report(args.manifest)
    if args.format == "json":
        print(render_json(report), end="")
    else:
        print(render_text(report), end="")
    raise SystemExit(EXIT_CODES[report["status"]])
