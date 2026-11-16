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

"""Data-only inventory of runtime communication and allocation receipts."""

import hashlib
import json
import re
from pathlib import Path

_RECEIPT = re.compile(
    r"rank-(\d+)\.(communication|mnnvl-allocation-([0-9a-f]{64}))\.json"
)
_SHA = re.compile(r"[0-9a-f]{64}")


def communication_inventory(paths: list[Path], *, run_id: str) -> dict[str, dict]:
    """Fingerprint the exact same-run receipts and verify allocation ancestry."""
    receipts = {}
    snapshots = {}
    allocations = []
    for path in paths:
        resolved = path.resolve(strict=True)
        match = _RECEIPT.fullmatch(path.name)
        if match is None or resolved != path or str(path) in receipts:
            raise ValueError("Invalid or duplicate communication receipt path")
        rank, _, allocation_sha = match.groups()
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Communication receipt byte budget exceeded")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        body = json.loads(raw)
        schema = (
            "tokenspeed.communication_evidence/v1"
            if allocation_sha is None
            else "tokenspeed.mnnvl_allocation_evidence/v1"
        )
        config_sha = body.get("config_sha256")
        if (
            body.get("schema") != schema
            or body.get("run_id") != run_id
            or type(body.get("rank")) is not int
            or body["rank"] != int(rank)
            or not isinstance(config_sha, str)
            or _SHA.fullmatch(config_sha) is None
        ):
            raise ValueError("Communication receipt run/rank/config identity mismatch")
        if allocation_sha is None:
            snapshots[(path.parent, int(rank))] = (sha, config_sha)
        else:
            if allocation_sha != sha:
                raise ValueError("Allocation receipt filename SHA256 mismatch")
            allocations.append(
                (path, int(rank), body["communication_snapshot_sha256"], config_sha)
            )
        receipts[str(path)] = {"path": str(path), "bytes": len(raw), "sha256": sha}
    for path, rank, snapshot_sha, config_sha in allocations:
        if snapshots.get((path.parent, rank)) != (snapshot_sha, config_sha):
            raise ValueError(
                "Allocation receipt lacks its sealed communication snapshot"
            )
    return receipts
