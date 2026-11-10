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

"""Validated graph-step measurements and a versioned local calibration cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

STARTUP_CONTEXT_ENDPOINTS = (8192, 32768, 131072, 524288)
STARTUP_WARMUP_REPLAYS = 2
STARTUP_MEASURED_REPLAYS = 8


def calibration_source_fingerprint(roots: Sequence[Path]) -> str:
    """Fingerprint installed Python code and native build artifacts once at boot.

    Python contents cover editable installs; native artifact size/mtime covers
    rebuilt extensions without reading multi-gigabyte libraries into memory.
    Missing roots are an error, not a reusable empty software identity.
    """
    digest = hashlib.sha256()
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise ValueError(f"calibration source root does not exist: {root}")
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.suffix in (".py", ".so", ".h", ".cuh", ".cu")
                or path.name == "ts-host"
            )
        )
        if not files:
            raise ValueError(f"calibration source root is empty: {root}")
        for path in files:
            digest.update(f"{index}/{path.relative_to(root)}\0".encode())
            if path.suffix == ".so" or path.name == "ts-host":
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            else:
                digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifyCalibration:
    batch_size: int
    min_context: int
    max_context: int
    verify_width: int
    native_step_ms: float
    adaptive_step_ms: tuple[tuple[int, float], ...]
    sample_count: int

    def __post_init__(self) -> None:
        for value in (
            self.batch_size,
            self.min_context,
            self.max_context,
            self.verify_width,
            self.sample_count,
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "calibration dimensions and sample count must be positive integers"
                )
        if self.max_context < self.min_context or self.verify_width < 2:
            raise ValueError("invalid calibration context range or verify width")
        if not math.isfinite(self.native_step_ms) or self.native_step_ms <= 0:
            raise ValueError("native step cost must be finite and positive")
        seen = set()
        for q, cost in self.adaptive_step_ms:
            if (
                type(q) is not int
                or q in seen
                or not self.batch_size <= q < self.batch_size * self.verify_width
            ):
                raise ValueError("invalid or duplicate adaptive query capacity")
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("adaptive step cost must be finite and positive")
            seen.add(q)

    @classmethod
    def measure(
        cls,
        *,
        batch_size: int,
        min_context: int,
        max_context: int,
        verify_width: int,
        native_samples: Sequence[float],
        adaptive_samples: Mapping[int, Sequence[float]],
        minimum_samples: int,
    ) -> VerifyCalibration:
        """Reduce warmed complete-step timings; the caller synchronizes ranks.

        Every sample should use the maximum rank time for the same step. Warmup
        samples must already be excluded. Kernel-time sums are not step timings.
        """
        if type(minimum_samples) is not int or minimum_samples < 1:
            raise ValueError("minimum_samples must be a positive integer")
        samples = [native_samples, *adaptive_samples.values()]
        if any(len(values) < minimum_samples for values in samples):
            raise ValueError("insufficient calibration samples")
        if any(not math.isfinite(v) or v <= 0 for values in samples for v in values):
            raise ValueError("calibration samples must be finite and positive")
        return cls(
            batch_size,
            min_context,
            max_context,
            verify_width,
            median(native_samples),
            tuple(
                (q, median(values)) for q, values in sorted(adaptive_samples.items())
            ),
            min(map(len, samples)),
        )

    def covers(self, *, batch_size: int, min_context: int, max_context: int) -> bool:
        return (
            batch_size == self.batch_size
            and self.min_context <= min_context <= max_context <= self.max_context
        )


class VerifyCalibrationCache:
    """Cache measurements under an exact, caller-supplied runtime identity.

    The identity must include model/code, hardware, topology, kernel backends,
    quantization, graph capacities and calibration settings. The cache does not
    infer compatibility across identities. Only the designated writer publishes;
    publication is atomic and readers treat malformed or stale files as misses.
    """

    VERSION = 1

    def __init__(self, directory: Path, *, identity: Mapping[str, object]):
        if not identity:
            raise ValueError("calibration cache requires a runtime identity")
        self.identity_json = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        digest = hashlib.sha256(self.identity_json.encode()).hexdigest()
        self.path = directory / f"verify-{digest}.json"

    @staticmethod
    def _validate(profiles: Sequence[VerifyCalibration]) -> None:
        for i, profile in enumerate(profiles):
            for previous in profiles[:i]:
                if (
                    previous.batch_size == profile.batch_size
                    and previous.verify_width == profile.verify_width
                    and max(previous.min_context, profile.min_context)
                    <= min(previous.max_context, profile.max_context)
                ):
                    raise ValueError("overlapping calibration profiles")

    def load(self) -> tuple[VerifyCalibration, ...] | None:
        try:
            raw = json.loads(self.path.read_text())
            if (
                set(raw) != {"version", "identity", "profiles"}
                or type(raw["version"]) is not int
                or raw["version"] != self.VERSION
                or raw["identity"] != json.loads(self.identity_json)
                or not isinstance(raw["profiles"], list)
                or not raw["profiles"]
            ):
                return None
            profiles = tuple(
                VerifyCalibration(
                    **{
                        **item,
                        "adaptive_step_ms": tuple(
                            tuple(pair) for pair in item["adaptive_step_ms"]
                        ),
                    }
                )
                for item in raw["profiles"]
            )
            self._validate(profiles)
            return profiles
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return None

    def save(self, profiles: Sequence[VerifyCalibration]) -> None:
        if not profiles:
            raise ValueError("cannot cache empty calibration")
        self._validate(profiles)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "identity": json.loads(self.identity_json),
            "profiles": [asdict(profile) for profile in profiles],
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, prefix=".verify-", delete=False
            ) as output:
                temporary = Path(output.name)
                json.dump(payload, output, allow_nan=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
