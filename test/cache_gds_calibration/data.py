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

"""Shared data validation and serialization for offline calibration."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


def read_rows(paths):
    return [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def write_rows(path, rows):
    with Path(path).open("x") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"expected a finite positive value, got {value}")
    return result


def integer(value, minimum):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"expected integer >= {minimum}, got {value}")
    return int(value)


@dataclass(frozen=True)
class PrefillCase:
    """Requested cached prefix and additional prompt tokens, excluding output."""

    prefix_tokens: int
    new_tokens: int
    split: str

    def __post_init__(self) -> None:
        if type(self.prefix_tokens) is not int or type(self.new_tokens) is not int:
            raise ValueError("prefill token counts must be integers")
        integer(self.prefix_tokens, 0)
        integer(self.new_tokens, 1)
        if self.split not in ("train", "validation"):
            raise ValueError(f"invalid prefill case: {self}")

    @property
    def prompt_tokens(self) -> int:
        return self.prefix_tokens + self.new_tokens


@dataclass(frozen=True)
class PrefillSample:
    case: PrefillCase
    repeat: int
    input_ids: list[int]

    @property
    def warmup(self) -> bool:
        return self.repeat < 0


def validate_cases(cases: Sequence[PrefillCase]) -> None:
    """Require distinct geometries, including across train and validation."""
    geometries = {(case.prefix_tokens, case.new_tokens) for case in cases}
    if not cases or len(geometries) != len(cases):
        raise ValueError("require nonempty, distinct prefill geometries")


def prefill_samples(
    cases: Sequence[PrefillCase],
    token_pool: Sequence[int],
    *,
    repeats: int,
    warmup_repeats: int,
    seed: int,
) -> Iterator[PrefillSample]:
    """Yield warmups before shuffled measurements, retaining exact token IDs.

    Random leading tokens isolate each prime/continuation pair. The remaining
    tokens use contiguous corpus spans. Only one prompt is materialized at once.
    """
    validate_cases(cases)
    integer(repeats, 1)
    integer(warmup_repeats, 1)
    if len(set(token_pool)) < 32:
        raise ValueError(
            "representative corpus requires at least 32 distinct token IDs"
        )
    rng = random.Random(seed)
    measured = [(case, repeat) for case in cases for repeat in range(repeats)]
    rng.shuffle(measured)
    ordered = [
        (case, -repeat - 1) for repeat in range(warmup_repeats) for case in cases
    ] + measured
    for case, repeat in ordered:
        ids = rng.choices(token_pool, k=min(case.prompt_tokens, 32))
        while len(ids) < case.prompt_tokens:
            start = rng.randrange(len(token_pool))
            ids.extend(
                token_pool[start : start + min(256, case.prompt_tokens - len(ids))]
            )
        yield PrefillSample(case, repeat, ids)


def validate_prefix(
    *,
    expected_prefix_tokens: int,
    expected_prompt_tokens: int,
    cached_tokens: int,
    prompt_tokens: int,
    max_prefix_shortfall: int,
) -> None:
    """Check actual recovery depth without replacing it with the requested depth."""
    integer(max_prefix_shortfall, 0)
    integer(cached_tokens, 0)
    integer(prompt_tokens, 1)
    if (
        prompt_tokens != expected_prompt_tokens
        or not 0 <= expected_prefix_tokens - cached_tokens <= max_prefix_shortfall
    ):
        raise ValueError(
            f"unexpected actual prefix/prompt: cached_tokens={cached_tokens}, prompt_tokens={prompt_tokens}"
        )


def completed_repeats(
    cases: Sequence[PrefillCase],
    paths: Sequence[str],
    *,
    repeats: int,
    deployment: str,
    max_prefix_shortfall: int,
) -> set[tuple[int, int, int]]:
    """Read completed native samples to skip when resuming a selected case set.

    Warmups and unselected geometries do not count. Reusing a sample still
    requires importing its original server log before fitting.
    """
    selected = {(case.prefix_tokens, case.prompt_tokens): case for case in cases}
    completed: set[tuple[int, int, int]] = set()
    for row in read_rows(paths):
        geometry = row["expected_a"], row["expected_b"]
        if row["warmup"] or geometry not in selected:
            continue
        if row["deployment"] != deployment or row["split"] != selected[geometry].split:
            raise ValueError("resumed sample has a different deployment or split")
        repeat = integer(row["repeat"], 0)
        if repeat >= repeats:
            raise ValueError("resumed repeat is outside the requested repeat count")
        completion = row["completion"]
        if (
            completion["finish_reason"] not in ("length", "stop")
            or completion["completion_tokens"] != 1
        ):
            raise ValueError("cannot resume from a failed native completion")
        validate_prefix(
            expected_prefix_tokens=geometry[0],
            expected_prompt_tokens=geometry[1],
            cached_tokens=completion["cached_tokens"],
            prompt_tokens=completion["prompt_tokens"],
            max_prefix_shortfall=max_prefix_shortfall,
        )
        key = (*geometry, repeat)
        if key in completed:
            raise ValueError("duplicate resumed geometry/repeat")
        completed.add(key)
    return completed
