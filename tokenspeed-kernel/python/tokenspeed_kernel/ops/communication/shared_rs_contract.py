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

"""CPU-only experiment configuration and deterministic layout mapping."""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class SharedRsTuning:
    """Explicit launch geometry; vectors are BF16x8 requests per thread."""

    ctas: int
    threads: int
    vectors_per_thread: int
    layout: str

    def validate(self, resident_ctas: int) -> None:
        if self.ctas not in (32, 64, 128) or self.ctas > resident_ctas:
            raise ValueError("CTA grid must fit the rank-uniform resident limit")
        if self.threads not in (256, 512, 1024):
            raise ValueError("threads must be 256, 512 or 1024")
        if self.vectors_per_thread not in (1, 2, 4):
            raise ValueError("vectors_per_thread must be 1, 2 or 4")
        if self.layout not in ("flat", "row"):
            raise ValueError("layout must be flat or row")

    def key(self) -> str:
        return f"{self.ctas}:{self.threads}:{self.vectors_per_thread}:{self.layout}"


def shared_rs_tunings() -> tuple[SharedRsTuning, ...]:
    """Return all 54 launch/layout candidates, flat before row."""
    return tuple(
        SharedRsTuning(c, t, v, layout)
        for layout, c, t, v in product(
            ("flat", "row"), (32, 64, 128), (256, 512, 1024), (1, 2, 4)
        )
    )


def vector_coordinate(
    tuning: SharedRsTuning, block: int, thread: int, iteration: int, vector: int
) -> tuple[int, int]:
    """Map one logical request to (row, BF16x8 column), before masking."""
    if tuning.layout == "row":
        rows = tuning.threads // 128
        return (
            (block + iteration * tuning.ctas) * rows * tuning.vectors_per_thread
            + thread // 128
            + vector * rows,
            thread % 128,
        )
    chunk = (
        (block + iteration * tuning.ctas) * tuning.threads * tuning.vectors_per_thread
        + thread
        + vector * tuning.threads
    )
    return divmod(chunk, 112)
