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

"""EAGLE-Harness round counts for tuning the chain width table: what the host
knows of each round (its live batch size, and the table's width where the
table decides it), never a device value.

``TOKENSPEED_EAGLE_HARNESS_ROUND_STATS=N`` (0, off, by default) makes rank 0
log the counts every ``N`` rounds."""

from __future__ import annotations

__all__ = ["RoundStats"]


class RoundStats:
    """Cumulative rounds and requests per live batch size bucket (sizes up to
    1, 2, 4, ..., 512), by verify width where the width table decides the
    round (``w<W>``), else ``rule`` (the confidence rule; its width stays on
    the device); rounds drafted after a prefill forward also count under
    ``prefill``. Every ``interval`` rounds ``record`` returns a log line, so
    a window's counts are the difference of the lines at its ends.

    Args:
        interval: Rounds between log lines.
    """

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError(f"round stats need a positive interval, got {interval}")
        self.interval = interval
        self.rounds = 0
        # Bucket -> kind -> [rounds, requests].
        self.counts: dict[int, dict[str, list[int]]] = {}

    @staticmethod
    def bucket(bs: int) -> int:
        """The smallest power of two at least ``bs``."""
        return 1 << (bs - 1).bit_length()

    def record(self, bs: int, width: int | None, prefill: bool) -> str | None:
        """Book a round of ``bs`` live requests the table drew ``width`` for
        (None: the confidence rule decided); the log line every ``interval``
        rounds, else None."""
        kinds = self.counts.setdefault(self.bucket(bs), {})
        for kind in ("rule" if width is None else f"w{width}",) + (
            ("prefill",) if prefill else ()
        ):
            counts = kinds.setdefault(kind, [0, 0])
            counts[0] += 1
            counts[1] += bs
        self.rounds += 1
        return self.line() if self.rounds % self.interval == 0 else None

    def line(self) -> str:
        """``EAGLE-Harness rounds <total>: bs<=<bucket> <kind> <rounds>r/<requests>q ...``"""
        buckets = []
        for bucket in sorted(self.counts):
            kinds = self.counts[bucket]
            buckets.append(
                f"bs<={bucket} "
                + " ".join(f"{k} {r}r/{q}q" for k, (r, q) in sorted(kinds.items()))
            )
        return f"EAGLE-Harness rounds {self.rounds}: " + "; ".join(buckets)
