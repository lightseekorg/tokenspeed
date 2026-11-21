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

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.gemm.gluon_gemv_autotune import (
    DEFAULT_ROUTING,
    ROWS_TILE,
    GemvRouting,
    decide_gemv_routing,
)
from utils import is_cdna4


class _Device:
    """Deterministic per-call times (us) of the kernels ``gemm.mm`` reaches.

    Dispatch mirrors ``gemm.mv``: the dense GEMV on its buckets, then the
    few-row tile, then the single-row GEMV, else mm's own path.
    """

    def __init__(
        self,
        default: dict[int, float],
        dense: dict[int, float],
        rowcta: float,
        rows: dict[int, float],
    ) -> None:
        self.default = default
        self.dense = dense
        self.rowcta = rowcta
        self.rows = rows
        self.timed: list[tuple[GemvRouting, ...]] = []

    def __call__(
        self, routings: Sequence[GemvRouting], ms: tuple[int, ...]
    ) -> list[float]:
        self.timed.append(tuple(routings))
        return [sum(self._time(routing, m) for m in ms) for routing in routings]

    def _time(self, routing: GemvRouting, m: int) -> float:
        bucket = 1 if m == 1 else 4 if m <= 4 else 16
        if bucket in routing.gemv_buckets:
            return self.dense[m]
        if m >= 2 and 1 << (m - 1).bit_length() in dict(routing.rows):
            return self.rows[m]
        if m == 1 and routing.single_row:
            return self.rowcta
        return self.default[m]


# Rounded medians the hand-made Qwen3.8-27B entries came from.
_DOWN = _Device(
    default={1: 34.0, 2: 34.2, 4: 36.0, 8: 40.0, 12: 42.0, 16: 44.0},
    dense={1: 29.4, 2: 32.0, 4: 33.4, 8: 37.0, 12: 40.0, 16: 44.3},
    rowcta=34.7,
    rows={2: 30.5, 4: 35.7},
)
_O_PROJ = _Device(
    default={1: 20.0, 2: 21.0, 4: 22.0, 8: 24.0, 12: 26.0, 16: 28.0},
    dense={1: 14.8, 2: 17.5, 4: 18.2, 8: 19.4, 12: 22.6, 16: 25.3},
    rowcta=12.9,
    rows={2: 14.6, 4: 18.3},
)


def test_down_projection_takes_the_dense_gemv_through_16_rows() -> None:
    routing = decide_gemv_routing(_DOWN, rows_tile_fits=True)

    # The 2-row tile wins alone, but the dense GEMV beats it plus the decode
    # GEMM over 2 and 4 rows; 5-16 rows win on the summed time although 16
    # rows alone are slower.
    assert routing == GemvRouting(frozenset({1, 4, 16}), (), False)


def test_output_projection_takes_the_row_cta_gemv_through_4_rows() -> None:
    routing = decide_gemv_routing(_O_PROJ, rows_tile_fits=True)

    assert routing == GemvRouting(
        frozenset({16}), ((2, ROWS_TILE), (4, ROWS_TILE)), True
    )


def test_untileable_weight_never_times_the_tile() -> None:
    device = _Device(_O_PROJ.default, _O_PROJ.dense, _O_PROJ.rowcta, _O_PROJ.rows)

    routing = decide_gemv_routing(device, rows_tile_fits=False)

    assert routing.rows == ()
    assert routing.gemv_buckets == {4, 16}
    assert all(not r.rows for timed in device.timed for r in timed)


def test_no_entry_without_a_lead() -> None:
    even = _Device(
        default={m: 10.0 for m in (1, 2, 4, 8, 12, 16)},
        dense={m: 9.9 for m in (1, 2, 4, 8, 12, 16)},
        rowcta=9.9,
        rows={2: 9.9, 4: 9.9},
    )

    assert decide_gemv_routing(even, rows_tile_fits=True) == DEFAULT_ROUTING


def test_routing_round_trips_through_json() -> None:
    routing = GemvRouting(frozenset({16}), ((2, (4, 4)), (4, (2, 8))), True)

    assert GemvRouting.from_json(routing.to_json()) == routing


# --------------------------------------------------------------------------- #
# On the GPU
# --------------------------------------------------------------------------- #

gpu = pytest.mark.skipif(not is_cdna4(), reason="AMD CDNA4 is required")

# Qwen3.8-27B's output projection, taken out of the tables for the test.
_N, _K = 5120, 6144


@pytest.fixture
def untabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from tokenspeed_kernel.ops.gemm import gluon_gemv_autotune as autotune
    from tokenspeed_kernel.selection import default_autotune_cache

    monkeypatch.setenv("TOKENSPEED_KERNEL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TOKENSPEED_GEMV_AUTOTUNE", "1")
    default_autotune_cache.cache_clear()
    autotune.gemv_autotune_enabled.cache_clear()
    saved = autotune._current_routing(_N, _K)
    autotune._apply(_N, _K, DEFAULT_ROUTING)
    autotune._settled.discard((_N, _K))
    yield tmp_path
    autotune._apply(_N, _K, saved)
    autotune._settled.discard((_N, _K))
    autotune.gemv_autotune_enabled.cache_clear()
    default_autotune_cache.cache_clear()


@gpu
def test_warmup_tunes_an_untabled_weight_once(untabled: Path) -> None:
    from tokenspeed_kernel.ops.gemm import _use_mv
    from tokenspeed_kernel.ops.gemm import gluon_gemv_autotune as autotune
    from tokenspeed_kernel.ops.gemm import mm
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import default_autotune_cache

    weight = torch.randn(_N, _K, device="cuda", dtype=torch.bfloat16) / _K**0.5
    x = torch.randn(1, _K, device="cuda", dtype=torch.bfloat16)

    mm(x, weight)
    # Outside warmup a miss keeps the default routing.
    assert autotune._current_routing(_N, _K) == DEFAULT_ROUTING

    with kernel_warmup(enabled=True):
        out = mm(x, weight)
    routing = autotune._current_routing(_N, _K)

    torch.testing.assert_close(
        out.float(), x.float() @ weight.float().T, atol=3e-2, rtol=3e-2
    )
    assert routing != DEFAULT_ROUTING
    assert (_N, _K) in autotune._settled
    (entry,) = default_autotune_cache()._entries.values()
    assert GemvRouting.from_json(entry) == routing
    # A single row leaves mm's own path only where a GEMV won.
    assert _use_mv(x, weight) == (routing.single_row or 1 in routing.gemv_buckets)

    # A fresh process applies the cached routing without measuring, even
    # outside warmup.
    autotune._apply(_N, _K, DEFAULT_ROUTING)
    autotune._settled.discard((_N, _K))
    default_autotune_cache.cache_clear()
    mm(x, weight)
    assert autotune._current_routing(_N, _K) == routing
