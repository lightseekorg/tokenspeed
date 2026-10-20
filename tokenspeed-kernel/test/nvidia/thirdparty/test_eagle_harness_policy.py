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

"""EAGLE-Harness device policy kernels: the cumulative ranked scores
select_verify publishes against torch's, bit for bit, at every shape the
policy ranks."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.eagle_harness_policy import (
    eagle_harness_policy_select_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="eagle_harness_policy CUDA kernel is NVIDIA-only",
)

_DEVICE = "cuda"
_MAX_ROWS, _MAX_COLS = 32, 64
# The ``State`` scalars (eagle_harness_policy.cu) a ranking uses; all others zero make a first unsampled round.
_RANKED_COUNT, _BATCH_SIZE, _VERIFY_COUNT, _FLAT, _NUM_STATE = 6, 7, 17, 18, 21


def _ranked_rows(
    rows: int, cols: int, binades: int, generator: torch.Generator
) -> torch.Tensor:
    """Each row's scores in rank order: descending scores in (0, 1) spread
    over ``binades`` binades (many: sums in another order round apart)."""
    scores = torch.rand((rows, cols), generator=generator, dtype=torch.float64)
    scale = torch.exp2(-torch.randint(0, binades, (rows, cols), generator=generator))
    return (scores * scale).float().sort(dim=-1, descending=True).values


@pytest.mark.parametrize("flat", [False, True])
def test_ranked_scores_match_torch(flat: bool) -> None:
    generator = torch.Generator().manual_seed(int(flat))
    # The kernel reads the live corner of a row-padded buffer.
    buffer = torch.full((_MAX_ROWS + 3, _MAX_COLS + 5), float("nan"), device=_DEVICE)
    top_scores = buffer[:, :_MAX_COLS]
    out = torch.empty(_MAX_ROWS * _MAX_COLS, device=_DEVICE)  # published scores
    state = torch.zeros(_NUM_STATE, dtype=torch.int32, device=_DEVICE)
    # Cost tables of one context without measured widths: the verify is width 1.
    i32 = dict(dtype=torch.int32, device=_DEVICE)
    f64 = dict(dtype=torch.float64, device=_DEVICE)
    tables = (
        torch.zeros((1, 1, 2), **i32),
        torch.zeros((1, 1, 8), **f64),
        torch.zeros((_MAX_ROWS, 1), **i32),
        torch.zeros((_MAX_ROWS, 1, 2), **f64),
    )
    recent, hist = torch.zeros(1, **i32), torch.zeros((_MAX_ROWS, 1), **i32)
    shapes, mismatches = [], []
    for binades in (1, 24):
        for rows in range(1, _MAX_ROWS + 1):
            for cols in range(1, _MAX_COLS + 1):
                shapes.append((binades, rows, cols))
                top = _ranked_rows(rows, cols, binades, generator).to(_DEVICE)
                expected = torch.cumsum(
                    top.view(-1) if flat else top.mean(dim=0), dim=0
                )
                buffer[:rows, :cols] = top
                state.zero_()
                state[_BATCH_SIZE] = rows
                state[_VERIFY_COUNT] = cols
                state[_FLAT] = flat
                eagle_harness_policy_select_verify(
                    state, recent, hist, out, top_scores, *tables, 1, 0
                )
                n = expected.numel()
                mismatches.append(
                    (state[_RANKED_COUNT] != n)
                    | (out[:n].view(torch.int32) != expected.view(torch.int32)).any()
                )
    failed = torch.stack(mismatches).nonzero().flatten().tolist()
    assert not failed, f"(binades, rows, cols) {[shapes[i] for i in failed[:5]]}"
