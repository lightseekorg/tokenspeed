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

import pytest
import torch
from tokenspeed_kernel.ops.attention.qsa import qwen4_exp_qsa_block_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="CuTe DSL QSA stream top-k requires NVIDIA Blackwell",
)


def _bucket_scores(capacity: int, device: str) -> list[tuple[str, torch.Tensor]]:
    columns = torch.arange(capacity, device=device, dtype=torch.int64)
    scores = [
        ("dense-ascending", (1 << 23) + columns),
        ("dense-descending", (1 << 23) + columns.flip(0)),
        ("all-equal", torch.full_like(columns, 1 << 23)),
    ]
    # The initial overlap tile contains both the definite winners and the
    # entire boundary bucket, including either side of its 256-entry capacity.
    for count in (2, 8, 16, 31, 32, 33, 127, 128, 255, 256, 257, 513, 1025):
        better = 512 - min(count // 2, 256)
        units = torch.zeros_like(columns)
        units[:better] = 1 << 25
        units[better : better + count] = (1 << 23) + torch.arange(count, device=device)
        scores.append((f"boundary-{count}", units))
        if count in (255, 256, 257):
            tied_units = units.clone()
            tied_units[better : better + count] = (1 << 23) + torch.arange(
                count, device=device
            ) % 67
            scores.append((f"partial-ties-{count}", tied_units))
    # Adjacent FP32 values straddle nearest-even FP16 rounding boundaries,
    # including coarse-group edges and both sides of a power of two.
    for midpoint in (
        (1 << 23) - 4096,
        (1 << 23) + 4096,
        (1 << 23) + 127 * 8192 + 4096,
        (1 << 23) + 128 * 8192 + 4096,
        (1 << 24) - 8192,
        (1 << 24) - 4096,
    ):
        for count in (32, 33, 256, 257):
            better = 512 - count // 2
            units = torch.zeros_like(columns)
            units[:better] = 1 << 25
            units[better : better + count] = (
                midpoint + torch.arange(count, device=device) - count // 2
            )
            scores.append((f"midpoint-{midpoint}-{count}", units))
    return scores


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    ("rows", "pages", "width"),
    [
        pytest.param(8, 64, 8, id="native-linear"),
        pytest.param(1, 128, 1, id="guarded-single"),
        pytest.param(2, 1024, 1, id="guarded-long"),
        pytest.param(64, 128, 1, id="native-many-rows"),
        pytest.param(8, 256, 8, id="split-prefill"),
        pytest.param(136, 32, 136, id="grouped-prefill"),
        pytest.param(136, 64, 136, id="dense-prefill"),
        pytest.param(136, 128, 136, id="overlap-prefill"),
        pytest.param(137, 32, 137, id="grouped-prefill-tail"),
        pytest.param(139, 64, 139, id="dense-prefill-tail"),
        pytest.param(143, 128, 143, id="overlap-prefill-tail"),
        pytest.param(15, 256, 15, id="split-prefill-tail"),
        pytest.param(136, 2048, 136, id="overlap-second-window"),
    ],
)
def test_stream_topk_exact_fp32_bucket_boundaries(
    device: str,
    rows: int,
    pages: int,
    width: int,
    enable_pdl: bool,
) -> None:
    """Bucket filtering must preserve exact scores, including after graph replay."""

    capacity = pages * 64
    query = torch.zeros(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(capacity, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    complete = torch.full((rows,), capacity, device=device, dtype=torch.int32)
    # Power-of-two scaling preserves the exact order while moving the cutoff
    # through FP16 zero, subnormal, normal, exponent and overflow boundaries.
    exponents = torch.tensor(
        [-25, -24, -15, -14, -1, 0, 1, 15, 16, 17, 40, 100],
        device=device,
        dtype=torch.int32,
    )
    row_ids = torch.arange(rows, device=device)
    digit_shifts = torch.tensor([24, 16, 8, 0], device=device)

    def invoke() -> torch.Tensor:
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 << 20,
            path="stream",
            max_logits_bytes=256 << 20,
            solution="cute_dsl",
            override=None,
            enable_pdl=enable_pdl,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()

    for step, (name, units) in enumerate(_bucket_scores(capacity, device)):
        # Four exactly representable BF16 base-256 digits make the MMA's FP32
        # score equal to units * 2**(exponent - 23). Only one head contributes.
        keys[:, 0, :4] = ((units[:, None] >> digit_shifts) & 255).to(torch.bfloat16)
        exponent = exponents[(row_ids + step) % exponents.numel()]
        query[:, 0, :4] = torch.exp2(
            (exponent[:, None] - 23 + digit_shifts).float()
        ).to(torch.bfloat16)
        for frontier in (capacity, capacity - 1, 513):
            complete.fill_(frontier)
            graph.replay()
            assert bool(((selected >= 0) & (selected < frontier)).all()), name
            ordered = selected.sort(dim=1).values
            assert bool((ordered[:, 1:] != ordered[:, :-1]).all()), name
            actual = units[selected.long()].sort(dim=1).values
            expected = units[:frontier].topk(512).values.sort().values
            torch.testing.assert_close(
                actual,
                expected[None].expand(rows, -1),
                rtol=0,
                atol=0,
                msg=lambda message: f"{name}, frontier={frontier}: {message}",
            )
