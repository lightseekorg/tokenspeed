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

"""GFX950/GFX1250 dense ``dsv4_prefill`` gluon checks for H=16 serving widths."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv4
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytest.importorskip("tokenspeed_triton")
pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

from utils import assert_no_triton_compile  # noqa: E402


def _prefill_name(width: int, heads: int = 16) -> str:
    q = torch.empty((1, heads, 512), dtype=torch.bfloat16)
    kv = torch.empty((width, 512), dtype=torch.bfloat16)
    return select_kernel(
        "attention",
        "dsv4_prefill",
        format_signature(
            q=dense_tensor_format(q.dtype),
            kv=dense_tensor_format(kv.dtype),
        ),
        traits={
            "head_dim": 512,
            "num_q_heads": heads,
            "cache_layout": "dense_workspace",
            "sinks": True,
            "selected_width": width,
            "metadata_dtypes": frozenset({torch.int32}),
        },
    ).name


def test_gluon_dsv4_prefill_is_selected_for_serving_widths():
    platform = current_platform()
    if platform.is_cdna4:
        expected = "gluon_dsv4_prefill_gfx950"
    elif platform.is_cdna5:
        expected = "gluon_dsv4_prefill_gfx1250"
    else:
        pytest.skip("AMD dense dsv4_prefill gluon")
    assert _prefill_name(128) == expected
    assert _prefill_name(640) == expected


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    result = torch.zeros_like(q)
    kv_rows = kv.reshape(-1, 512).float()
    for token in range(q.shape[0]):
        selected = indices[token, : int(lens[token])].long()
        selected = selected[(selected >= 0) & (selected < kv_rows.shape[0])]
        keys = kv_rows[selected]
        logits = q[token].float() @ keys.T * scale
        probabilities = torch.cat((logits, sink[:, None]), dim=1).softmax(dim=1)[:, :-1]
        result[token] = (probabilities @ keys).to(q.dtype)
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "heads, width",
    [
        (1, 128),
        (7, 128),
        (16, 128),
        (64, 128),
        (1, 640),
        (7, 640),
        (16, 640),
    ],
)
def test_gluon_dsv4_prefill_matches_reference(heads: int, width: int) -> None:
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("AMD dense dsv4_prefill gluon")
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    tokens = 5
    kv_rows = width + 32
    q = torch.randn((tokens, heads, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((kv_rows, 512), dtype=torch.bfloat16, device=device)
    indices = torch.randint(
        -2, kv_rows + 8, (tokens, width), dtype=torch.int32, device=device
    )
    lens = torch.tensor([width, 1, 0, width // 2, 3], dtype=torch.int32, device=device)
    sink = torch.linspace(-4, 4, heads, dtype=torch.float32, device=device)
    scale = 512**-0.5
    expected = _reference(q, kv, indices, lens, sink, scale)
    actual = dsv4.dsv4_prefill(q, kv, indices, lens, sink, scale)
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)
    triton = dsv4.dsv4_prefill(q, kv, indices, lens, sink, scale, solution="triton")
    torch.testing.assert_close(actual, triton, atol=8e-3, rtol=8e-3)
    assert torch.count_nonzero(actual[2]).item() == 0


def _serving_case(
    tokens: int, start: int, ratio: int, heads: int, seed: int
) -> tuple[torch.Tensor, ...]:
    """Inputs laid out like the V4.1 prefill workspace.

    Rows are [SWA prefix | current chunk | compressed history]; each token
    selects its 128-row sliding window (``-1`` before position 0) followed,
    when ``ratio`` is nonzero, by up to 512 top-k history rows padded with -1.
    """
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(seed)
    prefix = min(start, 127)
    positions = start + torch.arange(tokens, device=device)
    window = positions[:, None] - torch.arange(127, -1, -1, device=device)
    indices = torch.where(window >= 0, window - (start - prefix), -1)
    rows = prefix + tokens
    if ratio:
        history = (start + tokens) // ratio
        scores = torch.rand(tokens, history, device=device, generator=generator)
        visible = (
            torch.arange(history, device=device)[None, :]
            < ((positions + 1) // ratio)[:, None]
        )
        values, picked = scores.masked_fill(~visible, -1.0).topk(
            min(512, history), dim=1
        )
        picked = torch.where(values >= 0, picked + rows, -1)
        picked = torch.nn.functional.pad(picked, (0, 512 - picked.shape[1]), value=-1)
        indices = torch.cat((indices, picked), dim=1)
        rows += history
    q = torch.randn(
        (tokens, heads, 512), dtype=torch.bfloat16, device=device, generator=generator
    )
    kv = torch.randn(
        (rows, 512), dtype=torch.bfloat16, device=device, generator=generator
    )
    lens = torch.full((tokens,), indices.shape[1], dtype=torch.int32, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device, generator=generator)
    return q, kv, indices.to(torch.int32).contiguous(), lens, sink


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "tokens, start, ratio",
    [(70, 0, 2), (64, 4000, 2), (37, 100, 0), (128, 7872, 1)],
)
def test_gluon_dsv4_prefill_gfx950_serving_layout(
    tokens: int, start: int, ratio: int
) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 dsv4_prefill")
    q, kv, indices, lens, sink = _serving_case(tokens, start, ratio, 16, seed=3)
    scale = 512**-0.5
    actual = dsv4.dsv4_prefill(q, kv, indices, lens, sink, scale)
    expected = _reference(q, kv, indices, lens, sink, scale)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("width", [384, 512, 768, 1024, 1152])
def test_gluon_dsv4_prefill_gfx950_registered_widths(width: int) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 dsv4_prefill")
    torch.manual_seed(5)
    device = torch.device("cuda:0")
    tokens, heads, kv_rows = 6, 16, width + 40
    q = torch.randn((tokens, heads, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((kv_rows, 512), dtype=torch.bfloat16, device=device)
    indices = torch.randint(
        -1, kv_rows, (tokens, width), dtype=torch.int32, device=device
    )
    lens = torch.tensor(
        [width, width - 31, 33, 0, 1, width], dtype=torch.int32, device=device
    )
    sink = torch.full((heads,), -float("inf"), dtype=torch.float32, device=device)
    sink[::2] = 0.5
    scale = 512**-0.5
    actual = dsv4.dsv4_prefill(q, kv, indices, lens, sink, scale)
    expected = _reference(q, kv, indices, lens, sink, scale).nan_to_num(0.0)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_dsv4_prefill_gfx950_does_not_recompile() -> None:
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 dsv4_prefill")
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.prefill import (
        gluon_dsv4_prefill_gfx950,
    )

    scale = 512**-0.5
    # Token counts, kv rows, and lengths vary per batch; none may specialize.
    dsv4.dsv4_prefill(*_serving_case(17, 33, 2, 16, seed=0), scale)
    with assert_no_triton_compile(gluon_dsv4_prefill_gfx950):
        for tokens, start in ((1, 0), (16, 64), (32, 160), (129, 7000), (641, 3)):
            q, kv, indices, lens, sink = _serving_case(tokens, start, 2, 16, seed=1)
            lens[::3] = 17
            dsv4.dsv4_prefill(q, kv, indices, lens, sink, scale)
