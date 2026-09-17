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
            "num_heads": heads,
            "cache_layout": "dense_workspace",
            "support_sink": True,
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
