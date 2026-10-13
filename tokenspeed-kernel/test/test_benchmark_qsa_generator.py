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
import tokenspeed_kernel.benchmark.generators.qsa as qsa_generator
import torch
from tokenspeed_kernel.benchmark.harness import BenchmarkRequest


def test_selected_slots_follow_each_rows_prefix() -> None:
    positions = torch.tensor([3, 5, 9, 9])
    request_index = torch.tensor([0, 0, 1, 1])
    slots = qsa_generator.qsa_selected_slots(
        positions,
        request_index,
        region_slots=16,
        width=6,
        generator=torch.Generator().manual_seed(0),
    )

    assert slots.shape == (4, 6) and slots.dtype == torch.int32
    for row, (position, request) in enumerate(zip(positions, request_index)):
        valid = slots[row][slots[row] >= 0]
        start = 2 + int(request) * 16
        assert valid.numel() == min(int(position), 6)
        assert torch.all(slots[row][valid.numel() :] == -1)
        assert torch.all(valid[1:] > valid[:-1])
        assert torch.all((valid >= start) & (valid < start + int(position)))
    # A row whose prefix fits the width selects every slot of it.
    assert slots[0].tolist() == [2, 3, 4, -1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("phase", "q_len", "is_decode"), [("decode", 4, True), ("prefill", None, False)]
)
def test_qsa_generator_selection_matches_operation_api(
    monkeypatch: pytest.MonkeyPatch, phase: str, q_len: int | None, is_decode: bool
) -> None:
    import tokenspeed_kernel.ops.attention.qsa as qsa_ops

    captured: list[dict[str, object]] = []

    def select(family, mode, signature, **kwargs):
        captured.append(kwargs["traits"])
        raise RuntimeError("selection captured")

    monkeypatch.setattr(qsa_ops, "select_kernel", select)
    monkeypatch.setattr(qsa_generator, "load_builtin_kernels", lambda: None)
    parameters = {
        "model_profile": "qwen38_flash_tp2",
        "phase": phase,
        "batch": 2,
        "context_tokens": 8,
        "dtype": "bfloat16",
        "kv_dtype": "bfloat16",
    }
    if q_len is not None:
        parameters["q_len"] = q_len
    request = BenchmarkRequest(
        family="attention",
        mode="qsa_sparse_attention",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )
    with pytest.raises(RuntimeError, match="selection captured"):
        qsa_generator.prepare_qsa_sparse_attention(request, None)

    rows_per_request = q_len or 8
    assert captured[0] == {
        "batch_size": 2 if is_decode else 2 * rows_per_request,
        "q_len": rows_per_request if is_decode else 1,
        "num_q_heads": 12,
        "num_kv_heads": 1,
        "head_dim": 256,
        "value_head_dim": 256,
        "selected_width": 2051,
        "is_decode": is_decode,
    }
