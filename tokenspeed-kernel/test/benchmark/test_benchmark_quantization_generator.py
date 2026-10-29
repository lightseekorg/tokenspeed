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
import tokenspeed_kernel.benchmark.generators.quantization as quantization_generator
import torch
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest


def _request(registration: str | None, **updates: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="quantization",
        mode="fp8_with_scale",
        parameters={
            "M": 192,
            "K": 5120,
            "granularity": "token_group",
            "group_size": 32,
            "scale_encoding": "ue8m0",
            "dtype": "bfloat16",
            **updates,
        },
        solution=None,
        registration=registration,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.parametrize("registration", [None, "triton_quantize_fp8_group32_ue8m0"])
def test_fp8_with_scale_selection_matches_operation_api(
    monkeypatch: pytest.MonkeyPatch, registration: str | None
) -> None:
    from tokenspeed_kernel.ops import quantization as quantization_ops

    captured: dict[str, object] = {}

    def capture(source: str):
        def select(family, mode, signature, **kwargs):
            kwargs.pop("platform", None)
            if kwargs.get("features") is None:
                # An omitted features argument and features=None both leave
                # selection unconstrained; compare only real constraints.
                kwargs.pop("features", None)
            captured[source] = (family, mode, signature, kwargs)
            raise RuntimeError("selection captured")

        return select

    monkeypatch.setattr(quantization_ops, "select_kernel", capture("operation"))
    monkeypatch.setattr(quantization_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(quantization_generator, "select_kernel", capture("generator"))

    x = torch.empty((192, 5120), dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match="selection captured"):
        quantization_ops.quantize_fp8(
            x,
            granularity="token_group",
            group_size=32,
            scale_encoding="ue8m0",
            enable_pdl=False,
            override=registration,
            solution=None,
        )
    with pytest.raises(RuntimeError, match="selection captured"):
        quantization_generator.prepare_fp8_with_scale(_request(registration), None)

    assert captured["generator"] == captured["operation"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"block_size": [1, 32]}, "Unknown"),
        ({"granularity": "block"}, "granularity"),
        ({"group_size": 0}, "group_size"),
        ({"dtype": "float32"}, "dtype"),
    ],
)
def test_fp8_with_scale_rejects_invalid_cases(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(BenchmarkCaseError, match=message):
        quantization_generator.prepare_fp8_with_scale(_request(None, **updates), None)
