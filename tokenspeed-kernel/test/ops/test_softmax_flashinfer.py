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

"""Numerical and dependent-input coverage for the vendored CUDA softmax."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.flashinfer_softmax import softmax

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="Vendored softmax requires NVIDIA CUDA."
)

_SHAPES = [
    pytest.param(1, 4096, id="fused-cached"),
    pytest.param(1, 129280, id="map-reduce"),
    pytest.param(129, 65536, id="fused-uncached"),
]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@triton.jit
def _delayed_temperature(temperature_ptr, rows: tl.constexpr, BLOCK: tl.constexpr):
    # Bounded producer: completion never depends on consumer concurrency.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    value = tl.inline_asm_elementwise(
        """
        {
            .reg .u64 started, current, elapsed;
            .reg .pred waiting;
            mov.u64 started, %clock64;
        bounded_delay:
            mov.u64 current, %clock64;
            sub.u64 elapsed, current, started;
            setp.lt.u64 waiting, elapsed, $1;
            @waiting bra bounded_delay;
            mov.f32 $0, 0f3F800000;
        }
        """,
        constraints="=f,l",
        args=[tl.full((), 3000000, tl.uint64)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )
    offsets = tl.arange(0, BLOCK)
    tl.store(temperature_ptr + offsets, value, mask=offsets < rows)


def _check_pdl(device: str, enable_pdl: bool) -> None:
    if enable_pdl and torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("PDL requires compute capability 9.0 or newer.")


def _logits(rows: int, vocab: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(42)
    return torch.randn(rows, vocab, dtype=dtype, device=device, generator=generator)


@pytest.mark.parametrize(
    "rows,vocab",
    _SHAPES
    + [pytest.param(2, 4095, id="odd-vocab"), pytest.param(1, 1, id="singleton")],
)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("temperature_kind", ["default", "zero", "scalar", "tensor"])
def test_softmax_reference(
    rows: int,
    vocab: int,
    dtype: torch.dtype,
    enable_pdl: bool,
    temperature_kind: str,
    device: str,
    request: pytest.FixtureRequest,
) -> None:
    _check_pdl(device, enable_pdl)
    if temperature_kind in ("zero", "tensor") and vocab in (129280, 4095, 1):
        # Existing in the serialized original: padded -inf lanes multiply by
        # zero and contaminate the reduction. Do not hide a future fix (XPASS).
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="Pre-existing zero-temperature padding defect, separate from PDL.",
            )
        )
    logits = _logits(rows, vocab, dtype, device)
    temperature = {"default": None, "zero": 0.0, "scalar": 0.7}.get(temperature_kind)
    if temperature_kind == "tensor":
        temperature = torch.linspace(0.5, 1.5, rows, dtype=torch.float32, device=device)
        temperature[0] = 0.0
    values = torch.as_tensor(
        1.0 if temperature is None else temperature, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    inverse = torch.where(values == 0, 0.0, values.reciprocal())
    reference = torch.softmax(logits.float() * inverse, dim=-1)
    actual = softmax(logits, temperature=temperature, enable_pdl=enable_pdl)
    assert actual.dtype == torch.float32 and actual.shape == logits.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("rows,vocab", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_softmax_waits_for_temperature(
    rows: int,
    vocab: int,
    dtype: torch.dtype,
    enable_pdl: bool,
    graph_replay: bool,
    device: str,
) -> None:
    _check_pdl(device, True)  # The stress producer always uses PDL.
    logits = _logits(rows, vocab, dtype, device)
    temperature = torch.ones(rows, dtype=torch.float32, device=device)
    reference = softmax(logits, temperature=temperature, enable_pdl=False).cpu()

    def invoke() -> torch.Tensor:
        temperature.fill_(float("nan"))
        _delayed_temperature[(1,)](
            temperature,
            rows=rows,
            BLOCK=triton.next_power_of_2(rows),
            num_warps=1,
            launch_pdl=True,
        )
        return softmax(logits, temperature=temperature, enable_pdl=enable_pdl)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        invoke()  # Compile before capture; this output is not evidence.
    stream.synchronize()
    graph = None
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = invoke()
    for _ in range(2):
        if graph is not None:
            graph.replay()
        else:
            with torch.cuda.stream(stream):
                actual = invoke()
            stream.synchronize()
        copied = actual.cpu()
        assert torch.isfinite(copied).all()
        assert torch.equal(copied, reference)
        assert torch.equal(temperature.cpu(), torch.ones(rows))
    if graph is not None:
        graph.reset()
