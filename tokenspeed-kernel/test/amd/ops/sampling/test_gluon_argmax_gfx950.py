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
from utils import is_cdna4, is_cdna5

MODEL_VOCABS = {
    "dsv4": 129280,  # V4 Pro/Flash and V4.1 Flash share the output vocabulary.
    "kimi_k3": 163840,
    "glm_5_3_flash": 154880,
}

if is_cdna4():
    from tokenspeed_kernel_amd.ops.gfx950.sampling import argmax as argmax_impl

    _ARCH = "gfx950"
elif is_cdna5():
    from tokenspeed_kernel_amd.ops.gfx1250.sampling import argmax as argmax_impl

    _ARCH = "gfx1250"
else:
    pytest.skip(
        "AMD CDNA4 or CDNA5 is required for Gluon argmax tests",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_argmax_matches_torch_for_dtypes(dtype):
    torch.manual_seed(0xA950)
    x = torch.randn(8, 4096, device="cuda", dtype=dtype)
    out = argmax_impl.argmax(x, out=None)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


# Cover every _select_config bucket on gfx950 and gfx1250, with an odd M at
# each bucket boundary so non-power-of-two grids are exercised. fp16 and bf16
# share tile geometry, so only bf16 and fp32 are exercised here.
@pytest.mark.parametrize("N", MODEL_VOCABS.values(), ids=MODEL_VOCABS.keys())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("M", [1, 2, 3, 5, 17, 64, 65, 128, 129, 256, 513, 1024])
def test_argmax_matches_torch_for_model_shapes(M, N, dtype):
    torch.manual_seed(M ^ N)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    out = argmax_impl.argmax(x, out=None)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (1, MODEL_VOCABS["dsv4"], torch.float32),
        (4, MODEL_VOCABS["dsv4"], torch.float32),
        (8, MODEL_VOCABS["dsv4"], torch.float16),
        (128, MODEL_VOCABS["dsv4"], torch.bfloat16),
    ],
)
def test_argmax_all_nan_rows_return_sentinel(M, N, dtype):
    x = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    out = argmax_impl.argmax(x, out=None)
    expected = torch.full((M,), -1, device="cuda", dtype=out.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("M", [4, 128, 512])
def test_argmax_ignores_nan_but_preserves_valid_negative_infinity(M):
    N = MODEL_VOCABS["dsv4"]
    x = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
    x[0, 123] = 0.5
    x[0, 456] = 1.0
    x[1].fill_(-float("inf"))
    x[2, 7] = 3.0
    x[2, 5] = 3.0

    out = argmax_impl.argmax(x, out=None)
    expected = torch.full((M,), -1, device="cuda", dtype=out.dtype)
    expected[:3] = torch.tensor([456, 0, 5], device="cuda", dtype=out.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_argmax_returns_first_index_on_ties():
    M, N = 4, 4096
    x = torch.full((M, N), -100.0, device="cuda", dtype=torch.float32)
    plant_positions = [
        [0, 7, 9],
        [3, 4],
        [128, 1024, 2048],
        [N - 1, 17],
    ]
    for row, positions in enumerate(plant_positions):
        for pos in positions:
            x[row, pos] = 0.0
    torch.testing.assert_close(
        argmax_impl.argmax(x, out=None), torch.argmax(x, dim=-1), atol=0, rtol=0
    )


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_argmax_writes_into_strided_caller_buffer(out_dtype):
    M, N = 8, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    storage = torch.empty(M * 2, device="cuda", dtype=out_dtype)
    out = storage[::2]
    returned = argmax_impl.argmax(x, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.long(), torch.argmax(x, dim=-1), atol=0, rtol=0)


# M=16 routes through split scratch; M=256 is a one-stage launch without it.
@pytest.mark.parametrize("M", [16, 256])
def test_argmax_out_buffer_under_cuda_graph(M):
    N = MODEL_VOCABS["dsv4"]
    torch.manual_seed(M ^ N ^ 0xC0DE)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, dtype=torch.int32, device="cuda")

    argmax_impl.argmax(x, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        argmax_impl.argmax(x, out=out)

    new_x = 0.1 * torch.randn_like(x)
    x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.long(), torch.argmax(x, dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N",
    [(1, 4097), (5, 129281), (65, 154880), (513, 8192), (513, 65537)],
)
def test_argmax_strided_rows_and_partial_tiles(M, N):
    x = torch.randn(M, N + 7, device="cuda", dtype=torch.float32)[:, :N]
    x[:, -1] = 100.0
    torch.testing.assert_close(
        argmax_impl.argmax(x, out=None), torch.argmax(x, dim=-1), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
@pytest.mark.parametrize("M", [129, 513])
@pytest.mark.parametrize("N", [16385, 129281])
def test_argmax_masks_padding_with_negative_logits(M, N, dtype):
    # Offset each row and fill its padding with values that must not win.
    out_dtype = torch.int32
    storage = torch.full((M, N + 8), float("inf"), device="cuda", dtype=dtype)
    x = storage[:, 1 : N + 1]
    x.fill_(-2.0)
    x[:, -1] = -1.0
    x[0].fill_(float("nan"))
    x[1].fill_(-float("inf"))
    x[2, N // 2] = -1.0

    output_storage = torch.full((M * 2,), -7, device="cuda", dtype=out_dtype)
    out = output_storage[::2]
    returned = argmax_impl.argmax(x, out=out)
    assert returned is out
    expected = torch.full((M,), N - 1, device="cuda", dtype=out_dtype)
    expected[:3] = torch.tensor([-1, 0, N // 2], device="cuda", dtype=out_dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.all(output_storage[1::2] == -7)


@pytest.mark.parametrize("N", MODEL_VOCABS.values(), ids=MODEL_VOCABS.keys())
@pytest.mark.parametrize("M", [4, 16, 128, 512])
def test_argmax_ties_across_splits_and_infinities(M, N):
    x = torch.full((M, N), float("nan"), device="cuda")
    x[0, N // 32] = x[0, N - 1] = float("inf")
    x[1, N - 1] = -float("inf")
    x[2, N // 16] = x[2, N // 2] = 10.0
    expected = torch.full((M,), -1, device="cuda", dtype=torch.int64)
    expected[:3] = torch.tensor([N // 32, N - 1, N // 16], device="cuda")
    torch.testing.assert_close(
        argmax_impl.argmax(x, out=None), expected, atol=0, rtol=0
    )


@pytest.mark.parametrize("kind", ["small", "column_stride", "float64", "cpu", "empty"])
def test_argmax_unsupported_inputs_preserve_torch_fallback(kind):
    x = torch.randn(4, 8192, device="cuda")
    if kind == "small":
        x = x[:, :17]
    elif kind == "column_stride":
        x = x[:, ::2]
    elif kind == "float64":
        x = x.double()
    elif kind == "cpu":
        x = x.cpu()
    elif kind == "empty":
        x = x[:0]
    torch.testing.assert_close(
        argmax_impl.argmax(x, out=None), torch.argmax(x, dim=-1), atol=0, rtol=0
    )


@pytest.mark.parametrize("kind", ["shape", "dtype", "device"])
def test_argmax_rejects_invalid_output(kind):
    x = torch.empty((4, 4096), device="cuda")
    if kind == "shape":
        out = torch.empty(5, dtype=torch.int64, device="cuda")
    elif kind == "dtype":
        out = torch.empty(4, dtype=torch.float32, device="cuda")
    else:
        out = torch.empty(4, dtype=torch.int64, device="cpu")
    with pytest.raises(ValueError):
        argmax_impl.argmax(x, out=out)


def test_argmax_public_dispatch_and_repeated_graph_calls():
    from tokenspeed_kernel.ops.sampling import argmax
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    M, N = 4, MODEL_VOCABS["dsv4"]
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    outputs = [torch.empty(M, device="cuda", dtype=torch.int32) for _ in range(3)]
    selected = select_kernel(
        "sampling",
        "argmax",
        format_signature(logits=dense_tensor_format(x.dtype)),
        solution=None,
        override=None,
    )
    assert selected.name == f"gluon_argmax_{_ARCH}"
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        argmax(x, out=outputs[0], solution=None, override=None)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for out in outputs:
            returned = argmax(x, out=out, solution=None, override=None)
            assert returned is out
    for _ in range(10):
        x.normal_()
        graph.replay()
        expected = torch.argmax(x, dim=-1)
        for out in outputs:
            torch.testing.assert_close(out.long(), expected, atol=0, rtol=0)


def _scratch_cache_keys_for(logits, stream):
    prefix = (logits.device.index, stream.cuda_stream, logits.shape[0])
    return [key for key in argmax_impl._scratch_cache if key[:3] == prefix]


def test_argmax_concurrent_streams_use_independent_scratch():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    xs = [torch.randn(4, MODEL_VOCABS["dsv4"], device="cuda") for _ in streams]
    outputs = [torch.empty(4, dtype=torch.int64, device="cuda") for _ in streams]
    scratch_pointers = []
    for stream, x, out in zip(streams, xs, outputs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            argmax_impl.argmax(x, out=out)
        keys = _scratch_cache_keys_for(x, stream)
        assert keys
        scratch_pointers.append(
            {
                tensor.data_ptr()
                for key in keys
                for tensor in argmax_impl._scratch_cache[key]
            }
        )
    # Check ownership directly even if short kernels happen to run serially.
    assert scratch_pointers[0].isdisjoint(scratch_pointers[1])
    for _ in range(20):
        for stream, x, out in zip(streams, xs, outputs):
            with torch.cuda.stream(stream):
                argmax_impl.argmax(x, out=out)
    for stream in streams:
        stream.synchronize()
    for x, out in zip(xs, outputs):
        torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


def test_argmax_first_capture_does_not_cache_graph_pool_scratch():
    x = torch.randn(7, MODEL_VOCABS["dsv4"], device="cuda")
    out = torch.empty(7, dtype=torch.int64, device="cuda")
    # Compile on a different stream, leaving the capture stream's scratch cold.
    argmax_impl.argmax(x, out=out)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    for key in _scratch_cache_keys_for(x, stream):
        argmax_impl._scratch_cache.pop(key)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        argmax_impl.argmax(x, out=out)
    assert not _scratch_cache_keys_for(x, stream)
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        argmax_impl.argmax(x, out=out)
    stream.synchronize()
    assert _scratch_cache_keys_for(x, stream)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)
