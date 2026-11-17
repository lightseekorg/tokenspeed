# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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

"""PLE kernel correctness and exact-remainder CUDA-graph A/B benchmarks.

Normal: pytest test/nvidia/ops/test_ple_prototypes.py -s
Minimal dependency setup: python test_ple_prototypes.py -s
The latter loads the real PLE facade without initializing unrelated ops.
PDL-on cases require NVIDIA SM90+; PDL-off cases also run on AMD GPUs.
"""

import pathlib
import statistics
import sys
import types

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[3] / "python" / "tokenspeed_kernel"
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    for name, path in [
        ("tokenspeed_kernel", root),
        ("tokenspeed_kernel.ops", root / "ops"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    sys.modules["tokenspeed_kernel"].ops = sys.modules["tokenspeed_kernel.ops"]

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.numerics.torch.ple import torch_ple_gate_norm
from tokenspeed_kernel.ops.ple import (
    ple_conv_sequences,
    ple_gate_norm,
    ple_ngram_ids,
    prepare_ngram_reciprocals,
)
from tokenspeed_kernel.ops.ple.triton import _exact_remainder
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PLE kernel tests require a GPU"
)


@pytest.fixture(params=[False, True])
def pdl(request, monkeypatch):
    enabled = request.param
    # Check capability before overriding the production platform guard.
    if enabled and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA SM90+")
    monkeypatch.setattr("tokenspeed_kernel.ops.ple.pdl_enabled", lambda: enabled)
    return enabled


def _measure(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [fn() for _ in range(32)]
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        for _ in range(20):
            graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / (20 * 32))
    assert outputs
    return statistics.median(samples)


def _gate_inputs(tokens, d, dtype):
    torch.manual_seed(123)
    hc = 4
    # Projection split views: row stride exceeds logical width.
    projection = torch.randn(tokens, (hc + 1) * d, device="cuda", dtype=dtype)
    key, value = projection[:, : hc * d], projection[:, hc * d :]
    query = torch.randn_like(key)
    weights = [
        1 + torch.randn(hc * d, device="cuda", dtype=dtype) * 0.1 for _ in range(3)
    ]
    return key, query, value, *weights


def _gate_reference(inputs, d):
    return torch_ple_gate_norm(*inputs, d)


@pytest.mark.parametrize("tokens,d", [(1, 256), (1, 2560), (4, 2560), (16, 2560)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate(tokens, d, dtype, pdl):
    inputs = _gate_inputs(tokens, d, dtype)
    opts = dict(hc_count=4, hidden_size=d, eps=1e-6)
    expected = _gate_reference(inputs, d)
    actual = ple_gate_norm(*inputs, **opts)
    for ref, result in zip(expected, actual):
        # The fused GPU kernel differs from eager BF16 at rounding boundaries.
        ref_tol = 2e-5 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(result, ref, rtol=ref_tol, atol=ref_tol)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ple_gate_norm(*inputs, **opts)
    graph.replay()
    for a, b in zip(captured, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@triton.jit
def _remainder_test(x, ds, rs, out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x + i, i < N, other=0)
    d = tl.load(ds + i, i < N, other=1)
    r = tl.load(rs + i, i < N, other=0)
    tl.store(out + i, _exact_remainder(v, d, r), i < N)


def test_remainder_boundaries():
    import random

    rng = random.Random(13)
    divisors = [1, 2, 3, 7, 1013, 19997, 2**31 - 1, 2**32 + 15, 2**62 + 1, 2**63 - 1]
    divisors += [rng.randrange(1, 2**63) for _ in range(100)]
    pairs = [
        (n, d)
        for d in divisors
        for n in [
            0,
            1,
            d - 1,
            d,
            min(d + 1, 2**63 - 1),
            2**63 - 1,
            rng.randrange(2**63),
        ]
    ]
    x = torch.tensor([n for n, _ in pairs], device="cuda", dtype=torch.int64)
    ds = torch.tensor([d for _, d in pairs], device="cuda", dtype=torch.int64)
    rs = prepare_ngram_reciprocals([d for _, d in pairs], device=x.device)
    out = torch.empty_like(x)
    _remainder_test[(triton.cdiv(x.numel(), 128),)](x, ds, rs, out, x.numel(), 128)
    assert torch.equal(out, torch.remainder(x, ds))


def _ngram_case(lengths, hpn):
    n = 3
    total = sum(lengths)
    torch.manual_seed(43)
    ids = torch.randint(0, 50000, (total,), device="cuda")
    if total:
        ids[::3] = 7
    init = torch.randint(0, 50000, (len(lengths), n - 1), device="cuda")
    sizes = [1, 3, 1013, 19997, 2**31 - 1, 2**32 + 15, 2**62 + 1, 2**63 - 1] * hpn
    sizes = sizes[: (n - 1) * hpn]
    lens = torch.tensor(lengths, device="cuda")
    req = torch.repeat_interleave(torch.arange(len(lengths), device="cuda"), lens)
    col = torch.tensor(
        [c for length in lengths for c in range(length)],
        device="cuda",
        dtype=torch.int64,
    )
    starts = lens.cumsum(0) - lens
    mult = torch.tensor([12345678901, 31415926535, 27182818284], device="cuda")
    sizes_gpu = torch.tensor(sizes, device="cuda")
    offsets = torch.zeros(len(sizes), device="cuda", dtype=torch.int64)
    reciprocal = prepare_ngram_reciprocals(sizes, device=ids.device)
    args = (ids, init, req, col, lens, starts, mult, sizes_gpu, offsets)
    opts = dict(ngram_size=n, heads_per_ngram=hpn, eos_token_id=7, uniform_length=0)
    return args, opts, reciprocal


@pytest.mark.parametrize("lengths", [[1], [4], [0, 4, 2], [0, 0]])
@pytest.mark.parametrize("hpn", [3, 8])
def test_ngram(lengths, hpn, pdl):
    args, opts, reciprocal = _ngram_case(lengths, hpn)
    baseline = ple_ngram_ids(*args, **opts, mod_reciprocals=None, need_tail=True)
    actual = ple_ngram_ids(*args, **opts, mod_reciprocals=reciprocal, need_tail=True)
    for a, b in zip(actual, baseline):
        assert torch.equal(a, b)
    stride = max(lengths) + 1
    scratch = torch.full(
        ((len(lengths) + 1) * stride, 2), -1, device="cuda", dtype=torch.int64
    )
    graph = torch.cuda.CUDAGraph()

    # Warm the scatter specialization before capture.
    def run():
        return ple_ngram_ids(
            *args,
            **opts,
            mod_reciprocals=reciprocal,
            tail_out=scratch,
            tail_block_rows=stride,
        )

    run()
    with torch.cuda.graph(graph):
        out, _ = run()
    scratch.fill_(-1)
    graph.replay()
    assert torch.equal(out, baseline[0])
    assert torch.equal(
        scratch[torch.arange(len(lengths), device="cuda") * stride], args[1]
    )
    assert torch.equal(scratch[args[2] * stride + 1 + args[3]], baseline[1])
    assert (scratch[len(lengths) * stride :] == -1).all()


def _conv_case(lengths, *, windows):
    torch.manual_seed(73)
    channels, state = 37, 9
    total, batch = sum(lengths), len(lengths)
    values = torch.randn(total, channels, device="cuda") * 0.1
    initial = torch.randn(batch, channels, state, device="cuda") * 0.1
    weight = torch.randn(channels, 4, device="cuda") * 0.1
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int64)
    req = torch.repeat_interleave(torch.arange(batch, device="cuda"), lens)
    col = torch.tensor(
        [i for length in lengths for i in range(length)],
        device="cuda",
        dtype=torch.int64,
    )
    starts = lens.cumsum(0) - lens
    stride = max(lengths) + 1
    scratch = (
        torch.full((batch * stride, channels, state), -1.0, device="cuda")
        if windows
        else None
    )
    args = (values, initial, weight, req, col, lens, starts)
    opts = dict(
        total_tokens=total,
        batch_size=batch,
        dilation=3,
        kernel_size=4,
        state_len=state,
        weights_independent=True,
        compact_windows=False,
        windows=scratch,
        windows_block_rows=stride if windows else 0,
        scatter_windows=windows,
    )
    return args, opts


@pytest.mark.parametrize("lengths", [[1], [4], [2, 0, 3], [0, 0, 0], [1, 0, 0]])
@pytest.mark.parametrize(
    "write_final,windows", [(True, False), (True, True), (False, True)]
)
def test_conv_runtime_bounds(lengths, write_final, windows, pdl):
    args, opts = _conv_case(lengths, windows=windows)
    values, initial, weight = args[:3]
    expected = torch.empty_like(values)
    expected_final = torch.empty_like(initial)
    expected_windows = opts["windows"].clone() if windows else None
    start = 0
    for request, length in enumerate(lengths):
        sequence = torch.cat(
            (initial[request], values[start : start + length].T), dim=1
        )
        expected_final[request] = sequence[:, length : length + 9]
        if windows:
            base = request * opts["windows_block_rows"]
            expected_windows[base] = initial[request]
        for column in range(length):
            acc = (sequence[:, column : column + 10 : 3] * weight).sum(dim=1)
            expected[start + column] = acc * torch.sigmoid(acc)
            if windows:
                expected_windows[base + 1 + column] = sequence[
                    :, column + 1 : column + 10
                ]
        start += length

    def run():
        return ple_conv_sequences(*args, **opts, write_final=write_final)

    eager = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    if windows:
        opts["windows"].fill_(-1)
    graph.replay()
    for output, final, scratch in (eager, captured):
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)
        if write_final:
            torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
        else:
            assert final.shape[0] == 0
        if windows:
            torch.testing.assert_close(scratch, expected_windows, rtol=0, atol=0)


def test_conv_token_tiles_across_request_boundaries():
    if not current_platform().is_blackwell:
        pytest.skip("token-tiled convolution requires NVIDIA SM100 or SM103")

    lengths, channels, state = (521, 0, 510), 10240, 9
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    values = torch.randn(sum(lengths), channels, **opts)
    initial = torch.randn(len(lengths), channels, state, **opts)
    weight = torch.randn(channels, 4, **opts)
    gated, residual = [
        torch.randn(sum(lengths), channels + 16, **opts)[:, :channels] for _ in range(2)
    ]

    def run(lengths, values, initial, gated, residual):
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int64)
        starts = lens.cumsum(0) - lens
        req = torch.arange(len(lengths), device="cuda").repeat_interleave(lens)
        col = torch.arange(values.shape[0], device="cuda") - starts[req]
        return ple_conv_sequences(
            values,
            initial,
            weight,
            req,
            col,
            lens,
            starts,
            total_tokens=values.shape[0],
            batch_size=len(lengths),
            dilation=3,
            kernel_size=4,
            state_len=state,
            write_final=True,
            weights_independent=True,
            compact_windows=False,
            add_terms=(gated, residual),
        )[:2]

    reference = [
        run((length,), x, y[None], g, r)
        for length, x, y, g, r in zip(
            lengths,
            values.split(lengths),
            initial,
            gated.split(lengths),
            residual.split(lengths),
            strict=True,
        )
    ]
    result = run(lengths, values, initial, gated, residual)
    for actual, expected in zip(result, zip(*reference), strict=True):
        torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dim, dtype",
    [(160, torch.float8_e4m3fn), (512, torch.bfloat16)],
    ids=["flash-fp8", "max-bf16"],
)
def test_host_cache_keeps_aligned_and_tail_rows_until_invalidated(dim, dtype):
    from tokenspeed_kernel.ops.ple._host_cache_triton import (
        _ple_fill_staged_kernel,
        _ple_gather_staged_tail_kernel,
    )
    from tokenspeed_kernel.ops.ple.host_cache import PLEHostCache
    from utils import assert_no_triton_compile

    if not current_platform().is_nvidia or torch.cuda.get_device_capability()[
        0
    ] not in (9, 10):
        pytest.skip("PLE host cache requires Hopper or Blackwell")
    device = torch.device("cuda", torch.cuda.current_device())
    vocab_start = 20_000_000
    table = torch.ones(32, dim)
    table[1].fill_(2)
    table = table.to(dtype).pin_memory()
    scale = 0.125 if dtype == torch.float8_e4m3fn else None
    cache = PLEHostCache(
        slot_count=512,
        table=table,
        ngram_heads=16,
        output_dtype=torch.bfloat16,
        vocab_start=vocab_start,
        vocab_end=vocab_start + len(table),
        device=device,
    )

    def run(tokens, lookup_scale=scale):
        ids = torch.full((tokens, 16), vocab_start, device=device)
        ids[:, 1] += len(table)
        ids[-1, 0] += 1
        out = torch.empty((ids.numel(), dim), device=device, dtype=torch.bfloat16)
        assert cache.can_gather(ids)
        return cache.gather(ids, out, scale=lookup_scale).cpu().view(tokens, 16, dim)

    def expected(tokens):
        output = torch.full((tokens, 16, dim), scale or 1.0, dtype=torch.bfloat16)
        output[:, 1] = 0
        output[-1, 0] *= 2
        return output

    torch.testing.assert_close(run(257), expected(257), rtol=0, atol=0)
    table.view(torch.uint8).zero_()
    if scale is not None:
        torch.testing.assert_close(
            run(257, scale * 2), expected(257) * 2, rtol=0, atol=0
        )
    with assert_no_triton_compile(
        _ple_gather_staged_tail_kernel, _ple_fill_staged_kernel
    ):
        torch.testing.assert_close(run(259), expected(259), rtol=0, atol=0)
    cache.invalidate()
    assert not run(259).any()


def test_host_cache_reports_initialization_failures(monkeypatch, tmp_path):
    from tokenspeed_kernel.ops.ple.host_cache import PLEHostCache
    from tokenspeed_kernel.thirdparty.cuda import ple_host_gather as native

    table = torch.ones(32, 160, dtype=torch.bfloat16).pin_memory()
    args = dict(
        slot_count=512,
        table=table,
        ngram_heads=16,
        output_dtype=torch.bfloat16,
        vocab_start=100,
        vocab_end=132,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    for override, message in (
        ({"slot_count": 0}, "must be a power of two"),
        ({"device": torch.device("cpu")}, "NVIDIA CUDA"),
        ({"table": table.flatten()}, "shape"),
        ({"table": torch.ones_like(table, pin_memory=False)}, "pinned CPU"),
        ({"table": table[:, ::2]}, "contiguous"),
        ({"vocab_end": 133}, "shard bounds"),
        ({"ngram_heads": 8}, "supports 16 heads"),
        ({"table": table[:, :128].contiguous().pin_memory()}, "head dimensions"),
        ({"table": table.float().pin_memory()}, "storage"),
        ({"output_dtype": torch.float16}, "BF16 output"),
    ):
        with pytest.raises((ValueError, RuntimeError), match=message):
            PLEHostCache(**(args | override))
    native.ple_staged_gather.cache_clear()
    monkeypatch.setattr(native, "_MODULE_PATH", tmp_path / "missing.so")
    try:
        with pytest.raises(RuntimeError, match="Reinstall tokenspeed-kernel"):
            PLEHostCache(**args)
    finally:
        native.ple_staged_gather.cache_clear()


def test_conv_state_large_request_offset(pdl):
    channels, state = 4097, 9
    batch = 2**31 // (channels * state) + 2
    carried = torch.randn((1, channels, state), device="cuda", dtype=torch.bfloat16)
    initial = torch.empty((batch, channels, state), device="cuda", dtype=torch.bfloat16)
    initial[-1].copy_(carried[0])
    values = torch.randn((1, channels), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((channels, 4), device="cuda", dtype=torch.bfloat16)
    col = torch.zeros(1, device="cuda", dtype=torch.int64)
    lengths = torch.zeros(batch, device="cuda", dtype=torch.int64)
    lengths[-1] = 1
    starts = torch.zeros_like(lengths)

    def run(history, request, lens, offsets, count):
        windows = torch.empty((1, channels, state), device="cuda", dtype=torch.bfloat16)
        return ple_conv_sequences(
            values,
            history,
            weight,
            request,
            col,
            lens,
            offsets,
            total_tokens=1,
            batch_size=count,
            dilation=3,
            kernel_size=4,
            state_len=state,
            write_final=False,
            weights_independent=True,
            compact_windows=False,
            windows=windows,
            scatter_windows=False,
        )

    expected = run(carried, col, lengths[-1:], col, 1)
    actual = run(initial, col + batch - 1, lengths, starts, batch)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


def test_conv_reuses_specialization_across_shapes(pdl, monkeypatch):
    import tokenspeed_kernel.ops.ple as ple

    kernel = ple._ple_conv_state_kernel
    compiled = []

    class LaunchRecorder:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                result = kernel[grid](*args, **kwargs)
                compiled.append(result)
                return result

            return launch

    monkeypatch.setattr(ple, "_ple_conv_state_kernel", LaunchRecorder())
    for total, batch in [(3, 3), (5, 3), (7, 3), (3, 5), (3, 7)]:
        args, opts = _conv_case([total] + [0] * (batch - 1), windows=False)
        ple_conv_sequences(*args, **opts, write_final=True)
    torch.cuda.synchronize()
    assert compiled[0] is not None
    assert all(item is compiled[0] for item in compiled)


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_performance(tokens, pdl):
    args, opts, reciprocal = _ngram_case([tokens], 8)
    # Use embedding-sized moduli for timing, not the adversarial correctness mix.
    sizes = [1_000_003 + 2 * i for i in range(16)]
    args = (*args[:7], torch.tensor(sizes, device="cuda"), args[8])
    reciprocal = prepare_ngram_reciprocals(sizes, device=args[0].device)
    old = _measure(lambda: ple_ngram_ids(*args, **opts, mod_reciprocals=None))
    new = _measure(lambda: ple_ngram_ids(*args, **opts, mod_reciprocals=reciprocal))
    print(
        f"NGRAM tokens={tokens} pdl={pdl} div_us={old:.3f} reciprocal_us={new:.3f} speedup={old/new:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--confcutdir",
                str(pathlib.Path(__file__).parent),
                *sys.argv[1:],
            ]
        )
    )
