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

"""Ragged MLA prefill numerical, compilation and graph regressions."""

import sys
import warnings
from contextlib import contextmanager

import pytest
import torch
from cutlass.base_dsl.dsl import BaseDSL
from tokenspeed_mla.mla_prefill import (
    _compile_prefill_kernel,
    tokenspeed_mla_prefill,
    warmup_compile_prefill,
)


@contextmanager
def assert_no_cute_compile():
    """Observe real CuTe tracing without adding a kernel-package dependency."""
    compile_code = BaseDSL.generate_mlir.__code__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is compile_code:
            raise AssertionError("CuTe traced again after prefill warmup")
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)


class TestCompile:
    @pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_prefill_architectures(self, capability, causal):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _compile_prefill_kernel(
                torch.float8_e4m3fn,
                192,
                128,
                causal,
                True,
                use_pdl=False,
                enable_ex2_emulation=False,
                compute_capability=capability,
            )


def _inputs(q_lens, kv_lens, heads, kv_heads):
    generator = torch.Generator(device="cuda").manual_seed(42)
    tensors = [
        torch.randn(shape, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        for shape in (
            (sum(q_lens), heads, 192),
            (sum(kv_lens), kv_heads, 192),
            (sum(kv_lens), kv_heads, 128),
        )
    ]
    cumulative = [
        torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        for lens in (q_lens, kv_lens)
    ]
    return (*tensors, *cumulative)


def _reference(q, k, v, q_lens, kv_lens, causal):
    outputs, lses = [], []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        qi = q[q_start : q_start + q_len].float().cpu().transpose(0, 1)
        ki = k[kv_start : kv_start + kv_len].float().cpu().transpose(0, 1)
        vi = v[kv_start : kv_start + kv_len].float().cpu().transpose(0, 1)
        repeat = q.shape[1] // k.shape[1]
        ki, vi = ki.repeat_interleave(repeat, 0), vi.repeat_interleave(repeat, 0)
        scores = qi @ ki.transpose(1, 2) * 192**-0.5
        if causal:
            visible = torch.arange(q_len) + kv_len - q_len
            mask = torch.arange(kv_len)[None, :] <= visible[:, None]
            scores.masked_fill_(~mask, -torch.inf)
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).nan_to_num()
        outputs.append((probs @ vi).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
        q_start += q_len
        kv_start += kv_len
    return torch.cat(outputs).to(q.device), torch.cat(lses).to(q.device)


def _launch(inputs, q_lens, kv_lens, causal, lse, pdl, out=None):
    q, k, v, cum_q, cum_kv = inputs
    return tokenspeed_mla_prefill(
        q,
        k,
        v,
        cum_kv[1:] - cum_kv[:-1],
        cum_kv,
        max(kv_lens),
        len(q_lens),
        192**-0.5,
        is_causal=causal,
        return_lse=lse,
        cum_seq_lens_q=cum_q,
        max_seq_len_q=max(q_lens),
        enable_pdl=pdl,
        out=out,
    )


def _check(actual, reference):
    out, lse = actual
    expected, expected_lse = reference
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    # P is rounded to E4M3 before PV; cancellation needs an absolute bound.
    torch.testing.assert_close(out.float(), expected, atol=0.06, rtol=0.03)
    torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)
    error = (out.float() - expected).square().sum()
    assert error <= 0.001 * expected.square().sum().clamp_min(1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires SM100, SM103 or SM107",
)
class TestGPU:
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("pdl", [False, True])
    @pytest.mark.parametrize(
        "q_lens,kv_lens,heads,kv_heads",
        [
            ([1, 127, 257], [17, 259, 513], 4, 4),
            ([257, 0, 129], [257, 13, 129], 12, 12),
            ([33, 129, 1], [0, 513, 1], 4, 4),
            ([257], [4097], 96, 96),
            ([129, 3], [513, 257], 8, 2),
            ([33, 129], [0, 0], 4, 4),
            ([513, 3], [17, 1], 4, 4),
        ],
    )
    def test_ragged(self, q_lens, kv_lens, heads, kv_heads, causal, pdl):
        inputs = _inputs(q_lens, kv_lens, heads, kv_heads)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = _launch(inputs, q_lens, kv_lens, causal, True, pdl)
        _check(actual, _reference(*inputs[:3], q_lens, kv_lens, causal))

    @pytest.mark.parametrize("causal", [False, True])
    def test_output_reuse_without_lse(self, causal):
        q_lens, kv_lens = [129, 3], [513, 17]
        inputs = _inputs(q_lens, kv_lens, 4, 4)
        out = torch.full(
            (sum(q_lens), 4, 128), torch.nan, device="cuda", dtype=torch.bfloat16
        )
        actual = _launch(inputs, q_lens, kv_lens, causal, False, False, out)
        assert actual.data_ptr() == out.data_ptr()
        expected = _reference(*inputs[:3], q_lens, kv_lens, causal)[0]
        torch.testing.assert_close(actual.float(), expected, atol=0.06, rtol=0.03)

    @pytest.mark.parametrize("pdl", [False, True])
    def test_graph_and_shape_reuse(self, pdl):
        warmup_compile_prefill(enable_pdl=pdl)
        with assert_no_cute_compile():
            for q_lens, kv_lens, heads, kv_heads in (
                ([127, 3], [259, 17], 4, 4),
                ([257, 129, 1], [513, 257, 0], 12, 12),
                ([129, 3], [513, 257], 8, 2),
            ):
                inputs = _inputs(q_lens, kv_lens, heads, kv_heads)
                _launch(inputs, q_lens, kv_lens, True, True, pdl)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = _launch(inputs, q_lens, kv_lens, True, True, pdl)
                # Replaying must consume the updated input, including unaligned tails.
                inputs[0].copy_((inputs[0].float() * 0.5).to(inputs[0].dtype))
                graph.replay()
                _check(actual, _reference(*inputs[:3], q_lens, kv_lens, True))

    def test_large_packed_offsets(self):
        # Only the final row is live. Both Q and O element offsets exceed
        # 2**31 without computing attention over the unused allocation.
        tokens = 2**31 // 128 + 2
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            pytest.skip("requires 12 GiB free virtual tensor storage")
        q = torch.empty((tokens, 1, 192), device="cuda", dtype=torch.float8_e4m3fn)
        v = torch.empty((tokens, 1, 128), device="cuda", dtype=q.dtype)
        q[-1:].fill_(1)
        v[-1:].fill_(0.5)
        cum = torch.tensor([tokens - 1, tokens], device="cuda", dtype=torch.int32)
        out, lse = tokenspeed_mla_prefill(
            q,
            q,
            v,
            torch.ones(1, device="cuda", dtype=torch.int32),
            cum,
            1,
            1,
            192**-0.5,
            is_causal=True,
            return_lse=True,
            cum_seq_lens_q=cum,
            max_seq_len_q=1,
            enable_pdl=False,
        )
        torch.testing.assert_close(
            out[-1:].float(), torch.full_like(out[-1:], 0.5).float(), atol=0, rtol=0
        )
        torch.testing.assert_close(
            lse[-1:], torch.full_like(lse[-1:], 192**0.5), atol=2e-5, rtol=2e-5
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_existing_input_dtypes(self, dtype):
        q_lens, kv_lens = [127, 3], [259, 17]
        raw = _inputs(q_lens, kv_lens, 4, 4)
        inputs = tuple(t.to(dtype) for t in raw[:3]) + raw[3:]
        actual = _launch(inputs, q_lens, kv_lens, True, True, False)
        expected, expected_lse = _reference(*inputs[:3], q_lens, kv_lens, True)
        torch.testing.assert_close(actual[0].float(), expected, atol=0.005, rtol=0.005)
        torch.testing.assert_close(actual[1], expected_lse, atol=2e-5, rtol=2e-5)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
    def test_query_device_controls_compile_and_launch(self):
        q_lens, kv_lens = [127, 3], [259, 17]
        raw = _inputs(q_lens, kv_lens, 4, 4)
        inputs = tuple(t.to("cuda:1") for t in raw[:3]) + tuple(
            t.cpu() for t in raw[3:]
        )
        with torch.cuda.device(0):
            actual = _launch(inputs, q_lens, kv_lens, True, True, False)
            assert torch.cuda.current_device() == 0
        _check(actual, _reference(*inputs[:3], q_lens, kv_lens, True))

    def test_strided_inputs(self):
        q_lens, kv_lens = [127, 3], [259, 17]
        raw = _inputs(q_lens, kv_lens, 4, 4)
        views = []
        for src in raw[:3]:
            backing = torch.empty(
                (*src.shape[:-1], src.shape[-1] + 16),
                device=src.device,
                dtype=src.dtype,
            )
            view = backing[..., : src.shape[-1]]
            view.copy_(src)
            views.append(view)
        inputs = (*views, *raw[3:])
        _check(
            _launch(inputs, q_lens, kv_lens, True, True, False),
            _reference(*raw[:3], q_lens, kv_lens, True),
        )

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_unaligned_inputs_and_strided_lengths(self, dtype):
        q_lens, kv_lens = [127, 3], [259, 17]
        raw = _inputs(q_lens, kv_lens, 4, 4)
        tensors = []
        for src in raw[:3]:
            backing = torch.empty(src.numel() + 1, device=src.device, dtype=src.dtype)
            view = backing[1:].view(src.shape)
            view.copy_(src)
            tensors.append(view)
        for src in raw[3:]:
            backing = torch.empty(src.numel() * 2, device=src.device, dtype=dtype)
            view = backing[::2]
            view.copy_(src)
            tensors.append(view)
        actual = _launch(tuple(tensors), q_lens, kv_lens, True, True, False)
        _check(actual, _reference(*raw[:3], q_lens, kv_lens, True))

    @pytest.mark.parametrize(
        "case",
        [
            "dtype",
            "device",
            "kv_shape",
            "qk_dim",
            "heads",
            "cum_shape",
            "length_dtype",
            "unaligned_out",
        ],
    )
    def test_invalid_arguments(self, case):
        inputs = list(_inputs([127, 3], [259, 17], 4, 4))
        out = None
        if case == "dtype":
            inputs[1] = inputs[1].to(torch.float16)
        elif case == "device":
            inputs[1] = inputs[1].cpu()
        elif case == "kv_shape":
            inputs[2] = inputs[2][:-1]
        elif case == "qk_dim":
            inputs[1] = inputs[1][..., :128]
        elif case == "heads":
            inputs[0] = inputs[0][:, :3]
        elif case == "cum_shape":
            inputs[3] = inputs[3][:-1]
        elif case == "length_dtype":
            inputs[3] = inputs[3].float()
        else:
            out = torch.empty(130 * 4 * 128 + 1, device="cuda", dtype=torch.bfloat16)[
                1:
            ].view(130, 4, 128)
        with pytest.raises((TypeError, ValueError)):
            _launch(tuple(inputs), [127, 3], [259, 17], True, True, False, out)

    @pytest.mark.parametrize("batch", [0, 2])
    def test_empty_queries(self, batch):
        q, k, v, cq, ck = _inputs([0] * batch, [0] * batch, 4, 4)
        out = torch.empty((0, 4, 128), device="cuda", dtype=torch.bfloat16)
        result, lse = tokenspeed_mla_prefill(
            q,
            k,
            v,
            ck[1:] - ck[:-1],
            ck,
            0,
            batch,
            192**-0.5,
            is_causal=True,
            return_lse=True,
            cum_seq_lens_q=cq,
            max_seq_len_q=0,
            enable_pdl=False,
            out=out,
        )
        assert result is out
        assert result.shape == (0, 4, 128) and lse.shape == (0, 4)
