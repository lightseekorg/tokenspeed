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

"""GFX1250 FP8 MLA prefill launch paths: key splits and short prefixes."""

from __future__ import annotations

from itertools import accumulate

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 Gluon MLA tests", allow_module_level=True
    )

from tokenspeed_kernel_amd.ops.gfx1250.attention.mla import prefill  # noqa: E402

_SOFTMAX_SCALE = 192**-0.5


def _run(device, q_lens, kv_lens, is_causal):
    init = dict(device=device, dtype=torch.bfloat16)
    q = torch.randn(sum(q_lens), 12, 192, **init).to(torch.float8_e4m3fn)
    k = torch.randn(sum(kv_lens), 12, 192, **init).to(torch.float8_e4m3fn)
    v = torch.randn(sum(kv_lens), 12, 128, **init).to(torch.float8_e4m3fn)
    out, lse = prefill.launch_gluon_mla_prefill_gfx1250(
        q,
        k,
        v,
        torch.tensor([0, *accumulate(q_lens)], device=device).int(),
        torch.tensor([0, *accumulate(kv_lens)], device=device).int(),
        max(q_lens),
        max(kv_lens),
        _SOFTMAX_SCALE,
        is_causal=is_causal,
        return_lse=True,
    )
    return q, k, v, out, lse


@pytest.mark.parametrize(
    "q_lens,kv_lens,is_causal,split_kv",
    [
        pytest.param((300,), (5000,), False, True, id="split-prefix"),
        pytest.param(
            (200, 700), (3000, 0), False, True, id="split-prefix-and-new-sequence"
        ),
        pytest.param(
            (128, 128), (32768, 2048), False, True, id="split-unequal-prefixes"
        ),
        pytest.param((2048,), (2048,), True, True, id="split-causal-self"),
        pytest.param((1000,), (3000,), True, True, id="split-causal-with-prefix"),
        pytest.param(
            (200, 700), (560, 0), False, False, id="short-prefix-then-new-sequence"
        ),
        pytest.param(
            (700, 200), (0, 560), False, False, id="short-new-sequence-then-prefix"
        ),
        pytest.param(
            (128, 384, 256),
            (1000, 64, 1024),
            False,
            False,
            id="short-unequal-prefixes",
        ),
    ],
)
def test_launch_paths(
    device: str,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    is_causal: bool,
    split_kv: bool,
) -> None:
    """Underfilled launches split keys; short prefixes use memory clauses."""
    q, k, v, out, lse = _run(device, q_lens, kv_lens, is_causal)
    config = prefill.get_config(
        q=q,
        k=k,
        cu_seqlens_q=torch.zeros(len(q_lens) + 1),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        softmax_scale=1.0,
        is_causal=is_causal,
    )
    assert config.split_kv == split_kv
    assert config.llvm_fn_attrs == (
        None if split_kv else "amdgpu-sched-strategy=max-memory-clause"
    )

    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        rows = slice(q_offset, q_offset + q_len)
        ref, ref_lse = torch_dense_attention(
            q[rows].float(),
            k[kv_offset : kv_offset + kv_len].float(),
            v[kv_offset : kv_offset + kv_len].float(),
            scale=_SOFTMAX_SCALE,
            causal_offset=kv_len - q_len if is_causal else None,
            window_left=None,
            sinks=None,
        )
        torch.testing.assert_close(out[rows].float(), ref, rtol=2e-1, atol=2e-1)
        torch.testing.assert_close(lse[rows], ref_lse, rtol=8e-2, atol=8e-2)
        q_offset += q_len
        kv_offset += kv_len


def test_reuses_kernels_across_splits_and_sequences(device: str) -> None:
    """Split counts and sequence counts are runtime values, not constexprs."""
    # Warm both kernels of each mode with and without splits, and the
    # short-prefix scheduling.
    for is_causal in (False, True):
        _run(device, (853, 1045), (853, 1045), is_causal)
        _run(device, (300,), (5000,), is_causal)
    _run(device, (300,), (500,), False)
    with assert_no_triton_compile(
        prefill.gluon_mla_prefill_gfx1250, prefill.gluon_mla_prefill_merge_gfx1250
    ):
        for q_lens, kv_lens in (
            ((848,), (32768,)),
            ((848, 7344), (16384, 0)),
            ((8192,), (16384,)),
            ((257, 31, 1000), (4097, 65, 12000)),
            ((288, 7904), (560, 0)),
            ((128, 128, 128), (64, 1024, 0)),
        ):
            _run(device, q_lens, kv_lens, False)
        for q_lens in ((848,), (2048,), (5376,), (8192,)):
            _run(device, q_lens, q_lens, True)
    torch.cuda.synchronize()
