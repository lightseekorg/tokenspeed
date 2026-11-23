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

"""Exercise the actual MLA leaf, kernel bridge and prefill output contract."""

import sys

import pytest
import torch
from cutlass.base_dsl.dsl import BaseDSL

from tokenspeed.runtime.layers.attention.backends.paged.tokenspeed_mla import (
    CuteDSLMLABackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires SM100, SM103 or SM107",
)


def _leaf(dtype, cache_dtype, device):
    spec = MLAConfig(
        backend_name="tokenspeed_mla",
        num_attention_heads=4,
        num_kv_heads=1,
        head_dim=576,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
    )
    config = AttnConfig(
        device=device,
        dtype=dtype,
        kv_cache_dtype=cache_dtype,
        kv_cache_quant_method="",
        context_len=1024,
        max_bs=4,
        max_resident_requests=4,
        components=(spec,),
    )
    return CuteDSLMLABackend(config, spec, kernel_page_size=64)


def _inputs(dtype, device):
    generator = torch.Generator(device=device).manual_seed(31)
    q = torch.randn((130, 4, 192), device=device, generator=generator).to(dtype)
    k = torch.randn((276, 4, 192), device=device, generator=generator).to(dtype)
    # The framework's kv_b_proj output interleaves K_nope and V.
    kv = torch.randn((276, 4, 256), device=device, generator=generator).to(dtype)
    return q, k, kv[..., 128:]


def _call(leaf, inputs, out, cap):
    q, k, v = inputs
    cq = torch.tensor([0, 127, 130], device=q.device, dtype=torch.int32)
    ck = torch.tensor([0, 259, 276], device=q.device, dtype=torch.int32)
    return leaf.forward_extend_chunked(
        q,
        k,
        v,
        192**-0.5,
        cap,
        cum_seq_lens_q=cq,
        cum_seq_lens_kv=ck,
        max_q_len=127,
        max_kv_len=259,
        seq_lens=ck[1:] - ck[:-1],
        batch_size=2,
        causal=True,
        out=out,
    )


@pytest.mark.parametrize(
    "dtype,cache_dtype",
    [
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
    ],
)
@pytest.mark.parametrize("reuse", [False, True])
def test_leaf_preserves_output_and_natural_lse(dtype, cache_dtype, reuse):
    leaf = _leaf(dtype, cache_dtype, "cuda:0")
    inputs = _inputs(cache_dtype, "cuda:0")
    destination = (
        torch.full((130, 4, 128), torch.nan, device="cuda", dtype=dtype)
        if reuse
        else None
    )
    output, lse = _call(leaf, inputs, destination, None)
    assert output.dtype == dtype
    if reuse:
        assert output.data_ptr() == destination.data_ptr()
    q, k, v = [t.float().cpu() for t in inputs]
    for row, begin, end in (
        (0, 0, 133),
        (126, 0, 259),
        (127, 259, 274),
        (129, 259, 276),
    ):
        scores = torch.einsum("hd,khd->hk", q[row], k[begin:end]) * 192**-0.5
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), v[begin:end])
        tol = 0.06 if cache_dtype == torch.float8_e4m3fn else 0.005
        torch.testing.assert_close(
            output[row].float().cpu(), expected, atol=tol, rtol=0.03
        )
        torch.testing.assert_close(
            lse[row].cpu(), scores.logsumexp(-1), atol=2e-5, rtol=2e-5
        )


def test_leaf_rejects_unsupported_soft_cap():
    leaf = _leaf(torch.bfloat16, torch.float8_e4m3fn, "cuda:0")
    with pytest.raises(ValueError, match="logits_soft_cap"):
        _call(leaf, _inputs(torch.float8_e4m3fn, "cuda:0"), None, 20.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_leaf_warmup_uses_config_device():
    with torch.cuda.device(0):
        leaf = _leaf(torch.bfloat16, torch.float8_e4m3fn, "cuda:1")
        inputs = _inputs(torch.float8_e4m3fn, "cuda:1")
        compile_code = BaseDSL.generate_mlir.__code__

        def observe(frame, event, arg):
            if event == "call" and frame.f_code is compile_code:
                raise AssertionError("prefill compiled again after leaf warmup")

        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            actual = _call(leaf, inputs, None, None)
        finally:
            sys.setprofile(previous)
        assert torch.cuda.current_device() == 0
    assert actual[0].device == torch.device("cuda:1")
    assert torch.isfinite(actual[0]).all()
