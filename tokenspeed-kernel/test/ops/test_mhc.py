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

import pytest
import tokenspeed_kernel.ops.mhc.trtllm as trtllm_mhc
import torch
from tokenspeed_kernel.ops.mhc import has_trtllm_mhc, supports_trtllm_mhc
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels


def test_trtllm_mhc_registration_matches_extension_availability():
    load_builtin_kernels()
    names = (
        "trtllm_mhc_big_fuse",
        "trtllm_mhc_fused_hc",
        "trtllm_mhc_post_mapping",
    )
    registered = tuple(
        KernelRegistry.get().get_by_name(name) is not None for name in names
    )
    if has_trtllm_mhc():
        assert all(registered)
    else:
        assert not any(registered)


def test_trtllm_mhc_capability_rejects_unsupported_configs():
    assert not supports_trtllm_mhc(torch.device("cpu"), 4, 4096)
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    is_cuda_blackwell = (
        torch.version.cuda is not None
        and torch.cuda.get_device_capability(device)[0] == 10
    )
    assert not supports_trtllm_mhc(device, 2, 4096)
    assert not supports_trtllm_mhc(device, 4, 8)
    expected = is_cuda_blackwell and has_trtllm_mhc()
    assert supports_trtllm_mhc(device, 4, 4096) is expected
    assert supports_trtllm_mhc(device, 4, 7168) is expected


@pytest.mark.parametrize(
    ("num_tokens", "backend", "tile_n"),
    ((1, 3, 1), (33, 2, 0)),
)
def test_trtllm_mhc_fused_wrapper_owns_backend_selection(
    monkeypatch,
    num_tokens,
    backend,
    tile_n,
):
    if not has_trtllm_mhc():
        pytest.skip("TRT-LLM mHC extension is unavailable")

    calls = []
    monkeypatch.setattr(trtllm_mhc, "_mhc_fused_hc", lambda *args: calls.append(args))
    hc_mult, hidden_size = 4, 8
    x_prev = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16)
    residual_prev = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    post_prev = torch.empty(num_tokens, hc_mult, dtype=torch.float32)
    comb_prev = torch.empty(num_tokens, hc_mult, hc_mult, dtype=torch.float32)
    weight = torch.empty(
        hc_mult * (2 + hc_mult), hc_mult * hidden_size, dtype=torch.float32
    )
    hc_scale = torch.empty(3, dtype=torch.float32)
    hc_base = torch.empty(hc_mult * (2 + hc_mult), dtype=torch.float32)
    residual_cur = torch.empty_like(residual_prev)
    post_cur = torch.empty_like(post_prev)
    comb_cur = torch.empty_like(comb_prev)
    layer_input = torch.empty_like(x_prev)
    y_acc = torch.empty(num_tokens, hc_mult * (2 + hc_mult), dtype=torch.float32)
    r_acc = torch.empty(num_tokens, dtype=torch.float32)
    done_counter = torch.empty(num_tokens, dtype=torch.int32)

    trtllm_mhc.trtllm_mhc_fused_hc(
        x_prev,
        residual_prev,
        post_prev,
        comb_prev,
        weight,
        hc_scale,
        hc_base,
        residual_cur,
        post_cur,
        comb_cur,
        layer_input,
        y_acc,
        r_acc,
        done_counter,
        1e-6,
        1e-5,
        2,
    )

    assert len(calls) == 1
    assert calls[0][14:] == (
        num_tokens,
        hidden_size,
        hc_mult,
        1e-6,
        1e-5,
        1e-5,
        2.0,
        2,
        backend,
        tile_n,
        1,
        0,
        1,
        None,
        0.0,
    )


def test_trtllm_mhc_boundary_wrappers_own_raw_shapes(monkeypatch):
    if not has_trtllm_mhc():
        pytest.skip("TRT-LLM mHC extension is unavailable")

    big_fuse_calls = []
    post_calls = []
    monkeypatch.setattr(
        trtllm_mhc, "_mhc_big_fuse", lambda *args: big_fuse_calls.append(args)
    )
    monkeypatch.setattr(
        trtllm_mhc, "_mhc_post_mapping", lambda *args: post_calls.append(args)
    )
    num_tokens, hc_mult, hidden_size, num_splits = 3, 4, 8, 2
    residual = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    y_acc = torch.empty(
        num_splits, num_tokens, hc_mult * (2 + hc_mult), dtype=torch.float32
    )
    r_acc = torch.empty(num_splits, num_tokens, dtype=torch.float32)
    hc_scale = torch.empty(3, dtype=torch.float32)
    hc_base = torch.empty(hc_mult * (2 + hc_mult), dtype=torch.float32)
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16)
    trtllm_mhc.trtllm_mhc_big_fuse(
        y_acc,
        r_acc,
        residual,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        1e-6,
        1e-5,
        2,
    )

    hidden_states = torch.empty_like(layer_input)
    output = torch.empty_like(residual)
    trtllm_mhc.trtllm_mhc_post_mapping(
        residual,
        hidden_states,
        post_mix,
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        output,
    )

    assert big_fuse_calls[0][8:] == (
        num_tokens,
        hc_mult * hidden_size,
        hidden_size,
        1e-6,
        1e-5,
        1e-5,
        2.0,
        2,
        num_splits,
        256,
    )
    assert post_calls[0][5:] == (num_tokens, hidden_size)
