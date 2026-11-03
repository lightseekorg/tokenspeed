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

"""The sandwich norm boundary helper: when it fuses, and what it runs otherwise.

CPU only: the fused launch, the all-reduce and the Triton norms are recorded,
not run.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed.runtime.layers.layernorm as layernorm  # noqa: E402
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402

HIDDEN = 256
GROUP = (0, 1, 2, 3)


class Calls:
    def __init__(self, fused_result) -> None:
        self.log: list[tuple] = []
        self.fused_result = fused_result

    def fused(self, *args, **kwargs):
        self.log.append(("fused", args, kwargs))
        return self.fused_result

    def all_reduce(self, x, group):
        self.log.append(("all_reduce", x, group))
        return x + 1

    def norm(self, x, weight, eps, **kwargs):
        self.log.append(("norm", x, weight, eps, kwargs))
        out = x * 2
        return (out, x + 3) if "residual" in kwargs else out


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The same storage (``Parameter.data`` is a new tensor at every access)."""
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.dtype == b.dtype


def _norm(eps: float, dtype=torch.bfloat16) -> layernorm.RMSNorm:
    norm = layernorm.RMSNorm(HIDDEN, eps)
    norm.weight.data = torch.rand(HIDDEN).to(dtype)
    return norm


@pytest.fixture
def calls(monkeypatch):
    recorder = Calls(fused_result=None)
    monkeypatch.setattr(layernorm, "allreduce_sandwich_rmsnorm", recorder.fused)
    monkeypatch.setattr(layernorm, "all_reduce", recorder.all_reduce)
    monkeypatch.setattr(layernorm, "triton_rmsnorm", recorder.norm)
    monkeypatch.setattr(layernorm, "_get_process_group", lambda group: ("pg", group))
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", True)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 64)
    return recorder


def _run(
    tokens=8,
    group=GROUP,
    post=None,
    norm=None,
    x_dtype=torch.bfloat16,
    residual_dtype=torch.bfloat16,
    **kwargs,
):
    x = torch.randn(tokens, HIDDEN).to(x_dtype)
    residual = torch.randn(tokens, HIDDEN).to(residual_dtype)
    post = post or _norm(1e-5)
    norm = norm or _norm(1e-5)
    out = layernorm.sandwich_rmsnorm_with_allreduce(
        x, residual, post, norm, rank=2, group=group, **kwargs
    )
    return out, (x, residual, post, norm)


def _assert_unfused(calls, inputs, *, reduced: bool, pair=(None, None), post_eps=1e-5):
    x, residual, post, norm = inputs
    kinds = [entry[0] for entry in calls.log if entry[0] != "fused"]
    assert kinds == (["all_reduce", "norm", "norm"] if reduced else ["norm", "norm"])
    tail = calls.log[-2:]
    reduced_x = x + 1 if reduced else x
    _, a_in, a_w, a_eps, a_kw = tail[0]
    assert (
        torch.equal(a_in, reduced_x) and _same(a_w, post.weight) and a_eps == post_eps
    )
    assert a_kw == {"enable_pdl": None}
    _, n_in, n_w, n_eps, n_kw = tail[1]
    assert (
        torch.equal(n_in, reduced_x * 2) and _same(n_w, norm.weight) and n_eps == 1e-5
    )
    assert n_kw["residual"] is residual and n_kw["round_residual_sum_bf16"] is True
    assert n_kw["enable_pdl"] is None
    assert (n_kw["x_scale"], n_kw["residual_scale"]) == pair


@pytest.mark.parametrize("tokens", [1, 64])  # 64: --comm-fusion-max-num-tokens
def test_a_served_call_is_one_fused_launch(calls, tokens):
    calls.fused_result = ("out", "residual_out")
    out, (x, residual, post, norm) = _run(
        tokens=tokens, x_scale=0.5, residual_scale=0.75
    )
    assert out == ("out", "residual_out")
    ((kind, args, kwargs),) = calls.log
    assert kind == "fused"
    assert torch.equal(args[0], x) and torch.equal(args[1], residual)
    assert _same(args[2], post.weight) and _same(args[3], norm.weight)
    assert args[4:] == (2, ("pg", GROUP))
    assert kwargs == dict(eps=1e-5, x_scale=0.5, residual_scale=0.75, max_token_num=64)


@pytest.mark.parametrize(
    "pair", [(0.5, None), (None, 0.5)], ids=["x-scale-only", "residual-scale-only"]
)
@pytest.mark.parametrize("tokens", [1, 65])  # 65: above --comm-fusion-max-num-tokens
def test_a_one_sided_pair_reaches_either_path(calls, pair, tokens):
    """One multiplier alone is a valid call on both paths (the missing one is
    1.0 in the fused wrapper and in rmsnorm): at T = 1 it goes to the fused
    launch, and above the fusion limit to the unfused chain, unchanged."""
    calls.fused_result = ("out", "residual_out") if tokens == 1 else ("never",)
    out, inputs = _run(tokens=tokens, x_scale=pair[0], residual_scale=pair[1])
    if tokens == 1:
        assert out == ("out", "residual_out")
        ((kind, _, kwargs),) = calls.log
        assert kind == "fused"
        assert (kwargs["x_scale"], kwargs["residual_scale"]) == pair
    else:
        _assert_unfused(calls, inputs, reduced=True, pair=pair)


def test_a_call_the_kernel_declines_runs_the_unfused_chain(calls):
    out, inputs = _run(x_scale=0.5, residual_scale=0.75)
    assert calls.log[0][0] == "fused"
    _assert_unfused(calls, inputs, reduced=True, pair=(0.5, 0.75))


def test_without_the_launch_option_nothing_is_fused(calls, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", False)
    calls.fused_result = ("never",)
    _, inputs = _run()
    _assert_unfused(calls, inputs, reduced=True)


@pytest.mark.parametrize("tokens", [0, 65])
def test_token_counts_outside_the_fusion_range_run_unfused(calls, tokens):
    calls.fused_result = ("never",)
    _, inputs = _run(tokens=tokens)
    _assert_unfused(calls, inputs, reduced=True)


def test_non_bf16_weights_run_unfused(calls):
    calls.fused_result = ("never",)
    _, inputs = _run(post=_norm(1e-5, torch.float32))
    _assert_unfused(calls, inputs, reduced=True)


@pytest.mark.parametrize("tensor", ["x", "residual"])
def test_non_bf16_activations_run_unfused(calls, tensor):
    calls.fused_result = ("never",)
    _, inputs = _run(**{f"{tensor}_dtype": torch.float16})
    _assert_unfused(calls, inputs, reduced=True)


@pytest.mark.parametrize("which", ["post", "norm"])
def test_norms_with_a_weight_offset_raise_before_any_call(calls, which):
    """GemmaRMSNorm scales by 1 + weight: both entry points refuse it."""
    gemma = layernorm.GemmaRMSNorm(HIDDEN, 1e-5)
    gemma.weight.data = gemma.weight.data.to(torch.bfloat16)
    with pytest.raises(ValueError, match=r"weight \+ 1\.0"):
        _run(**{which: gemma})
    x = torch.randn(4, HIDDEN).to(torch.bfloat16)
    norms = {"post": _norm(1e-5), "norm": _norm(1e-5), which: gemma}
    with pytest.raises(ValueError, match=r"weight \+ 1\.0"):
        layernorm.sandwich_rmsnorm(x, x, norms["post"], norms["norm"])
    assert calls.log == []


def test_two_epsilons_run_unfused(calls):
    """The fused kernel takes one epsilon for both norms."""
    calls.fused_result = ("never",)
    _, inputs = _run(post=_norm(1e-6))
    _assert_unfused(calls, inputs, reduced=True, post_eps=1e-6)


def test_a_group_of_one_only_normalizes(calls):
    calls.fused_result = ("never",)
    _, inputs = _run(group=(0,))
    _assert_unfused(calls, inputs, reduced=False)


def test_sandwich_rmsnorm_is_the_two_triton_calls(calls):
    x = torch.randn(4, HIDDEN).to(torch.bfloat16)
    residual = torch.randn(4, HIDDEN).to(torch.bfloat16)
    post, norm = _norm(1e-6), _norm(1e-5)
    layernorm.sandwich_rmsnorm(x, residual, post, norm, x_scale=2.0, residual_scale=0.5)
    _assert_unfused(
        calls, (x, residual, post, norm), reduced=False, pair=(2.0, 0.5), post_eps=1e-6
    )


def _fp8_buffers(tokens=8):
    return (
        torch.empty(tokens, HIDDEN, dtype=torch.float8_e4m3fn),
        torch.empty(tokens, 1, dtype=torch.float32),
    )


def test_token_fp8_out_reaches_the_fused_launch(calls):
    """Per-token FP8 buffers go to the fused launch with the BF16 call's
    arguments; nothing else runs."""
    calls.fused_result = ("out", "residual_out")
    buffers = _fp8_buffers()
    out, _ = _run(x_scale=0.5, residual_scale=0.75, token_fp8_out=buffers)
    assert out == ("out", "residual_out")
    ((kind, _, kwargs),) = calls.log
    assert kind == "fused"
    assert kwargs.pop("token_fp8_out") is buffers
    assert kwargs == dict(eps=1e-5, x_scale=0.5, residual_scale=0.75, max_token_num=64)


@pytest.mark.parametrize("fusion", [True, False])
def test_token_fp8_out_reaches_the_unfused_second_norm(calls, monkeypatch, fusion):
    """A declined (or unfused) call writes the codes in the second Triton
    norm, the one whose output they quantize; the first norm and the rest of
    the chain keep the BF16 call's arguments."""
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", fusion)
    buffers = _fp8_buffers()
    _, inputs = _run(x_scale=0.5, residual_scale=0.75, token_fp8_out=buffers)
    if fusion:
        assert calls.log[0][2]["token_fp8_out"] is buffers
    assert calls.log[-1][4].pop("token_fp8_out") is buffers
    _assert_unfused(calls, inputs, reduced=True, pair=(0.5, 0.75))


def test_sandwich_rmsnorm_passes_token_fp8_out_to_the_second_norm(calls):
    x = torch.randn(4, HIDDEN).to(torch.bfloat16)
    residual = torch.randn(4, HIDDEN).to(torch.bfloat16)
    post, norm = _norm(1e-6), _norm(1e-5)
    buffers = _fp8_buffers(4)
    layernorm.sandwich_rmsnorm(
        x, residual, post, norm, x_scale=2.0, residual_scale=0.5, token_fp8_out=buffers
    )
    assert calls.log[-1][4].pop("token_fp8_out") is buffers
    _assert_unfused(
        calls, (x, residual, post, norm), reduced=False, pair=(2.0, 0.5), post_eps=1e-6
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
