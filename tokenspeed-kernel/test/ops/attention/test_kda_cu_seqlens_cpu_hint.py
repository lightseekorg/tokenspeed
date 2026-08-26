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

"""Plumbing tests for the required ``cu_seqlens_cpu`` host boundaries.

Every KDA prefill solution plans its chunk indices on the host from the
``cu_seqlens`` contents. Reading them from the device tensor instead is a
stream-synchronizing D2H per layer per chunk that stalls the launch thread
behind all queued GPU work (and serializes the chunk pipeline's stages), so
``kda_paged_prefill`` REQUIRES a host int64 copy and forwards it to every
solution. These tests pin that path from the dispatch layer down to the
wrapper call, entirely on CPU with the wrapper entry points stubbed out.
"""

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.attention.cutedsl_kda as cutedsl_op
import torch
from tokenspeed_kernel.ops.attention.triton.kda_dispatch import (
    KdaPrefillResult,
    _nvidia_kda_prefill,
)
from tokenspeed_kernel.selection import SelectedKernel

H, HV, K, V, T = 2, 2, 128, 128, 32


def _inputs(tokens: int = T, batch: int = 1):
    q = torch.zeros(batch, tokens, H, K, dtype=torch.bfloat16)
    k = torch.zeros(batch, tokens, H, K, dtype=torch.bfloat16)
    v = torch.zeros(batch, tokens, HV, V, dtype=torch.bfloat16)
    g = torch.zeros(batch, tokens, HV, K, dtype=torch.bfloat16)
    beta = torch.zeros(batch, tokens, HV, dtype=torch.bfloat16)
    a_log = torch.zeros(HV, dtype=torch.float32)
    dt_bias = torch.zeros(HV * K, dtype=torch.float32)
    return q, k, v, g, beta, a_log, dt_bias


@pytest.fixture
def stubbed_wrapper(monkeypatch):
    """Stub the CuteDSL wrapper entry points and record their kwargs."""
    seen: dict = {}

    def fake_check_config(lower_bound: float) -> None:
        seen["lower_bound"] = lower_bound

    def fake_workspace_size(boundaries, heads, cu_seqlens_cpu=None) -> int:
        seen["ws_cu_seqlens_cpu"] = cu_seqlens_cpu
        return 0

    def fake_forward(q, k, v, g, a_log, dt_bias, beta, boundaries, state, **kw):
        seen["fwd_cu_seqlens_cpu"] = kw.get("cu_seqlens_cpu")
        seen["boundaries"] = boundaries
        return v.clone(), state.transpose(-1, -2).clone()

    monkeypatch.setattr(cutedsl_op, "cutedsl_kda_check_config", fake_check_config)
    monkeypatch.setattr(cutedsl_op, "cutedsl_kda_workspace_size", fake_workspace_size)
    monkeypatch.setattr(cutedsl_op, "cutedsl_kda_forward", fake_forward)
    return seen


def test_hint_reaches_wrapper_calls(stubbed_wrapper):
    q, k, v, g, beta, a_log, dt_bias = _inputs()
    cu = torch.tensor([0, 10, T], dtype=torch.int32)
    hint = torch.tensor([0, 10, T], dtype=torch.int64)

    cutedsl_op.cutedsl_kda_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state=None,
        cu_seqlens=cu,
        cu_seqlens_cpu=hint,
        lower_bound=-5.0,
    )

    assert stubbed_wrapper["ws_cu_seqlens_cpu"] is hint
    assert stubbed_wrapper["fwd_cu_seqlens_cpu"] is hint
    assert stubbed_wrapper["boundaries"].dtype == torch.int64


def test_hint_length_mismatch_raises(stubbed_wrapper):
    q, k, v, g, beta, a_log, dt_bias = _inputs()
    cu = torch.tensor([0, 10, T], dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_cpu"):
        cutedsl_op.cutedsl_kda_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            initial_state=None,
            cu_seqlens=cu,
            cu_seqlens_cpu=torch.tensor([0, T], dtype=torch.int64),
            lower_bound=-5.0,
        )


def test_batch_fallback_synthesizes_hint(stubbed_wrapper):
    q, k, v, g, beta, a_log, dt_bias = _inputs(tokens=16, batch=2)

    cutedsl_op.cutedsl_kda_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state=None,
        cu_seqlens=None,
        lower_bound=-5.0,
    )

    synthesized = stubbed_wrapper["ws_cu_seqlens_cpu"]
    assert isinstance(synthesized, torch.Tensor)
    assert synthesized.tolist() == [0, 16, 32]
    assert stubbed_wrapper["fwd_cu_seqlens_cpu"] is synthesized


def test_dispatch_always_forwards_host_boundaries():
    calls = []

    def impl(*args, **kwargs):
        calls.append(kwargs)
        out = torch.zeros(1, T, HV, V)
        return out, torch.zeros(1, HV, K, V)

    q, k, v, g, beta, a_log, dt_bias = _inputs()
    cu = torch.tensor([0, T], dtype=torch.int32)
    cu_cpu = torch.tensor([0, T], dtype=torch.int64)

    _nvidia_kda_prefill(
        impl,
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state=torch.zeros(1, HV, K, V),
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        lower_bound=-5.0,
    )
    assert calls[-1]["cu_seqlens_cpu"] is cu_cpu


def test_facade_requires_host_boundaries(monkeypatch):
    import tokenspeed_kernel.ops.attention as attn

    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return KdaPrefillResult(
            out=torch.zeros(1, T, HV, V), final_state=torch.zeros(1, HV, K, V)
        )

    selected = SelectedKernel("fake_kda_prefill", fake_kernel)
    monkeypatch.setattr(attn, "select_kernel", lambda *a, **kw: selected)

    q, k, v, g, beta, a_log, dt_bias = _inputs()
    cu = torch.tensor([0, T], dtype=torch.int32)
    cu_cpu = torch.tensor([0, T], dtype=torch.int64)
    common = dict(
        initial_state=torch.zeros(1, HV, K, V), cu_seqlens=cu, lower_bound=-5.0
    )

    with pytest.raises(TypeError):
        attn.kda_paged_prefill(q, k, v, g, beta, a_log, dt_bias, **common)

    with pytest.raises(ValueError, match="host int64 tensor"):
        attn.kda_paged_prefill(
            q, k, v, g, beta, a_log, dt_bias, cu_seqlens_cpu=(0, T), **common
        )

    attn.kda_paged_prefill(
        q, k, v, g, beta, a_log, dt_bias, cu_seqlens_cpu=cu_cpu, **common
    )
    assert calls[-1]["cu_seqlens_cpu"] is cu_cpu


def test_solution_wrappers_forward_host_boundaries(monkeypatch):
    import tokenspeed_kernel.ops.attention.triton.kda_dispatch as kd

    received = []

    def fake_prefill(implementation, *args, **kwargs):
        received.append(kwargs)
        return KdaPrefillResult(
            out=torch.zeros(1, T, HV, V), final_state=torch.zeros(1, HV, K, V)
        )

    monkeypatch.setattr(kd, "_nvidia_kda_prefill", fake_prefill)

    q, k, v, g, beta, a_log, dt_bias = _inputs()
    cu = torch.tensor([0, T], dtype=torch.int32)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g_raw=g,
        beta_logits=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=torch.zeros(1, HV, K, V),
        cu_seqlens=cu,
        lower_bound=-5.0,
        cu_seqlens_cpu=torch.tensor([0, T], dtype=torch.int64),
    )

    kd.triton_nvidia_kda_paged_prefill(**dict(kwargs))
    assert received[-1]["cu_seqlens_cpu"] is kwargs["cu_seqlens_cpu"]

    kd.flashkda_nvidia_kda_paged_prefill(**dict(kwargs))
    assert received[-1]["cu_seqlens_cpu"] is kwargs["cu_seqlens_cpu"]
