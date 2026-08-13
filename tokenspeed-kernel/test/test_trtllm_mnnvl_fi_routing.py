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

"""Tests for AR-fusion routing between the upstream flashinfer MNNVL workspace
and the private mnnvl kernel (which now only backs the K3-specific epilogues)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia and torch.cuda.is_available()),
    reason="trtllm MNNVL routing is NVIDIA/CUDA only",
)

WORLD = 8
HIDDEN = 7168
DTYPE = torch.bfloat16


class _FakeUpstreamWorkspace:
    """Stands in for flashinfer's MNNVLAllReduceFusionWorkspace: only the
    attributes the wrapper reads, with an always-sufficient buffer."""

    rank = 0
    tp_size = WORLD
    buffer_size_bytes = 1 << 62

    def is_buffer_size_sufficient(self, tp_size, num_tokens, hidden_dim, dtype):
        return True


def _fi_wrapper(max_token_num: int = 2048):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        FlashinferMnnvlAllReduceFusionWorkspace,
    )

    return FlashinferMnnvlAllReduceFusionWorkspace(
        _FakeUpstreamWorkspace(), max_token_num
    )


def test_fi_supports_generic_patterns_only():
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern as P

    ws = _fi_wrapper()
    assert ws.supports(16, HIDDEN, DTYPE, WORLD, P.kAllReduce)
    assert ws.supports(16, HIDDEN, DTYPE, WORLD, P.kARResidualRMSNorm)
    # K3-specific epilogues stay on the private kernel.
    assert not ws.supports(16, HIDDEN, DTYPE, WORLD, P.kARResidualAttnResCombine)
    assert not ws.supports(16, HIDDEN + 3584, DTYPE, WORLD, P.kAllReduceLatentNorm)
    # Quant/partial epilogues have no mnnvl home at all.
    assert not ws.supports(
        16, HIDDEN, DTYPE, WORLD, P.kARResidualRMSNormFP8BlockWiseQuant
    )


def test_fi_supports_shape_gates():
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern as P

    ws = _fi_wrapper(max_token_num=2048)
    assert not ws.supports(0, HIDDEN, DTYPE, WORLD, P.kAllReduce)
    assert not ws.supports(16, HIDDEN, torch.float32, WORLD, P.kAllReduce)
    assert not ws.supports(16, HIDDEN, DTYPE, WORLD - 4, P.kAllReduce)
    assert not ws.supports(
        16, HIDDEN, DTYPE, WORLD, P.kAllReduce, residual_reduce_scattered=True
    )
    # Capacity is judged by the actual buffer bytes, not the creation-time
    # max_token_num (the multicast allocation granularity dwarfs any request,
    # so one workspace serves every caller of its group).
    ws.workspace.is_buffer_size_sufficient = (
        lambda tp_size, num_tokens, hidden_dim, dtype: False
    )
    assert not ws.supports(16, HIDDEN, DTYPE, WORLD, P.kAllReduce)


def test_routing_token_threshold_between_mnnvl_workspaces(monkeypatch):
    """Single-node: decode-sized generic calls stay private, mid-M generic
    calls go to fi, K3 epilogues stay private, IPC takes latent-norm and
    large payloads. Each mnnvl workspace is the other's shape fallback."""
    import tokenspeed_kernel.ops.communication.trtllm as trtllm_mod
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern as P

    class _FakePrivate:
        backend = "mnnvl"

        def __init__(self, ok=True):
            self.ok = ok

        def supports(self, *args, **kwargs):
            return self.ok

    fi = _fi_wrapper()
    priv = _FakePrivate()
    ipc = torch.empty(1, dtype=torch.int64)
    mgr = trtllm_mod._workspace_manager
    monkeypatch.setattr(mgr, "mnnvl_fi_workspace", fi)
    monkeypatch.setattr(mgr, "mnnvl_workspace", priv)
    monkeypatch.setattr(mgr, "workspace_tensor", ipc)
    monkeypatch.setattr(mgr, "world_size", WORLD)

    route = trtllm_mod._ar_fusion_workspace
    small = trtllm_mod.MNNVL_FLASHINFER_MIN_TOKENS - 1
    mid = trtllm_mod.MNNVL_FLASHINFER_MIN_TOKENS
    big = trtllm_mod.MNNVL_PREFER_IPC_BYTES // (HIDDEN * DTYPE.itemsize) + 8

    assert route(small, HIDDEN, DTYPE, P.kAllReduce, True) is priv
    assert route(mid, HIDDEN, DTYPE, P.kAllReduce, True) is fi
    assert route(mid, HIDDEN, DTYPE, P.kARResidualRMSNorm, True) is fi
    assert route(small, HIDDEN, DTYPE, P.kARResidualAttnResCombine, True) is priv
    assert route(mid, HIDDEN, DTYPE, P.kARResidualAttnResCombine, True) is priv
    assert route(small, HIDDEN + 3584, DTYPE, P.kAllReduceLatentNorm, True) is ipc
    assert route(big, HIDDEN, DTYPE, P.kARResidualRMSNorm, True) is ipc
    # Shape fallback: private rejects a small generic call -> fi takes it.
    monkeypatch.setattr(mgr, "mnnvl_workspace", _FakePrivate(ok=False))
    assert route(small, HIDDEN, DTYPE, P.kAllReduce, True) is fi


def test_routing_cross_node_without_fi_falls_back_to_private(monkeypatch):
    """No IPC (cross-node) and no fi (old flashinfer): the private kernel must
    keep serving the generic patterns as before."""
    import tokenspeed_kernel.ops.communication.trtllm as trtllm_mod
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern as P

    class _FakePrivate:
        backend = "mnnvl"

        def supports(self, *args, **kwargs):
            return True

    priv = _FakePrivate()
    mgr = trtllm_mod._workspace_manager
    monkeypatch.setattr(mgr, "mnnvl_fi_workspace", None)
    monkeypatch.setattr(mgr, "mnnvl_workspace", priv)
    monkeypatch.setattr(mgr, "workspace_tensor", None)
    monkeypatch.setattr(mgr, "world_size", WORLD)

    route = trtllm_mod._ar_fusion_workspace
    assert route(16, HIDDEN, DTYPE, P.kAllReduce, True) is priv
    assert route(16, HIDDEN, DTYPE, P.kARResidualRMSNorm, True) is priv
    # With fi armed, cross-node prefers it at every M (no IPC, no in-situ
    # small-M data; fi wins everywhere measured).
    monkeypatch.setattr(mgr, "mnnvl_fi_workspace", _fi_wrapper())
    assert route(16, HIDDEN, DTYPE, P.kAllReduce, True) is mgr.mnnvl_fi_workspace
