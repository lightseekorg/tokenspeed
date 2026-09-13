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

"""Tests for the MNNVL capability gate on the trtllm one-shot all-reduce path."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia and torch.cuda.is_available()),
    reason="trtllm MNNVL gate is NVIDIA/CUDA only",
)


def _probe():
    import tokenspeed_kernel.ops.communication.trtllm as trtllm_mod

    return trtllm_mod, trtllm_mod._mnnvl_locally_available


def test_cross_host_group_requires_fabric(monkeypatch):
    """A group wider than the host's GPUs needs working fabric memory.

    Without it, symm_mem.rendezvous() hangs instead of failing, so the gate
    must reject the workspace up front.
    """
    trtllm_mod, probe = _probe()
    monkeypatch.setattr(trtllm_mod, "fabric_allocation_supported", lambda _: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert probe(16) is False


def test_cross_host_group_allowed_with_fabric(monkeypatch):
    trtllm_mod, probe = _probe()
    monkeypatch.setattr(trtllm_mod, "fabric_allocation_supported", lambda _: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    # Still subject to the other capability checks, so only assert that the
    # cross-host rule alone no longer vetoes the group.
    assert probe(16) == probe(8)


def test_intra_host_group_ignores_fabric(monkeypatch):
    """Groups inside one host ride NVLS multicast, so fabric must not gate them."""
    trtllm_mod, probe = _probe()
    monkeypatch.setattr(
        trtllm_mod,
        "fabric_allocation_supported",
        lambda _: pytest.fail("fabric probe must not run for intra-host groups"),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    probe(8)


def test_unsupported_world_size_rejected():
    _, probe = _probe()

    assert probe(3) is False


def test_the_oneshot_cap_follows_the_call_width_not_the_armed_lane():
    """Arming is grow-only, so a retired wide lane must not pin narrow calls."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import MnnvlAllReduceFusionWorkspace

    def ws(armed, cap, max_token_num=2048):
        return MnnvlAllReduceFusionWorkspace(
            tp_rank=0,
            tp_size=8,
            max_token_num=max_token_num,
            hidden_dim=armed,
            buffer_size_bytes=0,
            multicast_ptr=1,
            peer_ptrs=None,
            local_ptr=1,
            buffer_flags=None,
            oneshot_token_cap=cap,
            refs=(),
        )

    # K3 arms 3584 + 7168 for a lane-norm path that is retired, and every live
    # all-reduce is 7168 wide. At the armed width the cap is 6; at the width in
    # hand it is 9, which is what an eight-token spec-decode step needs.
    k3 = ws(10752, 6)
    assert k3.resolve_use_oneshot(8, None, 10752) is False
    assert k3.resolve_use_oneshot(8, None, 7168) is True
    assert k3.resolve_use_oneshot(10, None, 7168) is False
    assert k3.resolve_use_oneshot(8, False, 7168) is False

    # Scaling never promises more rows than the buffer was armed for.
    assert ws(8192, 4096, max_token_num=64).resolve_use_oneshot(65, None, 4096) is False


def test_every_resolution_passes_the_call_width():
    """Resolution is authoritative wherever it runs, so none may omit the width.

    Upstream wrappers resolve, then the launcher resolves again; a site that
    left the width out would recompute the armed-width answer and undo the
    decision, which is invisible to a unit test calling the method directly.
    """
    import ast
    import pathlib

    import tokenspeed_kernel

    # Walk the package *and* the tests beside it: a stale two-argument call in a
    # distributed test is a TypeError that only a multi-rank run would reach.
    root = pathlib.Path(tokenspeed_kernel.__file__).parents[2]
    sites = 0
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve_use_oneshot"
            ):
                sites += 1
                assert len(node.args) == 3, f"{path}:{node.lineno} omits the width"
    assert sites >= 5, f"expected every resolution site to be checked, saw {sites}"
