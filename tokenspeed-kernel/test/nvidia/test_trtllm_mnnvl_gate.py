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

"""MNNVL capability, workspace and mHC fusion contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel.ops.communication import trtllm as comm
from tokenspeed_kernel.platform import current_platform
from torch.utils._python_dispatch import TorchDispatchMode

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


class _RejectTensorOperations(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"admission must only inspect metadata, called {func}")


@pytest.mark.parametrize("use_oneshot", [False, True])
@pytest.mark.parametrize(
    "world,tokens,condition",
    [(world, 17, "exact_capacity") for world in (1, 2, 3, 4, 8, 16, 32)]
    + [(4, tokens, "exact_capacity") for tokens in (0, 1, 16, 2048, 2049)]
    + [
        (4, 17, condition)
        for condition in (
            "short_capacity",
            "wrong_group",
            "fp32",
            "norm_dtype",
            "norm_shape",
            "norm_device",
            "norm_stride",
        )
    ],
)
def test_mhc_admission_uses_collective_size(
    monkeypatch, world, tokens, use_oneshot, condition
):
    from tokenspeed_kernel.thirdparty.cuda import trtllm as native

    x = torch.empty(tokens, 5120, dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(5120, dtype=x.dtype, device=x.device)
    lane_tokens = tokens * world if use_oneshot else 2 * -(-tokens // world) * world
    required_bytes = lane_tokens * 5120 * x.element_size()
    workspace = native.MnnvlAllReduceFusionWorkspace(
        tp_rank=0,
        tp_size=2 * world if condition == "wrong_group" else world,
        max_token_num=tokens,
        hidden_dim=5120,
        buffer_size_bytes=required_bytes - int(condition == "short_capacity"),
        multicast_ptr=0,
        peer_ptrs=torch.empty(world, dtype=torch.int64),
        local_ptr=0,
        buffer_flags=torch.empty(9, dtype=torch.uint32),
        oneshot_token_cap=tokens if use_oneshot else 0,
        refs=(),
    )
    manager = SimpleNamespace(
        initialized=True,
        world_size=world,
        use_fp32_lamport=condition == "fp32",
        mnnvl_workspace=workspace,
    )
    monkeypatch.setattr(comm, "_manager_for_group", lambda group: manager)
    if condition == "norm_dtype":
        weight = torch.empty_like(weight, dtype=torch.float32)
    elif condition == "norm_shape":
        weight = weight.new_empty(5121)
    elif condition == "norm_device":
        weight = torch.empty(5120, dtype=weight.dtype)
    elif condition == "norm_stride":
        weight = weight.new_empty(10240)[::2]
    with _RejectTensorOperations():
        supported = comm.supports_allreduce_mhc_post_norm(x, weight, None)
    assert supported == (
        condition == "exact_capacity"
        and world in (2, 4, 8, 16)
        and 0 < tokens <= native.MNNVL_TWOSHOT_MAX_TOKEN
    )


@pytest.fixture(params=[(8, True), (24, True), (25, False)])
def admitted_mhc(monkeypatch, request):
    from tokenspeed_kernel.thirdparty.cuda import trtllm as native

    tokens, use_oneshot = request.param
    x = torch.empty(tokens, 5120, dtype=torch.bfloat16, device="cuda")
    lane_tokens = tokens * 2 if use_oneshot else 4 * -(-tokens // 2)
    args = dict(
        x=x,
        residual=torch.empty(tokens, 4, 5120, dtype=x.dtype, device=x.device),
        post=torch.empty(tokens, 4, device=x.device),
        comb=torch.empty(tokens, 4, 4, device=x.device),
        pre=torch.empty(tokens, 4, device=x.device),
        weight=torch.empty(5120, dtype=x.dtype, device=x.device),
        eps=1e-6,
        group=None,
    )
    workspace = native.MnnvlAllReduceFusionWorkspace(
        tp_rank=0,
        tp_size=2,
        max_token_num=tokens,
        hidden_dim=5120,
        buffer_size_bytes=lane_tokens * 5120 * x.element_size(),
        multicast_ptr=0,
        peer_ptrs=torch.empty(2, dtype=torch.int64),
        local_ptr=0,
        buffer_flags=torch.empty(9, dtype=torch.uint32),
        oneshot_token_cap=tokens if use_oneshot else 0,
        refs=(),
    )
    manager = SimpleNamespace(
        initialized=True,
        rank=0,
        world_size=2,
        use_fp32_lamport=False,
        mnnvl_workspace=workspace,
        graph_consumed=False,
    )
    monkeypatch.setattr(comm, "_manager_for_group", lambda group: manager)
    assert comm.supports_allreduce_mhc_post_norm(x, args["weight"], args["group"])
    monkeypatch.setattr(
        comm,
        "supports_allreduce_mhc_post_norm",
        Mock(side_effect=AssertionError("execution must not repeat admission")),
    )
    launch = Mock()
    monkeypatch.setattr(
        native,
        "_load_trtllm_mhc_module",
        lambda: SimpleNamespace(trtllm_mnnvl_mhc=launch),
    )
    return args, manager, launch


@pytest.mark.parametrize("kernel_error", [False, True])
def test_mhc_execution_returns_outputs_or_propagates_error(admitted_mhc, kernel_error):
    args, manager, launch = admitted_mhc
    if kernel_error:
        launch.side_effect = RuntimeError("native kernel failed")
        with pytest.raises(RuntimeError, match="native kernel failed"):
            comm.allreduce_mhc_post_norm(**args)
    else:
        residual, normalized = comm.allreduce_mhc_post_norm(**args)
        assert residual is launch.call_args.args[6]
        assert normalized is launch.call_args.args[7]
        assert residual.shape == args["residual"].shape
        assert normalized.shape == args["x"].shape
    launch.assert_called_once()
    assert launch.call_args.args[-2] is manager.mnnvl_workspace.resolve_use_oneshot(
        args["x"].shape[0], None, 5120
    )


@pytest.mark.parametrize(
    "violation",
    [
        "rows",
        "zero_tokens",
        "token_limit",
        "dtype",
        "stride",
        "hc_shape",
        "post_dtype",
        "device",
        "eps",
    ],
)
def test_mhc_producer_contract_violation_raises_before_launch(admitted_mhc, violation):
    from tokenspeed_kernel.thirdparty.cuda import trtllm as native

    args, _, launch = admitted_mhc
    x = args["x"]
    tokens = x.shape[0]
    if violation == "rows":
        args["x"] = x[:-1]
    elif violation == "zero_tokens":
        args["x"] = x[:0]
    elif violation == "token_limit":
        args["x"] = x.new_empty(native.MNNVL_TWOSHOT_MAX_TOKEN + 1, 5120)
    elif violation == "dtype":
        args["x"] = x.float()
    elif violation == "stride":
        args["x"] = x.new_empty(tokens, 10240)[:, ::2]
    elif violation == "hc_shape":
        args["residual"] = args["residual"].view(tokens, 5120, 4)
    elif violation == "post_dtype":
        args["post"] = args["post"].to(torch.bfloat16)
    elif violation == "device":
        args["pre"] = torch.empty(tokens, 4)
    else:
        args["eps"] = 0
    with pytest.raises(ValueError, match="mHC fusion requires"):
        comm.allreduce_mhc_post_norm(**args)
    launch.assert_not_called()


def test_mhc_lost_workspace_raises_before_launch(admitted_mhc):
    args, manager, launch = admitted_mhc
    manager.mnnvl_workspace = None
    with pytest.raises(RuntimeError, match="prepared MNNVL workspace"):
        comm.allreduce_mhc_post_norm(**args)
    launch.assert_not_called()
