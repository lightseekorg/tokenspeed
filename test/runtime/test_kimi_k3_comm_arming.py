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

"""Deferred-finalize arming of the K3 latent tail.

The arming gate must be the experts kernel plan's own
``supports_deferred_finalize`` capability bit, not a use_trtllm proxy: the
trtllm solution spans kernels with either capability (the nvfp4/mxfp4 SiTU
variants emit the deferred triple, mxfp4 SwiGLU does not), and a mis-armed
TAIL_FUSION request crashes the experts layer with
``MoELayer does not support do_finalize=False``.
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=2, suite="runtime-1gpu")

# The iris cases drive the CDNA4 branch, which imports an AMD-only package.
needs_iris = pytest.mark.skipif(
    find_spec("iris") is None, reason="iris is packaged for ROCm only"
)

from tokenspeed.runtime.models.kimi_k3_comm import (  # noqa: E402
    ATTN_AR_MAX_TOKENS,
    _tail_finalize_top_k,
    attn_ar_eligible,
)


def test_arming_requires_experts_capability_bit():
    plan = SimpleNamespace(fused_moe_ar=True, use_trtllm=True)
    # A kernel without the deferred capability (e.g. mxfp4 SwiGLU) ->
    # materialized-input tail (finalize_top_k=None), even though
    # use_trtllm is True.
    assert _tail_finalize_top_k(10, plan, False) is None
    # Deferred-capable kernel (either SiTU variant) -> deferred triple.
    assert _tail_finalize_top_k(10, plan, True) == 10


def test_arming_requires_fused_moe_ar():
    plan = SimpleNamespace(fused_moe_ar=False, use_trtllm=True)
    assert _tail_finalize_top_k(10, plan, True) is None
    assert _tail_finalize_top_k(10, plan, False) is None


@needs_iris
def test_iris_preparation_deduplicates_equal_groups(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3_comm

    group = tuple(range(8))
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=8, tp_group=group),
        moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3_comm,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3_comm, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3_comm.prepare_k3_all_reduce_buffers(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden_size=3584,
        max_num_tokens=16384,
    )
    prepare.assert_called_once_with(
        group,
        staged_max_numel=8192 * 7168,
        producer_direct_max_numel=8192 * (7168 + 3584),
        attnres_max_numel=16 * 7168,
        attnres_max_rows=16,
        dtype=torch.bfloat16,
        backend=None,
    )


@needs_iris
def test_iris_preparation_handles_distinct_groups(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3_comm

    attn_group = (0, 1, 2, 3)
    moe_group = tuple(range(8))
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=4, tp_group=attn_group),
        moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=moe_group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3_comm,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3_comm, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3_comm.prepare_k3_all_reduce_buffers(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden_size=3584,
        max_num_tokens=8192,
    )
    assert prepare.call_args_list == [
        call(
            attn_group,
            staged_max_numel=8192 * 7168,
            producer_direct_max_numel=0,
            attnres_max_numel=0,
            attnres_max_rows=0,
            dtype=torch.bfloat16,
            backend=None,
        ),
        call(
            moe_group,
            staged_max_numel=8192 * 7168,
            producer_direct_max_numel=48 * (7168 + 3584),
            attnres_max_numel=0,
            attnres_max_rows=0,
            dtype=torch.bfloat16,
            backend=None,
        ),
    ]


@needs_iris
def test_iris_preparation_handles_moe_only_group(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3_comm

    attn_group = (0,)
    moe_group = tuple(range(8))
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=1, tp_group=attn_group),
        moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=moe_group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3_comm,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3_comm, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3_comm.prepare_k3_all_reduce_buffers(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden_size=3584,
        max_num_tokens=8192,
    )
    prepare.assert_called_once_with(
        moe_group,
        staged_max_numel=8192 * 7168,
        producer_direct_max_numel=48 * (7168 + 3584),
        attnres_max_numel=0,
        attnres_max_rows=0,
        dtype=torch.bfloat16,
        backend=None,
    )


@needs_iris
def test_iris_preparation_keeps_baseline_window_for_equal_tp4(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3_comm

    group = tuple(range(4))
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=4, tp_group=group),
        moe=SimpleNamespace(tp_ep_size=4, tp_ep_group=group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3_comm,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3_comm, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3_comm.prepare_k3_all_reduce_buffers(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden_size=3584,
        max_num_tokens=8192,
    )
    prepare.assert_called_once_with(
        group,
        staged_max_numel=8192 * 7168,
        producer_direct_max_numel=48 * (7168 + 3584),
        attnres_max_numel=0,
        attnres_max_rows=0,
        dtype=torch.bfloat16,
        backend=None,
    )


def test_attention_collective_gate():
    # Literals: asserting the constant against itself would pin nothing.
    assert ATTN_AR_MAX_TOKENS == 8
    # An unarmed group never takes the collective; shape cannot override that.
    assert not attn_ar_eligible(
        armed=False, has_prefix=True, num_tokens=1, fusion_max_tokens=2048
    )
    # The window edge is ours; anything wider is the vendor's.
    assert attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=8, fusion_max_tokens=2048
    )
    assert not attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=9, fusion_max_tokens=2048
    )
    # Block-write layers keep no residual for this epilogue to fold in.
    assert not attn_ar_eligible(
        armed=True, has_prefix=False, num_tokens=1, fusion_max_tokens=2048
    )
    assert not attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=0, fusion_max_tokens=2048
    )


def test_the_collective_is_what_serves_an_eligible_reduce():
    """The predicate is half the contract; the branch must hand it the operands."""
    from tokenspeed.runtime.models.kimi_k3_comm import K3AttnComm

    reduced = torch.zeros(1, 8)
    collective = Mock(return_value=(reduced, "shared"))
    vendor = Mock(return_value=(None, "vendor-residual", None))
    comm = K3AttnComm.__new__(K3AttnComm)
    comm.state = SimpleNamespace(
        cute_ar=collective,
        dummy_norm=SimpleNamespace(
            weight="gamma", forward_with_allreduce_fusion=vendor
        ),
        attn_ar_fusion_ok=True,
    )
    comm.mapping = SimpleNamespace(attn=SimpleNamespace(tp_rank=0, tp_group=(0, 1)))

    partial, prefix = torch.zeros(1, 8), torch.zeros(1, 8)
    out, mixed = comm.attn_reduce(partial, prefix, None, mlp_wp=None)

    # Both operands are [m, hidden] bf16, so assert identity, not arrival.
    args, kwargs = collective.call_args
    assert args[0] is partial and args[1] is prefix
    assert kwargs["include_reduce_scatter"] is False
    assert kwargs["include_routed"] is True
    assert collective.call_count == 1
    assert out is reduced and mixed is None

    # Assert the vendor took over: an exception would also give call_count zero.
    collective.reset_mock()
    wide = torch.zeros(9, 8)
    comm.attn_reduce(wide, wide, None, mlp_wp=None)
    assert collective.call_count == 0
    assert vendor.call_count == 1


def test_the_operator_can_forbid_the_fused_attention_reduce():
    """A negative window is how a server forbids fusing this reduce at all."""
    # server_args sets -1 when attn and dense TP disagree; 0 is reachable too.
    for window in (-1, 0):
        assert not attn_ar_eligible(
            armed=True, has_prefix=True, num_tokens=1, fusion_max_tokens=window
        )
    # A window narrower than the kernel's own ceiling still binds.
    assert attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=4, fusion_max_tokens=4
    )
    assert not attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=5, fusion_max_tokens=4
    )


def _arming_world(monkeypatch, *, multicast: bool, shape_ok: bool, peers_agree: bool):
    """Stand up K3AttnCommState's collaborators so arming can be exercised."""
    from tokenspeed.runtime.models import kimi_k3_comm as mod

    recorded = {"ops": [], "groups": []}

    class FakeDist:
        ReduceOp = torch.distributed.ReduceOp

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, *, op, group):
            # Required, not defaulted: dropping either in production must fail here.
            recorded["ops"].append(op)
            recorded["groups"].append(group)
            if not peers_agree:
                tensor.zero_()

    monkeypatch.setattr(mod, "dist", FakeDist)
    monkeypatch.setattr(mod, "prepare_all_reduce_lane", lambda *a, **k: True)
    monkeypatch.setattr(mod, "prepare_all_reduce_fusion", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_get_process_group", lambda g: "the-group")
    monkeypatch.setattr(mod, "multicast_backend_available", lambda g: multicast)
    monkeypatch.setattr(mod, "attn_reduce_shape_supported", lambda **k: shape_ok)
    monkeypatch.setattr(
        mod, "global_server_args_dict", {"comm_fusion_max_num_tokens": 2048}
    )
    monkeypatch.setattr(
        mod, "RMSNorm", lambda h, eps: SimpleNamespace(weight=torch.ones(1))
    )
    builder = Mock(return_value="collective")
    monkeypatch.setattr(mod, "build_attn_reduce_collective", builder)
    recorded["builder"] = builder
    return mod, recorded


_ARMING_MAPPING = SimpleNamespace(
    attn=SimpleNamespace(tp_size=8, tp_rank=3, tp_group=object())
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the vote is a cuda tensor")
def test_arming_builds_only_when_every_rank_agrees(monkeypatch):
    """A rank that armed alone would sit in a rendezvous its peers never join."""
    mod, rec = _arming_world(
        monkeypatch, multicast=True, shape_ok=True, peers_agree=True
    )
    state = mod.K3AttnCommState(mapping=_ARMING_MAPPING, hidden_size=7168)
    assert state.cute_ar == "collective"
    # MIN is what makes one dissenting rank stop all of them.
    assert rec["ops"] == [torch.distributed.ReduceOp.MIN]
    assert rec["groups"] == ["the-group"]
    kwargs = rec["builder"].call_args.kwargs
    assert kwargs["rank"] == 3 and kwargs["tp_size"] == 8  # rank is not size
    assert kwargs["hidden_size"] == 7168
    assert kwargs["max_tokens"] == mod.ATTN_AR_MAX_TOKENS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the vote is a cuda tensor")
@pytest.mark.parametrize(
    "multicast,shape_ok,peers_agree",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_arming_declines_when_any_probe_or_peer_says_no(
    monkeypatch, multicast, shape_ok, peers_agree
):
    """Each term is load-bearing: the constructor raises, it does not decline."""
    mod, rec = _arming_world(
        monkeypatch, multicast=multicast, shape_ok=shape_ok, peers_agree=peers_agree
    )
    state = mod.K3AttnCommState(mapping=_ARMING_MAPPING, hidden_size=7168)
    assert state.cute_ar is None
    assert rec["builder"].call_count == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
