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

from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from tokenspeed.runtime.models.kimi_k3_comm import (
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
    # Literals, not ATTN_AR_MAX_TOKENS: asserting the constant against itself
    # passes for every value and pins nothing.
    assert ATTN_AR_MAX_TOKENS == 8
    # An unarmed group never takes the collective; shape cannot override that.
    assert not attn_ar_eligible(
        armed=False, has_prefix=True, num_tokens=1, fusion_max_tokens=2048
    )
    # The vendor AR owns everything wider than the one-shot window, and the
    # window edge itself belongs to us.
    assert attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=8, fusion_max_tokens=2048
    )
    assert not attn_ar_eligible(
        armed=True, has_prefix=True, num_tokens=9, fusion_max_tokens=2048
    )
    # Block-write layers hand the prefix to the snapshot, so there is no
    # residual left for the collective to fold in.
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

    # Operand order is not observable from shapes -- both are [m, hidden] bf16 --
    # so assert identity, not just that the kernel was reached.
    args, kwargs = collective.call_args
    assert args[0] is partial and args[1] is prefix
    assert kwargs["include_reduce_scatter"] is False
    assert kwargs["include_routed"] is True
    assert collective.call_count == 1
    assert out is reduced and mixed is None

    # Nine tokens belong to the vendor. Assert it took over, not merely that we
    # did not: an exception on the way there would satisfy a call_count of zero.
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


def test_widths_the_collective_cannot_serve_are_declined_not_raised():
    """The constructor raises; the gate has to answer before it is reached."""
    from tokenspeed.runtime.models.kimi_k3_comm import _attn_collective_shape_ok

    assert _attn_collective_shape_ok(8, 7168)
    assert _attn_collective_shape_ok(16, 7168)
    # Cluster width is tp_size here, and the kernel takes powers of two <= 16.
    for tp in (7, 12, 24, 32):
        assert not _attn_collective_shape_ok(tp, 7168)
