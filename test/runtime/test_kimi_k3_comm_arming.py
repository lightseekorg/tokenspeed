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

"""Attention and AMD communication preparation for Kimi K3."""

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
    attn_ar_eligible,
)


@needs_iris
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize(
    "max_rows,tail_rows",
    [(39, 0), (40, 40), (47, 40), (511, 504), (8192, 8192), (16384, 8192)],
)
def test_iris_preparation_caps_attnres_for_equal_tp8_groups(
    monkeypatch, pp_size, max_rows, tail_rows
):
    from tokenspeed.runtime.models import kimi_k3

    group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=pp_size,
        attn=SimpleNamespace(tp_size=8, tp_group=group),
        moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_size=8, tp_ep_group=group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3.prepare_k3_all_reduce_buffers(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden_size=3584,
        max_num_tokens=max_rows,
    )
    prepare.assert_called_once_with(
        group,
        staged_max_numel=min(max_rows, 8192) * 7168,
        producer_direct_max_numel=min(max_rows, 8192) * (7168 + 3584),
        attnres_max_numel=16 * 7168,
        attnres_max_rows=16,
        enable_lamport=True,
        moe_tail_max_rows=tail_rows if pp_size == 1 else 0,
        dtype=torch.bfloat16,
        backend=None,
    )


@needs_iris
def test_iris_preparation_handles_distinct_groups(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3

    attn_group = (0, 1, 2, 3)
    moe_group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=1,
        attn=SimpleNamespace(tp_size=4, tp_group=attn_group),
        moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=moe_group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3.prepare_k3_all_reduce_buffers(
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
            enable_lamport=False,
            moe_tail_max_rows=0,
            dtype=torch.bfloat16,
            backend=None,
        ),
        call(
            moe_group,
            staged_max_numel=8192 * 7168,
            producer_direct_max_numel=48 * (7168 + 3584),
            attnres_max_numel=0,
            attnres_max_rows=0,
            enable_lamport=False,
            moe_tail_max_rows=0,
            dtype=torch.bfloat16,
            backend=None,
        ),
    ]


@needs_iris
def test_iris_preparation_handles_moe_only_group(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3

    attn_group = (0,)
    moe_group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=1,
        attn=SimpleNamespace(tp_size=1, tp_group=attn_group),
        moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=moe_group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3.prepare_k3_all_reduce_buffers(
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
        enable_lamport=False,
        moe_tail_max_rows=0,
        dtype=torch.bfloat16,
        backend=None,
    )


@needs_iris
def test_iris_preparation_keeps_baseline_window_for_equal_tp4(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3

    group = tuple(range(4))
    mapping = SimpleNamespace(
        pp_size=1,
        attn=SimpleNamespace(tp_size=4, tp_group=group),
        moe=SimpleNamespace(tp_ep_size=4, tp_ep_group=group),
    )
    prepare = Mock(return_value=True)
    monkeypatch.setattr(
        kimi_k3,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(kimi_k3, "prepare_all_reduce_buffers", prepare)

    assert kimi_k3.prepare_k3_all_reduce_buffers(
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
        enable_lamport=False,
        moe_tail_max_rows=0,
        dtype=torch.bfloat16,
        backend=None,
    )


@needs_iris
@pytest.mark.parametrize(
    "world,attn_tp,moe_tp,moe_ep,expected",
    [
        (8, 8, 8, 1, True),
        (16, 8, 8, 1, True),
        (8, 8, 1, 8, False),
        (8, 8, 4, 2, False),
        (8, 4, 8, 1, False),
        (8, 1, 8, 1, False),
        (16, 8, 8, 2, False),
        (4, 4, 4, 1, False),
    ],
)
def test_iris_lamport_requires_attention_and_moe_tp8(
    monkeypatch, world, attn_tp, moe_tp, moe_ep, expected
):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models import kimi_k3

    monkeypatch.setattr(
        kimi_k3, "current_platform", lambda: SimpleNamespace(is_cdna4=True)
    )
    for rank in range(world):
        mapping = Mapping(
            rank=rank,
            world_size=world,
            attn_tp_size=attn_tp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
        )
        prepare = Mock(return_value=True)
        monkeypatch.setattr(kimi_k3, "prepare_all_reduce_buffers", prepare)

        assert kimi_k3.prepare_k3_all_reduce_buffers(
            mapping=mapping,
            hidden_size=7168,
            routed_hidden_size=3584,
            max_num_tokens=8,
        )

        assert prepare.called
        for request in prepare.call_args_list:
            assert request.kwargs["enable_lamport"] is expected
        # Disabling Lamport must preserve the producer-direct pull path.
        moe_request = next(
            request
            for request in prepare.call_args_list
            if request.args[0] == mapping.moe.tp_ep_group
        )
        assert moe_request.kwargs["producer_direct_max_numel"] == 8 * 10752


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
    comm.cute_ar = collective
    comm.dummy_norm = SimpleNamespace(
        weight="gamma", forward_with_allreduce_fusion=vendor
    )
    comm.attn_ar_fusion_ok = True
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


def _arming_world(monkeypatch, *, multicast: bool, shape_ok: bool):
    """Stand up K3AttnComm's collaborators so arming can be exercised."""
    from tokenspeed.runtime.models import kimi_k3_comm as mod

    monkeypatch.setattr(mod.K3AttnComm, "_prepared_hidden_size", None)
    monkeypatch.setattr(mod.K3AttnComm, "attn_ar_fusion_ok", False)
    monkeypatch.setattr(mod.K3AttnComm, "dummy_norm", None)
    monkeypatch.setattr(mod.K3AttnComm, "cute_ar", None)
    monkeypatch.setattr(mod, "dist", SimpleNamespace(is_initialized=lambda: True))
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
    return mod, builder


_ARMING_MAPPING = SimpleNamespace(
    attn=SimpleNamespace(tp_size=8, tp_rank=3, tp_group=object())
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the dummy norm is on CUDA")
def test_arming_shares_workspaces_across_instances(monkeypatch):
    mod, builder = _arming_world(monkeypatch, multicast=True, shape_ok=True)
    comm = mod.K3AttnComm(mapping=_ARMING_MAPPING, hidden_size=7168)
    dummy_norm = comm.dummy_norm
    other = mod.K3AttnComm(mapping=_ARMING_MAPPING, hidden_size=7168)
    assert other is not comm
    assert other.cute_ar is comm.cute_ar is builder.return_value
    assert other.dummy_norm is dummy_norm
    builder.assert_called_once_with(
        group="the-group",
        rank=3,
        tp_size=8,
        hidden_size=7168,
        max_tokens=mod.ATTN_AR_MAX_TOKENS,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the dummy norm is on CUDA")
@pytest.mark.parametrize("multicast,shape_ok", [(False, True), (True, False)])
def test_arming_declines_unsupported_collectives(monkeypatch, multicast, shape_ok):
    mod, builder = _arming_world(monkeypatch, multicast=multicast, shape_ok=shape_ok)
    comm = mod.K3AttnComm(mapping=_ARMING_MAPPING, hidden_size=7168)
    dummy_norm = comm.dummy_norm
    other = mod.K3AttnComm(mapping=_ARMING_MAPPING, hidden_size=7168)
    assert comm.cute_ar is None
    assert other.cute_ar is None
    assert other.dummy_norm is dummy_norm
    builder.assert_not_called()


@needs_iris
@pytest.mark.parametrize(
    "rows,is_prefill,sharded_moe_supported,eligible",
    [
        (0, True, True, False),
        (15, True, True, False),
        (16, True, True, True),
        (37, True, True, True),
        (8192, True, True, True),
        (8193, True, True, False),
        (8192, False, True, False),
        (8192, True, False, False),
    ],
)
def test_attention_prefill_producer_window(
    monkeypatch, rows, is_prefill, sharded_moe_supported, eligible
):
    from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod
    from tokenspeed.runtime.models import kimi_k3_comm as module

    group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=1,
        attn=SimpleNamespace(tp_size=8, tp_group=group),
        moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
    )
    comm = module.K3AttnComm.__new__(module.K3AttnComm)
    comm.mapping = mapping
    like = torch.empty((rows, 7168), dtype=torch.bfloat16)
    projection = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(),
        weight=torch.empty((7168, 1536), dtype=torch.bfloat16),
        bias=None,
        reduce_results=False,
        input_is_parallel=True,
    )
    destination = Mock()
    acquire = Mock(return_value=(destination,))
    capability = Mock(return_value=True)
    monkeypatch.setattr(
        module, "current_platform", lambda: SimpleNamespace(is_cdna4=True)
    )
    monkeypatch.setattr(module, "can_acquire_all_reduce_outputs", capability)
    monkeypatch.setattr(module, "acquire_all_reduce_outputs", acquire)
    out = comm.acquire_prefill_projection_output(
        like,
        projection,
        is_prefill=is_prefill,
        sharded_moe_supported=sharded_moe_supported,
    )
    assert (out is destination) == eligible
    assert acquire.call_count == int(eligible)
    if rows == 16 and eligible:
        projection.reduce_results = True
        assert (
            comm.acquire_prefill_projection_output(
                like, projection, is_prefill=True, sharded_moe_supported=True
            )
            is None
        )
        projection.reduce_results = False
        mapping.moe.ep_size = 8
        assert (
            comm.acquire_prefill_projection_output(
                like, projection, is_prefill=True, sharded_moe_supported=True
            )
            is None
        )


@pytest.mark.parametrize(
    "rows,is_cdna4,eligible",
    [
        (48, True, False),
        (56, True, True),
        (57, True, False),
        (8192, True, True),
        (8200, True, False),
        (512, False, False),
    ],
)
def test_attention_prefill_mix_window(monkeypatch, rows, is_cdna4, eligible):
    from tokenspeed.runtime.models import kimi_k3_comm as module

    group = tuple(range(8))
    comm = module.K3AttnComm.__new__(module.K3AttnComm)
    comm.mapping = SimpleNamespace(attn=SimpleNamespace(tp_group=group))
    partial = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    history = torch.empty((4, rows, 7168), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((7168,), dtype=torch.bfloat16, device="meta")
    expected = (object(), object())
    operation = Mock(return_value=expected)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.communication.iris",
        SimpleNamespace(iris_attention_mix=operation),
    )
    monkeypatch.setattr(
        module, "current_platform", lambda: SimpleNamespace(is_cdna4=is_cdna4)
    )
    monkeypatch.setattr(module, "_get_process_group", lambda _: "owner")
    result = comm.prefill_mix_for_moe(
        partial,
        None,
        history,
        weight,
        weight,
        eps=1e-6,
        out_norm_weight=weight,
        out_norm_eps=1e-5,
        num_valid_blocks=4,
    )
    if eligible:
        assert result is expected
        operation.assert_called_once_with(
            partial,
            None,
            history,
            weight,
            weight,
            eps=1e-6,
            out_norm_weight=weight,
            out_norm_eps=1e-5,
            num_valid_blocks=4,
            group="owner",
        )
    else:
        assert result is None
        operation.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
