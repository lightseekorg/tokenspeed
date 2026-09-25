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


@needs_iris
@pytest.mark.parametrize(
    "rows,is_prefill,sharded_moe_supported,eligible",
    [
        (48, True, True, False),
        (511, True, True, False),
        (512, True, True, True),
        (513, True, True, False),
        (848, True, True, True),
        (1024, True, True, True),
        (2048, True, True, True),
        (4088, True, True, True),
        (4095, True, True, False),
        (4096, True, True, True),
        (4097, True, True, False),
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
    comm = module.K3AttnComm(SimpleNamespace(mapping=mapping))
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
    if eligible:
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


@pytest.mark.parametrize("has_prefix", [False, True])
@pytest.mark.parametrize("producer_direct", [False, True])
def test_attention_prefill_fallback_preserves_residual_ownership(
    monkeypatch, has_prefix, producer_direct
):
    from tokenspeed.runtime.models import kimi_k3_comm as module

    group = tuple(range(8))
    comm = module.K3AttnComm(
        SimpleNamespace(mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=group)))
    )
    partial = torch.zeros((8, 64), dtype=torch.bfloat16)
    prefix = torch.ones_like(partial) if has_prefix else None
    output = torch.full_like(partial, 3)
    fallback = Mock(side_effect=lambda value, _: value.add_(3))
    monkeypatch.setattr(module, "all_reduce", fallback)
    retained, delta = comm.prefill_reduce_for_attnres(
        partial, prefix, producer_direct=producer_direct
    )
    assert fallback.call_count == 1
    assert fallback.call_args.args[1] == group
    reduced = delta if has_prefix else retained
    torch.testing.assert_close(reduced, output)
    if has_prefix:
        assert retained is prefix
    else:
        assert delta is None
    if producer_direct:
        assert reduced.data_ptr() != partial.data_ptr()
        torch.testing.assert_close(partial, torch.zeros_like(partial))
        partial.fill_(9)  # The next producer cannot corrupt this retained result.
        torch.testing.assert_close(reduced, output)
    else:
        assert reduced is partial


@pytest.mark.parametrize(
    "rows,eligible",
    [
        (511, False),
        (512, True),
        (513, False),
        (848, True),
        (1024, True),
        (2048, True),
        (4088, True),
        (4095, False),
        (4096, True),
        (4097, False),
        (6224, True),
        (8192, True),
        (8193, False),
    ],
)
def test_attention_prefill_mix_window(monkeypatch, rows, eligible):
    from tokenspeed_kernel.ops.communication import iris_prefill

    from tokenspeed.runtime.models import kimi_k3_comm as module

    group = tuple(range(8))
    comm = module.K3AttnComm(
        SimpleNamespace(mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=group)))
    )
    partial = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    history = torch.empty((4, rows, 7168), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((7168,), dtype=torch.bfloat16, device="meta")
    expected = (object(), object())
    operation = Mock(return_value=expected)
    monkeypatch.setattr(iris_prefill, "iris_attention_prefill_mix", operation)
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


@pytest.mark.parametrize(
    "producer_direct,accepted", [(True, True), (True, False), (False, False)]
)
def test_sharded_attention_residual_is_gathered_before_moe_fallback(
    monkeypatch, producer_direct, accepted
):
    from tokenspeed.runtime.models import kimi_k3_comm as module

    group = tuple(range(8))
    routed = torch.empty((4096, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((4096, 7168), dtype=torch.bfloat16, device="meta")
    shard = torch.empty((512, 7168), dtype=torch.bfloat16, device="meta")
    full = torch.empty_like(shared)
    expected = object()
    owner = SimpleNamespace(
        mapping=SimpleNamespace(
            pp_size=1,
            attn=SimpleNamespace(tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group),
        ),
        up_proj=SimpleNamespace(
            narrowed=False,
            solution="auto",
            weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
        ),
        routed_hidden=3584,
        routed_norm=None,
        execution_plan=SimpleNamespace(
            lane_latent_norm_ar=False, comm_fusion_max_num_tokens=16
        ),
        _projection_tail=Mock(return_value=expected),
    )
    candidate = Mock(return_value=expected if accepted else None)
    gather = Mock(return_value=full)
    joined = Mock(return_value=(routed, shared))
    monkeypatch.setattr(module, "iris_kimi3_moe_tail", candidate)
    monkeypatch.setattr(module, "all_gather", gather)
    monkeypatch.setattr(module, "kimi3_join_reduce_moe", joined)
    monkeypatch.setattr(module, "_get_process_group", lambda _: "owner")
    result = module.K3MoeTailComm._tail_fused_lane_ar_replicated(
        owner,
        routed,
        shared,
        shard,
        None,
        (routed, shared) if producer_direct else None,
        4096,
        7168,
        prefix_is_sharded=True,
    )
    assert result is expected
    if producer_direct:
        assert candidate.call_args.kwargs["prefix_is_sharded"] is True
        assert candidate.call_args.args[2] is shard
    else:
        candidate.assert_not_called()
    if accepted:
        gather.assert_not_called()
        joined.assert_not_called()
        owner._projection_tail.assert_not_called()
    else:
        gather.assert_called_once_with(shard, group, dim=0, backend=None)
        owner._projection_tail.assert_called_once_with(routed, shared, full, 4096, 7168)


def test_non_iris_moe_tier_materializes_a_sharded_residual(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3_comm as module

    group = tuple(range(8))
    shard, full, routed, shared, output = (object() for _ in range(5))
    gather = Mock(return_value=full)
    monkeypatch.setattr(module, "all_gather", gather)
    tail = Mock(return_value=output)
    owner = SimpleNamespace(
        mapping=SimpleNamespace(moe=SimpleNamespace(tp_ep_group=group)),
        _tail_separate_reduce=tail,
    )
    result = module.K3MoeTailComm.run(
        owner,
        SimpleNamespace(tier=module.K3MoETailTier.SEPARATE_REDUCE),
        routed,
        shared,
        shard,
        4096,
        7168,
        None,
        prefix_is_sharded=True,
    )
    assert result is output
    gather.assert_called_once_with(shard, group, dim=0, backend=None)
    tail.assert_called_once_with(routed, shared, full, 4096, 7168)


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
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize(
    "max_rows,tail_rows",
    [(511, 0), (512, 512), (519, 512), (8192, 8192), (16384, 8192)],
)
def test_iris_preparation_caps_attnres_for_equal_tp8_groups(
    monkeypatch, pp_size, max_rows, tail_rows
):
    from tokenspeed.runtime.models import kimi_k3_comm

    group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=pp_size,
        attn=SimpleNamespace(tp_size=8, tp_group=group),
        moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_size=8, tp_ep_group=group),
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
    from tokenspeed.runtime.models import kimi_k3_comm

    attn_group = (0, 1, 2, 3)
    moe_group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=1,
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
    from tokenspeed.runtime.models import kimi_k3_comm

    attn_group = (0,)
    moe_group = tuple(range(8))
    mapping = SimpleNamespace(
        pp_size=1,
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
        enable_lamport=False,
        moe_tail_max_rows=0,
        dtype=torch.bfloat16,
        backend=None,
    )


@needs_iris
def test_iris_preparation_keeps_baseline_window_for_equal_tp4(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3_comm

    group = tuple(range(4))
    mapping = SimpleNamespace(
        pp_size=1,
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
    from tokenspeed.runtime.models import kimi_k3_comm

    monkeypatch.setattr(
        kimi_k3_comm, "current_platform", lambda: SimpleNamespace(is_cdna4=True)
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
        monkeypatch.setattr(kimi_k3_comm, "prepare_all_reduce_buffers", prepare)

        assert kimi_k3_comm.prepare_k3_all_reduce_buffers(
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


@pytest.mark.parametrize(
    "rows,producer_direct,tp,ep,narrowed,solution,accepted,attempted,pp_size",
    [
        (1, True, 8, 1, False, "auto", True, False, 1),
        (48, True, 8, 1, False, "auto", True, False, 1),
        (504, True, 8, 1, False, "auto", True, False, 1),
        (512, True, 8, 1, False, "auto", True, True, 1),
        (848, True, 8, 1, False, "auto", True, True, 1),
        (8192, True, 8, 1, False, "auto", True, True, 1),
        (8200, True, 8, 1, False, "auto", True, False, 1),
        (8192, False, 8, 1, False, "auto", True, False, 1),
        (8192, True, 1, 8, False, "auto", True, False, 1),
        (8192, True, 8, 1, True, "auto", True, False, 1),
        (8192, True, 8, 1, False, "torch", True, False, 1),
        (8192, True, 8, 1, False, "auto", False, True, 1),
        (8192, True, 8, 1, False, "auto", True, False, 2),
    ],
)
@pytest.mark.parametrize("has_norm", [False, True])
def test_row_sharded_moe_tail_selection_and_fallback(
    monkeypatch,
    rows,
    producer_direct,
    tp,
    ep,
    narrowed,
    solution,
    accepted,
    attempted,
    has_norm,
    pp_size,
):
    from tokenspeed.runtime.models import kimi_k3_comm as mod

    # Check dispatch here; GPU tests cover ownership and numerical correctness.
    routed = torch.empty((rows, 3584), dtype=torch.bfloat16, device="meta")
    shared = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    prefix = torch.empty_like(shared)
    expected = torch.empty_like(shared)
    fallback = torch.empty_like(shared)
    group = tuple(range(8))
    process_group = object()
    norm = (
        SimpleNamespace(
            weight=torch.empty(3584, dtype=torch.bfloat16, device="meta"),
            variance_epsilon=1e-5,
        )
        if has_norm
        else None
    )
    projection = SimpleNamespace(
        narrowed=narrowed,
        solution=solution,
        weight=torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta"),
    )
    owner = SimpleNamespace(
        mapping=SimpleNamespace(
            pp_size=pp_size,
            attn=SimpleNamespace(tp_size=8, tp_group=group),
            moe=SimpleNamespace(tp_size=tp, ep_size=ep, tp_ep_group=group),
        ),
        routed_hidden=3584,
        routed_norm=norm,
        up_proj=projection,
        execution_plan=SimpleNamespace(
            lane_latent_norm_ar=False, comm_fusion_max_num_tokens=16
        ),
        _projection_tail=Mock(return_value=fallback),
    )
    candidate = Mock(return_value=expected if accepted else None)
    joined = Mock(return_value=(routed, shared))
    resolve = Mock(return_value=process_group)
    monkeypatch.setattr(mod, "iris_kimi3_moe_tail", candidate)
    monkeypatch.setattr(mod, "kimi3_join_reduce_moe", joined)
    monkeypatch.setattr(mod, "_get_process_group", resolve)
    symm_outputs = (routed, shared) if producer_direct else None

    output = mod.K3MoeTailComm._tail_fused_lane_ar_replicated(
        owner,
        routed,
        shared,
        prefix,
        None,
        symm_outputs,
        rows,
        7168,
        prefix_is_sharded=False,
    )

    if attempted:
        candidate.assert_called_once_with(
            routed,
            shared,
            prefix,
            projection.weight,
            prefix_is_sharded=False,
            norm_weight=norm.weight if has_norm else None,
            eps=norm.variance_epsilon if has_norm else None,
            group=process_group,
        )
        resolve.assert_called_once_with(group)
    else:
        candidate.assert_not_called()
        resolve.assert_not_called()
    if attempted and accepted:
        assert output is expected
        joined.assert_not_called()
        owner._projection_tail.assert_not_called()
    else:
        assert output is fallback
        joined.assert_called_once_with(
            routed,
            shared,
            lane=None,
            symm_outputs=symm_outputs,
            routed_hidden=3584,
            routed_norm=norm,
            group=group,
            enable_lane_norm=False,
            max_token_num=16,
        )
        owner._projection_tail.assert_called_once_with(
            routed, shared, prefix, rows, 7168
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
