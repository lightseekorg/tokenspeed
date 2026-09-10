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

from tokenspeed.runtime.models.kimi_k3_comm import _tail_finalize_top_k


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
