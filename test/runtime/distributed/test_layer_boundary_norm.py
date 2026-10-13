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

"""``--layer-boundary-norm``: CommManager's add+norm at a layer boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.utils.env import global_server_args_dict

HIDDEN = 16


class StubNorm:
    """Records how it was called; fused form normalizes the fp32 sum."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

    def __call__(self, x, residual=None):
        if residual is None:
            self.calls.append(1)
            return self._norm(x)
        self.calls.append(2)
        total = x.float() + residual.float()
        return self._norm(total).to(x.dtype), total.to(x.dtype)


def _manager(mode: str, monkeypatch, norm: StubNorm) -> CommManager:
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", mode)
    monkeypatch.setitem(global_server_args_dict, "moe_combine_order", "rank")
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", False)
    mapping = SimpleNamespace(
        has_attn_tp=False,
        attn=SimpleNamespace(
            tp_rank=0, tp_group=(0,), tp_size=1, dp_size=1, qcp_size=1, has_qcp=False
        ),
        dense=SimpleNamespace(tp_size=1),
        moe=SimpleNamespace(tp_ep_size=1),
    )
    return CommManager(
        mapping=mapping,
        layer_id=0,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
        input_layernorm=norm,
        post_attn_layernorm=norm,
    )


def _inputs():
    torch.manual_seed(0)
    hidden = torch.randn(3, HIDDEN).to(torch.bfloat16)
    residual = torch.randn(3, HIDDEN).to(torch.bfloat16)
    return hidden, residual


def test_unfused_materializes_the_bf16_sum_before_a_standalone_norm(monkeypatch):
    norm = StubNorm()
    manager = _manager("unfused", monkeypatch, norm)
    hidden, residual = _inputs()
    out, new_residual = manager.input_reduce_norm(hidden.clone(), residual.clone())
    bf16_sum = hidden + residual
    assert norm.calls == [1]
    assert torch.equal(new_residual, bf16_sum)
    assert torch.equal(out, StubNorm._norm(bf16_sum))
    assert out.dtype == torch.bfloat16


def test_fused_keeps_the_add_norm_kernel(monkeypatch):
    norm = StubNorm()
    manager = _manager("fused", monkeypatch, norm)
    hidden, residual = _inputs()
    out, new_residual = manager.input_reduce_norm(hidden.clone(), residual.clone())
    assert norm.calls == [2]
    # The fused form normalizes the fp32 sum; it differs from the unfused
    # norm of the rounded sum, which is the whole point of the switch.
    assert torch.equal(new_residual, hidden + residual)
    assert torch.equal(
        out, StubNorm._norm(hidden.float() + residual.float()).to(out.dtype)
    )


@pytest.mark.parametrize("mode", ["fused", "unfused"])
def test_first_layer_has_no_residual_to_add(monkeypatch, mode):
    norm = StubNorm()
    manager = _manager(mode, monkeypatch, norm)
    hidden, _ = _inputs()
    out, new_residual = manager.input_reduce_norm(hidden, None)
    assert new_residual is hidden
    assert norm.calls == [1]
    assert torch.equal(out, StubNorm._norm(hidden))


@pytest.mark.parametrize("mode", ["fused", "unfused"])
def test_intra_layer_norm_stays_fused_under_every_mode(monkeypatch, mode):
    norm = StubNorm()
    manager = _manager(mode, monkeypatch, norm)
    hidden, residual = _inputs()
    manager.intra_layer_add_norm(hidden.clone(), residual.clone())
    assert norm.calls == [2]


def test_final_norm_follows_the_switch(monkeypatch):
    hidden, residual = _inputs()
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False), query_shard=None
    )

    norm = StubNorm()
    manager = _manager("unfused", monkeypatch, norm)
    out, residual_out = manager.final_norm(hidden.clone(), residual.clone(), ctx, norm)
    assert norm.calls == [1]
    assert torch.equal(residual_out, hidden + residual)
    assert torch.equal(out, StubNorm._norm(hidden + residual))

    norm = StubNorm()
    manager = _manager("fused", monkeypatch, norm)
    manager.final_norm(hidden.clone(), residual.clone(), ctx, norm)
    assert norm.calls == [2]


def test_unfused_vetoes_the_fused_all_reduce_norm_where_it_is_relied_upon(
    monkeypatch,
):
    # Even with the fusion flag left on and a TP topology that would fuse,
    # should_fuse itself says no: the unfused boundary norm needs the bf16
    # sum materialized first.
    monkeypatch.setitem(global_server_args_dict, "moe_combine_order", "rank")
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 1024)
    mapping = SimpleNamespace(
        has_attn_tp=True,
        attn=SimpleNamespace(
            tp_rank=0, tp_group=(0, 1), tp_size=2, dp_size=1, qcp_size=1, has_qcp=False
        ),
        dense=SimpleNamespace(tp_size=2),
        moe=SimpleNamespace(tp_ep_size=2),
    )
    for mode, fuses in (("fused", True), ("unfused", False)):
        monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", mode)
        monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", True)
        manager = CommManager(
            mapping=mapping,
            layer_id=1,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=False,
            query_sharded=False,
        )
        assert manager.should_fuse(4) is fuses, mode
