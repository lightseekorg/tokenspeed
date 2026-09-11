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

"""Qwen's complete DeepEP break boundary, using small deterministic experts.

These seam tests exercise the actual model forward and shared-expert callback.
The distributed DeepEP test separately validates vendor dispatch/combine: the
toy experts here intentionally do not stand in for collective-liveness tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.moe import utils as moe_utils
from tokenspeed.runtime.models import qwen3_5_moe
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

_HIDDEN = 4
_BUCKET = 8


class _Router(nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.events.append(("router", x.shape[0]))
        return torch.stack((x[:, 0], -x[:, 0]), dim=-1), None


class _TopK(nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> SimpleNamespace:
        self.events.append(("topk", x.shape[0]))
        return SimpleNamespace(ids=logits.argmax(dim=-1))

    def empty_topk_output(
        self,
        device: torch.device,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> SimpleNamespace:
        self.events.append(("empty_topk", hidden_states.shape[0]))
        return SimpleNamespace(ids=torch.empty(0, device=device, dtype=torch.int64))


class _Experts(nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_output: SimpleNamespace,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        low_latency: bool,
        overlap_fn,
    ) -> torch.Tensor:
        self.events.append(
            (
                "experts",
                hidden_states.shape[0],
                num_global_tokens,
                max_num_tokens_per_gpu,
                low_latency,
            )
        )
        if overlap_fn is not None:
            overlap_fn()
        return hidden_states * (topk_output.ids[:, None] + 1)


class _SharedExpert(nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events
        self.scale = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.events.append(("shared", x.shape[0]))
        # A fresh result each call catches a callback output accidentally
        # retained from capture or an earlier replay.
        return x * self.scale


def _merge_shared(
    x: torch.Tensor,
    weight: torch.Tensor,
    shared: torch.Tensor,
    routed: torch.Tensor,
) -> None:
    routed.add_(torch.sigmoid(x @ weight)[:, None] * shared)


@pytest.fixture(autouse=True)
def _runtime_seams(monkeypatch):
    monkeypatch.setattr(moe_utils, "get_deepep_mode", lambda: moe_utils.DeepEPMode.AUTO)
    monkeypatch.setattr(qwen3_5_moe, "fused_gate_sigmoid_mul_add", _merge_shared)


def _block(rank: int, shared: bool, gated: bool, device: str):
    # Retain the actual Qwen forward/adapter while avoiding checkpoint loading
    # and vendor kernels unrelated to the break's lifetime contract.
    block = Qwen3_5MoeSparseMoeBlock.__new__(Qwen3_5MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.use_deepep = True
    block.mapping = Mapping(
        rank=rank,
        world_size=4,
        attn_tp_size=2,
        attn_dp_size=2,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
    )
    block.comm_manager = CommManager(
        mapping=block.mapping,
        layer_id=0,
        is_moe=True,
        prev_is_moe=True,
        input_layernorm=None,
        post_attn_layernorm=None,
    )
    events = []
    block.gate = _Router(events)
    block.topk = _TopK(events)
    block.experts = _Experts(events)
    block.shared_expert = _SharedExpert(events) if shared else None
    block.shared_expert_gate = nn.Linear(_HIDDEN, 1, bias=False) if gated else None
    if block.shared_expert_gate is not None:
        with torch.no_grad():
            block.shared_expert_gate.weight.fill_(0.25)
    block.to(device)
    return block, events


def _ctx(mode: ForwardMode, all_decode_or_idle: bool) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=int(mode == ForwardMode.EXTEND),
        input_num_tokens=_BUCKET,
        forward_mode=mode,
        global_num_tokens=[_BUCKET] * 4,
        all_decode_or_idle=all_decode_or_idle,
        all_extend=mode == ForwardMode.EXTEND,
    )


def _reference(x: torch.Tensor, shared_scale: float, gated: bool) -> torch.Tensor:
    # Negative first coordinates choose expert 1; nonnegative choose expert 0.
    result = x * (1 + (x[:, :1] < 0).to(x.dtype))
    if shared_scale:
        shared = x * shared_scale
        result += (
            torch.sigmoid(x.sum(dim=-1, keepdim=True) * 0.25) * shared
            if gated
            else shared
        )
    return result


@pytest.mark.parametrize("shared,gated", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("rank,real,live", [(0, 5, 4), (1, 5, 1), (1, 1, 0), (1, 8, 4)])
def test_adapter_routes_only_physical_live_rows_and_clears_tail(
    monkeypatch, shared: bool, gated: bool, rank: int, real: int, live: int
) -> None:
    block, events = _block(rank, shared, gated, "cpu")
    monkeypatch.setattr(qwen3_5_moe, "current_valid_rows", lambda: real)
    x = torch.full((4, _HIDDEN), torch.nan)
    x[:live] = torch.arange(live * _HIDDEN).reshape(live, _HIDDEN) - 2.0
    dst = torch.full_like(x, torch.nan)
    result = block._forward_deepep_bcg_into(x, _ctx(ForwardMode.EXTEND, False), dst)

    assert result is dst
    expected = torch.zeros_like(x)
    expected[:live] = _reference(x[:live], 3.0 if shared else 0.0, gated)
    torch.testing.assert_close(result, expected)
    assert ("router", live) in events
    assert ("empty_topk" if live == 0 else "topk", live) in events
    # In particular, the empty source rank must still enter the experts call.
    assert ("experts", live, 16, 4, False) in events
    assert (("shared", live) in events) is shared


def test_adapter_refreshes_shared_output_counts_and_mode(monkeypatch) -> None:
    block, events = _block(1, True, True, "cpu")
    monkeypatch.setattr(qwen3_5_moe, "current_valid_rows", lambda: None)
    x = torch.arange(4 * _HIDDEN).reshape(4, _HIDDEN).float() - 5
    dst = torch.empty_like(x)
    for mode, low_latency, scale in [
        (ForwardMode.EXTEND, False, 3.0),
        (ForwardMode.DECODE, True, 7.0),
        (ForwardMode.EXTEND, False, 2.0),
    ]:
        block.shared_expert.scale = scale
        ctx = _ctx(mode, low_latency)
        # These hints are recomputed from each context, not frozen scalar args.
        ctx.global_num_tokens = [8, 8, int(scale) * 2, int(scale) * 2]
        block._forward_deepep_bcg_into(x, ctx, dst)
        torch.testing.assert_close(dst, _reference(x, scale, True))
        assert (
            "experts",
            4,
            8 + int(scale) * 2,
            max(4, int(scale)),
            low_latency,
        ) in events


def test_eager_forward_preserves_callers_count_hints() -> None:
    block, events = _block(0, True, False, "cpu")
    x = torch.ones(3, _HIDDEN)
    result = block(x, 71, 9, _ctx(ForwardMode.EXTEND, False))
    torch.testing.assert_close(result, x * 4)
    assert ("experts", 3, 71, 9, False) in events


def test_structural_stub_skips_complete_moe_and_owns_distinct_outputs(monkeypatch):
    block, events = _block(1, True, True, "cpu")
    ctx = _ctx(ForwardMode.EXTEND, False)
    x = torch.full((4, _HIDDEN), torch.nan)
    recorded = []

    def record(fn, dst, *args, capture_stub, **kwargs):
        recorded.append((fn, dst, args))
        return capture_stub(*args, **kwargs)

    monkeypatch.setattr(qwen3_5_moe, "is_breakable_capture_active", lambda: True)
    monkeypatch.setattr(qwen3_5_moe, "break_here", record)
    first = block(x, 1234, 1234, ctx)
    second = block(x, 5678, 5678, ctx)

    assert not events
    assert first.data_ptr() != second.data_ptr()
    assert torch.count_nonzero(first) == 0
    assert torch.count_nonzero(second) == 0
    for fn, dst, args in recorded:
        assert fn == block._forward_deepep_bcg_into
        assert args[0] is x
        assert args[1] is ctx
        assert args[2] is dst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_capture_replays_complete_moe_and_preserves_delayed_output() -> None:
    block, events = _block(1, True, True, "cuda")
    x = torch.zeros(4, _HIDDEN, device="cuda")
    capture_ctx = _ctx(ForwardMode.EXTEND, False)

    def forward():
        first = block(x + 1, 16, 4, capture_ctx)
        second = block(x * -2, 16, 4, capture_ctx)
        # The first same-shaped handoff stays live across the second break.
        return first + second

    with torch.no_grad(), active_forward(capture_ctx):
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        events.clear()
        cap = BreakableCapture(pool=None, stream=None)
        with cap:
            captured_output = forward()
        assert not events, "structural capture must never enter routing or experts"

        for real, live, scale in [(5, 1, 3.0), (1, 0, 7.0), (8, 4, 2.0)]:
            fresh = torch.randn_like(x)
            fresh[live:] = torch.nan
            x.copy_(fresh)
            block.shared_expert.scale = scale
            live_ctx = _ctx(ForwardMode.EXTEND, False)
            with active_forward(live_ctx):
                cap.replay(valid_rows=real)
            torch.cuda.synchronize()

            expected = torch.zeros_like(x)
            expected[:live] = _reference(fresh[:live] + 1, scale, True) + _reference(
                fresh[:live] * -2, scale, True
            )
            torch.testing.assert_close(captured_output, expected)
            assert ("experts", live, 16, 4, False) in events
            assert ("shared", live) in events
