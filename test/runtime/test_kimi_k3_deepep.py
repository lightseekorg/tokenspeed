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

"""K3 DeepEP token homes and shared-expert TP composition."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models import kimi_k3_deepep as deep
from tokenspeed.runtime.models.kimi_k3_deepep import KimiLinearMoEDeepEP


@pytest.mark.parametrize("tp,dp,pp", [(8, 1, 4), (8, 4, 1), (1, 4, 1)])
def test_deepep_accepts_stage_local_tp_dp_layout(monkeypatch, tp, dp, pp):
    monkeypatch.setattr(deep, "get_moe_backend", lambda: object())
    monkeypatch.setattr(
        deep.Kimi3MoEExecutionPlan,
        "build",
        mock.Mock(return_value=SimpleNamespace(use_marlin=True)),
    )
    gate = mock.Mock(side_effect=RuntimeError("validated layout"))
    monkeypatch.setattr(deep, "KimiLinearMoEGate", gate)
    config = SimpleNamespace(
        num_experts=64,
        num_experts_per_token=2,
        routed_scaling_factor=1.0,
        routed_expert_hidden_size=32,
        hidden_size=64,
        hidden_act="situ",
        activation_situ_beta=1.0,
        activation_situ_linear_beta=None,
    )
    with pytest.raises(RuntimeError, match="validated layout"):
        KimiLinearMoEDeepEP(
            config=config,
            mapping=SimpleNamespace(
                world_size=tp * dp * pp,
                attn=SimpleNamespace(tp_size=tp, dp_size=dp, cp_size=1),
                moe=SimpleNamespace(tp_size=1, dp_size=1, ep_size=tp * dp),
            ),
            layer_index=0,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )
    gate.assert_called_once_with(64, 64)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("num_tokens", [0, 3])
def test_deepep_routes_unique_rows_and_joins_attention_tp(
    monkeypatch, rank, num_tokens
):
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    prefix = torch.ones_like(hidden) * 7
    locations = torch.tensor([1, 0, 3])[:num_tokens]
    counts = [(num_tokens + 1) // 2, num_tokens // 2]
    start = sum(counts[:rank])
    source = hidden[start : start + counts[rank]]
    live = (locations > 0).to(hidden.dtype)[:, None]
    full_routed = hidden * 2 * live
    events = []

    class TopK:
        def __call__(self, rows, logits, *, output_format):
            return SimpleNamespace(
                topk_ids=torch.zeros((len(rows), 1), dtype=torch.int32),
                topk_weights=torch.ones((len(rows), 1)),
            )

        def empty_topk_output(self, device, *, hidden_states, router_logits):
            return self(hidden_states, router_logits, output_format=None)

    def experts(
        *,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
        do_finalize,
        low_latency,
        overlap_fn
    ):
        events.append("dispatch")
        torch.testing.assert_close(hidden_states, source)
        expected_live = locations[start : start + len(source)] > 0
        assert torch.equal(topk_output.topk_ids[:, 0] >= 0, expected_live)
        overlap_fn()
        events.append("combine")
        return source * 2 * expected_live[:, None]

    def gather(rows, *, group, scattered_num_tokens):
        assert group == (4, 5) and scattered_num_tokens == counts
        torch.testing.assert_close(rows, full_routed[start : start + len(source)])
        return full_routed

    def reduce(rows, group):
        assert group == (4, 5)
        expected = hidden * 3
        expected[:, rank * 2 : rank * 2 + 2] += (
            full_routed[:, rank * 2 : rank * 2 + 2] * 5
        )
        torch.testing.assert_close(rows, expected)
        return hidden * 6 + full_routed * 5

    monkeypatch.setattr(deep, "token_all_gather", gather)
    monkeypatch.setattr(deep, "all_reduce", reduce)
    monkeypatch.setattr(deep, "use_deepep_low_latency", lambda ctx, dp: True)
    layer = SimpleNamespace(
        mapping=SimpleNamespace(
            attn=SimpleNamespace(tp_size=2, tp_rank=rank, tp_group=(4, 5), dp_size=4)
        ),
        num_experts=8,
        routed_hidden=4,
        gate=lambda rows: torch.zeros(len(rows), 8),
        topk=TopK(),
        routed_expert_down_proj=lambda rows: (rows, None),
        shared_experts=lambda rows, down_out: rows * 3,
        experts=experts,
        routed_expert_norm=None,
        _shard_up_projection=True,
        routed_expert_up_proj=SimpleNamespace(
            shard_slice=(rank * 2, 2),
            project_shard=lambda rows: rows[:, rank * 2 : rank * 2 + 2] * 5,
        ),
    )
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        attn_backend=SimpleNamespace(decode_window_locations=lambda: locations),
    )
    output = KimiLinearMoEDeepEP.forward(
        layer,
        hidden,
        prefix,
        num_global_tokens=num_tokens * 4,
        max_num_tokens_per_gpu=num_tokens,
        ctx=ctx,
    )
    assert events == ["dispatch", "combine"]
    torch.testing.assert_close(output, prefix + hidden * 6 + full_routed * 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
