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

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import DeepseekV41DSpark
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout


@pytest.mark.parametrize(
    "batches",
    [
        (16, 16),
        (3, 19),
        (0, 8),
        (8, 0),
        (0, 0),
        (8, 8, 8, 8),
        (3, 9, 0, 7),
        (0, 0, 8, 0),
        (0, 0, 0, 0),
        (4,) * 8,
        (0, 0, 1, 0, 0, 2, 0, 0),
    ],
)
def test_draft_and_idle_use_proposal_counts(batches):
    tp = 16 // len(batches)
    bs = [count for count in batches for _ in range(tp)]
    verify = [n * 6 for n in bs]
    expected = [n * 5 for n in bs]
    drafter = NS(attn_dp_size=len(batches), spec_num_tokens=6, block_size=5)
    assert DeepseekV41DSpark.idle_forward_global_num_tokens(
        drafter, verify, bs, global_num_decodes=bs
    ) == ([expected] if any(bs) else [])


@pytest.mark.parametrize(
    "left,right,left_extends,right_extends",
    [(2, 3, 0, 0), (3, 0, 0, 2), (16, 16, 0, 0), (2, 3, 1, 2)],
)
@pytest.mark.parametrize("markov_vocab", [8, 4])
def test_active_draft_passes_its_collective_rows(
    monkeypatch, left, right, left_extends, right_extends, markov_vocab
):
    from tokenspeed.runtime.execution.drafter import deepseek_v41_dspark as impl

    bs = [left + left_extends] * 8 + [right + right_extends] * 8
    decodes = [left] * 8 + [right] * 8
    rows = [left_extends * 73 + left * 6] * 8 + [right_extends * 73 + right * 6] * 8
    requests = torch.arange(left).repeat_interleave(6)
    history = torch.zeros(left, 128, dtype=torch.int32)
    backend = NS(
        query_metadata=lambda mode: NS(request_indices=requests),
        window_slots=lambda *args: (history, None),
    )
    ctx = ForwardContext(
        bs=left + left_extends,
        num_extends=left_extends,
        input_num_tokens=rows[0],
        forward_mode=ForwardMode.MIXED if left_extends else ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(left_extends, left_extends, left, 6),
        attn_backend=backend,
        token_to_kv_pool=object(),
        all_decode_or_idle=not (left_extends or right_extends),
        global_bs=bs,
        global_num_decodes=decodes,
        global_num_tokens=rows,
    )
    seen = []

    def backbone(bonus, start, slots, pool, draft_ctx):
        assert draft_ctx.input_num_tokens == left * 5
        assert draft_ctx.global_num_tokens == [n * 5 for n in decodes]
        assert draft_ctx.global_num_decodes == decodes
        assert draft_ctx.global_bs == bs
        assert draft_ctx.forward_mode == ForwardMode.DECODE
        seen.append(draft_ctx)
        return torch.zeros(left, 5, 4)

    drafter = NS(
        attn_dp_size=2,
        spec_num_tokens=6,
        block_size=5,
        model=NS(
            forward_backbone=backbone,
            local_base_logits=lambda *args: torch.zeros(left, 5, 8),
            markov_head=object(),
            markov_projection=NS(num_embeddings_per_partition=markov_vocab),
        ),
        lm_head=NS(weight=torch.zeros(8, 4)),
        tp_group=(),
        candidates=None,
        partials=None,
        draft_tokens_buf=torch.full((left, 5), 7, dtype=torch.int32),
    )
    monkeypatch.setattr(impl, "sample_dspark_block_greedy", Mock())
    next_tokens = torch.full((left + left_extends, 6), -1, dtype=torch.int32)
    if markov_vocab != 8:
        with pytest.raises(ValueError, match="local vocabulary"):
            DeepseekV41DSpark._draft_decode_rows(
                drafter, ctx, torch.zeros(left, dtype=torch.int64), next_tokens
            )
        return
    DeepseekV41DSpark._draft_decode_rows(
        drafter, ctx, torch.zeros(left, dtype=torch.int64), next_tokens
    )
    assert len(seen) == 1
    assert torch.equal(
        next_tokens[left_extends:, 1:], torch.full((left, 5), 7, dtype=torch.int32)
    )
    assert torch.all(next_tokens[:left_extends] == -1)
    assert torch.all(next_tokens[:, 0] == -1)


@pytest.mark.parametrize("decodes", [[0, 0], [0, 3], [2, 1]])
def test_idle_schedule_uses_decode_counts_in_mixed_round(decodes):
    drafter = NS(block_size=5)
    assert DeepseekV41DSpark.idle_forward_global_num_tokens(
        drafter,
        [73 + decodes[0] * 6, 146 + decodes[1] * 6],
        [1 + decodes[0], 2 + decodes[1]],
        global_num_decodes=decodes,
    ) == ([[count * 5 for count in decodes]] if any(decodes) else [])


@pytest.mark.parametrize("dp", [2, 4, 8])
def test_supported_replica_count_reaches_stage_validation(dp):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models import deepseek_v41_dspark as model

    mapping = Mapping(
        rank=0,
        world_size=16,
        attn_tp_size=16 // dp,
        attn_dp_size=dp,
        dense_tp_size=16 // dp,
        moe_tp_size=1,
        moe_ep_size=16,
        lm_head_tp_size=16 // dp,
    )
    config = NS(
        dspark_num_stages=0,
        num_nextn_predict_layers=0,
        dspark_block_size=5,
        dspark_target_layer_ids=(0,),
        sliding_window=128,
        num_hidden_layers=40,
    )
    with pytest.raises(ValueError, match="Invalid V4.1 DSpark stage"):
        model.DeepseekV41DSparkModel(config, mapping, None, "mtp")


@pytest.mark.parametrize(
    "attention_tp,dense_tp,expected",
    [
        (8, 8, True),
        (4, 4, True),
        (4, 8, True),
        (4, 2, False),
        (2, 4, True),
        (2, 8, True),
        (8, 4, False),
    ],
)
def test_shared_expert_tp_can_span_attention_replicas(attention_tp, dense_tp, expected):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.deepseek_v41_parallel import supports_ascend_dp_moe

    mapping = Mapping(
        rank=0,
        world_size=16,
        attn_tp_size=attention_tp,
        attn_dp_size=16 // attention_tp,
        dense_tp_size=dense_tp,
        moe_tp_size=1,
        moe_ep_size=16,
    )
    assert supports_ascend_dp_moe(mapping, "ascend_w4a4", True) is expected
    assert not supports_ascend_dp_moe(mapping, "ascend_w4a4", False)


@pytest.mark.parametrize("decodes", [[0, 0], [0, 3]])
def test_prefill_only_rank_joins_remote_draft(decodes):
    context = ForwardContext(
        attn_backend=object(),
        token_to_kv_pool=object(),
        bs=1,
        num_extends=1,
        input_num_tokens=73,
        forward_mode=ForwardMode.EXTEND,
        output_layout=ForwardOutputLayout(1, 1, 0, 6),
        global_bs=[1, decodes[1]],
        global_num_tokens=[73, decodes[1] * 6],
        global_num_decodes=decodes,
        all_decode_or_idle=False,
    )
    runner = Mock()
    drafter = NS(block_size=5, draft_model_runner=runner)
    tokens = torch.full((1, 6), -1, dtype=torch.int32)
    DeepseekV41DSpark._draft_decode_rows(
        drafter, context, torch.empty(0, dtype=torch.int64), tokens
    )
    if any(decodes):
        draft = runner.forward.call_args.args[0]
        assert draft.forward_mode == ForwardMode.IDLE
        assert draft.input_num_tokens == 0
        assert draft.global_num_tokens == [count * 5 for count in decodes]
        assert draft.global_num_decodes == decodes
    else:
        runner.forward.assert_not_called()
    assert torch.all(tokens == -1)
