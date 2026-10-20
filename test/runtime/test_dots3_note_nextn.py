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


"""MTP checkpoint ownership and accepted-prefix narrowing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.models.dots3_note_nextn import Dots3NoteForCausalLMNextN


def test_draft_loader_preserves_own_embedding_and_shares_only_head(monkeypatch):
    from tokenspeed.runtime.execution.factory import _share_target_embed_and_head

    draft = Dots3NoteForCausalLMNextN.__new__(Dots3NoteForCausalLMNextN)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(mtp_layer_offset=46)
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(8, 4)
    draft.model.eh_proj = nn.Linear(8, 4, bias=False)
    draft.model.decoder = nn.Module()
    draft.model.decoder.input_layernorm = nn.LayerNorm(4, bias=False)
    draft.lm_head = nn.Linear(4, 8, bias=False)
    draft.post_load_weights = Mock()
    own_embedding = draft.model.embed_tokens.weight
    loaded = draft.load_weights(
        [
            ("model.embed_tokens.weight", torch.full((8, 4), 99.0)),
            ("model.mtp.embed_tokens.weight", torch.full((8, 4), 2.0)),
            ("model.layers.46.eh_proj.weight", torch.full((4, 8), 3.0)),
            ("model.layers.46.input_layernorm.weight", torch.full((4,), 4.0)),
            ("model.layers.45.input_layernorm.weight", torch.full((4,), 99.0)),
        ]
    )
    assert loaded == {
        "model.embed_tokens.weight",
        "model.eh_proj.weight",
        "model.decoder.input_layernorm.weight",
    }
    assert torch.all(own_embedding == 2)
    assert torch.all(draft.model.eh_proj.weight == 3)
    assert torch.all(draft.model.decoder.input_layernorm.weight == 4)
    main_embedding = nn.Parameter(torch.ones(8, 4))
    main_head = nn.Parameter(torch.zeros(8, 4))
    layout = SimpleNamespace(tp_group=None, tp_size=1, dp_lm_head_tp=False)
    draft.logits_processor = layout
    target = SimpleNamespace(
        logits_processor=layout,
        get_embed_and_head=lambda: (main_embedding, main_head),
        lm_head=SimpleNamespace(quant_method=None),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    runner = SimpleNamespace(
        model=target, mapping=SimpleNamespace(has_pp=False, is_last_pp_rank=True)
    )
    _share_target_embed_and_head(runner, draft)
    assert draft.model.embed_tokens.weight is own_embedding
    assert draft.lm_head.weight is main_head
    with pytest.raises(KeyError):
        draft.load_weights([("model.layers.46.unknown.weight", torch.ones(1))])


@pytest.mark.parametrize("mode_name", ["MIXED", "DECODE"])
def test_first_draft_writes_all_rows_but_attends_only_live_rows(monkeypatch, mode_name):
    import contextlib

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.models import dots3_note_nextn as impl

    mode = ForwardMode[mode_name]
    attn = impl.Dots3NoteDraftAttention.__new__(impl.Dots3NoteDraftAttention)
    nn.Module.__init__(attn)
    attn.num_local_heads = 1
    attn.head_start = 0
    attn.v_head_dim = attn.kv_lora_rank = 2
    attn.w_vc = None
    attn.q_a_proj = attn.kv_a_proj_with_mqa = attn.q_b_proj = lambda x: (x, None)
    attn.normalize_latents = lambda q, kv: (q, kv)
    attn.g_proj = lambda x: (torch.zeros(x.shape[0], 1), None)
    attn.o_proj = lambda x: (x, None)
    rows = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    positions = torch.arange(5)
    live = torch.tensor([1, 4])
    events = []

    def absorb(q, latent, pos, ctx, slots):
        events.append("write")
        assert ctx.forward_mode == ForwardMode.DECODE
        torch.testing.assert_close(q, rows)
        torch.testing.assert_close(pos, positions)
        torch.testing.assert_close(slots, positions)
        return q

    def core(q, k, v, *, positions, ctx, record_kv_cache):
        events.append("attend")
        torch.testing.assert_close(q, rows[live])
        assert record_kv_cache == (mode_name == "MIXED")
        return q

    attn._absorb = absorb
    attn.attn_mqa = core
    backend = SimpleNamespace(
        forward_write_locations=lambda layer, mode: positions,
        override_num_extends=lambda n: contextlib.nullcontext(),
    )
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=2,
        num_extends=1 if mode_name == "MIXED" else 0,
        input_num_tokens=5,
        forward_mode=mode,
        output_layout=None,
        gather_ids=live,
        draft_narrowing=SimpleNamespace(
            publish_accepted_prefix=lambda: events.append("publish")
        ),
    )
    monkeypatch.setattr(
        impl, "mla_project_value", lambda x, w, *, out: out.copy_(x.flatten(1))
    )
    result = attn(positions, rows, ctx, SimpleNamespace(pre_attn_comm=lambda x, c: x))
    assert events == ["publish", "write", "attend"]
    torch.testing.assert_close(result, rows[live] / 2)
    torch.testing.assert_close(
        impl.Dots3NoteDraftDecoderLayer.narrow_residual(None, rows, ctx), rows[live]
    )


def test_target_prefill_marks_pd_ready_after_all_cache_writes():
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.models.dots3_note import Dots3NoteAttention

    events = []
    backend = SimpleNamespace(
        step_counter=SimpleNamespace(record_cache=lambda: events.append("ready"))
    )
    backend.record_pd_cache_step = (
        lambda *a, **kw: AttentionBackend.record_pd_cache_step(backend, *a, **kw)
    )
    attn = Dots3NoteAttention.__new__(Dots3NoteAttention)
    nn.Module.__init__(attn)
    rows = torch.ones(2, 4)

    def compute(*args):
        events.append("write_latent_and_index")
        return rows

    attn._forward = compute
    ctx = SimpleNamespace(attn_backend=backend, forward_mode=ForwardMode.EXTEND)
    assert attn(torch.arange(2), rows, ctx, None) is rows
    assert events == ["write_latent_and_index", "ready"]
    events.clear()
    ctx.forward_mode = ForwardMode.DECODE
    attn(torch.arange(2), rows, ctx, None)
    assert events == ["write_latent_and_index"]
