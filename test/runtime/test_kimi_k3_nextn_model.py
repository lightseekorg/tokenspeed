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

"""Embed/head contract of the Kimi-K3 NextN (MTP) draft on a prefill pipeline.

Only the last stage builds this draft (``create_model_runner``), and that stage
holds the target head but not its embedding, so the draft keeps the embedding
shard its checkpoint ships and shares only the head; off the pipeline both are
shared. Building the real NextN layer needs a GPU, so these tests check the
contract on hand-built modules.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.kimi_k3_nextn import (  # noqa: E402
    KimiK3ForConditionalGenerationNextN,
    KimiK3NextNForCausalLM,
)

NUM_TARGET_LAYERS = 93
NEXTN_PREFIX = f"model.layers.{NUM_TARGET_LAYERS}."


def _mapping(*, has_pp: bool) -> SimpleNamespace:
    return SimpleNamespace(
        has_pp=has_pp,
        is_last_pp_rank=True,
        attn=SimpleNamespace(tp_rank=0, tp_size=1, has_dp=False),
        moe=SimpleNamespace(ep_rank=0, ep_size=1),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=NUM_TARGET_LAYERS,
        num_experts=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
    )


def _draft(*, has_pp: bool) -> KimiK3NextNForCausalLM:
    """The draft with a tiny embedding and head standing in for the GPU-only
    NextN layer."""
    draft = KimiK3NextNForCausalLM.__new__(KimiK3NextNForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.config = _config()
    draft.mapping = _mapping(has_pp=has_pp)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = torch.nn.Embedding(6, 4)
    draft.lm_head = torch.nn.Linear(4, 6, bias=False)
    draft.logits_processor = None
    return draft


@pytest.fixture
def no_cuda_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def test_filter_selects_the_nextn_layer() -> None:
    draft = _draft(has_pp=True)
    assert draft.checkpoint_weight_name_filter(f"{NEXTN_PREFIX}embed_tokens.weight")
    assert draft.checkpoint_weight_name_filter(f"{NEXTN_PREFIX}shared_head.head.weight")
    assert not draft.checkpoint_weight_name_filter("model.layers.92.mlp.weight")


def test_last_stage_keeps_its_own_embedding_and_shares_the_head(no_cuda_sync) -> None:
    draft = _draft(has_pp=True)
    own_embedding = draft.model.embed_tokens.weight
    target_head = torch.nn.Parameter(torch.full((6, 4), 7.0))

    draft.set_embed_and_head(None, target_head)

    assert draft.model.embed_tokens.weight is own_embedding
    assert draft.lm_head.weight is target_head
    # What the factory checks after sharing: the draft still owns an embedding.
    reported_embed, reported_head = draft.get_embed_and_head()
    assert reported_embed is own_embedding
    assert reported_head is target_head


def test_single_stage_shares_both_target_weights(no_cuda_sync) -> None:
    draft = _draft(has_pp=False)
    target_embed = torch.nn.Parameter(torch.full((6, 4), 3.0))
    target_head = torch.nn.Parameter(torch.full((6, 4), 7.0))

    draft.set_embed_and_head(target_embed, target_head)

    reported_embed, reported_head = draft.get_embed_and_head()
    assert reported_embed is target_embed
    assert reported_head is target_head


def test_multimodal_wrapper_forwards_the_contract(no_cuda_sync) -> None:
    inner = _draft(has_pp=True)
    wrapper = KimiK3ForConditionalGenerationNextN.__new__(
        KimiK3ForConditionalGenerationNextN
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = inner
    target_head = torch.nn.Parameter(torch.full((6, 4), 7.0))

    wrapper.set_embed_and_head(None, target_head)

    reported_embed, reported_head = wrapper.get_embed_and_head()
    assert reported_embed is inner.model.embed_tokens.weight
    assert reported_head is target_head


@pytest.mark.parametrize("has_pp", [False, True])
def test_pipeline_checkpoint_must_ship_the_embedding_shard(has_pp) -> None:
    # Off the pipeline the target embedding replaces the shard, so a
    # checkpoint without one loads; the pipeline's last stage drafts with it.
    draft = _draft(has_pp=has_pp)
    weights = [(f"{NEXTN_PREFIX}shared_head.head.weight", torch.ones(6, 4))]
    with mock.patch.object(draft, "post_load_weights") as post_load:
        if has_pp:
            with pytest.raises(ValueError, match="embed_tokens.weight"):
                draft.load_weights(iter(weights))
            post_load.assert_not_called()
        else:
            draft.load_weights(iter(weights))
            post_load.assert_called_once_with()
    torch.testing.assert_close(draft.lm_head.weight.data, torch.ones(6, 4))


def test_pipeline_checkpoint_embedding_shard_lands_in_the_draft() -> None:
    draft = _draft(has_pp=True)
    shard = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    weights = [
        (f"{NEXTN_PREFIX}embed_tokens.weight", shard),
        (f"{NEXTN_PREFIX}shared_head.head.weight", torch.ones(6, 4)),
        ("model.layers.0.ignored.weight", torch.zeros(1)),
    ]
    with mock.patch.object(draft, "post_load_weights") as post_load:
        draft.load_weights(iter(weights))
    post_load.assert_called_once_with()
    torch.testing.assert_close(draft.model.embed_tokens.weight.data, shard)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
