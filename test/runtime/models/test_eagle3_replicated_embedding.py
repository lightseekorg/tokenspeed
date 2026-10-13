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

"""An EAGLE-3 draft under attention TP drafts from a replicated, in-place refreshed target embedding."""

import os
import sys

import pytest
import torch
from transformers import LlamaConfig

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.models import llama_eagle3  # noqa: E402
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB, HIDDEN, TP = 64, 32, 2


def _peer_rows(shard: torch.Tensor) -> torch.Tensor:
    """Rank 0's shard followed by a stand-in for rank 1's."""
    return torch.cat([shard, shard + 100])


def _tp2_draft(monkeypatch: pytest.MonkeyPatch) -> llama_eagle3.LlamaForCausalLMEagle3:
    """Rank 0 of a TP2 draft; without a process group the gathers return a stand-in for rank 1."""
    mapping = Mapping(
        rank=0,
        world_size=TP,
        attn_tp_size=TP,
        attn_dp_size=1,
        dense_tp_size=TP,
        dense_dp_size=1,
        moe_tp_size=TP,
        moe_ep_size=1,
        moe_dp_size=1,
        nprocs_per_node=TP,
        nnodes=1,
    )
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 256)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 256)
    monkeypatch.setattr(
        llama_eagle3, "all_gather", lambda tensor, group, dim: _peer_rows(tensor)
    )
    monkeypatch.setattr(
        llama_eagle3,
        "all_gather_single",
        lambda output, input, group: output.copy_(_peer_rows(input)),
    )
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        draft_vocab_size=VOCAB,
    )
    with torch.device("cuda"):
        return llama_eagle3.LlamaForCausalLMEagle3(config, mapping)


def test_draft_looks_up_the_full_vocab_from_the_gathered_target_shard(monkeypatch):
    draft = _tp2_draft(monkeypatch)
    shard = torch.randn(VOCAB // TP, HIDDEN, device="cuda")
    draft.set_embed_and_head(shard, None)
    ids = torch.tensor([0, 31, 32, 63], device="cuda")

    rows = _peer_rows(shard)
    torch.testing.assert_close(draft.model.embed_tokens(ids), rows[ids], rtol=0, atol=0)


def test_refresh_regathers_the_updated_shard_in_place(monkeypatch):
    draft = _tp2_draft(monkeypatch)
    shard = torch.randn(VOCAB // TP, HIDDEN, device="cuda")
    draft.set_embed_and_head(shard, None)
    address = draft.model.embed_tokens.weight.data_ptr()
    ids = torch.tensor([5, 40], device="cuda")

    shard.mul_(0.5)  # An in-place target weight update.
    draft.refresh_target_embed()

    # Captured CUDA graphs keep reading the same table.
    assert draft.model.embed_tokens.weight.data_ptr() == address
    rows = _peer_rows(shard)
    torch.testing.assert_close(draft.model.embed_tokens(ids), rows[ids], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
