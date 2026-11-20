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

"""The DeepSeek-layout NextN loader the DeepSeek and GLM-MoE-DSA drafts share."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HIDDEN, Q_LORA, KV_A = 256, 128, 128 + 64
# The NextN layer sits right after the two target layers.
PREFIX = "model.layers.2"


@pytest.fixture
def make_draft(monkeypatch):
    from transformers import DeepseekV3Config, GlmMoeDsaConfig

    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
    from tokenspeed.runtime.models.glm_moe_dsa_nextn import GlmMoeDsaForCausalLMNextN
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "none")
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")

    def make(kind: str):
        shape = dict(
            vocab_size=512,
            hidden_size=HIDDEN,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=Q_LORA,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=128,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            max_position_embeddings=256,
        )
        if kind == "deepseek_v3":
            config, cls = DeepseekV3Config(**shape), DeepseekV3ForCausalLMNextN
        else:
            config = GlmMoeDsaConfig(
                **shape, index_topk=8, index_head_dim=128, index_n_heads=4
            )
            cls = GlmMoeDsaForCausalLMNextN
        # Fields the runtime reads that these config classes lack or default otherwise.
        config.topk_method = "noaux_tc"
        config.scoring_func = "sigmoid"
        config.rope_scaling = None
        config.moe_layer_freq = 1
        with torch.device("cuda"):
            return cls(config, Mapping(rank=0, world_size=1), None)

    return make


def _weight(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).cuda()


@pytest.mark.parametrize("kind", ["deepseek_v3", "glm_moe_dsa"])
def test_nextn_layer_loads_onto_the_draft_and_its_decoder(make_draft, kind):
    draft = make_draft(kind)
    norm = draft.model.decoder.input_layernorm.weight.clone()
    eh_proj = _weight(HIDDEN, 2 * HIDDEN, seed=0)
    q_a, kv_a = _weight(Q_LORA, HIDDEN, seed=1), _weight(KV_A, HIDDEN, seed=2)
    draft.load_weights(
        [
            (f"{PREFIX}.self_attn.q_a_proj.weight", q_a),
            (f"{PREFIX}.eh_proj.weight", eh_proj),
            # A target layer's tensor is not the draft's.
            ("model.layers.1.input_layernorm.weight", _weight(HIDDEN, seed=3)),
            (f"{PREFIX}.self_attn.kv_a_proj_with_mqa.weight", kv_a),
        ]
    )
    attn = draft.model.decoder.self_attn
    assert torch.equal(draft.model.eh_proj.weight, eh_proj)
    assert torch.equal(attn.fused_qkv_a_proj_with_mqa.weight, torch.cat([q_a, kv_a]))
    assert torch.equal(draft.model.decoder.input_layernorm.weight, norm)


def test_glm_indexer_weights_fill_the_fused_projection_mid_stream(make_draft):
    draft = make_draft("glm_moe_dsa")
    wk, weights_proj = _weight(128, HIDDEN, seed=4), _weight(4, HIDDEN, seed=5)
    eh_proj = _weight(HIDDEN, 2 * HIDDEN, seed=6)
    draft.load_weights(
        [
            (f"{PREFIX}.self_attn.indexer.wk.weight", wk),
            (f"{PREFIX}.eh_proj.weight", eh_proj),
            (f"{PREFIX}.self_attn.indexer.weights_proj.weight", weights_proj),
        ]
    )
    indexer = draft.model.decoder.self_attn.indexer
    assert torch.equal(indexer.wk_weights_proj.weight, torch.cat([wk, weights_proj]))
    assert torch.equal(draft.model.eh_proj.weight, eh_proj)


def test_glm_pairs_q_a_and_kv_a_across_update_chunks(make_draft):
    from tokenspeed.runtime.models.base.weight_update import weight_update_session

    draft = make_draft("glm_moe_dsa")
    q_a, kv_a = _weight(Q_LORA, HIDDEN, seed=7), _weight(KV_A, HIDDEN, seed=8)
    with weight_update_session([draft], require_complete=False, cpu_group=None):
        draft.load_weights([(f"{PREFIX}.self_attn.q_a_proj.weight", q_a)])
        draft.load_weights([(f"{PREFIX}.self_attn.kv_a_proj_with_mqa.weight", kv_a)])
    fused = draft.model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight
    assert torch.equal(fused, torch.cat([q_a, kv_a]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
