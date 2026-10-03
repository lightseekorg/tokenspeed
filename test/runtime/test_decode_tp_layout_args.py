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

"""Server-side selection of the decode TP layouts: argument constraints, the
LM-head resolution and the module-level refusals (no distributed runtime)."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import ServerArgs

DP8 = dict(model="x", world_size=8, attn_tp_size=1, data_parallel_size=8)


class TestServerArgs:
    def test_defaults_keep_todays_layout(self):
        args = ServerArgs(**DP8)
        args.mapping.rank = 5
        assert not args.mapping.attn.has_head_tp
        assert args.mapping.attn.head_tp_group == (5,)
        assert not args.mapping.lm_head.has_tp
        assert args.tp_batch_invariant == "none"
        tp8 = ServerArgs(model="x", world_size=8, attn_tp_size=8)
        tp8.mapping.rank = 5
        assert tp8.mapping.lm_head.tp_group == tp8.mapping.attn.tp_group

    def test_decode_preset_resolves_one_node_local_group(self):
        args = ServerArgs(
            **DP8,
            attn_head_tp_size=8,
            lm_head_tp_size=8,
            dense_tp_size=8,
            tp_batch_invariant="attn+dense",
            disaggregation_mode="decode",
        )
        args.mapping.rank = 3
        group = tuple(range(8))
        assert args.mapping.attn.has_head_tp
        assert args.mapping.attn.head_tp_group == group
        assert args.mapping.lm_head.tp_group == group
        assert args.mapping.dense.tp_group == group
        assert global_server_args_dict["tp_batch_invariant"] == "none"

    def test_head_tp_is_decode_only(self):
        with pytest.raises(ValueError, match="disaggregation-mode decode"):
            ServerArgs(**DP8, attn_head_tp_size=8)
        with pytest.raises(ValueError, match="disaggregation-mode decode"):
            ServerArgs(**DP8, attn_head_tp_size=8, disaggregation_mode="prefill")

    def test_head_tp_needs_attention_tp_1(self):
        with pytest.raises(ValueError, match="attention TP 1"):
            ServerArgs(
                model="x",
                world_size=8,
                attn_tp_size=2,
                data_parallel_size=4,
                attn_head_tp_size=8,
                disaggregation_mode="decode",
            )

    def test_batch_invariant_attn_needs_head_tp(self):
        with pytest.raises(ValueError, match="attn-head-tp-size"):
            ServerArgs(**DP8, tp_batch_invariant="attn")

    def test_batch_invariant_dense_needs_dense_tp(self):
        with pytest.raises(ValueError, match="dense-tp-size"):
            ServerArgs(
                **DP8,
                attn_head_tp_size=8,
                disaggregation_mode="decode",
                tp_batch_invariant="attn+dense",
            )

    def test_batch_invariant_refuses_quantized_weights(self):
        with pytest.raises(ValueError, match="unquantized"):
            ServerArgs(
                **DP8,
                attn_head_tp_size=8,
                disaggregation_mode="decode",
                tp_batch_invariant="attn",
                quantization="fp8",
            )

    def test_batch_invariant_rejects_unknown_selection(self):
        with pytest.raises(ValueError, match="tp-batch-invariant"):
            ServerArgs(
                **DP8,
                attn_head_tp_size=8,
                disaggregation_mode="decode",
                tp_batch_invariant="dense",
            )

    def test_lm_head_tp_excludes_dp_sampling(self):
        with pytest.raises(ValueError, match="dp-sampling"):
            ServerArgs(**DP8, lm_head_tp_size=8, dp_sampling=True)
        args = ServerArgs(**DP8, lm_head_tp_size=8)
        args.mapping.rank = 0
        assert args.mapping.lm_head.has_tp

    def test_rl_bitwise_keeps_the_layouts_explicit(self):
        """The envelope neither selects nor refuses a layout."""
        args = ServerArgs(
            **DP8,
            numerics="rl-bitwise",
            attn_head_tp_size=8,
            disaggregation_mode="decode",
        )
        assert args.tp_batch_invariant == "none"
        assert args.batch_invariant_collectives


class _StubModel(BaseCausalLM):
    model_cls = None

    def resolve_model(self, config, mapping, quant_config, prefix):
        return SimpleNamespace(embed_tokens=object())


def _causal_lm(mapping: Mapping) -> _StubModel:
    config = SimpleNamespace(
        hidden_size=16, vocab_size=64, tie_word_embeddings=False, model_type="t"
    )
    return _StubModel(config, mapping)


class TestLmHeadResolution:
    def test_dp_default_is_replicated(self):
        model = _causal_lm(
            Mapping(rank=1, world_size=4, attn_tp_size=1, attn_cp_size=1, attn_dp_size=4)
        )
        assert isinstance(model.lm_head, ReplicatedLinear)
        assert model.logits_processor.skip_all_gather
        assert not model.logits_processor.dp_lm_head_tp
        assert model.logits_processor.tp_size == 1

    def test_dp_with_lm_head_tp_shards_over_the_group(self):
        model = _causal_lm(
            Mapping(
                rank=1,
                world_size=4,
                attn_tp_size=1,
                attn_cp_size=1,
                attn_dp_size=4,
                lm_head_tp_size=4,
            )
        )
        assert isinstance(model.lm_head, ParallelLMHead)
        assert model.lm_head.tp_group == (0, 1, 2, 3)
        assert model.lm_head.tp_rank == 1
        assert model.lm_head.weight.shape[0] == 64 // 4
        processor = model.logits_processor
        assert processor.dp_lm_head_tp and processor.skip_all_gather
        assert processor.tp_group == (0, 1, 2, 3) and processor.tp_size == 4

    def test_attention_tp_is_unchanged(self):
        model = _causal_lm(Mapping(rank=2, world_size=4, attn_tp_size=4))
        assert isinstance(model.lm_head, ParallelLMHead)
        assert model.lm_head.tp_group == (0, 1, 2, 3)
        assert not model.logits_processor.skip_all_gather
        assert not model.logits_processor.dp_lm_head_tp

    def test_processor_flag_needs_a_sharded_head_under_skip_all_gather(self):
        config = SimpleNamespace(model_type="t", vocab_size=8)
        with pytest.raises(ValueError, match="dp_lm_head_tp"):
            LogitsProcessor(config, skip_all_gather=True, dp_lm_head_tp=True)
        with pytest.raises(ValueError, match="dp_lm_head_tp"):
            LogitsProcessor(
                config,
                skip_all_gather=False,
                tp_rank=0,
                tp_size=2,
                tp_group=(0, 1),
                dp_lm_head_tp=True,
            )


class TestModuleRefusals:
    @pytest.fixture(autouse=True)
    def _plain_layout(self, monkeypatch):
        monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "none")

    def test_dense_batch_invariant_constraints(self):
        from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP

        dp = Mapping(
            rank=0, world_size=4, attn_tp_size=1, attn_cp_size=1, attn_dp_size=4, dense_tp_size=4
        )
        with pytest.raises(ValueError, match="shared experts"):
            DeepseekV3MLP(16, 32, "silu", dp, None, "m", True, batch_invariant=True)
        with pytest.raises(ValueError, match="unquantized"):
            DeepseekV3MLP(16, 32, "silu", dp, object(), "m", False, batch_invariant=True)
        single = Mapping(
            rank=0, world_size=4, attn_tp_size=1, attn_cp_size=1, attn_dp_size=4, dense_tp_size=1
        )
        with pytest.raises(ValueError, match="dense TP group"):
            DeepseekV3MLP(16, 32, "silu", single, None, "m", False, batch_invariant=True)
        mlp = DeepseekV3MLP(16, 32, "silu", dp, None, "m", False, batch_invariant=True)
        assert mlp.down_proj.weight.shape == (16 // 4, 32)
        assert mlp(torch.empty(0, 16)).shape == (0, 16 // 4)

    def _attention(self, mapping, cls=None):
        from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

        cls = cls or DeepseekV3AttentionMLA
        return cls(
            config=SimpleNamespace(rms_norm_eps=1e-6),
            mapping=mapping,
            hidden_size=16,
            num_heads=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=8,
            kv_lora_rank=8,
            quant_config=None,
            layer_id=0,
            reduce_attn_results=False,
        )

    def test_head_tp_shards_the_head_projections(self):
        attn = self._attention(
            Mapping(
                rank=1,
                world_size=4,
                attn_tp_size=1,
                attn_cp_size=1,
                attn_dp_size=4,
                attn_head_tp_size=4,
            )
        )
        assert attn.has_head_tp and attn.num_local_heads == 2
        assert attn.q_b_proj.tp_group == (0, 1, 2, 3) and attn.q_b_proj.tp_rank == 1
        assert attn.kv_b_proj.weight.shape[0] == 2 * (8 + 4)
        assert attn.o_proj.weight.shape == (16, 2 * 4)
        assert not attn.o_proj.reduce_results
        assert attn.attn_mqa.tp_q_head_num == 8
        assert attn.attn_mha.tp_q_head_num == 2

    def test_heads_must_split_over_the_head_group(self):
        with pytest.raises(ValueError, match="divisible by the head TP size"):
            self._attention(
                Mapping(
                    rank=0,
                    world_size=16,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    attn_dp_size=16,
                    attn_head_tp_size=16,
                )
            )

    def test_batch_invariant_o_proj_needs_head_tp(self, monkeypatch):
        monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "attn")
        with pytest.raises(ValueError, match="attn-head-tp-size"):
            self._attention(
                Mapping(rank=0, world_size=4, attn_tp_size=1, attn_cp_size=1, attn_dp_size=4)
            )
        attn = self._attention(
            Mapping(
                rank=0,
                world_size=4,
                attn_tp_size=1,
                attn_cp_size=1,
                attn_dp_size=4,
                attn_head_tp_size=4,
            )
        )
        assert type(attn.o_proj).__name__ == "ColumnParallelLinear"
        assert attn.o_proj.weight.shape == (16 // 4, 8 * 4)

    def test_forward_override_must_opt_in(self):
        from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

        class Overriding(DeepseekV3AttentionMLA):
            def forward(self, *args, **kwargs):
                raise AssertionError

        class OptedIn(Overriding):
            supports_head_tp = True

        head_tp = Mapping(
            rank=0,
            world_size=4,
            attn_tp_size=1,
            attn_cp_size=1,
            attn_dp_size=4,
            attn_head_tp_size=4,
        )
        with pytest.raises(NotImplementedError, match="head-TP"):
            self._attention(head_tp, Overriding)
        assert self._attention(head_tp, OptedIn).has_head_tp
        # Without head TP an overriding subclass is untouched.
        plain = Mapping(rank=0, world_size=4, attn_tp_size=1, attn_cp_size=1, attn_dp_size=4)
        assert not self._attention(plain, Overriding).has_head_tp
