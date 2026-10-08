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

    def test_head_tp_turns_the_prefill_graph_off(self):
        # The prefill graph records extend forwards this layout never runs.
        args = ServerArgs(**DP8, attn_head_tp_size=8, disaggregation_mode="decode")
        assert args.disable_prefill_graph
        plain = ServerArgs(**DP8, disaggregation_mode="decode")
        assert not plain.disable_prefill_graph

    def test_batch_invariant_judges_the_resolved_quantization(self):
        # --quantization alone does not decide: the checkpoint's resolved
        # method and its disable_quant_module do (checked from ModelConfig).
        args = ServerArgs(
            **DP8,
            attn_head_tp_size=8,
            dense_tp_size=8,
            disaggregation_mode="decode",
            tp_batch_invariant="attn+dense",
            quantization="fp8",
        )
        args.validate_tp_batch_invariant_weights(None, ())
        args.validate_tp_batch_invariant_weights("fp8", ("self_attn", "dense_mlp"))
        args.validate_tp_batch_invariant_weights("fp8", ("self_attn", "mlps"))
        with pytest.raises(ValueError, match="unquantized o_proj"):
            args.validate_tp_batch_invariant_weights("fp8", ("dense_mlp",))
        with pytest.raises(ValueError, match="unquantized dense down_proj"):
            args.validate_tp_batch_invariant_weights("fp8", ("self_attn",))
        attn_only = ServerArgs(
            **DP8,
            attn_head_tp_size=8,
            disaggregation_mode="decode",
            tp_batch_invariant="attn",
        )
        attn_only.validate_tp_batch_invariant_weights("fp8", ("self_attn",))
        with pytest.raises(ValueError, match="unquantized o_proj"):
            attn_only.validate_tp_batch_invariant_weights("fp8", ())
        plain = ServerArgs(**DP8)
        plain.validate_tp_batch_invariant_weights("fp8", ())

    def test_head_tp_decode_engine_caps_the_generation_budget(self):
        """The D-role scheduler never retracts a request whose generation
        fits one safe-step window, so the engine, which cannot run a recovery
        prefill under head TP, admits only those."""
        from tokenspeed.runtime.engine.request_handler import RequestHandler
        from tokenspeed.runtime.engine.scheduler_utils import RETRACTION_SAFE_STEPS

        args = ServerArgs(**DP8, attn_head_tp_size=8, disaggregation_mode="decode")
        args.mapping.rank = 0

        def admit(server_args, max_new_tokens):
            handler = RequestHandler.__new__(RequestHandler)
            handler.max_req_len = 65536
            handler.max_new_tokens_budget = (
                RETRACTION_SAFE_STEPS if server_args.mapping.attn.has_head_tp else None
            )
            spec = SimpleNamespace(max_new_tokens=0)
            state = SimpleNamespace(
                sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
                prompt_input_ids=[1, 2, 3],
                finished_reason=None,
            )
            RequestHandler._apply_generation_budget(handler, spec, state)
            return spec, state

        spec, state = admit(args, RETRACTION_SAFE_STEPS)
        assert spec.max_new_tokens == RETRACTION_SAFE_STEPS
        assert state.finished_reason is None
        spec, state = admit(args, RETRACTION_SAFE_STEPS + 1)
        assert state.finished_reason is not None
        assert "attn-head-tp-size" in state.finished_reason.message
        # Undeclared budgets clamp to the context and exceed the window too.
        _, state = admit(args, None)
        assert state.finished_reason is not None
        plain = ServerArgs(**DP8, disaggregation_mode="decode")
        plain.mapping.rank = 0
        _, state = admit(plain, None)
        assert state.finished_reason is None

    def test_lm_head_tp_under_dp_refuses_prompt_logprobs(self):
        """The LM-head TP group exchanges its logits rows once per forward;
        the prompt-logprob chunk loop would run that exchange a per-rank
        number of times, so the engine aborts such requests at admission."""
        from tokenspeed.runtime.engine.request_handler import RequestHandler

        def admit(server_args, wants_input_logprobs):
            handler = RequestHandler.__new__(RequestHandler)
            mapping = server_args.mapping
            handler.supports_input_logprobs = not (
                mapping.attn.has_dp and mapping.lm_head.has_tp
            )
            state = SimpleNamespace(
                wants_input_logprobs=wants_input_logprobs, finished_reason=None
            )
            RequestHandler._refuse_unsupported_input_logprobs(handler, state)
            return state

        sharded = ServerArgs(**DP8, lm_head_tp_size=8)
        sharded.mapping.rank = 0
        assert admit(sharded, False).finished_reason is None
        state = admit(sharded, True)
        assert state.finished_reason is not None
        assert "lm-head-tp-size" in state.finished_reason.message
        replicated = ServerArgs(**DP8)
        replicated.mapping.rank = 0
        assert admit(replicated, True).finished_reason is None

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
            # rl-bitwise folds the MoE routes in slot order, which needs MoE
            # TP 1: the DP8 world runs its experts under EP.
            ep_size=8,
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
            Mapping(rank=1, world_size=4, attn_tp_size=1, attn_dp_size=4)
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
            rank=0,
            world_size=4,
            attn_tp_size=1,
            attn_dp_size=4,
            dense_tp_size=4,
        )
        with pytest.raises(ValueError, match="shared experts"):
            DeepseekV3MLP(16, 32, "silu", dp, None, "m", True, batch_invariant=True)
        with pytest.raises(ValueError, match="unquantized"):
            DeepseekV3MLP(
                16, 32, "silu", dp, object(), "m", False, batch_invariant=True
            )
        single = Mapping(
            rank=0,
            world_size=4,
            attn_tp_size=1,
            attn_dp_size=4,
            dense_tp_size=1,
        )
        with pytest.raises(ValueError, match="dense TP group"):
            DeepseekV3MLP(
                16, 32, "silu", single, None, "m", False, batch_invariant=True
            )
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

    def test_query_sharding_replicates_the_heads_unless_head_tp_is_asked(self):
        """Under QCP the default head group is this rank alone (every head on
        every rank, no exchange); ``--attn-head-tp-size`` equal to the shard
        group shards them. One core layer either way: under head TP it
        declares every head (the exchange delivers them), and the
        replicated-row forwards (the drafter's decode steps) that exchange
        nothing hand it the attention-TP slice -- the DSA core reads the head
        count from the query."""
        replicated = self._attention(
            Mapping(rank=1, world_size=4, attn_tp_size=4, attn_qcp_size=4)
        )
        assert not replicated.has_head_tp and replicated.num_local_heads == 8
        assert replicated.q_b_proj.tp_size == 1 and replicated.o_proj.tp_size == 1
        assert replicated.attn_mqa.tp_q_head_num == 8

        sharded = self._attention(
            Mapping(
                rank=1,
                world_size=4,
                attn_tp_size=4,
                attn_qcp_size=4,
                attn_head_tp_size=4,
            )
        )
        assert sharded.has_head_tp and sharded.num_local_heads == 2
        assert (
            sharded.q_b_proj.tp_group == (0, 1, 2, 3) and sharded.q_b_proj.tp_rank == 1
        )
        assert sharded.kv_b_proj.weight.shape[0] == 2 * (8 + 4)
        assert sharded.o_proj.weight.shape == (16, 2 * 4)
        assert not sharded.o_proj.reduce_results
        assert sharded.attn_mqa.tp_q_head_num == 8
        # The exchange follows the forward: a sharded extend exchanges, a
        # replicated decode step (no shard) does not.
        from tokenspeed.runtime.execution.query_shard import QueryShardPlan

        plan = QueryShardPlan.from_forward(
            total_tokens=6, input_lengths=[4, 2], size=4, rank=1
        )
        extend = SimpleNamespace(query_shard=plan)
        decode = SimpleNamespace(query_shard=None)
        assert sharded.head_tp_exchanges(extend)
        assert not sharded.head_tp_exchanges(decode)
        assert not replicated.head_tp_exchanges(extend)
        # Attention DP exchanges every forward and has no replicated rows.
        dp = self._attention(
            Mapping(
                rank=1,
                world_size=4,
                attn_tp_size=1,
                attn_dp_size=4,
                attn_head_tp_size=4,
            )
        )
        assert dp.head_tp_exchanges(decode)

    def test_an_empty_query_shard_runs_the_core(self):
        """The sparse core joins its group's history gathers, so a rank whose
        shard is empty calls it like every other rank, through both value
        hooks; only an idle attention-DP rank (no rows, no shard) skips the
        core. Nothing is projected for zero rows either way."""
        from tokenspeed.runtime.execution.query_shard import QueryShardPlan

        attn = self._attention(
            Mapping(rank=3, world_size=4, attn_tp_size=4, attn_qcp_size=4)
        )
        calls: list[tuple[str, tuple[int, ...]]] = []

        class Core:
            layer_id = 0

            def __call__(self, Q, **kwargs):
                calls.append(("decode", tuple(Q.shape)))
                return Q.new_empty(0, Q.shape[1] * attn.kv_lora_rank)

        def sparse_prefill(*, q, layer, **kwargs):
            calls.append(("sparse", tuple(q.shape)))
            return q.new_empty(0, q.shape[1] * attn.kv_lora_rank)

        del attn.attn_mqa
        attn.attn_mqa = Core()
        # Rank 3 holds no rows of the 3-row span.
        plan = QueryShardPlan.from_forward(
            total_tokens=3, input_lengths=[3], size=4, rank=3
        )
        assert plan.local_rows == 0
        ctx = SimpleNamespace(
            query_shard=plan,
            num_extends=1,
            attn_backend=SimpleNamespace(
                forward_sparse_prefill=sparse_prefill,
                supports_mla_projected_value_decode=False,
            ),
            token_to_kv_pool=None,
        )
        Q = torch.zeros(0, 8, attn.kv_lora_rank + 4)
        output = torch.full((0, 8 * 4), 1.0)
        assert attn.forward_absorb_attn_v_proj(Q, ctx, output) is output
        assert (
            attn.sparse_prefill_attn_v_proj(
                Q,
                ctx,
                output,
                kv_seq_lens=None,
                topk_slots=torch.empty(0, 0, dtype=torch.int32),
                topk_lens=torch.empty(0, dtype=torch.int32),
                max_seq_len=0,
            )
            is output
        )
        assert calls == [("decode", (0, 8, 12)), ("sparse", (0, 8, 12))]
        # An idle rank of a head group over attention-DP ranks (no shard)
        # skips the dense core; its exchange legs are the collectives.
        idle = SimpleNamespace(**{**vars(ctx), "query_shard": None, "num_extends": 0})
        plain = self._attention(
            Mapping(rank=3, world_size=4, attn_tp_size=1, attn_dp_size=4)
        )
        del plain.attn_mqa
        plain.attn_mqa = Core()
        calls.clear()
        assert plain.forward_absorb_attn_v_proj(Q, idle, output) is output
        assert calls == []

    def test_the_expanded_prefill_keeps_refusing_under_head_tp(self):
        """``forward`` (the dense MLA path with the expanded prologue) refuses
        extend rows on every head-sharded layout: a query shard outright, and
        a whole-span extend because the head-sharded kv_b_proj cannot expand
        every head's K/V."""
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.execution.query_shard import QueryShardPlan

        def ctx(**overrides):
            fields = dict(
                num_extends=1,
                bs=1,
                input_num_tokens=3,
                forward_mode=ForwardMode.EXTEND,
                query_shard=None,
            )
            fields.update(overrides)
            return SimpleNamespace(**fields)

        hidden = torch.zeros(3, 16)
        positions = torch.arange(3)
        for mapping in (
            Mapping(
                rank=0,
                world_size=4,
                attn_tp_size=4,
                attn_qcp_size=4,
                attn_head_tp_size=4,
            ),
            Mapping(
                rank=0,
                world_size=4,
                attn_tp_size=1,
                attn_dp_size=4,
                attn_head_tp_size=4,
            ),
        ):
            attn = self._attention(mapping)
            with pytest.raises(RuntimeError, match="expanded prefill"):
                attn(positions, hidden, ctx(), comm_manager=None)
        sharded = self._attention(
            Mapping(
                rank=0,
                world_size=4,
                attn_tp_size=4,
                attn_qcp_size=4,
                attn_head_tp_size=4,
            )
        )
        plan = QueryShardPlan.from_forward(
            total_tokens=3, input_lengths=[3], size=4, rank=0
        )
        with pytest.raises(RuntimeError, match="cannot take a query shard"):
            sharded(positions, hidden, ctx(query_shard=plan), comm_manager=None)

    def test_heads_must_split_over_the_head_group(self):
        with pytest.raises(ValueError, match="divisible by the head TP size"):
            self._attention(
                Mapping(
                    rank=0,
                    world_size=16,
                    attn_tp_size=1,
                    attn_dp_size=16,
                    attn_head_tp_size=16,
                )
            )

    def test_batch_invariant_o_proj_needs_head_tp(self, monkeypatch):
        monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "attn")
        with pytest.raises(ValueError, match="attn-head-tp-size"):
            self._attention(
                Mapping(rank=0, world_size=4, attn_tp_size=1, attn_dp_size=4)
            )
        attn = self._attention(
            Mapping(
                rank=0,
                world_size=4,
                attn_tp_size=1,
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
            attn_dp_size=4,
            attn_head_tp_size=4,
        )
        with pytest.raises(NotImplementedError, match="head-TP"):
            self._attention(head_tp, Overriding)
        assert self._attention(head_tp, OptedIn).has_head_tp
        # Without head TP an overriding subclass is untouched.
        plain = Mapping(rank=0, world_size=4, attn_tp_size=1, attn_dp_size=4)
        assert not self._attention(plain, Overriding).has_head_tp
