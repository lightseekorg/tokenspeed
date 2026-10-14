import argparse
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import MethodType, SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsv4.cuda import (
    has_indexer_topk_prefill,
    indexer_topk_prefill,
)
from tokenspeed_kernel.ops.attention.dsv4.triton import (
    dsv4_compute_global_topk_indices_and_lens,
)
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda import (
    hash_softplus_sqrt_topk_flash,
    softplus_sqrt_topk_flash,
)

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    _derive_num_attention_layers,
    configure_deepseek_v4_attention,
    is_deepseek_v4,
    is_deepseek_v4_nextn,
)
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (
    DeepseekV4DSpark,
    _dspark_decode_position_plan,
    _dspark_prefill_position_plan,
)
from tokenspeed.runtime.execution.drafter.eagle import (
    _advance_draft_forward_metadata_if_supported,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import (
    ForwardStepRunner,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_KV_GROUP_ID,
    deepseek_v4_cache_layout_from_config,
    deepseek_v4_indexer_fp8_row_bytes,
    deepseek_v4_indexer_mxfp4_row_bytes,
    deepseek_v4_nope_dim,
    deepseek_v4_swa_row_bytes,
    deepseek_v4_swa_token_stride,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import (
    derive_history_groups_by_layer,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    HybridDeepseekV4TokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
    v4_compressed_kv_spec,
    v4_compressor_state_spec,
    v4_indexer_kv_spec,
    v4_indexer_state_spec,
    v4_swa_kv_spec,
)
from tokenspeed.runtime.layers.attention.page_table import (
    group_slot_mapping_from_raw as _group_slot_mapping_from_raw,
)
from tokenspeed.runtime.layers.attention.page_table import (
    mask_invalid_graph_tokens as _mask_invalid_graph_tokens,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.moe import topk as topk_module
from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
from tokenspeed.runtime.layers.quantization import (
    QUANTIZATION_METHODS,
    Fp8Config,
    Mxfp4Config,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    initialize_dummy_integer_weights,
)
from tokenspeed.runtime.models import deepseek_v41_moe as deepseek_v4_model
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.attention import (
    _dspark_output_projection,
    _normalize_query_per_head,
    _quantize_dspark_non_rope,
    _rmsnorm,
    dspark_fp8_quant_dequant,
    get_dspark_topk_idxs_batched,
)
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads import (
    dspark_greedy_workspace,
    sample_dspark_block_greedy,
)
from tokenspeed.runtime.models.deepseek_v41_moe import (
    DeepseekV4MLP,
    DeepseekV4MoE,
    DeepseekV4MoEGate,
    DeepseekV4TopK,
    _deepseek_v4_mega_moe_max_num_tokens,
    _deepseek_v4_routed_expert_quant_config,
)
from tokenspeed.runtime.pd.cache_protocol import build_cache_fields_by_producer_step
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import (
    global_server_args_dict,
    global_server_args_dict_update,
)
from tokenspeed.runtime.utils.hf_transformers_utils import (
    _CONFIG_REGISTRY,
    _wrap_deepseek_v4_tokenizer,
    get_tokenizer,
    prefers_deepseek_v4_tokenizer,
)
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.spec_block_geometry import count_dspark_stages


def _v4_spec_set(hf_config, *, layer_ratio):
    """The spec set a ratio vector declares, in the recipe's own order."""
    ratios = {int(ratio) for ratio in layer_ratio}
    specs = [v4_swa_kv_spec(hf_config)]
    for ratio in sorted(r for r in ratios if r > 1):
        specs.append(v4_compressor_state_spec(ratio))
        specs.append(v4_compressed_kv_spec(ratio))
    if 4 in ratios:
        specs.append(v4_indexer_kv_spec())
        specs.append(v4_indexer_state_spec())
    return tuple(specs)


def _v4_recipe(
    hf_config, *, prefix_granularity: int = 256, decode_input_tokens: int = 1
):
    """A V4 recipe over one hf_config, with tiny scheduler limits."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
        DeepseekV4Recipe,
    )

    num_layers = len(hf_config.compress_ratios)
    return DeepseekV4Recipe(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            max_total_tokens=None,
            chunked_prefill_size=prefix_granularity,
            attention_use_fp4_indexer_cache=True,
            speculative_algorithm=None,
            disaggregation_mode="null",
            enable_prefix_caching=True,
        ),
        model_config=SimpleNamespace(
            hf_config=hf_config, num_attention_layers=num_layers
        ),
        attn_config=SimpleNamespace(
            prefix_granularity=prefix_granularity,
            max_bs=1,
            context_len=4096,
            pd_disaggregation_enabled=False,
            kvp_size=1,
        ),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 34,
        probe_batch_rows=None,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=0,
    )


def _v4_layout(hf_config, **kwargs):
    """``(recipe, groups, packed layout)`` -- group then pack, as setup does."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

    recipe = _v4_recipe(hf_config, **kwargs)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    recipe.check_layout(layout)
    return recipe, groups, layout


def _fake_pool(*group_specs, **arena_attrs) -> SimpleNamespace:
    """A cache-view double: the arena publishes, the view just names it."""
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=tuple(group_specs), **arena_attrs)
    )


def _make_planned_deepseek_v4_pool(
    layout,
    hf_config,
    *,
    num_lcm_blocks: int = 2,
):
    _, groups, packed = _v4_layout(hf_config)
    plan = packed.bind(num_lcm_blocks)
    max_packing = max(group.cache_blocks_per_lcm_block for group in plan.groups)
    pool_size = num_lcm_blocks * max_packing * plan.prefix_granularity
    specs = tuple(spec for spec, _ in groups)
    from cache_pool_test_utils import make_pool

    _, pool = make_pool(
        HybridDeepseekV4TokenToKVPool,
        plan,
        device="cpu",
        layout=layout,
        layer_num=len(layout.layer_ratio),
        rank=0,
        cache_group_specs=specs,
        token_capacity=pool_size,
    )
    return pool, plan


def _mhc_sinkhorn_reference(
    mixes: torch.Tensor, iters: int, eps: float
) -> torch.Tensor:
    mixes = torch.softmax(mixes, dim=-1) + eps
    mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        mixes = mixes / (mixes.sum(dim=-1, keepdim=True) + eps)
        mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    return mixes


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, _ = residual.shape
    x = residual.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_base, post_base, comb_base = torch.split(
        hc_base.float(), [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0].float() + pre_base) + hc_eps
    post = (torch.sigmoid(post_raw * hc_scale[1].float() + post_base) * 2.0).unsqueeze(
        -1
    )
    comb = _mhc_sinkhorn_reference(
        comb_raw.reshape(num_tokens, hc_mult, hc_mult) * hc_scale[2].float()
        + comb_base.reshape(1, hc_mult, hc_mult),
        sinkhorn_iters,
        hc_eps,
    )
    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input.to(residual.dtype), post, comb


def _mhc_post_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if post.dim() == 2:
        post = post.unsqueeze(-1)
    mixed_residual = torch.einsum("tnm,tnh->tmh", comb.float(), residual.float())
    block_update = post.float() * hidden_states.float().unsqueeze(1)
    return (mixed_residual + block_update).to(hidden_states.dtype)


class TestDeepseekV4Config(unittest.TestCase):
    quant_config: ClassVar = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
    }

    def test_config_registry(self):
        self.assertEqual(DeepseekV4Config.model_type, "deepseek_v4")
        self.assertIs(_CONFIG_REGISTRY["deepseek_v4"], DeepseekV4Config)

    def test_fp4_expert_contract_overrides_model_wide_fp8_config(self):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            scale_fmt="ue8m0",
        )

        routed_quant_config, is_block_fp8 = _deepseek_v4_routed_expert_quant_config(
            SimpleNamespace(expert_dtype="fp4"), quant_config
        )

        self.assertFalse(is_block_fp8)
        self.assertIsInstance(routed_quant_config, Mxfp4Config)
        self.assertTrue(routed_quant_config.is_checkpoint_mxfp4_serialized)

    def test_fp8_expert_contract_keeps_model_wide_fp8_config(self):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            scale_fmt="ue8m0",
        )

        for expert_dtype in (None, "fp8"):
            with self.subTest(expert_dtype=expert_dtype):
                routed_quant_config, is_block_fp8 = (
                    _deepseek_v4_routed_expert_quant_config(
                        SimpleNamespace(expert_dtype=expert_dtype), quant_config
                    )
                )

                self.assertTrue(is_block_fp8)
                self.assertIs(routed_quant_config, quant_config)

    def test_forward_mode_mixed_predicate(self):
        self.assertTrue(ForwardMode.MIXED.is_mixed())
        self.assertFalse(ForwardMode.EXTEND.is_mixed())
        self.assertFalse(ForwardMode.DECODE.is_mixed())
        self.assertTrue(ForwardMode.EXTEND.is_extend_or_mixed())
        self.assertTrue(ForwardMode.MIXED.is_extend_or_mixed())
        self.assertFalse(ForwardMode.DECODE.is_extend_or_mixed())
        self.assertTrue(ForwardMode.DECODE.is_decode_or_idle())
        self.assertTrue(ForwardMode.IDLE.is_decode_or_idle())
        self.assertFalse(ForwardMode.EXTEND.is_decode_or_idle())
        self.assertEqual(ForwardMode.from_num_extends(0, 0), ForwardMode.IDLE)
        self.assertEqual(ForwardMode.from_num_extends(0, 2), ForwardMode.DECODE)
        self.assertEqual(ForwardMode.from_num_extends(2, 2), ForwardMode.EXTEND)
        self.assertEqual(ForwardMode.from_num_extends(1, 2), ForwardMode.MIXED)

    def test_model_runner_forwards_supported_spec_step_idx(self):
        class ModelWithSpecStep:
            def __init__(self):
                self.received_spec_step_idx = None

            def forward(
                self,
                ctx,
                input_ids,
                positions,
                spec_step_idx=0,
            ):
                self.received_spec_step_idx = spec_step_idx
                return spec_step_idx

        runner = object.__new__(ModelRunner)
        runner.model = ModelWithSpecStep()
        runner.is_generation = True
        runner._model_forward_accepts_spec_step_idx = (
            ModelRunner._forward_accepts_kwarg(runner.model, "spec_step_idx")
        )
        self.assertTrue(runner.forward_accepts_spec_step_idx)

        empty = torch.empty(0, dtype=torch.int32)
        result = runner.forward(
            ctx=None,
            input_ids=empty,
            positions=empty,
            spec_step_idx=2,
        )

        self.assertEqual(result, 2)
        self.assertEqual(runner.model.received_spec_step_idx, 2)

    def test_model_runner_omits_unsupported_spec_step_idx(self):
        class ModelWithoutSpecStep:
            def forward(
                self,
                ctx,
                input_ids,
                positions,
            ):
                return "ok"

        runner = object.__new__(ModelRunner)
        runner.model = ModelWithoutSpecStep()
        runner.is_generation = True
        runner._model_forward_accepts_spec_step_idx = (
            ModelRunner._forward_accepts_kwarg(runner.model, "spec_step_idx")
        )
        self.assertFalse(runner.forward_accepts_spec_step_idx)

        empty = torch.empty(0, dtype=torch.int32)
        result = runner.forward(
            ctx=None,
            input_ids=empty,
            positions=empty,
            spec_step_idx=2,
        )

        self.assertEqual(result, "ok")

    def test_model_runner_does_not_forward_spec_step_idx_to_var_kwargs(self):
        class ModelWithKwargs:
            def __init__(self):
                self.received_kwargs = None

            def forward(
                self,
                ctx,
                input_ids,
                positions,
                **kwargs,
            ):
                self.received_kwargs = kwargs
                return "ok"

        runner = object.__new__(ModelRunner)
        runner.model = ModelWithKwargs()
        runner.is_generation = True
        runner._model_forward_accepts_spec_step_idx = (
            ModelRunner._forward_accepts_kwarg(runner.model, "spec_step_idx")
        )
        self.assertFalse(runner.forward_accepts_spec_step_idx)

        empty = torch.empty(0, dtype=torch.int32)
        result = runner.forward(
            ctx=None,
            input_ids=empty,
            positions=empty,
            spec_step_idx=2,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(runner.model.received_kwargs, {})

    def test_spec_helpers_preserve_non_v4_backend_contracts(self):
        seq_lens = object()
        calls = []

        class V4LikeBackend:
            def advance_draft_forward_metadata(self, actual_seq_lens):
                calls.append(actual_seq_lens)

        _advance_draft_forward_metadata_if_supported(V4LikeBackend(), seq_lens)
        _advance_draft_forward_metadata_if_supported(SimpleNamespace(), seq_lens)
        self.assertEqual(calls, [seq_lens])

    def _bind_deepseek_v4_moe_methods(self, moe):
        for name in (
            "_renormalize_routing_weights",
            "_compute_topk_output",
            "_forward_routed_experts",
            "_forward_shared_experts",
            "forward_mega_moe",
            "forward_normal",
        ):
            setattr(moe, name, MethodType(getattr(DeepseekV4MoE, name), moe))
        return moe

    def _make_fake_deepseek_v4_moe(
        self,
        hidden_states,
        input_ids,
        stream_fork,
        calls,
        *,
        norm_topk_prob,
    ):
        def routing_inputs(states, ids):
            calls.append("routing_inputs")
            self.assertIs(states, hidden_states)
            self.assertIs(ids, input_ids)
            return torch.zeros(states.shape[0], 4), None, None, ids

        def topk(states, router_logits, **kwargs):
            calls.append("topk")
            self.assertIs(states, hidden_states)
            self.assertEqual(tuple(router_logits.shape), (states.shape[0], 4))
            return SimpleNamespace(scale=2.0)

        def routed_experts(**kwargs):
            calls.append("routed")
            self.assertIs(kwargs["hidden_states"], hidden_states)
            self.assertIsNotNone(kwargs["topk_output"])
            return (hidden_states + 1) * kwargs["topk_output"].scale

        def shared_experts(states):
            calls.append("shared")
            self.assertIs(states, hidden_states)
            return hidden_states + 3

        moe = SimpleNamespace(
            use_mega_moe=False,
            n_shared_experts=1,
            shared_experts=shared_experts,
            stream_fork=stream_fork,
            routed_scaling_factor=2.0,
            config=SimpleNamespace(norm_topk_prob=norm_topk_prob),
            experts=routed_experts,
            topk=topk,
            _routing_inputs=routing_inputs,
        )
        return self._bind_deepseek_v4_moe_methods(moe)

    def test_deepseek_v4_topk_builds_precomputed_output(self):
        hidden_states = torch.ones((2, 3))
        router_logits = torch.zeros((2, 4))
        correction_bias = torch.zeros(4)
        topk_weights = torch.tensor([[1.4, 0.6], [1.1, 0.9]])
        topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
        calls = []

        def fake_moe_topk(*args, **kwargs):
            calls.append((args, kwargs))
            return topk_weights, topk_ids

        with patch.object(deepseek_v4_model, "moe_topk", fake_moe_topk):
            output = DeepseekV4TopK(
                top_k=2,
                renormalize=True,
                correction_bias=correction_bias,
                routed_scaling_factor=2.0,
                hash_routing=False,
            )(hidden_states, router_logits)

        self.assertIs(output.topk_weights, topk_weights)
        self.assertIs(output.topk_ids, topk_ids)
        self.assertIs(output.router_logits, router_logits)
        self.assertEqual(calls[0][0][2:6], ("sqrt_softplus", "topk", True, 2.0))

    def test_deepseek_v4_topk_routes_simulated_logits(self):
        router_logits = torch.zeros((2, 4))
        seen = []

        def fake_moe_topk(logits, *args, **kwargs):
            seen.append(logits)
            return torch.ones((2, 2)), torch.zeros((2, 2), dtype=torch.int32)

        with (
            patch.dict(os.environ, {"TOKENSPEED_MOE_ROUTING_SIMULATION": "uniform"}),
            patch.dict(topk_module._simulated_logits, clear=True),
            patch.object(deepseek_v4_model, "moe_topk", fake_moe_topk),
        ):
            output = DeepseekV4TopK(
                top_k=2,
                renormalize=True,
                correction_bias=torch.zeros(4),
                routed_scaling_factor=1.0,
                hash_routing=False,
            )(torch.ones((2, 3)), router_logits)
            expected = topk_module.simulated_router_logits(router_logits)

        torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(seen[0], router_logits))
        self.assertIs(output.router_logits, seen[0])

    def test_deepseek_v4_moe_stream_fork_disabled_order(self):
        calls = []
        hidden_states = torch.ones(2, 3)
        input_ids = torch.arange(2)
        moe = self._make_fake_deepseek_v4_moe(
            hidden_states,
            input_ids,
            StreamFork(None),
            calls,
            norm_topk_prob=True,
        )

        actual = DeepseekV4MoE.forward(
            moe,
            hidden_states,
            input_ids,
            num_global_tokens=2,
            max_num_tokens_per_gpu=2,
        )

        self.assertEqual(calls, ["routing_inputs", "topk", "routed", "shared"])
        self.assertTrue(
            torch.equal(actual, (hidden_states + 1) * 2 + hidden_states + 3)
        )

    def test_deepseek_v4_moe_always_builds_topk_output(self):
        hidden_states = torch.ones(2, 3)
        input_ids = torch.arange(2)

        for norm_topk_prob in (False, True):
            with self.subTest(norm_topk_prob=norm_topk_prob):
                calls = []
                moe = self._make_fake_deepseek_v4_moe(
                    hidden_states,
                    input_ids,
                    StreamFork(None),
                    calls,
                    norm_topk_prob=norm_topk_prob,
                )

                actual = DeepseekV4MoE.forward(
                    moe,
                    hidden_states,
                    input_ids,
                    num_global_tokens=2,
                    max_num_tokens_per_gpu=2,
                )

                expected = (hidden_states + 1) * 2 + hidden_states + 3
                self.assertEqual(calls, ["routing_inputs", "topk", "routed", "shared"])
                self.assertTrue(torch.equal(actual, expected))

    def test_deepseek_v4_moe_topk_owns_routed_scaling(self):
        captured = {}
        topk_config = {}

        class FakeExperts:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.topk_output_format = object()

        class FakeTopK:
            def __init__(self, **kwargs):
                topk_config.update(kwargs)

        backend = SimpleNamespace(
            is_mega_moe=lambda: False,
            is_gluon_petit=lambda: False,
            is_flashinfer_trtllm=lambda: True,
        )
        config = SimpleNamespace(
            n_shared_experts=None,
            routed_scaling_factor=2.5,
            scoring_func="sqrtsoftplus",
            n_routed_experts=8,
            num_experts_per_tok=2,
            hidden_size=256,
            moe_intermediate_size=128,
            hidden_act="silu",
            norm_topk_prob=True,
        )
        mapping = SimpleNamespace(
            attn=SimpleNamespace(tp_size=1),
            moe=SimpleNamespace(
                ep_size=1,
                tp_size=1,
                tp_ep_size=1,
                tp_rank=0,
                ep_rank=0,
            ),
        )
        gate = SimpleNamespace(e_score_correction_bias=None, tid2eid=None)

        with (
            patch.object(deepseek_v4_model, "get_moe_backend", return_value=backend),
            patch.object(deepseek_v4_model, "DeepseekV4MoEGate", return_value=gate),
            patch.object(
                deepseek_v4_model,
                "_deepseek_v4_routed_expert_quant_config",
                return_value=(None, False),
            ),
            patch.object(deepseek_v4_model, "MoELayer", FakeExperts),
            patch.object(deepseek_v4_model, "DeepseekV4TopK", FakeTopK),
        ):
            moe = DeepseekV4MoE(config, mapping, None, 0, "model.layers.0.ffn")

        self.assertEqual(moe.routed_scaling_factor, 2.5)
        self.assertEqual(captured["routing_mode"], "precomputed_topk")
        self.assertEqual(captured["routing_config"]["routed_scaling_factor"], 1.0)
        self.assertEqual(topk_config["routed_scaling_factor"], 2.5)

    def test_deepseek_v4_shared_mlp_uses_dense_tp(self):
        mapping = Mapping(
            rank=1,
            world_size=4,
            attn_tp_size=1,
            attn_dp_size=4,
            dense_tp_size=1,
            dense_dp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
            moe_dp_size=1,
        )

        shared_mlp = DeepseekV4MLP(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
            mapping=mapping,
            quant_config=None,
            prefix="model.layers.0.ffn.shared_experts",
        )

        self.assertEqual(shared_mlp.tp_rank, mapping.dense.tp_rank)
        self.assertEqual(shared_mlp.tp_size, mapping.dense.tp_size)
        self.assertEqual(shared_mlp.tp_group, mapping.dense.tp_group)
        self.assertNotEqual(shared_mlp.tp_size, mapping.moe.tp_ep_size)

    def _make_fake_mega_deepseek_v4_moe(
        self, hidden_states, input_ids, shared_experts, calls
    ):
        def routing_inputs(states, ids):
            calls.append("routing_inputs")
            self.assertIs(states, hidden_states)
            self.assertIs(ids, input_ids)
            return torch.zeros(states.shape[0], 4), None, None, ids

        def topk(states, router_logits, **kwargs):
            calls.append("topk")
            self.assertIs(states, hidden_states)
            self.assertEqual(tuple(router_logits.shape), (states.shape[0], 4))
            return object()

        def routed_experts(**kwargs):
            calls.append("routed")
            self.assertIs(kwargs["hidden_states"], hidden_states)
            self.assertIsNotNone(kwargs["topk_output"])
            return hidden_states + 1

        moe = SimpleNamespace(
            use_mega_moe=True,
            config=SimpleNamespace(num_experts_per_tok=2, norm_topk_prob=True),
            n_shared_experts=1,
            shared_experts=shared_experts,
            stream_fork=StreamFork(None),
            routed_scaling_factor=1.0,
            experts=routed_experts,
            topk=topk,
            _routing_inputs=routing_inputs,
        )
        return self._bind_deepseek_v4_moe_methods(moe)

    def test_deepseek_v4_mega_moe_dense_tp_one_skips_shared_rsag(self):
        calls = []
        hidden_states = torch.ones(2, 3)
        input_ids = torch.arange(2)
        test_case = self

        class SharedExperts:
            tp_rank = 0
            tp_size = 1
            tp_group = (0,)

            def __call__(self, states):
                calls.append("shared")
                test_case.assertIs(states, hidden_states)
                return states + 3

        moe = self._make_fake_mega_deepseek_v4_moe(
            hidden_states, input_ids, SharedExperts(), calls
        )
        ctx = object()

        class FakeCommManager:
            def pre_dense_comm(self, states, actual_ctx):
                test_case.assertIs(actual_ctx, ctx)
                return states

            def post_dense_comm(self, states, residual, actual_ctx):
                test_case.assertIs(actual_ctx, ctx)
                return states, residual

        actual = DeepseekV4MoE.forward(
            moe,
            hidden_states,
            input_ids,
            num_global_tokens=2,
            max_num_tokens_per_gpu=2,
            ctx=ctx,
            comm_manager=FakeCommManager(),
        )

        self.assertEqual(calls, ["routing_inputs", "topk", "routed", "shared"])
        self.assertTrue(torch.equal(actual, hidden_states + 1 + hidden_states + 3))

    def test_deepseek_v4_mega_moe_shared_uses_comm_manager(self):
        calls = []
        hidden_states = torch.ones(2, 3)
        input_ids = torch.arange(2)
        ctx = object()
        test_case = self

        class SharedExperts:
            tp_rank = 1
            tp_size = 2
            tp_group = (2, 3)

            def __call__(self, states):
                calls.append("shared")
                test_case.assertTrue(torch.equal(states, hidden_states + 2))
                return states + 3

        moe = self._make_fake_mega_deepseek_v4_moe(
            hidden_states, input_ids, SharedExperts(), calls
        )
        comm_calls = []

        class FakeCommManager:
            def pre_dense_comm(self, states, actual_ctx):
                comm_calls.append(("pre", actual_ctx))
                test_case.assertIs(actual_ctx, ctx)
                test_case.assertIs(states, hidden_states)
                return states + 2

            def post_dense_comm(self, states, residual, actual_ctx):
                comm_calls.append(("post", actual_ctx))
                test_case.assertIsNone(residual)
                test_case.assertIs(actual_ctx, ctx)
                test_case.assertTrue(torch.equal(states, hidden_states + 5))
                return states - 2, residual

        actual = DeepseekV4MoE.forward(
            moe,
            hidden_states,
            input_ids,
            num_global_tokens=2,
            max_num_tokens_per_gpu=2,
            ctx=ctx,
            comm_manager=FakeCommManager(),
        )

        self.assertEqual(calls, ["routing_inputs", "topk", "routed", "shared"])
        self.assertEqual(comm_calls, [("pre", ctx), ("post", ctx)])
        self.assertTrue(torch.equal(actual, hidden_states + 1 + hidden_states + 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_moe_stream_fork_aux_path_matches_serial(self):
        calls = []
        hidden_states = torch.ones(2, 3, device="cuda")
        input_ids = torch.arange(2, device="cuda")
        moe = self._make_fake_deepseek_v4_moe(
            hidden_states,
            input_ids,
            StreamFork(torch.cuda.Stream()),
            calls,
            norm_topk_prob=True,
        )

        with patch.object(deepseek_v4_model, "get_is_capture_mode", return_value=True):
            actual = DeepseekV4MoE.forward(
                moe,
                hidden_states,
                input_ids,
                num_global_tokens=2,
                max_num_tokens_per_gpu=2,
            )
        torch.cuda.synchronize()

        self.assertEqual(calls, ["routing_inputs", "topk", "routed", "shared"])
        self.assertTrue(
            torch.equal(actual, (hidden_states + 1) * 2 + hidden_states + 3)
        )

    def test_cuda_graph_replay_keeps_idle_actual_bs_with_padded_group_tables(self):
        captured = {}

        class FakeBackend:
            draft_block_decode = False
            cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window"})

            def refresh_decode_metadata(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        wrapper = object.__new__(ForwardStepRunner)
        wrapper.attn_backend = FakeBackend()
        wrapper.draft_attn_backend = None
        wrapper.max_tokens_per_req = 1
        wrapper.token_to_kv_pool = _fake_pool()
        wrapper._placeholder_tables = {
            "v4.swa": torch.zeros((4, 1), dtype=torch.int32),
        }

        wrapper._prepare_decode_metadata(
            4,
            0,
            torch.zeros(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            use_graph=True,
        )

        # (padded_bs, actual_bs) are the leading positional args.
        self.assertEqual(captured["args"][0], 4)
        self.assertEqual(captured["args"][1], 0)
        self.assertEqual(
            captured["kwargs"]["block_tables"]["v4.swa"].shape,
            (4, 1),
        )

    def test_cuda_graph_replay_forwards_group_tables_to_draft_backend(self):
        captured = {"target": {}, "draft": {}}

        class FakeBackend:
            draft_block_decode = False

            def __init__(self, key):
                self.key = key

            def refresh_decode_metadata(self, *args, **kwargs):
                captured[self.key]["args"] = args
                captured[self.key]["kwargs"] = kwargs

        wrapper = object.__new__(ForwardStepRunner)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.drafter = SimpleNamespace(
            draft_seq_lens_buf=torch.zeros(4, dtype=torch.int32),
        )
        wrapper.max_tokens_per_req = 4
        wrapper.token_to_kv_pool = _fake_pool()
        wrapper.draft_token_to_kv_pool = _fake_pool()

        table = torch.tensor([[7], [8]], dtype=torch.int32)
        wrapper._prepare_decode_metadata(
            4,
            2,
            torch.zeros(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            use_graph=True,
            block_tables={"v4.swa": table},
        )

        # The draft consumes its own groups out of the SAME delivered dict;
        # row padding is the backend's job now.
        draft_kwargs = captured["draft"]["kwargs"]
        self.assertIs(draft_kwargs["block_tables"]["v4.swa"], table)
        self.assertEqual(draft_kwargs["forward_mode"], ForwardMode.DECODE)
        draft_seq_lens = captured["draft"]["args"][3]
        self.assertEqual(
            draft_seq_lens.data_ptr(),
            wrapper.drafter.draft_seq_lens_buf.data_ptr(),
        )
        self.assertEqual(wrapper.drafter.draft_seq_lens_buf.tolist(), [1, 1, 1, 1])

    def test_cuda_graph_eager_draft_extend_round_runs_init_then_refresh(self):
        captured = {"target": [], "draft": []}

        class FakeBackend:
            draft_block_decode = False
            cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window"})

            def __init__(self, key):
                self.key = key

            def init_forward_metadata(self, *args, **kwargs):
                captured[self.key].append(("init", args, kwargs))

            def refresh_decode_metadata(self, *args, **kwargs):
                captured[self.key].append(("refresh", args, kwargs))

        wrapper = object.__new__(ForwardStepRunner)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            draft_seq_lens_buf=torch.tensor([0, 0], dtype=torch.int32),
        )
        wrapper.token_to_kv_pool = _fake_pool()
        wrapper.draft_token_to_kv_pool = _fake_pool()

        seq_lens = torch.tensor([21, 22], dtype=torch.int32)
        wrapper._init_forward_metadata(
            padded_bs=2,
            num_extends=2,
            req_pool_indices=torch.zeros(2, dtype=torch.int32),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
        )

        # The unified draft contract is two steps: prefill init, then the
        # same decode refresh the pure-decode path uses.
        self.assertEqual(
            [kind for kind, _, _ in captured["draft"]], ["init", "refresh"]
        )
        _, _, init_kwargs = captured["draft"][0]
        self.assertEqual(init_kwargs["forward_mode"], ForwardMode.EXTEND)
        # The init reads the accepted-prefix view, never the drafter's mutable
        # buffer; the buffer is only seeded for the round.
        self.assertEqual(init_kwargs["seq_lens"].data_ptr(), seq_lens.data_ptr())
        self.assertEqual(wrapper.drafter.draft_seq_lens_buf.tolist(), [21, 22])
        # The refresh prepares plain 1-token rows over the drafter buffer.
        _, refresh_args, refresh_kwargs = captured["draft"][1]
        self.assertEqual(refresh_kwargs["forward_mode"], ForwardMode.DECODE)
        self.assertNotIn("num_tokens", refresh_kwargs)
        self.assertEqual(
            refresh_args[3].data_ptr(),
            wrapper.drafter.draft_seq_lens_buf.data_ptr(),
        )

    def test_cuda_graph_eager_v4_draft_uses_only_its_cache_group_subset(self):
        captured = {"target": [], "draft": []}

        class FakeBackend:
            draft_block_decode = False
            cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window"})

            def __init__(self, key):
                self.key = key

            def init_forward_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

            def refresh_decode_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

        swa_table = torch.ones((1, 1), dtype=torch.int32)
        state_table = torch.full((1, 1), 2, dtype=torch.int32)

        wrapper = object.__new__(ForwardStepRunner)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            draft_seq_lens_buf=torch.zeros(1, dtype=torch.int32),
        )
        wrapper.token_to_kv_pool = _fake_pool(
            SimpleNamespace(group_id="v4.swa_kv", attn_kind="full_history"),
            SimpleNamespace(group_id="v4.state", attn_kind="state"),
        )
        wrapper.draft_token_to_kv_pool = _fake_pool(
            SimpleNamespace(group_id="v4.swa_kv", attn_kind="full_history"),
        )

        wrapper._init_forward_metadata(
            padded_bs=1,
            num_extends=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            forward_mode=ForwardMode.EXTEND,
            block_tables={"v4.swa_kv": swa_table, "v4.state": state_table},
            block_tables_cpu={"v4.swa_kv": swa_table, "v4.state": state_table},
        )

        # Two steps per round (init + decode refresh); both receive the same
        # delivered dict — each node consumes its own groups, so there is no
        # per-side table subsetting (and no cache-metadata side channel).
        self.assertEqual(len(captured["draft"]), 2)
        for _, draft_kwargs in captured["draft"]:
            self.assertIs(draft_kwargs["block_tables"]["v4.swa_kv"], swa_table)
            self.assertIs(draft_kwargs["block_tables"]["v4.state"], state_table)

    def test_cuda_graph_eager_draft_decode_aliases_drafter_seq_lens_buffer(self):
        captured = {"target": [], "draft": []}

        class FakeBackend:
            draft_block_decode = False
            cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window"})

            def __init__(self, key):
                self.key = key

            def refresh_decode_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

        wrapper = object.__new__(ForwardStepRunner)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            draft_seq_lens_buf=torch.tensor([11, 12], dtype=torch.int32),
            owns_eager_decode_metadata=False,
        )
        wrapper.token_to_kv_pool = _fake_pool()
        wrapper.draft_token_to_kv_pool = _fake_pool()

        seq_lens = torch.tensor([21, 22], dtype=torch.int32)
        wrapper._prepare_decode_metadata(
            2,
            2,
            torch.zeros(2, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            use_graph=False,
        )

        draft_args, draft_kwargs = captured["draft"][-1]
        # Draft decode metadata aliases the drafter-owned mutable buffer,
        # freshly seeded from the batch seq_lens.
        self.assertEqual(
            draft_args[3].data_ptr(),
            wrapper.drafter.draft_seq_lens_buf.data_ptr(),
        )
        self.assertEqual(wrapper.drafter.draft_seq_lens_buf.tolist(), [21, 22])
        self.assertEqual(draft_kwargs["forward_mode"], ForwardMode.DECODE)

    def test_deepseek_v4_tokenizer_wrapper_uses_model_encoder(self):
        calls = []

        class DummyTokenizer:
            vocab_size = 5

            def __call__(self, text, add_special_tokens=False, **kwargs):
                self.last_call = (text, add_special_tokens, kwargs)
                return {"input_ids": [len(text)]}

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {"<extra>": 5}

        def encode_messages(messages, **kwargs):
            calls.append((messages, kwargs))
            return "<encoded>"

        tokenizer = _wrap_deepseek_v4_tokenizer(DummyTokenizer(), encode_messages)

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            truncation=True,
            max_length=16,
        )

        self.assertEqual(prompt, "<encoded>")
        self.assertEqual(token_ids, [9])
        self.assertEqual(len(tokenizer), 6)
        self.assertEqual(calls[0][1]["thinking_mode"], "thinking")
        self.assertIsNone(calls[0][1]["reasoning_effort"])
        self.assertEqual(calls[1][1]["thinking_mode"], "chat")
        self.assertEqual(
            tokenizer.last_call,
            ("<encoded>", False, {"truncation": True, "max_length": 16}),
        )

    def test_deepseek_v4_tokenizer_is_auto_selected_by_architecture(self):
        self.assertTrue(prefers_deepseek_v4_tokenizer(["DeepseekV4ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(["KimiK2ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(None))

    def test_auto_tokenizer_mode_wraps_deepseek_v4_architecture(self):
        class DummyTokenizer:
            vocab_size = 5

            def __call__(self, text, add_special_tokens=False, **kwargs):
                return {"input_ids": [len(text)]}

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {}

        def encode_messages(messages, **kwargs):
            return "<encoded>"

        with (
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                return_value="/nonexistent/tokenizer-snapshot",
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils._load_deepseek_v4_encode_messages",
                return_value=encode_messages,
            ),
        ):
            tokenizer = get_tokenizer(
                "deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_mode="auto",
                architectures=["DeepseekV4ForCausalLM"],
            )

        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
            ),
            [9],
        )

    def test_deepseek_v4_server_args_cli_flags_round_trip(self):
        # Defaults match dataclass declaration
        self.assertEqual(ServerArgs.deepseek_v4_mega_moe_max_num_tokens, 0)
        self.assertEqual(ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb, 512)
        self.assertEqual(ServerArgs.deepseek_v4_prefill_chunk_size, 4)
        self.assertFalse(hasattr(ServerArgs, "deepseek_v4_prefix_state_policy"))

        # CLI flags parse
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--model=stub",
                    "--deepseek-v4-prefix-state-policy=zero-replay",
                ]
            )
        ns = parser.parse_args(
            [
                "--model=stub",
                "--deepseek-v4-mega-moe-max-num-tokens=128",
                "--deepseek-v4-indexer-prefill-max-logits-mb=256",
                "--deepseek-v4-prefill-chunk-size=8",
            ]
        )
        args = ServerArgs.from_cli_args(ns)
        self.assertEqual(args.deepseek_v4_mega_moe_max_num_tokens, 128)
        self.assertEqual(args.deepseek_v4_indexer_prefill_max_logits_mb, 256)
        self.assertEqual(args.deepseek_v4_prefill_chunk_size, 8)

        # Propagation into global_server_args_dict
        snapshot = dict(global_server_args_dict)
        try:
            global_server_args_dict_update(args)
            self.assertEqual(
                global_server_args_dict["deepseek_v4_mega_moe_max_num_tokens"], 128
            )
            self.assertEqual(
                global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"],
                256,
            )
            self.assertEqual(
                global_server_args_dict["deepseek_v4_prefill_chunk_size"], 8
            )
        finally:
            global_server_args_dict.clear()
            global_server_args_dict.update(snapshot)

    def test_deepseek_v4_mega_moe_max_num_tokens_uses_current_server_args(self):
        snapshot = dict(global_server_args_dict)
        try:
            global_server_args_dict.update(
                {
                    "deepseek_v4_mega_moe_max_num_tokens": 0,
                    "chunked_prefill_size": 16,
                    "prefill_graph_max_tokens": 32,
                    "max_cudagraph_capture_size": 64,
                    "max_num_seqs": 128,
                    "cuda_graph_max_bs": 4096,
                    "cuda_graph_max_tokens": 4096,
                    "max_running_requests": 4096,
                }
            )
            self.assertEqual(_deepseek_v4_mega_moe_max_num_tokens(), 128)

            global_server_args_dict["deepseek_v4_mega_moe_max_num_tokens"] = 256
            self.assertEqual(_deepseek_v4_mega_moe_max_num_tokens(), 256)
        finally:
            global_server_args_dict.clear()
            global_server_args_dict.update(snapshot)

    def test_fp8_quantization_config(self):
        quantization = QUANTIZATION_METHODS["fp8"]

        config = quantization.from_config(self.quant_config)

        self.assertEqual(quantization.get_name(), "fp8")
        self.assertIsNone(
            quantization.override_quantization_method(self.quant_config, None)
        )
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertTrue(config.is_checkpoint_fp8_serialized)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_qkv_rmsnorm_matches_separate(self):
        torch.manual_seed(0)
        q = torch.randn(8, 1536, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        q_norm = RMSNorm(1536, eps=1e-6).cuda().to(torch.bfloat16)
        kv_norm = RMSNorm(512, eps=1e-6).cuda().to(torch.bfloat16)
        fused_norm = FusedRMSNorm(q_norm, kv_norm)

        q_out = torch.empty_like(q)
        kv_out = torch.empty_like(kv)
        try:
            fused_norm(q, kv, output_q_a=q_out, output_kv_a=kv_out)
        except RuntimeError as exc:
            self.skipTest(str(exc))

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(q_out, q_norm(q)))
        self.assertTrue(torch.equal(kv_out, kv_norm(kv)))

    def test_model_config_maps_deepseek_v4_to_standard_fp8(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = None

        model_config._verify_quantization()

        self.assertEqual(model_config.quantization, "fp8")

    def test_model_config_overrides_default_block_size_for_deepseek_v4(self):
        def make_hf_config():
            return SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                model_type="deepseek_v4",
                head_dim=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                rope_scaling=None,
                hidden_size=4096,
                num_attention_heads=8,
                num_key_value_heads=8,
                num_hidden_layers=1,
                vocab_size=32000,
                quantization_config=None,
            )

        def build(prefix_granularity):
            server_args = SimpleNamespace(
                mapping=None,
                prefix_granularity=prefix_granularity,
                speculative_algorithm=None,
                load_format="auto",
                ext_yaml=None,
                validate_tp_batch_invariant_weights=lambda *args: None,
            )
            hf_config = make_hf_config()
            with (
                patch(
                    "tokenspeed.runtime.configs.model_config.get_config",
                    return_value=hf_config,
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_generation_config",
                    return_value=SimpleNamespace(eos_token_id=None),
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_hf_text_config",
                    return_value=hf_config,
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_context_length",
                    return_value=4096,
                ),
                patch.object(ModelConfig, "_verify_quantization"),
            ):
                ModelConfig(
                    "stub",
                    model_override_args="{}",
                    server_args=server_args,
                )
            return server_args

        self.assertEqual(build(64).prefix_granularity, 256)
        self.assertEqual(build(128).prefix_granularity, 128)

    def test_model_config_advertises_v4_dspark_prefix_replay_requirement(self):
        hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLMDSpark"],
            model_type="deepseek_v4",
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            rope_scaling=None,
            hidden_size=4096,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=1,
            vocab_size=32000,
            quantization_config=None,
            dspark_block_size=5,
            dspark_window_size=96,
        )
        server_args = SimpleNamespace(
            mapping=None,
            speculative_algorithm="DSPARK",
            enable_prefix_caching=True,
            speculative_num_steps=5,
            speculative_num_draft_tokens=6,
            prefix_granularity=256,
            load_format="auto",
            ext_yaml=None,
        )
        with (
            patch(
                "tokenspeed.runtime.configs.model_config.get_config",
                return_value=hf_config,
            ),
            patch(
                "tokenspeed.runtime.configs.model_config.get_generation_config",
                return_value=SimpleNamespace(eos_token_id=None),
            ),
            patch(
                "tokenspeed.runtime.configs.model_config.get_hf_text_config",
                return_value=hf_config,
            ),
            patch(
                "tokenspeed.runtime.configs.model_config.count_dspark_stages",
                return_value=2,
            ) as count_stages,
            patch(
                "tokenspeed.runtime.configs.model_config.get_context_length",
                return_value=4096,
            ),
            patch.object(ModelConfig, "_verify_quantization"),
        ):
            model_config = ModelConfig(
                "external-draft",
                model_override_args="{}",
                is_draft_worker=True,
                server_args=server_args,
            )
        self.assertEqual(model_config.dspark_prefix_replay_tokens, 96)
        self.assertEqual(model_config.num_attention_layers, 2)
        count_stages.assert_called_once_with("external-draft", revision=None)

    def test_model_config_keeps_incompatible_user_quantization_error(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = "mxfp4"

        with self.assertRaisesRegex(ValueError, "does not match"):
            model_config._verify_quantization()

    def test_deepseek_v4_flashmla_wrapper_exposes_required_api(self):
        if not current_platform().is_hopper_plus:
            self.skipTest("FlashMLA requires NVIDIA Hopper or newer")

        from tokenspeed_kernel.ops.attention.mla.cuda import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        self.assertTrue(callable(flash_mla_with_kvcache))
        self.assertTrue(callable(flash_mla_sparse_fwd))
        self.assertTrue(callable(get_mla_metadata))

    def test_deepseek_v4_model_config_uses_mla_runtime_metadata(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            rope_scaling=None,
        )

        self.assertTrue(is_deepseek_v4(model_config.hf_config))

        configure_deepseek_v4_attention(model_config, ServerArgs(model="x"))

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 512)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.qk_nope_head_dim, 448)
        self.assertEqual(model_config.v_head_dim, 512)
        self.assertEqual(model_config.index_head_dim, 128)
        self.assertAlmostEqual(model_config.scaling, 512**-0.5)

    def test_deepseek_v4_cache_helpers_match_attention_contract(self):
        head_dim = 512
        rope_dim = 64
        index_head_dim = 128

        self.assertEqual(deepseek_v4_nope_dim(head_dim, rope_dim), 448)
        self.assertEqual(deepseek_v4_swa_token_stride(head_dim, rope_dim), 576)
        self.assertEqual(deepseek_v4_swa_row_bytes(head_dim, rope_dim), 584)
        self.assertEqual(deepseek_v4_indexer_fp8_row_bytes(index_head_dim), 132)
        self.assertEqual(deepseek_v4_indexer_mxfp4_row_bytes(index_head_dim), 68)

    def test_deepseek_v4_nextn_architecture_uses_v4_runtime_metadata(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLMNextN"],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            rope_scaling=None,
        )

        self.assertTrue(is_deepseek_v4(model_config.hf_config))
        self.assertTrue(is_deepseek_v4_nextn(model_config.hf_config))
        self.assertTrue(
            is_deepseek_v4(
                SimpleNamespace(architectures=["DeepseekV4ForCausalLMDSpark"])
            )
        )

        configure_deepseek_v4_attention(model_config, ServerArgs(model="x"))

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 512)
        self.assertEqual(model_config.qk_nope_head_dim, 448)
        self.assertEqual(
            _derive_num_attention_layers(
                SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLMNextN"],
                    num_nextn_predict_layers=1,
                ),
                num_hidden_layers=43,
            ),
            1,
        )
        self.assertFalse(is_deepseek_v4(SimpleNamespace(architectures=None)))
        self.assertFalse(is_deepseek_v4_nextn(SimpleNamespace()))
        self.assertEqual(
            _derive_num_attention_layers(
                SimpleNamespace(architectures=None),
                num_hidden_layers=43,
            ),
            43,
        )

    def test_dspark_stage_count_requires_contiguous_namespaces(self):
        with tempfile.TemporaryDirectory() as model_path:
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "weight_map": {
                            "mtp.0.main_proj.weight": "a.safetensors",
                            "mtp.1.attn.wq_a.weight": "b.safetensors",
                            "mtp.2.norm.weight": "c.safetensors",
                        }
                    },
                    handle,
                )
            self.assertEqual(count_dspark_stages(model_path), 3)

            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "weight_map": {
                            "mtp.0.main_proj.weight": "a.safetensors",
                            "mtp.2.norm.weight": "c.safetensors",
                        }
                    },
                    handle,
                )
            with self.assertRaisesRegex(ValueError, "contiguous from zero"):
                count_dspark_stages(model_path)

    def test_dspark_stage_count_resolves_hub_index(self):
        with tempfile.TemporaryDirectory() as model_path:
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "weight_map": {
                            "mtp.0.main_proj.weight": "a.safetensors",
                            "mtp.1.norm.weight": "b.safetensors",
                        }
                    },
                    handle,
                )
            with patch(
                "tokenspeed.runtime.utils.spec_block_geometry.hf_hub_download",
                return_value=index_path,
            ) as download:
                self.assertEqual(
                    count_dspark_stages("public/model", revision="revision-a"),
                    2,
                )

        download.assert_called_once_with(
            repo_id="public/model",
            filename="model.safetensors.index.json",
            revision="revision-a",
        )

    def test_dspark_stage_count_fails_closed_when_hub_index_is_unavailable(self):
        with patch(
            "tokenspeed.runtime.utils.spec_block_geometry.hf_hub_download",
            side_effect=OSError("offline"),
        ):
            self.assertIsNone(count_dspark_stages("public/model"))

    def test_dspark_graph_safe_context_indices_mask_unwritten_slots(self):
        indices = get_dspark_topk_idxs_batched(
            window_size=4,
            block_size=2,
            start_pos=torch.tensor([0, 2, 8]),
        )

        self.assertEqual(tuple(indices.shape), (3, 2, 6))
        self.assertTrue(torch.equal(indices[0, 0], torch.tensor([0, -1, -1, -1, 4, 5])))
        self.assertTrue(torch.equal(indices[1, 0], torch.tensor([0, 1, 2, -1, 4, 5])))
        self.assertTrue(torch.equal(indices[2, 0], torch.tensor([0, 1, 2, 3, 4, 5])))
        self.assertTrue(torch.equal(indices[:, 0], indices[:, 1]))

    def test_dspark_bonus_selection_matches_verify_layout(self):
        output = torch.tensor(
            [
                9,
                10,
                11,
                12,
                13,
                14,
                20,
                21,
                22,
                23,
                24,
                25,
            ],
            dtype=torch.int32,
        )
        out = torch.empty(2, dtype=torch.int32)

        DeepseekV4DSpark._bonus_tokens_from_output(
            output,
            torch.tensor([1, 4], dtype=torch.int32),
            num_extends=0,
            verify_width=6,
            out=out,
        )

        self.assertTrue(torch.equal(out, torch.tensor([9, 23], dtype=torch.int32)))

    def test_dspark_decode_positions_follow_target_absolute_positions(self):
        (
            interim_positions,
            interim_valid,
            main_positions,
            next_context_lengths,
        ) = _dspark_decode_position_plan(
            old_context_lengths=torch.tensor([128, 128]),
            accepted=torch.tensor([1, 4]),
            block_offsets=torch.arange(5),
        )

        self.assertTrue(
            torch.equal(
                interim_positions,
                torch.tensor(
                    [
                        [128, 129, 130, 131, 132],
                        [128, 129, 130, 131, 132],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                interim_valid,
                torch.tensor(
                    [
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(main_positions, torch.tensor([128, 131])))
        self.assertTrue(torch.equal(next_context_lengths, torch.tensor([129, 132])))

    def test_dspark_prefill_positions_are_not_shifted(self):
        target_positions = torch.tensor([[124, 125, 126, 127]])

        window_positions, next_context_lengths = _dspark_prefill_position_plan(
            target_positions
        )

        self.assertTrue(torch.equal(window_positions, target_positions))
        self.assertTrue(torch.equal(next_context_lengths, torch.tensor([128])))

    @staticmethod
    def _greedy_fixture(vocab, rank, rows, block, tp_size, device):
        """A single vocabulary shard plus the reference argmax over it."""
        torch.manual_seed(0)
        shard = SimpleNamespace(
            num_org_elements=vocab,
            num_added_elements=0,
            org_vocab_start_index=0,
        )
        embedding = SimpleNamespace(
            weight=torch.randn(vocab, rank, device=device).to(torch.bfloat16)
        )
        projection = SimpleNamespace(
            weight=torch.randn(vocab, rank, device=device).to(torch.bfloat16),
            shard_indices=shard,
        )
        markov = SimpleNamespace(embedding=embedding, projection=projection)
        lm_head = SimpleNamespace(shard_indices=shard)
        logits = torch.randn(rows, block, vocab, device=device)
        bonus = torch.randint(0, vocab, (rows,), device=device, dtype=torch.int32)
        expected = torch.empty(rows, block, dtype=torch.int64, device=device)
        previous = bonus.long()
        for step in range(block):
            bias = embedding.weight[previous].float() @ projection.weight.float().T
            previous = torch.argmax(logits[:, step] + bias, dim=-1)
            expected[:, step] = previous
        candidates, partials = dspark_greedy_workspace(tp_size, rows + 3, vocab, device)
        return markov, lm_head, logits, bonus, expected, candidates, partials

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_dspark_single_rank_block_sampler_matches_reference_without_gathers(self):
        markov, lm_head, logits, bonus, expected, candidates, partials = (
            self._greedy_fixture(300, 32, 5, 4, 1, "cuda")
        )
        output = torch.empty(5, 4, dtype=torch.int32, device="cuda")

        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads.all_gather_single"
        ) as gather:
            sample_dspark_block_greedy(
                logits, bonus, markov, lm_head, object(), candidates, partials, output
            )

        gather.assert_not_called()
        self.assertIsNone(partials)
        self.assertEqual(candidates.shape[:2], (2, 1))
        self.assertTrue(torch.equal(output.long(), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_dspark_tp_block_sampler_gathers_one_candidate_buffer_per_step(self):
        markov, lm_head, logits, bonus, expected, candidates, partials = (
            self._greedy_fixture(300, 32, 5, 4, 4, "cuda")
        )
        output = torch.empty(5, 4, dtype=torch.int32, device="cuda")

        def fake_all_gather(gathered, scored, group):
            # Every rank contributes this shard: the packed keys tie and the
            # resolve must still pick this shard's winner.
            self.assertTrue(gathered.is_contiguous() and scored.is_contiguous())
            self.assertEqual(gathered.numel(), 4 * scored.numel())
            gathered.view(4, -1).copy_(scored.expand(4, -1))

        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads.all_gather_single",
            side_effect=fake_all_gather,
        ) as gather:
            sample_dspark_block_greedy(
                logits, bonus, markov, lm_head, object(), candidates, partials, output
            )

        self.assertEqual(gather.call_count, 4)
        self.assertTrue(torch.equal(output.long(), expected))

        with self.assertRaisesRegex(ValueError, "does not cover the batch"):
            sample_dspark_block_greedy(
                logits,
                bonus,
                markov,
                lm_head,
                object(),
                candidates[:, :, :2].contiguous(),
                partials[:2],
                output,
            )
        lm_head.shard_indices.num_added_elements = 8
        with self.assertRaisesRegex(ValueError, "without added vocabulary"):
            sample_dspark_block_greedy(
                logits, bonus, markov, lm_head, object(), candidates, partials, output
            )

    def test_dspark_wire_target_uses_draft_head(self):
        draft_head = SimpleNamespace(weight=torch.ones(8, 3), tp_size=1)
        target_head = SimpleNamespace(weight=torch.ones(4, 3), tp_size=2)
        drafter = object.__new__(DeepseekV4DSpark)
        drafter.draft_model = SimpleNamespace(lm_head=draft_head)
        target_model = SimpleNamespace(
            lm_head=target_head,
            logits_processor=SimpleNamespace(tp_group=(0, 1)),
            set_dspark_layers_to_capture=Mock(),
        )

        drafter.wire_target(target_model)

        self.assertIs(drafter.target_model, target_model)
        self.assertIs(drafter.lm_head, draft_head)
        self.assertEqual(drafter.tp_group, (0, 1))
        # The draft model's configure_target installs the capture layers
        # before any drafter exists; wiring only binds execution resources.
        target_model.set_dspark_layers_to_capture.assert_not_called()

    def test_dspark_tp_only_contract_uses_resolved_mapping(self):
        mapping = SimpleNamespace(attn=SimpleNamespace(dp_size=1))
        DeepseekV4DSpark._validate_tp_only_mapping(mapping)

        invalid = SimpleNamespace(attn=SimpleNamespace(dp_size=2))
        with self.assertRaisesRegex(ValueError, "tensor parallelism only"):
            DeepseekV4DSpark._validate_tp_only_mapping(invalid)

    def test_dspark_padding_slots_reset_before_every_graph_replay(self):
        drafter = object.__new__(DeepseekV4DSpark)
        drafter.device = torch.device("cpu")
        drafter.first_padding_slot = 3
        drafter.padding_slots = torch.arange(3, 7, dtype=torch.int64)
        drafter.slot_indices_buf = drafter.padding_slots.clone()
        drafter.kv_windows = torch.full((7, 2, 4, 8), 7.0)
        drafter.context_lengths = torch.full((7,), 123, dtype=torch.int64)
        drafter._request_by_pool_slot = [None, "request-1", None]
        drafter.input_buffers = SimpleNamespace(
            max_bs=4,
            req_pool_indices_buf=torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            extend_prefix_lens_cpu=torch.zeros(4, dtype=torch.int64),
        )

        active_window = drafter.kv_windows[1].clone()
        active_context = drafter.context_lengths[1].clone()
        for _ in range(10):
            drafter.kv_windows[3:].fill_(9)
            drafter.context_lengths[3:].fill_(1048575)
            drafter.prepare_request_state(["request-1"], [1], num_extends=0)
            self.assertTrue(
                torch.equal(
                    drafter.kv_windows[3:],
                    torch.zeros_like(drafter.kv_windows[3:]),
                )
            )
            self.assertTrue(
                torch.equal(
                    drafter.context_lengths[3:],
                    torch.zeros_like(drafter.context_lengths[3:]),
                )
            )
            self.assertTrue(
                torch.equal(
                    drafter.slot_indices_buf,
                    torch.tensor([1, 4, 5, 6]),
                )
            )

        self.assertTrue(torch.equal(drafter.kv_windows[1], active_window))
        self.assertTrue(torch.equal(drafter.context_lengths[1], active_context))

        drafter.kv_windows[3:].fill_(11)
        drafter.context_lengths[3:].fill_(999)
        drafter.prepare_request_state([], [], num_extends=0)
        self.assertTrue(torch.equal(drafter.slot_indices_buf, drafter.padding_slots))
        self.assertTrue(
            torch.equal(
                drafter.kv_windows[3:],
                torch.zeros_like(drafter.kv_windows[3:]),
            )
        )
        self.assertTrue(
            torch.equal(
                drafter.context_lengths[3:],
                torch.zeros_like(drafter.context_lengths[3:]),
            )
        )

    def test_dspark_fp8_quant_dequant_matches_ue8m0_reference(self):
        values = torch.linspace(-7.0, 7.0, 256, dtype=torch.float32).reshape(2, 128)
        actual = dspark_fp8_quant_dequant(values, 64)

        blocks = values.unflatten(-1, (-1, 64))
        absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        scales = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
        expected = (
            (blocks / scales)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .float()
            .mul(scales)
            .flatten(-2)
        )

        self.assertTrue(torch.equal(actual, expected))
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            dspark_fp8_quant_dequant(values[:, :-1], 64)

    def test_dspark_rmsnorm_cpu_matches_reference(self):
        values = torch.linspace(-7.0, 7.0, 1024, dtype=torch.bfloat16).reshape(2, 512)
        weight = torch.linspace(-0.5, 0.5, 512, dtype=torch.bfloat16)

        actual = _rmsnorm(values, weight, 1e-6)
        normalized = values.float()
        normalized.mul_(torch.rsqrt(normalized.square().mean(-1, keepdim=True) + 1e-6))
        expected = (weight.float() * normalized).to(values.dtype)

        self.assertTrue(torch.equal(actual, expected))

    def test_dspark_query_per_head_norm_cpu_matches_reference(self):
        values = torch.linspace(-7.0, 7.0, 2 * 3 * 4 * 8).reshape(2, 3, 4, 8)
        expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)

        actual = _normalize_query_per_head(values.clone(), head_dim=8, eps=1e-6)

        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_dspark_rmsnorm_cuda_graph_replays_exact_reference(self):
        torch.manual_seed(7)
        values = torch.randn((8, 5, 512), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((512,), device="cuda", dtype=torch.bfloat16)

        _rmsnorm(values, weight, 1e-6)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _rmsnorm(values, weight, 1e-6)

        values.copy_(torch.randn_like(values))
        graph.replay()
        torch.cuda.synchronize()
        normalized = values.float()
        normalized.mul_(torch.rsqrt(normalized.square().mean(-1, keepdim=True) + 1e-6))
        expected = (weight.float() * normalized).to(values.dtype)

        self.assertTrue(torch.equal(actual, expected))

    def test_dspark_kv_quantizes_only_non_rope_channels(self):
        tensor = torch.linspace(-8.0, 8.0, 512, dtype=torch.float32).reshape(1, 1, 512)

        actual = _quantize_dspark_non_rope(tensor, rope_head_dim=64)

        self.assertTrue(torch.equal(actual[..., -64:], tensor[..., -64:]))
        self.assertTrue(
            torch.equal(
                actual[..., :-64],
                dspark_fp8_quant_dequant(tensor[..., :-64], 64),
            )
        )
        self.assertFalse(torch.equal(actual[..., :-64], tensor[..., :-64]))

    def test_dspark_output_projection_keeps_wo_a_input_in_bf16(self):
        output = torch.linspace(
            -3.0,
            3.0,
            128,
            dtype=torch.bfloat16,
        ).reshape(1, 1, 1, 128)
        wo_a = torch.linspace(
            -0.5,
            0.5,
            128 * 128,
            dtype=torch.bfloat16,
        ).reshape(128, 128)
        wo_b = torch.linspace(
            -0.25,
            0.25,
            4 * 128,
            dtype=torch.bfloat16,
        ).reshape(4, 128)

        bf16_intermediate = torch.einsum(
            "bsgd,grd->bsgr",
            output,
            wo_a.view(1, 128, 128),
        ).flatten(2)
        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark_ops.attention.dspark_fp8_quant_dequant",
            wraps=dspark_fp8_quant_dequant,
        ) as quantize:
            actual = _dspark_output_projection(
                output,
                wo_a,
                wo_b,
                n_groups=1,
                o_lora_rank=128,
            )

        expected = F.linear(
            dspark_fp8_quant_dequant(bf16_intermediate, 128),
            wo_b,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(quantize.call_count, 1)
        quantize_input, quantize_block_size = quantize.call_args.args
        self.assertTrue(torch.equal(quantize_input, bf16_intermediate))
        self.assertEqual(quantize_block_size, 128)

    def test_dspark_speculative_config_uses_block_plus_bonus_width(self):
        server_args = ServerArgs(
            model="unused",
            disable_kvstore=True,
            enable_prefix_caching=False,
            speculative_config=json.dumps(
                {"method": "dspark", "num_speculative_tokens": 5}
            ),
        )

        self.assertEqual(server_args.speculative_algorithm, "DSPARK")
        self.assertEqual(server_args.speculative_num_steps, 5)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_dspark_allows_prefix_cache_when_kvstore_is_disabled(self):
        server_args = ServerArgs(
            model="unused",
            disable_kvstore=True,
            speculative_config=json.dumps(
                {"method": "dspark", "num_speculative_tokens": 5}
            ),
        )

        self.assertTrue(server_args.enable_prefix_caching)
        self.assertTrue(server_args.draft_model_path_use_base)

    def test_dspark_same_checkpoint_keeps_kvstore_default(self):
        # Whether same-checkpoint DSpark can use KVStore depends on where the
        # draft keeps its windows, which only the resolved draft config knows;
        # argument parsing no longer pre-empts that decision.
        server_args = ServerArgs(
            model="same-checkpoint",
            speculative_config=json.dumps(
                {
                    "method": "dspark",
                    "model": "same-checkpoint",
                    "num_speculative_tokens": 5,
                }
            ),
        )

        self.assertTrue(server_args.enable_kvstore)
        self.assertTrue(server_args.draft_model_path_use_base)

    def test_dspark_explicit_external_checkpoint_preserves_cache_behavior(self):
        server_args = ServerArgs(
            model="target-checkpoint",
            speculative_algorithm="DSPARK",
            speculative_draft_model_path="external-draft",
            speculative_num_steps=5,
        )

        self.assertTrue(server_args.enable_kvstore)
        self.assertTrue(server_args.enable_prefix_caching)
        self.assertFalse(server_args.draft_model_path_use_base)
        self.assertEqual(server_args.speculative_draft_model_path, "external-draft")

    def test_decode_kvstore_without_prefix_cache_preserves_generic_behavior(self):
        server_args = ServerArgs(
            model="target-checkpoint",
            enable_prefix_caching=False,
            disaggregation_mode="decode",
        )

        self.assertTrue(server_args.enable_kvstore)
        self.assertFalse(server_args.enable_prefix_caching)

    def test_dspark_external_decode_preserves_generic_cache_behavior(self):
        server_args = ServerArgs(
            model="target-checkpoint",
            enable_prefix_caching=False,
            disaggregation_mode="decode",
            speculative_algorithm="DSPARK",
            speculative_draft_model_path="external-draft",
            speculative_num_steps=5,
        )

        self.assertTrue(server_args.enable_kvstore)
        self.assertFalse(server_args.enable_prefix_caching)
        self.assertFalse(server_args.draft_model_path_use_base)
        self.assertEqual(server_args.speculative_draft_model_path, "external-draft")

    def test_dspark_external_checkpoint_preserves_generic_cache_behavior(self):
        server_args = ServerArgs(
            model="target-checkpoint",
            speculative_config=json.dumps(
                {
                    "method": "dspark",
                    "model": "external-draft",
                    "num_speculative_tokens": 6,
                }
            ),
        )

        self.assertTrue(server_args.enable_kvstore)
        self.assertTrue(server_args.enable_prefix_caching)
        self.assertFalse(server_args.draft_model_path_use_base)
        self.assertEqual(server_args.speculative_draft_model_path, "external-draft")
        self.assertEqual(server_args.speculative_num_steps, 5)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_deepseek_v4_attention_layout_matches_compressed_cache_contract(self):
        config = SimpleNamespace(
            compress_ratios=[0, 4, 128],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=False,
        )
        layout_fp4 = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        self.assertEqual(layout.layer_ratio, (1, 4, 128))
        self.assertEqual(layout.swa_token_stride, 576)
        self.assertEqual(layout.swa_scale_dim, 8)
        self.assertEqual(layout.swa_row_bytes, 584)
        self.assertEqual(layout.state_width(0), 512)
        self.assertEqual(layout.state_width(1), 1024)
        self.assertEqual(layout.state_width(2), 512)
        self.assertEqual(layout.state_width(1, indexer=True), 256)
        self.assertEqual(layout.indexer_row_bytes, 132)
        self.assertEqual(layout_fp4.indexer_row_bytes, 68)

    def test_deepseek_v4_cache_layout_can_slice_mtp_layer_range(self):
        config = SimpleNamespace(
            compress_ratios=[0, 4, 128, 0],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )

        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
            layer_indices=range(3, 4),
        )

        self.assertEqual(layout.layer_ratio, (1,))
        with self.assertRaisesRegex(ValueError, "out of range"):
            deepseek_v4_cache_layout_from_config(
                config,
                page_size=64,
                use_fp4_indexer_cache=True,
                layer_indices=range(4, 5),
            )

    def test_deepseek_v4_attention_layout_rejects_unknown_ratio(self):
        config = SimpleNamespace(
            compress_ratios=[8],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        with self.assertRaisesRegex(ValueError, "compress_ratio=8"):
            deepseek_v4_cache_layout_from_config(
                config,
                page_size=64,
                use_fp4_indexer_cache=False,
            )

    def test_deepseek_v4_kv_pool_allocates_v4_cache_families(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        pool, plan = _make_planned_deepseek_v4_pool(layout, config)

        self.assertEqual(
            tuple(pool.get_swa_kv_buffer(0).shape),
            (plan.group("v4.swa_kv").page_count, 37440),
        )
        self.assertIsNone(pool.compressed_kv_buffer[0])
        self.assertEqual(
            tuple(pool.get_compressed_kv_buffer_2d(1).shape),
            (plan.group("v4.c4a.compressed_kv").page_count, 37440),
        )
        self.assertEqual(
            tuple(pool.get_compressor_state_buffer(1).shape),
            (plan.group("v4.c4a.compressor_state").page_count, 4, 2048),
        )
        self.assertEqual(
            tuple(pool.get_compressor_state_buffer(2).shape),
            (plan.group("v4.c128a.compressor_state").page_count, 8, 1024),
        )
        self.assertEqual(pool.get_compressor_state_buffer(1).dtype, torch.float32)
        self.assertEqual(pool.get_compressor_state_buffer(2).dtype, torch.float32)
        self.assertEqual(
            tuple(pool.get_indexer_kv_buffer_2d(1).shape),
            (plan.group(V4_INDEXER_KV_GROUP_ID).page_count, 64 * 68),
        )
        self.assertEqual(
            tuple(pool.get_indexer_state_buffer(1).shape),
            (
                plan.group("v4.c4a.indexer_compressor_state").page_count,
                4,
                512,
            ),
        )
        self.assertEqual(pool.get_indexer_state_buffer(1).dtype, torch.float32)

    def test_deepseek_v4_kv_pool_binds_no_paged_attention_layer(self):
        # Every V4 group stores rows of token history, and a layer's fused
        # attention reads several of them at once (its SWA window beside its
        # compressed chain), so the generic per-layer derivation has no single
        # group to hand PagedAttention -- the view says so instead of raising.
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )
        pool, _ = _make_planned_deepseek_v4_pool(layout, config)

        self.assertTrue(
            all(spec.attn_kind != "state" for spec in pool.arena.cache_group_specs)
        )
        with self.assertRaisesRegex(ValueError, "more than one history"):
            derive_history_groups_by_layer(
                pool.arena,
                first_layer=0,
                num_layers=len(layout.layer_ratio),
                kv_planes=pool.layer_plane_bindings,
            )
        self.assertEqual(pool.history_group_by_layer(), {})
        self.assertEqual(pool.paged_group_ids, ())
        bind_cache_groups(torch.nn.Module(), pool)

    def test_deepseek_v4_kv_pool_uses_compressed_storage_blocks_for_page256(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=256,
            use_fp4_indexer_cache=True,
        )
        pool, plan = _make_planned_deepseek_v4_pool(layout, config)

        self.assertEqual(pool.swa_block_size, 64)
        self.assertEqual(pool.get_compressed_block_size(1), 64)
        self.assertEqual(pool.get_compressed_block_size(2), 2)
        self.assertEqual(
            tuple(pool.get_compressed_kv_buffer_2d(1).shape),
            (plan.group("v4.c4a.compressed_kv").page_count, 37440),
        )
        self.assertEqual(
            tuple(pool.get_indexer_kv_buffer_2d(1).shape),
            (plan.group(V4_INDEXER_KV_GROUP_ID).page_count, 64 * 68),
        )

    def test_deepseek_v4_plan_rejects_nonpositive_capacity(self):
        # Capacity is a plan quantity (the pool derives its size from the
        # arena), so the planner owns this guard.
        config = SimpleNamespace(
            compress_ratios=[1],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        cache_layout = _v4_layout(config)[2]

        for capacity in (0, -1):
            with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                cache_layout.bind(capacity)

    def test_deepseek_v4_group_slot_mapping_expands_per_request_indices(self):
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([0, 1, 2, 64, 65, 66], dtype=torch.int64),
            req_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            rows_per_page=64,
        )

        self.assertTrue(
            torch.equal(slots, torch.tensor([640, 641, 642, 1344, 1345, 1346]))
        )

    def test_deepseek_v4_kv_pool_requires_matching_layout_layers(self):
        config = SimpleNamespace(
            compress_ratios=[1],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        from cache_pool_test_utils import make_pool

        with self.assertRaisesRegex(ValueError, "layer_num"):
            make_pool(
                HybridDeepseekV4TokenToKVPool,
                _v4_layout(config)[2].bind(1),
                device="cpu",
                layout=layout,
                layer_num=2,
                rank=0,
            )

    def test_deepseek_v4_group_slot_mapping_from_raw(self):
        block_table = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([0, 63, 64, 9, 10], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
            block_table=block_table,
            rows_per_page=64,
            entry_stride_tokens=1,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([640, 703, 704, 1289, 1290])))

        compressed_slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([0, 255, 256, 511], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 0, 1], dtype=torch.int32),
            block_table=block_table,
            rows_per_page=64,
            entry_stride_tokens=4,
        )
        self.assertTrue(
            torch.equal(compressed_slots, torch.tensor([640, 703, 704, -1]))
        )

    def test_deepseek_v4_slot_mapping_masks_invalid_tokens(self):
        slots = _mask_invalid_graph_tokens(
            torch.tensor([10, 20, -1, 40], dtype=torch.int64),
            torch.tensor([True, False, True, False]),
        )

        self.assertTrue(torch.equal(slots, torch.tensor([10, -1, -1, -1])))

    def test_deepseek_v4_spec_geometry_is_prefix_granularity_free(self):
        """The spec constructors take no P; block spans come from the kernel
        page registry (256 // ratio rows x ratio-token stride)."""
        specs = {
            spec.group_id: spec
            for spec in _v4_spec_set(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4, 128),
            )
        }
        c4 = specs["v4.c4a.compressed_kv"]
        self.assertEqual(c4.rows_per_page, 64)
        self.assertEqual(c4.entry_stride_tokens, 4)
        self.assertEqual(c4.block_granularity, 256)
        c128 = specs["v4.c128a.compressed_kv"]
        self.assertEqual(c128.rows_per_page, 2)
        self.assertEqual(c128.entry_stride_tokens, 128)
        self.assertEqual(c128.block_granularity, 256)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_prefill_topk_cuda_op_matches_torch_topk(self):
        if not has_indexer_topk_prefill():
            self.skipTest("DeepSeek V4 prefill top-k op is unavailable")

        torch.manual_seed(0)
        lengths = torch.tensor([0, 3, 17, 33], device="cuda", dtype=torch.int32)
        logits = torch.randn((lengths.numel(), 40), device="cuda", dtype=torch.float32)
        row_starts = torch.zeros_like(lengths)
        out = torch.full((lengths.numel(), 8), -1, device="cuda", dtype=torch.int32)

        indexer_topk_prefill(logits, row_starts, lengths, out, out.shape[-1])
        torch.cuda.synchronize()

        for row, raw_len in enumerate(lengths.cpu().tolist()):
            selected = min(raw_len, out.shape[-1])
            actual = out[row, :selected].sort().values.cpu()
            if selected == 0:
                self.assertTrue(torch.equal(out[row], torch.full_like(out[row], -1)))
                continue
            expected = (
                torch.topk(
                    logits[row, :raw_len],
                    k=selected,
                    dim=-1,
                    sorted=False,
                )
                .indices.sort()
                .values.cpu()
                .to(torch.int32)
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(
                torch.equal(
                    out[row, selected:],
                    torch.full_like(out[row, selected:], -1),
                )
            )

    def test_deepseek_v4_global_topk_cpu_masks_invalid_req_before_indexing(self):
        indices, lens = dsv4_compute_global_topk_indices_and_lens(
            topk_indices=torch.tensor([[0, 4], [0, 1]], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 99], dtype=torch.int32),
            block_table=torch.tensor([[10]], dtype=torch.int32),
            block_size=4,
            is_valid_token=torch.tensor([True, False]),
        )

        self.assertTrue(
            torch.equal(
                indices,
                torch.tensor([[40, -1], [-1, -1]], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(lens, torch.tensor([2, 0], dtype=torch.int32)))

    def test_hidden_compression_reference_preserves_expected_shapes(self):
        torch.manual_seed(0)
        tokens, hc_mult, hidden = 3, 4, 5
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.float32)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(mix_hc, dtype=torch.float32)

        layer_input, post, comb = _mhc_pre_reference(
            residual,
            fn,
            scale,
            base,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=2,
        )
        updated = _mhc_post_reference(layer_input, residual, post, comb)

        self.assertEqual(tuple(layer_input.shape), (tokens, hidden))
        self.assertEqual(tuple(post.shape), (tokens, hc_mult, 1))
        self.assertEqual(tuple(comb.shape), (tokens, hc_mult, hc_mult))
        self.assertEqual(tuple(updated.shape), tuple(residual.shape))

    def test_hidden_compression_pre_reference_matches_math(self):
        torch.manual_seed(1)
        tokens, hc_mult, hidden = 2, 3, 4
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.bfloat16)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.tensor([0.7, 1.1, 0.5], dtype=torch.float32)
        base = torch.randn(mix_hc, dtype=torch.float32)
        eps = 1e-5

        layer_input, post, comb = _mhc_pre_reference(
            residual, fn, scale, base, rms_eps=1e-6, hc_eps=eps, sinkhorn_iters=3
        )

        x = residual.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        mixes = F.linear(x, fn) * rsqrt
        pre_raw, post_raw, comb_raw = torch.split(
            mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        pre_base, post_base, comb_base = torch.split(
            base, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        expected_pre = torch.sigmoid(pre_raw * scale[0] + pre_base) + eps
        expected_post = (
            torch.sigmoid(post_raw * scale[1] + post_base) * 2.0
        ).unsqueeze(-1)
        expected_comb = (
            F.softmax(
                comb_raw.reshape(tokens, hc_mult, hc_mult) * scale[2]
                + comb_base.reshape(1, hc_mult, hc_mult),
                dim=-1,
            )
            + eps
        )
        expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(2):
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-1, keepdim=True) + eps
            )
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-2, keepdim=True) + eps
            )
        expected_layer_input = torch.sum(
            expected_pre.unsqueeze(-1) * residual.float(), dim=1
        ).to(residual.dtype)

        self.assertTrue(torch.allclose(layer_input, expected_layer_input))
        self.assertTrue(torch.allclose(post, expected_post))
        self.assertTrue(torch.allclose(comb, expected_comb))

    def test_hidden_compression_post_reference_matches_lane_orientation(self):
        hidden_states = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        post = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
        comb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)

        updated = _mhc_post_reference(hidden_states, residual, post, comb)

        expected = torch.empty_like(residual)
        expected[:, 0] = (
            comb[:, 0, 0:1] * residual[:, 0]
            + comb[:, 1, 0:1] * residual[:, 1]
            + post[:, 0] * hidden_states
        )
        expected[:, 1] = (
            comb[:, 0, 1:2] * residual[:, 0]
            + comb[:, 1, 1:2] * residual[:, 1]
            + post[:, 1] * hidden_states
        )
        self.assertTrue(torch.allclose(updated, expected))

    def test_deepseek_v4_score_routing_does_not_forward_input_ids(self):
        hidden_states = torch.ones((2, 4))
        input_ids = torch.arange(2)
        gate = Mock(return_value=torch.zeros((2, 8)))
        gate.e_score_correction_bias = torch.zeros(8)
        gate.tid2eid = None
        moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
        moe.config = SimpleNamespace(n_routed_experts=8)
        moe.gate = gate

        _, _, hash_indices_table, routing_input_ids = moe._routing_inputs(
            hidden_states, input_ids
        )

        self.assertIsNone(hash_indices_table)
        self.assertIsNone(routing_input_ids)

    def test_deepseek_v4_router_matches_noaux_bias_semantics(self):
        logits = torch.tensor(
            [
                [0.2, 1.0, -0.5, 0.7],
                [1.5, -0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)

        topk_weights, topk_ids = moe_topk(
            logits,
            top_k=2,
            score_function="sqrt_softplus",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=1.0,
            correction_bias=bias,
            solution="torch",
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=2, dim=-1, sorted=False)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_hash_router_uses_table_ids_and_gate_scores(self):
        logits = torch.tensor(
            [
                [0.5, 1.0, -0.5, 0.1],
                [-0.2, 0.3, 1.4, 0.0],
            ],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([3, 1], dtype=torch.long)
        table = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 0],
                [3, 1],
            ],
            dtype=torch.int32,
        )

        topk_weights, topk_ids = moe_topk(
            logits,
            top_k=2,
            score_function="sqrt_softplus",
            selection_method="hash",
            renormalize=True,
            routed_scaling_factor=1.0,
            hash_indices_table=table,
            input_ids=input_ids,
            solution="torch",
        )

        expected_ids = torch.tensor([[3, 1], [2, 3]], dtype=torch.int32)
        expected_scores = F.softplus(logits).sqrt()
        expected_weights = expected_scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_gate_cpu_returns_fp32_logits(self):
        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            num_hash_layers=0,
            topk_method=None,
        )
        gate = DeepseekV4MoEGate(config, layer_index=1)
        with torch.no_grad():
            gate.weight.copy_(torch.randn_like(gate.weight))
        hidden_states = torch.randn(3, config.hidden_size)

        logits = gate(hidden_states)
        expected = F.linear(hidden_states, gate.weight, None).float()

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.allclose(logits, expected))

    def test_deepseek_v4_dummy_hash_table_spreads_tokens_over_experts(self):
        config = SimpleNamespace(
            n_routed_experts=16,
            hidden_size=8,
            num_hash_layers=1,
            vocab_size=64,
            num_experts_per_tok=4,
        )
        gate = DeepseekV4MoEGate(config, layer_index=0)
        gate.tid2eid.data.fill_(-1)

        initialize_dummy_integer_weights(gate)

        table = gate.tid2eid
        self.assertTrue(bool(((table >= 0) & (table < 16)).all()))
        self.assertTrue(all(row.unique().numel() == 4 for row in table))
        self.assertEqual(table.unique().numel(), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_gate_dsv3_router_gemm_shape(self):
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("DSV3 router GEMM requires SM90+")

        config = SimpleNamespace(
            n_routed_experts=256,
            hidden_size=4096,
            num_hash_layers=0,
            topk_method=None,
        )
        gate = DeepseekV4MoEGate(config, layer_index=1).cuda().to(torch.bfloat16)
        hidden_states = torch.randn(
            2, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        try:
            logits = gate(hidden_states)
        except RuntimeError as exc:
            if "dsv3_gemm library not found" not in str(exc):
                raise
            self.skipTest(str(exc))
        torch.cuda.synchronize()

        self.assertEqual(tuple(logits.shape), (2, config.n_routed_experts))
        self.assertEqual(logits.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_softplus_sqrt_topk_matches_reference(self):
        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(3, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)
        topk_weights = torch.empty(3, 6, device="cuda", dtype=torch.float32)
        topk_ids = torch.empty(3, 6, device="cuda", dtype=torch.int32)

        try:
            softplus_sqrt_topk_flash(logits, bias, topk_ids, topk_weights, 1.0, True)
        except (AttributeError, RuntimeError) as exc:
            self.skipTest(f"fused DeepSeek V4 router op unavailable: {exc}")
        torch.cuda.synchronize()

        scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_moe_topk_matches_reference(self):
        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(2, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)

        topk_weights, topk_ids = moe_topk(
            logits,
            top_k=6,
            score_function="sqrt_softplus",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=1.0,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_hash_topk_matches_reference(self):
        logits = torch.linspace(
            -2.0, 2.0, 256, device="cuda", dtype=torch.float32
        ).repeat(3, 1)
        input_ids = torch.tensor([1, 0, 1], device="cuda", dtype=torch.long)
        table = torch.tensor(
            [[5, 7, 11, 13, 17, 19], [23, 29, 31, 37, 41, 43]],
            device="cuda",
            dtype=torch.int32,
        )
        topk_weights = torch.empty(3, 6, device="cuda", dtype=torch.float32)
        topk_ids = torch.empty(3, 6, device="cuda", dtype=torch.int32)

        try:
            hash_softplus_sqrt_topk_flash(
                logits, input_ids, table, topk_ids, topk_weights, 1.0, True
            )
        except (AttributeError, RuntimeError) as exc:
            self.skipTest(f"fused DeepSeek V4 hash router op unavailable: {exc}")
        torch.cuda.synchronize()

        expected_ids = table[input_ids]
        scores = F.softplus(logits).sqrt()
        expected_weights = scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))


if __name__ == "__main__":
    unittest.main()


def test_v4_merged_solve_draft_is_a_continuation_layer():
    """One big model on V4: the MTP draft layer is simply the next layer of
    the merged model (compress_ratios already carries it), so one solve
    yields one plan whose draft fields share the target groups' packing."""
    hf_config = SimpleNamespace(
        num_hidden_layers=6,
        compress_ratios=[0, 4, 128],
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        sliding_window=128,
        index_head_dim=128,
    )
    layout = deepseek_v4_cache_layout_from_config(
        hf_config, page_size=64, use_fp4_indexer_cache=False
    )
    merged = _v4_layout(hf_config)[2]
    plan = merged.bind(2)
    # Every layer's swa field (all 6 layers incl. the MTP continuation
    # layer) shares the one v4.swa_kv group — one packing, one page-id
    # space for target and draft alike.
    swa_fields = [f for f in plan.fields if f.group_id == "v4.swa_kv"]
    assert len(swa_fields) == len(layout.layer_ratio)
    packing = dict(merged.group_packing)
    group = plan.group("v4.swa_kv")
    assert group.page_count == 1 + 2 * packing["v4.swa_kv"]


def test_v4_pd_recipe_and_readiness_follow_cache_producers():
    setup = DeepseekV4Recipe(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            speculative_algorithm=None,
            attention_use_fp4_indexer_cache=False,
            max_total_tokens=64 * 1024,
            chunked_prefill_size=256,
            disaggregation_mode="prefill",
            enable_prefix_caching=True,
        ),
        model_config=SimpleNamespace(
            num_attention_layers=3,
            hf_config=SimpleNamespace(
                compress_ratios=(1, 4, 128),
                head_dim=512,
                qk_rope_head_dim=64,
                sliding_window=128,
                index_head_dim=128,
                attention_config={},
            ),
        ),
        attn_config=SimpleNamespace(
            kvp_size=1,
            pd_disaggregation_enabled=True,
            prefix_granularity=256,
            max_bs=2,
            context_len=1024,
        ),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 30,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    ).setup()
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
        CacheLayerOwnership,
        cache_field_placement,
    )

    _, schedules = cache_field_placement(
        setup.spec.memory_plan, (CacheLayerOwnership(3, 0, (0, 3)),)
    )
    schedule = build_cache_fields_by_producer_step(
        setup.spec.memory_plan, producer_fields_by_step=schedules[0]
    )
    assert schedule.step_count == 3
    assert all(schedule.fields_by_step)
