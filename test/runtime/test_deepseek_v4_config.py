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
from unittest.mock import patch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch
import torch.nn.functional as F
from tokenspeed_kernel import (
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_select_experts,
)
from tokenspeed_kernel.ops.attention.cuda.dsv4 import (
    has_indexer_topk_prefill,
    indexer_topk_prefill,
)
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
from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    CudaGraphWrapper,
    _should_update_mamba_state_after_mtp_verify,
)
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (
    DeepseekV4DSpark,
    _dspark_decode_position_plan,
    _dspark_prefill_position_plan,
)
from tokenspeed.runtime.execution.drafter.eagle import (
    _advance_draft_forward_metadata_if_supported,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.attention.backends import (
    deepseek_v4 as deepseek_v4_backend,
)
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
    DeepseekV4IndexerDecodePlan,
    DeepseekV4IndexerPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    deepseek_v4_cache_layout_from_config,
    deepseek_v4_indexer_fp8_row_bytes,
    deepseek_v4_indexer_mxfp4_row_bytes,
    deepseek_v4_nope_dim,
    deepseek_v4_swa_row_bytes,
    deepseek_v4_swa_token_stride,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
    HybridDeepseekV4TokenToKVPool,
    _group_slot_mapping_from_raw,
    _mask_invalid_graph_tokens,
    _split_block_tables_into_v4_metadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
    v4_c4_state_window,
    v4_compressed_kv_spec,
    v4_compressor_state_spec,
    v4_indexer_state_spec,
    v4_swa_kv_spec,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.quantization import (
    QUANTIZATION_METHODS,
    Fp8Config,
    Mxfp4Config,
)
from tokenspeed.runtime.models import deepseek_v4 as deepseek_v4_model
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Indexer,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4MoE,
    DeepseekV4MoEGate,
    _deepseek_v4_expert_scale_parameter_name,
    _deepseek_v4_forward_metadata,
    _deepseek_v4_indexer_decode_max_len,
    _deepseek_v4_indexer_decode_plan,
    _deepseek_v4_indexer_prefill_max_logits_bytes,
    _deepseek_v4_indexer_prefill_metadata,
    _deepseek_v4_indexer_prefill_request_chunks,
    _deepseek_v4_indexer_prefill_request_gather_plan,
    _deepseek_v4_indexer_token_split,
    _deepseek_v4_indexer_topk_from_logits,
    _deepseek_v4_mega_moe_max_num_tokens,
    _deepseek_v4_reorder_c4_ape_2604,
    _deepseek_v4_routed_expert_quant_config,
    _DeepseekV4TopKBuffer,
    deepseek_v4_rope_config,
    hc_head,
    mhc_post,
    mhc_pre,
    pack_topk_as_router_logits,
)
from tokenspeed.runtime.models.deepseek_v4_dspark import (
    _ATTENTION_CHECKPOINT_TENSORS,
    DeepseekV4DSparkModel,
    DeepseekV4ForCausalLMDSpark,
    _apply_dspark_hc_head,
    _is_zero_initialized_expert_bias,
    count_dspark_stages,
)
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.attention import (
    _dspark_output_projection,
    _quantize_dspark_non_rope,
    dspark_fp8_quant_dequant,
    get_dspark_topk_idxs_batched,
)
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads import _local_vocab_argmax
from tokenspeed.runtime.models.deepseek_v4_mtp import DeepseekV4ForCausalLMNextN
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


def _v4_spec_set(hf_config, *, layer_ratio, decode_input_tokens: int = 1):
    """The spec set a ratio vector declares, in the recipe's own order."""
    ratios = {int(ratio) for ratio in layer_ratio}
    window = v4_c4_state_window(decode_input_tokens)
    specs = [v4_swa_kv_spec(hf_config)]
    for ratio in sorted(r for r in ratios if r > 1):
        specs.append(v4_compressor_state_spec(ratio, c4_state_window=window))
        specs.append(v4_compressed_kv_spec(ratio))
    if 4 in ratios:
        specs.append(v4_indexer_state_spec(c4_state_window=window))
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
            max_total_tokens=None,
            chunked_prefill_size=prefix_granularity,
            attention_use_fp4_indexer_cache=True,
            speculative_algorithm=None,
        ),
        model_config=SimpleNamespace(
            hf_config=hf_config, num_attention_layers=num_layers
        ),
        attn_config=SimpleNamespace(
            prefix_granularity=prefix_granularity,
            max_bs=1,
            context_len=4096,
            pd_disaggregation_enabled=False,
        ),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 34,
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
        model_dtype=torch.bfloat16,
        layout=layout,
        layer_num=len(layout.layer_ratio),
        rank=0,
        cache_group_specs=specs,
        token_capacity=pool_size,
    )
    return pool, plan


def _make_deepseek_v4_forward_metadata(
    *,
    page_size,
    req_pool_indices,
    page_table,
    seq_lens,
    query_lens,
    query_start_loc,
    token_to_req_indices,
    block_tables=None,
    block_table_base_offsets=None,
    swa_page_table=None,
    swa_base_logical_page=None,
    compressor_state_block_tables=None,
    compressor_state_base_logical_pages=None,
    indexer_state_block_table=None,
    indexer_state_base_logical_page=None,
    **kwargs,
):
    (
        split_swa,
        split_compressor_state,
        split_indexer_state,
        split_swa_base,
        split_compressor_state_base,
        split_indexer_state_base,
    ) = _split_block_tables_into_v4_metadata(
        block_tables or {},
        block_table_base_offsets,
    )
    if swa_page_table is None:
        swa_page_table = split_swa
    if swa_base_logical_page is None:
        swa_base_logical_page = split_swa_base
    if compressor_state_block_tables is None:
        compressor_state_block_tables = split_compressor_state
    if compressor_state_base_logical_pages is None:
        compressor_state_base_logical_pages = split_compressor_state_base
    if indexer_state_block_table is None:
        indexer_state_block_table = split_indexer_state
    if indexer_state_base_logical_page is None:
        indexer_state_base_logical_page = split_indexer_state_base

    cache = DeepseekV4CacheMetadata(
        page_size=page_size,
        page_table=page_table,
        block_tables=block_tables or {},
        block_table_base_offsets=(block_table_base_offsets or {}),
        swa_page_table=swa_page_table,
        swa_base_logical_page=swa_base_logical_page,
        compressor_state_block_tables=compressor_state_block_tables,
        compressor_state_base_logical_pages=compressor_state_base_logical_pages,
        indexer_state_block_table=indexer_state_block_table,
        indexer_state_base_logical_page=indexer_state_base_logical_page,
    )
    return DeepseekV4ForwardMetadata(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        query_lens=query_lens,
        query_start_loc=query_start_loc,
        token_to_req_indices=token_to_req_indices,
        cache=cache,
        **kwargs,
    )


def _v4_compressed_kv_tables(
    *,
    c4: torch.Tensor | None = None,
    c128: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    tables: dict[str, torch.Tensor] = {}
    if c4 is not None:
        tables["v4.c4a.compressed_kv"] = c4
    if c128 is not None:
        tables["v4.c128a.compressed_kv"] = c128
    return tables


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

    def test_expert_scale_name_follows_expert_format_and_backend(self):
        cases = (
            ("fp4", False, "weight_scale"),
            ("fp4", True, "weight_scale"),
            ("fp8", False, "weight_scale_inv"),
            ("fp8", True, "weight_scale"),
            (None, False, "weight_scale_inv"),
        )
        for expert_dtype, use_mega_moe, expected in cases:
            with self.subTest(
                expert_dtype=expert_dtype,
                use_mega_moe=use_mega_moe,
            ):
                self.assertEqual(
                    _deepseek_v4_expert_scale_parameter_name(
                        SimpleNamespace(expert_dtype=expert_dtype),
                        use_mega_moe=use_mega_moe,
                    ),
                    expected,
                )

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
                out_cache_loc,
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

        empty = torch.empty(0, dtype=torch.int32)
        result = runner.forward(
            ctx=None,
            input_ids=empty,
            positions=empty,
            out_cache_loc=empty,
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
                out_cache_loc,
            ):
                return "ok"

        runner = object.__new__(ModelRunner)
        runner.model = ModelWithoutSpecStep()
        runner.is_generation = True
        runner._model_forward_accepts_spec_step_idx = (
            ModelRunner._forward_accepts_kwarg(runner.model, "spec_step_idx")
        )

        empty = torch.empty(0, dtype=torch.int32)
        result = runner.forward(
            ctx=None,
            input_ids=empty,
            positions=empty,
            out_cache_loc=empty,
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
                out_cache_loc,
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

        empty = torch.empty(0, dtype=torch.int32)
        result = runner.forward(
            ctx=None,
            input_ids=empty,
            positions=empty,
            out_cache_loc=empty,
            spec_step_idx=2,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(runner.model.received_kwargs, {})

    def test_deepseek_v4_indexer_token_split_treats_spec_modes_as_decode(self):
        metadata = SimpleNamespace(num_prefill_tokens=2)
        metadata.decode_token_count = lambda: 3

        self.assertEqual(
            _deepseek_v4_indexer_token_split(ForwardMode.MIXED, metadata, 5),
            (2, 3),
        )
        self.assertEqual(
            _deepseek_v4_indexer_token_split(ForwardMode.EXTEND, metadata, 5),
            (5, 0),
        )
        self.assertEqual(
            _deepseek_v4_indexer_token_split(ForwardMode.DECODE, metadata, 5),
            (0, 5),
        )

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
            "_forward_shared_experts",
            "forward_mega_moe",
            "forward_normal",
        ):
            setattr(moe, name, MethodType(getattr(DeepseekV4MoE, name), moe))
        return moe

    def _make_fake_deepseek_v4_moe(self, hidden_states, input_ids, stream_fork, calls):
        def select_experts(states, ids):
            calls.append("select")
            self.assertIs(states, hidden_states)
            self.assertIs(ids, input_ids)
            topk_shape = (states.shape[0], 2)
            return (
                torch.ones(topk_shape, device=states.device),
                torch.zeros(topk_shape, device=states.device, dtype=torch.int32),
                None,
            )

        def make_topk_output(states, weights, ids, scores):
            del weights, ids, scores
            calls.append("topk")
            return states

        def routed_experts(**kwargs):
            calls.append("routed")
            self.assertIs(kwargs["hidden_states"], hidden_states)
            return hidden_states + 1

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
            experts=routed_experts,
            _select_experts=select_experts,
            _make_topk_output=make_topk_output,
        )
        return self._bind_deepseek_v4_moe_methods(moe)

    def test_deepseek_v4_moe_stream_fork_disabled_order(self):
        calls = []
        hidden_states = torch.ones(2, 3)
        input_ids = torch.arange(2)
        moe = self._make_fake_deepseek_v4_moe(
            hidden_states, input_ids, StreamFork(None), calls
        )

        actual = DeepseekV4MoE.forward(
            moe,
            hidden_states,
            input_ids,
            num_global_tokens=2,
            max_num_tokens_per_gpu=2,
        )

        self.assertEqual(calls, ["select", "topk", "routed", "shared"])
        self.assertTrue(
            torch.equal(actual, (hidden_states + 1) * 2 + hidden_states + 3)
        )

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
        def select_experts(states, ids):
            calls.append("select")
            self.assertIs(states, hidden_states)
            self.assertIs(ids, input_ids)
            topk_shape = (states.shape[0], 2)
            return (
                torch.ones(topk_shape, device=states.device),
                torch.zeros(topk_shape, device=states.device, dtype=torch.int32),
                None,
            )

        def routed_experts(states, topk_weights, topk_ids, activation_clamp=None):
            del topk_weights, activation_clamp
            calls.append("routed")
            self.assertIs(states, hidden_states)
            self.assertEqual(topk_ids.dtype, torch.int64)
            return hidden_states + 1

        moe = SimpleNamespace(
            use_mega_moe=True,
            config=SimpleNamespace(num_experts_per_tok=2),
            n_shared_experts=1,
            shared_experts=shared_experts,
            stream_fork=StreamFork(None),
            routed_scaling_factor=1.0,
            experts=routed_experts,
            _select_experts=select_experts,
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

        self.assertEqual(calls, ["select", "routed", "shared"])
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

        self.assertEqual(calls, ["select", "routed", "shared"])
        self.assertEqual(comm_calls, [("pre", ctx), ("post", ctx)])
        self.assertTrue(torch.equal(actual, hidden_states + 1 + hidden_states + 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_moe_stream_fork_aux_path_matches_serial(self):
        calls = []
        hidden_states = torch.ones(2, 3, device="cuda")
        input_ids = torch.arange(2, device="cuda")
        moe = self._make_fake_deepseek_v4_moe(
            hidden_states, input_ids, StreamFork(torch.cuda.Stream()), calls
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

        self.assertEqual(calls, ["select", "topk", "routed", "shared"])
        self.assertTrue(
            torch.equal(actual, (hidden_states + 1) * 2 + hidden_states + 3)
        )

    def test_cuda_graph_group_table_padding_uses_invalid_page_rows(self):
        table = torch.tensor([[5, -1]], dtype=torch.int32)
        padded = CudaGraphWrapper._pad_block_tables_to_padded_bs(
            {"v4.swa": table},
            actual_bs=1,
            padded_bs=3,
        )

        self.assertEqual(padded["v4.swa"].tolist(), [[5, -1], [-1, -1], [-1, -1]])

    def test_cuda_graph_replay_keeps_idle_actual_bs_with_padded_group_tables(self):
        captured = {}

        class FakeBackend:
            needs_group_block_tables = True
            uses_padded_decode_token_mask = True

            def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend()
        wrapper.draft_attn_backend = None
        wrapper.max_tokens_per_req = 1

        wrapper._init_replay_metadata(
            padded_bs=4,
            actual_bs=0,
            req_pool_indices=torch.zeros(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            group_placeholder_tables={
                "v4.swa": torch.zeros((4, 1), dtype=torch.int32),
            },
        )

        # padded_bs is the first positional arg.
        self.assertEqual(captured["args"][0], 4)
        self.assertEqual(captured["kwargs"]["actual_bs"], 0)
        self.assertEqual(
            captured["kwargs"]["block_tables"]["v4.swa"].shape,
            (4, 1),
        )

    def test_cuda_graph_replay_forwards_group_tables_to_draft_backend(self):
        captured = {"target": {}, "draft": {}}

        class FakeBackend:
            needs_group_block_tables = True
            uses_padded_decode_token_mask = False

            def __init__(self, key):
                self.key = key

            def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
                captured[self.key]["args"] = args
                captured[self.key]["kwargs"] = kwargs

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.drafter = SimpleNamespace(
            cache_view=SimpleNamespace(table=torch.zeros((1, 1), dtype=torch.int32)),
            draft_seq_lens_buf=torch.zeros(4, dtype=torch.int32),
        )
        wrapper.max_tokens_per_req = 4
        wrapper.use_v4_mtp_paged_metadata = True

        table = torch.tensor([[7], [8]], dtype=torch.int32)
        offsets = {"v4.swa": torch.tensor([1, 2], dtype=torch.int64)}
        wrapper._init_replay_metadata(
            padded_bs=4,
            actual_bs=2,
            req_pool_indices=torch.zeros(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            group_placeholder_tables={"v4.swa": table},
            block_table_base_offsets=offsets,
        )

        draft_kwargs = captured["draft"]["kwargs"]
        self.assertEqual(
            draft_kwargs["block_tables"]["v4.swa"].tolist(),
            [[7], [8], [-1], [-1]],
        )
        self.assertEqual(
            draft_kwargs["block_table_base_offsets"]["v4.swa"].tolist(),
            [1, 2, 0, 0],
        )
        self.assertEqual(draft_kwargs["forward_mode"], ForwardMode.DECODE)
        draft_seq_lens = captured["draft"]["args"][2]
        self.assertEqual(
            draft_seq_lens.data_ptr(),
            wrapper.drafter.draft_seq_lens_buf.data_ptr(),
        )
        self.assertEqual(wrapper.drafter.draft_seq_lens_buf.tolist(), [1, 1, 1, 1])

    def test_cuda_graph_mamba_verify_state_update_keeps_decode_mode_speculation(self):
        class BackendWithMambaUpdate:
            def update_mamba_state_after_mtp_verify(self, accepted_length, model):
                pass

        backend = BackendWithMambaUpdate()
        drafter = object()

        self.assertTrue(
            _should_update_mamba_state_after_mtp_verify(
                drafter, backend, ForwardMode.DECODE
            )
        )
        self.assertFalse(
            _should_update_mamba_state_after_mtp_verify(
                drafter, backend, ForwardMode.EXTEND
            )
        )
        self.assertFalse(
            _should_update_mamba_state_after_mtp_verify(
                None, backend, ForwardMode.DECODE
            )
        )
        self.assertFalse(
            _should_update_mamba_state_after_mtp_verify(
                drafter, object(), ForwardMode.DECODE
            )
        )

    def test_cuda_graph_eager_draft_prefill_uses_single_non_v4_metadata_init(self):
        captured = {"target": [], "draft": []}

        class FakeBackend:
            needs_group_block_tables = False

            def __init__(self, key):
                self.key = key

            def init_forward_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            cache_view=SimpleNamespace(table=torch.zeros((1, 1), dtype=torch.int32)),
            draft_seq_lens_buf=torch.tensor([0, 0], dtype=torch.int32),
        )
        wrapper.use_v4_mtp_paged_metadata = False

        seq_lens = torch.tensor([21, 22], dtype=torch.int32)
        wrapper._init_forward_metadata(
            padded_bs=2,
            num_extends=2,
            req_pool_indices=torch.zeros(2, dtype=torch.int32),
            seq_lens=seq_lens,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
        )

        self.assertEqual(len(captured["draft"]), 1)
        _, draft_kwargs = captured["draft"][0]
        self.assertEqual(draft_kwargs["forward_mode"], ForwardMode.EXTEND)
        self.assertEqual(
            draft_kwargs["seq_lens"].data_ptr(),
            wrapper.drafter.draft_seq_lens_buf.data_ptr(),
        )
        self.assertEqual(wrapper.drafter.draft_seq_lens_buf.tolist(), [21, 22])

    def test_cuda_graph_eager_v4_draft_uses_only_its_cache_group_subset(self):
        captured = {"target": [], "draft": []}

        class FakeBackend:
            needs_group_block_tables = True
            uses_cache_groups = True

            def __init__(self, key):
                self.key = key

            def init_forward_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

        target_cache_metadata = object()
        forward_batch = object()
        swa_table = torch.ones((1, 1), dtype=torch.int32)
        state_table = torch.full((1, 1), 2, dtype=torch.int32)

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            cache_view=SimpleNamespace(table=torch.zeros((1, 1), dtype=torch.int32)),
            draft_seq_lens_buf=torch.zeros(1, dtype=torch.int32),
        )
        wrapper.use_v4_mtp_paged_metadata = True
        wrapper.token_to_kv_pool = _fake_pool(
            SimpleNamespace(group_id="v4.swa_kv", family="history"),
            SimpleNamespace(group_id="v4.state", family="state"),
        )
        wrapper.draft_token_to_kv_pool = _fake_pool(
            SimpleNamespace(group_id="v4.swa_kv", family="history"),
        )

        wrapper._init_forward_metadata(
            padded_bs=1,
            num_extends=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.EXTEND,
            block_tables={"v4.swa_kv": swa_table, "v4.state": state_table},
            cache_metadata=target_cache_metadata,
            forward_batch=forward_batch,
        )

        self.assertIs(captured["target"][0][1]["cache_metadata"], target_cache_metadata)
        self.assertIs(captured["target"][0][1]["forward_batch"], forward_batch)
        self.assertEqual(len(captured["draft"]), 2)
        draft_prefill_kwargs = captured["draft"][0][1]
        self.assertNotIn("cache_metadata", draft_prefill_kwargs)
        self.assertNotIn("forward_batch", draft_prefill_kwargs)
        self.assertEqual(tuple(draft_prefill_kwargs["block_tables"]), ("v4.swa_kv",))
        self.assertIs(draft_prefill_kwargs["block_tables"]["v4.swa_kv"], swa_table)

    def test_cuda_graph_eager_draft_decode_preserves_non_v4_seq_lens_alias(self):
        captured = {"target": [], "draft": []}

        class FakeBackend:
            needs_group_block_tables = False

            def __init__(self, key):
                self.key = key

            def init_forward_metadata(self, *args, **kwargs):
                captured[self.key].append((args, kwargs))

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend("target")
        wrapper.draft_attn_backend = FakeBackend("draft")
        wrapper.max_tokens_per_req = 4
        wrapper.drafter = SimpleNamespace(
            cache_view=SimpleNamespace(table=torch.zeros((1, 1), dtype=torch.int32)),
            draft_seq_lens_buf=torch.tensor([11, 12], dtype=torch.int32),
        )

        seq_lens = torch.tensor([21, 22], dtype=torch.int32)
        wrapper.use_v4_mtp_paged_metadata = False
        wrapper._init_forward_metadata(
            padded_bs=2,
            num_extends=0,
            req_pool_indices=torch.zeros(2, dtype=torch.int32),
            seq_lens=seq_lens,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        _, non_v4_kwargs = captured["draft"][-1]
        self.assertEqual(
            non_v4_kwargs["seq_lens"].data_ptr(),
            wrapper.drafter.draft_seq_lens_buf.data_ptr(),
        )
        self.assertEqual(non_v4_kwargs["forward_mode"], ForwardMode.DECODE)

        wrapper.use_v4_mtp_paged_metadata = True
        wrapper._init_forward_metadata(
            padded_bs=2,
            num_extends=0,
            req_pool_indices=torch.zeros(2, dtype=torch.int32),
            seq_lens=seq_lens,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        _, v4_kwargs = captured["draft"][-1]
        self.assertEqual(v4_kwargs["seq_lens"].data_ptr(), seq_lens.data_ptr())
        self.assertEqual(v4_kwargs["forward_mode"], ForwardMode.DECODE)

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

    def test_deepseek_v4_indexer_prefill_max_logits_uses_server_arg(self):
        snapshot = dict(global_server_args_dict)
        try:
            global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"] = 7

            self.assertEqual(
                _deepseek_v4_indexer_prefill_max_logits_bytes(),
                7 * 1024 * 1024,
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
                load_format="auto",
                ext_yaml=None,
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
                "tokenspeed.runtime.models.deepseek_v4_dspark.count_dspark_stages",
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
        try:
            from tokenspeed_kernel.ops.attention.flash_mla import (
                flash_mla_sparse_fwd,
                flash_mla_with_kvcache,
                get_mla_metadata,
            )
            from tokenspeed_kernel.registry import error_fn
        except Exception as exc:  # noqa: BLE001 - optional kernel import
            self.skipTest(f"FlashMLA wrapper unavailable: {exc}")
        if (
            flash_mla_with_kvcache is error_fn
            or flash_mla_sparse_fwd is error_fn
            or get_mla_metadata is error_fn
        ):
            self.skipTest("FlashMLA wrapper unavailable on this platform")

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

        configure_deepseek_v4_attention(model_config)

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

        configure_deepseek_v4_attention(model_config)

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

    def test_deepseek_v4_mtp_checkpoint_name_remap(self):
        model = object.__new__(DeepseekV4ForCausalLMNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=43,
            num_nextn_predict_layers=1,
        )

        self.assertEqual(
            model._map_checkpoint_name("mtp.0.emb.tok_emb.weight"),
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            model._map_checkpoint_name("mtp.0.norm.weight"),
            "model.layers.43.shared_head.norm.weight",
        )
        self.assertEqual(
            model._map_checkpoint_name("mtp.0.attn.wq_a.weight"),
            "model.layers.43.mtp_block.attn.wq_a.weight",
        )
        self.assertEqual(
            model._map_checkpoint_name("mtp.0.ffn.experts.7.w1.scale"),
            "model.layers.43.mtp_block.ffn.experts.7.w1.weight_scale",
        )
        self.assertIsNone(model._map_checkpoint_name("mtp.0.head.weight"))
        self.assertIsNone(model._map_checkpoint_name("model.layers.43.head.weight"))
        self.assertIsNone(model._map_checkpoint_name("model.layers.1.attn.wq_a.weight"))

    def test_deepseek_v4_mtp_rejects_dspark_checkpoint_during_shard_filter(self):
        model = object.__new__(DeepseekV4ForCausalLMNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=43,
            num_nextn_predict_layers=1,
        )

        with self.assertRaisesRegex(
            ValueError,
            "MTP cannot load a DSpark checkpoint",
        ):
            model.checkpoint_weight_name_filter("mtp.0.main_proj.weight")

        self.assertTrue(model.checkpoint_weight_name_filter("mtp.0.e_proj.weight"))

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
                "huggingface_hub.hf_hub_download",
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
            "huggingface_hub.hf_hub_download",
            side_effect=OSError("offline"),
        ):
            self.assertIsNone(count_dspark_stages("public/model"))

    def test_dspark_checkpoint_name_mapping_is_stage_stable(self):
        model = object.__new__(DeepseekV4ForCausalLMDSpark)
        model.model = SimpleNamespace(num_stages=3)

        self.assertEqual(
            model._map_checkpoint_name("mtp.0.main_proj.scale"),
            "model.stages.0.main_proj.weight_scale_inv",
        )
        self.assertEqual(
            model._map_checkpoint_name("mtp.2.markov_head.markov_w1.weight"),
            "model.markov_embedding.weight",
        )
        self.assertIsNone(model._map_checkpoint_name("mtp.3.norm.weight"))

    def test_target_expert_scale_mapping_follows_expert_format(self):
        model = object.__new__(DeepseekV4ForCausalLM)
        non_mega = SimpleNamespace(is_mega_moe=lambda: False)
        with patch(
            "tokenspeed.runtime.models.deepseek_v4.get_moe_backend",
            return_value=non_mega,
        ):
            model.config = SimpleNamespace(expert_dtype="fp4")
            self.assertEqual(
                model._map_weight_name("layers.1.ffn.experts.7.w1.scale"),
                "model.layers.1.ffn.experts.7.w1.weight_scale",
            )
            model.config = SimpleNamespace(expert_dtype="fp8")
            self.assertEqual(
                model._map_weight_name("layers.1.ffn.experts.7.w1.scale"),
                "model.layers.1.ffn.experts.7.w1.weight_scale_inv",
            )

    def test_dspark_expert_scale_mapping_follows_expert_format(self):
        model = object.__new__(DeepseekV4ForCausalLMDSpark)
        model.model = SimpleNamespace(num_stages=3)

        mega = SimpleNamespace(is_mega_moe=lambda: True)
        non_mega = SimpleNamespace(is_mega_moe=lambda: False)
        model.config = SimpleNamespace(expert_dtype="fp4")
        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark.get_moe_backend",
            return_value=non_mega,
        ):
            self.assertEqual(
                model._map_checkpoint_name("mtp.1.ffn.experts.7.w1.scale"),
                "model.stages.1.block.ffn.experts.7.w1.weight_scale",
            )

        model.config = SimpleNamespace(expert_dtype="fp8")
        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark.get_moe_backend",
            return_value=mega,
        ):
            self.assertEqual(
                model._map_checkpoint_name("mtp.1.ffn.experts.7.w1.scale"),
                "model.stages.1.block.ffn.experts.7.w1.weight_scale",
            )
        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark.get_moe_backend",
            return_value=non_mega,
        ):
            self.assertEqual(
                model._map_checkpoint_name("mtp.1.ffn.experts.7.w1.scale"),
                "model.stages.1.block.ffn.experts.7.w1.weight_scale_inv",
            )

    def test_dspark_attention_contract_uses_checkpoint_namespace(self):
        self.assertIn("attn.attn_sink", _ATTENTION_CHECKPOINT_TENSORS)
        self.assertIn("attn.wq_a.weight", _ATTENTION_CHECKPOINT_TENSORS)
        self.assertIn("attn.wo_b.scale", _ATTENTION_CHECKPOINT_TENSORS)
        self.assertNotIn("wq_a.weight", _ATTENTION_CHECKPOINT_TENSORS)

    def test_dspark_only_allows_implicit_zero_expert_bias_parameters(self):
        self.assertTrue(
            _is_zero_initialized_expert_bias(
                "model.stages.0.block.ffn.experts.w13_weight_bias"
            )
        )
        self.assertTrue(
            _is_zero_initialized_expert_bias(
                "model.stages.2.block.ffn.experts.w2_weight_bias"
            )
        )
        self.assertFalse(
            _is_zero_initialized_expert_bias(
                "model.stages.0.block.ffn.experts.w13_weight"
            )
        )

    def test_dspark_target_capture_layers_fail_closed(self):
        model = object.__new__(DeepseekV4Model)
        model.layers = [SimpleNamespace(layer_id=i) for i in range(4)]

        model.set_dspark_layers_to_capture([1, 3])
        self.assertEqual(model.dspark_layers_to_capture, (1, 3))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model.set_dspark_layers_to_capture([3, 1])
        with self.assertRaisesRegex(ValueError, "unique"):
            model.set_dspark_layers_to_capture([1, 1])
        with self.assertRaisesRegex(ValueError, "out of range"):
            model.set_dspark_layers_to_capture([1, 4])

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

    def test_dspark_hc_head_preserves_batch_and_block_axes(self):
        batch_size, prefix_granularity, hc_mult, hidden_size = 2, 5, 4, 8
        hidden_states = torch.randn(
            batch_size,
            prefix_granularity,
            hc_mult,
            hidden_size,
        )
        hc_fn = torch.randn(hc_mult, hc_mult * hidden_size)
        hc_scale = torch.randn(1)
        hc_base = torch.randn(hc_mult)

        actual = _apply_dspark_hc_head(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
        )
        expected = torch.stack(
            [
                hc_head(
                    hidden_states[:, step],
                    hc_fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                )
                for step in range(prefix_granularity)
            ],
            dim=1,
        )

        self.assertEqual(
            tuple(actual.shape), (batch_size, prefix_granularity, hidden_size)
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_dspark_context_projection_materializes_strided_decode_slice(self):
        captured_hidden_states = torch.arange(48).reshape(2, 3, 8)[:, :2]
        self.assertFalse(captured_hidden_states.is_contiguous())

        def assert_contiguous(input_):
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(torch.equal(input_, captured_hidden_states))
            raise RuntimeError("projection reached")

        model = SimpleNamespace(
            stages=[SimpleNamespace(main_proj=assert_contiguous)],
        )
        with self.assertRaisesRegex(RuntimeError, "projection reached"):
            DeepseekV4DSparkModel.write_context_windows_batched(
                model,
                captured_hidden_states,
                torch.zeros((2, 2), dtype=torch.int64),
                torch.zeros(2, dtype=torch.int64),
                torch.ones((2, 2), dtype=torch.bool),
                torch.empty(0),
                0,
            )

    def test_dspark_tp_argmax_uses_contiguous_full_workspace_for_partial_batch(self):
        local_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        lm_head = SimpleNamespace(
            shard_indices=SimpleNamespace(
                num_org_elements=4,
                num_org_elements_padded=4,
                num_added_elements=0,
                org_vocab_start_index=0,
                added_vocab_start_index=4,
            )
        )
        gathered_values = torch.empty(4, 8)
        gathered_ids = torch.empty(4, 8, dtype=torch.int64)

        def fake_all_gather(output, input_, group):
            self.assertTrue(output.is_contiguous())
            self.assertEqual(output.numel(), 4)
            output.copy_(input_.repeat(4))

        with patch(
            "tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads.all_gather_into_tensor",
            side_effect=fake_all_gather,
        ) as gather:
            token_ids = _local_vocab_argmax(
                local_logits,
                lm_head,
                object(),
                gathered_values,
                gathered_ids,
            )

        self.assertEqual(gather.call_count, 2)
        self.assertTrue(torch.equal(token_ids, torch.tensor([3], dtype=torch.int32)))

        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            _local_vocab_argmax(
                local_logits,
                lm_head,
                object(),
                gathered_values[:, :1],
                gathered_ids[:, :1],
            )

    def test_dspark_tp_only_contract_uses_resolved_mapping(self):
        mapping = SimpleNamespace(attn=SimpleNamespace(dp_size=1, cp_size=1))
        DeepseekV4DSpark._validate_tp_only_mapping(mapping)

        for field in ("dp_size", "cp_size"):
            invalid = SimpleNamespace(attn=SimpleNamespace(dp_size=1, cp_size=1))
            setattr(invalid.attn, field, 2)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "tensor parallelism only"),
            ):
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

    def test_dspark_base_logits_use_public_fp32_head_math(self):
        model = object.__new__(DeepseekV4DSparkModel)
        hidden = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
        head = SimpleNamespace(
            weight=torch.tensor([[0.5, 0.25], [-1.0, 2.0]], dtype=torch.bfloat16)
        )

        logits = model.local_base_logits(hidden, head)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.equal(logits, hidden.float() @ head.weight.float().T))

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

    def test_dspark_same_checkpoint_rejects_kvstore(self):
        with self.assertRaisesRegex(
            ValueError,
            "does not support KVStore",
        ):
            ServerArgs(
                model="unused",
                speculative_config=json.dumps(
                    {"method": "dspark", "num_speculative_tokens": 5}
                ),
            )

    def test_dspark_explicit_same_checkpoint_rejects_kvstore(self):
        with self.assertRaisesRegex(ValueError, "does not support KVStore"):
            ServerArgs(
                model="same-checkpoint",
                speculative_config=json.dumps(
                    {
                        "method": "dspark",
                        "model": "same-checkpoint",
                        "num_speculative_tokens": 5,
                    }
                ),
            )

    def test_dspark_explicit_redirected_same_checkpoint_rejects_kvstore(self):
        with (
            patch(
                "tokenspeed.runtime.utils.server_args.maybe_model_redirect",
                side_effect=lambda model: (
                    "resolved-checkpoint" if model == "model-alias" else model
                ),
            ),
            self.assertRaisesRegex(ValueError, "does not support KVStore"),
        ):
            ServerArgs(
                model="model-alias",
                speculative_algorithm="DSPARK",
                speculative_draft_model_path="model-alias",
                speculative_num_steps=5,
            )

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

    def test_dspark_same_checkpoint_decode_still_rejects_kvstore(self):
        with self.assertRaisesRegex(ValueError, "does not support KVStore"):
            ServerArgs(
                model="same-checkpoint",
                enable_prefix_caching=False,
                disaggregation_mode="decode",
                speculative_algorithm="DSPARK",
                speculative_draft_model_path="same-checkpoint",
                speculative_num_steps=5,
            )

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

    def test_deepseek_v4_rope_config_matches_layer_type(self):
        config = SimpleNamespace(
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )

        swa_base, swa_scaling = deepseek_v4_rope_config(config, compress_ratio=1)
        csa_base, csa_scaling = deepseek_v4_rope_config(config, compress_ratio=4)

        self.assertEqual(swa_base, 10000.0)
        self.assertIsNone(swa_scaling)
        self.assertEqual(csa_base, 160000.0)
        self.assertIsNot(csa_scaling, config.rope_scaling)
        self.assertEqual(csa_scaling["rope_type"], "deepseek_yarn")
        self.assertEqual(csa_scaling["factor"], 16)
        self.assertEqual(csa_scaling["mscale"], 0)
        self.assertEqual(csa_scaling["mscale_all_dim"], 0)

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
            (plan.group("v4.c4a.compressed_kv").page_count, 64 * 68),
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
            (plan.group("v4.c4a.compressed_kv").page_count, 64 * 68),
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
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )
        cache_layout = _v4_layout(config)[2]

        for capacity in (0, -1):
            with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                cache_layout.bind(capacity)

    def test_deepseek_v4_group_slot_mapping_consumes_compact_base_offsets(self):
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([128, 129, 192, 64], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            rows_per_page=64,
            base_offsets=torch.tensor([2, 1], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(slots, torch.tensor([640, 641, -1, 1280])))

    def test_deepseek_v4_group_slot_mapping_expands_per_request_indices(self):
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([0, 1, 2, 64, 65, 66], dtype=torch.int64),
            req_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            rows_per_page=64,
            base_offsets=torch.tensor([0, 1], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(slots, torch.tensor([640, 641, 642, 1280, 1281, 1282]))
        )

    def test_deepseek_v4_backend_preserves_compact_cache_contract(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )
        compact = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        base = torch.tensor([2, 1], dtype=torch.int32)

        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            block_tables={"v4.swa_kv": compact},
            block_table_base_offsets={"v4.swa_kv": base},
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(torch.equal(metadata.cache.swa_page_table, compact))
        self.assertTrue(torch.equal(metadata.cache.swa_base_logical_page, base))

    def test_deepseek_v4_lcm_graph_tables_keep_absolute_logical_positions(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=4,
                head_dim=512,
                context_len=4096,
                sliding_window_tokens=128,
            )
        )
        backend.prefix_granularity = 256
        specs = _v4_spec_set(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(4, 128),
        )
        widths = {
            spec.group_id: backend._cuda_graph_group_table_width(
                spec,
                max_tokens_per_req=4,
                overlap_schedule_depth=1,
            )
            for spec in specs
        }

        # Scheduler tables retain absolute CacheBlock positions and represent
        # expired sliding entries as holes. Graph buffers must therefore cover
        # the full request extent at each group's own CacheBlock granularity.
        self.assertEqual(widths[V4_SWA_KV_GROUP_ID], 65)
        self.assertEqual(widths[v4_compressor_state_group_id(4)], 1026)
        self.assertEqual(widths[v4_compressor_state_group_id(128)], 513)
        self.assertEqual(widths[V4_INDEXER_COMPRESSOR_STATE_GROUP_ID], 1026)
        self.assertEqual(widths[v4_compressed_kv_group_id(4)], 17)
        self.assertEqual(widths[v4_compressed_kv_group_id(128)], 17)

    def test_deepseek_v4_lcm_tables_are_already_kernel_ready(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
                sliding_window_tokens=128,
            )
        )
        table = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)
        backend._expected_cache_group_ids = (V4_SWA_KV_GROUP_ID,)
        backend._cache_group_raw_tokens_per_page = {V4_SWA_KV_GROUP_ID: 64}
        backend._cache_group_max_page_ids = {V4_SWA_KV_GROUP_ID: 100}

        materialized = backend._prepare_cache_group_tables(
            {V4_SWA_KV_GROUP_ID: table},
            bs=1,
            actual_bs=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            device=torch.device("cpu"),
            phase="test",
        )

        self.assertEqual(materialized[V4_SWA_KV_GROUP_ID].tolist(), table.tolist())

    def _make_deepseek_v4_cache_group_contract_backend(self):
        backend = DeepseekV4AttentionBackend.__new__(DeepseekV4AttentionBackend)
        backend._expected_cache_group_ids = ("fine", "coarse")
        backend._cache_group_raw_tokens_per_page = {"fine": 4, "coarse": 256}
        backend._cache_group_max_page_ids = {"fine": 20000, "coarse": 1024}
        return backend

    def test_deepseek_v4_runtime_configures_eager_cache_group_contract(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )
        specs = (
            CacheGroupSpec(
                group_id="fine",
                retention="full_history",
                rows_per_page=4,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
            ),
            CacheGroupSpec(
                group_id="coarse",
                retention="full_history",
                rows_per_page=256,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
            ),
        )
        counts = {"fine": 20001, "coarse": 1025}

        backend.configure_runtime(
            cache_group_specs=specs,
            cache_group_page_counts=counts,
        )
        tables = {
            "fine": torch.ones((1, 1), dtype=torch.int32),
            "coarse": torch.ones((1, 1), dtype=torch.int32),
        }
        materialized = backend._prepare_cache_group_tables(
            tables,
            bs=1,
            actual_bs=1,
            seq_lens=torch.ones(1, dtype=torch.int32),
            device=torch.device("cpu"),
            phase="eager",
        )
        self.assertEqual(tuple(materialized), ("fine", "coarse"))

        # Graph setup may repeat the same contract but must not replace it.
        backend.init_cuda_graph_state(
            max_bs=1,
            cache_group_specs=specs,
            cache_group_page_counts=counts,
        )
        changed = {"fine": 20002, "coarse": 1025}
        with self.assertRaisesRegex(RuntimeError, "changed after initialization"):
            backend._configure_cache_group_contract(specs, changed)

    def test_deepseek_v4_cache_group_contract_covers_64k_boundary(self):
        backend = self._make_deepseek_v4_cache_group_contract_backend()
        tables = {
            "fine": torch.ones((1, 16384), dtype=torch.int32),
            "coarse": torch.ones((1, 256), dtype=torch.int32),
        }

        materialized = backend._prepare_cache_group_tables(
            tables,
            bs=1,
            actual_bs=1,
            seq_lens=torch.tensor([65536], dtype=torch.int32),
            device=torch.device("cpu"),
            phase="test",
        )
        self.assertEqual(tuple(materialized), ("fine", "coarse"))

        with self.assertRaisesRegex(RuntimeError, "missing a real page"):
            backend._prepare_cache_group_tables(
                tables,
                bs=1,
                actual_bs=1,
                seq_lens=torch.tensor([65537], dtype=torch.int32),
                device=torch.device("cpu"),
                phase="test",
            )

    def test_deepseek_v4_cache_group_contract_fails_closed(self):
        backend = self._make_deepseek_v4_cache_group_contract_backend()
        valid = {
            "fine": torch.ones((2, 2), dtype=torch.int32),
            "coarse": torch.ones((2, 2), dtype=torch.int32),
        }
        malformed = []
        malformed.append(({"fine": valid["fine"]}, "group mismatch"))
        malformed.append(
            (
                {
                    "fine": valid["fine"],
                    "coarse": valid["coarse"],
                    "extra": valid["fine"],
                },
                "group mismatch",
            )
        )
        malformed.append(
            (
                {"coarse": valid["coarse"], "fine": valid["fine"]},
                "wrong order",
            )
        )
        malformed.append(
            (
                {**valid, "fine": valid["fine"].to(torch.int64)},
                "torch.int32",
            )
        )
        malformed.append(({**valid, "fine": valid["fine"][0]}, "rank 2"))
        malformed.append(({**valid, "fine": valid["fine"][:1]}, "rows"))
        malformed.append(
            (
                {**valid, "fine": torch.empty((2, 0), dtype=torch.int32)},
                "zero width",
            )
        )

        for tables, error in malformed:
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                backend._prepare_cache_group_tables(
                    tables,
                    bs=2,
                    actual_bs=2,
                    seq_lens=torch.ones(2, dtype=torch.int32),
                    device=torch.device("cpu"),
                    phase="test",
                )

        null_active = {**valid, "fine": valid["fine"].clone()}
        null_active["fine"][0, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "missing a real page"):
            backend._prepare_cache_group_tables(
                null_active,
                bs=2,
                actual_bs=2,
                seq_lens=torch.ones(2, dtype=torch.int32),
                device=torch.device("cpu"),
                phase="test",
            )

        # A zero-token idle row has no active logical page and may keep the
        # reserved null entry. Negative lengths are always malformed.
        idle = {key: torch.zeros_like(table) for key, table in valid.items()}
        backend._prepare_cache_group_tables(
            idle,
            bs=2,
            actual_bs=2,
            seq_lens=torch.zeros(2, dtype=torch.int32),
            device=torch.device("cpu"),
            phase="idle",
        )
        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            backend._prepare_cache_group_tables(
                idle,
                bs=2,
                actual_bs=1,
                seq_lens=torch.tensor([-1, 0], dtype=torch.int32),
                device=torch.device("cpu"),
                phase="test",
            )

    def test_deepseek_v4_cache_group_replay_refreshes_only_live_rows(self):
        backend = self._make_deepseek_v4_cache_group_contract_backend()
        output_buffers = {
            "fine": torch.zeros((2, 4), dtype=torch.int32),
            "coarse": torch.zeros((2, 4), dtype=torch.int32),
        }

        for step in range(1, 11):
            tables = {
                "fine": torch.tensor([[step], [0]], dtype=torch.int32),
                "coarse": torch.tensor([[step], [0]], dtype=torch.int32),
            }
            materialized = backend._prepare_cache_group_tables(
                tables,
                bs=2,
                actual_bs=1,
                seq_lens=torch.ones(2, dtype=torch.int32),
                device=torch.device("cpu"),
                phase="replay",
                output_buffers=output_buffers,
            )
            for table in materialized.values():
                self.assertEqual(int(table[0, 0]), step)
                self.assertTrue(bool((table[:, 1:] == -1).all()))

        zero_tables = {
            "fine": torch.zeros((2, 1), dtype=torch.int32),
            "coarse": torch.zeros((2, 1), dtype=torch.int32),
        }
        backend._prepare_cache_group_tables(
            zero_tables,
            bs=2,
            actual_bs=0,
            seq_lens=torch.ones(2, dtype=torch.int32),
            device=torch.device("cpu"),
            phase="idle",
            output_buffers=output_buffers,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_deepseek_v4_target_mtp_graph_replay_refreshes_flat_tables(self):
        device = torch.device("cuda")
        target_specs = tuple(
            _v4_spec_set(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4, 128),
            )
        )
        draft_specs = tuple(
            _v4_spec_set(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1,),
            )
        )
        target_counts = {str(spec.group_id): 4096 for spec in target_specs}
        draft_counts = {str(spec.group_id): 4096 for spec in draft_specs}

        def make_backend(*, is_draft: bool) -> DeepseekV4AttentionBackend:
            return DeepseekV4AttentionBackend(
                SimpleNamespace(
                    prefix_granularity=64,
                    kernel_page_size=64,
                    device="cuda",
                    num_attention_heads=64,
                    num_kv_heads=1,
                    attn_tp_size=1,
                    dtype=torch.bfloat16,
                    is_draft=is_draft,
                    speculative_num_draft_tokens=4,
                    speculative_num_steps=1,
                    head_dim=512,
                    qk_rope_head_dim=64,
                    context_len=65536,
                )
            )

        target = make_backend(is_draft=False)
        draft = make_backend(is_draft=True)
        target.init_cuda_graph_state(
            max_bs=2,
            cache_group_specs=target_specs,
            cache_group_page_counts=target_counts,
            max_tokens_per_req=4,
        )
        draft.init_cuda_graph_state(
            max_bs=2,
            cache_group_specs=draft_specs,
            cache_group_page_counts=draft_counts,
            max_tokens_per_req=4,
        )

        common = {
            "bs": 2,
            "req_pool_indices": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "seq_lens": torch.tensor([65536, 0], dtype=torch.int32, device=device),
            "forward_mode": ForwardMode.DECODE,
            "num_tokens": 8,
        }
        target_capture = {
            group_id: torch.zeros_like(table)
            for group_id, table in target._cuda_graph_block_tables.items()
        }
        draft_capture = {
            group_id: torch.zeros_like(table)
            for group_id, table in draft._cuda_graph_block_tables.items()
        }
        target.init_forward_metadata_capture_cuda_graph(
            **common,
            block_tables=target_capture,
        )
        draft.init_forward_metadata_capture_cuda_graph(
            **common,
            block_tables=draft_capture,
        )
        target_metadata = target.forward_metadata
        draft_metadata = draft.forward_metadata
        assert target_metadata is not None
        assert draft_metadata is not None

        target_ids = tuple(target._expected_cache_group_ids or ())
        draft_ids = tuple(draft._expected_cache_group_ids or ())
        observed = torch.empty(
            len(target_ids) + len(draft_ids), dtype=torch.int32, device=device
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for index, group_id in enumerate(target_ids):
                observed[index].copy_(
                    target_metadata.cache.block_tables[group_id][0, 0]
                )
            for index, group_id in enumerate(draft_ids, start=len(target_ids)):
                observed[index].copy_(draft_metadata.cache.block_tables[group_id][0, 0])

        target_live = {
            group_id: torch.zeros_like(table)
            for group_id, table in target_capture.items()
        }
        draft_live = {group_id: target_live[group_id] for group_id in draft_ids}

        def replay(step: int) -> None:
            for index, table in enumerate(target_live.values()):
                table[0].fill_(step + index + 1)
                table[1].zero_()
            target.init_forward_metadata_replay_cuda_graph(
                **common,
                actual_bs=1,
                block_tables=target_live,
            )
            draft.init_forward_metadata_replay_cuda_graph(
                **common,
                actual_bs=1,
                block_tables=draft_live,
            )
            graph.replay()

        for step in range(10):
            replay(step)
            torch.cuda.synchronize()
            target_expected = [step + index + 1 for index in range(len(target_ids))]
            draft_expected = [
                step + target_ids.index(group_id) + 1 for group_id in draft_ids
            ]
            self.assertEqual(observed.cpu().tolist(), target_expected + draft_expected)

        self.assertEqual(
            target.forward_metadata.is_valid_token.tolist(),
            [True, True, True, True, False, False, False, False],
        )
        self.assertEqual(
            draft.forward_metadata.is_valid_token.tolist(),
            [True, True, True, True, False, False, False, False],
        )
        base_group_id = v4_compressed_kv_group_id(4)
        base_value = 9 + target_ids.index(base_group_id) + 1
        base_width = target_live[base_group_id].shape[1]
        self.assertTrue(
            bool(
                (
                    target.forward_metadata.cache.page_table[0, :base_width]
                    == base_value
                ).all()
            )
        )
        self.assertTrue(
            bool((target.forward_metadata.cache.page_table[0, base_width:] == 0).all())
        )
        self.assertTrue(bool((target.forward_metadata.cache.page_table[1] == 0).all()))
        self.assertTrue(bool((draft.forward_metadata.cache.page_table == 0).all()))

        torch.cuda.synchronize()
        reserved_before = torch.cuda.memory_reserved()
        for step in range(50):
            replay(step + 100)
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.memory_reserved(), reserved_before)

    def test_deepseek_v4_mixed_metadata_keeps_decode_rows_single_token(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )

        backend.init_forward_metadata(
            bs=3,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([7, 10, 4], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            page_table=torch.zeros((3, 1), dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
            num_extends=1,
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.query_lens.tolist(), [7, 1, 1])
        self.assertEqual(metadata.query_lens_cpu.tolist(), [7, 1, 1])
        self.assertEqual(metadata.num_prefill_reqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 7)
        self.assertEqual(metadata.decode_req_count(), 2)
        self.assertEqual(metadata.decode_token_count(), 2)
        self.assertEqual(
            metadata.token_to_req_indices.tolist(),
            [0, 0, 0, 0, 0, 0, 0, 1, 2],
        )

    def test_deepseek_v4_mixed_metadata_uses_runtime_verify_width(self):
        for verify_width in (1, 2, 4, 8):
            with self.subTest(verify_width=verify_width):
                backend = DeepseekV4AttentionBackend(
                    SimpleNamespace(
                        prefix_granularity=64,
                        kernel_page_size=64,
                        device="cpu",
                        num_attention_heads=64,
                        num_kv_heads=1,
                        attn_tp_size=1,
                        dtype=torch.bfloat16,
                        is_draft=False,
                        speculative_num_draft_tokens=verify_width,
                        head_dim=512,
                        context_len=16384,
                    )
                )
                prefill_tokens = 8192 - 2 * verify_width
                total_tokens = prefill_tokens + 2 * verify_width
                backend.init_forward_metadata(
                    bs=3,
                    num_tokens=total_tokens,
                    req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
                    seq_lens=torch.tensor(
                        [prefill_tokens, 100, 200], dtype=torch.int32
                    ),
                    forward_mode=ForwardMode.MIXED,
                    page_table=torch.zeros((3, 256), dtype=torch.int32),
                    extend_seq_lens_cpu=torch.tensor(
                        [prefill_tokens], dtype=torch.int32
                    ),
                    num_extends=1,
                )

                metadata = backend.forward_metadata
                self.assertIsNotNone(metadata)
                assert metadata is not None
                self.assertEqual(
                    metadata.query_lens.tolist(),
                    [prefill_tokens, verify_width, verify_width],
                )
                self.assertEqual(
                    metadata.query_lens_cpu.tolist(),
                    [prefill_tokens, verify_width, verify_width],
                )
                self.assertEqual(
                    metadata.query_start_loc.tolist(),
                    [
                        0,
                        prefill_tokens,
                        prefill_tokens + verify_width,
                        total_tokens,
                    ],
                )
                self.assertEqual(metadata.token_to_req_indices.numel(), total_tokens)
                self.assertEqual(
                    metadata.token_to_req_indices[-2 * verify_width :].tolist(),
                    [1] * verify_width + [2] * verify_width,
                )

    def test_deepseek_v4_mixed_metadata_rejects_packed_token_mismatch(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=4,
                head_dim=512,
                context_len=4096,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "mixed metadata token count mismatch",
        ):
            backend.init_forward_metadata(
                bs=2,
                num_tokens=10,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                seq_lens=torch.tensor([7, 20], dtype=torch.int32),
                forward_mode=ForwardMode.MIXED,
                page_table=torch.zeros((2, 64), dtype=torch.int32),
                extend_seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
                num_extends=1,
            )

    def test_deepseek_v4_prefill_metadata_uses_complete_cpu_mirrors(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )

        backend.init_forward_metadata(
            bs=2,
            num_tokens=14,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([17, 65], dtype=torch.int32),
            forward_mode=ForwardMode.EXTEND,
            page_table=torch.zeros((2, 2), dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([5, 9], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([12, 56], dtype=torch.int32),
            num_extends=2,
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.seq_lens_cpu.tolist(), [17, 65])
        self.assertEqual(metadata.query_lens_cpu.tolist(), [5, 9])
        self.assertEqual(metadata.num_prefill_reqs, 2)
        self.assertEqual(metadata.num_prefill_tokens, 14)
        self.assertEqual(metadata.token_to_req_indices.tolist(), [0] * 5 + [1] * 9)

    def test_deepseek_v4_prefill_metadata_requires_complete_cpu_mirrors(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "prefill metadata requires complete CPU sequence and query length mirrors",
        ):
            backend.init_forward_metadata(
                bs=1,
                num_tokens=3,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([5], dtype=torch.int32),
                forward_mode=ForwardMode.EXTEND,
                page_table=torch.zeros((1, 1), dtype=torch.int32),
                extend_prefix_lens_cpu=torch.tensor([2], dtype=torch.int32),
                num_extends=1,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "prefill metadata requires complete CPU sequence and query length mirrors",
        ):
            backend.init_forward_metadata(
                bs=1,
                num_tokens=3,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([131], dtype=torch.int32),
                forward_mode=ForwardMode.EXTEND,
                page_table=torch.zeros((1, 3), dtype=torch.int32),
                extend_seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
                num_extends=1,
            )

    def test_deepseek_v4_prefill_workspace_bounds_use_cpu_mirrors(self):
        self.assertEqual(
            DeepseekV4AttentionBackend._prefill_workspace_bounds(
                torch.tensor([17, 65], dtype=torch.int32),
                torch.tensor([5, 9], dtype=torch.int32),
                num_reqs=2,
                window_size=16,
                compress_ratio=4,
            ),
            (24, 16),
        )
        self.assertEqual(
            DeepseekV4AttentionBackend._prefill_workspace_bounds(
                torch.tensor([17], dtype=torch.int32),
                torch.tensor([5], dtype=torch.int32),
                num_reqs=1,
                window_size=16,
                compress_ratio=1,
            ),
            (17, 0),
        )
        self.assertEqual(
            DeepseekV4AttentionBackend._prefill_workspace_bounds(
                None,
                None,
                num_reqs=0,
                window_size=16,
                compress_ratio=4,
            ),
            (1, 0),
        )

    def test_deepseek_v4_prefill_workspace_bounds_fail_closed(self):
        invalid_cases = (
            (
                torch.tensor([17, 65], dtype=torch.int32),
                torch.tensor([5], dtype=torch.int32),
            ),
            (
                torch.tensor([4], dtype=torch.int32),
                torch.tensor([5], dtype=torch.int32),
            ),
        )
        for seq_lens_cpu, query_lens_cpu in invalid_cases:
            with (
                self.subTest(
                    seq_lens=seq_lens_cpu.tolist(),
                    query_lens=query_lens_cpu.tolist(),
                ),
                self.assertRaises(RuntimeError),
            ):
                DeepseekV4AttentionBackend._prefill_workspace_bounds(
                    seq_lens_cpu,
                    query_lens_cpu,
                    num_reqs=2 if seq_lens_cpu.numel() == 2 else 1,
                    window_size=16,
                    compress_ratio=4,
                )

        for seq_lens_cpu, query_lens_cpu in (
            (None, torch.tensor([5], dtype=torch.int32)),
            (torch.tensor([17], dtype=torch.int32), None),
            (
                torch.tensor([17, 65], dtype=torch.int32),
                torch.tensor([5, 9], dtype=torch.int32),
            ),
        ):
            with (
                self.subTest(
                    seq_lens_missing=seq_lens_cpu is None,
                    query_lens_missing=query_lens_cpu is None,
                    seq_lens_count=(
                        None if seq_lens_cpu is None else seq_lens_cpu.numel()
                    ),
                ),
                self.assertRaises(RuntimeError),
            ):
                DeepseekV4AttentionBackend._prefill_workspace_bounds(
                    seq_lens_cpu,
                    query_lens_cpu,
                    num_reqs=1,
                    window_size=16,
                    compress_ratio=4,
                )

        with self.assertRaisesRegex(ValueError, "num_reqs must be non-negative"):
            DeepseekV4AttentionBackend._prefill_workspace_bounds(
                None,
                None,
                num_reqs=-1,
                window_size=16,
                compress_ratio=4,
            )

    def test_deepseek_v4_chunked_prefill_uses_cpu_query_offsets(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )
        backend.prefill_chunk_size = 2
        backend.init_forward_metadata(
            bs=3,
            num_tokens=6,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([4, 7, 9], dtype=torch.int32),
            forward_mode=ForwardMode.EXTEND,
            page_table=torch.zeros((3, 1), dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([2, 1, 3], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([2, 6, 6], dtype=torch.int32),
            num_extends=3,
        )
        metadata = backend.forward_metadata
        assert metadata is not None
        calls = []

        def fake_prefill_chunk(**kwargs):
            chunk_metadata = backend.forward_metadata
            calls.append(
                (
                    kwargs["q"].shape[0],
                    kwargs["positions"].tolist(),
                    chunk_metadata.req_pool_indices.tolist(),
                    chunk_metadata.query_lens_cpu.tolist(),
                )
            )
            q = kwargs["q"]
            return q.new_zeros((q.shape[0], 1, 2))

        backend._forward_deepseek_v4_prefill_chunk = fake_prefill_chunk
        q = torch.zeros((6, 1, 2), dtype=torch.float32)
        common = {
            "q": q,
            "positions": torch.arange(6, dtype=torch.int32),
            "token_to_kv_pool": SimpleNamespace(),
            "layer_id": 0,
            "kind": "mla",
            "compress_ratio": 4,
            "num_local_heads": 1,
            "padded_heads": 1,
            "head_dim": 2,
            "window_size": 4,
            "softmax_scale": 1.0,
            "attn_sink": torch.zeros(1),
            "topk_indices": None,
        }
        out = backend.forward_deepseek_v4_prefill(**common)

        self.assertEqual(out.shape, (6, 1, 2))
        self.assertEqual(
            calls,
            [
                (3, [0, 1, 2], [0, 1], [2, 1]),
                (3, [3, 4, 5], [2], [3]),
            ],
        )

        metadata.query_lens_cpu = None
        with self.assertRaisesRegex(
            RuntimeError,
            "chunked prefill requires CPU query-length metadata",
        ):
            backend.forward_deepseek_v4_prefill(**common)

        metadata.query_lens_cpu = torch.tensor([2, 1, 2], dtype=torch.int32)
        with self.assertRaisesRegex(
            RuntimeError,
            "chunked prefill query lengths do not match token count",
        ):
            backend.forward_deepseek_v4_prefill(**common)

    def test_deepseek_v4_draft_keeps_mixed_step0_and_decode_step_metadata(self):
        verify_width = 4
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=True,
                speculative_num_draft_tokens=verify_width,
                head_dim=512,
                context_len=4096,
            )
        )
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        seq_lens = torch.tensor([7, 20], dtype=torch.int32)
        page_table = torch.zeros((2, 64), dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=7 + verify_width,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=ForwardMode.MIXED,
            page_table=page_table,
            extend_seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
            num_extends=1,
        )
        mixed_metadata = backend.forward_metadata
        self.assertIsNotNone(mixed_metadata)
        self.assertIs(backend.forward_prefill_metadata, mixed_metadata)

        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=page_table,
            num_extends=0,
        )
        decode_metadata = backend.forward_metadata
        self.assertIs(backend.forward_decode_metadata, decode_metadata)
        self.assertEqual(decode_metadata.query_lens.tolist(), [1, 1])

        mixed_ctx = SimpleNamespace(
            attn_backend=backend,
            forward_mode=ForwardMode.MIXED,
            input_num_tokens=7 + verify_width,
        )
        decode_ctx = SimpleNamespace(
            attn_backend=backend,
            forward_mode=ForwardMode.DECODE,
            input_num_tokens=2,
        )
        self.assertIs(_deepseek_v4_forward_metadata(mixed_ctx), mixed_metadata)
        self.assertIs(_deepseek_v4_forward_metadata(decode_ctx), decode_metadata)

    def test_deepseek_v4_cuda_graph_refresh_keeps_compact_table_columns(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )
        backend.init_cuda_graph_state(
            2,
            cache_group_specs=(
                CacheGroupSpec(
                    group_id="v4.swa_kv",
                    retention="sliding_window",
                    rows_per_page=64,
                    entry_stride_tokens=1,
                    sliding_window_tokens=128,
                ),
            ),
            cache_group_page_counts={"v4.swa_kv": 1024},
            max_tokens_per_req=1,
        )
        compact = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        refreshed = backend._refresh_cuda_graph_block_tables(
            2,
            {"v4.swa_kv": compact},
            pad_value=-1,
        )

        table = refreshed["v4.swa_kv"]
        self.assertTrue(torch.equal(table[:, :2], compact))
        self.assertTrue(torch.equal(table[:, 2:], torch.full_like(table[:, 2:], -1)))

    def test_deepseek_v4_metadata_splits_named_cache_groups(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )
        swa = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        c4_state = torch.tensor([[30], [40]], dtype=torch.int32)
        c128_state = torch.tensor([[50], [60]], dtype=torch.int32)
        indexer_state = torch.tensor([[70], [80]], dtype=torch.int32)
        c4_state_base = torch.tensor([3, 4], dtype=torch.int32)
        c128_state_base = torch.tensor([5, 6], dtype=torch.int32)
        indexer_state_base = torch.tensor([7, 8], dtype=torch.int32)

        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            block_tables={
                "v4.swa_kv": swa,
                "v4.c4a.compressor_state": c4_state,
                "v4.c128a.compressor_state": c128_state,
                "v4.c4a.indexer_compressor_state": indexer_state,
            },
            block_table_base_offsets={
                "v4.c4a.compressor_state": c4_state_base,
                "v4.c128a.compressor_state": c128_state_base,
                "v4.c4a.indexer_compressor_state": indexer_state_base,
            },
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        cache_metadata = metadata.cache
        self.assertTrue(torch.equal(cache_metadata.swa_page_table, swa))
        self.assertTrue(
            torch.equal(cache_metadata.compressor_state_block_tables[4], c4_state)
        )
        self.assertTrue(
            torch.equal(cache_metadata.compressor_state_block_tables[128], c128_state)
        )
        self.assertTrue(
            torch.equal(cache_metadata.indexer_state_block_table, indexer_state)
        )
        self.assertTrue(
            torch.equal(
                cache_metadata.compressor_state_base_logical_pages[4],
                c4_state_base,
            )
        )
        self.assertTrue(
            torch.equal(
                cache_metadata.compressor_state_base_logical_pages[128],
                c128_state_base,
            )
        )
        self.assertTrue(
            torch.equal(
                cache_metadata.indexer_state_base_logical_page,
                indexer_state_base,
            )
        )

    def test_deepseek_v4_metadata_slice_preserves_compact_base_offsets(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=4096,
            )
        )
        swa = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
        c4_state = torch.tensor([[40], [41], [42]], dtype=torch.int32)
        c128_state = torch.tensor([[50], [51], [52]], dtype=torch.int32)
        indexer_state = torch.tensor([[60], [61], [62]], dtype=torch.int32)
        raw_offsets = {
            "v4.swa_kv": torch.tensor([100, 200, 300], dtype=torch.int32),
            "v4.c4a.compressor_state": torch.tensor([400, 500, 600], dtype=torch.int32),
            "v4.c128a.compressor_state": torch.tensor(
                [700, 800, 900], dtype=torch.int32
            ),
            "v4.c4a.indexer_compressor_state": torch.tensor(
                [1000, 1100, 1200], dtype=torch.int32
            ),
        }
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=64,
            req_pool_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
            page_table=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
            query_lens=torch.tensor([2, 1, 3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32),
            block_tables={
                "v4.swa_kv": swa,
                "v4.c4a.compressor_state": c4_state,
                "v4.c128a.compressor_state": c128_state,
                "v4.c4a.indexer_compressor_state": indexer_state,
            },
            block_table_base_offsets=raw_offsets,
            swa_page_table=swa,
            swa_base_logical_page=raw_offsets["v4.swa_kv"],
            compressor_state_block_tables={4: c4_state, 128: c128_state},
            compressor_state_base_logical_pages={
                4: raw_offsets["v4.c4a.compressor_state"],
                128: raw_offsets["v4.c128a.compressor_state"],
            },
            indexer_state_block_table=indexer_state,
            indexer_state_base_logical_page=raw_offsets[
                "v4.c4a.indexer_compressor_state"
            ],
        )

        sliced = backend._metadata_slice(
            metadata,
            req_start=1,
            req_end=3,
            token_start=2,
            token_end=6,
            forward_mode=ForwardMode.EXTEND,
        )

        self.assertTrue(
            torch.equal(
                sliced.token_to_req_indices,
                torch.tensor([0, 1, 1, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.query_start_loc,
                torch.tensor([0, 1, 4], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(sliced.cache.swa_page_table, swa[1:3]))
        self.assertTrue(
            torch.equal(
                sliced.cache.swa_base_logical_page,
                raw_offsets["v4.swa_kv"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.cache.block_table_base_offsets["v4.swa_kv"],
                raw_offsets["v4.swa_kv"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.cache.compressor_state_base_logical_pages[4],
                raw_offsets["v4.c4a.compressor_state"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.cache.compressor_state_base_logical_pages[128],
                raw_offsets["v4.c128a.compressor_state"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.cache.indexer_state_base_logical_page,
                raw_offsets["v4.c4a.indexer_compressor_state"][1:3],
            )
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
                model_dtype=torch.bfloat16,
                layout=layout,
                layer_num=2,
                rank=0,
            )

    def test_deepseek_v4_metadata_maps_compressed_slots(self):
        compressed_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.tensor([[0, 1], [3, 4]], dtype=torch.int32),
            seq_lens=torch.tensor([70, 5], dtype=torch.int32),
            query_lens=torch.tensor([3, 5], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
            token_to_req_indices=torch.tensor(
                [0, 0, 0, 1, 1, 1, 1, 1],
                dtype=torch.int32,
            ),
            block_tables={"v4.c4a.compressed_kv": compressed_table},
        )

        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(metadata.cache.compressed_page_table(4), compressed_table)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "missing paged-cache block table",
        ):
            metadata.cache.compressed_page_table(128)

        slots = metadata.cache.compressed_slot_mapping(
            torch.tensor([3, 7, 127], dtype=torch.int64),
            compress_ratio=4,
            token_to_req_indices=metadata.token_to_req_indices,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([640, 641, 671])))
        masked_slots = metadata.cache.compressed_slot_mapping(
            torch.tensor([3, 7, 127], dtype=torch.int64),
            compress_ratio=4,
            token_to_req_indices=metadata.token_to_req_indices,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            is_valid_token=torch.tensor([True, False, True], dtype=torch.bool),
        )
        self.assertTrue(torch.equal(masked_slots, torch.tensor([640, -1, 671])))

        page256_metadata = _make_deepseek_v4_forward_metadata(
            page_size=256,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            page_table=torch.tensor([[5, 6]], dtype=torch.int32),
            seq_lens=torch.tensor([300], dtype=torch.int32),
            query_lens=torch.tensor([3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0], dtype=torch.int32),
            block_tables={
                "v4.c4a.compressed_kv": torch.tensor([[5, 6]], dtype=torch.int32),
            },
        )
        slots = page256_metadata.cache.compressed_slot_mapping(
            torch.tensor([255, 256, 511], dtype=torch.int64),
            compress_ratio=4,
            token_to_req_indices=page256_metadata.token_to_req_indices,
            query_start_loc=page256_metadata.query_start_loc,
            seq_lens=page256_metadata.seq_lens,
            kv_cache_block_size=64,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([383, -1, 447])))

        grouped_metadata = _make_deepseek_v4_forward_metadata(
            page_size=256,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
            seq_lens=torch.tensor([300, 10], dtype=torch.int32),
            query_lens=torch.tensor([3, 2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
            block_tables={
                "v4.c4a.compressed_kv": torch.tensor(
                    [[20, 21], [30, -1]], dtype=torch.int32
                )
            },
        )
        slots = grouped_metadata.cache.compressed_slot_mapping(
            torch.tensor([255, 256, 511, 2560, 4], dtype=torch.int64),
            compress_ratio=4,
            token_to_req_indices=grouped_metadata.token_to_req_indices,
            query_start_loc=grouped_metadata.query_start_loc,
            seq_lens=grouped_metadata.seq_lens,
            kv_cache_block_size=64,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([1343, -1, 1407, -1, -1])))

        decode_slots = grouped_metadata.cache._update_decode_compressed_slot_mapping(
            token_to_req_indices=grouped_metadata.token_to_req_indices,
            query_start_loc=grouped_metadata.query_start_loc,
            seq_lens=grouped_metadata.seq_lens,
            compress_ratio=4,
            kv_cache_block_size=64,
        )
        self.assertTrue(
            torch.equal(decode_slots[:5], torch.tensor([-1, -1, 1354, -1, -1]))
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

    def test_deepseek_v4_mixed_metadata_splits_prefill_and_decode(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=576,
                context_len=256,
            )
        )
        backend.configure_runtime(
            cache_group_specs=(
                SimpleNamespace(
                    group_id=V4_SWA_KV_GROUP_ID,
                    retention="sliding_window",
                    rows_per_page=64,
                    entry_stride_tokens=1,
                ),
            ),
            cache_group_page_counts={V4_SWA_KV_GROUP_ID: 128},
        )
        backend.init_forward_metadata(
            bs=3,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            block_tables={
                V4_SWA_KV_GROUP_ID: torch.tensor([[10], [20], [30]], dtype=torch.int32)
            },
            extend_seq_lens_cpu=torch.tensor([3, 1, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([2, 8, 11], dtype=torch.int32),
            num_extends=1,
        )
        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.num_prefill_reqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 3)
        self.assertEqual(metadata.decode_req_count(), 2)
        self.assertEqual(metadata.decode_token_count(), 2)
        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0, 0, 0, 1, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.seq_lens_cpu,
                torch.tensor([5, 9, 12], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.query_lens_cpu,
                torch.tensor([3, 1, 1], dtype=torch.int32),
            )
        )

        prefill = backend._metadata_slice(
            metadata,
            req_start=0,
            req_end=1,
            token_start=0,
            token_end=3,
            forward_mode=ForwardMode.EXTEND,
        )
        decode = backend._metadata_slice(
            metadata,
            req_start=1,
            req_end=3,
            token_start=3,
            token_end=5,
            forward_mode=ForwardMode.DECODE,
        )

        self.assertEqual(prefill.num_prefill_tokens, 3)
        self.assertEqual(decode.num_prefill_tokens, 0)
        self.assertTrue(
            torch.equal(prefill.token_to_req_indices, torch.tensor([0, 0, 0]))
        )
        self.assertTrue(torch.equal(decode.token_to_req_indices, torch.tensor([0, 1])))
        self.assertTrue(
            torch.equal(
                decode.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(decode.cache.swa_page_table[:, 0], torch.tensor([20, 30]))
        )
        self.assertTrue(bool((decode.cache.page_table == 0).all()))
        self.assertTrue(
            torch.equal(prefill.seq_lens_cpu, torch.tensor([5], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(decode.query_lens_cpu, torch.tensor([1, 1], dtype=torch.int32))
        )

    def test_deepseek_v4_mixed_metadata_accepts_prefill_prefix_lens_only(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=576,
                context_len=256,
            )
        )
        backend.init_forward_metadata(
            bs=4,
            req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12, 6], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            page_table=torch.tensor([[10], [20], [30], [40]], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([3, 4, 1, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([2, 5, 11], dtype=torch.int32),
            num_extends=3,
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.num_prefill_reqs, 3)
        self.assertEqual(metadata.num_prefill_tokens, 8)
        self.assertEqual(metadata.decode_req_count(), 1)
        self.assertEqual(metadata.decode_token_count(), 1)
        self.assertTrue(
            torch.equal(
                metadata.seq_lens_cpu,
                torch.tensor([5, 9, 12, 6], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.query_lens_cpu,
                torch.tensor([3, 4, 1, 1], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_mixed_backend_slices_prefill_and_decode(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=576,
                context_len=256,
            )
        )
        backend.init_forward_metadata(
            bs=3,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            page_table=torch.tensor([[10], [20], [30]], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([3, 1, 1], dtype=torch.int32),
            num_extends=1,
        )
        calls = []

        def fake_prefill(**kwargs):
            metadata = backend.forward_metadata
            calls.append(
                (
                    "prefill",
                    kwargs["q"].shape[0],
                    kwargs["positions"].tolist(),
                    kwargs["topk_indices"].tolist(),
                    metadata.req_pool_indices.tolist(),
                    metadata.token_to_req_indices.tolist(),
                    metadata.num_prefill_tokens,
                )
            )
            return kwargs["q"].new_full((3, 2, 4), 1.0)

        def fake_decode(**kwargs):
            metadata = backend.forward_metadata
            calls.append(
                (
                    "decode",
                    kwargs["q"].shape[0],
                    kwargs["positions"].tolist(),
                    kwargs["topk_indices"].tolist(),
                    metadata.req_pool_indices.tolist(),
                    metadata.token_to_req_indices.tolist(),
                    metadata.num_prefill_tokens,
                )
            )
            return kwargs["q"].new_full((2, 2, 4), 2.0)

        backend.forward_deepseek_v4_prefill = fake_prefill
        backend.forward_deepseek_v4_decode = fake_decode
        q = torch.zeros((5, 2, 4), dtype=torch.float32)
        topk = torch.arange(10, dtype=torch.int32).view(5, 2)
        out = backend.forward_deepseek_v4_mixed(
            q=q,
            positions=torch.arange(5, dtype=torch.int32),
            token_to_kv_pool=SimpleNamespace(),
            layer_id=0,
            kind="mla",
            compress_ratio=4,
            num_local_heads=2,
            padded_heads=2,
            head_dim=4,
            window_size=4,
            softmax_scale=1.0,
            attn_sink=torch.zeros(2),
            topk_indices=topk,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "prefill")
        self.assertEqual(calls[0][1], 3)
        self.assertEqual(calls[0][2], [0, 1, 2])
        self.assertEqual(calls[0][3], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(calls[0][4], [0])
        self.assertEqual(calls[0][5], [0, 0, 0])
        self.assertEqual(calls[0][6], 3)
        self.assertEqual(calls[1][0], "decode")
        self.assertEqual(calls[1][1], 2)
        self.assertEqual(calls[1][2], [3, 4])
        self.assertEqual(calls[1][3], [[6, 7], [8, 9]])
        self.assertEqual(calls[1][4], [1, 2])
        self.assertEqual(calls[1][5], [0, 1])
        self.assertEqual(calls[1][6], 0)
        self.assertTrue(torch.equal(out[:3], torch.ones((3, 2, 4))))
        self.assertTrue(torch.equal(out[3:], torch.full((2, 2, 4), 2.0)))

    def test_deepseek_v4_mixed_prefill_replaces_stale_slice(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=576,
                context_len=256,
            )
        )
        stale_prefill_metadata = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            num_prefill_reqs=1,
            req_pool_indices=torch.tensor([99], dtype=torch.int32),
            token_to_req_indices=torch.tensor([9, 9, 9], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
        )
        backend.forward_prefill_metadata = stale_prefill_metadata
        backend.init_forward_metadata(
            bs=3,
            num_tokens=5,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            page_table=torch.tensor([[10], [20], [30]], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([3, 1, 1], dtype=torch.int32),
            num_extends=1,
        )
        mixed_metadata = backend.forward_metadata
        self.assertIs(backend.forward_prefill_metadata, mixed_metadata)
        self.assertIsNot(backend.forward_prefill_metadata, stale_prefill_metadata)

        calls = []

        def fake_prefill_chunk(**kwargs):
            metadata = backend.forward_metadata
            calls.append(
                (
                    "prefill",
                    metadata.req_pool_indices.tolist(),
                    metadata.token_to_req_indices.tolist(),
                    metadata.forward_mode,
                )
            )
            q = kwargs["q"]
            return q.new_full((q.shape[0], 1, 2), 1.0)

        def fake_decode(**kwargs):
            metadata = backend.forward_metadata
            calls.append(
                (
                    "decode",
                    metadata.req_pool_indices.tolist(),
                    metadata.token_to_req_indices.tolist(),
                    metadata.forward_mode,
                )
            )
            q = kwargs["q"]
            return q.new_full((q.shape[0], 1, 2), 2.0)

        backend._forward_deepseek_v4_prefill_chunk = fake_prefill_chunk
        backend.forward_deepseek_v4_decode = fake_decode
        out = backend.forward_deepseek_v4_mixed(
            q=torch.zeros((5, 1, 2), dtype=torch.float32),
            positions=torch.arange(5, dtype=torch.int32),
            token_to_kv_pool=SimpleNamespace(),
            layer_id=0,
            kind="mla",
            compress_ratio=4,
            num_local_heads=1,
            padded_heads=1,
            head_dim=2,
            window_size=4,
            softmax_scale=1.0,
            attn_sink=torch.zeros(1),
            topk_indices=None,
        )

        self.assertEqual(calls[0][0], "prefill")
        self.assertEqual(calls[0][1], [0])
        self.assertEqual(calls[0][2], [0, 0, 0])
        self.assertTrue(calls[0][3].is_extend())
        self.assertEqual(calls[1][0], "decode")
        self.assertEqual(calls[1][1], [1, 2])
        self.assertEqual(calls[1][2], [0, 1])
        self.assertTrue(calls[1][3].is_decode())
        self.assertIs(backend.forward_metadata, mixed_metadata)
        self.assertTrue(torch.equal(out[:3], torch.ones((3, 1, 2))))
        self.assertTrue(torch.equal(out[3:], torch.full((2, 1, 2), 2.0)))

    def test_deepseek_v4_prefix_granularity_decoupled_from_kernel_page(self):
        """The scheduler grain only has to be a positive multiple of the
        kernel page: kernel geometry is registry-sourced, never P-derived."""

        def build(prefix_granularity):
            return DeepseekV4AttentionBackend(
                SimpleNamespace(
                    prefix_granularity=prefix_granularity,
                    kernel_page_size=None,
                    device="cpu",
                    num_attention_heads=64,
                    num_kv_heads=1,
                    attn_tp_size=1,
                    dtype=torch.bfloat16,
                    is_draft=False,
                    head_dim=512,
                    context_len=4096,
                    speculative_num_draft_tokens=1,
                )
            )

        for granularity in (256, 512, 1024):
            with self.subTest(prefix_granularity=granularity):
                backend = build(granularity)
                # The kernel scalar stays registry-sourced regardless of P.
                self.assertEqual(backend.kernel_page_size, 256)
                self.assertEqual(backend.max_num_pages, 4096 // 256)
        for granularity in (0, -256, 64, 128, 384):
            with self.subTest(prefix_granularity=granularity):
                with self.assertRaisesRegex(ValueError, "positive multiple"):
                    build(granularity)

    def test_deepseek_v4_spec_geometry_is_prefix_granularity_free(self):
        """The spec constructors take no P; block spans come from the kernel
        page registry (256 // ratio rows x ratio-token stride)."""
        specs = {
            spec.group_id: spec
            for spec in _v4_spec_set(
                SimpleNamespace(sliding_window=128),
                layer_ratio=(1, 4, 128),
                decode_input_tokens=1,
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

    def test_deepseek_v4_spec_metadata_requires_uniform_pack(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                head_dim=512,
                context_len=4096,
                speculative_num_draft_tokens=4,
            )
        )

        backend.init_forward_metadata(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([70, 3], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        )
        self.assertTrue(
            torch.equal(
                backend.forward_metadata.query_lens,
                torch.tensor([4, 4], dtype=torch.int32),
            )
        )
        self.assertEqual(backend.forward_metadata.forward_mode, ForwardMode.DECODE)
        self.assertEqual(backend.forward_metadata.num_prefill_reqs, 0)
        self.assertEqual(backend.forward_metadata.decode_req_count(), 2)
        self.assertEqual(backend.forward_metadata.decode_token_count(), 8)

        draft_backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=True,
                head_dim=512,
                context_len=4096,
                speculative_num_draft_tokens=4,
            )
        )
        draft_backend.init_forward_metadata(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([70, 3], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        )
        self.assertEqual(
            draft_backend.forward_metadata.forward_mode, ForwardMode.DECODE
        )
        self.assertIs(
            draft_backend.forward_prefill_metadata,
            draft_backend.forward_metadata,
        )
        self.assertIs(
            draft_backend.forward_decode_metadata, draft_backend.forward_metadata
        )

        with self.assertRaisesRegex(RuntimeError, "uniformly packed"):
            backend.init_forward_metadata(
                bs=2,
                num_tokens=7,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                seq_lens=torch.tensor([70, 3], dtype=torch.int32),
                forward_mode=ForwardMode.DECODE,
                page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            )

    def test_deepseek_v4_decode_metadata_defaults_to_one_token(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                head_dim=512,
                context_len=4096,
                speculative_num_draft_tokens=4,
            )
        )

        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([70, 3], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(
                backend.forward_metadata.query_lens,
                torch.tensor([1, 1], dtype=torch.int32),
            )
        )
        self.assertEqual(backend.forward_metadata.forward_mode, ForwardMode.DECODE)
        self.assertEqual(backend.forward_metadata.decode_token_count(), 2)

    def test_deepseek_v4_select_decode_metadata_ignores_prefill_fallback(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                head_dim=512,
                context_len=4096,
                speculative_num_draft_tokens=4,
            )
        )
        stale_prefill = _make_deepseek_v4_forward_metadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.zeros((2, 1), dtype=torch.int32),
            seq_lens=torch.tensor([70, 3], dtype=torch.int32),
            query_lens=torch.tensor([4, 4], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            forward_mode=ForwardMode.DECODE,
        )
        decode_metadata = _make_deepseek_v4_forward_metadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.zeros((2, 1), dtype=torch.int32),
            seq_lens=torch.tensor([72, 5], dtype=torch.int32),
            query_lens=torch.tensor([4, 4], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            forward_mode=ForwardMode.DECODE,
        )

        backend.forward_prefill_metadata = stale_prefill
        self.assertIsNone(backend._select_decode_metadata(8))
        backend.forward_decode_metadata = stale_prefill
        backend.forward_metadata = decode_metadata
        self.assertIs(backend._select_decode_metadata(8), decode_metadata)
        backend.forward_metadata = None
        backend.forward_decode_metadata = decode_metadata
        self.assertIs(backend._select_decode_metadata(8), decode_metadata)

    def test_deepseek_v4_cuda_graph_replay_without_num_tokens_uses_plain_decode(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                head_dim=512,
                context_len=4096,
                speculative_num_draft_tokens=4,
            )
        )
        backend.init_cuda_graph_state(max_bs=2, max_tokens_per_req=4)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([70, 3], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        backend.init_forward_metadata_replay_cuda_graph(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([70, 3], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        self.assertTrue(
            torch.equal(
                backend.forward_metadata.query_lens,
                torch.tensor([1, 1], dtype=torch.int32),
            )
        )
        self.assertEqual(backend.forward_metadata.decode_token_count(), 2)

    def test_deepseek_v4_decode_backend_maps_compressed_slots_batched(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=128,
            )
        )
        seq_lens = torch.tensor([70, 3], dtype=torch.int32)
        c4_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=c4_table,
            block_tables=_v4_compressed_kv_tables(c4=c4_table),
        )
        positions = seq_lens.to(torch.int64) - 1

        topk_indices = torch.tensor(
            [[1, 65, 3, -1], [0, -1, -1, -1]],
            dtype=torch.int32,
        )
        indices, lens = backend._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=4,
            block_size=64,
            topk_indices=topk_indices,
        )
        self.assertTrue(torch.equal(lens, torch.tensor([3, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                indices[:, 0, :4],
                torch.tensor(
                    [[641, 705, 643, -1], [1280, -1, -1, -1]],
                    dtype=torch.int32,
                ),
            )
        )

        seq_lens = torch.tensor([256, 129], dtype=torch.int32)
        c128_table = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
        )
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=c128_table,
            block_tables=_v4_compressed_kv_tables(c128=c128_table),
        )
        hca_positions = seq_lens.to(torch.int64) - 1
        indices, lens = backend._decode_compressed_attention_indices_and_lens(
            hca_positions,
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )
        self.assertTrue(torch.equal(lens, torch.tensor([2, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                indices[:, 0, :2],
                torch.tensor([[640, 641], [1280, -1]], dtype=torch.int32),
            )
        )
        cached_indices, cached_lens = (
            backend._decode_compressed_attention_indices_and_lens(
                hca_positions,
                compress_ratio=128,
                block_size=64,
                topk_indices=None,
            )
        )
        self.assertEqual(cached_indices.data_ptr(), indices.data_ptr())
        self.assertEqual(cached_lens.data_ptr(), lens.data_ptr())

    def test_deepseek_v4_decode_backend_capture_ignores_warmup_cache(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for capture cache semantics")
        device = torch.device("cuda")
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cuda",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=128,
            )
        )
        seq_lens = torch.tensor([128, 64], device=device, dtype=torch.int32)
        c128_table = torch.tensor(
            [[10, 11], [20, 21]],
            device=device,
            dtype=torch.int32,
        )
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], device=device, dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=c128_table,
            block_tables=_v4_compressed_kv_tables(c128=c128_table),
        )
        positions = seq_lens.to(torch.int64) - 1

        warmup_indices, _ = backend._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )
        metadata = backend.forward_metadata
        indices_cache = metadata.attention.decode_dense_compressed_indices_cache
        key = next(iter(indices_cache.keys()))
        metadata.attention.decode_dense_compressed_indices_capture_safe_keys.clear()

        original_capturing = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: True
        try:
            capture_indices, _ = backend._decode_compressed_attention_indices_and_lens(
                positions,
                compress_ratio=128,
                block_size=64,
                topk_indices=None,
            )
            reused_indices, _ = backend._decode_compressed_attention_indices_and_lens(
                positions,
                compress_ratio=128,
                block_size=64,
                topk_indices=None,
            )
        finally:
            torch.cuda.is_current_stream_capturing = original_capturing

        self.assertNotEqual(capture_indices.data_ptr(), warmup_indices.data_ptr())
        self.assertEqual(reused_indices.data_ptr(), capture_indices.data_ptr())
        self.assertIn(
            key,
            metadata.attention.decode_dense_compressed_indices_capture_safe_keys,
        )

    def test_deepseek_v4_c128a_prefill_local_compressed_indices_contract(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=1024,
            )
        )
        self.assertEqual(backend._dense_compressed_indices_width(128), 128)

        indices = backend._dense_prefill_local_compressed_indices(
            torch.tensor([0, 127, 128, 255], dtype=torch.int64),
            compress_ratio=128,
            width=backend._dense_compressed_indices_width(128),
        )
        self.assertEqual(tuple(indices.shape), (4, 128))
        self.assertTrue(
            torch.equal(indices[0, :2], torch.tensor([-1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(indices[1, :3], torch.tensor([0, -1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(indices[2, :3], torch.tensor([0, -1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(indices[3, :4], torch.tensor([0, 1, -1, -1], dtype=torch.int32))
        )
        cached = backend._dense_prefill_local_compressed_indices(
            torch.tensor([127], dtype=torch.int64),
            compress_ratio=128,
            width=backend._dense_compressed_indices_width(128),
        )
        self.assertEqual(cached.data_ptr(), indices.data_ptr())

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

    def test_deepseek_v4_decode_backend_masks_padding_tokens(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=128,
            )
        )
        seq_lens = torch.tensor([70, 3], dtype=torch.int32)
        compressed_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=compressed_table,
            block_tables=_v4_compressed_kv_tables(
                c4=compressed_table,
                c128=compressed_table,
            ),
        )
        metadata = backend.forward_metadata
        metadata.is_valid_token = torch.tensor([True, False])
        positions = seq_lens.to(torch.int64) - 1

        topk_indices = torch.tensor(
            [[1, 65, 3, -1], [0, -1, -1, -1]],
            dtype=torch.int32,
        )
        _, csa_lens = backend._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=4,
            block_size=64,
            topk_indices=topk_indices,
        )
        _, hca_lens = backend._decode_compressed_attention_indices_and_lens(
            torch.tensor([255, 128], dtype=torch.int64),
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )

        self.assertTrue(torch.equal(csa_lens, torch.tensor([3, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(hca_lens, torch.tensor([2, 0], dtype=torch.int32)))

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
        self.assertTrue(torch.equal(lens, torch.tensor([1, 0], dtype=torch.int32)))

    def test_deepseek_v4_cuda_graph_replay_marks_padding_tokens_invalid(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=128,
            )
        )
        backend.init_cuda_graph_state(max_bs=4)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        backend.init_forward_metadata_replay_cuda_graph(
            bs=4,
            actual_bs=2,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.tensor([70, 3, 1, 1], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                    [30, 31],
                    [40, 41],
                ],
                dtype=torch.int32,
            ),
        )

        metadata = backend.forward_metadata
        self.assertTrue(
            torch.equal(
                metadata.is_valid_token,
                torch.tensor([True, True, False, False]),
            )
        )
        self.assertEqual(metadata.decode_token_count(), 4)

    def test_deepseek_v4_cuda_graph_replay_preserves_padded_cache_tables(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                speculative_num_draft_tokens=1,
                head_dim=512,
                context_len=512,
            )
        )
        group_id = v4_compressed_kv_group_id(4)
        backend.init_cuda_graph_state(
            max_bs=4,
            cache_group_specs=(
                CacheGroupSpec(
                    group_id=group_id,
                    retention="full_history",
                    rows_per_page=64,
                    entry_stride_tokens=4,
                    sliding_window_tokens=None,
                ),
            ),
            cache_group_page_counts={group_id: 128},
        )
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={group_id: torch.ones((4, 2), dtype=torch.int32)},
        )
        active_table = torch.tensor(
            [[10, 11], [20, 21], [0, 0], [0, 0]], dtype=torch.int32
        )

        backend.init_forward_metadata_replay_cuda_graph(
            bs=4,
            actual_bs=2,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.tensor([70, 3, 1, 1], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={group_id: active_table},
        )

        table = backend.forward_metadata.cache.page_table
        self.assertTrue(torch.equal(table[:2, :2], active_table[:2]))
        self.assertTrue(
            torch.equal(table[2:, :2], torch.zeros((2, 2), dtype=torch.int32))
        )

    def test_deepseek_v4_indexer_metadata_refresh_masks_padding_tokens(self):
        key = (4, 4, 3)
        block_table = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            page_table=block_table,
            seq_lens=torch.tensor([9, 5, 3], dtype=torch.int32),
            query_lens=torch.ones(3, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            block_tables=_v4_compressed_kv_tables(c4=block_table),
            is_valid_token=torch.tensor([True, False, True]),
        )
        plan = DeepseekV4IndexerDecodePlan(
            context_lens=torch.empty((3, 1), dtype=torch.int32),
            page_table=torch.empty((3, 2), dtype=torch.int32),
            max_context_len=0,
        )
        metadata.indexer.decode_plan_cache[key] = plan

        def fake_compute(**kwargs):
            kwargs["out_context_lens"].copy_(
                torch.tensor([[2], [2], [1]], dtype=torch.int32)
            )
            kwargs["out_block_tables"].copy_(
                torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
            )

        with patch.object(
            deepseek_v4_backend,
            "dsv4_indexer_decode_metadata_compute",
            side_effect=fake_compute,
        ):
            deepseek_v4_backend._refresh_decode_indexer_plan_cache(
                metadata,
                max_context_len=256,
            )

        self.assertTrue(
            torch.equal(
                plan.context_lens,
                torch.tensor([[2], [0], [1]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.page_table,
                torch.tensor([[10, 11], [0, 0], [30, 31]], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_decode_plan_accepts_sliced_valid_mask(self):
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=4,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            seq_lens=torch.tensor([9, 5], dtype=torch.int32),
            query_lens=torch.ones(2, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        )

        def fake_compute(**kwargs):
            kwargs["out_context_lens"].copy_(
                torch.tensor([[2], [2]], dtype=torch.int32)
            )
            kwargs["out_block_tables"].copy_(
                torch.tensor([[10], [20]], dtype=torch.int32)
            )

        with patch.object(
            deepseek_v4_model,
            "dsv4_indexer_decode_metadata_compute",
            side_effect=fake_compute,
        ):
            plan = _deepseek_v4_indexer_decode_plan(
                positions=torch.tensor([8, 4], dtype=torch.int64),
                token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
                block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
                cache_block_size=4,
                compress_ratio=4,
                metadata=metadata,
                is_valid_token=torch.tensor([False, True]),
            )

        self.assertTrue(
            torch.equal(
                plan.context_lens,
                torch.tensor([[0], [2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.page_table,
                torch.tensor([[0], [20]], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_schedule_refresh_uses_decode_plan_lens(self):
        captured = {}

        def fake_dsv4_plan(*, seq_lens_2d, page_size, out):
            captured["context_lens"] = seq_lens_2d.clone()
            captured["cache_block_size"] = page_size
            captured["out"] = out
            out.fill_(9)
            return out

        key = (4, 4, 2)
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.tensor([[0], [0]], dtype=torch.int32),
            seq_lens=torch.tensor([5, 1], dtype=torch.int32),
            query_lens=torch.tensor([1, 1], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
            is_valid_token=torch.tensor([True, False]),
        )
        metadata.indexer.decode_plan_cache[key] = DeepseekV4IndexerDecodePlan(
            context_lens=torch.zeros((2, 1), dtype=torch.int32),
            page_table=torch.zeros((2, 1), dtype=torch.int32),
            max_context_len=0,
        )
        metadata.indexer.decode_schedule_metadata_cache[key] = torch.zeros(
            (2, 1),
            dtype=torch.int32,
        )

        with patch.object(deepseek_v4_backend, "dsv4_plan", side_effect=fake_dsv4_plan):
            deepseek_v4_backend._refresh_decode_indexer_schedule_metadata(metadata)

        self.assertTrue(
            torch.equal(
                captured["context_lens"], torch.zeros((2, 1), dtype=torch.int32)
            )
        )
        self.assertEqual(captured["cache_block_size"], 4)
        self.assertIs(
            captured["out"], metadata.indexer.decode_schedule_metadata_cache[key]
        )
        self.assertTrue(
            torch.equal(
                metadata.indexer.decode_schedule_metadata_cache[key],
                torch.full((2, 1), 9, dtype=torch.int32),
            )
        )

    def test_deepseek_v4_cuda_graph_decode_uses_packed_metadata(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                head_dim=512,
                context_len=128,
                speculative_num_draft_tokens=4,
            )
        )
        backend.init_cuda_graph_state(max_bs=4)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            num_tokens=16,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        metadata = backend.forward_metadata
        self.assertEqual(metadata.forward_mode, ForwardMode.DECODE)
        self.assertTrue(
            torch.equal(metadata.seq_lens, torch.full((4,), 4, dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(metadata.query_lens, torch.full((4,), 4, dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                metadata.query_start_loc,
                torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor(
                    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertEqual(metadata.decode_token_count(), 16)

        backend.init_forward_metadata_replay_cuda_graph(
            bs=4,
            actual_bs=2,
            num_tokens=16,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.tensor([70, 3, 1, 1], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                    [30, 31],
                    [40, 41],
                ],
                dtype=torch.int32,
            ),
        )

        metadata = backend.forward_metadata
        self.assertEqual(metadata.forward_mode, ForwardMode.DECODE)
        self.assertTrue(
            torch.equal(
                metadata.is_valid_token,
                torch.tensor(
                    [True] * 8 + [False] * 8,
                    dtype=torch.bool,
                ),
            )
        )
        self.assertEqual(metadata.decode_req_count(), 4)
        self.assertEqual(metadata.decode_token_count(), 16)

    def test_deepseek_v4_cuda_graph_packed_draft_decode_advances_metadata(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=True,
                head_dim=512,
                context_len=128,
                speculative_num_draft_tokens=4,
            )
        )
        backend.init_cuda_graph_state(max_bs=4)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            num_tokens=16,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )
        backend.init_forward_metadata_replay_cuda_graph(
            bs=4,
            actual_bs=2,
            num_tokens=16,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.tensor([70, 3, 1, 1], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            page_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                    [30, 31],
                    [40, 41],
                ],
                dtype=torch.int32,
            ),
        )

        self.assertIs(backend.forward_prefill_metadata, backend.forward_metadata)
        backend.advance_draft_forward_metadata()

        metadata = backend.forward_metadata
        self.assertEqual(metadata.forward_mode, ForwardMode.DECODE)
        self.assertTrue(
            torch.equal(
                metadata.seq_lens,
                torch.tensor([71, 4, 2, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.is_valid_token,
                torch.tensor([True, True, False, False], dtype=torch.bool),
            )
        )
        self.assertEqual(metadata.decode_token_count(), 4)

        first_decode_metadata = metadata
        cached_swa = torch.empty((4, 8), dtype=torch.int32)
        first_decode_metadata.attention.decode_swa_indices = cached_swa
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            num_tokens=16,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )
        backend.advance_draft_forward_metadata()
        self.assertIs(backend.forward_metadata, first_decode_metadata)
        self.assertIs(backend.forward_metadata.attention.decode_swa_indices, cached_swa)

    def test_deepseek_v4_draft_metadata_fallback_prefers_current_shape(self):
        prefill_metadata = SimpleNamespace(
            token_to_req_indices=torch.arange(4, dtype=torch.int32)
        )
        decode_metadata = SimpleNamespace(
            token_to_req_indices=torch.arange(1, dtype=torch.int32)
        )
        ctx = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_num_tokens=1,
            attn_backend=SimpleNamespace(
                forward_metadata=decode_metadata,
                forward_prefill_metadata=prefill_metadata,
            ),
        )

        self.assertIs(_deepseek_v4_forward_metadata(ctx), decode_metadata)

    def test_deepseek_v4_eager_draft_decode_refreshes_stale_graph_metadata(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=True,
                head_dim=512,
                context_len=128,
                speculative_num_draft_tokens=4,
            )
        )
        backend.init_cuda_graph_state(max_bs=4)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            num_tokens=16,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )
        self.assertEqual(backend._draft_decode_metadata.token_to_req_indices.numel(), 4)

        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([6], dtype=torch.int32)
        page_table = torch.tensor([[10]], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=1,
            num_tokens=6,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=ForwardMode.EXTEND,
            page_table=page_table,
            extend_seq_lens_cpu=seq_lens.cpu(),
            extend_prefix_lens_cpu=torch.zeros(1, dtype=torch.int32),
        )
        backend.init_forward_metadata(
            bs=1,
            num_tokens=1,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=page_table,
        )
        backend.advance_draft_forward_metadata()

        metadata = backend.forward_metadata
        self.assertEqual(metadata.forward_mode, ForwardMode.DECODE)
        self.assertEqual(metadata.token_to_req_indices.numel(), 1)
        self.assertEqual(metadata.decode_token_count(), 1)
        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_prefill_uses_prefill_metadata_slot(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                prefix_granularity=64,
                kernel_page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                is_draft=False,
                head_dim=512,
                context_len=128,
                speculative_num_draft_tokens=4,
            )
        )
        prefill_metadata = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            num_prefill_reqs=1,
            seq_lens=torch.tensor([6], dtype=torch.int32),
            token_to_req_indices=torch.zeros(6, dtype=torch.int32),
        )
        decode_metadata = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        backend.forward_prefill_metadata = prefill_metadata
        backend.forward_metadata = decode_metadata

        def fake_prefill_chunk(**kwargs):
            self.assertIs(backend.forward_metadata, prefill_metadata)
            q = kwargs["q"]
            return q.new_zeros((q.shape[0], 1, 2))

        backend._forward_deepseek_v4_prefill_chunk = fake_prefill_chunk
        out = backend.forward_deepseek_v4_prefill(
            q=torch.empty((6, 1, 2), dtype=torch.bfloat16),
            positions=torch.arange(6, dtype=torch.int64),
            token_to_kv_pool=SimpleNamespace(),
            layer_id=0,
            kind="test",
            compress_ratio=1,
            num_local_heads=1,
            padded_heads=1,
            head_dim=2,
            window_size=64,
            softmax_scale=1.0,
            attn_sink=torch.empty((1,), dtype=torch.float32),
            topk_indices=None,
        )

        self.assertEqual(out.shape, (6, 1, 2))
        self.assertIs(backend.forward_metadata, prefill_metadata)

    def test_deepseek_v4_indexer_decode_plan_batches_metadata(self):
        positions = torch.tensor([15, 7, 3], dtype=torch.int64)
        token_to_req_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        block_table = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
            dtype=torch.int32,
        )
        calls = []

        def fake_decode_metadata_compute(**kwargs):
            calls.append(kwargs)
            kwargs["out_context_lens"].copy_(
                torch.tensor([[4], [2], [1]], dtype=torch.int32)
            )
            kwargs["out_block_tables"].copy_(
                torch.tensor([[10], [20], [30]], dtype=torch.int32)
            )

        with (
            patch.dict(global_server_args_dict, {"max_model_len": None}),
            patch.object(
                deepseek_v4_model,
                "dsv4_indexer_decode_metadata_compute",
                fake_decode_metadata_compute,
            ),
        ):
            plan = deepseek_v4_model._deepseek_v4_indexer_decode_plan(
                positions=positions,
                token_to_req_indices=token_to_req_indices,
                block_table=block_table,
                cache_block_size=4,
                compress_ratio=4,
                is_valid_token=torch.tensor([True, False, True]),
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_blocks"], 1)
        self.assertEqual(plan.max_context_len, 4)
        self.assertTrue(
            torch.equal(
                plan.context_lens,
                torch.tensor([[4], [0], [1]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.page_table,
                torch.tensor([[10], [0], [30]], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_decode_max_len_uses_context_or_cache_window(self):
        block_table = torch.zeros((2, 257), dtype=torch.int32)

        with patch.dict(global_server_args_dict, {"max_model_len": 4096}):
            self.assertEqual(
                _deepseek_v4_indexer_decode_max_len(
                    block_table,
                    cache_block_size=64,
                    compress_ratio=4,
                ),
                1024,
            )

        with patch.dict(global_server_args_dict, {"max_model_len": None}):
            self.assertEqual(
                _deepseek_v4_indexer_decode_max_len(
                    block_table,
                    cache_block_size=64,
                    compress_ratio=4,
                ),
                4112,
            )

    def test_deepseek_v4_indexer_topk_requires_cuda_logits(self):
        logits = torch.tensor(
            [[0.0, 3.0, 1.0, -float("inf")]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "requires CUDA float32 logits"):
            _deepseek_v4_indexer_topk_from_logits(
                logits,
                lengths,
                topk_tokens=2,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_indexer_topk_rejects_unsupported_decode_topk(self):
        logits = torch.tensor(
            [[0.0, 3.0, 1.0, -float("inf")]],
            device="cuda",
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], device="cuda", dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "supports topk_tokens"):
            _deepseek_v4_indexer_topk_from_logits(
                logits,
                lengths,
                topk_tokens=4,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_indexer_topk_uses_local_prefill_op(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf"), -float("inf"), -float("inf")],
                [-float("inf"), -float("inf"), -float("inf"), 2.0, 8.0, 5.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        row_starts = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        row_ends = torch.tensor([3, 6], device="cuda", dtype=torch.int32)
        out = torch.empty((2, 4), device="cuda", dtype=torch.int32)

        try:
            actual = _deepseek_v4_indexer_topk_from_logits(
                logits,
                row_ends - row_starts,
                topk_tokens=4,
                use_prefill_topk_op=True,
                row_starts=row_starts,
                row_ends=row_ends,
                out=out,
            )
        except RuntimeError as exc:
            if "requires the CUDA prefill top-k op" not in str(exc):
                raise
            self.skipTest(str(exc))

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        expected = torch.tensor(
            [[0, 1, 2, -1], [0, 1, 2, -1]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_deepseek_v4_topk_buffer_grows_and_reuses(self):
        buffer = _DeepseekV4TopKBuffer(topk_tokens=3)

        first = buffer.get(2, torch.device("cpu"))
        second = buffer.get(1, torch.device("cpu"))
        third = buffer.get(4, torch.device("cpu"))

        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(second.shape, (1, 3))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(third.shape, (4, 3))
        self.assertGreaterEqual(buffer.buffer.shape[0], 4)

    def test_deepseek_v4_sparse_indexer_custom_op_registered(self):
        self.assertTrue(
            hasattr(torch.ops.tokenspeed, "deepseek_v4_sparse_attn_indexer")
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_sparse_indexer_custom_op_covers_decode_tokens(self):
        device = torch.device("cuda")
        n_head = 2
        head_dim = 4
        total_tokens = 3

        class FakeLinear:
            def __init__(self, out_features):
                self.out_features = out_features

            def __call__(self, x):
                return (
                    torch.zeros(
                        (x.shape[0], self.out_features),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    None,
                )

        self_obj = SimpleNamespace(
            use_fp4_cache=True,
            wq_b=FakeLinear(n_head * head_dim),
            weights_proj=FakeLinear(n_head),
            n_head=n_head,
            head_dim=head_dim,
            softmax_scale=1.0,
            compress_ratio=4,
            topk_tokens=2,
            topk_buffer=None,
            _persistent_topk_workspace=None,
            _prefill_gather_workspace=lambda rows, device: (
                torch.empty((0, 0), dtype=torch.uint8, device=device),
                torch.empty((0, 0), dtype=torch.uint8, device=device),
            ),
        )
        c4_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32, device=device),
            page_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
            seq_lens=torch.tensor([4], dtype=torch.int32, device=device),
            query_lens=torch.tensor([1], dtype=torch.int32, device=device),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            token_to_req_indices=torch.tensor(
                [0, 0, 0], dtype=torch.int32, device=device
            ),
            block_tables=_v4_compressed_kv_tables(c4=c4_table),
            num_prefill_tokens=1,
            num_prefill_reqs=1,
            seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
            query_lens_cpu=torch.tensor([1], dtype=torch.int32),
        )
        ctx = SimpleNamespace(forward_mode=ForwardMode.MIXED)
        captured = {}

        def fake_prepare_mxfp4(**kwargs):
            index_q = kwargs["index_q"]
            rows = index_q.shape[0]
            return (
                (
                    torch.empty(
                        (rows, n_head, head_dim // 2), dtype=torch.uint8, device=device
                    ),
                    torch.empty((rows, n_head, 1), dtype=torch.uint8, device=device),
                ),
                torch.empty((rows, n_head), dtype=torch.float32, device=device),
            )

        def fake_sparse_indexer(**kwargs):
            captured["packed_rows"] = kwargs["packed_q_values"].shape[0]
            captured["has_forward_metadata"] = "metadata" in kwargs
            captured["has_sparse_indexer_metadata"] = "indexer_metadata" in kwargs
            captured["has_indexer_cache"] = "indexer_cache" in kwargs
            captured["has_indexer_page_table"] = "indexer_page_table" in kwargs
            captured["cache_block_size"] = kwargs["indexer_block_size"]
            captured["cache_compress_ratio"] = kwargs["compress_ratio"]
            indexer_metadata = kwargs["indexer_metadata"]
            captured["num_prefill_tokens"] = (
                indexer_metadata.batch_metadata.num_prefill_tokens
            )
            captured["num_decode_tokens"] = (
                indexer_metadata.batch_metadata.num_decode_tokens
            )
            captured["prefill_chunks"] = len(indexer_metadata.prefill_metadata.chunks)
            captured["decode_max_context_len"] = (
                indexer_metadata.decode_plan.max_context_len
            )
            legacy_index_q_key = "fall" + "back_index_q"
            captured["has_reference_inputs"] = legacy_index_q_key in kwargs
            return torch.full(
                (total_tokens, self_obj.topk_tokens),
                7,
                dtype=torch.int32,
                device=device,
            )

        empty_prefill_metadata = DeepseekV4IndexerPrefillMetadata.empty(device)
        decode_metadata = SimpleNamespace(
            context_lens=torch.ones((2, 1), dtype=torch.int32, device=device),
            block_table=torch.zeros((2, 1), dtype=torch.int32, device=device),
            max_context_len=1,
        )

        with (
            # deep_gemm is unavailable on AMD runners; the guard only checks
            # for None and every consumer below is patched out.
            patch.object(
                deepseek_v4_model,
                "deep_gemm",
                deepseek_v4_model.deep_gemm or SimpleNamespace(),
            ),
            patch.object(
                deepseek_v4_model,
                "deepseek_v4_prepare_indexer_q_mxfp4",
                side_effect=fake_prepare_mxfp4,
            ),
            patch.object(
                deepseek_v4_model,
                "_deepseek_v4_deepgemm_fp4_indexer_available",
                return_value=True,
            ),
            patch.object(
                deepseek_v4_model,
                "_deepseek_v4_indexer_prefill_metadata",
                return_value=empty_prefill_metadata,
            ),
            patch.object(
                deepseek_v4_model,
                "_deepseek_v4_indexer_decode_plan",
                return_value=decode_metadata,
            ),
            patch.object(
                deepseek_v4_model,
                "_deepseek_v4_indexer_decode_schedule_metadata",
                return_value=None,
            ),
            patch.object(
                deepseek_v4_model,
                "_deepseek_v4_sparse_attn_indexer",
                side_effect=fake_sparse_indexer,
            ),
        ):
            actual = DeepseekV4Indexer._forward_sparse_indexer_custom_op(
                self_obj,
                hidden_states=torch.zeros((total_tokens, 8), device=device),
                qr=torch.zeros((total_tokens, 8), device=device),
                positions=torch.arange(total_tokens, dtype=torch.int64, device=device),
                metadata=metadata,
                ctx=ctx,
                indexer_cache=torch.empty((1, 1), dtype=torch.uint8, device=device),
                indexer_block_size=1,
                cos_sin_cache=torch.empty((1, 1), device=device),
            )

        self.assertEqual(tuple(actual.shape), (total_tokens, self_obj.topk_tokens))
        self.assertEqual(captured["packed_rows"], total_tokens)
        self.assertFalse(captured["has_forward_metadata"])
        self.assertTrue(captured["has_sparse_indexer_metadata"])
        self.assertTrue(captured["has_indexer_cache"])
        self.assertTrue(captured["has_indexer_page_table"])
        self.assertEqual(captured["cache_block_size"], 1)
        self.assertEqual(captured["cache_compress_ratio"], self_obj.compress_ratio)
        self.assertEqual(captured["prefill_chunks"], 0)
        self.assertEqual(captured["decode_max_context_len"], 1)
        self.assertFalse(captured["has_reference_inputs"])
        self.assertEqual(captured["num_prefill_tokens"], 1)
        self.assertEqual(captured["num_decode_tokens"], 2)

    def test_deepseek_v4_sparse_indexer_prefill_requires_metadata(self):
        with self.assertRaisesRegex(RuntimeError, "requires prepared chunk metadata"):
            deepseek_v4_model._deepseek_v4_sparse_attn_indexer_native(
                cache_2d=torch.empty((1, 1), dtype=torch.uint8),
                positions=torch.arange(1, dtype=torch.int64),
                token_to_req_indices=torch.zeros(1, dtype=torch.int32),
                block_table=torch.zeros((1, 1), dtype=torch.int32),
                seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
                query_lens_cpu=torch.tensor([1], dtype=torch.int32),
                prefill_chunk_specs=torch.empty((0, 5), dtype=torch.int64),
                prefill_chunk_offsets=torch.empty((0, 7), dtype=torch.int64),
                prefill_slots=torch.empty(0, dtype=torch.int64),
                prefill_cu_seq_lens=torch.empty(0, dtype=torch.int32),
                prefill_cu_seqlen_k_start=torch.empty(0, dtype=torch.int32),
                prefill_cu_seqlen_k_end=torch.empty(0, dtype=torch.int32),
                prefill_seq_lens_k=torch.empty(0, dtype=torch.int32),
                packed_q_values=torch.empty((1, 1, 1), dtype=torch.int8),
                packed_q_scales=torch.empty((1, 1), dtype=torch.int32),
                packed_weights=torch.empty((1, 1), dtype=torch.float32),
                decode_schedule_metadata=None,
                decode_context_lens=None,
                decode_block_table=None,
                decode_max_context_len=0,
                topk_indices_buffer=torch.empty((1, 1), dtype=torch.int32),
                prefill_gather_values_workspace=torch.empty((0, 1), dtype=torch.uint8),
                prefill_gather_scales_workspace=torch.empty((0, 1), dtype=torch.uint8),
                persistent_topk_workspace=torch.empty(0, dtype=torch.uint8),
                cache_block_size=1,
                compress_ratio=4,
                topk_tokens=1,
                num_prefill_tokens=1,
                num_decode_tokens=0,
            )

    def test_deepseek_v4_mixed_indexer_forward_uses_custom_op(self):
        base_block_table = torch.tensor([[1]], dtype=torch.int32)
        indexer_block_table = torch.tensor([[7]], dtype=torch.int32)
        captured = {}

        class FakeCompressor:
            def __init__(self):
                self.norm = SimpleNamespace(
                    weight=torch.ones(1),
                    variance_epsilon=1e-6,
                )

            def __call__(self, **kwargs):
                return None

        pool = SimpleNamespace(
            state_block_size=4,
            get_indexer_state_buffer=lambda layer_id: torch.empty((1, 1)),
            get_indexer_state_block_size=lambda layer_id: 4,
            get_indexer_block_size=lambda layer_id: 4,
            get_indexer_kv_buffer_2d=lambda layer_id: torch.empty((8, 128)),
        )
        metadata = _make_deepseek_v4_forward_metadata(
            page_size=4,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            page_table=base_block_table,
            seq_lens=torch.tensor([8], dtype=torch.int32),
            query_lens=torch.tensor([2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32),
            block_tables={
                "v4.c4a.compressed_kv": indexer_block_table,
                "v4.c4a.indexer_compressor_state": torch.tensor(
                    [[2, 3]], dtype=torch.int32
                ),
            },
            num_prefill_tokens=2,
            num_prefill_reqs=1,
            seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
            query_lens_cpu=torch.tensor([2], dtype=torch.int32),
        )
        ctx = SimpleNamespace(
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(forward_metadata=metadata),
            forward_mode=ForwardMode.MIXED,
        )
        self_obj = SimpleNamespace(
            use_fp4_cache=False,
            compressor=FakeCompressor(),
            compress_ratio=4,
            topk_tokens=2,
        )

        def fake_custom_op(**kwargs):
            captured["indexer_block_size"] = kwargs["indexer_block_size"]
            captured["indexer_cache"] = kwargs["indexer_cache"]
            return torch.full((2, 2), 3, dtype=torch.int32)

        self_obj._forward_sparse_indexer_custom_op = fake_custom_op

        with patch.object(
            deepseek_v4_model,
            "deepseek_v4_csa_indexer_cache_insert",
            return_value=None,
        ):
            topk = DeepseekV4Indexer.forward(
                self_obj,
                hidden_states=torch.zeros((2, 8)),
                qr=torch.zeros((2, 8)),
                positions=torch.tensor([6, 7], dtype=torch.int64),
                ctx=ctx,
                out_cache_loc=torch.zeros(2, dtype=torch.int64),
                layer_index=0,
                cos_sin_cache=torch.empty((1, 1)),
                compressor_slot_cache={},
            )

        self.assertEqual(captured["indexer_block_size"], 4)
        self.assertEqual(captured["indexer_cache"].shape, (8, 128))
        self.assertTrue(torch.equal(topk, torch.full((2, 2), 3, dtype=torch.int32)))

    def test_deepseek_v4_indexer_prefill_request_chunks_match_reference(self):
        chunks = _deepseek_v4_indexer_prefill_request_chunks(
            seq_lens_cpu=torch.tensor([16], dtype=torch.int32),
            query_lens_cpu=torch.tensor([6], dtype=torch.int32),
            compress_ratio=4,
            num_tokens=6,
            max_logits_bytes=32,
            workspace_size=100,
        )

        self.assertEqual(
            [
                (
                    c.req_start,
                    c.req_end,
                    c.query_start,
                    c.query_end,
                    c.token_start,
                    c.token_end,
                    c.skip_kv_gather,
                )
                for c in chunks
            ],
            [
                (0, 1, 0, 2, 0, 2, False),
                (0, 1, 2, 4, 2, 4, True),
                (0, 1, 4, 6, 4, 6, True),
            ],
        )

        chunks = _deepseek_v4_indexer_prefill_request_chunks(
            seq_lens_cpu=torch.tensor([16, 8], dtype=torch.int32),
            query_lens_cpu=torch.tensor([2, 2], dtype=torch.int32),
            compress_ratio=4,
            num_tokens=4,
            max_logits_bytes=128,
            workspace_size=100,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].req_start, chunks[0].req_end), (0, 2))
        self.assertEqual((chunks[0].token_start, chunks[0].token_end), (0, 4))
        self.assertFalse(chunks[0].skip_kv_gather)

    def test_deepseek_v4_indexer_prefill_request_gather_plan_matches_reference(self):
        slots, cu_start, cu_end, row_lens, max_len = (
            _deepseek_v4_indexer_prefill_request_gather_plan(
                seq_lens_cpu=torch.tensor([16, 8], dtype=torch.int32),
                query_lens_cpu=torch.tensor([4, 2], dtype=torch.int32),
                block_table=torch.tensor([[10], [20]], dtype=torch.int32),
                cache_block_size=4,
                compress_ratio=4,
                req_start=0,
                req_end=2,
                query_start=1,
                query_end=5,
            )
        )

        self.assertTrue(torch.equal(slots, torch.tensor([40, 41, 42, 43, 80, 81])))
        self.assertTrue(torch.equal(cu_start, torch.tensor([0, 0, 0, 4])))
        self.assertTrue(torch.equal(cu_end, torch.tensor([3, 3, 4, 5])))
        self.assertTrue(torch.equal(row_lens, torch.tensor([3, 3, 4, 1])))
        self.assertEqual(max_len, 4)

    def test_deepseek_v4_indexer_prefill_metadata_builds_chunk_plan(self):
        metadata = SimpleNamespace(
            seq_lens_cpu=torch.tensor([16, 8], dtype=torch.int32),
            query_lens_cpu=torch.tensor([4, 2], dtype=torch.int32),
            num_prefill_reqs=2,
            indexer=SimpleNamespace(prefill_plan_cache={}),
        )
        block_table = torch.tensor([[10], [20]], dtype=torch.int32)

        actual = _deepseek_v4_indexer_prefill_metadata(
            metadata=metadata,
            block_table=block_table,
            cache_block_size=4,
            compress_ratio=4,
            num_prefill_tokens=6,
        )
        cached = _deepseek_v4_indexer_prefill_metadata(
            metadata=metadata,
            block_table=block_table,
            cache_block_size=4,
            compress_ratio=4,
            num_prefill_tokens=6,
        )

        self.assertIs(actual, cached)
        self.assertEqual(len(actual.chunks), 1)
        chunk = actual.chunks[0]
        self.assertEqual(chunk.token_start, 0)
        self.assertEqual(chunk.token_end, 6)
        self.assertEqual(chunk.request_start, 0)
        self.assertEqual(chunk.request_end, 2)
        self.assertEqual(chunk.slot_start, 0)
        self.assertEqual(chunk.slot_end, 6)
        self.assertEqual(chunk.gather_row_start, 0)
        self.assertEqual(chunk.gather_row_end, 6)
        self.assertEqual(chunk.max_seq_len_k, 4)
        self.assertEqual(chunk.cu_seq_lens_start, 0)
        self.assertEqual(chunk.cu_seq_lens_end, 3)
        self.assertFalse(chunk.skip_kv_gather)
        self.assertEqual(actual.max_gather_rows(), 6)
        self.assertTrue(
            torch.equal(
                actual.chunk_specs,
                torch.tensor([[0, 6, 0, 2, 0]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                actual.chunk_offsets,
                torch.tensor([[0, 6, 0, 6, 4, 0, 3]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(actual.slots, torch.tensor([40, 41, 42, 43, 80, 81]))
        )
        self.assertTrue(
            torch.equal(actual.cu_seq_lens, torch.tensor([0, 4, 6], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(actual.cu_seqlen_k_start, torch.tensor([0, 0, 0, 0, 4, 4]))
        )
        self.assertTrue(
            torch.equal(actual.cu_seqlen_k_end, torch.tensor([3, 3, 3, 4, 5, 6]))
        )
        self.assertTrue(
            torch.equal(actual.seq_lens_k, torch.tensor([3, 3, 3, 4, 1, 2]))
        )

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

    def test_hidden_compression_runtime_requires_fast_kernel(self):
        tokens, hc_mult, hidden = 1, 2, 4
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.bfloat16)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(mix_hc, dtype=torch.float32)
        hidden_states = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        post = torch.ones(tokens, hc_mult, 1, dtype=torch.float32)
        comb = torch.eye(hc_mult, dtype=torch.float32).unsqueeze(0)

        with self.assertRaises(RuntimeError):
            mhc_pre(
                residual,
                fn,
                scale,
                base,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=2,
            )
        with self.assertRaises(RuntimeError):
            mhc_post(hidden_states, residual, post, comb)

    def test_hc_head_matches_shape_contract(self):
        tokens, hc_mult, hidden = 2, 4, 6
        x = torch.randn(tokens, hc_mult, hidden)
        fn = torch.randn(hc_mult, hc_mult * hidden)
        scale = torch.ones(1)
        base = torch.zeros(hc_mult)

        y = hc_head(x, fn, scale, base, rms_norm_eps=1e-6, hc_eps=1e-6)

        self.assertEqual(tuple(y.shape), (tokens, hidden))

    def test_deepseek_v4_router_matches_noaux_bias_semantics(self):
        logits = torch.tensor(
            [
                [0.2, 1.0, -0.5, 0.7],
                [1.5, -0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)

        topk_weights, topk_ids, scores = dsv4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=2, dim=-1, sorted=False)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
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

        topk_weights, topk_ids, _ = dsv4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_indices_table=table,
            input_ids=input_ids,
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
    def test_deepseek_v4_fused_select_experts_returns_scores(self):
        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(2, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)

        topk_weights, topk_ids, scores = dsv4_select_experts(
            logits,
            top_k=6,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
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

    def test_packed_topk_router_logits_recover_weights_after_softmax(self):
        topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3], [0.55, 0.45]], dtype=torch.float32)

        packed = pack_topk_as_router_logits(topk_weights, topk_ids, num_experts=4)
        recovered = packed.softmax(dim=-1).gather(1, topk_ids.long())

        self.assertTrue(torch.allclose(recovered, topk_weights))

    def test_c4_ape_reorder_matches_overlap_window_layout(self):
        ape = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

        reordered = _deepseek_v4_reorder_c4_ape_2604(ape)
        expected = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [16, 17, 18, 19, 24, 25, 26, 27],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [20, 21, 22, 23, 28, 29, 30, 31],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(reordered, expected))


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
            speculative_algorithm=None,
            attention_use_fp4_indexer_cache=False,
            max_total_tokens=64 * 1024,
            chunked_prefill_size=256,
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
            pd_disaggregation_enabled=True,
            prefix_granularity=256,
            max_bs=2,
            context_len=1024,
        ),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 30,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    ).setup()
    schedule = build_cache_fields_by_producer_step(
        setup.spec.memory_plan, num_target_layers=3
    )
    assert schedule.step_count == 3
    assert all(schedule.fields_by_step)

    records = []
    backend = object.__new__(DeepseekV4AttentionBackend)
    backend.step_counter = SimpleNamespace(record_cache=lambda: records.append(True))
    output = object()
    for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.IDLE):
        assert backend.record_layer_cache_ready(output, mode) is output
    assert records == [True]
