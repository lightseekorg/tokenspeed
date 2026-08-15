from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.attention.kv_cache.mha as mha_cache
from tokenspeed.runtime.cache.transfer.layout import combine_cache_transfer_layouts
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    build_hybrid_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    merge_continuation_layers,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    PagedCacheGroupSpec,
    build_paged_cache_group_specs,
)


def _mha_config() -> MHAConfig:
    return MHAConfig(
        device="cpu",
        backend_name="fa2",
        num_attention_heads=1,
        layer_types=(),
        kv_cache_mxfp8=False,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        prefix_granularity=64,
        kernel_page_size=64,
        kv_cache_quant_method="none",
        max_scheduled_tokens=128,
    )


def _mla_config() -> MLAConfig:
    return MLAConfig(
        device="cpu",
        backend_name="trtllm_mla",
        num_attention_heads=1,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=8,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        prefix_granularity=64,
        kernel_page_size=64,
        kv_cache_quant_method="none",
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        scaling=1.0,
        kv_cache_dim=6,
        max_scheduled_tokens=128,
    )


def _msa_config() -> MSAConfig:
    return MSAConfig(
        device="cpu",
        backend_name="msa",
        num_attention_heads=1,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        prefix_granularity=64,
        kernel_page_size=64,
        kv_cache_quant_method="none",
        compute_layer_types=("full_attention", "sparse_attention"),
        sparse_layer_ids=frozenset({1}),
        max_scheduled_tokens=128,
        index_head_dim=4,
        index_n_heads=1,
        index_block_size=64,
        index_topk_blocks=1,
        index_init_blocks=1,
        index_local_blocks=1,
    )


def _hybrid_setup_with_narrow_draft():
    # The recipe merges target and draft layers BEFORE the builder
    # (merge_continuation_layers); "state" here is a layer-external group,
    # plain tuple concatenation like Inkling's checkpoint columns.
    (
        fields,
        layer_types,
        group_ids,
        _,
        num_draft_layers,
    ) = merge_continuation_layers(
        fields=(
            CacheFieldSpec("full_attention", "layer.0.kv", "slot.0", (256,), 1),
            CacheFieldSpec("state", "layer.0.state", "slot.0", (128,), 1),
        ),
        layer_types=("full_attention",),
        group_ids=("full_attention",),
        draft_fields=(
            CacheFieldSpec("full_attention", "layer.0.kv", "slot.0", (256,), 1),
        ),
        draft_layer_types=("full_attention",),
        draft_group_ids=("full_attention",),
    )
    group_specs = (
        *build_paged_cache_group_specs(
            layer_types=layer_types,
            group_ids=group_ids,
            sliding_window_tokens=None,
            prefix_granularity=4,
        ),
        PagedCacheGroupSpec(
            group_id="state",
            retention="full_history",
            sliding_window_tokens=None,
            family="state",
            checkpoint_granularity=4,
        ),
    )
    return build_hybrid_cache_setup(
        family="inkling",
        server_args=SimpleNamespace(max_total_tokens=None),
        fields=fields,
        layer_types=layer_types,
        group_ids=group_ids,
        group_specs=group_specs,
        state_dtypes={},
        layer_kv_head_counts=None,
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=2_048,
        fixed_workspace_bytes=0,
        prefix_granularity=4,
        max_padding_fraction=1.0,
    )


def test_attention_configs_do_not_own_cache_setup() -> None:
    cache_setup_fields = {
        "conv_state_shape",
        "temporal_state_shape",
        "recurrent_state_shape",
        "conv_dtype",
        "ssm_dtype",
        "recurrent_dtype",
        "lcm_memory_plan",
        "layer_cache_group_ids",
        "token_capacity",
    }

    assert cache_setup_fields.isdisjoint(field.name for field in fields(MHAConfig))
    assert cache_setup_fields.isdisjoint(field.name for field in fields(MLAConfig))
    assert not hasattr(MHAConfig, "create_pool")
    assert not hasattr(MLAConfig, "create_pool")


def test_qwen_recipe_preserves_backend_kernel_page_size() -> None:
    text_config = SimpleNamespace(
        mamba2_cache_params=(
            (3, 2),
            (1, 2, 2),
            torch.bfloat16,
            torch.float32,
            (0,),
        ),
        linear_key_head_dim=1,
        linear_num_key_heads=1,
        linear_value_head_dim=1,
        linear_num_value_heads=1,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
    )
    attn_config = MHAConfig(
        device="cpu",
        backend_name="fa2",
        num_attention_heads=1,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        kv_cache_mxfp8=False,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        prefix_granularity=64,
        kernel_page_size=64,
        kv_cache_quant_method="none",
        max_scheduled_tokens=128,
    )
    server_args = SimpleNamespace(
        prefix_granularity=64,
        max_total_tokens=None,
        speculative_num_draft_tokens=0,
    )

    setup = prepare_cache_setup(
        family="qwen_gdn",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=16_384,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert server_args.prefix_granularity == 64
    assert attn_config.prefix_granularity == 64
    assert attn_config.kernel_page_size == 64
    assert setup.spec.memory_plan.prefix_granularity == 128
    assert setup.num_draft_layers == 0
    assert setup.spec.layer_group_ids == (
        f"{LINEAR_ATTENTION}_0",
        FULL_ATTENTION,
    )
    assert setup.spec.state_field_dtypes == {
        "layer.0.conv": torch.bfloat16,
        "layer.0.ssm": torch.float32,
    }
    assert not hasattr(attn_config, "lcm_memory_plan")
    pool = create_cache_pool(
        setup.spec,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is HybridMHATokenToKVPool
    assert pool.buffer is not None


@pytest.mark.parametrize(
    ("replay_enabled", "replay_supported", "expected_workspace_bytes"),
    ((False, True, 192), (True, False, 192), (True, True, 64)),
)
def test_qwen_recipe_sizes_verify_workspace_for_replay_ssm(
    monkeypatch,
    replay_enabled: bool,
    replay_supported: bool,
    expected_workspace_bytes: int,
) -> None:
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.attention.gdn_replay_commit_supported",
        lambda dtype: replay_supported,
    )
    text_config = SimpleNamespace(
        mamba2_cache_params=(
            (2, 2),
            (1, 2, 2),
            torch.bfloat16,
            torch.float32,
            (0,),
        )
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
    )
    attn_config = MHAConfig(
        device="cuda",
        backend_name="fa2",
        num_attention_heads=1,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        kv_cache_mxfp8=False,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        prefix_granularity=64,
        kernel_page_size=64,
        kv_cache_quant_method="none",
        max_scheduled_tokens=128,
    )
    draft_config = replace(
        attn_config,
        layer_types=(FULL_ATTENTION,),
    )
    server_args = SimpleNamespace(
        block_size=64,
        max_total_tokens=None,
        speculative_num_draft_tokens=3,
        enable_replay_ssm=replay_enabled,
    )

    setup = prepare_cache_setup(
        family="qwen_gdn",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=SimpleNamespace(num_attention_layers=1),
        draft_attn_config=draft_config,
        cache_budget_bytes=16_384,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.fixed_workspace_bytes == expected_workspace_bytes
    assert attn_config.replay_ssm is (replay_enabled and replay_supported)


def test_ordinary_mha_reserves_null_parent_within_cache_budget() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = _mha_config()
    server_args = SimpleNamespace(max_total_tokens=None)

    setup = prepare_cache_setup(
        family="mha",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=16_384,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.spec.family == "mha"
    assert setup.spec.memory_plan.prefix_granularity == 64
    assert setup.spec.memory_plan.num_lcm_blocks == 15
    assert setup.spec.memory_plan.arena_bytes <= 16_384
    assert setup.spec.token_capacity == 960
    assert setup.num_draft_layers == 0
    pool = create_cache_pool(
        setup.spec,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is MHATokenToKVPool
    assert pool.runtime_contract.token_capacity == setup.spec.token_capacity
    with pytest.raises(TypeError, match="incompatible with MHAConfig"):
        create_cache_pool(
            replace(setup.spec, family="kimi_k3"),
            attn_config,
            num_layers=2,
            rank=0,
            enable_memory_saver=False,
        )


def test_ordinary_mla_reserves_null_parent_within_cache_budget() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = _mla_config()
    server_args = SimpleNamespace(max_total_tokens=None)

    setup = prepare_cache_setup(
        family="mla",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=24_576,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.spec.family == "mla"
    assert setup.spec.memory_plan.prefix_granularity == 64
    assert setup.spec.memory_plan.num_lcm_blocks == 15
    assert setup.spec.memory_plan.arena_bytes <= 24_576
    assert setup.spec.token_capacity == 960
    assert setup.num_draft_layers == 0
    pool = create_cache_pool(
        setup.spec,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is MLATokenToKVPool
    assert pool.runtime_contract.token_capacity == setup.spec.token_capacity


@pytest.mark.parametrize(
    ("family", "target_config"),
    (("mla", _mla_config), ("msa", _msa_config)),
)
def test_ordinary_recipe_uses_the_draft_attention_family(
    family: str,
    target_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.pd.cache_protocol import build_pool_cache_transfer_contract

    model_config = SimpleNamespace(num_attention_layers=2, hf_config=SimpleNamespace())
    draft_model_config = SimpleNamespace(
        num_attention_layers=1, hf_config=SimpleNamespace()
    )
    target_attn_config = replace(target_config(), pd_disaggregation_enabled=True)
    draft_attn_config = replace(_mha_config(), pd_disaggregation_enabled=True)

    setup = prepare_cache_setup(
        family=family,
        server_args=SimpleNamespace(max_total_tokens=None),
        model_config=model_config,
        attn_config=target_attn_config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=65_536,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    # One arena, two concrete compute views: the MHA draft's fields are
    # continuation layers in the merged plan, but an MLA/MSA target pool must
    # not interpret them as target-shaped fields.
    assert setup.num_draft_layers == 1
    assert setup.num_target_layers == 2
    with pytest.raises(ValueError, match="bounds must be non-negative"):
        setup.spec.layer_view(first_layer=-1, num_layers=1)
    with pytest.raises(ValueError, match="exceeds the merged"):
        setup.spec.layer_view(first_layer=2, num_layers=2)
    draft_field_ids = {
        field.field_id
        for field in setup.spec.memory_plan.fields
        if field.field_id.startswith("layer.2.")
    }
    assert draft_field_ids  # the draft layer's fields are planned

    target_spec = setup.spec.layer_view(first_layer=0, num_layers=2)
    draft_spec = setup.spec.layer_view(
        first_layer=2,
        num_layers=1,
        family="mha",
        publish_runtime_contract=False,
    )
    target_pool = create_cache_pool(
        target_spec,
        target_attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    unbound_pool = CachePool(
        setup.spec.pool_size,
        target_pool.dtype,
        "cpu",
        setup.spec.memory_plan.prefix_granularity,
        0,
        setup.spec.memory_plan,
    )
    with pytest.raises(ValueError, match="must bind its fields before a view"):
        CachePool(
            setup.spec.pool_size,
            target_pool.dtype,
            "cpu",
            setup.spec.memory_plan.prefix_granularity,
            0,
            setup.spec.memory_plan,
            backing_pool=unbound_pool,
        )
    with pytest.raises(ValueError, match="must inherit, not republish"):
        CachePool(
            setup.spec.pool_size,
            target_pool.dtype,
            "cpu",
            setup.spec.memory_plan.prefix_granularity,
            0,
            setup.spec.memory_plan,
            paged_cache_group_specs=target_spec.paged_cache_group_specs,
            backing_pool=target_pool,
        )
    with pytest.raises(ValueError, match="only supported by ordinary MHA or MLA pools"):
        create_cache_pool(
            target_spec,
            _msa_config(),
            num_layers=2,
            rank=0,
            enable_memory_saver=False,
            backing_pool=target_pool,
        )
    draft_pool = create_cache_pool(
        draft_spec,
        draft_attn_config,
        num_layers=1,
        rank=0,
        enable_memory_saver=False,
        field_layer_offset=2,
        backing_pool=target_pool,
    )

    assert type(draft_pool) is MHATokenToKVPool
    assert draft_pool.buffer is target_pool.buffer
    assert draft_pool._fields is target_pool._fields
    assert draft_pool.runtime_contract is target_pool.runtime_contract
    assert draft_pool.layerwise_load_tracker is None
    assert set(target_pool._fields) == {
        field.field_id for field in setup.spec.memory_plan.fields
    }
    target_layout = target_pool.cache_transfer_layout()
    draft_layout = draft_pool.cache_transfer_layout()
    target_transfer_fields = {
        field_id for consumer in target_layout.consumers for field_id in consumer
    }
    draft_transfer_fields = {
        field_id for consumer in draft_layout.consumers for field_id in consumer
    }
    assert target_transfer_fields == set(target_pool._fields) - draft_field_ids
    assert draft_transfer_fields == draft_field_ids
    combined_layout = combine_cache_transfer_layouts(
        target_layout,
        draft_layout,
        group_ids=tuple(spec.group_id for spec in target_pool.paged_cache_group_specs),
    )
    assert len(combined_layout.consumers) == 3
    assert combined_layout.buffers == (target_pool.buffer,)
    assert {
        field.field_id for group in combined_layout.groups for field in group.fields
    } == set(target_pool._fields)
    contract, base_addr = build_pool_cache_transfer_contract(target_pool)
    assert contract.plan is target_pool.plan
    assert base_addr == target_pool.buffer.data_ptr()
    assert len(contract.field_dtypes) == len(target_pool._fields)

    target_last_layer = target_pool.get_key_buffer(1).clone()

    def _store_kv_cache(cache_k, cache_v, k_buffer, v_buffer, loc, *, enable_pdl):
        del enable_pdl
        k_buffer[loc] = cache_k
        v_buffer[loc] = cache_v

    monkeypatch.setattr(mha_cache, "store_kv_cache", _store_kv_cache)
    cache_k = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
    cache_v = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)
    draft_pool.set_kv_buffer(
        SimpleNamespace(layer_id=0),
        torch.tensor([0]),
        cache_k,
        cache_v,
    )
    assert torch.equal(draft_pool.get_key_buffer(0)[0], cache_k[0])
    assert torch.equal(draft_pool.get_value_buffer(0)[0], cache_v[0])
    assert torch.equal(target_pool.get_key_buffer(1), target_last_layer)

    # Sleep/wake repair visits both objects; only the allocation owner clears.
    draft_pool.clear_kv_buffers()
    assert torch.equal(draft_pool.get_key_buffer(0)[0], cache_k[0])
    target_pool.clear_kv_buffers()
    assert not torch.count_nonzero(draft_pool.get_key_buffer(0))


def test_heterogeneous_draft_guards_fail_fast() -> None:
    from tokenspeed.runtime.layers.attention.registry import (
        _create_draft_components,
        _resolve_heterogeneous_draft_family,
    )

    assert _resolve_heterogeneous_draft_family("mla", "mha") == "mha"
    assert _resolve_heterogeneous_draft_family("kimi_k3", "mla") == "mla"
    with pytest.raises(RuntimeError, match="require an MHA draft"):
        _resolve_heterogeneous_draft_family("mha", "mla")
    with pytest.raises(RuntimeError, match="support ordinary drafts only"):
        _create_draft_components(
            server_args=None,
            model_config=SimpleNamespace(num_attention_layers=1),
            config=object(),
            pool=None,
            cache_spec=object(),
            num_target_layers=1,
            full_attn_backend_name=None,
            is_hybrid_linear=True,
            is_kda=False,
            is_inkling=False,
        )


def test_deepseek_v4_draft_pd_is_rejected_for_an_ordinary_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed.runtime.layers.attention.registry as registry

    monkeypatch.setattr(
        registry,
        "is_deepseek_v4",
        lambda config: getattr(config, "is_deepseek_v4", False),
    )
    server_args = SimpleNamespace(
        attention_backend=None,
        drafter_attention_backend=None,
        disaggregation_mode="prefill",
    )
    target = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=("LlamaForCausalLM",),
            is_deepseek_v4=False,
        )
    )
    draft = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=("DeepseekV4ForCausalLMNextN",),
            is_deepseek_v4=True,
        )
    )

    with pytest.raises(NotImplementedError, match="target-only"):
        registry.create_attn_components(
            server_args,
            target,
            gpu_id=0,
            rank=0,
            gpu_memory=0,
            draft_model_config=draft,
        )


def test_hybrid_draft_layers_share_the_merged_plan() -> None:
    setup = _hybrid_setup_with_narrow_draft()

    # One big model: the draft layer's field is planned as a continuation
    # layer in the SAME plan; page ids come from the same shared groups.
    assert setup.num_draft_layers == 1
    plan = setup.spec.memory_plan
    target_field = plan.field("layer.0.kv")
    draft_field = plan.field("layer.1.kv")
    assert draft_field.group_id == target_field.group_id
    assert (
        plan.group(draft_field.group_id).page_count
        == plan.group(target_field.group_id).page_count
    )


def test_hybrid_draft_only_sliding_group_packs_by_ratio() -> None:
    """A draft-only sliding-window group (absent from a KDA-style target
    plan) participates in the draft solve with its own byte-ratio packing;
    shared groups keep the target's pinned packing.
    """
    (
        fields,
        layer_types,
        group_ids,
        _,
        num_draft_layers,
    ) = merge_continuation_layers(
        fields=(
            CacheFieldSpec(
                "full_attention",
                "layer.0.kv",
                "slot.0",
                (256,),
                1,
            ),
        ),
        layer_types=("full_attention",),
        group_ids=("full_attention",),
        draft_fields=(
            CacheFieldSpec(
                "full_attention",
                "layer.0.kv",
                "slot.0",
                (256,),
                1,
            ),
            # Draft-only sliding-window layers: not present in the target plan.
            CacheFieldSpec(
                "draft_swa",
                "layer.1.kv",
                "slot.1",
                (256,),
                1,
            ),
        ),
        draft_layer_types=("full_attention", "sliding_attention"),
        draft_group_ids=("full_attention", "draft_swa"),
    )
    # ONE spec derivation over the merged layers, per-layer windows.
    group_specs = build_paged_cache_group_specs(
        layer_types=layer_types,
        group_ids=group_ids,
        sliding_window_tokens=(None, None, 8),
        prefix_granularity=4,
    )
    setup = build_hybrid_cache_setup(
        family="inkling",
        server_args=SimpleNamespace(max_total_tokens=None),
        fields=fields,
        layer_types=layer_types,
        group_ids=group_ids,
        group_specs=group_specs,
        state_dtypes={},
        layer_kv_head_counts=None,
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=4_096,
        fixed_workspace_bytes=0,
        prefix_granularity=4,
        max_padding_fraction=1.0,
    )

    # One big model: both draft layers are continuation layers (global
    # layers 1 and 2) of the one merged plan. The full_attention group is
    # shared; the draft-only sliding group is planned alongside with its
    # own packing, and its spec joins the ONE published spec set.
    assert setup.num_draft_layers == 2
    plan = setup.spec.memory_plan
    assert plan.field("layer.1.kv").group_id == "full_attention"
    assert plan.field("layer.2.kv").group_id == "draft_swa"
    assert plan.group("draft_swa").cache_blocks_per_lcm_block >= 1
    assert setup.spec.layer_group_ids == (
        "full_attention",
        "full_attention",
        "draft_swa",
    )
    published = {spec.group_id for spec in setup.spec.paged_cache_group_specs}
    assert published == {"full_attention", "draft_swa"}


def test_union_contract_flows_draft_groups_to_scheduler_config() -> None:
    """No new contract: the one spec publishes draft-only
    groups as ordinary groups; pool publication and the scheduler config
    conversion carry them with their natural retention — the C++ side
    instantiates its existing SwaManager for them, no draft concept
    anywhere."""
    import torch

    from tokenspeed.runtime.engine.scheduler_utils import pool_to_paged_cache_groups
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool

    (
        fields,
        layer_types,
        group_ids,
        _,
        num_draft_layers,
    ) = merge_continuation_layers(
        fields=(CacheFieldSpec("full_attention", "layer.0.kv", "slot.0", (256,), 1),),
        layer_types=("full_attention",),
        group_ids=("full_attention",),
        draft_fields=(
            CacheFieldSpec("full_attention", "layer.0.kv", "slot.0", (256,), 1),
            CacheFieldSpec("draft_swa", "layer.1.kv", "slot.1", (256,), 1),
        ),
        draft_layer_types=("full_attention", "sliding_attention"),
        draft_group_ids=("full_attention", "draft_swa"),
    )
    group_specs = build_paged_cache_group_specs(
        layer_types=layer_types,
        group_ids=group_ids,
        sliding_window_tokens=(None, None, 8),
        prefix_granularity=4,
    )
    setup = build_hybrid_cache_setup(
        family="inkling",
        server_args=SimpleNamespace(max_total_tokens=None),
        fields=fields,
        layer_types=layer_types,
        group_ids=group_ids,
        group_specs=group_specs,
        state_dtypes={},
        layer_kv_head_counts=None,
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=4_096,
        fixed_workspace_bytes=0,
        prefix_granularity=4,
        max_padding_fraction=1.0,
    )
    pool = CachePool(
        size=setup.spec.pool_size,
        dtype=torch.uint8,
        device="cpu",
        prefix_granularity=4,
        rank=0,
        memory_plan=setup.spec.memory_plan,
        paged_cache_group_specs=setup.spec.paged_cache_group_specs,
        token_capacity=setup.spec.token_capacity,
    )
    groups = {g.group_id: g for g in pool_to_paged_cache_groups(pool)}
    assert set(groups) == {"full_attention", "draft_swa"}
    swa = groups["draft_swa"]
    assert swa.sliding_window_tokens == 8
    # Packing and page counts come from the ONE merged plan.
    plan_group = setup.spec.memory_plan.group("draft_swa")
    assert swa.cache_blocks_per_lcm_block == plan_group.cache_blocks_per_lcm_block
    assert swa.total_pages == plan_group.page_count


def test_draft_view_maps_local_layer_ids_to_continuation_planes() -> None:
    """Tripwire for the draft layer-map DIRECTION: a draft model's local
    layer 0 must resolve to the merged pool's continuation plane
    (num_target_layers), never to the target's layer 0. The inverse map
    (the hybrid {global: pool_idx} convention) silently corrupts the
    target's first layers with every draft KV write."""
    from tokenspeed.runtime.layers.attention.kv_cache.base import LayerMappedKVPool

    class _FakePool:
        page_size = 4

        def get_key_buffer(self, layer_id: int) -> int:
            return layer_id

    num_target_layers = 61
    draft_pool = LayerMappedKVPool(
        _FakePool(),
        [num_target_layers + local for local in range(1)],
        layer_map={local: num_target_layers + local for local in range(1)},
    )
    # Local draft layer 0 -> global continuation plane 61.
    assert draft_pool.get_key_buffer(0) == num_target_layers
    # A layer already carrying its global id (V4 MTP convention) passes through.
    assert draft_pool.get_key_buffer(num_target_layers) == num_target_layers

    # The hybrid default stays the inverse: global sparse ids -> compact slots.
    hybrid_pool = LayerMappedKVPool(_FakePool(), [3, 7, 11])
    assert hybrid_pool.get_key_buffer(7) == 1


def test_draft_view_maps_inkling_conv_checkpoint_accessors() -> None:
    """Same tripwire for the Inkling conv-checkpoint accessors: the
    __getattr__ pass-through would resolve them against the target's
    layers 0..n, silently restoring/publishing the wrong planes."""
    from tokenspeed.runtime.layers.attention.kv_cache.base import LayerMappedKVPool

    class _FakePool:
        page_size = 4

        def kvconv_checkpoint_buffers(self, layer_id: int):
            return ("kv", layer_id)

        def hiddenconv_checkpoint_buffer(self, layer_id: int, component: str):
            return ("hidden", layer_id, component)

    num_target_layers = 66
    draft_pool = LayerMappedKVPool(
        _FakePool(),
        [num_target_layers + local for local in range(3)],
        layer_map={local: num_target_layers + local for local in range(3)},
    )
    assert draft_pool.kvconv_checkpoint_buffers(0) == ("kv", 66)
    assert draft_pool.hiddenconv_checkpoint_buffer(2, "mlpconv") == (
        "hidden",
        68,
        "mlpconv",
    )
