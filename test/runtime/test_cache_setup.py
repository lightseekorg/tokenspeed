from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.attention.kv_cache.mha as mha_cache
from tokenspeed.runtime.cache.transfer.layout import combine_cache_transfer_layouts
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    CacheGroupSpec,
)


def _pool_over_new_arena(spec, config, *, num_layers: int, rank: int = 0):
    """Allocate an arena for ``spec`` and bind one compute view onto it."""
    arena = create_cache_arena(spec, device=config.device, enable_memory_saver=False)
    return create_cache_pool(
        spec,
        config,
        arena,
        num_layers=num_layers,
        rank=rank,
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


class _SyntheticHybridRecipe(CacheRecipe):
    """A minimal hybrid family, expressed the way a real one is.

    Its layer vocabulary and byte shapes are fixtures; everything else comes
    from the base pipeline. That a made-up family needs only these seams is
    the point -- the shared stages carry the rest.
    """

    family = "inkling"

    def __init__(
        self,
        *,
        layer_types,
        group_ids,
        num_draft_layers=0,
        windows=None,
        extra_state_group=None,
        cache_budget_bytes=2_048,
        **kwargs,
    ) -> None:
        super().__init__(
            server_args=SimpleNamespace(max_total_tokens=None),
            model_config=None,
            attn_config=SimpleNamespace(
                prefix_granularity=4, sliding_window_tokens=windows
            ),
            draft_model_config=None,
            draft_attn_config=None,
            cache_budget_bytes=cache_budget_bytes,
            decode_input_tokens=1,
            overlap_schedule_depth=0,
            **kwargs,
        )
        self._layer_types = tuple(layer_types)
        self._group_ids = tuple(group_ids)
        self._num_draft_layers = num_draft_layers
        self._extra_state_group = extra_state_group

    @property
    def layer_types(self):
        return self._layer_types

    @property
    def group_ids(self):
        return self._group_ids

    @property
    def num_draft_layers(self):
        return self._num_draft_layers

    @property
    def max_padding_fraction(self) -> float:
        return 1.0

    def fields_for_layer(self, layer_id, group_id, occurrence):
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.kv", f"slot.{occurrence}", (256,), "uint8"
            ),
        )

    def groups(self):
        groups = super().groups()
        if self._extra_state_group is None:
            return groups
        # A layer-external state group, like Inkling's checkpoint columns:
        # declared whole, its id written once.
        return groups + (
            (
                CacheGroupSpec(
                    group_id=self._extra_state_group,
                    retention="full_history",
                    sliding_window_tokens=None,
                    family="state",
                    checkpoint_granularity=self.prefix_granularity,
                ),
                (CacheFieldSpec("layer.0.state", "slot.0", (128,), "uint8"),),
            ),
        )


def _hybrid_setup_with_narrow_draft():
    """One KV group shared by target and draft, plus a state column."""
    return _SyntheticHybridRecipe(
        layer_types=("full_attention", "full_attention"),
        group_ids=("full_attention", "full_attention"),
        num_draft_layers=1,
        extra_state_group="state",
    ).setup()


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
    # The plan is the single source of field dtypes; no side channel.
    plan_dtypes = {
        field.field_id: field.dtype for field in setup.spec.memory_plan.fields
    }
    assert plan_dtypes["layer.0.conv"] == "bfloat16"
    assert plan_dtypes["layer.0.ssm"] == "float32"
    assert not hasattr(attn_config, "lcm_memory_plan")
    pool = _pool_over_new_arena(setup.spec, attn_config, num_layers=2)
    assert type(pool) is HybridMHATokenToKVPool
    assert pool.arena.buffer is not None


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
    pool = _pool_over_new_arena(setup.spec, attn_config, num_layers=2)
    assert type(pool) is MHATokenToKVPool
    assert pool.arena.runtime_contract.token_capacity == setup.spec.token_capacity
    with pytest.raises(TypeError, match="incompatible with MHAConfig"):
        _pool_over_new_arena(
            replace(setup.spec, family="kimi_k3"),
            attn_config,
            num_layers=2,
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
    pool = _pool_over_new_arena(setup.spec, attn_config, num_layers=2)
    assert type(pool) is MLATokenToKVPool
    assert pool.arena.runtime_contract.token_capacity == setup.spec.token_capacity


@pytest.mark.parametrize(
    ("family", "target_config"),
    (("mla", _mla_config), ("msa", _msa_config)),
)
def test_ordinary_recipe_uses_the_draft_attention_family(
    family: str,
    target_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.pd.cache_protocol import build_arena_cache_transfer_contract

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
    )
    # One arena, two compute views: the target's window and the draft's
    # continuation window, with no owner/view asymmetry to encode.
    arena = create_cache_arena(
        setup.spec, device=target_attn_config.device, enable_memory_saver=False
    )
    target_pool = create_cache_pool(
        target_spec,
        target_attn_config,
        arena,
        num_layers=2,
        rank=0,
    )
    draft_pool = create_cache_pool(
        draft_spec,
        draft_attn_config,
        arena,
        num_layers=1,
        rank=0,
        field_layer_offset=2,
    )

    assert type(draft_pool) is MHATokenToKVPool
    assert draft_pool.arena is target_pool.arena
    assert draft_pool.arena.buffer is target_pool.arena.buffer
    assert draft_pool.arena.runtime_contract is target_pool.arena.runtime_contract
    assert draft_pool.layerwise_load_tracker is None
    assert arena.field_ids() == {
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
    assert target_transfer_fields == arena.field_ids() - draft_field_ids
    assert draft_transfer_fields == draft_field_ids
    combined_layout = combine_cache_transfer_layouts(
        target_layout,
        draft_layout,
        group_ids=tuple(spec.group_id for spec in target_pool.arena.cache_group_specs),
    )
    assert len(combined_layout.consumers) == 3
    assert combined_layout.buffers == (target_pool.arena.buffer,)
    assert {
        field.field_id for group in combined_layout.groups for field in group.fields
    } == arena.field_ids()
    contract, base_addr = build_arena_cache_transfer_contract(target_pool.arena)
    assert contract.plan is target_pool.arena.plan
    assert base_addr == target_pool.arena.buffer.data_ptr()
    assert {field.field_id for field in contract.plan.fields} == arena.field_ids()

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

    # Sleep/wake repair visits both views; both name the one arena, so a
    # clear through either zeros the shared allocation exactly as well.
    draft_pool.clear_kv_buffers()
    assert not torch.count_nonzero(draft_pool.get_key_buffer(0))
    assert not torch.count_nonzero(target_pool.get_key_buffer(1))


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
            is_heterogeneous=True,
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
    setup = _SyntheticHybridRecipe(
        layer_types=("full_attention", "full_attention", "sliding_attention"),
        group_ids=("full_attention", "full_attention", "draft_swa"),
        num_draft_layers=2,
        windows=(None, None, 8),
        cache_budget_bytes=4_096,
    ).setup()

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
    published = {spec.group_id for spec in setup.spec.cache_group_specs}
    assert published == {"full_attention", "draft_swa"}


def test_union_contract_flows_draft_groups_to_scheduler_config() -> None:
    """No new contract: the one spec publishes draft-only
    groups as ordinary groups; pool publication and the scheduler config
    conversion carry them with their natural retention — the C++ side
    instantiates its existing SwaManager for them, no draft concept
    anywhere."""
    import torch
    from cache_pool_test_utils import MinimalCacheView

    from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups

    setup = _SyntheticHybridRecipe(
        layer_types=("full_attention", "full_attention", "sliding_attention"),
        group_ids=("full_attention", "full_attention", "draft_swa"),
        num_draft_layers=2,
        windows=(None, None, 8),
        cache_budget_bytes=4_096,
    ).setup()
    pool = MinimalCacheView(
        CacheArena(
            setup.spec.memory_plan,
            "cpu",
            cache_group_specs=setup.spec.cache_group_specs,
            token_capacity=setup.spec.token_capacity,
        ),
        torch.uint8,
        rank=0,
    )
    groups = {g.group_id: g for g in pool_to_cache_groups(pool)}
    assert set(groups) == {"full_attention", "draft_swa"}
    swa = groups["draft_swa"]
    assert swa.sliding_window_tokens == 8
    # Packing and page counts come from the ONE merged plan, carried across
    # the bridge by the contract rather than stamped onto the group specs.
    plan_group = setup.spec.memory_plan.group("draft_swa")
    contract = pool.arena.runtime_contract
    assert contract.group_packing["draft_swa"] == plan_group.cache_blocks_per_lcm_block
    assert swa.cache_blocks_per_lcm_block == plan_group.cache_blocks_per_lcm_block
    assert swa.total_pages == plan_group.page_count


def test_draft_view_maps_local_layer_ids_to_continuation_planes() -> None:
    """Tripwire for the draft window's DIRECTION and its bounds.

    A draft model numbers its layers locally, so local layer 0 must resolve to
    the merged plan's continuation plane (num_target_layers), never to the
    target's layer 0. And an id already carrying a global number must be
    REJECTED rather than offset a second time -- silently addressing another
    model's planes is how the KV of two models gets crossed.
    """
    from cache_pool_test_utils import MinimalCacheView

    class _Window(MinimalCacheView):
        """Just a layer window: the subject is _field_layer_id's arithmetic."""

        def __init__(self, *, first_layer: int, num_layers: int) -> None:
            self._field_layer_offset = first_layer
            self.layer_num = num_layers

    num_target_layers = 61
    draft = _Window(first_layer=num_target_layers, num_layers=3)

    assert draft._field_layer_id(0) == num_target_layers
    assert draft._field_layer_id(2) == num_target_layers + 2
    for outside in (num_target_layers, 3, -1):
        with pytest.raises(ValueError, match="outside this cache view"):
            draft._field_layer_id(outside)

    # The target view starts at 0, so its own ids pass through unchanged.
    target = _Window(first_layer=0, num_layers=num_target_layers)
    assert target._field_layer_id(7) == 7


# --- individual recipe seams ---


def test_qwen_mtp_padding_allowance_tracks_draft_planes() -> None:
    """The Qwen bound grows with the draft's mirrored K/V planes.

    p = 1 + 2 * draft_layers / full_attention_layers: no draft keeps the
    original 1.0, and each MTP layer buys headroom for the planes it adds.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
        QwenGDNRecipe,
    )

    def bound(*, full_attention_layers, draft_layers):
        recipe = QwenGDNRecipe.__new__(QwenGDNRecipe)
        recipe.__dict__["target_layer_types"] = (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ) * full_attention_layers
        recipe.draft_attn_config = object() if draft_layers else None
        recipe.draft_model_config = SimpleNamespace(num_attention_layers=draft_layers)
        return QwenGDNRecipe.max_padding_fraction.fget(recipe)

    for full_attention_layers in (6, 12):
        assert bound(full_attention_layers=full_attention_layers, draft_layers=0) == 1.0
        assert (
            abs(
                bound(full_attention_layers=full_attention_layers, draft_layers=1)
                - (1.0 + 2.0 / full_attention_layers)
            )
            < 1e-9
        )


def test_ordinary_profile_reserves_null_page_inside_budget() -> None:
    """Profiled capacity keeps the null page inside the budget.

    16_384 bytes at 16 bytes/token and P=64 buys 16 pages; one is the reserved
    null page, so 15 are schedulable.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        OrdinaryRecipe,
    )

    recipe = OrdinaryRecipe.__new__(OrdinaryRecipe)
    recipe.cache_budget_bytes = 16_384
    recipe.server_args = SimpleNamespace(max_total_tokens=None)
    recipe.attn_config = SimpleNamespace(
        prefix_granularity=64,
        cache_cell_size=lambda: 16,
        layer_types=(),
        sliding_window_tokens=None,
    )
    recipe.draft_attn_config = None
    recipe.model_config = SimpleNamespace(num_attention_layers=1)

    usable_pages = recipe.num_lcm_blocks(
        SimpleNamespace(lcm_block_bytes=1, prefix_granularity=64, group_packing=())
    )

    assert usable_pages == 15
    assert (usable_pages + 1) * 64 * 16 <= 16_384
