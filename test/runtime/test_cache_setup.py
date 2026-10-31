import functools
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

import tokenspeed.runtime.layers.attention.kv_cache.mha as mha_cache
from tokenspeed.runtime.cache.transfer.layout import (
    combine_cache_transfer_layouts,
    select_layer_fields,
)
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
    Qwen4ExpPLEBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
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
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    cache_recipe,
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.registry import _prepare_fixed_workspaces
from tokenspeed.runtime.pd.host_peer import build_host_peer  # noqa: E402

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 3])
def test_glm_workspace_budget_includes_kda_and_shared_kpool_tails(width):
    from test.runtime.test_glm53_flash_cache_spec import _recipe

    from tokenspeed.runtime.layers.attention import registry

    recipe = _recipe(
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        draft_layers=int(width > 1),
    )
    config = replace(recipe.attn_config, device="cuda", max_bs=2)
    recipe.attn_config = config
    if recipe.draft_attn_config is not None:
        recipe.draft_attn_config = replace(
            recipe.draft_attn_config, device="cuda", max_bs=2
        )
    recipe.probe_batch_rows = 4
    recipe.server_args.disable_kda_prefill_graph = True
    recipe.server_args.kda_backend = "auto"
    linear = config.component(LinearAttnConfig)
    num_layers = recipe.model_config.num_attention_layers
    model = SimpleNamespace(
        num_attention_layers=num_layers,
        attention_arch=registry.AttentionArch.DSA,
        hf_config=SimpleNamespace(
            architectures=["Glm53FlashForConditionalGeneration"],
            full_attention_layer_ids=[
                layer for layer in range(num_layers) if layer not in linear.layer_ids
            ],
        ),
    )
    setup = recipe.setup()
    arena = create_cache_arena(
        setup.spec, device=config.device, enable_memory_saver=False
    )
    backend, pool = registry._create_target_components(
        server_args=recipe.server_args,
        model_config=model,
        config=config,
        cache_spec=setup.spec.layer_view(first_layer=0, num_layers=num_layers),
        arena=arena,
        rank=0,
        full_attn_backend_name="dsa",
        linear_attention="kda",
        is_inkling=False,
        backend=None,
    )
    backend.set_cache_pool(pool)
    # The tail allocation is shared by target/draft views; count its storage
    # once, together with the verify buffers owned by the target backend.
    tail_bytes = pool._kpool_tail_workspace.storage.nbytes
    verify_bytes = backend.preallocate_verify_workspace(config.max_bs, width)
    assert setup.fixed_workspace_bytes == tail_bytes + verify_bytes


@pytest.mark.parametrize(
    "uses_paged_state_verify,expected_bytes",
    [(True, 128), (True, 0), (False, 128)],
)
def test_verify_preparation_calls_root_with_existing_budget_gate(
    uses_paged_state_verify, expected_bytes
):
    backend = Mock(spec=AttentionBackend)
    backend.preallocate_verify_workspace.return_value = 128
    draft_backend = Mock(spec=AttentionBackend)
    kwargs = dict(
        server_args=SimpleNamespace(speculative_num_draft_tokens=3),
        config=SimpleNamespace(max_bs=4, qcp_size=1),
        backend=backend,
        draft_backend=draft_backend,
        uses_paged_state_verify=uses_paged_state_verify,
        is_inkling=False,
    )
    _prepare_fixed_workspaces(**kwargs, expected_bytes=expected_bytes)
    if uses_paged_state_verify and expected_bytes:
        backend.preallocate_verify_workspace.assert_called_once_with(4, 3)
    else:
        backend.preallocate_verify_workspace.assert_not_called()
    draft_backend.preallocate_verify_workspace.assert_not_called()


def _model_wide_kwargs(**overrides) -> dict:
    """The AttnConfig (model-wide) tier the test configs share."""
    kwargs = dict(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=64,
        context_len=1024,
        max_bs=2,
    )
    kwargs.update(overrides)
    return kwargs


def _mha_config() -> AttnConfig:
    spec = MHAConfig(
        backend_name="fa2",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        cache_layer_types=(),
    )
    return AttnConfig(components=(spec,), **_model_wide_kwargs())


@pytest.mark.parametrize(
    "is_draft,width,num_layers,layerwise,ring_rows",
    [(False, 1, 3, True, 4), (True, 4, 1, False, 7)],
)
def test_inkling_construction_matches_ring_reservation(
    is_draft, width, num_layers, layerwise, ring_rows
):
    from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
        InklingAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import (
        _conv_ring_bytes,
    )

    config = _mha_config()
    spec = replace(
        config.component(MHAConfig),
        num_attention_heads=4,
        num_kv_heads=4,
        attn_tp_size=2,
    )
    config = replace(
        config,
        components=(spec,),
        is_draft=is_draft,
        speculative_num_draft_tokens=width,
    )
    text = SimpleNamespace(
        sconv_kernel_size=4,
        num_key_value_heads=4,
        head_dim=2,
        hidden_size=8,
    )
    inner = AttentionBackend(config, spec)
    backend = InklingAttnBackend.from_config(
        inner,
        text,
        config,
        num_layers=num_layers,
        is_draft=is_draft,
        enable_layerwise_cache_ready=layerwise,
    )
    assert backend.conv_pool.conv_state.shape == (num_layers, 4, ring_rows, 24)
    assert backend.conv_spec_num_tokens == width
    assert backend.enable_layerwise_cache_ready is layerwise
    assert backend.fixed_workspace_bytes() == _conv_ring_bytes(
        text_config=text,
        attn_config=config,
        num_layers=num_layers,
        spec_tokens=width,
    )


def _mla_config() -> AttnConfig:
    spec = MLAConfig(
        backend_name="trtllm_mla",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=8,
        attn_tp_size=1,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        scaling=1.0,
        kv_cache_dim=6,
    )
    return AttnConfig(components=(spec,), **_model_wide_kwargs())


def _mxfp8_mha_config() -> AttnConfig:
    config = _mha_config()
    # MXFP8 scales cover 32 head_dim elements.
    spec = replace(config.component(MHAConfig), head_dim=32)
    return replace(
        config,
        components=(spec,),
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_mxfp8=True,
    )


def _msa_config() -> AttnConfig:
    spec = MSAConfig(
        backend_name="msa",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        compute_layer_types=("full_attention", "sparse_attention"),
        sparse_layer_ids=frozenset({1}),
        index_head_dim=4,
        index_n_heads=1,
        index_topk_blocks=1,
        index_init_blocks=1,
        index_local_blocks=1,
    )
    return AttnConfig(components=(spec,), **_model_wide_kwargs())


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
        probe_batch_rows=None,
        **kwargs,
    ) -> None:
        super().__init__(
            server_args=SimpleNamespace(kv_offload_config=None, max_total_tokens=None),
            model_config=None,
            attn_config=_ns_config(
                pd_disaggregation_enabled=False,
                spec=SimpleNamespace(sliding_window_tokens=windows),
            ),
            draft_model_config=None,
            draft_attn_config=None,
            cache_budget_bytes=cache_budget_bytes,
            probe_batch_rows=probe_batch_rows,
            decode_input_tokens=1,
            overlap_schedule_depth=0,
            **kwargs,
        )
        self._layer_types = tuple(layer_types)
        self._group_ids = tuple(group_ids)
        self._num_draft_layers = num_draft_layers
        self._extra_state_group = extra_state_group

    @property
    def prefix_granularity(self):
        return 4

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
                    attn_kind="state",
                    sliding_window_tokens=None,
                    replayable=False,
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


def _ns_config(*, spec, **fields):
    """Namespace micro-stub honoring the component(cls) query with one spec."""
    return SimpleNamespace(
        components=(spec,), component=lambda cls, _s=spec: _s, **fields
    )


def _tiny_linear_attn():
    """Tiny GDN component whose state payloads (conv 8B, ssm 8B) keep the
    shared planes aligned with the 512-byte KV page stride."""
    from tokenspeed.runtime.layers.attention.configs.linear_attn import (
        LinearAttnConfig,
    )

    return LinearAttnConfig(
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=1,
        head_v_dim=1,
        conv_kernel_size=2,
        layer_ids=(0,),
        tp_size=1,
    )


def test_qwen_recipe_preserves_backend_kernel_page_size() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=SimpleNamespace()),
    )
    attn_config = AttnConfig(
        components=(
            MHAConfig(
                backend_name="fa2",
                num_attention_heads=1,
                num_kv_heads=1,
                head_dim=2,
                attn_tp_size=1,
                cache_layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
            ),
            _tiny_linear_attn(),
        ),
        **_model_wide_kwargs(),
    )
    server_args = SimpleNamespace(
        kv_offload_config=None,
        max_total_tokens=None,
        speculative_num_draft_tokens=0,
        speculative_eagle_topk=1,
        speculative_draft_tree=False,
    )

    setup = prepare_cache_setup(
        family="qwen_gdn",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=16_384,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert attn_config.kernel_page_size == 64
    assert setup.spec.memory_plan.prefix_granularity == 128
    assert setup.num_draft_layers == 0
    # The plan's field declarations are the single record of layer -> group.
    plan_groups = {
        field.field_id: field.group_id for field in setup.spec.memory_plan.fields
    }
    assert plan_groups["layer.0.conv"] == f"{LINEAR_ATTENTION}_0"
    assert plan_groups["layer.1.k"] == FULL_ATTENTION
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
    (
        "replay_enabled",
        "replay_supported",
        "topk",
        "draft_tokens",
        "expected_workspace_bytes",
    ),
    # Non-replay stages conv+ssm for 8 verify rows: 8 * (8 + 8). Replay: 64
    # conv staging bytes plus the captured payload (6 rows of 7 bf16
    # channels) and the fp32 A_log/dt_bias pairs -- 64 + 84 + 16.
    # A replayed draft tree (topk 2) adds one 8-byte ssm state per draft position: 2 * 3 * 8.
    (
        (False, True, 1, 3, 128),
        (True, False, 1, 3, 128),
        (True, True, 1, 3, 164),
        (False, True, 2, 3, 128),
        (True, True, 2, 3, 212),
        (True, True, 2, 7, 452),
        (True, True, 2, 8, 384),
        (True, True, 2, 16, 736),
        (True, True, 2, 17, 1052),
    ),
)
def test_qwen_recipe_sizes_verify_workspace_for_replay_ssm(
    monkeypatch,
    replay_enabled: bool,
    replay_supported: bool,
    topk: int,
    draft_tokens: int,
    expected_workspace_bytes: int,
) -> None:
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.attention.gdn.gdn_replay_commit_supported",
        lambda dtype: replay_supported,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=SimpleNamespace()),
    )
    target_spec = MHAConfig(
        backend_name="fa2",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        cache_layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
    )
    attn_config = AttnConfig(
        components=(target_spec, _tiny_linear_attn()),
        **_model_wide_kwargs(device="cuda"),
    )
    draft_config = replace(
        attn_config,
        components=(replace(target_spec, cache_layer_types=(FULL_ATTENTION,)),),
    )
    server_args = SimpleNamespace(
        kv_offload_config=None,
        block_size=64,
        max_total_tokens=None,
        speculative_num_draft_tokens=draft_tokens,
        speculative_eagle_topk=topk,
        speculative_draft_tree=topk > 1,
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
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.fixed_workspace_bytes == expected_workspace_bytes
    linear_attn = attn_config.component(LinearAttnConfig)
    assert linear_attn is not None
    assert linear_attn.replay_ssm is (replay_enabled and replay_supported)
    assert linear_attn.draft_tree is (topk > 1)


@pytest.mark.parametrize("speculative,width", [(False, 1), (True, 1), (True, 3)])
@pytest.mark.parametrize("ple_enabled", (False, True))
@pytest.mark.parametrize("role", ("null", "prefill", "decode"))
def test_qwen4_exp_workspace_budget_includes_preallocated_ple_commit_rows(
    speculative: bool, width: int, ple_enabled: bool, role: str
) -> None:
    text_config = SimpleNamespace(
        ple_layer_ids=(0, 1) if ple_enabled else (),
        short_conv_layer_ids=(0, 1),
        ngram_context_len=2,
        short_conv_state_shape=(4, 3),
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
        hf_text_config=text_config,
        num_attention_layers=2,
    )
    target_spec = MHAConfig(
        backend_name="fa2",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        cache_layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
    )
    attn_config = AttnConfig(
        components=(target_spec, _tiny_linear_attn()),
        speculative_num_draft_tokens=width,
        **_model_wide_kwargs(),
    )
    attn_config = replace(attn_config, pd_disaggregation_enabled=role != "null")
    draft_config = (
        replace(
            attn_config,
            components=(replace(target_spec, cache_layer_types=(FULL_ATTENTION,)),),
        )
        if speculative
        else None
    )
    server_args = SimpleNamespace(
        kv_offload_config=None,
        block_size=64,
        max_total_tokens=None,
        speculative_num_draft_tokens=width,
        speculative_eagle_topk=1,
        speculative_draft_tree=False,
        enable_replay_ssm=False,
        disaggregation_mode=role,
    )
    setup = prepare_cache_setup(
        family="qwen4_exp",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=1, hf_text_config=SimpleNamespace())
            if speculative
            else None
        ),
        draft_attn_config=draft_config,
        cache_budget_bytes=1 << 20,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    if not speculative or role == "prefill":
        assert setup.fixed_workspace_bytes == 0
        return

    backend = Qwen4ExpPLEBackend(attn_config, target_spec)
    backend.set_cache_pool(
        _pool_over_new_arena(
            setup.spec, attn_config, num_layers=len(setup.spec.layer_types), rank=0
        )
    )
    ple_bytes = backend.preallocate_verify_workspace(
        max_bs=attn_config.max_bs, draft_token_num=width
    )
    if width == 1:
        assert ple_bytes == setup.fixed_workspace_bytes == 0
        # Exercise the startup budget check with the real recipe's result.
        root = Qwen4ExpBackend(
            attn_config, AttentionBackend(attn_config, target_spec), backend, None
        )
        _prepare_fixed_workspaces(
            server_args=server_args,
            config=attn_config,
            backend=root,
            draft_backend=None,
            uses_paged_state_verify=True,
            is_inkling=False,
            expected_bytes=setup.fixed_workspace_bytes,
        )
        return
    # Eight verify rows: shared int64[2] context plus two bf16[4, 3]
    # conv states; the commit holds two int64 ids per request per layer.
    assert ple_bytes == (8 * (16 + 2 * 24) + 2 * 2 * 2 * 8 if ple_enabled else 0)
    assert setup.fixed_workspace_bytes == 128 + ple_bytes  # GDN conv + SSM: 128 B.


def test_parent_is_its_planes_whatever_the_group_packings() -> None:
    # Coprime packings, as Qwen4-Exp's PLE and QSA side planes get: a parent
    # rounded up to their LCM (5376 bytes here) would hold 4096 bytes no field
    # addresses. At Qwen3.8-Flash-Next TP1 that rounding made a 255 MB parent
    # 54 GB.
    def state(group_id: str) -> CacheGroupSpec:
        return CacheGroupSpec(
            group_id=group_id,
            attn_kind="state",
            sliding_window_tokens=None,
            replayable=False,
        )

    layout = pack(
        (
            (
                state("ssm"),
                (CacheFieldSpec("ssm.state", "plane.ssm", (4096,), "uint8"),),
            ),
            (
                state("side_a"),
                (CacheFieldSpec("side_a.key", "plane.a", (256,), "uint8"),),
            ),
            (
                state("side_b"),
                (CacheFieldSpec("side_b.key", "plane.b", (256,), "uint8"),),
            ),
        ),
        prefix_granularity=16,
        cache_blocks_per_lcm_block={"ssm": 1, "side_a": 3, "side_b": 7},
        alignment=256,
        max_padding_fraction=float("inf"),
    )
    plan = layout.bind(3)

    planes = sum(plane.bytes_per_lcm_block for plane in plan.planes)
    assert planes == 4096 + 3 * 256 + 7 * 256
    assert plan.lcm_block_bytes == planes
    assert plan.arena_bytes == 4 * planes
    # Every group's last page still ends inside the arena.
    for group in plan.groups:
        _, fields = plan.page_field_layout(group.group_id)
        for base, stride, size in fields:
            assert base + (group.page_count - 1) * stride + size <= plan.arena_bytes


@pytest.mark.parametrize(
    ("full_layers", "sliding_layers", "usable_pages"),
    ((2, 0, 15), (1, 4, 7)),
)
def test_ordinary_mha_reserves_null_parent_within_cache_budget(
    full_layers: int, sliding_layers: int, usable_pages: int
) -> None:
    model_config = SimpleNamespace(
        num_attention_layers=full_layers + sliding_layers,
        hf_config=SimpleNamespace(),
    )
    attn_config = _mha_config()
    mha = replace(
        attn_config.component(MHAConfig),
        cache_layer_types=(FULL_ATTENTION,) * full_layers
        + ("sliding_attention",) * sliding_layers,
        sliding_window_tokens=512,
    )
    attn_config = replace(attn_config, components=(mha,))
    server_args = SimpleNamespace(kv_offload_config=None, max_total_tokens=None)

    setup = prepare_cache_setup(
        family="mha",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=16_384,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.spec.family == "mha"
    assert setup.spec.memory_plan.prefix_granularity == 64
    assert setup.spec.memory_plan.num_lcm_blocks == usable_pages
    assert setup.spec.memory_plan.arena_bytes <= 16_384
    assert setup.spec.token_capacity == usable_pages * 64
    assert setup.num_draft_layers == 0
    pool = _pool_over_new_arena(
        setup.spec, attn_config, num_layers=model_config.num_attention_layers
    )
    assert type(pool) is MHATokenToKVPool
    assert pool.arena.runtime_contract.token_capacity == setup.spec.token_capacity
    with pytest.raises(TypeError, match="incompatible with MHAConfig"):
        _pool_over_new_arena(
            replace(setup.spec, family="kimi_k3"),
            attn_config,
            num_layers=model_config.num_attention_layers,
        )


def test_ordinary_mla_reserves_null_parent_within_cache_budget() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = _mla_config()
    server_args = SimpleNamespace(kv_offload_config=None, max_total_tokens=None)

    setup = prepare_cache_setup(
        family="mla",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=24_576,
        probe_batch_rows=None,
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
    ("family", "target_config", "target_overrides", "draft_overrides", "expected"),
    (
        # A 64-token fixed page, or a flexible leaf: the 64-token floor.
        ("mha", _mha_config, {}, None, 64),
        ("mha", _mha_config, {"kernel_page_size": None}, None, 64),
        # A wider fixed page pinned on either side widens the grain.
        ("mla", _mla_config, {"kernel_page_size": 128}, None, 128),
        ("mla", _mla_config, {}, {"kernel_page_size": 128}, 128),
        # The MSA sparse page and the MXFP8 scale tile are 128 tokens.
        ("msa", _msa_config, {}, None, 128),
        ("mha", _mxfp8_mha_config, {}, None, 128),
    ),
)
def test_ordinary_prefix_granularity_holds_whole_kernel_pages(
    family, target_config, target_overrides, draft_overrides, expected
) -> None:
    draft_attn_config = (
        None if draft_overrides is None else replace(_mha_config(), **draft_overrides)
    )
    recipe = cache_recipe(
        family,
        server_args=SimpleNamespace(kv_offload_config=None, max_total_tokens=None),
        model_config=SimpleNamespace(num_attention_layers=2),
        attn_config=replace(target_config(), **target_overrides),
        draft_model_config=(
            None
            if draft_attn_config is None
            else SimpleNamespace(num_attention_layers=1)
        ),
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=65_536,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert recipe.prefix_granularity == expected
    # One block spans the grain: every history group's span equals it.
    assert {spec.block_granularity for spec, _ in recipe.groups()} == {expected}


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
        server_args=SimpleNamespace(kv_offload_config=None, max_total_tokens=None),
        model_config=model_config,
        attn_config=target_attn_config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=65_536,
        probe_batch_rows=None,
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

    def _store_kv_cache(cache_k, cache_v, k_buffer, v_buffer, loc):
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

    assert (
        _resolve_heterogeneous_draft_family("mla", "mha", draft_family_declared=False)
        == "mha"
    )
    assert (
        _resolve_heterogeneous_draft_family(
            "kimi_k3", "mla", draft_family_declared=False
        )
        == "mla"
    )
    with pytest.raises(RuntimeError, match="require an MHA draft"):
        _resolve_heterogeneous_draft_family("mha", "mla", draft_family_declared=False)
    with pytest.raises(RuntimeError, match="support ordinary drafts only"):
        _create_draft_components(
            backend=None,
            server_args=None,
            model_config=SimpleNamespace(num_attention_layers=1),
            config=object(),
            pool=None,
            cache_spec=object(),
            num_target_layers=1,
            full_attn_backend_name=None,
            is_heterogeneous=True,
            linear_attention="gdn",
            is_inkling=False,
        )


def test_hybrid_draft_layers_share_plan_with_disjoint_views() -> None:
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
    target_fields, _ = select_layer_fields(
        plan.fields,
        first_layer=0,
        num_layers=setup.num_target_layers,
    )
    draft_fields, _ = select_layer_fields(
        plan.fields,
        first_layer=setup.num_target_layers,
        num_layers=setup.num_draft_layers,
    )
    assert target_fields.isdisjoint(draft_fields)
    assert target_fields | draft_fields == {field.field_id for field in plan.fields}


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
        probe_batch_rows=None,
    ).setup()

    # One big model: both draft layers are continuation layers (global
    # layers 1 and 2) of the one merged plan. The full_attention group is
    # shared; the draft-only sliding group is planned alongside with its
    # own packing, and its spec joins the ONE published spec set.
    assert setup.num_draft_layers == 2
    plan = setup.spec.memory_plan
    assert plan.field("layer.0.kv").group_id == "full_attention"
    assert plan.field("layer.1.kv").group_id == "full_attention"
    assert plan.field("layer.2.kv").group_id == "draft_swa"
    assert plan.group("draft_swa").cache_blocks_per_lcm_block >= 1
    published = {spec.group_id for spec in setup.spec.cache_group_specs}
    assert published == {"full_attention", "draft_swa"}


def test_union_contract_flows_draft_groups_to_scheduler_config() -> None:
    """No new contract: the one spec publishes draft-only
    groups as ordinary groups; pool publication and the scheduler config
    conversion carry them with their own kind -- the C++ side treats them as
    any sliding-window group, no draft concept anywhere."""
    from test.runtime.cache_pool_test_utils import MinimalCacheView

    import torch

    from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups

    setup = _SyntheticHybridRecipe(
        layer_types=("full_attention", "full_attention", "sliding_attention"),
        group_ids=("full_attention", "full_attention", "draft_swa"),
        num_draft_layers=2,
        windows=(None, None, 8),
        cache_budget_bytes=4_096,
        probe_batch_rows=None,
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
    from test.runtime.cache_pool_test_utils import MinimalCacheView

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
        recipe.attn_config = _mha_config()
        recipe.draft_attn_config = _mha_config() if draft_layers else None
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


def _qwen_gdn_setup(
    full_attention_layers: int,
    *,
    gdn_layers_per_block: int = 3,
    target_kv_dtype: torch.dtype = torch.bfloat16,
    draft_kv_heads: int | None = None,
):
    """Plan Qwen3.5 blocks of GDN layers and one attention layer; an SSM state is one bf16 K page."""
    labels = (
        (LINEAR_ATTENTION,) * gdn_layers_per_block + (FULL_ATTENTION,)
    ) * full_attention_layers
    target_spec = MHAConfig(
        backend_name="fa2",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=128,
        attn_tp_size=1,
        cache_layer_types=labels,
    )
    components = (target_spec,)
    if gdn_layers_per_block:
        components += (
            LinearAttnConfig(
                num_k_heads=8,
                num_v_heads=8,
                head_k_dim=32,
                head_v_dim=32,
                conv_kernel_size=4,
                layer_ids=tuple(
                    layer_id
                    for layer_id, label in enumerate(labels)
                    if label == LINEAR_ATTENTION
                ),
                tp_size=1,
            ),
        )
    attn_config = AttnConfig(
        components=components,
        **_model_wide_kwargs(kv_cache_dtype=target_kv_dtype, kernel_page_size=128),
    )
    draft_attn_config = None
    if draft_kv_heads is not None:
        draft_spec = replace(
            target_spec,
            num_attention_heads=draft_kv_heads,
            num_kv_heads=draft_kv_heads,
            cache_layer_types=(FULL_ATTENTION,),
        )
        draft_attn_config = replace(
            attn_config, components=(draft_spec,), kv_cache_dtype=torch.bfloat16
        )
    return prepare_cache_setup(
        family="qwen_gdn",
        server_args=SimpleNamespace(
            kv_offload_config=None,
            block_size=64,
            max_total_tokens=None,
            speculative_num_draft_tokens=3,
            speculative_eagle_topk=1,
            speculative_draft_tree=False,
            enable_replay_ssm=False,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=SimpleNamespace()),
        ),
        attn_config=attn_config,
        draft_model_config=SimpleNamespace(
            num_attention_layers=int(draft_kv_heads is not None)
        ),
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=1 << 28,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def _page_padding(memory_plan) -> dict[str, int]:
    """Bytes of each group's page that none of its fields use."""
    payload = {}
    for field in memory_plan.fields:
        payload[field.group_id] = payload.get(field.group_id, 0) + field.payload_bytes
    return {
        group.group_id: memory_plan.lcm_block_bytes // group.cache_blocks_per_lcm_block
        - payload[group.group_id]
        for group in memory_plan.groups
    }


@pytest.mark.parametrize(
    "draft_kv_heads,target_kv_dtype",
    [(4, torch.bfloat16), (1, torch.float8_e4m3fn)],
)
def test_qwen_recipe_plans_a_draft_layer_wider_than_a_target_attention_layer(
    draft_kv_heads, target_kv_dtype
) -> None:
    """GDN groups pad over a draft layer's K/V planes wider in bytes than a target attention layer's."""
    # More KV heads, or bf16 over an FP8 target: wider in bytes than a target attention layer.
    setup = _qwen_gdn_setup(
        2, target_kv_dtype=target_kv_dtype, draft_kv_heads=draft_kv_heads
    )

    fields_by_id = {field.field_id: field for field in setup.spec.memory_plan.fields}
    assert fields_by_id["layer.3.k"].shape == (128, 1, 128)
    assert fields_by_id["layer.8.k"].shape == (128, draft_kv_heads, 128)
    assert fields_by_id["layer.8.k"].dtype == "bfloat16"
    assert fields_by_id["layer.8.k"].group_id == FULL_ATTENTION
    assert setup.spec.token_capacity > 0


def test_qwen_state_blocks_carry_no_attention_planes() -> None:
    """A GDN state block holds its SSM states and conv windows only; KV pages pad by the conv windows."""
    plan = _qwen_gdn_setup(2, gdn_layers_per_block=2).spec.memory_plan
    conv_bytes = {
        field.payload_bytes for field in plan.fields if field.field_id.endswith(".conv")
    }.pop()

    # Two attention layers' K and V span four SSM states: all four GDN layers share one page.
    assert _page_padding(plan) == {
        FULL_ATTENTION: 4 * conv_bytes,
        "linear_attention_0": 0,
    }


def test_qwen_conv_windows_fill_a_free_state_segment() -> None:
    """When the SSM states leave a segment free, the conv windows pack into it and KV pages pad nothing."""
    plan = _qwen_gdn_setup(2).spec.memory_plan
    payload = {
        suffix: {
            field.payload_bytes
            for field in plan.fields
            if field.field_id.endswith(suffix)
        }.pop()
        for suffix in (".ssm", ".conv")
    }

    # Six GDN layers over four segments: two groups of three states, one segment left for their conv windows.
    assert plan.lcm_block_bytes == 4 * payload[".ssm"]
    assert _page_padding(plan) == {
        FULL_ATTENTION: 0,
        "linear_attention_0": payload[".ssm"] - 3 * payload[".conv"],
        "linear_attention_1": payload[".ssm"] - 3 * payload[".conv"],
    }


def test_qwen_mtp_draft_layer_fills_state_segments() -> None:
    """An MTP layer of a target attention layer's bytes takes segments the state layers fill too."""
    plan = _qwen_gdn_setup(2, draft_kv_heads=1).spec.memory_plan
    fields = {field.field_id: field for field in plan.fields}
    state_planes = {
        field.plane_id for field in plan.fields if field.field_id.endswith(".ssm")
    }

    assert fields["layer.8.k"].group_id == FULL_ATTENTION
    assert {fields["layer.8.k"].plane_id, fields["layer.8.v"].plane_id} <= state_planes
    assert _page_padding(plan)["linear_attention_0"] == 0


def test_ordinary_profile_reserves_null_page_inside_budget() -> None:
    from test.runtime.test_cache_capacity import mha_recipe

    recipe = mha_recipe(layers=1, budget=16 * 65536)
    setup = recipe.setup()
    assert setup.spec.memory_plan.num_lcm_blocks == 15
    assert setup.spec.memory_plan.arena_bytes == recipe.cache_budget_bytes


@pytest.mark.parametrize(
    "target_backend,draft_backend,error",
    [
        (None, None, None),
        ("tokenspeed_mla", None, None),
        (None, "tokenspeed_mla", None),
        ("trtllm_mla", None, "does not support MLA DCP"),
        (None, "trtllm_mla", r"\(DCP decode\) currently requires"),
        (None, "flashmla", None),
    ],
)
def test_kimi_kvp_resolves_target_and_draft_before_cache_allocation(
    monkeypatch, target_backend, draft_backend, error
):
    from test.runtime.conftest import kimi_recipe

    from tokenspeed.runtime.layers.attention import registry
    from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig

    base = kimi_recipe(tp_size=8).attn_config
    args = SimpleNamespace(
        attention_backend=target_backend,
        drafter_attention_backend=draft_backend,
        kv_parallel_size=2,
        disaggregation_mode="null",
        mapping=SimpleNamespace(world_size=8, world_group=tuple(range(8))),
        gpu_memory_utilization=0.9,
    )
    target = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
        model_profile=None,
        attention_arch=registry.AttentionArch.MLA,
    )
    draft = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["KimiK3ForConditionalGenerationNextN"]
        ),
        model_profile=None,
    )
    built_draft = []

    def create_config(server_args, model, is_draft=False):
        name = (
            server_args.drafter_attention_backend
            if is_draft
            else server_args.attention_backend
        )
        components = (replace(base.components[0], backend_name=name),)
        config = replace(
            base,
            device="cuda",
            kvp_size=2,
            kvp_group=(0, 1),
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            is_draft=is_draft,
            components=components if is_draft else components + base.components[1:],
        )
        if is_draft:
            built_draft.append(config)
        return config

    class ReadyForAllocation(Exception):
        pass

    def profile(**kwargs):
        config = kwargs["attn_config"]
        assert config.component(SoftmaxAttnConfig).backend_name == "tokenspeed_mla"
        assert built_draft[0].component(SoftmaxAttnConfig).backend_name == (
            draft_backend or "tokenspeed_mla"
        )
        raise ReadyForAllocation

    monkeypatch.setattr(
        registry, "current_platform", lambda: SimpleNamespace(is_amd=False)
    )

    # This test resolves NVIDIA backend capabilities without constructing them.
    # Their modules are not registered on AMD hosts.
    class DCPBackend(AttentionBackend):
        supports_mla_dcp = True

    for name in ("tokenspeed_mla", "flashmla"):
        monkeypatch.setitem(
            registry._BACKEND_REGISTRY,
            name,
            ({registry.AttentionArch.MLA}, DCPBackend),
        )
    monkeypatch.setattr(registry, "_create_attn_config", create_config)
    monkeypatch.setattr(registry, "profile_available_cache_memory_bytes", profile)
    expected = (
        pytest.raises(ValueError, match=error)
        if error
        else pytest.raises(ReadyForAllocation)
    )
    with expected:
        registry.create_attn_components(
            args,
            target,
            gpu_id=0,
            rank=0,
            gpu_memory=0,
            draft_model_config=draft,
            graph_reserve_bytes=0,
            post_profile_bytes=0,
            probe_batch_rows=None,
            profiled_cache_bytes=None,
            reuse_target_backend=None,
            reuse_draft_backend=None,
        )
    if draft_backend is not None:
        assert args.drafter_attention_backend == draft_backend


@pytest.mark.parametrize("degree", [1, 2, 8])
def test_kimi_dspark_preserves_backend_with_kvp(degree):
    from tokenspeed.runtime.layers.attention import registry

    def side(architecture):
        return registry._resolve_attn_side(
            SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]),
                model_profile=None,
            ),
            "tokenspeed_mla",
        )

    args = SimpleNamespace(
        attention_backend="tokenspeed_mla",
        drafter_attention_backend="tokenspeed_mla",
        kv_parallel_size=degree,
    )
    target = side("KimiK3ForConditionalGeneration")
    draft = side("K3DSparkModel")
    registry._apply_backend_overrides(args, target, draft)
    assert args.attention_backend == registry.HYBRID_LINEAR_ATTN_BACKEND
    assert args.drafter_attention_backend == "tokenspeed_mla"


class _Sized(Exception):
    pass


def _fake_around_the_budget(monkeypatch, free):
    """Fake what surrounds the real factory, profile and free-memory read.

    ``free`` gives this rank's free bytes; the factory stops where it would
    plan the cache and raises the budget it would plan with.
    """
    from tokenspeed.runtime.layers.attention import registry
    from tokenspeed.runtime.utils import common

    gpu = SimpleNamespace(
        device_count=lambda: 1,
        current_device=lambda: 0,
        empty_cache=lambda: None,
        mem_get_info=lambda gpu_id: (free(), 200 << 30),
    )
    monkeypatch.setattr(common.torch, "get_device_module", lambda device: gpu)
    side = SimpleNamespace(
        is_hybrid_linear=False,
        is_dspark=False,
        requested_backend=None,
    )
    monkeypatch.setattr(registry, "_resolve_attn_side", lambda *a: side)
    monkeypatch.setattr(registry, "_check_pd_support", lambda *a, **k: None)
    monkeypatch.setattr(registry, "_apply_backend_overrides", lambda *a, **k: None)
    config = SimpleNamespace(
        component=lambda cls: SimpleNamespace(), kvp_size=1, device="cuda"
    )
    monkeypatch.setattr(registry, "_create_attn_config", lambda *a, **k: config)
    monkeypatch.setattr(registry, "_resolve_cache_family", lambda *a: "mha")
    monkeypatch.setattr(
        registry, "_resolve_full_attn_backend_name", lambda *a, **k: "fake"
    )
    monkeypatch.setattr(
        registry, "_resolve_heterogeneous_draft_family", lambda *a, **k: None
    )
    monkeypatch.setattr(registry, "_pin_fixed_kernel_page", lambda config, *a: config)

    def prepare(family, *, cache_budget_bytes, probe_batch_rows, **kwargs):
        raise _Sized(cache_budget_bytes, probe_batch_rows)

    monkeypatch.setattr(registry, "cache_recipe", prepare)
    return registry


def _receive_pool_env(monkeypatch, slots, slot_mb):
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOTS", str(slots))
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOT_MB", str(slot_mb))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize(
    "mode, active, external, pool",
    [
        ("prefill", True, False, 4 << 30),
        ("prefill", False, False, 0),
        ("decode", True, False, 0),
        ("null", True, False, 0),
        ("null", True, True, 4 << 30),
        ("null", False, True, 0),
    ],
)
def test_every_boot_leaves_the_receive_pool_out_of_the_cache_budget(
    monkeypatch, rank, probe, mode, active, external, pool
) -> None:
    from tokenspeed.runtime.execution import device, factory

    _receive_pool_env(monkeypatch, 16, 256)
    _fake_around_the_budget(monkeypatch, lambda: 60 << 30)
    target = SimpleNamespace(
        model=object(),
        prepare_multimodal_runtime=lambda: None,
        prepare_communication_runtime=lambda tokens: None,
    )
    monkeypatch.setattr(factory, "create_model_runner", lambda *a: (target, None))
    monkeypatch.setattr(device, "probe_arena_floor", lambda *a, **k: 8)
    server_args = SimpleNamespace(
        disaggregation_mode=mode,
        enable_external_encoder=external,
        chunked_prefill_size=8192,
        attention_backend=None,
        drafter_attention_backend=None,
        disable_cudagraph_memory_reserve=not probe,
        enforce_eager=False,
        enable_memory_saver=False,
        gpu_memory_utilization=0.9,
        mapping=SimpleNamespace(world_size=1, world_group=None, is_first_pp_rank=True),
    )
    with pytest.raises(_Sized) as sized:
        device.build_device_side(
            server_args=server_args,
            model_config=SimpleNamespace(
                is_multimodal=True, is_multimodal_active=active
            ),
            draft_model_config=None,
            gpu_id=0,
            global_rank=rank,
            attn_tp_rank=rank,
            min_per_gpu_mem=100.0,
            overlap_schedule_depth=0,
            decode_input_tokens=1,
            max_batch_size=8,
            kv_transfer_builder=functools.partial(build_host_peer, outbox=None),
        )
    budget, rows = sized.value.args
    assert (rows is not None) == probe
    # 60 GiB free, 100 GiB at startup, 0.9 utilization: 10 GiB of headroom.
    assert abs(budget - ((50 << 30) - pool)) < (1 << 20)


def test_each_rank_leaves_its_own_pool_out_before_the_minimum(monkeypatch) -> None:
    import threading

    from tokenspeed.runtime.utils import common

    local = threading.local()
    registry = _fake_around_the_budget(monkeypatch, lambda: local.free)
    # One rank: just over 60 GiB free and a 4 GiB pool; its peer: 58 GiB free, no pool.
    ranks = [((60 << 30) + 4095, 4 << 30), (58 << 30, 0)]
    barrier, seen = threading.Barrier(len(ranks), timeout=10), []

    def all_reduce(tensor, op, group):
        seen.append(tensor.item())
        barrier.wait()
        tensor.fill_(min(seen))

    monkeypatch.setattr(common.torch.distributed, "all_reduce", all_reduce)
    args = SimpleNamespace(
        attention_backend=None,
        drafter_attention_backend=None,
        gpu_memory_utilization=0.75,
        mapping=SimpleNamespace(world_size=len(ranks), world_group=None),
    )
    budgets = [None] * len(ranks)

    def rank(index, free, pool):
        local.free = free
        with pytest.raises(_Sized) as sized:
            registry.create_attn_components(
                args,
                SimpleNamespace(),
                0,
                0,
                100.0,
                graph_reserve_bytes=0,
                post_profile_bytes=pool,
                probe_batch_rows=None,
                profiled_cache_bytes=None,
                reuse_target_backend=None,
                reuse_draft_backend=None,
            )
        budgets[index] = sized.value.args[0]

    threads = [
        threading.Thread(target=rank, args=(index, *spec))
        for index, spec in enumerate(ranks)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    # 100 GiB at startup at 0.75 utilization leaves 25 GiB of headroom.
    exact = (56 << 30) + 4095 - (25 << 30)
    assert budgets == [exact, exact]


@pytest.mark.parametrize("slots, slot_mb", [(2, 1), (0, 1), (-1, 1), (2, 0), (2, -1)])
@pytest.mark.parametrize("num_deepstack", [0, 3])
@pytest.mark.parametrize(
    "mode, active",
    [
        ("prefill", True),
        ("prefill", False),
        ("decode", True),
        ("null", True),
        ("encode", True),
    ],
)
def test_the_receive_builds_the_pool_exactly_where_the_budget_left_it_out(
    monkeypatch, slots, slot_mb, num_deepstack, mode, active
) -> None:
    from tokenspeed.runtime.epd import prefill_receive, recv_pool

    _receive_pool_env(monkeypatch, slots, slot_mb)
    server_args = SimpleNamespace(
        disaggregation_mode=mode,
        enable_external_encoder=False,
        mapping=SimpleNamespace(is_first_pp_rank=True),
    )
    facts = SimpleNamespace(
        device="cpu", hidden=16, num_deepstack=num_deepstack, dtype=torch.bfloat16
    )
    receive = prefill_receive.build_epd_receive(
        server_args,
        model_config=SimpleNamespace(is_multimodal_active=active),
        encoder_model_facts=lambda: facts,
        mapping=server_args.mapping,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pg_manager=None,
    )
    left_out = recv_pool.recv_pool_bytes(server_args, active)
    built = 0
    if receive is not None:
        settings = receive.settings
        built = settings.slots * (settings.slot_bytes + settings.deepstack_slot_bytes)
        # A slot splits between an item's rows and its deepstack rows within
        # the bytes the budget left out, losing at most their alignment.
        assert built >= left_out - 256 * (1 + num_deepstack) * settings.slots
    assert built <= left_out
    receives = mode == "prefill" and active and slots > 0 and slot_mb > 0
    assert (built > 0) == (left_out > 0) == receives


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
