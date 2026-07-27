"""Kimi-K3 FlatKV integration wiring: preflight, registry, and bridge.

Coverage:

- startup preflight admits Kimi-K3 on a FlatKV-built scheduler ext and still
  rejects any non-FlatKV build (FlatKV-only, no radix fallback);
- ``create_attn_components`` constructs a ``FlatHybridCachePool`` for Kimi-K3
  (never ``SimpleMambaPool`` / ``LayerMappedKVPool``), publishes the four
  scheduler group specs, and derives the cache budget from the same profiled
  free-memory bytes the other flat sizing paths consume;
- the assembled ``HybridLinearAttnBackend`` receives the contract pool: the
  CuteDSL MLA sub-backend resolves the ``full_attention`` table and latent
  views, the KDA sub-backend resolves the three ``linear_attention_*`` state
  groups' component views (dual-index metadata once per group per batch);
- ``validate_flat_scheduler_config`` admits the real wrapper + contract pool
  (the ``uses_flat_cache_groups`` multi-group check defers to the family
  guard on contract pools) and still rejects a non-FlatKV scheduler ext;
- the consumer family guard reads contract group families (never group-name
  heuristics) and leaves non-contract pools alone;
- scheduler bridge: geometry, four ``PagedCacheGroupConfig``s,
  ``bind_paged_cache_scheduler``, and the four-group per-layer table routing
  acceptance on CUDA;
- ``--max-total-tokens`` caps re-plan whole page sets; caps below the
  minimum admission requirement fail loudly;
- speculative decoding stays excluded and fails loudly at startup.

These tests use reduced TP1 shapes; full-checkpoint behavior is covered by
serving CI.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenspeed_kernel.platform import current_platform

# CI Registration (parsed via AST, runtime no-op)
# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``
# absolute imports) both need to be importable when run_ci_suite executes this
# file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.conftest import KIMI_GROUP_IDS as _GROUP_IDS
from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import MLA_KV_LORA_RANK as _KV_LORA_RANK
from test.runtime.conftest import MLA_QK_ROPE_DIM as _QK_ROPE_DIM
from test.runtime.conftest import (
    _poison,
    flat_metadata_for,
    make_kimi_pool,
    requires_cuda,
)

from ci_system.ci_register import register_cuda_ci

import tokenspeed.runtime.configs.paged_cache_spec as paged_cache_spec
from tokenspeed.runtime.configs.kimi_k3_cache_spec import plan_kimi_k3_flat_cache
from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.configs.paged_cache_spec import (
    preflight_kimi_k3_flat_consumers,
    validate_flat_scheduler_config,
)
from tokenspeed.runtime.engine.scheduler_utils import (
    pool_to_paged_cache_groups,
    scheduler_cache_geometry_from_pool,
)
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.attention.backends.flat_cache_metadata import (
    FlatCacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    HybridLinearAttnBackend,
    MambaAttnBackend,
)
from tokenspeed.runtime.layers.attention.kv_cache.flat_hybrid import (
    FlatHybridCachePool,
)

register_cuda_ci(est_time=180, suite="runtime-1gpu")

_K3_ARCHITECTURE = "KimiK3ForConditionalGeneration"
_BLOCK_SIZE = 64  # server --block-size (CuteDSL MLA kernel page)


def _tp1_reference_plan():
    """CPU-only planner probe for the TP1 page-set byte size."""
    return plan_kimi_k3_flat_cache(
        KimiLinearConfig(),
        flat_kvcache_enabled=True,
        tp_size=1,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        preferred_block_size=_BLOCK_SIZE,
        kernel_alignment=128,
        cache_budget_bytes=10**12,
    )


def _server_args(**overrides):
    args = SimpleNamespace(
        device="cuda",
        attention_backend=None,
        drafter_attention_backend=None,
        block_size=_BLOCK_SIZE,
        mapping=SimpleNamespace(
            world_size=1,
            world_group=None,
            attn=SimpleNamespace(tp_size=1, dp_size=1),
        ),
        gpu_memory_utilization=0.9,
        max_total_tokens=None,
        chunked_prefill_size=1024,
        max_mamba_cache_size=None,
        speculative_algorithm=None,
        speculative_num_draft_tokens=1,
        speculative_num_steps=0,
        enforce_eager=True,
        attn_tp_size=1,
        kv_cache_dtype="fp8_e4m3",
        kv_cache_quant_method="",
        max_cudagraph_capture_size=8,
        max_num_seqs=8,
        data_parallel_size=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _model_config():
    text_config = KimiLinearConfig()
    return SimpleNamespace(
        attention_arch=AttentionArch.MLA,
        hf_config=SimpleNamespace(
            architectures=[_K3_ARCHITECTURE],
            text_config=text_config,
        ),
        num_attention_layers=text_config.num_hidden_layers,
        num_hidden_layers=text_config.num_hidden_layers,
        mambaish_config=None,
        dtype=torch.bfloat16,
        context_len=8192,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=_KV_LORA_RANK + _QK_ROPE_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=128,
        qk_rope_head_dim=_QK_ROPE_DIM,
        v_head_dim=128,
        scaling=192**-0.5,
    )


@pytest.fixture()
def flat_runtime_env(monkeypatch):
    """Flat scheduler ext + CuteDSL server dict, without a real flat build."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setattr(paged_cache_spec, "scheduler_ext_flat_kvcache", lambda: True)
    monkeypatch.setattr(registry, "scheduler_ext_flat_kvcache", lambda: True)
    monkeypatch.setitem(global_server_args_dict, "kv_cache_dtype", "fp8_e4m3")
    # KimiLinearConfig.mamba2_cache_params reads the global mapping (as the
    # real runtime populates it from server args at startup).
    monkeypatch.setitem(
        global_server_args_dict,
        "mapping",
        SimpleNamespace(attn=SimpleNamespace(tp_size=1)),
    )


def _create_components(monkeypatch, budget_bytes, profile_calls=None, **arg_overrides):
    def fake_profile(**kwargs):
        if profile_calls is not None:
            profile_calls.append(kwargs)
        return budget_bytes

    monkeypatch.setattr(registry, "profile_available_cache_memory_bytes", fake_profile)
    return registry.create_attn_components(
        server_args=_server_args(**arg_overrides),
        model_config=_model_config(),
        gpu_id=0,
        rank=0,
        gpu_memory=budget_bytes,
    )


# ---------------------------------------------------------------------------
# Preflight (blocker 1): admit on FlatKV, reject anything else
# ---------------------------------------------------------------------------


def test_preflight_admits_kimi_on_flat_ext_and_rejects_radix(monkeypatch) -> None:
    kimi = SimpleNamespace(hf_config=SimpleNamespace(architectures=[_K3_ARCHITECTURE]))
    monkeypatch.setattr(paged_cache_spec, "scheduler_ext_flat_kvcache", lambda: True)
    preflight_kimi_k3_flat_consumers(kimi, None)

    monkeypatch.setattr(paged_cache_spec, "scheduler_ext_flat_kvcache", lambda: False)
    with pytest.raises(RuntimeError, match="FlatKV-only"):
        preflight_kimi_k3_flat_consumers(kimi)
    # Non-Kimi architectures never consult the ext.
    other = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
    )
    preflight_kimi_k3_flat_consumers(other, None)


# ---------------------------------------------------------------------------
# Registry wiring (blockers 2 + 3): FlatHybridCachePool + real backends
# ---------------------------------------------------------------------------


@requires_cuda
def test_registry_builds_flat_contract_pool_and_backends(
    flat_runtime_env, monkeypatch
) -> None:
    reference_plan = _tp1_reference_plan()
    page_set_bytes = reference_plan.diagnostics.bytes_per_page_set
    budget = 10 * page_set_bytes  # 10 page sets -> 1 null + 9 usable
    profile_calls: list[dict] = []

    (
        backend,
        pool,
        draft_backend,
        draft_pool,
        max_num_tokens,
        mamba_pool_total_chunks,
        mamba_pool,
    ) = _create_components(monkeypatch, budget, profile_calls, max_num_seqs=1)

    # --- pool: contract FlatHybridCachePool, never SimpleMambaPool ---
    assert isinstance(pool, FlatHybridCachePool)
    assert mamba_pool is None
    assert draft_backend is None and draft_pool is None
    assert mamba_pool_total_chunks == 0

    contract = pool.runtime_contract
    assert contract is not None
    assert contract.block_size == reference_plan.block_size
    assert contract.usable_pages == 9
    assert max_num_tokens == 3 * contract.block_size == pool.size
    assert contract.token_capacity < contract.usable_pages * contract.block_size

    # --- budget derivation: profiled free-memory bytes -> whole page sets ---
    assert len(profile_calls) == 1
    profile_kwargs = profile_calls[0]
    assert profile_kwargs["gpu_memory_utilization"] == 0.9
    assert profile_kwargs["tp_size"] == 1
    assert contract.usable_pages == budget // page_set_bytes - 1
    assert pool.allocated_bytes() == 10 * page_set_bytes

    # --- four published scheduler group specs ---
    specs = pool.paged_cache_group_specs
    assert tuple(spec.group_id for spec in specs) == _GROUP_IDS
    assert {spec.block_size for spec in specs} == {contract.block_size}
    assert [spec.family for spec in specs] == ["history", "state", "state", "state"]

    # --- backends: platform MLA history consumer + KDA state consumer ---
    if current_platform().is_amd:
        from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend

        mla_backend_cls = MLAAttnBackend
    else:
        from tokenspeed.runtime.layers.attention.backends.tokenspeed_mla import (
            CuteDSLMLABackend,
        )

        mla_backend_cls = CuteDSLMLABackend

    assert isinstance(backend, HybridLinearAttnBackend)
    assert isinstance(backend.full_attn_backend, mla_backend_cls)
    assert isinstance(backend.linear_attn_backend, MambaAttnBackend)
    assert backend.linear_attn_backend.is_kda is True
    assert backend.linear_attn_backend.pool is None  # no SimpleMambaPool
    assert backend.linear_attn_backend.kv_pool is pool
    assert backend.linear_attn_backend._flat_contract_bound is True
    assert backend.linear_attn_backend._flat_state_group_ids == _STATE_GROUPS
    assert backend.flat_cache_consumer_families == frozenset({"history", "state"})

    # --- blocker 3: the startup guard admits the real wrapper + pool ---
    validate_flat_scheduler_config(
        flat_kvcache_ext=True,
        paged_cache_groups=specs,
        attn_backend=backend,
        kv_pool=pool,
        speculative_algorithm=None,
    )
    # ... and still rejects a non-FlatKV scheduler ext for a contract pool.
    with pytest.raises(RuntimeError, match="runtime contract"):
        validate_flat_scheduler_config(
            flat_kvcache_ext=False,
            paged_cache_groups=specs,
            attn_backend=backend,
            kv_pool=pool,
            speculative_algorithm=None,
        )

    # --- scheduler bridge: geometry, group configs, bind ---
    geometry = scheduler_cache_geometry_from_pool(
        pool, fallback_token_capacity=128, fallback_page_size=128
    )
    assert geometry.page_size == contract.block_size
    assert geometry.num_device_pages == contract.usable_pages + 1
    assert geometry.token_capacity == pool.size
    scheduler_groups = pool_to_paged_cache_groups(pool)
    assert len(scheduler_groups) == 4
    assert [group.group_id for group in scheduler_groups] == list(_GROUP_IDS)
    pool.bind_paged_cache_scheduler(
        SimpleNamespace(
            available_kv_pages=lambda: contract.usable_pages,
            active_kv_pages=lambda: 0,
        )
    )

    # --- MLA + KDA backends resolve their tables/components ---
    mla_layer = next(
        binding.layer_id
        for binding in pool.plan.layer_bindings
        if binding.group_id == "full_attention"
    )
    key = pool.get_key_buffer(mla_layer)
    assert key.dtype == torch.float8_e4m3fn
    assert key.shape[-1] == _KV_LORA_RANK + _QK_ROPE_DIM
    assert pool.get_value_buffer(mla_layer).shape[-1] == _KV_LORA_RANK
    for group_id in _STATE_GROUPS:
        kda_layer = next(
            binding.layer_id
            for binding in pool.plan.layer_bindings
            if binding.group_id == group_id
        )
        assert pool.group_id_for_layer(kda_layer) == group_id
        conv = pool.get_component(kda_layer, "conv_state")
        recurrent = pool.get_component(kda_layer, "recurrent_state")
        assert conv.dtype == torch.bfloat16
        assert recurrent.dtype == torch.float32
        assert conv.shape[0] == recurrent.shape[0] == contract.usable_pages + 1

    # --- one decode forward's metadata flows into BOTH sub-backends ---
    block = contract.block_size
    seq_lens_cpu = [5, block + 2]
    full_table = np.array([[3, -1], [4, 5]], dtype=np.int32)
    arrays = {"full_attention": full_table}
    for offset, group_id in enumerate(_STATE_GROUPS):
        # Two columns: request 1's decode token crosses into its second
        # state page, so the dual-index gather reads column 1.
        arrays[group_id] = np.array(
            [[6 + offset, 7 + offset], [9 - offset, 8 - offset]], dtype=np.int32
        )
    metadata, forward_op = flat_metadata_for(contract, arrays, "cuda")
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    backend.init_forward_metadata(
        bs=2,
        num_extends=0,
        req_pool_indices=_poison((2,), dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens_cpu, device="cuda", dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        req_to_page=_poison((16, 8)),
        flat_block_tables=dict(metadata.tables(active_forward_op=forward_op)),
        flat_cache_metadata=metadata,
        flat_cache_forward_op=forward_op,
    )
    assert backend.full_attn_backend._flat_bound is True
    mla_md = backend.full_attn_backend.forward_decode_metadata
    assert mla_md.flat_out_cache_loc.tolist() == [3 * block + 4, 5 * block + 1]
    kda_md = backend.linear_attn_backend.forward_metadata
    assert tuple(sorted(kda_md.state_in_pages_by_group)) == _STATE_GROUPS
    assert tuple(sorted(kda_md.state_out_pages_by_group)) == _STATE_GROUPS
    for group_id in _STATE_GROUPS:
        assert kda_md.state_in_pages_by_group[group_id].shape[0] == 2
        assert kda_md.state_out_pages_by_group[group_id].shape[0] == 2

    del pool, backend
    torch.cuda.empty_cache()


@requires_cuda
def test_registry_max_total_tokens_replans_whole_page_sets(
    flat_runtime_env, monkeypatch
) -> None:
    reference_plan = _tp1_reference_plan()
    page_set_bytes = reference_plan.diagnostics.bytes_per_page_set
    budget = 25 * page_set_bytes  # 24 usable shared pages uncapped
    cap_pages = 2
    cap_tokens = cap_pages * reference_plan.block_size

    _, pool, _, _, max_num_tokens, _, _ = _create_components(
        monkeypatch, budget, max_total_tokens=cap_tokens, max_num_seqs=1
    )
    assert isinstance(pool, FlatHybridCachePool)
    # History pages + three state groups * (one live + one chunk page).
    assert pool.runtime_contract.usable_pages == 8
    assert pool.runtime_contract.token_capacity == cap_tokens
    assert max_num_tokens == cap_tokens

    del pool
    torch.cuda.empty_cache()


def test_registry_rejects_token_cap_below_one_full_page(
    flat_runtime_env, monkeypatch
) -> None:
    reference_plan = _tp1_reference_plan()
    page_set_bytes = reference_plan.diagnostics.bytes_per_page_set
    budget = 13 * page_set_bytes
    # Sub-page token caps cannot define usable scheduler geometry and must fail
    # before any GPU allocation (runs on CPU-only hosts too).
    with pytest.raises(ValueError, match="at least one full flat page"):
        _create_components(
            monkeypatch, budget, max_total_tokens=reference_plan.block_size - 1
        )


# ---------------------------------------------------------------------------
# Scheduler bridge on the real reduced pool
# ---------------------------------------------------------------------------


def test_reduced_k3_scheduler_geometry_is_exact(kimi_pool) -> None:
    contract = kimi_pool.runtime_contract
    usable_pages = contract.usable_pages
    geometry = scheduler_cache_geometry_from_pool(
        kimi_pool,
        fallback_token_capacity=999,
        fallback_page_size=128,
    )
    scheduler_groups = pool_to_paged_cache_groups(kimi_pool)

    assert usable_pages == 1
    assert contract.block_size == 1_536
    assert contract.token_capacity == usable_pages * contract.block_size
    assert kimi_pool.size == usable_pages * contract.block_size
    assert contract.num_device_pages_with_null == usable_pages + 1
    assert geometry.num_usable_pages == usable_pages
    assert geometry.num_device_pages == usable_pages + 1
    assert geometry.token_capacity == usable_pages * contract.block_size
    assert set(contract.group_page_counts.values()) == {usable_pages + 1}
    assert len(scheduler_groups) == 4
    assert {group.total_pages for group in scheduler_groups} == {usable_pages + 1}
    assert all(
        kimi_pool.raw_slab(slot).shape[0] == usable_pages + 1
        for slot in range(len(kimi_pool.plan.physical_slots))
    )


# ---------------------------------------------------------------------------
# Four-group per-layer table routing acceptance (CUDA)
# ---------------------------------------------------------------------------


def _forward_op_for_page(contract, page_id: int):
    arrays = {}
    for width, spec in enumerate(contract.group_specs, start=1):
        table = np.full((1, width), -1, dtype=np.int32)
        table[0, 0] = page_id
        arrays[spec.group_id] = table
    return SimpleNamespace(flat_block_tables_arrays=lambda: arrays)


class FakeFourGroupConsumer:
    def __init__(self) -> None:
        self.calls = 0
        self.routes: list[tuple[int, str]] = []

    def __call__(self, metadata, pool, active_forward_op) -> None:
        self.calls += 1
        page_id = pool.runtime_contract.usable_pages
        for binding in pool.plan.layer_bindings:
            group_id = pool.group_id_for_layer(binding.layer_id)
            table = metadata.table_for_layer(
                pool,
                binding.layer_id,
                active_forward_op=active_forward_op,
            )
            assert table is metadata.require_table(
                group_id,
                active_forward_op=active_forward_op,
            )
            assert table[0, 0].item() == page_id
            self.routes.append((binding.layer_id, group_id))
        for slot in range(len(pool.plan.physical_slots)):
            pool.raw_slab(slot)[page_id].fill_(slot + 1)


def _dispatch_fake_consumer(forward_op, pool, consumer):
    metadata = FlatCacheBatchMetadata.from_forward_op(
        forward_op,
        device="cuda",
        contract=pool.runtime_contract,
        num_requests=1,
    )
    consumer(metadata, pool, forward_op)
    return metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_k3_four_group_metadata_acceptance() -> None:
    pool = make_kimi_pool("cuda", usable_pages=1, with_mla_dims=False)
    contract = pool.runtime_contract
    usable_pages = contract.usable_pages

    rejected_consumer = FakeFourGroupConsumer()
    with pytest.raises(ValueError, match=r"outside -1\.\.1"):
        _dispatch_fake_consumer(
            _forward_op_for_page(contract, usable_pages + 1),
            pool,
            rejected_consumer,
        )
    assert rejected_consumer.calls == 0

    consumer = FakeFourGroupConsumer()
    source_forward_op = _forward_op_for_page(contract, usable_pages)
    metadata = _dispatch_fake_consumer(source_forward_op, pool, consumer)

    tables = tuple(metadata.tables(active_forward_op=source_forward_op).values())
    assert len(tables) == 4
    assert all(table.is_cuda and table.dtype == torch.int32 for table in tables)
    assert len({table.untyped_storage().data_ptr() for table in tables}) == 1
    assert metadata.max_page_id == usable_pages
    assert consumer.calls == 1
    assert all(
        torch.count_nonzero(pool.raw_slab(slot)[0]).item() == 0
        for slot in range(len(pool.plan.physical_slots))
    )
    assert all(
        torch.count_nonzero(pool.raw_slab(slot)[usable_pages]).item() > 0
        for slot in range(len(pool.plan.physical_slots))
    )

    config = KimiLinearConfig()
    family_by_group = {spec.group_id: spec.family for spec in contract.group_specs}
    routes_by_layer = dict(consumer.routes)
    mla_routes = [
        routes_by_layer[layer_id]
        for layer_id in range(config.num_hidden_layers)
        if not config.is_kda_layer(layer_id)
    ]
    kda_routes = [
        routes_by_layer[layer_id]
        for layer_id in range(config.num_hidden_layers)
        if config.is_kda_layer(layer_id)
    ]
    state_group_ids = tuple(
        spec.group_id for spec in contract.group_specs if spec.family == "state"
    )

    assert len(mla_routes) == 24
    assert {family_by_group[group_id] for group_id in mla_routes} == {"history"}
    assert len(kda_routes) == 69
    assert {family_by_group[group_id] for group_id in kda_routes} == {"state"}
    assert Counter(kda_routes) == Counter(
        {group_id: 23 for group_id in state_group_ids}
    )
