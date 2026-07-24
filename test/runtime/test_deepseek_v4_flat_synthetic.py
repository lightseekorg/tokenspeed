# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""One B200 smoke across the complete DeepSeek V4 flat-cache write path.

Detailed scheduler lifecycle and kernel numerics live in their respective unit
tests. This gate proves that the production V4 plan is accepted by the flat
scheduler, copied through persistent staging, normalized by the V4 backend,
and consumed by a real cache writer without crossing its assigned pages.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CI registration is parsed from the AST; the marker is a runtime no-op.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_ROOT)
try:
    from ci_system.ci_register import register_cuda_ci
finally:
    assert sys.path[0] == _TEST_ROOT
    sys.path.pop(0)

register_cuda_ci(est_time=120, suite="deepseek-v4-flat-synthetic")

torch = pytest.importorskip("torch")
ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (  # noqa: E402
    V4_SWA_KV_GROUP_ID,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    FlatBlockTableStagingBuffers,
    make_config,
    make_spec,
    pool_to_flat_block_pools,
    pool_to_paged_cache_groups,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.flat_cache_tables import FlatCacheTableOwnerView  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (  # noqa: E402
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (  # noqa: E402
    fused_qnorm_rope_kv_insert,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (  # noqa: E402
    V4FlatArenaSet,
    _group_slot_mapping_from_raw,
    build_deepseek_v4_flat_memory_plan,
    deepseek_v4_cache_layout_from_config,
)

PAGE_SIZE = 64
HEAD_DIM = 512
ROPE_DIM = 64
TOKEN_COUNT = 129


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        index_head_dim=128,
        sliding_window=128,
    )


def _layout():
    return deepseek_v4_cache_layout_from_config(
        _hf_config(),
        page_size=PAGE_SIZE,
        use_fp4_indexer_cache=False,
    )


def _plan():
    return build_deepseek_v4_flat_memory_plan(
        target_layout=_layout(),
        target_hf_config=_hf_config(),
        target_layer_num=3,
        target_max_live_requests=1,
        target_max_context_len=256,
        target_max_graph_bs=1,
        max_scheduled_tokens=256,
        max_total_tokens=256,
    )


def _scheduler(plan):
    pools_by_id = {pool.pool_id: pool for pool in plan.pools}
    pool_view = SimpleNamespace(
        flat_memory_plan=plan,
        paged_cache_group_specs=plan.target_owner_group_specs,
        scheduler_group_specs=plan.scheduler_group_specs,
        paged_cache_group_page_counts={
            spec.group_id: pools_by_id[spec.pool_id].total_blocks
            for spec in plan.target_owner_group_specs
        },
        scheduler_group_page_counts={
            spec.group_id: pools_by_id[spec.pool_id].total_blocks
            for spec in plan.scheduler_group_specs
        },
    )
    config = make_config(
        num_device_pages=0,
        max_scheduled_tokens=256,
        max_batch_size=1,
        page_size=PAGE_SIZE,
        scheduler_backend="flat",
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        prefetch_threshold=0,
        role="null",
        disable_prefix_cache=True,
        paged_cache_groups=pool_to_paged_cache_groups(pool_view),
        flat_block_pools=pool_to_flat_block_pools(pool_view),
    )
    assert config.uses_explicit_flat_pools
    return ts.Scheduler(config)


def _backend(plan) -> DeepseekV4AttentionBackend:
    metadata = plan.runtime_metadata
    backend = DeepseekV4AttentionBackend(
        SimpleNamespace(
            device=torch.device("cuda"),
            num_attention_heads=1,
            num_kv_heads=1,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=HEAD_DIM,
            is_draft=False,
            speculative_num_draft_tokens=1,
            speculative_num_steps=0,
            page_size=PAGE_SIZE,
            context_len=256,
            qk_rope_head_dim=ROPE_DIM,
            deepseek_v4_prefill_chunk_size=4,
        )
    )
    backend.init_cuda_graph_state(
        max_bs=metadata.graph_batch_rows,
        paged_cache_group_specs=plan.target_owner_group_specs,
        flat_capture_cols_by_group=metadata.graph_capture_cols_by_group("target"),
        flat_graph_batch_rows=metadata.graph_batch_rows,
    )
    return backend


def _changed_pages(before: torch.Tensor, after: torch.Tensor) -> set[int]:
    changed = (before != after).reshape(before.shape[0], -1).any(dim=1)
    return {int(page) for page in changed.nonzero().flatten().cpu().tolist()}


@pytest.fixture(scope="module", autouse=True)
def _require_b200_flat_runtime():
    assert torch.cuda.is_available(), "V4 flat synthetic gate requires CUDA"
    assert ts.FLAT_KVCACHE is True, "loaded scheduler extension is not flat KV"
    assert torch.cuda.get_device_capability(0) == (
        10,
        0,
    ), "V4 flat synthetic gate must run on B200/SM100"
    yield
    torch.cuda.synchronize()


def test_scheduler_plan_staging_backend_and_swa_writer_share_one_page_contract():
    torch.manual_seed(20260714)
    plan = _plan()
    scheduler = _scheduler(plan)
    scheduler.submit_requests([make_spec("request", list(range(TOKEN_COUNT)))])
    execution_plan = scheduler.next_execution_plan()
    assert len(execution_plan.forward) == 1
    op = execution_plan.forward[0]

    group_ids = {spec.group_id for spec in plan.scheduler_group_specs}
    assert set(op.flat_block_table_group_ids()) == group_ids
    # The production plan, not a parallel test schema, sizes persistent staging.
    staging = FlatBlockTableStagingBuffers(plan, device="cuda")
    with torch.cuda.nvtx.range("v4_flat_stage"):
        source = staging.stage(op, num_reqs=1)

    backend = _backend(plan)
    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([TOKEN_COUNT], dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=torch.tensor([TOKEN_COUNT], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.zeros(1, dtype=torch.int32),
        num_tokens=TOKEN_COUNT,
        num_extends=1,
        cache_table_source=source,
    )
    metadata = backend.forward_metadata
    assert metadata is not None
    assert not metadata.cache.has_legacy_block_table
    assert set(metadata.cache.paged_cache_block_tables) == group_ids
    assert set(metadata.cache.paged_cache_block_table_base_offsets) == group_ids

    target_source = FlatCacheTableOwnerView(
        tuple(spec.group_id for spec in plan.target_owner_group_specs)
    ).bind(source, owner="target")
    assert target_source is not None
    with torch.cuda.nvtx.range("v4_flat_graph_unpack"):
        backend._refresh_cuda_graph_flat_packed(target_source, bs=1, actual_bs=1)
    for group_id, table in target_source.tables.items():
        graph_table = backend._cuda_graph_paged_cache_block_tables[group_id]
        cols = table.shape[1]
        assert torch.equal(graph_table[:, :cols], table)
        assert torch.all(graph_table[:, cols:] == -1)
        assert torch.equal(
            backend._cuda_graph_paged_cache_base_offsets[group_id],
            target_source.base_offsets[group_id],
        )

    # One request crosses c4 and c128 compression boundaries. This proves the
    # same staged group set preserves heterogeneous raw spans without assuming
    # physical page adjacency.
    arena = V4FlatArenaSet(plan, device="cuda", enable_memory_saver=False)
    positions = torch.arange(TOKEN_COUNT, dtype=torch.int64, device="cuda")
    c4_slots = metadata.cache.compressed_slot_mapping(
        positions,
        4,
        token_to_req_indices=metadata.token_to_req_indices,
        query_start_loc=metadata.query_start_loc,
        seq_lens=metadata.seq_lens,
        kv_cache_block_size=_layout().storage_block_size(4),
        capacity_pages=arena.tensor(
            "target", v4_compressed_kv_group_id(4), 1, "compressed_kv"
        ).shape[0],
    )
    c128_slots = metadata.cache.compressed_slot_mapping(
        positions,
        128,
        token_to_req_indices=metadata.token_to_req_indices,
        query_start_loc=metadata.query_start_loc,
        seq_lens=metadata.seq_lens,
        kv_cache_block_size=_layout().storage_block_size(128),
        capacity_pages=arena.tensor(
            "target", v4_compressed_kv_group_id(128), 2, "compressed_kv"
        ).shape[0],
    )
    assert int(c4_slots[3]) >= 0 and int(c128_slots[127]) >= 0
    assert torch.all(c4_slots[(positions + 1) % 4 != 0] == -1)
    assert torch.all(c128_slots[(positions + 1) % 128 != 0] == -1)

    swa = arena.tensor("target", V4_SWA_KV_GROUP_ID, 0, "swa_kv")
    before = swa.clone()
    swa_spec = next(
        spec
        for spec in plan.scheduler_group_specs
        if spec.group_id == V4_SWA_KV_GROUP_ID
    )
    swa_slots = _group_slot_mapping_from_raw(
        positions,
        metadata.token_to_req_indices,
        metadata.cache.swa_block_table,
        swa_spec.rows_per_page,
        base_offsets=metadata.cache.swa_base_logical_page,
        capacity_pages=swa.shape[0],
    )
    valid_slots = swa_slots[swa_slots >= 0]
    expected_pages = {
        int(page)
        for page in torch.div(
            valid_slots, swa_spec.rows_per_page, rounding_mode="floor"
        )
        .cpu()
        .tolist()
    }
    assert expected_pages and 0 not in expected_pages

    fused_qnorm_rope_kv_insert(
        q=torch.randn(TOKEN_COUNT, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
        kv=torch.randn(TOKEN_COUNT, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
        swa_kv_cache_2d=swa,
        slot_mapping=swa_slots,
        positions=positions,
        cos_sin_cache=torch.randn(256, ROPE_DIM, dtype=torch.float32, device="cuda"),
        rms_norm_eps=1.0e-6,
        block_size=swa_spec.rows_per_page,
    )
    torch.cuda.synchronize()

    assert _changed_pages(before, swa) == expected_pages
    assert torch.equal(swa[0], before[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
