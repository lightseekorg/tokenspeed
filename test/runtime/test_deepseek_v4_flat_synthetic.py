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

"""B200 hard gate for the checkpoint-free DeepSeek V4 flat-cache chain.

This test deliberately does not construct a fake ``FlatForwardOp``.  The C++
flat scheduler owns allocation and emits the heterogeneous tables; the
nanobind direct-copy API fills persistent pinned staging; the real V4 backend
normalizes those tables into metadata; and the production TokenSpeed-kernel
writers/readers touch the five physical V4 data classes.

Structured acceptance permutations are covered by the scheduler completion
ledger; this hardware test keeps one end-to-end cache-write confinement smoke.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import tokenspeed_scheduler as ts
import torch

# CI registration is parsed from the AST; the marker is a runtime no-op.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_ROOT)
try:
    from ci_system.ci_register import register_cuda_ci
finally:
    assert sys.path[0] == _TEST_ROOT
    sys.path.pop(0)

register_cuda_ci(est_time=240, suite="deepseek-v4-flat-synthetic")

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    V4_C4_HISTORY_POOL_ID,
    V4_C4_STATE_POOL_ID,
    V4_C128_HISTORY_POOL_ID,
    V4_C128_STATE_POOL_ID,
    V4_INDEX_STATE_POOL_ID,
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_PRODUCER_TARGET_INDEXER,
    V4_PRODUCER_TARGET_MAIN,
    V4_SWA_KV_GROUP_ID,
    V4_SWA_POOL_ID,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.engine.scheduler_utils import FlatBlockTableStagingBuffers
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    deepseek_v4_csa_compress_kv_cache_insert,
    deepseek_v4_csa_indexer_cache_insert,
    deepseek_v4_hca_compress_kv_cache_insert,
    dequantize_deepseek_v4_fp8_ds_mla_cache,
    fused_qnorm_rope_kv_insert,
    read_deepseek_v4_indexer_fp8_cache,
    save_deepseek_v4_compressor_state,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4CacheLayout,
    _group_slot_mapping_from_raw,
)

PAGE_SIZE = 64
TOTAL_BLOCKS = 128
HEAD_DIM = 512
ROPE_DIM = 64
INDEX_HEAD_DIM = 128
C4_RATIO = 4
C128_RATIO = 128
C4_STATE_ROWS = 4
C128_STATE_ROWS = 8
SWA_WINDOW = 128
STATE_C4_WINDOW = 8
STATE_C128_WINDOW = 128

GROUP_TO_POOL = {
    V4_SWA_KV_GROUP_ID: V4_SWA_POOL_ID,
    v4_compressor_state_group_id(C4_RATIO): V4_C4_STATE_POOL_ID,
    v4_compressed_kv_group_id(C4_RATIO): V4_C4_HISTORY_POOL_ID,
    v4_compressor_state_group_id(C128_RATIO): V4_C128_STATE_POOL_ID,
    v4_compressed_kv_group_id(C128_RATIO): V4_C128_HISTORY_POOL_ID,
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID: V4_INDEX_STATE_POOL_ID,
}
POOL_IDS = tuple(sorted(set(GROUP_TO_POOL.values())))


def _pool_config(pool_id: str) -> ts.FlatBlockPoolConfig:
    pool = ts.FlatBlockPoolConfig()
    pool.pool_id = pool_id
    pool.total_blocks = TOTAL_BLOCKS
    # Scheduler metadata only; component tensors remain Python-owned.  A
    # positive distinct byte weight keeps per-pool observability meaningful.
    pool.bytes_per_block = 4096 + 256 * POOL_IDS.index(pool_id)
    return pool


def _group_config(
    *,
    group_id: str,
    rows: int,
    stride: int,
    retention,
    pool_id: str,
    required_domains: int,
    window: int | None = None,
    state: bool = False,
) -> ts.PagedCacheGroupConfig:
    return ts.PagedCacheGroupConfig(
        group_id=group_id,
        rows_per_page=rows,
        entry_stride_tokens=stride,
        total_pages=TOTAL_BLOCKS,
        retention=retention,
        sliding_window_tokens=window,
        family=(
            ts.PagedCacheGroupFamily.State
            if state
            else ts.PagedCacheGroupFamily.History
        ),
        block_size=rows * stride,
        pool_id=pool_id,
        prefix_role=(
            ts.PagedCachePrefixRole.ContinuationState
            if state
            else ts.PagedCachePrefixRole.HistoryAnchor
        ),
        table_layout=(
            ts.PagedCacheTableLayout.BoundedWindow
            if state
            else ts.PagedCacheTableLayout.Absolute
        ),
        required_producer_domain_mask=required_domains,
        owner_mask=1,
    )


def _v4_groups() -> list[ts.PagedCacheGroupConfig]:
    main = V4_PRODUCER_TARGET_MAIN
    indexer = V4_PRODUCER_TARGET_INDEXER
    return [
        _group_config(
            group_id=V4_SWA_KV_GROUP_ID,
            rows=PAGE_SIZE,
            stride=1,
            retention=ts.PagedCacheRetention.SlidingWindow,
            pool_id=V4_SWA_POOL_ID,
            required_domains=main,
            window=SWA_WINDOW,
            state=True,
        ),
        _group_config(
            group_id=v4_compressor_state_group_id(C4_RATIO),
            rows=C4_STATE_ROWS,
            stride=1,
            retention=ts.PagedCacheRetention.SlidingWindow,
            pool_id=V4_C4_STATE_POOL_ID,
            required_domains=main,
            window=STATE_C4_WINDOW,
            state=True,
        ),
        _group_config(
            group_id=v4_compressed_kv_group_id(C4_RATIO),
            rows=PAGE_SIZE,
            stride=C4_RATIO,
            retention=ts.PagedCacheRetention.FullHistory,
            pool_id=V4_C4_HISTORY_POOL_ID,
            required_domains=main | indexer,
        ),
        _group_config(
            group_id=v4_compressor_state_group_id(C128_RATIO),
            rows=C128_STATE_ROWS,
            stride=1,
            retention=ts.PagedCacheRetention.SlidingWindow,
            pool_id=V4_C128_STATE_POOL_ID,
            required_domains=main,
            window=STATE_C128_WINDOW,
            state=True,
        ),
        _group_config(
            group_id=v4_compressed_kv_group_id(C128_RATIO),
            rows=2,
            stride=C128_RATIO,
            retention=ts.PagedCacheRetention.FullHistory,
            pool_id=V4_C128_HISTORY_POOL_ID,
            required_domains=main,
        ),
        _group_config(
            group_id=V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
            rows=C4_STATE_ROWS,
            stride=1,
            retention=ts.PagedCacheRetention.SlidingWindow,
            pool_id=V4_INDEX_STATE_POOL_ID,
            required_domains=indexer,
            window=STATE_C4_WINDOW,
            state=True,
        ),
    ]


def _scheduler_config() -> ts.SchedulerConfig:
    config = ts.SchedulerConfig()
    config.block_size = PAGE_SIZE
    config.num_device_pages = 0
    config.num_host_pages = 0
    config.max_scheduled_tokens = 512
    config.max_batch_size = 8
    config.decode_input_tokens = 1
    config.overlap_schedule_depth = 0
    config.disable_l2_cache = True
    config.enable_l3_storage = False
    config.disable_prefix_cache = True
    config.enable_mixed_prefill_decode = True
    config.enable_structured_flat_kv_completion = True
    config.flat_block_pools = [_pool_config(pool_id) for pool_id in POOL_IDS]
    config.paged_cache_groups = _v4_groups()
    assert config.uses_structured_flat_admission
    return config


def _request(request_id: str, tokens: list[int]) -> ts.RequestSpec:
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    return spec


def _only_forward(plan):
    assert len(plan.forward) == 1
    op = plan.forward[0]
    assert set(op.flat_block_table_group_ids()) == set(GROUP_TO_POOL)
    return op


def _advance_ready(
    scheduler: ts.Scheduler,
    op,
    request_id: str,
    tokens: list[int],
) -> None:
    completion_input = next(
        item for item in op.flat_kv_completion_inputs if item.request_id == request_id
    )
    completion = ts.ForwardEvent.FlatKVCompletion()
    completion.request_id = request_id
    completion.table_generation = completion_input.table_generation
    completion.dispatch_seq = completion_input.dispatch_seq
    completion.accepted_raw_end = completion_input.dispatch_raw_end
    completion.protected_raw_end = completion_input.protected_raw_end
    groups = []
    for group_config in _v4_groups():
        group = ts.ForwardEvent.FlatKVGroupCompletion()
        group.group_id = group_config.group_id
        required_mask = group_config.required_producer_domain_mask
        valid_end = (
            completion_input.dispatch_raw_end
            // group_config.entry_stride_tokens
            * group_config.entry_stride_tokens
        )
        group.completed_domain_mask = required_mask
        group.domain_valid_ends = [valid_end] * required_mask.bit_count()
        groups.append(group)
    completion.groups = groups

    result = ts.ForwardEvent.ExtendResult()
    result.request_id = request_id
    result.tokens = tokens
    result.flat_kv_completion = completion
    event = ts.ExecutionEvent()
    event.add_event(result)
    scheduler.advance(event)


def _staging_plan() -> SimpleNamespace:
    max_rows = 2
    depth = 2
    max_cols = TOTAL_BLOCKS
    groups = tuple(
        SimpleNamespace(group_id=group_id, max_export_cols=max_cols)
        for group_id in sorted(GROUP_TO_POOL)
    )
    pools = tuple(
        SimpleNamespace(pool_id=pool_id, total_blocks=TOTAL_BLOCKS)
        for pool_id in POOL_IDS
    )
    specs = tuple(
        SimpleNamespace(group_id=group_id, pool_id=pool_id)
        for group_id, pool_id in sorted(GROUP_TO_POOL.items())
    )
    staging_bytes = 4 * depth * max_rows * sum(max_cols + 1 for _ in groups)
    return SimpleNamespace(
        group_table_plans=groups,
        pools=pools,
        scheduler_group_specs=specs,
        forward_buffer_depth=depth,
        max_scheduled_batch_rows=max_rows,
        forward_input_bytes=staging_bytes,
        cpu_forward_staging_bytes=staging_bytes,
    )


def _backend() -> DeepseekV4AttentionBackend:
    config = SimpleNamespace(
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
        context_len=512,
        qk_rope_head_dim=ROPE_DIM,
        deepseek_v4_prefill_chunk_size=4,
    )
    return DeepseekV4AttentionBackend(config)


def _init_metadata(
    backend: DeepseekV4AttentionBackend,
    *,
    tables: dict[str, torch.Tensor],
    bases: dict[str, torch.Tensor],
    seq_lens: list[int],
    query_lens: list[int],
    num_extends: int,
    mode: ForwardMode,
):
    bs = len(seq_lens)
    backend.init_forward_metadata(
        bs=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        forward_mode=mode,
        extend_seq_lens_cpu=torch.tensor(
            query_lens[:num_extends], dtype=torch.int32, device="cpu"
        ),
        extend_prefix_lens_cpu=torch.zeros(
            num_extends, dtype=torch.int32, device="cpu"
        ),
        num_tokens=sum(query_lens),
        num_extends=num_extends,
        flat_block_tables=tables,
        flat_block_table_base_offsets=bases,
    )
    metadata = backend.forward_metadata
    assert metadata is not None
    assert metadata.cache.table_source_kind == "flat"
    assert set(metadata.cache.paged_cache_block_tables) == set(GROUP_TO_POOL)
    assert set(metadata.cache.paged_cache_block_table_base_offsets) == set(
        GROUP_TO_POOL
    )
    return metadata


def _uint8_plane(shape: tuple[int, ...], sentinel: int) -> torch.Tensor:
    plane = torch.full(shape, sentinel, dtype=torch.uint8, device="cuda")
    plane[0].zero_()
    return plane


def _float_plane(shape: tuple[int, ...], sentinel: float) -> torch.Tensor:
    plane = torch.full(shape, sentinel, dtype=torch.float32, device="cuda")
    plane[0].zero_()
    return plane


def _slot_pages(slots: torch.Tensor, rows_per_page: int) -> set[int]:
    valid = slots[slots >= 0]
    assert valid.numel() > 0
    pages = torch.div(valid, rows_per_page, rounding_mode="floor")
    result = {int(page) for page in pages.cpu().tolist()}
    assert 0 not in result
    return result


def _changed_pages(before: torch.Tensor, after: torch.Tensor) -> set[int]:
    changed = (before != after).reshape(before.shape[0], -1).any(dim=1)
    return {int(page) for page in changed.nonzero().flatten().cpu().tolist()}


def _run_confined_write(
    planes: dict[str, torch.Tensor],
    target: str,
    expected_pages: set[int],
    writer,
) -> None:
    snapshots = {name: plane.clone() for name, plane in planes.items()}
    writer()
    torch.cuda.synchronize()

    assert _changed_pages(snapshots[target], planes[target]) == expected_pages
    assert torch.equal(planes[target][0], snapshots[target][0])
    assert int(torch.count_nonzero(planes[target][0]).item()) == 0
    guard_page = max(expected_pages) + 1
    assert guard_page < planes[target].shape[0]
    assert guard_page not in expected_pages
    assert torch.equal(
        planes[target][guard_page], snapshots[target][guard_page]
    ), f"{target} crossed into adjacent guard page {guard_page}"

    for name, snapshot in snapshots.items():
        if name != target:
            assert torch.equal(
                planes[name], snapshot
            ), f"{target} writer corrupted component plane {name}"


@pytest.fixture(scope="module", autouse=True)
def _require_b200_flat_runtime():
    assert torch.cuda.is_available(), "B200 synthetic gate requires CUDA"
    assert ts.FLAT_KVCACHE is True, "synthetic gate loaded a radix scheduler extension"
    assert torch.cuda.get_device_capability(0) == (
        10,
        0,
    ), "deepseek-v4 flat synthetic gate must run on a B200/SM100 GPU"
    yield
    torch.cuda.synchronize()


def test_real_flat_scheduler_to_v4_five_plane_kernel_chain():
    torch.manual_seed(20260714)
    scheduler = ts.Scheduler(_scheduler_config())

    # 257 tokens make raw position 255 a c4 and c128 compressed boundary while
    # reserving the following rows.  The second schedule rolls every bounded
    # state table before adding a four-token prefill request to the decode row.
    scheduler.submit_requests([_request("decode", list(range(257)))])
    initial_op = _only_forward(scheduler.next_execution_plan())

    staging = FlatBlockTableStagingBuffers(_staging_plan(), device="cuda")
    initial_tables, initial_bases = staging.stage(initial_op, num_reqs=1)

    backend = _backend()
    initial = _init_metadata(
        backend,
        tables=initial_tables,
        bases=initial_bases,
        seq_lens=[257],
        query_lens=[257],
        num_extends=1,
        mode=ForwardMode.EXTEND,
    )

    layout = DeepseekV4CacheLayout(
        layer_ratio=(C4_RATIO, C128_RATIO),
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        page_size=PAGE_SIZE,
        use_fp4_indexer_cache=False,
        index_head_dim=INDEX_HEAD_DIM,
    )
    c4_block = layout.storage_block_size(C4_RATIO)
    c128_block = layout.storage_block_size(C128_RATIO)
    index_block = max(PAGE_SIZE, c4_block)
    planes = {
        "swa_kv": _uint8_plane((TOTAL_BLOCKS, layout.swa_block_bytes(PAGE_SIZE)), 0xA1),
        "compressed_kv_c4": _uint8_plane(
            (TOTAL_BLOCKS, layout.swa_block_bytes(c4_block)), 0xA2
        ),
        "compressed_kv_c128": _uint8_plane(
            (TOTAL_BLOCKS, layout.swa_block_bytes(c128_block)), 0xA3
        ),
        "compressor_state_c4": _float_plane(
            (TOTAL_BLOCKS, C4_STATE_ROWS, HEAD_DIM * 4), -401.0
        ),
        "compressor_state_c128": _float_plane(
            (TOTAL_BLOCKS, C128_STATE_ROWS, HEAD_DIM * 2), -12801.0
        ),
        "indexer_kv": _uint8_plane(
            (TOTAL_BLOCKS, index_block * (INDEX_HEAD_DIM + 4)), 0xA4
        ),
        "indexer_compressor_state": _float_plane(
            (TOTAL_BLOCKS, C4_STATE_ROWS, INDEX_HEAD_DIM * 4), -404.0
        ),
    }

    positions = torch.arange(257, dtype=torch.int64, device="cuda")
    token_to_req = initial.token_to_req_indices
    assert token_to_req.numel() == positions.numel()

    swa_slots = _group_slot_mapping_from_raw(
        positions,
        token_to_req,
        initial.cache.swa_block_table,
        PAGE_SIZE,
        base_offsets=initial.cache.swa_base_logical_page,
        capacity_pages=TOTAL_BLOCKS,
    )
    c4_state_slots = _group_slot_mapping_from_raw(
        positions,
        token_to_req,
        initial.cache.compressor_state_block_tables[C4_RATIO],
        C4_STATE_ROWS,
        base_offsets=initial.cache.compressor_state_base_logical_pages[C4_RATIO],
        capacity_pages=TOTAL_BLOCKS,
    )
    c128_state_slots = _group_slot_mapping_from_raw(
        positions,
        token_to_req,
        initial.cache.compressor_state_block_tables[C128_RATIO],
        C128_STATE_ROWS,
        base_offsets=initial.cache.compressor_state_base_logical_pages[C128_RATIO],
        capacity_pages=TOTAL_BLOCKS,
    )
    index_state_slots = _group_slot_mapping_from_raw(
        positions,
        token_to_req,
        initial.cache.indexer_state_block_table,
        C4_STATE_ROWS,
        base_offsets=initial.cache.indexer_state_base_logical_page,
        capacity_pages=TOTAL_BLOCKS,
    )
    c4_slots = initial.cache.compressed_slot_mapping(
        positions,
        C4_RATIO,
        token_to_req_indices=token_to_req,
        query_start_loc=initial.query_start_loc,
        seq_lens=initial.seq_lens,
        kv_cache_block_size=c4_block,
        capacity_pages=TOTAL_BLOCKS,
    )
    c128_slots = initial.cache.compressed_slot_mapping(
        positions,
        C128_RATIO,
        token_to_req_indices=token_to_req,
        query_start_loc=initial.query_start_loc,
        seq_lens=initial.seq_lens,
        kv_cache_block_size=c128_block,
        capacity_pages=TOTAL_BLOCKS,
    )
    assert int(c4_slots[3].item()) >= 0 and int(c4_slots[255].item()) >= 0
    assert int(c128_slots[127].item()) >= 0 and int(c128_slots[255].item()) >= 0
    assert torch.all(c4_slots[(positions + 1) % C4_RATIO != 0] == -1)
    assert torch.all(c128_slots[(positions + 1) % C128_RATIO != 0] == -1)

    cos_sin = torch.randn(512, ROPE_DIM, dtype=torch.float32, device="cuda") * 0.02
    c4_kv = torch.randn(257, HEAD_DIM * 2, dtype=torch.bfloat16, device="cuda")
    c4_score = torch.randn_like(c4_kv) * 0.05
    c4_ape = (
        torch.randn(C4_RATIO, HEAD_DIM * 2, dtype=torch.float32, device="cuda") * 0.01
    )
    c128_kv = torch.randn(257, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    c128_score = torch.randn_like(c128_kv) * 0.05
    c128_ape = (
        torch.randn(C128_RATIO, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.01
    )
    index_kv = torch.randn(257, INDEX_HEAD_DIM * 2, dtype=torch.bfloat16, device="cuda")
    index_score = torch.randn_like(index_kv) * 0.05
    index_ape = (
        torch.randn(C4_RATIO, INDEX_HEAD_DIM * 2, dtype=torch.float32, device="cuda")
        * 0.01
    )

    _run_confined_write(
        planes,
        "compressor_state_c4",
        _slot_pages(c4_state_slots, C4_STATE_ROWS),
        lambda: save_deepseek_v4_compressor_state(
            kv=c4_kv,
            score=c4_score,
            ape=c4_ape,
            state_cache=planes["compressor_state_c4"],
            slot_mapping=c4_state_slots,
            positions=positions,
            block_size=C4_STATE_ROWS,
            compress_ratio=C4_RATIO,
        ),
    )
    _run_confined_write(
        planes,
        "compressor_state_c128",
        _slot_pages(c128_state_slots, C128_STATE_ROWS),
        lambda: save_deepseek_v4_compressor_state(
            kv=c128_kv,
            score=c128_score,
            ape=c128_ape,
            state_cache=planes["compressor_state_c128"],
            slot_mapping=c128_state_slots,
            positions=positions,
            block_size=C128_STATE_ROWS,
            compress_ratio=C128_RATIO,
        ),
    )
    _run_confined_write(
        planes,
        "indexer_compressor_state",
        _slot_pages(index_state_slots, C4_STATE_ROWS),
        lambda: save_deepseek_v4_compressor_state(
            kv=index_kv,
            score=index_score,
            ape=index_ape,
            state_cache=planes["indexer_compressor_state"],
            slot_mapping=index_state_slots,
            positions=positions,
            block_size=C4_STATE_ROWS,
            compress_ratio=C4_RATIO,
        ),
    )

    q = torch.randn(257, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    swa_kv = torch.randn(257, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _run_confined_write(
        planes,
        "swa_kv",
        _slot_pages(swa_slots, PAGE_SIZE),
        lambda: fused_qnorm_rope_kv_insert(
            q=q,
            kv=swa_kv,
            swa_kv_cache_2d=planes["swa_kv"],
            slot_mapping=swa_slots,
            positions=positions,
            cos_sin_cache=cos_sin,
            rms_norm_eps=1.0e-6,
            block_size=PAGE_SIZE,
        ),
    )
    _run_confined_write(
        planes,
        "compressed_kv_c4",
        _slot_pages(c4_slots, c4_block),
        lambda: deepseek_v4_csa_compress_kv_cache_insert(
            state_cache=planes["compressor_state_c4"],
            token_to_req_indices=token_to_req,
            positions=positions,
            compressor_slot_mapping=c4_state_slots,
            block_table=initial.cache.compressor_state_block_tables[C4_RATIO],
            block_table_base_offsets=(
                initial.cache.compressor_state_base_logical_pages[C4_RATIO]
            ),
            compressor_block_size=C4_STATE_ROWS,
            rms_norm_weight=torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda"),
            rms_norm_eps=1.0e-6,
            cos_sin_cache=cos_sin,
            kv_cache_2d=planes["compressed_kv_c4"],
            kv_slot_mapping=c4_slots,
            kv_cache_block_size=c4_block,
            compress_ratio=C4_RATIO,
        ),
    )
    _run_confined_write(
        planes,
        "compressed_kv_c128",
        _slot_pages(c128_slots, c128_block),
        lambda: deepseek_v4_hca_compress_kv_cache_insert(
            state_cache=planes["compressor_state_c128"],
            token_to_req_indices=token_to_req,
            positions=positions,
            compressor_slot_mapping=c128_state_slots,
            block_table=initial.cache.compressor_state_block_tables[C128_RATIO],
            block_table_base_offsets=(
                initial.cache.compressor_state_base_logical_pages[C128_RATIO]
            ),
            compressor_block_size=C128_STATE_ROWS,
            rms_norm_weight=torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda"),
            rms_norm_eps=1.0e-6,
            cos_sin_cache=cos_sin,
            kv_cache_2d=planes["compressed_kv_c128"],
            kv_slot_mapping=c128_slots,
            kv_cache_block_size=c128_block,
            compress_ratio=C128_RATIO,
        ),
    )
    _run_confined_write(
        planes,
        "indexer_kv",
        _slot_pages(c4_slots, index_block),
        lambda: deepseek_v4_csa_indexer_cache_insert(
            state_cache=planes["indexer_compressor_state"],
            token_to_req_indices=token_to_req,
            positions=positions,
            compressor_slot_mapping=index_state_slots,
            block_table=initial.cache.indexer_state_block_table,
            block_table_base_offsets=initial.cache.indexer_state_base_logical_page,
            compressor_block_size=C4_STATE_ROWS,
            rms_norm_weight=torch.ones(
                INDEX_HEAD_DIM, dtype=torch.float32, device="cuda"
            ),
            rms_norm_eps=1.0e-6,
            cos_sin_cache=cos_sin,
            kv_cache_2d=planes["indexer_kv"],
            kv_slot_mapping=c4_slots,
            kv_cache_block_size=index_block,
            use_fp4_cache=False,
            compress_ratio=C4_RATIO,
        ),
    )

    for decoded in (
        dequantize_deepseek_v4_fp8_ds_mla_cache(
            planes["swa_kv"],
            swa_slots[[0, 255]],
            PAGE_SIZE,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
        ),
        dequantize_deepseek_v4_fp8_ds_mla_cache(
            planes["compressed_kv_c4"],
            c4_slots[[3, 255]],
            c4_block,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
        ),
        dequantize_deepseek_v4_fp8_ds_mla_cache(
            planes["compressed_kv_c128"],
            c128_slots[[127, 255]],
            c128_block,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
        ),
        read_deepseek_v4_indexer_fp8_cache(
            planes["indexer_kv"], c4_slots[[3, 255]], index_block
        ),
    ):
        assert torch.isfinite(decoded).all()
        assert float(decoded.abs().sum().item()) > 0.0

    # Complete the first operation only after its physical writes are consumed.
    # The second direct-copy ring slot is then a true mixed op whose compact SWA
    # base has rolled, while the c4 history page IDs for B@3 and A@255 are
    # deliberately non-contiguous (physical adjacency is never assumed).
    _advance_ready(scheduler, initial_op, "decode", [10_000])
    scheduler.submit_requests([_request("prefill", list(range(20_000, 20_004)))])
    mixed_op = _only_forward(scheduler.next_execution_plan())
    assert list(mixed_op.request_ids) == ["prefill", "decode"]
    assert list(mixed_op.input_lengths) == [4, 1]
    assert mixed_op.num_extends() == 1

    mixed_tables, mixed_bases = staging.stage(mixed_op, num_reqs=2)
    assert int(mixed_bases[V4_SWA_KV_GROUP_ID][1].item()) > 0
    mixed = _init_metadata(
        backend,
        tables=mixed_tables,
        bases=mixed_bases,
        seq_lens=[4, 258],
        query_lens=[4, 1],
        num_extends=1,
        mode=ForwardMode.MIXED,
    )
    assert mixed.query_start_offsets == (0, 4, 5)
    assert mixed.token_to_req_indices.tolist() == [0, 0, 0, 0, 1]

    mixed_positions = torch.tensor([0, 1, 2, 3, 257], device="cuda")
    mixed_swa_slots = _group_slot_mapping_from_raw(
        mixed_positions,
        mixed.token_to_req_indices,
        mixed.cache.swa_block_table,
        PAGE_SIZE,
        base_offsets=mixed.cache.swa_base_logical_page,
        capacity_pages=TOTAL_BLOCKS,
    )
    mixed_boundary_positions = torch.tensor([3, 255], device="cuda")
    mixed_boundary_reqs = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    mixed_c4_slots = mixed.cache.compressed_slot_mapping(
        mixed_boundary_positions,
        C4_RATIO,
        token_to_req_indices=mixed_boundary_reqs,
        query_start_loc=mixed.query_start_loc,
        seq_lens=mixed.seq_lens,
        kv_cache_block_size=c4_block,
        capacity_pages=TOTAL_BLOCKS,
    )
    mixed_c4_pages = sorted(_slot_pages(mixed_c4_slots, c4_block))
    assert len(mixed_c4_pages) == 2
    assert mixed_c4_pages[1] - mixed_c4_pages[0] > 1

    mixed_q = torch.randn(5, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    mixed_kv = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _run_confined_write(
        planes,
        "swa_kv",
        _slot_pages(mixed_swa_slots, PAGE_SIZE),
        lambda: fused_qnorm_rope_kv_insert(
            q=mixed_q,
            kv=mixed_kv,
            swa_kv_cache_2d=planes["swa_kv"],
            slot_mapping=mixed_swa_slots,
            positions=mixed_positions,
            cos_sin_cache=cos_sin,
            rms_norm_eps=1.0e-6,
            block_size=PAGE_SIZE,
        ),
    )
