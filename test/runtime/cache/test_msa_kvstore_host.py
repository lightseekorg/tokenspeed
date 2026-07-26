from __future__ import annotations

from unittest import mock

import pytest
import torch

from tokenspeed.runtime.cache.kv_cache_host import MSATokenToKVPoolHost
from tokenspeed.runtime.layers.attention.kv_cache.msa import MSATokenToKVPool

PAGE_SIZE = 128
HEAD_NUM = 1
HEAD_DIM = 128
INDEX_HEAD_DIM = 128
LAYER_NUM = 4
INDEXED_LAYER_IDS = frozenset({0, 2})
SIZE = 2 * PAGE_SIZE
KV_DTYPE = torch.float8_e4m3fn
INDEX_DTYPE = torch.bfloat16

_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MSA KVStore transfer requires CUDA"
)


def _make_device_pool() -> MSATokenToKVPool:
    # This test exercises the radix host pool. Force the legacy per-layer
    # device layout even when pytest runs with a flat scheduler extension.
    with mock.patch(_PKG_FLAT_PROBE, return_value=False):
        return MSATokenToKVPool(
            size=SIZE,
            dtype=KV_DTYPE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=8,
            max_context_len=4 * PAGE_SIZE,
            page_size=PAGE_SIZE,
            rank=0,
            index_head_dim=INDEX_HEAD_DIM,
            index_dtype=INDEX_DTYPE,
            indexed_layer_ids=INDEXED_LAYER_IDS,
        )


def _make_host_pool(device_pool: MSATokenToKVPool) -> MSATokenToKVPoolHost:
    return MSATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="layer_first",
        device="cpu",
    )


@pytest.fixture(scope="module")
def msa_pools():
    # cudaHostRegister-backed host memory is not released between pools in one
    # process, so register a single device/host pool pair and share it.
    device_pool = _make_device_pool()
    host_pool = _make_host_pool(device_pool)
    return device_pool, host_pool


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _fill_page_byte_pattern(tensor: torch.Tensor, rows: slice, *, seed: int) -> None:
    page_bytes = tensor[rows].view(torch.uint8).reshape(rows.stop - rows.start, -1)
    row_ids = torch.arange(
        rows.start,
        rows.stop,
        dtype=torch.int64,
        device=tensor.device,
    ).unsqueeze(1)
    column_ids = torch.arange(
        page_bytes.shape[1],
        dtype=torch.int64,
        device=tensor.device,
    ).unsqueeze(0)
    pattern = (row_ids * 17 + column_ids * 29 + seed) % 256
    page_bytes.copy_(pattern.to(torch.uint8))


@cuda_only
def test_msa_host_pool_sizing_and_sparse_layer_mapping(msa_pools):
    from tokenspeed.runtime.cache.executor.memory_executor import (
        _pool_size_per_token,
    )

    device_pool, host_pool = msa_pools

    kv_bytes = HEAD_NUM * HEAD_DIM * LAYER_NUM * KV_DTYPE.itemsize * 2
    index_k_bytes = INDEX_HEAD_DIM * INDEX_DTYPE.itemsize * len(INDEXED_LAYER_IDS)

    assert host_pool.size_per_token == kv_bytes + index_k_bytes
    assert _pool_size_per_token(device_pool) == host_pool.size_per_token
    assert host_pool.get_ksize_per_token() == kv_bytes // 2 + index_k_bytes
    assert tuple(device_pool.index_k_buffer) == tuple(sorted(INDEXED_LAYER_IDS))
    assert tuple(host_pool.index_k_host_buffer) == tuple(sorted(INDEXED_LAYER_IDS))
    assert len(host_pool.index_k_data_refs) == len(INDEXED_LAYER_IDS)
    assert host_pool.index_k_data_ptrs.numel() == len(INDEXED_LAYER_IDS)
    assert host_pool.index_k_row_bytes == INDEX_HEAD_DIM * INDEX_DTYPE.itemsize

    for layer_id in INDEXED_LAYER_IDS:
        buffer = host_pool.index_k_host_buffer[layer_id]
        assert buffer.shape == (host_pool.size, INDEX_HEAD_DIM)
        assert buffer.dtype == INDEX_DTYPE


@cuda_only
@pytest.mark.parametrize("io_backend", ["direct", "kernel"])
def test_msa_host_pool_roundtrip_preserves_kv_and_index_k(msa_pools, io_backend):
    device_pool, host_pool = msa_pools

    if io_backend == "direct" and getattr(torch.version, "hip", None) is not None:
        pytest.skip("direct KVStore transfer is only available on NVIDIA")

    device_rows = slice(PAGE_SIZE, 3 * PAGE_SIZE)
    host_rows = slice(0, 2 * PAGE_SIZE)

    # Make every byte depend on its source row and byte column. Layer- and
    # buffer-specific seeds additionally catch layer, K/V, and index-K pointer
    # mixups; non-contiguous sparse layers exercise the actual layer-id mapping.
    for layer_id in range(LAYER_NUM):
        for page_id in (1, 2):
            rows = slice(page_id * PAGE_SIZE, (page_id + 1) * PAGE_SIZE)
            _fill_page_byte_pattern(
                device_pool.k_buffer[layer_id],
                rows,
                seed=layer_id * 37 + 3,
            )
            _fill_page_byte_pattern(
                device_pool.v_buffer[layer_id],
                rows,
                seed=layer_id * 37 + 101,
            )
            if layer_id in INDEXED_LAYER_IDS:
                _fill_page_byte_pattern(
                    device_pool.index_k_buffer[layer_id],
                    rows,
                    seed=layer_id * 37 + 199,
                )

    # Transfer device pages [1, 2] into host pages [0, 1] (skip padded page 0).
    # Match HostExecutor._prepare_indices: kernel consumes CUDA indices, while
    # direct sorts and consumes CPU indices.
    indices_device = "cuda" if io_backend == "kernel" else "cpu"
    device_indices = torch.arange(
        PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64, device=indices_device
    )
    host_indices = torch.arange(
        0, 2 * PAGE_SIZE, dtype=torch.int64, device=indices_device
    )

    orig_k = {
        layer_id: device_pool.k_buffer[layer_id][device_rows].clone()
        for layer_id in range(LAYER_NUM)
    }
    orig_v = {
        layer_id: device_pool.v_buffer[layer_id][device_rows].clone()
        for layer_id in range(LAYER_NUM)
    }
    orig_index_k = {
        layer_id: device_pool.index_k_buffer[layer_id][device_rows].clone()
        for layer_id in INDEXED_LAYER_IDS
    }

    # The module-scoped pool is reused by both backend parameters. Poison the
    # destination before every writeback so a missing D2H copy cannot pass by
    # observing data left behind by the preceding backend.
    for layer_id in range(LAYER_NUM):
        host_pool.k_buffer[layer_id][host_rows].view(torch.uint8).fill_(0xA5)
        host_pool.v_buffer[layer_id][host_rows].view(torch.uint8).fill_(0x5A)
    for layer_id in INDEXED_LAYER_IDS:
        host_pool.index_k_host_buffer[layer_id][host_rows].view(torch.uint8).fill_(0x3C)

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend
    )
    torch.cuda.synchronize()

    # Check D2H independently, so matching mistakes in writeback and loadback
    # cannot cancel each other out and make the final roundtrip look correct.
    for layer_id in range(LAYER_NUM):
        _assert_same_bytes(
            host_pool.k_buffer[layer_id][host_rows], orig_k[layer_id].cpu()
        )
        _assert_same_bytes(
            host_pool.v_buffer[layer_id][host_rows], orig_v[layer_id].cpu()
        )
    for layer_id in INDEXED_LAYER_IDS:
        _assert_same_bytes(
            host_pool.index_k_host_buffer[layer_id][host_rows],
            orig_index_k[layer_id].cpu(),
        )

    for layer_id in range(LAYER_NUM):
        device_pool.k_buffer[layer_id][device_rows].zero_()
        device_pool.v_buffer[layer_id][device_rows].zero_()
        if layer_id in INDEXED_LAYER_IDS:
            device_pool.index_k_buffer[layer_id][device_rows].zero_()

    for layer_id in range(LAYER_NUM):
        host_pool.load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
        )
    torch.cuda.synchronize()

    for layer_id in range(LAYER_NUM):
        _assert_same_bytes(
            device_pool.k_buffer[layer_id][device_rows], orig_k[layer_id]
        )
        _assert_same_bytes(
            device_pool.v_buffer[layer_id][device_rows], orig_v[layer_id]
        )
    for layer_id in INDEXED_LAYER_IDS:
        _assert_same_bytes(
            device_pool.index_k_buffer[layer_id][device_rows],
            orig_index_k[layer_id],
        )
