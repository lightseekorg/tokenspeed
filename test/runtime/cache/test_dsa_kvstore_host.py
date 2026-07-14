from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.cache.kv_cache_host import DSATokenToKVPoolHost
from tokenspeed.runtime.layers.attention.configs.dsa import dsa_index_k_row_bytes
from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

PAGE_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
LAYER_NUM = 2
SIZE = 2 * PAGE_SIZE

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSA KVStore transfer requires CUDA"
)


def _make_device_pool() -> DSATokenToKVPool:
    return DSATokenToKVPool(
        size=SIZE,
        dtype=torch.float8_e4m3fn,
        model_dtype=torch.bfloat16,
        quant_method="none",
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device="cuda",
        enable_memory_saver=False,
        max_batch_size=8,
        max_context_len=256,
        page_size=PAGE_SIZE,
        rank=0,
        index_head_dim=INDEX_HEAD_DIM,
    )


def _make_host_pool(device_pool: DSATokenToKVPool) -> DSATokenToKVPoolHost:
    return DSATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="layer_first",
        device="cpu",
    )


@pytest.fixture(scope="module")
def dsa_pools():
    # cudaHostRegister-backed host memory is not released between pools in one
    # process, so register a single device/host pool pair and share it.
    device_pool = _make_device_pool()
    host_pool = _make_host_pool(device_pool)
    return device_pool, host_pool


@cuda_only
def test_dsa_host_pool_sizing_includes_index_k(dsa_pools):
    _, host_pool = dsa_pools

    row_bytes = dsa_index_k_row_bytes(INDEX_HEAD_DIM)
    latent_bytes = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * LAYER_NUM  # uint8 store dtype
    assert host_pool.index_k_row_bytes == row_bytes
    assert host_pool.size_per_token == latent_bytes + row_bytes * LAYER_NUM
    # Host index-K mirrors the device layout (layer_num, host_size, row_bytes).
    assert host_pool.index_k_buffer.shape == (LAYER_NUM, host_pool.size, row_bytes)
    assert host_pool.index_k_buffer.dtype == torch.uint8


@cuda_only
def test_dsa_host_pool_rejects_page_first_layout(dsa_pools):
    device_pool, _ = dsa_pools
    # The layout guard raises before any host memory is registered.
    with pytest.raises(NotImplementedError):
        DSATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="page_first",
            device="cpu",
        )


@cuda_only
@pytest.mark.parametrize("io_backend", ["direct", "kernel"])
def test_dsa_host_pool_roundtrip_preserves_index_k(dsa_pools, io_backend):
    device_pool, host_pool = dsa_pools

    torch.manual_seed(0)
    # Fill latent + index-K with random bytes; byte-exact round trip proves the
    # block-split index-K page layout survives the page-contiguous copy.
    for layer_id in range(LAYER_NUM):
        device_pool.kv_buffer[layer_id].copy_(
            torch.randint(
                0, 256, device_pool.kv_buffer[layer_id].shape, dtype=torch.uint8
            ).cuda()
        )
        device_pool.index_k_buffer[layer_id].copy_(
            torch.randint(
                0, 256, device_pool.index_k_buffer[layer_id].shape, dtype=torch.uint8
            ).cuda()
        )

    # Transfer device pages [1, 2] into host pages [0, 1] (skip padded page 0).
    # The kernel backend requires CUDA indices (the cache controller moves them
    # to device before the transfer); direct accepts them too.
    device_indices = torch.arange(
        PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64, device="cuda"
    )
    host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64, device="cuda")

    orig_latent = [
        device_pool.kv_buffer[i][PAGE_SIZE : 3 * PAGE_SIZE].clone()
        for i in range(LAYER_NUM)
    ]
    orig_index_k = [
        device_pool.index_k_buffer[i][PAGE_SIZE : 3 * PAGE_SIZE].clone()
        for i in range(LAYER_NUM)
    ]

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend
    )
    torch.cuda.synchronize()

    for layer_id in range(LAYER_NUM):
        device_pool.kv_buffer[layer_id][PAGE_SIZE : 3 * PAGE_SIZE].zero_()
        device_pool.index_k_buffer[layer_id][PAGE_SIZE : 3 * PAGE_SIZE].zero_()

    for layer_id in range(LAYER_NUM):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
    torch.cuda.synchronize()

    for layer_id in range(LAYER_NUM):
        assert torch.equal(
            device_pool.kv_buffer[layer_id][PAGE_SIZE : 3 * PAGE_SIZE],
            orig_latent[layer_id],
        )
        assert torch.equal(
            device_pool.index_k_buffer[layer_id][PAGE_SIZE : 3 * PAGE_SIZE],
            orig_index_k[layer_id],
        )
