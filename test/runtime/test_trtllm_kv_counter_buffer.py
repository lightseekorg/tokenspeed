"""The trtllm MHA leaf passes FlashInfer one persistent multi-CTA KV counter
buffer instead of letting it zero-fill a fresh one on every call.

Without the buffer, ``trtllm_batch_{decode,context}_with_kv_cache`` allocate
``torch.zeros(counter_bytes, uint8)`` per call: one extra fill kernel per
attention layer per step. The kernel resets its counters at the end of each
launch, so one buffer zeroed at construction serves every call. A batch the
buffer cannot hold falls back to FlashInfer's own allocation.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.paged import trtllm
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

PAGE = 64
HEADS = 8
HEAD_DIM = 128
MAX_BS = 8
SM_COUNT = 148


def _counter_bytes(batch_size: int, num_qo_heads: int, sm_count: int) -> int:
    # FlashInfer's get_trtllm_gen_multi_ctas_kv_counter_bytes.
    return -(-max(batch_size * num_qo_heads, sm_count) // 8) * 8 * 4


def _backend(device: str = "cpu") -> trtllm.TRTLLMMHAAttnBackend:
    spec = MHAConfig(
        backend_name="trtllm",
        num_attention_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    cfg = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=PAGE,
        kernel_page_size=PAGE,
        context_len=4096,
        max_bs=MAX_BS,
        kv_cache_quant_method="none",
        components=(spec,),
    )
    return trtllm.TRTLLMMHAAttnBackend(cfg, spec, kernel_page_size=PAGE)


@pytest.fixture
def backend(monkeypatch):
    """A CPU backend given the buffer a CUDA backend allocates, with the
    kernel calls captured instead of launched."""
    monkeypatch.setattr(
        trtllm, "get_trtllm_gen_multi_ctas_kv_counter_bytes", _counter_bytes
    )
    be = _backend()
    assert be._kv_counter_buffer is None
    be._sm_count = SM_COUNT
    be._kv_counter_buffer = torch.zeros(
        _counter_bytes(MAX_BS, HEADS, SM_COUNT), dtype=torch.uint8
    )
    be._save_kv_and_prepare_q = lambda q, *args: q.view(-1, HEADS, HEAD_DIM)
    be._get_kv_cache_permuted = lambda *args: (None, None)
    be.calls = []

    def launch(**kwargs):
        be.calls.append(kwargs)
        return torch.zeros_like(kwargs["query"])

    monkeypatch.setattr(trtllm, "trtllm_batch_decode_with_kv_cache", launch)
    monkeypatch.setattr(trtllm, "trtllm_batch_context_with_kv_cache", launch)
    return be


def _layer():
    return SimpleNamespace(
        tp_q_head_num=HEADS, head_dim=HEAD_DIM, scaling=1.0, sliding_window_size=-1
    )


def _metadata(bs: int, q_len: int = 1) -> trtllm.TRTLLMMHAMetadata:
    cu = torch.arange(0, bs * q_len + 1, q_len, dtype=torch.int32)
    return trtllm.TRTLLMMHAMetadata(
        cache_seqlens_int32=torch.full((bs,), 128, dtype=torch.int32),
        max_seq_len_q=q_len,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        page_table=torch.zeros((bs, 2), dtype=torch.int32),
    )


def _decode(be, bs: int):
    be.forward_decode_metadata = _metadata(bs)
    q = torch.zeros(bs, HEADS * HEAD_DIM)
    be.forward_decode(q, None, None, _layer(), None, None, bs)
    return be.calls[-1]["multi_ctas_kv_counter_buffer"]


def _extend(be, bs: int, q_len: int):
    be.forward_prefill_metadata = _metadata(bs, q_len)
    q = torch.zeros(bs * q_len, HEADS * HEAD_DIM)
    be.forward_extend(q, None, None, _layer(), None, None, bs)
    return be.calls[-1]["multi_ctas_kv_counter_buffer"]


def test_decode_and_extend_reuse_one_counter_buffer(backend):
    buf = backend._kv_counter_buffer
    assert _decode(backend, 1) is buf
    assert _decode(backend, MAX_BS) is buf
    assert _extend(backend, 3, 16) is buf
    assert _extend(backend, MAX_BS, 4) is buf


def test_batch_past_the_buffer_falls_back_to_flashinfer_allocation(backend):
    # 8 heads x 20 requests = 160 counters > the 152 the buffer holds.
    assert _decode(backend, 20) is None
    assert _extend(backend, 20, 2) is None
    # Below sm_count every batch needs the same sm_count-sized buffer.
    assert _decode(backend, MAX_BS + 1) is backend._kv_counter_buffer


def test_counter_buffer_outlives_a_cache_pool_rebind(backend):
    buf = backend._kv_counter_buffer
    backend.cache_pool = object()
    backend.set_cache_pool(object())
    assert backend._kv_counter_buffer is buf
    assert _decode(backend, 1) is buf


def test_cpu_backend_has_no_counter_buffer():
    be = _backend()
    assert be._kv_counter_buffer is None
    assert be._kv_counter_buffer_for(1, HEADS) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="needs an NVIDIA GPU",
)
def test_cuda_backend_allocates_a_zeroed_counter_buffer():
    be = _backend("cuda")
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    buf = be._kv_counter_buffer
    assert buf is not None and buf.dtype == torch.uint8 and buf.is_cuda
    assert buf.numel() == trtllm.get_trtllm_gen_multi_ctas_kv_counter_bytes(
        MAX_BS, HEADS, sm_count
    )
    assert not buf.any()
    assert be._kv_counter_buffer_for(MAX_BS, HEADS) is buf
