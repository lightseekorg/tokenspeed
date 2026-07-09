"""Regression: trtllm decode metadata must clamp KV seqlens to >=
spec_num_tokens on the multi-token (MTP target-verify) path.

Root cause analyzed in dashllm1.log (DP4+EP4 Qwen3.5-397B, MTP, CUDA graph ON):

    accept_rate decays to 0 over multiple rounds. Localized end-to-end:
      * CUDA-graph padded / idle decode rows are filled with seq_len=1
        (InputBuffer.seq_lens_buf[batch_size:].fill_(1)).
      * In MTP target-verify q_len_per_req == spec_num_draft_tokens (e.g. 4).
      * A row with seq_len=1 < q_len=4 gives query positions whose causal key
        span is empty; the trtllm decode kernel returns NaN for those rows.
      * That NaN enters the residual stream of a full-attn layer and propagates
        to every downstream layer, including the linear-attn (mamba) layers,
        whose conv/ssm state then accumulates NaN -> accept_rate collapses.

The MHA backend already guards this (``seq_lens[:bs].clamp_min(
self.spec_num_tokens)``). This test pins the equivalent guard for the trtllm
backend: ``_init_multi_token_metadata`` (and the CUDA-graph capture builder)
must expose ``cache_seqlens_int32 >= spec_num_tokens`` so padded rows keep a
non-empty causal span. Plain single-token decode (q_len=1) is unaffected.

The test builds a CPU-only backend (``device="cpu"``) so it needs no GPU and no
real KV pool; it exercises the metadata builders directly.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.trtllm import (
    TRTLLMMHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

SPEC_NUM_TOKENS = 4


def _make_backend(is_draft: bool = False) -> TRTLLMMHAAttnBackend:
    cfg = MHAConfig(
        device="cpu",
        backend_name="trtllm",
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=128,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        context_len=4096,
        max_bs=8,
        max_graph_bs=8,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=SPEC_NUM_TOKENS,
        is_draft=is_draft,
    )
    return TRTLLMMHAAttnBackend(cfg)


def _req_to_page(req_pool_size: int, max_pages: int) -> torch.Tensor:
    # Each request maps to a couple of distinct page ids; row 0 is the
    # reserved/padding row. Values are arbitrary but valid (>=0).
    table = torch.zeros((req_pool_size + 1, max_pages), dtype=torch.int32)
    for r in range(req_pool_size + 1):
        table[r, 0] = r
    return table


def test_multi_token_metadata_clamps_padded_seqlen_runtime():
    """Runtime (non-CUDA-graph) verify path clamps seq_len < spec_num_tokens."""
    be = _make_backend()
    bs = 4
    # Two real rows (long context) + two "padded" rows with seq_len=1, exactly
    # the layout that triggered the NaN (real_bs=2, padded to 4).
    seq_lens = torch.tensor([512, 300, 1, 1], dtype=torch.int32)
    req_pool_indices = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    req_to_page = _req_to_page(req_pool_size=8, max_pages=be.max_num_pages)

    be._init_multi_token_metadata(
        bs, SPEC_NUM_TOKENS, req_pool_indices, seq_lens, req_to_page
    )
    cache_seqlens = be.forward_prefill_metadata.cache_seqlens_int32

    # Every row must have at least spec_num_tokens keys so no query row in the
    # uniform q_len=spec_num_tokens block gets an empty causal span.
    assert int(cache_seqlens.min()) >= SPEC_NUM_TOKENS
    # Real rows must be left untouched (clamp is a no-op for them).
    assert int(cache_seqlens[0]) == 512
    assert int(cache_seqlens[1]) == 300
    # Padded rows get raised to spec_num_tokens.
    assert int(cache_seqlens[2]) == SPEC_NUM_TOKENS
    assert int(cache_seqlens[3]) == SPEC_NUM_TOKENS


def test_clamp_does_not_mutate_shared_seq_lens():
    """The clamp must write into a private buffer, never the caller's tensor."""
    be = _make_backend()
    bs = 3
    seq_lens = torch.tensor([1, 1, 256], dtype=torch.int32)
    original = seq_lens.clone()
    req_pool_indices = torch.tensor([0, 0, 1], dtype=torch.int32)
    req_to_page = _req_to_page(req_pool_size=8, max_pages=be.max_num_pages)

    be._init_multi_token_metadata(
        bs, SPEC_NUM_TOKENS, req_pool_indices, seq_lens, req_to_page
    )

    # Caller's seq_lens is unchanged; clamped values live in a separate buffer.
    assert torch.equal(seq_lens, original)
    cache_seqlens = be.forward_prefill_metadata.cache_seqlens_int32
    assert cache_seqlens.data_ptr() != seq_lens.data_ptr()
    assert int(cache_seqlens.min()) >= SPEC_NUM_TOKENS


def test_plain_decode_seqlen_not_clamped():
    """Plain single-token decode (q_len=1) must NOT be clamped: a seq_len of 1
    is valid there (1 query attends to 1 key)."""
    be = _make_backend()
    bs = 3
    seq_lens = torch.tensor([1, 5, 9], dtype=torch.int32)
    req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int32)
    req_to_page = _req_to_page(req_pool_size=8, max_pages=be.max_num_pages)

    be._init_decode_metadata(bs, req_pool_indices, seq_lens, req_to_page)
    cache_seqlens = be.forward_decode_metadata.cache_seqlens_int32

    # Decode path aliases seq_lens verbatim (no clamp): seq_len=1 stays 1.
    assert int(cache_seqlens[0]) == 1
    assert torch.equal(cache_seqlens, seq_lens[:bs])


def test_cuda_graph_capture_builder_clamps():
    """CUDA-graph capture builder points cache_seqlens at the clamped buffer."""
    be = _make_backend()
    max_bs = 8
    seq_lens_buf = torch.ones((max_bs,), dtype=torch.int32)  # all padded -> 1
    be.init_cuda_graph_state(max_bs, seq_lens_buf)

    bs = 4
    be._init_multi_token_metadata_capture(bs, SPEC_NUM_TOKENS)
    cache_seqlens = be.forward_prefill_metadata.cache_seqlens_int32

    assert int(cache_seqlens.min()) >= SPEC_NUM_TOKENS
    # Must be the dedicated clamped buffer, not the shared seq_lens_buf.
    assert cache_seqlens.data_ptr() == be.spec_cache_seqlens_buf.data_ptr()
    assert cache_seqlens.data_ptr() != seq_lens_buf.data_ptr()


def test_draft_replay_refreshes_spec_cache_seqlens_buf():
    """Draft replay must refresh spec_cache_seqlens_buf (draft step 1 is multi-token)."""
    be = _make_backend(is_draft=True)
    max_bs = 8
    # At capture time, seq_lens_buf is all 1s (padded rows).
    seq_lens_buf = torch.ones((max_bs,), dtype=torch.int32)
    be.init_cuda_graph_state(max_bs, seq_lens_buf)

    bs = 4
    # Capture: multi-token metadata (step 1) + decode metadata (steps 2+).
    be._init_multi_token_metadata_capture(bs, SPEC_NUM_TOKENS)
    be._init_decode_metadata_capture(bs, seq_lens_buf)

    # At capture, spec_cache_seqlens_buf was seeded from all-ones → clamped to 4.
    capture_vals = be.spec_cache_seqlens_buf[:bs].clone()
    assert int(capture_vals.min()) >= SPEC_NUM_TOKENS
    assert int(capture_vals[0]) == SPEC_NUM_TOKENS  # 1 → clamped to 4

    # Replay with real seq_lens (two real rows + two padded).
    real_seq_lens = torch.tensor([512, 300, 1, 1], dtype=torch.int32)
    req_pool_indices = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    be.init_forward_metadata_replay_cuda_graph(
        bs,
        req_pool_indices,
        real_seq_lens,
        ForwardMode.DECODE,
        req_to_page=None,  # skip page-table gather (Triton kernel)
    )

    # spec_cache_seqlens_buf must now reflect the clamped real values.
    replay_vals = be.spec_cache_seqlens_buf[:bs]
    assert int(replay_vals[0]) == 512  # real, no clamp needed
    assert int(replay_vals[1]) == 300  # real, no clamp needed
    assert int(replay_vals[2]) == SPEC_NUM_TOKENS  # 1 → clamped to 4
    assert int(replay_vals[3]) == SPEC_NUM_TOKENS  # 1 → clamped to 4

    # forward_prefill_metadata must point at spec_cache_seqlens_buf.
    prefill_cache_seqlens = be.forward_prefill_metadata.cache_seqlens_int32
    assert prefill_cache_seqlens.data_ptr() == be.spec_cache_seqlens_buf.data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
