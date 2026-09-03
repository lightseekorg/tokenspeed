"""Draft seq_lens must stay visible to the backend now that it owns the buffer.

The drafter edits ``draft_seq_lens_buf`` in place inside the captured graph --
``add_(1)`` per multi-step iteration, and ``_apply_correction`` trimming the
rejected tail at step 0. Backends read their own buffer, so both edits must be
published via ``advance_draft_forward_metadata`` or the draft attends over a
stale prefix and the accept rate silently drops. CPU-only: drives the metadata
builders and the hook directly, no GPU or KV pool.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    get_capture_warmup_seq_len,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.msa import (
    MSAAttnBackend,
    MSAHybridAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.trtllm import (
    TRTLLMMHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig


def _cfg() -> AttnConfig:
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=128,
        attn_tp_size=1,
    )
    return AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=64,
        kernel_page_size=64,
        context_len=4096,
        max_bs=8,
        max_graph_bs=8,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        is_draft=True,
        components=(spec,),
    )


def _mha_backend(backend_cls=MHAAttnBackend):
    cfg = _cfg()
    return backend_cls(cfg, cfg.component(SoftmaxAttnConfig))


def _seqlens_field(be, metadata):
    # The KV cache-seqlens field differs by backend; both alias the owned buffer.
    if isinstance(be, TRTLLMMHAAttnBackend):
        return metadata.cache_seqlens_int32
    return metadata.seq_lens


@pytest.mark.parametrize("spec_width", [1, 2, 4, 8])
def test_mtp_capture_warmup_seq_len_survives_worst_case_verify(spec_width):
    capture_len = get_capture_warmup_seq_len(spec_width, has_drafter=True)
    post_verify_len = capture_len - (spec_width - 1)
    assert post_verify_len >= spec_width


def test_non_spec_capture_warmup_seq_len_is_unchanged():
    assert get_capture_warmup_seq_len(1, has_drafter=False) == 1


@pytest.mark.parametrize("backend_cls", [MHAAttnBackend, TRTLLMMHAAttnBackend])
def test_advance_updates_draft_decode_metadata(backend_cls):
    be = _mha_backend(backend_cls)
    max_bs, bs = 8, 4
    be.init_cuda_graph_state(max_bs)

    # Capture single-token decode metadata (plain draft path, is_draft=True).
    req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int32)
    capture_seq_lens = torch.full((max_bs,), 5, dtype=torch.int32)
    be.init_forward_metadata_capture_cuda_graph(
        bs, req_pool_indices, capture_seq_lens, ForwardMode.DECODE
    )
    field = _seqlens_field(be, be.forward_decode_metadata)

    # The metadata field must be a view of the owned buffer.
    owned = (
        be.cuda_graph_cache_seqlens
        if isinstance(be, TRTLLMMHAAttnBackend)
        else be.cuda_graph_seq_lens
    )
    assert field.data_ptr() == owned[:bs].data_ptr()

    # Simulate the drafter advancing lengths in-graph across draft steps.
    for step_len in (11, 12, 13):
        draft_seq_lens = torch.full((bs,), step_len, dtype=torch.int32)
        be.advance_draft_forward_metadata(draft_seq_lens)
        got = _seqlens_field(be, be.forward_decode_metadata)[:bs]
        assert torch.equal(got, draft_seq_lens), (
            f"{backend_cls.__name__}: draft decode seqlens did not advance to "
            f"{step_len}; got {got.tolist()}"
        )


def test_advance_does_not_mutate_caller_tensor():
    be = _mha_backend()
    be.init_cuda_graph_state(8)
    draft_seq_lens = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    original = draft_seq_lens.clone()
    be.advance_draft_forward_metadata(draft_seq_lens)
    # advance copies OUT of the caller tensor; it must not be written to.
    assert torch.equal(draft_seq_lens, original)
    assert torch.equal(be.cuda_graph_seq_lens[:4], original)


def _msa_backend() -> MSAAttnBackend:
    spec = MSAConfig(
        backend_name="msa",
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=128,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=64,
        kernel_page_size=64,
        context_len=4096,
        max_bs=8,
        max_graph_bs=8,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        is_draft=True,
        components=(spec,),
    )
    return MSAAttnBackend(config, spec)


def test_msa_init_cuda_graph_state_matches_helper_signature():
    """msa must take the post-ownership signature (no seq_lens_buf).

    It was left on the old signature while the shared parameter was dropped
    from init_backend_cuda_graph_state, so every MiniMax cuda-graph startup
    raised TypeError: missing 1 required positional argument: 'seq_lens_buf'.
    """
    be = _msa_backend()
    init_backend_cuda_graph_state(be, 8, cache_group_specs=())
    # And it must own the buffer rather than alias a controller tensor.
    assert be.cuda_graph_seq_lens.shape[0] == 8
    assert be.cuda_graph_seq_lens.dtype == torch.int32


def test_msa_inherits_default_advance():
    """msa's buffer uses the default name, so the base implementation applies."""
    be = _msa_backend()
    init_backend_cuda_graph_state(be, 8, cache_group_specs=())
    seq_lens = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    be.advance_draft_forward_metadata(seq_lens)
    assert torch.equal(be.cuda_graph_seq_lens[:4], seq_lens)


def test_msa_hybrid_composes_cache_contract_from_children():
    class ChildBackend:
        def __init__(self, families):
            self.cache_consumer_families = frozenset(families)
            self.cache_pool = None

        def set_cache_pool(self, cache_pool):
            self.cache_pool = cache_pool

    dense = ChildBackend({"history"})
    sparse = ChildBackend({"history"})
    backend = object.__new__(MSAHybridAttnBackend)
    backend.full_attn_backend = dense
    backend.sparse_attn_backend = sparse
    pool = object()

    backend.set_cache_pool(pool)

    assert backend.cache_consumer_families == frozenset({"history"})
    assert backend.cache_pool is pool
    assert dense.cache_pool is pool
    assert sparse.cache_pool is pool


def test_default_advance_is_a_noop_before_graph_state_exists():
    """The hook may fire before init_cuda_graph_state (eager runs); it must not
    raise, just do nothing."""
    be = _mha_backend()
    be.advance_draft_forward_metadata(torch.tensor([5, 6], dtype=torch.int32))


@pytest.mark.parametrize("backend_cls", [MHAAttnBackend, TRTLLMMHAAttnBackend])
def test_capture_seeds_owned_seqlens(backend_cls):
    """Capture must seed the owned buffer: the capture run reads it.

    The buffer is zero-initialized and only replay copies live lengths in, so
    without seeding the warmup/capture forward attends over seq_len 0 (empty
    causal span -> NaN, or a schedule recorded against zero lengths).
    """
    be = _mha_backend(backend_cls)
    max_bs, bs = 8, 4
    be.init_cuda_graph_state(max_bs)
    capture_seq_lens = torch.full((max_bs,), 37, dtype=torch.int32)
    be.init_forward_metadata_capture_cuda_graph(
        bs,
        torch.arange(1, bs + 1, dtype=torch.int32),
        capture_seq_lens,
        ForwardMode.DECODE,
    )
    got = _seqlens_field(be, be.forward_decode_metadata)[:bs]
    assert torch.equal(
        got, torch.full((bs,), 37, dtype=torch.int32)
    ), f"{backend_cls.__name__}: capture left seqlens at {got.tolist()}"


def test_hybrid_composite_forwards_advance_to_full_attn_child():
    """Composite backends own no buffer; they must forward to the child."""
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        HybridLinearAttnBackend,
    )

    full = _mha_backend()
    full.init_cuda_graph_state(8)
    hybrid = object.__new__(HybridLinearAttnBackend)
    hybrid.full_attn_backend = full
    hybrid.linear_attn_backend = None

    seq_lens = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    hybrid.advance_draft_forward_metadata(seq_lens)
    assert torch.equal(full.cuda_graph_seq_lens[:4], seq_lens)


# Step 0: `_apply_correction` trims the rejected tail (vc + N -> vc + a).


def _correction_models():
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3DraftAttentionMLA
    from tokenspeed.runtime.models.glm_moe_dsa_nextn import GlmMoeDsaForCausalLMNextN
    from tokenspeed.runtime.models.llama_eagle3 import LlamaAttention
    from tokenspeed.runtime.models.qwen3_5_nextn import (
        Qwen3_5DraftAttentionDecoderLayer,
    )

    return [
        ("llama_eagle3", LlamaAttention._apply_correction),
        ("qwen3_5_nextn", Qwen3_5DraftAttentionDecoderLayer._apply_correction),
        ("deepseek_v3", DeepseekV3DraftAttentionMLA._apply_correction),
        ("glm_moe_dsa_nextn", GlmMoeDsaForCausalLMNextN._apply_first_step_correction),
    ]


@pytest.mark.parametrize("name,correction", _correction_models())
def test_first_step_correction_reaches_backend(name, correction):
    from types import SimpleNamespace

    be = _mha_backend()
    max_bs, bs = 8, 4
    be.init_cuda_graph_state(max_bs)

    # Target verify left seq_lens at vc + N (vc=100, N=4).
    draft_seq_lens = torch.full((bs,), 104, dtype=torch.int32)
    be.init_forward_metadata_capture_cuda_graph(
        bs,
        torch.arange(1, bs + 1, dtype=torch.int32),
        draft_seq_lens,
        ForwardMode.DECODE,
    )

    accept_lengths = torch.tensor([2, 1, 4, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        draft_seq_lens_buf=draft_seq_lens,
        accept_lengths=accept_lengths,
        num_extends=0,
        bs=bs,
        attn_backend=be,
    )
    # Bound methods on the class need an explicit (unused) self.
    try:
        correction(ctx)
    except TypeError:
        correction(None, ctx)

    expected = torch.tensor([102, 101, 104, 103], dtype=torch.int32)  # vc + a
    assert torch.equal(draft_seq_lens, expected), f"{name}: correction wrong"
    assert torch.equal(be.forward_decode_metadata.seq_lens[:bs], expected), (
        f"{name}: trimmed seqlens never reached the backend; draft step 0 "
        f"would attend over the rejected tail"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
