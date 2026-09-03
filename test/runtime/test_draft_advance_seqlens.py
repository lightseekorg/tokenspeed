"""Draft seq_lens must stay visible to the backend now that it owns the buffer.

The drafter edits ``draft_seq_lens_buf`` in place inside the captured graph --
``add_(1)`` per multi-step iteration -- and publishes the accepted frontier
for step 0 through ``AcceptedPrefixPublisher`` (the ``ctx.draft_narrowing``
handle the draft model fires when it switches to its live rows). Backends read
their own buffer, so both must land via ``advance_draft_forward_metadata`` or
the draft attends over a stale prefix and the accept rate silently drops.
CPU-only: drives the leaf metadata builders and the hook directly, no GPU or
KV pool.
"""

from __future__ import annotations

import pytest
import torch

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

MAX_NUM_PAGES = 64  # context 4096 / kernel page 64


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
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        is_draft=True,
        components=(spec,),
    )


def _mha_backend(backend_cls=MHAAttnBackend):
    cfg = _cfg()
    return backend_cls(cfg, cfg.component(SoftmaxAttnConfig), kernel_page_size=64)


def _seqlens_field(be, metadata):
    # The KV cache-seqlens field differs by backend; both alias the owned buffer.
    if isinstance(be, TRTLLMMHAAttnBackend):
        return metadata.cache_seqlens_int32
    return metadata.seq_lens


def _capture(be, bs, seq_lens):
    page_table = torch.zeros((bs, be.max_num_pages), dtype=torch.int32)
    be.init_forward_metadata_capture_cuda_graph(bs, seq_lens, page_table)


@pytest.mark.parametrize("backend_cls", [MHAAttnBackend, TRTLLMMHAAttnBackend])
def test_advance_updates_draft_decode_metadata(backend_cls):
    be = _mha_backend(backend_cls)
    max_bs, bs = 8, 4
    be.init_cuda_graph_state(max_bs)

    # Capture single-token decode metadata (plain draft path, is_draft=True).
    capture_seq_lens = torch.full((max_bs,), 5, dtype=torch.int32)
    _capture(be, bs, capture_seq_lens)
    field = _seqlens_field(be, be.forward_decode_metadata)

    # The metadata field must be a view of the owned buffer (one home for
    # every leaf: decode_seq_lens_buffer).
    assert field.data_ptr() == be.decode_seq_lens_buffer[:bs].data_ptr()

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
    assert torch.equal(be.decode_seq_lens_buffer[:4], original)


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
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        is_draft=True,
        components=(spec,),
    )
    return MSAAttnBackend(config, spec, kernel_page_size=64)


def test_msa_owns_its_graph_seqlens_buffer():
    be = _msa_backend()
    be.init_cuda_graph_state(8)
    # It must own the buffer rather than alias a controller tensor.
    assert be.decode_seq_lens_buffer.shape[0] == 8
    assert be.decode_seq_lens_buffer.dtype == torch.int32


def test_msa_inherits_default_advance():
    """One buffer name for every leaf, so the base implementation applies."""
    be = _msa_backend()
    be.init_cuda_graph_state(8)
    seq_lens = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    be.advance_draft_forward_metadata(seq_lens)
    assert torch.equal(be.decode_seq_lens_buffer[:4], seq_lens)


def test_msa_hybrid_fans_pool_binding_out_to_both_routers():
    class ChildRouter:
        def __init__(self):
            self.cache_pool = None

        def set_cache_pool(self, cache_pool):
            self.cache_pool = cache_pool

    dense = ChildRouter()
    sparse = ChildRouter()
    backend = object.__new__(MSAHybridAttnBackend)
    backend.full_router = dense
    backend.sparse_router = sparse
    pool = object()

    backend.set_cache_pool(pool)

    assert backend.cache_pool is pool
    assert dense.cache_pool is pool
    assert sparse.cache_pool is pool


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
    _capture(be, bs, capture_seq_lens)
    got = _seqlens_field(be, be.forward_decode_metadata)[:bs]
    assert torch.equal(
        got, torch.full((bs,), 37, dtype=torch.int32)
    ), f"{backend_cls.__name__}: capture left seqlens at {got.tolist()}"


def test_router_fans_advance_out_to_every_leaf():
    """The runner-facing router owns no buffer; it must fan out to leaves."""
    from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
        CacheGroupGeometry,
    )
    from tokenspeed.runtime.layers.attention.backends.router import (
        CacheGroupRouter,
    )

    leaf = _mha_backend()
    leaf.init_cuda_graph_state(8)
    router = CacheGroupRouter(None, is_draft=True, spec_num_tokens=4, device="cpu")
    router.bind(
        CacheGroupGeometry(
            granularities={"full_attention": 64},
            families={"full_attention": "history"},
            full_history_group_id="full_attention",
        ),
        {"full_attention": leaf},
    )

    seq_lens = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    router.advance_draft_forward_metadata(seq_lens)
    assert torch.equal(leaf.decode_seq_lens_buffer[:4], seq_lens)


def test_hybrid_composite_forwards_advance_to_full_attn_child():
    """Composite backends own no buffer; they must forward to the child."""
    from tokenspeed.runtime.layers.attention.backends.hybrid import (
        HybridLinearAttnBackend,
    )

    full = _mha_backend()
    full.init_cuda_graph_state(8)
    hybrid = object.__new__(HybridLinearAttnBackend)
    hybrid.full_attn_backend = full
    hybrid.linear_attn_backend = None

    seq_lens = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    hybrid.advance_draft_forward_metadata(seq_lens)
    assert torch.equal(full.decode_seq_lens_buffer[:4], seq_lens)


# Step 0: the drafter publishes the accepted frontier (vc + N -> vc + a); the
# model only names the moment.


def _make_eagle_for_frontier(bs: int, seq_lens, req_pool_indices, valid_cache_lengths):
    from types import SimpleNamespace

    from tokenspeed.runtime.execution.drafter.eagle import Eagle

    drafter = Eagle.__new__(Eagle)
    drafter.input_buffers = SimpleNamespace(
        seq_lens_buf=torch.tensor(seq_lens, dtype=torch.int32),
        req_pool_indices_buf=torch.tensor(req_pool_indices, dtype=torch.int64),
    )
    drafter.runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.tensor(valid_cache_lengths, dtype=torch.int32)
    )
    return drafter


def test_accepted_frontier_is_vc_plus_accept_for_decode_rows():
    from types import SimpleNamespace

    # Target verify left seq_lens at vc + N (vc=100, N=4); prompt rows keep
    # their own lengths.
    drafter = _make_eagle_for_frontier(
        bs=4,
        seq_lens=[17, 104, 104, 104],
        req_pool_indices=[3, 0, 1, 2],
        valid_cache_lengths=[100, 100, 100, 5],
    )
    draft_input = SimpleNamespace(
        num_extends=1,
        accept_lengths=torch.tensor([1, 2, 1, 4], dtype=torch.int32),
    )

    frontier = drafter._accepted_frontier(4, draft_input)

    assert frontier.tolist() == [17, 102, 101, 104]
    # A fresh tensor: publishing it must not alias the runner's seq_lens buffer.
    assert frontier.data_ptr() != drafter.input_buffers.seq_lens_buf.data_ptr()


def test_publish_accepted_prefix_reaches_backend_and_is_idempotent():
    from tokenspeed.runtime.execution.drafter.eagle import AcceptedPrefixPublisher

    be = _mha_backend()
    max_bs, bs = 8, 4
    be.init_cuda_graph_state(max_bs)

    draft_seq_lens = torch.full((bs,), 104, dtype=torch.int32)
    _capture(be, bs, draft_seq_lens)

    frontier = torch.tensor([102, 101, 104, 103], dtype=torch.int32)  # vc + a
    narrowing = AcceptedPrefixPublisher(be, frontier)
    narrowing.publish_accepted_prefix()
    assert torch.equal(be.forward_decode_metadata.seq_lens[:bs], frontier), (
        "accepted frontier never reached the backend; draft step 0 would "
        "attend over the rejected tail"
    )
    # Every layer of a multi-layer draft may fire it: no double trim.
    narrowing.publish_accepted_prefix()
    assert torch.equal(be.forward_decode_metadata.seq_lens[:bs], frontier)
    # The drafter's own buffer is untouched -- the backend owns the copy.
    assert torch.equal(draft_seq_lens, torch.full((bs,), 104, dtype=torch.int32))


def test_draft_narrowing_is_the_only_step0_flag_on_forward_context():
    """The drafter tensors left ForwardContext: the narrowing handle is the
    step-0 discriminator, with no ``accept_lengths`` / ``draft_seq_lens_buf``
    fields to fall back on."""
    from dataclasses import fields

    from tokenspeed.runtime.execution.context import ForwardContext

    names = {f.name for f in fields(ForwardContext)}
    assert "draft_narrowing" in names
    assert not {"accept_lengths", "draft_seq_lens_buf"} & names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
