# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Qwen4 cache consumers are assembled per view and budgeted independently."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tokenspeed.runtime.layers.attention.registry as registry
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import (
    _compose_qwen4_exp_backend,
    _prepare_verify_workspace,
)


def _config(*, is_draft: bool, width: int):
    spec = SimpleNamespace(
        num_attention_heads=8, num_kv_heads=1, attn_tp_size=2, head_dim=16
    )
    return SimpleNamespace(
        component=lambda component_type: spec,
        device="cpu",
        dtype=torch.bfloat16,
        is_draft=is_draft,
        speculative_num_draft_tokens=width,
        context_len=512,
        max_bs=4,
    )


@pytest.mark.parametrize("has_qsa", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize(
    "is_draft,has_ple", [(False, False), (False, True), (True, False)]
)
def test_composition_uses_local_fields_without_requiring_linear_layers(
    has_ple, has_qsa, is_draft, hybrid, width
):
    config = _config(is_draft=is_draft, width=width)
    groups = [QWEN4_EXP_PLE_CACHE_GROUP] if has_ple else []
    if has_qsa:
        groups += [QWEN4_EXP_QSA_CACHE_GROUP, QWEN4_EXP_QSA_RECENT_CACHE_GROUP]
    fields = [
        SimpleNamespace(field_id=f"layer.3.{group}", group_id=group) for group in groups
    ]
    # A shared arena also publishes other views' fields. They must not create
    # consumers in this view merely because a family exists in the contract.
    fields += [
        SimpleNamespace(field_id=f"layer.9.{group}", group_id=group)
        for group in (
            QWEN4_EXP_PLE_CACHE_GROUP,
            QWEN4_EXP_QSA_CACHE_GROUP,
            QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
        )
    ]
    pool = SimpleNamespace(
        field_layer_range=range(3, 4),
        layer_num=1,
        arena=SimpleNamespace(plan=SimpleNamespace(fields=fields)),
    )
    full = SimpleNamespace(device="cpu", spec_num_tokens=width)
    attention = (
        HybridLinearAttnBackend(full, SimpleNamespace(), [3]) if hybrid else full
    )
    backend = _compose_qwen4_exp_backend(config, pool, attention)
    assert backend.attention_backend is attention
    assert (backend.ple_backend is not None) == has_ple
    assert (backend.indexer_backend is not None) == has_qsa
    # The public contract is available even when the child has no head/dtype fields.
    assert backend.device == "cpu"
    assert backend.dtype == torch.bfloat16
    assert backend.is_draft is is_draft
    assert backend.spec_num_tokens == width
    assert backend.num_qo_heads == 4
    assert backend.num_kv_heads == 1
    assert backend.head_dim == 16
    assert backend.cache_pool is None


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("has_gdn", [False, True])
@pytest.mark.parametrize(
    "has_ple,has_qsa", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("is_draft", [False, True])
def test_verify_workspace_counts_each_consumer_once_and_checks_zero_budget(
    width, has_gdn, has_ple, has_qsa, is_draft
):
    calls = []

    def consumer(name, nbytes):
        def preallocate(max_bs, draft_token_num):
            calls.append((name, max_bs, draft_token_num))
            return nbytes

        return SimpleNamespace(preallocate_verify_workspace=preallocate)

    attention = SimpleNamespace(device="cpu")
    if has_gdn:
        attention = HybridLinearAttnBackend(attention, consumer("gdn", 3), [0])
    config = _config(is_draft=is_draft, width=width)
    config.max_bs = 2
    root = Qwen4ExpBackend(
        config,
        attention,
        consumer("ple", 5) if has_ple else None,
        consumer("qsa", 7) if has_qsa else None,
    )
    kwargs = dict(
        server_args=SimpleNamespace(speculative_num_draft_tokens=width),
        config=config,
        backend=root,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
    )
    target_verify = width > 1 and not is_draft
    expected_bytes = (3 * has_gdn + 5 * has_ple + 7 * has_qsa) if target_verify else 0
    _prepare_verify_workspace(**kwargs, expected_bytes=expected_bytes)
    assert calls == (
        ([("gdn", 2, width)] if has_gdn else [])
        + ([("ple", 2, width)] if has_ple else [])
        + ([("qsa", 2, width)] if has_qsa else [])
        if target_verify
        else []
    )
    with pytest.raises(RuntimeError, match="does not match allocated tensors"):
        _prepare_verify_workspace(**kwargs, expected_bytes=expected_bytes + 1)


@pytest.mark.parametrize("width,expected_bytes", [(1, 0), (4, 12)])
def test_registry_preallocates_qwen4_without_inspecting_children(width, expected_bytes):
    # A registry-facing composite needs only the operation, not its child fields.
    root = object.__new__(Qwen4ExpBackend)
    root.preallocate_verify_workspace = Mock(return_value=expected_bytes)
    _prepare_verify_workspace(
        server_args=SimpleNamespace(speculative_num_draft_tokens=width),
        config=SimpleNamespace(max_bs=2),
        backend=root,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
        expected_bytes=expected_bytes,
    )
    root.preallocate_verify_workspace.assert_called_once_with(2, width)


@pytest.mark.parametrize("is_qwen4", [False, True])
@pytest.mark.parametrize(
    "has_linear_config,has_local_state", [(False, False), (True, False), (True, True)]
)
def test_hybrid_factory_binds_gdn_only_for_local_state(
    monkeypatch, is_qwen4, has_linear_config, has_local_state
):
    # Load the subclass before replacing its base class with a constructor mock.
    import_module("tokenspeed.runtime.layers.attention.backends.state.kda")
    full = SimpleNamespace(device="cpu", spec_num_tokens=1)
    gdn = SimpleNamespace(set_kv_pool=Mock(), commit_verified_state=Mock())
    factory = Mock(return_value=gdn)
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.backends.state.mamba.MambaAttnBackend",
        factory,
    )
    config = _config(is_draft=False, width=1)
    components = {
        SoftmaxAttnConfig: config.component(SoftmaxAttnConfig),
        LinearAttnConfig: (
            SimpleNamespace(layer_ids=(0 if has_local_state else 3,))
            if has_linear_config
            else None
        ),
    }
    config.component = components.get
    pool = SimpleNamespace(
        state_group_by_layer={0: "linear_attention"} if has_local_state else {},
        field_layer_range=range(2),
        layer_num=2,
        arena=SimpleNamespace(plan=SimpleNamespace(fields=[])),
    )
    monkeypatch.setattr(registry, "is_qwen4_exp", lambda hf_config: is_qwen4)
    monkeypatch.setattr(
        registry, "_create_attn_backend_with_name", lambda name, arch, config: full
    )
    backend = registry._create_hybrid_linear_attn_backend(
        SimpleNamespace(speculative_algorithm=None, kda_backend="auto"),
        SimpleNamespace(
            hf_config=SimpleNamespace(full_attention_layer_ids=[1]),
            attention_arch="mha",
        ),
        config,
        pool=pool,
        full_attn_backend_name=None,
        is_kda=False,
    )
    assert isinstance(backend, Qwen4ExpBackend) == is_qwen4
    attention = backend.attention_backend if is_qwen4 else backend
    if has_local_state:
        assert isinstance(attention, HybridLinearAttnBackend)
        assert attention.full_attn_backend is full
        factory.assert_called_once_with(config, components[SoftmaxAttnConfig])
        gdn.set_kv_pool.assert_called_once_with(pool)
        accepted = torch.tensor([1, 3], dtype=torch.int32)
        backend.update_mamba_state_after_mtp_verify(accepted)
        gdn.commit_verified_state.assert_called_once_with(accepted)
    else:
        factory.assert_not_called()
        assert attention is full


@pytest.fixture(params=[False, True], ids=["router", "hybrid"])
def attention_root(request):
    router = CacheGroupRouter(
        lambda group, page_size: None,
        is_draft=False,
        spec_num_tokens=4,
        device="cpu",
        consumed_group_ids=(FULL_ATTENTION,),
    )
    attention = (
        HybridLinearAttnBackend(router, SimpleNamespace(), [0])
        if request.param
        else router
    )
    return (
        Qwen4ExpBackend(_config(is_draft=False, width=4), attention, None, None),
        router,
    )


def test_draft_hooks_and_sparse_share_reach_the_full_router(attention_root):
    root, router = attention_root
    seq_lens = torch.zeros(2, dtype=torch.int32)
    leaf = SimpleNamespace(
        advance_draft_forward_metadata=Mock(wraps=seq_lens.copy_),
        fill_block_decode_seq_lens=lambda bs, out: out[:bs].copy_(seq_lens[:bs]),
    )
    router.leaves = {FULL_ATTENTION: leaf}
    indexer = SimpleNamespace(
        advance_draft_forward_metadata=Mock(),
        update_draft_forward_metadata=Mock(),
        fill_block_decode_seq_lens=Mock(),
    )
    root.indexer_backend = indexer
    advance = torch.tensor([8, 12], dtype=torch.int32)
    frontier = torch.tensor([5, 9], dtype=torch.int32)
    root.advance_draft_forward_metadata(advance)
    torch.testing.assert_close(seq_lens, advance)
    root.update_draft_forward_metadata(frontier)
    lengths = torch.full((3,), -1, dtype=torch.int32)
    root.fill_block_decode_seq_lens(2, lengths)
    assert lengths.tolist() == [5, 9, -1]
    assert leaf.advance_draft_forward_metadata.call_count == 2
    indexer.advance_draft_forward_metadata.assert_called_once_with(advance)
    indexer.update_draft_forward_metadata.assert_called_once_with(frontier)
    indexer.fill_block_decode_seq_lens.assert_called_once_with(2, lengths)
    root.sparse_topk.decode = frontier
    assert router.sparse_topk.decode is frontier
    router.sparse_topk.clear()
    assert root.sparse_topk.decode is None


def test_runtime_and_draft_setup_reach_attention_leaves(attention_root):
    root, router = attention_root
    leaf = SimpleNamespace(
        configure_runtime=Mock(),
        init_prefill_graph_state=Mock(),
        supports_layer_sliding_window=True,
    )
    router.leaves = {FULL_ATTENTION: leaf}
    pool = object()
    root.configure_runtime(token_to_kv_pool=pool)
    root.init_prefill_graph_state(16, 4)
    leaf.configure_runtime.assert_called_once_with(token_to_kv_pool=pool)
    leaf.init_prefill_graph_state.assert_called_once_with(16, 4)
    assert root.supports_layer_sliding_window
    slots = torch.empty(4, dtype=torch.int32)
    starts = torch.tensor([8, 12], dtype=torch.int32)
    router.draft_write_locations_uniform = Mock(return_value=slots)
    assert root.draft_write_locations_uniform(slots, starts, 2) is slots
    router.draft_write_locations_uniform.assert_called_once_with(slots, starts, 2)
