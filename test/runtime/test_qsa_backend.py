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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.paged.qsa import QSAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_raw_key_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.qsa.runtime import (
    QSARuntime,
    bind_qsa_indexers,
)
from tokenspeed.runtime.layers.attention.registry import create_paged_router


def _make_qsa_backend(*, max_bs: int, is_draft: bool, device: str) -> CacheGroupRouter:
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        layer_types=(),
        sliding_window_tokens=None,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=64,
        kernel_page_size=64,
        context_len=1024,
        max_bs=max_bs,
        pd_disaggregation_enabled=False,
        speculative_num_steps=0,
        speculative_num_draft_tokens=4,
        is_draft=is_draft,
        draft_block_decode=False,
        components=(spec,),
    )
    return create_paged_router(config, AttentionArch.MHA, backend_name="qsa")


class _TestWrapper:
    """Hybrid wrapper surface over a QSA full-attention child."""

    commit_speculative_state_after_verify = (
        AttentionBackend.commit_speculative_state_after_verify
    )
    register_speculative_state_backend = (
        AttentionBackend.register_speculative_state_backend
    )

    def __init__(self, full_backend) -> None:
        self.full_attn_backend = full_backend

    def child_backends(self):
        return (self.full_attn_backend,)


class _TestIndexer:
    """Stand-in exposing only the backend-owned staging contract."""

    def __init__(
        self,
        layer_id: int,
        *,
        index_head_dim: int,
        compress_ratio: int,
        recent_page_size: int,
        num_pages: int,
    ) -> None:
        self.layer_id = layer_id
        self.index_head_dim = index_head_dim
        self.compress_ratio = compress_ratio
        self.recent_page_size = recent_page_size
        self.qsa_runtime = None
        self.raw = torch.zeros(
            num_pages,
            compress_ratio,
            1,
            index_head_dim,
            dtype=torch.bfloat16,
        )
        self.position_cache = torch.zeros(num_pages, 3, dtype=torch.int64)

    def verify_commit_fields(self, pool):
        del pool
        return self.raw, self.position_cache


def _indexer(layer_id: int, *, compress_ratio: int) -> _TestIndexer:
    return _TestIndexer(
        layer_id,
        index_head_dim=8,
        compress_ratio=compress_ratio,
        recent_page_size=64,
        num_pages=4,
    )


@pytest.fixture
def commit_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    import tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa as module

    calls: list[tuple[tuple, dict]] = []

    def commit(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "qwen4_exp_qsa_commit_verify_layers", commit)
    return calls


def _stage_round(
    backend: CacheGroupRouter,
    indexers: list[_TestIndexer],
    *,
    bs: int,
    width: int,
    pool,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows = bs * width
    head_dim = indexers[0].index_head_dim
    position_values = torch.arange(rows * 3, dtype=torch.int64).view(rows, 3)
    logical_positions = torch.arange(rows, dtype=torch.int64)
    recent_locs = torch.arange(1, rows + 1, dtype=torch.int32)
    views = []
    for slot, indexer in enumerate(indexers):
        token_k = torch.full((rows, 1, head_dim), float(slot + 1), dtype=torch.bfloat16)
        destinations = backend.runtime.verify_staging_buffers(
            indexer,
            token_k,
            position_values,
            logical_positions,
            recent_locs,
            bs,
            pool,
        )
        destinations[0].copy_(token_k.view(bs, width, 1, head_dim))
        destinations[1].copy_(position_values.view(bs, width, 3))
        destinations[2].copy_(logical_positions.view(bs, width))
        destinations[3].copy_(recent_locs.view(bs, width))
        views.append(destinations)
    return views


def test_qsa_backend_uses_the_ordinary_router_and_registered_leaf() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False, device="cpu")

    assert type(backend) is CacheGroupRouter
    assert isinstance(backend.runtime, QSARuntime)
    assert backend.runtime.dtype is torch.bfloat16
    assert isinstance(backend._leaf_factory("full_attention", 64), QSAAttnBackend)


def test_qsa_backend_commits_every_layer_in_one_launch(commit_calls) -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(1, compress_ratio=4), _indexer(3, compress_ratio=4)]
    pool = SimpleNamespace()

    assert bind_qsa_indexers(wrapper, indexers) is backend.runtime
    assert bind_qsa_indexers(wrapper, indexers) is backend.runtime
    assert [indexer.qsa_runtime for indexer in indexers] == [
        backend.runtime,
        backend.runtime,
    ]

    views = _stage_round(backend, indexers, bs=2, width=4, pool=pool)
    staging = backend.runtime._staging[4]
    assert staging.token_k.shape == (2, 2, 4, 1, 8)
    assert views[0][0].data_ptr() == staging.token_k[0].data_ptr()
    assert views[1][0].data_ptr() == staging.token_k[1].data_ptr()
    assert views[0][1].data_ptr() == views[1][1].data_ptr()

    wrapper.commit_speculative_state_after_verify(
        torch.tensor([9, 1, 3], dtype=torch.int32),
        num_extends=1,
    )

    assert len(commit_calls) == 1
    args, kwargs = commit_calls[0]
    assert [int(value) for value in args[0]] == [
        indexer.raw.data_ptr() for indexer in indexers
    ]
    assert [int(value) for value in args[1]] == [
        indexer.position_cache.data_ptr() for indexer in indexers
    ]
    assert args[2] is staging.token_k
    torch.testing.assert_close(
        args[6], torch.tensor([1, 3], dtype=torch.int32), atol=0, rtol=0
    )
    assert args[3].shape == (8,)
    assert args[5].shape == (8, 3)
    assert kwargs["verify_width"] == 4


def test_qsa_backend_one_buffer_serves_every_batch_size(commit_calls) -> None:
    backend = _make_qsa_backend(max_bs=16, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0, compress_ratio=4), _indexer(1, compress_ratio=4)]
    pool = SimpleNamespace()
    bind_qsa_indexers(wrapper, indexers)

    _stage_round(backend, indexers, bs=8, width=4, pool=pool)
    staging = backend.runtime._staging[4]
    backend.commit_speculative_state_after_verify(
        torch.tensor([2] * 8, dtype=torch.int32), num_extends=0
    )
    tables = backend.runtime._commit_tables

    _stage_round(backend, indexers, bs=4, width=4, pool=pool)
    backend.commit_speculative_state_after_verify(
        torch.tensor([3] * 4, dtype=torch.int32), num_extends=0
    )

    assert list(backend.runtime._staging) == [4]
    assert backend.runtime._staging[4] is staging
    assert backend.runtime._commit_tables is tables
    assert len(commit_calls) == 2
    assert commit_calls[1][0][3].shape == (16,)
    assert commit_calls[0][0][2] is commit_calls[1][0][2]


def test_qsa_backend_sizes_staging_from_config_bound() -> None:
    backend = _make_qsa_backend(max_bs=64, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0, compress_ratio=4), _indexer(1, compress_ratio=4)]
    bind_qsa_indexers(wrapper, indexers)

    _stage_round(backend, indexers, bs=3, width=4, pool=SimpleNamespace())

    staging = backend.runtime._staging[4]
    assert staging.capacity == 64
    assert staging.token_k.shape == (2, 64, 4, 1, 8)
    assert staging.position_values.shape == (64, 4, 3)
    assert staging.logical_positions.shape == (64, 4)
    assert staging.recent_locs.shape == (64, 4)


def test_qsa_backend_never_resizes_staging() -> None:
    backend = _make_qsa_backend(max_bs=1, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0, compress_ratio=4)]
    bind_qsa_indexers(wrapper, indexers)
    pool = SimpleNamespace()

    _stage_round(backend, indexers, bs=4, width=4, pool=pool)
    staged_before = backend.runtime._staging[4].token_k

    with pytest.raises(RuntimeError, match="must never be resized"):
        _stage_round(backend, indexers, bs=5, width=4, pool=pool)

    assert backend.runtime._staging[4].token_k is staged_before


def test_qsa_backend_rejects_mismatched_indexer_geometry() -> None:
    wrapper = _TestWrapper(_make_qsa_backend(max_bs=2, is_draft=False, device="cpu"))

    with pytest.raises(RuntimeError, match="disagrees with layer 0"):
        bind_qsa_indexers(
            wrapper, [_indexer(0, compress_ratio=4), _indexer(1, compress_ratio=8)]
        )


def test_qsa_backend_rejects_duplicate_layer_ids() -> None:
    wrapper = _TestWrapper(_make_qsa_backend(max_bs=2, is_draft=False, device="cpu"))

    with pytest.raises(RuntimeError, match="distinct layer ids"):
        bind_qsa_indexers(
            wrapper, [_indexer(2, compress_ratio=4), _indexer(2, compress_ratio=4)]
        )


def test_qsa_backend_rejects_staging_from_unbound_layer() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    bind_qsa_indexers(wrapper, [_indexer(0, compress_ratio=4)])

    with pytest.raises(RuntimeError, match="without being bound"):
        _stage_round(
            backend,
            [_indexer(7, compress_ratio=4)],
            bs=1,
            width=4,
            pool=SimpleNamespace(),
        )


def test_qsa_backend_rejects_two_pools_in_one_forward() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0, compress_ratio=4), _indexer(1, compress_ratio=4)]
    bind_qsa_indexers(wrapper, indexers)

    _stage_round(backend, indexers[:1], bs=1, width=4, pool=SimpleNamespace())
    with pytest.raises(RuntimeError, match="two KV pools"):
        _stage_round(
            backend,
            indexers[1:],
            bs=1,
            width=4,
            pool=SimpleNamespace(),
        )


def test_qsa_indexer_staging_requires_binding() -> None:
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer

    indexer = object.__new__(QSAIndexer)
    indexer.qsa_runtime = None

    with pytest.raises(RuntimeError, match="before indexers were bound"):
        indexer._verify_staging_buffers(
            torch.zeros(4, 1, 8, dtype=torch.bfloat16),
            torch.zeros(4, 3, dtype=torch.int64),
            torch.arange(4, dtype=torch.int64),
            torch.arange(1, 5, dtype=torch.int32),
            1,
            object(),
        )


def test_qsa_backend_commit_without_staging_is_silent(commit_calls) -> None:
    wrapper = _TestWrapper(_make_qsa_backend(max_bs=2, is_draft=False, device="cpu"))
    bind_qsa_indexers(wrapper, [_indexer(0, compress_ratio=4)])

    wrapper.commit_speculative_state_after_verify(
        torch.tensor([2], dtype=torch.int32), num_extends=0
    )

    assert commit_calls == []


def test_qsa_backend_rejects_invalid_extend_prefix() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False, device="cpu")
    bind_qsa_indexers(_TestWrapper(backend), [_indexer(0, compress_ratio=4)])

    with pytest.raises(ValueError, match="invalid extend prefix"):
        backend.commit_speculative_state_after_verify(
            torch.tensor([1, 2], dtype=torch.int32), num_extends=3
        )


def test_qsa_backend_draft_never_commits_target_acceptance(commit_calls) -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=True, device="cpu")
    wrapper = _TestWrapper(backend)
    indexer = _indexer(0, compress_ratio=4)

    assert bind_qsa_indexers(wrapper, [indexer]) is None
    wrapper.commit_speculative_state_after_verify(
        torch.tensor([4], dtype=torch.int32), num_extends=0
    )

    assert not hasattr(wrapper, "_speculative_state_backends")
    assert indexer.qsa_runtime is None
    assert commit_calls == []


def test_qsa_backend_rejects_rebinding_to_another_model() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False, device="cpu")
    wrapper = _TestWrapper(backend)
    bind_qsa_indexers(wrapper, [_indexer(0, compress_ratio=4)])

    with pytest.raises(RuntimeError, match="cannot be rebound"):
        bind_qsa_indexers(wrapper, [_indexer(1, compress_ratio=4)])


def _qsa_pool() -> SimpleNamespace:
    groups = {
        FULL_ATTENTION: 256,
        QWEN4_EXP_QSA_CACHE_GROUP: 256,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP: 64,
    }
    fields = [
        SimpleNamespace(
            group_id=QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
            field_id=qsa_raw_key_field(layer_id),
            shape=(4, 1, 8),
        )
        # Layer 4 is the adjacent draft view; it must not get target staging.
        for layer_id in (1, 3, 4)
    ]
    return SimpleNamespace(
        arena=SimpleNamespace(
            plan=SimpleNamespace(fields=fields),
            cache_group_specs=tuple(
                SimpleNamespace(
                    group_id=gid,
                    block_granularity=granularity,
                    family="history",
                    retention="full_history",
                )
                for gid, granularity in groups.items()
            ),
        ),
        paged_group_ids=tuple(groups),
        field_layer_range=range(4),
    )


def test_qsa_registry_preallocates_one_runtime_before_model_binding(
    commit_calls,
) -> None:
    router = _make_qsa_backend(max_bs=8, is_draft=False, device="cpu")
    runtime = router.runtime
    pool = _qsa_pool()
    router.set_cache_pool(pool)

    assert router.runtime is runtime
    assert len(router.leaves) == 3
    assert all(isinstance(leaf, QSAAttnBackend) for leaf in router.leaves.values())
    assert all(not hasattr(leaf, "runtime") for leaf in router.leaves.values())
    expected_bytes = 2 * 8 * 4 * 8 * 2 + 8 * 4 * 36
    assert router.preallocate_verify_workspace(8, 4) == expected_bytes
    staging = runtime._staging[4]
    assert staging.token_k.shape == (2, 8, 4, 1, 8)

    indexers = [_indexer(1, compress_ratio=4), _indexer(3, compress_ratio=4)]
    wrapper = _TestWrapper(router)
    for _ in range(2):
        assert bind_qsa_indexers(wrapper, indexers) is runtime
    _stage_round(router, indexers, bs=2, width=4, pool=pool)
    wrapper.commit_speculative_state_after_verify(
        torch.tensor([1, 3], dtype=torch.int32), num_extends=0
    )

    assert runtime._staging[4] is staging
    assert len(commit_calls) == 1
    assert not hasattr(wrapper, "_speculative_state_backends")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_graph", [False, True])
def test_qsa_runtime_refreshes_layout_and_commits_live_verify_rows(
    use_graph: bool,
) -> None:
    router = _make_qsa_backend(max_bs=4, is_draft=False, device="cuda")
    pool = _qsa_pool()
    router.set_cache_pool(pool)
    router.preallocate_verify_workspace(4, 4)
    router.init_cuda_graph_state(4)
    runtime = router.runtime
    indexers = [_indexer(1, compress_ratio=4), _indexer(3, compress_ratio=4)]
    for indexer in indexers:
        indexer.raw = indexer.raw.cuda()
        indexer.position_cache = indexer.position_cache.cuda()
    bind_qsa_indexers(router, indexers)
    staging = runtime._staging[4]
    source = torch.arange(2 * 8 * 8, dtype=torch.float32, device="cuda").reshape(
        2, 8, 1, 8
    )
    source = source.to(torch.bfloat16)
    seq_lens = torch.tensor([4, 12], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    tables = {
        gid: torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
        for gid in router.group_ids
    }
    ctx = SimpleNamespace(bs=2, forward_mode=ForwardMode.DECODE)

    def refresh(actual_bs: int) -> None:
        router.refresh_decode_metadata(
            2,
            actual_bs,
            slots,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=use_graph,
        )

    def stage() -> None:
        layout = runtime.qsa_forward_layout(
            ctx,
            8,
            compressed_token_page_size=256,
            recent_page_size=64,
            compress_ratio=4,
            reset_draft_tags=None,
        )
        positions = layout.logical_positions[:, None].expand(-1, 3)
        for layer, indexer in enumerate(indexers):
            destinations = runtime.verify_staging_buffers(
                indexer,
                source[layer],
                positions,
                layout.logical_positions,
                layout.recent_locs,
                2,
                pool,
            )
            destinations[0].copy_(source[layer].view(2, 4, 1, 8))
            destinations[1].copy_(positions.view(2, 4, 3))
            destinations[2].copy_(layout.logical_positions.view(2, 4))
            destinations[3].copy_(layout.recent_locs.view(2, 4))

    for _ in range(2):
        refresh(2)
        stage()
    torch.cuda.synchronize()
    graph = None
    if use_graph:
        refresh(2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage()

    expected_keys = [
        torch.zeros_like(indexer.raw, device="cpu") for indexer in indexers
    ]
    expected_positions = [
        torch.zeros_like(indexer.position_cache, device="cpu") for indexer in indexers
    ]
    commit_tables = None
    for lengths, accepted, actual_bs in [([4, 12], [1, 3], 2), ([8, 16], [4, 0], 1)]:
        seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        source.add_(1)
        refresh(actual_bs)
        if graph is None:
            stage()
        else:
            graph.replay()
        router.commit_speculative_state_after_verify(
            torch.tensor(accepted, dtype=torch.int32, device="cuda"), num_extends=0
        )
        assert runtime._staging[4] is staging
        if commit_tables is not None:
            assert runtime._commit_tables is commit_tables
        commit_tables = runtime._commit_tables
        source_cpu = source.cpu()
        for layer, indexer in enumerate(indexers):
            for request, count in enumerate(accepted):
                for step in range(count):
                    position = lengths[request] - 4 + step
                    expected_keys[layer][request + 1, position % 4] = source_cpu[
                        layer, request * 4 + step
                    ]
                    if position % 4 == 0:
                        expected_positions[layer][request + 1].fill_(position)
            torch.testing.assert_close(
                indexer.raw.cpu(), expected_keys[layer], atol=0, rtol=0
            )
            torch.testing.assert_close(
                indexer.position_cache.cpu(), expected_positions[layer], atol=0, rtol=0
            )
