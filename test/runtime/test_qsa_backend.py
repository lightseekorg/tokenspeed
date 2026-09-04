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

from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.specific.qsa import (
    QSAAttnBackend,
    bind_qsa_indexers,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig


def _make_qsa_backend(*, max_bs: int, is_draft: bool) -> QSAAttnBackend:
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
        device="cpu",
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
        speculative_num_draft_tokens=1,
        is_draft=is_draft,
        draft_block_decode=False,
        components=(spec,),
    )
    return QSAAttnBackend(config, spec)


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
        self.qsa_coordinator = None
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


def _indexer(layer_id: int, *, compress_ratio: int = 4) -> _TestIndexer:
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
    backend: QSAAttnBackend,
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
        destinations = backend.verify_staging_buffers(
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


def test_qsa_backend_is_the_registered_group_router() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False)

    assert isinstance(backend, CacheGroupRouter)
    assert backend.data_type is torch.bfloat16


def test_qsa_backend_commits_every_layer_in_one_launch(commit_calls) -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False)
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(1), _indexer(3)]
    pool = SimpleNamespace()

    assert bind_qsa_indexers(wrapper, indexers) is backend
    assert bind_qsa_indexers(wrapper, indexers) is backend
    assert [indexer.qsa_coordinator for indexer in indexers] == [backend, backend]

    views = _stage_round(backend, indexers, bs=2, width=4, pool=pool)
    staging = backend._staging[4]
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
    backend = _make_qsa_backend(max_bs=16, is_draft=False)
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0), _indexer(1)]
    pool = SimpleNamespace()
    bind_qsa_indexers(wrapper, indexers)

    _stage_round(backend, indexers, bs=8, width=4, pool=pool)
    staging = backend._staging[4]
    backend.commit_after_mtp_verify(
        torch.tensor([2] * 8, dtype=torch.int32), num_extends=0
    )
    tables = backend._commit_tables

    _stage_round(backend, indexers, bs=4, width=4, pool=pool)
    backend.commit_after_mtp_verify(
        torch.tensor([3] * 4, dtype=torch.int32), num_extends=0
    )

    assert list(backend._staging) == [4]
    assert backend._staging[4] is staging
    assert backend._commit_tables is tables
    assert len(commit_calls) == 2
    assert commit_calls[1][0][3].shape == (16,)
    assert commit_calls[0][0][2] is commit_calls[1][0][2]


def test_qsa_backend_sizes_staging_from_config_bound() -> None:
    backend = _make_qsa_backend(max_bs=64, is_draft=False)
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0), _indexer(1)]
    bind_qsa_indexers(wrapper, indexers)

    _stage_round(backend, indexers, bs=3, width=4, pool=SimpleNamespace())

    staging = backend._staging[4]
    assert staging.capacity == 64
    assert staging.token_k.shape == (2, 64, 4, 1, 8)
    assert staging.position_values.shape == (64, 4, 3)
    assert staging.logical_positions.shape == (64, 4)
    assert staging.recent_locs.shape == (64, 4)


def test_qsa_backend_never_resizes_staging() -> None:
    backend = _make_qsa_backend(max_bs=1, is_draft=False)
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0)]
    bind_qsa_indexers(wrapper, indexers)
    pool = SimpleNamespace()

    _stage_round(backend, indexers, bs=4, width=4, pool=pool)
    staged_before = backend._staging[4].token_k

    with pytest.raises(RuntimeError, match="must never be resized"):
        _stage_round(backend, indexers, bs=5, width=4, pool=pool)

    assert backend._staging[4].token_k is staged_before


def test_qsa_backend_rejects_mismatched_indexer_geometry() -> None:
    wrapper = _TestWrapper(_make_qsa_backend(max_bs=2, is_draft=False))

    with pytest.raises(RuntimeError, match="disagrees with layer 0"):
        bind_qsa_indexers(wrapper, [_indexer(0), _indexer(1, compress_ratio=8)])


def test_qsa_backend_rejects_duplicate_layer_ids() -> None:
    wrapper = _TestWrapper(_make_qsa_backend(max_bs=2, is_draft=False))

    with pytest.raises(RuntimeError, match="distinct layer ids"):
        bind_qsa_indexers(wrapper, [_indexer(2), _indexer(2)])


def test_qsa_backend_rejects_staging_from_unbound_layer() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False)
    wrapper = _TestWrapper(backend)
    bind_qsa_indexers(wrapper, [_indexer(0)])

    with pytest.raises(RuntimeError, match="without being bound"):
        _stage_round(
            backend,
            [_indexer(7)],
            bs=1,
            width=4,
            pool=SimpleNamespace(),
        )


def test_qsa_backend_rejects_two_pools_in_one_forward() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False)
    wrapper = _TestWrapper(backend)
    indexers = [_indexer(0), _indexer(1)]
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
    indexer.qsa_coordinator = None

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
    wrapper = _TestWrapper(_make_qsa_backend(max_bs=2, is_draft=False))
    bind_qsa_indexers(wrapper, [_indexer(0)])

    wrapper.commit_speculative_state_after_verify(
        torch.tensor([2], dtype=torch.int32), num_extends=0
    )

    assert commit_calls == []


def test_qsa_backend_rejects_invalid_extend_prefix() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False)
    bind_qsa_indexers(_TestWrapper(backend), [_indexer(0)])

    with pytest.raises(ValueError, match="invalid extend prefix"):
        backend.commit_after_mtp_verify(
            torch.tensor([1, 2], dtype=torch.int32), num_extends=3
        )


def test_qsa_backend_draft_never_commits_target_acceptance(commit_calls) -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=True)
    wrapper = _TestWrapper(backend)
    indexer = _indexer(0)

    assert bind_qsa_indexers(wrapper, [indexer]) is None
    wrapper.commit_speculative_state_after_verify(
        torch.tensor([4], dtype=torch.int32), num_extends=0
    )

    assert not hasattr(wrapper, "_speculative_state_backends")
    assert indexer.qsa_coordinator is None
    assert commit_calls == []


def test_qsa_backend_rejects_rebinding_to_another_model() -> None:
    backend = _make_qsa_backend(max_bs=2, is_draft=False)
    wrapper = _TestWrapper(backend)
    bind_qsa_indexers(wrapper, [_indexer(0)])

    with pytest.raises(RuntimeError, match="cannot be rebound"):
        bind_qsa_indexers(wrapper, [_indexer(1)])
