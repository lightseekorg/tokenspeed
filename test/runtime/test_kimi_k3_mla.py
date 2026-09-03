"""Kimi-K3 MLA attention on the cache-group full-attention table.

Coverage:

- pool MLA adapter surface (`get_key_buffer` / `set_mla_kv_buffer` /
  `get_mla_kv_buffer`) delegating to the generic ``latent_kv`` component view;
- write-location oracles proving latent writes land at
  ``page_id * P + offset`` (decode, eager/chunked prefill, page boundaries);
- decode and chunked-prefill numerical parity between the router-expanded
  (GroupTableStacks) kernel-page table and a hand-built one, plus a naive
  fp32 reference.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenspeed_kernel.platform import current_platform

# CI Registration (parsed via AST, runtime no-op)
# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``
# absolute imports) both need to be importable when run_ci_suite executes this
# file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.conftest import MLA_KV_LORA_RANK as _KV_LORA_RANK
from test.runtime.conftest import MLA_LATENT_DIM as _LATENT_DIM
from test.runtime.conftest import MLA_QK_ROPE_DIM as _QK_ROPE_DIM
from test.runtime.conftest import full_attention_metadata_for as _metadata_for
from test.runtime.conftest import kda_layer_id as _kda_layer_id
from test.runtime.conftest import make_kimi_pool as _make_pool
from test.runtime.conftest import mla_layer_id as _mla_layer_id
from test.runtime.conftest import (
    requires_cuda,
)

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-1gpu")

_HEADS = 16
_KERNEL_PAGE = 64
_FULL = "full_attention"


def _fake_layer(layer_id: int, scaling: float = _LATENT_DIM**-0.5):
    return SimpleNamespace(
        layer_id=layer_id,
        tp_q_head_num=_HEADS,
        head_dim=_LATENT_DIM,
        v_head_dim=_KV_LORA_RANK,
        scaling=scaling,
        k_scale_float=None,
        logit_cap=0.0,
    )


def _token_locs(pages: list[int], positions: torch.Tensor, page_size: int):
    page_tensor = torch.tensor(pages, device=positions.device, dtype=torch.int64)
    return page_tensor[positions // page_size] * page_size + positions % page_size


def _kernel_page_table(
    logical_rows: list[list[int]],
    prefix_granularity: int,
    device,
    *,
    max_num_pages: int,
):
    """Hand-built kernel-page table padded to the leaf's ``max_num_pages``:
    each logical page expands to its ratio consecutive kernel pages; -1 holes
    collapse to the null page 0 (the router's padding contract)."""
    ratio = prefix_granularity // _KERNEL_PAGE
    rows = []
    for pages in logical_rows:
        row: list[int] = []
        for page in pages:
            if page > 0:
                row.extend(page * ratio + k for k in range(ratio))
            else:
                row.extend(0 for _ in range(ratio))
        row.extend(0 for _ in range(max_num_pages - len(row)))
        rows.append(row)
    return torch.tensor(rows, device=device, dtype=torch.int32)


def _expand_via_stacks(backend, pool, logical_rows, device="cuda"):
    """Router-side expansion stage: raw scheduler blocks -> the leaf's
    ``[bs, max_num_pages]`` kernel-page table (GroupTableStacks.fill)."""
    from tokenspeed.runtime.layers.attention.backends.group_tables import (
        GroupTableSpec,
        GroupTableStacks,
    )

    bs = len(logical_rows)
    stacks = GroupTableStacks(
        [
            GroupTableSpec(
                group_id=_FULL,
                block_granularity=pool.arena.prefix_granularity,
                kernel_page_size=backend.kernel_page_size,
                max_num_pages=backend.max_num_pages,
            )
        ],
        max_bs=max(bs, 4),
        max_tokens_per_req=backend.spec_num_tokens,
        device=device,
    )
    raw = torch.tensor(logical_rows, dtype=torch.int32, device=device)
    stacks.fill(bs, bs, {_FULL: raw})
    return stacks.table(_FULL, bs)


# ---------------------------------------------------------------------------
# Bridge: per-group table views
# ---------------------------------------------------------------------------


def test_bridge_exposes_full_attention_table() -> None:
    pool = _make_pool("cpu", usable_pages=2)
    table = np.array([[1, 2]], dtype=np.int32)
    metadata, forward_op = _metadata_for(pool, table, "cpu")

    tables = metadata.tables(active_forward_op=forward_op)
    assert tuple(tables) == metadata.group_ids
    assert torch.equal(tables["full_attention"], torch.tensor(table, dtype=torch.int32))

    with pytest.raises(RuntimeError, match="stale"):
        metadata.tables(active_forward_op=object())


# ---------------------------------------------------------------------------
# Pool MLA adapter surface
# ---------------------------------------------------------------------------


def test_pool_mla_views_are_no_copy_and_layer_gated() -> None:
    pool = _make_pool("cpu", usable_pages=2)
    layer_id = _mla_layer_id(pool)
    page_size = pool.arena.prefix_granularity

    key = pool.get_key_buffer(layer_id)

    assert key.shape == ((2 * 12 + 1) * page_size, 1, _LATENT_DIM)
    assert key.dtype == torch.float8_e4m3fn
    assert (
        key.untyped_storage().data_ptr()
        == pool.arena.buffer.untyped_storage().data_ptr()
    )

    value = pool.get_value_buffer(layer_id)
    assert value.shape[-1] == _KV_LORA_RANK
    assert value.data_ptr() == key.data_ptr()
    key2, value2 = pool.get_kv_buffer(layer_id)
    assert key2.data_ptr() == key.data_ptr()
    assert value2.shape == value.shape

    # The two surfaces are layer-gated in both directions: latent KV is read
    # through the kv buffers, KDA state only through get_component.
    with pytest.raises(ValueError, match="state layer"):
        pool.get_key_buffer(_kda_layer_id(pool))
    with pytest.raises(ValueError, match="no KDA state"):
        pool.get_component(layer_id, "latent_kv")


@requires_cuda
def test_pool_write_location_oracle_page_id_times_p_plus_offset() -> None:
    """Latent writes land at ``[page_id, token_offset]``."""
    torch.manual_seed(0)
    pool = _make_pool("cuda", usable_pages=3)
    layer_id = _mla_layer_id(pool)
    layer = _fake_layer(layer_id)
    page_size = pool.arena.prefix_granularity
    # Page contiguity is a plan invariant (exact_page_stride), so the flat slot
    # view folds into pages -- which is the oracle under test.
    paged_latent = (
        pool.get_key_buffer(layer_id)
        .view(torch.uint8)
        .view(-1, page_size, 1, _LATENT_DIM)
    )

    # Page-boundary coverage: last slot of page 1, first slot of page 2,
    # interior of page 3.
    locs = torch.tensor(
        [1 * page_size + (page_size - 1), 2 * page_size, 3 * page_size + 7],
        device="cuda",
        dtype=torch.int64,
    )
    latent = torch.randn(3, 1, _LATENT_DIM, device="cuda", dtype=torch.bfloat16)
    pool.set_mla_kv_buffer(
        layer, locs, latent[..., :_KV_LORA_RANK], latent[..., _KV_LORA_RANK:]
    )
    torch.cuda.synchronize()

    expected = latent.to(torch.float8_e4m3fn).view(torch.uint8)
    for row, (page, off) in enumerate([(1, page_size - 1), (2, 0), (3, 7)]):
        got_bytes = paged_latent[page, off, 0]
        assert torch.equal(got_bytes, expected[row, 0]), (page, off)

    # Null page 0 must stay untouched.
    assert torch.count_nonzero(paged_latent[0]).item() == 0

    # Round-trip through the read adapter.
    nope, rope = pool.get_mla_kv_buffer(layer, locs, torch.bfloat16)
    ref = latent.to(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(nope, ref[..., :_KV_LORA_RANK].contiguous())
    assert torch.equal(rope, ref[..., _KV_LORA_RANK:].contiguous())


# ---------------------------------------------------------------------------
# Backend: kernel-page table expansion parity (GPU)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuda_env():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed.runtime.utils.env import global_server_args_dict

    saved = {
        key: global_server_args_dict.get(key)
        for key in ("kv_cache_dtype", "chunked_prefill_size", "mla_chunk_multiplier")
    }
    global_server_args_dict["kv_cache_dtype"] = "fp8_e4m3"
    global_server_args_dict["chunked_prefill_size"] = 256
    global_server_args_dict["mla_chunk_multiplier"] = 1
    yield
    global_server_args_dict.update(saved)


@pytest.fixture(scope="module")
def gpu_pool(cuda_env):
    return _make_pool("cuda", usable_pages=6)


@pytest.fixture(scope="module")
def backend_factory(cuda_env, gpu_pool):
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    def make():
        # The CuteDSL leaf is SM100 (Blackwell) only; everywhere else the
        # generic MLA leaf serves the same interface, so the parity claims
        # under test are identical.
        is_blackwell = (
            not current_platform().is_amd
            and torch.cuda.get_device_capability()[0] >= 10
        )
        if is_blackwell:
            from tokenspeed.runtime.layers.attention.backends.tokenspeed_mla import (
                CuteDSLMLABackend,
            )

            backend_cls = CuteDSLMLABackend
            backend_name = "tokenspeed_mla"
        else:
            from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend

            backend_cls = MLAAttnBackend
            backend_name = "mla"
        spec = MLAConfig(
            backend_name=backend_name,
            num_attention_heads=_HEADS,
            num_kv_heads=1,
            head_dim=_LATENT_DIM,
            attn_tp_size=1,
            kv_lora_rank=_KV_LORA_RANK,
            qk_nope_head_dim=128,
            qk_rope_head_dim=_QK_ROPE_DIM,
            v_head_dim=128,
            scaling=192**-0.5,
            kv_cache_dim=_LATENT_DIM,
        )
        config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.float8_e4m3fn,
            prefix_granularity=_KERNEL_PAGE,
            kernel_page_size=_KERNEL_PAGE,
            context_len=4 * gpu_pool.arena.prefix_granularity,
            max_bs=8,
            kv_cache_quant_method="",
            components=(spec,),
        )
        backend = backend_cls(config, spec, kernel_page_size=_KERNEL_PAGE)
        backend.set_cache_pool(gpu_pool)
        # Production order: set_cache_pool binds the pool, then the router
        # allocates the persistent decode buffers unconditionally.
        backend.init_cuda_graph_state(max_bs=8)
        return backend

    return make


def _write_history(pool, layer, logical_rows, lengths, seed=0):
    torch.manual_seed(seed)
    page_size = pool.arena.prefix_granularity
    for pages, length in zip(logical_rows, lengths):
        positions = torch.arange(length, device="cuda", dtype=torch.int64)
        locs = _token_locs(pages, positions, page_size)
        latent = torch.randn(
            length, 1, _LATENT_DIM, device="cuda", dtype=torch.bfloat16
        )
        pool.set_mla_kv_buffer(
            layer, locs, latent[..., :_KV_LORA_RANK], latent[..., _KV_LORA_RANK:]
        )
    torch.cuda.synchronize()


def _refresh_decode(backend, page_table, seq_lens_cpu):
    bs = page_table.shape[0]
    seq_lens = torch.tensor(seq_lens_cpu, device="cuda", dtype=torch.int32)
    backend.refresh_decode_metadata(bs, bs, seq_lens, page_table)


@requires_cuda
def test_decode_grouped_matches_single_table_and_reference(
    backend_factory, gpu_pool
) -> None:
    pool = gpu_pool
    page_size = pool.arena.prefix_granularity
    layer = _fake_layer(_mla_layer_id(pool), scaling=_LATENT_DIM**-0.5)
    # req0 crosses a logical page boundary; req1 ends exactly at one.
    logical_rows = [[2, 4], [1, -1]]
    seq_lens_cpu = [page_size + 42, page_size]
    _write_history(pool, layer, logical_rows, seq_lens_cpu, seed=1)

    grouped_backend = backend_factory()
    _refresh_decode(
        grouped_backend,
        _expand_via_stacks(grouped_backend, pool, logical_rows),
        seq_lens_cpu,
    )
    grouped_md = grouped_backend.forward_decode_metadata

    bs = len(logical_rows)
    torch.manual_seed(2)
    q = torch.randn(bs, _HEADS, _LATENT_DIM, device="cuda", dtype=torch.bfloat16)
    out_grouped = grouped_backend.forward_decode(
        q, None, None, layer, None, pool, bs, save_kv_cache=False
    )

    # Single-table arm: byte-equivalent hand-built kernel-page table.
    single_table_backend = backend_factory()
    page_table = _kernel_page_table(
        logical_rows,
        page_size,
        "cuda",
        max_num_pages=single_table_backend.max_num_pages,
    )
    _refresh_decode(single_table_backend, page_table, seq_lens_cpu)
    single_table_md = single_table_backend.forward_decode_metadata
    width = page_table.shape[1]
    assert torch.equal(
        grouped_md.page_table[:, :width],
        single_table_md.page_table[:, :width],
    )
    out_single_table = single_table_backend.forward_decode(
        q, None, None, layer, None, pool, bs, save_kv_cache=False
    )
    torch.cuda.synchronize()
    assert torch.equal(out_grouped, out_single_table)

    # Naive fp32 reference over the fp8 history.
    key_buffer = pool.get_key_buffer(layer.layer_id).float()
    for row, (pages, seq_len) in enumerate(zip(logical_rows, seq_lens_cpu)):
        positions = torch.arange(seq_len, device="cuda", dtype=torch.int64)
        history = key_buffer[_token_locs(pages, positions, page_size), 0]
        q_fp8 = q[row].to(torch.float8_e4m3fn).float()
        weights = torch.softmax((q_fp8 @ history.T) * layer.scaling, dim=-1)
        reference = weights @ history[:, :_KV_LORA_RANK]
        got = out_grouped[row].view(_HEADS, _KV_LORA_RANK).float()
        max_err = (reference - got).abs().max().item()
        assert max_err < 0.05 * reference.abs().max().item(), max_err


def _init_prefill(backend, page_table, prefix, extend):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    bs = page_table.shape[0]
    seq_lens = torch.tensor(
        [p + e for p, e in zip(prefix, extend)], device="cuda", dtype=torch.int32
    )
    backend.init_forward_metadata(
        bs,
        bs,
        seq_lens,
        page_table,
        ForwardMode.EXTEND,
        extend_prefix_lens=torch.tensor(prefix, device="cuda", dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor(prefix, dtype=torch.int32),
        extend_seq_lens=torch.tensor(extend, device="cuda", dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor(extend, dtype=torch.int32),
        extend_with_prefix=any(p > 0 for p in prefix),
    )
    return seq_lens


@requires_cuda
def test_chunked_prefill_grouped_matches_single_table_and_reference(
    backend_factory, gpu_pool
) -> None:
    from tokenspeed_kernel.ops.attention import attn_merge_state

    pool = gpu_pool
    page_size = pool.arena.prefix_granularity
    layer = _fake_layer(_mla_layer_id(pool), scaling=192**-0.5)
    # req0's prefix crosses a page boundary mid-page; req1's prefix ends
    # exactly on one, and its extend tokens cross into a fresh page.
    logical_rows = [[2, 4, 6, 8, 10], [1, 3, -1, -1, -1]]
    prefix = [page_size + 36, page_size]
    extend = [300, 40]
    _write_history(pool, layer, logical_rows, prefix, seed=4)

    grouped_backend = backend_factory()
    single_table_backend = backend_factory()
    _init_prefill(
        grouped_backend,
        _expand_via_stacks(grouped_backend, pool, logical_rows),
        prefix,
        extend,
    )
    _init_prefill(
        single_table_backend,
        _kernel_page_table(
            logical_rows,
            page_size,
            "cuda",
            max_num_pages=single_table_backend.max_num_pages,
        ),
        prefix,
        extend,
    )
    assert grouped_backend.chunked_prefill_metadata.extend_prefix_lens_cpu.tolist() == (
        prefix
    )
    assert (
        grouped_backend.chunked_prefill_metadata.extend_seq_lens_cpu.tolist() == extend
    )

    # Chunked-prefill metadata parity with the hand-built page_table build.
    grouped_cm = grouped_backend.chunked_prefill_metadata
    single_table_cm = single_table_backend.chunked_prefill_metadata
    assert grouped_cm.chunked_loop_num == single_table_cm.chunked_loop_num
    assert grouped_cm.chunked_loop_num >= 2, "test must exercise multiple chunks"
    for grouped_idx, single_table_idx in zip(
        grouped_cm.chunk_kv_indices_list, single_table_cm.chunk_kv_indices_list
    ):
        assert torch.equal(grouped_idx, single_table_idx)
    assert torch.equal(grouped_cm.chunked_seq_len, single_table_cm.chunked_seq_len)
    assert torch.equal(
        grouped_cm.cu_chunked_seq_len, single_table_cm.cu_chunked_seq_len
    )
    assert grouped_cm.max_chunk_len_per_loop == single_table_cm.max_chunk_len_per_loop

    bs = len(logical_rows)
    total = sum(extend)
    torch.manual_seed(5)
    q = torch.randn(total, _HEADS, 192, device="cuda", dtype=torch.bfloat16)
    new_latent = torch.randn(total, 1, _LATENT_DIM, device="cuda", dtype=torch.bfloat16)

    def head_map(latent):
        keys = torch.cat(
            [latent[..., :128], latent[..., _KV_LORA_RANK:]], dim=-1
        ).expand(-1, _HEADS, -1)
        values = latent[..., 128:256].expand(-1, _HEADS, -1)
        return keys, values

    def run(backend):
        cmeta = backend.chunked_prefill_metadata
        out = torch.empty(total, _HEADS, 128, device="cuda", dtype=torch.bfloat16)
        k_new, v_new = head_map(new_latent)
        _, lse = backend.forward_extend_chunked(
            q,
            k_new.contiguous(),
            v_new.contiguous(),
            layer.scaling,
            0.0,
            cum_seq_lens_q=cmeta.cum_extend_seq_lens,
            cum_seq_lens_kv=cmeta.cum_extend_seq_lens,
            max_q_len=cmeta.max_extend_seq_len,
            max_kv_len=cmeta.max_extend_seq_len,
            seq_lens=cmeta.extend_seq_lens,
            batch_size=bs,
            causal=True,
            out=out,
        )
        for loop_idx in range(cmeta.chunked_loop_num):
            chunk_idx = cmeta.chunk_kv_indices_list[loop_idx]
            kv_a, k_pe = pool.get_mla_kv_buffer(layer, chunk_idx, torch.bfloat16)
            k_hist, v_hist = head_map(torch.cat([kv_a, k_pe], dim=-1))
            chunk_out, chunk_lse = backend.forward_extend_chunked(
                q,
                k_hist.contiguous(),
                v_hist.contiguous(),
                layer.scaling,
                0.0,
                cum_seq_lens_q=cmeta.cum_extend_seq_lens,
                cum_seq_lens_kv=cmeta.cu_chunked_seq_len[loop_idx],
                max_q_len=cmeta.max_extend_seq_len,
                max_kv_len=cmeta.max_chunk_len_per_loop[loop_idx],
                seq_lens=cmeta.chunked_seq_len[loop_idx],
                batch_size=bs,
                causal=False,
            )
            attn_merge_state(out, lse, chunk_out, chunk_lse, inplace=True)
        return out

    out_grouped = run(grouped_backend)
    out_single_table = run(single_table_backend)
    torch.cuda.synchronize()
    assert torch.equal(out_grouped, out_single_table)

    # Naive fp32 reference (history read back through the fp8 cache).
    key_buffer = pool.get_key_buffer(layer.layer_id)
    offset = 0
    for row, (pages, prefix_len, extend_len) in enumerate(
        zip(logical_rows, prefix, extend)
    ):
        positions = torch.arange(prefix_len, device="cuda", dtype=torch.int64)
        history_latent = key_buffer[_token_locs(pages, positions, page_size)].to(
            torch.bfloat16
        )
        k_hist, v_hist = head_map(history_latent)
        k_new, v_new = head_map(new_latent[offset : offset + extend_len])
        k_all = torch.cat([k_hist, k_new], dim=0).float()
        v_all = torch.cat([v_hist, v_new], dim=0).float()
        q_req = q[offset : offset + extend_len].float()
        scores = torch.einsum("thd,shd->hts", q_req, k_all) * layer.scaling
        causal_mask = torch.ones(
            extend_len, prefix_len + extend_len, device="cuda", dtype=torch.bool
        )
        for t in range(extend_len):
            causal_mask[t, prefix_len + t + 1 :] = False
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        reference = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), v_all)
        got = out_grouped[offset : offset + extend_len].float()
        max_err = (reference - got).abs().max().item()
        assert max_err < 0.05 * reference.abs().max().item(), (row, max_err)
        offset += extend_len
