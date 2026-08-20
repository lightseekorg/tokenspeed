"""Kimi-K3 KDA recurrent state on the cache contract path.

Coverage:

- per-group KDA state metadata: ``state_in_blocks_by_group`` /
  ``state_out_blocks_by_group``
  mappings keyed by state group id, dual-index computed ONCE per group per
  batch, with a proof the three groups' indices are independent and selected
  per layer via ``pool.group_id_for_layer``;
- structural binding of the two KDA components (``conv_state`` /
  ``recurrent_state``) from the LCM pool's no-copy component views;
- eager prefill and decode over dual state page indices compared against a
  naive fp32 recurrence AND the FLA one-shot oracle: initial zero state,
  same-page evolution, boundary crossing, prefix resume, copy-on-write, and
  isolation between requests;
- the KDA multi-group CUDA-graph state buffer capture/replay logic (the MLA
  half lives in ``test_kimi_k3_cudagraph.py``).
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenspeed_kernel.platform import current_platform

# The chunked-prefill KDA path resolves to the flash-linear-attention ("fla")
# solution, an optional reference dependency not installed in CI. Tests that
# exercise it skip when it is absent (mirrors the kernel-level KDA tests).
requires_fla = pytest.mark.skipif(
    find_spec("fla") is None,
    reason="flash-linear-attention (fla) required for the KDA prefill kernel",
)

# CI Registration (parsed via AST, runtime no-op)
# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``
# absolute imports) both need to be importable when run_ci_suite executes this
# file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import cache_metadata_for as _metadata_for
from test.runtime.conftest import layer_for_group as _kda_layer_for_group
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from test.runtime.conftest import requires_cuda

from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import hybrid_kda, hybrid_linear_attn
from tokenspeed.runtime.layers.attention.backends.hybrid_kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    compute_state_block_indices,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)

register_cuda_ci(est_time=240, suite="runtime-1gpu")

_LOWER_BOUND = -5.0


def _backend_config(device: str, *, spec_tokens: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=128,
        is_draft=False,
        speculative_num_draft_tokens=spec_tokens,
        max_bs=8,
    )


def _backend(device: str, *, contract_pool, spec_tokens: int = 1) -> KdaAttnBackend:
    kda_backend = "auto" if current_platform().is_amd else "fla"
    backend = KdaAttnBackend(
        _backend_config(device, spec_tokens=spec_tokens),
        kda_backend=kda_backend,
    )
    backend.set_kv_pool(contract_pool)
    return backend


def _stub_contract(*, prefix_granularity: int, usable_pages: int):
    """Small-P contract with the Kimi group topology (1 history + 3 state)."""
    group_ids = ("full_attention", *_STATE_GROUPS)
    specs = tuple(
        (
            CacheGroupSpec(
                group_id=group_id,
                retention="full_history",
                rows_per_page=prefix_granularity,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
            )
            if group_id == "full_attention"
            else CacheGroupSpec(
                group_id=group_id,
                retention="full_history",
                sliding_window_tokens=None,
                family="state",
                checkpoint_granularity=prefix_granularity,
            )
        )
        for group_id in group_ids
    )
    return CacheRuntimeContract(
        prefix_granularity=prefix_granularity,
        num_lcm_blocks=usable_pages,
        token_capacity=usable_pages * prefix_granularity,
        group_specs=specs,
        group_page_counts={spec.group_id: usable_pages + 1 for spec in specs},
        group_packing={spec.group_id: 1 for spec in specs},
    )


class _StubContractPool:
    """Minimal contract pool: one KDA layer per state group, per-layer
    conv/recurrent component slabs indexed by page id (page 0 = null)."""

    def __init__(self, contract, device, conv_dim, width, num_heads, head_dim):
        # The arena publishes the contract; a view only names its arena.
        self.arena = SimpleNamespace(runtime_contract=contract)
        num_pages = contract.group_page_counts[_STATE_GROUPS[0]]
        self._groups = {i: _STATE_GROUPS[i] for i in range(3)}
        self._components = {
            layer_id: {
                "conv_state": torch.zeros(
                    num_pages, conv_dim, width - 1, device=device, dtype=torch.bfloat16
                ),
                "recurrent_state": torch.zeros(
                    num_pages,
                    num_heads,
                    head_dim,
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                ),
            }
            for layer_id in self._groups
        }

    def group_id_for_layer(self, layer_id: int) -> str:
        return self._groups[layer_id]

    def get_component(self, layer_id: int, name: str) -> torch.Tensor:
        return self._components[layer_id][name]


def _naive_kda_scan(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound, S0):
    """Naive fp32 KDA recurrence (per-channel safe-gated delta rule).

    q/k/v/g_raw: [T, H, D]; beta_raw: [T, H]; A_log: [H]; dt_bias: [H*D];
    S0: [H, D, D] (K-major). Mirrors fla's in-kernel semantics: l2norm with
    eps 1e-6, gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)),
    beta = sigmoid(beta_raw), scale = D ** -0.5.
    """
    T, H, D = q.shape
    qf = q.float()
    kf = k.float()
    qf = qf / (qf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    kf = kf / (kf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    g = lower_bound * torch.sigmoid(
        A_log.float().exp().view(H, 1) * (g_raw.float() + dt_bias.float().view(H, D))
    )
    beta = torch.sigmoid(beta_raw.float())
    scale = D**-0.5
    S = S0.float().clone()
    vf = v.float()
    outs = []
    for t in range(T):
        S = S * g[t].exp().unsqueeze(-1)
        kt = kf[t]
        v_delta = vf[t] - (kt.unsqueeze(-1) * S).sum(-2)
        S = S + (beta[t].unsqueeze(-1) * kt).unsqueeze(-1) * v_delta.unsqueeze(-2)
        outs.append(((qf[t] * scale).unsqueeze(-1) * S).sum(-2))
    return torch.stack(outs), S


# ---------------------------------------------------------------------------
# Contract binding (set_kv_pool)
# ---------------------------------------------------------------------------


def test_set_kv_pool_binds_contract_state_groups() -> None:
    pool = _make_kimi_pool("cpu")
    backend = _backend("cpu", contract_pool=pool)
    assert backend.state_paging_active
    assert backend._state_group_ids == _STATE_GROUPS
    assert backend._checkpoint_granularity == pool.arena.prefix_granularity


# ---------------------------------------------------------------------------
# Per-group dual-index metadata
# ---------------------------------------------------------------------------


def _kimi_tables(bs: int, width: int, *, base: int = 1) -> dict[str, np.ndarray]:
    """Distinct page ids per group so cross-group mixups are detectable."""
    tables = {}
    page = base
    for group_id in _STATE_GROUPS:
        rows = np.zeros((bs, width), dtype=np.int32)
        for row in range(bs):
            for col in range(width):
                rows[row, col] = page
                page += 1
        tables[group_id] = rows
    return tables


def test_dual_index_reuses_one_slot_plan_and_groups_are_independent(
    monkeypatch,
) -> None:
    pool = _make_kimi_pool("cpu", usable_pages=24)
    backend = _backend("cpu", contract_pool=pool)
    page_size = pool.arena.prefix_granularity

    plan_calls = 0
    real = hybrid_linear_attn._compute_state_block_index_plan

    def counting(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hybrid_linear_attn, "_compute_state_block_index_plan", counting)

    bs = 2
    tables = _kimi_tables(bs, width=2)
    metadata, forward_op = _metadata_for(pool.arena.runtime_contract, tables, "cpu")
    # Decode: request 0 crosses into its second page, request 1 stays inside
    # its first page.
    seq_lens = torch.tensor([page_size + 1, 5], dtype=torch.int32)
    backend.init_forward_metadata(
        bs=bs,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_lens,
        forward_mode=ForwardMode.DECODE,
        cache_metadata=metadata,
        forward_batch=forward_op,
    )

    # Conversion and slot arithmetic run once for the whole batch. Each state
    # group only gathers its independent page ids from that shared plan.
    assert plan_calls == 1

    md = backend.forward_metadata
    assert tuple(sorted(md.state_in_blocks_by_group)) == _STATE_GROUPS
    assert tuple(sorted(md.state_out_blocks_by_group)) == _STATE_GROUPS
    for group_id in _STATE_GROUPS:
        rows = tables[group_id]
        # req 0: before=P -> in slot 0, out slot 1; req 1: before=4 -> slot 0.
        assert md.state_in_blocks_by_group[group_id].tolist() == [
            int(rows[0, 0]),
            int(rows[1, 0]),
        ]
        assert md.state_out_blocks_by_group[group_id].tolist() == [
            int(rows[0, 1]),
            int(rows[1, 0]),
        ]
    # Three groups carry three disjoint page-id sets (independence).
    all_pages = [
        page
        for group_id in _STATE_GROUPS
        for page in md.state_out_blocks_by_group[group_id].tolist()
    ]
    assert len(set(all_pages)) == len(all_pages)


def test_cuda_graph_replay_refreshes_buffers_in_place() -> None:
    # Replay refreshes the SAME buffers (same data_ptr) from a fresh forward
    # op, per group, and pads dummy rows to the pad slot id.
    pool = _make_kimi_pool("cpu", usable_pages=24)
    backend = _backend("cpu", contract_pool=pool)
    backend.init_cuda_graph_state(max_num_tokens=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    captured_ptrs = {
        gid: backend.state_in_by_group[gid][1].data_ptr() for gid in _STATE_GROUPS
    }
    # Distinct per-group tables so the refresh is provably group-specific.
    tables = _kimi_tables(bs=2, width=4, base=1)
    metadata, forward_op = _metadata_for(pool.arena.runtime_contract, tables, "cpu")
    # bs 2 requests, one padding row (real_bs 1). Decode: before = seq-1.
    backend.init_forward_metadata_replay_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        num_padding=1,
        cache_metadata=metadata,
        forward_batch=forward_op,
    )
    md = backend.forward_metadata
    for gid in _STATE_GROUPS:
        buf = backend.state_in_by_group[gid][1]
        # Same storage as capture (in-place refresh, no realloc).
        assert buf.data_ptr() == captured_ptrs[gid]
        assert md.state_in_blocks_by_group[gid].data_ptr() == captured_ptrs[gid]
        # Real row 0 refreshed from this group's table; padded row 1 -> pad.
        expected_in, _ = compute_state_block_indices(
            torch.as_tensor(tables[gid][:1]),
            pool.arena.runtime_contract.prefix_granularity,
            torch.tensor([4]),  # before = seq_len 5 - 1
            torch.tensor([5]),
            validate=False,
            group_id=gid,
        )
        assert buf[0].item() == expected_in[0].item()
        assert buf[1].item() == backend.pad_slot_id


# ---------------------------------------------------------------------------
# GPU: KDA prefill/decode over dual state page indices vs naive + FLA
# ---------------------------------------------------------------------------


class _KDAHarness:
    """Drives KdaAttnBackend forwards over a contract pool and mirrors
    them with naive fp32 + FLA oracles."""

    H = 4
    D = 128
    WIDTH = 4

    def __init__(self, pool, contract, layer_ids, device="cuda", seed=0):
        torch.manual_seed(seed)
        self.pool = pool
        self.contract = contract
        self.device = device
        self.layer_ids = list(layer_ids)
        self.key_dim = self.H * self.D
        self.value_dim = self.H * self.D
        self.conv_dim = 3 * self.key_dim
        self.backend = _backend(device, contract_pool=pool)
        # Per-layer weights and full token streams (so each group is proven
        # independent, not accidentally identical).
        self.params = {}
        for layer_id in self.layer_ids:
            self.params[layer_id] = dict(
                conv_weights=torch.randn(
                    self.conv_dim, self.WIDTH, device=device, dtype=torch.bfloat16
                )
                * 0.1,
                bias=torch.randn(self.conv_dim, device=device, dtype=torch.bfloat16)
                * 0.1,
                A_log=torch.randn(self.H, device=device, dtype=torch.float32) * 0.1,
                dt_bias=torch.randn(self.H * self.D, device=device, dtype=torch.float32)
                * 0.1,
            )

    def token_stream(self, total):
        return dict(
            mixed=torch.randn(
                total, self.conv_dim, device=self.device, dtype=torch.bfloat16
            ),
            g_raw=torch.randn(
                total, self.H * self.D, device=self.device, dtype=torch.bfloat16
            )
            * 0.5,
            beta_raw=torch.randn(
                total, self.H, device=self.device, dtype=torch.bfloat16
            ),
        )

    def common_kwargs(self, layer_id):
        p = self.params[layer_id]
        return dict(
            conv_weights=p["conv_weights"],
            bias=p["bias"],
            activation="silu",
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            attention_tp_size=1,
            head_k_dim=self.D,
            head_v_dim=self.D,
            A_log=p["A_log"],
            dt_bias=p["dt_bias"],
            layer_id=layer_id,
            a=None,
            b=None,
            lower_bound=_LOWER_BOUND,
        )

    def init_metadata(self, tables, seq_lens, mode, extend_prefix_lens=None):
        bs = len(seq_lens)
        np_tables = {
            gid: np.asarray(rows, dtype=np.int32) for gid, rows in tables.items()
        }
        metadata, forward_op = _metadata_for(self.contract, np_tables, self.device)
        kwargs = dict(
            cache_metadata=metadata,
            forward_batch=forward_op,
        )
        if extend_prefix_lens is not None:
            kwargs["extend_prefix_lens"] = torch.tensor(
                extend_prefix_lens, dtype=torch.int32, device=self.device
            )
        if mode.is_extend_or_mixed():
            # The executor guarantees host extend lengths for every extend
            # batch; model that contract here.
            prefix = extend_prefix_lens or [0] * bs
            kwargs["extend_seq_lens_cpu"] = torch.tensor(
                [int(s) - int(p) for s, p in zip(seq_lens, prefix)],
                dtype=torch.int32,
            )
        self.backend.init_forward_metadata(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            forward_mode=mode,
            **kwargs,
        )

    def extend(self, layer_id, mixed, g_raw, beta_raw, bs=1):
        seq_len = mixed.shape[0]
        # The conv kernels write their output IN PLACE into mixed_qkv (a
        # fresh projection in production); clone so the oracle can reuse the
        # raw token stream afterwards.
        return self.backend.forward_extend(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=self.pool,
            bs=bs,
            forward_mode=ForwardMode.EXTEND,
            mixed_qkv=mixed.clone(),
            g_raw=g_raw,
            beta_raw=beta_raw,
            seq_len=seq_len,
            **self.common_kwargs(layer_id),
        ).flatten(0, 1)

    def decode(self, layer_id, mixed, g_raw, beta_raw, bs):
        # mixed cloned for the same reason as in extend().
        return self.backend.forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=self.pool,
            bs=bs,
            mixed_qkv=mixed.clone(),
            g_raw=g_raw,
            beta_raw=beta_raw,
            **self.common_kwargs(layer_id),
        ).flatten(0, 1)

    def oracle(self, layer_id, mixed, g_raw, beta_raw):
        """(naive_out, naive_state, fla_out, fla_state) over one contiguous
        sequence starting from the zero state."""
        from tokenspeed_kernel.ops.attention.triton.linear.kda import (
            kda_chunk_prefill,
        )

        from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
            causal_conv1d_fn,
        )

        p = self.params[layer_id]
        total = mixed.shape[0]
        conv_state = torch.zeros(
            1,
            self.conv_dim,
            self.WIDTH - 1,
            device=self.device,
            dtype=torch.bfloat16,
        )
        conv_out = causal_conv1d_fn(
            mixed.transpose(0, 1),
            p["conv_weights"],
            p["bias"],
            activation="silu",
            conv_states=conv_state,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=self.device),
            cache_indices=torch.zeros(1, dtype=torch.int32, device=self.device),
            query_start_loc=torch.tensor(
                [0, total], dtype=torch.int32, device=self.device
            ),
            seq_lens_cpu=torch.tensor([total], dtype=torch.int32),
        ).transpose(0, 1)[:total]
        q, k, v = torch.split(
            conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.view(total, self.H, self.D)
        k = k.view(total, self.H, self.D)
        v = v.view(total, self.H, self.D)
        g = g_raw.view(total, self.H, self.D)
        naive_out, naive_state = _naive_kda_scan(
            q,
            k,
            v,
            g,
            beta_raw,
            p["A_log"],
            p["dt_bias"],
            _LOWER_BOUND,
            torch.zeros(
                self.H, self.D, self.D, device=self.device, dtype=torch.float32
            ),
        )
        fla_out, fla_state = kda_chunk_prefill(
            q.view(1, total, self.H, self.D),
            k.view(1, total, self.H, self.D),
            v.view(1, total, self.H, self.D),
            g.view(1, total, self.H, self.D),
            beta_raw.view(1, total, self.H),
            p["A_log"],
            p["dt_bias"],
            initial_state=torch.zeros(
                1, self.H, self.D, self.D, device=self.device, dtype=torch.float32
            ),
            cu_seqlens=torch.tensor([0, total], dtype=torch.int32, device=self.device),
            lower_bound=_LOWER_BOUND,
            beta_is_logit=True,
        )
        return (
            naive_out.flatten(0, 1),
            naive_state,
            fla_out[0].flatten(0, 1),
            fla_state[0].float(),
        )


def _assert_close(actual, expected, what, mean_tol=2e-3, atol=1e-1, rtol=1e-2):
    diff = (actual.float() - expected.float()).abs()
    assert diff.mean().item() < mean_tol, f"{what}: mean diff {diff.mean().item()}"
    assert torch.allclose(
        actual.float(), expected.float(), atol=atol, rtol=rtol
    ), f"{what}: max diff {diff.max().item()}"


@requires_cuda
@requires_fla
def test_kda_three_groups_zero_state_same_page_and_crossing() -> None:
    """Prefill from zero state, same-page decode, boundary-crossing decode,
    per-group table independence — vs naive fp32 recurrence AND FLA."""
    P, usable = 4, 8
    contract = _stub_contract(prefix_granularity=P, usable_pages=usable)
    pool = _StubContractPool(
        contract, "cuda", conv_dim=3 * 4 * 128, width=4, num_heads=4, head_dim=128
    )
    h = _KDAHarness(pool, contract, layer_ids=[0, 1, 2])
    # Distinct pages per group: writes to the wrong group's pages would land
    # in rows this test asserts stay zero.
    tables = {
        "linear_attention_0": [[1, 4]],
        "linear_attention_1": [[2, 5]],
        "linear_attention_2": [[3, 6]],
    }
    streams = {layer_id: h.token_stream(6) for layer_id in [0, 1, 2]}

    outputs = {layer_id: [] for layer_id in [0, 1, 2]}
    # Prefill 3 tokens: in = null page 0 (zero state), out = slot 0.
    h.init_metadata(
        tables, seq_lens=[3], mode=ForwardMode.EXTEND, extend_prefix_lens=[0]
    )
    md = h.backend.forward_metadata
    for gid in _STATE_GROUPS:
        assert md.state_in_blocks_by_group[gid].tolist() == [0]
        assert md.state_out_blocks_by_group[gid].tolist() == [tables[gid][0][0]]
    for layer_id in [0, 1, 2]:
        s = streams[layer_id]
        outputs[layer_id].append(
            h.extend(layer_id, s["mixed"][:3], s["g_raw"][:3], s["beta_raw"][:3])
        )

    # Decode token 4 (same page), 5 (crossing), 6 (same new page).
    expected_pages = {
        gid: [
            (rows[0][0], rows[0][0]),
            (rows[0][0], rows[0][1]),
            (rows[0][1], rows[0][1]),
        ]
        for gid, rows in tables.items()
    }
    snapshot = {}
    for step, pos in enumerate(range(3, 6)):
        h.init_metadata(tables, seq_lens=[pos + 1], mode=ForwardMode.DECODE)
        md = h.backend.forward_metadata
        for gid in _STATE_GROUPS:
            assert md.state_in_blocks_by_group[gid].tolist() == [
                expected_pages[gid][step][0]
            ]
            assert md.state_out_blocks_by_group[gid].tolist() == [
                expected_pages[gid][step][1]
            ]
        for layer_id in [0, 1, 2]:
            s = streams[layer_id]
            outputs[layer_id].append(
                h.decode(
                    layer_id,
                    s["mixed"][pos : pos + 1],
                    s["g_raw"][pos : pos + 1],
                    s["beta_raw"][pos : pos + 1],
                    bs=1,
                )
            )
        if pos == 3:
            # Page written at the P boundary becomes a read-only snapshot.
            for layer_id, gid in zip([0, 1, 2], _STATE_GROUPS):
                page = tables[gid][0][0]
                snapshot[layer_id] = (
                    pool.get_component(layer_id, "conv_state")[page].clone(),
                    pool.get_component(layer_id, "recurrent_state")[page].clone(),
                )

    for layer_id, gid in zip([0, 1, 2], _STATE_GROUPS):
        s = streams[layer_id]
        naive_out, naive_state, fla_out, fla_state = h.oracle(
            layer_id, s["mixed"], s["g_raw"], s["beta_raw"]
        )
        flat_out = torch.cat(outputs[layer_id], dim=0)
        _assert_close(flat_out, naive_out, f"layer {layer_id} out vs naive")
        _assert_close(flat_out, fla_out, f"layer {layer_id} out vs FLA")
        ssm = pool.get_component(layer_id, "recurrent_state")
        conv = pool.get_component(layer_id, "conv_state")
        final_page = tables[gid][0][1]
        _assert_close(ssm[final_page], naive_state, f"layer {layer_id} state vs naive")
        _assert_close(ssm[final_page], fla_state, f"layer {layer_id} state vs FLA")
        # Snapshot page untouched by the boundary-crossing decode.
        snap_page = tables[gid][0][0]
        assert torch.equal(conv[snap_page], snapshot[layer_id][0])
        assert torch.equal(ssm[snap_page], snapshot[layer_id][1])
        assert ssm[snap_page].abs().max().item() > 0.0
        # Null page 0 is never written.
        assert conv[0].abs().max().item() == 0.0
        assert ssm[0].abs().max().item() == 0.0
        # Independence: pages belonging to the OTHER groups stay zero in this
        # layer's slabs — the layer consumed only its own group's indices.
        other_pages = [
            page
            for other_gid, rows in tables.items()
            if other_gid != gid
            for page in rows[0]
        ]
        for page in other_pages:
            assert ssm[page].abs().max().item() == 0.0, (layer_id, page)
            assert conv[page].abs().max().item() == 0.0, (layer_id, page)


@requires_cuda
@requires_fla
def test_kda_prefix_resume_copy_on_write_and_isolation() -> None:
    """Prefix hit at a P boundary: the shared snapshot page is read-only for
    both branches (CoW), and batched decode keeps requests isolated."""
    P, usable = 4, 12
    contract = _stub_contract(prefix_granularity=P, usable_pages=usable)
    pool = _StubContractPool(
        contract, "cuda", conv_dim=3 * 4 * 128, width=4, num_heads=4, head_dim=128
    )
    h = _KDAHarness(pool, contract, layer_ids=[0, 1, 2], seed=1)
    layer_id = 0
    gid = "linear_attention_0"

    def tables_for(rows0):
        # Unused groups still need valid rows of the same width (init
        # computes indices for every group once per batch).
        return {
            "linear_attention_0": rows0,
            "linear_attention_1": [[p + 4 for p in row] for row in rows0],
            "linear_attention_2": [[p + 8 for p in row] for row in rows0],
        }

    # Shared prefix: 4 tokens (exactly one page). Branch A continues with 2
    # decodes; branch B resumes from the snapshot with a 3-token extend and
    # one decode. Total streams: A = prefix + a4 a5; B = prefix + b4 b5 b6 b7.
    prefix = h.token_stream(4)
    a_new = h.token_stream(2)
    b_new = h.token_stream(4)

    def cat(*streams):
        return {
            key: torch.cat([s[key] for s in streams], dim=0)
            for key in ("mixed", "g_raw", "beta_raw")
        }

    a_full = cat(prefix, a_new)
    b_full = cat(prefix, b_new)

    # 1) A prefills the shared 4-token prefix -> snapshot page 1.
    h.init_metadata(
        tables_for([[1]]), seq_lens=[4], mode=ForwardMode.EXTEND, extend_prefix_lens=[0]
    )
    a_outs = [h.extend(layer_id, prefix["mixed"], prefix["g_raw"], prefix["beta_raw"])]
    conv = pool.get_component(layer_id, "conv_state")
    ssm = pool.get_component(layer_id, "recurrent_state")
    snap_conv = conv[1].clone()
    snap_ssm = ssm[1].clone()
    assert snap_ssm.abs().max().item() > 0.0

    # 2) A decodes token 5: crossing out of the snapshot (in=1, out=2).
    h.init_metadata(tables_for([[1, 2]]), seq_lens=[5], mode=ForwardMode.DECODE)
    md = h.backend.forward_metadata
    assert md.state_in_blocks_by_group[gid].tolist() == [1]
    assert md.state_out_blocks_by_group[gid].tolist() == [2]
    a_outs.append(
        h.decode(
            layer_id,
            a_new["mixed"][:1],
            a_new["g_raw"][:1],
            a_new["beta_raw"][:1],
            bs=1,
        )
    )
    assert torch.equal(conv[1], snap_conv)
    assert torch.equal(ssm[1], snap_ssm)

    # 3) B resumes from the shared snapshot: extend prefix=4 -> 7 tokens
    #    (in = shared page 1, out = fresh page 3; page 1 must stay intact).
    h.init_metadata(
        tables_for([[1, 3]]),
        seq_lens=[7],
        mode=ForwardMode.EXTEND,
        extend_prefix_lens=[4],
    )
    md = h.backend.forward_metadata
    assert md.state_in_blocks_by_group[gid].tolist() == [1]
    assert md.state_out_blocks_by_group[gid].tolist() == [3]
    b_outs = [
        h.extend(
            layer_id, b_new["mixed"][:3], b_new["g_raw"][:3], b_new["beta_raw"][:3]
        )
    ]
    assert torch.equal(conv[1], snap_conv)
    assert torch.equal(ssm[1], snap_ssm)

    # 4) Batched decode: A token 6 (in-page evolution on page 2) and B token 8
    #    (crossing to page 5) in ONE forward — isolation between requests.
    h.init_metadata(
        tables_for([[1, 2], [1, 3]]),
        seq_lens=[6, 8],
        mode=ForwardMode.DECODE,
    )
    md = h.backend.forward_metadata
    assert md.state_in_blocks_by_group[gid].tolist() == [2, 3]
    assert md.state_out_blocks_by_group[gid].tolist() == [2, 3]
    mixed = torch.cat([a_new["mixed"][1:2], b_new["mixed"][3:4]], dim=0)
    g_raw = torch.cat([a_new["g_raw"][1:2], b_new["g_raw"][3:4]], dim=0)
    beta_raw = torch.cat([a_new["beta_raw"][1:2], b_new["beta_raw"][3:4]], dim=0)
    both = h.decode(layer_id, mixed, g_raw, beta_raw, bs=2)
    a_outs.append(both[: h.H])
    b_outs.append(both[h.H :])

    # Shared snapshot page is still byte-identical after everything.
    assert torch.equal(conv[1], snap_conv)
    assert torch.equal(ssm[1], snap_ssm)
    assert ssm[0].abs().max().item() == 0.0

    naive_a, state_a, fla_a, fla_state_a = h.oracle(
        layer_id, a_full["mixed"], a_full["g_raw"], a_full["beta_raw"]
    )
    naive_b, state_b, fla_b, fla_state_b = h.oracle(
        layer_id, b_full["mixed"], b_full["g_raw"], b_full["beta_raw"]
    )
    flat_a = torch.cat(a_outs, dim=0)  # tokens 1..6
    flat_b = torch.cat(b_outs, dim=0)  # tokens 5..8 (B computes only its new)
    _assert_close(flat_a, naive_a, "branch A out vs naive")
    _assert_close(flat_a, fla_a, "branch A out vs FLA")
    _assert_close(flat_b, naive_b[4 * h.H :], "branch B out vs naive")
    _assert_close(flat_b, fla_b[4 * h.H :], "branch B out vs FLA")
    _assert_close(ssm[2], state_a, "branch A final state vs naive")
    _assert_close(ssm[2], fla_state_a, "branch A final state vs FLA")
    _assert_close(ssm[3], state_b, "branch B final state vs naive")
    _assert_close(ssm[3], fla_state_b, "branch B final state vs FLA")


@requires_cuda
@requires_fla
@pytest.mark.skipif(
    not current_platform().is_amd,
    reason="indexed paged cache KDA decode is an AMD-specific contract",
)
def test_kda_cache_pool_component_views_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real LCM pool's page-strided component views flow
    through the conv + KDA kernels: prefill + decodes on one KDA layer of
    each of the three groups, vs naive + FLA."""
    pool = _make_kimi_pool("cuda", usable_pages=4)
    contract = pool.arena.runtime_contract
    layer_ids = [_kda_layer_for_group(pool, gid) for gid in _STATE_GROUPS]

    # Kimi TP8 geometry: 12 heads, D=128, conv_dim 4608.
    class _KimiHarness(_KDAHarness):
        H = 12

    h = _KimiHarness(pool, contract, layer_ids=layer_ids, seed=2)
    # Distinct pages per group; all sequences stay inside one logical page
    # (P = 1536), so in == out after the first write (same-page evolution).
    tables = {gid: [[i + 1]] for i, gid in enumerate(_STATE_GROUPS)}
    total = 10
    streams = {layer_id: h.token_stream(total) for layer_id in layer_ids}

    outputs = {layer_id: [] for layer_id in layer_ids}
    h.init_metadata(
        tables, seq_lens=[8], mode=ForwardMode.EXTEND, extend_prefix_lens=[0]
    )
    for layer_id in layer_ids:
        s = streams[layer_id]
        outputs[layer_id].append(
            h.extend(layer_id, s["mixed"][:8], s["g_raw"][:8], s["beta_raw"][:8])
        )

    from tokenspeed_kernel.thirdparty.triton import fla_kda_recurrent

    def _unexpected_megafuse(*args, **kwargs):
        raise AssertionError("AMD paged cache decode must bypass the FLA KDA megafuse")

    indexed_decode_calls = 0
    indexed_decode = hybrid_kda.kda_paged_decode

    def _indexed_decode_spy(*args, **kwargs):
        nonlocal indexed_decode_calls
        indexed_decode_calls += 1
        return indexed_decode(*args, **kwargs)

    monkeypatch.setattr(
        fla_kda_recurrent,
        "fused_recurrent_kda_megafuse",
        _unexpected_megafuse,
    )
    monkeypatch.setattr(
        hybrid_kda,
        "kda_paged_decode",
        _indexed_decode_spy,
    )
    for pos in range(8, total):
        h.init_metadata(tables, seq_lens=[pos + 1], mode=ForwardMode.DECODE)
        for layer_id in layer_ids:
            s = streams[layer_id]
            outputs[layer_id].append(
                h.decode(
                    layer_id,
                    s["mixed"][pos : pos + 1],
                    s["g_raw"][pos : pos + 1],
                    s["beta_raw"][pos : pos + 1],
                    bs=1,
                )
            )

    assert indexed_decode_calls == len(layer_ids) * (total - 8)

    for layer_id, gid in zip(layer_ids, _STATE_GROUPS):
        s = streams[layer_id]
        naive_out, naive_state, fla_out, fla_state = h.oracle(
            layer_id, s["mixed"], s["g_raw"], s["beta_raw"]
        )
        flat_out = torch.cat(outputs[layer_id], dim=0)
        _assert_close(flat_out, naive_out, f"group {gid} out vs naive")
        _assert_close(flat_out, fla_out, f"group {gid} out vs FLA")
        ssm = pool.get_component(layer_id, "recurrent_state")
        page = tables[gid][0][0]
        # Each group's state landed at its OWN page id (pages differ per
        # group, so a cross-group index mixup would read a wrong state here).
        # NOTE: same-index bindings of different groups may alias one raw
        # slab, so pages written by the other groups are allowed to appear
        # through this layer's view — only page 0 must stay zero.
        _assert_close(ssm[page], naive_state, f"group {gid} state vs naive")
        _assert_close(ssm[page], fla_state, f"group {gid} state vs FLA")
        assert ssm[0].abs().max().item() == 0.0


def test_mask_fresh_initial_state_zeroes_recycled_bytes() -> None:
    """Fresh sequences must not inherit a recycled page's stale bytes as
    their initial recurrent state: only rows with real history keep the
    gathered state."""
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        _mask_fresh_initial_state,
    )

    stale = torch.full((3, 2, 4, 4), float("nan"))
    stale[1] = 7.0  # the one resuming row carries a real (finite) snapshot

    # None => every sequence fresh => all zeros.
    out = _mask_fresh_initial_state(stale, None)
    assert (out == 0).all()

    has_init = torch.tensor([False, True, False])
    out = _mask_fresh_initial_state(stale, has_init)
    assert (out[0] == 0).all() and (out[2] == 0).all()
    assert (out[1] == 7.0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
