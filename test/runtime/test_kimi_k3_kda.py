"""Kimi-K3 KDA recurrent state on the cache contract path.

Coverage:

- per-group KDA state metadata: ``state_in_blocks_by_group`` /
  ``state_out_blocks_by_group``
  mappings keyed by state group id, dual-index computed ONCE per group per
  batch, with a proof the three groups' indices are independent and selected
  per layer via ``pool.state_group_by_layer``;
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
from tokenspeed_kernel.numerics.torch.attention.kda import (
    kda_delta_rule,
    kda_gate,
)
from tokenspeed_kernel.ops.attention.kda import kda_recurrent_layout
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
from test.runtime.conftest import block_tables_for as _tables_for
from test.runtime.conftest import layer_for_group as _kda_layer_for_group
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from test.runtime.conftest import requires_cuda

from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
from tokenspeed.runtime.layers.attention.backends.state import kda, mamba
from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
    compute_state_block_indices,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)

register_cuda_ci(est_time=240, suite="runtime-1gpu")

_LOWER_BOUND = -5.0


@pytest.mark.parametrize("lengths", [(4, 2), (1,)])
def test_prefill_hands_the_stored_state_to_the_op_untouched(
    lengths, monkeypatch
) -> None:
    backend = object.__new__(KdaAttnBackend)
    backend.kda_recurrent_layout = "v_major"
    backend.kda_backend = "auto"
    backend._kda_gate = lambda g_raw, *_args: g_raw
    backend._prefill_plan_rows = {7: 0, 9: 1}
    # Prepare boundaries once; repeated preparation must preserve their identity.
    boundaries = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    scan_boundaries = backend._prepare_prefill_scan_query_start_loc(boundaries)
    assert scan_boundaries.dtype == torch.int64
    assert torch.equal(scan_boundaries, boundaries)
    assert (
        backend._prepare_prefill_scan_query_start_loc(scan_boundaries)
        is scan_boundaries
    )
    # The scan takes the boundaries it is handed, never the metadata's cached
    # ones, even when those have the same row count.
    full_boundaries = torch.tensor([0, 5, 8], dtype=torch.int32)
    full_scan_boundaries = full_boundaries.to(torch.int64)
    plan = object()
    backend.forward_metadata = mamba.MambaForwardMetadata(
        query_start_loc=full_boundaries,
        scan_query_start_loc=full_scan_boundaries,
        query_start_loc_int64=full_scan_boundaries,
        prefill_scan_plan=plan,
    )
    num_tokens = sum(lengths)
    stored = torch.arange(len(lengths) * 24, dtype=torch.float32).view(
        len(lengths), 2, 3, 4
    )
    final = torch.empty_like(stored)
    captured = {}

    def fake_prefill(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(out=torch.empty(1, num_tokens, 2, 4), final_state=final)

    monkeypatch.setattr(kda, "kda_paged_prefill", fake_prefill)
    query = torch.empty(1, num_tokens, 2, 3)
    value = torch.empty(1, num_tokens, 2, 4)
    _, final_state = backend._prefill_scan(
        query,
        query,
        value,
        stored,
        scan_boundaries,
        layer_id=9,
        A_log=torch.empty(2),
        dt_bias=torch.empty(2, 3),
        D=None,
        a=None,
        b=None,
        g_raw=torch.empty_like(query),
        f_a_out=None,
        f_b_weight=None,
        beta_raw=torch.empty(1, num_tokens, 2),
        seq_len=num_tokens,
        num_real_tokens=num_tokens,
        lower_bound=-5.0,
        cu_seqlens_cpu=boundaries.to(torch.int64),
    )
    assert captured["initial_state"] is stored
    assert captured["cu_seqlens"] is scan_boundaries
    assert captured["recurrent_layout"] == "v_major"
    # Layers share the forward's plan, each through its own handoff-flag row:
    # one row reused across layers would hand a layer stale handoff flags.
    assert captured["plan"] is plan
    assert captured["plan_layer"] == 1
    assert final_state is final
    assert "out" not in captured


def _backend_config(device: str, *, spec_tokens: int = 1):
    """(AttnConfig, softmax spec): model-wide facts + softmax geometry."""
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    spec = MHAConfig(
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=128,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=8,
        max_resident_requests=8,
        is_draft=False,
        speculative_num_draft_tokens=spec_tokens,
        components=(spec,),
    )
    return config, spec


def _backend(device: str, *, contract_pool, spec_tokens: int = 1) -> KdaAttnBackend:
    kda_backend = "auto" if current_platform().is_amd else "fla"
    config, spec = _backend_config(device, spec_tokens=spec_tokens)
    backend = KdaAttnBackend(
        config, spec, enable_prefill_graph=False, kda_backend=kda_backend
    )
    backend.set_kv_pool(contract_pool)
    # RuntimeStates.stopped and the executor's per-row request ends, as the
    # executor binds them; no request ends here.
    backend.bind_request_stop_flags(
        torch.zeros(
            RequestSlotLayout(config.max_resident_requests).capacity,
            dtype=torch.bool,
            device=device,
        ),
        torch.zeros(config.max_bs, dtype=torch.int32, device=device),
    )
    return backend


def _stub_contract(*, prefix_granularity: int, usable_pages: int):
    """Small-P contract with the Kimi group topology (1 history + 3 state)."""
    group_ids = ("full_attention", *_STATE_GROUPS)
    specs = tuple(
        (
            CacheGroupSpec(
                group_id=group_id,
                attn_kind="full_history",
                rows_per_page=prefix_granularity,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
                replayable=False,
            )
            if group_id == "full_attention"
            else CacheGroupSpec(
                group_id=group_id,
                attn_kind="state",
                sliding_window_tokens=None,
                replayable=False,
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

    def __init__(
        self,
        contract,
        device,
        conv_dim,
        width,
        num_heads,
        head_dim,
        *,
        recurrent_dtype: torch.dtype = torch.float32,
    ):
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
                    dtype=recurrent_dtype,
                ),
            }
            for layer_id in self._groups
        }

    @property
    def state_group_by_layer(self) -> dict[int, str]:
        return self._groups

    def get_component(self, layer_id: int, name: str) -> torch.Tensor:
        return self._components[layer_id][name]


def _logical_checkpoint(backend, layer, page, request_slot):
    """The page's dense state followed by its request slot's accepted records."""
    pool = backend.kv_pool
    state = pool.get_component(layer, "recurrent_state")[page].float().clone()
    if backend._retained_records is None:
        # No decode has allocated the request-slot workspace yet.
        return state
    # Every layer shares the slot's count.
    length = int(backend._replay_length[request_slot].item())
    if length:
        layer_row = backend._descriptor_row_by_layer[layer]
        correction, key, decay = backend._retained_records[request_slot, layer_row]
        for token in range(length):
            state = (
                state * decay[:, token, None, :]
                + correction[:, token, :, None] * key[:, token, None, :]
            )
    return state


def _naive_kda_scan(q, k, v, g_raw, beta_raw, A_log, dt_bias, lower_bound, S0):
    """Naive fp32 KDA recurrence (per-channel safe-gated delta rule).

    q/k/v/g_raw: [T, H, D]; beta_raw: [T, H]; A_log: [H]; dt_bias: [H*D];
    S0: [H, D, D] (K-major). Mirrors fla's in-kernel semantics: l2norm with
    eps 1e-6, gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)),
    beta = sigmoid(beta_raw), scale = D ** -0.5.
    """
    T, H, D = q.shape
    qf, kf = q.float(), k.float()
    qf = qf / (qf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    kf = kf / (kf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    outputs, state = kda_delta_rule(
        qf * D**-0.5,
        kf,
        v,
        kda_gate(g_raw, A_log, dt_bias.view(H, D), lower_bound=lower_bound),
        torch.sigmoid(beta_raw.float()),
        S0.transpose(-1, -2),
    )
    return outputs, state.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Contract binding (set_kv_pool)
# ---------------------------------------------------------------------------


def test_set_kv_pool_binds_contract_state_groups() -> None:
    pool = _make_kimi_pool("cpu")
    backend = _backend("cpu", contract_pool=pool)
    assert backend.state_paging_active
    assert backend._state_group_ids == _STATE_GROUPS
    assert backend.deferred_state_commit
    # The default capacity, whatever the window.
    assert backend._retained_capacity == 16


def test_retained_state_groups_defer_their_decode_state() -> None:
    """Every KDA State group keeps decode records in request slots until a
    request ends or is retracted, which the scheduler's peer fence relies on;
    a composite reports its KDA child's groups and nothing else."""
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )

    pool = _make_kimi_pool("cpu")
    backend = _backend("cpu", contract_pool=pool)
    assert backend.deferred_decode_state_groups() == frozenset(_STATE_GROUPS)
    config, spec = _backend_config("cpu")
    full = AttentionBackend(config, spec)
    assert full.deferred_decode_state_groups() == frozenset()
    root = HybridLinearAttnBackend(full, backend, [])
    assert root.deferred_decode_state_groups() == frozenset(_STATE_GROUPS)


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


def test_two_slot_resolve_runs_once_per_group_and_groups_are_independent(
    monkeypatch,
) -> None:
    pool = _make_kimi_pool("cpu", usable_pages=24)
    backend = _backend("cpu", contract_pool=pool)

    resolve_calls = 0
    resolved_tables = 0
    # KDA always defers its decode state, so decode resolves through verify.
    real = mamba.verify_state_groups

    def counting(*args, **kwargs):
        nonlocal resolve_calls, resolved_tables
        resolve_calls += 1
        resolved_tables += len(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(mamba, "verify_state_groups", counting)

    bs = 2
    tables = _kimi_tables(bs, width=2)
    # Request 0 is the first decode after prompt end (the rotated checkpoint
    # in slot 0); request 1 evolves its live block in place.
    for group_id in _STATE_GROUPS:
        tables[group_id][1, 0] = 0
    delivered = _tables_for(pool.arena.runtime_contract, tables, "cpu")
    seq_lens = torch.tensor([65, 5], dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=bs)
    backend.refresh_decode_metadata(
        bs,
        bs,
        torch.tensor([0, 1], dtype=torch.int32),
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        block_tables=delivered,
    )

    # One resolve per state group over its own table, never per layer; a
    # verify resolves every group in one call.
    assert resolved_tables == len(_STATE_GROUPS)
    assert resolve_calls == 1

    md = backend.forward_metadata
    assert tuple(sorted(md.state_in_blocks_by_group)) == _STATE_GROUPS
    assert tuple(sorted(md.state_out_blocks_by_group)) == _STATE_GROUPS
    for group_id in _STATE_GROUPS:
        rows = tables[group_id]
        assert md.state_in_blocks_by_group[group_id].tolist() == [
            int(rows[0, 0]),
            int(rows[1, 1]),
        ]
        assert md.state_out_blocks_by_group[group_id].tolist() == [
            int(rows[0, 1]),
            int(rows[1, 1]),
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
    backend.init_cuda_graph_state(max_bs=2)
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
    delivered = _tables_for(pool.arena.runtime_contract, tables, "cpu")
    # bs 2 requests, one padding row (real_bs 1). Decode: before = seq-1.
    backend.refresh_decode_metadata(
        2,
        1,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([5, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        for_graph_replay=True,
        block_tables=delivered,
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
            torch.tensor([4]),  # before = seq_len 5 - 1
            validate=False,
            group_id=gid,
        )
        assert buf[0].item() == expected_in[0].item()
        assert buf[1].item() == backend.pad_slot_id


# ---------------------------------------------------------------------------
# GPU: KDA prefill/decode over two-slot state tables vs naive + FLA
# ---------------------------------------------------------------------------


class _KDAHarness:
    """Drives KdaAttnBackend forwards over a contract pool and mirrors
    them with naive fp32 + FLA oracles."""

    H = 4
    D = 128
    WIDTH = 4
    MAX_BS = 4

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
        # RuntimeStates.stopped, as the executor binds it: a test marks the
        # slot of a request whose round ends it, before that round's commit.
        self.stopped = torch.zeros(16, dtype=torch.bool, device=device)
        self.request_ends = torch.zeros(self.MAX_BS, dtype=torch.int32, device=device)
        self.backend.bind_request_stop_flags(self.stopped, self.request_ends)
        self.backend.init_cuda_graph_state(max_bs=self.MAX_BS)
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

    def _delivered(self, tables):
        np_tables = {
            gid: np.asarray(rows, dtype=np.int32) for gid, rows in tables.items()
        }
        return _tables_for(self.contract, np_tables, self.device)

    def extend_metadata(
        self, tables, seq_lens, extend_prefix_lens, num_extends, request_slots=None
    ):
        """An EXTEND batch, or a MIXED one whose rows past ``num_extends``
        decode one token each; the executor's host mirrors carry each row's
        new-token and prefix lengths."""
        bs = len(seq_lens)
        prefix_cpu = torch.tensor(extend_prefix_lens, dtype=torch.int32)
        new_cpu = torch.tensor(seq_lens, dtype=torch.int32) - prefix_cpu
        self.backend.init_forward_metadata(
            bs=bs,
            num_extends=num_extends,
            req_pool_indices=torch.tensor(
                range(1, bs + 1) if request_slots is None else request_slots,
                dtype=torch.int32,
                device=self.device,
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            forward_mode=ForwardMode.from_num_extends(num_extends, bs),
            block_tables=self._delivered(tables),
            extend_seq_lens=new_cpu.to(self.device),
            extend_seq_lens_cpu=new_cpu,
            extend_prefix_lens=prefix_cpu.to(self.device),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_replay_lens_cpu=torch.zeros_like(prefix_cpu),
            extend_prompt_lens_cpu=prefix_cpu + new_cpu,
            extend_with_prefix=bool(prefix_cpu.any()),
            query_shard=None,
        )

    def decode_metadata(self, tables, seq_lens):
        """A decode step: the same unpadded refresh the runner issues."""
        bs = len(seq_lens)
        self.backend.refresh_decode_metadata(
            bs,
            bs,
            torch.arange(1, bs + 1, dtype=torch.int32, device=self.device),
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            forward_mode=ForwardMode.DECODE,
            block_tables=self._delivered(tables),
        )

    def extend(self, layer_id, mixed, g_raw, beta_raw, bs=1, *, forward_mode):
        seq_len = mixed.shape[0]
        # The model hands over the q|k|v slice of its merged projection rows
        # (here [q|k|v|g]); it used to pass a compact copy. From the same
        # states, both must leave the same output and state bytes. The slice
        # runs last, so its states carry on. Fresh rows keep the raw token
        # stream intact for the oracle.
        rows = torch.cat((mixed, g_raw), dim=-1)
        states = [
            self.pool.get_component(layer_id, name)
            for name in ("conv_state", "recurrent_state")
        ]
        initial = [state.clone() for state in states]
        results = []
        for mixed_qkv in (
            rows[:, : mixed.shape[1]].contiguous(),
            rows[:, : mixed.shape[1]],
        ):
            for state, start in zip(states, initial):
                state.copy_(start)
            out = self.backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=self.pool,
                bs=bs,
                forward_mode=forward_mode,
                mixed_qkv=mixed_qkv,
                save_kv_cache=True,
                g_raw=g_raw,
                beta_raw=beta_raw,
                seq_len=seq_len,
                **self.common_kwargs(layer_id),
            ).flatten(0, 1)
            results.append([out, *(state.clone() for state in states)])
        for compact, view in zip(*results):
            assert view.dtype == compact.dtype
            assert torch.equal(
                view.contiguous().view(torch.uint8),
                compact.contiguous().view(torch.uint8),
            )
        return results[1][0]

    def commit_decode(self, bs):
        self.backend.commit_state_after_acceptance(
            torch.ones(bs, dtype=torch.int32, device=self.device),
            num_extends=0,
            accepted_path=None,
        )

    def decode(self, layer_id, mixed, g_raw, beta_raw, bs):
        # The conv kernels may write their output in place into mixed_qkv (a
        # fresh projection in production); clone so the oracle can reuse the
        # raw token stream afterwards.
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
        ).reshape(-1, self.D)

    def oracle(self, layer_id, mixed, g_raw, beta_raw):
        from test.runtime.kda_torch_reference import kda_torch_reference

        params = self.params[layer_id]
        output, states, _ = kda_torch_reference(
            mixed.unsqueeze(0),
            torch.zeros(
                1, self.conv_dim, self.WIDTH - 1, device=self.device, dtype=mixed.dtype
            ),
            torch.zeros(1, self.H, self.D, self.D, device=self.device),
            beta_raw,
            conv_weights=params["conv_weights"],
            bias=params["bias"],
            gate=g_raw,
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            lower_bound=_LOWER_BOUND,
        )
        return output.reshape(-1, self.D), states[0, -1]


def _to_slab_layout(state: torch.Tensor) -> torch.Tensor:
    """Oracles build K-major states; the slab holds the platform's own layout."""
    if kda_recurrent_layout() == "v_major":
        return state.transpose(-1, -2)
    return state


def _assert_close(actual, expected, what, mean_tol=2e-3, atol=1e-1, rtol=1e-2):
    diff = (actual.float() - expected.float()).abs()
    assert diff.mean().item() < mean_tol, f"{what}: mean diff {diff.mean().item()}"
    assert torch.allclose(
        actual.float(), expected.float(), atol=atol, rtol=rtol
    ), f"{what}: max diff {diff.max().item()}"


@requires_cuda
@requires_fla
def test_kda_three_groups_zero_state_same_page_and_crossing() -> None:
    """Prefill from zero state, the first decode off the rotated prompt-end
    checkpoint, in-place decodes, per-group table independence -- vs naive
    fp32 recurrence AND FLA."""
    P, usable = 4, 8
    contract = _stub_contract(prefix_granularity=P, usable_pages=usable)
    pool = _StubContractPool(
        contract, "cuda", conv_dim=3 * 4 * 128, width=4, num_heads=4, head_dim=128
    )
    h = _KDAHarness(pool, contract, layer_ids=[0, 1, 2])
    # Distinct blocks per group: writes to the wrong group's blocks would land
    # in rows this test asserts stay zero. Block (checkpoint, live) per group.
    blocks = {
        "linear_attention_0": (1, 4),
        "linear_attention_1": (2, 5),
        "linear_attention_2": (3, 6),
    }
    streams = {layer_id: h.token_stream(6) for layer_id in [0, 1, 2]}

    outputs = {layer_id: [] for layer_id in [0, 1, 2]}
    # Prefill 3 tokens from the zero state (slot 0 null) into the checkpoint block.
    h.extend_metadata(
        {gid: [[0, checkpoint]] for gid, (checkpoint, _) in blocks.items()},
        seq_lens=[3],
        extend_prefix_lens=[0],
        num_extends=1,
    )
    md = h.backend.forward_metadata
    for gid in _STATE_GROUPS:
        assert md.state_in_blocks_by_group[gid].tolist() == [0]
        assert md.state_out_blocks_by_group[gid].tolist() == [blocks[gid][0]]
    for layer_id in [0, 1, 2]:
        s = streams[layer_id]
        outputs[layer_id].append(
            h.extend(
                layer_id,
                s["mixed"][:3],
                s["g_raw"][:3],
                s["beta_raw"][:3],
                forward_mode=ForwardMode.EXTEND,
            )
        )
    # The prompt-end block is published: read-only from here on.
    snapshot = {}
    for layer_id, gid in zip([0, 1, 2], _STATE_GROUPS):
        page = blocks[gid][0]
        snapshot[layer_id] = (
            pool.get_component(layer_id, "conv_state")[page].clone(),
            pool.get_component(layer_id, "recurrent_state")[page].clone(),
        )

    # Decode token 4 reads the rotated checkpoint and writes the fresh live
    # block; tokens 5 and 6 evolve the live block in place (slot 0 released).
    decode_tables = [
        {gid: [[checkpoint, live]] for gid, (checkpoint, live) in blocks.items()},
        {gid: [[0, live]] for gid, (_, live) in blocks.items()},
        {gid: [[0, live]] for gid, (_, live) in blocks.items()},
    ]
    expected_pages = {
        gid: [(checkpoint, live), (live, live), (live, live)]
        for gid, (checkpoint, live) in blocks.items()
    }
    for step, pos in enumerate(range(3, 6)):
        h.decode_metadata(decode_tables[step], seq_lens=[pos + 1])
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
        h.commit_decode(1)

    for layer_id, gid in zip([0, 1, 2], _STATE_GROUPS):
        s = streams[layer_id]
        naive_out, naive_state = h.oracle(
            layer_id, s["mixed"], s["g_raw"], s["beta_raw"]
        )
        flat_out = torch.cat(outputs[layer_id], dim=0)
        _assert_close(flat_out, naive_out, f"layer {layer_id} out vs naive")
        ssm = pool.get_component(layer_id, "recurrent_state")
        conv = pool.get_component(layer_id, "conv_state")
        snap_page, final_page = blocks[gid]
        _assert_close(
            _logical_checkpoint(h.backend, layer_id, final_page, 1),
            naive_state,
            f"layer {layer_id} state vs naive",
        )
        # The published checkpoint is untouched by the decode that read it.
        assert torch.equal(conv[snap_page], snapshot[layer_id][0])
        assert torch.equal(ssm[snap_page], snapshot[layer_id][1])
        assert ssm[snap_page].abs().max().item() > 0.0
        # Null page 0 is never written.
        assert conv[0].abs().max().item() == 0.0
        assert ssm[0].abs().max().item() == 0.0
        # Independence: blocks belonging to the OTHER groups stay zero in this
        # layer's slabs -- the layer consumed only its own group's indices.
        other_pages = [
            page
            for other_gid, pages in blocks.items()
            if other_gid != gid
            for page in pages
        ]
        for page in other_pages:
            assert ssm[page].abs().max().item() == 0.0, (layer_id, page)
            assert conv[page].abs().max().item() == 0.0, (layer_id, page)


@requires_cuda
@requires_fla
def test_kda_prefix_resume_copy_on_write_and_isolation() -> None:
    """Prefix hit at a published prompt-end checkpoint: the shared block is
    read-only for both branches, and a MIXED forward and batched decode keep
    requests isolated."""
    P, usable = 4, 16
    contract = _stub_contract(prefix_granularity=P, usable_pages=usable)
    pool = _StubContractPool(
        contract, "cuda", conv_dim=3 * 4 * 128, width=4, num_heads=4, head_dim=128
    )
    h = _KDAHarness(pool, contract, layer_ids=[0, 1, 2], seed=1)
    layer_id = 0
    gid = "linear_attention_0"

    def tables_for(rows0):
        # The other groups' layers run every forward too, on their own
        # blocks: one count serves every layer of a request.
        return {
            "linear_attention_0": rows0,
            "linear_attention_1": [[p + 4 if p else 0 for p in row] for row in rows0],
            "linear_attention_2": [[p + 8 if p else 0 for p in row] for row in rows0],
        }

    def extend_all(*args, **kwargs):
        """Run every layer's extend; the checks follow layer 0's."""
        return [h.extend(layer, *args, **kwargs) for layer in h.layer_ids][0]

    def decode_all(*args, **kwargs):
        """Run every layer's verify before a commit; the checks follow layer 0's."""
        return [h.decode(layer, *args, **kwargs) for layer in h.layer_ids][0]

    # Shared prefix: 4 tokens (exactly one page). Branch A continues with 3
    # decodes; branch B resumes from the snapshot with a 3-token extend and
    # one decode. Total streams: A = prefix + a4 a5 a6; B = prefix + b4 b5 b6 b7.
    prefix = h.token_stream(4)
    a_new = h.token_stream(3)
    b_new = h.token_stream(4)

    def cat(*streams):
        return {
            key: torch.cat([s[key] for s in streams], dim=0)
            for key in ("mixed", "g_raw", "beta_raw")
        }

    a_full = cat(prefix, a_new)
    b_full = cat(prefix, b_new)

    # 1) A prefills the shared 4-token prefix -> its prompt-end state, block 1.
    h.extend_metadata(
        tables_for([[0, 1]]), seq_lens=[4], extend_prefix_lens=[0], num_extends=1
    )
    a_outs = [
        extend_all(
            prefix["mixed"],
            prefix["g_raw"],
            prefix["beta_raw"],
            forward_mode=ForwardMode.EXTEND,
        )
    ]
    conv = pool.get_component(layer_id, "conv_state")
    ssm = pool.get_component(layer_id, "recurrent_state")
    snap_conv = conv[1].clone()
    snap_ssm = ssm[1].clone()
    assert snap_ssm.abs().max().item() > 0.0

    # 2) A's first decode (token 5) reads the rotated checkpoint (slot 0 =
    #    block 1) and writes the fresh live block 2.
    h.decode_metadata(tables_for([[1, 2]]), seq_lens=[5])
    md = h.backend.forward_metadata
    assert md.state_in_blocks_by_group[gid].tolist() == [1]
    assert md.state_out_blocks_by_group[gid].tolist() == [2]
    a_outs.append(
        decode_all(
            a_new["mixed"][:1],
            a_new["g_raw"][:1],
            a_new["beta_raw"][:1],
            bs=1,
        )
    )
    h.commit_decode(1)
    assert torch.equal(conv[1], snap_conv)
    assert torch.equal(ssm[1], snap_ssm)

    # 3) One MIXED forward: B hits the published checkpoint, extend prefix=4
    #    -> 7 tokens (in = shared block 1, out = fresh block 3; block 1 must
    #    stay intact), while A decodes token 6 in place on block 2 (slot 0
    #    released).
    h.extend_metadata(
        tables_for([[1, 3], [0, 2]]),
        seq_lens=[7, 6],
        extend_prefix_lens=[4, 5],
        num_extends=1,
        request_slots=[2, 1],
    )
    md = h.backend.forward_metadata
    assert md.state_in_blocks_by_group[gid].tolist() == [1, 2]
    assert md.state_out_blocks_by_group[gid].tolist() == [3, 2]
    both = extend_all(
        torch.cat([b_new["mixed"][:3], a_new["mixed"][1:2]], dim=0),
        torch.cat([b_new["g_raw"][:3], a_new["g_raw"][1:2]], dim=0),
        torch.cat([b_new["beta_raw"][:3], a_new["beta_raw"][1:2]], dim=0),
        bs=2,
        forward_mode=ForwardMode.MIXED,
    )
    h.backend.commit_state_after_acceptance(
        torch.ones(2, dtype=torch.int32, device=h.device),
        num_extends=1,
        accepted_path=None,
    )
    b_outs = [both[: 3 * h.H]]
    a_outs.append(both[3 * h.H :])
    assert torch.equal(conv[1], snap_conv)
    assert torch.equal(ssm[1], snap_ssm)

    # 4) Batched decode: A token 7 (in place on block 2) and B's first
    #    decode, token 8 (its prompt-end block 3 rotated into slot 0, fresh
    #    live block 5), in ONE forward -- isolation between requests.
    h.decode_metadata(tables_for([[0, 2], [3, 5]]), seq_lens=[7, 8])
    md = h.backend.forward_metadata
    assert md.state_in_blocks_by_group[gid].tolist() == [2, 3]
    assert md.state_out_blocks_by_group[gid].tolist() == [2, 5]
    b_prompt_end = (conv[3].clone(), ssm[3].clone())
    mixed = torch.cat([a_new["mixed"][2:3], b_new["mixed"][3:4]], dim=0)
    g_raw = torch.cat([a_new["g_raw"][2:3], b_new["g_raw"][3:4]], dim=0)
    beta_raw = torch.cat([a_new["beta_raw"][2:3], b_new["beta_raw"][3:4]], dim=0)
    both = decode_all(mixed, g_raw, beta_raw, bs=2)
    h.commit_decode(2)
    a_outs.append(both[: h.H])
    b_outs.append(both[h.H :])

    # Both published checkpoints are still byte-identical after everything.
    assert torch.equal(conv[1], snap_conv)
    assert torch.equal(ssm[1], snap_ssm)
    assert torch.equal(conv[3], b_prompt_end[0])
    assert torch.equal(ssm[3], b_prompt_end[1])
    assert ssm[0].abs().max().item() == 0.0

    naive_a, state_a = h.oracle(
        layer_id, a_full["mixed"], a_full["g_raw"], a_full["beta_raw"]
    )
    naive_b, state_b = h.oracle(
        layer_id, b_full["mixed"], b_full["g_raw"], b_full["beta_raw"]
    )
    flat_a = torch.cat(a_outs, dim=0)  # tokens 1..7
    flat_b = torch.cat(b_outs, dim=0)  # tokens 5..8 (B computes only its new)
    _assert_close(flat_a, naive_a, "branch A out vs naive")
    _assert_close(flat_b, naive_b[4 * h.H :], "branch B out vs naive")
    _assert_close(
        _logical_checkpoint(h.backend, layer_id, 2, 1),
        state_a,
        "branch A final state vs naive",
    )
    _assert_close(
        _logical_checkpoint(h.backend, layer_id, 5, 2),
        state_b,
        "branch B final state vs naive",
    )


@requires_cuda
@requires_fla
@pytest.mark.skipif(
    not current_platform().is_amd,
    reason="indexed cache-group KDA decode is an AMD-specific contract",
)
def test_kda_cache_pool_component_views_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real LCM pool's page-strided component views flow
    through the conv + KDA kernels: prefill + decodes on every KDA layer,
    one layer of each of the three groups checked vs naive + FLA. Every
    layer of a request verifies each round, as its one count requires."""
    pool = _make_kimi_pool("cuda", usable_pages=4)
    contract = pool.arena.runtime_contract
    checked = [_kda_layer_for_group(pool, gid) for gid in _STATE_GROUPS]
    layer_ids = sorted(pool.state_group_by_layer)

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
    h.extend_metadata(tables, seq_lens=[8], extend_prefix_lens=[0], num_extends=1)
    for layer_id in layer_ids:
        s = streams[layer_id]
        outputs[layer_id].append(
            h.extend(
                layer_id,
                s["mixed"][:8],
                s["g_raw"][:8],
                s["beta_raw"][:8],
                forward_mode=ForwardMode.EXTEND,
            )
        )

    for pos in range(8, total):
        h.decode_metadata(tables, seq_lens=[pos + 1])
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
        h.commit_decode(1)

    for layer_id, gid in zip(checked, _STATE_GROUPS):
        s = streams[layer_id]
        naive_out, naive_state = h.oracle(
            layer_id, s["mixed"], s["g_raw"], s["beta_raw"]
        )
        flat_out = torch.cat(outputs[layer_id], dim=0)
        _assert_close(flat_out, naive_out, f"group {gid} out vs naive")
        ssm = pool.get_component(layer_id, "recurrent_state")
        page = tables[gid][0][0]
        # Each group's state landed at its OWN page id (pages differ per
        # group, so a cross-group index mixup would read a wrong state here).
        # NOTE: same-index bindings of different groups may alias one raw
        # slab, so pages written by the other groups are allowed to appear
        # through this layer's view — only page 0 must stay zero.
        _assert_close(
            _logical_checkpoint(h.backend, layer_id, page, 1),
            naive_state,
            f"group {gid} state vs naive",
        )
        assert ssm[0].abs().max().item() == 0.0


def test_prefill_state_inputs_zero_fresh_rows_without_reading_null_page() -> None:
    """Fresh sequences must not inherit a recycled page's stale bytes as
    their initial recurrent state: only the row that resumes real history
    keeps the gathered snapshot, and no row reads physical page 0."""
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        _prepare_cache_prefill_state_inputs,
    )

    # Pages 1, 3, 4 are freshly allocated working pages still carrying a
    # previous tenant's bytes; page 2 is a real (finite) snapshot.
    ssm_states = torch.full((5, 2, 4, 4), float("nan"))
    ssm_states[0] = 0.0
    ssm_states[2] = 7.0
    conv_states = torch.arange(5, dtype=torch.float32).view(5, 1, 1).expand(5, 3, 2)
    conv_states = conv_states.clone()
    state_in = torch.tensor([0, 2, 0], dtype=torch.int32)
    state_out = torch.tensor([1, 3, 4], dtype=torch.int64)

    recurrent_state, has_initial_state = _prepare_cache_prefill_state_inputs(
        conv_states, ssm_states, state_in, state_out, dtype=torch.float32
    )

    assert has_initial_state.tolist() == [False, True, False]
    assert (recurrent_state[0] == 0).all() and (recurrent_state[2] == 0).all()
    assert (recurrent_state[1] == 7.0).all()
    # The resumed row copies its snapshot's conv page into its working page;
    # fresh rows keep their own working page and never touch page 0.
    assert (conv_states[3] == 2.0).all()
    assert (conv_states[1] == 1.0).all() and (conv_states[4] == 4.0).all()
    assert (conv_states[0] == 0.0).all()


@requires_cuda
@pytest.mark.parametrize("prefix", [0, 1024])
@pytest.mark.parametrize("backend_name", ["cutedsl_kda", "cudnn"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kda_prefill_fused_staging_matches_pytorch(
    prefix, backend_name, state_dtype, monkeypatch
):
    """Native KDA outputs and persistent state match the old staging ops."""
    from tokenspeed_kernel.ops.attention.kda.cute_dsl import cutedsl_kda_supported

    if not cutedsl_kda_supported():
        pytest.skip("requires the native CuteDSL KDA prefill package")

    contract = _stub_contract(prefix_granularity=1024, usable_pages=8)
    pools = [
        _StubContractPool(
            contract, "cuda", conv_dim=3 * 4 * 128, width=4, num_heads=4, head_dim=128
        )
        for _ in range(2)
    ]
    for layer_id in range(3):
        for pool in pools:
            pool._components[layer_id]["recurrent_state"] = pool.get_component(
                layer_id, "recurrent_state"
            ).to(state_dtype)
        for component in ("conv_state", "recurrent_state"):
            before = pools[0].get_component(layer_id, component)
            before.normal_(0, 0.1)
            pools[1].get_component(layer_id, component).copy_(before)
    harnesses = [
        _KDAHarness(pool, contract, layer_ids=[0, 1, 2], device="cuda", seed=42)
        for pool in pools
    ]
    tables = {
        gid: [[2 * row + 1, 2 * row + 2]] for row, gid in enumerate(_STATE_GROUPS)
    }
    for harness in harnesses:
        harness.backend.kda_backend = backend_name
        harness.extend_metadata(
            tables,
            seq_lens=[prefix + 868],
            extend_prefix_lens=[prefix],
            num_extends=1,
        )

    def pytorch_staging(conv, state, sources, destinations, *, dtype):
        history = sources > 0
        safe = torch.where(history, sources, destinations).long()
        conv[destinations.long()] = conv[safe]
        recurrent = state[safe]
        recurrent.masked_fill_(~history[:, None, None, None], 0)
        return recurrent.to(dtype), history

    before, after = harnesses
    for layer_id in range(3):
        stream = before.token_stream(868)
        with monkeypatch.context() as patch:
            patch.setattr(mamba, "_prepare_cache_prefill_state_inputs", pytorch_staging)
            expected = before.extend(
                layer_id,
                stream["mixed"],
                stream["g_raw"],
                stream["beta_raw"],
                bs=1,
                forward_mode=ForwardMode.EXTEND,
            )
        actual = after.extend(
            layer_id,
            stream["mixed"],
            stream["g_raw"],
            stream["beta_raw"],
            bs=1,
            forward_mode=ForwardMode.EXTEND,
        )
        assert torch.equal(actual, expected)
        for component in ("conv_state", "recurrent_state"):
            assert torch.equal(
                pools[0].get_component(layer_id, component),
                pools[1].get_component(layer_id, component),
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
