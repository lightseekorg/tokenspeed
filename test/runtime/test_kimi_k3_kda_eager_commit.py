"""Retained KDA runtime commitment, graph replay and PyTorch references."""

import os
import sys

import numpy as np
import pytest
import torch

# CI registration and absolute test imports use the same paths as the suite.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-1gpu")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.cache_pool_test_utils import (
    assert_no_alias,
    storages_of,
)
from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import block_tables_for as _tables_for
from test.runtime.conftest import kimi_recipe as _kimi_recipe
from test.runtime.conftest import kimi_recipe_with_history
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from test.runtime.conftest import make_kimi_pool_with_history
from types import SimpleNamespace  # noqa: E402

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
    _packed_qkv_views,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.registry import _prepare_fixed_workspaces

_LOWER_BOUND = -5.0
H, D, D_FA = 12, 128, 128
KEY_DIM = H * D
CONV_DIM = 3 * KEY_DIM
T = 3
DEV = "cuda"
_CHECKPOINT_COMPONENTS = (
    "conv_state",
    "recurrent_state",
    "replay_correction",
    "replay_key",
    "replay_decay",
    "replay_length",
)


class _Harness:
    H = H

    def __init__(
        self,
        *,
        draft_tokens: int | None,
        seed: int = 0,
        max_bs: int = 8,
        usable_pages: int = 24,
        history_capacity: int,
        mamba_ssm_dtype: str = "float32",
    ):
        torch.manual_seed(seed)
        self.pool = make_kimi_pool_with_history(
            DEV,
            usable_pages=usable_pages,
            history_capacity=history_capacity,
            mamba_ssm_dtype=mamba_ssm_dtype,
        )
        self.contract = self.pool.arena.runtime_contract
        self.key_dim = self.H * D
        self.conv_dim = 3 * self.key_dim
        self.width = max(int(draft_tokens or 1), 1)
        spec = SoftmaxAttnConfig(
            num_attention_heads=self.H,
            num_kv_heads=self.H,
            attn_tp_size=1,
            head_dim=D,
        )
        config = AttnConfig(
            device=DEV,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_quant_method=None,
            context_len=4096,
            is_draft=False,
            speculative_num_draft_tokens=draft_tokens,
            max_bs=max_bs,
            max_resident_requests=max_bs,
            components=(spec,),
        )
        self.config = config
        self.backend = KdaAttnBackend(
            config, spec, enable_prefill_graph=False, kda_backend="auto"
        )
        self.backend.set_kv_pool(self.pool)
        # The persistent decode buffers exist from construction, as at the
        # wrapper (the verify refresh below writes into them).
        self.backend.init_cuda_graph_state(config.max_bs)
        self.backend.preallocate_verify_workspace(config.max_bs, self.width)
        self.layer_ids = list(self.backend._state_layer_ids())
        self.params = {
            layer_id: dict(
                conv_weights=torch.randn(
                    self.conv_dim, 4, device=DEV, dtype=torch.bfloat16
                )
                * 0.1,
                f_b_weight=torch.randn(
                    self.key_dim, D_FA, device=DEV, dtype=torch.bfloat16
                )
                * 0.05,
                A_log=torch.randn(self.H, device=DEV, dtype=torch.float32) * 0.1,
                dt_bias=torch.randn(self.key_dim, device=DEV, dtype=torch.float32)
                * 0.1,
                lower_bound=_LOWER_BOUND,
            )
            for layer_id in self.layer_ids
        }

    def inputs(self, bs, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=generator).to(DEV, torch.bfloat16)

        return {
            "mixed_qkv": rnd(bs * self.width, self.conv_dim),
            "f_a_out": rnd(bs * self.width, D_FA),
            "beta_raw": rnd(bs * self.width, self.H),
        }

    def prepare_metadata(self, rpis, pages, seq_lens):
        bs = len(rpis)
        tables = {
            group_id: np.asarray(
                [[0, page] for page in pages[group_id]], dtype=np.int32
            )
            for group_id in _STATE_GROUPS
        }
        delivered = _tables_for(self.contract, tables, DEV)
        self.backend.refresh_decode_metadata(
            bs,
            bs,
            torch.tensor(rpis, dtype=torch.int32, device=DEV),
            torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            block_tables=delivered,
        )

    def forward(self, inputs, bs):
        outputs = []
        for layer_id in self.layer_ids:
            params = self.params[layer_id]
            outputs.append(
                self.backend.forward_decode(
                    None,
                    None,
                    None,
                    layer=None,
                    out_cache_loc=None,
                    token_to_kv_pool=self.pool,
                    bs=bs,
                    mixed_qkv=inputs["mixed_qkv"].clone(),
                    f_a_out=inputs["f_a_out"],
                    beta_raw=inputs["beta_raw"],
                    g_raw=inputs.get("g_raw"),
                    conv_weights=params["conv_weights"],
                    bias=params.get("bias"),
                    activation="silu",
                    key_dim=self.key_dim,
                    value_dim=self.key_dim,
                    attention_tp_size=1,
                    head_k_dim=D,
                    head_v_dim=D,
                    A_log=params["A_log"],
                    dt_bias=params["dt_bias"],
                    f_b_weight=params["f_b_weight"],
                    lower_bound=params["lower_bound"],
                    layer_id=layer_id,
                    seq_len=bs * self.width,
                    a=None,
                    b=None,
                )
            )
        return outputs

    def round(self, rpis, pages, seq_lens, seed, accepted):
        bs = len(rpis)
        self.prepare_metadata(rpis, pages, seq_lens)
        self.forward(self.inputs(bs, seed), bs)
        self.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV), accepted_path=None
        )


def _assert_committed_pages_equal(left, right, pages):
    for layer_id in left.layer_ids:
        group_id = left.pool.state_group_by_layer[layer_id]
        for page in pages[group_id]:
            torch.testing.assert_close(
                left.pool.get_component(layer_id, "conv_state")[page],
                right.pool.get_component(layer_id, "conv_state")[page],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                _logical_state(left.pool, layer_id, [page])[0],
                _logical_state(right.pool, layer_id, [page])[0],
                atol=1e-5,
                rtol=1e-3,
            )


def test_packed_qkv_views_keep_projection_storage_and_stride():
    rows, row_width, production_steps = 64, 6416, 4
    packed_width = 3 * 16 * D
    projection = torch.randn(rows, row_width, dtype=torch.bfloat16, device=DEV)
    packed = projection[:, :packed_width]

    query, key, value = _packed_qkv_views(
        packed,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        head_q=D,
        head_k=D,
        head_v=D,
    )

    storage_ptr = projection.untyped_storage().data_ptr()
    for tensor in (query, key, value):
        assert tensor.untyped_storage().data_ptr() == storage_ptr
        assert tensor.stride()[1:] == (row_width, D, 1)
        recurrent_view = tensor.view(16, production_steps, 16, D)
        assert recurrent_view.stride() == (
            production_steps * row_width,
            row_width,
            D,
            1,
        )


def _reference_round(harness, pages, inputs, accepted, *, prefill):
    from test.runtime.kda_torch_reference import (
        checked_convolution,
        kda_prefill_reference,
        kda_torch_reference,
    )

    batch = len(accepted)
    states, outputs = {}, []
    for layer in harness.layer_ids:
        params = harness.params[layer]
        ids = pages[harness.pool.state_group_by_layer[layer]]
        gate = inputs.get("g_raw")
        if gate is None:
            gate = torch.nn.functional.linear(inputs["f_a_out"], params["f_b_weight"])
        raw = inputs["mixed_qkv"].view(batch, -1, harness.conv_dim)
        conv = harness.pool.get_component(layer, "conv_state")[ids]
        options = {}
        if prefill:
            reference = kda_prefill_reference
            options["solution"] = (
                None
                if harness.backend.kda_backend == "auto"
                else harness.backend.kda_backend
            )
        else:
            reference = kda_torch_reference
            options["convolved"] = checked_convolution(
                raw,
                conv,
                conv_weights=params["conv_weights"],
                bias=params.get("bias"),
                prefill=False,
            )[0]
        output, tape, windows = reference(
            inputs["mixed_qkv"].view(batch, -1, harness.conv_dim),
            harness.pool.get_component(layer, "conv_state")[ids],
            _logical_state(harness.pool, layer, ids),
            inputs["beta_raw"],
            conv_weights=params["conv_weights"],
            bias=params.get("bias"),
            gate=gate,
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            lower_bound=params["lower_bound"],
            **options,
        )
        rows = torch.arange(batch, device=DEV)
        counts = torch.tensor(accepted, device=DEV).clamp(min=1) - 1
        states[layer] = (
            (tape, windows) if prefill else (tape[rows, counts], windows[rows, counts])
        )
        outputs.append(output.reshape(-1, harness.H, D))
    return outputs, states


def _reference_committed_states(harness, pages, inputs, accepted):
    return _reference_round(harness, pages, inputs, accepted, prefill=False)[1]


def test_retained_commit_matches_torch_over_multiple_rounds_and_layers():
    """All layer groups commit the accepted prefix of the independent recurrence."""
    harness = _Harness(draft_tokens=T, seed=17, history_capacity=16)
    rpis = [0, 1, 2]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    accepts = ([0, 1, T], [T, 0, 2], [1, T, 0], [2, 1, T])
    for round_index, accepted in enumerate(accepts):
        inputs = harness.inputs(len(rpis), 100 + round_index)
        # Keep gate-projection rounding outside the accepted-state check.
        inputs["g_raw"] = inputs["mixed_qkv"][:, : harness.key_dim]
        expected = _reference_committed_states(harness, pages, inputs, accepted)
        harness.prepare_metadata(rpis, pages, seq_lens)
        harness.forward(inputs, len(rpis))
        harness.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV),
            accepted_path=None,
        )
        for layer_id, (state, conv) in expected.items():
            ids = pages[harness.pool.state_group_by_layer[layer_id]]
            torch.testing.assert_close(
                _logical_state(harness.pool, layer_id, ids),
                state,
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                harness.pool.get_component(layer_id, "conv_state")[ids],
                conv,
                atol=0,
                rtol=0,
            )
        seq_lens = [length + max(1, count) for length, count in zip(seq_lens, accepted)]


def test_graph_replay_then_post_forward_commit_matches_eager_over_rounds():
    """The captured verify graph plus the existing post-forward hook follows eager."""
    captured = _Harness(draft_tokens=T, seed=29, history_capacity=16)
    eager = _Harness(draft_tokens=T, seed=29, history_capacity=16)
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    bs = len(rpis)

    warm_inputs = captured.inputs(bs, 211)
    captured.prepare_metadata(rpis, pages, seq_lens)
    captured.forward(warm_inputs, bs)
    torch.cuda.synchronize()

    req_pool_indices = torch.tensor(rpis, dtype=torch.int32, device=DEV)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    capture_tables = {
        group_id: torch.full((bs, 2), -1, dtype=torch.int32, device=DEV)
        for group_id in _STATE_GROUPS
    }
    captured.backend.init_forward_metadata_capture_cuda_graph(
        bs,
        req_pool_indices,
        seq_lens_tensor,
        ForwardMode.DECODE,
        block_tables=capture_tables,
    )
    stable_inputs = captured.inputs(bs, 223)
    stable_accepted = torch.ones(bs, dtype=torch.int32, device=DEV)
    accepted_source = torch.ones_like(stable_accepted)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        captured.forward(stable_inputs, bs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.forward(stable_inputs, bs)
        # Stand in for the sampler/accept kernel in the full round graph: the
        # post-forward commit consumes this in-graph product.
        stable_accepted.copy_(accepted_source)

    first_layer = captured.layer_ids[0]
    payload = captured.backend._retained_qkv[
        captured.backend._descriptor_row_by_layer[first_layer]
    ]
    payload_ptr = payload.data_ptr()
    captured_payload = payload.clone()
    accepts = ([1, T], [T, 1], [2, T])
    for round_index, accepted in enumerate(accepts):
        tables = {
            group_id: np.asarray(
                [[0, page] for page in pages[group_id]], dtype=np.int32
            )
            for group_id in _STATE_GROUPS
        }
        delivered = _tables_for(captured.contract, tables, DEV)
        seq_lens_tensor.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=DEV))
        captured.backend.refresh_decode_metadata(
            bs,
            bs,
            req_pool_indices,
            seq_lens_tensor,
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables=delivered,
        )
        replay_inputs = captured.inputs(bs, 227 + round_index)
        for name, value in replay_inputs.items():
            stable_inputs[name].copy_(value)
        accepted_source.copy_(torch.tensor(accepted, dtype=torch.int32, device=DEV))
        graph.replay()
        captured.backend.commit_verified_state(stable_accepted, accepted_path=None)

        eager.prepare_metadata(rpis, pages, seq_lens)
        eager.forward(replay_inputs, bs)
        eager.backend.commit_verified_state(stable_accepted, accepted_path=None)
        torch.cuda.synchronize()
        _assert_committed_pages_equal(captured, eager, pages)
        seq_lens = [length + count for length, count in zip(seq_lens, accepted)]

    assert payload.data_ptr() == payload_ptr
    assert not torch.equal(payload, captured_payload)

    # Negative control: replaying the next round without the required
    # post-forward commit must diverge from the multi-round eager oracle.
    tables = {
        group_id: np.asarray([[0, page] for page in pages[group_id]], dtype=np.int32)
        for group_id in _STATE_GROUPS
    }
    delivered = _tables_for(captured.contract, tables, DEV)
    seq_lens_tensor.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=DEV))
    captured.backend.refresh_decode_metadata(
        bs,
        bs,
        req_pool_indices,
        seq_lens_tensor,
        forward_mode=ForwardMode.DECODE,
        for_graph_replay=True,
        block_tables=delivered,
    )
    replay_inputs = captured.inputs(bs, 251)
    for name, value in replay_inputs.items():
        stable_inputs[name].copy_(value)
    accepted_source.fill_(T)
    graph.replay()  # Deliberately omit commit_verified_state.
    eager.prepare_metadata(rpis, pages, seq_lens)
    eager.forward(replay_inputs, bs)
    eager.backend.commit_verified_state(stable_accepted, accepted_path=None)
    torch.cuda.synchronize()
    with pytest.raises(AssertionError):
        _assert_committed_pages_equal(captured, eager, pages)


@pytest.mark.parametrize("draft_tokens", [None, 1, 4])
def test_standard_decode_replay_workspace_matches_recipe(draft_tokens):
    """Decode and MTP reserve the bytes used by their real replay buffers."""
    harness = _Harness(draft_tokens=draft_tokens, history_capacity=16)
    recipe = _kimi_recipe(
        max_bs=harness.backend.max_bs,
        speculative_algorithm=None if harness.width == 1 else "EAGLE3",
        speculative_num_draft_tokens=draft_tokens,
    )
    planned_bytes = recipe.setup().fixed_workspace_bytes
    allocated_bytes = harness.backend.preallocate_verify_workspace(
        harness.backend.max_bs, harness.width
    )
    assert planned_bytes == allocated_bytes
    _prepare_fixed_workspaces(
        server_args=recipe.server_args,
        config=harness.config,
        backend=harness.backend,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
        qsa_bf16_scratch=None,
        expected_bytes=planned_bytes,
    )


def test_replay_planning_matches_allocation_and_rejects_drift():
    """The cache recipe and startup allocator share one byte contract."""
    harness = _Harness(draft_tokens=T, history_capacity=16)
    recipe = _kimi_recipe(
        draft_layers=5,
        max_bs=8,
        speculative_algorithm="EAGLE3",
        speculative_num_draft_tokens=T,
    )
    planned_bytes = recipe.setup().fixed_workspace_bytes
    allocated_bytes = harness.backend.preallocate_verify_workspace(
        harness.backend.max_bs, T
    )
    assert planned_bytes == allocated_bytes
    server_args = SimpleNamespace(speculative_num_draft_tokens=T)
    config = SimpleNamespace(max_bs=8, max_resident_requests=8, qcp_size=1, kvp_size=1)
    backend = HybridLinearAttnBackend(
        AttentionBackend(harness.config, harness.config.components[0]),
        harness.backend,
        [],
    )

    _prepare_fixed_workspaces(
        server_args=server_args,
        config=config,
        backend=backend,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
        qsa_bf16_scratch=None,
        expected_bytes=planned_bytes,
    )

    with pytest.raises(
        RuntimeError,
        match="planned fixed workspace does not match allocated tensors",
    ):
        _prepare_fixed_workspaces(
            server_args=server_args,
            config=config,
            backend=backend,
            draft_backend=None,
            uses_paged_state_verify=True,
            is_inkling=False,
            qsa_bf16_scratch=None,
            expected_bytes=planned_bytes + 1,
        )


def test_equal_geometry_pool_replacement_rebinds_batched_replay():
    harness = _Harness(draft_tokens=T, history_capacity=16)
    pages = {group: [2] for group in _STATE_GROUPS}
    harness.prepare_metadata([0], pages, [8 + T])
    harness.forward(harness.inputs(1, 711), 1)

    replacement = _make_kimi_pool(DEV, usable_pages=24)
    # Prefill snapshots also refer to the old pool; publishing a replacement
    # must drop them before eager preparation or startup recapture rebuilds them.
    harness.backend._prefill_metadata[8, 1] = object()
    harness.backend._prefill_metadata_pool = harness.pool
    harness.backend.set_kv_pool(replacement)
    assert not harness.backend._prefill_metadata
    assert harness.backend._prefill_metadata_pool is None
    harness.pool = replacement
    harness.contract = replacement.arena.runtime_contract
    harness.prepare_metadata([0], pages, [8 + T])
    harness.forward(harness.inputs(1, 712), 1)


def test_uneven_state_groups_bind_batched_replay():
    """Replay uses each layer's cache group instead of equal-size partitions."""
    harness = _Harness(draft_tokens=T, history_capacity=16)
    first_group, second_group = _STATE_GROUPS[:2]
    moved_layer = next(
        layer_id
        for layer_id in harness.layer_ids
        if harness.pool.state_group_by_layer[layer_id] == second_group
    )
    harness.pool.state_group_by_layer[moved_layer] = first_group
    harness.backend.set_kv_pool(harness.pool)

    pages = {group: [2] for group in _STATE_GROUPS}
    harness.prepare_metadata([0], pages, [8 + T])
    harness.forward(harness.inputs(1, 713), 1)

    expected = [
        harness.backend._replay_group_rows[harness.pool.state_group_by_layer[layer_id]]
        for layer_id in harness.layer_ids
    ]
    assert harness.backend._replay_group_indices.tolist() == expected
    assert expected.count(0) != expected.count(1)


def test_rebinding_the_pool_reissues_the_retained_workspace():
    """Rebinding drops the previous records workspace and its references to the old pool."""
    harness = _Harness(draft_tokens=T, history_capacity=16)
    harness.backend.preallocate_verify_workspace(harness.config.max_bs, T)
    old_pool = storages_of(
        *(
            harness.backend._state_components(layer_id)[component]
            for layer_id in harness.layer_ids
            for component in (0, 1)
        )
    )
    replacement = _make_kimi_pool(DEV, usable_pages=24)

    harness.backend.set_kv_pool(replacement)
    assert harness.backend._retained_qkv is None
    harness.backend.preallocate_verify_workspace(harness.config.max_bs, T)
    assert_no_alias(harness.backend, old_pool)


def test_rebinding_the_pool_commits_like_a_fresh_bind():
    """Verify/commit after a rebind lands in the new pool exactly as a fresh bind's does."""
    rebound = _Harness(draft_tokens=T, history_capacity=16)
    fresh = _Harness(draft_tokens=T, history_capacity=16)
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    rebound.round(rpis, pages, seq_lens, 100, [1, T])
    old_slabs = storages_of(
        *(
            rebound.pool.get_component(layer, component)
            for layer in rebound.layer_ids
            for component in _CHECKPOINT_COMPONENTS
        )
    )

    replacement = _make_kimi_pool(DEV, usable_pages=24)
    rebound.backend.set_kv_pool(replacement)
    rebound.backend.init_cuda_graph_state(rebound.config.max_bs)
    rebound.backend.preallocate_verify_workspace(rebound.config.max_bs, rebound.width)
    assert_no_alias(rebound.backend, old_slabs)
    rebound.pool = replacement
    rebound.contract = replacement.arena.runtime_contract
    for round_index, accepted in enumerate(([0, T], [T, 2])):
        for harness in (rebound, fresh):
            harness.round(rpis, pages, seq_lens, 200 + round_index, accepted)
        _assert_committed_pages_equal(rebound, fresh, pages)
        seq_lens = [length + count for length, count in zip(seq_lens, accepted)]
    layer_id = fresh.layer_ids[0]
    page = pages[fresh.pool.state_group_by_layer[layer_id]][0]
    for harness in (rebound, fresh):
        for component in ("conv_state", "recurrent_state"):
            value = (
                _logical_state(harness.pool, layer_id, [page])[0]
                if component == "recurrent_state"
                else harness.pool.get_component(layer_id, component)[page]
            )
            assert value.abs().sum() > 0


def test_reordered_state_groups_are_the_same_geometry():
    """Group ids name the groups; the contract's enumeration order is not geometry."""
    harness = _Harness(draft_tokens=T, history_capacity=16)
    specs = harness.pool.arena.runtime_contract.group_specs
    reordered = SimpleNamespace(
        arena=SimpleNamespace(
            cache_group_specs=harness.pool.arena.cache_group_specs,
            runtime_contract=SimpleNamespace(group_specs=tuple(reversed(specs))),
        ),
        paged_group_ids=harness.pool.paged_group_ids,
        state_group_by_layer=harness.pool.state_group_by_layer,
        get_component=harness.pool.get_component,
    )

    harness.backend.validate_cache_pool(reordered)


def test_a_rejected_hybrid_rebind_moves_no_child():
    """The router accepts a pool the state backend rejects; two-phase binding must move nothing."""
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )

    harness = _Harness(draft_tokens=T, history_capacity=16)
    leaves = []

    def leaf_factory(group_id, granularity):
        leaf = SimpleNamespace(group_id=group_id, cache_pool=None)
        leaf.validate_cache_pool = lambda pool: None
        leaf.set_cache_pool = leaf._bind = lambda pool: setattr(
            leaf, "cache_pool", pool
        )
        leaves.append(leaf)
        return leaf

    router = CacheGroupRouter(
        leaf_factory,
        is_draft=False,
        spec_num_tokens=T,
        device=DEV,
        consumed_group_ids=None,
    )
    hybrid = HybridLinearAttnBackend(router, harness.backend, [])
    hybrid.set_cache_pool(harness.pool)
    assert leaves and all(leaf.cache_pool is harness.pool for leaf in leaves)

    def retyped(layer, name):
        component = harness.pool.get_component(layer, name)
        if name == "conv_state":
            return component.to(
                torch.float32 if component.dtype != torch.float32 else torch.float16
            )
        return component

    retyped_pool = SimpleNamespace(
        arena=harness.pool.arena,
        paged_group_ids=harness.pool.paged_group_ids,
        state_group_by_layer=harness.pool.state_group_by_layer,
        get_component=retyped,
    )
    router.validate_cache_pool(retyped_pool)
    with pytest.raises(RuntimeError, match="different state geometry"):
        hybrid.set_cache_pool(retyped_pool)

    assert hybrid.cache_pool is harness.pool
    assert router.cache_pool is harness.pool
    assert all(leaf.cache_pool is harness.pool for leaf in leaves)
    assert harness.backend.kv_pool is harness.pool


def test_a_rejected_hybrid_rebind_publishes_nothing():
    harness = _Harness(draft_tokens=T, history_capacity=16)
    full = SimpleNamespace(device=DEV, cache_pool=harness.pool)
    full.set_cache_pool = full._bind = lambda pool: setattr(full, "cache_pool", pool)
    full.validate_cache_pool = lambda pool: None
    hybrid = HybridLinearAttnBackend(full, harness.backend, [])
    hybrid.set_cache_pool(harness.pool)

    first_layer = harness.backend._state_layer_ids()[0]
    recurrent = harness.pool.get_component(first_layer, "recurrent_state")
    widened = torch.zeros(
        (recurrent.shape[0], recurrent.shape[1] * 2, *recurrent.shape[2:]),
        dtype=recurrent.dtype,
        device=recurrent.device,
    )

    def get_component(layer_id, name):
        if layer_id == first_layer and name == "recurrent_state":
            return widened
        return harness.pool.get_component(layer_id, name)

    coarser = SimpleNamespace(
        arena=harness.pool.arena,
        state_group_by_layer=harness.pool.state_group_by_layer,
        get_component=get_component,
    )
    with pytest.raises(RuntimeError, match="different state geometry"):
        hybrid.set_cache_pool(coarser)

    assert hybrid.cache_pool is harness.pool
    assert harness.backend.kv_pool is harness.pool


def test_retained_workspace_fits_the_graph_probe_pool():
    """Records workspace is independent of the small cache used during graph probing."""
    harness = _Harness(draft_tokens=T, history_capacity=16)
    harness.backend.set_kv_pool(_make_kimi_pool(DEV, usable_pages=4))

    allocated_bytes = harness.backend.preallocate_verify_workspace(
        harness.config.max_bs, T
    )
    recipe = _kimi_recipe(
        max_bs=harness.config.max_bs,
        speculative_algorithm="EAGLE3",
        speculative_num_draft_tokens=T,
    )
    assert allocated_bytes == recipe.workspace_bytes()


def test_hybrid_rebinding_reaches_the_state_child():
    harness = _Harness(draft_tokens=T, history_capacity=16)
    full = SimpleNamespace(device=DEV, cache_pool=None)
    full.set_cache_pool = full._bind = lambda pool: setattr(full, "cache_pool", pool)
    full.validate_cache_pool = lambda pool: None
    full.init_prefill_graph_state = lambda max_num_tokens, max_bs: None
    hybrid = HybridLinearAttnBackend(full, harness.backend, [])
    share = harness.backend.sparse_topk
    share.prefill = share.decode = object()

    old_slabs = storages_of(
        *(
            harness.backend._state_components(layer_id)[component]
            for layer_id in harness.backend._state_layer_ids()
            for component in (0, 1)
        )
    )
    replacement = _make_kimi_pool(DEV, usable_pages=24)
    hybrid.set_cache_pool(replacement)
    assert full.cache_pool is replacement
    assert harness.backend.kv_pool is replacement
    assert share.prefill is None and share.decode is None
    assert_no_alias(hybrid, old_slabs)
    assert_no_alias(harness.backend, old_slabs)


def test_padded_graph_verify_keeps_live_rows_and_commits_isolated():
    """Padding may have undefined outputs, but cannot change live state."""
    captured = _Harness(draft_tokens=T, seed=982, history_capacity=16)
    eager = _Harness(draft_tokens=T, seed=982, history_capacity=16)
    bs, real_bs = 4, 2
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * real_bs + i for i in range(real_bs)]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * real_bs
    captured.prepare_metadata(rpis, pages, seq_lens)
    captured.forward(captured.inputs(real_bs, 901), real_bs)
    capture_tables = {
        group_id: torch.full((bs, 2), -1, dtype=torch.int32, device=DEV)
        for group_id in _STATE_GROUPS
    }
    captured.backend.init_forward_metadata_capture_cuda_graph(
        bs,
        torch.arange(bs, dtype=torch.int32, device=DEV),
        torch.full((bs,), 8 + T, dtype=torch.int32, device=DEV),
        ForwardMode.DECODE,
        block_tables=capture_tables,
    )
    stable_inputs = captured.inputs(bs, 902)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        captured.forward(stable_inputs, bs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = captured.forward(stable_inputs, bs)

    for round_index, counts in enumerate(([0, T], [1, 2])):
        tables = {
            group_id: np.asarray(
                [[0, page] for page in pages[group_id]], dtype=np.int32
            )
            for group_id in _STATE_GROUPS
        }
        captured.backend.refresh_decode_metadata(
            bs,
            real_bs,
            torch.tensor(rpis, dtype=torch.int32, device=DEV),
            torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables=_tables_for(captured.contract, tables, DEV),
        )
        inputs = captured.inputs(bs, 903 + round_index)
        for name, value in inputs.items():
            stable_inputs[name].copy_(value)
        before = {
            (layer, component): captured.pool.get_component(layer, component).clone()
            for layer in captured.layer_ids
            for component in _CHECKPOINT_COMPONENTS
        }
        graph.replay()
        accepted = torch.tensor(counts, dtype=torch.int32, device=DEV)
        captured.backend.commit_verified_state(accepted, accepted_path=None)
        eager.prepare_metadata(rpis, pages, seq_lens)
        eager_outputs = eager.forward(
            {name: value[: real_bs * T] for name, value in inputs.items()}, real_bs
        )
        eager.backend.commit_verified_state(accepted, accepted_path=None)
        for actual, expected in zip(graph_outputs, eager_outputs, strict=True):
            torch.testing.assert_close(
                actual.reshape(-1, KEY_DIM)[: real_bs * T],
                expected.reshape(-1, KEY_DIM),
                atol=2e-2,
                rtol=2e-2,
            )
        # Groups share physical pages; guard every page outside all live destinations.
        destination_pages = [
            page for group_pages in pages.values() for page in group_pages
        ]
        for layer_id in captured.layer_ids:
            ids = pages[captured.pool.state_group_by_layer[layer_id]]
            torch.testing.assert_close(
                _logical_state(captured.pool, layer_id, ids),
                _logical_state(eager.pool, layer_id, ids),
                atol=2e-6,
                rtol=2e-5,
            )
            for component in _CHECKPOINT_COMPONENTS:
                committed = captured.pool.get_component(layer_id, component)
                untouched = torch.ones(committed.shape[0], device=DEV, dtype=torch.bool)
                untouched[destination_pages] = False
                torch.testing.assert_close(
                    committed[untouched],
                    before[layer_id, component][untouched],
                    atol=0,
                    rtol=0,
                )
                if component in ("conv_state", "recurrent_state", "replay_length"):
                    torch.testing.assert_close(
                        committed,
                        eager.pool.get_component(layer_id, component),
                        atol=1e-5 if component == "recurrent_state" else 0,
                        rtol=1e-3 if component == "recurrent_state" else 0,
                    )
        seq_lens = [length + count for length, count in zip(seq_lens, counts)]


def test_commit_plan_resolves_pages_when_packed_storage_moves():
    """Resolve the new scheduler pages after replacing packed table storage."""
    harness = _Harness(
        draft_tokens=T,
        seed=173,
        max_bs=2,
        usable_pages=24,
        history_capacity=16,
    )
    first_pages = {
        group: [1 + 2 * index, 2 + 2 * index]
        for index, group in enumerate(_STATE_GROUPS)
    }
    next_pages = {
        group: [7 + 2 * index, 8 + 2 * index]
        for index, group in enumerate(_STATE_GROUPS)
    }
    harness.prepare_metadata([0, 1], first_pages, [64 + T, 64 + T])
    # Retain the previous views so the allocator cannot reuse their address.
    _previous_tables = harness.backend._verify_commit_ctx[0]
    harness.prepare_metadata([0, 1], next_pages, [64 + T, 64 + T])

    accepted = torch.tensor([2, T], dtype=torch.int32, device=DEV)
    pages, steps = harness.backend._resolve_verify_commit_pages(
        accepted, harness.backend._replay_group_ids
    )
    expected = torch.tensor(
        [next_pages[group] for group in harness.backend._replay_group_ids],
        dtype=torch.int32,
        device=DEV,
    )
    torch.testing.assert_close(pages, expected, atol=0, rtol=0)
    torch.testing.assert_close(steps, accepted, atol=0, rtol=0)
    torch.testing.assert_close(
        harness.backend._replay_commit_reads[: expected.numel()].view_as(expected),
        expected,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("batch", [2, 5])
def test_commit_plan_queued_table_updates_keep_each_round_pages(batch):
    """Queued descriptor uploads preserve each round's source and live pages."""
    harness = _Harness(
        draft_tokens=T,
        seed=174,
        max_bs=batch,
        usable_pages=24,
        history_capacity=16,
    )
    group_ids = harness.backend._replay_group_ids
    request_indices = torch.arange(batch, dtype=torch.int32, device=DEV)
    seq_lens = torch.full((batch,), 64 + T, dtype=torch.int32, device=DEV)
    accepted = torch.arange(batch, dtype=torch.int32, device=DEV) % (T + 2)
    rounds = []
    for iteration in range(12):
        tables = {}
        expected_reads, expected_pages = [], []
        for group_index, group in enumerate(group_ids):
            rows, sources, destinations = [], [], []
            for row in range(batch):
                live = 1 + (iteration * 3 + group_index + row) % 20
                checkpoint = (
                    1 + (iteration + group_index * 7 + row) % 20
                    if (iteration + row) % 2
                    else 0
                )
                if iteration % 3 == 0 and row == batch - 1:
                    checkpoint = live = -1
                rows.append(
                    [checkpoint, live] + [-1] * (1 + (iteration + group_index) % 3)
                )
                sources.append(checkpoint if checkpoint > 0 else live)
                destinations.append(live)
            # Retain independently allocated, strided tables across rounds.
            tables[group] = torch.tensor(rows, dtype=torch.int32, device=DEV)[:, :2]
            expected_reads.append(sources)
            expected_pages.append(destinations)
        rounds.append((tables, expected_reads, expected_pages))

    def plan(tables):
        harness.backend.refresh_decode_metadata(
            batch,
            batch,
            request_indices,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables=tables,
        )
        return harness.backend._resolve_verify_commit_pages(accepted, group_ids)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Compile and warm the metadata kernels before queuing delayed updates.
        for tables, _, _ in rounds:
            plan(tables)
    stream.synchronize()

    snapshots = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        for tables, _, _ in rounds:
            pages, steps = plan(tables)
            reads = harness.backend._replay_commit_reads[: len(group_ids) * batch]
            snapshots.append(
                (pages.clone(), reads.view_as(pages).clone(), steps.clone())
            )
    torch.cuda.current_stream().wait_stream(stream)
    for (_, expected_reads, expected_pages), (pages, reads, steps) in zip(
        rounds, snapshots, strict=True
    ):
        torch.testing.assert_close(
            pages,
            torch.tensor(expected_pages, dtype=torch.int32, device=DEV),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            reads,
            torch.tensor(expected_reads, dtype=torch.int32, device=DEV),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(steps, accepted.clamp(1, T), atol=0, rtol=0)


@pytest.mark.parametrize("history_capacity", [16])
@pytest.mark.parametrize("draft_tokens", [None, 1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_decode_records_commit_after_acceptance(
    draft_tokens, captured, history_capacity
):
    """Verify preserves live state; acceptance commits only the selected prefixes."""
    actual = _Harness(
        draft_tokens=draft_tokens,
        seed=431,
        history_capacity=history_capacity,
    )
    eager = _Harness(
        draft_tokens=draft_tokens,
        seed=431,
        history_capacity=history_capacity,
    )
    width = actual.width
    recipe = kimi_recipe_with_history(
        history_capacity=history_capacity,
        max_bs=actual.backend.max_bs,
        speculative_num_draft_tokens=draft_tokens,
        speculative_algorithm=None if width == 1 else "EAGLE3",
    )
    planned_bytes = recipe.workspace_bytes()
    _prepare_fixed_workspaces(
        server_args=recipe.server_args,
        config=actual.config,
        backend=actual.backend,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
        qsa_bf16_scratch=None,
        expected_bytes=planned_bytes,
    )

    def workspace_tensors():
        backend = actual.backend
        return tuple(
            tensor
            for tensor in (
                backend._retained_qkv,
                backend._replay_descriptors,
                backend._replay_group_indices,
                backend._replay_commit_tables,
                backend._replay_commit_reads,
                backend._replay_commit_pages,
                backend._replay_commit_steps,
                backend._replay_record_metadata,
                backend._replay_record_stamps,
                *backend.query_start_loc_list,
            )
            if tensor is not None
        )

    # Retain the allocations so a replacement cannot reuse their addresses.
    initial_workspace = workspace_tensors()
    workspace_storage = [
        (tensor.data_ptr(), tensor.nbytes) for tensor in initial_workspace
    ]
    batch_sizes = (4, 8, 3, 4)
    live_counts = (2, 1, 3, 2)
    max_live = max(live_counts)
    all_pages = {
        group: [2 + index * max_live + row for row in range(max_live)]
        for index, group in enumerate(_STATE_GROUPS)
    }
    all_lengths = [8 + width] * max_live
    stable_inputs = actual.inputs(actual.config.max_bs, 432)
    # Compare accepted states from the same explicit gate in eager and graphs.
    stable_inputs["g_raw"] = stable_inputs["mixed_qkv"][:, : actual.key_dim]

    def assert_live_state_unchanged(snapshots):
        for (layer, component), snapshot in snapshots.items():
            # Verify may overwrite candidate records beyond the accepted length.
            if component in ("replay_correction", "replay_key", "replay_decay"):
                continue
            torch.testing.assert_close(
                actual.pool.get_component(layer, component), snapshot, atol=0, rtol=0
            )

    before = {
        (layer, component): actual.pool.get_component(layer, component).clone()
        for layer in actual.layer_ids
        for component in _CHECKPOINT_COMPONENTS
    }
    # Bind layer descriptors before capture, as serving warmup does.
    actual.prepare_metadata(list(range(max_live)), all_pages, all_lengths)
    actual.forward(actual.inputs(max_live, 433), max_live)
    assert_live_state_unchanged(before)
    graphs = {}
    if captured:
        stream = torch.cuda.Stream()
        for batch in dict.fromkeys(batch_sizes):
            actual.backend.init_forward_metadata_capture_cuda_graph(
                batch,
                torch.arange(batch, dtype=torch.int32, device=DEV),
                torch.full((batch,), 8 + width, dtype=torch.int32, device=DEV),
                ForwardMode.DECODE,
            )
            capture_inputs = {
                name: value[: batch * width] for name, value in stable_inputs.items()
            }
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual.forward(capture_inputs, batch)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            assert_live_state_unchanged(before)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = actual.forward(capture_inputs, batch)
            graphs[batch] = graph, outputs
            assert_live_state_unchanged(before)
    assert [
        (tensor.data_ptr(), tensor.nbytes) for tensor in workspace_tensors()
    ] == workspace_storage

    for round_index, (batch, live) in enumerate(
        zip(batch_sizes, live_counts, strict=True)
    ):
        if round_index:
            all_pages = {group: ids[::-1] for group, ids in all_pages.items()}
            all_lengths.reverse()
        pages = {group: ids[:live] for group, ids in all_pages.items()}
        lengths = all_lengths[:live]
        rpis = list(reversed(range(live)))
        inputs = actual.inputs(batch, 440 + round_index)
        inputs["g_raw"] = inputs["mixed_qkv"][:, : actual.key_dim]
        live_inputs = {name: value[: live * width] for name, value in inputs.items()}
        counts = [1 + (round_index + row) % width for row in range(live)]
        expected = _reference_committed_states(eager, pages, live_inputs, counts)
        tables = {
            group: np.asarray([[0, page] for page in ids], dtype=np.int32)
            for group, ids in pages.items()
        }
        actual.backend.refresh_decode_metadata(
            batch,
            live,
            torch.tensor(rpis, dtype=torch.int32, device=DEV),
            torch.tensor(lengths, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=captured,
            block_tables=_tables_for(actual.contract, tables, DEV),
        )
        before = {
            (layer, component): actual.pool.get_component(layer, component).clone()
            for layer in actual.layer_ids
            for component in _CHECKPOINT_COMPONENTS
        }
        for name, value in inputs.items():
            stable_inputs[name][: batch * width].copy_(value)
        first_outputs = None
        for _ in range(2):
            if captured:
                graph, outputs = graphs[batch]
                graph.replay()
            else:
                outputs = actual.forward(
                    {
                        name: value[: batch * width]
                        for name, value in stable_inputs.items()
                    },
                    batch,
                )
            assert_live_state_unchanged(before)
            live_outputs = [
                output.reshape(-1, KEY_DIM)[: live * width] for output in outputs
            ]
            if first_outputs is None:
                first_outputs = [output.clone() for output in live_outputs]
            else:
                for output, previous in zip(live_outputs, first_outputs, strict=True):
                    torch.testing.assert_close(output, previous, atol=0, rtol=0)

        eager.prepare_metadata(rpis, pages, lengths)
        eager_outputs = eager.forward(live_inputs, live)
        for output, reference in zip(outputs, eager_outputs, strict=True):
            torch.testing.assert_close(
                output.reshape(-1, KEY_DIM)[: live * width],
                reference.reshape(-1, KEY_DIM),
                atol=2e-2,
                rtol=2e-2,
            )
        accepted = torch.tensor(counts, dtype=torch.int32, device=DEV)
        for harness in (actual, eager):
            root = HybridLinearAttnBackend(
                AttentionBackend(harness.config, harness.config.components[0]),
                harness.backend,
                [],
            )
            root.commit_state_after_acceptance(
                accepted, num_extends=0, accepted_path=None
            )
        # State groups share physical planes, so a view includes the other
        # groups' pages. Exclude all accepted destinations from the guard.
        destination_pages = [
            page for group_pages in pages.values() for page in group_pages
        ]
        for layer, (state, conv) in expected.items():
            ids = pages[actual.pool.state_group_by_layer[layer]]
            torch.testing.assert_close(
                (
                    _logical_state(actual.pool, layer, ids)
                    if history_capacity
                    else actual.pool.get_component(layer, "recurrent_state")[ids]
                ),
                state,
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                actual.pool.get_component(layer, "conv_state")[ids],
                conv,
                atol=0,
                rtol=0,
            )
            for component in _CHECKPOINT_COMPONENTS:
                committed = actual.pool.get_component(layer, component)
                untouched = torch.ones(committed.shape[0], device=DEV, dtype=torch.bool)
                untouched[destination_pages] = False
                torch.testing.assert_close(
                    committed[untouched],
                    before[layer, component][untouched],
                    atol=0,
                    rtol=0,
                )
                if component in ("conv_state", "recurrent_state", "replay_length"):
                    torch.testing.assert_close(
                        committed,
                        eager.pool.get_component(layer, component),
                        atol=0,
                        rtol=0,
                    )
        assert [
            (tensor.data_ptr(), tensor.nbytes) for tensor in workspace_tensors()
        ] == workspace_storage
        assert (
            actual.backend.preallocate_verify_workspace(actual.config.max_bs, width)
            == planned_bytes
        )
        all_lengths[:live] = [length + count for length, count in zip(lengths, counts)]


@pytest.mark.parametrize("width", [1, 4])
def test_verify_convolution_bias_matches_torch(width):
    from tokenspeed_kernel.ops.attention.kda import kda_verify_conv_update

    torch.manual_seed(871)
    channels = CONV_DIM
    raw = torch.randn(2 * width, channels, device=DEV, dtype=torch.bfloat16) * 0.1
    state = torch.randn(3, channels, 3, device=DEV, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(channels, 4, device=DEV, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(channels, device=DEV, dtype=torch.bfloat16) * 0.1
    pages = torch.tensor([2, -1], device=DEV, dtype=torch.int32)
    history = torch.stack((state[2], torch.zeros_like(state[0]))).float()
    sequence = torch.cat(
        (history, raw.view(2, width, channels).transpose(1, 2).float()), dim=-1
    )
    expected = torch.nn.functional.conv1d(
        sequence, weight.float().unsqueeze(1), bias.float(), groups=channels
    )
    expected = torch.nn.functional.silu(expected).transpose(1, 2).reshape_as(raw)
    actual = kda_verify_conv_update(
        raw,
        weight,
        state,
        pages,
        num_heads=H,
        head_dim=D,
        draft_token_num=width,
        recurrent_layout="v_major",
        bias=bias,
        query_start_loc=torch.arange(3, device=DEV, dtype=torch.int32) * width,
    )
    torch.testing.assert_close(actual.float(), expected, atol=1e-4, rtol=8e-3)


@pytest.mark.parametrize("draft_tokens", [None, 4])
@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
def test_record_decode_preserves_raw_gate_bias_and_layer_subset(
    draft_tokens, lower_bound
):
    harness = _Harness(draft_tokens=draft_tokens, seed=901, history_capacity=16)
    if not harness.backend._verify_split_producers:
        pytest.skip("fused verify kernel does not support a conv bias")
    harness.layer_ids = harness.layer_ids[:1]
    for layer in harness.layer_ids:
        harness.params[layer]["bias"] = (
            torch.randn(CONV_DIM, device=DEV, dtype=torch.bfloat16) * 0.1
        )
        harness.params[layer]["f_b_weight"] = None
        harness.params[layer]["lower_bound"] = lower_bound
    pages = {group: [2 + i] for i, group in enumerate(_STATE_GROUPS)}
    for round_index in range(3):
        inputs = harness.inputs(1, 910 + round_index)
        inputs["g_raw"] = (
            torch.randn(harness.width, KEY_DIM, device=DEV, dtype=torch.bfloat16) * 0.1
        )
        count = max(1, harness.width - 1)
        reference = _reference_committed_states(harness, pages, inputs, [count])
        harness.prepare_metadata([0], pages, [8 + harness.width + round_index * count])
        harness.forward(inputs, 1)
        harness.backend.commit_state_after_acceptance(
            torch.tensor([count], device=DEV, dtype=torch.int32),
            num_extends=0,
            accepted_path=None,
        )
        for layer, (state, conv) in reference.items():
            ids = pages[harness.pool.state_group_by_layer[layer]]
            torch.testing.assert_close(
                _logical_state(harness.pool, layer, ids),
                state,
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                harness.pool.get_component(layer, "conv_state")[ids],
                conv,
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("same_page", [False, True])
@pytest.mark.parametrize("draft_tokens", [None, 4])
def test_mixed_decode_commit_matches_separate_forwards(
    draft_tokens, same_page, checkpoint
):
    history_capacity = 16

    # Prefill consumes the full cache field, including every per-rank head.
    # With ``checkpoint`` the extend row also stores its state 128 tokens in,
    # into a third block, as an EXTEND forward of it alone does.
    class _FullWidthHarness(_Harness):
        H = 12

    actual = _FullWidthHarness(
        draft_tokens=draft_tokens,
        seed=941,
        history_capacity=history_capacity,
    )
    width, prompt = actual.width, 200 if checkpoint else 3
    pages = {group: [2 + 2 * i, 3 + 2 * i] for i, group in enumerate(_STATE_GROUPS)}
    stores = {group: 8 + i for i, group in enumerate(_STATE_GROUPS)}
    sources = {
        group: ids[0] if same_page else ids[0] + 12 for group, ids in pages.items()
    }
    source_snapshots = {}
    if history_capacity:
        for layer in actual.layer_ids:
            source = sources[actual.pool.state_group_by_layer[layer]]
            for name, value in (("correction", 0.03), ("key", 0.02), ("decay", 0.9)):
                actual.pool.get_component(layer, f"replay_{name}")[source].fill_(value)
            actual.pool.get_component(layer, "replay_length")[source].fill_(3)
            source_snapshots[layer] = _logical_state(actual.pool, layer, [source])
    inputs = {
        name: value[: prompt + width]
        for name, value in actual.inputs(prompt + width, 943).items()
    }
    # Isolate mixed page routing from the inline gate producer's rounding.
    inputs["g_raw"] = inputs["mixed_qkv"][:, : actual.key_dim]

    def metadata(harness, *, mixed):
        bs = 2 if mixed else 1
        lengths = torch.tensor(
            [2 + prompt, 8 + width][:bs], device=DEV, dtype=torch.int32
        )
        new_cpu = torch.tensor([prompt], dtype=torch.int32)
        prefix_cpu = torch.full((1,), 2, dtype=torch.int32)
        tables = {
            group: np.asarray(
                [
                    [sources[group] if row == 0 else 0, page]
                    for row, page in enumerate(ids[:bs])
                ],
                dtype=np.int32,
            )
            for group, ids in pages.items()
        }
        if checkpoint:
            tables = {
                group: np.column_stack((rows, [stores[group], -1][:bs]))
                for group, rows in tables.items()
            }
        harness.backend.init_forward_metadata(
            bs,
            1,
            torch.arange(bs, device=DEV, dtype=torch.int32),
            lengths,
            ForwardMode.MIXED if mixed else ForwardMode.EXTEND,
            block_tables=_tables_for(harness.contract, tables, DEV),
            extend_seq_lens=new_cpu.to(DEV),
            extend_seq_lens_cpu=new_cpu,
            extend_prefix_lens=prefix_cpu.to(DEV),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_replay_lens_cpu=torch.zeros_like(prefix_cpu),
            extend_prompt_lens_cpu=prefix_cpu + new_cpu,
            extend_with_prefix=True,
            extend_state_checkpoint_lens_cpu=torch.tensor(
                [128 if checkpoint else 0], dtype=torch.int32
            ),
            query_shard=None,
        )

    def forward(harness, values, *, batch, mode):
        outputs = []
        for layer in harness.layer_ids:
            params = harness.params[layer]
            outputs.append(
                harness.backend.forward_extend(
                    None,
                    None,
                    None,
                    layer=None,
                    token_to_kv_pool=harness.pool,
                    bs=batch,
                    forward_mode=mode,
                    mixed_qkv=values["mixed_qkv"].clone(),
                    f_a_out=values["f_a_out"],
                    g_raw=values["g_raw"],
                    beta_raw=values["beta_raw"],
                    conv_weights=params["conv_weights"],
                    bias=None,
                    activation="silu",
                    key_dim=harness.key_dim,
                    value_dim=harness.key_dim,
                    attention_tp_size=1,
                    head_k_dim=D,
                    head_v_dim=D,
                    A_log=params["A_log"],
                    dt_bias=params["dt_bias"],
                    f_b_weight=params["f_b_weight"],
                    lower_bound=params["lower_bound"],
                    layer_id=layer,
                    seq_len=values["mixed_qkv"].shape[0],
                ).reshape(-1, harness.H, D)
            )
        return outputs

    count = max(1, width - 1)
    prompt_outputs, prompt_states = _reference_round(
        actual,
        {group: [source] for group, source in sources.items()},
        {name: value[:prompt] for name, value in inputs.items()},
        [prompt],
        prefill=True,
    )
    decode_outputs, decode_states = _reference_round(
        actual,
        {group: ids[1:] for group, ids in pages.items()},
        {name: value[prompt:] for name, value in inputs.items()},
        [count],
        prefill=False,
    )
    checkpoint_states = None
    if checkpoint:
        _, checkpoint_states = _reference_round(
            actual,
            {group: [source] for group, source in sources.items()},
            {name: value[:128] for name, value in inputs.items()},
            [128],
            prefill=True,
        )
    stored_before = {
        (layer, name): actual.pool.get_component(layer, name)[
            stores[actual.pool.state_group_by_layer[layer]]
        ].clone()
        for layer in actual.layer_ids
        for name in ("conv_state", "recurrent_state", "replay_length")
    }
    metadata(actual, mixed=True)
    actual_outputs = forward(actual, inputs, batch=2, mode=ForwardMode.MIXED)
    actual.backend.commit_state_after_acceptance(
        torch.tensor([1, count], device=DEV, dtype=torch.int32),
        num_extends=1,
        accepted_path=None,
    )
    for layer, output, prefill, decode in zip(
        actual.layer_ids, actual_outputs, prompt_outputs, decode_outputs, strict=True
    ):
        expected = torch.cat((prefill, decode.reshape(-1, actual.H, D)))
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
        store = stores[actual.pool.state_group_by_layer[layer]]
        for component, index in (("recurrent_state", 0), ("conv_state", 1)):
            expected_checkpoint = (
                checkpoint_states[layer][index][0]
                if checkpoint
                else stored_before[layer, component]
            )
            torch.testing.assert_close(
                actual.pool.get_component(layer, component)[store],
                expected_checkpoint,
                atol=2e-6 if index == 0 else 0,
                rtol=2e-5 if index == 0 else 0,
            )
        expected_length = 0 if checkpoint else stored_before[layer, "replay_length"]
        assert (
            actual.pool.get_component(layer, "replay_length")[store] == expected_length
        )
        ids = pages[actual.pool.state_group_by_layer[layer]]
        torch.testing.assert_close(
            actual.pool.get_component(layer, "conv_state")[ids],
            torch.cat((prompt_states[layer][1], decode_states[layer][1])),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            (
                _logical_state(actual.pool, layer, ids)
                if history_capacity
                else actual.pool.get_component(layer, "recurrent_state")[ids]
            ),
            torch.cat((prompt_states[layer][0], decode_states[layer][0])),
            atol=2e-6,
            rtol=2e-5,
        )
        if history_capacity:
            assert actual.pool.get_component(layer, "replay_length")[ids[0]].item() == 0
            if not same_page:
                source = sources[actual.pool.state_group_by_layer[layer]]
                torch.testing.assert_close(
                    _logical_state(actual.pool, layer, [source]),
                    source_snapshots[layer],
                    atol=0,
                    rtol=0,
                )


def _logical_state(pool, layer, ids):
    state = pool.get_component(layer, "recurrent_state")[ids].float().clone()
    fields = {
        name: pool.get_component(layer, f"replay_{name}")[ids]
        for name in ("correction", "key", "decay", "length")
    }
    for row, length in enumerate(fields["length"].flatten().tolist()):
        for index in range(length):
            state[row] = (
                state[row] * fields["decay"][row, :, index, None, :]
                + fields["correction"][row, :, index, :, None]
                * fields["key"][row, :, index, None, :]
            )
    return state


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_retained_rounds_reconstruct_endpoint_and_reuse_rejected(width, captured):
    actual = _Harness(draft_tokens=width, seed=1201, history_capacity=16)
    untouched_layer = actual.layer_ids[1]
    untouched = {
        name: actual.pool.get_component(untouched_layer, name).clone()
        for name in ("conv_state", "replay_length")
    }
    actual.layer_ids = actual.layer_ids[:1]
    pages = {group: [2] for group in _STATE_GROUPS}
    layer = actual.layer_ids[0]
    for name, value in (("correction", 0.03), ("key", 0.02), ("decay", 0.9)):
        actual.pool.get_component(layer, f"replay_{name}")[2].fill_(value)
    actual.pool.get_component(layer, "replay_length")[2].fill_(3)
    actual.prepare_metadata([0], pages, [8 + width])
    inputs = actual.inputs(1, 1202)
    # The oracle checks history reuse using an identical raw gate.
    inputs["g_raw"] = inputs["mixed_qkv"][:, : actual.key_dim]
    actual.forward(inputs, 1)
    graph = None
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = actual.forward(inputs, 1)
    for iteration in range(20):
        values = actual.inputs(1, 1210 + iteration)
        values["g_raw"] = values["mixed_qkv"][:, : actual.key_dim]
        for name in inputs:
            inputs[name].copy_(values[name])
        actual.prepare_metadata([0], pages, [8 + width + iteration])
        count = width if iteration % 2 else 1
        expected_output, expected = _reference_round(
            actual, pages, values, [count], prefill=False
        )
        if graph is None:
            output = actual.forward(inputs, 1)
        else:
            graph.replay()
        torch.testing.assert_close(
            output[0].reshape(-1, actual.H, D),
            expected_output[0],
            atol=2e-3,
            rtol=2e-3,
        )
        # Alternate full and partial acceptance; next verify overwrites rejected slots.
        accepted = torch.tensor(
            [width if iteration % 2 else 1], device=DEV, dtype=torch.int32
        )
        actual.backend.commit_verified_state(accepted, accepted_path=None)
        for name, snapshot in untouched.items():
            torch.testing.assert_close(
                actual.pool.get_component(untouched_layer, name),
                snapshot,
                atol=0,
                rtol=0,
            )
        layer = actual.layer_ids[0]
        torch.testing.assert_close(
            _logical_state(actual.pool, layer, [2]),
            expected[layer][0],
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            actual.pool.get_component(layer, "conv_state")[[2]],
            expected[layer][1],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("same_page", [False, True])
@pytest.mark.parametrize("mamba_ssm_dtype", ["float32", "bfloat16"])
def test_retained_prefill_gathers_endpoint_and_resets_final_length(
    same_page, mamba_ssm_dtype
):
    harness = _Harness(
        draft_tokens=4, history_capacity=16, mamba_ssm_dtype=mamba_ssm_dtype
    )
    layer = harness.layer_ids[0]
    source, destination = 2, 2 if same_page else 3
    pool = harness.pool
    pool.get_component(layer, "recurrent_state")[source].fill_(0.1)
    for name, value in (("correction", 0.03), ("key", 0.02), ("decay", 0.9)):
        pool.get_component(layer, f"replay_{name}")[source].fill_(value)
    pool.get_component(layer, "replay_length")[source].fill_(3)
    expected = _logical_state(pool, layer, [source])
    snapshots = {
        name: pool.get_component(layer, name)[source].clone()
        for name in (
            "recurrent_state",
            "replay_correction",
            "replay_key",
            "replay_decay",
            "replay_length",
        )
    }
    reads = torch.tensor([source], device=DEV, dtype=torch.int32)
    writes = torch.tensor([destination], device=DEV, dtype=torch.int32)
    conv, recurrent = harness.backend._state_components(layer)
    gathered, _ = harness.backend._prepare_prefill_state_inputs(
        layer, conv, recurrent, reads, writes, dtype=torch.float32
    )
    torch.testing.assert_close(gathered, expected, atol=2e-7, rtol=2e-6)
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(
            pool.get_component(layer, name)[source], snapshot, atol=0, rtol=0
        )
    final = gathered + 0.25
    harness.backend._store_prefill_state(layer, final, recurrent, writes)
    torch.testing.assert_close(
        recurrent[destination], final[0].to(recurrent.dtype), atol=0, rtol=0
    )
    assert pool.get_component(layer, "replay_length")[destination].item() == 0
    if not same_page:
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(
                pool.get_component(layer, name)[source], snapshot, atol=0, rtol=0
            )


def test_retained_workspace_matches_recipe_and_rebind_invalidates_history():
    harness = _Harness(draft_tokens=4, history_capacity=16)
    recipe = kimi_recipe_with_history(
        max_bs=8, speculative_num_draft_tokens=4, history_capacity=16
    )
    assert (
        harness.backend.preallocate_verify_workspace(8, 4) == recipe.workspace_bytes()
    )
    assert harness.backend._retained_qkv.shape == (69, 32, CONV_DIM)
    pages = {group: [2] for group in _STATE_GROUPS}
    harness.layer_ids = harness.layer_ids[:1]
    harness.prepare_metadata([0], pages, [12])
    harness.forward(harness.inputs(1, 1301), 1)
    old_history = harness.backend._retained_histories[harness.layer_ids[0], 1]
    harness.prepare_metadata([0], pages, [13])
    harness.forward(harness.inputs(1, 1302), 1)
    assert harness.backend._retained_histories[harness.layer_ids[0], 1] is old_history
    # Rebind while the cached history still refers to the old pool.
    old_arena = harness.pool.arena.buffer.clone()
    replacement = make_kimi_pool_with_history(DEV, usable_pages=26, history_capacity=16)
    harness.backend.set_kv_pool(replacement)
    assert not harness.backend._retained_histories
    old_pool = harness.pool
    harness.pool = replacement
    harness.contract = replacement.arena.runtime_contract
    harness.backend.init_cuda_graph_state(8)
    assert (
        harness.backend.preallocate_verify_workspace(8, 4) == recipe.workspace_bytes()
    )
    inputs = harness.inputs(1, 1303)
    inputs["g_raw"] = inputs["mixed_qkv"][:, : harness.key_dim]
    expected = _reference_committed_states(harness, pages, inputs, [3])
    harness.prepare_metadata([0], pages, [12])
    harness.forward(inputs, 1)
    harness.backend.commit_verified_state(
        torch.tensor([3], device=DEV, dtype=torch.int32), accepted_path=None
    )
    layer = harness.layer_ids[0]
    torch.testing.assert_close(
        _logical_state(replacement, layer, [2]),
        expected[layer][0],
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        replacement.get_component(layer, "conv_state")[[2]],
        expected[layer][1],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(old_pool.arena.buffer, old_arena, atol=0, rtol=0)
    # Rebuilding graph metadata must also release the cached history views.
    harness.backend.init_cuda_graph_state(8)
    assert not harness.backend._retained_histories
    with pytest.raises((ValueError, RuntimeError), match="retained|history"):
        harness.backend.set_kv_pool(
            make_kimi_pool_with_history(DEV, history_capacity=8)
        )


@pytest.mark.parametrize("same_page", [False, True])
@pytest.mark.parametrize("initial_length", [12, 13])
def test_retained_source_live_capacity_boundary(initial_length, same_page):
    actual = _Harness(draft_tokens=4, seed=1401, history_capacity=16)
    actual.layer_ids = actual.layer_ids[:1]
    layer = actual.layer_ids[0]
    for name, value in (("correction", 0.03), ("key", 0.02), ("decay", 0.9)):
        actual.pool.get_component(layer, f"replay_{name}")[1].fill_(value)
    actual.pool.get_component(layer, "replay_length")[1].fill_(initial_length)
    logical_source = _logical_state(actual.pool, layer, [1])
    before = {
        name: actual.pool.get_component(layer, name)[1].clone()
        for name in (
            "conv_state",
            "recurrent_state",
            "replay_correction",
            "replay_key",
            "replay_decay",
            "replay_length",
        )
    }
    destinations = [1] if same_page else [2, 3]
    batch = len(destinations)
    accepted = [4] if same_page else [1, 4]
    tables = {
        group: np.asarray([[1, page] for page in destinations], dtype=np.int32)
        for group in _STATE_GROUPS
    }
    inputs = actual.inputs(batch, 1402)
    # Check consolidation and copy-on-write independently of gate projection.
    inputs["g_raw"] = inputs["mixed_qkv"][:, : actual.key_dim]
    expected_output, expected = _reference_round(
        actual,
        {group: [1] * batch for group in _STATE_GROUPS},
        inputs,
        accepted,
        prefill=False,
    )
    actual.backend.refresh_decode_metadata(
        batch,
        batch,
        torch.arange(batch, device=DEV, dtype=torch.int32),
        torch.full((batch,), 20, device=DEV, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables_for(actual.contract, tables, DEV),
    )
    output = actual.forward(inputs, batch)[0]
    torch.testing.assert_close(
        output.reshape_as(expected_output[0]), expected_output[0], atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        _logical_state(actual.pool, layer, destinations),
        logical_source.expand(batch, -1, -1, -1),
        atol=2e-7,
        rtol=2e-6,
    )
    retained_before_commit = 12 if same_page and initial_length == 12 else 0
    assert (
        actual.pool.get_component(layer, "replay_length")[destinations]
        .flatten()
        .tolist()
        == [retained_before_commit] * batch
    )
    actual.backend.commit_verified_state(
        torch.tensor(accepted, device=DEV, dtype=torch.int32), accepted_path=None
    )
    assert actual.pool.get_component(layer, "replay_length")[
        destinations
    ].flatten().tolist() == [retained_before_commit + count for count in accepted]
    torch.testing.assert_close(
        _logical_state(actual.pool, layer, destinations),
        expected[layer][0],
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        actual.pool.get_component(layer, "conv_state")[destinations],
        expected[layer][1],
        atol=0,
        rtol=0,
    )
    if not same_page:
        for name, snapshot in before.items():
            torch.testing.assert_close(
                actual.pool.get_component(layer, name)[1], snapshot, atol=0, rtol=0
            )


def test_retained_nonsquare_pool_is_rejected_before_publication():
    harness = _Harness(draft_tokens=4, history_capacity=16)
    malformed = make_kimi_pool_with_history(DEV, history_capacity=16)
    for layer in malformed.state_group_by_layer:
        conv, state = malformed._state_buffers_by_layer[layer]
        malformed._state_buffers_by_layer[layer] = (conv, state[..., :64])
        malformed._replay_key[layer] = malformed._replay_key[layer][..., :64]
        malformed._replay_decay[layer] = malformed._replay_decay[layer][..., :64]
    fresh = KdaAttnBackend(
        harness.config,
        harness.config.components[0],
        enable_prefill_graph=False,
        kda_backend="auto",
    )
    with pytest.raises((ValueError, RuntimeError), match="geometry"):
        fresh.set_kv_pool(malformed)
    assert fresh.kv_pool is None
    old_pool = harness.backend.kv_pool
    with pytest.raises((ValueError, RuntimeError), match="geometry"):
        harness.backend.set_kv_pool(malformed)
    assert harness.backend.kv_pool is old_pool


def test_dense_only_kda_pool_rejected_before_publication():
    pool = make_kimi_pool_with_history(DEV, history_capacity=0)
    spec = SoftmaxAttnConfig(
        num_attention_heads=H, num_kv_heads=H, attn_tp_size=1, head_dim=D
    )
    config = AttnConfig(
        device=DEV,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_quant_method=None,
        context_len=4096,
        max_bs=8,
        max_resident_requests=8,
        components=(spec,),
    )
    backend = KdaAttnBackend(
        config, spec, enable_prefill_graph=False, kda_backend="auto"
    )
    with pytest.raises(ValueError, match="retained"):
        backend.set_kv_pool(pool)
    assert backend.kv_pool is None


@pytest.mark.parametrize(
    "invalid", ["conv_dtype", "conv_stride", "missing_history", "draft"]
)
def test_unsupported_retained_pool_rejected_before_publication(invalid):
    from dataclasses import replace

    harness = _Harness(draft_tokens=4, history_capacity=16)
    pool = make_kimi_pool_with_history(DEV, history_capacity=16)
    layer = min(pool.state_group_by_layer)
    conv, recurrent = pool.get_state_buffers(layer)
    if invalid == "conv_dtype":
        pool._state_buffers_by_layer[layer] = (conv.float(), recurrent)
    elif invalid == "conv_stride":
        strided = torch.empty((*conv.shape, 2), device=DEV, dtype=conv.dtype)[..., 0]
        pool._state_buffers_by_layer[layer] = (strided, recurrent)
    elif invalid == "missing_history":
        pool._replay_decay[layer] = None
    config = replace(harness.config, is_draft=invalid == "draft")
    if invalid == "draft":
        with pytest.raises(ValueError, match="KDA.*draft|draft.*KDA"):
            KdaAttnBackend(
                config,
                config.components[0],
                enable_prefill_graph=False,
                kda_backend="auto",
            )
        return
    backend = KdaAttnBackend(
        config, config.components[0], enable_prefill_graph=False, kda_backend="auto"
    )
    with pytest.raises(ValueError, match="retained"):
        backend.set_kv_pool(pool)
    assert backend.kv_pool is None


@pytest.mark.parametrize("draft_tokens", [None, T])
def test_verify_fuses_producers_and_commits_like_split(monkeypatch, draft_tokens):
    """Verify projects the gate and convolves inside its kernel, as split would."""
    import tokenspeed.runtime.layers.attention.backends.state.kda as kda_backend

    fused = _Harness(draft_tokens=draft_tokens, history_capacity=16, seed=1201)
    split = _Harness(draft_tokens=draft_tokens, history_capacity=16, seed=1201)
    if not fused.backend._verify_split_producers:
        pytest.skip("split verify producers unavailable")
    convolutions = []
    conv_update = kda_backend.kda_verify_conv_update

    def counted_conv_update(*args, **kwargs):
        convolutions.append(kwargs["draft_token_num"])
        return conv_update(*args, **kwargs)

    monkeypatch.setattr(kda_backend, "kda_verify_conv_update", counted_conv_update)
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    width = fused.width
    seq_lens = [8 + width] * len(rpis)
    accepted = [1, width]
    for round_index in range(3):
        outputs = {}
        for name, harness in (("fused", fused), ("split", split)):
            with monkeypatch.context() as patch:
                if name == "split":
                    patch.setattr(
                        kda_backend,
                        "kda_fused_paged_verify_fuses_producers",
                        lambda *args, **kwargs: False,
                    )
                harness.prepare_metadata(rpis, pages, seq_lens)
                outputs[name] = harness.forward(
                    harness.inputs(len(rpis), 1300 + round_index), len(rpis)
                )
                harness.backend.commit_verified_state(
                    torch.tensor(accepted, dtype=torch.int32, device=DEV),
                    accepted_path=None,
                )
        for got, want in zip(outputs["fused"], outputs["split"]):
            # The gate projection may round one raw gate a BF16 step apart.
            torch.testing.assert_close(
                got.float(),
                want.float(),
                atol=1e-3 * want.float().abs().max().item(),
                rtol=2.0**-7,
            )
        seq_lens = [length + count for length, count in zip(seq_lens, accepted)]
    # Only the split harness ran the separate convolution producer.
    assert convolutions == [width] * (3 * len(split.layer_ids))
    _assert_committed_pages_equal(fused, split, pages)
