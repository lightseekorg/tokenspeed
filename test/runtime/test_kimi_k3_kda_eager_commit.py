"""Retained KDA runtime commitment, graph replay and PyTorch references."""

import gc
import os
import subprocess
import sys
from pathlib import Path

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
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from types import SimpleNamespace  # noqa: E402

from tokenspeed_kernel.ops.kvcache.triton import store_state_rows

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    _packed_qkv_views,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.registry import _prepare_fixed_workspaces

_LOWER_BOUND = -5.0
H, D, D_FA = 12, 128, 128
KEY_DIM = H * D
CONV_DIM = 3 * KEY_DIM
T = 3
DEV = "cuda"
_CHECKPOINT_COMPONENTS = ("conv_state", "recurrent_state")


class _Harness:
    H = H

    def __init__(
        self,
        *,
        draft_tokens: int | None,
        seed: int = 0,
        max_bs: int = 8,
        usable_pages: int = 24,
        max_resident_requests: int | None = None,
        mamba_ssm_dtype: str = "float32",
        history_capacity: int | None = None,
        pool=None,
    ):
        torch.manual_seed(seed)
        self.pool = pool or _make_kimi_pool(
            DEV,
            usable_pages=usable_pages,
            mamba_ssm_dtype=mamba_ssm_dtype,
        )
        # The arena's memory saver (a no-op unless the arena enables it). As
        # in serving, the backend's state lives in the restorable
        # "attention_state" region and the executor's in "runtime_states".
        saver = self.pool.arena.memory_saver_adapter
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
        components = (spec,)
        if history_capacity is not None:
            # --kda-history-capacity reaches the backend through this component.
            components += (
                LinearAttnConfig(
                    num_k_heads=self.H,
                    num_v_heads=self.H,
                    head_k_dim=D,
                    head_v_dim=D,
                    conv_kernel_size=4,
                    layer_ids=tuple(sorted(self.pool.state_group_by_layer)),
                    tp_size=1,
                    kda_history_capacity=history_capacity,
                ),
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
            max_resident_requests=(
                max_bs if max_resident_requests is None else max_resident_requests
            ),
            components=components,
        )
        self.config = config
        with saver.named_region("attention_state"):
            self.backend = KdaAttnBackend(
                config, spec, enable_prefill_graph=False, kda_backend="auto"
            )
            self.backend.set_kv_pool(self.pool)
        with saver.named_region("runtime_states"):
            # RuntimeStates.stopped, as the executor binds it: no request ends
            # unless a test marks its slot.
            self.stopped = torch.zeros(
                RequestSlotLayout(config.max_resident_requests).capacity,
                dtype=torch.bool,
                device=DEV,
            )
            # The executor's per-row request ends: 0 for every row here, raised
            # to REQUEST_ENDED_INCOMPLETE by a commit that could not complete a
            # state.
            self.request_ends = torch.zeros(
                config.max_bs, dtype=torch.int32, device=DEV
            )
        self.backend.bind_request_stop_flags(self.stopped, self.request_ends)
        # The persistent decode buffers exist from construction, as at the
        # wrapper (the verify refresh below writes into them).
        with saver.named_region("attention_state"):
            self.backend.init_cuda_graph_state(config.max_bs)
            self.backend.preallocate_verify_workspace(config.max_bs, self.width)
        self.layer_ids = list(self.backend._state_layer_ids())
        with saver.named_region("runtime_states"):
            self.params = self._layer_params()

    def _layer_params(self):
        return {
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


def _assert_committed_pages_equal(left, right, pages, slots):
    for layer in left.layer_ids:
        ids = pages[left.pool.state_group_by_layer[layer]]
        torch.testing.assert_close(
            left.pool.get_component(layer, "conv_state")[ids],
            right.pool.get_component(layer, "conv_state")[ids],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            _logical_state(left, layer, ids, slots),
            _logical_state(right, layer, ids, slots),
            atol=1e-5,
            rtol=1e-3,
        )


@pytest.mark.parametrize("draft_tokens", [1, 4])
def test_preconv_ring_holds_each_committed_convolution_window(draft_tokens):
    """Between materializations, the three ring rows before a request's head
    are its last accepted raw inputs: every layer's live page holds them as
    its convolution state after each commit, as the ring head wraps."""
    harness = _Harness(draft_tokens=draft_tokens, seed=517)
    width, rings = harness.width, harness.backend._preconv_ring
    rows = width + 3
    rpis = [3, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + width] * len(rpis)
    state = harness.backend._replay_state
    # A fresh slot's ring holds nothing; its first prepare seeds it.
    assert state[rpis, 1].tolist() == [0, 0]
    heads = [3, 3]
    for round_index in range(5):
        accepted = [1 + (round_index + row) % width for row in range(len(rpis))]
        harness.round(rpis, pages, seq_lens, 520 + round_index, accepted)
        heads = [(head + take) % rows for head, take in zip(heads, accepted)]
        assert state[rpis, 1].tolist() == [head + 1 for head in heads]
        for layer_id in harness.layer_ids:
            ring = rings[:, harness.backend._descriptor_row_by_layer[layer_id]]
            conv = harness.pool.get_component(layer_id, "conv_state")
            group = harness.pool.state_group_by_layer[layer_id]
            for row, slot in enumerate(rpis):
                past = [(heads[row] - 3 + i) % rows for i in range(3)]
                torch.testing.assert_close(
                    ring[slot, past].T, conv[pages[group][row]], atol=0, rtol=0
                )
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]


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


def _reference_round(harness, pages, inputs, accepted, *, slots, prefill, layers=None):
    """The torch reference of a round for ``layers``, by default every layer
    the harness forwards."""
    from test.runtime.kda_torch_reference import (
        checked_convolution,
        kda_prefill_reference,
        kda_torch_reference,
    )

    batch = len(accepted)
    states, outputs = {}, []
    for layer in harness.layer_ids if layers is None else layers:
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
            _logical_state(harness, layer, ids, slots),
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


def _reference_committed_states(harness, pages, inputs, accepted, slots, layers=None):
    return _reference_round(
        harness, pages, inputs, accepted, slots=slots, prefill=False, layers=layers
    )[1]


@pytest.mark.parametrize(
    "history_capacity", [None, T, 2 * T], ids=["default", "window", "two_windows"]
)
def test_retained_commit_matches_torch_over_multiple_rounds_and_layers(
    history_capacity,
):
    """All layer groups commit the accepted prefix of the independent recurrence.

    A capacity of one window folds every round that accepts tokens; two
    windows fold after a round of full acceptance.
    """
    harness = _Harness(draft_tokens=T, seed=17, history_capacity=history_capacity)
    expected_capacity = 16 if history_capacity is None else history_capacity
    assert harness.backend._retained_records.shape[4] == expected_capacity
    rpis = [3, 1, 2]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    accepts = ([0, 1, T], [T, 0, 2], [1, T, 0], [2, 1, T])
    counts = [0] * len(rpis)
    for round_index, accepted in enumerate(accepts):
        inputs = harness.inputs(len(rpis), 100 + round_index)
        # Keep gate-projection rounding outside the accepted-state check.
        inputs["g_raw"] = inputs["mixed_qkv"][:, : harness.key_dim]
        expected = _reference_committed_states(harness, pages, inputs, accepted, rpis)
        harness.prepare_metadata(rpis, pages, seq_lens)
        harness.forward(inputs, len(rpis))
        harness.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV),
            accepted_path=None,
        )
        # Each request keeps one count for every layer: a window that would
        # overflow folds the records first, and a round accepts one token at least.
        counts = [
            (0 if count + T > expected_capacity else count) + max(1, accept)
            for count, accept in zip(counts, accepted)
        ]
        assert harness.backend._replay_length[rpis].flatten().tolist() == counts
        for layer_id, (state, conv) in expected.items():
            ids = pages[harness.pool.state_group_by_layer[layer_id]]
            torch.testing.assert_close(
                _logical_state(harness, layer_id, ids, rpis),
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
    captured = _Harness(draft_tokens=T, seed=29)
    eager = _Harness(draft_tokens=T, seed=29)
    rpis = [2, 1]
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
    # The graph writes each request's raw rows into its slot's ring.
    payload = captured.backend._preconv_ring[
        :, captured.backend._descriptor_row_by_layer[first_layer]
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
        _assert_committed_pages_equal(captured, eager, pages, rpis)
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
        _assert_committed_pages_equal(captured, eager, pages, rpis)


@pytest.mark.parametrize("draft_tokens", [None, 1, 4])
def test_standard_decode_replay_workspace_matches_recipe(draft_tokens):
    """Decode and MTP reserve the bytes used by their real replay buffers."""
    harness = _Harness(draft_tokens=draft_tokens)
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
    harness = _Harness(draft_tokens=T)
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
    harness = _Harness(draft_tokens=T)
    pages = {group: [2] for group in _STATE_GROUPS}
    harness.prepare_metadata([1], pages, [8 + T])
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
    harness.prepare_metadata([1], pages, [8 + T])
    harness.forward(harness.inputs(1, 712), 1)


def test_uneven_state_groups_bind_batched_replay():
    """Replay uses each layer's cache group instead of equal-size partitions."""
    harness = _Harness(draft_tokens=T)
    first_group, second_group = _STATE_GROUPS[:2]
    moved_layer = next(
        layer_id
        for layer_id in harness.layer_ids
        if harness.pool.state_group_by_layer[layer_id] == second_group
    )
    harness.pool.state_group_by_layer[moved_layer] = first_group
    harness.backend.set_kv_pool(harness.pool)

    pages = {group: [2] for group in _STATE_GROUPS}
    harness.prepare_metadata([1], pages, [8 + T])
    harness.forward(harness.inputs(1, 713), 1)

    expected = [
        harness.backend._replay_group_rows[harness.pool.state_group_by_layer[layer_id]]
        for layer_id in harness.layer_ids
    ]
    assert harness.backend._replay_group_indices.tolist() == expected
    assert expected.count(0) != expected.count(1)


def test_rebinding_the_pool_reissues_the_retained_workspace():
    """Rebinding drops the previous records workspace and its references to the old pool."""
    harness = _Harness(draft_tokens=T)
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
    assert harness.backend._preconv_ring is None
    harness.backend.preallocate_verify_workspace(harness.config.max_bs, T)
    assert_no_alias(harness.backend, old_pool)


def test_rebinding_the_pool_commits_like_a_fresh_bind():
    """Verify/commit after a rebind lands in the new pool exactly as a fresh bind's does."""
    rebound = _Harness(draft_tokens=T)
    fresh = _Harness(draft_tokens=T)
    rpis = [2, 1]
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
        _assert_committed_pages_equal(rebound, fresh, pages, rpis)
        seq_lens = [length + count for length, count in zip(seq_lens, accepted)]
    layer_id = fresh.layer_ids[0]
    page = pages[fresh.pool.state_group_by_layer[layer_id]][0]
    for harness in (rebound, fresh):
        for component in ("conv_state", "recurrent_state"):
            value = (
                _logical_state(harness, layer_id, [page], rpis[:1])[0]
                if component == "recurrent_state"
                else harness.pool.get_component(layer_id, component)[page]
            )
            assert value.abs().sum() > 0


def test_reordered_state_groups_are_the_same_geometry():
    """Group ids name the groups; the contract's enumeration order is not geometry."""
    harness = _Harness(draft_tokens=T)
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

    harness = _Harness(draft_tokens=T)
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
    harness = _Harness(draft_tokens=T)
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
    harness = _Harness(draft_tokens=T)
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
    harness = _Harness(draft_tokens=T)
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
    captured = _Harness(draft_tokens=T, seed=982)
    eager = _Harness(draft_tokens=T, seed=982)
    bs, real_bs = 4, 2
    rpis = [2, 1]
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
        torch.arange(1, bs + 1, dtype=torch.int32, device=DEV),
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
                _logical_state(captured, layer_id, ids, rpis),
                _logical_state(eager, layer_id, ids, rpis),
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
                torch.testing.assert_close(
                    committed,
                    eager.pool.get_component(layer_id, component),
                    atol=1e-5 if component == "recurrent_state" else 0,
                    rtol=1e-3 if component == "recurrent_state" else 0,
                )
        # Padding rows report nothing and leave every count alone; the live
        # rows' counts follow the eager forward's: one accepted token at
        # least, every round.
        assert not captured.request_ends.any()
        assert torch.equal(
            captured.backend._replay_length, eager.backend._replay_length
        )
        assert captured.backend._replay_length[rpis].flatten().tolist() == [
            sum(max(1, accepts[row]) for accepts in ([0, T], [1, 2])[: round_index + 1])
            for row in range(real_bs)
        ]
        seq_lens = [length + count for length, count in zip(seq_lens, counts)]


def test_commit_plan_resolves_pages_when_packed_storage_moves():
    """Resolve the new scheduler pages after replacing packed table storage."""
    harness = _Harness(
        draft_tokens=T,
        seed=173,
        max_bs=2,
        usable_pages=24,
    )
    first_pages = {
        group: [1 + 2 * index, 2 + 2 * index]
        for index, group in enumerate(_STATE_GROUPS)
    }
    next_pages = {
        group: [7 + 2 * index, 8 + 2 * index]
        for index, group in enumerate(_STATE_GROUPS)
    }
    harness.prepare_metadata([1, 2], first_pages, [64 + T, 64 + T])
    # Retain the previous views so the allocator cannot reuse their address.
    _previous_tables = harness.backend._verify_commit_ctx[0]
    harness.prepare_metadata([1, 2], next_pages, [64 + T, 64 + T])

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
    )
    group_ids = harness.backend._replay_group_ids
    request_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEV)
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


@pytest.mark.parametrize("draft_tokens", [None, 1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_decode_records_commit_after_acceptance(draft_tokens, captured):
    """Verify preserves live state; acceptance commits only the selected prefixes."""
    actual = _Harness(draft_tokens=draft_tokens, seed=431)
    eager = _Harness(draft_tokens=draft_tokens, seed=431)
    width = actual.width
    recipe = _kimi_recipe(
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
                backend._preconv_ring,
                backend._replay_state,
                backend._replay_snapshot,
                backend._retained_records,
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
    all_slots = list(range(1, max_live + 1))
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
            torch.testing.assert_close(
                actual.pool.get_component(layer, component), snapshot, atol=0, rtol=0
            )

    before = {
        (layer, component): actual.pool.get_component(layer, component).clone()
        for layer in actual.layer_ids
        for component in _CHECKPOINT_COMPONENTS
    }
    # Bind layer descriptors before capture, as serving warmup does.
    actual.prepare_metadata(all_slots, all_pages, all_lengths)
    actual.forward(actual.inputs(max_live, 433), max_live)
    assert_live_state_unchanged(before)
    graphs = {}
    if captured:
        stream = torch.cuda.Stream()
        for batch in dict.fromkeys(batch_sizes):
            actual.backend.init_forward_metadata_capture_cuda_graph(
                batch,
                torch.arange(1, batch + 1, dtype=torch.int32, device=DEV),
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
            all_slots.reverse()
        pages = {group: ids[:live] for group, ids in all_pages.items()}
        lengths = all_lengths[:live]
        rpis = all_slots[:live]
        inputs = actual.inputs(batch, 440 + round_index)
        inputs["g_raw"] = inputs["mixed_qkv"][:, : actual.key_dim]
        live_inputs = {name: value[: live * width] for name, value in inputs.items()}
        counts = [1 + (round_index + row) % width for row in range(live)]
        expected = _reference_committed_states(eager, pages, live_inputs, counts, rpis)
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
                _logical_state(actual, layer, ids, rpis),
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
    from tokenspeed_kernel.ops.attention.kda import (
        KdaPreconvRing,
        kda_verify_conv_update,
    )

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
    # Request 0's ring holds page 2's taps before its head, as its prepare
    # seeds it; request 1 is padding (page -1): zero history, no ring row.
    ring = torch.zeros(2, width + 3, channels, device=DEV, dtype=torch.bfloat16)
    ring[0, :3] = state[2].T
    before = ring.clone()
    actual = kda_verify_conv_update(
        raw,
        weight,
        pages,
        live_indices=pages,
        preconv=KdaPreconvRing(
            ring, torch.full((2,), 3, device=DEV, dtype=torch.int32)
        ),
        record_slots=torch.arange(2, device=DEV, dtype=torch.int32),
        page_count=state.shape[0],
        num_heads=H,
        head_dim=D,
        draft_token_num=width,
        recurrent_layout="v_major",
        bias=bias,
        query_start_loc=torch.arange(3, device=DEV, dtype=torch.int32) * width,
    )
    torch.testing.assert_close(actual.float(), expected, atol=1e-4, rtol=8e-3)
    before[0, 3:] = raw[:width]
    torch.testing.assert_close(ring, before, atol=0, rtol=0)


@pytest.mark.parametrize("draft_tokens", [None, 4])
@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
def test_record_decode_preserves_raw_gate_and_bias(draft_tokens, lower_bound):
    """Every layer verifies with a raw gate and a convolution bias; the first
    layer's committed state follows the torch reference."""
    harness = _Harness(draft_tokens=draft_tokens, seed=901)
    if not harness.backend._verify_split_producers:
        pytest.skip("fused verify kernel does not support a conv bias")
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
        reference = _reference_committed_states(
            harness, pages, inputs, [count], [1], layers=harness.layer_ids[:1]
        )
        harness.prepare_metadata([1], pages, [8 + harness.width + round_index * count])
        harness.forward(inputs, 1)
        harness.backend.commit_state_after_acceptance(
            torch.tensor([count], device=DEV, dtype=torch.int32),
            num_extends=0,
            accepted_path=None,
        )
        for layer, (state, conv) in reference.items():
            ids = pages[harness.pool.state_group_by_layer[layer]]
            torch.testing.assert_close(
                _logical_state(harness, layer, ids, [1]),
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
    # With ``checkpoint`` the extend row also stores its state 128 tokens in,
    # into a third block, as an EXTEND forward of it alone does.
    actual = _Harness(draft_tokens=draft_tokens, seed=941)
    width, prompt = actual.width, 200 if checkpoint else 3
    pages = {group: [2 + 2 * i, 3 + 2 * i] for i, group in enumerate(_STATE_GROUPS)}
    stores = {group: 8 + i for i, group in enumerate(_STATE_GROUPS)}
    sources = {
        group: ids[0] if same_page else ids[0] + 12 for group, ids in pages.items()
    }
    source_snapshots = {}
    if same_page:
        _seed_retained_history(actual, slot=1, length=3)
    for layer in actual.layer_ids:
        source = sources[actual.pool.state_group_by_layer[layer]]
        actual.pool.get_component(layer, "recurrent_state")[source].fill_(0.1)
        source_snapshots[layer] = _logical_state(actual, layer, [source], [1])
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
            torch.arange(1, bs + 1, device=DEV, dtype=torch.int32),
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
        slots=[1],
        prefill=True,
    )
    decode_outputs, decode_states = _reference_round(
        actual,
        {group: ids[1:] for group, ids in pages.items()},
        {name: value[prompt:] for name, value in inputs.items()},
        [count],
        slots=[2],
        prefill=False,
    )
    checkpoint_states = None
    if checkpoint:
        _, checkpoint_states = _reference_round(
            actual,
            {group: [source] for group, source in sources.items()},
            {name: value[:128] for name, value in inputs.items()},
            [128],
            slots=[1],
            prefill=True,
        )
    stored_before = {
        (layer, name): actual.pool.get_component(layer, name)[
            stores[actual.pool.state_group_by_layer[layer]]
        ].clone()
        for layer in actual.layer_ids
        for name in ("conv_state", "recurrent_state")
    }
    metadata(actual, mixed=True)
    # Every layer's gather reads the extend row's count, which clears once
    # the last layer's gather has read it.
    assert actual.backend._replay_length[1].item() == (3 if same_page else 0)
    actual_outputs = forward(actual, inputs, batch=2, mode=ForwardMode.MIXED)
    assert actual.backend._replay_length[1].item() == 0
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
        ids = pages[actual.pool.state_group_by_layer[layer]]
        torch.testing.assert_close(
            actual.pool.get_component(layer, "conv_state")[ids],
            torch.cat((prompt_states[layer][1], decode_states[layer][1])),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            _logical_state(actual, layer, ids, [1, 2]),
            torch.cat((prompt_states[layer][0], decode_states[layer][0])),
            atol=2e-6,
            rtol=2e-5,
        )
        if not same_page:
            source = sources[actual.pool.state_group_by_layer[layer]]
            torch.testing.assert_close(
                _logical_state(actual, layer, [source], [1]),
                source_snapshots[layer],
                atol=0,
                rtol=0,
            )


def _logical_state(harness, layer, pages, slots, counts=None):
    """Reconstruct each dense page using its request slot's accepted updates.

    ``counts`` are the slots' INT32 [slots, 1] counts to apply, by default the
    accepted counts every layer shares.
    """
    pool = harness.pool
    state = pool.get_component(layer, "recurrent_state")[pages].float().clone()
    if counts is None:
        counts = harness.backend._replay_length
    lengths = counts[slots].flatten().tolist()
    layer_row = harness.backend._descriptor_row_by_layer[layer]
    for row, (slot, length) in enumerate(zip(slots, lengths, strict=True)):
        correction, key, decay = harness.backend._retained_records[slot, layer_row]
        for index in range(length):
            state[row] = (
                state[row] * decay[:, index, None, :]
                + correction[:, index, :, None] * key[:, index, None, :]
            )
    return state


@pytest.mark.parametrize("width", [1, T])
@pytest.mark.parametrize("mamba_ssm_dtype", ["float32", "bfloat16"])
def test_ending_round_commit_leaves_a_dense_live_page(width, mamba_ssm_dtype):
    """The commit of the forward that ends a request folds its records.

    Its live page alone then holds the accepted endpoint, which its finish
    publishes, and its slot counts no record. A request still decoding keeps
    the dense base of its page and counts its accepted records.
    """
    harness = _Harness(draft_tokens=width, seed=1401, mamba_ssm_dtype=mamba_ssm_dtype)
    rpis = [3, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + width] * len(rpis)
    for round_index, accepted in enumerate(([width, 1], [1, width], [width, 1])):
        ending = round_index == 2
        if ending:
            # The sampling epilogue marks slot 3 before the commit runs.
            harness.stopped[3] = True
        inputs = harness.inputs(len(rpis), 1410 + round_index)
        inputs["g_raw"] = inputs["mixed_qkv"][:, : harness.key_dim]
        expected = _reference_committed_states(harness, pages, inputs, accepted, rpis)
        dense_before = {
            layer: harness.pool.get_component(layer, "recurrent_state")[
                pages[harness.pool.state_group_by_layer[layer]]
            ].clone()
            for layer in harness.layer_ids
        }
        harness.prepare_metadata(rpis, pages, seq_lens)
        harness.forward(inputs, len(rpis))
        harness.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV), accepted_path=None
        )
        for layer, (state, _) in expected.items():
            ids = pages[harness.pool.state_group_by_layer[layer]]
            dense = harness.pool.get_component(layer, "recurrent_state")[ids]
            torch.testing.assert_close(
                _logical_state(harness, layer, ids, rpis),
                state,
                atol=2e-6 if mamba_ssm_dtype == "float32" else 2e-3,
                rtol=2e-5 if mamba_ssm_dtype == "float32" else 2e-2,
            )
            counts = harness.backend._replay_length[rpis].flatten()
            if ending:
                assert counts[0].item() == 0
                torch.testing.assert_close(
                    dense[0].float(),
                    state[0].to(dense.dtype).float(),
                    atol=2e-6 if mamba_ssm_dtype == "float32" else 1e-5,
                    rtol=2e-5 if mamba_ssm_dtype == "float32" else 8e-3,
                )
            else:
                assert counts[0].item() > 0
                torch.testing.assert_close(
                    dense[0], dense_before[layer][0], atol=0, rtol=0
                )
            # Slot 1 keeps decoding: its page keeps the dense base.
            assert counts[1].item() > 0
            torch.testing.assert_close(dense[1], dense_before[layer][1], atol=0, rtol=0)
        seq_lens = [length + max(1, count) for length, count in zip(seq_lens, accepted)]


def _seed_retained_history(harness, *, slot, length):
    """Give ``slot`` ``length`` accepted records in every layer."""
    correction, key, decay = harness.backend._retained_records[slot].unbind(1)
    correction[:, :, :length].fill_(0.03)
    key[:, :, :length].fill_(0.02)
    decay[:, :, :length].fill_(0.9)
    harness.backend._replay_length[slot].fill_(length)


def test_a_fold_completes_the_listed_pages_and_leaves_the_rest():
    """``fold_deferred_state`` folds each listed slot's accepted records into
    the page a retraction's plan lists for the slot in that layer's group,
    and clears the count. A listed slot without records, restored and not
    yet decoding, keeps its page; no page any group lists changes otherwise.
    (The State groups share page memory, each page id held by one group at a
    time, so a layer's view shows another group's listed page change.)"""
    harness = _Harness(draft_tokens=T, seed=1501)
    pool = harness.pool
    _seed_retained_history(harness, slot=3, length=5)
    folds = {
        group_id: [(3, 2 + 2 * group), (1, 3 + 2 * group)]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    expected, before = {}, {}
    for layer in harness.layer_ids:
        listed = folds[pool.state_group_by_layer[layer]]
        expected[layer] = _logical_state(
            harness, layer, [page for _, page in listed], [slot for slot, _ in listed]
        )
        before[layer] = pool.get_component(layer, "recurrent_state").clone()

    harness.backend.fold_deferred_state(
        {
            group_id: torch.tensor(pairs, dtype=torch.int32)
            for group_id, pairs in folds.items()
        },
        stream=torch.cuda.current_stream(),
    )

    listed = {page for pairs in folds.values() for _, page in pairs}
    for layer in harness.layer_ids:
        pages = [page for _, page in folds[pool.state_group_by_layer[layer]]]
        dense = pool.get_component(layer, "recurrent_state")
        torch.testing.assert_close(
            dense[pages].float(), expected[layer], atol=2e-6, rtol=2e-5
        )
        others = [page for page in range(dense.shape[0]) if page not in listed]
        torch.testing.assert_close(dense[others], before[layer][others], atol=0, rtol=0)
    # Every layer folded from the count before one launch cleared it.
    assert harness.backend._replay_length[[3, 1]].flatten().tolist() == [0, 0]


def test_a_landed_slot_folds_nothing_its_previous_owner_left():
    """A request ended on the host leaves its accepted records counted in its
    slot. A PD landing gives the slot to a request whose prompt ran on
    another node, and a retraction may fold that request before its first
    forward. The landing's reset leaves the fold nothing to apply, so no
    page changes: neither the landed request's own nor one the previous
    owner wrote, which another request may hold by now."""
    harness = _Harness(draft_tokens=T, seed=1503)
    pool = harness.pool
    _seed_retained_history(harness, slot=3, length=5)
    before = {
        layer: pool.get_component(layer, "recurrent_state").clone()
        for layer in harness.layer_ids
    }
    stream = torch.cuda.current_stream()

    harness.backend.reset_request_workspace(
        torch.tensor([3], device=DEV), stream=stream
    )
    harness.backend.fold_deferred_state(
        {
            group_id: torch.tensor([[3, 2 + group]], dtype=torch.int32)
            for group, group_id in enumerate(_STATE_GROUPS)
        },
        stream=stream,
    )

    for layer in harness.layer_ids:
        torch.testing.assert_close(
            pool.get_component(layer, "recurrent_state"), before[layer], atol=0, rtol=0
        )


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_retained_rounds_reconstruct_endpoint_and_reuse_rejected(width, captured):
    """Twenty rounds of alternating acceptance over every layer: the first
    layer follows the torch reference, and the one count its slot keeps
    advances by each round's acceptance until a window would overflow the
    history, when the prepare folds the records and the count starts over."""
    actual = _Harness(draft_tokens=width, seed=1201)
    # Each group holds its own page: every layer acts on the request.
    pages = {group: [2 + index] for index, group in enumerate(_STATE_GROUPS)}
    layer = actual.layer_ids[0]
    page = pages[actual.pool.state_group_by_layer[layer]]
    _seed_retained_history(actual, slot=1, length=3)
    expected_count = 3
    actual.prepare_metadata([1], pages, [8 + width])
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
        # The reference reads the count before the round's prepare.
        count = width if iteration % 2 else 1
        expected_output, expected = _reference_round(
            actual, pages, values, [count], slots=[1], prefill=False, layers=[layer]
        )
        actual.prepare_metadata([1], pages, [8 + width + iteration])
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
        if expected_count + width > actual.backend._retained_capacity:
            expected_count = 0
        expected_count += count
        assert actual.backend._replay_length.flatten().tolist()[1] == expected_count
        torch.testing.assert_close(
            _logical_state(actual, layer, page, [1]),
            expected[layer][0],
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            actual.pool.get_component(layer, "conv_state")[page],
            expected[layer][1],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("same_page", [False, True])
@pytest.mark.parametrize("mamba_ssm_dtype", ["float32", "bfloat16"])
def test_retained_prefill_gathers_endpoint_and_resets_final_length(
    same_page, mamba_ssm_dtype
):
    """Every layer's gather applies the records the row's count covers, and
    the count clears once, after the last layer's gather has read it."""
    harness = _Harness(draft_tokens=4, mamba_ssm_dtype=mamba_ssm_dtype)
    layer = harness.layer_ids[0]
    source, destination = 2, 2 if same_page else 3
    pool = harness.pool
    for each in harness.layer_ids:
        pool.get_component(each, "recurrent_state")[source].fill_(0.1)
    # A distinct source is a published dense checkpoint. A continuation on
    # its own live page can still have records in the request's slot.
    if same_page:
        _seed_retained_history(harness, slot=1, length=3)
    expected = {
        each: _logical_state(harness, each, [source], [1]) for each in harness.layer_ids
    }
    snapshots = {
        name: pool.get_component(layer, name)[source].clone()
        for name in _CHECKPOINT_COMPONENTS
    }
    prefix = torch.tensor([8], dtype=torch.int32)
    new_tokens = torch.tensor([1], dtype=torch.int32)
    tables = {group: [[source, destination]] for group in _STATE_GROUPS}
    harness.backend.init_forward_metadata(
        1,
        1,
        torch.tensor([1], device=DEV, dtype=torch.int32),
        (prefix + new_tokens).to(DEV),
        ForwardMode.EXTEND,
        block_tables=_tables_for(harness.contract, tables, DEV),
        extend_seq_lens=new_tokens.to(DEV),
        extend_seq_lens_cpu=new_tokens,
        extend_prefix_lens=prefix.to(DEV),
        extend_prefix_lens_cpu=prefix,
        extend_replay_lens_cpu=torch.zeros_like(prefix),
        extend_prompt_lens_cpu=prefix + new_tokens,
        extend_with_prefix=True,
        extend_state_checkpoint_lens_cpu=torch.zeros_like(prefix),
        query_shard=None,
    )
    count = 3 if same_page else 0
    assert harness.backend._replay_length[1].item() == count
    reads = torch.tensor([source], device=DEV, dtype=torch.int32)
    writes = torch.tensor([destination], device=DEV, dtype=torch.int32)
    # Layers gather in model order; the count holds until the last one.
    for each in harness.layer_ids:
        assert harness.backend._replay_length[1].item() == count, each
        conv, recurrent = harness.backend._state_components(each)
        gathered, _ = harness.backend._prepare_prefill_state_inputs(
            each, conv, recurrent, reads, writes, dtype=torch.float32
        )
        torch.testing.assert_close(gathered, expected[each], atol=2e-7, rtol=2e-6)
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(
            pool.get_component(layer, name)[source], snapshot, atol=0, rtol=0
        )
    final = gathered + 0.25
    store_state_rows(final, recurrent, writes)
    torch.testing.assert_close(
        recurrent[destination], final[0].to(recurrent.dtype), atol=0, rtol=0
    )
    assert harness.backend._replay_length[1].item() == 0
    if not same_page:
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(
                pool.get_component(layer, name)[source], snapshot, atol=0, rtol=0
            )


@pytest.mark.parametrize(
    "dtype,expected", [("float32", 1 + 3 / 256), ("bfloat16", 1 + 2 / 128)]
)
def test_materializing_a_checkpoint_rounds_only_after_all_accepted_updates(
    dtype, expected
):
    harness = _Harness(draft_tokens=4, mamba_ssm_dtype=dtype)
    layer = harness.layer_ids[0]
    layer_row = harness.backend._descriptor_row_by_layer[layer]
    slot, page = 3, 2
    dense = harness.pool.get_component(layer, "recurrent_state")[page]
    length = harness.backend._replay_length[slot]
    records = harness.backend._retained_records[slot, layer_row]
    harness.layer_ids = [layer]
    # The overflowing count folds in every layer of the request, so each
    # group keeps its own page: groups may share page memory.
    own = harness.pool.state_group_by_layer[layer]
    pages = {
        group: [page if group == own else 5 + index]
        for index, group in enumerate(_STATE_GROUPS)
    }
    # The prepare ahead of the verify folds the old records before its four
    # candidates overflow the 16-entry workspace. Only three old updates are
    # nonzero.
    length.fill_(13)
    records.fill_(float("nan"))
    correction, key, decay = records.unbind(0)
    dense.fill_(1)
    correction[:, :13].zero_()
    correction[:, :3].fill_(1 / 256)
    key[:, :13].fill_(1)
    decay[:, :13].fill_(1)
    harness.prepare_metadata([slot], pages, [32])
    assert torch.all(dense == expected)
    # The count stays for the commit; past 16 - 4, the verify reads none of it.
    assert length.item() == 13
    # Verify reads the folded block and writes no block or count of its own.
    harness.forward(harness.inputs(1, 1311), 1)
    assert torch.all(dense == expected)
    assert length.item() == 13


def test_retained_workspace_matches_recipe_and_rebind_invalidates_history():
    harness = _Harness(draft_tokens=4, max_resident_requests=13)
    recipe = _kimi_recipe(
        max_bs=8, max_resident_requests=13, speculative_num_draft_tokens=4
    )
    assert (
        harness.backend.preallocate_verify_workspace(8, 4) == recipe.workspace_bytes()
    )
    # Every resident slot (13 and the null and padding slots) keeps a ring of
    # T + 3 pre-convolution rows per layer, in the convolution state's dtype.
    assert harness.backend._preconv_ring.shape == (15, 69, 7, CONV_DIM)
    assert harness.backend._preconv_ring.dtype == torch.bfloat16
    # Each group holds its own page: the last round commits every layer.
    pages = {group: [2 + index] for index, group in enumerate(_STATE_GROUPS)}
    page = pages[harness.pool.state_group_by_layer[harness.layer_ids[0]]]
    harness.layer_ids = harness.layer_ids[:1]
    harness.prepare_metadata([13], pages, [12])
    harness.forward(harness.inputs(1, 1301), 1)
    old_history = harness.backend._retained_histories[harness.layer_ids[0], 1]
    arena_storage = harness.pool.arena.buffer.untyped_storage().data_ptr()
    assert old_history.length.dtype == torch.int32
    assert old_history.length.numel() == harness.config.max_resident_requests + 2
    assert old_history.length.untyped_storage().data_ptr() != arena_storage
    for records in (old_history.correction, old_history.key, old_history.decay):
        assert records.untyped_storage().data_ptr() != arena_storage
        assert records.shape == (harness.config.max_resident_requests + 2, H, 16, D)
        assert records.dtype == torch.float32
    harness.prepare_metadata([13], pages, [13])
    harness.forward(harness.inputs(1, 1302), 1)
    assert harness.backend._retained_histories[harness.layer_ids[0], 1] is old_history
    # Rebind while the cached history still refers to the old pool.
    old_arena = harness.pool.arena.buffer.clone()
    replacement = _make_kimi_pool(DEV, usable_pages=26)
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
    layer = harness.layer_ids[0]
    expected = _reference_committed_states(harness, pages, inputs, [3], [13])
    # Every layer verifies the round its commit publishes.
    harness.layer_ids = list(harness.backend._state_layer_ids())
    harness.prepare_metadata([13], pages, [12])
    harness.forward(inputs, 1)
    harness.backend.commit_verified_state(
        torch.tensor([3], device=DEV, dtype=torch.int32), accepted_path=None
    )
    torch.testing.assert_close(
        _logical_state(harness, layer, page, [13]),
        expected[layer][0],
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        replacement.get_component(layer, "conv_state")[page],
        expected[layer][1],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(old_pool.arena.buffer, old_arena, atol=0, rtol=0)
    # Rebuilding graph metadata must also release the cached history views.
    harness.backend.init_cuda_graph_state(8)
    assert not harness.backend._retained_histories


@pytest.mark.parametrize(
    "same_page,initial_length", [(True, 12), (True, 13), (False, 0)]
)
def test_retained_source_live_capacity_boundary(initial_length, same_page):
    """A row whose count fits beside its window reads it whole; one past
    capacity - window starts from a folded block and reads none; one that
    moves to private pages starts from a copy of its source. Every layer
    verifies; the first follows the torch reference. Each group holds its
    own pages, as groups may share page memory."""
    actual = _Harness(draft_tokens=4, seed=1401)
    layer = actual.layer_ids[0]
    own = actual.pool.state_group_by_layer[layer]
    groups = {group: index for index, group in enumerate(_STATE_GROUPS)}
    # Group g reads source page 1 + 3g; it writes back in place, or to the
    # two private pages after it.
    sources = {group: 1 + 3 * index for group, index in groups.items()}
    destinations = {
        group: [source] if same_page else [source + 1, source + 2]
        for group, source in sources.items()
    }
    for each in actual.layer_ids:
        source = sources[actual.pool.state_group_by_layer[each]]
        actual.pool.get_component(each, "recurrent_state")[source].fill_(0.125)
    if same_page:
        _seed_retained_history(actual, slot=1, length=initial_length)
    logical_source = _logical_state(actual, layer, [sources[own]], [1])
    before = {
        name: actual.pool.get_component(layer, name)[sources[own]].clone()
        for name in _CHECKPOINT_COMPONENTS
    }
    batch = len(destinations[own])
    slots = [1] if same_page else [3, 5]
    accepted = [4] if same_page else [1, 4]
    tables = {
        group: np.asarray(
            [[sources[group], page] for page in destinations[group]], dtype=np.int32
        )
        for group in _STATE_GROUPS
    }
    inputs = actual.inputs(batch, 1402)
    # Check consolidation and copy-on-write independently of gate projection.
    inputs["g_raw"] = inputs["mixed_qkv"][:, : actual.key_dim]
    expected_output, expected = _reference_round(
        actual,
        {group: [source] * batch for group, source in sources.items()},
        inputs,
        accepted,
        slots=slots,
        prefill=False,
        layers=[layer],
    )
    actual.backend.refresh_decode_metadata(
        batch,
        batch,
        torch.tensor(slots, device=DEV, dtype=torch.int32),
        torch.full((batch,), 20, device=DEV, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables_for(actual.contract, tables, DEV),
    )
    output = actual.forward(inputs, batch)[0]
    torch.testing.assert_close(
        output.reshape_as(expected_output[0]), expected_output[0], atol=2e-3, rtol=2e-3
    )
    counts = actual.backend._replay_length
    # The records the round's verify read: none of a count past
    # capacity - window, whose records the prepare folded.
    read = torch.where(counts > actual.backend._retained_capacity - 4, 0, counts)
    torch.testing.assert_close(
        _logical_state(actual, layer, destinations[own], slots, counts=read),
        logical_source.expand(batch, -1, -1, -1),
        atol=2e-7,
        rtol=2e-6,
    )
    retained_before_commit = 12 if same_page and initial_length == 12 else 0
    assert read[slots].flatten().tolist() == [retained_before_commit] * batch
    # The prepare leaves the count to the commit.
    assert (
        actual.backend._replay_length[slots].flatten().tolist()
        == [initial_length if same_page else 0] * batch
    )
    actual.backend.commit_verified_state(
        torch.tensor(accepted, device=DEV, dtype=torch.int32), accepted_path=None
    )
    assert actual.backend._replay_length[slots].flatten().tolist() == [
        retained_before_commit + count for count in accepted
    ]
    torch.testing.assert_close(
        _logical_state(actual, layer, destinations[own], slots),
        expected[layer][0],
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        actual.pool.get_component(layer, "conv_state")[destinations[own]],
        expected[layer][1],
        atol=0,
        rtol=0,
    )
    if not same_page:
        for name, snapshot in before.items():
            torch.testing.assert_close(
                actual.pool.get_component(layer, name)[sources[own]],
                snapshot,
                atol=0,
                rtol=0,
            )


def test_retained_nonsquare_pool_is_rejected_before_publication():
    harness = _Harness(draft_tokens=4)
    malformed = _make_kimi_pool(DEV)
    for layer in malformed.state_group_by_layer:
        conv, state = malformed._state_buffers_by_layer[layer]
        malformed._state_buffers_by_layer[layer] = (conv, state[..., :64])
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


@pytest.mark.parametrize(
    "invalid", ["conv_dtype", "conv_stride", "recurrent_dtype", "draft"]
)
def test_unsupported_retained_pool_rejected_before_publication(invalid):
    from dataclasses import replace

    harness = _Harness(draft_tokens=4)
    pool = _make_kimi_pool(DEV)
    layer = min(pool.state_group_by_layer)
    conv, recurrent = pool.get_state_buffers(layer)
    if invalid == "conv_dtype":
        pool._state_buffers_by_layer[layer] = (conv.float(), recurrent)
    elif invalid == "conv_stride":
        strided = torch.empty((*conv.shape, 2), device=DEV, dtype=conv.dtype)[..., 0]
        pool._state_buffers_by_layer[layer] = (strided, recurrent)
    elif invalid == "recurrent_dtype":
        # Supported alone, but one commit and fold launch serves every layer.
        other = torch.bfloat16 if recurrent.dtype == torch.float32 else torch.float32
        pool._state_buffers_by_layer[layer] = (conv, recurrent.to(other))
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
    with pytest.raises(ValueError, match="retained|geometry|length"):
        backend.set_kv_pool(pool)
    assert backend.kv_pool is None


@pytest.mark.parametrize("draft_tokens", [None, T])
def test_verify_fuses_producers_and_commits_like_split(monkeypatch, draft_tokens):
    """Verify projects the gate and convolves inside its kernel, as split would."""
    import tokenspeed.runtime.layers.attention.backends.state.kda as kda_backend

    fused = _Harness(draft_tokens=draft_tokens, seed=1201)
    split = _Harness(draft_tokens=draft_tokens, seed=1201)
    if not fused.backend._verify_split_producers:
        pytest.skip("split verify producers unavailable")
    convolutions = []
    conv_update = kda_backend.kda_verify_conv_update

    def counted_conv_update(*args, **kwargs):
        convolutions.append(kwargs["draft_token_num"])
        return conv_update(*args, **kwargs)

    monkeypatch.setattr(kda_backend, "kda_verify_conv_update", counted_conv_update)
    rpis = [2, 1]
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
    _assert_committed_pages_equal(fused, split, pages, rpis)


def test_a_page_move_copies_the_base_and_keeps_the_count():
    """A request whose input page differs from its live page in one State
    group gets its dense base copied there, with no record applied: the
    records its count covers stay valid on top of the copy, so its count and
    the other groups' pages stay as they are, and the round verifies and
    commits as if the pages had not moved."""
    harness = _Harness(draft_tokens=T, seed=1601)
    moved = _STATE_GROUPS[0]
    live = {group: 2 + index for index, group in enumerate(_STATE_GROUPS)}
    pages = {group: [page] for group, page in live.items()}
    harness.round([1], pages, [8 + T], 1601, [T])
    assert harness.backend._replay_length[1].item() == T
    # The moved group's old live block becomes its input checkpoint and a
    # new private block goes live; the other groups keep theirs.
    fresh = 2 + len(_STATE_GROUPS)
    destinations = {
        group: [fresh if group == moved else page] for group, page in live.items()
    }
    inputs = harness.inputs(1, 1602)
    inputs["g_raw"] = inputs["mixed_qkv"][:, : harness.key_dim]
    expected = _reference_committed_states(harness, pages, inputs, [2], [1])
    before = {
        layer: harness.pool.get_component(layer, "recurrent_state").clone()
        for layer in harness.layer_ids
    }
    tables = {
        group: np.asarray(
            [[page if group == moved else 0, destinations[group][0]]], dtype=np.int32
        )
        for group, page in live.items()
    }
    harness.backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1], dtype=torch.int32, device=DEV),
        torch.tensor([8 + 2 * T], dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables_for(harness.contract, tables, DEV),
    )
    assert harness.backend._replay_length[1].item() == T
    for layer, pool in before.items():
        group = harness.pool.state_group_by_layer[layer]
        recurrent = harness.pool.get_component(layer, "recurrent_state")
        if group == moved:
            torch.testing.assert_close(
                recurrent[fresh], pool[live[moved]], atol=0, rtol=0
            )
        else:
            torch.testing.assert_close(recurrent, pool, atol=0, rtol=0)
    harness.forward(inputs, 1)
    harness.backend.commit_verified_state(
        torch.tensor([2], dtype=torch.int32, device=DEV), accepted_path=None
    )
    assert harness.backend._replay_length[1].item() == T + 2
    for layer, (state, conv) in expected.items():
        ids = destinations[harness.pool.state_group_by_layer[layer]]
        torch.testing.assert_close(
            _logical_state(harness, layer, ids, [1]), state, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            harness.pool.get_component(layer, "conv_state")[ids],
            conv,
            atol=0,
            rtol=0,
        )
    # The moved group's input checkpoint is read, never written.
    for layer, pool in before.items():
        if harness.pool.state_group_by_layer[layer] == moved:
            torch.testing.assert_close(
                harness.pool.get_component(layer, "recurrent_state")[live[moved]],
                pool[live[moved]],
                atol=0,
                rtol=0,
            )


def test_graph_replays_read_no_record_past_the_capacity_like_eager():
    """Rounds that accept a full window carry the shared count past
    capacity - window. The prepare then folds the records into the live
    block, and the verify and the commit read none, so the next count is the
    round's acceptance alone. Replays of one captured verify graph follow
    eager forwards round for round, across both overflows."""
    captured = _Harness(draft_tokens=T, seed=1801)
    eager = _Harness(draft_tokens=T, seed=1801)
    rpis, bs = [1], 1
    pages = {group: [2 + index] for index, group in enumerate(_STATE_GROUPS)}
    seq_lens = [64]

    warm_inputs = captured.inputs(bs, 1800)
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
    stable_inputs = captured.inputs(bs, 1801)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        captured.forward(stable_inputs, bs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.forward(stable_inputs, bs)

    # Verifies write candidates only past the count; both harnesses start
    # from the same twelve records.
    capacity = captured.backend._retained_capacity
    assert capacity == 16
    for harness in (captured, eager):
        _seed_retained_history(harness, slot=1, length=capacity - T - 1)
    accepted = torch.full((bs,), T, dtype=torch.int32, device=DEV)
    counts = []
    for round_index in range(7):
        tables = {
            group_id: np.asarray(
                [[0, page] for page in pages[group_id]], dtype=np.int32
            )
            for group_id in _STATE_GROUPS
        }
        seq_lens_tensor.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=DEV))
        captured.backend.refresh_decode_metadata(
            bs,
            bs,
            req_pool_indices,
            seq_lens_tensor,
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables=_tables_for(captured.contract, tables, DEV),
        )
        replay_inputs = captured.inputs(bs, 1810 + round_index)
        for name, value in replay_inputs.items():
            stable_inputs[name].copy_(value)
        graph.replay()
        captured.backend.commit_verified_state(accepted, accepted_path=None)

        eager.prepare_metadata(rpis, pages, seq_lens)
        eager.forward(replay_inputs, bs)
        eager.backend.commit_verified_state(accepted, accepted_path=None)
        torch.cuda.synchronize()
        counts.append(eager.backend._replay_length[1].item())
        assert captured.backend._replay_length[1].item() == counts[-1]
        _assert_committed_pages_equal(captured, eager, pages, rpis)
        seq_lens = [seq_lens[0] + T]
    # The twelve seeded records fit beside a 3-token window (a window reads
    # up to 16 - 3 = 13), so the first round counts 12 + 3 = 15. That is past
    # the 13 records, so the next round reads none and counts its own 3; at
    # 12 + 3 = 15 that repeats.
    assert counts == [15, 3, 6, 9, 12, 15, 3]


def test_an_ended_request_some_layers_reject_reports_incomplete():
    """Only the first layer verifies the round that ends a request: the other
    layers reject its commit, so its report says the live block lacks
    accepted tokens. The request ended, so its count clears."""
    from tokenspeed_kernel.ops.metadata import (
        REQUEST_ENDED,
        REQUEST_ENDED_INCOMPLETE,
    )

    harness = _Harness(draft_tokens=T, seed=1702)
    pages = {group: [2 + index] for index, group in enumerate(_STATE_GROUPS)}
    harness.round([1], pages, [8 + T], 1702, [T])
    harness.layer_ids = harness.layer_ids[:1]
    harness.stopped[1] = True
    harness.request_ends[0] = REQUEST_ENDED
    harness.round([1], pages, [8 + 2 * T], 1703, [T])
    assert harness.request_ends[0].item() == REQUEST_ENDED_INCOMPLETE
    assert harness.backend._replay_length[1].item() == 0


def test_a_masked_overlap_row_keeps_its_count():
    """A masked overlap row -- every State group's pages invalid, its request
    ended by an earlier forward -- carries no layer: it reports nothing new,
    and its slot keeps its count and convolution windows."""
    from tokenspeed_kernel.ops.metadata import REQUEST_ENDED

    harness = _Harness(draft_tokens=T, seed=1706)
    pages = {group: [2 + index] for index, group in enumerate(_STATE_GROUPS)}
    harness.round([1], pages, [8 + T], 1706, [T])
    harness.stopped[1] = True
    harness.request_ends[0] = REQUEST_ENDED
    conv_before = {
        layer: harness.pool.get_component(layer, "conv_state").clone()
        for layer in harness.layer_ids
    }
    tables = {
        group: np.asarray([[0, page] for page in pages[group]], dtype=np.int32)
        for group in _STATE_GROUPS
    }
    delivered = _tables_for(harness.contract, tables, DEV)
    for group in _STATE_GROUPS:
        delivered[group][0, 1] = -1
    harness.backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1], dtype=torch.int32, device=DEV),
        torch.tensor([8 + 2 * T], dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        block_tables=delivered,
    )
    harness.forward(harness.inputs(1, 1707), 1)
    harness.backend.commit_verified_state(
        torch.tensor([T], dtype=torch.int32, device=DEV), accepted_path=None
    )
    assert harness.request_ends[0].item() == REQUEST_ENDED
    assert harness.backend._replay_length[1].item() == T
    for layer, before in conv_before.items():
        torch.testing.assert_close(
            harness.pool.get_component(layer, "conv_state"), before, atol=0, rtol=0
        )


def _memory_saver_kimi_pool(usable_pages):
    """The Kimi-K3 test pool on an arena whose storage the memory saver manages."""
    from test.runtime.cache_pool_test_utils import make_arena, make_pool
    from test.runtime.conftest import MLA_KV_LORA_RANK, MLA_QK_ROPE_DIM, kimi_layout

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )

    recipe = _kimi_recipe()
    groups, layout = kimi_layout(recipe)
    plan = layout.bind(usable_pages)
    arena = make_arena(
        plan,
        DEV,
        cache_group_specs=tuple(spec for spec, _ in groups),
        enable_memory_saver=True,
    )
    _, pool = make_pool(
        HybridKDATokenToKVPool,
        plan,
        device=DEV,
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=MLA_KV_LORA_RANK,
        qk_rope_head_dim=MLA_QK_ROPE_DIM,
        layer_num=KimiLinearConfig().num_hidden_layers,
        rank=0,
        layer_types=recipe.layer_types,
    )
    return pool


def _check_retained_workspace_survives_release_on_cuda():
    """Seed the rings, release and resume weights three times, then verify again.

    The backend allocates its retained workspace (records, the pre-convolution
    ring, the replay state and snapshot, the commit descriptors) where serving
    does, in the "attention_state" region, and the arena keeps the cache
    pages. After draining and each release no ordinary CUDA allocator segment
    may stay; after each resume every persistent tensor keeps its address and
    contents, each descriptor still addresses its layer's ring, and the next
    verify reproduces the bits it gave before the first release. A verify
    reads the ring history and records and rewrites the same candidate rows,
    so running it again is idempotent.
    """
    pool = _memory_saver_kimi_pool(usable_pages=24)
    saver = pool.arena.memory_saver_adapter
    width, rpis = 4, [3, 1]
    harness = _Harness(draft_tokens=width, seed=601, pool=pool)
    backend = harness.backend
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }

    def refresh(seq_lens):
        # Serving builds the block tables in "attention_state" and the batch
        # in the executor's inputs; the refresh copies them into its buffers.
        with saver.named_region("attention_state"):
            tables = _tables_for(
                harness.contract,
                {
                    group_id: np.asarray(
                        [[0, page] for page in pages[group_id]], dtype=np.int32
                    )
                    for group_id in _STATE_GROUPS
                },
                DEV,
            )
        with saver.named_region("runtime_states"):
            indices = torch.tensor(rpis, dtype=torch.int32, device=DEV)
            lengths = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
        backend.refresh_decode_metadata(
            len(rpis),
            len(rpis),
            indices,
            lengths,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
        )

    # Serving's decode capture (prepare_capture_metadata, in this region)
    # memoizes the verify scratch grid of each captured batch size before any
    # refresh reads it; this eager run memoizes its batch size the same way.
    with saver.named_region("attention_state"):
        backend._verify_scratch_grid(len(rpis), width)
    seq_lens = [8 + width] * len(rpis)
    for round_index in range(3):
        accepted = [1 + (round_index + row) % width for row in range(len(rpis))]
        refresh(seq_lens)
        with saver.named_region("runtime_states"):
            inputs = harness.inputs(len(rpis), 610 + round_index)
            taken = torch.tensor(accepted, dtype=torch.int32, device=DEV)
        harness.forward(inputs, len(rpis))
        backend.commit_verified_state(taken, accepted_path=None)
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    assert all(state > 0 for state in backend._replay_state[rpis, 1].tolist())

    refresh(seq_lens)
    with saver.named_region("runtime_states"):
        inputs = harness.inputs(len(rpis), 620)
    expected = [out.cpu() for out in harness.forward(inputs, len(rpis))]
    ring = backend._preconv_ring
    descriptors = backend._replay_descriptors.view(torch.int64)
    ring_pointers = [ring[:, row].data_ptr() for row in range(ring.shape[1])]
    retained = [
        (tensor, tensor.data_ptr(), tensor.cpu())
        for tensor in (
            ring,
            backend._replay_state,
            backend._replay_snapshot,
            descriptors,
            backend._retained_records,
            *inputs.values(),
        )
    ]
    gc.collect()
    for _ in range(3):
        torch.cuda.synchronize()
        saver.pause(tag="weights")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ordinary_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == torch.cuda.current_device()
            and tuple(segment["segment_pool_id"]) == (0, 0)
        )
        assert ordinary_bytes == 0, ordinary_bytes
        saver.resume(tag="weights")
        for tensor, pointer, reference in retained:
            assert tensor.data_ptr() == pointer
            torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)
        assert descriptors[:, 0].cpu().tolist() == ring_pointers
    for got, want in zip(harness.forward(inputs, len(rpis)), expected, strict=True):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.version.hip is not None, reason="requires native CUDA VMM memory-saver hooks"
)
def test_retained_workspace_survives_weights_release_and_resume():
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_kimi_k3_kda_eager_commit import "
                "_check_retained_workspace_survives_release_on_cuda; "
                "_check_retained_workspace_survives_release_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=900,
        )
