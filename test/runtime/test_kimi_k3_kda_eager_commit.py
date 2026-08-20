"""Eager KDA replay commit versus the existing per-position scratch path."""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import cache_metadata_for as _metadata_for
from test.runtime.conftest import kimi_recipe as _kimi_recipe
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from types import SimpleNamespace  # noqa: E402

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
)
from tokenspeed.runtime.layers.attention.registry import _prepare_verify_workspace

_LOWER_BOUND = -5.0
H, D, D_FA = 4, 128, 128
KEY_DIM = H * D
CONV_DIM = 3 * KEY_DIM
T = 3
DEV = "cuda"


class _Harness:
    def __init__(self, *, eager_replay: bool, seed: int = 0):
        torch.manual_seed(seed)
        self.pool = _make_kimi_pool(DEV, usable_pages=24)
        self.contract = self.pool.arena.runtime_contract
        config = SimpleNamespace(
            device=DEV,
            num_attention_heads=H,
            num_kv_heads=H,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=D,
            is_draft=False,
            speculative_num_draft_tokens=T,
            max_bs=8,
        )
        self.backend = KdaAttnBackend(config)
        self.backend.set_kv_pool(self.pool)
        if eager_replay and not self.backend._replay_active:
            pytest.skip("KDA replay commit kernel unavailable")
        if not eager_replay:
            self.backend._replay_active = False
            self.backend._verify_scratch = None
            self.backend.preallocate_verify_workspace(config.max_bs, T)
        self.layer_ids = list(self.backend._state_layer_ids())
        self.params = {
            layer_id: dict(
                conv_weights=torch.randn(CONV_DIM, 4, device=DEV, dtype=torch.bfloat16)
                * 0.1,
                f_b_weight=torch.randn(KEY_DIM, D_FA, device=DEV, dtype=torch.bfloat16)
                * 0.05,
                A_log=torch.randn(H, device=DEV, dtype=torch.float32) * 0.1,
                dt_bias=torch.randn(KEY_DIM, device=DEV, dtype=torch.float32) * 0.1,
            )
            for layer_id in self.layer_ids
        }

    def inputs(self, bs, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=generator).to(DEV, torch.bfloat16)

        return {
            "mixed_qkv": rnd(bs * T, CONV_DIM),
            "f_a_out": rnd(bs * T, D_FA),
            "beta_raw": rnd(bs * T, H),
        }

    def prepare_metadata(self, rpis, pages, seq_lens):
        bs = len(rpis)
        tables = {
            group_id: np.asarray([[page] for page in pages[group_id]], dtype=np.int32)
            for group_id in _STATE_GROUPS
        }
        metadata, op = _metadata_for(self.contract, tables, DEV)
        self.backend.init_forward_metadata(
            bs=bs,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            cache_metadata=metadata,
            forward_batch=op,
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
                    g_raw=None,
                    conv_weights=params["conv_weights"],
                    bias=None,
                    activation="silu",
                    key_dim=KEY_DIM,
                    value_dim=KEY_DIM,
                    attention_tp_size=1,
                    head_k_dim=D,
                    head_v_dim=D,
                    A_log=params["A_log"],
                    dt_bias=params["dt_bias"],
                    f_b_weight=params["f_b_weight"],
                    lower_bound=_LOWER_BOUND,
                    layer_id=layer_id,
                    seq_len=bs * T,
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
            torch.tensor(accepted, dtype=torch.int32, device=DEV)
        )


def _assert_committed_pages_equal(left, right, pages):
    for layer_id in left.layer_ids:
        group_id = left.pool.group_id_for_layer(layer_id)
        for page in pages[group_id]:
            torch.testing.assert_close(
                left.pool.get_component(layer_id, "conv_state")[page],
                right.pool.get_component(layer_id, "conv_state")[page],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                left.pool.get_component(layer_id, "recurrent_state")[page],
                right.pool.get_component(layer_id, "recurrent_state")[page],
                atol=1e-5,
                rtol=1e-3,
            )


def test_eager_replay_matches_scratch_over_multiple_rounds_and_layers():
    """Absolute oracle: independent scratch and replay arms evolve identically."""
    replay = _Harness(eager_replay=True, seed=17)
    scratch = _Harness(eager_replay=False, seed=17)
    rpis = [0, 1, 2]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    accepts = ([0, 1, T], [T, 0, 2], [1, T, 0], [2, 1, T])
    for round_index, accepted in enumerate(accepts):
        for harness in (replay, scratch):
            harness.round(rpis, pages, seq_lens, 100 + round_index, accepted)
        _assert_committed_pages_equal(replay, scratch, pages)
        seq_lens = [length + count for length, count in zip(seq_lens, accepted)]


def test_graph_replay_then_post_forward_commit_matches_eager_over_rounds():
    """The captured verify graph plus the existing post-forward hook follows eager."""
    captured = _Harness(eager_replay=True, seed=29)
    eager = _Harness(eager_replay=True, seed=29)
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    bs = len(rpis)

    captured.backend.init_cuda_graph_state(captured.backend.max_bs)
    warm_inputs = captured.inputs(bs, 211)
    captured.prepare_metadata(rpis, pages, seq_lens)
    captured.forward(warm_inputs, bs)
    torch.cuda.synchronize()

    req_pool_indices = torch.tensor(rpis, dtype=torch.int32, device=DEV)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    capture_tables = {
        group_id: torch.full((bs, 1), -1, dtype=torch.int32, device=DEV)
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
    payload = captured.backend._replay_payload(first_layer)[0]
    payload_ptr = payload.data_ptr()
    captured_payload = payload.clone()
    accepts = ([1, T], [T, 1], [2, T])
    for round_index, accepted in enumerate(accepts):
        tables = {
            group_id: np.asarray([[page] for page in pages[group_id]], dtype=np.int32)
            for group_id in _STATE_GROUPS
        }
        metadata, op = _metadata_for(captured.contract, tables, DEV)
        seq_lens_tensor.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=DEV))
        captured.backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens_tensor,
            ForwardMode.DECODE,
            cache_metadata=metadata,
            forward_batch=op,
        )
        replay_inputs = captured.inputs(bs, 227 + round_index)
        for name, value in replay_inputs.items():
            stable_inputs[name].copy_(value)
        accepted_source.copy_(torch.tensor(accepted, dtype=torch.int32, device=DEV))
        graph.replay()
        captured.backend.commit_verified_state(stable_accepted)

        eager.prepare_metadata(rpis, pages, seq_lens)
        eager.forward(replay_inputs, bs)
        eager.backend.commit_verified_state(stable_accepted)
        torch.cuda.synchronize()
        _assert_committed_pages_equal(captured, eager, pages)
        seq_lens = [length + count for length, count in zip(seq_lens, accepted)]

    assert payload.data_ptr() == payload_ptr
    assert not torch.equal(payload, captured_payload)

    # Negative control: replaying the next round without the required
    # post-forward commit must diverge from the multi-round eager oracle.
    tables = {
        group_id: np.asarray([[page] for page in pages[group_id]], dtype=np.int32)
        for group_id in _STATE_GROUPS
    }
    metadata, op = _metadata_for(captured.contract, tables, DEV)
    seq_lens_tensor.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=DEV))
    captured.backend.init_forward_metadata_replay_cuda_graph(
        bs,
        req_pool_indices,
        seq_lens_tensor,
        ForwardMode.DECODE,
        cache_metadata=metadata,
        forward_batch=op,
    )
    replay_inputs = captured.inputs(bs, 251)
    for name, value in replay_inputs.items():
        stable_inputs[name].copy_(value)
    accepted_source.fill_(T)
    graph.replay()  # Deliberately omit commit_verified_state.
    eager.prepare_metadata(rpis, pages, seq_lens)
    eager.forward(replay_inputs, bs)
    eager.backend.commit_verified_state(stable_accepted)
    torch.cuda.synchronize()
    with pytest.raises(AssertionError):
        _assert_committed_pages_equal(captured, eager, pages)


def test_replay_planning_matches_allocation_and_rejects_drift():
    """The cache recipe and startup allocator share one byte contract."""
    harness = _Harness(eager_replay=True)
    recipe = _kimi_recipe(
        draft_layers=5,
        max_bs=8,
        speculative_algorithm="EAGLE3",
        speculative_num_draft_tokens=T,
    )
    planned_bytes = recipe.setup().fixed_workspace_bytes
    harness.backend.preallocate_verify_workspace(harness.backend.max_bs, T)
    conv_bytes = sum(
        pair[0].nbytes for pair in harness.backend._verify_scratch.values()
    )
    payload_bytes = sum(payload.nbytes for payload in harness.backend._replay_payloads)
    assert planned_bytes == conv_bytes + payload_bytes
    server_args = SimpleNamespace(speculative_num_draft_tokens=T)
    config = SimpleNamespace(max_bs=8)
    backend = SimpleNamespace(linear_attn_backend=harness.backend)

    _prepare_verify_workspace(
        server_args=server_args,
        config=config,
        backend=backend,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
        expected_bytes=planned_bytes,
    )

    with pytest.raises(
        RuntimeError,
        match="planned paged-state verify workspace does not match allocated tensors",
    ):
        _prepare_verify_workspace(
            server_args=server_args,
            config=config,
            backend=backend,
            draft_backend=None,
            uses_paged_state_verify=True,
            is_inkling=False,
            expected_bytes=planned_bytes + 1,
        )


def test_descriptor_binding_rejects_nonuniform_conv_width():
    harness = _Harness(eager_replay=True)
    last = harness.layer_ids[-1]
    harness.params[last]["conv_weights"] = harness.params[last]["conv_weights"][:, :3]
    harness.prepare_metadata([0], {group: [2] for group in _STATE_GROUPS}, [8 + T])
    with pytest.raises(RuntimeError, match="uniform geometry"):
        harness.forward(harness.inputs(1, 701), 1)


def test_equal_geometry_pool_replacement_rebinds_batched_replay():
    harness = _Harness(eager_replay=True)
    pages = {group: [2] for group in _STATE_GROUPS}
    harness.prepare_metadata([0], pages, [8 + T])
    harness.forward(harness.inputs(1, 711), 1)
    assert harness.backend._batched_replay_ready

    replacement = _make_kimi_pool(DEV, usable_pages=24)
    harness.backend.set_kv_pool(replacement)
    harness.pool = replacement
    harness.contract = replacement.arena.runtime_contract
    harness.prepare_metadata([0], pages, [8 + T])
    harness.forward(harness.inputs(1, 712), 1)
    assert harness.backend._batched_replay_ready


def test_verify_scratch_cannot_grow_after_graph_capture():
    """A late transient-row reallocation fails instead of invalidating graphs."""
    harness = _Harness(eager_replay=True)
    harness.backend.preallocate_verify_workspace(harness.backend.max_bs, T)
    capacity = next(iter(harness.backend._verify_scratch.values()))[0].shape[0]
    harness.backend._graphs_captured = True
    harness.backend.query_start_loc_list.extend([None] * (capacity + 1))
    with pytest.raises(RuntimeError, match="after CUDA graphs were captured"):
        harness.backend._ensure_verify_scratch(1, T)


def test_no_store_fused_verify_matches_decomposed_outputs():
    """Replay fusion agrees with the shared conv/gate/recurrent verify path."""
    from types import MethodType

    fused = _Harness(eager_replay=True, seed=61)
    decomposed = _Harness(eager_replay=False, seed=61)
    decomposed.backend._verify = MethodType(
        MambaAttnBackend._verify, decomposed.backend
    )
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    inputs = fused.inputs(len(rpis), 67)
    for harness in (fused, decomposed):
        harness.prepare_metadata(rpis, pages, [11, 11])
    fused_out = fused.forward(inputs, len(rpis))
    decomposed_out = decomposed.forward(inputs, len(rpis))
    for actual, expected in zip(fused_out, decomposed_out):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    # Negative control: perturbing the decomposed base state must break the oracle.
    layer_id = decomposed.layer_ids[0]
    group_id = decomposed.pool.group_id_for_layer(layer_id)
    recurrent = decomposed.pool.get_component(layer_id, "recurrent_state")
    recurrent[torch.tensor(pages[group_id], device=DEV)] += 1
    wrong = decomposed.forward(inputs, len(rpis))[0]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(fused_out[0], wrong, atol=2e-2, rtol=2e-2)
