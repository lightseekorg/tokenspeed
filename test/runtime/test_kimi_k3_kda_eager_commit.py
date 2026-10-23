"""Eager KDA replay commit versus the existing per-position scratch path."""

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
    binding_state,
    storages_of,
)
from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import block_tables_for as _tables_for
from test.runtime.conftest import kimi_recipe as _kimi_recipe
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from types import MethodType, SimpleNamespace  # noqa: E402

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
H, D, D_FA = 4, 128, 128
KEY_DIM = H * D
CONV_DIM = 3 * KEY_DIM
T = 3
DEV = "cuda"


class _Harness:
    H = H

    def __init__(
        self,
        *,
        eager_replay: bool,
        draft_tokens: int | None,
        seed: int = 0,
        max_bs: int = 8,
        usable_pages: int = 24,
    ):
        torch.manual_seed(seed)
        self.pool = _make_kimi_pool(DEV, usable_pages=usable_pages)
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
            prefix_granularity=128,
            context_len=4096,
            is_draft=False,
            speculative_num_draft_tokens=draft_tokens,
            max_bs=max_bs,
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
        if eager_replay and not self.backend._replay_active:
            pytest.skip("KDA replay commit kernel unavailable")
        if not eager_replay:
            self.backend._replay_active = False
            self.backend._verify_scratch = None
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
                left.pool.get_component(layer_id, "recurrent_state")[page],
                right.pool.get_component(layer_id, "recurrent_state")[page],
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


def test_direct_committed_read_matches_seeded_verify_and_commit_bitwise():
    """The fallback may skip only the recurrent-state seed copy."""
    direct = _Harness(draft_tokens=T, eager_replay=False, seed=73)
    seeded = _Harness(draft_tokens=T, eager_replay=False, seed=73)
    for harness in (direct, seeded):
        harness.backend._verify = MethodType(MambaAttnBackend._verify, harness.backend)
    seeded.backend._verify_reads_committed_recurrent_state = False

    def seeded_verify_scan(
        self,
        query,
        key,
        value,
        ssm_comp,
        ssm_scratch,
        state_in_blocks,
        output_indices,
        **kwargs,
    ):
        del ssm_comp, state_in_blocks
        return KdaAttnBackend._verify_scan(
            self,
            query,
            key,
            value,
            ssm_scratch,
            ssm_scratch,
            output_indices[: kwargs["batch_size"], 0] - 1,
            output_indices,
            **kwargs,
        )

    seeded.backend._verify_scan = MethodType(seeded_verify_scan, seeded.backend)
    rpis = [0, 1, 2]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    generator = torch.Generator(device=DEV).manual_seed(79)
    for direct_layer, seeded_layer in zip(direct.layer_ids, seeded.layer_ids):
        direct_group = direct.pool.state_group_by_layer[direct_layer]
        seeded_group = seeded.pool.state_group_by_layer[seeded_layer]
        assert direct_group == seeded_group
        page_ids = torch.tensor(pages[direct_group], dtype=torch.int64, device=DEV)
        for component in ("conv_state", "recurrent_state"):
            direct_pool = direct.pool.get_component(direct_layer, component)
            seeded_pool = seeded.pool.get_component(seeded_layer, component)
            values = torch.randn(
                (len(page_ids), *direct_pool.shape[1:]),
                dtype=direct_pool.dtype,
                device=DEV,
                generator=generator,
            )
            direct_pool[page_ids] = values
            seeded_pool[page_ids] = values

    committed_before = {
        (layer_id, component): direct.pool.get_component(layer_id, component).clone()
        for layer_id in direct.layer_ids
        for component in ("conv_state", "recurrent_state")
    }

    seq_lens = [8 + T] * len(rpis)
    for harness in (direct, seeded):
        harness.prepare_metadata(rpis, pages, seq_lens)
    inputs = direct.inputs(len(rpis), 83)
    direct_out = direct.forward(inputs, len(rpis))
    seeded_out = seeded.forward(inputs, len(rpis))
    torch.cuda.synchronize()

    write_rows = direct.backend.forward_metadata.mamba_output_indices.long()
    for layer_id, actual, expected in zip(direct.layer_ids, direct_out, seeded_out):
        assert torch.equal(actual, expected)
        direct_conv, direct_state = direct.backend._verify_scratch[layer_id]
        seeded_conv, seeded_state = seeded.backend._verify_scratch[layer_id]
        assert torch.equal(direct_conv[write_rows], seeded_conv[write_rows])
        assert torch.equal(direct_state[write_rows], seeded_state[write_rows])
        for component in ("conv_state", "recurrent_state"):
            assert torch.equal(
                direct.pool.get_component(layer_id, component),
                committed_before[(layer_id, component)],
            )

    accepted = torch.tensor([1, T, 2], dtype=torch.int32, device=DEV)
    direct.backend.commit_verified_state(accepted, accepted_path=None)
    seeded.backend.commit_verified_state(accepted, accepted_path=None)
    torch.cuda.synchronize()
    for layer_id in direct.layer_ids:
        for component in ("conv_state", "recurrent_state"):
            assert torch.equal(
                direct.pool.get_component(layer_id, component),
                seeded.pool.get_component(layer_id, component),
            )


def test_packed_qkv_views_match_materialized_split_across_commits():
    """Packed views preserve verifier outputs and committed state over rounds."""
    rpis = [0, 1]
    viewed = _Harness(
        draft_tokens=T,
        eager_replay=False,
        seed=89,
        max_bs=len(rpis),
        usable_pages=8,
    )
    materialized = _Harness(
        draft_tokens=T,
        eager_replay=False,
        seed=89,
        max_bs=len(rpis),
        usable_pages=8,
    )
    for harness in (viewed, materialized):
        harness.backend._verify = MethodType(MambaAttnBackend._verify, harness.backend)
    materialized.backend._verify_packed_qkv_views = False

    # One layer per state group covers routing without recompiling all layers.
    for harness in (viewed, materialized):
        harness.layer_ids = [
            next(
                layer_id
                for layer_id in harness.layer_ids
                if harness.pool.state_group_by_layer[layer_id] == group_id
            )
            for group_id in _STATE_GROUPS
        ]

    pages = {
        group_id: [
            2 + group_index * len(rpis) + request_index
            for request_index in range(len(rpis))
        ]
        for group_index, group_id in enumerate(_STATE_GROUPS)
    }
    all_pages = [page for group_pages in pages.values() for page in group_pages]
    assert len(all_pages) == len(set(all_pages))
    page_ids = torch.tensor(all_pages, dtype=torch.int64, device=DEV)
    generator = torch.Generator(device=DEV).manual_seed(93)
    for viewed_layer, materialized_layer in zip(
        viewed.layer_ids, materialized.layer_ids, strict=True
    ):
        group_id = viewed.pool.state_group_by_layer[viewed_layer]
        assert group_id == materialized.pool.state_group_by_layer[materialized_layer]
        for component in ("conv_state", "recurrent_state"):
            viewed_state = viewed.pool.get_component(viewed_layer, component)
            viewed_state[page_ids] = torch.randn(
                (len(page_ids), *viewed_state.shape[1:]),
                dtype=viewed_state.dtype,
                device=DEV,
                generator=generator,
            )
            materialized.pool.get_component(materialized_layer, component).copy_(
                viewed_state
            )

    seq_lens = [8 + T] * len(rpis)
    accepts = ([0, T], [2, 1], [T, 0], [1, 2])
    for round_index, accepted in enumerate(accepts):
        for harness in (viewed, materialized):
            harness.prepare_metadata(rpis, pages, seq_lens)
        inputs = viewed.inputs(len(rpis), 97 + round_index)
        viewed_out = viewed.forward(inputs, len(rpis))
        materialized_out = materialized.forward(inputs, len(rpis))
        torch.cuda.synchronize()

        write_rows = viewed.backend.forward_metadata.mamba_output_indices.long()
        for viewed_layer, actual, expected in zip(
            viewed.layer_ids, viewed_out, materialized_out, strict=True
        ):
            assert torch.equal(actual, expected), f"layer {viewed_layer}"
            viewed_conv, viewed_state = viewed.backend._verify_scratch[viewed_layer]
            materialized_conv, materialized_state = (
                materialized.backend._verify_scratch[viewed_layer]
            )
            assert torch.equal(viewed_conv[write_rows], materialized_conv[write_rows])
            assert torch.equal(viewed_state[write_rows], materialized_state[write_rows])

        accepted_tensor = torch.tensor(accepted, dtype=torch.int32, device=DEV)
        viewed.backend.commit_verified_state(accepted_tensor, accepted_path=None)
        materialized.backend.commit_verified_state(accepted_tensor, accepted_path=None)
        torch.cuda.synchronize()
        for viewed_layer in viewed.layer_ids:
            for component in ("conv_state", "recurrent_state"):
                assert torch.equal(
                    viewed.pool.get_component(viewed_layer, component),
                    materialized.pool.get_component(viewed_layer, component),
                )
        seq_lens = [
            length + count for length, count in zip(seq_lens, accepted, strict=True)
        ]


def _reference_committed_states(harness, pages, inputs, accepted):
    """Evaluate accepted recurrence independently from each path's producers."""
    bs = len(accepted)
    width = harness.width
    raw = inputs["mixed_qkv"].float().view(bs, width, CONV_DIM)
    beta = inputs["beta_raw"].float().sigmoid().view(bs, width, H)
    expected = {}
    for layer_id in harness.layer_ids:
        params = harness.params[layer_id]
        ids = pages[harness.pool.state_group_by_layer[layer_id]]
        full_conv = harness.pool.get_component(layer_id, "conv_state")[ids].float()
        full_state = harness.pool.get_component(layer_id, "recurrent_state")[
            ids
        ].clone()
        conv = full_conv[:, :CONV_DIM]
        state = full_state[:, :H]
        weight = params["conv_weights"].float()
        raw_gate = inputs.get("g_raw")
        gate = (
            raw_gate.float()
            if raw_gate is not None
            else (
                inputs["f_a_out"].float()[:, None, :]
                * params["f_b_weight"].float()[None, :, :]
            ).sum(-1)
        )
        # Records verification uses activation-dtype convolution values,
        # including AMD's fused producer. The decomposed scratch path
        # does too; NVIDIA's fused scratch path retains FP32 intermediates.
        materialized_producers = (harness.backend._replay_active) or (
            not harness.backend._replay_active and torch.version.hip is not None
        )
        if materialized_producers:
            from tokenspeed_kernel.ops.attention.kda import kda_verify_conv_update

            if (
                raw_gate is None
                and harness.backend._replay_active
                and torch.version.hip
            ):
                # AMD's fused verify projects the gate in FP32 and rounds it
                # once to the activation dtype; a BLAS GEMM may round a few
                # channels the other way.
                gate = gate.to(inputs["f_a_out"].dtype).float()
            elif raw_gate is None:
                gate = torch.nn.functional.linear(
                    inputs["f_a_out"], params["f_b_weight"]
                ).float()
            if harness.backend._replay_active:
                produced_qkv = kda_verify_conv_update(
                    inputs["mixed_qkv"],
                    params["conv_weights"],
                    harness.pool.get_component(layer_id, "conv_state"),
                    torch.tensor(ids, device=DEV, dtype=torch.int32),
                    num_heads=H,
                    head_dim=D,
                    draft_token_num=width,
                    recurrent_layout="v_major",
                    bias=params.get("bias"),
                ).view(bs, width, CONV_DIM)
            else:
                from tokenspeed_kernel.ops.conv import causal_conv1d_update

                # The scratch path uses the common convolution producer,
                # whose intermediate rounding differs from the KDA producer.
                produced_qkv = causal_conv1d_update(
                    inputs["mixed_qkv"]
                    .view(bs, width, CONV_DIM)
                    .transpose(1, 2)
                    .clone(),
                    conv.to(inputs["mixed_qkv"].dtype).contiguous(),
                    params["conv_weights"],
                    bias=params.get("bias"),
                    activation="silu",
                    intermediate_conv_window=torch.empty(
                        bs,
                        width,
                        CONV_DIM,
                        3,
                        dtype=inputs["mixed_qkv"].dtype,
                        device=DEV,
                    ),
                    parent_indices=None,
                ).transpose(1, 2)
            produced_qkv = produced_qkv.float()
        gate = gate.view(bs, width, H, D) + params["dt_bias"].view(H, D)
        scale = params["A_log"].exp()[None, None, :, None]
        lower_bound = params["lower_bound"]
        log_decay = (
            -scale * torch.nn.functional.softplus(gate)
            if lower_bound is None
            else lower_bound * torch.sigmoid(scale * gate)
        )
        decay = log_decay.exp()
        states, windows = [], []
        for token in range(width):
            current = raw[:, token]
            if materialized_producers:
                qkv = produced_qkv[:, token]
            else:
                acc = current * weight[:, 3]
                for tap in range(3):
                    acc = torch.addcmul(acc, conv[:, :, tap], weight[:, tap])
                qkv = torch.nn.functional.silu(acc)
            _, key, value = (part.view(bs, H, D) for part in qkv.chunk(3, -1))
            key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            state = state * decay[:, token, :, None, :]
            correction = value - (state * key[:, :, None, :]).sum(-1)
            correction = correction * beta[:, token, :, None]
            state = state + correction[..., None] * key[:, :, None, :]
            conv = torch.cat((conv[:, :, 1:], current[..., None]), dim=-1)
            states.append(state.clone())
            windows.append(conv.clone())
        full_state[:, :H] = torch.stack(
            [states[max(1, count) - 1][i] for i, count in enumerate(accepted)]
        )
        full_conv[:, :CONV_DIM] = torch.stack(
            [windows[max(1, count) - 1][i] for i, count in enumerate(accepted)]
        )
        expected[layer_id] = full_state, full_conv
    return expected


def test_eager_replay_matches_scratch_over_multiple_rounds_and_layers():
    """Both paths commit their accepted prefix, with their producer precision."""
    replay = _Harness(draft_tokens=T, eager_replay=True, seed=17)
    scratch = _Harness(draft_tokens=T, eager_replay=False, seed=17)
    rpis = [0, 1, 2]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    accepts = ([0, 1, T], [T, 0, 2], [1, T, 0], [2, 1, T])
    for round_index, accepted in enumerate(accepts):
        for harness in (replay, scratch):
            inputs = harness.inputs(len(rpis), 100 + round_index)
            expected = _reference_committed_states(harness, pages, inputs, accepted)
            harness.round(rpis, pages, seq_lens, 100 + round_index, accepted)
            for layer_id, (state, conv) in expected.items():
                ids = pages[harness.pool.state_group_by_layer[layer_id]]
                torch.testing.assert_close(
                    harness.pool.get_component(layer_id, "recurrent_state")[ids],
                    state,
                    atol=2e-6,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    harness.pool.get_component(layer_id, "conv_state")[ids].float(),
                    conv,
                    atol=0,
                    rtol=0,
                )
        seq_lens = [length + max(1, count) for length, count in zip(seq_lens, accepted)]


def test_graph_replay_then_post_forward_commit_matches_eager_over_rounds():
    """The captured verify graph plus the existing post-forward hook follows eager."""
    captured = _Harness(draft_tokens=T, eager_replay=True, seed=29)
    eager = _Harness(draft_tokens=T, eager_replay=True, seed=29)
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
    payload = captured.backend._replay_payload(first_layer)[0]
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
    harness = _Harness(draft_tokens=draft_tokens, eager_replay=True)
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
        expected_bytes=planned_bytes,
    )


def test_replay_planning_matches_allocation_and_rejects_drift():
    """The cache recipe and startup allocator share one byte contract."""
    harness = _Harness(draft_tokens=T, eager_replay=True)
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
    config = SimpleNamespace(max_bs=8, qcp_size=1)
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
            expected_bytes=planned_bytes + 1,
        )


def test_descriptor_binding_rejects_nonuniform_conv_width():
    harness = _Harness(draft_tokens=T, eager_replay=True)
    last = harness.layer_ids[-1]
    harness.prepare_metadata([0], {group: [2] for group in _STATE_GROUPS}, [8 + T])
    harness.forward(harness.inputs(1, 701), 1)
    # Verify kernels reject invalid conv widths before descriptor binding.
    # Rebind directly to exercise the batched descriptor's geometry check.
    weights = harness.backend._replay_weights[last]
    weights = (weights[0][:, :3], *weights[1:])
    harness.backend._replay_weights[last] = weights
    harness.backend._replay_descriptor_bound.remove(last)
    with pytest.raises(RuntimeError, match="uniform geometry"):
        harness.backend._bind_replay_descriptor(last, weights)


def test_equal_geometry_pool_replacement_rebinds_batched_replay():
    harness = _Harness(draft_tokens=T, eager_replay=True)
    pages = {group: [2] for group in _STATE_GROUPS}
    harness.prepare_metadata([0], pages, [8 + T])
    harness.forward(harness.inputs(1, 711), 1)
    assert harness.backend._batched_replay_ready

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
    assert harness.backend._batched_replay_ready


def test_uneven_state_groups_bind_batched_replay():
    """Replay uses each layer's cache group instead of equal-size partitions."""
    harness = _Harness(draft_tokens=T, eager_replay=True)
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
    assert harness.backend._batched_replay_ready
    assert harness.backend._replay_group_indices.tolist() == expected
    assert expected.count(0) != expected.count(1)


def test_verify_scratch_cannot_grow_after_preallocation():
    """Scratch is allocated once; any later capacity overrun fails loudly."""
    harness = _Harness(draft_tokens=T, eager_replay=True)
    harness.backend.preallocate_verify_workspace(harness.backend.max_bs, T)
    capacity = next(iter(harness.backend._verify_scratch.values()))[0].shape[0]
    with pytest.raises(RuntimeError, match="preallocated scratch holds"):
        harness.backend._ensure_verify_scratch(capacity + 1, T)


def test_no_store_fused_verify_matches_decomposed_outputs():
    """Replay fusion agrees with the shared conv/gate/recurrent verify path."""
    from types import MethodType

    fused = _Harness(draft_tokens=T, eager_replay=True, seed=61)
    decomposed = _Harness(draft_tokens=T, eager_replay=False, seed=61)
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
    group_id = decomposed.pool.state_group_by_layer[layer_id]
    recurrent = decomposed.pool.get_component(layer_id, "recurrent_state")
    recurrent[torch.tensor(pages[group_id], device=DEV)] += 1
    wrong = decomposed.forward(inputs, len(rpis))[0]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(fused_out[0], wrong, atol=2e-2, rtol=2e-2)


def test_rebinding_the_pool_reissues_the_raw_gate_scratch(monkeypatch):
    """Rebinding drops the previous records workspace and its references to the old pool."""
    harness = _Harness(draft_tokens=T, eager_replay=True)
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
    assert harness.backend._verify_scratch is None
    harness.backend.preallocate_verify_workspace(harness.config.max_bs, T)
    assert_no_alias(harness.backend, old_pool)
    for layer_id in harness.layer_ids:
        rows = harness.backend._verify_scratch[layer_id][0]
        assert rows.device == replacement.get_component(layer_id, "conv_state").device
        assert rows.shape[0] >= harness.config.max_bs


def test_rebinding_the_pool_commits_like_a_fresh_bind():
    """Verify/commit after a rebind lands in the new pool exactly as a fresh bind's does."""
    rebound = _Harness(draft_tokens=T, eager_replay=True)
    fresh = _Harness(draft_tokens=T, eager_replay=True)
    rpis = [0, 1]
    pages = {
        group_id: [2 + group * len(rpis) + i for i in range(len(rpis))]
        for group, group_id in enumerate(_STATE_GROUPS)
    }
    seq_lens = [8 + T] * len(rpis)
    rebound.round(rpis, pages, seq_lens, 100, [1, T])

    replacement = _make_kimi_pool(DEV, usable_pages=24)
    rebound.backend.set_kv_pool(replacement)
    rebound.backend.init_cuda_graph_state(rebound.config.max_bs)
    rebound.backend.preallocate_verify_workspace(rebound.config.max_bs, rebound.width)
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
            assert harness.pool.get_component(layer_id, component)[page].abs().sum() > 0


def test_reordered_state_groups_are_the_same_geometry():
    """Group ids name the groups; the contract's enumeration order is not geometry."""
    harness = _Harness(draft_tokens=T, eager_replay=False)
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

    harness = _Harness(draft_tokens=T, eager_replay=False)
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
    harness = _Harness(draft_tokens=T, eager_replay=False)
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


def test_a_rebound_backend_matches_a_fresh_one_field_by_field():
    """After the documented re-initialisation, no attribute tells a rebind from a fresh bind."""
    rebound = _Harness(draft_tokens=T, eager_replay=True)
    fresh = _Harness(draft_tokens=T, eager_replay=True)
    pages = {group: [2] for group in _STATE_GROUPS}
    rebound.round([0], pages, [8 + T], 100, [1])
    old_slabs = storages_of(
        *(
            rebound.backend._state_components(layer_id)[component]
            for layer_id in rebound.backend._state_layer_ids()
            for component in (0, 1)
        )
    )

    rebound.backend.set_kv_pool(_make_kimi_pool(DEV, usable_pages=24))
    for harness in (rebound, fresh):
        harness.backend.init_cuda_graph_state(harness.config.max_bs)
        harness.backend.preallocate_verify_workspace(harness.config.max_bs, T)

    rebound_state = binding_state(rebound.backend)
    fresh_state = binding_state(fresh.backend)
    # Tapes record raw device addresses, which differ between any two backends.
    assert rebound.backend._replay_state_tapes.keys() == (
        fresh.backend._replay_state_tapes.keys()
    )
    rebound_state.pop("_replay_state_tapes")
    fresh_state.pop("_replay_state_tapes")
    assert rebound_state == fresh_state
    assert_no_alias(rebound.backend, old_slabs)


def test_a_pool_too_small_for_the_raw_gate_scratch_is_refused(monkeypatch):
    """Records workspace is independent of the small cache used during graph probing."""
    harness = _Harness(draft_tokens=T, eager_replay=True)
    harness.backend.set_kv_pool(_make_kimi_pool(DEV, usable_pages=4))

    harness.backend.preallocate_verify_workspace(harness.config.max_bs, T)
    for layer_id in harness.layer_ids:
        conv = harness.backend._state_components(layer_id)[0]
        scratch = harness.backend._verify_scratch[layer_id][0]
        assert scratch.shape[0] >= harness.config.max_bs
        assert scratch.untyped_storage().data_ptr() != conv.untyped_storage().data_ptr()


def test_hybrid_rebinding_reaches_the_state_child():
    harness = _Harness(draft_tokens=T, eager_replay=False)
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
    captured = _Harness(draft_tokens=T, eager_replay=True, seed=982)
    eager = _Harness(draft_tokens=T, eager_replay=True, seed=982)
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
        for layer_id in captured.layer_ids:
            for component in ("conv_state", "recurrent_state"):
                torch.testing.assert_close(
                    captured.pool.get_component(layer_id, component),
                    eager.pool.get_component(layer_id, component),
                    atol=0 if component == "conv_state" else 1e-5,
                    rtol=0 if component == "conv_state" else 1e-3,
                )
        seq_lens = [length + count for length, count in zip(seq_lens, counts)]


def test_commit_plan_resolves_pages_when_packed_storage_moves():
    """Resolve the new scheduler pages after replacing packed table storage."""
    harness = _Harness(
        draft_tokens=T, eager_replay=True, seed=173, max_bs=2, usable_pages=24
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
        draft_tokens=T, eager_replay=True, seed=174, max_bs=batch, usable_pages=24
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


@pytest.mark.parametrize("draft_tokens", [None, 1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_decode_records_commit_after_acceptance(draft_tokens, captured):
    """Verify preserves live state; acceptance commits only the selected prefixes."""
    actual = _Harness(draft_tokens=draft_tokens, eager_replay=True, seed=431)
    eager = _Harness(draft_tokens=draft_tokens, eager_replay=True, seed=431)
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
        expected_bytes=planned_bytes,
    )

    def workspace_tensors():
        backend = actual.backend
        return tuple(
            tensor
            for tensor in (
                *backend._replay_payloads,
                backend._replay_descriptors,
                backend._replay_group_indices,
                backend._replay_commit_tables,
                backend._replay_commit_reads,
                backend._replay_commit_pages,
                backend._replay_commit_steps,
                backend._replay_record_metadata,
                backend._replay_record_stamps,
                *(pair[0] for pair in backend._verify_scratch.values()),
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

    def assert_live_state_unchanged(snapshots):
        for (layer, component), snapshot in snapshots.items():
            torch.testing.assert_close(
                actual.pool.get_component(layer, component), snapshot, atol=0, rtol=0
            )

    before = {
        (layer, component): actual.pool.get_component(layer, component).clone()
        for layer in actual.layer_ids
        for component in ("conv_state", "recurrent_state")
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
            for component in ("conv_state", "recurrent_state")
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
                actual.pool.get_component(layer, "recurrent_state")[ids],
                state,
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                actual.pool.get_component(layer, "conv_state")[ids].float(),
                conv,
                atol=0,
                rtol=0,
            )
            for component in ("conv_state", "recurrent_state"):
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
    )
    torch.testing.assert_close(actual.float(), expected, atol=1e-4, rtol=8e-3)


@pytest.mark.parametrize("draft_tokens", [None, 4])
@pytest.mark.parametrize("lower_bound", [None, _LOWER_BOUND])
def test_record_decode_preserves_raw_gate_bias_and_layer_subset(
    draft_tokens, lower_bound
):
    harness = _Harness(draft_tokens=draft_tokens, eager_replay=True, seed=901)
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
                harness.pool.get_component(layer, "recurrent_state")[ids],
                state,
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                harness.pool.get_component(layer, "conv_state")[ids].float(),
                conv,
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("draft_tokens", [None, 4])
def test_mixed_decode_commit_matches_separate_forwards(draft_tokens):
    # Prefill consumes the full cache field, including every per-rank head.
    class _FullWidthHarness(_Harness):
        H = 12

    actual = _FullWidthHarness(draft_tokens=draft_tokens, eager_replay=True, seed=941)
    reference = _FullWidthHarness(
        draft_tokens=draft_tokens, eager_replay=True, seed=941
    )
    width, prompt = actual.width, 3
    pages = {group: [2 + 2 * i, 3 + 2 * i] for i, group in enumerate(_STATE_GROUPS)}
    inputs = {
        name: value[: prompt + width]
        for name, value in actual.inputs(prompt + width, 943).items()
    }

    def metadata(harness, *, mixed):
        bs = 2 if mixed else 1
        lengths = torch.tensor([prompt, 8 + width][:bs], device=DEV, dtype=torch.int32)
        new_cpu = torch.tensor([prompt], dtype=torch.int32)
        prefix_cpu = torch.zeros(1, dtype=torch.int32)
        tables = {
            group: np.asarray([[0, page] for page in ids[:bs]], dtype=np.int32)
            for group, ids in pages.items()
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
            extend_replay_lens_cpu=prefix_cpu,
            extend_prompt_lens_cpu=new_cpu,
            extend_with_prefix=False,
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

    metadata(actual, mixed=True)
    actual_outputs = forward(actual, inputs, batch=2, mode=ForwardMode.MIXED)
    count = max(1, width - 1)
    actual.backend.commit_state_after_acceptance(
        torch.tensor([1, count], device=DEV, dtype=torch.int32),
        num_extends=1,
        accepted_path=None,
    )
    metadata(reference, mixed=False)
    prompt_outputs = forward(
        reference,
        {name: value[:prompt] for name, value in inputs.items()},
        batch=1,
        mode=ForwardMode.EXTEND,
    )
    reference.prepare_metadata(
        [1], {group: ids[1:] for group, ids in pages.items()}, [8 + width]
    )
    decode_outputs = reference.forward(
        {name: value[prompt:] for name, value in inputs.items()}, 1
    )
    reference.backend.commit_state_after_acceptance(
        torch.tensor([count], device=DEV, dtype=torch.int32),
        num_extends=0,
        accepted_path=None,
    )
    for layer, output, prefill, decode in zip(
        actual.layer_ids, actual_outputs, prompt_outputs, decode_outputs, strict=True
    ):
        expected = torch.cat((prefill, decode.reshape(-1, actual.H, D)))
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
        for component in ("conv_state", "recurrent_state"):
            torch.testing.assert_close(
                actual.pool.get_component(layer, component),
                reference.pool.get_component(layer, component),
                atol=0,
                rtol=0,
            )
