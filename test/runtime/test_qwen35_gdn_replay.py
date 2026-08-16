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

"""Qwen3.5 GDN ReplaySSM integration through MambaAttnBackend."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``
# absolute imports) both need to be importable when run_ci_suite executes
# this file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.test_gdn_state_paging import _CacheMetadata, _ContractPool

from tokenspeed_kernel.ops.attention import gdn_replay_commit_supported

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
)

BATCH = 2
DRAFT_TOKENS = 3
NUM_K_HEADS = 2
NUM_V_HEADS = 4
HEAD_K_DIM = 32
HEAD_V_DIM = 24
CONV_WIDTH = 4
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
DEVICE = "cuda"


def _config(*, replay: bool):
    return SimpleNamespace(
        device=DEVICE,
        num_attention_heads=NUM_K_HEADS,
        num_kv_heads=NUM_K_HEADS,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=HEAD_K_DIM,
        is_draft=False,
        speculative_num_draft_tokens=DRAFT_TOKENS,
        replay_ssm=replay,
    )


def _make_backend(conv_state, recurrent_state, *, replay: bool):
    if replay and not gdn_replay_commit_supported(torch.bfloat16):
        pytest.skip("GDN ReplaySSM kernel unavailable on this platform")
    pool = _ContractPool(
        4,
        {0: ("linear_attention", conv_state, recurrent_state)},
    )
    backend = MambaAttnBackend(_config(replay=replay))
    backend.set_kv_pool(pool)
    return backend, pool


def _inputs(seed=11):
    torch.manual_seed(seed)
    return dict(
        mixed_qkv=torch.randn(
            BATCH * DRAFT_TOKENS,
            CONV_DIM,
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        a=torch.randn(
            BATCH * DRAFT_TOKENS,
            NUM_V_HEADS,
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        b=torch.randn(
            BATCH * DRAFT_TOKENS,
            NUM_V_HEADS,
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        conv_weights=torch.randn(
            CONV_DIM,
            CONV_WIDTH,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        * 0.1,
        A_log=torch.randn(NUM_V_HEADS, device=DEVICE, dtype=torch.float32) * 0.1,
        dt_bias=torch.randn(NUM_V_HEADS, device=DEVICE, dtype=torch.float32) * 0.1,
    )


def _forward_verify(backend, pool, inputs, *, layer_id=0):
    return backend.forward_decode(
        None,
        None,
        None,
        layer=None,
        out_cache_loc=None,
        token_to_kv_pool=pool,
        bs=BATCH,
        mixed_qkv=inputs["mixed_qkv"].clone(),
        conv_weights=inputs["conv_weights"],
        bias=None,
        activation="silu",
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        attention_tp_size=1,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        layer_id=layer_id,
        seq_len=BATCH * DRAFT_TOKENS,
    )


def _prepare_verify(backend, pool, inputs):
    tables = torch.tensor([[1, 5], [2, 6]], dtype=torch.int32, device=DEVICE)
    backend.init_forward_metadata(
        bs=BATCH,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        seq_lens=torch.tensor([7, 7], dtype=torch.int32, device=DEVICE),
        forward_mode=ForwardMode.DECODE,
        tokens_per_req=DRAFT_TOKENS,
        cache_metadata=_CacheMetadata({"linear_attention": tables}),
    )
    return _forward_verify(backend, pool, inputs)


def _initial_pools(seed=23, state_dtype=torch.float32):
    torch.manual_seed(seed)
    conv = (
        torch.randn(
            8,
            CONV_DIM,
            CONV_WIDTH - 1,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    recurrent = (
        torch.randn(
            8,
            NUM_V_HEADS,
            HEAD_V_DIM,
            HEAD_K_DIM,
            device=DEVICE,
            dtype=state_dtype,
        )
        * 0.02
    )
    return conv, recurrent


def test_qwen_verify_caches_kv_without_draft_recurrent_states():
    conv, recurrent = _initial_pools()
    backend, pool = _make_backend(conv, recurrent, replay=True)
    rows = BATCH * (DRAFT_TOKENS + 1)
    expected_conv_workspace = rows * conv[0].nbytes

    assert backend.preallocate_verify_workspace(BATCH, DRAFT_TOKENS) == (
        expected_conv_workspace
    )
    conv_scratch, recurrent_scratch = backend._verify_scratch[0]
    assert conv_scratch.shape[0] == rows
    assert recurrent_scratch is None

    before = recurrent.clone()
    inputs = _inputs()
    _prepare_verify(backend, pool, inputs)
    torch.cuda.synchronize()

    torch.testing.assert_close(recurrent, before)
    workspace = backend._gdn_replay
    assert workspace is not None
    payload = workspace.payload[0]
    a = payload[:, KEY_DIM + VALUE_DIM : -NUM_V_HEADS]
    b = payload[:, -NUM_V_HEADS:]
    key = payload[:, :KEY_DIM]
    value = payload[:, KEY_DIM : KEY_DIM + VALUE_DIM]
    assert key.shape == (BATCH * DRAFT_TOKENS, KEY_DIM)
    assert value.shape == (BATCH * DRAFT_TOKENS, VALUE_DIM)
    torch.testing.assert_close(a, inputs["a"])
    torch.testing.assert_close(b, inputs["b"])
    assert workspace.initialized_layers == {0}
    torch.testing.assert_close(workspace.parameters[0, 0], inputs["A_log"])
    torch.testing.assert_close(workspace.parameters[0, 1], inputs["dt_bias"])


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_qwen_replay_commit_matches_per_position_scratch_fallback(state_dtype):
    conv, recurrent = _initial_pools(state_dtype=state_dtype)
    replay_backend, replay_pool = _make_backend(
        conv.clone(), recurrent.clone(), replay=True
    )
    scratch_backend, scratch_pool = _make_backend(
        conv.clone(), recurrent.clone(), replay=False
    )
    inputs = _inputs()

    replay_out = _prepare_verify(replay_backend, replay_pool, inputs)
    scratch_out = _prepare_verify(scratch_backend, scratch_pool, inputs)
    accepted = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    replay_backend.commit_verified_state(accepted)
    scratch_backend.commit_verified_state(accepted)
    torch.cuda.synchronize()

    torch.testing.assert_close(replay_out, scratch_out, atol=0.0, rtol=0.0)
    replay_conv = replay_pool.get_component(0, "conv_state")
    scratch_conv = scratch_pool.get_component(0, "conv_state")
    replay_state = replay_pool.get_component(0, "recurrent_state")
    scratch_state = scratch_pool.get_component(0, "recurrent_state")
    committed_pages = torch.tensor([5, 6], device=DEVICE)
    torch.testing.assert_close(
        replay_conv[committed_pages],
        scratch_conv[committed_pages],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        replay_state[committed_pages],
        scratch_state[committed_pages],
        atol=1e-6,
        rtol=1e-5,
    )
    assert replay_backend._verify_commit_ctx is None


def test_qwen_replay_payload_and_commit_survive_cuda_graph_replay():
    conv, recurrent = _initial_pools()
    backend, pool = _make_backend(conv, recurrent, replay=True)
    backend.init_cuda_graph_state(BATCH)
    backend.preallocate_verify_workspace(BATCH, DRAFT_TOKENS)
    inputs = _inputs(seed=37)

    # Eager warmup compiles every kernel before stream capture.
    _prepare_verify(backend, pool, inputs)
    torch.cuda.synchronize()

    req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([7, 7], dtype=torch.int32, device=DEVICE)
    backend.init_forward_metadata_capture_cuda_graph(
        BATCH,
        req_pool_indices,
        seq_lens,
        ForwardMode.DECODE,
    )
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _forward_verify(backend, pool, inputs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _forward_verify(backend, pool, inputs)

    workspace = backend._gdn_replay
    assert workspace is not None
    payload = workspace.payload[0]
    payload_ptr = payload.data_ptr()
    captured_key = payload[:, :KEY_DIM].clone()
    tables = torch.tensor([[1, 5], [2, 6]], dtype=torch.int32, device=DEVICE)
    backend.init_forward_metadata_replay_cuda_graph(
        BATCH,
        req_pool_indices,
        seq_lens,
        ForwardMode.DECODE,
        cache_metadata=_CacheMetadata({"linear_attention": tables}),
    )
    inputs["mixed_qkv"].normal_(mean=0.5, std=0.2)
    inputs["a"].normal_(mean=-0.5, std=0.2)
    inputs["b"].normal_(mean=0.25, std=0.2)
    recurrent[torch.tensor([5, 6], device=DEVICE)].fill_(float("nan"))
    graph.replay()
    backend.commit_verified_state(
        torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    )
    torch.cuda.synchronize()

    assert payload.data_ptr() == payload_ptr
    assert not torch.equal(payload[:, :KEY_DIM], captured_key)
    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(recurrent[torch.tensor([5, 6], device=DEVICE)]).all())


def test_qwen_replay_commits_all_layers_with_one_kernel_call(monkeypatch):
    conv0, recurrent0 = _initial_pools(seed=73)
    conv1, recurrent1 = _initial_pools(seed=79)

    def make_backend(replay, conv_states, recurrent_states):
        pool = _ContractPool(
            4,
            {
                layer_id: (
                    "linear_attention",
                    conv_state,
                    recurrent_state,
                )
                for layer_id, (conv_state, recurrent_state) in enumerate(
                    zip(conv_states, recurrent_states)
                )
            },
        )
        if replay and not gdn_replay_commit_supported(torch.bfloat16):
            pytest.skip("GDN ReplaySSM kernel unavailable on this platform")
        backend = MambaAttnBackend(_config(replay=replay))
        backend.set_kv_pool(pool)
        return backend, pool

    replay_backend, replay_pool = make_backend(
        True,
        (conv0.clone(), conv1.clone()),
        (recurrent0.clone(), recurrent1.clone()),
    )
    scratch_backend, scratch_pool = make_backend(
        False,
        (conv0.clone(), conv1.clone()),
        (recurrent0.clone(), recurrent1.clone()),
    )
    inputs = (_inputs(seed=83), _inputs(seed=89))

    _prepare_verify(replay_backend, replay_pool, inputs[0])
    _forward_verify(replay_backend, replay_pool, inputs[1], layer_id=1)
    _prepare_verify(scratch_backend, scratch_pool, inputs[0])
    _forward_verify(scratch_backend, scratch_pool, inputs[1], layer_id=1)

    from tokenspeed.runtime.layers.attention.backends import (
        hybrid_linear_attn as backend_ops,
    )

    original = backend_ops.gdn_replay_commit
    launch_calls = 0

    def counted_commit(*args, **kwargs):
        nonlocal launch_calls
        launch_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(backend_ops, "gdn_replay_commit", counted_commit)
    accepted = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    replay_backend.commit_verified_state(accepted)
    scratch_backend.commit_verified_state(accepted)
    torch.cuda.synchronize()

    assert launch_calls == 1
    workspace = replay_backend._gdn_replay
    assert workspace is not None
    packed = workspace.payload
    assert packed.shape[:2] == (2, BATCH * DRAFT_TOKENS)
    assert packed.is_contiguous()

    committed_pages = torch.tensor([5, 6], device=DEVICE)
    for layer_id in (0, 1):
        torch.testing.assert_close(
            replay_pool.get_component(layer_id, "conv_state")[committed_pages],
            scratch_pool.get_component(layer_id, "conv_state")[committed_pages],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            replay_pool.get_component(layer_id, "recurrent_state")[committed_pages],
            scratch_pool.get_component(layer_id, "recurrent_state")[committed_pages],
            atol=1e-6,
            rtol=1e-5,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
