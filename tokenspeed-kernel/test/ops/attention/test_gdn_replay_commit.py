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

"""Correctness coverage for Qwen GDN ReplaySSM accepted-prefix commits."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.gdn import (
    gdn_gate,
    gdn_l2norm,
    torch_gdn_recurrence,
)
from tokenspeed_kernel.ops.attention.gdn import (
    gdn_decode_mtp,
    gdn_replay_commit,
)
from tokenspeed_kernel.ops.attention.gdn.triton import _gdn_replay_commit_kernel
from tokenspeed_kernel.registry import KernelRegistry
from utils import assert_no_triton_compile

_STATE_TOLERANCES = {
    torch.bfloat16: {"rtol": 2e-2, "atol": 1e-3},
    torch.float16: {"rtol": 1e-3, "atol": 1e-6},
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
}


def test_gdn_replay_commit_registration_stays_portable():
    spec = KernelRegistry.get().get_by_name("triton_gdn_replay_commit")

    assert spec is not None
    assert spec.solution == "triton"
    assert spec.capability.vendors == frozenset({"nvidia", "amd"})


def _reference_states(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial: torch.Tensor,
    accepted: torch.Tensor,
) -> torch.Tensor:
    """The K-last state after each request's accepted prefix; FP32 arithmetic."""
    key = gdn_l2norm(k.float(), 1e-6)
    _, states = torch_gdn_recurrence(
        key,
        key,
        v,
        gdn_gate(a.float(), dt_bias.float(), A_log.float()),
        torch.sigmoid(b.float()),
        initial,
        scale=1.0,
        parents=None,
        compute_dtype=torch.float32,
    )
    final = initial.float().clone()
    for batch, count in enumerate(accepted.clamp(max=k.shape[1]).tolist()):
        if count > 0:
            final[batch] = states[batch, count - 1]
    return final


def _inputs(
    device: str,
    state_dtype: torch.dtype,
    *,
    head_k_dim: int = 32,
    head_v_dim: int = 24,
    pool_size: int = 20,
):
    torch.manual_seed(17)
    batch, draft_tokens = 4, 4
    num_k_heads, num_v_heads = 2, 4
    input_dtype = torch.float16 if state_dtype == torch.float16 else torch.bfloat16
    k = torch.randn(
        batch,
        draft_tokens,
        num_k_heads,
        head_k_dim,
        device=device,
        dtype=input_dtype,
    )
    v = torch.randn(
        batch,
        draft_tokens,
        num_v_heads,
        head_v_dim,
        device=device,
        dtype=input_dtype,
    )
    a = torch.randn(
        batch,
        draft_tokens,
        num_v_heads,
        device=device,
        dtype=input_dtype,
    )
    b = torch.randn_like(a)
    A_log = torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
    pool = (
        torch.randn(
            pool_size,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        * 0.02
    )
    read = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int32)
    write = torch.tensor([8, 9, 10, 11], device=device, dtype=torch.int32)
    accepted = torch.tensor([0, 1, 3, 4], device=device, dtype=torch.int32)
    return k, v, a, b, A_log, dt_bias, pool, read, write, accepted


def _replay(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    pool: torch.Tensor,
    read: torch.Tensor,
    write: torch.Tensor,
    accepted: torch.Tensor,
) -> None:
    """Pack one layer into the same layer-major API used by the runtime."""
    batch, draft_tokens, num_k_heads, head_k_dim = k.shape
    num_v_heads, head_v_dim = v.shape[2:]
    payload = torch.cat(
        (
            k.reshape(batch, draft_tokens, -1),
            v.reshape(batch, draft_tokens, -1),
            a,
            b,
        ),
        dim=-1,
    ).reshape(1, batch * draft_tokens, -1)
    gdn_replay_commit(
        payload,
        torch.stack((A_log, dt_bias)).unsqueeze(0),
        state_addresses=torch.tensor(
            [pool.data_ptr()], dtype=torch.uint64, device=pool.device
        ),
        state_row_strides=torch.tensor(
            [pool.stride(0)], dtype=torch.int64, device=pool.device
        ),
        read_indices=read.unsqueeze(0),
        write_indices=write.unsqueeze(0),
        accepted_length=accepted,
        draft_token_num=draft_tokens,
        geometry=(num_k_heads, num_v_heads, head_k_dim, head_v_dim),
        state_dtype=pool.dtype,
        solution="triton",
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("head_dims", [(32, 24), (33, 65), (128, 65), (128, 128)])
def test_gdn_replay_commit_matches_accepted_prefix_reference(
    device: str,
    state_dtype: torch.dtype,
    head_dims: tuple[int, int],
    require,
):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, write, accepted = _inputs(
        device,
        state_dtype,
        head_k_dim=head_dims[0],
        head_v_dim=head_dims[1],
    )
    expected = _reference_states(
        k, v, a, b, A_log, dt_bias, pool[read.long()], accepted
    )
    original = pool.clone()

    _replay(
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool,
        read,
        write,
        accepted,
    )

    torch.testing.assert_close(
        pool[write.long()].float(),
        expected.to(state_dtype).float(),
        **_STATE_TOLERANCES[state_dtype],
    )
    torch.testing.assert_close(pool[read.long()], original[read.long()])
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in set(write.tolist())],
        device=device,
    )
    torch.testing.assert_close(pool[untouched], original[untouched])


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("storage_offset", [1, 4])
def test_gdn_replay_commit_preserves_unaligned_pool_padding(
    device: str, state_dtype: torch.dtype, storage_offset: int, require
):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, write, accepted = _inputs(
        device, state_dtype, head_k_dim=128, head_v_dim=65
    )
    row_elements = pool[0].numel()
    stride = row_elements + 1
    backing = torch.randn(
        pool.shape[0] * stride + storage_offset, device=device, dtype=state_dtype
    )
    padded_pool = torch.as_strided(
        backing,
        pool.shape,
        (stride, pool.stride(1), pool.stride(2), 1),
        storage_offset=storage_offset,
    )
    padded_pool.copy_(pool)
    write[1] = read[1]  # Cover in-place replay as well as separate destinations.
    accepted[0], accepted[-1] = -2, 9
    expected = _reference_states(
        k, v, a, b, A_log, dt_bias, padded_pool[read.long()], accepted
    )
    original = backing.clone()

    _replay(k, v, a, b, A_log, dt_bias, padded_pool, read, write, accepted)

    torch.testing.assert_close(
        padded_pool[write.long()].float(),
        expected.to(state_dtype).float(),
        **_STATE_TOLERANCES[state_dtype],
    )
    written = torch.zeros_like(backing, dtype=torch.bool)
    for row in write.tolist():
        start = storage_offset + row * stride
        written[start : start + row_elements] = True
    torch.testing.assert_close(backing[~written], original[~written], rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gdn_replay_commit_reuses_shared_chunks_without_reading_rejected_tokens(
    device: str, state_dtype: torch.dtype, require
):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, write, _ = _inputs(
        device, state_dtype, head_k_dim=128, head_v_dim=65
    )
    draft_tokens = 19
    k, v, a, b = (
        torch.randn(
            tensor.shape[0],
            draft_tokens,
            *tensor.shape[2:],
            device=device,
            dtype=tensor.dtype,
        )
        for tensor in (k, v, a, b)
    )
    accepted = torch.tensor([0, 1, 9, 23], dtype=torch.int32, device=device)
    k[:, 1, 0] = 0
    a[:, 0, 0] = 128  # Exercise the linear softplus branch.
    b[:, 0, 0] = -128
    for row, length in enumerate(accepted.tolist()):
        for tensor in (k, v, a, b):
            tensor[row, length:] = float("nan")
    expected = _reference_states(
        k, v, a, b, A_log, dt_bias, pool[read.long()], accepted
    )

    _replay(k, v, a, b, A_log, dt_bias, pool, read, write, accepted)

    torch.testing.assert_close(
        pool[write.long()].float(),
        expected.to(state_dtype).float(),
        **_STATE_TOLERANCES[state_dtype],
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_replay_commit_batch_window_and_capacity_do_not_recompile(
    device: str, state_dtype: torch.dtype, require
):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    H, HV, K, V = 2, 4, 128, 128
    width = H * K + HV * V + 2 * HV

    def commit(batch: int, tokens: int, capacity: int, layers: int):
        payload = torch.randn(
            layers, capacity, width, dtype=torch.bfloat16, device=device
        )
        pool = torch.zeros(
            layers, batch * 2, HV, V, K, dtype=state_dtype, device=device
        )
        read = torch.arange(batch, dtype=torch.int32, device=device).repeat(layers, 1)
        gdn_replay_commit(
            payload,
            torch.zeros(layers, 2, HV, device=device),
            state_addresses=torch.tensor(
                [pool[layer].data_ptr() for layer in range(layers)],
                dtype=torch.uint64,
                device=device,
            ),
            state_row_strides=torch.full(
                (layers,), HV * V * K, dtype=torch.int64, device=device
            ),
            read_indices=read,
            write_indices=read + batch,
            accepted_length=torch.full(
                (batch,), tokens, dtype=torch.int32, device=device
            ),
            draft_token_num=tokens,
            geometry=(H, HV, K, V),
            state_dtype=pool.dtype,
            solution="triton",
        )
        torch.cuda.synchronize()

    commit(batch=1, tokens=1, capacity=7, layers=2)
    with assert_no_triton_compile(_gdn_replay_commit_kernel):
        commit(batch=4, tokens=3, capacity=13, layers=3)
        commit(batch=17, tokens=16, capacity=281, layers=1)


@pytest.mark.parametrize("batch", [16, 32, 64])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_replay_commit_tp4_graph_refresh(
    device: str, batch: int, state_dtype: torch.dtype, require
):
    """TP4 head grouping preserves state as graph acceptance and padding change."""
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    torch.manual_seed(3109)
    layers, tokens, H, HV, K, V = 2, 4, 4, 12, 128, 128
    width = H * K + HV * V + 2 * HV
    payload = torch.randn(
        layers, batch * tokens, width, device=device, dtype=torch.bfloat16
    )
    original_payload = payload.clone()
    parameters = torch.randn(layers, 2, HV, device=device) * 0.1
    pool = (
        torch.randn(layers, batch * 2 + 3, HV, V, K, device=device, dtype=state_dtype)
        * 0.02
    )
    read = torch.arange(1, batch + 1, device=device, dtype=torch.int32).repeat(
        layers, 1
    )
    read[1].copy_(read[1].roll(1))
    read[:, 0] = -1
    write = torch.arange(
        batch + 1, batch * 2 + 1, device=device, dtype=torch.int32
    ).repeat(layers, 1)
    write[:, 1] = read[:, 1]  # Each in-place row belongs to only one request.
    write[:, -1] = -1
    accepted = torch.arange(batch, device=device, dtype=torch.int32) % (tokens + 1)
    addresses = torch.tensor(
        [pool[layer].data_ptr() for layer in range(layers)],
        device=device,
        dtype=torch.uint64,
    )
    strides = torch.full((layers,), pool.stride(1), device=device, dtype=torch.int64)

    def commit():
        gdn_replay_commit(
            payload,
            parameters,
            state_addresses=addresses,
            state_row_strides=strides,
            read_indices=read,
            write_indices=write,
            accepted_length=accepted,
            draft_token_num=tokens,
            geometry=(H, HV, K, V),
            state_dtype=state_dtype,
            solution="triton",
        )

    commit()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit()

    packed = payload.view(layers, batch, tokens, width)
    for offset in (0, 1):
        accepted.copy_((torch.arange(batch, device=device) + offset) % (tokens + 1))
        payload.copy_(original_payload)
        rejected = torch.arange(tokens, device=device)[None, :] >= accepted[:, None]
        packed[:, rejected] = float("nan")
        original = pool.clone()
        expected = []
        for layer in range(layers):
            initial = original[layer, read[layer].clamp_min(0).long()].clone()
            initial[read[layer] < 0] = 0
            expected.append(
                _reference_states(
                    packed[layer, :, :, : H * K].reshape(batch, tokens, H, K),
                    packed[layer, :, :, H * K : H * K + HV * V].reshape(
                        batch, tokens, HV, V
                    ),
                    packed[layer, :, :, H * K + HV * V : -HV],
                    packed[layer, :, :, -HV:],
                    parameters[layer, 0],
                    parameters[layer, 1],
                    initial,
                    accepted,
                )
            )

        graph.replay()
        torch.cuda.synchronize()

        for layer in range(layers):
            live = write[layer] >= 0
            torch.testing.assert_close(
                pool[layer, write[layer, live].long()].float(),
                expected[layer][live].to(state_dtype).float(),
                **_STATE_TOLERANCES[state_dtype],
            )
            written = torch.zeros(pool.shape[1], device=device, dtype=torch.bool)
            written[write[layer, live].long()] = True
            torch.testing.assert_close(
                pool[layer, ~written], original[layer, ~written], rtol=0, atol=0
            )


def test_gdn_replay_commit_ignores_rejected_suffix(device: str, require):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, write, accepted = _inputs(
        device, torch.float32
    )
    baseline = pool.clone()
    changed = pool.clone()

    _replay(
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        baseline,
        read,
        write,
        accepted,
    )

    k_changed, v_changed, a_changed, b_changed = (
        tensor.clone() for tensor in (k, v, a, b)
    )
    for row, length in enumerate(accepted.tolist()):
        k_changed[row, length:].normal_(mean=3.0, std=0.5)
        v_changed[row, length:].normal_(mean=-3.0, std=0.5)
        a_changed[row, length:].fill_(2.0)
        b_changed[row, length:].fill_(-2.0)
    _replay(
        k_changed,
        v_changed,
        a_changed,
        b_changed,
        A_log,
        dt_bias,
        changed,
        read,
        write,
        accepted,
    )

    torch.testing.assert_close(changed[write.long()], baseline[write.long()])


def test_gdn_replay_commit_handles_fresh_and_padding_rows(device: str, require):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, write, accepted = _inputs(
        device, torch.float32
    )
    read[0] = -1  # A fresh request starts from the logical zero state.
    write[-1] = -1  # CUDA graph padding must not write any pool row.
    accepted[0] = 2
    expected_initial = pool[read.clamp_min(0).long()].clone()
    expected_initial[0].zero_()
    expected = _reference_states(k, v, a, b, A_log, dt_bias, expected_initial, accepted)
    original = pool.clone()

    _replay(
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool,
        read,
        write,
        accepted,
    )

    torch.testing.assert_close(
        pool[write[:-1].long()].float(),
        expected[:-1].float(),
        rtol=2e-2,
        atol=3e-2,
    )
    # The padded request's original destination remains untouched.
    torch.testing.assert_close(pool[11], original[11])


def test_gdn_replay_commit_matches_flashinfer_qwen_geometry(device: str, require):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    require("attention", "gdn_decode_mtp", "flashinfer", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, _, accepted = _inputs(
        device,
        torch.float32,
        head_k_dim=128,
        head_v_dim=128,
        pool_size=32,
    )
    batch, draft_tokens = k.shape[:2]
    accepted = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    scratch_rows = torch.arange(
        8,
        8 + batch * draft_tokens,
        device=device,
        dtype=torch.int32,
    ).view(batch, draft_tokens)
    write = torch.arange(24, 24 + batch, device=device, dtype=torch.int32)
    flashinfer_pool = pool.clone()
    replay_pool = pool.clone()

    # Query affects only attention output, not the state recurrence, so K is a
    # sufficient query input for this state-only cross-backend comparison.
    gdn_decode_mtp(
        k,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=flashinfer_pool,
        initial_state_indices=read,
        output_state_indices=scratch_rows,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution="flashinfer",
        tree_ancestors=None,
    )
    _replay(
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        replay_pool,
        read,
        write,
        accepted,
    )
    expected = torch.stack(
        [flashinfer_pool[scratch_rows[row, accepted[row] - 1]] for row in range(batch)]
    )
    torch.testing.assert_close(
        replay_pool[write.long()], expected, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("head_dims", [(32, 24), (128, 128)])
def test_gdn_replay_commit_replays_disjoint_layer_pools_in_one_launch(
    device: str, require, head_dims: tuple[int, int]
):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    k, v, a, b, A_log, dt_bias, pool, read, write, accepted = _inputs(
        device,
        torch.float32,
        head_k_dim=head_dims[0],
        head_v_dim=head_dims[1],
    )
    batch, draft_tokens, num_k_heads, head_k_dim = k.shape
    num_v_heads, head_v_dim = v.shape[2:]
    num_layers = 3
    key_width = num_k_heads * head_k_dim
    value_width = num_v_heads * head_v_dim
    payload_width = key_width + value_width + 2 * num_v_heads
    payload = torch.empty(
        num_layers,
        batch * draft_tokens,
        payload_width,
        dtype=k.dtype,
        device=device,
    )
    parameters = torch.empty(
        num_layers, 2, num_v_heads, dtype=torch.float32, device=device
    )
    pools = []
    reads = []
    writes = []
    expected = []
    for layer in range(num_layers):
        layer_k = (k.float() + layer * 0.05).to(k.dtype)
        layer_v = (v.float() - layer * 0.04).to(v.dtype)
        layer_a = (a.float() + layer * 0.03).to(a.dtype)
        layer_b = (b.float() - layer * 0.02).to(b.dtype)
        layer_A_log = A_log + layer * 0.01
        layer_dt_bias = dt_bias - layer * 0.01
        state_row_elements = pool[0].numel()
        state_row_stride = state_row_elements + 17 + layer
        state_backing = torch.empty(
            pool.shape[0] * state_row_stride,
            dtype=pool.dtype,
            device=device,
        )
        layer_pool = torch.as_strided(
            state_backing,
            size=pool.shape,
            stride=(
                state_row_stride,
                head_v_dim * head_k_dim,
                head_k_dim,
                1,
            ),
        )
        layer_pool.copy_(pool + layer * 0.005)
        layer_read = (read + layer).remainder(6) + 1
        layer_write = write + layer

        packed = payload[layer]
        packed[:, :key_width].copy_(layer_k.reshape(batch * draft_tokens, -1))
        packed[:, key_width : key_width + value_width].copy_(
            layer_v.reshape(batch * draft_tokens, -1)
        )
        packed[:, key_width + value_width : -num_v_heads].copy_(
            layer_a.reshape(batch * draft_tokens, -1)
        )
        packed[:, -num_v_heads:].copy_(layer_b.reshape(batch * draft_tokens, -1))
        parameters[layer, 0].copy_(layer_A_log)
        parameters[layer, 1].copy_(layer_dt_bias)
        expected.append(
            _reference_states(
                layer_k,
                layer_v,
                layer_a,
                layer_b,
                layer_A_log,
                layer_dt_bias,
                layer_pool[layer_read.long()],
                accepted,
            )
        )
        pools.append(layer_pool)
        reads.append(layer_read)
        writes.append(layer_write)

    state_addresses = torch.tensor(
        [layer_pool.data_ptr() for layer_pool in pools],
        dtype=torch.uint64,
        device=device,
    )
    state_row_strides = torch.tensor(
        [layer_pool.stride(0) for layer_pool in pools],
        dtype=torch.int64,
        device=device,
    )
    gdn_replay_commit(
        payload,
        parameters,
        state_addresses=state_addresses,
        state_row_strides=state_row_strides,
        read_indices=torch.stack(reads),
        write_indices=torch.stack(writes),
        accepted_length=accepted,
        draft_token_num=draft_tokens,
        geometry=(num_k_heads, num_v_heads, head_k_dim, head_v_dim),
        state_dtype=torch.float32,
        solution="triton",
    )

    for layer_pool, layer_write, layer_expected in zip(pools, writes, expected):
        torch.testing.assert_close(
            layer_pool[layer_write.long()],
            layer_expected,
            rtol=2e-2,
            atol=3e-2,
        )


def test_gdn_replay_commit_batch_heads_exceed_grid_z_limit(device: str, require):
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    torch.manual_seed(23)
    batch, draft_tokens, num_k_heads, num_v_heads = 8200, 2, 2, 8
    head_k_dim, head_v_dim = 32, 16
    assert batch * num_v_heads > 65535
    k = torch.randn(batch, draft_tokens, num_k_heads, head_k_dim, device=device)
    v = torch.randn(batch, draft_tokens, num_v_heads, head_v_dim, device=device)
    k, v = k.bfloat16(), v.bfloat16()
    a = torch.randn(batch, draft_tokens, num_v_heads, device=device).bfloat16()
    b = torch.randn_like(a)
    A_log = torch.randn(num_v_heads, device=device) * 0.1
    dt_bias = torch.randn(num_v_heads, device=device) * 0.1
    pool = torch.randn(
        2 * batch, num_v_heads, head_v_dim, head_k_dim, device=device
    ).mul_(0.02)
    read = torch.arange(batch, device=device, dtype=torch.int32)
    write = read + batch
    accepted = torch.randint(0, draft_tokens + 1, (batch,), device=device)
    accepted = accepted.to(torch.int32)
    expected = _reference_states(
        k, v, a, b, A_log, dt_bias, pool[read.long()], accepted
    )

    _replay(k, v, a, b, A_log, dt_bias, pool, read, write, accepted)

    torch.testing.assert_close(pool[write.long()], expected, rtol=2e-2, atol=3e-2)


def test_gdn_replay_commit_layer_offsets_beyond_int32(device: str, require):
    """Qwen3.8 TP1 payload [48 layers, 88 x 64 rows, 8288]: the last layer's base
    is past 2**31 elements and must commit like a payload holding only it."""
    require("attention", "gdn_replay_commit", "triton", torch.bfloat16, "q")
    L, H, HV, K, V, T, B, rows = 48, 16, 48, 128, 128, 64, 2, 88 * 64
    width = H * K + HV * V + 2 * HV
    assert (L - 1) * rows * width > 2**31
    gen = torch.Generator(device=device).manual_seed(0)
    payload = torch.empty(L, rows, width, dtype=torch.bfloat16, device=device)
    payload[:, : B * T] = torch.randn(
        L, B * T, width, generator=gen, device=device
    ).bfloat16()
    parameters = torch.randn(L, 2, HV, generator=gen, device=device)
    pools = torch.randn(L, 4, HV, V, K, generator=gen, device=device) * 0.1
    initial = pools.clone()

    def commit(payload, parameters, pools):
        layers = payload.shape[0]
        gdn_replay_commit(
            payload,
            parameters,
            state_addresses=torch.tensor(
                [pools[i].data_ptr() for i in range(layers)],
                dtype=torch.uint64,
                device=device,
            ),
            state_row_strides=torch.full(
                (layers,), HV * V * K, dtype=torch.int64, device=device
            ),
            read_indices=torch.tensor(
                [[0, 1]] * layers, dtype=torch.int32, device=device
            ),
            write_indices=torch.tensor(
                [[2, 3]] * layers, dtype=torch.int32, device=device
            ),
            accepted_length=torch.tensor([5, 7], dtype=torch.int32, device=device),
            draft_token_num=T,
            geometry=(H, HV, K, V),
            state_dtype=torch.float32,
            solution="triton",
        )

    commit(payload, parameters, pools)
    last = initial[L - 1 :].clone()
    commit(payload[L - 1 :, : B * T].contiguous(), parameters[L - 1 :], last)
    assert torch.equal(pools[L - 1, 2:4], last[0, 2:4])
