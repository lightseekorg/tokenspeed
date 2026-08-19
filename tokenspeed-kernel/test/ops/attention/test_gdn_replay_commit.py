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
from tokenspeed_kernel.ops.attention import (
    gdn_decode_mtp,
    gdn_replay_commit,
)
from tokenspeed_kernel.registry import KernelRegistry


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
    """Token-by-token GDN state oracle; all arithmetic is FP32."""
    batch, draft_tokens, num_k_heads, _ = k.shape
    num_v_heads = v.shape[2]
    group = num_v_heads // num_k_heads
    state = initial.float().transpose(-2, -1).clone()  # [B, HV, K, V]
    head_map = torch.arange(num_v_heads, device=k.device) // group
    for step in range(draft_tokens):
        key = k[:, step].float()
        key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        key = key.index_select(1, head_map)
        x = a[:, step].float() + dt_bias.float()
        softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
        decay = torch.exp(-torch.exp(A_log.float()) * softplus)
        beta = torch.sigmoid(b[:, step].float())
        next_state = state * decay[..., None, None]
        predicted = torch.einsum("bhk,bhkv->bhv", key, next_state)
        delta = beta[..., None] * (v[:, step].float() - predicted)
        next_state = next_state + key[..., None] * delta[..., None, :]
        active = (accepted > step).view(batch, 1, 1, 1)
        state = torch.where(active, next_state, state)
    return state.transpose(-2, -1)


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
    k = torch.randn(
        batch,
        draft_tokens,
        num_k_heads,
        head_k_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        batch,
        draft_tokens,
        num_v_heads,
        head_v_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    a = torch.randn(
        batch,
        draft_tokens,
        num_v_heads,
        device=device,
        dtype=torch.bfloat16,
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


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dims", [(32, 24), (128, 65), (128, 128)])
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
        rtol=2e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(pool[read.long()], original[read.long()])
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in set(write.tolist())],
        device=device,
    )
    torch.testing.assert_close(pool[untouched], original[untouched])


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
