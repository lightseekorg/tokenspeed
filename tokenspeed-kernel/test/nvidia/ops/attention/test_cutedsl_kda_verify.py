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

"""The CuTe DSL KDA verify through the dispatch facade.

On sm100/sm103 the no-store verify must produce what sequential decode outputs
at every draft position, fill the raw-gate replay payload the batched replay
consumes, and leave the committed state and convolution pools untouched. Its
inputs are column slices of one projection row, as in the model.
"""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.kda import torch_kda_conv_decode
from tokenspeed_kernel.platform import ArchVersion, current_platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
    pytest.skip("CuTe DSL KDA verify targets sm100/sm103", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda import (  # noqa: E402
    kda_batched_replay_uses_raw_gate,
    kda_fused_paged_verify_uses_split_producers,
    resolve_kda_batched_replay_commit,
    try_kda_fused_paged_verify,
)
from tokenspeed_kernel.ops.attention.kda._cute_dsl import verify  # noqa: E402

HV, K, D_FA = 4, 128, 128
P = HV * K
# Projection row: q|k|v, f_a, beta, padding to a 16 B multiple.
WIDTH = 3 * P + D_FA + HV + 4
LOWER_BOUND = -5.0
PAGES = 16


def _inputs(n, t, seed, state_dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to("cuda", dtype)

    projection = rnd(n * t, WIDTH)
    x = dict(
        qkv=projection[:, : 3 * P],
        f_a=projection[:, 3 * P : 3 * P + D_FA],
        beta=projection[:, 3 * P + D_FA : 3 * P + D_FA + HV],
        conv_w=rnd(3 * P, 4, scale=0.3),
        conv=rnd(PAGES, 3 * P, 3),
        w_fb=rnd(P, D_FA, scale=0.05),
        A_log=rnd(HV, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(P, dtype=torch.float32),
        state=rnd(PAGES, HV, K, K, dtype=torch.float32).to(state_dtype),
    )
    rows = n * t
    x["payload"] = dict(
        qkv=torch.full(
            (rows, 3 * P), float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        gate=torch.full((rows, P), float("nan"), dtype=torch.bfloat16, device="cuda"),
        beta=torch.full((rows, HV), float("nan"), dtype=torch.bfloat16, device="cuda"),
    )
    return x


def _verify(x, reads, t):
    return try_kda_fused_paged_verify(
        x["qkv"],
        x["conv_w"],
        x["conv"],
        x["conv"],
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        state_pool=x["state"],
        state_scratch=None,
        read_indices=reads,
        write_indices=reads,
        num_heads=HV,
        head_dim=K,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
        store_states=False,
        recurrent_layout="v_major",
        replay_mixed_qkv=x["payload"]["qkv"],
        replay_gate=x["payload"]["gate"],
        replay_beta=x["payload"]["beta"],
    )


def _reference_outputs(x, reads, t, gate):
    """FP64 sequential decode over every draft position: ``[rows, HV, K]``."""
    width = 3 * P
    w = x["conv_w"].double()
    exp_a = torch.exp(x["A_log"].double())[:, None]
    out = []
    for request, page in enumerate(reads.tolist()):
        if page >= 0:
            window = x["conv"][page].double()
            h = x["state"][page].double().transpose(-1, -2)
        else:
            window = torch.zeros(width, 3, dtype=torch.float64, device="cuda")
            h = torch.zeros(HV, K, K, dtype=torch.float64, device="cuda")
        for i in range(t):
            row = request * t + i
            raw = x["qkv"][row].double()
            acc = (window * w[:, :3]).sum(-1) + raw * w[:, 3]
            y = acc * torch.sigmoid(acc)
            window = torch.cat([window[:, 1:], raw[:, None]], dim=1)
            q, k, v = (y[j * P : (j + 1) * P].view(HV, K) for j in range(3))
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            g = (gate[row].double() + x["dt_bias"].double()).view(HV, K)
            h = h * torch.exp(LOWER_BOUND * torch.sigmoid(exp_a * g))[:, :, None]
            residual = v - torch.einsum("hkv,hk->hv", h, k)
            residual = residual * torch.sigmoid(x["beta"][row].double())[:, None]
            h = h + torch.einsum("hk,hv->hkv", k, residual)
            out.append(K**-0.5 * torch.einsum("hk,hkv->hv", q, h))
    return torch.stack(out)


def test_blackwell_pairs_the_fused_verify_with_the_raw_gate_replay():
    assert not kda_fused_paged_verify_uses_split_producers(
        torch.bfloat16,
        store_states=False,
        recurrent_layout="v_major",
        num_heads=HV,
        head_dim=K,
    )
    assert kda_batched_replay_uses_raw_gate(torch.bfloat16, num_heads=HV, head_dim=K)
    x = _inputs(2, 2, seed=0, state_dtype=torch.bfloat16)
    reads = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    _verify(x, reads, 2)
    # Only the CuTe verify writes the replay payload.
    assert not x["payload"]["qkv"].isnan().any()


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("t", [2, 4, 6])
def test_outputs_match_sequential_decode_and_fill_the_payload(state_dtype, t):
    n = 5
    x = _inputs(n, t, seed=t, state_dtype=state_dtype)
    reads = torch.tensor([3, -1, 7, 1, 12], dtype=torch.int32, device="cuda")
    state0, conv0 = x["state"].clone(), x["conv"].clone()

    out = _verify(x, reads, t)

    payload = x["payload"]
    torch.testing.assert_close(payload["qkv"], x["qkv"], atol=0, rtol=0)
    torch.testing.assert_close(payload["beta"], x["beta"], atol=0, rtol=0)
    torch.testing.assert_close(
        payload["gate"], x["f_a"] @ x["w_fb"].t(), atol=0, rtol=0
    )
    reference = _reference_outputs(x, reads, t, payload["gate"])
    assert out.shape == (1, n * t, HV, K) and out.dtype == torch.bfloat16
    # FP32 recurrence vs FP64 on identical inputs; the output is rounded to
    # BF16 once.
    torch.testing.assert_close(
        out[0].double(), reference, atol=1e-5, rtol=torch.finfo(torch.bfloat16).eps
    )
    torch.testing.assert_close(x["state"], state0, atol=0, rtol=0)
    torch.testing.assert_close(x["conv"], conv0, atol=0, rtol=0)


def test_the_payload_replays_to_sequential_decode():
    """Verify, then the batched replay from its payload, commits what
    ``accepted`` ordinary decode steps produce."""
    n, t = 4, 4
    x = _inputs(n, t, seed=11, state_dtype=torch.float32)
    reads = torch.tensor([2, -1, 5, 9], dtype=torch.int32, device="cuda")
    writes = torch.tensor([[13, 14, 15, -1]], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([4, 2, 0, 3], dtype=torch.int32, device="cuda")
    _verify(x, reads, t)

    payload = x["payload"]
    columns = (
        payload["qkv"],
        x["conv_w"],
        x["conv"],
        x["f_a"],
        x["w_fb"],
        payload["beta"],
        x["A_log"],
        x["dt_bias"],
        x["state"],
        payload["gate"],
    )
    descriptors = torch.tensor(
        [[c.data_ptr() for c in columns]], dtype=torch.uint64, device="cuda"
    )
    replay = resolve_kda_batched_replay_commit()
    replay(
        descriptors=descriptors,
        group_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        read_indices=reads.view(1, n),
        write_indices=writes,
        accepted_length=accepted,
        draft_token_num=t,
        num_heads=HV,
        head_dim=K,
        f_a_dim=D_FA,
        qkv_stride=payload["qkv"].stride(0),
        conv_stride=x["conv"].stride(0),
        f_a_stride=x["f_a"].stride(0),
        beta_stride=payload["beta"].stride(0),
        state_stride=x["state"].stride(0),
        state_dtype=x["state"].dtype,
        gate_stride=payload["gate"].stride(0),
        conv_width=4,
        lower_bound=LOWER_BOUND,
    )

    source = reads.clone()
    source[source < 0] = 0
    fresh = source == 0
    x["conv"][0].zero_()
    x["state"][0].zero_()
    windows, states = torch_kda_conv_decode(
        x["qkv"],
        x["conv_w"],
        x["conv"],
        x["state"],
        source,
        torch.eye(P, device="cuda"),
        payload["gate"],
        x["A_log"],
        x["dt_bias"],
        x["beta"],
        accepted,
        tokens_per_request=t,
        num_heads=HV,
        head_k_dim=K,
        head_v_dim=K,
        lower_bound=LOWER_BOUND,
    )
    assert fresh.tolist() == [False, True, False, False]
    for i, page in enumerate(writes[0].tolist()):
        if page < 0:
            continue
        torch.testing.assert_close(x["conv"][page].float(), windows[i], atol=0, rtol=0)
        torch.testing.assert_close(
            x["state"][page].transpose(-1, -2), states[i], atol=1e-5, rtol=1e-2
        )


def test_batch_sizes_share_one_compiled_kernel():
    """The batch size is a runtime argument: no recompilation per batch."""
    t = 4
    x = _inputs(3, t, seed=3, state_dtype=torch.bfloat16)
    _verify(x, torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda"), t)
    compiled = len(verify._COMPILED)
    y = _inputs(6, t, seed=4, state_dtype=torch.bfloat16)
    _verify(y, torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda"), t)
    assert len(verify._COMPILED) == compiled
