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

"""The CuTe DSL batched KDA replay against sequential decode.

The selected all-layer replay must commit what ``accepted_length`` ordinary
decode steps produce, ignore the rejected suffix, leave a zero-accept page
untouched, start fresh pages from zero, and skip unwritten pages.
"""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.kda import torch_kda_conv_decode
from tokenspeed_kernel.platform import ArchVersion, current_platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
    pytest.skip("CuTe DSL KDA replay targets sm100/sm103", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda import (  # noqa: E402
    resolve_kda_batched_replay_commit,
)
from tokenspeed_kernel.registry import KernelRegistry  # noqa: E402

HV, K, D_FA = 4, 128, 128
P = HV * K
LOWER_BOUND = -5.0
PAGES = 24
ZERO_PAGE = 0


def _layer(n, t, seed, state_dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to("cuda", dtype)

    layer = dict(
        qkv=rnd(n * t, 3 * P),
        conv_w=rnd(3 * P, 4, scale=0.3),
        conv=rnd(PAGES, 3 * P, 3),
        f_a=rnd(n * t, D_FA),
        w_fb=rnd(P, D_FA, scale=0.05),
        beta=rnd(n * t, HV),
        A_log=rnd(HV, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(P, dtype=torch.float32),
        state=rnd(PAGES, HV, K, K, dtype=torch.float32).to(state_dtype),
        gate=torch.empty(n * t, P, device="cuda", dtype=torch.float32),
    )
    layer["conv"][ZERO_PAGE].zero_()
    layer["state"][ZERO_PAGE].zero_()
    return layer


def _descriptors(layers):
    columns = ("qkv", "conv_w", "conv", "f_a", "w_fb", "beta", "A_log", "dt_bias")
    rows = [
        [x[c].data_ptr() for c in columns]
        + [x["state"].data_ptr(), x["gate"].data_ptr()]
        for x in layers
    ]
    return torch.tensor(rows, dtype=torch.uint64, device="cuda")


def _replay(kernel, layers, groups, reads, writes, accepted, t):
    x = layers[0]
    kernel(
        descriptors=_descriptors(layers),
        group_indices=torch.tensor(groups, dtype=torch.int32, device="cuda"),
        read_indices=reads,
        write_indices=writes,
        accepted_length=accepted,
        draft_token_num=t,
        num_heads=HV,
        head_dim=K,
        f_a_dim=D_FA,
        qkv_stride=x["qkv"].stride(0),
        conv_stride=x["conv"].stride(0),
        f_a_stride=x["f_a"].stride(0),
        beta_stride=x["beta"].stride(0),
        state_stride=x["state"].stride(0),
        state_dtype=x["state"].dtype,
        gate_stride=x["gate"].stride(0),
        conv_width=4,
        lower_bound=LOWER_BOUND,
    )


def _reference(x, reads, t, accepted):
    return torch_kda_conv_decode(
        x["qkv"],
        x["conv_w"],
        x["conv"],
        x["state"],
        reads,
        x["w_fb"],
        x["f_a"],
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


def _pages(rows):
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


@pytest.fixture(scope="module")
def kernel():
    selected = resolve_kda_batched_replay_commit()
    assert selected is not None
    assert selected.name == "cutedsl_nvidia_kda_batched_replay_commit"
    return selected


@pytest.mark.parametrize("t", [1, 4])
def test_replay_matches_sequential_decode(kernel, t):
    """Two cache groups, every accepted length in [0, t], fresh and skipped pages."""
    n = 6
    layers = [
        _layer(n, t, seed=10 * t + i, state_dtype=torch.float32) for i in range(3)
    ]
    groups = [0, 0, 1]
    # Group 1 reads a fresh page (-1) for request 0 and skips request 5's write.
    reads = _pages([[1, 2, 3, 4, 5, 6], [-1, 8, 9, 10, 11, 12]])
    writes = _pages([[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, -1]])
    accepted = torch.tensor(
        [i % (t + 1) for i in range(n)], dtype=torch.int32, device="cuda"
    )
    before = [(x["state"].clone(), x["conv"].clone()) for x in layers]
    _replay(kernel, layers, groups, reads, writes, accepted, t)
    for x, group, (state0, conv0) in zip(layers, groups, before, strict=True):
        source = reads[group].clone()
        source[source < 0] = ZERO_PAGE
        windows, states = _reference(x, source, t, accepted)
        written = set()
        for i in range(n):
            w = int(writes[group, i])
            if w < 0:
                continue
            written.add(w)
            torch.testing.assert_close(x["conv"][w].float(), windows[i], atol=0, rtol=0)
            torch.testing.assert_close(
                x["state"][w].transpose(-1, -2), states[i], atol=1e-5, rtol=1e-2
            )
        others = [p for p in range(PAGES) if p not in written]
        torch.testing.assert_close(x["state"][others], state0[others], atol=0, rtol=0)
        torch.testing.assert_close(x["conv"][others], conv0[others], atol=0, rtol=0)


def test_bf16_state_matches_the_triton_replay(kernel):
    """BF16 storage: both replays round the same FP32 recurrence once."""
    n, t = 5, 4
    layers = [_layer(n, t, seed=40 + i, state_dtype=torch.bfloat16) for i in range(2)]
    triton_layers = [{k: v.clone() for k, v in x.items()} for x in layers]
    reads = _pages([[1, 2, 3, 4, 5]])
    writes = _pages([[13, 14, 15, 16, 17]])
    accepted = _pages([4, 0, 2, 3, 1])
    _replay(kernel, layers, [0, 0], reads, writes, accepted, t)
    triton = KernelRegistry.get().get_impl("triton_nvidia_kda_batched_replay_commit")
    _replay(triton, triton_layers, [0, 0], reads, writes, accepted, t)
    for x, y in zip(layers, triton_layers, strict=True):
        torch.testing.assert_close(x["conv"], y["conv"], atol=0, rtol=0)
        torch.testing.assert_close(
            x["state"], y["state"], atol=1e-6, rtol=torch.finfo(torch.bfloat16).eps
        )


def test_rejected_suffix_and_zero_accept_commit_in_place(kernel):
    """Rejected rows cannot reach the state; zero accepted leaves the page as is."""
    n, t = 4, 4
    accepted = _pages([2, 0, 4, 1])
    pages = _pages([[1, 2, 3, 4]])
    base = _layer(n, t, seed=7, state_dtype=torch.bfloat16)
    perturbed = {k: v.clone() for k, v in base.items()}
    for i, a in enumerate(accepted.tolist()):
        rows = slice(i * t + a, (i + 1) * t)
        for name in ("qkv", "f_a", "beta"):
            perturbed[name][rows] = perturbed[name][rows] * 3 + 1
    before = base["state"][2].clone(), base["conv"][2].clone()
    _replay(kernel, [base], [0], pages, pages, accepted, t)
    _replay(kernel, [perturbed], [0], pages, pages, accepted, t)
    torch.testing.assert_close(perturbed["state"], base["state"], atol=0, rtol=0)
    torch.testing.assert_close(perturbed["conv"], base["conv"], atol=0, rtol=0)
    torch.testing.assert_close(base["state"][2], before[0], atol=0, rtol=0)
    torch.testing.assert_close(base["conv"][2], before[1], atol=0, rtol=0)
