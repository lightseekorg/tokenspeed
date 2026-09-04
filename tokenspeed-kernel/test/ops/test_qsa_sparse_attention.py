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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import qsa_sparse_attention
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QSA sparse attention requires CUDA or ROCm"
)


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    output = torch.zeros(
        (q.shape[0], q.shape[1], v_cache.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    group_size = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        slots = selected_slots[row][selected_slots[row] > 0].long()
        for head in range(q.shape[1]):
            kv_head = head // group_size
            keys = k_cache[slots, kv_head].float() * k_scale
            values = v_cache[slots, kv_head].float() * v_scale
            scores = q[row, head].float() @ keys.T
            output[row, head] = torch.softmax(scores * scale, dim=-1) @ values
    return output.to(q.dtype)


def test_qsa_sparse_attention_validates_uniform_query_length(device: str) -> None:
    q = torch.empty((6, 2, 32), dtype=torch.bfloat16, device=device)
    cache = torch.empty((16, 1, 32), dtype=torch.bfloat16, device=device)
    selected = torch.ones((6, 1), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="positive"):
        qsa_sparse_attention(q, cache, cache, selected, scale=1.0, max_seqlen_q=0)
    with pytest.raises(ValueError, match="divisible"):
        qsa_sparse_attention(q, cache, cache, selected, scale=1.0, max_seqlen_q=4)


@pytest.mark.parametrize("rows", [1, 4, 9])
def test_qsa_sparse_attention_blackwell_cluster_matches_reference(
    device: str,
    rows: int,
) -> None:
    platform = current_platform()
    if platform.arch_version != ArchVersion(10, 0):
        pytest.skip("cluster QSA sparse attention is specialized for NVIDIA SM100")

    torch.manual_seed(67 + rows)
    cache_slots, q_heads, kv_heads, head_dim, width = 4096, 6, 1, 256, 2051
    q_storage = torch.randn(
        rows,
        q_heads,
        head_dim * 2,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_storage[..., ::2]
    k_cache = (
        torch.randn(
            cache_slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    v_cache = (
        torch.randn(
            cache_slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    slot_storage = torch.full((rows, width * 2), -1, dtype=torch.int32, device=device)
    slots = slot_storage[:, ::2]
    slots[:, :2049] = torch.randint(
        1, cache_slots, (rows, 2049), dtype=torch.int32, device=device
    )
    slots[:, 5::11] = -1
    slots[:, 9::17] = 0
    if rows > 1:
        slots[-1].fill_(-1)

    traits = {
        "batch_size": rows,
        "head_dim": head_dim,
        "value_head_dim": head_dim,
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "selected_width": width,
    }
    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        k_cache=dense_tensor_format(k_cache.dtype),
        v_cache=dense_tensor_format(v_cache.dtype),
    )
    selected_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        platform=platform,
        traits=traits,
    )
    assert selected_kernel.name == "cute_dsl_blackwell_qsa_sparse_attention"
    mtp3_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        platform=platform,
        traits={**traits, "batch_size": 32, "q_len": 4},
    )
    assert mtp3_kernel.name == "cute_dsl_blackwell_qsa_sparse_attention"

    scale = head_dim**-0.5
    k_scale, v_scale = 0.5, 0.25
    actual = qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        slots,
        scale=scale,
        max_seqlen_q=(4 if rows == 4 else 1),
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        _reference(
            q,
            k_cache,
            v_cache,
            slots,
            scale,
            k_scale=k_scale,
            v_scale=v_scale,
        ).float(),
        rtol=3.5e-2,
        atol=3.5e-2,
    )


def test_qsa_sparse_attention_blackwell_cluster_supports_graph_replay(
    device: str,
) -> None:
    platform = current_platform()
    if platform.arch_version != ArchVersion(10, 0):
        pytest.skip("cluster QSA sparse attention is specialized for NVIDIA SM100")
    if (
        KernelRegistry.get().get_by_name("cute_dsl_blackwell_qsa_sparse_attention")
        is None
    ):
        pytest.skip("CuTe DSL QSA sparse attention is unavailable")

    torch.manual_seed(79)
    cache_slots, width = 4096, 2051
    q = torch.randn(1, 6, 256, device=device, dtype=torch.bfloat16)
    k_cache = (
        torch.randn(cache_slots, 1, 256, device=device, dtype=torch.bfloat16) * 0.25
    ).to(torch.float8_e4m3fn)
    v_cache = (
        torch.randn(cache_slots, 1, 256, device=device, dtype=torch.bfloat16) * 0.25
    ).to(torch.float8_e4m3fn)
    selected = torch.randint(
        1, cache_slots, (1, width), device=device, dtype=torch.int32
    )
    scale = 256**-0.5
    k_scale, v_scale = 0.5, 0.25
    kwargs = {
        "scale": scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "override": "cute_dsl_blackwell_qsa_sparse_attention",
    }

    qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)

    replay_query = torch.randn_like(q)
    q.copy_(replay_query)
    selected[:, 1024:].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.float(),
        _reference(
            replay_query,
            k_cache,
            v_cache,
            selected,
            scale,
            k_scale=k_scale,
            v_scale=v_scale,
        ).float(),
        rtol=3.5e-2,
        atol=3.5e-2,
    )
