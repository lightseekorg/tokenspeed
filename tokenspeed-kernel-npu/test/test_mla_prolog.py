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

import pytest
import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def _random_bf16(shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    return (torch.randn(shape, device=device) * 0.01).to(torch.bfloat16)


def _rms_norm(value: torch.Tensor, width: int) -> torch.Tensor:
    value = value[..., :width].float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)).to(
        torch.bfloat16
    )


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize(("hidden", "heads"), ((3072, 32), (4096, 64)))
@pytest.mark.parametrize(
    ("tokens", "index_dtype"), ((2, torch.int64), (32, torch.int32))
)
@torch.inference_mode()
def test_lite_mla_prolog_nope_packed_cache_and_graph_replay(
    hidden: int, heads: int, tokens: int, index_dtype: torch.dtype
):
    import torch_npu
    from tokenspeed_kernel_npu.ops.mla_prolog import mla_prolog, mla_prolog_available

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(7303 + heads)
    device = torch.device("npu:0")
    x = _random_bf16((tokens, hidden), device=device)
    weight_dq_nd = _random_bf16((hidden, 1536), device=device)
    weight_uq_qr_nd = _random_bf16((1536, heads * 192), device=device)
    weight_uk = _random_bf16((heads, 128, 512), device=device)
    weight_dkv_kr_nd = _random_bf16((hidden, 576), device=device)
    gamma_cq = torch.ones(1536, dtype=torch.bfloat16, device=device)
    gamma_ckv = torch.ones(512, dtype=torch.bfloat16, device=device)
    weight_dq = torch_npu.npu_format_cast(weight_dq_nd, 29)
    weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr_nd, 29)
    weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr_nd, 29)
    cache_index = torch.arange(63, 63 + tokens, dtype=index_dtype, device=device)
    cache = torch.zeros(2, 64, 1, 576, dtype=torch.bfloat16, device=device)

    assert mla_prolog_available()
    query, query_rope = mla_prolog(
        x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        gamma_cq,
        gamma_ckv,
        cache,
        cache_index,
        rmsnorm_epsilon_cq=1e-5,
        rmsnorm_epsilon_ckv=1e-5,
    )
    cq = _rms_norm(x @ weight_dq_nd, 1536)
    qc_qr = (cq @ weight_uq_qr_nd).view(tokens, heads, 192)
    expected_query = torch.einsum("bnd,ndc->bnc", qc_qr[..., :128], weight_uk)
    ckv_kr = x @ weight_dkv_kr_nd
    expected_cache = torch.cat((_rms_norm(ckv_kr, 512), ckv_kr[..., 512:]), dim=-1)
    torch.testing.assert_close(query, expected_query, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(query_rope, qc_qr[..., 128:], rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(
        cache.view(-1, 576)[cache_index], expected_cache, rtol=1e-2, atol=1e-3
    )

    graph_x = x.clone()
    graph_index = cache_index.clone()
    graph_cache = torch.zeros_like(cache)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_query, graph_query_rope = mla_prolog(
            graph_x,
            weight_dq,
            weight_uq_qr,
            weight_uk,
            weight_dkv_kr,
            gamma_cq,
            gamma_ckv,
            graph_cache,
            graph_index,
            rmsnorm_epsilon_cq=1e-5,
            rmsnorm_epsilon_ckv=1e-5,
        )
    graph_x.copy_(x * 0.5)
    graph_index.copy_(cache_index + 1)
    graph.replay()

    eager_cache = torch.zeros_like(cache)
    eager_query, eager_query_rope = mla_prolog(
        graph_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        gamma_cq,
        gamma_ckv,
        eager_cache,
        graph_index,
        rmsnorm_epsilon_cq=1e-5,
        rmsnorm_epsilon_ckv=1e-5,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(graph_query, eager_query, rtol=0, atol=0)
    torch.testing.assert_close(graph_query_rope, eager_query_rope, rtol=0, atol=0)
    torch.testing.assert_close(graph_cache, eager_cache, rtol=0, atol=0)
