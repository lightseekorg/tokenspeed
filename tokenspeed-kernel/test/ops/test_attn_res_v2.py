"""Parity tests for the single-kernel AttnRes v2 forward (``attn_res.fwd_v2``).

fwd_v2 folds the optional residual accumulate (``prefix += delta``, written
back in place), the online-softmax candidate mix, and the following RMSNorm
into one launch. The torch solution is validated against a direct fp32
reference; the SM100 CUDA solution (when built) is validated against the torch
solution, including strided block layouts and CUDA-graph capture.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attn_res import attn_res_fwd_v2
from tokenspeed_kernel.ops.attn_res.torch import torch_attn_res_fwd_v2

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

H = 7168  # K3 hidden size; the only H the CUDA kernel instantiates.


def _cuda_v2_built() -> bool:
    from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL_V2

    return _HAS_CUDA_KERNEL_V2


def _reference(prefix, delta, blocks, res_w, rms_w, out_w, eps, out_eps):
    """Direct softmax over the KB+1 candidate logits, fp32; returns
    (normed mix, updated prefix)."""
    prefix = prefix.float()
    if delta is not None:
        # The kernel accumulates and rounds in bf16 before scoring.
        prefix = (prefix + delta.float()).to(torch.bfloat16).float()
    cands = torch.cat([blocks.float(), prefix.unsqueeze(0)], dim=0)
    wq = (rms_w.float() * res_w.float())[None, None, :]
    rms = torch.rsqrt(cands.pow(2).mean(-1, keepdim=True) + eps)
    logits = (cands * rms * wq).sum(-1)  # [KB+1, T]
    w = torch.softmax(logits, dim=0)[..., None]
    mix = (cands * w).sum(0)
    rms_o = torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + out_eps)
    return mix * rms_o * out_w.float(), prefix


def _inputs(T, KB, use_delta, seed):
    torch.manual_seed(seed)
    prefix = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    delta = (
        torch.randn(T, H, dtype=torch.bfloat16, device="cuda") if use_delta else None
    )
    blocks = torch.randn(KB, T, H, dtype=torch.bfloat16, device="cuda")
    res_w = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05
    rms_w = torch.rand(H, dtype=torch.bfloat16, device="cuda") + 0.5
    out_w = torch.rand(H, dtype=torch.bfloat16, device="cuda") + 0.5
    return prefix, delta, blocks, res_w, rms_w, out_w


@pytest.mark.parametrize("T,KB", [(1, 1), (1, 8), (4, 3)])
@pytest.mark.parametrize("use_delta", [False, True])
def test_torch_solution_vs_reference(T, KB, use_delta):
    prefix, delta, blocks, res_w, rms_w, out_w = _inputs(T, KB, use_delta, T + KB)
    eps = 1e-5
    p = prefix.clone()
    out = torch_attn_res_fwd_v2(
        prefix=p,
        delta=delta,
        block_residual=blocks,
        res_weight=res_w,
        rms_weight=rms_w,
        out_norm_weight=out_w,
        eps=eps,
        out_norm_eps=eps,
    )
    ref, ref_prefix = _reference(prefix, delta, blocks, res_w, rms_w, out_w, eps, eps)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    # The delta accumulate is applied in place on the prefix.
    torch.testing.assert_close(p.float(), ref_prefix, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _cuda_v2_built(), reason="SM100 fwd_v2 kernel not built")
@pytest.mark.parametrize("T,KB", [(1, 1), (1, 3), (1, 8), (4, 7), (33, 8), (300, 8)])
@pytest.mark.parametrize("use_delta", [False, True])
@pytest.mark.parametrize("layout", ["block_major", "token_major"])
def test_cuda_solution_vs_reference(T, KB, use_delta, layout):
    prefix, delta, _, res_w, rms_w, out_w = _inputs(T, KB, use_delta, T * 10 + KB)
    if layout == "block_major":
        # Leading-dim slice of a larger buffer, like block_residual[:KB].
        blocks = torch.randn(KB + 2, T, H, dtype=torch.bfloat16, device="cuda")[:KB]
    else:
        big = torch.randn(T, KB + 1, H, dtype=torch.bfloat16, device="cuda")
        blocks = big[:, :KB].permute(1, 0, 2)
    eps = 1e-5
    p = prefix.clone()
    out = attn_res_fwd_v2(
        p, delta, blocks, res_w, rms_w, out_w, eps, eps, enable_pdl=True
    )
    ref, ref_prefix = _reference(prefix, delta, blocks, res_w, rms_w, out_w, eps, eps)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(p.float(), ref_prefix, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _cuda_v2_built(), reason="SM100 fwd_v2 kernel not built")
def test_cuda_solution_graph_capture():
    """Decode-shape capture/replay: the runtime replays fwd_v2 inside CUDA
    graphs with PDL enabled."""
    T, KB = 1, 7
    prefix, delta, blocks, res_w, rms_w, out_w = _inputs(T, KB, True, 7)
    out = torch.empty_like(prefix)
    p = prefix.clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            attn_res_fwd_v2(
                p, delta, blocks, res_w, rms_w, out_w, 1e-5, 1e-5, True, out=out
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    p.copy_(prefix)
    with torch.cuda.graph(graph):
        attn_res_fwd_v2(
            p, delta, blocks, res_w, rms_w, out_w, 1e-5, 1e-5, True, out=out
        )
    p.copy_(prefix)
    graph.replay()
    torch.cuda.synchronize()
    ref, ref_prefix = _reference(prefix, delta, blocks, res_w, rms_w, out_w, 1e-5, 1e-5)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(p.float(), ref_prefix, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("T", [1, 4])
def test_deferred_delta_chain_matches_eager(T):
    """The deferred-FFN-accumulate arrangement: layer i's residual add is
    carried as the NEXT mix's fused delta instead of a standalone kernel.

    Chains three mock layers and checks every mix output and the running
    prefix (what a block-write snapshot after the mix would copy) against the
    eager arrangement that materializes each add first and mixes with
    delta=None. Runs on whichever solution is selected (torch now, CUDA once
    the SM100 kernel is built) -- both implement the same bf16 in-place
    accumulate contract.
    """
    KB, eps = 3, 1e-5
    torch.manual_seed(T)
    init = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    blocks = torch.randn(KB, T, H, dtype=torch.bfloat16, device="cuda")
    ffn_deltas = [
        torch.randn(T, H, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    ]
    res_w = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05
    rms_w = torch.rand(H, dtype=torch.bfloat16, device="cuda") + 0.5
    out_w = torch.rand(H, dtype=torch.bfloat16, device="cuda") + 0.5

    # Eager arrangement: materialize the accumulate, then mix with delta=None.
    prefix_eager = init.clone()
    outs_eager = []
    for delta in ffn_deltas:
        prefix_eager = prefix_eager + delta
        outs_eager.append(
            attn_res_fwd_v2(prefix_eager, None, blocks, res_w, rms_w, out_w, eps, eps)
        )

    # Deferred arrangement: the add rides the mix as its fused delta and the
    # prefix is updated in place.
    prefix_chain = init.clone()
    outs_chain = []
    for delta in ffn_deltas:
        outs_chain.append(
            attn_res_fwd_v2(prefix_chain, delta, blocks, res_w, rms_w, out_w, eps, eps)
        )

    for got, want in zip(outs_chain, outs_eager):
        torch.testing.assert_close(got.float(), want.float(), atol=3e-2, rtol=3e-2)
    # The in-place prefix (what a block-write snapshot would copy after the
    # mix) matches the eagerly accumulated stream.
    torch.testing.assert_close(
        prefix_chain.float(), prefix_eager.float(), atol=1e-2, rtol=1e-2
    )


def test_out_of_range_shapes_fall_back():
    """H != 7168 or KB > 8 must route to the torch solution, not crash."""
    T, KB, h = 2, 9, 4096
    torch.manual_seed(0)
    prefix = torch.randn(T, h, dtype=torch.bfloat16, device="cuda")
    blocks = torch.randn(KB, T, h, dtype=torch.bfloat16, device="cuda")
    w = torch.rand(h, dtype=torch.bfloat16, device="cuda") + 0.5
    out = attn_res_fwd_v2(prefix, None, blocks, w, w, w, 1e-5, 1e-5)
    assert out.shape == (T, h) and out.dtype == torch.bfloat16
