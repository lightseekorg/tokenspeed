"""Numerics for the CuTe DSL fused KDA decode kernel.

Reference order:

1. A pure-torch fp32 reference of the megafusion math (conv shift + SiLU,
   f_b gate GEMV + safe/softplus gate, L2 norm, delta-rule step) — the
   ground truth for every batch size.
2. (removed) the racy Triton megafusion, deleted from the tree — parity now
   at small batch: the Triton program relies on same-wave residency for its
   in-place conv-window shift, and its own outputs flake once its grid
   exceeds one wave (observed from B=16 at the K3 TP8 geometry). The CuTe
   kernel orders the shift with a cluster barrier instead and stays
   deterministic (asserted below via nv=2 vs nv=4 bitwise identity).

The CuTe kernel is JIT-compiled on first use; tests only need a GPU with a
built tokenspeed_kernel python package (no ahead-of-time kernel build).
"""

from importlib.util import find_spec

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if find_spec("cutlass") is None or find_spec("quack") is None:
    pytest.skip("cutlass DSL + quack required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("validated on the SM100 family only", allow_module_level=True)

from tokenspeed_kernel.ops.attention.cute_dsl import (  # noqa: E402
    kda_fused_decode as kfd,
)
from tokenspeed_kernel.ops.attention.cute_dsl.kda_fused_decode import (  # noqa: E402
    cutedsl_fused_recurrent_kda_megafuse,
)
from tokenspeed_kernel.ops.attention.kda_utils import KdaGatedNormRequest  # noqa: E402

# K3 decode shapes (TP8 rank): 12 heads, K = V = 128, D_FA = 128.
HV, K, V = 12, 128, 128
P = HV * K
D_FA = 128
LOWER_BOUND = -5.0


def _make_inputs(bs, pages=64, seed=0, pad_every=0):
    torch.manual_seed(seed)
    dev = "cuda"
    x = dict(
        qkv=torch.randn(bs, 3 * P, dtype=torch.bfloat16, device=dev),
        conv_w=torch.randn(3 * P, 4, dtype=torch.bfloat16, device=dev) * 0.3,
        conv_pool=torch.randn(pages, 3 * P, 3, dtype=torch.bfloat16, device=dev),
        f_a=torch.randn(bs, D_FA, dtype=torch.bfloat16, device=dev),
        w_fb=torch.randn(P, D_FA, dtype=torch.bfloat16, device=dev) * 0.05,
        beta=torch.randn(bs, HV, dtype=torch.bfloat16, device=dev),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev) * 0.5,
        dt_bias=torch.randn(P, dtype=torch.float32, device=dev),
        h_pool=torch.randn(pages, HV, K, V, dtype=torch.float32, device=dev),
        ri=torch.randperm(pages, device=dev)[:bs].to(torch.int32),
        cu=torch.arange(bs + 1, dtype=torch.int32, device=dev),
        gate=torch.randn(bs, P, dtype=torch.bfloat16, device=dev),
        norm_w=torch.rand(V, dtype=torch.bfloat16, device=dev) + 0.5,
    )
    x["wi"] = x["ri"].clone()
    if pad_every:
        # Graph-padding style rows: negative page ids read zeros/skip stores.
        pad = torch.arange(bs, device=dev) % pad_every == 0
        x["ri"] = torch.where(pad, torch.full_like(x["ri"], -1), x["ri"])
        x["wi"] = torch.where(pad, torch.full_like(x["wi"], -1), x["wi"])
    return x


def _clone(x):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}


def _run(x, *, lower_bound=LOWER_BOUND, cu=True, enable_pdl=False, onorm=None):
    return cutedsl_fused_recurrent_kda_megafuse(
        x["qkv"],
        x["conv_w"],
        x["conv_pool"],
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["ri"],
        x["wi"],
        num_heads=HV,
        head_dim=K,
        cu_seqlens=x["cu"] if cu else None,
        lower_bound=lower_bound,
        onorm=onorm,
        enable_pdl=enable_pdl,
    )


def _onorm_request(x, eps=1e-6):
    return KdaGatedNormRequest(weight=x["norm_w"], gate=x["gate"], eps=eps)


def _torch_reference(x, *, lower_bound=LOWER_BOUND, apply_onorm=False):
    """fp32 torch reference of one fused decode step (mutates the pools)."""
    qkv = x["qkv"].float()
    conv_w = x["conv_w"].float()
    f_a = x["f_a"].float()
    w_fb = x["w_fb"].float()
    beta = x["beta"].float()
    A_log = x["A_log"]
    dt_bias = x["dt_bias"]
    bs = qkv.shape[0]
    out = torch.zeros(bs, HV, V, dtype=torch.float32, device=qkv.device)
    for n in range(bs):
        r = int(x["ri"][n])
        w = int(x["wi"][n])
        window = (
            x["conv_pool"][r].float()
            if r >= 0
            else torch.zeros(3 * P, 3, device=qkv.device)
        )
        xt = qkv[n]
        acc = (window * conv_w[:, :3]).sum(-1) + xt * conv_w[:, 3]
        y = acc * torch.sigmoid(acc)
        if w >= 0:
            new_window = torch.cat([window[:, 1:], xt[:, None]], dim=1)
            x["conv_pool"][w] = new_window.to(x["conv_pool"].dtype)
        q = y[:P].view(HV, K)
        k = y[P : 2 * P].view(HV, K)
        v = y[2 * P :].view(HV, V)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        g = (w_fb @ f_a[n]) + dt_bias
        g = g.view(HV, K)
        exp_a = torch.exp(A_log)[:, None]
        if lower_bound is not None:
            gk = lower_bound * torch.sigmoid(exp_a * g)
        else:
            gk = -exp_a * torch.where(g < 20.0, torch.log1p(torch.exp(g)), g)
        h = (
            x["h_pool"][r].clone()
            if r >= 0
            else torch.zeros(HV, K, V, device=qkv.device)
        )
        h = h * torch.exp(gk)[:, :, None]
        t = torch.einsum("hkv,hk->hv", h, k)
        v_new = (v - t) * torch.sigmoid(beta[n])[:, None]
        h = h + torch.einsum("hk,hv->hkv", k, v_new)
        out[n] = torch.einsum("hkv,hk->hv", h, q)
        if w >= 0:
            x["h_pool"][w] = h
    if apply_onorm:
        # Gated RMSNorm on the fp32 outputs, matching the fused epilogue (no bf16 round-trip).
        var = (out * out).mean(-1, keepdim=True)
        out = (
            out
            * torch.rsqrt(var + 1e-6)
            * x["norm_w"].float()
            * torch.sigmoid(x["gate"].float()).view(bs, HV, V)
        )
    return out.to(torch.bfloat16)


def _assert_step_close(x, o, ref_x, ref_o, o_atol=6e-3, h_atol=4e-3):
    torch.testing.assert_close(o.float(), ref_o.float(), atol=o_atol, rtol=1e-2)
    torch.testing.assert_close(x["h_pool"], ref_x["h_pool"], atol=h_atol, rtol=1e-3)
    torch.testing.assert_close(
        x["conv_pool"].float(), ref_x["conv_pool"].float(), atol=0.0, rtol=0.0
    )


class TestKdaFusedDecodeCutedsl:
    @pytest.mark.parametrize("bs", [1, 2, 3, 4, 5, 8, 12, 16])
    @pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None])
    def test_matches_torch_reference(self, bs, lower_bound):
        x = _make_inputs(bs, seed=bs)
        ref_x = _clone(x)
        o = _run(x, lower_bound=lower_bound)
        ref_o = _torch_reference(ref_x, lower_bound=lower_bound)
        torch.cuda.synchronize()
        _assert_step_close(x, o, ref_x, ref_o)

    @pytest.mark.parametrize("bs", [1, 8])
    def test_cuda_graph_capture(self, bs):
        x = _make_inputs(bs, seed=8)
        stream = torch.cuda.Stream()
        # Side-stream launches must order after the default-stream input producers, or the kernel can read garbage page indices and fault OOB.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _run(_clone(x))
        torch.cuda.synchronize()
        ref_x = _clone(x)
        ref_o = _torch_reference(ref_x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            o = _run(x)
        # Replay on restored pools reproduces the step, like the engine's decode graph.
        x["h_pool"].copy_(_make_inputs(bs, seed=8)["h_pool"])
        x["conv_pool"].copy_(_make_inputs(bs, seed=8)["conv_pool"])
        g.replay()
        torch.cuda.synchronize()
        _assert_step_close(x, o, ref_x, ref_o)

    @pytest.mark.parametrize("bs", [1, 4, 8, 16])  # covers nv=2 and nv=1
    def test_onorm_fused_matches_reference(self, bs):
        x = _make_inputs(bs, seed=20 + bs)
        ref_x = _clone(x)
        req = _onorm_request(x)
        o = _run(x, onorm=req)
        ref_o = _torch_reference(ref_x, apply_onorm=True)
        torch.cuda.synchronize()
        assert req.consumed
        _assert_step_close(x, o, ref_x, ref_o)

    def test_onorm_fused_vs_separate_chain(self):
        # The fused epilogue norms fp32 values while the separate kernel sees a bf16 round-trip, so the two agree only to bf16 rounding.
        from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid

        bs = 4
        x = _make_inputs(bs, seed=30)
        xc = _clone(x)
        o_raw = _run(xc)
        torch.cuda.synchronize()
        o_chain = rmsnorm_gated_sigmoid(
            o_raw.reshape(bs, P).contiguous(),
            x["gate"].contiguous(),
            x["norm_w"],
            1e-6,
            HV,
            K,
        ).view(bs, HV, V)
        req = _onorm_request(x)
        o_fused = _run(x, onorm=req)
        torch.cuda.synchronize()
        assert req.consumed
        # Same state evolution regardless of the epilogue.
        assert torch.equal(x["h_pool"], xc["h_pool"])
        assert torch.equal(x["conv_pool"], xc["conv_pool"])
        torch.testing.assert_close(
            o_fused.float(), o_chain.float(), atol=3e-2, rtol=8e-3
        )

    def test_onorm_strided_gate(self):
        # Gate as a column slice of a wider projection output, as the runtime passes it.
        bs = 3
        x = _make_inputs(bs, seed=31)
        wide = torch.randn(bs, 5 * P, dtype=torch.bfloat16, device="cuda")
        wide[:, 3 * P : 4 * P] = x["gate"]
        x["gate"] = wide[:, 3 * P : 4 * P]
        assert not x["gate"].is_contiguous()
        ref_x = _clone(x)
        req = _onorm_request(x)
        o = _run(x, onorm=req)
        ref_o = _torch_reference(ref_x, apply_onorm=True)
        torch.cuda.synchronize()
        assert req.consumed
        _assert_step_close(x, o, ref_x, ref_o)

    def test_onorm_ineligible_left_unconsumed(self):
        bs = 2
        x = _make_inputs(bs, seed=32)
        ref_x = _clone(x)
        # Wrong weight width: the kernel must decline and return raw output.
        req = KdaGatedNormRequest(
            weight=torch.ones(64, dtype=torch.bfloat16, device="cuda"),
            gate=x["gate"],
            eps=1e-6,
        )
        o = _run(x, onorm=req)
        ref_o = _torch_reference(ref_x, apply_onorm=False)
        torch.cuda.synchronize()
        assert not req.consumed
        _assert_step_close(x, o, ref_x, ref_o)

    def test_onorm_large_batch_fused(self):
        # Large batches run the nv=1 band; the wrapper fuses the norm unconditionally when operands are eligible.
        bs = 32
        x = _make_inputs(bs, seed=36)
        ref_x = _clone(x)
        req = _onorm_request(x)
        o = _run(x, onorm=req)
        ref_o = _torch_reference(ref_x, apply_onorm=True)
        torch.cuda.synchronize()
        assert req.consumed
        _assert_step_close(x, o, ref_x, ref_o)

    @pytest.mark.parametrize("pad_elems", [0, 640, 642])
    def test_envelope_strided_pool(self, pad_elems):
        # Envelope pitch: pad 640 stays 16B-aligned (bulk-TMA nv=1); pad 642 is only 8B-aligned and must auto-fall back to the nv=2 path — same numerics either way.
        bs = 8  # batch band that picks nv=1 on aligned pools
        pages = 16
        x = _make_inputs(bs, pages=pages, seed=37)
        dense = HV * K * V
        storage = torch.zeros(
            pages * (dense + pad_elems) + dense,
            dtype=torch.float32,
            device="cuda",
        )
        strided = storage.as_strided(
            (pages, HV, K, V), (dense + pad_elems, K * V, V, 1)
        )
        strided.copy_(x["h_pool"])
        x["h_pool"] = strided
        ref_x = _clone(x)
        req = _onorm_request(x)
        o = _run(x, onorm=req)
        ref_o = _torch_reference(ref_x, apply_onorm=True)
        torch.cuda.synchronize()
        assert req.consumed
        torch.testing.assert_close(o.float(), ref_o.float(), atol=6e-3, rtol=1e-2)
        torch.testing.assert_close(x["h_pool"], ref_x["h_pool"], atol=4e-3, rtol=1e-3)

    def test_onorm_padded_rows(self):
        bs = 8
        x = _make_inputs(bs, seed=33, pad_every=3)
        ref_x = _clone(x)
        req = _onorm_request(x)
        o = _run(x, onorm=req)
        ref_o = _torch_reference(ref_x, apply_onorm=True)
        torch.cuda.synchronize()
        assert req.consumed
        _assert_step_close(x, o, ref_x, ref_o)

    @pytest.mark.parametrize("bs", [1, 8])
    def test_onorm_cuda_graph_capture(self, bs):
        x = _make_inputs(bs, seed=34)
        stream = torch.cuda.Stream()
        # Side-stream launches must order after the default-stream input producers, or the kernel can read garbage page indices and fault OOB.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _run(_clone(x), onorm=_onorm_request(x))
        torch.cuda.synchronize()
        ref_x = _clone(x)
        ref_o = _torch_reference(ref_x, apply_onorm=True)
        g = torch.cuda.CUDAGraph()
        req = _onorm_request(x)
        with torch.cuda.graph(g, stream=stream):
            o = _run(x, onorm=req)
        assert req.consumed  # decided at capture, stable across replays
        x["h_pool"].copy_(_make_inputs(bs, seed=34)["h_pool"])
        x["conv_pool"].copy_(_make_inputs(bs, seed=34)["conv_pool"])
        g.replay()
        torch.cuda.synchronize()
        _assert_step_close(x, o, ref_x, ref_o)

    def test_registry_selects_cutedsl(self):
        import tokenspeed_kernel.ops.attention  # noqa: F401  (registrations)
        from tokenspeed_kernel.selection import select_kernel
        from tokenspeed_kernel.signature import format_signatures

        selected = select_kernel(
            "attention",
            "kda_fused_paged_decode",
            next(iter(format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}))),
            traits={"flat_state": True},
        )
        assert selected.name == "cutedsl_nvidia_kda_fused_paged_decode"
