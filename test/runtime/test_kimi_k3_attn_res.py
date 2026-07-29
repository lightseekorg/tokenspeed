"""Kimi-K3 AttnRes mixing + router numerics tests (cheap; kernel parity on GPU).

Covers the ``attn_res_fwd`` op the model routes AttnRes mixing through: the
torch fallback must match the reference ``modeling_kimi.py::_apply_attn_res``
math, the model wiring must slice candidates correctly, and (when a Blackwell
kernel build is present) the CUDA kernel must match the torch fallback.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attn_res import attn_res_fwd  # noqa: E402
from tokenspeed_kernel.ops.attn_res.torch import torch_attn_res_fwd  # noqa: E402

from tokenspeed.runtime.layers.layernorm import RMSNorm  # noqa: E402
from tokenspeed.runtime.models import kimi_k3  # noqa: E402

_HIDDEN = 64
_BLOCKS = 4
_EPS = 1e-5


def _make_inputs(num_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    prefix_sum = torch.randn(num_tokens, _HIDDEN, dtype=torch.bfloat16)
    # Block-major [num_blocks, T, hidden], matching the model's scratch layout.
    block_residual = torch.randn(_BLOCKS, num_tokens, _HIDDEN, dtype=torch.bfloat16)
    norm = RMSNorm(_HIDDEN, eps=_EPS)
    norm.weight.data.uniform_(0.5, 1.5)
    proj = torch.nn.Linear(_HIDDEN, 1, bias=False)
    return prefix_sum, block_residual, proj, norm


def _reference_apply_attn_res(prefix_sum, block_residual, proj, norm):
    """Verbatim math from the checkpoint's modeling_kimi.py::_apply_attn_res,
    over block-major candidates [N, T, H] = blocks then the current stream."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(0)), dim=0)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)  # [N, T]
    probs = scores.softmax(0)
    return (probs.unsqueeze(-1) * v_float).sum(0).to(v.dtype)


class AttnResTests(unittest.TestCase):
    def test_torch_fallback_matches_reference(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(17)
        got = torch_attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
        )
        ref = _reference_apply_attn_res(prefix_sum, block_residual, proj, norm)
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_model_wiring_slices_valid_blocks(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(9, seed=2)
        got = kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 2)
        ref = _reference_apply_attn_res(prefix_sum, block_residual[:2], proj, norm)
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_zero_valid_blocks_is_identity(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(5, seed=3)
        got = kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 0)
        self.assertIs(got, prefix_sum)

    def test_cuda_kernel_matches_torch_fallback(self):
        # Only runs where the Blackwell attn_res build is present (e.g. B300 CI).
        try:
            from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
        except ImportError:
            _HAS_CUDA_KERNEL = False
        if not (_HAS_CUDA_KERNEL and torch.cuda.is_available()):
            self.skipTest("Blackwell attn_res kernel not available")
        from tokenspeed_kernel.ops.attn_res import attn_res_fwd

        torch.manual_seed(0)
        T, H, K = 128, 7168, 8  # kernel-eligible shape (H in supported set)
        dev = "cuda"
        prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
        blocks = torch.randn(K, T, H, dtype=torch.bfloat16, device=dev)
        res_w = torch.randn(H, dtype=torch.bfloat16, device=dev)
        rms_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
        got = attn_res_fwd(prefix, blocks, res_w, rms_w, _EPS)  # cuda path
        ref = torch_attn_res_fwd(
            layer_residual=prefix,
            block_residual=blocks,
            res_weight=res_w,
            rms_weight=rms_w,
            eps=_EPS,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_cuda_kernel_single_token_dispatches_match_torch(self):
        """T=1 specialized dispatches.

        N in {1, 2, 4, 8, 12} at H=7168 routes to the single-CTA / split-K
        single-token kernels when no fused out-norm is requested, and to the
        online kernel (which fuses the norm) otherwise. N counts the layer
        residual, so K = N - 1 blocks. Cover both, plus N=13 (> N_MAX) which
        must stay on the torch fallback.
        """
        try:
            from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
        except ImportError:
            _HAS_CUDA_KERNEL = False
        if not (_HAS_CUDA_KERNEL and torch.cuda.is_available()):
            self.skipTest("Blackwell attn_res kernel not available")
        from tokenspeed_kernel.ops.attn_res import attn_res_fwd

        torch.manual_seed(0)
        T, H = 1, 7168
        dev = "cuda"
        for K in (1, 3, 7, 11, 12):
            prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
            blocks = torch.randn(K, T, H, dtype=torch.bfloat16, device=dev)
            res_w = torch.randn(H, dtype=torch.bfloat16, device=dev)
            rms_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
            got = attn_res_fwd(prefix, blocks, res_w, rms_w, _EPS)
            ref = torch_attn_res_fwd(
                layer_residual=prefix,
                block_residual=blocks,
                res_weight=res_w,
                rms_weight=rms_w,
                eps=_EPS,
            )
            torch.testing.assert_close(
                got, ref, atol=2e-2, rtol=2e-2, msg=f"N={K} no-norm"
            )
            out_norm_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
            got_n = attn_res_fwd(
                prefix, blocks, res_w, rms_w, _EPS, out_norm_weight=out_norm_w
            )
            ref_n = _manual_rmsnorm(ref, out_norm_w, _EPS)
            torch.testing.assert_close(
                got_n, ref_n, atol=2e-2, rtol=2e-2, msg=f"N={K} out-norm"
            )


def _manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    xf = x.float()
    rs = (xf.square().mean(-1, keepdim=True) + eps).rsqrt()
    return (xf * rs * weight.float()).to(x.dtype)


class AttnResOutNormTests(unittest.TestCase):
    def test_output_eps_is_ignored_without_output_norm(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(11, seed=4)
        kwargs = {
            "layer_residual": prefix_sum,
            "block_residual": block_residual,
            "res_weight": proj.weight.reshape(-1).to(torch.bfloat16),
            "rms_weight": norm.weight.to(torch.bfloat16),
            "eps": _EPS,
        }
        expected = attn_res_fwd(**kwargs)
        actual = attn_res_fwd(**kwargs, out_norm_eps=2 * _EPS)
        torch.testing.assert_close(actual, expected)

    def test_public_fallback_supports_distinct_output_eps(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(11, seed=5)
        out_norm = RMSNorm(_HIDDEN, eps=_EPS)
        out_norm.weight.data.uniform_(0.5, 1.5)
        output_eps = 2 * _EPS
        fused = attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
            out_norm_weight=out_norm.weight.to(torch.bfloat16),
            out_norm_eps=output_eps,
        )
        mixed = attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
        )
        ref = _manual_rmsnorm(mixed, out_norm.weight, output_eps)
        torch.testing.assert_close(fused, ref, atol=2e-2, rtol=2e-2)

    def test_model_helper_out_norm_wiring(self):
        if not torch.cuda.is_available():
            self.skipTest("RMSNorm.forward requires CUDA")
        prefix_sum, block_residual, proj, norm = _make_inputs(6, seed=6)
        prefix_sum = prefix_sum.cuda()
        block_residual = block_residual.cuda()
        proj = proj.to(torch.bfloat16).cuda()
        norm = norm.cuda()
        out_norm = RMSNorm(_HIDDEN, eps=_EPS).to(torch.bfloat16).cuda()
        out_norm.weight.data.uniform_(0.5, 1.5)
        got = kimi_k3._apply_attn_res(
            prefix_sum, block_residual, proj, norm, 2, out_norm=out_norm
        )
        mixed = kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 2)
        torch.testing.assert_close(
            got, _manual_rmsnorm(mixed, out_norm.weight, _EPS), atol=2e-2, rtol=2e-2
        )
        # Zero valid blocks: the helper must still apply the out-norm.
        got0 = kimi_k3._apply_attn_res(
            prefix_sum, block_residual, proj, norm, 0, out_norm=out_norm
        )
        torch.testing.assert_close(
            got0,
            _manual_rmsnorm(prefix_sum, out_norm.weight, _EPS),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_cuda_kernel_out_norm_matches_torch(self):
        try:
            from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
        except ImportError:
            _HAS_CUDA_KERNEL = False
        if not (_HAS_CUDA_KERNEL and torch.cuda.is_available()):
            self.skipTest("Blackwell attn_res kernel not available")
        from tokenspeed_kernel.ops.attn_res import attn_res_fwd

        torch.manual_seed(1)
        T, H, K = 64, 7168, 8
        dev = "cuda"
        prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
        blocks = torch.randn(K, T, H, dtype=torch.bfloat16, device=dev)
        res_w = torch.randn(H, dtype=torch.bfloat16, device=dev)
        rms_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
        out_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
        got = attn_res_fwd(prefix, blocks, res_w, rms_w, _EPS, out_norm_weight=out_w)
        ref = torch_attn_res_fwd(
            layer_residual=prefix,
            block_residual=blocks,
            res_weight=res_w,
            rms_weight=rms_w,
            eps=_EPS,
            out_norm_weight=out_w,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


class RouterGateTests(unittest.TestCase):
    def test_router_dispatch_matches_fp32_gemm(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        gate = kimi_k3.KimiLinearMoEGate(hidden_size=7168, num_experts=896).cuda()
        torch.manual_seed(0)
        gate.weight.data.copy_(torch.randn(896, 7168, dtype=torch.bfloat16).cuda())
        x = torch.randn(3, 7168, dtype=torch.bfloat16, device="cuda")
        got = gate(x)
        ref = torch.nn.functional.linear(x.float(), gate.weight)
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
        # Routing consumes top-k ids: they must be identical.
        self.assertTrue(
            torch.equal(
                got.topk(16, dim=-1).indices.sort(-1).values,
                ref.topk(16, dim=-1).indices.sort(-1).values,
            )
        )

    def test_router_gemm_runs_in_fp32(self):
        gate = kimi_k3.KimiLinearMoEGate(hidden_size=_HIDDEN, num_experts=8)
        torch.manual_seed(0)
        # fp32 at rest: the checkpoint's bf16 weight is cast once at load time
        # (default_weight_loader copy_), mirrored here.
        loaded = torch.randn(8, _HIDDEN, dtype=torch.bfloat16)
        gate.weight.data.copy_(loaded)
        self.assertEqual(gate.weight.dtype, torch.float32)
        x = torch.randn(6, _HIDDEN, dtype=torch.bfloat16)
        logits = gate(x)
        # fp32 GEMM: logits must match the fully up-cast bf16 reference.
        self.assertEqual(logits.dtype, torch.float32)
        ref = torch.nn.functional.linear(x.float(), loaded.float())
        torch.testing.assert_close(logits, ref)

    def test_bf16_router_weight_uses_fp32_fallback(self):
        gate = kimi_k3.KimiLinearMoEGate(
            hidden_size=_HIDDEN,
            num_experts=8,
        ).to(dtype=torch.bfloat16)
        torch.manual_seed(1)
        gate.weight.data.copy_(torch.randn_like(gate.weight))
        x = torch.randn(2, _HIDDEN, dtype=torch.bfloat16)

        logits = gate(x)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(x.float(), gate.weight.float()),
        )


class KimiKDAMergedProjTests(unittest.TestCase):
    def test_loader_layout_and_forward_parity(self):
        torch.manual_seed(0)
        hidden, head_dim, num_heads, tp = 16, 4, 8, 2
        proj = num_heads * head_dim
        ws = {
            n: torch.randn(proj, hidden, dtype=torch.bfloat16)
            for n in ("q", "k", "v", "g")
        }
        ws["f_a"] = torch.randn(head_dim, hidden, dtype=torch.bfloat16)
        ws["b"] = torch.randn(num_heads, hidden, dtype=torch.bfloat16)
        for rank in range(tp):
            m = kimi_k3.KimiKDAMergedProj(
                hidden_size=hidden,
                proj=proj,
                num_heads=num_heads,
                head_dim=head_dim,
                tp_rank=rank,
                tp_size=tp,
            )
            for sid, w in ws.items():
                m.weight.weight_loader(m.weight, w, sid)
            x = torch.randn(3, hidden, dtype=torch.bfloat16)
            mixed_qkv, gate, f_a_out, beta = m(x)
            pl = proj // tp
            hl = num_heads // tp

            def ref(w, rows, rk=rank):
                return x @ w[rk * rows : (rk + 1) * rows].t()

            torch.testing.assert_close(
                mixed_qkv,
                torch.cat(
                    [ref(ws["q"], pl), ref(ws["k"], pl), ref(ws["v"], pl)], dim=-1
                ),
            )
            self.assertTrue(mixed_qkv.is_contiguous())
            torch.testing.assert_close(gate, ref(ws["g"], pl))
            # f_a is replicated: full output on every rank.
            torch.testing.assert_close(f_a_out, x @ ws["f_a"].t())
            torch.testing.assert_close(beta, ref(ws["b"], hl))
            # Rows are padded to the tactic-friendly multiple.
            self.assertEqual(m.weight.shape[0] % m._ROW_ALIGN, 0)

    def test_decode_single_row_slice_is_zero_copy(self):
        m = kimi_k3.KimiKDAMergedProj(
            hidden_size=8, proj=8, num_heads=2, head_dim=4, tp_rank=0, tp_size=1
        )
        torch.nn.init.normal_(m.weight)
        mixed, _, _, _ = m(torch.randn(1, 8, dtype=torch.bfloat16))
        # [1, 3p] slice of a [1, total] row is already contiguous: no copy.
        self.assertTrue(mixed.is_contiguous())


if __name__ == "__main__":
    unittest.main()
