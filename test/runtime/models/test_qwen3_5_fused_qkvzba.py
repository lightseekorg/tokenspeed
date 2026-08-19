"""Correctness + perf guard for fused_qkvzba_split_reshape_cat_contiguous.

The kernel's v/z access was reworked from one V_PER_GROUP * HEAD_V arange
into a per-head static_range loop so any integer head ratio works (Triton
arange requires a power-of-2 span; ratio 3 * 128 = 384 is not one).

- correctness: bit-exact vs a plain torch split reference for ratios
  1/2/3/4, plus 16-byte alignment of every output (flashinfer's CuteDSL
  gdn_decode_mtp requirement).
- perf: the looped kernel must not regress vs the previous wide-arange
  implementation on the power-of-2 ratios it used to handle (1/2/4).
"""

import os

# CI Registration (parsed via AST, runtime no-op)
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

try:
    import torch
    import triton
    import triton.language as tl

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

if HAS_CUDA:

    @triton.jit
    def _legacy_wide_arange_kernel(
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        stride_qkvz,
        stride_ba,
        NUM_HEADS_QK: tl.constexpr,
        NUM_HEADS_V: tl.constexpr,
        HEAD_QK: tl.constexpr,
        HEAD_V: tl.constexpr,
    ):
        # Pre-rework implementation (power-of-2 ratios only), kept here as
        # the perf baseline for the regression bound.
        i_bs, i_qk = tl.program_id(0), tl.program_id(1)
        V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
        TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
        TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
        TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V
        QKV_DIM_T: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V

        blk_q_ptr = (
            mixed_qkvz + i_bs * stride_qkvz + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
        )
        blk_k_ptr = (
            mixed_qkvz
            + i_bs * stride_qkvz
            + TOTAL_Q
            + i_qk * HEAD_QK
            + tl.arange(0, HEAD_QK)
        )
        blk_v_ptr = (
            mixed_qkvz
            + i_bs * stride_qkvz
            + TOTAL_Q
            + TOTAL_K
            + i_qk * V_PER_GROUP * HEAD_V
            + tl.arange(0, V_PER_GROUP * HEAD_V)
        )
        blk_z_ptr = (
            mixed_qkvz
            + i_bs * stride_qkvz
            + TOTAL_Q
            + TOTAL_K
            + TOTAL_V
            + i_qk * V_PER_GROUP * HEAD_V
            + tl.arange(0, V_PER_GROUP * HEAD_V)
        )
        blk_q_st_ptr = (
            mixed_qkv + i_bs * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
        )
        blk_k_st_ptr = (
            mixed_qkv
            + i_bs * QKV_DIM_T
            + NUM_HEADS_QK * HEAD_QK
            + i_qk * HEAD_QK
            + tl.arange(0, HEAD_QK)
        )
        blk_v_st_ptr = (
            mixed_qkv
            + i_bs * QKV_DIM_T
            + NUM_HEADS_QK * HEAD_QK * 2
            + i_qk * V_PER_GROUP * HEAD_V
            + tl.arange(0, V_PER_GROUP * HEAD_V)
        )
        blk_z_st_ptr = (
            z
            + i_bs * NUM_HEADS_V * HEAD_V
            + i_qk * V_PER_GROUP * HEAD_V
            + tl.arange(0, V_PER_GROUP * HEAD_V)
        )
        tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
        tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
        tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
        tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
        for i in tl.static_range(V_PER_GROUP):
            blk_b_ptr = mixed_ba + i_bs * stride_ba + i_qk * V_PER_GROUP + i
            blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
            tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
        for i in tl.static_range(V_PER_GROUP):
            blk_a_ptr = (
                mixed_ba + i_bs * stride_ba + NUM_HEADS_V + i_qk * V_PER_GROUP + i
            )
            blk_a_st_ptr = a + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
            tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))

    def _legacy_launch(mixed_qkvz, mixed_ba, nk, nv, head_qk, head_v):
        batch = mixed_qkvz.shape[0]
        qkv_dim_t = nk * head_qk * 2 + nv * head_v
        mixed_qkv = torch.empty(
            [batch, qkv_dim_t], dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
        )
        z = torch.empty(
            [batch, nv, head_v], dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
        )
        b = torch.empty([batch, nv], dtype=mixed_ba.dtype, device=mixed_ba.device)
        a = torch.empty_like(b)
        _legacy_wide_arange_kernel[(batch, nk)](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            mixed_qkvz.stride(0),
            mixed_ba.stride(0),
            nk,
            nv,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
        return mixed_qkv, z, b, a


HEAD_QK = HEAD_V = 128


def _make_inputs(batch, nk, nv, device="cuda"):
    qkvz_dim = nk * HEAD_QK * 2 + nv * HEAD_V * 2
    mixed_qkvz = torch.randn(batch, qkvz_dim, dtype=torch.bfloat16, device=device)
    mixed_ba = torch.randn(batch, nv * 2, dtype=torch.bfloat16, device=device)
    return mixed_qkvz, mixed_ba


def _bench_interleaved(fn_a, fn_b, rounds=5):
    """Time two candidates fairly and return their fastest observations."""
    best_a = best_b = float("inf")
    for _ in range(rounds):
        best_a = min(
            best_a, triton.testing.do_bench(fn_a, warmup=25, rep=50, return_mode="min")
        )
        best_b = min(
            best_b, triton.testing.do_bench(fn_b, warmup=25, rep=50, return_mode="min")
        )
    return best_a, best_b


@unittest.skipUnless(HAS_CUDA, "needs CUDA + triton")
class FusedQkvzbaTest(unittest.TestCase):
    def _fused(self):
        from tokenspeed.runtime.models.qwen3_5 import (
            fused_qkvzba_split_reshape_cat_contiguous,
        )

        return fused_qkvzba_split_reshape_cat_contiguous

    def test_matches_split_reference_for_all_integer_ratios(self):
        fused = self._fused()
        torch.manual_seed(0)
        for nk, nv in [(4, 4), (4, 8), (4, 12), (4, 16), (2, 6)]:
            mixed_qkvz, mixed_ba = _make_inputs(64, nk, nv)
            mixed_qkv, z, b, a = fused(mixed_qkvz, mixed_ba, nk, nv, HEAD_QK, HEAD_V)

            q_r, k_r, v_r, z_r = mixed_qkvz.split(
                [nk * HEAD_QK, nk * HEAD_QK, nv * HEAD_V, nv * HEAD_V], dim=-1
            )
            b_r, a_r = mixed_ba.split([nv, nv], dim=-1)

            ratio = nv // nk
            self.assertTrue(
                torch.equal(mixed_qkv, torch.cat([q_r, k_r, v_r], dim=-1)),
                f"qkv mismatch at ratio {ratio}",
            )
            self.assertTrue(
                torch.equal(z.reshape(z.shape[0], -1), z_r),
                f"z mismatch at ratio {ratio}",
            )
            self.assertTrue(torch.equal(b, b_r), f"b mismatch at ratio {ratio}")
            self.assertTrue(torch.equal(a, a_r), f"a mismatch at ratio {ratio}")
            for name, t in [("mixed_qkv", mixed_qkv), ("z", z), ("b", b), ("a", a)]:
                self.assertEqual(
                    t.data_ptr() % 16, 0, f"{name} misaligned at ratio {ratio}"
                )
                self.assertTrue(
                    t.is_contiguous(), f"{name} non-contiguous at ratio {ratio}"
                )

    def test_supports_noncontiguous_input_rows(self):
        fused = self._fused()
        torch.manual_seed(1)
        nk, nv = 4, 12
        mixed_qkvz, mixed_ba = _make_inputs(32, nk, nv)
        wide_qkvz = torch.cat([mixed_qkvz, mixed_qkvz], dim=-1)[
            :, : mixed_qkvz.shape[1]
        ]
        # build a strided (non-contiguous-row) view with identical content
        padded = torch.zeros(
            32, mixed_qkvz.shape[1] + 64, dtype=torch.bfloat16, device="cuda"
        )
        padded[:, : mixed_qkvz.shape[1]] = mixed_qkvz
        strided = padded[:, : mixed_qkvz.shape[1]]
        self.assertFalse(strided.is_contiguous())
        del wide_qkvz

        out_s = fused(strided, mixed_ba, nk, nv, HEAD_QK, HEAD_V)
        out_c = fused(mixed_qkvz, mixed_ba, nk, nv, HEAD_QK, HEAD_V)
        for ts, tc in zip(out_s, out_c):
            self.assertTrue(torch.equal(ts, tc))

    def test_perf_no_regression_vs_legacy_wide_arange(self):
        fused = self._fused()
        torch.manual_seed(2)
        results = {}
        for nk, nv in [(4, 4), (4, 8), (4, 16)]:  # legacy-supported ratios
            mixed_qkvz, mixed_ba = _make_inputs(8192, nk, nv)  # prefill-sized
            args = (mixed_qkvz, mixed_ba, nk, nv, HEAD_QK, HEAD_V)
            t_old, t_new = _bench_interleaved(
                lambda a=args: _legacy_launch(*a),
                lambda a=args: fused(*a),
            )
            results[nv // nk] = (t_old, t_new)

        for ratio, (t_old, t_new) in results.items():
            # pow2 ratios compile to the same wide-arange path as before;
            # allow 20% noise margin on a memory-bound microkernel
            self.assertLessEqual(
                t_new,
                t_old * 1.20,
                f"ratio {ratio}: kernel {t_new:.4f}ms vs "
                f"legacy {t_old:.4f}ms (>20% regression)",
            )

    def test_perf_ratio3_beats_torch_fallback(self):
        fused = self._fused()
        torch.manual_seed(3)
        nk, nv = 4, 12
        mixed_qkvz, mixed_ba = _make_inputs(8192, nk, nv)

        def torch_fallback():
            # mirrors fix_query_key_value_ordering + the cat in forward()
            q, k, v, z = mixed_qkvz.split(
                [nk * HEAD_QK, nk * HEAD_QK, nv * HEAD_V, nv * HEAD_V], dim=-1
            )
            b, a = mixed_ba.split([nv, nv], dim=-1)
            b, a = b.contiguous(), a.contiguous()
            z = z.reshape(z.size(0), nv, HEAD_V)
            return torch.cat((q, k, v), dim=-1), z, b, a

        args = (mixed_qkvz, mixed_ba, nk, nv, HEAD_QK, HEAD_V)
        t_fused, t_torch = _bench_interleaved(
            lambda: fused(*args),
            torch_fallback,
        )
        self.assertLessEqual(
            t_fused,
            t_torch,
            f"ratio 3 fused {t_fused:.4f}ms slower than torch "
            f"fallback {t_torch:.4f}ms",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
