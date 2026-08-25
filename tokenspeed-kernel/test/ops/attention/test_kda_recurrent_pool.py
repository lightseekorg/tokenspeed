"""Parity tests for the in-kernel state-pool KDA decode variant.

The pool kernel must match the reference path (gather -> fla fused_recurrent_kda
-> scatter) bit-for-bit on the math while adding dual-index page addressing:
negative write ids skip the store, read ids independent of write ids, and pages
not addressed by the batch stay untouched.
"""

from importlib.util import find_spec

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.linear.kda import (
    kda_recurrent_decode,
    kda_recurrent_decode_pool,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if find_spec("fla") is None:
    pytest.skip(
        "flash-linear-attention (fla) required for the KDA reference kernel",
        allow_module_level=True,
    )

# K3 decode shapes (TP8 rank): 12 heads, K=V=128; lower_bound from config.
HV, K, V = 12, 128, 128
LOWER_BOUND = -5.0


def _make_inputs(bs, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, bs, HV, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, bs, HV, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, bs, HV, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, bs, HV, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(1, bs, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV * K, dtype=torch.float32, device=dev)
    cu = torch.arange(bs + 1, dtype=torch.int32, device=dev)
    return q, k, v, g, beta, A_log, dt_bias, cu


def _reference(q, k, v, g, beta, A_log, dt_bias, cu, pool, read_idx, write_idx):
    """The pre-pool path: python-side gather/scatter around the fla kernel."""
    # The pool is V-major; the fla reference reads and writes K-major states.
    state = pool[read_idx.long()].transpose(-1, -2)
    out, new_state = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state=state,
        cu_seqlens=cu,
        lower_bound=LOWER_BOUND,
        beta_is_logit=True,
    )
    keep = (write_idx >= 0).view(-1, 1, 1, 1)
    out_idx = write_idx.long().clamp_min(0)
    pool[out_idx] = torch.where(
        keep, new_state.transpose(-1, -2).to(pool.dtype), pool[out_idx]
    )
    return out


class TestKdaRecurrentPool:
    @pytest.mark.parametrize("bs", [1, 4])
    def test_matches_reference(self, bs):
        q, k, v, g, beta, A_log, dt_bias, cu = _make_inputs(bs)
        num_pages = 64
        pool_ref = torch.randn(num_pages, HV, K, V, dtype=torch.float32, device="cuda")
        pool_new = pool_ref.clone()
        read_idx = torch.randperm(num_pages, device="cuda")[:bs].to(torch.int32)
        write_idx = read_idx.clone()

        ref = _reference(
            q, k, v, g, beta, A_log, dt_bias, cu, pool_ref, read_idx, write_idx
        )
        got = kda_recurrent_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            h_pool=pool_new,
            read_indices=read_idx,
            write_indices=write_idx,
            cu_seqlens=cu,
            lower_bound=LOWER_BOUND,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(pool_new, pool_ref, atol=1e-4, rtol=1e-4)

    def test_dual_index_page_crossing(self):
        # Read page != write page (logical-page boundary crossing): the old page
        # must stay unmodified and the new page must receive the state.
        bs = 3
        q, k, v, g, beta, A_log, dt_bias, cu = _make_inputs(bs, seed=1)
        pool_ref = torch.randn(32, HV, K, V, dtype=torch.float32, device="cuda")
        pool_new = pool_ref.clone()
        read_idx = torch.tensor([0, 5, 9], dtype=torch.int32, device="cuda")
        write_idx = torch.tensor([1, 6, 9], dtype=torch.int32, device="cuda")

        ref = _reference(
            q, k, v, g, beta, A_log, dt_bias, cu, pool_ref, read_idx, write_idx
        )
        got = kda_recurrent_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            h_pool=pool_new,
            read_indices=read_idx,
            write_indices=write_idx,
            cu_seqlens=cu,
            lower_bound=LOWER_BOUND,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(pool_new, pool_ref, atol=1e-4, rtol=1e-4)
        # Read pages 0/5 must be untouched (write went to 1/6).
        assert torch.equal(pool_new[0], pool_ref[0])
        assert torch.equal(pool_new[5], pool_ref[5])

    def test_negative_write_skips_store(self):
        bs = 2
        q, k, v, g, beta, A_log, dt_bias, cu = _make_inputs(bs, seed=2)
        pool = torch.randn(16, HV, K, V, dtype=torch.float32, device="cuda")
        snapshot = pool.clone()
        read_idx = torch.tensor([3, 7], dtype=torch.int32, device="cuda")
        write_idx = torch.tensor([-1, -1], dtype=torch.int32, device="cuda")

        kda_recurrent_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            h_pool=pool,
            read_indices=read_idx,
            write_indices=write_idx,
            cu_seqlens=cu,
            lower_bound=LOWER_BOUND,
        )
        assert torch.equal(pool, snapshot)

    def test_strided_qkv_views_match_contiguous(self):
        # Production feeds zero-copy slices of the packed conv output: q/k/v
        # share a [T, 2*H*K + HV*V] buffer, so their token stride exceeds
        # H*K. The kernel must honor the strides (the original port assumed
        # dense layout and cross-read tokens at bs > 1).
        bs = 5
        torch.manual_seed(7)
        dev = "cuda"
        packed = torch.randn(bs, 2 * HV * K + HV * V, dtype=torch.bfloat16, device=dev)
        q = packed[:, : HV * K].view(1, bs, HV, K)
        k = packed[:, HV * K : 2 * HV * K].view(1, bs, HV, K)
        v = packed[:, 2 * HV * K :].view(1, bs, HV, V)
        g_buf = torch.randn(bs, HV * K + 32, dtype=torch.bfloat16, device=dev)
        g = g_buf[:, : HV * K].view(1, bs, HV, K)
        beta_buf = torch.randn(bs, HV + 8, dtype=torch.bfloat16, device=dev)
        beta = beta_buf[:, :HV].view(1, bs, HV)
        A_log = torch.randn(HV, dtype=torch.float32, device=dev)
        dt_bias = torch.randn(HV * K, dtype=torch.float32, device=dev)
        cu = torch.arange(bs + 1, dtype=torch.int32, device=dev)

        pool_ref = torch.randn(64, HV, K, V, dtype=torch.float32, device=dev)
        pool_new = pool_ref.clone()
        read_idx = torch.randperm(64, device=dev)[:bs].to(torch.int32)
        write_idx = read_idx.clone()

        ref = _reference(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            A_log,
            dt_bias,
            cu,
            pool_ref,
            read_idx,
            write_idx,
        )
        got = kda_recurrent_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            h_pool=pool_new,
            read_indices=read_idx,
            write_indices=write_idx,
            cu_seqlens=cu,
            lower_bound=LOWER_BOUND,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(pool_new, pool_ref, atol=1e-4, rtol=1e-4)

    def test_negative_read_uses_zero_state(self):
        bs = 1
        q, k, v, g, beta, A_log, dt_bias, cu = _make_inputs(bs, seed=3)
        pool = torch.randn(8, HV, K, V, dtype=torch.float32, device="cuda")
        zero_page = 4
        pool[zero_page].zero_()

        got_neg = kda_recurrent_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            h_pool=pool,
            read_indices=torch.tensor([-1], dtype=torch.int32, device="cuda"),
            write_indices=torch.tensor([-1], dtype=torch.int32, device="cuda"),
            cu_seqlens=cu,
            lower_bound=LOWER_BOUND,
        )
        got_zero = kda_recurrent_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            h_pool=pool,
            read_indices=torch.tensor([zero_page], dtype=torch.int32, device="cuda"),
            write_indices=torch.tensor([-1], dtype=torch.int32, device="cuda"),
            cu_seqlens=cu,
            lower_bound=LOWER_BOUND,
        )
        torch.testing.assert_close(got_neg, got_zero)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
