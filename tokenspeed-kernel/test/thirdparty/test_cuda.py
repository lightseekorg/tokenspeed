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

"""
Unit tests for tokenspeed_kernel CUDA kernels.

Run: pytest tokenspeed-kernel/test/thirdparty/test_cuda.py -v
"""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="NVIDIA CUDA required",
)


# ─── Helpers ───


def _torch_rope_neox(query, key, positions, cos_sin_cache, head_size):
    """Pure-torch NeoX-style RoPE reference (slow but correct)."""
    nnz = query.shape[0]
    q = query.view(nnz, -1, head_size).float()
    k = key.view(nnz, -1, head_size).float()
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2

    for i in range(nnz):
        pos = positions[i].item()
        cos_vals = cos_sin_cache[pos, :half]
        sin_vals = cos_sin_cache[pos, half:]
        for h in range(q.shape[1]):
            x1 = q[i, h, :half].clone()
            x2 = q[i, h, half:rotary_dim].clone()
            q[i, h, :half] = x1 * cos_vals - x2 * sin_vals
            q[i, h, half:rotary_dim] = x2 * cos_vals + x1 * sin_vals
        for h in range(k.shape[1]):
            x1 = k[i, h, :half].clone()
            x2 = k[i, h, half:rotary_dim].clone()
            k[i, h, :half] = x1 * cos_vals - x2 * sin_vals
            k[i, h, half:rotary_dim] = x2 * cos_vals + x1 * sin_vals

    return q.to(query.dtype).view(nnz, -1), k.to(key.dtype).view(nnz, -1)


# ─── apply_rope_with_cos_sin_cache_inplace ───


class TestApplyRopeWithCosSinCacheInplace:
    """apply_rope_with_cos_sin_cache_inplace

    Supports both in-place and out-of-place mode (output_q_rope / output_k_rope).

    Signature:
      apply_rope_with_cos_sin_cache_inplace(
          positions, query, key, head_size, cos_sin_cache,
          is_neox=True, fused_set_kv_buffer_arg=None,
          output_q_rope=None, output_k_rope=None)
    """

    NNZ = 32
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_SIZE = 128
    ROTARY_DIM = 128
    MAX_SEQ_LEN = 4096

    def _make_inputs(self, seed=42):
        torch.manual_seed(seed)
        positions = torch.randint(0, self.MAX_SEQ_LEN, (self.NNZ,), device="cuda")
        query = torch.randn(
            self.NNZ,
            self.NUM_Q_HEADS * self.HEAD_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            self.NNZ,
            self.NUM_KV_HEADS * self.HEAD_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        cos_sin_cache = torch.randn(
            self.MAX_SEQ_LEN, self.ROTARY_DIM, device="cuda", dtype=torch.float32
        )
        return positions, query, key, cos_sin_cache

    def test_outofplace_correctness(self):
        """Out-of-place RoPE matches torch reference."""
        from tokenspeed_kernel.thirdparty.cuda import (
            apply_rope_with_cos_sin_cache_inplace as tk_rope,
        )

        positions, query, key, cos_sin_cache = self._make_inputs()
        q_ref, k_ref = _torch_rope_neox(
            query.clone(), key.clone(), positions, cos_sin_cache, self.HEAD_SIZE
        )

        output_q = torch.empty_like(query)
        output_k = torch.empty_like(key)
        tk_rope(
            positions,
            query,
            key,
            self.HEAD_SIZE,
            cos_sin_cache,
            is_neox=True,
            output_q_rope=output_q,
            output_k_rope=output_k,
        )
        assert torch.allclose(output_q, q_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(output_k, k_ref, atol=1e-2, rtol=1e-2)

    def test_inplace(self):
        """In-place RoPE works."""
        from tokenspeed_kernel.thirdparty.cuda import (
            apply_rope_with_cos_sin_cache_inplace as tk_rope,
        )

        positions, query, key, cos_sin_cache = self._make_inputs()
        q_orig = query.clone()
        tk_rope(positions, query, key, self.HEAD_SIZE, cos_sin_cache, is_neox=True)
        assert not torch.equal(query, q_orig), "query should be modified in-place"

    def test_outofplace(self):
        """Out-of-place RoPE works."""
        from tokenspeed_kernel.thirdparty.cuda import (
            apply_rope_with_cos_sin_cache_inplace as tk_rope,
        )

        positions, query, key, cos_sin_cache = self._make_inputs()
        q_orig = query.clone()
        output_q = torch.empty_like(query)
        output_k = torch.empty_like(key)
        tk_rope(
            positions,
            query,
            key,
            self.HEAD_SIZE,
            cos_sin_cache,
            is_neox=True,
            output_q_rope=output_q,
            output_k_rope=output_k,
        )
        assert torch.equal(
            query, q_orig
        ), "query should NOT be modified in out-of-place mode"

    def test_outofplace_matches_inplace(self):
        """Out-of-place output matches in-place output."""
        from tokenspeed_kernel.thirdparty.cuda import (
            apply_rope_with_cos_sin_cache_inplace as tk_rope,
        )

        positions, query, key, cos_sin_cache = self._make_inputs()

        # In-place
        q_ip = query.clone()
        k_ip = key.clone()
        tk_rope(positions, q_ip, k_ip, self.HEAD_SIZE, cos_sin_cache, is_neox=True)

        # Out-of-place
        output_q = torch.empty_like(query)
        output_k = torch.empty_like(key)
        tk_rope(
            positions,
            query,
            key,
            self.HEAD_SIZE,
            cos_sin_cache,
            is_neox=True,
            output_q_rope=output_q,
            output_k_rope=output_k,
        )
        assert torch.equal(output_q, q_ip)
        assert torch.equal(output_k, k_ip)

    def test_correctness(self):
        """Matches torch reference."""
        from tokenspeed_kernel.thirdparty.cuda import (
            apply_rope_with_cos_sin_cache_inplace as tk_rope,
        )

        positions, query, key, cos_sin_cache = self._make_inputs()
        q_ref, k_ref = _torch_rope_neox(
            query.clone(), key.clone(), positions, cos_sin_cache, self.HEAD_SIZE
        )

        q_tk = query.clone()
        k_tk = key.clone()
        tk_rope(positions, q_tk, k_tk, self.HEAD_SIZE, cos_sin_cache, is_neox=True)

        assert torch.allclose(q_tk, q_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(k_tk, k_ref, atol=1e-2, rtol=1e-2)

    def test_float16(self):
        """Works with float16 (not just bfloat16)."""
        from tokenspeed_kernel.thirdparty.cuda import (
            apply_rope_with_cos_sin_cache_inplace as tk_rope,
        )

        positions, _, _, cos_sin_cache = self._make_inputs()
        query = torch.randn(
            self.NNZ,
            self.NUM_Q_HEADS * self.HEAD_SIZE,
            device="cuda",
            dtype=torch.float16,
        )
        key = torch.randn(
            self.NNZ,
            self.NUM_KV_HEADS * self.HEAD_SIZE,
            device="cuda",
            dtype=torch.float16,
        )
        q_orig = query.clone()
        tk_rope(positions, query, key, self.HEAD_SIZE, cos_sin_cache, is_neox=True)
        assert not torch.equal(query, q_orig)


# ─── dsv3_router_gemm ───


class TestDsv3RouterGemm:
    """dsv3_router_gemm

    Supports num_tokens > 16 via cuBLAS fallback.

    Signature:
      dsv3_router_gemm(hidden_states, router_weights, out_dtype=torch.float32)
        -> torch.Tensor[num_tokens, num_experts]
    """

    NUM_EXPERTS = 256
    HIDDEN_DIM = 7168

    def _make_inputs(self, num_tokens=8, seed=42):
        torch.manual_seed(seed)
        hidden = torch.randn(
            num_tokens, self.HIDDEN_DIM, device="cuda", dtype=torch.bfloat16
        )
        weights = torch.randn(
            self.NUM_EXPERTS, self.HIDDEN_DIM, device="cuda", dtype=torch.bfloat16
        )
        return hidden, weights

    def _torch_ref(self, hidden, weights):
        """Torch fp32 reference: hidden @ weights.T"""
        return (hidden.float() @ weights.float().T).float()

    def test_basic(self):
        """Produces correct shape/dtype."""
        from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as tk_gemm

        hidden, weights = self._make_inputs()
        out = tk_gemm(hidden, weights, out_dtype=torch.float32)
        assert out.shape == (8, self.NUM_EXPERTS)
        assert out.dtype == torch.float32

    def test_correctness(self):
        """Matches torch reference."""
        from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as tk_gemm

        hidden, weights = self._make_inputs()
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights, out_dtype=torch.float32)
        assert torch.allclose(out, ref, atol=1e-1, rtol=1e-2)

    def test_large_batch(self):
        """Supports num_tokens > 16 (cuBLAS fallback)."""
        from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(num_tokens=64)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights, out_dtype=torch.float32)
        assert out.shape == (64, self.NUM_EXPERTS)
        assert torch.allclose(out, ref, atol=1e-1, rtol=1e-2)

    def test_fp32_weights(self):
        """Works with fp32 router weights."""
        from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as tk_gemm

        hidden, weights = self._make_inputs()
        weights_fp32 = weights.float()
        ref = self._torch_ref(hidden, weights_fp32)
        out = tk_gemm(hidden, weights_fp32, out_dtype=torch.float32)
        assert torch.allclose(out, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("num_tokens", [1, 8, 16])
    def test_correctness_varied_tokens(self, num_tokens):
        """Matches torch ref at various token counts."""
        from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(num_tokens=num_tokens)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights, out_dtype=torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
        )
        assert cos_sim > 0.99, f"cosine similarity {cos_sim} < 0.99"

    def test_large_batch_correctness(self):
        """Matches torch ref at num_tokens=64."""
        from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(num_tokens=64)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights, out_dtype=torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
        )
        assert cos_sim > 0.99, f"cosine similarity {cos_sim} < 0.99"


# ─── fp32_router_gemm ───


class TestFp32RouterGemm:
    """fp32_router_gemm

    Fused kernel for activation(bf16/fp32) x weight(fp32) -> fp32.
    Custom kernel for M<=32 (E=256, H=3072), cuBLAS fallback for larger M.

    Signature:
      fp32_router_gemm(hidden_states, router_weights)
        -> torch.Tensor[num_tokens, num_experts]
    """

    NUM_EXPERTS = 256
    HIDDEN_DIM = 3072

    def _make_inputs(self, num_tokens=8, input_dtype=torch.bfloat16, seed=42):
        torch.manual_seed(seed)
        hidden = torch.randn(
            num_tokens, self.HIDDEN_DIM, device="cuda", dtype=input_dtype
        )
        weights = torch.randn(
            self.NUM_EXPERTS, self.HIDDEN_DIM, device="cuda", dtype=torch.float32
        )
        return hidden, weights

    def _torch_ref(self, hidden, weights):
        """Torch fp32 reference: hidden.float() @ weights.float().T"""
        return (hidden.float() @ weights.float().T).float()

    def test_basic(self):
        """Produces correct shape/dtype."""
        from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm as tk_gemm

        hidden, weights = self._make_inputs()
        out = tk_gemm(hidden, weights)
        assert out.shape == (8, self.NUM_EXPERTS)
        assert out.dtype == torch.float32

    def test_correctness_bf16_input(self):
        """bf16 input x fp32 weight matches torch reference."""
        from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(input_dtype=torch.bfloat16)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights)
        assert torch.allclose(out, ref, atol=1e-1, rtol=1e-2)

    def test_correctness_fp32_input(self):
        """fp32 input x fp32 weight matches torch reference."""
        from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(input_dtype=torch.float32)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights)
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("num_tokens", [1, 8, 16, 32])
    def test_correctness_varied_tokens(self, num_tokens):
        """Matches torch ref at various token counts (custom kernel path)."""
        from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(num_tokens=num_tokens)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights)
        cos_sim = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
        )
        assert cos_sim > 0.99, f"cosine similarity {cos_sim} < 0.99"

    def test_large_batch(self):
        """M=64 falls back to cuBLAS and still correct."""
        from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm as tk_gemm

        hidden, weights = self._make_inputs(num_tokens=64)
        ref = self._torch_ref(hidden, weights)
        out = tk_gemm(hidden, weights)
        assert out.shape == (64, self.NUM_EXPERTS)
        cos_sim = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
        )
        assert cos_sim > 0.99, f"cosine similarity {cos_sim} < 0.99"


# ─── ScalarType / scalar_types ───


class TestScalarTypes:
    """ScalarType / scalar_types

    Pure Python — no CUDA kernel. Uses the local copy at compressed_tensors/scalar_type.py.
    """

    def test_local_importable(self):
        """Local copy is importable from compressed_tensors."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            ScalarType,
            scalar_types,
        )

        assert hasattr(scalar_types, "uint4b8")
        assert hasattr(scalar_types, "uint8b128")
        assert isinstance(scalar_types.uint4b8, ScalarType)

    def test_utils_uses_local(self):
        """utils.py now imports from local copy, not flashinfer."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            ScalarType as LocalScalarType,
        )
        from tokenspeed.runtime.layers.quantization.utils import ScalarType

        assert ScalarType is LocalScalarType

    def test_local_id_values(self):
        """Local ScalarType IDs have expected values and are ints."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            scalar_types as local_st,
        )

        # Verify known types exist and have integer IDs
        for name in ["uint4b8", "uint8b128", "uint4", "int8"]:
            st = getattr(local_st, name)
            assert isinstance(st.id, int), f"{name}.id should be an int"
            assert st.id >= 0, f"{name}.id should be non-negative"

    def test_local_scalar_type_fields(self):
        """Local ScalarType fields (exponent, mantissa, signed, bias) are correct for known types."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            scalar_types as local_st,
        )

        for name in ["uint4b8", "uint8b128", "uint4", "int4", "int8", "float8_e4m3fn"]:
            val = getattr(local_st, name)
            assert isinstance(val.exponent, int), f"{name} exponent should be int"
            assert isinstance(val.mantissa, int), f"{name} mantissa should be int"
            assert isinstance(val.signed, bool), f"{name} signed should be bool"
            assert isinstance(val.bias, int), f"{name} bias should be int"
            # Sanity: size_bits should be positive
            assert val.size_bits > 0, f"{name} size_bits should be positive"


# ─── gptq_marlin_repack ───


class TestGptqMarlinRepack:
    """gptq_marlin_repack

    Repacks GPTQ quantized weights into Marlin layout for efficient inference.

    Signature:
      gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
        -> torch.Tensor[int32]
    """

    def _make_inputs(self, size_k=256, size_n=64, num_bits=4, has_perm=False, seed=42):
        torch.manual_seed(seed)
        pack_factor = 32 // num_bits
        b_q_weight = torch.randint(
            0,
            2**31 - 1,
            (size_k // pack_factor, size_n),
            device="cuda",
            dtype=torch.int32,
        )
        if has_perm:
            perm = torch.randperm(size_k, device="cuda").to(torch.int32)
        else:
            perm = torch.empty((0,), device="cuda", dtype=torch.int32)
        return b_q_weight, perm

    def test_basic_4bit(self):
        """4-bit repack produces correct shape."""
        from tokenspeed_kernel.thirdparty.cuda import gptq_marlin_repack as tk_repack

        b_q, perm = self._make_inputs(num_bits=4)
        out = tk_repack(b_q, perm, 256, 64, 4)
        assert out.shape == (256 // 16, 64 * 16 // 8)
        assert out.dtype == torch.int32

    def test_basic_8bit(self):
        """8-bit repack produces correct shape."""
        from tokenspeed_kernel.thirdparty.cuda import gptq_marlin_repack as tk_repack

        b_q, perm = self._make_inputs(num_bits=8)
        out = tk_repack(b_q, perm, 256, 64, 8)
        assert out.shape == (256 // 16, 64 * 16 // 4)
        assert out.dtype == torch.int32

    def test_with_perm(self):
        """Repack with act_order permutation."""
        from tokenspeed_kernel.thirdparty.cuda import gptq_marlin_repack as tk_repack

        b_q, perm = self._make_inputs(num_bits=4, has_perm=True)
        out = tk_repack(b_q, perm, 256, 64, 4)
        assert out.shape == (256 // 16, 64 * 16 // 8)

    @pytest.mark.parametrize("num_bits", [4, 8])
    @pytest.mark.parametrize("has_perm", [False, True])
    def test_deterministic(self, num_bits, has_perm):
        """Deterministic (bitwise identical on repeated calls)."""
        from tokenspeed_kernel.thirdparty.cuda import gptq_marlin_repack as tk_repack

        b_q, perm = self._make_inputs(num_bits=num_bits, has_perm=has_perm)
        out1 = tk_repack(b_q, perm, 256, 64, num_bits)
        out2 = tk_repack(b_q, perm, 256, 64, num_bits)
        assert torch.equal(
            out1, out2
        ), "repeated calls should produce bitwise identical output"


# ─── routing_flash ───


class TestRoutingFlash:
    """routing_flash

    Fused softmax + top-k + correction bias + zero-expert masking.
    Only supports num_experts in {384, 576, 768, 896}.

    Signature:
      routing_flash(input, correction_bias, topk_indices, topk_weights,
                    num_experts_real, scaling_factor, renorm) -> None
    """

    NUM_EXPERTS = 384
    NUM_REAL_EXPERTS = 256
    TOPK = 12
    SCALE = 6.0

    def _make_inputs(self, num_tokens=16, seed=42):
        torch.manual_seed(seed)
        inp = torch.randn(
            num_tokens, self.NUM_EXPERTS, device="cuda", dtype=torch.float32
        )
        bias = torch.randn(self.NUM_EXPERTS, device="cuda", dtype=torch.float32)
        idx = torch.empty(num_tokens, self.TOPK, device="cuda", dtype=torch.int32)
        wts = torch.empty(num_tokens, self.TOPK, device="cuda", dtype=torch.float32)
        return inp, bias, idx, wts

    def _torch_ref(self, inp, bias, num_tokens):
        scores = inp.softmax(dim=-1)
        scores_biased = scores + bias.unsqueeze(0)
        topk_idx = torch.topk(scores_biased, k=self.TOPK, dim=-1, sorted=True)[1]
        topk_wts = scores.gather(1, topk_idx)
        # Zero-expert masking
        mask = topk_idx >= self.NUM_REAL_EXPERTS
        topk_idx[mask] = -1
        topk_wts *= self.SCALE
        return topk_idx.to(torch.int32), topk_wts

    def test_basic(self):
        """Produces correct shape."""
        from tokenspeed_kernel.thirdparty.cuda import routing_flash as tk_route

        inp, bias, idx, wts = self._make_inputs()
        tk_route(inp, bias, idx, wts, self.NUM_REAL_EXPERTS, self.SCALE, False)
        assert idx.shape == (16, self.TOPK)
        assert wts.shape == (16, self.TOPK)

    def test_correctness(self):
        """Matches torch reference."""
        from tokenspeed_kernel.thirdparty.cuda import routing_flash as tk_route

        inp, bias, idx, wts = self._make_inputs()
        inp_clone = inp.clone()
        ref_idx, ref_wts = self._torch_ref(inp_clone, bias, 16)
        tk_route(inp, bias, idx, wts, self.NUM_REAL_EXPERTS, self.SCALE, False)
        torch.testing.assert_close(idx, ref_idx)
        torch.testing.assert_close(wts, ref_wts, rtol=1e-3, atol=8e-2)


# ───────────────────────────────────────────────────────────────────────────────
# verify_chain_greedy & chain_speculative_sampling_target_only
# ───────────────────────────────────────────────────────────────────────────────


class TestVerifyChainGreedy:
    """verify_chain_greedy

    Correctness validated against known reference values.
    """

    DEVICE = "cuda"

    def _make_inputs(self, bs, num_draft_tokens):
        """Create standard test inputs for verify_chain_greedy."""
        candidates = torch.tensor(
            [[23958, 1266, 9400, 61749][:num_draft_tokens] for _ in range(bs)],
            dtype=torch.int32,
            device=self.DEVICE,
        )
        target_predict = torch.tensor(
            [[1266, 9400, 61749, 6620][:num_draft_tokens] for _ in range(bs)],
            dtype=torch.int64,
            device=self.DEVICE,
        )
        predicts = torch.empty_like(target_predict, dtype=torch.int32)
        accept_index = torch.full(
            (bs, num_draft_tokens), -1, dtype=torch.int32, device=self.DEVICE
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=self.DEVICE)
        return candidates, target_predict, predicts, accept_index, accept_length

    def test_basic(self):
        """All-matching candidates are accepted."""
        from tokenspeed_kernel.thirdparty.cuda import verify_chain_greedy

        bs, ndt = 4, 4
        candidates, target_predict, predicts, accept_index, accept_length = (
            self._make_inputs(bs, ndt)
        )
        verify_chain_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates,
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=ndt,
        )
        # candidates[1:] == target_predict[:3], so 3 accepted
        assert (accept_length == 3).all()
        assert torch.equal(predicts, target_predict.to(torch.int32))

    @pytest.mark.parametrize("bs", [1, 8, 32, 64, 127])
    def test_batch_sizes(self, bs):
        """Correct across various batch sizes."""
        from tokenspeed_kernel.thirdparty.cuda import verify_chain_greedy

        ndt = 4
        candidates, target_predict, predicts, accept_index, accept_length = (
            self._make_inputs(bs, ndt)
        )
        verify_chain_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates,
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=ndt,
        )
        assert (accept_length == 3).all()
        assert torch.equal(predicts, target_predict.to(torch.int32))

    def test_no_match(self):
        """Verify behavior when no candidates match target predictions."""
        from tokenspeed_kernel.thirdparty.cuda import verify_chain_greedy

        bs, ndt = 2, 4
        candidates = torch.tensor(
            [[100, 200, 300, 400] for _ in range(bs)],
            dtype=torch.int32,
            device=self.DEVICE,
        )
        target_predict = torch.tensor(
            [[999, 998, 997, 996] for _ in range(bs)],
            dtype=torch.int64,
            device=self.DEVICE,
        )
        predicts = torch.empty(bs, ndt, dtype=torch.int32, device=self.DEVICE)
        accept_index = torch.full((bs, ndt), -1, dtype=torch.int32, device=self.DEVICE)
        accept_length = torch.empty(bs, dtype=torch.int32, device=self.DEVICE)

        verify_chain_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates,
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=ndt,
        )
        assert (accept_length == 0).all()

    def test_partial_match(self):
        """Verify behavior when only first candidate matches."""
        from tokenspeed_kernel.thirdparty.cuda import verify_chain_greedy

        bs, ndt = 2, 4
        candidates = torch.tensor(
            [[10, 20, 30, 40] for _ in range(bs)],
            dtype=torch.int32,
            device=self.DEVICE,
        )
        # Only first target matches candidates[1] (20), rest differ
        target_predict = torch.tensor(
            [[20, 999, 998, 997] for _ in range(bs)],
            dtype=torch.int64,
            device=self.DEVICE,
        )
        predicts = torch.empty(bs, ndt, dtype=torch.int32, device=self.DEVICE)
        accept_index = torch.full((bs, ndt), -1, dtype=torch.int32, device=self.DEVICE)
        accept_length = torch.empty(bs, dtype=torch.int32, device=self.DEVICE)

        verify_chain_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates,
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=ndt,
        )
        assert (accept_length == 1).all()


class TestChainSpeculativeSamplingTargetOnly:
    """chain_speculative_sampling_target_only

    Correctness validated against known reference values.
    """

    DEVICE = "cuda"

    def _make_deterministic_inputs(self, batch_size, num_draft_tokens=4, vocab_size=16):
        """Create deterministic test inputs matching fork's reference test."""
        candidates = torch.tensor(
            [[10, 11, 12, 13][:num_draft_tokens] for _ in range(batch_size)],
            dtype=torch.int32,
            device=self.DEVICE,
        )
        target_probs = torch.zeros(
            (batch_size, num_draft_tokens, vocab_size), device=self.DEVICE
        )
        target_probs[:, 0, 11] = 0.95  # high prob > threshold_single (0.9)
        target_probs[:, 1, 12] = 0.1  # low prob
        target_probs[:, 1, 5] = 1.0  # high resample prob for vocab 5

        uniform_samples = torch.tensor(
            [[0.5, 0.9, 0.5, 0.5][:num_draft_tokens] for _ in range(batch_size)],
            dtype=torch.float32,
            device=self.DEVICE,
        )
        uniform_samples_final = torch.tensor(
            [0.1] * batch_size, dtype=torch.float32, device=self.DEVICE
        )

        predicts = torch.zeros(
            batch_size * num_draft_tokens, dtype=torch.int32, device=self.DEVICE
        )
        accept_index = torch.zeros(
            (batch_size, num_draft_tokens), dtype=torch.int32, device=self.DEVICE
        )
        accept_token_num = torch.zeros(
            batch_size, dtype=torch.int32, device=self.DEVICE
        )
        draft_probs = torch.zeros_like(target_probs)

        return (
            candidates,
            target_probs,
            uniform_samples,
            uniform_samples_final,
            predicts,
            accept_index,
            accept_token_num,
            draft_probs,
        )

    def test_deterministic(self):
        """Deterministic chain speculative sampling."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        bs, ndt = 4, 4
        (
            candidates,
            target_probs,
            uniform_samples,
            uniform_samples_final,
            predicts,
            accept_index,
            accept_token_num,
            draft_probs,
        ) = self._make_deterministic_inputs(bs, ndt)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            draft_probs,
            0.9,
            1.0,
        )
        assert (accept_token_num == 1).all()
        pred_reshaped = predicts.reshape(bs, ndt)
        for i in range(bs):
            assert pred_reshaped[i, 0].item() == 11  # accepted
            assert pred_reshaped[i, 1].item() == 5  # resampled

    @pytest.mark.parametrize("bs", [1, 8, 32, 64, 127])
    def test_batch_sizes(self, bs):
        """Deterministic sampling across batch sizes."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        ndt, vocab_size = 4, 16
        (
            candidates,
            target_probs,
            uniform_samples,
            uniform_samples_final,
            predicts,
            accept_index,
            accept_token_num,
            draft_probs,
        ) = self._make_deterministic_inputs(bs, ndt, vocab_size)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            draft_probs,
            0.9,
            1.0,
        )
        assert (accept_token_num == 1).all()
        pred_reshaped = predicts.reshape(bs, ndt)
        for i in range(bs):
            assert pred_reshaped[i, 0].item() == 11
            assert pred_reshaped[i, 1].item() == 5

    def test_accept_index_values(self):
        """Verify accept_index values match expected pattern."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        bs, ndt, vocab = 4, 4, 16
        (
            candidates,
            target_probs,
            uniform_samples,
            uniform_samples_final,
            predicts,
            accept_index,
            accept_token_num,
            draft_probs,
        ) = self._make_deterministic_inputs(bs, ndt, vocab)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            draft_probs,
            0.9,
            1.0,
        )
        expected_accept_index = torch.zeros(
            (bs, ndt), dtype=torch.int32, device=self.DEVICE
        )
        for i in range(bs):
            expected_accept_index[i, 0] = i * 4
            expected_accept_index[i, 1] = i * 4 + 1
        torch.testing.assert_close(accept_index, expected_accept_index)

    def test_all_accepted(self):
        """Test when all draft tokens match target (high threshold)."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        bs, ndt, vocab = 2, 4, 16
        candidates = torch.tensor(
            [[5, 6, 7, 8] for _ in range(bs)],
            dtype=torch.int32,
            device=self.DEVICE,
        )
        target_probs = torch.zeros((bs, ndt, vocab), device=self.DEVICE)
        for t in range(ndt):
            target_probs[:, t, candidates[0, t].item()] = 1.0

        draft_probs = torch.zeros_like(target_probs)
        uniform_samples = torch.full(
            (bs, ndt), 0.01, dtype=torch.float32, device=self.DEVICE
        )
        uniform_samples_final = torch.full(
            (bs,), 0.01, dtype=torch.float32, device=self.DEVICE
        )
        predicts = torch.zeros(bs * ndt, dtype=torch.int32, device=self.DEVICE)
        accept_index = torch.zeros((bs, ndt), dtype=torch.int32, device=self.DEVICE)
        accept_token_num = torch.zeros(bs, dtype=torch.int32, device=self.DEVICE)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            draft_probs,
            0.0,  # threshold_single=0 means always accept
            1.0,
        )
        # ndt-1 draft tokens accepted (last position generates new prediction)
        assert (accept_token_num == ndt - 1).all()

    # ------------------------------------------------------------------
    # draft_probs=None path: kernel skips the GMEM round-trip and tracks
    # the rejected token's id in a register instead. Output triple must
    # match the legacy draft_probs=zeros path bit-exactly.
    # ------------------------------------------------------------------

    def test_none_draft_probs_deterministic(self):
        """draft_probs=None matches the deterministic baseline outputs."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        bs, ndt = 4, 4
        (
            candidates,
            target_probs,
            uniform_samples,
            uniform_samples_final,
            predicts,
            accept_index,
            accept_token_num,
            _draft_probs,
        ) = self._make_deterministic_inputs(bs, ndt)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            None,
            0.9,
            1.0,
        )
        assert (accept_token_num == 1).all()
        pred_reshaped = predicts.reshape(bs, ndt)
        for i in range(bs):
            assert pred_reshaped[i, 0].item() == 11  # accepted
            assert pred_reshaped[i, 1].item() == 5  # resampled

    @pytest.mark.parametrize("bs", [1, 8, 32, 64, 127])
    def test_none_draft_probs_matches_zeros_buffer(self, bs):
        """draft_probs=None produces bit-identical (predicts, accept_index,
        accept_token_num) to passing torch.zeros_like(target_probs)."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        ndt, vocab = 4, 16

        # First call: legacy path with explicit zeros buffer.
        (
            candidates_a,
            target_probs_a,
            uniform_samples_a,
            uniform_samples_final_a,
            predicts_a,
            accept_index_a,
            accept_token_num_a,
            draft_probs_a,
        ) = self._make_deterministic_inputs(bs, ndt, vocab)
        chain_speculative_sampling_target_only(
            predicts_a,
            accept_index_a,
            accept_token_num_a,
            candidates_a,
            uniform_samples_a,
            uniform_samples_final_a,
            target_probs_a,
            draft_probs_a,
            0.9,
            1.0,
        )

        # Second call: new path with draft_probs=None.
        (
            candidates_b,
            target_probs_b,
            uniform_samples_b,
            uniform_samples_final_b,
            predicts_b,
            accept_index_b,
            accept_token_num_b,
            _,
        ) = self._make_deterministic_inputs(bs, ndt, vocab)
        chain_speculative_sampling_target_only(
            predicts_b,
            accept_index_b,
            accept_token_num_b,
            candidates_b,
            uniform_samples_b,
            uniform_samples_final_b,
            target_probs_b,
            None,
            0.9,
            1.0,
        )

        torch.testing.assert_close(predicts_a, predicts_b, rtol=0, atol=0)
        torch.testing.assert_close(accept_index_a, accept_index_b, rtol=0, atol=0)
        torch.testing.assert_close(
            accept_token_num_a, accept_token_num_b, rtol=0, atol=0
        )

    def test_none_draft_probs_all_accepted(self):
        """All-accepted path also works with draft_probs=None."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        bs, ndt, vocab = 2, 4, 16
        candidates = torch.tensor(
            [[5, 6, 7, 8] for _ in range(bs)],
            dtype=torch.int32,
            device=self.DEVICE,
        )
        target_probs = torch.zeros((bs, ndt, vocab), device=self.DEVICE)
        for t in range(ndt):
            target_probs[:, t, candidates[0, t].item()] = 1.0
        uniform_samples = torch.full(
            (bs, ndt), 0.01, dtype=torch.float32, device=self.DEVICE
        )
        uniform_samples_final = torch.full(
            (bs,), 0.01, dtype=torch.float32, device=self.DEVICE
        )
        predicts = torch.zeros(bs * ndt, dtype=torch.int32, device=self.DEVICE)
        accept_index = torch.zeros((bs, ndt), dtype=torch.int32, device=self.DEVICE)
        accept_token_num = torch.zeros(bs, dtype=torch.int32, device=self.DEVICE)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            None,
            0.0,
            1.0,
        )
        assert (accept_token_num == ndt - 1).all()

    def test_draft_probs_writeback_preserved(self):
        """When draft_probs IS provided, the kernel must still write back the
        rejected position once at kernel exit (legacy observable behavior)."""
        from tokenspeed_kernel.thirdparty.cuda import (
            chain_speculative_sampling_target_only,
        )

        bs, ndt = 4, 4
        (
            candidates,
            target_probs,
            uniform_samples,
            uniform_samples_final,
            predicts,
            accept_index,
            accept_token_num,
            draft_probs,
        ) = self._make_deterministic_inputs(bs, ndt)

        # Sentinel: fill draft_probs with -1 so unmodified positions are
        # easily distinguished from kernel writes.
        draft_probs.fill_(-1.0)

        chain_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            uniform_samples,
            uniform_samples_final,
            target_probs,
            draft_probs,
            0.9,
            1.0,
        )
        # Expected: at rejection, kernel writes
        # draft_probs[batch, rejected_pos, draft_id] = target_probs[same].
        # For this fixture: rejection happens at pos 1, draft_id = 12.
        # target_probs[:, 1, 12] = 0.1.
        for i in range(bs):
            assert draft_probs[i, 1, 12].item() == target_probs[i, 1, 12].item()


# ───────────────────────────────────────────────────────────────────────────────
# rmsnorm_fused_parallel
# ───────────────────────────────────────────────────────────────────────────────


def _rmsnorm_ref(x, weight, eps):
    """Reference RMSNorm implementation in PyTorch."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


class TestRMSNormFusedParallel:
    """rmsnorm_fused_parallel"""

    DEVICE = "cuda"

    def test_basic_bf16(self):
        """Basic fused parallel RMSNorm with bfloat16."""
        from tokenspeed_kernel.thirdparty.cuda import rmsnorm_fused_parallel

        bs, dim1, dim2 = 4, 128, 64
        eps = 1e-6
        input1 = torch.randn(bs, dim1, device=self.DEVICE, dtype=torch.bfloat16)
        weight1 = torch.randn(dim1, device=self.DEVICE, dtype=torch.bfloat16)
        output1 = torch.empty_like(input1)
        input2 = torch.randn(bs, dim2, device=self.DEVICE, dtype=torch.bfloat16)
        weight2 = torch.randn(dim2, device=self.DEVICE, dtype=torch.bfloat16)
        output2 = torch.empty_like(input2)

        rmsnorm_fused_parallel(input1, weight1, output1, input2, weight2, output2, eps)

        ref1 = _rmsnorm_ref(input1, weight1, eps)
        ref2 = _rmsnorm_ref(input2, weight2, eps)
        torch.testing.assert_close(output1, ref1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(output2, ref2, atol=1e-2, rtol=1e-2)

    def test_basic_fp16(self):
        """Basic fused parallel RMSNorm with float16."""
        from tokenspeed_kernel.thirdparty.cuda import rmsnorm_fused_parallel

        bs, dim1, dim2 = 4, 256, 128
        eps = 1e-5
        input1 = torch.randn(bs, dim1, device=self.DEVICE, dtype=torch.float16)
        weight1 = torch.randn(dim1, device=self.DEVICE, dtype=torch.float16)
        output1 = torch.empty_like(input1)
        input2 = torch.randn(bs, dim2, device=self.DEVICE, dtype=torch.float16)
        weight2 = torch.randn(dim2, device=self.DEVICE, dtype=torch.float16)
        output2 = torch.empty_like(input2)

        rmsnorm_fused_parallel(input1, weight1, output1, input2, weight2, output2, eps)

        ref1 = _rmsnorm_ref(input1, weight1, eps)
        ref2 = _rmsnorm_ref(input2, weight2, eps)
        torch.testing.assert_close(output1, ref1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(output2, ref2, atol=1e-2, rtol=1e-2)

    def test_inplace(self):
        """In-place output (output == input)."""
        from tokenspeed_kernel.thirdparty.cuda import rmsnorm_fused_parallel

        bs, dim1, dim2 = 4, 128, 64
        eps = 1e-6
        input1 = torch.randn(bs, dim1, device=self.DEVICE, dtype=torch.bfloat16)
        input2 = torch.randn(bs, dim2, device=self.DEVICE, dtype=torch.bfloat16)
        weight1 = torch.randn(dim1, device=self.DEVICE, dtype=torch.bfloat16)
        weight2 = torch.randn(dim2, device=self.DEVICE, dtype=torch.bfloat16)

        ref1 = _rmsnorm_ref(input1, weight1, eps)
        ref2 = _rmsnorm_ref(input2, weight2, eps)

        # In-place: output is the same tensor as input
        rmsnorm_fused_parallel(input1, weight1, input1, input2, weight2, input2, eps)

        torch.testing.assert_close(input1, ref1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(input2, ref2, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("bs", [1, 16, 64])
    def test_batch_sizes(self, bs):
        """Various batch sizes."""
        from tokenspeed_kernel.thirdparty.cuda import rmsnorm_fused_parallel

        dim1, dim2 = 1536, 512
        eps = 1e-5
        input1 = torch.randn(bs, dim1, device=self.DEVICE, dtype=torch.bfloat16)
        weight1 = torch.randn(dim1, device=self.DEVICE, dtype=torch.bfloat16)
        output1 = torch.empty_like(input1)
        input2 = torch.randn(bs, dim2, device=self.DEVICE, dtype=torch.bfloat16)
        weight2 = torch.randn(dim2, device=self.DEVICE, dtype=torch.bfloat16)
        output2 = torch.empty_like(input2)

        rmsnorm_fused_parallel(input1, weight1, output1, input2, weight2, output2, eps)

        ref1 = _rmsnorm_ref(input1, weight1, eps)
        ref2 = _rmsnorm_ref(input2, weight2, eps)
        torch.testing.assert_close(output1, ref1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(output2, ref2, atol=1e-2, rtol=1e-2)

    def test_deepseek_dims(self):
        """DeepSeek model dimensions (q_a=1536, kv_a=512)."""
        from tokenspeed_kernel.thirdparty.cuda import rmsnorm_fused_parallel

        bs = 32
        q_a_dim, kv_a_dim = 1536, 512
        eps = 1e-6
        input_q_a = torch.randn(bs, q_a_dim, device=self.DEVICE, dtype=torch.bfloat16)
        weight_q_a = torch.ones(q_a_dim, device=self.DEVICE, dtype=torch.bfloat16)
        output_q_a = torch.empty_like(input_q_a)
        input_kv_a = torch.randn(bs, kv_a_dim, device=self.DEVICE, dtype=torch.bfloat16)
        weight_kv_a = torch.ones(kv_a_dim, device=self.DEVICE, dtype=torch.bfloat16)
        output_kv_a = torch.empty_like(input_kv_a)

        rmsnorm_fused_parallel(
            input_q_a, weight_q_a, output_q_a, input_kv_a, weight_kv_a, output_kv_a, eps
        )

        ref_q = _rmsnorm_ref(input_q_a, weight_q_a, eps)
        ref_kv = _rmsnorm_ref(input_kv_a, weight_kv_a, eps)
        torch.testing.assert_close(output_q_a, ref_q, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(output_kv_a, ref_kv, atol=1e-2, rtol=1e-2)


# ───────────────────────────────────────────────────────────────────────────────
# silu_and_mul_fuse_block_quant
# ───────────────────────────────────────────────────────────────────────────────


def _silu_ref(x):
    """Reference SiLU activation."""
    return x * torch.sigmoid(x)


class TestSiluAndMulFuseBlockQuant:
    """silu_and_mul_fuse_block_quant

     Scale tensors must be column-major (stride(1) > stride(0)).
    Uses torch.empty(...).mT.contiguous().mT for proper Fortran-order allocation.
    """

    DEVICE = "cuda"
    BLOCK_SIZE = 128

    def _col_major_scale(self, rows, cols):
        """Allocate a column-major (Fortran-order) scale tensor."""
        return (
            torch.zeros(cols, rows, device=self.DEVICE, dtype=torch.float32)
            .contiguous()
            .t()
        )

    def _col_major_scale_3d(self, batch, rows, cols):
        """Allocate a 3D scale tensor with column-major last 2 dims."""
        return (
            torch.zeros(batch, cols, rows, device=self.DEVICE, dtype=torch.float32)
            .contiguous()
            .permute(0, 2, 1)
        )

    def test_basic(self):
        """Basic silu_and_mul_fuse_block_quant."""
        from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

        num_tokens, hidden_size = 16, 256
        x = torch.randn(
            num_tokens, 2 * hidden_size, device=self.DEVICE, dtype=torch.bfloat16
        )
        out = torch.empty(
            num_tokens, hidden_size, device=self.DEVICE, dtype=torch.float8_e4m3fn
        )
        num_blocks = hidden_size // self.BLOCK_SIZE
        scale_out = self._col_major_scale(num_tokens, num_blocks)

        result_out, result_scale = silu_and_mul_fuse_block_quant(
            x, scale_out, out, enable_pdl=False
        )

        assert result_out.dtype == torch.float8_e4m3fn
        assert result_out.shape == (num_tokens, hidden_size)
        # Scale should have been written (not all zeros)
        assert result_scale.abs().sum() > 0

    @pytest.mark.parametrize("num_tokens", [4, 16, 64])
    def test_batch_sizes(self, num_tokens):
        """Various batch sizes."""
        from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

        hidden_size = 256
        x = torch.randn(
            num_tokens, 2 * hidden_size, device=self.DEVICE, dtype=torch.bfloat16
        )
        out = torch.empty(
            num_tokens, hidden_size, device=self.DEVICE, dtype=torch.float8_e4m3fn
        )
        num_blocks = hidden_size // self.BLOCK_SIZE
        scale_out = self._col_major_scale(num_tokens, num_blocks)

        result_out, result_scale = silu_and_mul_fuse_block_quant(
            x, scale_out, out, enable_pdl=False
        )
        assert result_out.dtype == torch.float8_e4m3fn
        assert result_out.shape == (num_tokens, hidden_size)

    def test_ep_variant(self):
        """EP variant with num_tokens_per_expert."""
        from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

        num_experts, max_tokens, hidden_size = 4, 8, 256
        x = torch.randn(
            num_experts,
            max_tokens,
            2 * hidden_size,
            device=self.DEVICE,
            dtype=torch.bfloat16,
        )
        out = torch.empty(
            num_experts,
            max_tokens,
            hidden_size,
            device=self.DEVICE,
            dtype=torch.float8_e4m3fn,
        )
        num_blocks = hidden_size // self.BLOCK_SIZE
        scale_out = self._col_major_scale_3d(num_experts, max_tokens, num_blocks)

        num_tokens_per_expert = torch.full(
            (num_experts,), max_tokens, device=self.DEVICE, dtype=torch.int32
        )

        result_out, result_scale = silu_and_mul_fuse_block_quant(
            x,
            scale_out,
            out,
            enable_pdl=False,
            num_tokens_per_expert=num_tokens_per_expert,
            num_tokens_hint=max_tokens,
            num_experts=num_experts,
        )
        assert result_out.dtype == torch.float8_e4m3fn
        assert result_out.shape == (num_experts, max_tokens, hidden_size)

    def test_correctness(self):
        """FP8 output approximately matches SiLU+Mul reference."""
        from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

        num_tokens, hidden_size = 16, 256
        x = torch.randn(
            num_tokens, 2 * hidden_size, device=self.DEVICE, dtype=torch.bfloat16
        )
        out = torch.empty(
            num_tokens, hidden_size, device=self.DEVICE, dtype=torch.float8_e4m3fn
        )
        num_blocks = hidden_size // self.BLOCK_SIZE  # = 2 for hidden=256
        scale_out = self._col_major_scale(num_tokens, num_blocks)

        silu_and_mul_fuse_block_quant(x, scale_out, out, enable_pdl=False)

        # Reference
        gate = x[..., :hidden_size].float()
        up = x[..., hidden_size:].float()
        ref = _silu_ref(gate) * up

        # Dequantize block-quantized FP8: scale_out is (num_tokens, num_blocks) col-major
        # Each block of 128 elements in the output row has its own scale
        deq = torch.zeros_like(ref)
        for t in range(num_tokens):
            for b in range(num_blocks):
                s = scale_out[t, b]
                deq[t, b * self.BLOCK_SIZE : (b + 1) * self.BLOCK_SIZE] = (
                    out[t, b * self.BLOCK_SIZE : (b + 1) * self.BLOCK_SIZE].float() * s
                )

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), deq.flatten().unsqueeze(0)
        )
        assert cos_sim > 0.99, f"cosine similarity {cos_sim} < 0.99"

    def test_fp16(self):
        """Float16 input."""
        from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

        num_tokens, hidden_size = 8, 256
        x = torch.randn(
            num_tokens, 2 * hidden_size, device=self.DEVICE, dtype=torch.float16
        )
        out = torch.empty(
            num_tokens, hidden_size, device=self.DEVICE, dtype=torch.float8_e4m3fn
        )
        num_blocks = hidden_size // self.BLOCK_SIZE
        scale_out = self._col_major_scale(num_tokens, num_blocks)

        result_out, _ = silu_and_mul_fuse_block_quant(
            x, scale_out, out, enable_pdl=False
        )
        assert result_out.dtype == torch.float8_e4m3fn
