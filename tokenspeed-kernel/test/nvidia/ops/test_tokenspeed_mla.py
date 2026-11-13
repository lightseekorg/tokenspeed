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

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mla import torch_mla_kv_pack
from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla import (
    mla_kv_pack_quantize_fp8,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="tokenspeed_mla kernels are NVIDIA-only",
)

# K2.5 / DSv3 chunked-prefill shape with TP=4.
S = 256
H = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _make_kv_slice_inputs(device: str, dtype: torch.dtype = torch.bfloat16):
    """Mirror the deepseek_v3.py call site: k_nope and v are slice views of
    a packed kv tensor produced by kv_b_proj."""
    torch.manual_seed(0)
    kv = torch.randn(S, H, QK_NOPE + V_HEAD, device=device, dtype=dtype)
    k_nope = kv[..., :QK_NOPE]
    v = kv[..., QK_NOPE:]
    k_pe = torch.randn(S, 1, QK_ROPE, device=device, dtype=dtype)
    return k_nope, k_pe, v


def _reference(k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype):
    return torch_mla_kv_pack(k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype)


def test_pure_cast_strided_inputs(device: str) -> None:
    """k_nope/v are non-contiguous slices, scale=1.0 — the prefill call site."""
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    assert not k_nope.is_contiguous()
    assert not v.is_contiguous()

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert k_out.shape == (S, H, QK_NOPE + QK_ROPE)
    assert v_out.shape == (S, H, V_HEAD)
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_scaled_independent_k_v(device: str) -> None:
    """k and v use different scales; output reflects each independently."""
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    k_scale_inv, v_scale_inv = 0.5, 1.7

    k_ref, v_ref = _reference(
        k_nope, k_pe, v, k_scale_inv, v_scale_inv, torch.float8_e4m3fn
    )
    k_out, v_out = mla_kv_pack_quantize_fp8(
        k_nope, k_pe, v, k_scale_inv=k_scale_inv, v_scale_inv=v_scale_inv
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_k_pe_2d_and_3d_equivalent(device: str) -> None:
    """k_pe is accepted as both [s, 1, rope] and [s, rope]; same output."""
    k_nope, k_pe_3d, v = _make_kv_slice_inputs(device)
    k_pe_2d = k_pe_3d.squeeze(1)

    k_3d, v_3d = mla_kv_pack_quantize_fp8(k_nope, k_pe_3d, v)
    k_2d, v_2d = mla_kv_pack_quantize_fp8(k_nope, k_pe_2d, v)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_3d, k_2d)
    assert _bitwise_equal(v_3d, v_2d)


def test_contiguous_inputs(device: str) -> None:
    """Standalone (non-slice) k_nope, v inputs also work."""
    torch.manual_seed(1)
    k_nope = torch.randn(S, H, QK_NOPE, device=device, dtype=torch.bfloat16)
    v = torch.randn(S, H, V_HEAD, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(S, 1, QK_ROPE, device=device, dtype=torch.bfloat16)
    assert k_nope.is_contiguous()
    assert v.is_contiguous()

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_fp16_input(device: str) -> None:
    torch.manual_seed(2)
    kv = torch.randn(S, H, QK_NOPE + V_HEAD, device=device, dtype=torch.float16)
    k_nope = kv[..., :QK_NOPE]
    v = kv[..., QK_NOPE:]
    k_pe = torch.randn(S, 1, QK_ROPE, device=device, dtype=torch.float16)

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_e5m2_output(device: str) -> None:
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e5m2)
    k_out, v_out = mla_kv_pack_quantize_fp8(
        k_nope, k_pe, v, fp8_dtype=torch.float8_e5m2
    )
    torch.cuda.synchronize()

    assert k_out.dtype == torch.float8_e5m2
    assert v_out.dtype == torch.float8_e5m2
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_preallocated_outputs(device: str) -> None:
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    k_out = torch.empty(
        (S, H, QK_NOPE + QK_ROPE), dtype=torch.float8_e4m3fn, device=device
    )
    v_out = torch.empty((S, H, V_HEAD), dtype=torch.float8_e4m3fn, device=device)

    k_ret, v_ret = mla_kv_pack_quantize_fp8(k_nope, k_pe, v, k_out=k_out, v_out=v_out)
    torch.cuda.synchronize()

    assert k_ret.data_ptr() == k_out.data_ptr()
    assert v_ret.data_ptr() == v_out.data_ptr()

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


@pytest.mark.parametrize("k_scale_inv,v_scale_inv", [(1.0, 1.0), (0.5, 1.7)])
def test_pdl_off_matches_pdl_on(
    device: str, k_scale_inv: float, v_scale_inv: float
) -> None:
    """PDL is a scheduling hint; output must be bitwise-identical regardless."""
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper+ (SM≥90)")
    k_nope, k_pe, v = _make_kv_slice_inputs(device)

    k_off, v_off = mla_kv_pack_quantize_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        enable_pdl=False,
    )
    k_on, v_on = mla_kv_pack_quantize_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        enable_pdl=True,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(k_off, k_on)
    assert _bitwise_equal(v_off, v_on)


def test_kv_pack_reuses_compilation() -> None:
    # This contract belongs to the in-tree kernel; the pinned wheel can
    # precede it. Isolate the import from the wheel used by the other tests.
    repo_root = Path(__file__).resolve().parents[4]
    script = textwrap.dedent("""
        import torch
        from tokenspeed_mla.mla_kv_pack_quantize_fp8 import (
            _mla_kv_pack_quantize_fp8_kernel,
            mla_kv_pack_quantize_fp8,
        )
        from utils import assert_no_triton_compile

        torch.manual_seed(0)
        kv = torch.randn(256, 16, 256, device="cuda", dtype=torch.bfloat16)
        pe = torch.randn(256, 1, 64, device="cuda", dtype=torch.bfloat16)

        def run(tokens):
            # With 12 heads, row counts can be non-multiples of 16.
            nope, rope, v = kv[:tokens, :12, :128], pe[:tokens], kv[:tokens, :12, 128:]
            k_out, v_out = mla_kv_pack_quantize_fp8(nope, rope, v)
            k_ref = torch.cat((nope, rope.expand(-1, 12, -1)), dim=-1).to(k_out.dtype)
            v_ref = v.to(v_out.dtype)
            assert torch.equal(k_out.view(torch.uint8), k_ref.view(torch.uint8))
            assert torch.equal(v_out.view(torch.uint8), v_ref.view(torch.uint8))

        run(256)
        with assert_no_triton_compile(_mla_kv_pack_quantize_fp8_kernel):
            for tokens in (1, 3, 17, 255):
                run(tokens)
        """)
    env = os.environ.copy()
    python_path = [
        str(repo_root / "tokenspeed-mla" / "python"),
        str(repo_root / "tokenspeed-kernel" / "test"),
    ]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        "MLA KV packing compilation reuse failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_fp8_block_decode_batch_sizes_reuse_compiled_kernels() -> None:
    """Every batch size of one drafter decode shape reuses four compiled kernels."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("tokenspeed_mla decode requires SM100, SM103 or SM107")
    # The in-tree kernel owns this contract; the pinned wheel can precede it.
    repo_root = Path(__file__).resolve().parents[4]
    script = textwrap.dedent("""
        import torch
        from tokenspeed_mla import tokenspeed_mla_decode
        from utils import assert_no_cute_compile

        # A Kimi-K3.1 block drafter at TP8: 12 heads, a 7-token causal block.
        heads, q_len, page, max_seq_len = 12, 7, 64, 32776
        workspace = torch.empty(512 * 1024**2, dtype=torch.int8, device="cuda")
        kv = torch.randn(256, page, 576, device="cuda").to(torch.float8_e4m3fn)

        def run(batch):
            query = torch.randn(batch, q_len, heads, 576, device="cuda")
            tables = torch.randint(0, 256, (batch, 514), device="cuda").int()
            lengths = [q_len + (300 * b) % 8000 for b in range(batch)]
            out = tokenspeed_mla_decode(
                query=query.to(torch.float8_e4m3fn),
                kv_cache=kv,
                workspace_buffer=workspace,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=tables,
                seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
                max_seq_len=max_seq_len,
                softmax_scale=192**-0.5,
                causal_mask=True,
            )
            assert torch.isfinite(out).all()

        # One batch per reducer capacity (32, 16, and 8 splits), then no reducer.
        for batch in (1, 8, 16, 64):
            run(batch)
        with assert_no_cute_compile():
            for batch in range(1, 65):
                run(batch)
        """)
    env = os.environ.copy()
    python_path = [
        str(repo_root / "tokenspeed-mla" / "python"),
        str(repo_root / "tokenspeed-kernel" / "test"),
    ]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        "MLA decode recompiled across batch sizes:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_sm103_bf16_decode_causal_variants_compile() -> None:
    """The SM103 BF16 decode path compiles for both mask variants."""
    repo_root = Path(__file__).resolve().parents[4]
    mla_python = repo_root / "tokenspeed-mla" / "python"
    script = textwrap.dedent("""
        import torch
        import tokenspeed_mla.mla_decode as mla_decode

        # Cross-target compilation must not query occupancy from the host GPU.
        mla_decode.get_max_active_clusters = lambda _cluster_size: 1

        for causal_mask in (False, True):
            mla_decode._get_compiled_mla_kernel.cache_clear()
            compiled_kernel = mla_decode._get_compiled_mla_kernel(
                torch_dtype=torch.bfloat16,
                page_size=64,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                is_persistent=False,
                is_var_seq=True,
                is_var_split_kv=False,
                is_workspace_size_zero=False,
                fold_sq_factor=4,
                causal_mask=causal_mask,
                num_heads=16,
                seq_len_q=4,
                cp_world=1,
                use_pdl=False,
                return_lse=False,
                has_local_visible_lens=False,
                compute_capability=(10, 3),
                partial_fp16=False,
            )
            assert compiled_kernel is not None
        """)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_103a",
            "CUTE_DSL_DISABLE_FILE_CACHING": "1",
            "CUTE_DSL_NO_CACHE": "1",
        }
    )
    python_path = [str(mla_python)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        "SM103 BF16 decode compilation failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
