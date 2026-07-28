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

"""SiTU sidecar deferred-finalize (do_finalize=False) memory-safety contract.

The raw-FFI deferred branch must confine all device writes to the tensors it
allocates. tokenspeed_situ 0.1.0.post20260726 violates this: the routing
workspace (permuted->expanded map, histograms, CTA tables; tens of KB) is
carved starting at ``align512(end of expanded_idx_to_permuted_idx)`` but the
allocation request covers only the ``[num_tokens * top_k]`` int32 output, so
every deferred call writes far past its block. Under multi-size CUDA-graph
capture this bakes out-of-bounds writes at fixed addresses; replay then
corrupts neighboring allocations, or raises an illegal memory access when
the block lands near the end of a caching-allocator segment (the serving
crash this test was distilled from).

The test plants byte sentinels beyond the returned ``expanded_idx`` block
and fails if the sidecar dirties them. It is marked ``xfail`` for sidecar
versions known to carry the bug, so it turns green (XPASS -> pass) exactly
when a fixed wheel is installed and acts as the acceptance gate for
re-enabling the deferred-finalize tail (reverted in d3d7d6b).
"""

from __future__ import annotations

import ctypes

import pytest
import torch

E, TOPK, HID, ISPP = 896, 16, 3584, 384

# Sidecar versions with the confirmed deferred-branch workspace overrun.
KNOWN_BAD_SIDECAR_VERSIONS = {"0.1.0.post20260726"}


def _sidecar_version() -> str | None:
    try:
        from tokenspeed_situ.loader import bundle_info

        return str(bundle_info()["package_version"])
    except Exception:  # noqa: BLE001 - absent/broken sidecar handled by skip
        return None


def _deferred_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        from tokenspeed_situ.op import (  # noqa: F401
            trtllm_fp4_block_scale_moe_raw,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _deferred_available(),
    reason="requires the SiTU sidecar raw FFI on sm103",
)

_cudart = ctypes.CDLL("libcudart.so")


def _d2h(ptr: int, nbytes: int) -> bytes:
    buf = (ctypes.c_uint8 * nbytes)()
    rc = _cudart.cudaMemcpy(
        ctypes.byref(buf), ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), 2
    )
    assert rc == 0, f"cudaMemcpy D2H failed rc={rc}"
    return bytes(buf)


def _h2d(ptr: int, data: bytes) -> None:
    src = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    rc = _cudart.cudaMemcpy(
        ctypes.c_void_p(ptr), ctypes.byref(src), ctypes.c_size_t(len(data)), 1
    )
    assert rc == 0, f"cudaMemcpy H2D failed rc={rc}"


@pytest.fixture(scope="module")
def situ_weights():
    """Synthetic expert weights in the sidecar's final layout (scale=1.0)."""
    torch.manual_seed(0)
    dev = "cuda"
    w = {}
    w["w13"] = torch.randint(
        0, 256, (E, 2 * ISPP, HID // 2), dtype=torch.uint8, device=dev
    )
    w["w13_scale"] = torch.full(
        (E, 2 * ISPP, HID // 32), 127, dtype=torch.uint8, device=dev
    ).view(torch.float8_e4m3fn)
    w["w2"] = torch.randint(0, 256, (E, HID, ISPP // 2), dtype=torch.uint8, device=dev)
    w["w2_scale"] = torch.full(
        (E, HID, ISPP // 32), 127, dtype=torch.uint8, device=dev
    ).view(torch.float8_e4m3fn)
    w["alpha"] = torch.ones(E, dtype=torch.float32, device=dev)
    w["beta"] = torch.ones(E, dtype=torch.float32, device=dev)
    return w


def _deferred_raw(w, ids, weights, x):
    from tokenspeed_situ.op import (
        ActivationType,
        RoutingInputMode,
        trtllm_fp4_block_scale_moe_raw,
    )

    output = torch.empty(
        x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device
    )
    result = trtllm_fp4_block_scale_moe_raw(
        RoutingInputMode.UNPACKED_PRECOMPUTED,
        None,
        ids,
        weights,
        None,
        x,
        None,
        w["w13"],
        w["w13_scale"],
        None,
        w["alpha"],
        w["beta"],
        None,
        w["w2"],
        w["w2_scale"],
        None,
        None,
        None,
        None,
        None,
        E,
        TOPK,
        None,
        None,
        ISPP,
        0,
        E,
        None,
        0,
        False,  # do_finalize
        False,  # enable_pdl
        ActivationType.SITU,
        output,
        [-1, -1],
        True,
        None,
    )
    as_torch = lambda v: v if isinstance(v, torch.Tensor) else torch.from_dlpack(v)
    return as_torch(result[0]), as_torch(result[2])


def _routing(t: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(t)])
    weights = torch.softmax(torch.randn(t, TOPK, generator=g), dim=-1)
    return ids.to(torch.int32).cuda(), weights.to(torch.bfloat16).cuda()


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_deferred_writes_stay_inside_returned_buffers(situ_weights, num_tokens):
    if _sidecar_version() in KNOWN_BAD_SIDECAR_VERSIONS:
        pytest.xfail(
            "tokenspeed_situ "
            f"{_sidecar_version()} carves the deferred routing workspace "
            "past the expanded_idx allocation (confirmed OOB write; serving "
            "IMA under multi-size graph capture). Install a fixed sidecar "
            "wheel to turn this contract test green."
        )
    ids, weights = _routing(num_tokens, 42)
    x = torch.randn(num_tokens, HID, dtype=torch.bfloat16, device="cuda") * 0.1

    # Steady-state call so the caching allocator settles into a fixed block
    # for the returned index tensor, then sentinel far past its end.
    g2, idx = _deferred_raw(situ_weights, ids, weights, x)
    torch.cuda.synchronize()
    idx_end = idx.data_ptr() + idx.numel() * idx.element_size()
    g2_end = g2.data_ptr() + g2.numel() * g2.element_size()
    del g2, idx

    tail = 1 << 17  # cover the full observed carve-out span (~64KB+)
    sentinel = bytes([0xAB]) * tail
    _h2d(idx_end, sentinel)
    _h2d(g2_end, sentinel[: 1 << 12])
    torch.cuda.synchronize()

    g2b, idxb = _deferred_raw(situ_weights, ids, weights, x)
    torch.cuda.synchronize()
    assert idxb.data_ptr() + idxb.numel() * 4 == idx_end, (
        "allocator did not reuse the probed block; rerun (layout-dependent)"
    )

    tail_bytes = _d2h(idx_end, tail)
    dirty = sum(1 for b in tail_bytes if b != 0xAB)
    gemm2_tail = _d2h(g2_end, 1 << 12)
    gemm2_dirty = sum(1 for b in gemm2_tail if b != 0xAB)
    assert gemm2_dirty == 0, f"gemm2_output overran by {gemm2_dirty} bytes"
    assert dirty == 0, (
        f"sidecar wrote {dirty} bytes beyond the expanded_idx allocation "
        f"(T={num_tokens}); the deferred-finalize path must not be enabled "
        "with this sidecar build"
    )
