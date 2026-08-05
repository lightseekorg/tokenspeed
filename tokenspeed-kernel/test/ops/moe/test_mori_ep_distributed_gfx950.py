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
"""Multi-GPU (gfx950) correctness for the MORI EP dispatch/combine path.

Real all-to-all across 2 or 4 MI350X ranks + the ``mori``/``flydsl`` packages, so a
plain one-GPU ``pytest`` run skips it. Exercise it with, e.g.:

    torchrun --standalone --nproc-per-node=2 -m pytest -q \
        tokenspeed-kernel/test/ops/moe/test_mori_ep_distributed_gfx950.py

Each rank dispatches its OWN tokens to the expert-owning ranks, runs the local expert
GEMM (bf16 masked grouped-GEMM, or the native MXFP4 grouped-GEMM), combines back, and
the per-rank result is compared (cosine) to a replicated full-MoE reference computed
locally over all experts. This is the direct analog of the ``none``-backend distributed
reference test, but through MORI's real dispatch/combine instead of masked-replicate.
"""

from __future__ import annotations

import os

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


if not _is_gfx950():
    pytest.skip("MORI EP is gfx950 (CDNA4) only", allow_module_level=True)

import tokenspeed_kernel  # noqa: E402
import torch.distributed as dist  # noqa: E402
from kimi3_reference import dequantize_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.communication.mori_ep import (  # noqa: E402
    get_dispatcher,
    mori_available,
)
from tokenspeed_kernel.ops.moe.mori.bf16 import masked_grouped_gemm  # noqa: E402
from tokenspeed_kernel.ops.moe.mori.mxfp4 import (  # noqa: E402
    _grouped_mxfp4_gemm_3d,
    _mori_mxfp4_weight_preprocessor,
)

# Fixed local experts per rank -> total experts scale linearly with world size, so
# the test is world-size-agnostic (works for any world >= 2, not just powers of two).
E_LOCAL, H, I, K, T, CAP = 4, 1024, 512, 4, 32, 128


def _reference_moe(x, topk_ids, topk_w, w13, w2):
    """Replicated full MoE over ALL experts (bf16 weights), origin-token order."""
    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in range(x.shape[0]):
        for slot in range(K):
            e = int(topk_ids[t, slot].item())
            gu = xf[t] @ w13[e].float().t()
            inter = torch.nn.functional.silu(gu[:I]) * gu[I:]
            out[t] += topk_w[t, slot].float() * (inter @ w2[e].float().t())
    return out


def _run(quant: str) -> None:
    world = _world_size()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    e_local = E_LOCAL  # experts owned by this rank
    e_total = e_local * world  # global expert count, scales linearly with world size

    gen = torch.Generator(device="cuda").manual_seed(1234)
    w13 = (
        torch.randn(
            e_total, 2 * I, H, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        * 0.05
    )
    w2 = (
        torch.randn(e_total, H, I, generator=gen, device="cuda", dtype=torch.bfloat16)
        * 0.05
    )

    lo, hi = rank * e_local, (rank + 1) * e_local
    if quant == "bf16":
        w13_local = w13[lo:hi].contiguous()
        w2_local = w2[lo:hi].contiguous()
        ref_w13, ref_w2 = w13, w2
    else:  # mxfp4: quantize per expert, build swizzled local weight module + dequant reference

        def q(w2d):
            return tokenspeed_kernel.quantize_mxfp4(
                w2d.contiguous(), scale_size=32, scale_layout="linear"
            )

        w13_p = torch.stack([q(w13[e])[0] for e in range(e_total)])
        w13_s = torch.stack([q(w13[e])[1] for e in range(e_total)])
        w2_p = torch.stack([q(w2[e])[0] for e in range(e_total)])
        w2_s = torch.stack([q(w2[e])[1] for e in range(e_total)])
        # reference uses the *dequantized* weights (what the kernel effectively multiplies)
        ref_w13 = dequantize_mxfp4(w13_p, w13_s)
        ref_w2 = dequantize_mxfp4(w2_p, w2_s)
        wmod = torch.nn.Module()
        wmod.w13_weight, wmod.w13_weight_scale = (
            w13_p[lo:hi].contiguous(),
            w13_s[lo:hi].contiguous(),
        )
        wmod.w2_weight, wmod.w2_weight_scale = (
            w2_p[lo:hi].contiguous(),
            w2_s[lo:hi].contiguous(),
        )
        wmod.top_k = K
        _mori_mxfp4_weight_preprocessor({}, wmod)

    torch.manual_seed(100 + rank)
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1
    topk_ids = torch.stack(
        [torch.randperm(e_total, device="cuda")[:K] for _ in range(T)]
    ).to(torch.int32)
    raw = torch.rand(T, K, device="cuda")
    topk_w = (raw / raw.sum(-1, keepdim=True)).float()

    disp = get_dispatcher(
        rank=rank,
        world_size=world,
        hidden_dim=H,
        num_local_experts=e_local,
        num_experts_per_token=K,
        max_num_inp_token_per_rank=CAP,
        data_type=torch.bfloat16,
    )
    handle = disp.dispatch(x, topk_w, topk_ids)
    packed = handle["packed_x"]
    if quant == "bf16":
        # masked_grouped_gemm returns a fresh tensor -> copy into the buffer
        packed.copy_(masked_grouped_gemm(packed, handle["counts"], w13_local, w2_local))
    else:
        # native mxfp4 bridge writes into packed_x IN PLACE (returns None); bound rows by the
        # global-token upper bound (world * tokens/rank * top_k), as the registered kernel does.
        n_recv_bound = world * T * K
        _grouped_mxfp4_gemm_3d(packed, handle["counts"], wmod, n_recv_bound)
    out = disp.combine(handle)[:T].float()

    ref = _reference_moe(x, topk_ids, topk_w, ref_w13, ref_w2)
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0
    ).item()
    # The mxfp4 path is W4A4 on the gfx950 Gluon kernels: it quantizes weights AND activations
    # (and the intermediate) to MXFP4, while the reference dequantizes weights to bf16 and keeps
    # activations in bf16 -- so ~0.97 cos is expected-correct for mxfp4 (end-to-end accuracy is
    # gated by the aime25 eval). bf16 has no such quantization and stays tight.
    threshold = 0.98 if quant == "bf16" else 0.97
    assert cos > threshold, f"{quant} rank{rank}: cos={cos}"


@pytest.mark.skipif(
    _world_size() < 2,
    reason="expert parallel needs >=2 ranks; launch with torchrun (any world size >= 2)",
)
@pytest.mark.skipif(not mori_available(), reason="requires the mori + flydsl packages")
def test_mori_ep_bf16_matches_replicated_reference() -> None:
    _run("bf16")


@pytest.mark.skipif(
    _world_size() < 2,
    reason="expert parallel needs >=2 ranks; launch with torchrun (any world size >= 2)",
)
@pytest.mark.skipif(not mori_available(), reason="requires the mori + flydsl packages")
def test_mori_ep_mxfp4_matches_replicated_reference() -> None:
    _run("mxfp4")
