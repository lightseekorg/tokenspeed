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

"""Single-GPU reproducer for the SM100 flashinfer `trtllm_bf16_moe`
illegal-memory-access seen in the qwen3.5-397b-a17b-nvfp4 agentic perf job.

The failing GEMM reported by the runner is

    bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_s6_et128x8_m128x8x16
    _c1x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_rgTma_clmp_dynB_sm100f
    numBatches=512 GemmMNK=1 4096 128

which is the second per-expert GEMM inside
`flashinfer.fused_moe.core.trtllm_bf16_moe` with the Qwen3.5-397B-A17B
shape at attn-tp=moe-tp=8: num_experts=512, hidden_size=4096,
moe_intermediate_size_per_rank = 1024 / 8 = 128. The kernel runs fine
in eager but blows up the first time it is captured inside a
`torch.cuda.CUDAGraph`, which is exactly the path the runtime hits the
first time CudaGraphWrapper captures the MTP/NextN drafter at bs=1.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not platform.is_blackwell,
    reason="flashinfer trtllm_bf16_moe is only registered for SM100 Blackwell.",
)


# Qwen3.5-397B-A17B-NVFP4 MoE shape under attn-tp=moe-tp=8.
NUM_EXPERTS = 512
TOP_K = 10
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE_PER_RANK = 128  # moe_intermediate_size (1024) // tp_size (8)
BLOCK_K = 128


def _build_inputs(
    *,
    bs: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> dict:
    """Build the call-site arguments for `flashinfer.trtllm_bf16_moe`.

    Weight layout follows `WeightLayout.BlockMajorK`. The flashinfer
    launcher reads per-expert weights as `[E, K / block_k, Mn, block_k]`
    (see `check_weights_shape` in `trtllm_fused_moe_kernel_launcher.cu`),
    so for a gated swiglu activation with K = hidden_size and
    Mn = 2 * intermediate_size on gemm1, and K = intermediate_size,
    Mn = hidden_size on gemm2:
      gemm1_weights: [E, H // BLOCK_K, 2 * I, BLOCK_K]
      gemm2_weights: [E, I // BLOCK_K, H, BLOCK_K]
    where I = INTERMEDIATE_SIZE_PER_RANK, H = HIDDEN_SIZE, E = NUM_EXPERTS.
    """
    assert HIDDEN_SIZE % BLOCK_K == 0
    assert INTERMEDIATE_SIZE_PER_RANK % BLOCK_K == 0

    g = generator
    hidden_states = torch.randn(
        bs, HIDDEN_SIZE, device=device, dtype=dtype, generator=g
    )
    routing_logits = torch.randn(
        bs, NUM_EXPERTS, device=device, dtype=dtype, generator=g
    )
    gemm1_weights = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE // BLOCK_K,
        2 * INTERMEDIATE_SIZE_PER_RANK,
        BLOCK_K,
        device=device,
        dtype=dtype,
        generator=g,
    )
    gemm2_weights = torch.randn(
        NUM_EXPERTS,
        INTERMEDIATE_SIZE_PER_RANK // BLOCK_K,
        HIDDEN_SIZE,
        BLOCK_K,
        device=device,
        dtype=dtype,
        generator=g,
    )
    # Scale weights down so the gemm2 output stays in range, since
    # otherwise inf/nan accumulators can mask the IMA we want to catch.
    gemm1_weights.mul_(1.0 / (HIDDEN_SIZE**0.5))
    gemm2_weights.mul_(1.0 / (INTERMEDIATE_SIZE_PER_RANK**0.5))

    return dict(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE_SIZE_PER_RANK,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        # 1 = Renormalize (TopK -> Softmax), what
        # flashinfer_trtllm_unquant_moe_apply picks for Qwen3.5 by default.
        routing_method_type=1,
    )


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_trtllm_bf16_moe_eager(device: str, bs: int) -> None:
    """Sanity check: the failing kernel returns finite output in eager.

    The runtime hits the IMA only under cuda-graph capture; running the
    same shape eagerly is expected to succeed and acts as a positive
    control next to `test_trtllm_bf16_moe_cuda_graph` below.
    """
    from flashinfer.fused_moe.core import trtllm_bf16_moe

    torch.manual_seed(0)
    inputs = _build_inputs(bs=bs, device=torch.device(device))
    # Match the wrapper's behaviour: tune for the actual batch size.
    out = trtllm_bf16_moe(**inputs, tune_max_num_tokens=max(1, bs))
    if isinstance(out, (list, tuple)):
        out = out[0]
    torch.cuda.synchronize()
    assert out.shape == (bs, HIDDEN_SIZE)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "trtllm_bf16_moe produced non-finite output"


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_trtllm_bf16_moe_cuda_graph(device: str, bs: int) -> None:
    """Capture the failing kernel inside a CUDAGraph and replay it.

    This is the path that reproduces the qwen3.5-397b-a17b-nvfp4 agentic
    perf IMA: the first time CudaGraphWrapper captures the MTP/NextN
    drafter at bs=1, flashinfer's gemm2 launch raises
    `Error in function 'run' at trtllm_batched_gemm_runner.cu:278`. We
    want CI to fail loudly here so a fix can be developed without
    spinning up the full 8-GPU server.
    """
    from flashinfer.fused_moe.core import trtllm_bf16_moe

    torch.manual_seed(0)
    inputs = _build_inputs(bs=bs, device=torch.device(device))

    # Warm up on a side stream, then capture on the same stream as the
    # runtime's CudaGraphWrapper does.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        warm = trtllm_bf16_moe(**inputs, tune_max_num_tokens=max(1, bs))
        if isinstance(warm, (list, tuple)):
            warm = warm[0]
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    captured_out: list[torch.Tensor] = []
    with torch.cuda.graph(g):
        out = trtllm_bf16_moe(**inputs, tune_max_num_tokens=max(1, bs))
        if isinstance(out, (list, tuple)):
            out = out[0]
        captured_out.append(out)

    g.replay()
    torch.cuda.synchronize()

    captured = captured_out[0]
    assert captured.shape == (bs, HIDDEN_SIZE)
    assert captured.dtype == torch.bfloat16
    assert torch.isfinite(
        captured
    ).all(), "trtllm_bf16_moe produced non-finite output under cuda-graph replay"
