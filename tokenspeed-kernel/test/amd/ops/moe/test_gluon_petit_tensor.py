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

import importlib

import pytest
import torch
import triton.experimental.gluon as g
from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel

petit_kernel = load_petit_kernel()

from lib.gemm.rocm.intrinsics import (
    BufferResource,
    amdgcn_cvt_scalef32_pk_fp4_f32,
    amdgcn_s_waitcnt_barrier,
)
from lib.moe.rocm.mem.input_mxfp4_packed import PackedInputState
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.tal.tensor_ops import (
    atomic_add,
    atomic_or,
    group_gather,
    group_max,
    load_words,
    store_words,
    transpose_packed_bytes,
    wave_inclusive_sum,
)
from triton.experimental.gluon import language as l
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires GFX950",
)


@g.jit
def _buffer_resource_copy(source, copied, width: l.constexpr, masked: l.constexpr):
    lane = l.arange(0, 256, layout=l.BlockedLayout([1], [64], [4], [0]))
    source_resource = MakeBufferResource(source, (256 * width + 4) * 4)
    target_resource = MakeBufferResource(copied, (256 * width + 4) * 4)
    offset = lane * width * 4
    if width == 2:
        value = BufferResource.LoadU64(source_resource, offset, 16, 0)
        BufferResource.StoreU64(target_resource, offset, 16, value, 0)
    else:
        value = BufferResource.LoadU32(source_resource, offset, 16, 0)
        if masked:
            BufferResource.StoreU32(
                target_resource, offset, 16, value, 0, predicate=lane % 3 != 0
            )
        else:
            BufferResource.StoreU32(
                target_resource, offset, 16, value, 0, predicate=True
            )


@pytest.mark.parametrize("width,masked", ((1, False), (1, True), (2, False)))
def test_buffer_resource_copy(width, masked):
    source = torch.randint(
        -(1 << 31), 1 << 31, (256 * width + 8,), device="cuda", dtype=torch.int32
    )
    copied = torch.full_like(source, -7)
    _buffer_resource_copy[(1,)](source, copied, width, masked)
    expected = torch.full_like(source, -7)
    expected[4:-4] = source[4:-4]
    if masked:
        expected[4:-4:3] = -7
    torch.testing.assert_close(copied, expected, atol=0, rtol=0)


@g.jit
def _masked_memory(source, copied, summed, bits, count: l.constexpr):
    lane = l.arange(0, 256, layout=l.BlockedLayout([1], [64], [4], [0]))
    mask = (lane < count) & (lane % 3 != 0)
    value = load_words(MakeBufferResource(source, 1024), lane * 4, 0, 1, 0, mask)
    store_words(MakeBufferResource(copied, 1024), lane * 4, 0, value, 1, 0, mask)
    atomic_add(MakeBufferResource(summed, 4), 0, 0, 1, 0, mask)
    atomic_or(MakeBufferResource(bits, 4), 0, 0, 1 << (lane % 16), 0, mask)


@pytest.mark.parametrize("count", (0, 1, 31, 64, 65, 255, 256))
def test_masked_memory(count):
    source = torch.arange(256, device="cuda", dtype=torch.int32)
    copied = torch.full_like(source, -7)
    summed = torch.zeros(1, device="cuda", dtype=torch.int32)
    bits = torch.zeros_like(summed)
    _masked_memory[(1,)](source, copied, summed, bits, count)
    mask = (source < count) & (source % 3 != 0)
    torch.testing.assert_close(copied, torch.where(mask, source, -7), atol=0, rtol=0)
    assert summed.item() == mask.sum().item()
    expected = 0
    for lane in range(count):
        if lane % 3:
            expected |= 1 << (lane % 16)
    assert bits.item() == expected


@g.jit
def _tensor_groups(
    source,
    maxima,
    gathered,
    scanned,
    transposed,
    reduced,
    count: l.constexpr,
    width: l.constexpr,
):
    index = l.arange(0, count, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0]))
    value = l.load(source + index)
    l.store(gathered + index, group_gather(value, value % width, width))
    l.store(scanned + index, wave_inclusive_sum(value))
    l.store(transposed + index, transpose_packed_bytes(value.to(l.uint32)))
    l.store(reduced + index, group_max(l.load(maxima + index), 8))


@pytest.mark.parametrize("count,num_warps", ((64, 1), (64, 4), (256, 4), (512, 8)))
@pytest.mark.parametrize("width", (4, 32, 64))
def test_tensor_groups_preserve_wave_boundaries(count, num_warps, width):
    source = torch.randint(0, 1 << 30, (count,), device="cuda", dtype=torch.int32)
    maxima = torch.randn(count, device="cuda")
    maxima[5] = float("nan")
    gathered, scanned, transposed = [torch.empty_like(source) for _ in range(3)]
    reduced = torch.empty_like(maxima)
    kernel = _tensor_groups[(1,)](
        source,
        maxima,
        gathered,
        scanned,
        transposed,
        reduced,
        count,
        width,
        num_warps=num_warps,
    )
    groups = source.reshape(-1, width)
    expected_gather = groups.gather(1, (groups % width).long()).flatten()
    expected_scan = source.reshape(-1, 64).cumsum(1).flatten().to(torch.int32)
    words = source.long().reshape(-1, 4)
    byte_shifts = torch.arange(4, device="cuda") * 8
    byte_matrix = (words[:, :, None] >> byte_shifts) & 0xFF
    expected_transpose = (byte_matrix << byte_shifts[None, :, None]).sum(1)
    expected_max = maxima.reshape(-1, 8).amax(1).repeat_interleave(8)
    torch.testing.assert_close(gathered, expected_gather, atol=0, rtol=0)
    torch.testing.assert_close(scanned, expected_scan, atol=0, rtol=0)
    torch.testing.assert_close(
        transposed, expected_transpose.flatten().to(torch.int32), atol=0, rtol=0
    )
    torch.testing.assert_close(reduced, expected_max, atol=0, rtol=0, equal_nan=True)
    # These operations must not introduce communication between independent waves.
    assert kernel.metadata.shared == 0
    assert "s_barrier" not in kernel.asm["amdgcn"]


@g.jit
def _round_fp4(source, out):
    lane = l.arange(0, 64, layout=l.BlockedLayout([1], [64], [1], [0]))
    value = l.load(source + lane)
    packed = amdgcn_cvt_scalef32_pk_fp4_f32(0, value, value, 1.0, 0)
    l.store(out + lane, packed & 15)


def test_fp4_ties_and_signed_zero():
    values = [
        0.0,
        -0.0,
        0.25,
        0.75,
        1.25,
        1.75,
        2.5,
        3.5,
        5.0,
        8.0,
        -0.25,
        -0.75,
        -5.0,
        -8.0,
        1.0,
        -1.0,
    ]
    codes = [0, 8, 0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 14, 15, 2, 10]
    source = torch.tensor(values * 4, device="cuda")
    out = torch.empty(64, device="cuda", dtype=torch.int32)
    _round_fp4[(1,)](source, out, num_warps=1)
    torch.testing.assert_close(
        out, torch.tensor(codes * 4, device="cuda", dtype=torch.int32), atol=0, rtol=0
    )


@g.jit
def _scale_load(source, out, Input: l.constexpr):
    THREADS: l.constexpr = Input.kThreads
    WORDS: l.constexpr = Input.kShmStageWords * 2
    tid = l.arange(0, THREADS, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0]))
    storage = l.allocate_shared_memory(
        l.uint32, [Input.kShmStageWords * 2], l.SwizzledSharedLayout(1, 1, 1, [0])
    )
    shm = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    # Guard the activation region and unused scale lanes against stray writes.
    for start in l.static_range(0, WORDS, THREADS):
        index = start + tid
        l.store(shm + index, 0xDEADBEEF, index < Input.kShmStageWords * 2)
    l.barrier()
    resource = MakeBufferResource(source, Input.kGroupM * Input.kRowStride)
    state = PackedInputState(resource, 0, Input.kGroupM, 0, 0)
    Input.LoadScalesAsync(state, shm, tid // 64, tid % 64)
    amdgcn_s_waitcnt_barrier(0)
    for start in l.static_range(0, WORDS, THREADS):
        index = start + tid
        value = l.load(shm + index, index < Input.kShmStageWords * 2, other=0)
        l.store(out + index, value, index < Input.kShmStageWords * 2)
    storage._keep_alive()


@pytest.mark.parametrize("num_tokens", (1, 256, 1024))
def test_partial_scale_load_preserves_lds(num_tokens):
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=128,
        topk=4,
        model_dim=2880,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=petit_kernel.MegaMoeActivationFunction.swiglu,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=True,
    )
    mega = importlib.import_module("lib.moe.rocm.mega_moe")
    stage1, _, _ = mega._MegaMoESolutions()[
        config._solution_id_for_tokens(num_tokens)
    ].Kernels(True, num_tokens >= 256, num_tokens >= 1024)
    policy = stage1.Input
    source = torch.arange(
        policy.kGroupM * policy.kRowStride // 4, dtype=torch.int32, device="cuda"
    )
    out = torch.empty(policy.kShmStageWords * 2, dtype=torch.int32, device="cuda")
    _scale_load[(1,)](source, out, policy, num_warps=policy.kNumWarps)
    expected = torch.full_like(out, -559038737)
    scales = source.reshape(policy.kGroupM, policy.kRowStride // 4)[
        :, policy.kValueBytes // 4 :
    ].flatten()
    for stage in range(2):
        begin = stage * policy.kShmStageWords + policy.kShmActWords
        expected[begin : begin + policy.kScaleWordsPerStage] = scales[
            stage
            * policy.kScaleWordsPerStage : (stage + 1)
            * policy.kScaleWordsPerStage
        ]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("groups", (1, 90, 224))
def test_tensor_quantization_padding_and_zero_groups(groups):
    mega = importlib.import_module("lib.moe.rocm.mega_moe")
    values = [
        0.0,
        -0.0,
        0.25,
        0.75,
        1.25,
        1.75,
        2.5,
        3.5,
        5.0,
        6.0,
        -0.25,
        -0.75,
        -5.0,
        -6.0,
        1.0,
        -1.0,
    ]
    codes = [0, 8, 0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 14, 15, 2, 10]
    # Include a strided input row and a final partial wave of quantization groups.
    source = torch.full(
        (3, groups * 32 + 32), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    source[:, : groups * 32] = torch.tensor(values * (groups * 2), device="cuda")
    source[0, : groups * 32] = 0
    scale_stride = (groups + 15) // 16 * 16
    out = torch.full(
        (3, groups * 16 + scale_stride), 255, device="cuda", dtype=torch.uint8
    )
    mega.MegaMoEQuantizeMxFp4Kernel[((groups + 63) // 64, 3)](
        source,
        out,
        groups,
        source.stride(0),
        num_warps=1,
    )
    expected = torch.zeros_like(out)
    packed = [codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2)]
    expected[1:, : groups * 16] = torch.tensor(
        packed * (groups * 2), device="cuda", dtype=torch.uint8
    )
    expected[1:, groups * 16 : groups * 16 + groups] = 127
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("num_experts", "topk", "model_dim", "activation_function", "has_bias"),
    (
        (
            128,
            4,
            2880,
            petit_kernel.MegaMoeActivationFunction.swiglu,
            True,
        ),
        (
            384,
            6,
            7168,
            petit_kernel.MegaMoeActivationFunction.silu,
            False,
        ),
        (
            896,
            16,
            3584,
            petit_kernel.MegaMoeActivationFunction.kimi_situ,
            False,
        ),
    ),
)
@pytest.mark.parametrize("num_tokens", (1, 256, 1024))
def test_token_counts_do_not_recompile(
    num_tokens: int,
    num_experts: int,
    topk: int,
    model_dim: int,
    activation_function: petit_kernel.MegaMoeActivationFunction,
    has_bias: bool,
) -> None:
    # New token counts must not trigger JIT compilation during serving.
    mega_moe = importlib.import_module("lib.moe.rocm.mega_moe")
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=num_experts,
        topk=topk,
        model_dim=model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=activation_function,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=has_bias,
    )
    adapter = mega_moe._MegaMoESolutions()[config._solution_id_for_tokens(num_tokens)]
    stage1, _, combine = adapter.Kernels(
        True,
        num_tokens >= 256,
        num_tokens >= (1024 if num_experts == 128 else 256),
    )
    device = torch.device("cuda", 0)
    uint8 = torch.empty(1, dtype=torch.uint8, device=device)
    int32 = torch.empty(1, dtype=torch.int32, device=device)
    float32 = torch.empty(1, dtype=torch.float32, device=device)
    bfloat16 = torch.empty(1, dtype=torch.bfloat16, device=device)
    bias = bfloat16 if has_bias else None

    # Warm each runtime integer specialization class before varying row counts.
    def warm_rows(rows):
        mega_moe.MegaMoEStage1.warmup(
            uint8,
            uint8,
            rows,
            bias,
            uint8,
            0,
            uint8,
            int32,
            float32,
            stage1,
            grid=(stage1.kNumSMs,),
            num_warps=stage1.kNumWarps,
            enable_fp_fusion=False,
        )
        mega_moe.MegaMoECombine.warmup(
            bfloat16,
            rows,
            config.compute_model_dim,
            uint8,
            0,
            combine,
            grid=(combine.kNumSMs,),
            num_warps=combine.kNumWarps,
            enable_fp_fusion=False,
        )

    for rows in (0, 1, 2, 16):
        warm_rows(rows)
    with assert_no_triton_compile(mega_moe.MegaMoEStage1), assert_no_triton_compile(
        mega_moe.MegaMoECombine
    ):
        for rows in (3, 17, 32, 63, 127, 256, 1024):
            warm_rows(rows)
