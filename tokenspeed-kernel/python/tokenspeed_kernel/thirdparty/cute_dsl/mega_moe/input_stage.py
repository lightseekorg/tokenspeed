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

"""Quantize BF16 rows and stage routes directly into the MegaMoE workspace."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch


@cute.jit
def _aligned(tensor: cute.Tensor, alignment: cutlass.Constexpr):
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(
            pointer.dtype,
            pointer.toint(),
            pointer.memspace,
            assumed_align=alignment,
        ),
        tensor.layout,
    )


class _Nvfp4InputStageKernel:
    def __init__(self, hidden: int, top_k: int):
        self.hidden = hidden
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        input_scale: cute.Tensor,
        ids: cute.Tensor,
        scores: cute.Tensor,
        q_out: cute.Tensor,
        sf_out: cute.Tensor,
        ids_out: cute.Tensor,
        scores_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.stage(
            x, input_scale, ids, scores, q_out, sf_out, ids_out, scores_out
        ).launch(grid=(q_out.shape[0], 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def stage(
        self,
        x: cute.Tensor,
        input_scale: cute.Tensor,
        ids: cute.Tensor,
        scores: cute.Tensor,
        q_out: cute.Tensor,
        sf_out: cute.Tensor,
        ids_out: cute.Tensor,
        scores_out: cute.Tensor,
    ):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        live = row < x.shape[0]
        if live:
            source = cute.zipped_divide(x[row, None], (16,))
            target = cute.zipped_divide(q_out[row, None], (16,))
            load = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN, num_bits_per_copy=64
            )
            norm = cutlass.Float32(input_scale[0])
            decode_global = cute.arch.rcp_approx(norm)
            reciprocal_six = cute.arch.rcp_approx(cutlass.Float32(6.0))
            for block in cutlass.range(tid, self.hidden // 16, 128, unroll=1):
                values_bf16 = cute.make_rmem_tensor((16,), cutlass.BFloat16)
                cute.copy(load, _aligned(source[(None,), (block,)], 16), values_bf16)
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                values.store(values_bf16.load().to(cutlass.Float32))
                amax = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(16):
                    amax = cute.arch.fmax(amax, cute.arch.fmax(values[i], -values[i]))
                scale = (norm * (amax * reciprocal_six)).to(cutlass.Float8E4M3FN)
                # Match FlashInfer CUDA's encode scale. norm * rcp(scale)
                # differs at FP4 ties from rcp(scale * rcp(norm)).
                encode = cutlass.Float32(0.0)
                if amax != cutlass.Float32(0.0):
                    encode = cute.arch.rcp_approx(
                        cutlass.Float32(scale) * decode_global
                    )
                packed = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
                scaled = values.load() * cute.full_like(values, encode, cutlass.Float32)
                packed.store(scaled.to(cutlass.Float4E2M1FN))
                cute.copy(store, packed, _aligned(target[(None,), (block,)], 8))
                sf_out[row, block] = scale

        # Mask the entire capacity tail in the same launch. Counts are runtime
        # tensor extents, so eager and graphs use the same padding contract.
        if tid < self.top_k:
            expert = cutlass.Int64(-1)
            score = cutlass.Float32(0.0)
            if live:
                expert = cutlass.Int64(ids[row, tid])
                score = cutlass.Float32(scores[row, tid])
            ids_out[row, tid] = expert
            scores_out[row, tid] = score


def _view(tensor: torch.Tensor, alignment: int):
    result = cutlass_torch.from_dlpack(tensor, assumed_align=alignment)
    return result.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


class Nvfp4InputStage:
    def __init__(self, hidden: int, top_k: int):
        if hidden % 64 != 0 or not 0 < top_k <= 128:
            raise ValueError(
                "NVFP4 input staging requires hidden % 64 == 0 and top_k <= 128"
            )
        self.kernel = _Nvfp4InputStageKernel(hidden, top_k)
        self.compiled: dict[tuple, object] = {}

    def run(self, x, input_scale, ids, scores, q_out, sf_out, ids_out, scores_out):
        """Write live quantized rows and masked tail routes into caller-owned buffers."""
        key = (x.dtype, input_scale.dtype, ids.dtype, scores.dtype)
        arguments = (
            _view(x, 16),
            _view(input_scale, 4),
            _view(ids, 4),
            _view(scores, 4),
            _view(q_out.view(torch.float4_e2m1fn_x2), 16),
            _view(sf_out, 4),
            _view(ids_out, 4),
            _view(scores_out, 4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        compiled = self.compiled.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE input staging requires eager warmup before capture"
                )
            compiled = cute.compile(self.kernel, *arguments)
            self.compiled[key] = compiled
        compiled(*arguments)
