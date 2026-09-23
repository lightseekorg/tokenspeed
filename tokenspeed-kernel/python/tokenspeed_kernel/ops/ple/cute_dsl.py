# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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

"""Opt-in CuTe gate/norm prototype; not selected by the model runtime."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack


class _GraphTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


@cute.jit
def _warp_sum(value):
    for i in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, offset=1 << i)
    return value


@cute.jit
def _block_sum(value, scratch):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    value = _warp_sum(value)
    if lane == 0:
        scratch[tid // 32] = value
    cute.arch.sync_threads()
    value = Float32(0)
    if lane < 4:
        value = scratch[lane]
    value = _warp_sum(value)
    cute.arch.sync_threads()
    return value


class _GateNorm:
    def __init__(self, hidden_size, hc_count, eps, enable_pdl):
        self.d = hidden_size
        self.hc = hc_count
        self.eps = eps
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(self, key, query, value, kg, qg, cg, gated, norm, stream):
        self.kernel(key, query, value, kg, qg, cg, gated, norm).launch(
            grid=(key.shape[0], self.hc, 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(self, key, query, value, kg, qg, cg, gated, norm):
        token, branch, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        count = (self.d + 127) // 128
        smem = cutlass.utils.SmemAllocator()
        scratch = smem.allocate_tensor(
            Float32, cute.make_layout((8,)), byte_alignment=16
        )
        k = cute.make_rmem_tensor((count,), Float32)
        q = cute.make_rmem_tensor((count,), Float32)
        v = cute.make_rmem_tensor((count,), Float32)
        kw = cute.make_rmem_tensor((count,), Float32)
        qw = cute.make_rmem_tensor((count,), Float32)
        cw = cute.make_rmem_tensor((count,), Float32)
        for i in cutlass.range_constexpr(count):
            c = lane + i * 128
            kw[i], qw[i], cw[i] = Float32(0), Float32(0), Float32(0)
            if c < self.d:
                kw[i] = Float32(kg[branch * self.d + c])
                qw[i] = Float32(qg[branch * self.d + c])
                cw[i] = Float32(cg[branch * self.d + c])
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(count):
            c = lane + i * 128
            k[i], q[i], v[i] = Float32(0), Float32(0), Float32(0)
            if c < self.d:
                k[i] = Float32(key[token, branch * self.d + c])
                q[i] = Float32(query[token, branch * self.d + c])
                v[i] = Float32(value[token, c])
        ks = (k.load() * k.load()).reduce(cute.ReductionOp.ADD, Float32(0), 0)
        qs = (q.load() * q.load()).reduce(cute.ReductionOp.ADD, Float32(0), 0)
        # Key/query norms share one cross-warp exchange and its barriers.
        ks, qs = _warp_sum(ks), _warp_sum(qs)
        if lane % 32 == 0:
            scratch[lane // 32] = ks
            scratch[lane // 32 + 4] = qs
        cute.arch.sync_threads()
        ks, qs = Float32(0), Float32(0)
        if lane % 32 < 4:
            ks = scratch[lane % 32]
            qs = scratch[lane % 32 + 4]
        ks, qs = _warp_sum(ks), _warp_sum(qs)
        cute.arch.sync_threads()
        kr = cute.math.rsqrt(ks / self.d + self.eps)
        qr = cute.math.rsqrt(qs / self.d + self.eps)
        products = cute.make_rmem_tensor((count,), Float32)
        for i in cutlass.range_constexpr(count):
            kn = (k[i] * kr * kw[i]).to(gated.element_type).to(Float32)
            qn = (q[i] * qr * qw[i]).to(gated.element_type).to(Float32)
            products[i] = kn * qn
        dot = products.load().reduce(cute.ReductionOp.ADD, Float32(0), 0)
        dot = _block_sum(dot, scratch) * (1.0 / math.sqrt(self.d))
        magnitude = cute.math.sqrt(cute.arch.fmax(cute.math.abs(dot), Float32(1e-6)))
        signed = Float32(0)
        if dot > 0:
            signed = magnitude
        elif dot < 0:
            signed = -magnitude
        sigmoid = 1.0 / (1.0 + cute.math.exp(-signed))
        for i in cutlass.range_constexpr(count):
            v[i] = (sigmoid * v[i]).to(gated.element_type).to(Float32)
            c = lane + i * 128
            if c < self.d:
                gated[token, branch * self.d + c] = v[i].to(gated.element_type)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        vs = (v.load() * v.load()).reduce(cute.ReductionOp.ADD, Float32(0), 0)
        vr = cute.math.rsqrt(_block_sum(vs, scratch) / self.d + self.eps)
        for i in cutlass.range_constexpr(count):
            c = lane + i * 128
            if c < self.d:
                norm[token, branch * self.d + c] = (v[i] * vr * cw[i]).to(
                    norm.element_type
                )


_compile_cache = {}


def ple_gate_norm_cute(
    key,
    query,
    value,
    key_weight,
    query_weight,
    conv_weight,
    *,
    hc_count: int,
    hidden_size: int,
    eps: float,
    enable_pdl: bool,
):
    """Return gated and normalized values, preserving BF16/FP16 rounding.

    Inputs match ple_gate_norm; weights must already be ready independently
    of the preceding kernel. This opt-in prototype never silently falls back.
    Compile/warm up before capture. Dtype/shape/stride/device/PDL key the cache.
    """
    inputs = (key, query, value, key_weight, query_weight, conv_weight)
    if hidden_size <= 0 or hc_count <= 0:
        raise ValueError("hidden_size and hc_count must be positive")
    if any(t.device != key.device or t.dtype != key.dtype for t in inputs):
        raise ValueError("all inputs must have the same CUDA device and dtype")
    if not key.is_cuda or key.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("expected CUDA FP16/BF16/FP32 inputs")
    if any(t.stride(-1) != 1 for t in inputs):
        raise ValueError("the last input dimension must be dense")
    total = key.shape[0]
    width = hc_count * hidden_size
    if (
        key.shape != (total, width)
        or query.shape != key.shape
        or value.shape != (total, hidden_size)
    ):
        raise ValueError("inconsistent gate/norm shapes")
    if any(t.shape != (width,) for t in inputs[3:]):
        raise ValueError("gamma weights must cover all branches")
    gated = key.new_empty((total, width))
    norm = torch.empty_like(gated)
    if not total:
        return gated, norm
    tensors = tuple(from_dlpack(_GraphTensor(t)) for t in (*inputs, gated, norm))
    stream = cuda.CUstream(torch.cuda.current_stream(key.device).cuda_stream)
    signature = (
        key.device.index,
        key.dtype,
        tuple((tuple(t.shape), t.stride()) for t in inputs),
        hc_count,
        hidden_size,
        eps,
        enable_pdl,
    )
    compiled = _compile_cache.get(signature)
    if compiled is None:
        compiled = cute.compile(
            _GateNorm(hidden_size, hc_count, eps, enable_pdl), *tensors, stream
        )
        _compile_cache[signature] = compiled
    compiled(*tensors, stream)
    return gated, norm
