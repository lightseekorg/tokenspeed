# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lamport mailbox-to-local copy and reset."""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from .primitives import (
    NEG_ZERO_F32_BITS,
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    fragment_is_dirty,
    load_global_u32x4,
    packed_u32x4_to_bf16x8,
    pdl_enabled,
    store_global_u32x4,
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)


class LamportCopy:
    """Consume the local physical copy of an NVLS-multicast mailbox.

    With ``has_residual`` the copy fuses a same-shape addend into the gather
    (``out = mailbox + residual``, fp32 add, bf16 round — the same rounding as
    a separate eager add), absorbing the layer's residual accumulate.
    """

    def __init__(
        self,
        hidden_dim: int,
        ctas: int,
        threads: int,
        has_residual: bool = False,
        sentinel: int = NEG_ZERO_F32_BITS,
    ):
        # Bind in host Python; the JIT resolves names from self, not module globals.
        self.use_pdl = pdl_enabled()
        self.hidden_dim = hidden_dim
        self.ctas = ctas
        self.threads = threads
        self.has_residual = has_residual
        self.sentinel = sentinel

    @cute.jit
    def __call__(
        self,
        symmetric_mailbox: cute.Tensor,
        local_output: cute.Tensor,
        residual: cute.Tensor,
        m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(symmetric_mailbox, local_output, residual, m).launch(
            grid=(self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        symmetric_mailbox: cute.Tensor,
        local_output: cute.Tensor,
        residual: cute.Tensor,
        m: cutlass.Int32,
    ):
        # Mailbox inspection must not pass the producer GEMM's programmatic completion point.
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        thread = cutlass.Int64(block * self.threads + tidx)
        stride = cutlass.Int64(self.ctas * self.threads)
        fragments = cutlass.Int64(m) * cutlass.Int64(self.hidden_dim // VEC_BF16)

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            packed = load_global_u32x4(source, volatile=True)
            while fragment_is_dirty(packed, self.sentinel):
                packed = load_global_u32x4(source, volatile=True)

            if cutlass.const_expr(self.has_residual):
                res_ptr = cute.make_ptr(
                    cutlass.BFloat16,
                    (residual.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                res_packed = load_global_u32x4(res_ptr, volatile=False)
                gathered = packed_u32x4_to_bf16x8(packed)
                addend = packed_u32x4_to_bf16x8(res_packed)
                fused = cute.make_rmem_tensor(
                    cute.make_layout((VEC_BF16,)), cutlass.BFloat16
                )
                for i in cutlass.range_constexpr(VEC_BF16):
                    fused[i] = (
                        gathered[i].to(cutlass.Float32) + addend[i].to(cutlass.Float32)
                    ).to(cutlass.BFloat16)
                packed = bf16x8_to_packed_u32x4(fused.load())

            destination = cutlass.Int64((local_output.iterator + element).toint())
            store_global_u32x4(destination, packed, volatile=False)
            fragment = fragment + stride

        # The returned tensor is complete; a same-stream successor may overlap the mailbox cleanup below.
        cute.arch.griddepcontrol_launch_dependents()

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            store_lamport_sentinel_128(source, sentinel=self.sentinel)
            fragment = fragment + stride


# functools.cache keys on the call shape, so every caller must pass these alike.
@functools.cache
def compile_kernel(
    hidden_dim: int,
    max_m: int,
    ctas: int,
    threads: int,
    device_index: int,
    has_residual: bool = False,
    sentinel: int = NEG_ZERO_F32_BITS,
):
    if hidden_dim <= 0 or hidden_dim % VEC_BF16:
        raise ValueError("hidden_dim must be a positive multiple of 8")
    if max_m <= 0:
        raise ValueError("max_m must be positive")
    if ctas <= 0:
        raise ValueError("Lamport copy ctas must be positive")
    if not 1 <= threads <= 1024:
        raise ValueError("Lamport copy threads must be in [1, 1024]")
    with torch.accelerator.device_index(device_index):
        mailbox = make_fake_compact_tensor(
            cutlass.BFloat16, (max_m * hidden_dim,), assumed_align=16
        )
        output = make_fake_compact_tensor(
            cutlass.BFloat16,
            (cute.sym_int32(divisibility=VEC_BF16),),
            assumed_align=16,
        )
        residual = make_fake_compact_tensor(
            cutlass.BFloat16,
            (cute.sym_int32(divisibility=VEC_BF16),),
            assumed_align=16,
        )
        return cute.compile(
            LamportCopy(hidden_dim, ctas, threads, has_residual, sentinel),
            mailbox,
            output,
            residual,
            cutlass.Int32(max_m),
            make_fake_stream(),
        )


def launch(
    symmetric_mailbox: torch.Tensor,
    local_output: torch.Tensor,
    *,
    m: int,
    hidden_dim: int,
    max_m: int,
    ctas: int,
    threads: int,
    residual: torch.Tensor | None = None,
    sentinel: int = NEG_ZERO_F32_BITS,
) -> None:
    if not 1 <= m <= max_m:
        raise ValueError(f"runtime M={m} must be in [1, {max_m}]")
    if (
        symmetric_mailbox.ndim != 3
        or symmetric_mailbox.dtype != torch.bfloat16
        or not symmetric_mailbox.is_cuda
        or not symmetric_mailbox.is_contiguous()
        or symmetric_mailbox.shape[0] != 1
        or symmetric_mailbox.shape[2] != hidden_dim
    ):
        raise ValueError(
            f"symmetric_mailbox must be contiguous CUDA BF16 [1,M,{hidden_dim}]"
        )
    if symmetric_mailbox.shape[1] != max_m:
        raise ValueError("symmetric mailbox must retain its full max_m capacity")
    if (
        local_output.shape != (1, m, hidden_dim)
        or local_output.dtype != torch.bfloat16
        or local_output.device != symmetric_mailbox.device
        or not local_output.is_contiguous()
    ):
        raise ValueError("local_output must be contiguous CUDA BF16 [1,M,H]")

    if residual is not None and (
        residual.shape != (m, hidden_dim)
        or residual.dtype != torch.bfloat16
        or residual.device != symmetric_mailbox.device
        or not residual.is_contiguous()
    ):
        raise ValueError("residual must be contiguous CUDA BF16 [M,H]")

    device_index = symmetric_mailbox.device.index
    stream = cuda.CUstream(
        torch.cuda.current_stream(symmetric_mailbox.device).cuda_stream
    )
    residual_flat = local_output.flatten() if residual is None else residual.flatten()
    compile_kernel(
        hidden_dim,
        max_m,
        ctas,
        threads,
        device_index,
        has_residual=residual is not None,
        sentinel=sentinel,
    )(
        to_cute(symmetric_mailbox.flatten(), 16),
        to_cute_dynamic_m(
            local_output.flatten(),
            mode=0,
            assumed_align=16,
        ),
        to_cute_dynamic_m(
            residual_flat,
            mode=0,
            assumed_align=16,
        ),
        cutlass.Int32(m),
        stream,
    )


class LamportCopyKernel:
    """Copy a borrowed symmetric mailbox into a fresh local tensor."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_m: int,
        ctas: int,
        threads: int,
        sentinel: int = NEG_ZERO_F32_BITS,
    ) -> None:
        """Bind a gather to one mailbox.

        Args:
            hidden_dim: Width of a mailbox row.
            max_m: Rows the mailbox holds.
            ctas: Grid width of the gather.
            threads: Block width of the gather.
            sentinel: The empty word this mailbox is armed with. The arming
                host code and this kernel must name the same one; the caller
                that owns the mailbox owns the value.
        """
        self.hidden_dim = hidden_dim
        self.max_m = max_m
        self.ctas = ctas
        self.threads = threads
        self.sentinel = sentinel
        device_index = torch.accelerator.current_device_index()
        compile_kernel(
            hidden_dim,
            max_m,
            ctas,
            threads,
            device_index,
            has_residual=False,
            sentinel=sentinel,
        )
        compile_kernel(
            hidden_dim,
            max_m,
            ctas,
            threads,
            device_index,
            has_residual=True,
            sentinel=sentinel,
        )

    def __call__(
        self,
        symmetric_mailbox: torch.Tensor,
        *,
        m: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather the mailbox; with ``residual`` also fuse ``+ residual``."""
        if not symmetric_mailbox.is_cuda:
            raise ValueError("symmetric_mailbox must be a CUDA tensor")
        device = symmetric_mailbox.device
        with torch.accelerator.device_index(device.index):
            output = torch.empty(
                (1, m, self.hidden_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            launch(
                symmetric_mailbox,
                output,
                m=m,
                hidden_dim=self.hidden_dim,
                max_m=self.max_m,
                ctas=self.ctas,
                threads=self.threads,
                residual=residual,
                sentinel=self.sentinel,
            )
        return output
