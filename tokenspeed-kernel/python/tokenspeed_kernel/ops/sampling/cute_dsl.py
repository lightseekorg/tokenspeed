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

"""CuTe DSL based sampling kernels.

Wraps the upstream CuTe DSL ``ArgmaxKernel`` (derived from the Quack library and
ported through TensorRT-LLM) so the sampling public API can register it
without touching the third-party module directly.

Exports two entry points:

* :func:`argmax`: drop-in replacement for ``torch.argmax(logits, dim=-1)``.
  Returns int64 indices written by the kernel directly — no post-kernel cast
  on the hot path. Transparently falls back to ``torch.argmax`` when the CuTe
  DSL kernel is unavailable or its preconditions are not met
  (dtype/N/alignment/SM-version).
* :func:`argmax_pair`: row-wise ``(max_value, argmax_index)`` packed as a
  single ``(M, 2)`` float32 tensor. The kernel writes the max value and index
  into two separate tensors; this entry point assembles them back into the
  legacy ``(M, 2)`` layout (one extra elementwise copy off the hot path). The
  runtime no longer uses this layout — kept for tests / future logprob users.

The distributed argmax (:func:`distributed_argmax`) and the fused greedy
DSpark Markov step built on its slots (:func:`markov_argmax_step`) are
described in this directory's ``README.md``.

Platform support:

The CuTe DSL kernel ships only for NVIDIA Hopper/Blackwell (sm_90..<sm_120).
The common ``tokenspeed_kernel.ops.sampling.argmax`` API selects this
registered solution on NVIDIA.
"""

from dataclasses import dataclass

import torch
import torch.distributed as _dist
import torch.distributed._symmetric_memory as _symm_mem
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "MarkovArgmaxWorkspace",
    "argmax",
    "argmax_pair",
    "create_dist_argmax_state",
    "create_markov_argmax_workspace",
    "supports_dist_argmax_shape",
    "supports_markov_argmax_shape",
    "try_create_dist_argmax_state",
    "cute_dsl_argmax",
    "distributed_argmax",
    "is_available",
    "markov_argmax_step",
]

_argmax_kernel_impl = error_fn
_compile_cache: dict[tuple, object] = {}

# Minimum vocab size for the CuTe tiled kernel.
#
# The kernel hangs on B200 (sm_100) when ``_calculate_threads_per_row()``
# returns 32 AND ``tiler_mn[1] == N`` (i.e. ``is_even_N`` skips ``fill_oob``).
# Empirically that happens for N ∈ {256, 512, 1024, 2048, 3072} — every clean
# multiple in the upstream ``128 < N <= 3072`` band. Bumping the floor above
# 3072 sidesteps the bad band entirely; every real LLM vocab (≥ 30K) is far
# above this, so we never lose the kernel in practice.
_MIN_VOCAB_SIZE = 4096

# The async copy requires 128-byte alignment.
_VOCAB_SIZE_ALIGNMENT = 32

# Fused Markov step: 4 warp pairs per CTA, each halving one 16-row group's rank dim.
_MARKOV_WARPS = 8
_MARKOV_K_SPLIT = 2
_MARKOV_GROUPS_PER_CTA = _MARKOV_WARPS // _MARKOV_K_SPLIT
_MARKOV_MAX_ROW_TILES = 8  # 8-row MMA tiles: at most 64 batch rows per CTA
_MARKOV_KEY_NONE = -(1 << 63)  # empty candidate key, below every real one


def _ts_supported_arch() -> bool:
    """Gate: only NVIDIA Hopper/Blackwell run the CuTe DSL kernel.

    * Vendor must be NVIDIA — AMD ROCm and any future vendor get the torch
      fallback (CuTe DSL has no ROCm backend).
    * SM range ``[9.0, 12.0)``: ``redux.sync.max.f32`` exists from Blackwell
      (sm_100/sm_103); we run on Hopper too via the shuffle path. ``sm_120+``
      is excluded — upstream TRT-LLM reports CUTLASS DSL JIT instability there.
    * If platform detection itself raises (e.g. CPU-only host with no GPU),
      treat it as unsupported and let callers fall back transparently.
    """
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False
    sm = p.arch_version.major * 10 + p.arch_version.minor
    return 90 <= sm < 120


def _has_cluster_launch_support() -> bool:
    """Check if the GPU supports TMA cluster launches required by the kernel.

    The CUTLASS DSL ArgmaxKernel uses cluster dimensions > 1 via TMA, which
    requires hardware cluster launch support. NVIDIA H20 GPUs report sm_90
    (Hopper architecture) but lack the cluster launch capability, causing
    CUDA_ERROR_INVALID_CLUSTER_SIZE (error 912) at kernel launch time.

    This function uses a device-name heuristic to detect H20 SKUs and route
    them through the torch.argmax fallback instead.
    """
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False

    # Only Hopper (sm_90) SKUs are affected -- Blackwell always supports
    # cluster launches.
    if p.arch_version.major > 9:
        return True

    # Device-name heuristic: H20 is the only sm_90 SKU known to lack cluster
    # launch support. Other Hopper SKUs (H100, H200, H800) all support it.
    import re

    return not re.search(r"\bH20\b", p.device_name, re.IGNORECASE)


_ARCH_SUPPORTED = _ts_supported_arch()


# Only import the third-party CuTe DSL module on supported NVIDIA hardware
# with cluster launch capability. H20 GPUs (sm_90 without TMA cluster support)
# hit CUDA_ERROR_INVALID_CLUSTER_SIZE; route them through torch.argmax instead.
_CUTE_AVAILABLE = False
if _ARCH_SUPPORTED and _has_cluster_launch_support():
    try:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass._mlir.dialects import llvm
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute.typing import Int32
        from cutlass.cutlass_dsl import T, dsl_user_op
        from tokenspeed_kernel.thirdparty.cute_dsl.argmax import (
            ArgmaxKernel,
            CUDAGraphCompatibleWrapper,
            domain_offset_i64,
            elem_pointer,
            fill_oob,
            predicate_k,
            store_shared_remote,
            torch2cute_dtype_map,
            warp_argmax_redux,
            warp_reduce_argmax,
        )

        _CUTE_AVAILABLE = True
    except ImportError:
        _CUTE_AVAILABLE = False


def is_available() -> bool:
    """Whether the CuTe DSL argmax kernel can run on this platform."""
    return _CUTE_AVAILABLE


def supports_dist_argmax_shape(
    vocab_per_rank: int, dtype: torch.dtype, world_size: int
) -> bool:
    """Whether a shard's width, dtype and group size fit the distributed argmax.

    Deliberately excludes platform and multicast support, which can differ
    between ranks: this verdict follows from the model config alone, so a
    caller may reject on it before doing any collective work.

    Args:
        vocab_per_rank: Column count of one rank's logits shard.
        dtype: Value dtype of those logits.
        world_size: Ranks in the group the argmax reduces over.

    Returns:
        True when the shard is one :func:`distributed_argmax` will accept.
    """
    return (
        dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 1 <= world_size <= 32
        and vocab_per_rank >= _MIN_VOCAB_SIZE
        and vocab_per_rank % _VOCAB_SIZE_ALIGNMENT == 0
        and world_size * vocab_per_rank < 0x7FFFFFFF
    )


def _supports_cute(N: int, dtype: torch.dtype) -> bool:
    if not _CUTE_AVAILABLE:
        return False
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if N < _MIN_VOCAB_SIZE:
        return False
    if N % _VOCAB_SIZE_ALIGNMENT != 0:
        return False
    return True


def _convert_to_cute(t: torch.Tensor):
    """Wrap a torch tensor as a CuTe DSL tensor with a CUDA-graph-safe view."""
    return from_dlpack(
        CUDAGraphCompatibleWrapper(t.detach()), assumed_align=16
    ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))


def _convert_to_cute_1d(t: torch.Tensor):
    """1D-tensor variant of :func:`_convert_to_cute`."""
    return from_dlpack(
        CUDAGraphCompatibleWrapper(t.detach()), assumed_align=16
    ).mark_compact_shape_dynamic(mode=0, stride_order=(0,))


def _invoke_kernel(
    logits: torch.Tensor, out_max: torch.Tensor, out_idx: torch.Tensor
) -> None:
    """Launch ArgmaxKernel with separate ``(M,)`` max and idx output tensors.

    Caller is responsible for shape/dtype checks; this helper assumes inputs
    are already validated by :func:`_supports_cute`.
    """
    dtype = torch2cute_dtype_map[logits.dtype]
    x_tensor = _convert_to_cute(logits)
    max_tensor = _convert_to_cute_1d(out_max)
    idx_tensor = _convert_to_cute_1d(out_idx)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Blackwell (sm_100/103) supports redux.sync.max.f32; Hopper falls back to
    # warp shuffles.
    p = current_platform()
    sm = p.arch_version.major * 10 + p.arch_version.minor
    use_redux = 100 <= sm < 120

    N = logits.shape[1]
    # Cache by index dtype too: the kernel writes the index with the output
    # tensor's element type, so int64 vs int32 produce distinct compiled units.
    compile_key = (dtype, N, use_redux, out_idx.dtype)
    compiled = _compile_cache.get(compile_key)
    if compiled is None:
        kernel = ArgmaxKernel(dtype, N, use_redux=use_redux)
        compiled = cute.compile(kernel, x_tensor, max_tensor, idx_tensor, stream)
        _compile_cache[compile_key] = compiled

    compiled(x_tensor, max_tensor, idx_tensor, stream)


_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)


def _validate_argmax_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _validate_argmax_pair_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    M = logits.shape[0]
    if out.shape != (M, 2):
        raise ValueError(f"out must have shape (M, 2)={M, 2}, got {tuple(out.shape)}")
    if out.dtype != torch.float32 or out.device != logits.device:
        raise ValueError("out must be float32 on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch implementation of :func:`argmax`.

    Selected at import time on non-NVIDIA / unsupported-SM hosts (AMD ROCm,
    CPU-only, sm_80, sm_120+, missing ``nvidia-cutlass-dsl``). Also reached
    per-call from the cute path when the input fails the kernel's
    preconditions (1D / non-CUDA / fp16 / bf16 / small N / unaligned N).
    """
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _argmax_pair_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch implementation of :func:`argmax_pair`.

    Selected at import time on non-NVIDIA / unsupported-SM hosts, and reached
    per-call from the cute path when the input fails the kernel's
    preconditions.
    """
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    M = logits.shape[0]
    device = logits.device
    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=device)
    else:
        _validate_argmax_pair_out(logits, out)

    max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
    out[:, 0:1].copy_(max_vals.to(torch.float32))
    out[:, 1:2].copy_(max_indices.to(torch.float32))
    return out


def _register_cute_argmax(fn):
    if not _CUTE_AVAILABLE:
        return fn
    return register_kernel(
        "sampling",
        "argmax",
        name="cute_dsl_argmax",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(11, 9),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
    )(fn)


@_register_cute_argmax
def cute_dsl_argmax(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """CuTe DSL fast path for argmax.

    Falls back per-call to :func:`_argmax_torch_fallback` when the input
    isn't kernel-eligible (1D / non-CUDA / empty batch / unsupported dtype /
    small N / unaligned N). Only ever bound to the public ``argmax`` name on
    NVIDIA hosts with the cute DSL Python packages available — see the
    module-level dispatch below.
    """
    if out is not None:
        _validate_argmax_out(logits, out)

    if (
        logits.dim() != 2
        or not logits.is_cuda
        or logits.shape[0] == 0
        or not _supports_cute(logits.shape[1], logits.dtype)
    ):
        return _argmax_torch_fallback(logits, out=out)

    M = logits.shape[0]
    device = logits.device
    out_idx = (
        out if out is not None else torch.empty((M,), dtype=torch.int64, device=device)
    )

    # The max value is needed only inside the kernel reduction; the caller
    # never sees it. Allocate a scratch buffer so the kernel has somewhere to
    # write it.
    scratch_max = torch.empty((M,), dtype=torch.float32, device=device)
    _invoke_kernel(logits, scratch_max, out_idx)
    return out_idx


def _argmax_pair_cute(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """CuTe DSL fast path for argmax_pair. Falls back per-call when needed."""
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")

    M, N = logits.shape
    device = logits.device

    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=device)
    else:
        _validate_argmax_pair_out(logits, out)

    if not logits.is_cuda or M == 0 or not _supports_cute(N, logits.dtype):
        # Reuse the pure-torch packing path; pass our pre-allocated buffer so
        # the caller-supplied ``out`` is honored.
        return _argmax_pair_torch_fallback(logits, out=out)

    # Kernel writes into separate (M,) tensors; assemble into the legacy
    # (M, 2) layout for backward compatibility. This is off the runtime hot
    # path (callers use :func:`argmax` instead), so the extra copy/cast is OK.
    tmp_max = torch.empty((M,), dtype=torch.float32, device=device)
    tmp_idx = torch.empty((M,), dtype=torch.int64, device=device)
    _invoke_kernel(logits, tmp_max, tmp_idx)
    out[:, 0].copy_(tmp_max)
    out[:, 1].copy_(tmp_idx.to(torch.float32))
    return out


# Direct CuTe DSL module API. The common runtime-facing API lives in
# tokenspeed_kernel.ops.sampling and selects among registered solutions.
if _CUTE_AVAILABLE:
    argmax = cute_dsl_argmax
    argmax_pair = _argmax_pair_cute
    _argmax_kernel_impl = _invoke_kernel
else:
    argmax = _argmax_torch_fallback
    argmax_pair = _argmax_pair_torch_fallback


# Distributed (cross-rank) argmax — kernel + operator with symm-mem workspace.
if _CUTE_AVAILABLE:

    @dsl_user_op
    def ptx_multimem_st_release_u64(
        mc_ptr: cutlass.Int64, value: cutlass.Int64, *, loc=None, ip=None
    ) -> None:
        llvm.inline_asm(
            None,
            [
                cutlass.Int64(mc_ptr).ir_value(loc=loc, ip=ip),
                cutlass.Int64(value).ir_value(loc=loc, ip=ip),
            ],
            """multimem.st.release.sys.global.b64 [$0], $1;""",
            "l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def ptx_ld_acquire_sys_u64(
        addr: cutlass.Int64, *, loc=None, ip=None
    ) -> cutlass.Int64:
        return cutlass.Int64(
            llvm.inline_asm(
                T.i64(),
                [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
                """ld.acquire.sys.global.u64 $0, [$1];""",
                "=l,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def ptx_atomic_and_relaxed_sys_u64(
        addr: cutlass.Int64, mask: cutlass.Int64, *, loc=None, ip=None
    ) -> cutlass.Int64:
        return cutlass.Int64(
            llvm.inline_asm(
                T.i64(),
                [
                    cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
                    cutlass.Int64(mask).ir_value(loc=loc, ip=ip),
                ],
                """atom.and.relaxed.sys.global.b64 $0, [$1], $2;""",
                "=l,l,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    # u64 slot layout: [flag:63 | idx:32..62 | f32 value:0..31].
    @cute.jit
    def pack_argmax_payload_u64(
        val_f32: cutlass.Float32, global_idx: cutlass.Int32
    ) -> cutlass.Int64:
        val_bits = val_f32.bitcast(cutlass.Int32).to(cutlass.Int64) & cutlass.Int64(
            0xFFFFFFFF
        )
        idx_bits = cutlass.Int64(global_idx) & cutlass.Int64(0x7FFFFFFF)
        flag_bit = cutlass.Int64(1) << cutlass.Int64(63)
        return flag_bit | (idx_bits << cutlass.Int64(32)) | val_bits

    @cute.jit
    def unpack_argmax_payload_u64(packed: cutlass.Int64):
        val_bits32 = (packed & cutlass.Int64(0xFFFFFFFF)).to(Int32)
        val_f32 = val_bits32.bitcast(cutlass.Float32)
        idx_i32 = ((packed >> cutlass.Int64(32)) & cutlass.Int64(0x7FFFFFFF)).to(Int32)
        return val_f32, idx_i32

    @dsl_user_op
    def ptx_ld_global_u32(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int32:
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
                """ld.global.u32 $0, [$1];""",
                "=r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def ptx_atomic_add_acq_rel_sys_u32(
        addr: cutlass.Int64, val: cutlass.Int32, *, loc=None, ip=None
    ) -> cutlass.Int32:
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [
                    cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
                    cutlass.Int32(val).ir_value(loc=loc, ip=ip),
                ],
                """atom.add.acq_rel.sys.global.u32 $0, [$1], $2;""",
                "=r,l,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def ptx_st_relaxed_sys_u32(
        addr: cutlass.Int64, val: cutlass.Int32, *, loc=None, ip=None
    ) -> None:
        llvm.inline_asm(
            None,
            [
                cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(val).ir_value(loc=loc, ip=ip),
            ],
            """st.relaxed.sys.global.u32 [$0], $1;""",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    class DistArgmaxKernel(ArgmaxKernel):
        """Cross-rank argmax: per-rank local argmax + warp-level peer reduce.

        ``skip_ping_pong=False`` (default): kernel reads ``round_id`` and
        alternates between two slot bands so back-to-back calls are
        race-free on their own. ``skip_ping_pong=True``: hardcodes band
        0; caller must guarantee an external sync between consecutive
        calls (same contract as ``all_gather_inner``'s
        ``SKIP_ENTRY_SYNC=True``).
        """

        def __init__(
            self,
            dtype,
            N: int,
            use_redux: bool = False,
            world_size: int = 1,
            rank: int = 0,
            skip_ping_pong: bool = False,
        ):
            super().__init__(dtype, N, use_redux=use_redux)
            self.world_size = world_size
            self.rank = rank
            self.dist_tp_enabled = world_size > 1
            self.skip_ping_pong = skip_ping_pong

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mO_max: cute.Tensor,
            mO_idx: cute.Tensor,
            stream: cuda.CUstream,
            slot_ptrs: cutlass.Int64,
            slot_multicast_ptr: cutlass.Int64,
            round_id_ptr: cutlass.Int64,
            warps_done_ptr: cutlass.Int64,
        ):
            self._set_cluster_n()
            tiler_mn, tv_layout = self._get_tv_layout()
            num_threads = cute.size(tv_layout, mode=[0])
            num_warps = num_threads // cute.arch.WARP_SIZE

            self.kernel(
                mX,
                mO_max,
                mO_idx,
                tv_layout,
                tiler_mn,
                slot_ptrs,
                slot_multicast_ptr,
                round_id_ptr,
                warps_done_ptr,
            ).launch(
                grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
                block=[num_threads, 1, 1],
                cluster=(
                    [1, self.cluster_n, 1]
                    if cutlass.const_expr(self.cluster_n > 1)
                    else None
                ),
                smem=self._smem_size_in_bytes(tiler_mn, num_warps),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mO_max: cute.Tensor,
            mO_idx: cute.Tensor,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
            slot_ptrs: cutlass.Int64,
            slot_multicast_ptr: cutlass.Int64,
            round_id_ptr: cutlass.Int64,
            warps_done_ptr: cutlass.Int64,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, bidz = cute.arch.block_idx()

            if cutlass.const_expr(self.cluster_n > 1):
                cluster_y = cute.arch.block_idx()[1]
            else:
                cluster_y = cutlass.const_expr(0)

            shape = mX.shape
            idX = cute.make_identity_tensor(shape)

            mX = domain_offset_i64((bidx * tiler_mn[0], 0), mX)
            gX = cute.local_tile(mX, tiler_mn, (0, cluster_y))
            mO_max = domain_offset_i64((bidx * tiler_mn[0],), mO_max)
            mO_idx = domain_offset_i64((bidx * tiler_mn[0],), mO_idx)
            cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
                smem, tv_layout
            )

            copy_atom_load_X = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=128,
            )
            thr_copy_X = cute.make_tiled_copy(
                copy_atom_load_X, tv_layout, tiler_mn
            ).get_slice(tidx)

            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

            tvlayout_cX = cute.composition(cX, tv_layout)
            thr_coord = (tidx, (None, None))
            thr_cX = tvlayout_cX[thr_coord]

            tXrX = cute.make_fragment_like(tXgX)
            num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
            self._initialize_cluster(tidx, mbar_ptr, num_warps)

            is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
            tXpX = (
                predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
                if not is_even_N
                else None
            )

            if tXcX[0][0] < shape[0]:
                cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

            if cutlass.const_expr(not is_even_N):
                fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)

            current_max = -tXsX.element_type.inf
            current_argmax = Int32(0xFFFFFFFF)

            for i in cutlass.range_constexpr(thr_cX.shape[0]):
                for j in cutlass.range_constexpr(thr_cX.shape[1]):
                    col_idx = thr_cX[i, j][1]
                    linear_idx = i + j * thr_cX.shape[0]
                    element_value1 = x[linear_idx]
                    if element_value1 > current_max:
                        current_max = element_value1
                        current_argmax = Int32(col_idx)

            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            if cutlass.const_expr(self.use_redux):
                warp_max, warp_argmax = warp_argmax_redux(current_max, current_argmax)
            else:
                warp_max, warp_argmax = warp_reduce_argmax(current_max, current_argmax)

            if cutlass.const_expr(self.cluster_n == 1):
                warps_per_row = cute.size(reduction_buffer.shape[1])
                row_idx_buf, col_idx_buf = (
                    warp_idx // warps_per_row,
                    warp_idx % warps_per_row,
                )

                if lane_idx == 0:
                    reduction_buffer[row_idx_buf, col_idx_buf, 0, 0] = warp_max
                    reduction_buffer[row_idx_buf, col_idx_buf, 0, 1] = warp_argmax.to(
                        cutlass.Float32
                    )

                cute.arch.barrier()
                block_reduce_max = -tXsX.element_type.inf
                block_reduce_argmax = Int32(0xFFFFFFFF)

                if lane_idx < warps_per_row:
                    block_reduce_max = reduction_buffer[row_idx_buf, lane_idx, 0, 0]
                    block_reduce_argmax = reduction_buffer[
                        row_idx_buf, lane_idx, 0, 1
                    ].to(cutlass.Int32)

                if cutlass.const_expr(self.use_redux):
                    warp_max, warp_argmax = warp_argmax_redux(
                        block_reduce_max, block_reduce_argmax
                    )
                else:
                    warp_max, warp_argmax = warp_reduce_argmax(
                        block_reduce_max, block_reduce_argmax
                    )
            else:
                cute.arch.cluster_wait()
                warps_per_row, cluster_n = reduction_buffer.shape[1]
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                rows_per_block, (warps_per_row, cluster_n), _, _ = (
                    reduction_buffer.shape
                )
                row_idx_buf, col_idx_buf = (
                    warp_idx // warps_per_row,
                    warp_idx % warps_per_row,
                )

                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps_total = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps_total
                            * cluster_n
                            * 2
                            * reduction_buffer.element_type.width
                            // 8,
                        )

                if lane_idx < cluster_n:
                    store_shared_remote(
                        warp_max,
                        elem_pointer(
                            reduction_buffer,
                            (row_idx_buf, (col_idx_buf, cta_rank_in_cluster), 0, 0),
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                    store_shared_remote(
                        warp_argmax.to(cutlass.Float32),
                        elem_pointer(
                            reduction_buffer,
                            (row_idx_buf, (col_idx_buf, cta_rank_in_cluster), 0, 1),
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )

                cute.arch.mbarrier_wait(mbar_ptr, phase=0)
                block_reduce_val = -tXsX.element_type.inf
                block_reduce_argmax = Int32(0xFFFFFFFF)
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)

                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        element_max = reduction_buffer[row_idx_buf, idx, 0, 0]
                        element_argmax = reduction_buffer[row_idx_buf, idx, 0, 1].to(
                            cutlass.Int32
                        )
                        if element_max > block_reduce_val:
                            block_reduce_val = element_max
                            block_reduce_argmax = element_argmax
                        elif element_max == block_reduce_val:
                            if element_argmax < block_reduce_argmax:
                                block_reduce_argmax = element_argmax

                if cutlass.const_expr(self.use_redux):
                    warp_max, warp_argmax = warp_argmax_redux(
                        block_reduce_val, block_reduce_argmax
                    )
                else:
                    warp_max, warp_argmax = warp_reduce_argmax(
                        block_reduce_val, block_reduce_argmax
                    )

            row_idx = tXcX[0][0]
            warps_per_row = tv_layout.shape[0][0] // cute.arch.WARP_SIZE
            local_row_idx = row_idx - (bidx * tiler_mn[0])
            first_warp_for_row = local_row_idx * warps_per_row
            first_thread_for_row = first_warp_for_row * cute.arch.WARP_SIZE

            if cutlass.const_expr(self.dist_tp_enabled):
                row_is_valid = (
                    row_idx < shape[0]
                    and local_row_idx >= 0
                    and local_row_idx < tiler_mn[0]
                    and (self.cluster_n == 1 or bidy == 0)
                )
                is_leader_warp = (warp_idx == first_warp_for_row) and row_is_valid

                if is_leader_warp:
                    if cutlass.const_expr(self.skip_ping_pong):
                        round_bit = Int32(0)
                    else:
                        round_id_val = ptx_ld_global_u32(round_id_ptr)
                        round_bit = round_id_val & Int32(1)
                    band_off_u64 = (
                        cutlass.Int64(row_idx) * cutlass.Int64(2)
                        + cutlass.Int64(round_bit)
                    ) * cutlass.Int64(self.world_size)

                    if lane_idx == 0:
                        global_idx = Int32(self.rank * self.N) + warp_argmax
                        if warp_argmax == Int32(0xFFFFFFFF):
                            global_idx = Int32(0x7FFFFFFF)
                        packed = pack_argmax_payload_u64(warp_max, global_idx)
                        mc_addr = slot_multicast_ptr + (
                            band_off_u64 + cutlass.Int64(self.rank)
                        ) * cutlass.Int64(8)
                        ptx_multimem_st_release_u64(mc_addr, packed)

                    local_pad = ptx_ld_acquire_sys_u64(
                        slot_ptrs + cutlass.Int64(self.rank) * cutlass.Int64(8)
                    )

                    flag_bit_check = cutlass.Int64(1) << cutlass.Int64(63)
                    clear_mask = cutlass.Int64(0x7FFFFFFFFFFFFFFF)
                    peer_val = -tXsX.element_type.inf
                    peer_idx = Int32(0x7FFFFFFF)
                    if lane_idx < self.world_size:
                        slot_addr = local_pad + (
                            band_off_u64 + cutlass.Int64(lane_idx)
                        ) * cutlass.Int64(8)
                        v = cutlass.Int64(0)
                        while (v & flag_bit_check) == cutlass.Int64(0):
                            v = ptx_ld_acquire_sys_u64(slot_addr)
                        peer_val, peer_idx = unpack_argmax_payload_u64(v)
                        ptx_atomic_and_relaxed_sys_u64(slot_addr, clear_mask)

                    if cutlass.const_expr(self.use_redux):
                        warp_max, warp_argmax = warp_argmax_redux(peer_val, peer_idx)
                    else:
                        warp_max, warp_argmax = warp_reduce_argmax(peer_val, peer_idx)
                    # A row whose elements never beat -inf (all-NaN / all -inf) leaves
                    # the argmax at its 0xFFFFFFFF sentinel; emit the in-range index 0.
                    if warp_argmax == Int32(0x7FFFFFFF):
                        warp_argmax = Int32(0)

                    if cutlass.const_expr(not self.skip_ping_pong):
                        if lane_idx == 0:
                            old = ptx_atomic_add_acq_rel_sys_u32(
                                warps_done_ptr, cutlass.Int32(1)
                            )
                            if old == cutlass.Int32(shape[0]) - cutlass.Int32(1):
                                ptx_atomic_add_acq_rel_sys_u32(
                                    round_id_ptr, cutlass.Int32(1)
                                )
                                ptx_st_relaxed_sys_u32(warps_done_ptr, cutlass.Int32(0))

            if (
                tidx == first_thread_for_row
                and row_idx < shape[0]
                and local_row_idx >= 0
                and local_row_idx < tiler_mn[0]
                and (self.cluster_n == 1 or bidy == 0)
            ):
                mO_max[local_row_idx] = warp_max.to(mO_max.element_type)
                mO_idx[local_row_idx] = warp_argmax.to(mO_idx.element_type)

    # Fused greedy DSpark Markov step (see MarkovArgmaxStepKernel).

    def _inline_asm(result, code, constraints, operands, *, effects, loc, ip):
        """One inline-PTX block; a tuple ``result`` returns a tuple of values."""
        if result is None:
            ir_type = None
        elif isinstance(result, tuple):
            ir_type = llvm.StructType.get_literal([dtype.mlir_type for dtype in result])
        else:
            ir_type = result.mlir_type
        value = llvm.inline_asm(
            ir_type,
            [operand.ir_value(loc=loc, ip=ip) for operand in operands],
            code,
            constraints,
            has_side_effects=effects,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        if result is None:
            return None
        if isinstance(result, tuple):
            return tuple(
                dtype(llvm.extractvalue(dtype.mlir_type, value, [i], loc=loc, ip=ip))
                for i, dtype in enumerate(result)
            )
        return result(value)

    @dsl_user_op
    def _ld_global_nc_v4(addr: cutlass.Int64, *, loc=None, ip=None):
        return _inline_asm(
            (Int32,) * 4,
            "ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4];",
            "=r,=r,=r,=r,l",
            [cutlass.Int64(addr)],
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_shared_v4(addr: Int32, *, loc=None, ip=None):
        # Written by cp.async before a barrier: never hoisted above it.
        return _inline_asm(
            (Int32,) * 4,
            "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
            "=r,=r,=r,=r,r",
            [Int32(addr)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_global_b32(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
        return _inline_asm(
            Int32,
            "ld.global.b32 $0, [$1];",
            "=r,l",
            [cutlass.Int64(addr)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _cp_async_16(
        dst: Int32, src: cutlass.Int64, src_bytes: Int32, *, loc=None, ip=None
    ) -> None:
        # src_bytes 0 zero-fills the 16 destination bytes without reading src.
        _inline_asm(
            None,
            "cp.async.cg.shared.global [$0], [$1], 16, $2;",
            "r,l,r",
            [Int32(dst), cutlass.Int64(src), Int32(src_bytes)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _mma_m16n8k16(a, b, c, *, bf16: bool, loc=None, ip=None):
        """``c + a @ b`` on tensor cores: a 16x16, b 16x8, FP32 accumulators."""
        kind = "bf16" if bf16 else "f16"
        code = (
            f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
        )
        return _inline_asm(
            (cutlass.Float32,) * 4,
            code,
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            [*a, *b, *c],
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_half_as_f32(
        addr: cutlass.Int64, *, bf16: bool, loc=None, ip=None
    ) -> cutlass.Float32:
        kind = "bf16" if bf16 else "f16"
        return _inline_asm(
            cutlass.Float32,
            f"{{ .reg .b16 h; ld.global.nc.b16 h, [$1]; cvt.f32.{kind} $0, h; }}",
            "=f,l",
            [cutlass.Int64(addr)],
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _bias_add_rounded(
        bias: cutlass.Float32,
        base: cutlass.Float32,
        *,
        bf16: bool,
        loc=None,
        ip=None,
    ) -> cutlass.Float32:
        """``round(base + round(bias))`` in the 16-bit logits dtype, as FP32."""
        kind = "bf16" if bf16 else "f16"
        code = (
            f"{{ .reg .b16 h; .reg .f32 s; cvt.rn.{kind}.f32 h, $1; "
            f"cvt.f32.{kind} s, h; add.rn.f32 s, s, $2; "
            f"cvt.rn.{kind}.f32 h, s; cvt.f32.{kind} $0, h; }}"
        )
        return _inline_asm(
            cutlass.Float32,
            code,
            "=f,f,f",
            [cutlass.Float32(bias), cutlass.Float32(base)],
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _red_max_s64(
        addr: cutlass.Int64, value: cutlass.Int64, *, loc=None, ip=None
    ) -> None:
        _inline_asm(
            None,
            "red.relaxed.gpu.global.max.s64 [$0], $1;",
            "l,l",
            [cutlass.Int64(addr), cutlass.Int64(value)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _atom_add_acq_rel_gpu_u32(
        addr: cutlass.Int64, value: Int32, *, loc=None, ip=None
    ) -> Int32:
        return _inline_asm(
            Int32,
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            [cutlass.Int64(addr), Int32(value)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_relaxed_gpu_b64(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int64:
        return _inline_asm(
            cutlass.Int64,
            "ld.relaxed.gpu.global.b64 $0, [$1];",
            "=l,l",
            [cutlass.Int64(addr)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _st_relaxed_gpu_b64(
        addr: cutlass.Int64, value: cutlass.Int64, *, loc=None, ip=None
    ) -> None:
        _inline_asm(
            None,
            "st.relaxed.gpu.global.b64 [$0], $1;",
            "l,l",
            [cutlass.Int64(addr), cutlass.Int64(value)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _st_relaxed_gpu_u32(addr: cutlass.Int64, value: Int32, *, loc=None, ip=None):
        _inline_asm(
            None,
            "st.relaxed.gpu.global.u32 [$0], $1;",
            "l,r",
            [cutlass.Int64(addr), Int32(value)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _st_global_b32(addr: cutlass.Int64, value: Int32, *, loc=None, ip=None):
        _inline_asm(
            None,
            "st.global.b32 [$0], $1;",
            "l,r",
            [cutlass.Int64(addr), Int32(value)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _poll_slot(
        addr: cutlass.Int64, active: Int32, *, loc=None, ip=None
    ) -> cutlass.Int64:
        """The slot word, or a bare ready flag on an inactive lane.

        Flag and payload share one 64-bit word, so a relaxed load sees them
        together and independent polls can be in flight at once.
        """
        code = (
            "{ .reg .pred p; .reg .b64 v; setp.ne.s32 p, $2, 0; "
            "mov.b64 v, 0x8000000000000000; @p ld.relaxed.sys.global.b64 v, [$1]; "
            "mov.b64 $0, v; }"
        )
        return _inline_asm(
            cutlass.Int64,
            code,
            "=l,l,r",
            [cutlass.Int64(addr), Int32(active)],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _clear_slot_flag(
        addr: cutlass.Int64, active: Int32, *, loc=None, ip=None
    ) -> None:
        code = (
            "{ .reg .pred p; .reg .b64 old; setp.ne.s32 p, $2, 0; "
            "@p atom.relaxed.sys.global.and.b64 old, [$0], $1; }"
        )
        _inline_asm(
            None,
            code,
            "l,l,r",
            [
                cutlass.Int64(addr),
                cutlass.Int64(0x7FFFFFFFFFFFFFFF),
                Int32(active),
            ],
            effects=True,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _argmax_key(value: cutlass.Float32, index: Int32) -> cutlass.Int64:
        """Pack ``(value, index)`` into one int64 that orders like the argmax.

        A larger key is a larger value, then a lower index. -inf and NaN never
        win (the empty key), as under the distributed argmax's strict
        comparison; -0.0 ties with +0.0.
        """
        key = cutlass.Int64(_MARKOV_KEY_NONE)
        if value > cutlass.Float32(float("-inf")):
            bits = value.bitcast(Int32)
            if bits == Int32(-2147483648):
                bits = Int32(0)
            if bits < Int32(0):
                bits = bits ^ Int32(0x7FFFFFFF)
            key = (bits.to(cutlass.Int64) << cutlass.Int64(32)) | (
                Int32(0x7FFFFFFF) - index
            ).to(cutlass.Int64)
        return key

    @cute.jit
    def _decode_argmax_key(key: cutlass.Int64):
        """``(value, index)`` of a key; the empty key is ``(-inf, 0x7FFFFFFF)``."""
        value = cutlass.Float32(float("-inf"))
        index = Int32(0x7FFFFFFF)
        if key != cutlass.Int64(_MARKOV_KEY_NONE):
            bits = (key >> cutlass.Int64(32)).to(Int32)
            if bits < Int32(0):
                bits = bits ^ Int32(0x7FFFFFFF)
            value = bits.bitcast(cutlass.Float32)
            index = Int32(0x7FFFFFFF) - (key & cutlass.Int64(0xFFFFFFFF)).to(Int32)
        return value, index

    @cute.jit
    def _max_key(a: cutlass.Int64, b: cutlass.Int64) -> cutlass.Int64:
        result = a
        if b > a:
            result = b
        return result

    class MarkovArgmaxStepKernel:
        """One greedy DSpark Markov step over a vocabulary-sharded head.

        For row ``m`` and shard column ``v``: ``bias = embedding[prev[m]] .
        projection[v]`` accumulated in FP32 and rounded to the logits dtype,
        ``logit = round(base[m, v] + bias)``, and ``out[m]`` is the global
        token of the largest logit over every rank's shard (lowest token id
        on ties; 0 if no logit beats -inf).

        Grid ``(shard_rows / 64, ceil(rows / rows_per_cta))``. A CTA gathers
        its rows' previous-token latents into shared memory; each of its four
        warp pairs takes 16 projection rows, the two warps of a pair splitting
        the rank dimension, and multiplies them by the latents on tensor
        cores. The CTA's best logit per row is folded into one int64 key per
        row in global memory; the CTA that finishes last exchanges the keys
        with the other ranks through the distributed argmax's slots, writes
        ``out`` and resets the scratch for the next launch.
        """

        def __init__(
            self,
            dtype,
            vocab_per_rank: int,
            rank_dim: int,
            row_tiles: int,
            world_size: int,
            rank: int,
        ):
            self.bf16 = dtype is cutlass.BFloat16
            self.N = vocab_per_rank
            self.row_tiles = row_tiles
            self.rows_per_cta = 8 * row_tiles
            self.world_size = world_size
            self.rank = rank
            self.threads = 32 * _MARKOV_WARPS
            self.groups = vocab_per_rank // 16
            self.chunks_per_warp = rank_dim // (64 * _MARKOV_K_SPLIT)
            self.row_bytes = 2 * rank_dim
            # 64 spare bytes per row keep the B-fragment loads conflict-free.
            self.latent_stride = self.row_bytes + 64
            self.latent_chunks = self.rows_per_cta * self.row_bytes // 16
            self.peers = 1 << (world_size - 1).bit_length()
            self.peer_steps = self.peers.bit_length() - 1
            self.rows_per_warp_pass = 32 // self.peers
            rows_per_unroll = _MARKOV_WARPS * self.rows_per_warp_pass
            self.exchange_unroll = -(-self.rows_per_cta // rows_per_unroll)
            self.rows_per_pass = self.exchange_unroll * rows_per_unroll
            self.smem_bytes = (
                self.rows_per_cta * self.latent_stride
                + _MARKOV_GROUPS_PER_CTA * 4 * row_tiles * 32 * 4
                + _MARKOV_GROUPS_PER_CTA * self.rows_per_cta * 8
                + 16
                + 4 * 128
            )

        @cute.jit
        def __call__(
            self,
            base_ptr: cutlass.Int64,
            prev_ptr: cutlass.Int64,
            prev_stride: Int32,
            out_ptr: cutlass.Int64,
            out_stride: Int32,
            embedding_ptr: cutlass.Int64,
            embedding_rows: Int32,
            projection_ptr: cutlass.Int64,
            vocab_start: Int32,
            rows: Int32,
            keys_ptr: cutlass.Int64,
            ctas_done_ptr: cutlass.Int64,
            slot_ptrs: cutlass.Int64,
            slot_multicast_ptr: cutlass.Int64,
            round_id_ptr: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            grid_x = -(-self.groups // _MARKOV_GROUPS_PER_CTA)
            grid_y = (rows + (self.rows_per_cta - 1)) // self.rows_per_cta
            self.kernel(
                base_ptr,
                prev_ptr,
                prev_stride,
                out_ptr,
                out_stride,
                embedding_ptr,
                embedding_rows,
                projection_ptr,
                vocab_start,
                rows,
                keys_ptr,
                ctas_done_ptr,
                slot_ptrs,
                slot_multicast_ptr,
                round_id_ptr,
            ).launch(
                grid=[grid_x, grid_y, 1],
                block=[self.threads, 1, 1],
                smem=self.smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            base_ptr: cutlass.Int64,
            prev_ptr: cutlass.Int64,
            prev_stride: Int32,
            out_ptr: cutlass.Int64,
            out_stride: Int32,
            embedding_ptr: cutlass.Int64,
            embedding_rows: Int32,
            projection_ptr: cutlass.Int64,
            vocab_start: Int32,
            rows: Int32,
            keys_ptr: cutlass.Int64,
            ctas_done_ptr: cutlass.Int64,
            slot_ptrs: cutlass.Int64,
            slot_multicast_ptr: cutlass.Int64,
            round_id_ptr: cutlass.Int64,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            grid_x, grid_y, _ = cute.arch.grid_dim()
            warp = tidx // 32
            lane = tidx % 32
            g = lane // 4
            t = lane % 4
            row0 = bidy * self.rows_per_cta

            smem = cutlass.utils.SmemAllocator()
            latent = smem.allocate_array(
                cutlass.Int8,
                num_elems=self.rows_per_cta * self.latent_stride,
                byte_alignment=128,
            )
            partial = cute.make_tensor(
                smem.allocate_array(
                    cutlass.Float32,
                    num_elems=_MARKOV_GROUPS_PER_CTA * 4 * self.row_tiles * 32,
                    byte_alignment=128,
                ),
                cute.make_layout(
                    (_MARKOV_GROUPS_PER_CTA, 4 * self.row_tiles, 32),
                    stride=(4 * self.row_tiles * 32, 32, 1),
                ),
            )
            cand = cute.make_tensor(
                smem.allocate_array(
                    cutlass.Int64,
                    num_elems=_MARKOV_GROUPS_PER_CTA * self.rows_per_cta,
                    byte_alignment=128,
                ),
                cute.make_layout(
                    (_MARKOV_GROUPS_PER_CTA, self.rows_per_cta),
                    stride=(self.rows_per_cta, 1),
                ),
            )
            last_cta = cute.make_tensor(
                smem.allocate_array(Int32, num_elems=4, byte_alignment=16),
                cute.make_layout(4),
            )
            latent_base = latent.toint()

            # Groups past the shard (a partial last CTA) reload the last one.
            group_in_cta = warp // _MARKOV_K_SPLIT
            k_part = warp % _MARKOV_K_SPLIT
            group = bidx * _MARKOV_GROUPS_PER_CTA + group_in_cta
            load_group = group
            if group >= self.groups:
                load_group = Int32(self.groups - 1)
            row_g = projection_ptr + (load_group * 16 + g).to(cutlass.Int64) * (
                self.row_bytes
            )
            row_g8 = row_g + 8 * self.row_bytes
            # Lane t holds rank columns 8t.. and 32+8t.. of each 64-wide chunk.
            first_byte = k_part * (self.chunks_per_warp * 128) + t * 16
            a_words = []
            for c in cutlass.range_constexpr(self.chunks_per_warp):
                offset = (first_byte + c * 128).to(cutlass.Int64)
                a_words.append(
                    (
                        (
                            _ld_global_nc_v4(row_g + offset),
                            _ld_global_nc_v4(row_g8 + offset),
                        ),
                        (
                            _ld_global_nc_v4(row_g + offset + 64),
                            _ld_global_nc_v4(row_g8 + offset + 64),
                        ),
                    )
                )

            # Latent rows of the clamped previous tokens; padding rows zero-filled.
            chunks_per_row = self.row_bytes // 16
            for i in cutlass.range_constexpr(
                (self.latent_chunks + self.threads - 1) // self.threads
            ):
                chunk = tidx + i * self.threads
                if chunk < self.latent_chunks:
                    gather_row = chunk // chunks_per_row
                    gather_part = chunk % chunks_per_row
                    gather_req = row0 + gather_row
                    token = Int32(0)
                    src_bytes = Int32(0)
                    if gather_req < rows:
                        token = _ld_global_b32(
                            prev_ptr
                            + gather_req.to(cutlass.Int64)
                            * prev_stride.to(cutlass.Int64)
                            * 4
                        )
                        src_bytes = Int32(16)
                    if token < Int32(0):
                        token = Int32(0)
                    if token > embedding_rows - 1:
                        token = embedding_rows - 1
                    _cp_async_16(
                        latent_base
                        + gather_row * self.latent_stride
                        + gather_part * 16,
                        embedding_ptr
                        + token.to(cutlass.Int64) * self.row_bytes
                        + (gather_part * 16).to(cutlass.Int64),
                        src_bytes,
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            acc = []
            for j in cutlass.range_constexpr(self.row_tiles):
                acc.append(
                    (
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                    )
                )
            for c in cutlass.range_constexpr(self.chunks_per_warp):
                b_offset = first_byte + c * 128
                for j in cutlass.range_constexpr(self.row_tiles):
                    b_addr = latent_base + (8 * j + g) * self.latent_stride + b_offset
                    b_words = (_ld_shared_v4(b_addr), _ld_shared_v4(b_addr + 64))
                    for i in cutlass.range_constexpr(4):
                        # k slots (2t, 2t+1), (2t+8, 2t+9) = lane words w, w + 1.
                        a_g, a_g8 = a_words[c][i // 2]
                        b_vec = b_words[i // 2]
                        w = 2 * (i % 2)
                        acc[j] = _mma_m16n8k16(
                            (a_g[w], a_g8[w], a_g[w + 1], a_g8[w + 1]),
                            (b_vec[w], b_vec[w + 1]),
                            acc[j],
                            bf16=self.bf16,
                        )

            # The second warp of each pair hands its half of the dot over.
            if k_part == 1:
                for j in cutlass.range_constexpr(self.row_tiles):
                    for q in cutlass.range_constexpr(4):
                        partial[group_in_cta, 4 * j + q, lane] = acc[j][q]
            cute.arch.barrier()
            if k_part == 0:
                for j in cutlass.range_constexpr(self.row_tiles):
                    for e in cutlass.range_constexpr(2):
                        # acc[j][2h+e]: shard row 16*group+g+8h, request 8j+2t+e.
                        req = row0 + 8 * j + 2 * t + e
                        req_load = req
                        if req >= rows:
                            req_load = rows - 1
                        lane_best = cutlass.Int64(_MARKOV_KEY_NONE)
                        for h in cutlass.range_constexpr(2):
                            column = load_group * 16 + g + 8 * h
                            bias = (
                                acc[j][2 * h + e]
                                + partial[group_in_cta, 4 * j + 2 * h + e, lane]
                            )
                            base = _ld_half_as_f32(
                                base_ptr
                                + (
                                    req_load.to(cutlass.Int64) * self.N
                                    + column.to(cutlass.Int64)
                                )
                                * 2,
                                bf16=self.bf16,
                            )
                            logit = _bias_add_rounded(bias, base, bf16=self.bf16)
                            lane_best = _max_key(
                                lane_best, _argmax_key(logit, vocab_start + column)
                            )
                        if req >= rows:
                            lane_best = cutlass.Int64(_MARKOV_KEY_NONE)
                        if group >= self.groups:
                            lane_best = cutlass.Int64(_MARKOV_KEY_NONE)
                        for s in cutlass.range_constexpr(3):
                            lane_best = _max_key(
                                lane_best,
                                cute.arch.shuffle_sync_bfly(lane_best, offset=4 << s),
                            )
                        if g == 0:
                            cand[group_in_cta, 8 * j + 2 * t + e] = lane_best
            cute.arch.barrier()

            if tidx < self.rows_per_cta:
                cta_best = cand[0, tidx]
                for q in cutlass.range_constexpr(1, _MARKOV_GROUPS_PER_CTA):
                    cta_best = _max_key(cta_best, cand[q, tidx])
                key_row = row0 + tidx
                if key_row < rows:
                    if cta_best != cutlass.Int64(_MARKOV_KEY_NONE):
                        _red_max_s64(keys_ptr + key_row.to(cutlass.Int64) * 8, cta_best)
            # Publish this CTA's keys before counting it done.
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            if tidx == 0:
                done = _atom_add_acq_rel_gpu_u32(ctas_done_ptr, Int32(1))
                is_last = Int32(0)
                if done == grid_x * grid_y - 1:
                    is_last = Int32(1)
                last_cta[0] = is_last
            cute.arch.barrier()
            if last_cta[0] != Int32(0):
                cute.arch.fence_acq_rel_gpu()
                if tidx == 0:
                    _st_relaxed_gpu_u32(ctas_done_ptr, Int32(0))
                if cutlass.const_expr(self.world_size == 1):
                    for own_row in cutlass.range(tidx, rows, self.threads):
                        own_addr = keys_ptr + own_row.to(cutlass.Int64) * 8
                        own_key = _ld_relaxed_gpu_b64(own_addr)
                        _st_relaxed_gpu_b64(own_addr, cutlass.Int64(_MARKOV_KEY_NONE))
                        _, own_winner = _decode_argmax_key(own_key)
                        if own_winner == Int32(0x7FFFFFFF):
                            own_winner = Int32(0)
                        _st_global_b32(
                            out_ptr
                            + own_row.to(cutlass.Int64)
                            * out_stride.to(cutlass.Int64)
                            * 4,
                            own_winner,
                        )
                else:
                    self._exchange(
                        tidx,
                        warp,
                        lane,
                        rows,
                        out_ptr,
                        out_stride,
                        keys_ptr,
                        slot_ptrs,
                        slot_multicast_ptr,
                        round_id_ptr,
                    )

        @cute.jit
        def _exchange(
            self,
            tidx: Int32,
            warp: Int32,
            lane: Int32,
            rows: Int32,
            out_ptr: cutlass.Int64,
            out_stride: Int32,
            keys_ptr: cutlass.Int64,
            slot_ptrs: cutlass.Int64,
            slot_multicast_ptr: cutlass.Int64,
            round_id_ptr: cutlass.Int64,
        ):
            """The distributed argmax's handshake for every row, in one CTA.

            Same slots, band ping-pong and payload as ``DistArgmaxKernel``:
            each row's candidate is multicast to slot ``(row, band, rank)``
            on every rank, each rank reads all ``world_size`` slots of the row
            and clears their flags, and ``round_id`` advances once per launch
            after every row is done.
            """
            round_bit = (ptx_ld_global_u32(round_id_ptr) & Int32(1)).to(cutlass.Int64)
            for post_row in cutlass.range(tidx, rows, self.threads):
                post_addr = keys_ptr + post_row.to(cutlass.Int64) * 8
                post_key = _ld_relaxed_gpu_b64(post_addr)
                _st_relaxed_gpu_b64(post_addr, cutlass.Int64(_MARKOV_KEY_NONE))
                post_value, post_index = _decode_argmax_key(post_key)
                post_slot = (
                    post_row.to(cutlass.Int64) * 2 + round_bit
                ) * self.world_size + self.rank
                ptx_multimem_st_release_u64(
                    slot_multicast_ptr + post_slot * 8,
                    pack_argmax_payload_u64(post_value, post_index),
                )

            local_slots = ptx_ld_acquire_sys_u64(slot_ptrs + self.rank * 8)
            segment = lane // self.peers
            peer = lane % self.peers
            ready = cutlass.Int64(1) << cutlass.Int64(63)
            for first in cutlass.range(0, rows, self.rows_per_pass):
                # Issue every poll of this pass before waiting on any.
                row_ids = []
                actives = []
                addrs = []
                payloads = []
                for u in cutlass.range_constexpr(self.exchange_unroll):
                    row = (
                        first
                        + (u * _MARKOV_WARPS + warp) * self.rows_per_warp_pass
                        + segment
                    )
                    active = Int32(0)
                    if row < rows:
                        if peer < self.world_size:
                            active = Int32(1)
                    addr = (
                        local_slots
                        + (
                            (row.to(cutlass.Int64) * 2 + round_bit) * self.world_size
                            + peer.to(cutlass.Int64)
                        )
                        * 8
                    )
                    row_ids.append(row)
                    actives.append(active)
                    addrs.append(addr)
                    payloads.append(_poll_slot(addr, active))
                for u in cutlass.range_constexpr(self.exchange_unroll):
                    payload = payloads[u]
                    while (payload & ready) == cutlass.Int64(0):
                        payload = _poll_slot(addrs[u], actives[u])
                    _clear_slot_flag(addrs[u], actives[u])
                    peer_value, peer_index = unpack_argmax_payload_u64(payload)
                    peer_key = _argmax_key(peer_value, peer_index)
                    if actives[u] == Int32(0):
                        peer_key = cutlass.Int64(_MARKOV_KEY_NONE)
                    for s in cutlass.range_constexpr(self.peer_steps):
                        peer_key = _max_key(
                            peer_key,
                            cute.arch.shuffle_sync_bfly(peer_key, offset=1 << s),
                        )
                    if peer == 0:
                        if row_ids[u] < rows:
                            _, winner = _decode_argmax_key(peer_key)
                            if winner == Int32(0x7FFFFFFF):
                                winner = Int32(0)
                            _st_global_b32(
                                out_ptr
                                + row_ids[u].to(cutlass.Int64)
                                * out_stride.to(cutlass.Int64)
                                * 4,
                                winner,
                            )
            # Every row has read its band: advance the band once per launch.
            cute.arch.barrier()
            if tidx == 0:
                ptx_atomic_add_acq_rel_sys_u32(round_id_ptr, Int32(1))


_dist_argmax_compile_cache: dict[tuple, object] = {}


@dataclass
class DistArgmaxState:
    group: _dist.ProcessGroup
    rank_in_group: int
    world_size: int
    max_M: int
    dtype: torch.dtype
    device: torch.device
    slot_buffer: torch.Tensor
    slot_handle: object
    round_id_gpu: torch.Tensor
    warps_done_gpu: torch.Tensor
    use_redux: bool
    skip_ping_pong: bool = False


def _build_dist_argmax_state(
    group: _dist.ProcessGroup,
    rank_in_group: int,
    max_M: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    skip_ping_pong: bool = False,
    *,
    strict: bool,
) -> DistArgmaxState | None:
    assert dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), f"distributed argmax supports bf16/fp16/fp32 value dtype; got {dtype}"
    assert _ARCH_SUPPORTED and _CUTE_AVAILABLE, (
        "distributed_argmax requires CuTe DSL on NVIDIA Hopper+/Blackwell; "
        "current platform doesn't qualify."
    )
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    world_size = group.size()
    assert 1 <= world_size <= 32, (
        f"world_size={world_size} unsupported; must be 1..32 (cross-rank "
        f"reduce uses a single warp shuffle)"
    )
    p = current_platform()
    sm = p.arch_version.major * 10 + p.arch_version.minor
    use_redux = 100 <= sm < 120
    slots = _symm_mem.empty(
        (2 * max_M * world_size,),
        dtype=torch.int64,
        device=device,
    )
    slots.zero_()
    round_id_gpu = torch.zeros(1, dtype=torch.int32, device=device)
    warps_done_gpu = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.current_stream().synchronize()
    hdl = _symm_mem.rendezvous(slots, group=group)
    assert hdl.rank == rank_in_group and hdl.world_size == world_size, (
        f"symm-mem handle reports rank={hdl.rank}, world_size={hdl.world_size}, "
        f"but state was constructed with rank_in_group={rank_in_group}, "
        f"world_size={world_size}"
    )
    if not hdl.multicast_ptr:
        if not strict:
            return None
        raise RuntimeError(
            f"distributed_argmax requires CUDA multicast / NVLS, but the "
            f"symm-mem handle on device {device} reports multicast_ptr="
            f"{hdl.multicast_ptr}. The kernel uses multimem.st.release.sys "
            f"which needs NVSwitch + sm_90+ multicast support; non-NVLS "
            f"hardware (PCIe-only, passthrough, etc.) cannot run this op."
        )
    _dist.barrier(group=group, device_ids=[device.index])
    return DistArgmaxState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=world_size,
        max_M=max_M,
        dtype=dtype,
        device=device,
        slot_buffer=slots,
        slot_handle=hdl,
        round_id_gpu=round_id_gpu,
        warps_done_gpu=warps_done_gpu,
        use_redux=use_redux,
        skip_ping_pong=skip_ping_pong,
    )


def create_dist_argmax_state(
    group: _dist.ProcessGroup,
    rank_in_group: int,
    max_M: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    skip_ping_pong: bool = False,
) -> DistArgmaxState:
    """Build the cross-rank argmax state, requiring NVLS multicast.

    Args:
        group: The process group the argmax reduces over.
        rank_in_group: This process's rank within ``group``.
        max_M: Largest row count any call will pass.
        dtype: Value dtype of the logits; bf16, fp16 or fp32.
        device: CUDA device for the symmetric-memory slots; defaults to the
            current device.
        skip_ping_pong: Pin the slot band instead of alternating. Only safe
            when the caller synchronizes across ranks between calls.

    Returns:
        The state to pass to :func:`distributed_argmax`.

    Raises:
        RuntimeError: The group has no CUDA multicast / NVLS support.
    """
    return _build_dist_argmax_state(
        group, rank_in_group, max_M, dtype, device, skip_ping_pong, strict=True
    )


def try_create_dist_argmax_state(
    group: _dist.ProcessGroup,
    rank_in_group: int,
    max_M: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    skip_ping_pong: bool = False,
) -> DistArgmaxState | None:
    """Build the cross-rank argmax state, or report that NVLS is missing.

    Multicast support cannot be queried before the symmetric-memory
    rendezvous, so callers that would rather fall back than fail ask here
    instead of catching an exception. Every other precondition still asserts.

    Args:
        group: The process group the argmax reduces over.
        rank_in_group: This process's rank within ``group``.
        max_M: Largest row count any call will pass.
        dtype: Value dtype of the logits; bf16, fp16 or fp32.
        device: CUDA device for the symmetric-memory slots; defaults to the
            current device.
        skip_ping_pong: Pin the slot band instead of alternating. Only safe
            when the caller synchronizes across ranks between calls.

    Returns:
        The state, or ``None`` when the group has no multicast support. The
        verdict follows from the group's fabric, so every rank agrees on it.
    """
    return _build_dist_argmax_state(
        group, rank_in_group, max_M, dtype, device, skip_ping_pong, strict=False
    )


def _dist_argmax_validate_inputs(
    state: DistArgmaxState,
    logits: torch.Tensor,
    out_max: torch.Tensor | None,
    out_idx: torch.Tensor | None,
) -> tuple[int, int]:
    assert (
        logits.dim() == 2
    ), f"logits must be 2D (M, N); got shape {tuple(logits.shape)}"
    M, N = logits.shape
    assert (
        logits.device == state.device
    ), f"logits.device={logits.device} != state.device={state.device}"
    assert (
        logits.dtype == state.dtype
    ), f"logits.dtype={logits.dtype} != state.dtype={state.dtype}"
    assert (
        N >= _MIN_VOCAB_SIZE
    ), f"per-rank vocab N={N} below kernel floor {_MIN_VOCAB_SIZE}"
    assert (
        N % _VOCAB_SIZE_ALIGNMENT == 0
    ), f"per-rank vocab N={N} not aligned to {_VOCAB_SIZE_ALIGNMENT}"
    if out_max is not None:
        assert out_max.shape == (M,) and out_max.device == logits.device, (
            f"out_max must be shape ({M},) on logits.device; got "
            f"shape={tuple(out_max.shape)} device={out_max.device}"
        )
    if out_idx is not None:
        _validate_argmax_out(logits, out_idx)
    return M, N


def _dist_argmax_invoke_kernel(
    state: DistArgmaxState,
    logits: torch.Tensor,
    out_max: torch.Tensor,
    out_idx: torch.Tensor,
) -> None:
    dtype = torch2cute_dtype_map[logits.dtype]
    x_tensor = _convert_to_cute(logits)
    max_tensor = _convert_to_cute_1d(out_max)
    idx_tensor = _convert_to_cute_1d(out_idx)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    N = logits.shape[1]
    compile_key = (
        dtype,
        N,
        state.use_redux,
        out_max.dtype,
        out_idx.dtype,
        state.world_size,
        state.rank_in_group,
        state.skip_ping_pong,
    )
    round_id_ptr = state.round_id_gpu.data_ptr()
    warps_done_ptr = state.warps_done_gpu.data_ptr()
    compiled = _dist_argmax_compile_cache.get(compile_key)
    if compiled is None:
        kernel = DistArgmaxKernel(
            dtype,
            N,
            use_redux=state.use_redux,
            world_size=state.world_size,
            rank=state.rank_in_group,
            skip_ping_pong=state.skip_ping_pong,
        )
        compiled = cute.compile(
            kernel,
            x_tensor,
            max_tensor,
            idx_tensor,
            stream,
            state.slot_handle.buffer_ptrs_dev,
            state.slot_handle.multicast_ptr,
            round_id_ptr,
            warps_done_ptr,
        )
        _dist_argmax_compile_cache[compile_key] = compiled

    compiled(
        x_tensor,
        max_tensor,
        idx_tensor,
        stream,
        state.slot_handle.buffer_ptrs_dev,
        state.slot_handle.multicast_ptr,
        round_id_ptr,
        warps_done_ptr,
    )


def distributed_argmax(
    state: DistArgmaxState,
    logits: torch.Tensor,
    *,
    out_max: torch.Tensor | None = None,
    out_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed argmax over the vocab dim, across ``state.group``.

    Contract: every rank in ``state.group`` calls this in lockstep with
    matching ``M, N``, same ``state``, and the same call ordering — the
    cross-rank exchange is collective and slot bands are reused across
    calls. Do not share one ``DistArgmaxState`` across concurrent streams.
    """
    assert logits.is_contiguous(), (
        "distributed_argmax wraps its input as compact rows; a strided view "
        "would be read with the wrong row pitch"
    )
    M, N = _dist_argmax_validate_inputs(state, logits, out_max, out_idx)
    device = logits.device

    if state.world_size == 1:
        max_vals, idx_vals = logits.max(dim=-1)
        if out_max is not None:
            out_max.copy_(max_vals.to(out_max.dtype))
            max_vals = out_max
        if out_idx is not None:
            out_idx.copy_(idx_vals)
            idx_vals = out_idx
        return max_vals, idx_vals

    assert M <= state.max_M, f"batch size {M} exceeds state.max_M={state.max_M}"
    assert (
        state.world_size * N < 0x7FFFFFFF
    ), f"world_size*N = {state.world_size * N} overflows the 31-bit slot idx"

    if out_max is None:
        out_max = torch.empty((M,), dtype=logits.dtype, device=device)
    if out_idx is None:
        out_idx = torch.empty((M,), dtype=torch.int64, device=device)
    _dist_argmax_invoke_kernel(state, logits, out_max, out_idx)
    return out_max, out_idx


# Fused greedy DSpark Markov step — kernel + operator sharing the slots above.


@dataclass
class MarkovArgmaxWorkspace:
    """Scratch and compiled kernels of :func:`markov_argmax_step`.

    ``keys`` holds one packed candidate per row and ``ctas_done`` counts the
    CTAs of a launch. Every launch leaves both as it found them (empty keys,
    zero), so a captured step replays without host-side resets.
    """

    state: DistArgmaxState | None
    device: torch.device
    max_rows: int
    vocab_per_rank: int
    rank_dim: int
    dtype: torch.dtype
    keys: torch.Tensor
    ctas_done: torch.Tensor
    kernels: dict[int, object]


def supports_markov_argmax_shape(
    vocab_per_rank: int, rank_dim: int, dtype: torch.dtype
) -> bool:
    """Whether :func:`markov_argmax_step` covers a shard of this geometry.

    Args:
        vocab_per_rank: Shard columns of the base logits (projection rows).
        rank_dim: Markov rank: width of the embedding and projection rows.
        dtype: Dtype of the base logits, embedding and projection.

    Returns:
        True when the kernel covers the geometry. The verdict follows from
        the model config alone, so every rank reaches the same one.
    """
    return (
        dtype in (torch.bfloat16, torch.float16)
        and 0 < vocab_per_rank < 0x7FFFFFFF
        and vocab_per_rank % 16 == 0
        and rank_dim % 128 == 0
        and 128 <= rank_dim <= 512
    )


def _markov_row_tiles(rows: int) -> int:
    """MMA row tiles (8 rows each) one CTA covers for a ``rows``-row step."""
    tiles = 1
    while tiles < _MARKOV_MAX_ROW_TILES and 8 * tiles < rows:
        tiles *= 2
    return tiles


_markov_argmax_compile_cache: dict[tuple, object] = {}


def _compile_markov_argmax(
    dtype: torch.dtype,
    vocab_per_rank: int,
    rank_dim: int,
    row_tiles: int,
    world_size: int,
    rank: int,
):
    """Compile one specialization; pointers and sizes are runtime arguments."""
    kernel = MarkovArgmaxStepKernel(
        torch2cute_dtype_map[dtype],
        vocab_per_rank,
        rank_dim,
        row_tiles,
        world_size,
        rank,
    )
    # Placeholder values: every argument below is dynamic in the kernel.
    return cute.compile(
        kernel, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, cuda.CUstream(0)
    )


def create_markov_argmax_workspace(
    state: DistArgmaxState | None,
    *,
    device: torch.device,
    max_rows: int,
    vocab_per_rank: int,
    rank_dim: int,
    dtype: torch.dtype,
) -> MarkovArgmaxWorkspace:
    """Allocate the step's scratch and compile every kernel it can select.

    Call before CUDA graph capture: neither allocation nor compilation can be
    captured, and the step picks its kernel by row count.

    Args:
        state: Distributed-argmax state whose slots the step exchanges
            candidates through, or None when one rank holds the whole
            vocabulary.
        device: CUDA device of the logits; must be ``state.device`` when a
            state is given.
        max_rows: Most rows any step will pass.
        vocab_per_rank: Columns of this rank's shard.
        rank_dim: Markov rank.
        dtype: Dtype of the base logits, embedding and projection.

    Returns:
        The workspace to pass to :func:`markov_argmax_step`.

    Raises:
        RuntimeError: CuTe DSL is unavailable here, or a capture is running.
        ValueError: The geometry, group or row bound is unsupported.
    """
    if not _CUTE_AVAILABLE:
        raise RuntimeError("markov_argmax_step requires CuTe DSL on NVIDIA sm_90+")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Create the Markov argmax workspace before graph capture")
    if not supports_markov_argmax_shape(vocab_per_rank, rank_dim, dtype):
        raise ValueError(
            f"unsupported Markov argmax geometry: vocab_per_rank={vocab_per_rank}, "
            f"rank_dim={rank_dim}, dtype={dtype}"
        )
    world_size = 1 if state is None else state.world_size
    rank = 0 if state is None else state.rank_in_group
    if state is not None:
        if world_size < 2:
            raise ValueError("a distributed-argmax state needs at least two ranks")
        if state.skip_ping_pong:
            raise ValueError(
                "the step alternates slot bands; skip_ping_pong is unsupported"
            )
        if device != state.device:
            raise ValueError(f"device {device} differs from state {state.device}")
        if max_rows > state.max_M:
            raise ValueError(f"max_rows {max_rows} exceeds state.max_M {state.max_M}")
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    kernels = {}
    for rows in range(1, max_rows + 1):
        tiles = _markov_row_tiles(rows)
        key = (dtype, vocab_per_rank, rank_dim, tiles, world_size, rank)
        if key not in _markov_argmax_compile_cache:
            _markov_argmax_compile_cache[key] = _compile_markov_argmax(*key)
        kernels[tiles] = _markov_argmax_compile_cache[key]
    return MarkovArgmaxWorkspace(
        state=state,
        device=device,
        max_rows=max_rows,
        vocab_per_rank=vocab_per_rank,
        rank_dim=rank_dim,
        dtype=dtype,
        keys=torch.full(
            (max_rows,), _MARKOV_KEY_NONE, dtype=torch.int64, device=device
        ),
        ctas_done=torch.zeros(1, dtype=torch.int32, device=device),
        kernels=kernels,
    )


def markov_argmax_step(
    workspace: MarkovArgmaxWorkspace,
    base_logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
    vocab_start: int,
    out: torch.Tensor,
) -> None:
    """One greedy DSpark Markov step in one kernel launch.

    For row ``m``, with ``p = clamp(prev_tokens[m], 0, embedding.shape[0] -
    1)``: ``bias = embedding[p] @ projection.T`` accumulated in FP32 and
    rounded to the logits dtype, ``logits = base_logits[m] + bias`` rounded
    to the logits dtype, and ``out[m]`` receives the global token id
    (``vocab_start`` + shard column) of the largest logit over every rank of
    the workspace's group. Ties go to the lowest token id; a row whose
    logits never beat -inf (all -inf or NaN) yields 0. With a state, every
    rank calls in lockstep with the same row count, like
    :func:`distributed_argmax`, whose slots and band order it shares.

    Args:
        workspace: From :func:`create_markov_argmax_workspace`.
        base_logits: ``[rows, vocab_per_rank]`` contiguous base logits of
            this rank's shard.
        prev_tokens: ``[rows]`` int32 previous tokens, any stride.
        embedding: ``[vocab, rank_dim]`` contiguous Markov embedding.
        projection: ``[vocab_per_rank, rank_dim]`` contiguous projection rows
            of this rank's shard.
        vocab_start: Global token id of shard column 0.
        out: ``[rows]`` int32 destination, any stride. It may be another
            column of the table ``prev_tokens`` reads.

    Returns:
        None. ``out`` is written in place.

    Raises:
        ValueError: A tensor's shape, dtype, layout or device does not match
            the workspace, or ``rows`` exceeds ``workspace.max_rows``.
    """
    if base_logits.dim() != 2 or not base_logits.is_contiguous():
        raise ValueError("base_logits must be a contiguous [rows, vocab] tensor")
    rows, vocab = base_logits.shape
    expected = {
        "base_logits": (base_logits, (rows, workspace.vocab_per_rank), workspace.dtype),
        "embedding": (
            embedding,
            (embedding.shape[0], workspace.rank_dim),
            workspace.dtype,
        ),
        "projection": (
            projection,
            (workspace.vocab_per_rank, workspace.rank_dim),
            workspace.dtype,
        ),
        "prev_tokens": (prev_tokens, (rows,), torch.int32),
        "out": (out, (rows,), torch.int32),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} must be {dtype} of shape {shape}; got {tensor.dtype} "
                f"of shape {tuple(tensor.shape)}"
            )
        if tensor.device != workspace.device:
            raise ValueError(f"{name} is on {tensor.device}, not {workspace.device}")
    if not (embedding.is_contiguous() and projection.is_contiguous()):
        raise ValueError("embedding and projection must be contiguous")
    if embedding.data_ptr() % 16 or projection.data_ptr() % 16:
        raise ValueError("embedding and projection rows are read in 16-byte words")
    if embedding.shape[0] < 1:
        raise ValueError("embedding must have at least one row")
    if rows > workspace.max_rows:
        raise ValueError(f"rows {rows} exceeds workspace.max_rows {workspace.max_rows}")
    if not 0 <= vocab_start < 0x7FFFFFFF - vocab:
        raise ValueError(f"vocab_start {vocab_start} overflows the 31-bit token id")
    if rows == 0:
        return
    state = workspace.state
    workspace.kernels[_markov_row_tiles(rows)](
        base_logits.data_ptr(),
        prev_tokens.data_ptr(),
        prev_tokens.stride(0),
        out.data_ptr(),
        out.stride(0),
        embedding.data_ptr(),
        embedding.shape[0],
        projection.data_ptr(),
        vocab_start,
        rows,
        workspace.keys.data_ptr(),
        workspace.ctas_done.data_ptr(),
        0 if state is None else state.slot_handle.buffer_ptrs_dev,
        0 if state is None else state.slot_handle.multicast_ptr,
        0 if state is None else state.round_id_gpu.data_ptr(),
        cuda.CUstream(torch.cuda.current_stream(workspace.device).cuda_stream),
    )
