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


"""Collective, explicitly owned symmetric outputs for large-M up projection."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication.triton import (
    _alloc_symm,
    blockwise_barrier,
    sync_threads,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

TP = 8
HIDDEN = 7168
LATENT = 3584
SHARD = HIDDEN // TP


@dataclass(frozen=True)
class SymmetricUpProjectionTuning:
    """MMA tile and cluster geometry; compilation happens before capture.

    Args:
        tile_m: MMA M, including both CTAs for a two-CTA instruction.
        tile_n: MMA N. Divisors of 896 avoid padding at the owner boundary.
        two_cta: Use Blackwell tcgen05 CTA-group-two instructions.
        cluster_m: Cluster size in M, divisible by two for two-CTA MMA.
        cluster_n: Cluster size in N.
        enable_pdl: Allow dependent kernels to launch their prologues early.
        store_kind: TMA bulk store or register-to-global multimem vectors.
    """

    tile_m: int
    tile_n: int
    two_cta: bool
    cluster_m: int
    cluster_n: int
    enable_pdl: bool
    store_kind: str

    def validate(self) -> None:
        if self.store_kind not in ("tma", "multimem"):
            raise ValueError("store_kind must be tma or multimem")
        if self.tile_m not in ((128, 256) if self.two_cta else (64, 128)):
            raise ValueError("invalid MMA M for selected CTA group")
        if self.tile_n not in (32, 64, 128, 256):
            raise ValueError("unsupported MMA N")
        cluster = (self.cluster_m, self.cluster_n)
        if any(x <= 0 or x & (x - 1) for x in cluster):
            raise ValueError("cluster dimensions must be positive powers of two")
        if self.cluster_m * self.cluster_n > 16:
            raise ValueError("cluster size exceeds 16")
        if self.two_cta and self.cluster_m % 2:
            raise ValueError("two-CTA MMA requires an even cluster M")


@dataclass(frozen=True)
class SymmetricUpProjectionOverlapTuning(SymmetricUpProjectionTuning):
    """Explicit experimental output-pipeline controls; old tuning stays frozen.

    Args:
        c_stages: Output SMEM ring slots, zero for the established auto policy.
        release_acc_early: Release TMEM after its final read, before output waits.
        acquire_before_store: Wait for a reusable C slot immediately before writing.
        prefetch_acc_tile: Drain all accumulator subtiles before any output wait.
        epilogue_m: Output tile rows, zero together with epilogue_n for auto.
        epilogue_n: Output tile columns; controls TMA store transaction width.
        addend_cache_policy: Explicit no_allocate or normal L1 load policy.
        paired_addend_loads: Issue shared and residual vector loads as a pair.
        addend_stages: Zero for register loads, or one/two coalesced async SMEM stages.
    """

    c_stages: int
    release_acc_early: bool
    acquire_before_store: bool
    prefetch_acc_tile: bool
    epilogue_m: int
    epilogue_n: int
    addend_cache_policy: str
    paired_addend_loads: bool
    addend_stages: int

    def validate(self) -> None:
        super().validate()
        if self.store_kind != "tma":
            raise ValueError("overlap experiment requires TMA stores")
        if self.c_stages not in (0, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError("invalid output ring depth")
        if any(type(x) is not int for x in (self.epilogue_m, self.epilogue_n)):
            raise TypeError("epilogue dimensions must be explicit integers")
        if (self.epilogue_m, self.epilogue_n) != (0, 0) and (
            self.epilogue_m not in (32, 64, 128)
            or self.epilogue_n not in (32, 64, 128, 256)
        ):
            raise ValueError("invalid explicit epilogue tile")
        if self.addend_cache_policy not in ("no_allocate", "normal"):
            raise ValueError("invalid addend L1 cache policy")
        if type(self.addend_stages) is not int or self.addend_stages not in (0, 1, 2):
            raise ValueError("addend_stages must be the explicit integer 0, 1 or 2")
        if self.addend_stages:
            cta_m = self.tile_m // (2 if self.two_cta else 1)
            if (
                (cta_m, self.tile_n) != (128, 128)
                or (self.epilogue_m, self.epilogue_n) != (128, 64)
                or self.addend_cache_policy != "no_allocate"
                or self.paired_addend_loads
            ):
                raise ValueError(
                    "coalesced addends require CTA128x128/epi128x64 and unpaired no_allocate"
                )
        if any(
            type(x) is not bool
            for x in (
                self.release_acc_early,
                self.acquire_before_store,
                self.prefetch_acc_tile,
                self.paired_addend_loads,
            )
        ):
            raise ValueError("pipeline switches must be explicit booleans")


@triton.jit
def _fence_proxy_alias():
    tl.inline_asm_elementwise(
        "fence.proxy.alias;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _rank_barrier_kernel(
    signal_pad_ptrs,
    RANK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # One CTA: no per-producer-CTA rendezvous or full-output memory sweep.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    _fence_proxy_alias()
    blockwise_barrier(signal_pad_ptrs, None, RANK, 8, sem="acq_rel")
    sync_threads()
    _fence_proxy_alias()


def _vote(group: dist.ProcessGroup, payload: Any, error: str | None) -> None:
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, (payload, error), group=group)
    failures = [f"rank {i}: {entry[1]}" for i, entry in enumerate(records) if entry[1]]
    if failures or any(entry[0] != records[0][0] for entry in records):
        raise RuntimeError(f"collective up-projection validation failed: {records}")


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device != rhs.device or not lhs.numel() or not rhs.numel():
        return False

    # Inputs may be owner-column slices with full-hidden row strides.
    def end(t: torch.Tensor) -> int:
        span = sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
        return t.data_ptr() + span * t.element_size()

    return lhs.data_ptr() < end(rhs) and rhs.data_ptr() < end(lhs)


@dataclass(eq=False)
class SymmetricUpProjectionOutput:
    """A collectively allocated output owned by its caller.

    tensor is contiguous BF16 [M,7168] and is the actual multicast target.
    Keep this object alive for every graph that refers to it. A call overwrites
    its contents; earlier results persist only when they use another output.
    Allocate one slot for each simultaneously live layer/graph output. Reuse
    requires every rank's preceding consumers to complete in the same stream.
    """

    tensor: torch.Tensor
    handle: Any
    group: dist.ProcessGroup
    rank: int


def allocate_symmetric_up_projection_output(
    group: dist.ProcessGroup,
    num_tokens: int,
    *,
    device: torch.device | None,
) -> SymmetricUpProjectionOutput:
    """Collectively allocate one final output before capture.

    Args:
        group: Rank-consistent eight-GPU process group with NVLS multicast.
        num_tokens: Equal on every rank, in [256,8192].
        device: Local CUDA device, defaulting to the current device.

    Returns:
        Caller-owned output and rendezvous handle, separate from RS scratch.
    """
    error = None
    device = device or torch.device("cuda", torch.cuda.current_device())
    try:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("symmetric output allocation is forbidden during capture")
        if dist.get_world_size(group) != TP or not 256 <= num_tokens <= 8192:
            raise ValueError("requires TP8 and 256 <= M <= 8192")
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise ValueError("requires Blackwell SM100/SM103")
    except Exception as exc:
        error = str(exc)
    _vote(group, num_tokens, error)
    tensor, handle = _alloc_symm((num_tokens, HIDDEN), torch.bfloat16, device, group)
    rank = dist.get_rank(group)
    error = None
    if not handle.multicast_ptr or handle.rank != rank or handle.world_size != TP:
        error = "symmetric handle has no TP8 NVLS multicast mapping"
    _vote(group, num_tokens, error)
    return SymmetricUpProjectionOutput(tensor, handle, group, rank)


class BoundSymmetricUpProjection:
    """A precompiled up projection bound to tensors and one symmetric output.

    Prepare collectively outside capture, then update tensor contents in place.
    Calling the plan computes
    BF16(BF16(FP32(A @ W.T) + FP32(shared)) + FP32(residual_owner))
    and gathers owner shards directly into output.tensor. Its entry barrier
    protects remote readers of a previous use; its exit barrier makes all
    multicast stores visible before subsequent same-stream consumers.

    This is an out= API, not a fresh-allocation API. Cross-stream use, concurrent
    calls, overlapping inputs/output, and rank-local residuals are unsupported.
    """

    @classmethod
    def prepare(
        cls,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
        residual: torch.Tensor,
        output: SymmetricUpProjectionOutput,
        *,
        residual_is_replicated: bool,
        tuning: SymmetricUpProjectionTuning,
        skip_entry_sync: bool,
    ) -> BoundSymmetricUpProjection:
        """Validate and JIT-bind exact shapes before graph capture.

        Args:
            latent: Replicated BF16 [M,3584] normalized routed input.
            weight: Local contiguous BF16 [896,3584] up-projection weight.
            shared_shard: Contiguous BF16 [M,896] shared ReduceScatter result.
            residual: Bitwise-replicated BF16 [M,7168] residual.
            output: Distinct, caller-owned symmetric output for these tensors.
            residual_is_replicated: Must explicitly be True on every rank.
            tuning: Rank-identical MMA/cluster/PDL settings.
            skip_entry_sync: Caller guarantees a preceding all-rank barrier
                after every previous output consumer, such as the shared RS
                exit in the strict same-stream full-tail pipeline.

        Returns:
            A callable plan whose output aliases exactly output.tensor.
        """
        m = output.tensor.shape[0]
        error = None
        try:
            tuning.validate()
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare must run before capture")
            if residual_is_replicated is not True:
                raise ValueError("residual_is_replicated=True is required")
            if output.rank != dist.get_rank(output.group):
                raise ValueError("output rank disagrees with process group")
            for tensor, shape in (
                (latent, (m, LATENT)),
                (weight, (SHARD, LATENT)),
                (shared_shard, (m, SHARD)),
                (residual, (m, HIDDEN)),
                (output.tensor, (m, HIDDEN)),
            ):
                if (
                    tensor.shape != shape
                    or tensor.dtype != torch.bfloat16
                    or tensor.device != output.tensor.device
                    or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16
                ):
                    raise ValueError(f"invalid input layout, expected BF16 {shape}")
            if any(
                _overlaps(output.tensor, x)
                for x in (latent, weight, shared_shard, residual)
            ):
                raise ValueError("output overlaps an input")
        except Exception as exc:
            error = str(exc)
        _vote(output.group, (m, tuning, residual_is_replicated, skip_entry_sync), error)
        plan = cls()
        plan.output = output
        plan.tuning = tuning
        plan.skip_entry_sync = skip_entry_sync
        plan.inputs = (latent, weight, shared_shard, residual)
        error = None
        try:
            plan._compile()
            _rank_barrier_kernel.warmup(
                output.handle.signal_pad_ptrs_dev,
                RANK=output.rank,
                ENABLE_PDL=tuning.enable_pdl,
                num_warps=4,
                grid=(1,),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _vote(output.group, (m, tuning), error)
        # Compile the tiny barrier without any distributed device rendezvous.
        # Actual warm launch follows once every rank's compiler has succeeded.
        plan._barrier()
        torch.cuda.synchronize(output.tensor.device)
        return plan

    def _compile(self) -> None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.runtime import from_dlpack
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
            CUDAGraphCompatibleWrapper,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.gemm import (
            SymmetricUpProjectionGemm,
        )

        self._cutlass = cutlass
        self._cuda = cuda
        latent, weight, shared, residual = self.inputs
        owner = slice(self.output.rank * SHARD, (self.output.rank + 1) * SHARD)
        tensors = (
            latent,
            weight,
            self.output.tensor[:, owner],
            shared,
            residual[:, owner],
        )
        self._views = tuple(x.unsqueeze(-1) for x in tensors)
        self._cute_args = tuple(
            from_dlpack(CUDAGraphCompatibleWrapper(x.detach()), assumed_align=16)
            for x in self._views
        )
        with torch.cuda.device(latent.device):
            t = self.tuning
            self.kernel = SymmetricUpProjectionGemm(
                cutlass.Float32,
                t.two_cta,
                (t.tile_m, t.tile_n),
                (t.cluster_m, t.cluster_n),
                True,
                t.enable_pdl,
                t.store_kind,
            )
            if isinstance(t, SymmetricUpProjectionOverlapTuning):
                self.kernel.experimental_fused_n64 = (
                    getattr(t, "experimental_fused_n64", False) is True
                )
                self.kernel.configure_output_pipeline(
                    t.c_stages,
                    t.release_acc_early,
                    t.acquire_before_store,
                    t.prefetch_acc_tile,
                )
                self.kernel.configure_epilogue_tile(t.epilogue_m, t.epilogue_n)
                self.kernel.configure_addend_loads(
                    t.addend_cache_policy, t.paired_addend_loads
                )
                self.kernel.configure_addend_pipeline(t.addend_stages)
            self._configure_diagnostics()
            self.max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
                t.cluster_m * t.cluster_n
            )
            self.max_active_clusters = self._select_max_active_clusters(
                self.max_active_clusters
            )
            compile_options = {}
            if isinstance(t, SymmetricUpProjectionOverlapTuning):
                self._compile_dump = tempfile.TemporaryDirectory(
                    prefix=f"up-overlap-r{self.output.rank}-"
                )
                compile_options["options"] = (
                    f"--dump-dir={self._compile_dump.name} --keep-ptx --keep-cubin"
                )
            # Host-side descriptors are bound once; capture never enters DLPack/JIT.
            self.compiled = cute.compile(
                self.kernel,
                *self._cute_args,
                cutlass.Int64(
                    self.output.handle.multicast_ptr + self.output.rank * SHARD * 2
                ),
                cutlass.Boolean(True),
                self.max_active_clusters,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                **compile_options,
            )

    def _select_max_active_clusters(self, hardware_capacity: int) -> int:
        """Retain the hardware default; explicit experiments may lower it before JIT."""
        return hardware_capacity

    def _configure_diagnostics(self) -> None:
        """Host-only diagnostic hook; normal plans add no instrumentation."""

    def _barrier(self) -> None:
        kwargs = {"launch_pdl": True} if self.tuning.enable_pdl else {}
        _rank_barrier_kernel[(1,)](
            self.output.handle.signal_pad_ptrs_dev,
            RANK=self.output.rank,
            ENABLE_PDL=self.tuning.enable_pdl,
            num_warps=4,
            **kwargs,
        )

    def producer_only(self, *, multicast: bool) -> None:
        """Launch only GEMM for diagnostics; callers must synchronize all ranks.

        Without multicast, only this rank's output slice is written. No output
        reuse or cross-rank publication guarantees are supplied by this method.
        """
        address = (
            int(self.output.handle.multicast_ptr)
            if multicast
            else self.output.tensor.data_ptr()
        ) + self.output.rank * SHARD * 2
        self.compiled(
            *self._cute_args,
            self._cutlass.Int64(address),
            self._cutlass.Boolean(multicast),
            self._cuda.CUstream(
                torch.cuda.current_stream(self.output.tensor.device).cuda_stream
            ),
        )

    def __call__(self) -> torch.Tensor:
        return symmetric_up_projection(self)


@register_kernel(
    "communication",
    "symmetric_up_projection",
    name="cutedsl_symmetric_up_projection",
    features={"mnnvl", "multicast_gemm", "cuda_graph", "symmetric_output"},
    solution="cutedsl",
    signatures=frozenset(
        {
            format_signature(
                latent=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
                shared=dense_tensor_format(torch.bfloat16),
                residual=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    priority=Priority.SPECIALIZED,
    tags={"blackwell", "throughput", "experimental"},
)
def symmetric_up_projection(plan: BoundSymmetricUpProjection) -> torch.Tensor:
    """Run a prepared plan and return its explicitly owned symmetric output."""
    if not plan.skip_entry_sync:
        plan._barrier()
    plan.producer_only(multicast=True)
    plan._barrier()
    return plan.output.tensor
