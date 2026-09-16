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

"""Experimental raw-shared reduction inside up GEMM; no runtime registration."""

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.fused_rs_workspace import (
    SharedRsWorkspace,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    BoundSymmetricUpProjection,
    SymmetricUpProjectionOutput,
    SymmetricUpProjectionOverlapTuning,
    _overlaps,
    _rank_barrier_kernel,
    _vote,
)


@dataclass(frozen=True)
class FusedRsUpProjectionTuning(SymmetricUpProjectionOverlapTuning):
    """Fused-only request grouping; ordinary overlap tuning keeps its old schema.

    Args:
        reduce_vectors: Independent BF16x8 NVLS requests issued before consuming
            their results, one (frozen), two, four or eight. Groups above one
            currently require one addend stage for an isolated experiment.
    """

    reduce_vectors: int

    def validate(self):
        super().validate()
        if type(self.reduce_vectors) is not int or self.reduce_vectors not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("reduce_vectors must be the explicit integer 1, 2, 4 or 8")
        if self.reduce_vectors > 1 and self.addend_stages != 1:
            raise ValueError("grouped reduction currently requires one addend stage")


@dataclass(frozen=True)
class FusedRsN64UpProjectionTuning(FusedRsUpProjectionTuning):
    """Explicit M4096-only N64 experiment; old spec parsers never construct it."""

    @property
    def experimental_fused_n64(self):
        return True

    def validate(self):
        if (
            self.tile_m != 256
            or self.tile_n != 64
            or not self.two_cta
            or (self.cluster_m, self.cluster_n) != (2, 1)
            or (self.epilogue_m, self.epilogue_n) != (128, 64)
            or self.c_stages != 0
            or self.prefetch_acc_tile
            or self.addend_stages != 1
            or self.reduce_vectors != 4
            or self.enable_pdl
            or self.store_kind != "tma"
            or not self.release_acc_early
            or not self.acquire_before_store
            or self.addend_cache_policy != "no_allocate"
            or self.paired_addend_loads
        ):
            raise ValueError(
                "N64 experiment fixes all controls except the explicit N64 tile"
            )
        # Reuse all established scalar/type/control validation on its canonical
        # N128 counterpart; only this subclass authorizes the N64 layout contract.
        FusedRsUpProjectionTuning.validate(replace(self, tile_n=128))


def validate_fused_tuning(tuning):
    """Reject unimplemented geometry and arithmetic before distributed launch."""
    if not isinstance(tuning, SymmetricUpProjectionOverlapTuning):
        raise ValueError("fused RS requires explicit overlap tuning")
    tuning.validate()
    n64 = getattr(tuning, "experimental_fused_n64", False) is True
    if (
        tuning.tile_m != 256
        or tuning.tile_n != (64 if n64 else 128)
        or not tuning.two_cta
        or (tuning.cluster_m, tuning.cluster_n) != (2, 1)
        or tuning.enable_pdl
        or tuning.store_kind != "tma"
        or not tuning.release_acc_early
        or not tuning.acquire_before_store
        or (tuning.epilogue_m, tuning.epilogue_n) != (128, 64)
        or tuning.addend_cache_policy != "no_allocate"
        or tuning.paired_addend_loads
        or tuning.addend_stages not in (1, 2)
    ):
        raise ValueError("fused RS only supports canonical wide one/two-stage tuning")


class BoundFusedRsUpProjection(BoundSymmetricUpProjection):
    """One-stream raw symmetric partials → reduced/additive up GEMM → final AG.

    The input workspace and output must remain alive and non-overlapping until
    all captured graphs finish. Independent concurrent instances need independent
    workspaces. Sequential layers can reuse raw storage only after this plan's
    all-rank exit. There is no skip-publication option or standalone RS launch.
    """

    @classmethod
    def prepare_fused(
        cls,
        latent: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        workspace: SharedRsWorkspace,
        output: SymmetricUpProjectionOutput,
        *,
        residual_is_replicated: bool,
        tuning: SymmetricUpProjectionOverlapTuning,
    ):
        """Validate and bind outside capture, without materializing an RS shard.

        Args:
            latent: Replicated contiguous CUDA BF16 [M,3584] routed result.
            weight: Owner-local contiguous CUDA BF16 [896,3584] weight.
            residual: Replicated contiguous CUDA BF16 [M,7168] residual.
            workspace: Dedicated raw symmetric [max_M,7168] input allocation.
            output: Separate final symmetric [M,7168] output allocation.
            residual_is_replicated: Explicit True, matching the normal up contract.
            tuning: Canonical wide epilogue with one/two addend stages and PDL off.

        Returns:
            Plan exposing input_view for a copy or real producer out=, and a
            callable that always publishes raw input and completes all ranks.
        """
        state = workspace.state
        m = output.tensor.shape[0]
        raw, dummy = state.comm_buff[:m], state.local_buff[:m]
        error = None
        try:
            validate_fused_tuning(tuning)
            if getattr(tuning, "experimental_fused_n64", False) and m != 4096:
                raise ValueError("N64 fused experiment is restricted to exact M4096")
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("fused RS prepare must precede capture")
            if (
                not 256 <= m <= min(state.max_token_num, 8192)
                or state.group is not output.group
                or state.rank_in_group != output.rank
                or output.rank != dist.get_rank(output.group)
                or dist.get_world_size(output.group) != 8
                or not state.symm_mem_hdl.multicast_ptr
                or residual_is_replicated is not True
            ):
                raise ValueError(
                    "requires matching TP8 workspace/output and replicated residual"
                )
            for tensor, shape in (
                (raw, (m, 7168)),
                (dummy, (m, 896)),
                (latent, (m, 3584)),
                (weight, (896, 3584)),
                (residual, (m, 7168)),
                (output.tensor, (m, 7168)),
            ):
                if (
                    tensor.shape != shape
                    or tensor.dtype != torch.bfloat16
                    or tensor.device != output.tensor.device
                    or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16
                ):
                    raise ValueError(
                        f"invalid fused input layout; expected BF16 {shape}"
                    )
            physical = state.symm_mem_hdl.get_buffer(
                state.rank_in_group,
                (state.max_token_num, 7168),
                torch.bfloat16,
                storage_offset=0,
            )
            if physical.data_ptr() != state.comm_buff.data_ptr():
                raise ValueError("raw input does not match its symmetric mapping")
            for protected in (latent, weight, residual, dummy, output.tensor):
                if _overlaps(raw, protected):
                    raise ValueError("raw symmetric input aliases protected storage")
            if any(
                _overlaps(output.tensor, x) for x in (latent, weight, residual, dummy)
            ):
                raise ValueError("final output aliases an input")
        except Exception as exc:
            error = str(exc)
        identity = (m, tuning, residual_is_replicated, "fused-shared-rs-v1")
        _vote(output.group, identity, error)
        plan = cls()
        plan.workspace = workspace
        plan.input_view = raw
        plan.output = output
        plan.tuning = tuning
        plan.skip_entry_sync = False
        # The third tensor supplies only layout metadata. The specialized
        # kernel never reads or writes this legacy shard allocation.
        plan.inputs = (latent, weight, dummy, residual)
        error = None
        try:
            plan._compile()
            for signals in (
                state.symm_mem_hdl.signal_pad_ptrs_dev,
                output.handle.signal_pad_ptrs_dev,
            ):
                _rank_barrier_kernel.warmup(
                    signals,
                    RANK=output.rank,
                    ENABLE_PDL=False,
                    num_warps=4,
                    grid=(1,),
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _vote(output.group, identity, error)
        plan._barrier()
        torch.cuda.synchronize(output.tensor.device)
        return plan

    def _configure_diagnostics(self):
        from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.fused_shared_rs import (
            FusedSharedRsUpProjectionGemm,
        )

        kernel_type = FusedSharedRsUpProjectionGemm
        if getattr(self.tuning, "experimental_fused_n64", False):
            from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.fused_shared_rs_n64 import (
                FusedSharedRsN64UpProjectionGemm,
            )

            kernel_type = FusedSharedRsN64UpProjectionGemm
        self.kernel = kernel_type(
            int(self.workspace.state.symm_mem_hdl.multicast_ptr),
            self.output.rank,
            self.tuning.c_stages,
            self.tuning.prefetch_acc_tile,
            self.tuning.addend_stages,
            (
                self.tuning.reduce_vectors
                if isinstance(self.tuning, FusedRsUpProjectionTuning)
                else 1
            ),
        )

    def stage(self, partial: torch.Tensor) -> torch.Tensor:
        """Copy BF16 [M,7168] into input_view; real producers may use out= instead."""
        if (
            partial.shape != self.input_view.shape
            or partial.dtype != self.input_view.dtype
            or partial.device != self.input_view.device
            or not partial.is_contiguous()
        ):
            raise ValueError("stage requires colocated contiguous BF16 [M,7168]")
        self.input_view.copy_(partial)
        return self.input_view

    def __call__(self) -> torch.Tensor:
        # The entry's release/acquire and alias fences publish raw partials,
        # including same-stream copies/producers issued before the fixed HT.
        _rank_barrier_kernel[(1,)](
            self.workspace.state.symm_mem_hdl.signal_pad_ptrs_dev,
            RANK=self.output.rank,
            ENABLE_PDL=False,
            num_warps=4,
        )
        super().producer_only(multicast=True)
        # All CTAs have finished their raw reads before this complete-rank exit.
        # It protects both final output consumers and subsequent raw overwrites.
        self._barrier()
        return self.output.tensor

    def producer_only(self, *, multicast: bool) -> None:
        """Reject unsafe un-published standalone access to symmetric partials."""
        raise RuntimeError("use the complete fused plan; raw publication is mandatory")
