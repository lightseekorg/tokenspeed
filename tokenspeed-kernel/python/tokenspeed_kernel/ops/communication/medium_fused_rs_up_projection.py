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

"""Experimental medium-M fused collective binding; not registered in serving."""

import hashlib
import re
import tempfile

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_config import (
    MediumFusedRsUpProjectionTuning,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_fused_rs_up_projection import (
    BoundFusedRsUpProjection,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    _overlaps,
    _rank_barrier_kernel,
    _vote,
)


def _cuda_result(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"medium kernel CUDA resource query failed: {result}")
    return result[1] if len(result) == 2 else result[1:]


def _admit_compiled(plan):
    """Inspect the exact CUBIN and reject insufficient residency before launch."""
    cuda = plan._cuda
    ptx, cubin = plan.compiled.__ptx__, plan.compiled.__cubin__
    if type(ptx) is bytes:
        ptx = ptx.decode("utf-8")
    if type(ptx) is not str or type(cubin) not in (bytes, bytearray):
        raise ValueError("compiled PTX and CUBIN must be available for admission")
    symbols = re.findall(r"\.entry\s+([A-Za-z0-9_$]+)", ptx)
    if len(symbols) != 1:
        raise ValueError("requires one compiled fused producer symbol")
    if (
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2" not in ptx
        or "cp.async.bulk.tensor.2d.global.shared::cta" not in ptx
        or "griddepcontrol." in ptx
    ):
        raise ValueError("compiled producer lost fused NVLS/TMA or PDL-off contract")
    budget = plan.kernel.medium_smem_budget_bytes
    if type(budget) is not int or budget <= 0:
        raise ValueError("compiled shared-memory upper bound is unavailable")
    module = _cuda_result(cuda.cuModuleLoadData(cubin))
    try:
        function = _cuda_result(cuda.cuModuleGetFunction(module, symbols[0].encode()))
        resource = {}
        for key, name in (
            ("registers_per_thread", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
            ("local_bytes_per_thread", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
            ("static_shared_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"),
            ("max_threads_per_block", "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"),
        ):
            resource[key] = int(
                _cuda_result(
                    cuda.cuFuncGetAttribute(
                        getattr(cuda.CUfunction_attribute, name), function
                    )
                )
            )
        if resource["max_threads_per_block"] < 192 or resource["static_shared_bytes"]:
            raise ValueError("requires 192 threads and dynamic-only shared allocation")
        # Spill is reported as an optimization result, not mistaken for lack
        # of correctness; actual residency is still queried with all resources.
        _cuda_result(
            cuda.cuFuncSetAttribute(
                function,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                budget,
            )
        )
        attribute = cuda.CUlaunchAttribute()
        attribute.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attribute.value.clusterDim.x = plan.tuning.cluster_m
        attribute.value.clusterDim.y = plan.tuning.cluster_n
        attribute.value.clusterDim.z = 1
        config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = (
            plan.tuning.cluster_m,
            plan.tuning.cluster_n,
            plan.max_active_clusters,
        )
        config.blockDimX, config.blockDimY, config.blockDimZ = 192, 1, 1
        config.sharedMemBytes = budget
        config.attrs, config.numAttrs = [attribute], 1
        active = int(_cuda_result(cuda.cuOccupancyMaxActiveClusters(function, config)))
        # Full-grid work may queue more whole clusters than fit concurrently.
        # There is no grid-wide or cross-rank rendezvous inside this GEMM.
        # Persistent admission retains its complete-population capacity check.
        required = (
            1 if plan.tuning.scheduler_type == "full_grid" else plan.max_active_clusters
        )
        if active < required:
            raise ValueError(
                f"compiled capacity {active} below required resident clusters {required}"
            )
        return {
            **resource,
            "active_clusters_at_smem_upper_bound": active,
            "selected_cluster_cap": (
                plan.max_active_clusters
                if plan.tuning.scheduler_type == "static_persistent"
                else None
            ),
            "scheduler_type": plan.tuning.scheduler_type,
            "problem_clusters": plan.launch_geometry["problem_clusters"],
            "launched_clusters": plan.launch_geometry["launched_clusters"],
            "planned_launch_grid": plan.launch_geometry["grid"],
            "residency_query_grid": [
                plan.tuning.cluster_m,
                plan.tuning.cluster_n,
                plan.max_active_clusters,
            ],
            "minimum_required_resident_clusters": required,
            "dynamic_smem_upper_bound_bytes": budget,
            "actual_launch_dynamic_smem_observed": False,
            "mma_tile": [plan.tuning.tile_m, plan.tuning.tile_n],
            "cluster": [plan.tuning.cluster_m, plan.tuning.cluster_n],
            "num_ab_stage": plan.kernel.num_ab_stage,
            "requested_ab_stages": plan.tuning.ab_stages,
            "num_c_stage": plan.kernel.num_c_stage,
            "num_acc_stage": plan.kernel.num_acc_stage,
            "addend_smem_bytes": plan.kernel.addend_smem_bytes,
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        }
    finally:
        _cuda_result(cuda.cuModuleUnload(module))


class BoundMediumFusedRsUpProjection(BoundFusedRsUpProjection):
    """Same publication/math/output lifecycle, independent medium configuration.

    The fixed inputs and one explicit stream are bound before capture. The
    legacy shard allocation remains metadata only. Calls allocate nothing and
    use exactly the inherited raw-publication, producer and all-rank exit.
    """

    def _compile(self):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
            CUDAGraphCompatibleWrapper,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.fused_shared_rs_medium import (
            MediumFusedSharedRsUpProjectionGemm,
        )

        self._cuda, self._cutlass = cuda, cutlass
        latent, weight, dummy, residual = self.inputs
        owner = slice(self.output.rank * 896, (self.output.rank + 1) * 896)
        tensors = (
            latent,
            weight,
            self.output.tensor[:, owner],
            dummy,
            residual[:, owner],
        )
        self._views = tuple(x.unsqueeze(-1) for x in tensors)
        self._cute_args = tuple(
            from_dlpack(CUDAGraphCompatibleWrapper(x.detach()), assumed_align=16)
            for x in self._views
        )
        with torch.cuda.device(latent.device):
            self.kernel = MediumFusedSharedRsUpProjectionGemm(
                int(self.workspace.state.symm_mem_hdl.multicast_ptr),
                self.output.rank,
                self.tuning,
            )
            self._compile_dump = tempfile.TemporaryDirectory(
                prefix=f"up-medium-r{self.output.rank}-"
            )
            self.compiled = cute.compile(
                self.kernel,
                *self._cute_args,
                cutlass.Int64(
                    self.output.handle.multicast_ptr + self.output.rank * 896 * 2
                ),
                cutlass.Boolean(True),
                self.max_active_clusters,
                cuda.CUstream(self._bound_stream),
                options=f"--dump-dir={self._compile_dump.name} --keep-ptx --keep-cubin",
            )
            self.admission = _admit_compiled(self)

    def _check_stream(self):
        if (
            torch.cuda.current_stream(self.output.tensor.device).cuda_stream
            != self._bound_stream
        ):
            raise ValueError("medium fused plan must run/capture on its bound stream")

    def stage(self, partial):
        """Copy the contiguous BF16 partial into the exact symmetric input view."""
        self._check_stream()
        if (
            _overlaps(partial, self.input_view)
            and partial.data_ptr() != self.input_view.data_ptr()
        ):
            raise ValueError("staging source partially aliases its destination")
        return super().stage(partial)

    def __call__(self):
        """Publish raw partials, run fused RS/GEMM/AG, complete all ranks, return out."""
        self._check_stream()
        return super().__call__()


def prepare_medium_fused_rs_up_projection(
    latent, weight, residual, workspace, output, *, residual_is_replicated, tuning
):
    """Collectively bind and admit an unqualified medium configuration.

    Args:
        latent: Replicated contiguous CUDA BF16 [M,3584] normalized routed input.
        weight: Owner-local contiguous CUDA BF16 [896,3584] up-projection weight.
        residual: Replicated contiguous CUDA BF16 [M,7168], added once after RS.
        workspace: SharedRsWorkspace holding symmetric raw input and metadata shard.
        output: Separate symmetric BF16 [M,7168] output; a guarded prefix is allowed.
        residual_is_replicated: Explicit True on all eight ranks.
        tuning: Rank-identical MediumFusedRsUpProjectionTuning with all controls explicit.

    Returns:
        BoundMediumFusedRsUpProjection with input_view, stage, callable output and
        admission_records. M must be in [33,8192]; non-BT endpoints are diagnostic
        controls, never an implicit runtime qualification. All validation, cluster
        agreement, compilation and resource admission happen before any launch.
    """
    state = workspace.state
    m = output.tensor.shape[0]
    raw, dummy = state.comm_buff[:m], state.local_buff[:m]
    error, local = None, None
    try:
        if type(tuning) is not MediumFusedRsUpProjectionTuning:
            raise ValueError("requires explicit medium tuning type")
        tuning.validate()
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("medium preparation must precede capture")
        if (
            not 33 <= m <= min(state.max_token_num, 8192)
            or state.group is not output.group
            or state.rank_in_group != output.rank
            or output.rank != dist.get_rank(output.group)
            or dist.get_world_size(output.group) != 8
            or residual_is_replicated is not True
            or not state.symm_mem_hdl.multicast_ptr
            or not output.handle.multicast_ptr
        ):
            raise ValueError(
                "requires matching TP8 symmetric owners and replicated residual"
            )
        if torch.cuda.get_device_capability(output.tensor.device)[0] != 10:
            raise ValueError("medium fused kernel requires Blackwell SM100/SM103")
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
                raise ValueError(f"requires aligned contiguous CUDA BF16 {shape}")
        for handle, tensor, shape in (
            (state.symm_mem_hdl, raw, (state.max_token_num, 7168)),
            (output.handle, output.tensor, (m, 7168)),
        ):
            physical = handle.get_buffer(
                output.rank, shape, torch.bfloat16, storage_offset=0
            )
            if physical.data_ptr() != tensor.data_ptr():
                raise ValueError("symmetric tensor does not match its physical mapping")
        if any(
            _overlaps(raw, tensor)
            for tensor in (latent, weight, residual, dummy, output.tensor)
        ):
            raise ValueError("raw symmetric input aliases protected storage")
        if any(
            _overlaps(output.tensor, tensor)
            for tensor in (latent, weight, residual, dummy)
        ):
            raise ValueError("symmetric output aliases a live input")
        import cutlass.utils as utils

        with torch.cuda.device(output.tensor.device):
            capacity = utils.HardwareInfo().get_max_active_clusters(
                tuning.cluster_m * tuning.cluster_n
            )
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cluster capacity query returned no usable clusters")
        local = {"rank": output.rank, "hardware_cluster_capacity": capacity}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    identity = ("medium-fused-shared-rs-up-ag", m, tuning, residual_is_replicated)
    _vote(output.group, identity, error)
    capacity_records = [None] * 8
    dist.all_gather_object(capacity_records, local, group=output.group)
    minimum = min(row["hardware_cluster_capacity"] for row in capacity_records)
    selected = minimum if tuning.cluster_cap is None else tuning.cluster_cap
    _vote(
        output.group,
        identity,
        None if selected <= minimum else "requested cap exceeds all-rank capacity",
    )
    plan = BoundMediumFusedRsUpProjection()
    plan.workspace, plan.input_view, plan.output, plan.tuning = (
        workspace,
        raw,
        output,
        tuning,
    )
    plan.inputs = (latent, weight, dummy, residual)
    plan.skip_entry_sync = False
    plan.max_active_clusters = selected
    plan.launch_geometry = tuning.launch_geometry(m, minimum)
    plan.capacity_records = capacity_records
    plan._bound_stream = torch.cuda.current_stream(output.tensor.device).cuda_stream
    error = None
    try:
        plan._compile()
        for signals in (
            state.symm_mem_hdl.signal_pad_ptrs_dev,
            output.handle.signal_pad_ptrs_dev,
        ):
            _rank_barrier_kernel.warmup(
                signals, RANK=output.rank, ENABLE_PDL=False, num_warps=4, grid=(1,)
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _vote(output.group, identity, error)
    plan.admission_records = [None] * 8
    dist.all_gather_object(plan.admission_records, plan.admission, group=output.group)
    plan._barrier()
    torch.cuda.synchronize(output.tensor.device)
    return plan
