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

"""Default-off medium profile of the existing live-pointer serving adapter."""

import torch
from tokenspeed_kernel.ops.communication.cutedsl_fused_rs_up_projection import (
    FusedRsUpProjectionServing,
    _PreparedLaunch,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection import (
    prepare_medium_fused_rs_up_projection,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving_config import (
    integrated_fused_rs_serving_config,
    medium_fused_rs_serving_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
    SharedRsWorkspace,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    SymmetricUpProjectionOutput,
    allocate_symmetric_up_projection_output,
)


class MediumFusedRsUpProjectionServing(FusedRsUpProjectionServing):
    """Use the same validation, live operands and two barriers with medium binding.

    This experimental profile is unqualified. The caller must opt in explicitly;
    an enabled flag is not evidence of correctness or serving performance.
    """

    profile = staticmethod(medium_fused_rs_serving_config)

    def __init__(
        self,
        workspace: SharedRsWorkspace,
        max_tokens: int,
        *,
        output: SymmetricUpProjectionOutput | None = None,
    ):
        """Allocate one layer's output before KV sizing and graph capture.

        Args:
            workspace: Model-owned raw symmetric storage, used sequentially.
            max_tokens: Largest provisional bucket this output must cover.
            output: Optional model-owned symmetric output. Its prior consumers
                must finish before reuse; it must outlive every bound graph.
        """
        self.profile(max_tokens)
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("medium serving allocation must precede capture")
        if max_tokens > workspace.state.max_token_num:
            raise ValueError("medium output capacity exceeds raw workspace")
        self.workspace = workspace
        if output is None:
            output = allocate_symmetric_up_projection_output(
                workspace.state.group, max_tokens, device=workspace.state.device
            )
        if (
            output.tensor.shape != (max_tokens, 7168)
            or output.tensor.dtype != torch.bfloat16
            or output.tensor.device != workspace.state.device
            or output.group is not workspace.state.group
            or not output.tensor.is_contiguous()
        ):
            raise ValueError("serving output must match workspace group and capacity")
        self.output = output
        self._plans: dict[int, _PreparedLaunch] = {}

    def input_view(self, num_tokens: int) -> torch.Tensor:
        """Return the producer's exact symmetric BF16 [M,7168] destination."""
        self.profile(num_tokens)
        if num_tokens > self.output.tensor.shape[0]:
            raise ValueError("medium serving bucket exceeds allocated capacity")
        return self.workspace.state.comm_buff[:num_tokens]

    def _prepare(self, m, latent, weight, residual):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("medium serving compilation must finish during warmup")
        output = SymmetricUpProjectionOutput(
            self.output.tensor[:m],
            self.output.handle,
            self.output.group,
            self.output.rank,
        )
        bound = prepare_medium_fused_rs_up_projection(
            latent,
            weight,
            residual,
            self.workspace,
            output,
            residual_is_replicated=True,
            tuning=self.profile(m),
        )
        # Inherit the live-pointer launcher; never retain warmup latent/residual.
        result = _PreparedLaunch(
            compiled=bound.compiled,
            cuda=bound._cuda,
            cutlass=bound._cutlass,
            fixed_args=bound._cute_args[1:4],
            weight=weight,
            output=output,
            compile_dump=bound._compile_dump,
            capacity_records=bound.admission_records,
            kernel=bound.kernel,
            max_active_clusters=bound.max_active_clusters,
        )
        self._plans[m] = result
        return result


class IntegratedFusedRsUpProjectionServing(MediumFusedRsUpProjectionServing):
    """One live-pointer execution path with continuous medium/large tuning.

    Allocation, publication, masking and completion are inherited unchanged.
    This acceptance profile does not imply numerical or TTFT qualification.
    """

    profile = staticmethod(integrated_fused_rs_serving_config)


class IntegratedFusedRsOutputPool:
    """Two outputs for sequential K3 layers, never for concurrent executions.

    AttnRes snapshots and speculative hidden-state taps must own copies. The
    current layer's residual may reference the previous slot, not this slot.
    The inherited publication barrier orders prior consumers across ranks
    before multicast overwrites a slot; output completion precedes consumers.
    """

    def __init__(self, workspace: SharedRsWorkspace, max_tokens: int):
        """Allocate both model-owned slots before KV sizing and graph capture.

        Args:
            workspace: Shared raw workspace for one sequential model instance.
            max_tokens: Maximum output capacity, common to both slots.
        """
        integrated_fused_rs_serving_config(max_tokens)
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("output pool allocation must precede capture")
        if max_tokens > workspace.state.max_token_num:
            raise ValueError("output pool capacity exceeds raw workspace")
        self.workspace = workspace
        self.max_tokens = max_tokens
        self.outputs = tuple(
            allocate_symmetric_up_projection_output(
                workspace.state.group, max_tokens, device=workspace.state.device
            )
            for _ in range(2)
        )

    def bind_layer(self, layer_index: int) -> IntegratedFusedRsUpProjectionServing:
        """Return a layer-specific launch cache using its fixed parity slot.

        Args:
            layer_index: Nonnegative global decoder-layer index.

        Returns:
            Adapter with independent weight/plan bindings and a shared output.
        """
        if type(layer_index) is not int or layer_index < 0:
            raise ValueError("layer index must be a nonnegative integer")
        return IntegratedFusedRsUpProjectionServing(
            self.workspace, self.max_tokens, output=self.outputs[layer_index % 2]
        )
