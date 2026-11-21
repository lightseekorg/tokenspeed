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


import torch
from tokenspeed_kernel.compile_monitor import serving_selection
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm import plan_dense_gemm
from tokenspeed_kernel.selection import resolve_kernel_override
from torch.nn.parameter import Parameter

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.utils import set_weight_attrs


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def __init__(self):
        from tokenspeed.runtime.utils.env import global_server_args_dict

        self.batch_invariant = global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        self.kernel_plan = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        if self.batch_invariant:
            self.kernel_plan = plan_dense_gemm(
                mode="mm",
                input_size=input_size_per_partition,
                output_size=sum(output_partition_sizes),
                dtype=params_dtype,
                batch_size=None,
                batch_invariant=True,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.batch_invariant and self.kernel_plan is None:
            self.kernel_plan = plan_dense_gemm(
                mode="mm",
                input_size=layer.weight.shape[1],
                output_size=layer.weight.shape[0],
                dtype=layer.weight.dtype,
                batch_size=None,
                batch_invariant=True,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.batch_invariant:
            if self.kernel_plan is None:
                raise RuntimeError(
                    "Batch-invariant Linear requires a kernel plan before execution"
                )
            return kernel_mm(x, layer.weight, bias=bias, plan=self.kernel_plan)

        if resolve_kernel_override("gemm", "mm", None) is not None:
            return kernel_mm(x, layer.weight, bias=bias)

        if bias is None:
            from tokenspeed_kernel.ops.gemm import _try_gluon_largem_gfx1250

            largem = _try_gluon_largem_gfx1250(x, layer.weight)
            if largem is not None:
                return largem
        return kernel_mm(
            x,
            layer.weight,
            bias=bias,
        )

    @torch.no_grad()
    def warmup_row_counts(self, layer: torch.nn.Module, max_tokens: int) -> None:
        """Run the layer's GEMM once at every row count up to max_tokens.

        For a layer that eager code runs at each forward's real row count: the
        library GEMM this method reaches (cuBLAS on NVIDIA) picks its kernel
        per row count and loads a kernel on its first launch, which must not
        happen while serving (see
        ``tokenspeed_kernel.ops.gemm.warmup_dsv4_linear_fp32``). Kernels are
        selected as serving selects them.

        Args:
            layer: A bias-free layer this method serves.
            max_tokens: Largest row count the layer's caller passes.
        """
        if layer.bias is not None:
            raise ValueError("Row-count warm-up takes bias-free layers only")
        weight = layer.weight
        rows = torch.zeros(
            (max_tokens, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        with serving_selection():
            # Largest first, so the outputs each launch allocates reuse cached blocks.
            for tokens in range(max_tokens, 0, -1):
                self.apply(layer, rows[:tokens], None)
        torch.cuda.synchronize(weight.device)
