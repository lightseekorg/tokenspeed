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

"""CuTe DSL implementation of BF16/group128 quantization for DeepGEMM."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="cute_dsl_quantize_fp8_group128_packed",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "granularity": frozenset({"token_group_128"}),
        "scale_encoding": frozenset({"packed_ue8m0"}),
        "scale_layout": frozenset({"mn_major"}),
    },
    priority=Priority.PERFORMANT + 1,
)
def cute_dsl_quantize_fp8_group128_packed(
    x: torch.Tensor,
    *,
    granularity: str,
    group_size: int | None,
    scale_encoding: str,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 values and packed UE8M0 scales without a transpose or repack.

    Args:
        x: BF16 activations with positive K divisible by 128.
        granularity: Must be token_group.
        group_size: Must be 128.
        scale_encoding: Must be packed_ue8m0.
        enable_pdl: Enable producer waiting and dependent launch signaling.

    Returns:
        Values shaped like x and int32 scales [M, ceil(K/512)], with strides
        [1, round_up(M, 4)]. M flattens the leading dimensions. Padding is zero.
        Warm the chosen width and PDL setting before capturing a CUDA Graph.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.fp8_group128 import launch

    if (
        granularity != "token_group"
        or group_size != 128
        or scale_encoding != "packed_ue8m0"
    ):
        raise ValueError("requires token_group=128 with packed_ue8m0 scales")
    if (
        x.dtype != torch.bfloat16
        or x.ndim == 0
        or x.shape[-1] == 0
        or x.shape[-1] % 128
    ):
        raise ValueError("requires BF16 input with positive K divisible by 128")
    x = x.contiguous()
    # A contiguous slice can still have an unaligned storage offset.
    if x.data_ptr() % 32:
        x = x.clone()
    k = x.shape[-1]
    rows = x.numel() // k
    padded_rows = (rows + 3) // 4 * 4
    storage = torch.empty(
        ((k + 511) // 512, padded_rows), dtype=torch.int32, device=x.device
    )
    scales = storage.t()[:rows]
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if rows:
        threads = 32 if rows == 1 else 128 if rows <= 512 else 256
        launch(x, output, scales, threads=threads, use_pdl=enable_pdl)
    return output, scales
