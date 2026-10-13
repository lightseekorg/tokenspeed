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

import logging
from functools import cache

import torch
from tokenspeed_kernel.ops.communication.fabric import group_has_fabric
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.flashinfer.moe_alltoall import FlashInferMoeAlltoAll

logger = logging.getLogger(__name__)


@cache
def get_flashinfer_moe_alltoall(
    group: torch.distributed.ProcessGroup,
    model_scope: str,
    max_tokens: int,
    hidden_size: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    weights_dtype: torch.dtype,
) -> FlashInferMoeAlltoAll | None:
    """Create one workspace per model/group, or return None without a shared NVLink fabric.

    Args:
        group: Expert-parallel process group.
        model_scope: Model identity; overlapping target/draft graphs require separate storage.
        max_tokens: Maximum local token capacity, including prefill and verification.
        hidden_size: Width of routed activations and outputs.
        top_k: Number of selected experts per token.
        num_experts: Global expert count.
        dtype: Routed activation/output dtype.
        weights_dtype: Precomputed routing-weight dtype.

    Returns:
        A shared transport for sequential layers of this model, or None for the AG/RS fallback.
    """
    ranks = tuple(torch.distributed.get_process_group_ranks(group))
    if not current_platform().is_nvidia or not group_has_fabric(ranks):
        return None
    transport = FlashInferMoeAlltoAll(
        group, max_tokens, hidden_size, top_k, num_experts, dtype, weights_dtype
    )
    logger.info(
        f"K3 MoE communication: FlashInfer MNNVL all-to-all scope={model_scope!s} ep="
        f"{group.size():d} capacity={max_tokens:d} width={hidden_size:d}",
    )
    return transport
