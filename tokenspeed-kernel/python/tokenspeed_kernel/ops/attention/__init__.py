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

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

LSE_LN = math.log2(math.e)

# ---------------------------------------------------------------------------
# Selection traits
# ---------------------------------------------------------------------------
#
# The trait dicts the attention variants pass to ``select_kernel`` and the
# ``traits`` their registrations declare share one vocabulary:
#
# * Geometry is an int per dimension, named after the tensor axis it
#   describes: ``batch_size``, ``num_tokens``, ``q_len`` (query rows per
#   request), ``max_seqlen_q``, ``num_q_heads``/``num_kv_heads``/``num_heads``
#   (the last for kernels without a q/kv split), ``index_heads``, ``head_dim``
#   (the QK width), ``value_head_dim``, ``qk_nope_head_dim``, ``kv_lora_rank``,
#   ``qk_rope_head_dim``, ``page_size``, ``topk``, ``selected_width``, ... A
#   spec constrains a dimension with an exact set, or with ``<dim>_align`` /
#   ``<dim>_min`` bounds evaluated by ``selection.spec_matches_shape_traits``
#   (for example ``batch_size_align``).
# * A feature the call uses is a plain boolean named after the feature:
#   ``is_causal``, ``logit_cap``, ``return_lse``, ``sinks``, ``skip_softmax``,
#   ``sliding_window``. A spec lists the values it accepts, so ``{False}``
#   means the feature is unsupported and ``{False, True}`` that it is optional.
# * An optional input the call provides is ``has_<input>``: ``has_kv_cache``,
#   ``has_prefill_plan``, ``has_q_out``, ...
# * Everything else is an enum-like layout or format string (``cache_layout``,
#   ``index_k_format``, ``topk_layout``, ``recurrent_layout``, ...).
#
# Trait dicts list the geometry first, in the order above with each
# ``_align``/``_min`` bound right after its dimension, followed by the
# remaining traits alphabetically.


def attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    lse_scale_log2: float = LSE_LN,
    inplace: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two partial attention states.

    Args:
        out_a: First partial output with shape [total_q, num_heads, head_dim].
        lse_a: First partial log-sum-exp with shape [total_q, num_heads].
        out_b: Second partial output with shape [total_q, num_heads, head_dim].
        lse_b: Second partial log-sum-exp with shape [total_q, num_heads].
        lse_scale_log2: Multiplier that converts input LSE to log2 domain.
        inplace: Whether to write the merged state back into ``out_a``/``lse_a``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    This is shared by MHA and MLA because the merge only depends on partial
    attention outputs and LSE values, not on how the K/V states were produced.
    """

    signature = format_signature(
        out_a=dense_tensor_format(out_a.dtype),
        out_b=dense_tensor_format(out_b.dtype),
    )
    kernel = select_kernel(
        "attention",
        "attn_merge_state",
        signature,
        traits={"head_dim": out_a.shape[-1]},
        solution=solution,
        override=override,
    )

    shape_params = {
        "total_q": out_a.shape[0],
        "num_heads": out_a.shape[1],
        "head_dim": out_a.shape[2],
    }
    ShapeCapture.get().record(
        "attention",
        "attn_merge_state",
        kernel.name,
        out_a.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "attn_merge_state",
        out_a.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            out_a=out_a,
            lse_a=lse_a,
            out_b=out_b,
            lse_b=lse_b,
            lse_scale_log2=lse_scale_log2,
            inplace=inplace,
            enable_pdl=pdl_enabled(),
        )


# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.triton  # noqa: E402,F401

__all__ = [
    "attn_merge_state",
]
