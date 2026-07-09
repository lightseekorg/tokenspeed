"""Standalone Gluon bf16 MoE GEMM package for AMD MI350X (gfx950).

Two-stage unquantized (bf16 activation / bf16 weight) fused MoE in Gluon:

  Stage 1 (``stage1_kernel.py``): gate + up GEMM with fused SwiGLU.
  Stage 2 (``stage2_kernel.py``): down GEMM + routed-weight scale, written
     to a ``[num_tokens, topk, D]`` scratch, then reduced over topk.

  Block-align: ``moe_align_fused.py`` (decode, single-workgroup) and
     ``moe_align_device.py`` (prefill) produce the ``sorted_token_ids`` /
     ``sorted_expert_ids`` / ``sorted_weights`` contract the kernels consume.

  End-to-end (``moe.py``): ``gluon_bf16_moe(...)`` (decode split-K path for
     small M, XCD-remap prefill path otherwise).

Reference shape: DeepSeek-V3 (E=256, D=7168, I=256, topk=8); see ``README.md``.
"""

from .moe import gluon_bf16_moe
from .moe_align_device import moe_align_block_size_device
from .moe_align_fused import moe_align_block_size_fused
from .stage1_kernel import gluon_bf16_moe_stage1_kernel, invoke_stage1
from .stage1_splitk_kernel import auto_split_k, invoke_stage1_splitk
from .stage2_decode_kernel import invoke_stage2_decode
from .stage2_kernel import (
    gluon_bf16_moe_reduce_kernel,
    gluon_bf16_moe_stage2_kernel,
    invoke_stage2,
)

__all__ = [
    "gluon_bf16_moe",
    "invoke_stage1",
    "invoke_stage1_splitk",
    "auto_split_k",
    "invoke_stage2",
    "invoke_stage2_decode",
    "moe_align_block_size_device",
    "moe_align_block_size_fused",
    "gluon_bf16_moe_stage1_kernel",
    "gluon_bf16_moe_stage2_kernel",
    "gluon_bf16_moe_reduce_kernel",
]
