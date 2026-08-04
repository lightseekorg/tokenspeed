"""DeepSeek V4 DeepGEMM exports and model warmup helpers."""

from tokenspeed_kernel.ops.model.deepseek_v4.deep_gemm.warmup import (
    warmup_fp8_gemm_nt,
    warmup_fp8_gemm_nt_from_model,
    warmup_mega_moe_jit,
    warmup_prefill_jit,
)
from tokenspeed_kernel.ops.other.native import deep_gemm as _deep_gemm
from tokenspeed_kernel.ops.other.native.deep_gemm import *  # noqa: F403

__all__ = [
    *_deep_gemm.__all__,
    "warmup_fp8_gemm_nt",
    "warmup_fp8_gemm_nt_from_model",
    "warmup_mega_moe_jit",
    "warmup_prefill_jit",
]
