"""lm_head GEMM kernel wrapper (ported from TRT-LLM invokeFusedAGemm).

Computes:
    out = hidden_states @ weight.T

where:
    hidden_states: bf16 CUDA tensor [num_tokens, hidden_dim]
    weight:        bf16 CUDA tensor [vocab_shard, hidden_dim]   (row-major, contiguous)
    out:           bf16 CUDA tensor [num_tokens, vocab_shard]

Pipeline config (B200-tuned): 96 KB smem/CTA, 2 CTAs/SM, 8 stages
(tile_n=8) / 6 stages (tile_n=16). The launcher picks a static or
persistent-CTA grid internally based on num_tokens and hd_out — see
``should_use_persistent`` in ``csrc/lm_head_gemm.cu``.

Two gating helpers for callers:

  - ``is_supported(h, w)``: shape/dtype/CC compatibility — does the .so
    have a template instantiation that can run? Used by benchmark harnesses
    to enumerate what's compiled.

  - ``should_use_fused(h, w)``: bench-driven routing — does the fused
    kernel actually beat ``torch.matmul`` on this (M, N) pair? Production
    callers (``logits_processor.py``) should use this and fall back to
    ``torch.matmul`` when it returns False.

The bench thresholds in ``_FUSED_MAX_TOKENS`` come from the M=1..16 sweep
in ``tmp/opt_lm_head/integrate_and_replace.md`` §13. Re-run the sweep and
update this table when the kernel or HW changes.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import tvm_ffi


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


# (hidden_dim, vocab_shard) pairs that have explicit template instantiations
# compiled into the shared library. Keep in sync with INSTANTIATE_TILE_N in
# csrc/lm_head_gemm.cu.
_SUPPORTED_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {
        # DSv3 (vocab=129280)
        (7168, 16160),  # DSv3 TP=8
        (7168, 32320),  # DSv3 TP=4
        (7168, 129280),  # DSv3 no-TP
        # Kimi K2.5 (vocab=163840)
        (7168, 20480),  # K2.5 TP=8
        (7168, 40960),  # K2.5 TP=4
        (7168, 163840),  # K2.5 no-TP
    }
)


# Per-(hidden_dim, vocab_shard), the largest num_tokens for which the fused
# kernel beats torch.matmul on B200 (sm_100). 0 / absent → never beats torch
# on this shape, fall back unconditionally. From M=1..16 sweep, see
# tmp/opt_lm_head/integrate_and_replace.md §13.
_FUSED_MAX_TOKENS: dict[tuple[int, int], int] = {
    # DSv3 (vocab=129280)
    (
        7168,
        16160,
    ): 0,  # cuBLAS hmma already at peak at small N; ours ties at M=1, loses M>=2.
    (7168, 32320): 8,  # ours wins / ties M=1..8 by 2-5%; M>=9 loses up to 3%.
    (7168, 129280): 4,  # ours wins / ties M=1..4 (within 2%); M>=5 loses 2-10%.
    # Kimi K2.5 (vocab=163840)
    (7168, 20480): 16,  # ours wins / ties every M=1..16 by 0-7%.
    (
        7168,
        40960,
    ): 8,  # ours wins M=1..8 by 5-9%; M=9..15 loses 4-6% (M=16 a tiny win, treat as loss).
    (7168, 163840): 16,  # ours wins every M=1..16 by 4-16%.
}


@functools.cache
def _load_module():
    so_path = _objs_dir() / "lm_head_gemm" / "lm_head_gemm.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel lm_head_gemm library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def is_supported(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    num_tokens_max: int = 16,
) -> bool:
    """Return True if this lm_head shape can run through the fused kernel.

    Capability gate (does the .so have it?), independent of perf:
      * bf16 only
      * M = num_tokens <= num_tokens_max (default 16)
      * (K, N) = (hidden_dim, vocab_shard) matches a compiled instantiation
      * CC >= 9.0 (Hopper or newer; uses HMMA + cp.async + mbarrier + PDL)
    """
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if hidden_states.device.type != "cuda" or weight.device.type != "cuda":
        return False
    if hidden_states.dim() != 2 or weight.dim() != 2:
        return False
    num_tokens, hd_in = hidden_states.shape
    hd_out, hd_in_w = weight.shape
    if hd_in != hd_in_w or num_tokens > num_tokens_max:
        return False
    if (hd_in, hd_out) not in _SUPPORTED_SHAPES:
        return False
    major, _ = torch.cuda.get_device_capability(hidden_states.device)
    if major < 9:
        return False
    return True


def should_use_fused(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return True if the fused kernel beats torch.matmul on this shape.

    Combines the capability gate (``is_supported``) with the bench-derived
    perf gate (``_FUSED_MAX_TOKENS``). Production callers should use this —
    it returns False even for compiled shapes when torch.matmul wins (e.g.
    DSv3 TP=8 N=16160, where cuBLAS bf16 hmma is already at peak).
    """
    if not is_supported(hidden_states, weight):
        return False
    if not hidden_states.is_contiguous() or not weight.is_contiguous():
        return False
    num_tokens = hidden_states.shape[0]
    hd_in = hidden_states.shape[1]
    hd_out = weight.shape[0]
    cap = _FUSED_MAX_TOKENS.get((hd_in, hd_out), 0)
    return num_tokens <= cap


def lm_head_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Fused bf16 GEMM for lm_head / router-like projections.

    Args:
        hidden_states: ``[num_tokens, hidden_dim]`` bf16 contiguous CUDA tensor.
        weight:        ``[vocab_shard, hidden_dim]`` bf16 contiguous CUDA tensor
                       (i.e. the raw ``lm_head.weight``; no explicit transpose).
        out:           optional pre-allocated output ``[num_tokens, vocab_shard]``.
        enable_pdl:    whether to request Programmatic Dependent Launch.

    Returns:
        bf16 CUDA tensor ``[num_tokens, vocab_shard]``.
    """
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    num_tokens, _ = hidden_states.shape
    hd_out = weight.shape[0]
    if out is None:
        out = torch.empty(
            (num_tokens, hd_out),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
    else:
        assert out.is_contiguous()
        assert out.shape == (num_tokens, hd_out)
        assert out.dtype == torch.bfloat16
    # tile_n=8 is the minimum-latency config for M<=8; tile_n=16 amortizes
    # the store/epilogue when M>8.
    tile_n = 8 if num_tokens <= 8 else 16
    _load_module().lm_head_gemm(
        out, hidden_states, weight, int(tile_n), bool(enable_pdl)
    )
    return out
