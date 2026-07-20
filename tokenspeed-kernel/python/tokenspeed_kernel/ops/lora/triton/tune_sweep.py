"""Comprehensive autotune sweep for LoRA decode kernels across common shapes.

Covers the (N, K) pairs seen in production for the major model families and
TP configurations, across max_rank values of 16 / 32 / 64 / 128.  Saves all
picked configs to the on-disk JSON caches so fresh processes skip the sweep.

Usage::

    python -m tokenspeed_kernel.ops.lora.triton.tune_sweep

Estimated runtime: ~5 min on H100 (all shapes × all kernels).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.lora.triton.lora_expand import _lora_expand_kernel
from tokenspeed_kernel.ops.lora.triton.lora_gate_up_expand import (
    _lora_gate_up_expand_kernel,
)
from tokenspeed_kernel.ops.lora.triton.lora_qkv_expand import _lora_qkv_expand_kernel
from tokenspeed_kernel.ops.lora.triton.lora_shrink import _lora_shrink_kernel
from tokenspeed_kernel.ops.lora.triton.tune import (
    _BatchInfo,
    _make_batch,
    tune_expand,
    tune_gate_up,
    tune_qkv,
    tune_shrink,
)
from tokenspeed_kernel.ops.lora.triton.tuning import save_kernel_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class _ModelTP:
    name: str
    hidden: int
    intermediate_per_tp: int
    q_per_tp: int
    kv_per_tp: int


# ── Representative (model, TP) configs ──────────────────────────────────────
# Each entry represents one serving configuration: hidden size, per-rank
# intermediate, and per-rank Q / KV sizes after tensor parallelism sharding.
# Source model sizes:
#   Llama-3-8B:  hidden=4096, intermediate=14336, heads=32/8, head_dim=128
#   Llama-3-70B: hidden=8192, intermediate=28672, heads=64/8, head_dim=128
#   Qwen3-8B:    hidden=4096, intermediate=12288, heads=32/8, head_dim=128
_CONFIGS: list[_ModelTP] = [
    # ── Llama-3-8B ──────────────────────────────────────────────────────────
    _ModelTP("llama3-8b  TP=1", 4096, 14336, 4096, 1024),
    _ModelTP("llama3-8b  TP=2", 4096, 7168, 2048, 512),
    _ModelTP("llama3-8b  TP=4", 4096, 3584, 1024, 256),
    # ── Qwen3-8B ────────────────────────────────────────────────────────────
    _ModelTP("qwen3-8b   TP=1", 4096, 12288, 4096, 1024),
    _ModelTP("qwen3-8b   TP=2", 4096, 6144, 2048, 512),
    _ModelTP("qwen3-8b   TP=4", 4096, 3072, 1024, 256),
    # ── Llama-3-70B ─────────────────────────────────────────────────────────
    _ModelTP("llama3-70b TP=4", 8192, 7168, 2048, 256),
    _ModelTP("llama3-70b TP=8", 8192, 3584, 1024, 128),
]

# Max-rank values to cover — N in the shrink key is stack_num * max_rank.
_MAX_RANKS = [16, 32, 64, 128]


def _sweep_shrink(cfg: _ModelTP, max_rank: int) -> None:
    rank = max_rank  # tune at full rank so the K-loop is fully exercised
    # Attention shrink
    tune_shrink(in_dim=cfg.hidden, stack_num=3, rank=rank, max_rank=max_rank)
    tune_shrink(in_dim=cfg.q_per_tp, stack_num=1, rank=rank, max_rank=max_rank)
    # MLP shrink
    tune_shrink(in_dim=cfg.hidden, stack_num=2, rank=rank, max_rank=max_rank)
    tune_shrink(
        in_dim=cfg.intermediate_per_tp, stack_num=1, rank=rank, max_rank=max_rank
    )


def _sweep_expand(cfg: _ModelTP, max_rank: int) -> None:
    # Clear in-process cache so the autotuner sweeps all configs fresh
    # rather than reusing entries loaded from the on-disk JSON.
    for k in _lora_expand_kernel, _lora_qkv_expand_kernel, _lora_gate_up_expand_kernel:
        k.cache.clear()
    rank = max_rank
    # o_proj / down_proj
    tune_expand(out_dim=cfg.hidden, max_rank=max_rank, rank=rank)
    # QKV
    tune_qkv(
        q_per_tp=cfg.q_per_tp,
        kv_per_tp=cfg.kv_per_tp,
        max_rank=max_rank,
        rank=rank,
    )
    # gate/up
    tune_gate_up(
        intermediate_per_tp=cfg.intermediate_per_tp,
        max_rank=max_rank,
        rank=rank,
    )


def main() -> int:
    total_shrink = len(_CONFIGS) * len(_MAX_RANKS)
    total_expand = total_shrink
    done = 0

    for max_rank in _MAX_RANKS:
        for cfg in _CONFIGS:
            done += 1
            print(f"\n[{done}/{total_shrink}] shrink  {cfg.name}  max_rank={max_rank}")
            _sweep_shrink(cfg, max_rank)

    done = 0
    for max_rank in _MAX_RANKS:
        for cfg in _CONFIGS:
            done += 1
            print(f"\n[{done}/{total_expand}] expand  {cfg.name}  max_rank={max_rank}")
            _sweep_expand(cfg, max_rank)

    print("\n=== Saving caches ===")
    for kern in (
        _lora_shrink_kernel,
        _lora_expand_kernel,
        _lora_qkv_expand_kernel,
        _lora_gate_up_expand_kernel,
    ):
        path = save_kernel_cache(kern)
        print(f"  wrote {path}  ({len(kern.cache)} entries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
