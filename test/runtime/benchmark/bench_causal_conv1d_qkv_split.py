"""Benchmark: causal_conv1d + fused_qkv_split (sequential) vs each kernel alone.

Goal: measure what fraction of the sequential pipeline the QKV split (b0)
occupies. If split contributes a meaningful fraction (>~20%) at large T,
a fused kernel that eliminates the conv_out staging buffer would be worth
building.

Run:
  python/.venv/bin/python test/runtime/benchmark/bench_causal_conv1d_qkv_split.py

Config matches Qwen3.5 GDN prefill (nq=nk=nv=16, h=128, width=4, silu).
"""

from __future__ import annotations

import sys

import torch
import triton

sys.path.insert(
    0,
    "python",
)
sys.path.insert(
    0,
    "tokenspeed-kernel/python",
)

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
)
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Qwen3.5 GDN config (single-GPU, no TP)
NQ = NK = NV = 16
HEAD_DIM = 128
CONV_DIM = (NQ + NK + NV) * HEAD_DIM  # 3072 for Qwen3.5
CONV_WIDTH = 4  # kernel width (3 state tokens + 1)

WARMUP = 50
REPEAT = 200


def _make_conv_inputs(seq_len: int):
    """Single-sequence prefill inputs for causal_conv1d_fn."""
    # x: (dim, seq_len) channel-last → stride(0)=1, stride(1)=dim
    x = torch.randn(CONV_DIM, seq_len, dtype=DTYPE, device=DEVICE)
    x = x.as_strided(x.shape, (1, CONV_DIM))  # channel-last

    weight = torch.randn(CONV_DIM, CONV_WIDTH, dtype=DTYPE, device=DEVICE)

    num_cache_lines = 1
    conv_states = torch.zeros(
        num_cache_lines, CONV_DIM, CONV_WIDTH - 1, dtype=DTYPE, device=DEVICE
    )
    cache_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.zeros(1, dtype=torch.bool, device=DEVICE)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)

    return x, weight, conv_states, cache_indices, has_initial_state, query_start_loc


def bench_conv1d(seq_len: int) -> float:
    x, weight, conv_states, cache_indices, has_initial_state, query_start_loc = (
        _make_conv_inputs(seq_len)
    )

    def fn():
        causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
        )

    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REPEAT, return_mode="median")


def bench_split(seq_len: int) -> float:
    mixed_qkv = torch.randn(seq_len, CONV_DIM, dtype=DTYPE, device=DEVICE)

    def fn():
        fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=NQ,
            num_k_heads=NK,
            num_v_heads=NV,
            head_q=HEAD_DIM,
            head_k=HEAD_DIM,
            head_v=HEAD_DIM,
        )

    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REPEAT, return_mode="median")


def bench_sequential(seq_len: int) -> float:
    """Measure conv1d + split back-to-back (as in current hybrid_linear_attn.py)."""
    x, weight, conv_states, cache_indices, has_initial_state, query_start_loc = (
        _make_conv_inputs(seq_len)
    )

    def fn():
        conv_out = causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
        )
        # replicate hybrid_linear_attn.py:1044 post-processing
        mixed_qkv = conv_out.transpose(0, 1)[:seq_len].contiguous()
        fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=NQ,
            num_k_heads=NK,
            num_v_heads=NV,
            head_q=HEAD_DIM,
            head_k=HEAD_DIM,
            head_v=HEAD_DIM,
        )

    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REPEAT, return_mode="median")


def bench_fused(seq_len: int) -> float:
    """Measure fused causal_conv1d + QKV split in one kernel."""
    import sys

    sys.path.insert(0, "tokenspeed-kernel/python")
    from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
        causal_conv1d_qkv_split_gdn_prefill,
    )

    x, weight, conv_states, cache_indices, has_initial_state, query_start_loc = (
        _make_conv_inputs(seq_len)
    )
    seq_lens_cpu = [seq_len]

    def fn():
        causal_conv1d_qkv_split_gdn_prefill(
            x,
            weight,
            None,
            conv_states,
            query_start_loc,
            seq_lens_cpu,
            num_q_heads=NQ,
            num_k_heads=NK,
            num_v_heads=NV,
            head_q=HEAD_DIM,
            head_k=HEAD_DIM,
            head_v=HEAD_DIM,
            total_seq_len=seq_len,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
        )

    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REPEAT, return_mode="median")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    props = torch.cuda.get_device_properties(0)
    l2_mb = props.L2_cache_size / (1024**2)
    sm = f"{props.major}{props.minor}"
    print(f"GPU: {props.name}  L2: {l2_mb:.0f} MB  sm_{sm}")
    print(f"dtype={DTYPE}  conv_dim={CONV_DIM}  nq={NQ} nk={NK} nv={NV} h={HEAD_DIM}")
    print()

    header = (
        f"{'T':>6}  {'conv1d':>9}  {'split(b0)':>9}  {'seq(c+s)':>9}"
        f"  {'fused':>9}  {'gain%':>7}  {'conv_out':>9}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in [512, 2048, 4096, 8192, 16384]:
        conv_out_mb = CONV_DIM * seq_len * 2 // (1024**2)

        t_conv = bench_conv1d(seq_len)
        t_split = bench_split(seq_len)
        t_seq = bench_sequential(seq_len)
        t_fused = bench_fused(seq_len)

        gain_pct = (t_seq - t_fused) / t_seq * 100

        print(
            f"{seq_len:>6}  {t_conv*1e3:>7.1f}µs  {t_split*1e3:>7.1f}µs"
            f"  {t_seq*1e3:>7.1f}µs  {t_fused*1e3:>7.1f}µs"
            f"  {gain_pct:>6.1f}%  {conv_out_mb:>7d}MB"
        )

    print()
    print("gain% = (seq - fused) / seq  (positive = fused faster)")
    print(f"L2 capacity: {l2_mb:.0f} MB")


if __name__ == "__main__":
    main()
