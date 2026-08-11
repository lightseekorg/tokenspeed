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

"""Offline autotune driver for the LoRA Triton kernels.

Builds synthetic ``LoraBatchInfo`` batches for a few representative
segment shapes, calls each kernel once (triggering ``triton.autotune``
to benchmark all candidate configs and pick the fastest per ``(N, K)``
key), and then writes the picked configs to JSON via
:func:`tokenspeed_kernel.ops.lora.triton.tuning.save_kernel_cache`.

Usage::

    python -m tokenspeed_kernel.ops.lora.triton.tune \\
        --hidden 4096 --intermediate 12288 \\
        --q-per-tp 2048 --kv-per-tp 1024 \\
        --rank 16 --max-rank 64 --tp-size 2

The defaults match Qwen3-8B at attn_tp_size=2.  Shapes only affect which
``(N, K)`` keys get tuned; the actual launch parameters are independent
of which model the cache is shipped against.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.lora.triton.lora_expand import (
    _lora_expand_kernel,
    lora_expand_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_gate_up_expand import (
    _lora_gate_up_expand_kernel,
    lora_gate_up_expand_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_qkv_expand import (
    _lora_qkv_expand_kernel,
    lora_qkv_expand_fwd,
)
from tokenspeed_kernel.ops.lora.triton.lora_shrink import (
    _lora_shrink_kernel,
    lora_shrink_fwd,
)
from tokenspeed_kernel.ops.lora.triton.tuning import save_kernel_cache

logger = logging.getLogger(__name__)


@dataclass
class _BatchInfo:
    """Minimal stand-in for ``runtime.lora.lora_manager.LoraBatchInfo``."""

    bs: int
    max_len: int
    seg_lens: torch.Tensor
    seg_indptr: torch.Tensor
    weight_indices: torch.Tensor
    lora_ranks: torch.Tensor
    scalings: torch.Tensor
    permutation: torch.Tensor | None = None


def _make_batch(
    s_per_seg: int, n_segs: int, rank: int, device: str = "cuda"
) -> _BatchInfo:
    seg_lens = torch.full((n_segs,), s_per_seg, dtype=torch.int32, device=device)
    seg_indptr = torch.tensor(
        [i * s_per_seg for i in range(n_segs + 1)], dtype=torch.int32, device=device
    )
    # weight_indices: route every segment to real adapter slot 0.
    weight_indices = torch.zeros(n_segs, dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([rank], dtype=torch.int32, device=device)
    scalings = torch.tensor([1.0], dtype=torch.float32, device=device)
    return _BatchInfo(
        bs=n_segs,
        max_len=s_per_seg,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


def tune_shrink(*, in_dim: int, stack_num: int, rank: int, max_rank: int) -> None:
    """Drive ``_lora_shrink_kernel`` for one ``(stack_num, in_dim)`` shape.

    Uses a decode-shaped batch (``bs=32, max_len=1``) because that is where
    LoRA latency dominates the e2e (every decode step pays the kernel cost;
    prefill is amortized).  Tuning at prefill shapes picks block tiles that
    waste threads at decode-time.
    """
    device = "cuda"
    dtype = torch.bfloat16
    n_segs = 32
    s_per_seg = 1
    s = n_segs * s_per_seg
    x = torch.randn((s, in_dim), device=device, dtype=dtype)
    weights = torch.randn((2, stack_num * max_rank, in_dim), device=device, dtype=dtype)
    bi = _make_batch(s_per_seg, n_segs, rank=rank, device=device)
    lora_shrink_fwd(x, weights, bi, stack_num=stack_num)
    torch.cuda.synchronize()
    print(
        f"  shrink in_dim={in_dim} stack={stack_num}  →  best={_lora_shrink_kernel.best_config}"
    )


def tune_expand(*, out_dim: int, max_rank: int, rank: int) -> None:
    device = "cuda"
    dtype = torch.bfloat16
    n_segs = 32
    s_per_seg = 1
    s = n_segs * s_per_seg
    x = torch.randn((s, max_rank), device=device, dtype=dtype)
    weights = torch.randn((2, out_dim, max_rank), device=device, dtype=dtype)
    bi = _make_batch(s_per_seg, n_segs, rank=rank, device=device)
    out = torch.zeros((s, out_dim), device=device, dtype=dtype)
    lora_expand_fwd(x, weights, bi, base_output=out)
    torch.cuda.synchronize()
    print(
        f"  expand out_dim={out_dim} R={max_rank}  →  best={_lora_expand_kernel.best_config}"
    )


def tune_qkv(*, q_per_tp: int, kv_per_tp: int, max_rank: int, rank: int) -> None:
    device = "cuda"
    dtype = torch.bfloat16
    n_segs = 32
    s_per_seg = 1
    s = n_segs * s_per_seg
    x = torch.randn((s, 3 * max_rank), device=device, dtype=dtype)
    out_dim = q_per_tp + 2 * kv_per_tp
    weights = torch.randn((2, out_dim, max_rank), device=device, dtype=dtype)
    max_qkv = max(q_per_tp, kv_per_tp)
    output_offset = torch.tensor(
        [0, q_per_tp, q_per_tp + kv_per_tp, q_per_tp + 2 * kv_per_tp],
        dtype=torch.int32,
        device=device,
    )
    bi = _make_batch(s_per_seg, n_segs, rank=rank, device=device)
    out = torch.zeros((s, out_dim), device=device, dtype=dtype)
    lora_qkv_expand_fwd(x, weights, bi, output_offset, max_qkv, base_output=out)
    torch.cuda.synchronize()
    print(
        f"  qkv_expand max_qkv={max_qkv} R={max_rank}  →  best={_lora_qkv_expand_kernel.best_config}"
    )


def tune_gate_up(*, intermediate_per_tp: int, max_rank: int, rank: int) -> None:
    device = "cuda"
    dtype = torch.bfloat16
    n_segs = 32
    s_per_seg = 1
    s = n_segs * s_per_seg
    x = torch.randn((s, 2 * max_rank), device=device, dtype=dtype)
    weights = torch.randn(
        (2, 2 * intermediate_per_tp, max_rank), device=device, dtype=dtype
    )
    bi = _make_batch(s_per_seg, n_segs, rank=rank, device=device)
    out = torch.zeros((s, 2 * intermediate_per_tp), device=device, dtype=dtype)
    lora_gate_up_expand_fwd(x, weights, bi, intermediate_per_tp, base_output=out)
    torch.cuda.synchronize()
    print(
        f"  gate_up_expand out={intermediate_per_tp} R={max_rank}  →  best={_lora_gate_up_expand_kernel.best_config}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument(
        "--intermediate",
        type=int,
        default=12288,
        help="Full (un-sharded) intermediate_size",
    )
    p.add_argument("--q-per-tp", type=int, default=2048)
    p.add_argument("--kv-per-tp", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-rank", type=int, default=64)
    p.add_argument("--tp-size", type=int, default=2)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    intermediate_per_tp = args.intermediate // args.tp_size

    print("=== Tuning shrink (lora_shrink) ===")
    # Attention shrink: stack=3 (QKV) on hidden, stack=1 (o) on q_per_tp.
    tune_shrink(in_dim=args.hidden, stack_num=3, rank=args.rank, max_rank=args.max_rank)
    tune_shrink(
        in_dim=args.q_per_tp, stack_num=1, rank=args.rank, max_rank=args.max_rank
    )
    # MLP shrink: stack=2 (gate/up) on hidden, stack=1 (down) on intermediate_per_tp.
    tune_shrink(in_dim=args.hidden, stack_num=2, rank=args.rank, max_rank=args.max_rank)
    tune_shrink(
        in_dim=intermediate_per_tp, stack_num=1, rank=args.rank, max_rank=args.max_rank
    )

    print("\n=== Tuning expand (lora_expand) ===")
    # o_proj uses lora_expand directly (out_dim = hidden).
    tune_expand(out_dim=args.hidden, max_rank=args.max_rank, rank=args.rank)
    # down_proj also uses lora_expand (out_dim = hidden).
    # Same shape — autotune cache hit on the second call.

    print("\n=== Tuning qkv_expand (lora_qkv_expand) ===")
    tune_qkv(
        q_per_tp=args.q_per_tp,
        kv_per_tp=args.kv_per_tp,
        max_rank=args.max_rank,
        rank=args.rank,
    )

    print("\n=== Tuning gate_up_expand (lora_gate_up_expand) ===")
    tune_gate_up(
        intermediate_per_tp=intermediate_per_tp,
        max_rank=args.max_rank,
        rank=args.rank,
    )

    print("\n=== Saving caches ===")
    for kern in (
        _lora_shrink_kernel,
        _lora_expand_kernel,
        _lora_qkv_expand_kernel,
        _lora_gate_up_expand_kernel,
    ):
        path = save_kernel_cache(kern)
        print(f"  wrote {path} ({len(kern.cache)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
