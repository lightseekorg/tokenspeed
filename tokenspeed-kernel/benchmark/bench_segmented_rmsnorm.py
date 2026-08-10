"""Benchmark one segmented RMSNorm launch against five RMSNorms plus cat."""

from __future__ import annotations

import argparse

import torch
import triton.testing
from tokenspeed_kernel.ops.layernorm.triton import rmsnorm, segmented_rmsnorm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    print("tokens,unfused_us,fused_us,speedup")
    for tokens in args.tokens:
        x = torch.randn(
            tokens,
            args.segments,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        weight = torch.randn(
            args.segments,
            args.hidden_size,
            device=device,
            dtype=torch.float32,
        )

        def unfused() -> torch.Tensor:
            return torch.cat(
                [
                    rmsnorm(x[:, index, :].contiguous(), weight[index], eps)
                    for index in range(args.segments)
                ],
                dim=-1,
            )

        def fused() -> torch.Tensor:
            return segmented_rmsnorm(x, weight, eps).flatten(-2)

        expected = unfused()
        actual = fused()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        unfused_us = triton.testing.do_bench(
            unfused, warmup=args.warmup, rep=args.rep
        )
        fused_us = triton.testing.do_bench(fused, warmup=args.warmup, rep=args.rep)
        print(
            f"{tokens},{unfused_us:.3f},{fused_us:.3f},"
            f"{unfused_us / fused_us:.3f}"
        )


if __name__ == "__main__":
    main()
