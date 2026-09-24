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

"""Cold-cache benchmark for the Kimi K3 packed latent input projection."""

from __future__ import annotations

import argparse

import tokenspeed_triton.testing
import torch
from tokenspeed_kernel.ops.moe.triton.latent_input import triton_latent_input_packed
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_mediumm import (
    launch_gluon_latent_input_mediumm_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_prefill import (
    launch_gluon_latent_input_largem_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_small_batch import (
    launch_gluon_latent_input_small_batch_gfx950,
)


def _run(variant: str, hidden: torch.Tensor, packed: torch.Tensor) -> None:
    weights = packed.split((896, 3584, 1536))
    if variant == "triton":
        triton_latent_input_packed(hidden, *weights, gate_clamp=4.0, up_clamp=None)
    elif variant == "gluon_small":
        launch_gluon_latent_input_small_batch_gfx950(
            hidden, *weights, packed, beta=4.0, linear_beta=None
        )
    elif variant == "gluon_medium":
        launch_gluon_latent_input_mediumm_gfx950(
            hidden, *weights, packed, beta=4.0, linear_beta=None
        )
    else:
        launch_gluon_latent_input_largem_gfx950(
            hidden, *weights, packed, beta=4.0, linear_beta=None
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[256, 512, 1024, 2048, 3072]
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=24)
    parser.add_argument(
        "--variant",
        choices=("triton", "gluon_small", "gluon_medium", "gluon_large"),
        default="gluon_medium",
    )
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    if args.profile:
        if len(args.tokens) != 1:
            parser.error("--profile requires exactly one --tokens value")
        m = args.tokens[0]
        # Eight distinct 86 MB weights exceed MI350X's 256 MB Infinity Cache.
        # Each traced launch sees a different weight bank.
        banks = [
            (
                torch.randn((m, 7168), dtype=torch.bfloat16, device="cuda") * 0.002,
                torch.randn((6016, 7168), dtype=torch.bfloat16, device="cuda") * 0.002,
            )
            for _ in range(8)
        ]
        _run(args.variant, *banks[0])
        torch.cuda.synchronize()
        for iteration in range(args.profile_iterations):
            _run(args.variant, *banks[iteration % len(banks)])
        torch.cuda.synchronize()
        print(f"profiled M={m} variant={args.variant} rotations={len(banks)}")
        return

    packed = torch.randn((6016, 7168), dtype=torch.bfloat16, device="cuda") * 0.002
    variants = ("triton", "gluon_small", "gluon_medium", "gluon_large")
    print(
        f"device={torch.cuda.get_device_name()} "
        "method=tokenspeed_triton.testing.do_bench cache_flush=256MiB"
    )
    for m in args.tokens:
        hidden = torch.randn((m, 7168), dtype=torch.bfloat16, device="cuda") * 0.002
        for name in variants:
            ms = tokenspeed_triton.testing.do_bench(
                lambda: _run(name, hidden, packed),
                warmup=25,
                rep=80,
                quantiles=[0.5, 0.2, 0.8],
            )[0]
            tflops = 2 * m * 6016 * 7168 / (ms * 1e9)
            print(f"M={m:4d} {name:12s} {ms * 1000:9.2f} us {tflops:8.1f} TFLOP/s")


if __name__ == "__main__":
    main()
