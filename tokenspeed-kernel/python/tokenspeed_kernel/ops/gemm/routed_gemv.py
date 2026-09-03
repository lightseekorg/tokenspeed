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

"""Measured decode-GEMV routing for the served models' projection shapes.

Every entry in ``MEASURED_ROUTE`` was measured at the exact ``(M, N, K)`` the
serving decode path hands ``decode_gemv``. The tuner cycles eight weight copies
so the L2 never holds the operand between calls, matching serving where each
layer streams a different weight. Hot-cache numbers ranked backends wrongly;
cold-L2 reproduces serving per-shape times within ~5%.
``test/gemm_tuning/tune_route.py`` reproduces the sweep.

A backend earns an entry only by beating the incumbent selection by at least
4% -- above measurement noise, so a noise-level lead does not become a
maintenance obligation. Shapes not listed keep the selection they had
(rowcta at M == 1, torch.mm otherwise). The table is data, not policy:
re-run the sweep on new hardware or after a kernel change and replace the
literals wholesale.
"""

from __future__ import annotations

import functools
import threading
from types import MappingProxyType

import torch
from tokenspeed_kernel.ops.gemm.triton_gemv import _torch_decode_gemv
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# (m, n, k) -> backend; immutable so the registry's import-time view and the
# wrappers' per-call view cannot diverge. Measured; see module docstring.
MEASURED_ROUTE: MappingProxyType[tuple[int, int, int], str] = MappingProxyType(
    {
        # K3 drafters at TP16; shapes read from the model code, not configs.
        # both q_b  N=768 K=1536
        (2, 768, 1536): "skinny",
        (3, 768, 1536): "skinny",
        (4, 768, 1536): "skinny",
        (5, 768, 1536): "skinny",
        (6, 768, 1536): "skinny",
        (7, 768, 1536): "skinny",
        (8, 768, 1536): "skinny",
        (9, 768, 1536): "skinny",
        (10, 768, 1536): "skinny",
        (11, 768, 1536): "skinny",
        (12, 768, 1536): "skinny",  # re-swept: 1.49x vs cuBLAS; edges tgv by ~1%
        (13, 768, 1536): "tgv",
        (14, 768, 1536): "tgv",
        (15, 768, 1536): "tgv",
        (16, 768, 1536): "tgv",
        (17, 768, 1536): "tgv",
        (18, 768, 1536): "tgv",
        (19, 768, 1536): "tgv",
        (20, 768, 1536): "tgv",
        (21, 768, 1536): "tgv",
        (22, 768, 1536): "tgv",
        (23, 768, 1536): "tgv",
        (24, 768, 1536): "tgv",
        (25, 768, 1536): "tgv",
        (26, 768, 1536): "tgv",
        (27, 768, 1536): "tgv",
        (28, 768, 1536): "tgv",
        (17, 2304, 7168): "tgv",
        (18, 2304, 7168): "tgv",
        (19, 2304, 7168): "tgv",
        (21, 2304, 7168): "tgv",
        (22, 2304, 7168): "tgv",
        (24, 2304, 7168): "tgv",
        (32, 2304, 7168): "tgv",
        (29, 768, 1536): "tgv",
        (30, 768, 1536): "tgv",
        (31, 768, 1536): "tgv",
        (32, 768, 1536): "tgv",
        # dspark gate_up  N=1792 K=7168
        (2, 1792, 7168): "skinny",
        (3, 1792, 7168): "skinny",
        (4, 1792, 7168): "skinny",
        # dspark fused_qkv_a (2112x7168): handled by dsv3_fused_a_gemm upstream.
        # eagle3 fused_qkv_a  N=2112 K=14336
        (1, 2112, 14336): "skinny",
        (2, 2112, 14336): "skinny",
        (3, 2112, 14336): "skinny",
        # eagle3 gate_up  N=2304 K=7168
        (2, 2304, 7168): "skinny",
        (3, 2304, 7168): "skinny",
        # both o_proj  N=7168 K=512
        (1, 7168, 512): "tgv",
        # dspark down  N=7168 K=896
        (1, 7168, 896): "tgv",
        (2, 7168, 896): "tgv",
        (3, 7168, 896): "tgv",
        (4, 7168, 896): "tgv",
        (5, 7168, 896): "tgv",
        (6, 7168, 896): "tgv",
        (7, 7168, 896): "tgv",
        (8, 7168, 896): "tgv",
        # eagle3 down  N=7168 K=1152
        (1, 7168, 1152): "tgv",
        (2, 7168, 1152): "tgv",
        # eagle3 fc: (7168, 21504) was the unsharded width; TP16 shard unmeasured.
        # M values the 1/2/4/8 sweep skipped; GB200 reproduced 39/40 verdicts.
        # n1152_k1536  N=1152 K=1536
        (9, 1152, 1536): "skinny",
        (10, 1152, 1536): "tgv",
        (11, 1152, 1536): "tgv",
        (12, 1152, 1536): "tgv",
        (13, 1152, 1536): "tgv",
        (14, 1152, 1536): "tgv",
        (15, 1152, 1536): "tgv",
        (16, 1152, 1536): "tgv",
        # shared gate_up shard  N=1536 K=7168
        (3, 1536, 7168): "skinny",
        # MLA q_b  N=2304 K=1536
        (5, 2304, 1536): "tgv",
        (6, 2304, 1536): "tgv",
        (7, 2304, 1536): "tgv",
        # n2880_k7168  N=2880 K=7168
        (12, 2880, 7168): "tgv",  # 1.095-1.102x, three runs
        (13, 2880, 7168): "tgv",
        (15, 2880, 7168): "tgv",
        # MLA fused qkv_a + gate  N=3648 K=7168
        (3, 3648, 7168): "skinny",
        # KDA in_proj  N=6288 K=7168
        (3, 6288, 7168): "tgv",  # re-swept: 1.12x vs cuBLAS; edges skinny by ~1%
        (5, 6288, 7168): "tgv",
        (6, 6288, 7168): "tgv",
        # shared down shard  N=7168 K=768
        (3, 7168, 768): "tgv",
        (5, 7168, 768): "tgv",
        (6, 7168, 768): "tgv",
        (7, 7168, 768): "tgv",
        (1, 3584, 7168): "skinny",  # MoE latent down-proj, 92 calls/step
        # KDA o_proj shard  N=7168 K=1536
        (1, 7168, 1536): "tgv",
        (2, 7168, 1536): "tgv",
        (4, 7168, 1536): "tgv",
        (8, 7168, 1536): "tgv",
        (1, 1536, 7168): "skinny",
        (2, 1536, 7168): "skinny",
        (4, 1536, 7168): "skinny",
        (5, 1536, 7168): "skinny",
        (6, 1536, 7168): "skinny",
        # These waive the 4% margin: cuBLAS needs a split-K reduce, tgv does not.
        (7, 1536, 7168): "tgv",
        (8, 1536, 7168): "tgv",
        (5, 3584, 7168): "tgv",
        (6, 3584, 7168): "tgv",
        (7, 3584, 7168): "tgv",
        (8, 3584, 7168): "tgv",
        (1, 7168, 768): "tgv",
        (2, 7168, 768): "tgv",
        (4, 7168, 768): "tgv",
        (8, 7168, 768): "tgv",
        (1, 6288, 7168): "skinny",  # KDA in_proj (qkvgfab), 69 calls/step
        (1, 3648, 7168): "skinny",  # MLA fused qkv_a + gate, 24 calls/step
        # (1, 2304, 1536) mla_q_b stays on rowcta: 2.48 vs skinny 2.53.
        # M > 1 (small batches, speculative verify) vs the cublas incumbent.
        (2, 3584, 7168): "skinny",  # 9.20 vs 11.67 (1.27x)
        # TP8 DSpark drafter, shapes observed at the launch point on GB300.
        # eagle3 drafter TP8 widths; nothing past M=4 clears the margin.
        (2, 4608, 7168): "tgv",  # 1.10x
        (1, 7168, 2304): "tgv",  # 1.10x
        (2, 7168, 2304): "tgv",  # 1.10x
        (2, 1536, 1536): "skinny",  # 2.18x
        (3, 1536, 1536): "skinny",  # 1.95x
        (4, 1536, 1536): "skinny",  # 1.83x
        (5, 1536, 1536): "skinny",  # 1.68x
        (6, 1536, 1536): "skinny",  # 1.63x
        (7, 1536, 1536): "skinny",  # 1.55x
        (8, 1536, 1536): "tgv",  # 1.54x
        (1, 7168, 1024): "tgv",  # 1.24x
        (2, 7168, 1024): "tgv",  # 1.16x
        (3, 7168, 1024): "tgv",  # 1.15x
        (4, 7168, 1024): "tgv",  # 1.16x
        (5, 7168, 1024): "tgv",  # 1.15x
        (6, 7168, 1024): "tgv",  # 1.16x
        (7, 7168, 1024): "tgv",  # 1.15x
        (8, 7168, 1024): "tgv",  # 1.15x
        (1, 7168, 1792): "tgv",  # 1.13x
        (2, 7168, 1792): "tgv",  # 1.10x
        (4, 3584, 7168): "skinny",  # 10.38 vs 11.21 (1.08x)
        (2, 6288, 7168): "tgv",  # 15.29 vs 17.07 (1.12x)
        (4, 6288, 7168): "tgv",  # 15.25 vs 17.26 (1.13x)
        (8, 6288, 7168): "tgv",  # 15.38 vs 16.75 (1.09x)
        (2, 3648, 7168): "skinny",  # 9.33 vs 11.58 (1.24x)
        (4, 3648, 7168): "skinny",  # 10.40 vs 11.51 (1.11x)
        (8, 3648, 7168): "tgv",  # 10.65 vs 11.47 (1.08x)
        (2, 2304, 1536): "skinny",  # 2.52 vs 4.97 (1.97x)
        (4, 2304, 1536): "tgv",  # 2.86 vs 4.58 (1.60x)
        (8, 2304, 1536): "tgv",  # 2.86 vs 4.45 (1.56x)
        # TP16 (GB200, sm100). The shapes were read off the decode_gemv
        # dispatch during a live bs=1 run rather than scaled from TP8: only
        # 1152x1536 halves, 3584x7168 is not TP-sharded at all, 2880x7168 has
        # no TP8 analogue, and TP8's largest entry (6288x7168) never reaches
        (17, 1152, 1536): "tgv",
        (18, 1152, 1536): "tgv",
        (19, 1152, 1536): "tgv",
        (20, 1152, 1536): "tgv",
        (21, 1152, 1536): "tgv",
        (22, 1152, 1536): "tgv",
        (23, 1152, 1536): "tgv",
        (24, 1152, 1536): "tgv",
        (25, 1152, 1536): "tgv",
        (26, 1152, 1536): "tgv",
        (27, 1152, 1536): "tgv",
        (28, 1152, 1536): "tgv",
        (29, 1152, 1536): "tgv",
        (30, 1152, 1536): "tgv",
        (31, 1152, 1536): "tgv",
        (32, 1152, 1536): "tgv",
        # decode_gemv here. Same >= 4% margin, same cold-L2 tuner.
        (3, 3584, 7168): "skinny",  # 9.54 vs 11.23 (1.18x)
        (1, 2880, 7168): "skinny",  # 7.13 vs rowcta 8.61 (1.21x)
        (2, 2880, 7168): "skinny",  # 7.95 vs 9.80 (1.23x)
        (3, 2880, 7168): "skinny",  # 8.24 vs 9.77 (1.19x)
        (4, 2880, 7168): "skinny",  # 9.32 vs 9.78 (1.05x)
        # 3584 and 2880 leave skinny at M >= 5: skinny inverts there
        # (15.85 vs cublas 11.06 at M=8, 3584x7168), so the win is not simply
        # "skinny for small M" and the boundary has to be measured per width.
        (2, 1152, 1536): "skinny",  # 1.99 vs 4.85 (2.44x)
        (3, 1152, 1536): "skinny",  # 2.10 vs 4.47 (2.13x)
        (4, 1152, 1536): "skinny",  # 2.24 vs 4.40 (1.97x)
        (5, 1152, 1536): "skinny",  # 2.23 vs 4.40 (1.98x)
        (6, 1152, 1536): "skinny",  # 2.35 vs 4.42 (1.89x)
        (7, 1152, 1536): "skinny",  # 2.59 vs 4.45 (1.72x)
        (8, 1152, 1536): "skinny",  # 2.72 vs 4.46 (1.64x)
        # (1, 1152, 1536) stays on rowcta: 1.907 vs 1.943, inside the margin.
        # Qwen3.8-Flash-Next (GB200, sm100), BF16 dense decode linears.
        # Shapes were observed in FP8/NVFP4 MTP3/MTP7 serving, then every
        # exact M=1..32 was swept cold-L2. TP1 had no successful forward.
        # Entries require unanimous backend selection and >= 4% gain.
        # TP2/TP4 shared shapes (six independent sweeps).
        # 512x2560
        (3, 512, 2560): "ll_bf16",  # 2.147 vs 4.461 us (2.08x)
        (4, 512, 2560): "ll_bf16",  # 2.163 vs 4.468 us (2.07x)
        (5, 512, 2560): "ll_bf16",  # 2.149 vs 4.564 us (2.12x)
        (6, 512, 2560): "ll_bf16",  # 2.167 vs 4.505 us (2.08x)
        (7, 512, 2560): "ll_bf16",  # 2.170 vs 4.646 us (2.14x)
        (8, 512, 2560): "ll_bf16",  # 2.180 vs 4.598 us (2.11x)
        (9, 512, 2560): "ll_bf16",  # 2.165 vs 4.543 us (2.10x)
        (10, 512, 2560): "ll_bf16",  # 2.162 vs 4.427 us (2.05x)
        (11, 512, 2560): "ll_bf16",  # 2.161 vs 4.438 us (2.05x)
        (12, 512, 2560): "ll_bf16",  # 2.165 vs 4.475 us (2.07x)
        (13, 512, 2560): "ll_bf16",  # 2.150 vs 4.421 us (2.06x)
        (14, 512, 2560): "ll_bf16",  # 2.155 vs 4.475 us (2.08x)
        (15, 512, 2560): "ll_bf16",  # 2.159 vs 4.469 us (2.07x)
        (16, 512, 2560): "ll_bf16",  # 2.159 vs 4.658 us (2.16x)
        (17, 512, 2560): "ll_bf16",  # 2.260 vs 4.063 us (1.80x)
        (18, 512, 2560): "ll_bf16",  # 2.268 vs 4.063 us (1.79x)
        (19, 512, 2560): "ll_bf16",  # 2.268 vs 4.065 us (1.79x)
        (20, 512, 2560): "ll_bf16",  # 2.264 vs 4.065 us (1.80x)
        (21, 512, 2560): "ll_bf16",  # 2.271 vs 4.071 us (1.79x)
        (22, 512, 2560): "ll_bf16",  # 2.271 vs 4.067 us (1.79x)
        (23, 512, 2560): "ll_bf16",  # 2.287 vs 4.048 us (1.77x)
        (24, 512, 2560): "ll_bf16",  # 2.290 vs 4.017 us (1.75x)
        (25, 512, 2560): "ll_bf16",  # 2.627 vs 4.066 us (1.55x)
        (26, 512, 2560): "ll_bf16",  # 2.618 vs 4.078 us (1.56x)
        (27, 512, 2560): "ll_bf16",  # 2.615 vs 4.043 us (1.55x)
        (28, 512, 2560): "ll_bf16",  # 2.617 vs 4.062 us (1.55x)
        (29, 512, 2560): "ll_bf16",  # 2.611 vs 4.002 us (1.53x)
        (30, 512, 2560): "ll_bf16",  # 2.621 vs 4.067 us (1.55x)
        (31, 512, 2560): "ll_bf16",  # 2.617 vs 4.054 us (1.55x)
        (32, 512, 2560): "ll_bf16",  # 2.611 vs 4.598 us (1.76x)
        # 640x2560
        (2, 640, 2560): "skinny",  # 2.130 vs 4.397 us (2.06x)
        (3, 640, 2560): "skinny",  # 2.616 vs 4.415 us (1.69x)
        (5, 640, 2560): "ll_bf16",  # 2.805 vs 4.535 us (1.62x)
        (6, 640, 2560): "ll_bf16",  # 2.819 vs 4.543 us (1.61x)
        (7, 640, 2560): "ll_bf16",  # 2.840 vs 4.526 us (1.59x)
        (8, 640, 2560): "ll_bf16",  # 2.859 vs 4.559 us (1.60x)
        (9, 640, 2560): "ll_bf16",  # 3.595 vs 4.545 us (1.26x)
        (10, 640, 2560): "ll_bf16",  # 3.615 vs 4.561 us (1.26x)
        (11, 640, 2560): "ll_bf16",  # 3.619 vs 4.556 us (1.26x)
        (12, 640, 2560): "ll_bf16",  # 3.633 vs 4.552 us (1.25x)
        (13, 640, 2560): "ll_bf16",  # 3.661 vs 4.492 us (1.23x)
        (14, 640, 2560): "ll_bf16",  # 3.668 vs 4.572 us (1.25x)
        (15, 640, 2560): "ll_bf16",  # 3.699 vs 4.577 us (1.24x)
        (16, 640, 2560): "ll_bf16",  # 3.718 vs 4.623 us (1.24x)
        (17, 640, 2560): "ll_bf16",  # 3.287 vs 4.069 us (1.24x)
        (18, 640, 2560): "ll_bf16",  # 3.287 vs 4.064 us (1.24x)
        (19, 640, 2560): "ll_bf16",  # 3.281 vs 4.067 us (1.24x)
        (20, 640, 2560): "ll_bf16",  # 3.298 vs 4.066 us (1.23x)
        (21, 640, 2560): "ll_bf16",  # 3.314 vs 4.064 us (1.23x)
        (22, 640, 2560): "ll_bf16",  # 3.312 vs 4.068 us (1.23x)
        (23, 640, 2560): "ll_bf16",  # 3.338 vs 4.074 us (1.22x)
        (24, 640, 2560): "ll_bf16",  # 3.342 vs 4.071 us (1.22x)
        (25, 640, 2560): "ll_bf16",  # 3.352 vs 4.064 us (1.21x)
        (26, 640, 2560): "ll_bf16",  # 3.349 vs 4.062 us (1.21x)
        (27, 640, 2560): "ll_bf16",  # 3.343 vs 4.074 us (1.22x)
        (28, 640, 2560): "ll_bf16",  # 3.361 vs 4.062 us (1.21x)
        (29, 640, 2560): "ll_bf16",  # 3.370 vs 4.063 us (1.21x)
        (30, 640, 2560): "ll_bf16",  # 3.367 vs 4.065 us (1.21x)
        (31, 640, 2560): "ll_bf16",  # 3.399 vs 4.067 us (1.20x)
        (32, 640, 2560): "ll_bf16",  # 3.385 vs 4.674 us (1.38x)
        # 2560x2560
        (1, 2560, 2560): "ll_bf16",  # 3.072 vs 3.745 us (1.22x)
        (2, 2560, 2560): "ll_bf16",  # 3.062 vs 6.364 us (2.08x)
        (3, 2560, 2560): "ll_bf16",  # 3.179 vs 6.121 us (1.93x)
        (4, 2560, 2560): "ll_bf16",  # 3.125 vs 6.558 us (2.10x)
        (5, 2560, 2560): "ll_bf16",  # 3.189 vs 6.555 us (2.06x)
        (6, 2560, 2560): "ll_bf16",  # 3.145 vs 6.949 us (2.21x)
        (7, 2560, 2560): "ll_bf16",  # 3.216 vs 6.847 us (2.13x)
        (8, 2560, 2560): "ll_bf16",  # 3.213 vs 6.720 us (2.09x)
        (9, 2560, 2560): "ll_bf16",  # 4.144 vs 4.889 us (1.18x)
        (10, 2560, 2560): "ll_bf16",  # 4.085 vs 4.893 us (1.20x)
        (11, 2560, 2560): "ll_bf16",  # 4.134 vs 4.883 us (1.18x)
        (12, 2560, 2560): "ll_bf16",  # 4.033 vs 4.902 us (1.22x)
        (13, 2560, 2560): "ll_bf16",  # 4.175 vs 4.897 us (1.17x)
        (14, 2560, 2560): "ll_bf16",  # 4.159 vs 4.905 us (1.18x)
        (15, 2560, 2560): "ll_bf16",  # 4.241 vs 4.905 us (1.16x)
        (16, 2560, 2560): "ll_bf16",  # 4.236 vs 4.915 us (1.16x)
        (17, 2560, 2560): "ll_bf16",  # 4.007 vs 5.854 us (1.46x)
        (18, 2560, 2560): "ll_bf16",  # 4.019 vs 5.657 us (1.41x)
        (19, 2560, 2560): "ll_bf16",  # 4.019 vs 5.553 us (1.38x)
        (20, 2560, 2560): "ll_bf16",  # 4.018 vs 5.631 us (1.40x)
        (21, 2560, 2560): "ll_bf16",  # 4.027 vs 5.700 us (1.42x)
        (22, 2560, 2560): "ll_bf16",  # 4.036 vs 5.651 us (1.40x)
        (23, 2560, 2560): "ll_bf16",  # 4.037 vs 5.489 us (1.36x)
        (24, 2560, 2560): "ll_bf16",  # 4.042 vs 5.677 us (1.40x)
        (25, 2560, 2560): "ll_bf16",  # 4.038 vs 5.073 us (1.26x)
        (26, 2560, 2560): "ll_bf16",  # 4.061 vs 5.075 us (1.25x)
        (27, 2560, 2560): "ll_bf16",  # 4.058 vs 5.072 us (1.25x)
        (28, 2560, 2560): "ll_bf16",  # 4.032 vs 5.088 us (1.26x)
        (29, 2560, 2560): "ll_bf16",  # 4.073 vs 5.083 us (1.25x)
        (30, 2560, 2560): "ll_bf16",  # 4.068 vs 5.085 us (1.25x)
        (31, 2560, 2560): "ll_bf16",  # 4.058 vs 5.074 us (1.25x)
        (32, 2560, 2560): "ll_bf16",  # 4.106 vs 5.681 us (1.38x)
        # 12800x2560
        (2, 12800, 2560): "skinny",  # 11.489 vs 13.797 us (1.20x)
        (4, 12800, 2560): "ll_bf16",  # 13.113 vs 13.682 us (1.04x)
        # TP4-only shapes (three independent sweeps).
        # 320x2560
        (2, 320, 2560): "skinny",  # 1.969 vs 5.501 us (2.79x)
        (3, 320, 2560): "skinny",  # 2.104 vs 4.280 us (2.03x)
        (4, 320, 2560): "ll_bf16",  # 2.160 vs 4.214 us (1.95x)
        (5, 320, 2560): "ll_bf16",  # 2.159 vs 4.214 us (1.95x)
        (6, 320, 2560): "ll_bf16",  # 2.189 vs 4.286 us (1.96x)
        (7, 320, 2560): "ll_bf16",  # 2.183 vs 4.472 us (2.05x)
        (8, 320, 2560): "ll_bf16",  # 2.195 vs 4.481 us (2.04x)
        (9, 320, 2560): "ll_bf16",  # 2.211 vs 4.536 us (2.05x)
        (10, 320, 2560): "ll_bf16",  # 2.197 vs 4.554 us (2.07x)
        (11, 320, 2560): "ll_bf16",  # 2.203 vs 4.536 us (2.06x)
        (12, 320, 2560): "ll_bf16",  # 2.207 vs 4.517 us (2.05x)
        (13, 320, 2560): "ll_bf16",  # 2.197 vs 4.534 us (2.06x)
        (14, 320, 2560): "ll_bf16",  # 2.206 vs 4.528 us (2.05x)
        (15, 320, 2560): "ll_bf16",  # 2.203 vs 4.557 us (2.07x)
        (16, 320, 2560): "ll_bf16",  # 2.202 vs 4.534 us (2.06x)
        (17, 320, 2560): "ll_bf16",  # 2.180 vs 4.523 us (2.07x)
        (18, 320, 2560): "ll_bf16",  # 2.187 vs 4.527 us (2.07x)
        (19, 320, 2560): "ll_bf16",  # 2.210 vs 4.518 us (2.04x)
        (20, 320, 2560): "ll_bf16",  # 2.186 vs 4.523 us (2.07x)
        (21, 320, 2560): "ll_bf16",  # 2.169 vs 4.535 us (2.09x)
        (22, 320, 2560): "ll_bf16",  # 2.182 vs 4.520 us (2.07x)
        (23, 320, 2560): "ll_bf16",  # 2.204 vs 4.541 us (2.06x)
        (24, 320, 2560): "ll_bf16",  # 2.183 vs 4.540 us (2.08x)
        (25, 320, 2560): "ll_bf16",  # 2.183 vs 4.519 us (2.07x)
        (26, 320, 2560): "ll_bf16",  # 2.190 vs 4.522 us (2.06x)
        (27, 320, 2560): "ll_bf16",  # 2.195 vs 4.536 us (2.07x)
        (28, 320, 2560): "ll_bf16",  # 2.195 vs 4.536 us (2.07x)
        (29, 320, 2560): "ll_bf16",  # 2.188 vs 4.552 us (2.08x)
        (30, 320, 2560): "ll_bf16",  # 2.192 vs 4.578 us (2.09x)
        (31, 320, 2560): "ll_bf16",  # 2.212 vs 4.565 us (2.06x)
        (32, 320, 2560): "ll_bf16",  # 2.274 vs 4.557 us (2.00x)
        # 2560x160
        (1, 2560, 160): "tgv",  # 1.571 vs 1.878 us (1.20x)
        (2, 2560, 160): "tgv",  # 1.583 vs 2.747 us (1.74x)
        (3, 2560, 160): "tgv",  # 1.592 vs 2.744 us (1.72x)
        (4, 2560, 160): "tgv",  # 1.588 vs 2.747 us (1.73x)
        (5, 2560, 160): "tgv",  # 1.611 vs 2.748 us (1.71x)
        (6, 2560, 160): "tgv",  # 1.573 vs 1.961 us (1.25x)
        (7, 2560, 160): "tgv",  # 1.577 vs 1.912 us (1.21x)
        (8, 2560, 160): "tgv",  # 1.569 vs 1.887 us (1.20x)
        # 2560x1536
        (2, 2560, 1536): "ll_bf16",  # 2.439 vs 4.934 us (2.02x)
        (3, 2560, 1536): "ll_bf16",  # 2.458 vs 5.160 us (2.10x)
        (4, 2560, 1536): "ll_bf16",  # 2.458 vs 4.889 us (1.99x)
        (5, 2560, 1536): "ll_bf16",  # 2.473 vs 4.895 us (1.98x)
        (6, 2560, 1536): "ll_bf16",  # 2.493 vs 4.914 us (1.97x)
        (7, 2560, 1536): "ll_bf16",  # 2.504 vs 5.013 us (2.00x)
        (8, 2560, 1536): "ll_bf16",  # 2.538 vs 4.940 us (1.95x)
        (15, 2560, 1536): "tgv",  # 3.073 vs 3.253 us (1.06x)
        (17, 2560, 1536): "ll_bf16",  # 3.043 vs 4.374 us (1.44x)
        (18, 2560, 1536): "ll_bf16",  # 2.982 vs 4.459 us (1.50x)
        (19, 2560, 1536): "ll_bf16",  # 2.966 vs 4.418 us (1.49x)
        (20, 2560, 1536): "ll_bf16",  # 2.956 vs 4.205 us (1.42x)
        (21, 2560, 1536): "ll_bf16",  # 3.056 vs 4.185 us (1.37x)
        (22, 2560, 1536): "ll_bf16",  # 2.981 vs 4.215 us (1.41x)
        (23, 2560, 1536): "ll_bf16",  # 2.980 vs 4.215 us (1.41x)
        (24, 2560, 1536): "ll_bf16",  # 3.023 vs 4.255 us (1.41x)
        (25, 2560, 1536): "ll_bf16",  # 2.965 vs 3.279 us (1.11x)
        (26, 2560, 1536): "ll_bf16",  # 2.991 vs 3.233 us (1.08x)
        (27, 2560, 1536): "ll_bf16",  # 3.008 vs 3.243 us (1.08x)
        (28, 2560, 1536): "ll_bf16",  # 2.986 vs 3.223 us (1.08x)
        (29, 2560, 1536): "ll_bf16",  # 3.020 vs 3.282 us (1.09x)
        (30, 2560, 1536): "ll_bf16",  # 2.987 vs 3.238 us (1.08x)
        (31, 2560, 1536): "ll_bf16",  # 3.001 vs 3.207 us (1.07x)
        (32, 2560, 1536): "ll_bf16",  # 2.994 vs 3.220 us (1.08x)
        # 3584x2560
        (1, 3584, 2560): "ll_bf16",  # 4.100 vs 5.153 us (1.26x)
        (2, 3584, 2560): "ll_bf16",  # 4.159 vs 7.912 us (1.90x)
        (3, 3584, 2560): "ll_bf16",  # 4.169 vs 7.695 us (1.85x)
        (4, 3584, 2560): "ll_bf16",  # 4.176 vs 7.961 us (1.91x)
        (5, 3584, 2560): "ll_bf16",  # 4.185 vs 7.884 us (1.88x)
        (6, 3584, 2560): "ll_bf16",  # 4.206 vs 7.972 us (1.90x)
        (7, 3584, 2560): "ll_bf16",  # 4.204 vs 7.882 us (1.87x)
        (8, 3584, 2560): "ll_bf16",  # 4.215 vs 7.883 us (1.87x)
        (9, 3584, 2560): "ll_bf16",  # 4.387 vs 5.419 us (1.24x)
        (10, 3584, 2560): "ll_bf16",  # 4.388 vs 5.411 us (1.23x)
        (11, 3584, 2560): "ll_bf16",  # 4.384 vs 5.415 us (1.24x)
        (12, 3584, 2560): "ll_bf16",  # 4.399 vs 5.414 us (1.23x)
        (13, 3584, 2560): "ll_bf16",  # 4.409 vs 5.422 us (1.23x)
        (14, 3584, 2560): "ll_bf16",  # 4.444 vs 5.411 us (1.22x)
        (15, 3584, 2560): "ll_bf16",  # 4.422 vs 5.419 us (1.23x)
        (16, 3584, 2560): "ll_bf16",  # 4.456 vs 5.410 us (1.21x)
        (17, 3584, 2560): "ll_bf16",  # 4.955 vs 5.496 us (1.11x)
        (18, 3584, 2560): "ll_bf16",  # 4.917 vs 5.495 us (1.12x)
        (19, 3584, 2560): "ll_bf16",  # 4.953 vs 5.498 us (1.11x)
        (20, 3584, 2560): "ll_bf16",  # 4.955 vs 5.496 us (1.11x)
        (21, 3584, 2560): "ll_bf16",  # 4.898 vs 5.508 us (1.12x)
        (22, 3584, 2560): "ll_bf16",  # 4.909 vs 5.490 us (1.12x)
        (23, 3584, 2560): "ll_bf16",  # 4.857 vs 5.500 us (1.13x)
        (24, 3584, 2560): "ll_bf16",  # 4.830 vs 5.498 us (1.14x)
        (25, 3584, 2560): "ll_bf16",  # 4.835 vs 5.500 us (1.14x)
        (26, 3584, 2560): "ll_bf16",  # 4.829 vs 5.492 us (1.14x)
        (27, 3584, 2560): "ll_bf16",  # 4.837 vs 5.502 us (1.14x)
        (28, 3584, 2560): "ll_bf16",  # 4.832 vs 5.500 us (1.14x)
        (29, 3584, 2560): "ll_bf16",  # 4.832 vs 5.512 us (1.14x)
        (30, 3584, 2560): "ll_bf16",  # 4.894 vs 5.503 us (1.12x)
        (31, 3584, 2560): "ll_bf16",  # 4.871 vs 5.511 us (1.13x)
        (32, 3584, 2560): "ll_bf16",  # 4.896 vs 6.771 us (1.38x)
        # 4120x2560
        (1, 4120, 2560): "ll_bf16",  # 4.565 vs 5.499 us (1.20x)
        (2, 4120, 2560): "ll_bf16",  # 4.609 vs 7.739 us (1.68x)
        (3, 4120, 2560): "ll_bf16",  # 4.616 vs 7.711 us (1.67x)
        (4, 4120, 2560): "ll_bf16",  # 4.656 vs 7.872 us (1.69x)
        (5, 4120, 2560): "ll_bf16",  # 4.625 vs 7.763 us (1.68x)
        (6, 4120, 2560): "ll_bf16",  # 4.649 vs 7.990 us (1.72x)
        (7, 4120, 2560): "ll_bf16",  # 4.658 vs 7.961 us (1.71x)
        (8, 4120, 2560): "ll_bf16",  # 4.658 vs 7.894 us (1.69x)
        (9, 4120, 2560): "ll_bf16",  # 4.821 vs 5.644 us (1.17x)
        (10, 4120, 2560): "ll_bf16",  # 4.840 vs 5.670 us (1.17x)
        (11, 4120, 2560): "ll_bf16",  # 4.842 vs 5.652 us (1.17x)
        (12, 4120, 2560): "ll_bf16",  # 4.848 vs 5.662 us (1.17x)
        (13, 4120, 2560): "ll_bf16",  # 4.857 vs 5.646 us (1.16x)
        (14, 4120, 2560): "ll_bf16",  # 4.874 vs 5.672 us (1.16x)
        (15, 4120, 2560): "ll_bf16",  # 4.864 vs 5.653 us (1.16x)
        (16, 4120, 2560): "ll_bf16",  # 4.887 vs 5.671 us (1.16x)
        (17, 4120, 2560): "ll_bf16",  # 5.411 vs 5.736 us (1.06x)
        (18, 4120, 2560): "ll_bf16",  # 5.409 vs 5.750 us (1.06x)
        (19, 4120, 2560): "ll_bf16",  # 5.365 vs 5.741 us (1.07x)
        (20, 4120, 2560): "ll_bf16",  # 5.320 vs 5.755 us (1.08x)
        (21, 4120, 2560): "ll_bf16",  # 5.356 vs 5.745 us (1.07x)
        (22, 4120, 2560): "ll_bf16",  # 5.284 vs 5.750 us (1.09x)
        (23, 4120, 2560): "ll_bf16",  # 5.263 vs 5.739 us (1.09x)
        (24, 4120, 2560): "ll_bf16",  # 5.216 vs 5.757 us (1.10x)
        (25, 4120, 2560): "ll_bf16",  # 5.218 vs 5.740 us (1.10x)
        (26, 4120, 2560): "ll_bf16",  # 5.209 vs 5.754 us (1.10x)
        (27, 4120, 2560): "ll_bf16",  # 5.222 vs 5.744 us (1.10x)
        (28, 4120, 2560): "ll_bf16",  # 5.222 vs 5.767 us (1.10x)
        (29, 4120, 2560): "ll_bf16",  # 5.225 vs 5.753 us (1.10x)
        (30, 4120, 2560): "ll_bf16",  # 5.230 vs 5.764 us (1.10x)
        (31, 4120, 2560): "ll_bf16",  # 5.235 vs 5.754 us (1.10x)
        (32, 4120, 2560): "ll_bf16",  # 5.251 vs 6.830 us (1.30x)
        # TP2-only shapes (three independent sweeps).
        # 2560x320
        (1, 2560, 320): "tgv",  # 1.714 vs 2.259 us (1.32x)
        (2, 2560, 320): "tgv",  # 1.684 vs 3.122 us (1.85x)
        (3, 2560, 320): "tgv",  # 1.714 vs 2.263 us (1.32x)
        (4, 2560, 320): "tgv",  # 1.686 vs 2.229 us (1.32x)
        (5, 2560, 320): "tgv",  # 1.668 vs 2.228 us (1.34x)
        (6, 2560, 320): "tgv",  # 1.677 vs 2.288 us (1.36x)
        (7, 2560, 320): "tgv",  # 1.673 vs 2.160 us (1.29x)
        (8, 2560, 320): "tgv",  # 1.673 vs 2.162 us (1.29x)
        (18, 2560, 320): "tgv",  # 1.868 vs 1.995 us (1.07x)
        (19, 2560, 320): "tgv",  # 1.883 vs 1.998 us (1.06x)
        (20, 2560, 320): "tgv",  # 1.861 vs 1.974 us (1.06x)
        (22, 2560, 320): "tgv",  # 1.864 vs 1.980 us (1.06x)
        (24, 2560, 320): "tgv",  # 1.860 vs 1.985 us (1.07x)
        # 2560x3072
        (1, 2560, 3072): "ll_bf16",  # 3.537 vs 4.298 us (1.22x)
        (2, 2560, 3072): "ll_bf16",  # 3.468 vs 7.177 us (2.07x)
        (3, 2560, 3072): "ll_bf16",  # 3.475 vs 7.054 us (2.03x)
        (4, 2560, 3072): "ll_bf16",  # 3.508 vs 7.229 us (2.06x)
        (5, 2560, 3072): "ll_bf16",  # 3.494 vs 7.114 us (2.04x)
        (6, 2560, 3072): "ll_bf16",  # 3.497 vs 7.047 us (2.02x)
        (7, 2560, 3072): "ll_bf16",  # 3.516 vs 7.192 us (2.05x)
        (8, 2560, 3072): "ll_bf16",  # 3.513 vs 7.138 us (2.03x)
        (9, 2560, 3072): "ll_bf16",  # 4.599 vs 5.627 us (1.22x)
        (10, 2560, 3072): "ll_bf16",  # 4.626 vs 5.595 us (1.21x)
        (11, 2560, 3072): "ll_bf16",  # 4.624 vs 5.635 us (1.22x)
        (12, 2560, 3072): "ll_bf16",  # 4.655 vs 5.596 us (1.20x)
        (13, 2560, 3072): "ll_bf16",  # 4.669 vs 5.635 us (1.21x)
        (14, 2560, 3072): "ll_bf16",  # 4.706 vs 5.601 us (1.19x)
        (15, 2560, 3072): "ll_bf16",  # 4.698 vs 5.634 us (1.20x)
        (16, 2560, 3072): "ll_bf16",  # 4.761 vs 7.281 us (1.53x)
        (17, 2560, 3072): "ll_bf16",  # 4.569 vs 6.345 us (1.39x)
        (18, 2560, 3072): "ll_bf16",  # 4.578 vs 6.430 us (1.40x)
        (19, 2560, 3072): "ll_bf16",  # 4.552 vs 6.383 us (1.40x)
        (20, 2560, 3072): "ll_bf16",  # 4.548 vs 6.390 us (1.41x)
        (21, 2560, 3072): "ll_bf16",  # 4.617 vs 6.912 us (1.50x)
        (22, 2560, 3072): "ll_bf16",  # 4.541 vs 6.404 us (1.41x)
        (23, 2560, 3072): "ll_bf16",  # 4.630 vs 6.588 us (1.42x)
        (24, 2560, 3072): "ll_bf16",  # 4.594 vs 6.399 us (1.39x)
        (25, 2560, 3072): "ll_bf16",  # 4.618 vs 5.953 us (1.29x)
        (26, 2560, 3072): "ll_bf16",  # 4.653 vs 5.937 us (1.28x)
        (27, 2560, 3072): "ll_bf16",  # 4.618 vs 5.955 us (1.29x)
        (28, 2560, 3072): "ll_bf16",  # 4.641 vs 5.945 us (1.28x)
        (29, 2560, 3072): "ll_bf16",  # 4.628 vs 5.961 us (1.29x)
        (30, 2560, 3072): "ll_bf16",  # 4.583 vs 5.949 us (1.30x)
        (31, 2560, 3072): "ll_bf16",  # 4.641 vs 5.962 us (1.28x)
        (32, 2560, 3072): "ll_bf16",  # 4.635 vs 6.441 us (1.39x)
        # 6656x2560
        (2, 6656, 2560): "tgv",  # 6.671 vs 7.547 us (1.13x)
        (3, 6656, 2560): "tgv",  # 6.642 vs 7.282 us (1.10x)
        (4, 6656, 2560): "tgv",  # 6.639 vs 7.309 us (1.10x)
        (5, 6656, 2560): "tgv",  # 6.642 vs 7.084 us (1.07x)
        (6, 6656, 2560): "tgv",  # 6.663 vs 7.091 us (1.06x)
        (7, 6656, 2560): "tgv",  # 6.621 vs 7.067 us (1.07x)
        (8, 6656, 2560): "tgv",  # 6.670 vs 7.074 us (1.06x)
        # 8240x2560
        (1, 8240, 2560): "ll_bf16",  # 6.953 vs 8.111 us (1.17x)
        (2, 8240, 2560): "ll_bf16",  # 7.069 vs 8.478 us (1.20x)
        (3, 8240, 2560): "ll_bf16",  # 7.041 vs 8.228 us (1.17x)
        (4, 8240, 2560): "ll_bf16",  # 7.129 vs 8.218 us (1.15x)
        (5, 8240, 2560): "ll_bf16",  # 7.086 vs 7.897 us (1.11x)
        (6, 8240, 2560): "ll_bf16",  # 7.109 vs 7.890 us (1.11x)
        (7, 8240, 2560): "ll_bf16",  # 7.096 vs 7.861 us (1.11x)
        (8, 8240, 2560): "ll_bf16",  # 7.066 vs 7.849 us (1.11x)
        (9, 8240, 2560): "ll_bf16",  # 7.273 vs 7.954 us (1.09x)
        (10, 8240, 2560): "ll_bf16",  # 7.260 vs 7.948 us (1.09x)
        (11, 8240, 2560): "ll_bf16",  # 7.279 vs 7.966 us (1.09x)
        (12, 8240, 2560): "ll_bf16",  # 7.254 vs 7.958 us (1.10x)
        (13, 8240, 2560): "ll_bf16",  # 7.283 vs 7.993 us (1.10x)
        (14, 8240, 2560): "ll_bf16",  # 7.312 vs 7.956 us (1.09x)
        (15, 8240, 2560): "ll_bf16",  # 7.327 vs 7.984 us (1.09x)
        (16, 8240, 2560): "ll_bf16",  # 7.274 vs 7.969 us (1.10x)
        (25, 8240, 2560): "ll_bf16",  # 7.870 vs 8.520 us (1.08x)
        (26, 8240, 2560): "ll_bf16",  # 7.854 vs 8.541 us (1.09x)
        (27, 8240, 2560): "ll_bf16",  # 7.847 vs 8.521 us (1.09x)
        (28, 8240, 2560): "ll_bf16",  # 7.874 vs 8.549 us (1.09x)
        (29, 8240, 2560): "ll_bf16",  # 7.930 vs 8.529 us (1.08x)
        (30, 8240, 2560): "ll_bf16",  # 7.914 vs 8.549 us (1.08x)
        (31, 8240, 2560): "ll_bf16",  # 7.910 vs 8.525 us (1.08x)
        # Unstable exact keys stay on the incumbent: (2,512,2560),
        # (4,640,2560), (17/21/23,2560,320), (1,6656,2560), and
        # (3/4,12800,2560). For 12800x2560, M=5..32 also stay on cuBLAS.
    }
)

_BF16_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    }
)
# sm100 (GB200/B200) and sm103 (GB300) both run these kernels -- the skinny
# GEMM uses only generic CuTe primitives, and the sm103 gate recorded where the
# sweep had been run, not what the kernels require. Re-swept on GB200: the
# sm103 winners reproduce entry for entry at the TP8 shapes.
_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    vendors=frozenset({"nvidia"}),
)


# Shapes an eager call has already compiled and allocated for. Capture must
# not JIT or allocate, so an unwarmed shape falls back to torch.mm there.
_warmed: set[tuple[str, int, int, int, int]] = set()
_warmed_lock = threading.Lock()


def _usable_in_capture(backend: str, dev: int, m: int, n: int, k: int) -> bool:
    # Device-keyed: warmth on one GPU says nothing about another's modules.
    return (
        not torch.cuda.is_current_stream_capturing()
        or (backend, dev, m, n, k) in _warmed
    )


def _mark_warmed(backend: str, dev: int, m: int, n: int, k: int) -> None:
    # Only a successful eager call earns capture trust.
    if not torch.cuda.is_current_stream_capturing():
        with _warmed_lock:
            _warmed.add((backend, dev, m, n, k))


# Measured (m, n, k) -> (block_size, outputs_per_block, k_unroll, vector_width).
SKINNY_CONFIG_ROUTE: MappingProxyType[
    tuple[int, int, int], tuple[int, int, int, int]
] = MappingProxyType(
    {
        (2, 768, 1536): (96, 2, 1, 16),
        (3, 768, 1536): (96, 2, 1, 16),
        (4, 768, 1536): (96, 2, 1, 16),
        (2, 1152, 1536): (96, 4, 1, 16),
        (3, 1152, 1536): (96, 4, 1, 16),
        (2, 1536, 1536): (96, 4, 1, 16),
        (3, 1536, 1536): (96, 4, 1, 16),
        (2, 2304, 1536): (96, 4, 1, 16),
        (2, 320, 2560): (160, 2, 1, 16),
        (5, 1536, 7168): (224, 2, 1, 16),
        (6, 1536, 7168): (64, 2, 1, 16),
    }
)


@functools.lru_cache(maxsize=256)
def _skinny_config(m: int, n: int, k: int):
    """The measured config for this shape, else the vendor heuristic."""
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )

    tuned = SKINNY_CONFIG_ROUTE.get((m, n, k))
    if tuned is not None:
        return SkinnyGemmConfig(m, *tuned)
    return shape_dynamic_skinny_gemm.default_config(m, n, k)


def skinny_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` via the vendored CuTe skinny GEMM.

    Caller-owned invariant, as for every ``decode_gemv`` backend: ``out`` must
    not overlap ``x`` or ``weight`` storage.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm,
    )

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    # Table shapes only; anything else would grow the compile cache unbounded.
    if (
        MEASURED_ROUTE.get((m, n, k)) != "skinny"
        or x.dtype != torch.bfloat16
        or not _usable_in_capture("skinny", dev, m, n, k)
    ):
        return _torch_decode_gemv(x, weight, out)
    config = _skinny_config(m, n, k)
    # default_config can emit a config supports() rejects; fall back, don't raise.
    if not shape_dynamic_skinny_gemm.supports(config, m, n, k):
        return _torch_decode_gemv(x, weight, out)
    # supports() cannot see pointer alignment; the kernel asserts it at launch.
    align = config.vector_width * x.element_size()
    if x.data_ptr() % align or weight.data_ptr() % align:
        return _torch_decode_gemv(x, weight, out)
    # DLPack refuses requires_grad tensors; detach is a zero-copy view.
    result = shape_dynamic_skinny_gemm(x.detach(), weight.detach(), config, out=out)
    _mark_warmed("skinny", dev, m, n, k)
    return result


# TGV requires a bias; the routed GEMVs have none. Never an evicting cache: a
# captured graph replays against these exact tensors.
_tgv_biases: dict[tuple[int, int], torch.Tensor] = {}
_tgv_bias_lock = threading.Lock()


def _tgv_bias(n: int, device_index: int) -> torch.Tensor:
    key = (n, device_index)
    bias = _tgv_biases.get(key)
    if bias is None:
        with _tgv_bias_lock:
            # Re-check: a racing thread must not replace a graph-held entry.
            bias = _tgv_biases.get(key)
            if bias is None:
                bias = torch.zeros(
                    n, device=f"cuda:{device_index}", dtype=torch.bfloat16
                )
                _tgv_biases[key] = bias
    return bias


def tgv_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` via FlashInfer's TGV low-latency GEMM.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight; its transpose is the
            column-major ``(K, N)`` layout TGV wants, with no copy.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from flashinfer import mm_bf16

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    if (
        MEASURED_ROUTE.get((m, n, k)) != "tgv"
        or x.dtype != torch.bfloat16
        or not _usable_in_capture("tgv", dev, m, n, k)
    ):
        return _torch_decode_gemv(x, weight, out)
    bias = _tgv_bias(n, dev)
    # TGV is CuTe DSL inside FlashInfer: same DLPack no-grad rule.
    result = mm_bf16(
        x.detach(),
        weight.detach().t(),
        bias=bias,
        pdl=pdl_enabled(),
        backend="tgv",
        out=out,
    )
    _mark_warmed("tgv", dev, m, n, k)
    return result


def ll_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` via the low-latency BF16 GEMM.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from tokenspeed_kernel.ops.gemm.ll_bf16 import ll_bf16_mm, ll_bf16_mm_supported

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    if (
        MEASURED_ROUTE.get((m, n, k)) != "ll_bf16"
        or x.dtype != torch.bfloat16
        or not _usable_in_capture("ll_bf16", dev, m, n, k)
        or not ll_bf16_mm_supported(x, weight)
    ):
        return _torch_decode_gemv(x, weight, out)
    result = ll_bf16_mm(x.detach(), weight.detach(), out=out)
    _mark_warmed("ll_bf16", dev, m, n, k)
    return result


# Fused ``a + x @ W.T + c`` (K3 MoE latent up-proj epilogue): (m, n, k) ->
# (block_size, outputs_per_block, k_unroll). Cold-L2 vs the incumbent: M == 1
# 8.86us vs rowcta_gemv_add3 9.42, M == 2 10.05 vs composed 12.81. M == 4 was
# 1.04x, under the margin, so it keeps the composed path.
ADD3_ROUTE: MappingProxyType[tuple[int, int, int], tuple[int, int, int]] = (
    MappingProxyType(
        {
            (1, 7168, 3584): (64, 4, 2),
            (2, 7168, 3584): (64, 7, 2),
        }
    )
)


def decode_gemv_routed(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :data:`MEASURED_ROUTE` covers this call on this platform.

    Args:
        x: ``[M, K]`` activation.
        weight: ``[N, K]`` weight.

    Returns:
        True when ``decode_gemv`` would reach a measured backend rather than
        the portable fallback.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.ndim != 2
    ):
        return False
    m, k = x.shape
    return (m, weight.shape[0], k) in MEASURED_ROUTE and _is_routed_arch(
        x.device.index or 0
    )


@functools.lru_cache(maxsize=8)
def _is_routed_arch(device_index: int) -> bool:
    """MEASURED_ROUTE's arch floor: sm100 and up, matching its registration."""
    from tokenspeed_kernel.platform import current_platform

    if current_platform().vendor != "nvidia":
        return False
    return torch.cuda.get_device_capability(device_index) >= (10, 0)


@functools.lru_cache(maxsize=8)
def _is_measured_arch(device_index: int) -> bool:
    """ADD3_ROUTE's arch floor: its configs were only swept on sm103."""
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()
    if platform.vendor != "nvidia":
        return False
    return torch.cuda.get_device_capability(device_index) >= (10, 3)


def skinny_add3_supported(m: int, n: int, k: int, device: torch.device) -> bool:
    """Whether :func:`skinny_gemv_add3` has a measured config for this call.

    Args:
        m/n/k: the projection extents (``x[M, K] @ W[N, K].T``).
        device: the CUDA device the call would run on.

    Returns:
        True only for table shapes on the measured architecture.
    """
    return (m, n, k) in ADD3_ROUTE and _is_measured_arch(device.index or 0)


def _composed_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    result = torch.addmm(c, x, weight.t())
    result += a
    if out is not None:
        out.copy_(result)
        return out
    return result


def skinny_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` via the skinny GEMM's dual-residual epilogue.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight.
        a/c: ``[M, N]`` addends with unit inner stride (row stride free, so a
            column slice of a wider tensor is accepted).
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` result in ``x``'s dtype.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    tuned = ADD3_ROUTE.get((m, n, k))
    if (
        tuned is None
        or x.dtype != torch.bfloat16
        or not _is_measured_arch(dev)
        or not _usable_in_capture("skinny_add3", dev, m, n, k)
    ):
        return _composed_add3(x, weight, a, c, out)
    config = SkinnyGemmConfig(m, *tuned)
    if not shape_dynamic_skinny_gemm.supports(config, m, n, k):
        return _composed_add3(x, weight, a, c, out)
    result = shape_dynamic_skinny_gemm(
        x.detach(),
        weight.detach(),
        config,
        residual=a.detach(),
        residual2=c.detach(),
        out=out,
    )
    _mark_warmed("skinny_add3", dev, m, n, k)
    return result


def _register_route() -> None:
    impls = {"skinny": skinny_gemv, "tgv": tgv_gemv, "ll_bf16": ll_bf16_gemv}
    for (m, n, k), backend in MEASURED_ROUTE.items():
        register_kernel(
            "gemm",
            "decode_gemv",
            name=f"{backend}_gemv_m{m}_n{n}_k{k}",
            solution="flashinfer" if backend == "tgv" else "cute_dsl",
            capability=_CAPABILITY,
            signatures=_BF16_SIG,
            traits={
                "m": frozenset({m}),
                "n": frozenset({n}),
                "k": frozenset({k}),
            },
            # Above the M == 1 rowcta spec so a measured win takes the shape.
            priority=Priority.SPECIALIZED + 2,
        )(impls[backend])


_register_route()
