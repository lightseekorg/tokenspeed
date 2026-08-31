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

Every entry in ``MEASURED_ROUTE`` was measured on GB300 (sm103) at the exact
(N, K) the TP8 decode path hands ``decode_gemv`` -- extracted from an nsys
trace of serving, not assumed -- with the tuner cycling eight weight copies
so the L2 never holds the operand between calls, the way serving streams a
different layer's weight each launch. Hot-cache numbers ranked backends
wrongly (1.9x off serving at 6288x7168); cold-L2 reproduces serving per-shape
times within ~5%. ``test/gemm_tuning/tune_route.py`` reproduces the sweep.

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
        # TP4 (B200, sm100), bf16 dense linears: the fp8 checkpoint quantizes
        # only the routed experts. M is the captured decode bucket -- 5 was
        # swept but never observed.
        (1, 512, 2560): "skinny",  # mlp.gate; 1.83 vs 4.37 (2.39x)
        (2, 512, 2560): "skinny",  # 2.05 vs 4.38 (2.13x)
        (4, 512, 2560): "ll_bf16",  # 2.28 vs 4.43 (1.95x)
        (8, 512, 2560): "ll_bf16",  # 2.32 vs 4.64 (2.00x)
        (1, 320, 2560): "skinny",  # shared_expert gate_up; 1.80 vs 4.36 (2.42x)
        (2, 320, 2560): "skinny",  # 2.00 vs 5.75 (2.88x)
        (4, 320, 2560): "ll_bf16",  # 2.22 vs 4.74 (2.13x)
        (8, 320, 2560): "ll_bf16",  # 2.25 vs 4.73 (2.10x)
        # shared_expert down-proj. K = 160 is not a multiple of 128, so neither
        # the skinny GEMM nor ll_bf16 admits it and tgv takes every width.
        (1, 2560, 160): "tgv",  # 1.61 vs 1.88 (1.17x)
        (2, 2560, 160): "tgv",  # 1.65 vs 3.09 (1.87x)
        (4, 2560, 160): "tgv",  # 1.66 vs 3.09 (1.86x)
        (8, 2560, 160): "tgv",  # 1.65 vs 1.92 (1.16x)
        (1, 2560, 1536): "ll_bf16",  # attn o_proj; 2.55 vs 3.25 (1.27x)
        (2, 2560, 1536): "ll_bf16",  # 2.54 vs 4.90 (1.92x)
        (4, 2560, 1536): "ll_bf16",  # 2.58 vs 4.84 (1.88x)
        (8, 2560, 1536): "ll_bf16",  # 2.67 vs 4.67 (1.75x)
        (1, 4120, 2560): "ll_bf16",  # linear_attn in_proj; 4.84 vs 9.26 (1.91x)
        (2, 4120, 2560): "ll_bf16",  # 4.88 vs 8.55 (1.75x)
        (4, 4120, 2560): "ll_bf16",  # 4.87 vs 8.41 (1.73x)
        (8, 4120, 2560): "ll_bf16",  # 4.91 vs 8.38 (1.71x)
        (1, 3584, 2560): "ll_bf16",  # 4.35 vs 5.50 (1.26x)
        (2, 3584, 2560): "ll_bf16",  # 4.42 vs 7.50 (1.70x)
        (4, 3584, 2560): "ll_bf16",  # 4.41 vs 7.63 (1.73x)
        (8, 3584, 2560): "ll_bf16",  # 4.44 vs 7.28 (1.64x)
        # 640x2560 crosses over: skinny leads to M == 4, ll_bf16 from M == 8.
        (1, 640, 2560): "skinny",  # 1.88 vs 4.69 (2.49x)
        (2, 640, 2560): "skinny",  # 2.10 vs 4.62 (2.20x)
        (4, 640, 2560): "skinny",  # 2.89 vs 4.77 (1.65x)
        (8, 640, 2560): "ll_bf16",  # 3.01 vs 4.81 (1.60x)
        (1, 2560, 2560): "ll_bf16",  # 3.27 vs 5.06 (1.55x)
        (2, 2560, 2560): "ll_bf16",  # 3.36 vs 6.37 (1.90x)
        (4, 2560, 2560): "ll_bf16",  # 3.40 vs 6.08 (1.79x)
        (8, 2560, 2560): "ll_bf16",  # 3.40 vs 6.35 (1.87x)
        # 12800x2560's margins shrink as M grows and invert at 8 (cublas 13.70
        # vs ll_bf16 13.43, inside the margin), so M == 8 keeps cublas.
        (1, 12800, 2560): "skinny",  # 11.07 vs 14.62 (1.32x)
        (2, 12800, 2560): "skinny",  # 11.81 vs 14.38 (1.22x)
        (4, 12800, 2560): "ll_bf16",  # 13.31 vs 14.43 (1.08x)
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


# Measured skinny configs that beat ``default_config`` by >=4% under the
# tune_route.py methodology (cold L2: 8 weight copies cycled in-graph,
# 41-round medians, GB200/sm100 -- hence the ``_is_skinny_config_arch`` pin):
# (m, n, k) -> (block_size, outputs_per_block, k_unroll, vector_width).
# All use 256-bit loads (vector_width 16), which nvidia-cutlass-dsl accepts
# from 4.7; the heuristic predates that option. Wins are 4.2%-49.8%, biggest
# where one block tile covers K (96x16 == 1536, 160x16 == 2560). The K=7168
# and K=14336 families lost cold -- an L2-warm sweep had them ahead -- and
# keep the heuristic, except the 1536x7168 shard at M == 5 and 6.
SKINNY_CONFIG_ROUTE: MappingProxyType[
    tuple[int, int, int], tuple[int, int, int, int]
] = MappingProxyType(
    {
        (2, 768, 1536): (96, 2, 1, 16),
        (3, 768, 1536): (96, 2, 1, 16),
        (4, 768, 1536): (96, 2, 1, 16),
        (2, 1152, 1536): (96, 4, 1, 16),
        (3, 1152, 1536): (96, 4, 1, 16),
        (4, 1152, 1536): (96, 4, 1, 16),
        (5, 1152, 1536): (96, 4, 1, 16),
        (2, 1536, 1536): (96, 4, 1, 16),
        (3, 1536, 1536): (96, 4, 1, 16),
        (4, 1536, 1536): (96, 2, 1, 16),
        (2, 2304, 1536): (96, 4, 1, 16),
        (1, 320, 2560): (160, 2, 1, 16),
        (2, 320, 2560): (160, 2, 1, 16),
        (1, 512, 2560): (160, 2, 1, 16),
        (2, 512, 2560): (160, 2, 1, 16),
        (1, 640, 2560): (160, 4, 1, 16),
        (2, 640, 2560): (160, 2, 1, 16),
        (4, 640, 2560): (160, 2, 1, 16),
        (2, 12800, 2560): (160, 4, 1, 16),
        (5, 1536, 7168): (224, 2, 1, 16),
        (6, 1536, 7168): (64, 2, 1, 16),
    }
)


@functools.lru_cache(maxsize=8)
def _is_skinny_config_arch(device_index: int) -> bool:
    """SKINNY_CONFIG_ROUTE's arch pin: its configs were only swept on sm100."""
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()
    if platform.vendor != "nvidia":
        return False
    return torch.cuda.get_device_capability(device_index) == (10, 0)


# Unbounded is safe: the sole hot caller admits only MEASURED_ROUTE skinny
# shapes, so the key space is that table times the device count.
@functools.lru_cache(maxsize=None)
def _skinny_config(m: int, n: int, k: int, device_index: int):
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )

    if _is_skinny_config_arch(device_index):
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
    config = _skinny_config(m, n, k, dev)
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
