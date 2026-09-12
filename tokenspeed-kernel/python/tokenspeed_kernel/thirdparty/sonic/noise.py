# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
# Adapted for tokenspeed-kernel from sonic-sampler v1.0.0
# (https://github.com/tachyontrails/sonic-sampler, Apache-2.0): the Gumbel
# weights and acceptance coins are drawn in-kernel from
# Philox(seed[slot], offset[slot]) at the candidates the kernels read, instead
# of loaded from per-slot noise planes; Triton comes from the ``_triton`` shim.

"""In-kernel noise: Philox(seed[slot], offset[slot]) at the token indices a
kernel actually reads, replacing sonic's per-slot Gumbel and coin planes."""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton

jit = triton.jit


@jit
def step_seed(sd_ptr, of_ptr, batch_id, step, steps: tl.constexpr):
    """64-bit Philox key of one (slot, step): the slot's seed and its offset
    (the request's cache length) select the stream, so draws depend only on
    the request and advance every step. Streams never repeat while ``steps``
    is at least the largest per-step advance of the offset."""

    seed = tl.load(sd_ptr + batch_id).to(tl.int64)
    offset = tl.load(of_ptr + batch_id).to(tl.int64)

    lo, hi, _, _ = tl.randint4x(seed, offset * steps + step)

    return (hi.to(tl.int64) << 32) | lo.to(tl.int64)


@jit
def gumbel_noise(seed, indices):
    """log(Exp(1)) = log(-log U) at the given token indices, U from one Philox
    round per group of four consecutive tokens (matches a plane written with
    ``tl.rand4x`` over ``indices // 4``)."""

    u0, u1, u2, u3 = tl.rand4x(seed, indices // 4)

    lane = indices % 4
    u = tl.where(lane == 0, u0, tl.where(lane == 1, u1, tl.where(lane == 2, u2, u3)))

    return tl.log(-tl.log(tl.maximum(u, 1.0e-7)))
