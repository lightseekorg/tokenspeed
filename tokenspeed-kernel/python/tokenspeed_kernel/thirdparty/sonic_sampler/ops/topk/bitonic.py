# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton  # noqa.

cdiv, jit = triton.cdiv, triton.jit


"""
The following suite of kernels were adapted from the official Triton repository:

    https://github.com/triton-lang/triton/blob/main/python/triton/language/standard.py

with specific modifications and optimizations to allow for:

    • Revised bitonic folding that minimizes inter-warp barrier(s).

    • Selective reductions and reshape(s) for layouts to maximize occupancy.

    • Leveraging optimized IMNMX.U32.U32 instructions during CAS.

    • Support alternating write-outs across vocab block(s) / tile(s).

"""


@jit
def indicator(pair, outer: tl.constexpr, inner: tl.constexpr):
    """
    Expands the bit-`pair` into a tensor with shape [1, ..., 1, 2, 1, ..., 1]
    where the 2 is at the `inner`-th position from the end.

    """
    return tl.reshape(pair, ([1] * outer) + [2] + ([1] * inner))


@jit
def compare_and_swap(values, pair, flip, dim: tl.constexpr, total: tl.constexpr):

    outer: tl.constexpr = total - dim - 1

    # Flip along the middle dimension.

    result = values ^ tl.xor_sum(values, axis=outer, keep_dims=True)

    # Determine right / left position along the axis, expand, and flip.

    ordering = flip ^ indicator(pair, outer, dim)

    # Ensure CAS compiles down to optimized IMNMX.U32.U32 instruction(s).

    minima = tl.minimum(values, result)
    maxima = tl.maximum(values, result)

    # Conditional swap while preserving bitonic invariant(s) and ordering stability.

    return tl.where(ordering != 0, maxima, minima)


@jit
def hypercube_merge(values, pair, flip, stage: tl.constexpr, total: tl.constexpr):

    hypercube = values

    for i in tl.static_range(0, stage):

        hypercube = compare_and_swap(hypercube, pair, flip, stage - i - 1, total)

    return hypercube


@jit
def bitonic_fold(
    values,
    pair,
    rows: tl.constexpr,
    max_k: tl.constexpr,
    folds: tl.constexpr,
    descending: tl.constexpr,
    factor_n: tl.constexpr,
    factor_k: tl.constexpr,
):

    # Iteratively reduce across folds via bitonic reduction.

    block = values

    for i in tl.static_range(1, folds + 1):

        block = block.reshape((rows >> i, 2, max_k))

        left_block, right_block = tl.split(block.trans(0, 2, 1))
        block = tl.maximum(left_block, right_block)

        hypercube = tl.reshape(block, [2] * (factor_n - i - 1))

        if i < folds:

            flip = indicator(pair, folds - i - 1, factor_k)
            block = hypercube_merge(hypercube, pair, flip, factor_k, factor_n - i - 1)

        else:

            block = hypercube_merge(
                hypercube, pair, descending, factor_k, factor_n - i - 1
            )

    return block.reshape((max_k,))


@jit
def bitonic_reduction(
    values, rows: tl.constexpr, max_k: tl.constexpr, total_cols: tl.constexpr
):

    factor_n: tl.constexpr = tl.standard._log2(total_cols) + 1  # noqa.
    factor_k: tl.constexpr = tl.standard._log2(max_k)  # noqa.

    folds: tl.constexpr = tl.standard._log2(rows)  # noqa.

    pair = tl.arange(0, 2).to(tl.int1)

    return bitonic_fold(values, pair, rows, max_k, folds, 1, factor_n, factor_k)


@jit
def top_k(values, descending: tl.constexpr, max_k: tl.constexpr, block_n: tl.constexpr):
    """
    Ordered bitonic top-k via alternating reduction across hypercube folds with focused
    optimizations around CAS instructions, minimized inter-warp barrier(s), and maximized
    occupancy.

    Note(s):

        • `values` -> [`block_n`]

        • `factor_k` -> log_2(`max_k`)

        • `factor_n` -> log_2(`block_n`)

    """
    factor_n: tl.constexpr = tl.standard._log2(block_n)  # noqa.
    factor_k: tl.constexpr = tl.standard._log2(max_k)  # noqa.

    folds: tl.constexpr = factor_n - factor_k - 1
    rows: tl.constexpr = 1 << folds

    hypercube = tl.reshape(values, [2] * factor_n)
    pair = tl.arange(0, 2).to(tl.int1)

    for stage in tl.static_range(1, factor_k + 1):

        flip = indicator(pair, factor_n - stage - 1, stage)
        hypercube = hypercube_merge(hypercube, pair, flip, stage, factor_n)

    hypercube = tl.max(hypercube, axis=folds)

    if folds > 0:

        # Continue with alternating rows with non-zero fold(s) remaining.

        flip = indicator(pair, folds - 1, factor_k)
        hypercube = hypercube_merge(hypercube, pair, flip, factor_k, factor_n - 1)

        grouped = tl.reshape(hypercube, (rows, max_k))

        return bitonic_fold(
            grouped, pair, rows, max_k, folds, descending, factor_n, factor_k
        )

    else:

        # Short circuit folding with final ascending / descending merge.

        hypercube = hypercube_merge(hypercube, pair, descending, factor_k, factor_n - 1)

        return tl.reshape(hypercube, (max_k,))
