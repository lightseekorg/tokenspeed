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

"""Keep MXFP8 persistent narrow tiles within one N tile.

FlashInfer 0.7.1rc2 relaxed this restriction from 0.7.0, admitting MXFP8
candidates that fault during autotuning. Keep small-M swapped and split-K
candidates, with the same check when a cached tactic serves another shape.
"""

from __future__ import annotations

import functools
import inspect
import types

from flashinfer.gemm import gemm_base


def _uses_narrow_multi_tile(tactic, inputs) -> bool:
    tile, _, swap_ab, _, split_k_slices = tactic
    kernel_n = inputs[0].shape[0] if swap_ab else inputs[1].shape[1]
    return split_k_slices == 1 and tile[1] < 64 and kernel_n > tile[1]


@functools.cache
def _mxfp8_runner_type(*args, **kwargs):
    upstream = gemm_base._cute_dsl_gemm_mxfp8_runner(*args, **kwargs)

    class TokenspeedMxfp8GemmRunner(type(upstream)):
        def get_cache_key_extras(self, inputs):
            return (*super().get_cache_key_extras(inputs), "single-narrow-tile-v1")

        def get_valid_tactics(self, inputs, profile):
            return [
                tactic
                for tactic in super().get_valid_tactics(inputs, profile)
                if not _uses_narrow_multi_tile(tactic, inputs)
            ]

        def forward(self, inputs, tactic=None, do_preparation=False, **kwargs):
            if tactic is not None and tactic != -1:
                if _uses_narrow_multi_tile(tactic, inputs):
                    # Bucketed cache hits must also obey the actual shape's
                    # restriction. Upstream selects a valid default for this M.
                    tactic = -1
            return super().forward(
                inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
            )

    return TokenspeedMxfp8GemmRunner


def _mxfp8_runner(*args, **kwargs):
    return _mxfp8_runner_type(*args, **kwargs)()


@functools.cache
def _private_mm_mxfp8():
    raw = inspect.unwrap(gemm_base.mm_mxfp8)
    namespace = dict(raw.__globals__)
    namespace["_cute_dsl_gemm_mxfp8_runner"] = _mxfp8_runner
    clone = types.FunctionType(
        raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__
    )
    clone.__kwdefaults__ = raw.__kwdefaults__
    clone.__annotations__ = raw.__annotations__
    # Preserve upstream argument validation and backend selection around the
    # private body; only the cute-dsl runner factory is replaced.
    clone = gemm_base.flashinfer_api(trace=gemm_base.mm_mxfp8_trace)(clone)
    clone = gemm_base.backend_requirement(
        {
            "cutlass": gemm_base._cutlass_gemm_mxfp8_requirement,
            "trtllm": gemm_base._trtllm_gemm_mxfp8_requirement,
            "cute-dsl": gemm_base._cute_dsl_gemm_mxfp8_requirement,
            "cutedsl_low_latency": gemm_base._cutedsl_low_latency_gemm_mxfp8_requirement,
            "cudnn": gemm_base._cudnn_mm_mxfp8_requirement,
            "b12x": gemm_base._b12x_gemm_mxfp8_requirement,
        },
        common_check=gemm_base._check_mm_mxfp8_problem_size,
        heuristic_func=gemm_base._heuristic_func_mm_mxfp8,
    )(clone)
    namespace["mm_mxfp8"] = clone
    return clone


def mm_mxfp8(*args, **kwargs):
    return _private_mm_mxfp8()(*args, **kwargs)
