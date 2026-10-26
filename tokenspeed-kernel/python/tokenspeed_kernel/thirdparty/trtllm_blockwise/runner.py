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

"""TRT-LLM's FP8 tactic search adapted to the shared FlashInfer autotuner.

Candidate configurations follow CuteDSLFp8BlackwellRunner in TensorRT-LLM
commit ffddc5abd188f308adb5a2906112c2b1e7192d7d. Kernel source stays unchanged.
"""

import itertools
from functools import lru_cache

import cutlass
import torch
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
    autotuner_initializer_empty,
    autotuner_initializer_ones,
    autotuner_initializer_randn,
)
from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
    Sm100BlockwiseGemmKernel,
)
from tokenspeed_kernel.thirdparty.trtllm_blockwise._launch import _prepare, _run


def _token_bucket(tokens: int) -> int:
    return 1 << (max(1, tokens).bit_length() - 1)


def _token_buckets(tokens: int) -> tuple[int, ...]:
    return tuple(1 << i for i in range(max(1, tokens).bit_length()))


class BlockwiseGemmRunner(TunableRunner):
    """Search upstream tile, cluster and 2-CTA choices for block-FP8 GEMM."""

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec((0, 2, 4), (0, 0, 0), _token_buckets, _token_bucket),
        ),
        tensor_initializers=(
            (0, autotuner_initializer_randn),
            (2, autotuner_initializer_ones),
            (4, autotuner_initializer_empty),
        ),
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )

    def get_cache_key_extras(self, inputs) -> tuple:
        # Shape keys already include N/K. Separate devices and this scale contract.
        return ("e4m3_fp32_128x128_bf16_v1", inputs[0].device.index)

    def get_valid_tactics(self, inputs, profile) -> list:
        a, b = inputs[:2]
        m, k = a.shape
        n = b.shape[0]
        return [
            (two, tile, cluster)
            for two, tile, cluster in itertools.product(
                (False, True),
                ((64, 128), (128, 128), (256, 128)),
                (
                    (1, 1),
                    (1, 2),
                    (1, 4),
                    (2, 1),
                    (2, 2),
                    (2, 4),
                    (4, 1),
                    (4, 2),
                    (4, 4),
                ),
            )
            if Sm100BlockwiseGemmKernel.can_implement(
                cutlass.Float8E4M3FN,
                cutlass.Float32,
                cutlass.BFloat16,
                two,
                tile,
                cluster,
                m,
                n,
                k,
                1,
                "k",
                "k",
                "n",
            )
        ]

    def forward(self, inputs, tactic):
        return _run(*inputs, tactic)


_runner = BlockwiseGemmRunner()


@lru_cache(maxsize=512)
def _lookup_key(m, n, k, device):
    # Cache only shape metadata. Always read the current autotuner dictionaries,
    # so loading, clearing or replacing tactics needs no second invalidation path.
    shapes = ((m, k), (n, k), (m, k // 128), (n // 128, k // 128), (m, n))
    key = AutoTuner._get_cache_key(
        "trtllm_cutedsl_fp8_blockscale",
        _runner,
        shapes,
        _runner.tuning_config,
        ("e4m3_fp32_128x128_bf16_v1", device),
    )
    return key, key.file_key


def run(a, b, a_scales, b_scales, out):
    inputs = [a, b, a_scales, b_scales, out]
    tuner = AutoTuner.get()
    if not (
        tuner.is_tuning_mode
        or tuner._override_tuning_buckets is not None
        or tuner._override_round_up
        or tuner._effective_skip_ops
        or tuner._in_v2_context
        or tuner._effective_measure_policy is not None
        or tuner._active_managed_store is not None
    ):
        key, file_key = _lookup_key(
            _token_bucket(a.shape[0]), b.shape[0], a.shape[1], a.device.index
        )
        cached = tuner.profiling_cache.get(key)
        if cached is not None:
            return _run(*inputs, cached[0])
        cached = tuner._file_configs.get(file_key)
        if cached is not None and cached[0] == "BlockwiseGemmRunner":
            return _run(*inputs, cached[1])
        # A miss still uses native lookup, including legacy files and diagnostics.
    if (
        tuner.is_tuning_mode
        and "trtllm_cutedsl_fp8_blockscale" not in tuner._effective_skip_ops
    ):
        # Cache hits skip profiling, so prepare variants explicitly on restart.
        # All M/N/K remain dynamic; this work is done once per configuration/GPU.
        with torch.cuda.device(a.device):
            _prepare(a.device, _runner.get_valid_tactics(inputs, None))
            runner, tactic = tuner.choose_one(
                "trtllm_cutedsl_fp8_blockscale",
                [_runner],
                _runner.tuning_config,
                inputs,
            )
    else:
        runner, tactic = tuner.choose_one(
            "trtllm_cutedsl_fp8_blockscale", [_runner], _runner.tuning_config, inputs
        )
    return runner(inputs, tactic=tactic)
