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
import os
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
    is_in_profile_measurement,
)
from flashinfer.fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
    Sm100BlockwiseGemmKernel,
)
from tokenspeed_kernel.thirdparty.trtllm_blockwise._launch import _run


@lru_cache(maxsize=512)
def _nearest_token_buckets(tokens: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            get_hybrid_num_tokens_buckets(1 << 30),
            key=lambda bucket: (abs(bucket - tokens), bucket),
        )
    )


_CACHE_EXTRAS = ("e4m3_fp32_128x128_bf16_v1", "profile_l2_buffers")


class BlockwiseGemmRunner(TunableRunner):
    """Search upstream tile, cluster and 2-CTA choices for block-FP8 GEMM."""

    def __init__(self):
        self._profile_inputs: list[list[torch.Tensor]] = []
        self._profile_step = 0
        self._profile_base_repeat = 0
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0, 2, 4),
                    (0, 0, 0),
                    get_hybrid_num_tokens_buckets,
                    map_to_hybrid_bucket_uncapped,
                ),
            ),
            tensor_initializers=(
                (0, autotuner_initializer_randn),
                (2, autotuner_initializer_ones),
                (4, autotuner_initializer_empty),
            ),
            use_cold_l2_cache=False,
            use_cuda_graph=True,
            inputs_pre_hook=self._prepare_profile_inputs,
        )

    def __hash__(self) -> int:
        # FlashInfer includes the runner hash in in-memory cache keys.
        return id(self)

    def _prepare_profile_inputs(self, inputs):
        # Each profile bucket owns its temporary buffers only while tuning.
        self._profile_inputs = []
        self._profile_step = 0
        tuner = AutoTuner.get()
        l2_bytes = tuner._get_l2_cache_size_in_bytes(inputs[0].device.index)
        group_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs)
        groups = (l2_bytes + group_bytes - 1) // group_bytes
        self._profile_inputs = [inputs] + [
            [tensor.clone() for tensor in inputs] for _ in range(groups - 1)
        ]
        # Give every group the same number of calls in each timed graph.
        tuner.repeat = groups * max(
            1, (self._profile_base_repeat + groups - 1) // groups
        )
        return inputs

    def get_cache_key_extras(self, inputs) -> tuple:
        # Shape keys already include N/K. Separate devices and this scale contract.
        return (*_CACHE_EXTRAS, inputs[0].device.index)

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

    def forward(self, inputs, tactic=-1):
        if self._profile_inputs and is_in_profile_measurement():
            inputs = self._profile_inputs[
                self._profile_step % len(self._profile_inputs)
            ]
            self._profile_step += 1
        return _run(*inputs, tactic)


_runner = BlockwiseGemmRunner()


def run(a, b, a_scales, b_scales, out):
    inputs = [a, b, a_scales, b_scales, out]
    tuner = AutoTuner.get()
    op = "trtllm_cutedsl_fp8_blockscale"
    if tuner.is_tuning_mode and op not in tuner._effective_skip_ops:
        original_repeat = tuner.repeat
        _runner._profile_base_repeat = original_repeat
        try:
            with torch.cuda.device(a.device):
                runner, tactic = tuner.choose_one(
                    op,
                    [_runner],
                    _runner.tuning_config,
                    inputs,
                )
        finally:
            tuner.repeat = original_repeat
            _runner._profile_inputs = []
    else:
        runner, tactic = tuner.choose_one(op, [_runner], _runner.tuning_config, inputs)

    if tactic == -1 and not (
        tuner.is_tuning_mode
        or op in tuner._effective_skip_ops
        or tuner._get_override_stack()
        or tuner._in_v2_context
        or tuner._effective_measure_policy is not None
        or tuner._active_managed_store is not None
    ):
        # Avoid probing every bucket when this operation has no tuned entries.
        b_shape = tuple(b.shape)
        extras = (*_CACHE_EXTRAS, a.device.index)
        has_memory = any(
            key.custom_op == op
            and key.nearest_profile[1] == b_shape
            and key.extras == extras
            for key in tuner.profiling_cache
        )
        prefix = f"({op!r}, "
        b_shape_text = f", {b_shape!r}, "
        extras_suffix = f", {extras!r})"
        has_file = not has_memory and any(
            key.endswith(extras_suffix)
            and key.startswith(prefix)
            and b_shape_text in key
            for key in tuner._file_configs
        )
        if (
            has_memory
            or has_file
            or os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE") == "1"
        ):
            shapes = tuple(tuple(tensor.shape) for tensor in inputs)
            hit, _, _, _ = tuner.search_cache(
                op, [_runner], shapes, _runner.tuning_config, inputs=inputs
            )
            if not hit:
                # M is a signed 32-bit kernel argument. Query cached buckets in
                # distance order using FlashInfer's normal validation. The
                # current bucket was already checked above.
                m = a.shape[0]
                current_bucket = map_to_hybrid_bucket_uncapped(m)
                for bucket in _nearest_token_buckets(m):
                    if bucket == current_bucket:
                        continue
                    bucket_shapes = tuple(
                        (bucket, *shape[1:]) if index in (0, 2, 4) else shape
                        for index, shape in enumerate(shapes)
                    )
                    hit, _, cached_tactic, _ = tuner.search_cache(
                        op,
                        [_runner],
                        bucket_shapes,
                        _runner.tuning_config,
                        inputs=inputs,
                    )
                    if hit:
                        runner, tactic = _runner, cached_tactic
                        break
    return runner(inputs, tactic=tactic)
