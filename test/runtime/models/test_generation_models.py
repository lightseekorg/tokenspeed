"""
Usage:

To test a specific model:
1. Add it to ALL_OTHER_MODELS
2. Run `ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others`
"""

import os

# CI Registration (parsed via AST, runtime no-op)
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="runtime-1gpu")
register_cuda_ci(est_time=300, suite="runtime-2gpu")

import dataclasses
import multiprocessing as mp
import os
import subprocess
import sys
import time
import unittest
from typing import List

import torch
from tokenspeed_kernel.platform import current_platform

# Add project root directory to path for importing test.runners
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
from test.runners import DEFAULT_PROMPTS, HFRunner, RTRunner, check_close_model_outputs
from test.test_utils import is_in_ci


def get_available_gpu_count() -> int:
    """Get the number of available GPUs in the environment."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


_BLACKWELL_SYSTEM = current_platform().is_blackwell


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 5e-2
    rouge_l_tolerance: float = 1
    skip_long_prompt: bool = False
    trust_remote_code: bool = False
    enforce_eager: bool = False
    max_model_len: int = None
    min_gpu_memory_gb: float = 0
    blackwell_only: bool = False
    extra_kwargs: dict = dataclasses.field(default_factory=dict)


# Popular models that run on the CI
# tp_size is set to available GPU count at runtime
_AVAILABLE_GPUS = get_available_gpu_count()
CI_MODELS = [
    ModelCase(
        "openai/gpt-oss-120b",
        tp_size=_AVAILABLE_GPUS,
        skip_long_prompt=True,
        min_gpu_memory_gb=150,
        extra_kwargs={
            "disable_prefill_graph": True,
            "max_total_tokens": 32768,
            "max_model_len": 16384,
            **({"moe_backend": "flashinfer_mxfp4"} if _BLACKWELL_SYSTEM else {}),
            "speculative_algorithm": "EAGLE3",
            "speculative_draft_model_path": "nvidia/gpt-oss-120b-Eagle3-long-context",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "gpu_memory_utilization": 0.9,
        },
    ),
    ModelCase(
        "txn545/Qwen3.5-35B-A3B-NVFP4",
        tp_size=_AVAILABLE_GPUS,
        skip_long_prompt=True,
        blackwell_only=True,
        extra_kwargs={
            "disable_prefill_graph": True,
            "max_total_tokens": 32768,
            "max_model_len": 16384,
            "moe_backend": "flashinfer_cutlass",
            "speculative_algorithm": "MTP",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "gpu_memory_utilization": 0.9,
        },
    ),
]

# All other models that do not run on the CI
ALL_OTHER_MODELS = [
    ModelCase("Qwen/Qwen2-1.5B-Instruct"),
    ModelCase("Qwen/Qwen3.5-27B"),
    ModelCase("Qwen/Qwen3.5-35B-A3B"),
    ModelCase("Qwen/Qwen3.5-122B-A10B"),
]

TORCH_DTYPES = [torch.bfloat16]


class TestGenerationModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_logits_and_output_strs(
        self,
        prompts: List[str],
        model_case: ModelCase,
        torch_dtype: torch.dtype,
    ) -> None:
        model_path = model_case.model_path
        prefill_tolerance, decode_tolerance, rouge_l_tolerance = (
            model_case.prefill_tolerance,
            model_case.decode_tolerance,
            model_case.rouge_l_tolerance,
        )
        max_new_tokens = 32

        # with HFRunner(
        #     model_path,
        #     torch_dtype=torch_dtype,
        #     model_type="generation",
        #     trust_remote_code=model_case.trust_remote_code,
        #     tp_size=model_case.tp_size,
        #     max_model_len=model_case.max_model_len,
        # ) as hf_runner:
        #     hf_outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens)
        #     if torch.cuda.current_device() == 0:
        #         print(f"\n{'='*60}", flush=True)
        #         print(f"[HFRunner] model={model_path}", flush=True)
        #         for i, (prompt, output) in enumerate(
        #             zip(prompts, hf_outputs.output_strs)
        #         ):
        #             print(
        #                 f"  [{i}] prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
        #                 flush=True,
        #             )
        #             print(
        #                 f"  [{i}] output: {output[:100]}{'...' if len(output) > 100 else ''}",
        #                 flush=True,
        #             )
        #         print(f"{'='*60}\n", flush=True)

        with RTRunner(
            model_path,
            world_size=model_case.tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
            enforce_eager=model_case.enforce_eager,
            # port=None uses auto-incrementing port
            **model_case.extra_kwargs,
        ) as rt_runner:
            if "speculative_algorithm" in model_case.extra_kwargs:
                rt_outputs = rt_runner.batch_forward(
                    prompts, max_new_tokens=max_new_tokens
                )
            else:
                rt_outputs = rt_runner.forward(prompts, max_new_tokens=max_new_tokens)
            if torch.cuda.current_device() == 0:
                print(f"\n{'='*60}", flush=True)
                print(f"[RTRunner] model={model_path}", flush=True)
                for i, (prompt, output) in enumerate(
                    zip(prompts, rt_outputs.output_strs)
                ):
                    print(
                        f"  [{i}] prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                        flush=True,
                    )
                    print(
                        f"  [{i}] output: {output[:100]}{'...' if len(output) > 100 else ''}",
                        flush=True,
                    )
                print(f"{'='*60}\n", flush=True)

    def test_ci_models(self):
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        for model_case in CI_MODELS:
            if model_case.blackwell_only and not _BLACKWELL_SYSTEM:
                print(f"Skipping {model_case.model_path}: Blackwell-only model")
                continue
            total_memory_gb = gpu_memory_gb * model_case.tp_size
            if (
                model_case.min_gpu_memory_gb > 0
                and total_memory_gb < model_case.min_gpu_memory_gb
            ):
                print(
                    f"Skipping {model_case.model_path}: requires {model_case.min_gpu_memory_gb}GB, got {total_memory_gb:.0f}GB ({gpu_memory_gb:.0f}GB x {model_case.tp_size})"
                )
                continue
            for torch_dtype in TORCH_DTYPES:

                # Skip long prompts for models that do not have a long context
                prompts = DEFAULT_PROMPTS
                if model_case.skip_long_prompt:
                    prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

                # Assert the logits and output strs are close
                self.assert_close_logits_and_output_strs(
                    prompts, model_case, torch_dtype
                )

    def test_others(self):
        if is_in_ci():
            return

        for model_case in ALL_OTHER_MODELS:
            # Only run a specified model
            if (
                "ONLY_RUN" in os.environ
                and os.environ["ONLY_RUN"] != model_case.model_path
            ):
                continue

            # Skip long prompts for models that do not have a long context
            prompts = DEFAULT_PROMPTS
            if model_case.skip_long_prompt:
                prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

            # Assert the logits and output strs are close
            self.assert_close_logits_and_output_strs(prompts, model_case, torch.float16)


if __name__ == "__main__":
    unittest.main()
