"""
Tests for MiniMax-M2 model support.

Usage:

# Run generation comparison test (HF vs RT logits)
ONLY_RUN=MiniMaxAI/MiniMax-M2.5 python3 -m unittest test_minimax_models.TestMiniMaxGeneration.test_generation

# Run GSM8K accuracy test.
python3 test_minimax_models.py TestMiniMaxGSM8K
"""

import dataclasses
import multiprocessing as mp
import os
import sys
import unittest
from typing import List

import torch

# Add project root directory to path for importing test.runners
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
from test.runners import DEFAULT_PROMPTS, HFRunner, RTRunner, check_close_model_outputs
from test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    kill_process_tree,
    popen_api_server,
    run_evalscope,
)

from tokenspeed.runtime.configs.minimax_m2_config import MiniMaxM2Config
from tokenspeed.runtime.models.minimax_m2 import get_spec_layer_idx_from_weight_name


def get_available_gpu_count() -> int:
    """Get the number of available GPUs in the environment."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 5e-2
    rouge_l_tolerance: float = 1
    skip_long_prompt: bool = False
    trust_remote_code: bool = False
    disable_prefill_graph: bool = False
    max_model_len: int = None
    max_total_tokens: int = None


_AVAILABLE_GPUS = get_available_gpu_count()

MINIMAX_MODELS = [
    ModelCase(
        "MiniMaxAI/MiniMax-M2.5",
        tp_size=_AVAILABLE_GPUS,
        disable_prefill_graph=True,
        skip_long_prompt=True,
        max_total_tokens=32768,
        max_model_len=16384,
    ),
]


class TestMiniMaxConfig(unittest.TestCase):
    """Config compatibility checks for MiniMax-M2 family checkpoints."""

    def test_m27_public_config_metadata_fields_are_preserved(self):
        config = MiniMaxM2Config(
            architectures=["MiniMaxM2ForCausalLM"],
            attn_type_list=[1] * 62,
            dtype="bfloat16",
            max_position_embeddings=204800,
            quantization_config={
                "activation_scheme": "dynamic",
                "fmt": "float8_e4m3fn",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
            shared_intermediate_size=0,
            num_mtp_modules=3,
            mtp_transformer_layers=1,
            use_mtp=True,
        )

        self.assertEqual(config.model_type, "minimax_m2")
        self.assertEqual(config.architectures, ["MiniMaxM2ForCausalLM"])
        self.assertEqual(config.max_position_embeddings, 204800)
        self.assertEqual(config.attn_type_list, [1] * 62)
        self.assertEqual(config.dtype, "bfloat16")
        self.assertEqual(config.to_dict()["dtype"], "bfloat16")
        self.assertEqual(config.quantization_config["quant_method"], "fp8")
        self.assertEqual(config.quantization_config["weight_block_size"], [128, 128])
        self.assertEqual(config.to_dict()["shared_intermediate_size"], 0)
        self.assertEqual(config.num_mtp_modules, 3)
        self.assertEqual(config.mtp_transformer_layers, 1)
        self.assertTrue(config.use_mtp)

    def test_speculative_layers_after_main_model_are_skipped_by_loader(self):
        config = MiniMaxM2Config(
            num_hidden_layers=62,
            num_mtp_modules=3,
            mtp_transformer_layers=2,
        )

        self.assertIsNone(
            get_spec_layer_idx_from_weight_name(
                config, "model.layers.61.self_attn.q_proj.weight"
            )
        )
        self.assertEqual(
            get_spec_layer_idx_from_weight_name(
                config, "model.layers.62.self_attn.q_proj.weight"
            ),
            62,
        )
        self.assertEqual(
            get_spec_layer_idx_from_weight_name(
                config, "model.layers.64.block_sparse_moe.gate.weight"
            ),
            64,
        )
        self.assertEqual(
            get_spec_layer_idx_from_weight_name(
                config, "model.layers.67.block_sparse_moe.gate.weight"
            ),
            67,
        )
        self.assertIsNone(
            get_spec_layer_idx_from_weight_name(
                config, "model.layers.68.self_attn.q_proj.weight"
            )
        )


class TestMiniMaxGeneration(unittest.TestCase):
    """Compare HFRunner vs RTRunner output logits and strings for MiniMax-M2."""

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

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
            tp_size=model_case.tp_size,
            max_model_len=model_case.max_model_len,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens)
            if torch.cuda.current_device() == 0:
                print(f"\n{'=' * 60}", flush=True)
                print(f"[HFRunner] model={model_path}", flush=True)
                for i, (prompt, output) in enumerate(
                    zip(prompts, hf_outputs.output_strs)
                ):
                    print(
                        f"  [{i}] prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                        flush=True,
                    )
                    print(
                        f"  [{i}] output: {output[:100]}{'...' if len(output) > 100 else ''}",
                        flush=True,
                    )
                print(f"{'=' * 60}\n", flush=True)

        with RTRunner(
            model_path,
            world_size=model_case.tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
            disable_prefill_graph=model_case.disable_prefill_graph,
            max_total_tokens=model_case.max_total_tokens,
            max_model_len=model_case.max_model_len,
        ) as rt_runner:
            rt_outputs = rt_runner.forward(prompts, max_new_tokens=max_new_tokens)
            if torch.cuda.current_device() == 0:
                print(f"\n{'=' * 60}", flush=True)
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
                print(f"{'=' * 60}\n", flush=True)

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            rt_outputs=rt_outputs,
            prefill_tolerance=prefill_tolerance,
            decode_tolerance=decode_tolerance,
            rouge_l_tolerance=rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={prompts}",
        )

    def test_generation(self):
        """Test MiniMax-M2.5 generation output matches between HF and RT."""
        if is_in_ci():
            return

        for model_case in MINIMAX_MODELS:
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
            self.assert_close_logits_and_output_strs(
                prompts, model_case, torch.bfloat16
            )


class TestMiniMaxGSM8K(unittest.TestCase):
    """Launch MiniMax-M2 server and run GSM8K accuracy evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.model = "MiniMaxAI/MiniMax-M2.5"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_api_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        metrics = run_evalscope(
            base_url=self.base_url,
            model=self.model,
            dataset="gsm8k",
            limit=200,
            eval_batch_size=128,
            generation_config={"max_tokens": 512},
            dataset_args={"gsm8k": {"few_shot_num": 5, "few_shot_random": False}},
        )
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
