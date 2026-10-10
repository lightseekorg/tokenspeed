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

"""Unit tests for the offline batch inference example recipe."""

import json
import sys
import tempfile
from unittest.mock import MagicMock, patch

from examples.offline_batch_inference import (
    main,
    parse_args,
    run_batch_inference,
)


def test_parse_args_defaults():
    with patch.object(
        sys,
        "argv",
        [
            "offline_batch_inference.py",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
    ):
        args = parse_args()
        assert args.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert args.tensor_parallel_size == 1
        assert args.max_model_len == 8192
        assert args.gpu_memory_utilization == 0.90
        assert args.temperature == 0.0
        assert args.max_new_tokens == 256
        assert args.attention_backend is None
        assert args.sampling_backend is None
        assert args.output_file is None


def test_parse_args_custom_values():
    with patch.object(
        sys,
        "argv",
        [
            "offline_batch_inference.py",
            "--model-path",
            "Qwen/Qwen2.5-7B-Instruct",
            "--tp",
            "4",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.85",
            "--temperature",
            "0.7",
            "--max-new-tokens",
            "128",
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "triton",
            "--output-file",
            "eval_results.json",
        ],
    ):
        args = parse_args()
        assert args.model == "Qwen/Qwen2.5-7B-Instruct"
        assert args.tensor_parallel_size == 4
        assert args.max_model_len == 16384
        assert args.gpu_memory_utilization == 0.85
        assert args.temperature == 0.7
        assert args.max_new_tokens == 128
        assert args.attention_backend == "triton"
        assert args.sampling_backend == "triton"
        assert args.output_file == "eval_results.json"


def test_run_batch_inference_lifecycle():
    fake_engine = MagicMock()
    fake_outputs = [
        {"text": "Response 1", "output_ids": [10, 20, 30]},
        {"text": "Response 2", "output_ids": [40, 50]},
    ]
    fake_engine.generate.return_value = fake_outputs

    mock_engine_cls = MagicMock(return_value=fake_engine)
    mock_server_args = MagicMock()

    mock_engine_module = MagicMock(Engine=mock_engine_cls)
    mock_server_args_module = MagicMock(ServerArgs=mock_server_args)

    with patch.dict(
        sys.modules,
        {
            "tokenspeed.runtime.entrypoints.engine": mock_engine_module,
            "tokenspeed.runtime.utils.server_args": mock_server_args_module,
            "tokenspeed.runtime.entrypoints": MagicMock(Engine=mock_engine_cls),
            "tokenspeed.runtime.utils": MagicMock(ServerArgs=mock_server_args),
            "tokenspeed.runtime": MagicMock(),
            "tokenspeed": MagicMock(),
        },
    ):
        prompts = ["Question 1", "Question 2"]
        results = run_batch_inference(
            model="test-model",
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            temperature=0.0,
            max_new_tokens=100,
            prompts=prompts,
            attention_backend="triton",
            sampling_backend="triton",
        )

        mock_server_args.assert_called_once_with(
            model="test-model",
            attn_tp_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            attention_backend="triton",
            sampling_backend="triton",
            log_level="error",
        )
        mock_engine_cls.assert_called_once_with(
            server_args=mock_server_args.return_value
        )
        fake_engine.generate.assert_called_once_with(
            prompt=prompts,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": 100,
                "ignore_eos": False,
            },
        )
        fake_engine.shutdown.assert_called_once()
        assert results == fake_outputs


def test_run_batch_inference_shutdown_on_exception():
    fake_engine = MagicMock()
    fake_engine.generate.side_effect = RuntimeError("Forward pass failed")

    mock_engine_cls = MagicMock(return_value=fake_engine)
    mock_server_args = MagicMock()

    mock_engine_module = MagicMock(Engine=mock_engine_cls)
    mock_server_args_module = MagicMock(ServerArgs=mock_server_args)

    with patch.dict(
        sys.modules,
        {
            "tokenspeed.runtime.entrypoints.engine": mock_engine_module,
            "tokenspeed.runtime.utils.server_args": mock_server_args_module,
        },
    ):
        try:
            run_batch_inference(
                model="test-model",
                tensor_parallel_size=1,
                max_model_len=2048,
                gpu_memory_utilization=0.9,
                temperature=0.0,
                max_new_tokens=50,
                prompts=["Prompt"],
            )
        except RuntimeError:
            pass

        # Guarantee shutdown is invoked even if generate() raises
        fake_engine.shutdown.assert_called_once()


def test_main_with_output_file():
    fake_outputs = [
        {"text": "Output 1", "output_ids": [1, 2]},
        {"text": "Output 2", "output_ids": [3, 4, 5]},
        {"text": "Output 3", "output_ids": [6]},
        {"text": "Output 4", "output_ids": [7, 8]},
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    with patch(
        "examples.offline_batch_inference.run_batch_inference",
        return_value=fake_outputs,
    ), patch.object(
        sys,
        "argv",
        [
            "offline_batch_inference.py",
            "--model",
            "mock-model",
            "--output-file",
            tmp_path,
        ],
    ):
        exit_code = main()
        assert exit_code == 0

        with open(tmp_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert len(saved_data) == 4
        assert saved_data[0]["text"] == "Output 1"
        assert saved_data[0]["token_count"] == 2
