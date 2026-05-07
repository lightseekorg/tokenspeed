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

import torch

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.model_loader import get_model
from tokenspeed.runtime.utils import (
    get_available_gpu_memory,
    get_colorful_logger,
    set_cuda_arch,
)
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)

_DEEPSEEK_V3_ARCHITECTURES = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
    }
)


class WeightLoader:
    """Handles model weight loading from disk.

    This class is stateless and does not modify external state.
    It returns LoadedModel with all necessary information.
    """

    @staticmethod
    def _validate_fp8_kv_cache_args(
        model_config: ModelConfig,
        server_args: ServerArgs,
    ) -> None:
        if server_args.kv_cache_dtype != "fp8_e4m3":
            return

        architectures = getattr(
            getattr(model_config, "hf_config", None),
            "architectures",
            None,
        )
        is_deepseek_v3 = bool(
            architectures
            and any(arch in _DEEPSEEK_V3_ARCHITECTURES for arch in architectures)
        )
        if (
            is_deepseek_v3
            and server_args.attention_backend == "tokenspeed_mla"
            and server_args.quantization_param_path is None
        ):
            raise RuntimeError(
                "DeepSeek V3 with tokenspeed_mla requires FP8 MLA KV-cache "
                "scaling factors when --kv-cache-dtype fp8_e4m3 is used. Pass "
                "--quantization-param-path with KV-cache scales, or use an "
                "attention backend that supports an unquantized KV cache."
            )

    @staticmethod
    def load_model(
        model_config: ModelConfig,
        server_args: ServerArgs,
        device: str,
        gpu_id: int,
        memory_saver_adapter: TorchMemorySaverAdapter,
    ):
        """Load model from disk.

        Args:
            model_config: Model configuration
            server_args: Server arguments
            device: Device type ("cuda", "cpu")
            gpu_id: GPU ID
            memory_saver_adapter: Memory saver adapter

        Returns:
            LoadedModel with model and dtype
        """
        logger.info(
            "Load weight begin. avail mem=%.2f GB",
            get_available_gpu_memory(device, gpu_id),
        )

        # Reduce thread conflicts during weight loading
        if device != "cpu":
            torch.set_num_threads(1)

        set_cuda_arch()
        WeightLoader._validate_fp8_kv_cache_args(model_config, server_args)

        # Create load config
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            ext_yaml=server_args.ext_yaml,
            weight_loader_prefetch_checkpoints=server_args.weight_loader_prefetch_checkpoints,
            weight_loader_prefetch_num_threads=server_args.weight_loader_prefetch_num_threads,
        )

        # Load model with memory saver context
        with memory_saver_adapter.region():
            model = get_model(
                model_config=model_config,
                load_config=load_config,
                device_config=DeviceConfig(device),
            )

        # Load KV cache scaling factors if using FP8
        if server_args.kv_cache_dtype == "fp8_e4m3":
            if server_args.quantization_param_path is not None:
                if callable(getattr(model, "load_kv_cache_scales", None)):
                    model.load_kv_cache_scales(server_args.quantization_param_path)
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        f"model {model.__class__} does not support loading scaling factors."
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors provided. "
                    "Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        dtype = model_config.dtype

        logger.info(
            "Load weight end. type=%s, dtype=%s, avail mem=%.2f GB",
            type(model).__name__,
            dtype,
            get_available_gpu_memory(device, gpu_id),
        )

        return model
