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

"""Model configuration helpers and derived runtime metadata."""

import copy
import json
import math
import os
from enum import IntEnum, auto

import torch
import yaml
from transformers import PretrainedConfig

from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_generation_config,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)

_DEEPSEEK_V4_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
    }
)
_MLA_ARCHITECTURES = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
        "Eagle3DeepseekV2ForCausalLM",
        "LongcatFlashForCausalLM",
    }
)
_DOUBLE_ATTENTION_LAYER_ARCHITECTURES = frozenset(
    {
        "LongcatFlashForCausalLM",
    }
)


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()


def override_model_config(model_config, ext_yaml):
    with open(ext_yaml) as f:
        ext_config = yaml.safe_load(f)

    override_model_config: dict = ext_config.get("override_model_config", {})
    for k, v in override_model_config.items():
        if hasattr(model_config, k):
            old_v = model_config.__getattribute__(k)
            if isinstance(v, dict):
                new_v = copy.deepcopy(old_v)
                new_v.__dict__.update(v)
            else:
                new_v = v
            model_config.__setattr__(k, new_v)
            logger.info("Override model config: %s=%r", k, new_v)


def is_deepseek_v4(config: PretrainedConfig) -> bool:
    return (
        config.architectures is not None
        and config.architectures[0] in _DEEPSEEK_V4_ARCHITECTURES
    )


def configure_deepseek_v4_attention(model_config) -> None:
    """Derive DeepSeek V4's MLA-like dimensions for runtime setup."""

    hf_config = model_config.hf_config
    model_config.head_dim = hf_config.head_dim
    model_config.attention_arch = AttentionArch.MLA
    model_config.kv_lora_rank = hf_config.head_dim
    model_config.qk_rope_head_dim = hf_config.qk_rope_head_dim
    model_config.qk_nope_head_dim = hf_config.head_dim - hf_config.qk_rope_head_dim
    model_config.v_head_dim = hf_config.head_dim
    model_config.index_head_dim = getattr(hf_config, "index_head_dim", None)
    model_config.scaling = 1 / math.sqrt(model_config.head_dim)
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        scaling_factor = rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        model_config.scaling = model_config.scaling * mscale * mscale


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: str | None = None,
        context_length: int | None = None,
        model_override_args: dict | None = None,
        dtype: str = "auto",
        quantization: str | None = None,
        override_config_file: str | None = None,
        is_draft_worker: bool | None = False,
        server_args: ServerArgs = None,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self.quantization = quantization
        self.mapping = server_args.mapping

        # Parse args
        self.model_override_args = json.loads(model_override_args)
        kwargs = {}
        if override_config_file and override_config_file.strip():
            kwargs["_configuration_file"] = override_config_file.strip()

        self.hf_config = get_config(
            model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=self.model_override_args,
            is_draft_worker=is_draft_worker,
            **kwargs,
        )
        self.hf_generation_config = get_generation_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        self.hf_text_config = get_hf_text_config(self.hf_config)

        # Check model type
        self.is_generation = is_generation_model(self.hf_config.architectures)
        self.is_multimodal = is_multimodal_model(self.hf_config.architectures)
        self.is_multimodal_gen = is_multimodal_gen_model(self.hf_config.architectures)
        self.is_image_gen = is_image_gen_model(self.hf_config.architectures)
        self.is_audio_model = is_audio_model(self.hf_config.architectures)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)

        # Derive context length
        derived_context_len = get_context_length(self.hf_text_config)
        if context_length is not None:
            if context_length > derived_context_len:
                if envs.TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.get():
                    logger.warning(
                        "User-specified context_length (%s) is greater than the derived "
                        "context_length (%s). This may lead to incorrect model outputs or "
                        "CUDA errors.",
                        context_length,
                        derived_context_len,
                    )
                    self.context_len = context_length
                else:
                    raise ValueError(
                        f"User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                        f"This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config. "
                        f"To allow overriding this maximum, set the env var TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
                    )
            else:
                self.context_len = context_length
        else:
            self.context_len = derived_context_len

        # Unify the config keys for hf_text_config
        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )

        # MLA models carry per-head dimension metadata that does not follow the
        # standard hidden_size / num_attention_heads derivation above.
        if is_deepseek_v4(self.hf_config):
            configure_deepseek_v4_attention(self)
        elif any(arch in _MLA_ARCHITECTURES for arch in self.hf_config.architectures):
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
            self.v_head_dim = self.hf_config.v_head_dim

            # Handle rope scaling with yarn
            self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
            rope_scaling = getattr(self.hf_config, "rope_scaling", None)
            if rope_scaling and "factor" in rope_scaling:
                mscale_all_dim = self.hf_config.rope_scaling.get(
                    "mscale_all_dim", False
                )
                scaling_factor = self.hf_config.rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale

        elif "MiniCPM3ForCausalLM" in self.hf_config.architectures:
            self.head_dim = 128
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        else:
            self.attention_arch = AttentionArch.MHA

        self.num_attention_heads = self.hf_text_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )

        # for Dbrx and MPT models
        if self.hf_config.model_type in {"dbrx", "mpt"}:
            self.num_key_value_heads = getattr(
                self.hf_config.attn_config, "kv_n_heads", None
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.hf_text_config.hidden_size
        self.num_hidden_layers = getattr(self.hf_text_config, "num_hidden_layers", None)
        if self.num_hidden_layers is None:
            self.num_hidden_layers = self.hf_text_config.num_layers
        self.num_attention_layers = self.num_hidden_layers
        if any(
            arch in _DOUBLE_ATTENTION_LAYER_ARCHITECTURES
            for arch in self.hf_config.architectures
        ):
            self.num_attention_layers = self.num_hidden_layers * 2
        self.vocab_size = self.hf_text_config.vocab_size

        # Verify quantization
        self._verify_quantization()

        # Cache attributes
        self.hf_eos_token_id = self.get_hf_eos_token_id()
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)

        if server_args is not None and server_args.load_format == "extensible":
            override_model_config(self, server_args.ext_yaml)

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        if quant_cfg is None:
            # modelopt NVFP4 checkpoints store quant config in hf_quant_config.json
            # Resolve the local snapshot directory (model_path may be a HF hub ID)
            if os.path.isdir(self.model_path):
                model_dir = self.model_path
            else:
                try:
                    from huggingface_hub import snapshot_download

                    model_dir = snapshot_download(
                        self.model_path,
                        revision=self.revision,
                        allow_patterns=["*.json"],
                        local_files_only=True,
                    )
                except Exception:
                    model_dir = None
            if model_dir is not None:
                hf_quant_path = os.path.join(model_dir, "hf_quant_config.json")
                if os.path.isfile(hf_quant_path):
                    with open(hf_quant_path) as f:
                        hf_quant = json.load(f)
                    quant_algo = hf_quant.get("quantization", {}).get("quant_algo", "")
                    if quant_algo:
                        quant_cfg = {
                            "quant_method": "modelopt",
                            "quant_algo": quant_algo,
                        }
                        quant_cfg.update(hf_quant.get("quantization", {}))
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = [*QUANTIZATION_METHODS]

        optimized_quantization_methods = [
            "fp8",
            "nvfp4",
            "mxfp4",
            "compressed_tensors",
            "compressed-tensors",
            "w8a8_fp8",
        ]
        compatible_quantization_methods = {
            "w8a8_fp8": ["compressed-tensors", "compressed_tensors"],
        }
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()
            # Detect which checkpoint is it
            for _, method in QUANTIZATION_METHODS.items():
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization
                )
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                if (
                    self.quantization not in compatible_quantization_methods
                    or quant_method
                    not in compatible_quantization_methods[self.quantization]
                ):
                    raise ValueError(
                        "Quantization method specified in the model config "
                        f"({quant_method}) does not match the quantization "
                        f"method specified in the `quantization` argument "
                        f"({self.quantization})."
                    )

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}."
                )

            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.",
                    self.quantization,
                )

    def get_hf_eos_token_id(self) -> set[int] | None:
        eos_ids = getattr(self.hf_config, "eos_token_id", None)
        if eos_ids:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        if eos_ids is None:
            eos_ids = set()
        if self.hf_generation_config:
            generation_eos_ids = getattr(
                self.hf_generation_config, "eos_token_id", None
            )
            if generation_eos_ids:
                generation_eos_ids = (
                    {generation_eos_ids}
                    if isinstance(generation_eos_ids, int)
                    else set(generation_eos_ids)
                )
                eos_ids = eos_ids | generation_eos_ids
        return eos_ids


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    class_name = config.architectures[0]
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        #  We set `torch_dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in `python/tokenspeed/runtime/models/llava.py`.
        setattr(config, "torch_dtype", torch.float16)
        return config

    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    else:
        return config


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str | torch.dtype,
) -> torch.dtype:
    #  getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.bfloat16

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                if config.model_type == "gemma2":
                    logger.info(
                        "For Gemma 2, we downcast float32 to bfloat16 instead "
                        "of float16 by default. Please specify `dtype` if you "
                        "want to use float16."
                    )
                    torch_dtype = torch.bfloat16
                else:
                    # Following the common practice, we use float16 for float32
                    # models.
                    torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def is_generation_model(model_architectures: list[str]):
    return True


def is_multimodal_model(model_architectures: list[str]):
    multimodal_architectures = {
        "LlavaLlamaForCausalLM",
        "LlavaQwenForCausalLM",
        "LlavaMistralForCausalLM",
        "LlavaVidForCausalLM",
        "MllamaForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "MiniCPMV",
    }
    return any(arch in multimodal_architectures for arch in model_architectures)


def is_multimodal_gen_model(model_architectures: list[str]):
    return False


def is_image_gen_model(model_architectures: list[str]):
    return False


def is_audio_model(model_architectures: list[str]):
    return False


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
