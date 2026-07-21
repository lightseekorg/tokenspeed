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
from collections.abc import Callable
from dataclasses import dataclass
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
    resolve_architecture,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)

_DEEPSEEK_V4_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForCausalLMNextN",
    }
)
_MLA_ARCHITECTURES = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
        "Eagle3DeepseekV2ForCausalLM",
        "LongcatFlashForCausalLM",
        "KimiK25ForConditionalGeneration",
    }
)
_DSA_ARCHITECTURES = frozenset(
    {
        "GlmMoeDsaForCausalLM",
        "GlmMoeDsaForCausalLMNextN",
    }
)
_MSA_ARCHITECTURES = frozenset(
    {
        "MiniMaxM3SparseForConditionalGeneration",
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
    DSA = auto()
    MSA = auto()


@dataclass(frozen=True)
class _AttentionFamilySpec:
    name: str
    architectures: frozenset[str]
    configure: Callable[[object], None]
    default_backend: str | None = None
    supports_target_verify_forward_mode: bool = False
    default_block_size: int | None = None


def override_model_config(model_config, ext_yaml):
    with open(ext_yaml, encoding="utf-8") as f:
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
    return resolve_architecture(config) in _DEEPSEEK_V4_ARCHITECTURES


def is_deepseek_v4_nextn(config: PretrainedConfig) -> bool:
    return resolve_architecture(config) == "DeepseekV4ForCausalLMNextN"


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


def configure_glm_attention(model_config) -> None:
    mla_config = (
        model_config.hf_text_config
        if hasattr(model_config.hf_text_config, "kv_lora_rank")
        else model_config.hf_config
    )
    required_fields = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_topk",
        "index_head_dim",
        "index_n_heads",
    )
    missing_fields = [
        field for field in required_fields if not hasattr(mla_config, field)
    ]
    if missing_fields:
        raise ValueError(
            "GLM attention config is missing required fields: "
            + ", ".join(missing_fields)
        )

    model_config.head_dim = getattr(mla_config, "qk_head_dim", None)
    if model_config.head_dim is None:
        model_config.head_dim = (
            mla_config.qk_nope_head_dim + mla_config.qk_rope_head_dim
        )
    model_config.attention_arch = AttentionArch.DSA
    model_config.kv_lora_rank = mla_config.kv_lora_rank
    model_config.qk_nope_head_dim = mla_config.qk_nope_head_dim
    model_config.qk_rope_head_dim = mla_config.qk_rope_head_dim
    model_config.v_head_dim = mla_config.v_head_dim
    model_config.index_topk = mla_config.index_topk
    model_config.index_head_dim = mla_config.index_head_dim
    model_config.index_n_heads = mla_config.index_n_heads
    model_config.index_topk_pattern = getattr(mla_config, "index_topk_pattern", None)

    model_config.scaling = 1 / math.sqrt(
        model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    )
    rope_scaling = getattr(mla_config, "rope_scaling", None)
    if rope_scaling and "factor" in rope_scaling:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        scaling_factor = rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        model_config.scaling = model_config.scaling * mscale * mscale


def configure_mla_attention(model_config) -> None:
    mla_config = (
        model_config.hf_text_config
        if hasattr(model_config.hf_text_config, "kv_lora_rank")
        else model_config.hf_config
    )
    model_config.head_dim = 256
    model_config.attention_arch = AttentionArch.MLA
    model_config.kv_lora_rank = mla_config.kv_lora_rank
    model_config.qk_nope_head_dim = mla_config.qk_nope_head_dim
    model_config.qk_rope_head_dim = mla_config.qk_rope_head_dim
    model_config.v_head_dim = mla_config.v_head_dim

    model_config.scaling = 1 / math.sqrt(
        model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    )
    rope_scaling = getattr(mla_config, "rope_scaling", None)
    if rope_scaling and "factor" in rope_scaling:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        scaling_factor = rope_scaling["factor"]
        mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        model_config.scaling = model_config.scaling * mscale * mscale


def configure_minimax_m3_attention(model_config) -> None:
    model_config.attention_arch = AttentionArch.MSA


_ATTENTION_FAMILY_SPECS = (
    _AttentionFamilySpec(
        name="DeepSeek V4",
        architectures=_DEEPSEEK_V4_ARCHITECTURES,
        configure=configure_deepseek_v4_attention,
        supports_target_verify_forward_mode=True,
        default_block_size=256,
    ),
    _AttentionFamilySpec(
        name="GLM",
        architectures=_DSA_ARCHITECTURES,
        configure=configure_glm_attention,
        default_backend="dsa",
        supports_target_verify_forward_mode=True,
    ),
    _AttentionFamilySpec(
        name="MLA",
        architectures=_MLA_ARCHITECTURES,
        configure=configure_mla_attention,
    ),
    _AttentionFamilySpec(
        name="MiniMax MSA",
        architectures=_MSA_ARCHITECTURES,
        configure=configure_minimax_m3_attention,
        default_block_size=128,
    ),
)


def _model_architectures(
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> list[str]:
    return (
        [resolve_architecture(hf_config)]
        + list(getattr(hf_config, "architectures", None) or [])
        + list(getattr(hf_text_config, "architectures", []) or [])
    )


def _resolve_attention_family(
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> _AttentionFamilySpec | None:
    architectures = _model_architectures(hf_config, hf_text_config)
    for spec in _ATTENTION_FAMILY_SPECS:
        if any(arch in spec.architectures for arch in architectures):
            return spec
    return None


def _apply_attention_family_defaults(
    server_args: ServerArgs,
    spec: _AttentionFamilySpec,
) -> None:
    if spec.default_block_size is not None:
        block_size_default = ServerArgs.__dataclass_fields__["block_size"].default
        if server_args.block_size == block_size_default:
            logger.info(
                "%s default block_size=%d; pass --block-size with a value other "
                "than %d to keep that value.",
                spec.name,
                spec.default_block_size,
                block_size_default,
            )
            server_args.block_size = spec.default_block_size
    if spec.default_backend is not None and server_args.attention_backend is None:
        server_args.attention_backend = spec.default_backend


def _derive_num_attention_layers(
    hf_config: PretrainedConfig,
    num_hidden_layers: int,
) -> int:
    architectures = getattr(hf_config, "architectures", None) or []
    num_attention_layers = num_hidden_layers
    if is_deepseek_v4_nextn(hf_config):
        num_attention_layers = int(getattr(hf_config, "num_nextn_predict_layers", 1))
    if any(arch in _DOUBLE_ATTENTION_LAYER_ARCHITECTURES for arch in architectures):
        num_attention_layers = num_hidden_layers * 2
    return num_attention_layers


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
        if (
            is_draft_worker
            and resolve_architecture(self.hf_config)
            == "InklingForConditionalGenerationNextN"
        ):
            # The MTP head's depth blocks have their own local/full attention
            # pattern (mtp_config.local_layer_ids) and only depths
            # 0..steps-1 ever run; swap in the depth-specialized (and
            # steps-pruned) text config so layer construction, attention
            # metadata, and paged-cache layout all derive from it.
            from tokenspeed.runtime.configs.inkling_config import (
                inkling_mtp_text_config,
            )

            self.hf_text_config = inkling_mtp_text_config(
                self.hf_text_config,
                num_steps=getattr(server_args, "speculative_num_steps", None),
            )
            if hasattr(self.hf_config, "text_config"):
                self.hf_config.text_config = self.hf_text_config

        # Check model type
        self.is_generation = is_generation_model(self.hf_config.architectures)
        self.is_multimodal = is_multimodal_model(self.hf_config.architectures)
        self.is_multimodal_gen = is_multimodal_gen_model(self.hf_config.architectures)
        self.is_image_gen = is_image_gen_model(self.hf_config.architectures)
        self.is_audio_model = is_audio_model(self.hf_config.architectures)

        language_model_only = bool(getattr(server_args, "language_model_only", False))
        # Target-only flag; never apply to draft / auxiliary checkpoints.
        apply_language_model_only = language_model_only and not is_draft_worker
        if apply_language_model_only:
            if not self.is_multimodal:
                raise ValueError(
                    "--language-model-only requires a multimodal model checkpoint."
                )
            logger.info(
                "Running in language-model-only mode: vision/audio encoders will "
                "be skipped; requests with multimodal inputs will be rejected."
            )
        # ``is_multimodal`` is the architectural fact; this is the runtime gate.
        self.is_multimodal_active = self.is_multimodal and not apply_language_model_only
        if (
            not is_draft_worker
            and getattr(server_args, "mm_encoder_tp_mode", "weights") == "data"
        ):
            if not self.is_multimodal_active:
                raise ValueError("item-DP requires an active multimodal encoder")
        # Vision-only role (EPD encode): the inverse axis of language_model_only.
        # Build the vision tower (is_multimodal_active stays True) but SKIP LM
        # construction + LM weight load so a full ViT fits at encode TP=1.
        encoder_only = (
            getattr(server_args, "disaggregation_mode", None) == "encode"
            and not is_draft_worker
        )
        if encoder_only and not self.is_multimodal:
            raise ValueError(
                "disaggregation_mode=encode requires a multimodal checkpoint."
            )
        if encoder_only and apply_language_model_only:
            raise ValueError(
                "disaggregation_mode=encode (encoder-only) and language_model_only "
                "are mutually exclusive."
            )
        if encoder_only and self.is_audio_model:
            raise ValueError(
                "disaggregation_mode=encode does not support audio models; "
                "only image/video encoders are currently supported."
            )
        if encoder_only:
            # Single model-facing gate: Kimi reads hf_config.encoder_only directly;
            # Qwen3_5ForConditionalGeneration reads it to skip LM construction.
            self.hf_config.encoder_only = True
            logger.info(
                "Running in encoder-only mode: the language model will not "
                "be constructed or loaded (encode role)."
            )
        # Cap gpu_memory_utilization for VLMs in mm mode — the vision encoder
        # needs headroom that the global default doesn't account for.
        if (
            self.is_multimodal_active
            and getattr(server_args, "_gpu_memory_utilization_defaulted", False)
            and server_args.gpu_memory_utilization > 0.9
        ):
            logger.info(
                "Clamping gpu_memory_utilization %.2f -> 0.9 to leave headroom "
                "for the vision encoder.",
                server_args.gpu_memory_utilization,
            )
            server_args.gpu_memory_utilization = 0.9
        self.mm_attention_backend = getattr(server_args, "mm_attention_backend", None)
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

        # MLA/DSA families carry per-head dimension metadata that does not
        # follow the standard hidden_size / num_attention_heads derivation above.
        attention_family = _resolve_attention_family(
            self.hf_config,
            self.hf_text_config,
        )
        if attention_family is not None:
            _apply_attention_family_defaults(server_args, attention_family)
            attention_family.configure(self)
        elif "MiniCPM3ForCausalLM" in self.hf_config.architectures:
            self.head_dim = 128
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        else:
            self.attention_arch = AttentionArch.MHA

        self.use_v4_mtp_paged_metadata = (
            getattr(server_args, "speculative_algorithm", None) is not None
            and not is_draft_worker
            and attention_family is not None
            and attention_family.supports_target_verify_forward_mode
        )

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
        self.num_attention_layers = _derive_num_attention_layers(
            self.hf_config,
            self.num_hidden_layers,
        )
        if is_draft_worker:
            mtp_layers = getattr(self.hf_text_config, "mtp_num_hidden_layers", None)
            if mtp_layers is not None:
                self.num_attention_layers = mtp_layers
            else:
                nextn_layers = getattr(
                    self.hf_text_config, "num_nextn_predict_layers", None
                )
                if nextn_layers is not None and nextn_layers > 0:
                    self.num_attention_layers = nextn_layers
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
                except Exception as exc:
                    logger.debug(
                        "Unable to resolve local quantization config for %s: %s",
                        self.model_path,
                        exc,
                    )
                    model_dir = None
            if model_dir is not None:
                hf_quant_path = os.path.join(model_dir, "hf_quant_config.json")
                if os.path.isfile(hf_quant_path):
                    with open(hf_quant_path, encoding="utf-8") as f:
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
    class_name = resolve_architecture(config)
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        # We set `dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in
        # `python/tokenspeed/runtime/models/llava.py`.
        config.dtype = torch.float16
        return config

    if hasattr(config, "thinker_config"):
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            return thinker_config.text_config
        return thinker_config
    if hasattr(config, "text_config"):
        if not hasattr(config.text_config, "num_attention_heads"):
            raise ValueError("text_config must define num_attention_heads")
        return config.text_config
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
    # config.dtype can be missing or None.
    config_dtype = getattr(config, "dtype", None)
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


def is_multimodal_model(model_architectures: list[str] | None):
    multimodal_architectures = {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3ASRForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "InklingForConditionalGeneration",
        "MiniMaxM3SparseForConditionalGeneration",
    }
    return any(arch in multimodal_architectures for arch in model_architectures or [])


def is_multimodal_gen_model(model_architectures: list[str]):
    return False


def is_image_gen_model(model_architectures: list[str]):
    return False


def is_audio_model(model_architectures: list[str] | None):
    audio_architectures = {
        "InklingForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3ASRForConditionalGeneration",
    }
    return any(arch in audio_architectures for arch in model_architectures or [])


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
