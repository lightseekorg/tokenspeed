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

"""Runtime configuration exports."""

from tokenspeed.runtime.configs.base_config import BaseConfig
from tokenspeed.runtime.configs.deepseek_v2_config import DeepseekV2Config
from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config
from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.deepseek_v32_config import DeepseekV32Config
from tokenspeed.runtime.configs.glm53_flash_config import Glm53FlashConfig
from tokenspeed.runtime.configs.glm_moe_dsa_config import GlmMoeDsaConfig
from tokenspeed.runtime.configs.gpt_oss_config import GptOssConfig
from tokenspeed.runtime.configs.inkling_config import (
    InklingAudioConfig,
    InklingMMConfig,
    InklingModelConfig,
    InklingVisionConfig,
)
from tokenspeed.runtime.configs.kimi_k2_config import KimiK2Config
from tokenspeed.runtime.configs.kimi_k3_config import (
    KimiK3Config,
    KimiK3VisionConfig,
    KimiLinearConfig,
)
from tokenspeed.runtime.configs.kimi_k3_dspark_config import KimiK3DSparkConfig
from tokenspeed.runtime.configs.kimi_k25_config import KimiK25Config
from tokenspeed.runtime.configs.llama4_config import Llama4TextConfig
from tokenspeed.runtime.configs.llama_config import LlamaConfig
from tokenspeed.runtime.configs.longcat_flash_config import LongcatFlashConfig
from tokenspeed.runtime.configs.minimax_m3_config import MiniMaxM3Config
from tokenspeed.runtime.configs.qwen2_config import Qwen2Config
from tokenspeed.runtime.configs.qwen3_5_config import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from tokenspeed.runtime.configs.qwen3_asr_config import (
    Qwen3ASRAudioEncoderConfig,
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
)
from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.configs.qwen3_moe_config import Qwen3MoeConfig
from tokenspeed.runtime.configs.qwen3_omni_moe_config import Qwen3OmniMoeConfig
from tokenspeed.runtime.configs.qwen4_exp_config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)

_CONFIG_CLASSES = [
    DeepseekV2Config,
    DeepseekV3Config,
    DeepseekV32Config,
    DeepseekV4Config,
    Glm53FlashConfig,
    GlmMoeDsaConfig,
    GptOssConfig,
    InklingAudioConfig,
    InklingMMConfig,
    InklingModelConfig,
    InklingVisionConfig,
    KimiK2Config,
    KimiK25Config,
    KimiK3Config,
    KimiK3DSparkConfig,
    KimiK3VisionConfig,
    KimiLinearConfig,
    Llama4TextConfig,
    LlamaConfig,
    LongcatFlashConfig,
    MiniMaxM3Config,
    Qwen2Config,
    Qwen3ASRAudioEncoderConfig,
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3OmniMoeConfig,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
]


def get_config_class(model_type: str) -> type[BaseConfig] | None:
    """Get the configuration class by its model_type."""
    for cls in _CONFIG_CLASSES:
        if getattr(cls, "model_type", None) == model_type:
            return cls
    return None


__all__ = [cls.__name__ for cls in _CONFIG_CLASSES]
