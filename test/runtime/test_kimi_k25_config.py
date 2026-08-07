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

"""Regression tests for Kimi K2.5 / K3 nested text-config resolution.

``KimiK25Config`` and ``KimiK3Config`` nest their text backbone under
``text_config``. ``get_config`` resolves the class that parses that dict by
walking ``sub_configs``; without the mapping it falls back to the outer config
class and reads the outer (empty) ``attribute_map``, so a checkpoint's legacy
``num_local_experts`` spelling shadows a user's canonical ``n_routed_experts``
override.
"""

import json

from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config
from tokenspeed.runtime.configs.kimi_k3_config import (
    KimiK3Config,
    KimiK3VisionConfig,
    KimiLinearConfig,
)
from tokenspeed.runtime.configs.kimi_k25_config import (
    KimiK25Config,
    KimiK25VisionConfig,
)
from tokenspeed.runtime.configs.utils import get_config


def _write_kimi_k25_config(path, text_config: dict) -> None:
    raw = {
        "model_type": "kimi_k25",
        "architectures": ["KimiK25ForConditionalGeneration"],
        "text_config": text_config,
    }
    (path / "config.json").write_text(json.dumps(raw), encoding="utf-8")


def test_kimi_k25_declares_text_config_sub_class() -> None:
    assert KimiK25Config.sub_configs["text_config"] is DeepseekV3Config
    assert KimiK25Config.sub_configs["vision_config"] is KimiK25VisionConfig


def test_kimi_k3_declares_text_config_sub_class() -> None:
    assert KimiK3Config.sub_configs["text_config"] is KimiLinearConfig
    assert KimiK3Config.sub_configs["vision_config"] is KimiK3VisionConfig


def test_kimi_k25_canonical_override_wins_over_legacy_alias(tmp_path) -> None:
    _write_kimi_k25_config(tmp_path, {"num_local_experts": 64})

    config = get_config(
        str(tmp_path),
        model_override_args={"n_routed_experts": 8},
    )

    assert config.text_config.n_routed_experts == 8
