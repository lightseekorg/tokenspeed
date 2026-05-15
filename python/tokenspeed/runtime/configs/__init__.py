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

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.kimi_k2_config import KimiK2Config
from tokenspeed.runtime.configs.minimax_m2_config import MiniMaxM2Config
from tokenspeed.runtime.configs.qwen2_config import Qwen2Config
from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5Config, Qwen3_5MoeConfig
from tokenspeed.runtime.configs.qwen3_config import Qwen3Config

__all__ = [
    "DeepseekV4Config",
    "Qwen2Config",
    "Qwen3Config",
    "Qwen3_5Config",
    "Qwen3_5MoeConfig",
    "MiniMaxM2Config",
    "KimiK2Config",
]
