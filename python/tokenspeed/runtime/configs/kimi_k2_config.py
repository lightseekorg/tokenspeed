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

"""Kimi K2 model configuration.

Kimi K2 checkpoints (and their Eagle3 drafts) ship a custom ``model_type`` of
``kimi_k2`` without an ``auto_map`` entry, so ``transformers`` cannot resolve
the config on its own. Structurally the model matches DeepSeek-V3 (MLA + MoE
with ``sigmoid`` / ``noaux_tc`` routing), so we expose a thin subclass that
inherits ``DeepseekV3Config`` and only overrides the registered ``model_type``.
"""

from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config


class KimiK2Config(DeepseekV3Config):
    model_type = "kimi_k2"


__all__ = ["KimiK2Config"]
