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

"""DeepSeek-V3.2 MTP draft (``DeepseekV32ForCausalLMNextN``).

The V3.2 NextN/MTP draft shares the target's DSA backbone, so it reuses the
GLM-DSA NextN implementation; only the class name differs so the registry
resolves the ``DeepseekV32ForCausalLMNextN`` draft architecture.
"""

from __future__ import annotations

from tokenspeed.runtime.models.glm_moe_dsa_nextn import GlmMoeDsaForCausalLMNextN


class DeepseekV32ForCausalLMNextN(GlmMoeDsaForCausalLMNextN):
    """DeepSeek-V3.2 MTP draft: GLM-DSA NextN under the V3.2 architecture name."""


EntryClass = [DeepseekV32ForCausalLMNextN]
