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

"""DeepSeek-V3.2 (``DeepseekV32ForCausalLM``).

V3.2 is architecturally DeepSeek-V3's MLA/MoE backbone plus a DSA sparse
indexer, which is the same shape as GLM-DSA (``glm5.py``): identical indexer
geometry (``index_topk`` / ``index_head_dim`` / ``index_n_heads``), identical
indexer weight names (``self_attn.indexer.{wq_b,wk,weights_proj,k_norm}``, loaded
into the fused ``wk_weights_proj`` by the shared loader), and the same V3 MoE
(``DeepseekV3MoE`` with noaux_tc / sigmoid). It therefore reuses the GLM-DSA
implementation wholesale; only the class name differs so the model registry
resolves the ``DeepseekV32ForCausalLM`` checkpoint architecture.
"""

from __future__ import annotations

from tokenspeed.runtime.models.glm5 import GlmMoeDsaForCausalLM


class DeepseekV32ForCausalLM(GlmMoeDsaForCausalLM):
    """DeepSeek-V3.2: GLM-DSA backbone under the V3.2 architecture name."""


EntryClass = [DeepseekV32ForCausalLM]
