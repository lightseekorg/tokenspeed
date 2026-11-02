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

"""Standard MHA backend preserves target and block-draft geometry."""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

# Imported for register_backend() side effects; no symbols are referenced.
import tokenspeed.runtime.layers.attention.backends.paged.mha  # noqa: F401
import tokenspeed.runtime.layers.attention.backends.paged.mla  # noqa: F401
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import _get_backend_cls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def test_gluon_backend_is_rejected_for_mha():
    with pytest.raises(ValueError, match="does not support"):
        _get_backend_cls("gluon", AttentionArch.MHA)


@pytest.mark.parametrize("draft,heads", [(False, 48), (True, 40)])
@pytest.mark.parametrize(
    "name,solution", [("mha", None), ("triton", "triton"), (None, None)]
)
def test_mha_preserves_target_and_block_draft_geometry(draft, heads, name, solution):
    spec = MHAConfig(
        backend_name=name,
        num_attention_heads=heads,
        num_kv_heads=8,
        head_dim=128,
        attn_tp_size=8,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=50176,
        max_bs=16,
        speculative_num_draft_tokens=16,
        is_draft=draft,
        draft_block_decode=draft,
        components=(spec,),
    )
    cls = _get_backend_cls(name, AttentionArch.MHA)
    page_size = cls.resolve_kernel_page_size(config, 64)
    assert page_size == 64
    backend = cls(config, spec, kernel_page_size=page_size)
    assert backend.kernel_solution == solution
    assert backend.tp_q_head_num == (5 if draft else 6)
    assert backend.tp_kv_head_num == 1
    assert backend.block_decode_expansion == (16 if draft else 1)
    backend.init_cuda_graph_state(16)
    assert backend.page_table_buf.shape[0] == (256 if draft else 16)
    plan = backend.plan(window_left=4095 if draft else 511)
    if current_platform().is_cdna4:
        assert plan["extend_mode"] == ("postwrite" if solution is None else "prewrite")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
