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

"""Fixed-budget Flash MTP arena regression; no weights or GPU allocation.

Run directly with Python, as runtime CI does. Compare DCP disabled (degree one)
with DCP4 using the same model, concurrency, indexer format and per-rank budget.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
)

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)


def _recipe(degree, fp4):
    # Flash's 43 target layers and its real SWA-only MTP continuation layer.
    hf = SimpleNamespace(
        compress_ratios=(0, 0) + (4, 128) * 20 + (4, 0),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )
    return DeepseekV4Recipe(
        server_args=SimpleNamespace(
            max_total_tokens=None,
            chunked_prefill_size=8192,
            disaggregation_mode="null",
            enable_prefix_caching=True,
            attention_use_fp4_indexer_cache=fp4,
        ),
        model_config=SimpleNamespace(hf_config=hf, num_attention_layers=43),
        attn_config=SimpleNamespace(
            prefix_granularity=256,
            max_bs=16,
            context_len=4096,
            pd_disaggregation_enabled=False,
            dcp_size=degree,
        ),
        draft_model_config=SimpleNamespace(hf_config=hf, num_attention_layers=1),
        draft_attn_config=SimpleNamespace(dcp_size=degree),
        cache_budget_bytes=32 << 30,
        decode_input_tokens=4,
        overlap_schedule_depth=0,
    )


class Dsv4CacheArenaBenefitTest(unittest.TestCase):
    def test_dcp_increases_capacity_with_the_same_cache_budget(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                tp_recipe = _recipe(degree=1, fp4=fp4)
                dcp_recipe = _recipe(degree=4, fp4=fp4)
                self.assertEqual(
                    tp_recipe.cache_budget_bytes, dcp_recipe.cache_budget_bytes
                )
                tp = tp_recipe.setup().spec
                dcp = dcp_recipe.setup().spec
                for spec in (tp, dcp):
                    self.assertGreater(spec.token_capacity, 0)
                    self.assertLessEqual(
                        spec.memory_plan.arena_bytes, tp_recipe.cache_budget_bytes
                    )
                self.assertGreater(dcp.token_capacity, tp.token_capacity)
                print(
                    f"indexer={'FP4' if fp4 else 'FP8'} "
                    f"budget_bytes={tp_recipe.cache_budget_bytes} "
                    f"token_capacity(DCP1->DCP4)={tp.token_capacity}->{dcp.token_capacity} "
                    f"capacity_ratio={dcp.token_capacity / tp.token_capacity:.3f} "
                    f"arena_bytes(DCP1->DCP4)="
                    f"{tp.memory_plan.arena_bytes}->{dcp.memory_plan.arena_bytes}",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
