from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.paged_attention import (  # noqa: E402
    PagedAttention,
    validate_cache_group_ids,
)


class TestMlaCacheGroupId(unittest.TestCase):
    """Guards the rule DeepseekV3AttentionMLA now obeys: a PagedAttention that
    meets a multi-group pool must carry a group id.

    Untagged MLA modules went unnoticed while draft models had their own
    single-group pool; once the draft began sharing the target's unified pool
    -- which publishes every group the hybrid target owns -- the validation
    rejected them and K3 + EAGLE3 refused to start. The wiring itself (that
    ``DeepseekV3AttentionMLA.__init__`` applies the tag) is covered by serving,
    not here: constructing it needs distributed init and real weights.
    """

    def test_validation_rejects_an_untagged_layer(self) -> None:
        """Guard the other half: the check really fires when a tag is missing,
        so the test above cannot pass vacuously."""
        import torch.nn as nn

        from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
            FULL_ATTENTION,
        )

        class _Spec:
            def __init__(self, group_id: str) -> None:
                self.group_id = group_id

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = PagedAttention(
                    8, 576, 1.0, num_kv_heads=1, layer_id=0, v_head_dim=512
                )

        specs = (_Spec(FULL_ATTENTION), _Spec("linear_attention"))
        model = _Model()
        with self.assertRaises(ValueError):
            validate_cache_group_ids(model, specs)
        model.attn.group_id = FULL_ATTENTION
        validate_cache_group_ids(model, specs)  # must not raise


if __name__ == "__main__":
    unittest.main()
