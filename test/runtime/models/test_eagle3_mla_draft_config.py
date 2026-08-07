"""Config-adaptation regressions for the Eagle3 MLA draft path.

Both cases were hit bringing up Kimi-K3 + EAGLE3 (an MLA draft against a
vocab-parallel target).
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models import deepseek_v3 as dsv3


class DraftRopeScalingTest(unittest.TestCase):
    """Only yarn-style rope_scaling may reach the draft layer's yarn path."""

    def test_normalized_plain_rope_is_dropped(self):
        normalized = {"rope_theta": 50000.0, "rope_type": "default"}
        self.assertIsNone(dsv3._draft_rope_scaling(normalized))

    def test_yarn_configs_are_kept(self):
        for real in (
            {"rope_type": "yarn", "factor": 4.0},
            {"type": "deepseek_yarn", "factor": 40.0},
        ):
            self.assertIs(dsv3._draft_rope_scaling(real), real)

    def test_unsupported_scaling_is_dropped_not_misrouted(self):
        # A factor-bearing non-yarn schema must not reach the yarn path.
        self.assertIsNone(
            dsv3._draft_rope_scaling({"rope_type": "linear", "factor": 4.0})
        )
        self.assertIsNone(dsv3._draft_rope_scaling({"type": "dynamic", "alpha": 2.0}))

    def test_none_passthrough(self):
        self.assertIsNone(dsv3._draft_rope_scaling(None))


class _Embed(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(rows, 8))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = _Embed(64)


class _Draft:
    class _Cfg:
        pass

    config = _Cfg()
    load_lm_head_from_target = False

    def __init__(self, embed_loaded: bool):
        self.model = _Model()
        self.lm_head = _Embed(64)
        self._embed_loaded_from_checkpoint = embed_loaded

    set_embed_and_head = dsv3.Eagle3DeepseekV2ForCausalLM.set_embed_and_head


class EmbedSharingTest(unittest.TestCase):
    """The target's embedding may be a Kimi-K3-style TP shard while the
    draft's module is replicated; adopting the shard would read out of
    bounds, so sharing is skipped on mismatch."""

    def test_mismatched_shard_keeps_draft_embedding(self):
        draft = _Draft(embed_loaded=True)
        own = draft.model.embed_tokens.weight
        shard = torch.nn.Parameter(torch.ones(8, 8))  # 64/8 rows
        draft.set_embed_and_head(shard, None)
        self.assertIs(draft.model.embed_tokens.weight, own)

    def test_mismatch_without_draft_weights_fails_loudly(self):
        draft = _Draft(embed_loaded=False)
        shard = torch.nn.Parameter(torch.ones(8, 8))
        with self.assertRaises(ValueError):
            draft.set_embed_and_head(shard, None)

    def test_matching_shapes_share(self):
        draft = _Draft(embed_loaded=True)
        full = torch.nn.Parameter(torch.ones(64, 8))
        draft.set_embed_and_head(full, None)
        self.assertIs(draft.model.embed_tokens.weight, full)


if __name__ == "__main__":
    unittest.main()
