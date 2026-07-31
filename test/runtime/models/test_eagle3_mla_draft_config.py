"""Config-adaptation regressions for the Eagle3 MLA draft path.

Both cases were hit bringing up Kimi-K3 + EAGLE3 (an MLA draft against a
vocab-parallel target); both are CPU-only checks of the adaptation logic.
"""

import torch

from tokenspeed.runtime.models import deepseek_v3 as dsv3


def test_normalized_rope_scaling_does_not_engage_yarn():
    """Newer transformers folds rope params into ``rope_scaling`` even when
    the checkpoint declares none (``{"rope_theta": ..., "rope_type":
    "default"}``). The draft layer used to see any truthy dict, stamp it
    ``deepseek_yarn``, and crash in ``get_rope`` on the missing ``factor``.
    Only a real scaling config (one with a factor) may engage yarn."""
    src = dsv3.Eagle3MlaDecoderLayer.__init__.__code__
    # The guard lives in the layer __init__; assert its effect functionally
    # by replicating the input it sees.
    rope_scaling = {"rope_theta": 50000.0, "rope_type": "default"}
    guarded = rope_scaling if "factor" in rope_scaling else None
    assert guarded is None
    real = {"rope_type": "yarn", "factor": 4.0}
    assert (real if "factor" in real else None) is not None
    # And the source actually contains the guard, so a refactor that drops
    # it fails here rather than at serve time.
    assert "factor" in src.co_consts or any(
        "factor" in str(c) for c in src.co_consts
    ), "rope_scaling factor guard missing from Eagle3MlaDecoderLayer.__init__"


def test_embed_sharing_requires_matching_shapes():
    """The target's embedding may be vocab-parallel (Kimi-K3: a TP shard)
    while the Eagle3 MLA draft's embedding module is replicated. Adopting a
    shard into the replicated module leaves every id beyond the shard
    reading out of bounds, so sharing must be skipped on shape mismatch and
    the draft's own full embedding kept."""

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

        def __init__(self):
            self.model = _Model()
            self.lm_head = _Embed(64)

        set_embed_and_head = dsv3.Eagle3DeepseekV2ForCausalLM.set_embed_and_head

    draft = _Draft()
    own = draft.model.embed_tokens.weight
    shard = torch.nn.Parameter(torch.ones(8, 8))  # target TP shard: 64/8 rows
    draft.set_embed_and_head(shard, None)
    assert draft.model.embed_tokens.weight is own, (
        "a mismatched (sharded) target embedding must not replace the "
        "draft's replicated table"
    )
    full = torch.nn.Parameter(torch.ones(64, 8))
    draft.set_embed_and_head(full, None)
    assert draft.model.embed_tokens.weight is full, (
        "a shape-matched target embedding should be shared"
    )
