"""Tests for draft-to-target wiring.

Model-to-model wiring (shared embed/head, EAGLE3 capture ids) happens in
``factory._wire_draft_to_target_model`` right after both models load, so
shared weights are released before the KV-cache budget is profiled.
Drafter-instance wiring (capture hooks on the target) happens in
``BaseDrafter.wire_target`` implementations.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed.runtime.execution.factory as factory  # noqa: E402
from tokenspeed.runtime.execution.drafter import get_drafter_impl  # noqa: E402
from tokenspeed.runtime.execution.drafter.base import BaseDrafter  # noqa: E402
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4DSpark,
)
from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.execution.drafter.dspark import DSpark  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.drafter.mtp import Mtp  # noqa: E402


def _server_args(algo: str, capture_ids=None) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_algorithm=algo,
        eagle3_layers_to_capture=capture_ids,
    )


def test_get_drafter_impl_routing():
    from tokenspeed.runtime.models.deepseek_v4_dspark import (
        DeepseekV4ForCausalLMDSpark,
    )
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    assert get_drafter_impl("EAGLE3", mock.MagicMock()) is Eagle
    assert get_drafter_impl("MTP", mock.MagicMock()) is Eagle
    assert (
        get_drafter_impl(
            "MTP", mock.MagicMock(spec=InklingForConditionalGenerationNextN)
        )
        is Mtp
    )
    assert get_drafter_impl("DFLASH", mock.MagicMock()) is DFlash
    assert get_drafter_impl("DSPARK", mock.MagicMock()) is DSpark
    assert (
        get_drafter_impl("DSPARK", mock.MagicMock(spec=DeepseekV4ForCausalLMDSpark))
        is DeepseekV4DSpark
    )


def test_shares_target_embed_head_flags():
    # Eagle-like drafters and the checkpoint-local V4 DSpark reuse the
    # target's embed/LM head; DFlash and generic DSpark ship their own.
    assert Eagle.shares_target_embed_head
    assert Mtp.shares_target_embed_head
    assert DeepseekV4DSpark.shares_target_embed_head
    assert not DFlash.shares_target_embed_head
    assert not DSpark.shares_target_embed_head
    assert not BaseDrafter.shares_target_embed_head


def test_pd_layerwise_finalization_capability_matches_supported_drafters():
    assert Eagle.supports_pd_layerwise_finalization
    assert DFlash.supports_pd_layerwise_finalization
    assert DSpark.supports_pd_layerwise_finalization
    assert not Mtp.supports_pd_layerwise_finalization
    assert not DeepseekV4DSpark.supports_pd_layerwise_finalization
    assert not BaseDrafter.supports_pd_layerwise_finalization


def test_wire_eagle3_shares_embed_head_and_installs_capture_ids():
    target, draft = mock.MagicMock(), mock.MagicMock()
    target.model.get_embed_and_head.return_value = ("EMBED", "HEAD")
    draft.model_config.hf_config = {
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
    }

    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory._wire_draft_to_target_model(_server_args("EAGLE3"), target, draft)

    draft.model.set_embed_and_head.assert_called_once_with("EMBED", "HEAD")
    target.model.set_eagle3_layers_to_capture.assert_called_once_with([1, 2, 3])


class _ModuleSharingDraft:
    def __init__(self):
        self.shared = None
        self.legacy = None

    def set_embed_and_head_module(self, embed, lm_head):
        self.shared = (embed, lm_head)

    def set_embed_and_head(self, embed, head):
        self.legacy = (embed, head)


def test_wire_mtp_shares_complete_lm_head_for_opted_in_draft():
    lm_head = object()
    target = SimpleNamespace(
        model=SimpleNamespace(
            lm_head=lm_head,
            get_embed_and_head=lambda: ("EMBED", "HEAD_WEIGHT"),
        )
    )
    draft_model = _ModuleSharingDraft()
    draft = SimpleNamespace(model=draft_model)

    with mock.patch.object(factory, "get_drafter_impl", return_value=Mtp):
        factory._wire_draft_to_target_model(_server_args("MTP"), target, draft)

    assert draft_model.shared == ("EMBED", lm_head)
    assert draft_model.legacy is None


def test_wire_mtp_module_sharing_requires_target_lm_head():
    target = SimpleNamespace(
        model=SimpleNamespace(get_embed_and_head=lambda: ("EMBED", "HEAD_WEIGHT"))
    )
    draft = SimpleNamespace(model=_ModuleSharingDraft())

    with (
        mock.patch.object(factory, "get_drafter_impl", return_value=Mtp),
        pytest.raises(ValueError, match="complete lm_head module"),
    ):
        factory._wire_draft_to_target_model(_server_args("MTP"), target, draft)


def test_wire_eagle3_explicit_capture_ids_override_checkpoint():
    target, draft = mock.MagicMock(), mock.MagicMock()
    target.model.get_embed_and_head.return_value = ("E", "H")
    draft.model_config.hf_config = {
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
    }

    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory._wire_draft_to_target_model(
            _server_args("EAGLE3", capture_ids=[7, 8]), target, draft
        )

    target.model.set_eagle3_layers_to_capture.assert_called_once_with([7, 8])


def test_wire_dflash_keeps_own_embed_head():
    target, draft = mock.MagicMock(), mock.MagicMock()

    with mock.patch.object(factory, "get_drafter_impl", return_value=DFlash):
        factory._wire_draft_to_target_model(_server_args("DFLASH"), target, draft)

    target.model.get_embed_and_head.assert_not_called()
    draft.model.set_embed_and_head.assert_not_called()


def test_base_wire_target_is_a_noop():
    drafter = mock.MagicMock(spec=BaseDrafter)
    target_model = mock.MagicMock()
    BaseDrafter.wire_target(drafter, target_model)
    target_model.assert_not_called()


def test_dflash_wire_target_requires_capture_support():
    drafter = mock.MagicMock(spec=DFlash)
    drafter._incremental_proj_enabled = False
    target_model = mock.MagicMock(
        spec=["get_input_embeddings", "lm_head", "logits_processor"]
    )
    with pytest.raises(ValueError, match="set_dflash_layers_to_capture"):
        DFlash.wire_target(drafter, target_model)


def test_dflash_wire_target_installs_capture_hooks():
    drafter = mock.MagicMock(spec=DFlash)
    drafter._incremental_proj_enabled = False
    drafter.target_layer_ids = [4, 5]
    target_model = mock.MagicMock(
        spec=[
            "get_input_embeddings",
            "lm_head",
            "logits_processor",
            "set_dflash_layers_to_capture",
        ]
    )

    DFlash.wire_target(drafter, target_model)

    target_model.set_dflash_layers_to_capture.assert_called_once_with(
        [4, 5], incremental_callback=None, slot_bufs=None
    )
    assert drafter.embed_tokens is target_model.get_input_embeddings.return_value
    assert drafter.lm_head is target_model.lm_head


def test_dspark_wire_target_installs_capture_layers():
    drafter = mock.MagicMock(spec=DeepseekV4DSpark)
    drafter.target_layer_ids = [10, 20]
    target_model = mock.MagicMock(
        spec=["lm_head", "logits_processor", "set_dspark_layers_to_capture"]
    )

    DeepseekV4DSpark.wire_target(drafter, target_model)

    target_model.set_dspark_layers_to_capture.assert_called_once_with([10, 20])
    assert drafter.lm_head is target_model.lm_head


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
