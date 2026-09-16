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

"""CPU tests for the declared model-setup capture contract."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator


@pytest.fixture
def factory():
    # Only orchestration tests need the complete runtime dependency set.
    from tokenspeed.runtime.execution import factory

    return factory


class _BlockDraft(TargetCaptureConfigurator):
    def __init__(self, config):
        self.config = config
        self.calls = []

    def configure_target(self, target_model, target_config):
        self.calls.append((target_model, target_config))


@pytest.fixture
def dflash_model():
    import torch

    from tokenspeed.runtime.models.dflash import DFlashDraftModel

    model = DFlashDraftModel.__new__(DFlashDraftModel)
    torch.nn.Module.__init__(model)
    return model


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
@pytest.mark.parametrize("stage", [0, 1, 3])
def test_setup_configures_every_stage_without_creating_a_drafter(
    factory, monkeypatch, algorithm, stage
):
    implementation = mock.Mock(
        shares_target_embed_head=False,
        side_effect=AssertionError("drafter constructed during model setup"),
    )
    monkeypatch.setattr(factory, "get_drafter_impl", lambda algo, model: implementation)
    target = SimpleNamespace(set_dflash_layers_to_capture=mock.Mock())
    draft = _BlockDraft(
        SimpleNamespace(dflash_config={"target_layer_ids": [2, 23, 47]}, pp_rank=stage)
    )
    target_config = object()
    factory.configure_draft_target(
        SimpleNamespace(speculative_algorithm=algorithm),
        SimpleNamespace(
            model=target, model_config=SimpleNamespace(hf_text_config=target_config)
        ),
        SimpleNamespace(model=draft),
    )
    assert draft.calls == [(target, target_config)]
    implementation.assert_not_called()


def test_method_name_alone_does_not_opt_a_model_into_capture_setup(
    factory, monkeypatch
):
    monkeypatch.setattr(
        factory,
        "get_drafter_impl",
        lambda algo, model: SimpleNamespace(shares_target_embed_head=False),
    )
    draft = SimpleNamespace(configure_target=mock.Mock())
    with pytest.raises(TypeError, match="must implement TargetCaptureConfigurator"):
        factory.configure_draft_target(
            SimpleNamespace(speculative_algorithm="DSPARK"),
            SimpleNamespace(model=object()),
            SimpleNamespace(model=draft),
        )
    draft.configure_target.assert_not_called()


def test_capture_interface_requires_an_implementation():
    class MissingImplementation(TargetCaptureConfigurator):
        pass

    with pytest.raises(TypeError, match="abstract"):
        MissingImplementation()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("stream", ["prefix", "attn_res"])
def test_dflash_model_capture_uses_checkpoint_fields(dflash_model, nested, stream):
    values = {"target_layer_ids": [2, 7], "aux_hidden_stream": stream}
    config = (
        SimpleNamespace(dflash_config=values) if nested else SimpleNamespace(**values)
    )
    target = SimpleNamespace(
        set_dflash_layers_to_capture=mock.Mock(),
        set_dflash_aux_hidden_stream=mock.Mock(),
    )
    dflash_model.config = config
    dflash_model.configure_target(target, None)
    target.set_dflash_layers_to_capture.assert_called_once_with([2, 7])
    target.set_dflash_aux_hidden_stream.assert_called_once_with(stream)


def test_nested_capture_config_overrides_top_level_values(dflash_model):
    target = SimpleNamespace(
        set_dflash_layers_to_capture=mock.Mock(),
        set_dflash_aux_hidden_stream=mock.Mock(),
    )
    config = SimpleNamespace(
        target_layer_ids=[1],
        aux_hidden_stream="prefix",
        dflash_config={"target_layer_ids": [2, 7], "aux_hidden_stream": "ATTN_RES"},
    )
    dflash_model.config = config
    dflash_model.configure_target(target, None)
    target.set_dflash_layers_to_capture.assert_called_once_with([2, 7])
    target.set_dflash_aux_hidden_stream.assert_called_once_with("attn_res")


def test_missing_taps_fail_during_model_setup(dflash_model):
    dflash_model.config = SimpleNamespace()
    with pytest.raises(ValueError, match="target_layer_ids"):
        dflash_model.configure_target(object(), None)


def test_target_without_capture_support_fails_during_model_setup(dflash_model):
    dflash_model.config = SimpleNamespace(target_layer_ids=[1])
    with pytest.raises(ValueError, match="set_dflash_layers_to_capture"):
        dflash_model.configure_target(object(), None)


def test_unsupported_stream_fails_before_installing_capture(dflash_model):
    setter = mock.Mock()
    target = SimpleNamespace(set_dflash_layers_to_capture=setter)
    dflash_model.config = SimpleNamespace(
        target_layer_ids=[1], aux_hidden_stream="attn_res"
    )
    with pytest.raises(ValueError, match="only supply 'prefix'"):
        dflash_model.configure_target(target, None)
    setter.assert_not_called()


@pytest.mark.parametrize("override,expected", [(None, [2, 5]), ([3, 7], [3, 7])])
@pytest.mark.parametrize("wrapped", [False, True])
def test_eagle_setup_preserves_capture_override_and_weight_sharing(
    factory, monkeypatch, override, expected, wrapped
):
    monkeypatch.setattr(
        factory,
        "get_drafter_impl",
        lambda algo, model: SimpleNamespace(shares_target_embed_head=True),
    )
    capture = mock.Mock()
    target = SimpleNamespace(
        get_embed_and_head=lambda: ("embedding", "head"),
        set_eagle3_layers_to_capture=capture,
    )
    shared = mock.Mock()
    checkpoint = {"eagle_config": {"eagle_aux_hidden_state_layer_ids": [2, 5]}}
    if wrapped:
        checkpoint = {"text_config": checkpoint}
    factory.configure_draft_target(
        SimpleNamespace(
            speculative_algorithm="EAGLE3", eagle3_layers_to_capture=override
        ),
        SimpleNamespace(model=target),
        SimpleNamespace(
            model=SimpleNamespace(set_embed_and_head=shared),
            model_config=SimpleNamespace(hf_config=checkpoint),
        ),
    )
    shared.assert_called_once_with("embedding", "head")
    capture.assert_called_once_with(expected)


def test_mtp_setup_preserves_module_sharing_without_capture(factory, monkeypatch):
    monkeypatch.setattr(
        factory,
        "get_drafter_impl",
        lambda algo, model: SimpleNamespace(shares_target_embed_head=True),
    )
    calls = []

    class Draft:
        def set_embed_and_head_module(self, embedding, head):
            calls.append((embedding, head))

    head = object()
    target = SimpleNamespace(
        get_embed_and_head=lambda: ("embedding", "head-weight"), lm_head=head
    )
    factory.configure_draft_target(
        SimpleNamespace(speculative_algorithm="MTP"),
        SimpleNamespace(model=target),
        SimpleNamespace(model=Draft()),
    )
    assert calls == [("embedding", head)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
