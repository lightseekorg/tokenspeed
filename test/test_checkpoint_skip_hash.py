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

import builtins
import json
from test.test_l3_kvstore import _deployment_facts

import pytest

from tokenspeed.runtime.cache.l3.identity import (
    _local_checkpoint_fingerprint,
    l3_checkpoint_id,
    storage_deployment_id,
)


@pytest.fixture(autouse=True)
def clean_identity_environment(monkeypatch):
    for name in (
        "TOKENSPEED_SKIP_CHECKPOINT_HASH",
        "TOKENSPEED_CHECKPOINT_IDENTITY_MODE",
        "TOKENSPEED_CHECKPOINT_RELEASES",
    ):
        monkeypatch.delenv(name, raising=False)
    _local_checkpoint_fingerprint.cache_clear()
    yield
    _local_checkpoint_fingerprint.cache_clear()


def identify(model, **overrides):
    return l3_checkpoint_id(
        str(model),
        **{
            "hf_config": None,
            "revision": "v1",
            "load_format": "auto",
            "model_loader_extra_config": {},
            "ext_yaml": "",
            **overrides,
        },
    )


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"test weights v1")
    return tmp_path


def test_skip_never_opens_weights_and_needs_no_release_mapping(
    checkpoint, monkeypatch, caplog
):
    original_open = builtins.open

    def reject_weight_read(file, *args, **kwargs):
        if str(file).endswith(".safetensors"):
            raise AssertionError("weight content must not be read")
        return original_open(file, *args, **kwargs)

    # Simulate unavailable weight I/O after the model has already loaded.
    monkeypatch.setattr(builtins, "open", reject_weight_read)
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", "1")
    first = identify(checkpoint)
    assert first.startswith("unchecked-")
    assert identify(checkpoint) == first
    assert "weight scan skipped" in caplog.text
    # The same fixture proves that the default policy does access the weights.
    monkeypatch.delenv("TOKENSPEED_SKIP_CHECKPOINT_HASH")
    with pytest.raises(AssertionError, match="weight content must not be read"):
        identify(checkpoint)


@pytest.mark.parametrize("skip", [None, "0"])
def test_disabled_skip_preserves_content_identity(checkpoint, monkeypatch, skip):
    baseline = identify(checkpoint)
    if skip is not None:
        monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", skip)
    assert identify(checkpoint) == baseline
    (checkpoint / "model.safetensors").write_bytes(b"test weights v2")
    _local_checkpoint_fingerprint.cache_clear()
    assert identify(checkpoint) != baseline


def test_skip_does_not_detect_in_place_weight_replacement(checkpoint, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", "1")
    first = identify(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"replacement weights")
    _local_checkpoint_fingerprint.cache_clear()
    assert identify(checkpoint) == first
    assert identify(checkpoint, revision="v2") != first


def test_skip_distinguishes_models_loaders_and_content_mode(checkpoint, monkeypatch):
    content = identify(checkpoint)
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", "1")
    first = identify(checkpoint)
    assert first != content
    assert identify("org/other-model") != first
    assert identify(checkpoint, load_format="pt") != first
    assert identify(checkpoint, model_loader_extra_config={"pattern": "other"}) != first


def test_skip_overrides_legacy_release_configuration(checkpoint, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_IDENTITY_MODE", "release")
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", "1")
    first = identify(checkpoint)  # No releases map is required.
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_RELEASES", "invalid leftover JSON")
    assert identify(checkpoint) == first


def test_release_policy_is_unchanged_when_skip_is_disabled(checkpoint, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", "0")
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_IDENTITY_MODE", "release")
    monkeypatch.setenv(
        "TOKENSPEED_CHECKPOINT_RELEASES", json.dumps({str(checkpoint): "release-v1"})
    )
    first = identify(checkpoint)
    assert first.startswith("release-")
    monkeypatch.setenv(
        "TOKENSPEED_CHECKPOINT_RELEASES", json.dumps({str(checkpoint): "release-v2"})
    )
    assert identify(checkpoint) != first
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_RELEASES", "{}")
    with pytest.raises(KeyError):
        identify(checkpoint)


@pytest.mark.parametrize("value", ["", "true", "2"])
def test_invalid_skip_value_is_rejected(checkpoint, monkeypatch, value):
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", value)
    with pytest.raises(ValueError, match="TOKENSPEED_SKIP_CHECKPOINT_HASH"):
        identify(checkpoint)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "org/other"},
        {"cache_signature": "different-layout"},
        {"cache_quantization": "different-quantization"},
        {"draft_revision": "different-draft"},
    ],
)
def test_skip_preserves_deployment_isolation(checkpoint, monkeypatch, overrides):
    monkeypatch.setenv("TOKENSPEED_SKIP_CHECKPOINT_HASH", "1")
    facts = {**_deployment_facts(), "revision": identify(checkpoint)}
    assert storage_deployment_id(**facts) != storage_deployment_id(
        **{**facts, **overrides}
    )
