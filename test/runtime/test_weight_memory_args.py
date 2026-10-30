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

"""Memory policy flags fail before a server allocates model storage."""

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs, prepare_server_args


def discard_args(**overrides):
    args = dict(
        model="test-model",
        enable_memory_saver=True,
        weight_memory_policy="discard",
        model_update_config="{}",
        model_update_sdk_module="test_sdk",
        model_update_engine_type="fluent_llm",
        model_update_draft_weights="retain",
    )
    args.update(overrides)
    return args


def test_memory_saving_requires_an_explicit_weight_policy():
    with pytest.raises(ValueError, match="requires --weight-memory-policy"):
        ServerArgs(model="test-model", enable_memory_saver=True)


def test_cli_selects_cpu_backup_without_requiring_an_updater():
    args = prepare_server_args(
        ["test-model", "--enable-memory-saver", "--weight-memory-policy", "cpu_backup"]
    )
    assert args.enable_memory_saver
    assert args.weight_memory_policy == "cpu_backup"
    assert args.model_update_config is None


@pytest.mark.parametrize("draft_policy", ["retain", "refresh"])
def test_discard_uses_explicit_draft_update_semantics(draft_policy):
    args = ServerArgs(**discard_args(model_update_draft_weights=draft_policy))
    assert args.weight_memory_policy == "discard"
    assert args.model_update_draft_weights == draft_policy


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"enable_memory_saver": False}, "requires memory saver"),
        ({"model_update_config": None}, "requires memory saver"),
        ({"model_update_draft_weights": None}, "explicit draft"),
        ({"quantization": "fp8"}, "unquantized"),
        ({"enable_eplb": True}, "without EPLB"),
    ],
)
def test_discard_rejects_unsupported_contracts_at_argument_resolution(
    overrides, reason
):
    with pytest.raises(ValueError, match=reason):
        ServerArgs(**discard_args(**overrides))
