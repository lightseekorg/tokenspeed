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

"""The opt-in shared-expert auxiliary stream of DeepSeek V4/V4.1 MoE under Petit.

Import-level checks (no kernel runs): the switch defaults off, and only with
it on does a Petit layer fork its shared expert onto the auxiliary stream.
"""

import logging

import pytest

from tokenspeed.runtime.models.deepseek_v41_moe import (
    log_shared_expert_stream,
    shared_expert_stream,
    shared_expert_stream_label,
)
from tokenspeed.runtime.utils.env import envs

_SWITCH = "TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM"


def test_switch_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv(_SWITCH, raising=False)
    assert envs.TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM.get() is False


@pytest.mark.parametrize("value", [None, "0", "false"])
def test_switch_off_keeps_todays_streams(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv(_SWITCH, raising=False)
    else:
        monkeypatch.setenv(_SWITCH, value)
    aux = object()
    # Petit: both branches on the main stream; every other backend: aux.
    assert shared_expert_stream(aux, use_gluon_petit=True) is None
    assert shared_expert_stream(aux, use_gluon_petit=False) is aux
    assert shared_expert_stream(None, use_gluon_petit=False) is None


@pytest.mark.parametrize("value", ["1", "true"])
def test_switch_on_forks_petit_shared_expert(monkeypatch, value) -> None:
    monkeypatch.setenv(_SWITCH, value)
    aux = object()
    assert shared_expert_stream(aux, use_gluon_petit=True) is aux
    # Other backends are unaffected by the switch.
    assert shared_expert_stream(aux, use_gluon_petit=False) is aux
    assert shared_expert_stream(None, use_gluon_petit=True) is None


@pytest.mark.parametrize(
    "value,use_gluon_petit,has_aux,label",
    [
        ("1", True, True, "aux (TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM)"),
        ("0", True, True, "main (Gluon Petit default)"),
        ("0", False, True, "aux (backend default)"),
        ("1", True, False, "main (no aux stream)"),
    ],
)
def test_shared_expert_stream_label(
    monkeypatch, value, use_gluon_petit, has_aux, label
) -> None:
    monkeypatch.setenv(_SWITCH, value)
    aux = object() if has_aux else None
    assert shared_expert_stream_label(aux, use_gluon_petit=use_gluon_petit) == label


@pytest.mark.parametrize(
    "value,stream", [("1", "aux"), ("0", "main")], ids=["aux", "main"]
)
def test_layer_zero_logs_the_shared_expert_stream(
    monkeypatch, caplog, value, stream
) -> None:
    monkeypatch.setenv(_SWITCH, value)
    caplog.set_level(logging.INFO, logger="tokenspeed.runtime.models.deepseek_v41_moe")
    for layer_index in range(3):
        log_shared_expert_stream(
            layer_index,
            1,
            object(),
            use_gluon_petit=True,
            prefix=f"model.layers.{layer_index}.mlp",
            moe_backend="gluon_petit",
        )
    # No shared expert, nothing to log.
    log_shared_expert_stream(
        0,
        0,
        object(),
        use_gluon_petit=True,
        prefix="model.layers.0.mlp",
        moe_backend="gluon_petit",
    )
    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        f"DeepSeek V4 MoE shared expert stream: {stream} ("
        + (
            "TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM"
            if stream == "aux"
            else "Gluon Petit default"
        )
        + ") prefix=model.layers.0.mlp moe_backend=gluon_petit"
    ]
