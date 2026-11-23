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

import pytest

from tokenspeed.runtime.models.deepseek_v41_moe import shared_expert_stream
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
