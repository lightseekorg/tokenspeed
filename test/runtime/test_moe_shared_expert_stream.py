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

"""The model-neutral shared-expert stream label and its log line.

Import-level checks (no kernel runs): ``shared_expert_stream_label`` names the
stream a shared expert runs on, and GLM-5.3 Flash and Kimi K3 log that choice
once per model in the line format DeepSeek V4 uses.
"""

import logging
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.layers.moe.utils import shared_expert_stream_label
from tokenspeed.runtime.models import glm53_flash, kimi_k3

_SWITCH = "TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM"


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


# (switch, use_gluon_petit, has_aux, label): the switch on and off, with and
# without an aux stream, and the non-Petit default.
_STREAM_CASES = [
    ("1", True, True, "aux (TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM)"),
    ("0", True, True, "main (Gluon Petit default)"),
    ("1", True, False, "main (no aux stream)"),
    ("0", True, False, "main (no aux stream)"),
    ("0", False, True, "aux (backend default)"),
]
_STREAM_IDS = ["on-aux", "off-aux", "on-no-aux", "off-no-aux", "non-petit"]


@pytest.mark.parametrize(
    "value,use_gluon_petit,has_aux,label", _STREAM_CASES, ids=_STREAM_IDS
)
def test_glm53_flash_logs_the_shared_expert_stream_once(
    monkeypatch, caplog, value, use_gluon_petit, has_aux, label
) -> None:
    monkeypatch.setenv(_SWITCH, value)
    caplog.set_level(logging.INFO, logger=glm53_flash.__name__)
    config = SimpleNamespace(mlp_layer_types=["dense"] * 3 + ["sparse"] * 3)
    aux = object() if has_aux else None
    # The target's MoE layers 3..5 log on layer 3 only.
    for layer_index in range(3, 6):
        glm53_flash.log_shared_expert_stream(
            config,
            layer_index,
            aux,
            use_gluon_petit=use_gluon_petit,
            prefix=f"model.layers.{layer_index}.mlp",
            moe_backend="gluon_petit",
        )
    # The MTP draft's one MoE layer is layer_id 0.
    glm53_flash.log_shared_expert_stream(
        config,
        0,
        aux,
        use_gluon_petit=use_gluon_petit,
        prefix="model.decoder.mlp",
        moe_backend="gluon_petit",
    )
    assert [r.getMessage() for r in caplog.records] == [
        f"GLM-5.3 Flash MoE shared expert stream: {label} "
        f"prefix={prefix} moe_backend=gluon_petit"
        for prefix in ("model.layers.3.mlp", "model.decoder.mlp")
    ]


@pytest.mark.parametrize(
    "value,use_gluon_petit,has_aux,label", _STREAM_CASES, ids=_STREAM_IDS
)
def test_kimi_k3_logs_the_shared_expert_stream_once(
    monkeypatch, caplog, value, use_gluon_petit, has_aux, label
) -> None:
    monkeypatch.setenv(_SWITCH, value)
    caplog.set_level(logging.INFO, logger=kimi_k3.__name__)
    config = SimpleNamespace(first_k_dense_replace=1)
    aux = object() if has_aux else None
    for layer_index in range(1, 4):
        kimi_k3.log_shared_expert_stream(
            config,
            layer_index,
            aux,
            use_gluon_petit=use_gluon_petit,
            prefix=f"model.layers.{layer_index}.mlp",
            moe_backend="gluon_petit",
        )
    assert [r.getMessage() for r in caplog.records] == [
        f"Kimi K3 MoE shared expert stream: {label} "
        "prefix=model.layers.1.mlp moe_backend=gluon_petit"
    ]
