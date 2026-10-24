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

"""Draft expert selection must not leak into the target or retain FlatQuant."""

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.execution import model_runner


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("draft", ["ascend_bf16", "ascend_w4a4", None])
def test_target_preserved_across_draft_loading(monkeypatch, draft, fail):
    args = SimpleNamespace(
        device="npu",
        moe_backend="ascend_w4a4",
        draft_moe_backend=draft,
        npu_flatquant_config="target.json",
    )
    snapshots = []
    monkeypatch.setattr(
        model_runner,
        "global_server_args_dict_update",
        lambda a: snapshots.append((a.moe_backend, a.npu_flatquant_config)),
    )
    monkeypatch.setattr(model_runner, "initialize_moe_config", lambda a: None)
    with model_runner.draft_moe_configuration(args, is_draft_worker=False):
        assert args.moe_backend == "ascend_w4a4"
        assert args.npu_flatquant_config == "target.json"
    try:
        with model_runner.draft_moe_configuration(args, is_draft_worker=True):
            assert args.moe_backend == (draft or "ascend_w4a4")
            assert args.npu_flatquant_config == (
                None if draft == "ascend_bf16" else "target.json"
            )
            if fail:
                raise RuntimeError("weight loader failed")
    except RuntimeError as error:
        assert fail and str(error) == "weight loader failed"
    assert args.moe_backend == "ascend_w4a4"
    assert args.npu_flatquant_config == "target.json"
    if draft == "ascend_bf16":
        assert snapshots[-1] == ("ascend_w4a4", "target.json")
    else:
        assert not snapshots
