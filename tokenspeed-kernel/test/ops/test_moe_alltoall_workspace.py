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

"""Workspace ownership for K3's optional all-to-all transport."""

from types import SimpleNamespace
from unittest import mock

import torch
from tokenspeed_kernel.ops.communication import flashinfer


def test_workspaces_are_shared_by_layers_but_separate_for_models(monkeypatch) -> None:
    flashinfer.get_flashinfer_moe_alltoall.cache_clear()
    group = mock.Mock()
    group.size.return_value = 2
    factory = mock.Mock(side_effect=[object(), object()])
    monkeypatch.setattr(flashinfer, "FlashInferMoeAlltoAll", factory)
    monkeypatch.setattr(
        flashinfer, "current_platform", lambda: SimpleNamespace(is_nvidia=True)
    )
    monkeypatch.setattr(flashinfer, "group_has_fabric", lambda ranks: True)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1]
    )
    kwargs = dict(
        group=group,
        max_tokens=8,
        hidden_size=32,
        top_k=2,
        num_experts=8,
        dtype=torch.bfloat16,
        weights_dtype=torch.bfloat16,
    )
    target = flashinfer.get_flashinfer_moe_alltoall(model_scope="target", **kwargs)
    assert target is flashinfer.get_flashinfer_moe_alltoall(
        model_scope="target", **kwargs
    )
    assert target is not flashinfer.get_flashinfer_moe_alltoall(
        model_scope="draft", **kwargs
    )
    assert factory.call_count == 2
    flashinfer.get_flashinfer_moe_alltoall.cache_clear()


def test_missing_fabric_keeps_reference_transport(monkeypatch) -> None:
    flashinfer.get_flashinfer_moe_alltoall.cache_clear()
    factory = mock.Mock(
        side_effect=AssertionError("workspace allocated without fabric")
    )
    monkeypatch.setattr(flashinfer, "FlashInferMoeAlltoAll", factory)
    monkeypatch.setattr(
        flashinfer, "current_platform", lambda: SimpleNamespace(is_nvidia=True)
    )
    monkeypatch.setattr(flashinfer, "group_has_fabric", lambda ranks: False)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1]
    )
    assert (
        flashinfer.get_flashinfer_moe_alltoall(
            group=object(),
            model_scope="target",
            max_tokens=8,
            hidden_size=32,
            top_k=2,
            num_experts=8,
            dtype=torch.bfloat16,
            weights_dtype=torch.bfloat16,
        )
        is None
    )
    factory.assert_not_called()
    flashinfer.get_flashinfer_moe_alltoall.cache_clear()
