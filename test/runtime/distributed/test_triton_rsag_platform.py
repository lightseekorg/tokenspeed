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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion

import tokenspeed.runtime.distributed.comm_backend.triton_rsag as triton_rsag


class _Fallback:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def token_all_gather(self, tensor, group, scattered_num_tokens):
        self.calls.append(("token_all_gather", (tensor, group, scattered_num_tokens)))
        return tensor + 1

    def token_reduce_scatter(self, tensor, group, scattered_num_tokens):
        self.calls.append(
            ("token_reduce_scatter", (tensor, group, scattered_num_tokens))
        )
        return tensor + 2

    def all_gather(self, tensor, group, dim=0):
        self.calls.append(("all_gather", (tensor, group, dim)))
        return tensor + 3


def _platform(major: int, minor: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        is_nvidia=True,
        arch_version=ArchVersion(major, minor),
    )


def test_sm120_falls_back_for_all_triton_rsag_entrypoints(monkeypatch) -> None:
    fallback = _Fallback()
    backend = triton_rsag.TritonRSAGBackend(fallback=fallback)
    monkeypatch.setattr(triton_rsag, "current_platform", lambda: _platform(12))
    monkeypatch.setattr(
        backend,
        "_get_or_create",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SM120 must not create Triton RSAG state")
        ),
    )
    tensor = torch.zeros(1, 8, dtype=torch.bfloat16)
    group = (0, 1, 2, 3)
    scattered = [1, 1, 1, 1]

    gathered = backend.token_all_gather(tensor, group, scattered)
    scattered_out = backend.token_reduce_scatter(tensor, group, scattered)
    inner = backend.all_gather(tensor, group, dim=-1)

    torch.testing.assert_close(gathered, tensor + 1)
    torch.testing.assert_close(scattered_out, tensor + 2)
    torch.testing.assert_close(inner, tensor + 3)
    assert [name for name, _args in fallback.calls] == [
        "token_all_gather",
        "token_reduce_scatter",
        "all_gather",
    ]


@pytest.mark.parametrize("major", [9, 10], ids=["h100", "b200"])
def test_existing_nvidia_architectures_preserve_triton_rsag_token_path(
    major: int,
    monkeypatch,
) -> None:
    fallback = _Fallback()
    backend = triton_rsag.TritonRSAGBackend(fallback=fallback)
    monkeypatch.setattr(triton_rsag, "current_platform", lambda: _platform(major))
    state = object()
    monkeypatch.setattr(backend, "_get_or_create", lambda *_args: state)
    expected = torch.ones(4, 8, dtype=torch.bfloat16)
    call: dict[str, object] = {}

    def fake_all_gather(state_arg, tensor, token_list_in_group):
        call.update(
            state=state_arg,
            tensor=tensor,
            token_list_in_group=token_list_in_group,
        )
        return expected

    monkeypatch.setattr(triton_rsag, "all_gather", fake_all_gather)
    tensor = torch.zeros(1, 8, dtype=torch.bfloat16)

    actual = backend.token_all_gather(tensor, (0, 1, 2, 3), [1, 1, 1, 1])

    assert actual is expected
    assert call == {
        "state": state,
        "tensor": tensor,
        "token_list_in_group": [1, 1, 1, 1],
    }
    assert fallback.calls == []
