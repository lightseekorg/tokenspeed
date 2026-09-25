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

"""Guards for the trainer weight-update NCCL group in
``tokenspeed.runtime.execution.weight_update_group``.

CPU-only, no real NCCL: ``model_runner`` itself cannot be imported on a host
without the compiled ``tokenspeed_kernel`` / ``tokenspeed_triton``
extensions, so these two helpers live in their own dependency-free module and
are exercised here with fakes instead of a real process group.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.weight_update_group import (
    _assert_not_split,
    _no_default_group_split,
)


def test_noop_when_not_initialized(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    # Any access to _get_default_group would raise, since there is none;
    # the context manager must not even look at it when not initialized.
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    with _no_default_group_split():
        pass


def test_clears_and_restores_bound_device(monkeypatch):
    fake_default_pg = SimpleNamespace(bound_device_id=torch.device("cuda", 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: fake_default_pg,
    )

    with _no_default_group_split():
        assert fake_default_pg.bound_device_id is None

    assert fake_default_pg.bound_device_id == torch.device("cuda", 0)


def test_restores_bound_device_when_body_raises(monkeypatch):
    fake_default_pg = SimpleNamespace(bound_device_id=torch.device("cuda", 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: fake_default_pg,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with _no_default_group_split():
            assert fake_default_pg.bound_device_id is None
            raise RuntimeError("boom")

    assert fake_default_pg.bound_device_id == torch.device("cuda", 0)


def test_assert_not_split_raises_when_split_from_set():
    fake_backend = SimpleNamespace(options=SimpleNamespace(split_from=object()))
    fake_pg = SimpleNamespace(_get_backend=lambda device: fake_backend)

    with pytest.raises(RuntimeError, match="split"):
        _assert_not_split(fake_pg, torch.device("cuda", 0))


def test_assert_not_split_passes_when_clean():
    fake_backend = SimpleNamespace(options=SimpleNamespace(split_from=None))
    fake_pg = SimpleNamespace(_get_backend=lambda device: fake_backend)

    _assert_not_split(fake_pg, torch.device("cuda", 0))  # no raise


def test_assert_not_split_passes_when_backend_unavailable():
    def _raise_no_backend(device):
        raise RuntimeError("no backend for device")

    fake_pg = SimpleNamespace(_get_backend=_raise_no_backend)

    _assert_not_split(fake_pg, torch.device("cuda", 0))  # no raise
