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

"""Model weight validity across allocator release, remap and full updates.

The owner lives on the forward thread. Remapping discarded pages never makes
weights usable: only a complete successful update does. Immutable model
constants have small explicit CPU snapshots; old parameters are never copied.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class DiscardableModel(Protocol):
    supports_weight_discard: bool

    def required_weight_update_names(self) -> set[str]: ...

    def weight_discard_preserved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def parameters(self): ...

    def named_parameters(self, *, remove_duplicate: bool): ...


def _layout(tensor: torch.Tensor) -> tuple:
    return (
        tensor.device,
        tensor.dtype,
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.stride(),
    )


def _storage(tensor: torch.Tensor) -> tuple:
    return tensor.device, tensor.untyped_storage().data_ptr()


@dataclass(frozen=True)
class PreservedTensor:
    tensor: torch.Tensor
    layout: tuple
    value: torch.Tensor

    def restore(self) -> None:
        if _layout(self.tensor) != self.layout:
            raise RuntimeError("A preserved model tensor changed its captured storage")
        self.tensor.copy_(self.value)


def backup_weights(
    *, policy: str | None, is_draft: bool, draft_policy: str | None
) -> bool:
    """Resolve allocation backup independently for target and frozen draft."""
    if policy not in (None, "cpu_backup", "discard"):
        raise ValueError(f"Invalid weight memory policy: {policy}")
    if policy != "discard":
        return True
    if draft_policy not in ("retain", "refresh"):
        raise ValueError("Discard requires an explicit draft update policy")
    return is_draft and draft_policy == "retain"


class WeightMemoryState:
    """A failed transition poisons the deployment until it is restarted."""

    def __init__(self, *, policy: str, models: Sequence[nn.Module]) -> None:
        if policy not in ("cpu_backup", "discard"):
            raise ValueError(f"Invalid weight memory policy: {policy}")
        self.policy = policy
        self.resident = True
        self.valid = True
        self.failed = False
        self.updating = False
        self.local_update_success: bool | None = None
        self.kv_released = False
        self.models = tuple(models)
        self.preserved: list[PreservedTensor] = []
        self.parameter_layouts: list[dict[str, tuple]] = []
        if policy != "discard":
            return
        if not self.models:
            raise ValueError("Discard needs at least one model")
        parameter_storages = {
            _storage(p)
            for model in self.models
            for p in model.parameters()
            if p.numel()
        }
        saved = set()
        for model in self.models:
            if (
                not isinstance(model, DiscardableModel)
                or not model.supports_weight_discard
            ):
                raise ValueError(
                    f"{type(model).__name__} does not support weight discard"
                )
            # Fail at startup if the model cannot describe a complete replacement.
            if not isinstance(model.required_weight_update_names(), set):
                raise TypeError("Required checkpoint names must be a set")
            self.parameter_layouts.append(
                {
                    name: _layout(value)
                    for name, value in model.named_parameters(remove_duplicate=False)
                }
            )
            for tensor in model.weight_discard_preserved_tensors():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Preserved model state must contain only tensors")
                if tensor.numel() and _storage(tensor) in parameter_storages:
                    raise ValueError(
                        "Preserved model state must not alias parameter storage"
                    )
                layout = _layout(tensor)
                if layout not in saved:
                    self.preserved.append(
                        PreservedTensor(tensor, layout, tensor.detach().cpu().clone())
                    )
                    saved.add(layout)

    def require_healthy(self) -> None:
        if self.failed:
            raise RuntimeError("A weight transition failed; restart the deployment")

    def require_usable(self) -> None:
        self.require_healthy()
        if not self.resident or not self.valid or self.updating:
            raise RuntimeError(
                "Weights are unavailable; a complete weight update is required"
            )

    def mark_kv_released(self) -> None:
        self.require_healthy()
        self.kv_released = True

    def before_kv_resume(self) -> None:
        self.require_healthy()
        if self.policy == "discard":
            self.require_usable()

    def mark_kv_resumed(self) -> None:
        self.kv_released = False

    def release(self, operation: Callable[[], None]) -> None:
        self.require_healthy()
        if not self.resident or self.updating:
            raise RuntimeError("Cannot release unavailable or updating weights")
        if self.policy == "discard" and not self.kv_released:
            raise RuntimeError("Release KV before discarding weights")
        self.resident = False
        if self.policy == "discard":
            self.valid = False
        try:
            operation()
        except BaseException:
            self.failed = True
            raise

    def resume(self, operation: Callable[[], None]) -> None:
        self.require_healthy()
        if self.resident:
            raise RuntimeError("Weight storage is already resident")
        try:
            operation()
            with torch.no_grad():
                for saved in self.preserved:
                    saved.restore()
            self.resident = True
        except BaseException:
            self.failed = True
            raise

    def check_update(self) -> None:
        """Check the local load prerequisites without starting a transition."""
        self.require_healthy()
        if not self.resident or self.updating:
            raise RuntimeError("Resume weight storage before updating")
        if self.policy == "discard" and not self.kv_released:
            raise RuntimeError("Release KV before a discard-policy weight update")

    def begin_update(self) -> bool:
        self.check_update()
        self.valid = False
        self.updating = True
        self.local_update_success = None
        return self.policy == "discard"

    def finish_update(self, *, success: bool) -> None:
        if not self.updating:
            raise RuntimeError("No weight update is active")
        if not success:
            self.updating = False
            self.local_update_success = False
            self.failed = True
            return
        try:
            for model, expected in zip(self.models, self.parameter_layouts):
                actual = {
                    name: _layout(value)
                    for name, value in model.named_parameters(remove_duplicate=False)
                }
                if actual != expected:
                    raise RuntimeError(
                        "Weight update changed captured parameter storage"
                    )
            if self.policy == "discard":
                actual_constants = {
                    _layout(tensor)
                    for model in self.models
                    for tensor in model.weight_discard_preserved_tensors()
                }
                if actual_constants != {saved.layout for saved in self.preserved}:
                    raise RuntimeError("Weight update replaced preserved model storage")
            self.local_update_success = True
        except BaseException:
            self.updating = False
            self.local_update_success = False
            self.failed = True
            raise

    def commit_update(self, success: bool) -> None:
        """Apply the replica-wide result of a load every rank started.

        The control plane commits only after all ranks passed
        ``check_update`` and attempted the load; a refusal before loading
        never reaches here. A failure therefore poisons every rank, including
        one whose own load succeeded: a peer may hold a partial version.
        """
        if not success:
            self.valid = False
            self.failed = True
            self.updating = False
            self.local_update_success = None
            return
        self.require_healthy()
        if not self.updating or self.local_update_success is not True:
            self.valid = False
            self.failed = True
            raise RuntimeError("Cannot commit weights before a complete local update")
        self.valid = True
        self.updating = False
        self.local_update_success = None
