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

"""Explicit draft-model contract for target capture configuration at startup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


class TargetCaptureConfigurator(ABC):
    """A draft model that configures the target features its checkpoint consumes.

    Model setup invokes this interface on every pipeline stage, before cache
    construction and before any drafter exists. Drafter resource binding and
    per-forward hooks never install or replace this capture configuration.
    Each concrete draft owns its target-family adapter; this interface does
    not require every target to implement an algorithm-named setter.
    """

    @abstractmethod
    def configure_target(self, target_model: nn.Module, target_config) -> None:
        """Validate checkpoint compatibility and install trained capture semantics.

        Args:
            target_model: The loaded target model on this rank.
            target_config: The resolved target text configuration.
        """
