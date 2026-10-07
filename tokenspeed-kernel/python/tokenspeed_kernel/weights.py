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

"""Storage-level weight layouts and broker-owned tensor replacement.

The transformation function identifies a storage layout; ``None`` means the
canonical loaded layout. Views share a layout regardless of shape or dtype.
Participating layers opt into checking enrollment before kernel selection;
subsequent weight writes must pass through the broker.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    import torch

UNKNOWN_LAYOUT = object()
WeightSlot = tuple[object, str]


class WeightBroker:
    """Track weight storage without retaining tensors or their allocations.

    Ordinary views share their storage's record automatically. A new
    allocation is untracked until enrolled or produced through ``preprocess``.
    No tensor attributes or raw allocation addresses are used.
    """

    def __init__(self) -> None:
        self._layouts: WeakKeyDictionary = WeakKeyDictionary()

    def enroll(self, tensor: torch.Tensor, layout: Callable | None = None) -> None:
        """Record storage produced by a transformation, or a canonical load."""
        self._layouts[tensor.untyped_storage()] = layout

    def layout(
        self, tensor: torch.Tensor, *, allow_unknown: bool = False
    ) -> Callable | None:
        """Return the layout; optionally treat nonparticipating storage as canonical."""
        layout = self.tracked_layout(tensor)
        if layout is UNKNOWN_LAYOUT:
            if allow_unknown:
                return None
            raise ValueError(
                "untracked weight storage; load or replace weights through the broker"
            )
        return layout

    def tracked_layout(self, tensor: torch.Tensor) -> Callable | None | object:
        """Return the layout or ``UNKNOWN_LAYOUT`` for unenrolled storage."""
        return self._layouts.get(tensor.untyped_storage(), UNKNOWN_LAYOUT)

    def preprocess(
        self,
        preprocessor: tuple[tuple[str, ...], Callable],
        owner: object,
        *,
        config=None,
    ) -> dict[str, WeightSlot]:
        """Transform named owner attributes and install the returned tensors.

        Args:
            preprocessor: Source attribute names and a function accepting those
                tensors as keyword arguments. Missing optional attributes are
                passed as ``None``.
                Returns an output tensor dictionary, optionally followed by a
                destination dictionary and a selected layout function. The
                function itself identifies the layout unless one is returned;
                an explicit ``None`` selects canonical storage.
            owner: Object containing the source attributes. Missing or ``None``
                destinations mean ``(owner, output_name)``; explicit destinations
                are ordinary ``(object, attribute_name)`` pairs.
            config: Optional metadata passed as the ``config`` keyword between
                the layer, preprocessor, and eventual compatible kernel. Must
                not retain additional tensor copies or select or pin a kernel;
                return tensor results through the broker-managed outputs.

        Returns:
            Installed attribute bindings by output name. Superseded source
            references are cleared. Existing Parameters retain their identities
            and loader metadata, updated by attributes on the returned tensors.
            All tensor outputs receive the selected layout,
            including unchanged companion tensors. Transformations may mutate
            inputs in place; there is no rollback if the function fails.
        """
        import torch

        names, transform = preprocessor
        inputs = {name: getattr(owner, name, None) for name in names}
        with torch.no_grad():
            outputs = (
                transform(**inputs, config=config)
                if config is not None
                else transform(**inputs)
            )
        layout = transform
        # Producers may bind optional metadata with partial; the underlying
        # preprocessing function, not its wrapper, identifies the layout.
        while isinstance(layout, partial):
            layout = layout.func
        destinations = {}
        if isinstance(outputs, tuple):
            if len(outputs) == 3:
                outputs, destinations, layout = outputs
            else:
                outputs, destinations = outputs
        destinations = destinations or {}
        installed = {}
        for name, tensor in outputs.items():
            destination, attr = destinations.get(name) or (owner, name)
            previous = getattr(destination, attr, None)
            if tensor is not None and isinstance(previous, torch.nn.Parameter):
                previous.data = tensor
                previous.__dict__.update(tensor.__dict__)
                tensor = previous
            else:
                if (
                    tensor is not None
                    and isinstance(destination, torch.nn.Module)
                    and attr not in destination._buffers
                ):
                    parameter = torch.nn.Parameter(tensor, requires_grad=False)
                    parameter.__dict__.update(tensor.__dict__)
                    tensor = parameter
                setattr(destination, attr, tensor)
            if tensor is not None:
                self.enroll(tensor, layout)
                installed[name] = (destination, attr)
        destination_slots = {(id(obj), attr) for obj, attr in installed.values()}
        for name in names:
            if (id(owner), name) not in destination_slots:
                setattr(owner, name, None)
        return installed


_weight_broker = WeightBroker()


def get_weight_broker() -> WeightBroker:
    """Return the process-wide, weakly held storage-layout registry."""
    return _weight_broker
