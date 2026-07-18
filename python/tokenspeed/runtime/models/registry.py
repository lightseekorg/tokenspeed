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

"""Model registry helpers for resolving runtime model entry classes."""

import importlib
import pkgutil
from collections.abc import Set
from dataclasses import dataclass, field
from functools import lru_cache

import torch.nn as nn

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, type[nn.Module] | str] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys()

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        if model_arch not in self.models:
            return None

        return self.models[model_arch]

    def _normalize_archs(
        self,
        architectures: str | list[str],
    ) -> list[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        return architectures

    def resolve_model_cls(
        self,
        architectures: str | list[str],
    ) -> tuple[type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


@lru_cache
def import_model_classes():
    """Import model modules and collect their ``EntryClass`` exports."""
    model_arch_name_to_cls = {}
    package_name = "tokenspeed.runtime.models"
    package = importlib.import_module(package_name)

    for _, name, _ in pkgutil.iter_modules(package.__path__, package_name + "."):

        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Failed to import model module {name}.") from exc
        if hasattr(module, "EntryClass"):
            entry = module.EntryClass
            if isinstance(
                entry, list
            ):  # To support multiple model classes in one module
                for tmp in entry:
                    if tmp.__name__ in model_arch_name_to_cls:
                        raise ValueError(
                            f"Duplicated model implementation for {tmp.__name__}"
                        )
                    model_arch_name_to_cls[tmp.__name__] = tmp
            else:
                if entry.__name__ in model_arch_name_to_cls:
                    raise ValueError(
                        f"Duplicated model implementation for {entry.__name__}"
                    )
                model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry(import_model_classes())
