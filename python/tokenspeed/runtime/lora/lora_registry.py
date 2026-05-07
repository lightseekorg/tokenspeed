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

"""In-process registry that tracks loaded LoRA adapters and maps names to IDs."""

from __future__ import annotations

from typing import Iterator, Optional

from tokenspeed.runtime.lora.lora_config import LoraConfig
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Sentinel value meaning "no adapter" — maps cleanly to int for scheduling.
NO_LORA_ID: int = 0


class LoraRegistry:
    """Thread-unsafe registry; call from the scheduler/engine main thread only.

    TODO: add locking when multi-threaded engine support is needed.
    """

    def __init__(self, max_loras: int) -> None:
        self.max_loras = max_loras
        self._configs: dict[str, LoraConfig] = {}     # name → config
        self._name_to_id: dict[str, int] = {}          # name → integer ID
        self._id_to_name: dict[int, str] = {}          # integer ID → name
        self._next_id: int = 1                         # 0 is reserved for "no lora"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, config: LoraConfig) -> int:
        """Register a new adapter and return its integer ID.

        Raises ``ValueError`` if the adapter is already registered or the
        registry is at capacity.
        """
        if config.name in self._name_to_id:
            raise ValueError(f"LoRA adapter '{config.name}' is already registered.")
        if not config.pinned and len(self._evictable_names()) >= self.max_loras:
            raise ValueError(
                f"LoRA registry is full ({self.max_loras} non-pinned adapters). "
                "Unload an adapter before loading a new one."
            )
        lora_id = self._next_id
        self._next_id += 1
        self._configs[config.name] = config
        self._name_to_id[config.name] = lora_id
        self._id_to_name[lora_id] = config.name
        logger.info("Registered LoRA adapter '%s' → id=%d", config.name, lora_id)
        return lora_id

    def unregister(self, name: str) -> None:
        """Remove an adapter from the registry.

        Raises ``KeyError`` if the name is not registered.
        """
        if name not in self._name_to_id:
            raise KeyError(f"LoRA adapter '{name}' is not registered.")
        lora_id = self._name_to_id.pop(name)
        del self._id_to_name[lora_id]
        del self._configs[name]
        logger.info("Unregistered LoRA adapter '%s' (id=%d)", name, lora_id)

    def get_id(self, name: str) -> Optional[int]:
        """Return the integer ID for an adapter name, or None if not found."""
        return self._name_to_id.get(name)

    def get_config(self, name: str) -> Optional[LoraConfig]:
        """Return the LoraConfig for a registered adapter name."""
        return self._configs.get(name)

    def get_config_by_id(self, lora_id: int) -> Optional[LoraConfig]:
        name = self._id_to_name.get(lora_id)
        return self._configs.get(name) if name else None

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_id

    def __len__(self) -> int:
        return len(self._name_to_id)

    def __iter__(self) -> Iterator[LoraConfig]:
        return iter(self._configs.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evictable_names(self) -> list[str]:
        return [n for n, cfg in self._configs.items() if not cfg.pinned]
