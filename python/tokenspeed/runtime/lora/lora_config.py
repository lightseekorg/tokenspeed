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

"""LoRA adapter configuration and metadata."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoraConfig:
    """Configuration for a single LoRA adapter.

    Loaded from the adapter's ``adapter_config.json`` (PEFT format).
    """

    # Identifier used at request time (e.g. "sql-expert")
    name: str

    # Filesystem path to the adapter directory or file
    path: str

    # LoRA rank (r)
    r: int = 16

    # LoRA alpha scaling factor
    lora_alpha: int = 16

    # Target modules (e.g. ["q_proj", "v_proj"])
    target_modules: list[str] = field(default_factory=list)

    # Whether this adapter is pinned in GPU memory (never evicted)
    pinned: bool = False

    # Base model name for compatibility checking
    base_model_name_or_path: Optional[str] = None

    @classmethod
    def from_path(cls, name: str, path: str, pinned: bool = False) -> "LoraConfig":
        """Load LoraConfig from a PEFT adapter directory."""
        config_file = os.path.join(path, "adapter_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"adapter_config.json not found at {config_file}. "
                "The path must point to a PEFT-format adapter directory."
            )
        with open(config_file) as f:
            raw = json.load(f)

        return cls(
            name=name,
            path=path,
            r=raw.get("r", 16),
            lora_alpha=raw.get("lora_alpha", 16),
            target_modules=raw.get("target_modules") or [],
            pinned=pinned,
            base_model_name_or_path=raw.get("base_model_name_or_path"),
        )

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r if self.r > 0 else 1.0
