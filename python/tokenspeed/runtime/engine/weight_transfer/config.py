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

"""Weight-transfer configuration (vLLM parity).

Mirrors ``vllm.config.weight_transfer.WeightTransferConfig`` so a trainer
written against vLLM can target tokenspeed unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

# The backends tokenspeed implements. vLLM types the field as
# ``Literal["nccl", "ipc"] | str`` and validates against its engine registry;
# we validate up front against the backends we actually support so misuse
# fails fast with a clear message instead of deep in the worker.
SUPPORTED_BACKENDS: tuple[str, ...] = ("nccl", "ipc")

WeightTransferBackend = Literal["nccl", "ipc"]


@dataclass
class WeightTransferConfig:
    """Configuration for vLLM-compatible RL weight transfer.

    Attributes:
        backend: Transport used to move weights out-of-band from the trainer
            to the inference workers. ``"nccl"`` (disaggregated: separate GPUs
            for train vs. infer) or ``"ipc"`` (colocated: trainer + inference
            share GPUs).
    """

    backend: str = "nccl"

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported weight transfer backend: {self.backend!r}. "
                f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}."
            )

    @classmethod
    def from_json(cls, raw: str | None) -> "WeightTransferConfig":
        """Parse a ``--weight-transfer-config`` JSON string into a config.

        ``None`` / empty string yields the default (``backend="nccl"``), which
        matches vLLM's default when the flag is omitted.
        """
        if not raw:
            return cls()
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--weight-transfer-config must be valid JSON, got {raw!r}: {e}"
            ) from e
        if not isinstance(data, dict):
            raise ValueError(
                f"--weight-transfer-config must be a JSON object, got {type(data).__name__}"
            )
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(
                f"Unknown weight-transfer-config keys: {sorted(unknown)}. "
                f"Allowed: {sorted(cls.__dataclass_fields__)}."
            )
        return cls(**data)
