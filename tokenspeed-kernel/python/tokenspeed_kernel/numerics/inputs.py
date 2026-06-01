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

from typing import Any, Callable

import torch
from tokenspeed_kernel.signature import FormatSignature

__all__ = [
    "InputGenerator",
    "get_benchmark_shapes",
    "get_input_generator",
    "get_standard_shapes",
    "set_benchmark_shapes",
    "set_input_generator",
    "set_standard_shapes",
]

InputGeneratorFactory = Callable[..., "InputGenerator"]

_INPUT_GENERATORS: dict[tuple[str, str], InputGeneratorFactory] = {}
_STANDARD_SHAPES: dict[tuple[str, str], list[dict[str, Any]]] = {}
_BENCHMARK_SHAPES: dict[tuple[str, str], list[dict[str, Any]]] = {}


class InputGenerator:
    """Generates test inputs for a given operator family/mode."""

    def __init__(
        self,
        op_family: str,
        op_mode: str,
        dtype: torch.dtype,
        traits: dict,
        *,
        format_signature: FormatSignature | None = None,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.op_family = op_family
        self.op_mode = op_mode
        self.dtype = dtype
        self.traits = traits
        self.format_signature = format_signature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        rng_device = "cuda" if self.device.startswith("cuda") else "cpu"
        self.rng = torch.Generator(device=rng_device).manual_seed(seed)

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


def set_input_generator(
    op_family: str,
    op_mode: str,
    generator_factory: InputGeneratorFactory,
) -> None:
    _INPUT_GENERATORS[(op_family, op_mode)] = generator_factory


def set_standard_shapes(
    op_family: str,
    op_mode: str,
    shapes: list[dict[str, Any]],
) -> None:
    _STANDARD_SHAPES[(op_family, op_mode)] = [dict(shape) for shape in shapes]


def set_benchmark_shapes(
    op_family: str,
    op_mode: str,
    shapes: list[dict[str, Any]],
) -> None:
    _BENCHMARK_SHAPES[(op_family, op_mode)] = [dict(shape) for shape in shapes]


def get_input_generator(
    op_family: str,
    op_mode: str,
    dtype: torch.dtype,
    traits: dict,
    *,
    format_signature: FormatSignature | None = None,
    device: str | None = None,
    seed: int = 42,
) -> InputGenerator:
    factory = _INPUT_GENERATORS.get((op_family, op_mode))
    if factory is None:
        known = ", ".join(f"{f}.{m}" for f, m in sorted(_INPUT_GENERATORS)) or "none"
        raise KeyError(
            f"No input generator registered for {op_family}.{op_mode}. Known: {known}"
        )
    return factory(
        op_family,
        op_mode,
        dtype,
        traits,
        format_signature=format_signature,
        device=device,
        seed=seed,
    )


def get_standard_shapes(op_family: str, op_mode: str) -> list[dict[str, Any]]:
    shapes = _STANDARD_SHAPES.get((op_family, op_mode))
    if shapes is None:
        known = ", ".join(f"{f}.{m}" for f, m in sorted(_STANDARD_SHAPES)) or "none"
        raise KeyError(
            f"No standard shapes registered for {op_family}.{op_mode}. Known: {known}"
        )
    return [dict(shape) for shape in shapes]


def get_benchmark_shapes(op_family: str, op_mode: str) -> list[dict[str, Any]]:
    shapes = _BENCHMARK_SHAPES.get((op_family, op_mode))
    if shapes is not None:
        return [dict(shape) for shape in shapes]
    return get_standard_shapes(op_family, op_mode)
