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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch

__all__ = [
    "FormatSignature",
    "ScaleFormat",
    "TensorFormat",
    "dense_tensor_format",
    "format_signature",
    "tensor_format",
    "format_signatures",
]


@dataclass(frozen=True)
class ScaleFormat:
    """Metadata representation for one tensor scale sidecar.

    Args:
        storage_dtype: Physical dtype used by the scale tensor.
        granularity: Scale granularity, such as "tensor", "channel", "block".
        block_shape: Logical block shape covered by each scale value when
            granularity is block-based. Required for block scales unless
            ``dynamic_block_shape`` is true.
        dynamic_block_shape: Whether a block scale intentionally supports a
            runtime-selected block shape and should not match on one fixed
            shape.
    """

    storage_dtype: torch.dtype
    granularity: str
    block_shape: tuple[int, ...] | None = None
    dynamic_block_shape: bool = False

    def __post_init__(self) -> None:
        if self.block_shape is not None:
            block_shape = tuple(self.block_shape)
            if not block_shape or any(dim <= 0 for dim in block_shape):
                raise ValueError("block_shape must contain positive dimensions")
            object.__setattr__(self, "block_shape", block_shape)

        if self.granularity == "block":
            if self.block_shape is not None and self.dynamic_block_shape:
                raise ValueError(
                    "block_shape and dynamic_block_shape are mutually exclusive"
                )
            if self.block_shape is None and not self.dynamic_block_shape:
                raise ValueError(
                    "block scale format requires block_shape or dynamic_block_shape=True"
                )
            return

        if self.block_shape is not None:
            raise ValueError("block_shape is only valid for block scale formats")
        if self.dynamic_block_shape:
            raise ValueError(
                "dynamic_block_shape is only valid for block scale formats"
            )

    def __str__(self) -> str:
        parts = [self.granularity, f"storage={self.storage_dtype}"]
        if self.block_shape is not None:
            parts.append(f"block={self.block_shape}")
        elif self.dynamic_block_shape:
            parts.append("block=dynamic")
        return "scale(" + ", ".join(parts) + ")"


@dataclass(frozen=True)
class TensorFormat:
    """Metadata representation for one logical tensor.

    Args:
        storage_dtype: Physical dtype used by the main tensor payload.
        format: Logical representation format, such as "dense",
            "scaled-fp8", "mxfp8", "mxfp4", or "nvfp4".
            Use "dense" with an FP8 storage dtype for unscaled FP8 tensors.
        scale: Optional scale sidecar metadata bundled with this tensor role.
    """

    storage_dtype: torch.dtype
    format: str = "dense"
    scale: ScaleFormat | None = None

    def __post_init__(self) -> None:
        if self.format == "fp8":
            raise ValueError(
                "TensorFormat format=fp8 is ambiguous; use dense for "
                "unscaled FP8 or scaled-fp8 with scale metadata"
            )
        if self.format == "scaled-fp8" and self.scale is None:
            raise ValueError("scaled-fp8 tensor format requires scale metadata")

    def __str__(self) -> str:
        if self.scale is None:
            return f"{self.format}[storage={self.storage_dtype}]"
        return f"{self.format}[storage={self.storage_dtype}, {self.scale}]"


@dataclass(frozen=True)
class FormatSignature:
    """One concrete set of role-indexed tensor formats for all tensor operands.

    Each role appears at most once and maps to exactly one ``TensorFormat``.
    A kernel that supports alternatives for a role represents them as multiple
    ``FormatSignature`` values in ``KernelSpec.format_signatures`` rather than
    as multiple formats inside one signature.
    """

    roles: tuple[tuple[str, TensorFormat], ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        normalized: list[tuple[str, TensorFormat]] = []
        for role, tensor_format in sorted(self.roles, key=lambda item: item[0]):
            if role in seen:
                raise ValueError(f"duplicate format role {role!r}")
            seen.add(role)
            normalized.append((role, tensor_format))
        object.__setattr__(self, "roles", tuple(normalized))

    def format_for(self, role: str) -> TensorFormat | None:
        """Return the format for role, or None if it is absent."""
        for role_name, tensor_format in self.roles:
            if role_name == role:
                return tensor_format
        return None

    def primary_storage_dtype(self) -> torch.dtype | None:
        """Return the dtype used by dtype-oriented tooling and filters.

        The registry and selection path use the full signature. This helper is
        only for compatibility with user-facing commands that still ask for a
        single dtype filter, such as numerics and benchmark CLIs.
        """
        preferred_roles = (
            "q",
            "a",
            "x",
            "input",
            "logits",
            "out_a",
            "hidden",
        )
        for role in preferred_roles:
            tensor_format = self.format_for(role)
            if tensor_format is not None:
                return tensor_format.storage_dtype
        return self.roles[0][1].storage_dtype if self.roles else None

    def __str__(self) -> str:
        return (
            ", ".join(f"{role}={tensor_format}" for role, tensor_format in self.roles)
            or "none"
        )


def tensor_format(
    format: str,
    storage_dtype: torch.dtype,
    *,
    scale: ScaleFormat | None = None,
) -> TensorFormat:
    """Construct a format for one tensor role.

    Args:
        format: Logical representation format, such as "dense",
            "scaled-fp8", "mxfp8", "mxfp4", or "nvfp4".
            Use "dense" with an FP8 storage dtype for unscaled FP8 tensors.
        storage_dtype: Physical dtype used by the main tensor payload.
        scale: Optional scale sidecar metadata bundled with this tensor role.
    """
    return TensorFormat(storage_dtype=storage_dtype, format=format, scale=scale)


def dense_tensor_format(storage_dtype: torch.dtype) -> TensorFormat:
    """Construct a dense, unscaled tensor format for storage_dtype."""
    return tensor_format("dense", storage_dtype)


def format_signature(**roles: TensorFormat) -> FormatSignature:
    """Construct one concrete role-indexed format signature.

    Keyword names are logical tensor roles for the operator, for example
    a/b for GEMM or q/k_cache/v_cache for attention.
    Values are the exact formats required for those roles. Each role gets
    exactly one ``TensorFormat``; represent alternatives by constructing
    multiple ``FormatSignature`` values.

    Examples:
        >>> import torch
        >>> format_signature(
        ...     a=dense_tensor_format(torch.bfloat16),
        ...     b=tensor_format("mxfp4", torch.uint8),
        ... )

        This creates one signature equivalent to:

        >>> FormatSignature(
        ...     (
        ...         ("a", dense_tensor_format(torch.bfloat16)),
        ...         ("b", tensor_format("mxfp4", torch.uint8)),
        ...     )
        ... )
    """
    return FormatSignature(tuple(roles.items()))


def format_signatures(
    roles: str | Iterable[str],
    format: str,
    storage_dtypes: Iterable[torch.dtype],
    *,
    scale: ScaleFormat | None = None,
) -> frozenset[FormatSignature]:
    """Construct same-format signatures for each storage dtype.

    Args:
        roles: Logical tensor roles. Pass a string for one role or an iterable
            for multiple roles.
        format: Logical representation format assigned to every role.
        storage_dtypes: Physical dtypes used by every role, one signature per
            dtype.
        scale: Optional scale sidecar metadata assigned to every role.

    This helper expands dtype alternatives into separate signatures; it does
    not put multiple formats on one role. Use ``format_signature`` directly for
    mixed-role combinations such as dense activations with quantized weights.

    Examples:
        >>> import torch
        >>> format_signatures(
        ...     ("q", "k_cache", "v_cache"),
        ...     "dense",
        ...     {torch.float16, torch.bfloat16},
        ... )

        This expands to a ``frozenset`` containing one signature per dtype,
        equivalent to:

        >>> frozenset(
        ...     {
        ...         format_signature(
        ...             q=dense_tensor_format(torch.float16),
        ...             k_cache=dense_tensor_format(torch.float16),
        ...             v_cache=dense_tensor_format(torch.float16),
        ...         ),
        ...         format_signature(
        ...             q=dense_tensor_format(torch.bfloat16),
        ...             k_cache=dense_tensor_format(torch.bfloat16),
        ...             v_cache=dense_tensor_format(torch.bfloat16),
        ...         ),
        ...     }
        ... )
    """
    normalized_roles = (roles,) if isinstance(roles, str) else tuple(roles)
    return frozenset(
        format_signature(
            **{
                role: tensor_format(format, storage_dtype, scale=scale)
                for role in normalized_roles
            }
        )
        for storage_dtype in storage_dtypes
    )
