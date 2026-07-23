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

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tokenspeed_kernel.signature import FormatSignature

if TYPE_CHECKING:
    from tokenspeed_kernel.registry import KernelSpec

__all__ = [
    "DuplicateOperationError",
    "OperationRegistry",
    "OperationSchema",
    "UnknownOperationError",
]


SignatureValidator = Callable[[frozenset[FormatSignature]], None]
TraitValidator = Callable[[Mapping[str, frozenset[Any]]], None]


class DuplicateOperationError(ValueError):
    """Raised when two schemas define the same family and mode."""


class UnknownOperationError(LookupError):
    """Raised when a requested operation has no published schema."""


def _validate_name(value: str, description: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value or value != value.strip() or "." in value:
        raise ValueError(
            f"{description} must be a non-empty, trimmed string without '.'"
        )


def _inspect_callable(value: Callable, description: str) -> inspect.Signature:
    if not callable(value):
        raise TypeError(f"{description} must be callable")
    try:
        return inspect.signature(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{description} must have an inspectable signature") from exc


def _validate_implementation_abi(
    canonical: inspect.Signature,
    implementation: Callable,
) -> None:
    """Check the canonical launch form when the implementation is inspectable."""
    if not callable(implementation):
        raise TypeError("kernel implementation must be callable")
    try:
        implementation_signature = inspect.signature(implementation)
    except (TypeError, ValueError):
        # Compiled, PyBind, and framework callables often expose no Python
        # signature. Their call shape is still enforced when the frontend runs.
        return
    sentinels = {name: object() for name in canonical.parameters}
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for name, parameter in canonical.parameters.items():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(sentinels[name])
        else:
            kwargs[name] = sentinels[name]

    try:
        implementation_signature.bind(*args, **kwargs)
    except TypeError as exc:
        implementation_name = getattr(
            implementation,
            "__qualname__",
            getattr(implementation, "__name__", type(implementation).__name__),
        )
        raise TypeError(
            f"kernel implementation {implementation_name!r} cannot accept the "
            f"canonical call {canonical}: {exc}"
        ) from exc


@dataclass(frozen=True)
class OperationSchema:
    """Registration contract for one kernel family and mode.

    The reference callable defines the canonical kernel arguments and operation
    semantics. Per-operation validators inspect the same format signatures and
    trait dictionaries already accepted by :func:`register_kernel`. Parameters
    before ``*`` define positional launch arguments; keyword-only parameters
    define keyword launch arguments.
    """

    family: str
    mode: str
    reference: Callable
    validate_signatures: SignatureValidator
    validate_traits: TraitValidator
    signature: inspect.Signature = field(init=False)

    def __post_init__(self) -> None:
        _validate_name(self.family, "operation family")
        _validate_name(self.mode, "operation mode")
        if not callable(self.validate_signatures):
            raise TypeError("signature validator must be callable")
        if not callable(self.validate_traits):
            raise TypeError("trait validator must be callable")

        signature = _inspect_callable(self.reference, "operation reference")
        variadic = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if variadic:
            names = ", ".join(repr(name) for name in variadic)
            raise TypeError(
                "operation reference must list every canonical parameter; "
                f"variadic parameter(s): {names}"
            )
        object.__setattr__(self, "signature", signature)

    @property
    def id(self) -> tuple[str, str]:
        return (self.family, self.mode)

    def publish(self) -> OperationSchema:
        """Publish this schema in the process-wide operation catalog."""
        OperationRegistry.get().register(self)
        return self

    def validate_registration(self, spec: KernelSpec, implementation: Callable) -> None:
        """Validate one implementation before it enters the kernel registry."""
        if (spec.family, spec.mode) != self.id:
            raise ValueError(
                f"kernel {spec.name!r} belongs to {spec.family}.{spec.mode}, "
                f"not {self.family}.{self.mode}"
            )
        self.validate_signatures(spec.format_signatures)
        self.validate_traits(spec.traits)
        _validate_implementation_abi(self.signature, implementation)


class OperationRegistry:
    """Process-wide catalog of operation schemas, keyed by string identity.

    Schemas are normally published as their modules are imported. The catalog
    then supplies contracts to future kernel registrations and validates any
    kernels that were registered before their schema became available.
    """

    _instance: OperationRegistry | None = None

    def __init__(self) -> None:
        self._schemas: dict[tuple[str, str], OperationSchema] = {}

    @classmethod
    def get(cls) -> OperationRegistry:
        """Return the shared operation catalog, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, schema: OperationSchema) -> None:
        """Publish a schema after validating kernels already using its identity."""
        if not isinstance(schema, OperationSchema):
            raise TypeError("operation registry accepts OperationSchema instances")
        existing_schema = self._schemas.get(schema.id)
        if existing_schema is schema:
            return
        if existing_schema is not None:
            raise DuplicateOperationError(
                f"operation {schema.family}.{schema.mode} already has a schema"
            )

        from tokenspeed_kernel.registry import KernelRegistry

        # Plugins may register kernels before the corresponding contract module
        # is imported. Validate that backlog before making the schema visible so
        # registration order does not weaken the contract.
        existing = KernelRegistry.get().get_for_operator(*schema.id)
        for spec in existing:
            implementation = KernelRegistry.get().get_impl(spec.name)
            if implementation is None:
                raise RuntimeError(f"kernel {spec.name!r} has no registered callable")
            schema.validate_registration(spec, implementation)
        self._schemas[schema.id] = schema

    def find(self, family: str, mode: str) -> OperationSchema | None:
        """Return a schema when present, for callers where it is optional."""
        return self._schemas.get((family, mode))

    def lookup(self, family: str, mode: str) -> OperationSchema:
        """Return a required schema, raising when the operation is unknown."""
        schema = self.find(family, mode)
        if schema is None:
            raise UnknownOperationError(f"unknown operation {family}.{mode}")
        return schema

    def schemas(self) -> tuple[OperationSchema, ...]:
        """Return all schemas in deterministic family-and-mode order."""
        return tuple(self._schemas[key] for key in sorted(self._schemas))
