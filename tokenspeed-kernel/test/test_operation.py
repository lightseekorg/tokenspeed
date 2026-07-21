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

import importlib
import inspect

import pytest
import torch
from tokenspeed_kernel.operation import (
    DuplicateOperationError,
    OperationRegistry,
    OperationSchema,
    UnknownOperationError,
)
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@pytest.fixture
def operation_catalog(monkeypatch: pytest.MonkeyPatch, fresh_registry) -> None:
    monkeypatch.setattr(OperationRegistry, "_instance", None)


def _signature():
    return format_signature(x=dense_tensor_format(torch.float32))


def _reference(x, scale=1.0, *, out=None):
    result = x * scale
    if out is not None:
        return out.copy_(result)
    return result


def _schema(
    *,
    family: str = "test_operation",
    mode: str = "apply",
    validate_signatures=lambda signatures: None,
    validate_traits=lambda traits: None,
) -> OperationSchema:
    return OperationSchema(
        family=family,
        mode=mode,
        reference=_reference,
        validate_signatures=validate_signatures,
        validate_traits=validate_traits,
    )


def test_schema_exposes_and_calls_reference(operation_catalog) -> None:
    schema = _schema().publish()
    tensor = torch.tensor([2.0])

    assert schema.signature == inspect.signature(_reference)
    assert torch.equal(schema.reference(tensor, 3.0), torch.tensor([6.0]))
    assert OperationRegistry.get().lookup(*schema.id) is schema
    assert OperationRegistry.get().schemas() == (schema,)
    assert KernelRegistry.get().get_for_operator(*schema.id) == []


def test_duplicate_and_unknown_schemas_fail(operation_catalog) -> None:
    _schema().publish()

    with pytest.raises(DuplicateOperationError, match="already has a schema"):
        _schema().publish()
    with pytest.raises(UnknownOperationError, match="unknown operation missing.mode"):
        OperationRegistry.get().lookup("missing", "mode")


@pytest.mark.parametrize("parameter", ["args", "kwargs"])
def test_reference_must_define_canonical_parameters(
    operation_catalog, parameter: str
) -> None:
    if parameter == "args":

        def reference(x, *args):
            return x, args

    else:

        def reference(x, **kwargs):
            return x, kwargs

    with pytest.raises(TypeError, match="must list every canonical parameter"):
        OperationSchema(
            "test_operation",
            "apply",
            reference,
            lambda signatures: None,
            lambda traits: None,
        )


def test_late_schema_publication_validates_existing_kernels(operation_catalog) -> None:
    spec = KernelSpec(
        name="already_registered",
        family="test_operation",
        mode="apply",
        format_signatures=frozenset({_signature()}),
    )
    KernelRegistry.get().register(spec, _reference)

    schema = _schema().publish()

    assert OperationRegistry.get().lookup(*schema.id) is schema


def test_legacy_decorator_routes_through_schema_validators(operation_catalog) -> None:
    seen = []

    def validate_signatures(signatures):
        seen.append(("signatures", signatures))

    def validate_traits(traits):
        seen.append(("traits", traits))

    schema = _schema(
        validate_signatures=validate_signatures,
        validate_traits=validate_traits,
    ).publish()
    signatures = frozenset({_signature()})
    traits = {"aligned": frozenset({True})}

    @register_kernel(
        *schema.id,
        name="valid_adapter",
        solution="test",
        signatures=signatures,
        traits=traits,
    )
    def adapter(x, scale=1.0, *, out=None):
        return _reference(x, scale, out=out)

    spec = KernelRegistry.get().get_by_name("valid_adapter")
    assert spec is not None
    assert seen == [
        ("signatures", signatures),
        ("traits", traits),
    ]


def test_variadic_adapter_can_preserve_legacy_registration(operation_catalog) -> None:
    schema = _schema().publish()

    @register_kernel(
        *schema.id,
        name="variadic_adapter",
        solution="test",
        signatures={_signature()},
    )
    def adapter(*args, **kwargs):
        return _reference(*args, **kwargs)

    assert KernelRegistry.get().get_impl("variadic_adapter") is adapter


def test_uninspectable_callable_can_preserve_legacy_registration(
    operation_catalog,
) -> None:
    class OpaqueAdapter:
        __signature__ = object()

        def __call__(self, x, scale=1.0, *, out=None):
            return _reference(x, scale, out=out)

    adapter = OpaqueAdapter()
    schema = _schema().publish()
    register_kernel(
        *schema.id,
        name="opaque_adapter",
        solution="test",
        signatures={_signature()},
    )(adapter)

    assert KernelRegistry.get().get_impl("opaque_adapter") is adapter


def test_reference_parameter_kinds_define_launch_form(operation_catalog) -> None:
    schema = _schema().publish()

    with pytest.raises(TypeError, match="cannot accept the canonical"):
        register_kernel(
            *schema.id,
            name="one_call_form",
            solution="test",
            signatures={_signature()},
        )(lambda *, x, scale=1.0, out=None: _reference(x, scale, out=out))


def test_keyword_only_reference_accepts_keyword_only_kernel(operation_catalog) -> None:
    def reference(*, x, scale=1.0):
        return x * scale

    schema = OperationSchema(
        "test_operation",
        "keyword_apply",
        reference,
        lambda signatures: None,
        lambda traits: None,
    ).publish()

    @register_kernel(
        *schema.id,
        name="keyword_adapter",
        solution="test",
        signatures={_signature()},
    )
    def adapter(*, x, scale=1.0):
        return reference(x=x, scale=scale)

    assert KernelRegistry.get().get_impl("keyword_adapter") is adapter


def test_invalid_metadata_is_rejected_before_registry_mutation(
    operation_catalog,
) -> None:
    def validate_traits(traits):
        unknown = set(traits) - {"aligned"}
        if unknown:
            raise ValueError(f"unknown traits: {sorted(unknown)}")

    schema = _schema(validate_traits=validate_traits).publish()

    @register_kernel(
        *schema.id,
        name="replace_me",
        solution="test",
        signatures={_signature()},
    )
    def original(x, scale=1.0, *, out=None):
        return _reference(x, scale, out=out)

    with pytest.raises(ValueError, match="unknown traits"):

        @register_kernel(
            *schema.id,
            name="replace_me",
            solution="test",
            signatures={_signature()},
            traits={"typo": frozenset({True})},
        )
        def replacement(x, scale=1.0, *, out=None):
            return _reference(x, scale, out=out)

    assert KernelRegistry.get().get_impl("replace_me") is original


def test_incompatible_callable_is_rejected(operation_catalog) -> None:
    schema = _schema().publish()

    with pytest.raises(TypeError, match="cannot accept the canonical call"):

        @register_kernel(
            *schema.id,
            name="missing_out",
            solution="test",
            signatures={_signature()},
        )
        def adapter(x, scale=1.0):
            return x * scale

    assert KernelRegistry.get().get_by_name("missing_out") is None


def test_direct_registry_registration_cannot_bypass_schema(operation_catalog) -> None:
    def validate_signatures(signatures):
        if signatures != frozenset({_signature()}):
            raise ValueError("invalid signatures")

    schema = _schema(validate_signatures=validate_signatures).publish()
    spec = KernelSpec(
        name="direct",
        family=schema.family,
        mode=schema.mode,
        format_signatures=frozenset(),
    )

    with pytest.raises(ValueError, match="invalid signatures"):
        KernelRegistry.get().register(spec, _reference)
    assert KernelRegistry.get().get_by_name("direct") is None


def test_unmodeled_registration_is_unchanged(operation_catalog) -> None:
    @register_kernel(
        "legacy",
        "unmodeled",
        name="legacy_kernel",
        solution="test",
        signatures=set(),
    )
    def legacy(value):
        return value

    assert KernelRegistry.get().get_impl("legacy_kernel") is legacy


def test_kernel_registry_reset_does_not_remove_schemas(operation_catalog) -> None:
    schema = _schema().publish()

    KernelRegistry.reset()

    assert OperationRegistry.get().lookup(*schema.id) is schema


def test_registered_traits_are_copied(operation_catalog) -> None:
    schema = _schema().publish()
    traits = {"aligned": frozenset({True})}

    @register_kernel(
        *schema.id,
        name="immutable_traits",
        solution="test",
        signatures={_signature()},
        traits=traits,
    )
    def adapter(x, scale=1.0, *, out=None):
        return _reference(x, scale, out=out)

    traits["typo"] = frozenset({True})
    spec = KernelRegistry.get().get_by_name("immutable_traits")
    assert spec is not None
    assert "typo" not in spec.traits


def test_contract_package_publishes_complete_catalog() -> None:
    import tokenspeed_kernel.contracts as contracts
    import tokenspeed_kernel.contracts.ops as ops_contracts

    exports = {("attention", "mha_prefill"): "MHA_PREFILL"}
    catalog = {schema.id: schema for schema in contracts.list_operation_schemas()}

    assert catalog.keys() == exports.keys()
    for (family, mode), export in exports.items():
        family_contracts = importlib.import_module(
            f"tokenspeed_kernel.contracts.ops.{family}"
        )
        mode_contract = importlib.import_module(
            f"tokenspeed_kernel.contracts.ops.{family}.{mode}"
        )
        schema = getattr(contracts, export)

        assert getattr(ops_contracts, export) is schema
        assert getattr(family_contracts, export) is schema
        assert getattr(mode_contract, export) is schema
        assert catalog[(family, mode)] is schema
        assert contracts.get_operation_schema(family, mode) is schema
