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

"""Typed launch options contributed by runtime plugins.

Only JSON scalars cross the process boundary. Models share a frozen argument
object owned by their launch; model-specific state never enters that object.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import re
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclasses.dataclass(frozen=True)
class _Schema:
    config_type: type
    validate: Callable[[ServerArgs], None] | None


_SCHEMAS: dict[str, _Schema] = {}


def _argument_fields(config_type):
    return tuple(
        field
        for field in dataclasses.fields(config_type)
        if not field.metadata.get("native_server_arg", False)
    )


def _scalar_type(annotation):
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        choices = [item for item in get_args(annotation) if item is not type(None)]
        if len(choices) != 1:
            raise TypeError(
                "Extra server arguments require one scalar type, optionally None"
            )
        return _scalar_type(choices[0])
    if origin is Literal:
        choices = get_args(annotation)
        if not choices or len({type(item) for item in choices}) != 1:
            raise TypeError("Extra server argument choices must have one scalar type")
        return type(choices[0]), choices
    if annotation not in (str, bool, int, float):
        raise TypeError(f"Unsupported extra server argument type: {annotation}")
    return annotation, None


def register_extra_server_args(
    namespace: str,
    config_type: type,
    *,
    validate: Callable[[ServerArgs], None] | None,
) -> None:
    """Register a plugin dataclass and its optional cross-argument validator.

    Frozen dataclass fields accept str/bool/int/float, homogeneous Literal choices
    and optional None. Fields tagged native_server_arg bind the matching native
    ServerArgs value and cannot be supplied through extra JSON or plugin flags.
    Model-derived state belongs to its execution owner, outside this schema.
    CLI names are --<namespace>-<field-name>; boolean flags take true/false.
    Programmatic callers supply a flat mapping in ServerArgs.extra_server_args.
    Namespace ownership stays in the registry; field names must be unique.
    Failed plugin activation rolls this registration back with its other entries.
    """
    if re.fullmatch(r"[a-z][a-z0-9_]*", namespace) is None:
        raise ValueError(f"Invalid extra server argument namespace: {namespace!r}")
    if namespace in _SCHEMAS:
        raise ValueError(f"Extra server arguments {namespace!r} are already registered")
    if not isinstance(config_type, type) or not dataclasses.is_dataclass(config_type):
        raise TypeError("Extra server arguments require a dataclass type")
    if validate is not None and not callable(validate):
        raise TypeError("Extra server argument validator must be callable or None")
    if not config_type.__dataclass_params__.frozen:
        raise TypeError("Extra server arguments require a frozen dataclass")
    hints = get_type_hints(config_type)
    for field in dataclasses.fields(config_type):
        if not field.init:
            raise TypeError(
                f"Extra server arguments cannot contain derived state: {field.name}"
            )
        if re.fullmatch(r"[a-z][a-z0-9_]*", field.name) is None:
            raise TypeError(f"Invalid extra server argument field: {field.name}")
        _scalar_type(hints[field.name])
    owned = {
        field.name
        for schema in _SCHEMAS.values()
        for field in _argument_fields(schema.config_type)
    }
    conflicts = owned & {field.name for field in _argument_fields(config_type)}
    if conflicts:
        raise ValueError(
            f"Extra server argument fields are already registered: {sorted(conflicts)}"
        )
    _SCHEMAS[namespace] = _Schema(config_type, validate)

    def undo():
        del _SCHEMAS[namespace]

    from tokenspeed.runtime.plugins.registry import record_external

    record_external("extra server arguments", namespace, undo)


def _boolean(value: str) -> bool:
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def add_extra_server_args(parser: argparse.ArgumentParser) -> None:
    """Add registered flags and the flat JSON transport to a parser."""
    parser.add_argument(
        "--extra-server-args",
        type=json.loads,
        default=argparse.SUPPRESS,
        help="JSON options registered by installed runtime plugins.",
    )
    for namespace, schema in _SCHEMAS.items():
        group = parser.add_argument_group(f"{namespace} server arguments")
        hints = get_type_hints(schema.config_type)
        for field in _argument_fields(schema.config_type):
            scalar, choices = _scalar_type(hints[field.name])
            group.add_argument(
                f"--{namespace.replace('_', '-')}-{field.name.replace('_', '-')}",
                dest=f"extra_server_args.{field.name}",
                type=_boolean if scalar is bool else scalar,
                choices=choices,
                default=argparse.SUPPRESS,
                help=field.metadata.get("help"),
            )


def collect_extra_server_args(args: argparse.Namespace) -> dict:
    """Combine JSON and individual flags, rejecting contradictory values."""
    raw = vars(args).get("extra_server_args", {})
    if not isinstance(raw, dict):
        raise TypeError("extra_server_args must be a JSON object")
    result = copy.deepcopy(raw)
    for key, value in vars(args).items():
        if not key.startswith("extra_server_args."):
            continue
        _, name = key.split(".")
        if name in result and (
            type(result[name]) is not type(value) or result[name] != value
        ):
            raise ValueError(f"Conflicting extra server argument {name}")
        result[name] = value
    return normalize_extra_server_args(result)


def normalize_extra_server_args(values: dict) -> dict:
    """Validate and copy plain values for CLI, Python and deserialized launches."""
    if not isinstance(values, dict):
        raise TypeError("extra_server_args must be a JSON object")
    if any(not isinstance(name, str) for name in values):
        raise TypeError("extra_server_args field names must be strings")
    result = copy.deepcopy(values)
    known = {
        field.name
        for schema in _SCHEMAS.values()
        for field in _argument_fields(schema.config_type)
    }
    unknown = result.keys() - known
    if unknown:
        raise ValueError(f"Unknown extra server arguments: {sorted(unknown)}")
    for namespace, schema in _SCHEMAS.items():
        hints = get_type_hints(schema.config_type)
        names = {field.name for field in _argument_fields(schema.config_type)}
        options = {key: value for key, value in result.items() if key in names}
        if not options:
            continue
        for name, value in options.items():
            annotation = hints[name]
            nullable = type(None) in get_args(annotation)
            scalar, choices = _scalar_type(annotation)
            if value is None and nullable:
                continue
            if (
                type(value) is not scalar
                or choices is not None
                and value not in choices
            ):
                raise ValueError(f"Invalid {name}: {value!r}; expected {annotation}")
            if scalar is float and not math.isfinite(value):
                raise ValueError(f"Invalid {name}: expected a finite float")
        schema.config_type(**options)
    return result


def validate_extra_server_args(server_args: ServerArgs) -> None:
    """Run cross-argument checks after plugin discovery, before numerics resolve."""
    server_args.extra_server_args = normalize_extra_server_args(
        server_args.extra_server_args
    )
    for schema in tuple(_SCHEMAS.values()):
        if schema.validate is not None:
            schema.validate(server_args)


def get_extra_server_args(server_args: ServerArgs, namespace: str, config_type: type):
    """Return one immutable configuration per resolved launch and plugin.

    Numerical resolvers can inspect provisional values before ServerArgs finishes
    initialization. Published values are shared by every model in that launch.
    """
    schema = _SCHEMAS.get(namespace)
    if schema is None or schema.config_type is not config_type:
        raise ValueError(f"Unregistered extra server argument schema {namespace!r}")
    names = {field.name for field in _argument_fields(config_type)}
    values = {
        key: value
        for key, value in server_args.extra_server_args.items()
        if key in names
    }
    values.update(
        {
            field.name: getattr(server_args, field.name)
            for field in dataclasses.fields(config_type)
            if field.metadata.get("native_server_arg", False)
        }
    )
    if not server_args._extra_server_args_ready:
        return config_type(**values)
    cached = server_args._extra_server_args_cache.get(namespace)
    if cached is not None:
        previous_type, previous_values, options = cached
        if previous_type is not config_type or previous_values != values:
            raise ValueError(
                f"Extra server arguments {namespace!r} changed after publication; create a new ServerArgs"
            )
        return options
    options = config_type(**values)
    server_args._extra_server_args_cache[namespace] = (config_type, values, options)
    return options
