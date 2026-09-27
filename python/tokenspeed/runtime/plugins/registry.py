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

"""The registration surface of runtime plugins.

These functions are a plugin's whole import surface for registration. Each
writes one registry whose initial content is the in-tree table it replaced,
so in-tree and plugin entries resolve through the same lookup. A name that is
already registered raises unless the caller passes ``override=True``; an
override is logged, so replacing an in-tree entry is always a visible
decision in the plugin's source.

Registrations made while :func:`tokenspeed.runtime.plugins.ensure_loaded`
runs a plugin's ``register()`` are recorded under that plugin; if
``register()`` raises, they are undone so a failed plugin leaves no partial
state behind.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig

    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend

logger = get_colorful_logger(__name__)


def register_drafter(
    algorithm: str,
    drafter_cls: type,
    *,
    model_cls: type | None = None,
    defaults_to_base_checkpoint: bool = False,
    override: bool = False,
) -> None:
    """Register a speculative-decoding drafter; see the drafter registry.

    Thin re-export of
    :func:`tokenspeed.runtime.execution.drafter.register_drafter`, imported
    lazily so registration stays cheap in lightweight contexts.
    """
    from tokenspeed.runtime.execution.drafter import (
        register_drafter as _register_drafter,
    )

    _register_drafter(
        algorithm,
        drafter_cls,
        model_cls=model_cls,
        defaults_to_base_checkpoint=defaults_to_base_checkpoint,
        override=override,
    )


__all__ = [
    "register_drafter",
    "ModelProfile",
    "RegisteredModel",
    "register_attention_backend",
    "register_cache_pool",
    "register_cache_recipe",
    "register_config",
    "register_linear_attention_backend",
    "register_model",
    "registered_architectures",
    "registered_model",
    "resolve_model_profile",
]


@dataclass(frozen=True)
class RegisteredModel:
    """A model class registered by a plugin under one architecture name.

    Attributes:
        cls: The model entry class; it defines ``model_profile(hf_config)``.
        override: Whether the registration may replace an in-tree class of
            the same architecture name.
    """

    cls: type[nn.Module]
    override: bool


@dataclass
class _Recording:
    """Registrations made by one plugin's ``register()``."""

    names: dict[str, list[str]] = field(default_factory=dict)
    undo: list[Callable[[], None]] = field(default_factory=list)


# Plugin model classes, keyed by architecture name. In-tree classes stay in
# ``tokenspeed.runtime.models.registry``, which consults this table first;
# keeping it separate lets every process resolve a profile without importing
# every in-tree model module.
_MODELS: dict[str, RegisteredModel] = {}
_MISSING = object()
_active: _Recording | None = None


@contextlib.contextmanager
def recording() -> Iterator[_Recording]:
    """Collect the registrations of one plugin ``register()`` call.

    Used by the plugin loader only. On an exception inside the block, every
    registration made in it is undone before the exception propagates.
    """
    global _active
    if _active is not None:
        raise RuntimeError("plugin registrations cannot be recorded re-entrantly")
    _active = _Recording()
    record = _active
    try:
        yield record
    except BaseException:
        for undo in reversed(record.undo):
            undo()
        raise
    finally:
        _active = None


def _put(table: dict, key: Any, value: Any, *, kind: str, override: bool) -> None:
    previous = table.get(key, _MISSING)
    if previous is not _MISSING:
        if not override:
            raise ValueError(
                f"{kind} {key!r} is already registered; pass override=True to "
                "replace it"
            )
        logger.warning(f"Plugin registration overrides {kind} {key!r}")
    table[key] = value

    def undo() -> None:
        if previous is _MISSING:
            del table[key]
        else:
            table[key] = previous

    if _active is not None:
        _active.names.setdefault(kind, []).append(str(key))
        _active.undo.append(undo)


def record_external(kind: str, name: str, undo) -> None:
    """Let a registry outside this module join a plugin recording.

    Args:
        kind: Registration kind for the loader's per-plugin summary line.
        name: The registered name, for the same summary.
        undo: Zero-argument callable reverting the registration; run (in
            reverse order) when the plugin's ``register()`` raises.
    """
    if _active is not None:
        _active.names.setdefault(kind, []).append(str(name))
        _active.undo.append(undo)


def register_model(
    cls: type[nn.Module],
    *,
    architectures: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register a model entry class.

    Args:
        cls: The entry class. It must define a ``model_profile(hf_config)``
            classmethod returning a :class:`ModelProfile`.
        architectures: HF ``architectures`` names the class serves; empty
            means the class name.
        override: Allow replacing a class already registered, in tree or by
            another plugin, under one of these names. A collision with an
            in-tree class is detected when the model loader first resolves
            the architecture.

    Raises:
        TypeError: ``cls`` defines no ``model_profile`` classmethod.
        ValueError: A name is already registered and ``override`` is False.
    """
    if not callable(getattr(cls, "model_profile", None)):
        raise TypeError(
            f"{cls.__name__} must define a model_profile(hf_config) classmethod "
            "returning a ModelProfile"
        )
    for name in architectures or (cls.__name__,):
        _put(
            _MODELS,
            name,
            RegisteredModel(cls=cls, override=override),
            kind="model",
            override=override,
        )


def registered_model(architecture: str) -> RegisteredModel | None:
    """Return the plugin registration for ``architecture``, if any."""
    return _MODELS.get(architecture)


def registered_architectures() -> frozenset[str]:
    """Return every architecture name registered by plugins."""
    return frozenset(_MODELS)


def resolve_model_profile(
    architectures: Iterable[str], hf_config: PretrainedConfig
) -> ModelProfile | None:
    """Return the profile of the first registered architecture, if any.

    Args:
        architectures: Candidate architecture names in resolution order.
        hf_config: The checkpoint config handed to ``model_profile``.

    Returns:
        The class's profile, or None when no candidate is registered by a
        plugin (in-tree models still resolve through the architecture tables).
    """
    for architecture in architectures:
        registered = _MODELS.get(architecture)
        if registered is None:
            continue
        profile = registered.cls.model_profile(hf_config)
        if not isinstance(profile, ModelProfile):
            raise TypeError(
                f"{registered.cls.__name__}.model_profile returned "
                f"{type(profile).__name__}, not ModelProfile"
            )
        return profile
    return None


def register_config(
    cls: type[PretrainedConfig],
    *,
    model_type: str | None,
    architectures: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register an HF config class for checkpoints of a plugin model.

    ``get_config`` resolves a checkpoint's ``model_type`` first and then its
    first ``architectures`` entry, so a checkpoint whose ``config.json``
    carries no ``model_type`` is still parsed by its own class.

    Args:
        cls: The ``PretrainedConfig`` subclass.
        model_type: ``config.json`` ``model_type`` the class parses, or None
            when checkpoints are identified by architecture only.
        architectures: Architecture names the class parses.
        override: Allow replacing an existing registration.

    Raises:
        ValueError: Nothing to key the class by, or a key is already
            registered and ``override`` is False.
    """
    from transformers import AutoConfig

    from tokenspeed.runtime.utils import hf_transformers_utils

    if model_type is None and not architectures:
        raise ValueError("register_config needs a model_type or architectures")
    if model_type is not None:
        _put(
            hf_transformers_utils._CONFIG_REGISTRY,
            model_type,
            cls,
            kind="config model_type",
            override=override,
        )
        # Mirrors the in-tree table: Transformers' own lookups (e.g. from
        # AutoTokenizer) resolve the type too. Best effort, as in tree — but
        # a failed plugin must not leave its config class resolvable through
        # AutoConfig, so the global mutation joins the recording's rollback.
        with contextlib.suppress(ValueError):
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            extra = CONFIG_MAPPING._extra_content
            prior = extra.get(model_type, _MISSING)
            AutoConfig.register(model_type, cls)

            def undo_autoconfig(
                model_type: str = model_type, prior: Any = prior
            ) -> None:
                if prior is _MISSING:
                    extra.pop(model_type, None)
                else:
                    extra[model_type] = prior

            record_external("AutoConfig model_type", model_type, undo_autoconfig)
    for architecture in architectures:
        _put(
            hf_transformers_utils._ARCHITECTURE_CONFIG_REGISTRY,
            architecture,
            cls,
            kind="config architecture",
            override=override,
        )


def register_attention_backend(
    name: str,
    archs: set[AttentionArch],
    cls: type[AttentionBackend],
    *,
    override: bool = False,
) -> None:
    """Register a full-attention backend selectable by ``--attention-backend``.

    Args:
        name: Backend name.
        archs: Attention architectures the backend serves.
        cls: The backend class.
        override: Allow replacing an existing backend of this name.
    """
    # Built-in backends register when this package imports; load them before
    # the plugin's entry, so replacing a built-in name collides (or, with
    # override=True, wins) here instead of being silently overwritten by the
    # built-in import later in startup.
    import tokenspeed.runtime.layers.attention.backends  # noqa: F401
    from tokenspeed.runtime.layers.attention import registry as attention_registry

    _put(
        attention_registry._BACKEND_REGISTRY,
        name,
        (set(archs), cls),
        kind="attention backend",
        override=override,
    )


def register_linear_attention_backend(
    name: str,
    factory: Callable[..., AttentionBackend],
    *,
    override: bool = False,
) -> None:
    """Register the backend for a model's linear-attention layers.

    A profile's ``linear_attention`` names one of these. The hybrid wrapper
    routes the model's linear layers to the backend the factory builds.

    Args:
        name: Linear-attention backend name.
        factory: ``factory(server_args, config) -> AttentionBackend``, where
            ``config`` is the model's ``AttnConfig``.
        override: Allow replacing an existing backend of this name.
    """
    from tokenspeed.runtime.layers.attention import registry as attention_registry

    _put(
        attention_registry._LINEAR_ATTN_BACKENDS,
        name,
        factory,
        kind="linear attention backend",
        override=override,
    )


def register_cache_recipe(
    family: str,
    recipe: Callable[..., Any],
    *,
    override: bool = False,
) -> None:
    """Register the cache recipe of a cache family.

    Args:
        family: Cache family name a profile's ``cache_family`` refers to.
        recipe: A ``CacheRecipe`` subclass (or factory) accepting the recipe
            keyword arguments; a subclass must declare ``family == family``.
        override: Allow replacing an existing recipe of this family.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes import setup

    if isinstance(recipe, type):
        try:
            declared = recipe.family
        except AttributeError:
            raise ValueError(
                f"recipe {recipe.__name__} must declare family = {family!r}"
            ) from None
    else:
        declared = family
    if declared != family:
        raise ValueError(
            f"recipe {recipe.__name__} declares family {declared!r}, "
            f"registered as {family!r}"
        )
    _put(setup._RECIPES, family, recipe, kind="cache recipe", override=override)


def register_cache_pool(
    family: str,
    factory: Callable[..., Any],
    *,
    override: bool = False,
) -> None:
    """Register the cache pool factory of a cache family.

    Args:
        family: Cache family name.
        factory: ``factory(spec, config, arena, *, num_layers, rank,
            field_layer_offset) -> CachePool``.
        override: Allow replacing an existing factory of this family.
    """
    from tokenspeed.runtime.layers.attention.kv_cache import factory as pool_factory

    _put(
        pool_factory._POOL_FACTORIES,
        family,
        factory,
        kind="cache pool",
        override=override,
    )
