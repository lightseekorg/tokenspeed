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

import copy
import json
import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import torch

logger = logging.getLogger(__name__)

_ConfigT = TypeVar("_ConfigT", bound=type)


def _config_dataclass(cls: _ConfigT) -> _ConfigT:
    """Turn a config into a keyword-only dataclass that preserves unknown fields.

    Standard dataclass constructors reject checkpoint extension fields and
    legacy aliases. Config files need to retain those for forward/backward
    compatibility, so the generated constructor separates declared fields from
    extensions and passes the latter to ``__post_init__``.

    Applied automatically by :meth:`BaseConfig.__init_subclass__` to configs
    that declare their fields as class-level annotations and define no custom
    ``__init__``; no explicit ``@_config_dataclass`` decorator is required.
    """
    cls = dataclass(cls, kw_only=True, repr=False, eq=False)
    generated_init = cls.__init__
    config_fields = fields(cls)

    @wraps(generated_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            raise TypeError(
                f"{type(self).__name__} accepts keyword arguments only, "
                f"got {len(args)} positional argument(s)"
            )

        for item in config_fields:
            if item.init and item.name in kwargs:
                value = kwargs.pop(item.name)
            elif item.default is not MISSING:
                value = item.default
            elif item.default_factory is not MISSING:
                value = item.default_factory()
            else:
                raise TypeError(f"Missing required config field: {item.name!r}")
            setattr(self, item.name, value)

        self.__post_init__(**kwargs)

    cls.__init__ = __init__
    return cls


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseConfig):
        return value.to_dict()
    if callable(to_dict := getattr(value, "to_dict", None)):
        return _serialize(to_dict())
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, set):
        return sorted(_serialize(item) for item in value)
    return copy.deepcopy(value)


def _normalize_dtype(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    dtype = getattr(torch, value.removeprefix("torch."), None)
    return dtype if isinstance(dtype, torch.dtype) else value


def _normalize_architectures(value: Any) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        value = [value]
    elif isinstance(value, tuple):
        value = list(value)
    elif not isinstance(value, list):
        raise TypeError(
            "`architectures` must be a string, list of strings, or None, "
            f"got {type(value).__name__}"
        )

    if not all(isinstance(arch, str) and arch for arch in value):
        raise ValueError(
            "`architectures` entries must be non-empty strings, " f"got {value!r}"
        )

    return value


def _is_classvar(annotation: Any) -> bool:
    # Under ``from __future__ import annotations`` the annotation is a string.
    if isinstance(annotation, str):
        return annotation.startswith("ClassVar") or annotation.startswith(
            "typing.ClassVar"
        )
    return getattr(annotation, "__origin__", None) is ClassVar


@lru_cache(maxsize=None)
def _classvar_names(cls: type) -> frozenset[str]:
    """Names annotated ``ClassVar`` anywhere in ``cls``'s MRO."""
    names: set[str] = set()
    for klass in cls.__mro__:
        for name, annotation in getattr(klass, "__annotations__", {}).items():
            if _is_classvar(annotation):
                names.add(name)
    return frozenset(names)


def get_rope_parameters(config):
    """Return the config's standardized ``rope_parameters`` dict.

    MRoPE extensions such as ``mrope_section`` live in the public
    ``rope_parameters`` field (validators skip them via
    ``ignore_keys_at_rope_validation``), so no private field is involved.
    """
    return getattr(config, "rope_parameters", None) or {}


def get_rope_theta(config, default=10000.0):
    """Return the RoPE base theta, nested ``rope_parameters`` first.

    ``rope_parameters["rope_theta"]`` wins when present; otherwise fall back to
    the top-level ``rope_theta`` field, then ``default`` if neither is set.
    """
    theta = get_rope_parameters(config).get("rope_theta")
    if theta is not None:
        return theta
    return getattr(config, "rope_theta", None) or default


class RotaryConfigMixin:
    """Standardize and validate the RoPE fields consumed by TokenSpeed."""

    default_theta = 10_000.0
    ignore_keys_at_rope_validation = set()

    @property
    def rope_scaling(self) -> dict[str, Any] | None:
        return get_rope_parameters(self) or None

    def convert_rope_params_to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Normalize legacy top-level RoPE keyword arguments.

        Args:
            **kwargs: Remaining checkpoint configuration fields.

        Returns:
            Fields not consumed during RoPE normalization.
        """
        rope_scaling = kwargs.pop("rope_scaling", MISSING)
        current = getattr(self, "rope_parameters", None)
        if rope_scaling is MISSING:
            self.rope_parameters = current or {}
        else:
            # Presence, rather than truthiness, determines override priority:
            # explicit ``None`` and ``{}`` both disable checkpoint scaling.
            self.rope_parameters = rope_scaling or {}

        # Mirror transformers' RotaryEmbeddingConfigMixin: when the config
        # declares neither a ``rope_theta`` field nor any ``rope_parameters``
        # entries, the model's ``default_theta`` still supplies the base
        # (MiniMax-M3, LongCat, and Qwen3-Omni rely on this). Inject it before
        # standardization so an empty ``rope_parameters`` does not trip the
        # no-RoPE early return in ``standardize_rope_params`` below.
        if not self.rope_parameters and getattr(self, "rope_theta", None) is None:
            self.rope_parameters["rope_theta"] = self.default_theta

        # Determine whether the dictionary is global or keyed by layer type
        # before adding common defaults. ``standardize_rope_params`` applies
        # ``rope_theta`` and ``partial_rotary_factor`` to the correct level.
        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self) -> None:
        """Normalize legacy/global RoPE dictionaries to ``rope_parameters``.

        YaRN deliberately preserves a missing
        ``original_max_position_embeddings`` field. Its absence means
        ``max_position_embeddings`` is the pre-extension length, so the usable
        context is ``factor`` times it; its presence means
        ``max_position_embeddings`` is already the extended length. Defaulting
        the field here would erase that distinction and make
        :func:`~tokenspeed.runtime.configs.utils.get_context_length` silently
        divide the derived context length by ``factor``. The rotary embedding
        consumer supplies the concrete fallback where it is actually needed.

        Llama 3 and LongRoPE use this field as their actual pre-extension
        training length. They therefore require an explicitly supplied value:
        inferring it from ``max_position_embeddings`` would silently construct
        incorrect rotary frequencies. An explicitly declared top-level value
        is still copied into the normalized dictionary below.
        """
        rope_theta = getattr(self, "rope_theta", None)
        partial = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        layer_types = getattr(self, "layer_types", None)

        if not (rope_parameters or rope_theta):
            return

        is_layered = bool(
            layer_types
            and rope_parameters
            and set(rope_parameters).issubset(set(layer_types))
        )
        parameter_sets = rope_parameters.values() if is_layered else (rope_parameters,)
        for parameters in parameter_sets:
            parameters.setdefault("rope_type", parameters.get("type", "default"))
            parameters.setdefault(
                "rope_theta",
                rope_theta if rope_theta is not None else self.default_theta,
            )
            if partial is not None:
                parameters["partial_rotary_factor"] = partial
            rope_type = parameters["rope_type"]
            if rope_type in {"llama3", "longrope", "yarn"}:
                # Do not infer an original context length from the extended
                # maximum. For YaRN, its absence informs context-length
                # derivation; for Llama 3 and LongRoPE, it is required by the
                # rotary algorithm and validation must reject its absence.
                original_max = getattr(self, "original_max_position_embeddings", None)
                if original_max is not None:
                    parameters.setdefault(
                        "original_max_position_embeddings", original_max
                    )
        self.rope_parameters = rope_parameters

    def validate_rope(self) -> None:
        """Validate standardized RoPE dictionaries."""
        rope_parameters = getattr(self, "rope_parameters", None)
        if not rope_parameters:
            return

        layer_types = getattr(self, "layer_types", None)
        is_layered = bool(
            layer_types and set(rope_parameters).issubset(set(layer_types))
        )
        parameter_sets = rope_parameters.values() if is_layered else (rope_parameters,)
        for parameters in parameter_sets:
            rope_type = parameters.get("rope_type", parameters.get("type", "default"))
            parameters["rope_type"] = rope_type
            validator = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            if validator is None:
                logger.warning("No RoPE validator is registered for %r", rope_type)
                continue
            validator(parameters, self.ignore_keys_at_rope_validation)

    @staticmethod
    def _check_rope_keys(
        parameters: dict[str, Any],
        required: set[str],
        optional: set[str] | None = None,
        ignore_keys: set[str] | None = None,
    ) -> None:
        received = set(parameters)
        if "type" in received:
            received.remove("type")
        optional = set(optional or ()) | {"partial_rotary_factor"}
        received -= set(ignore_keys or ())
        missing = required - received
        if missing:
            raise KeyError(
                "Missing required keys in `rope_parameters` for "
                f"rope_type={parameters.get('rope_type')!r}: {missing}"
            )
        unused = received - required - optional
        if unused:
            logger.warning(
                "Unrecognized keys in `rope_parameters` for rope_type=%r: %s",
                parameters.get("rope_type"),
                unused,
            )

    def _validate_default_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        self._check_rope_keys(parameters, {"rope_type"}, {"rope_theta"}, ignore_keys)

    def _validate_linear_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        self._validate_factor_rope(parameters, ignore_keys)

    def _validate_dynamic_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        self._validate_factor_rope(parameters, ignore_keys)

    def _validate_factor_rope(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None
    ) -> None:
        self._check_rope_keys(
            parameters,
            {"rope_type", "factor"},
            {"rope_theta"},
            ignore_keys,
        )
        factor = parameters["factor"]
        if not isinstance(factor, (int, float)) or factor < 1:
            logger.warning("RoPE factor must be a number >= 1, got %r", factor)

    def _validate_yarn_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        # ``original_max_position_embeddings`` is optional for YaRN. When it is
        # omitted, ``max_position_embeddings`` is the pre-extension length and
        # the rotary embedding consumer falls back to it.
        self._check_rope_keys(
            parameters,
            {"rope_type", "factor"},
            {
                "rope_theta",
                "original_max_position_embeddings",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
                "truncate",
            },
            ignore_keys,
        )
        factor = parameters["factor"]
        if not isinstance(factor, (int, float)) or factor < 1:
            logger.warning("RoPE factor must be a number >= 1, got %r", factor)
        beta_fast = parameters.get("beta_fast")
        beta_slow = parameters.get("beta_slow")
        for name, value in (("beta_fast", beta_fast), ("beta_slow", beta_slow)):
            if value is not None and not isinstance(value, (int, float)):
                logger.warning("RoPE %s must be a number, got %r", name, value)
        if (beta_fast or 32) < (beta_slow or 1):
            logger.warning(
                "RoPE beta_fast must be greater than beta_slow, got "
                "beta_fast=%s beta_slow=%s",
                beta_fast,
                beta_slow,
            )
        attn_factor = parameters.get("attn_factor")
        if attn_factor is not None and (
            not isinstance(attn_factor, (int, float)) or attn_factor < 0
        ):
            logger.warning(
                "RoPE attn_factor must be a number >= 0, got %r",
                attn_factor,
            )

    def _validate_longrope_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        self._check_rope_keys(
            parameters,
            {
                "rope_type",
                "short_factor",
                "long_factor",
                "original_max_position_embeddings",
            },
            {"rope_theta", "attn_factor", "factor"},
            ignore_keys,
        )
        partial = parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(
            self,
            "head_dim",
            self.hidden_size // self.num_attention_heads,
        )
        expected = int(head_dim * partial) // 2
        for name in ("short_factor", "long_factor"):
            values = parameters[name]
            if not isinstance(values, list) or not all(
                isinstance(value, (int, float)) for value in values
            ):
                logger.warning("LongRoPE %s must be a list of numbers", name)
            elif len(values) != expected:
                logger.warning(
                    "LongRoPE %s must have length %s, got %s",
                    name,
                    expected,
                    len(values),
                )
        factor = parameters.get("factor")
        if factor is not None and (not isinstance(factor, (int, float)) or factor < 1):
            logger.warning("RoPE factor must be a number >= 1, got %r", factor)
        attn_factor = parameters.get("attn_factor")
        if attn_factor is not None and (
            not isinstance(attn_factor, (int, float)) or attn_factor < 0
        ):
            logger.warning(
                "RoPE attn_factor must be a number >= 0, got %r",
                attn_factor,
            )

    def _validate_llama3_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        self._check_rope_keys(
            parameters,
            {
                "rope_type",
                "factor",
                "original_max_position_embeddings",
                "low_freq_factor",
                "high_freq_factor",
                "rope_theta",
            },
            ignore_keys=ignore_keys,
        )
        factor = parameters["factor"]
        if not isinstance(factor, (int, float)) or factor < 1:
            logger.warning("RoPE factor must be a number >= 1, got %r", factor)
        low = parameters["low_freq_factor"]
        high = parameters["high_freq_factor"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            logger.warning("Llama3 frequency factors must be numeric")
        elif high <= low:
            logger.warning("Llama3 high_freq_factor must exceed low_freq_factor")
        original_max = parameters.get("original_max_position_embeddings")
        if original_max is not None and not isinstance(original_max, int):
            logger.warning(
                "RoPE original_max_position_embeddings must be an integer, got %r",
                original_max,
            )
        max_position = getattr(self, "max_position_embeddings", None)
        if (
            original_max is not None
            and max_position is not None
            and original_max >= max_position
        ):
            logger.warning(
                "RoPE original_max_position_embeddings must be less than "
                "max_position_embeddings, got %s and %s",
                original_max,
                max_position,
            )

    def _validate_proportional_rope_parameters(
        self, parameters: dict[str, Any], ignore_keys: set[str] | None = None
    ) -> None:
        self._check_rope_keys(
            parameters,
            {"rope_type", "rope_theta"},
            {"factor"},
            ignore_keys,
        )


_ALLOWED_LAYER_TYPES = frozenset(
    {
        "attention",
        "chunked_attention",
        "compressed_sparse_attention",
        "conv",
        "deepseek_sparse_attention",
        "dense",
        "full_attention",
        "hash_moe",
        "heavily_compressed_attention",
        "hybrid",
        "linear_attention",
        "mamba",
        "minimax_m3_sparse",
        "moe",
        "sliding_attention",
        "sparse",
        "swa_attention",
    }
)


@dataclass(kw_only=True, repr=False, eq=False)
class BaseConfig(RotaryConfigMixin):
    """Minimal runtime configuration contract for TokenSpeed-owned models."""

    model_type: ClassVar[str] = ""
    base_config_key: ClassVar[str] = ""
    sub_configs: ClassVar[dict[str, type["BaseConfig"]]] = {}
    attribute_map: ClassVar[dict[str, str]] = {}
    keys_to_ignore_at_inference: ClassVar[list[str]] = []
    base_model_tp_plan: ClassVar[dict[str, Any] | None] = None
    base_model_pp_plan: ClassVar[dict[str, Any] | None] = None
    base_model_ep_plan: ClassVar[dict[str, Any] | None] = None

    architectures: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    dtype: str | torch.dtype | None = None
    torch_dtype: str | torch.dtype | None = None
    _name_or_path: str = field(default="", init=False, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Make field-annotated configs into kw-only dataclasses automatically.

        Configs that declare their fields as class-level annotations (e.g.
        ``DeepseekV3Config``) are turned into keyword-only dataclasses and
        wrapped so unknown/legacy checkpoint fields are routed to
        ``__post_init__``. Configs with a custom ``__init__`` keep it, and
        subclasses that merely rename a parent (no annotations, no ``__init__``)
        inherit the parent's constructor untouched.
        """
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__ or not cls.__dict__.get("__annotations__"):
            return
        _config_dataclass(cls)

    def __init__(
        self,
        *,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        dtype: str | torch.dtype | None = None,
        torch_dtype: str | torch.dtype | None = None,
        name_or_path: str = "",
        **kwargs: Any,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.dtype = dtype
        self.torch_dtype = torch_dtype
        self._name_or_path = str(name_or_path)

        # Existing model configs still call this handwritten constructor. New
        # dataclass configs reach the same finalization through ``__post_init__``.
        BaseConfig.__post_init__(self, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        """Finalize common fields after the complete config has been assigned."""
        # Keep the legacy alias out of the generic setattr loop. Some configs
        # expose ``rope_scaling`` as a computed, read-only runtime view, while
        # checkpoint loading still needs to accept it as an input alias.
        legacy_rope_scaling = kwargs.pop("rope_scaling", MISSING)

        self.dtype = _normalize_dtype(
            self.dtype if self.dtype is not None else self.torch_dtype
        )
        # ``torch_dtype`` is an input alias, not serialized runtime state.
        self.__dict__.pop("torch_dtype", None)

        if "name_or_path" in kwargs:
            self._name_or_path = str(kwargs.pop("name_or_path"))

        classvar_names = _classvar_names(type(self))
        for key, value in kwargs.items():
            if key in classvar_names:
                continue
            setattr(self, key, value)

        self.architectures = _normalize_architectures(self.architectures)

        # Normalize RoPE after applying the remaining checkpoint fields. RoPE
        # standardization can depend on those fields, for example
        # ``original_max_position_embeddings`` or a ``layer_types`` property
        # backed by ``full_attention_interval``.
        #
        # Inspect only fields owned by this config (``self.__dict__``), not
        # attributes forwarded through ``__getattr__``: composite configs
        # forward missing attributes to their nested text config, and treating
        # those as outer fields would duplicate/mutate the nested RoPE
        # dictionary on the wrapper itself.
        if (
            "rope_parameters" in self.__dict__
            or self.__dict__.get("rope_theta") is not None
            or legacy_rope_scaling is not MISSING
        ):
            # A checkpoint may ship canonical ``rope_parameters`` alongside a
            # stale legacy ``rope_scaling`` (often ``null``). The canonical
            # field is authoritative during construction; only the override
            # path (``update()``) treats an explicit ``rope_scaling=None/{}``
            # as a request to clear checkpoint scaling.
            if legacy_rope_scaling is not MISSING and self.__dict__.get(
                "rope_parameters"
            ):
                self.convert_rope_params_to_dict()
            else:
                self.convert_rope_params_to_dict(rope_scaling=legacy_rope_scaling)
            self.validate_rope()

        self.validate_layer_type()
        self.validate_token_ids()

    def __setattr__(self, name: str, value: Any) -> None:
        mapped_name = type(self).attribute_map.get(name, name)
        object.__setattr__(self, mapped_name, value)

    def __getattribute__(self, name: str) -> Any:
        if name not in {"attribute_map", "__class__"}:
            name = type(self).attribute_map.get(name, name)
        return object.__getattribute__(self, name)

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value: str) -> None:
        self._name_or_path = str(value)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "BaseConfig":
        """Construct a config from checkpoint fields.

        Args:
            config_dict: Serialized checkpoint configuration.
            **kwargs: Values that override serialized fields.

        Returns:
            An initialized config instance.
        """
        values = copy.deepcopy(config_dict)
        values.update(kwargs)
        return cls(**values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "BaseConfig":
        """Load a config from a local path or Hugging Face repository.

        Args:
            pretrained_model_name_or_path: Directory, JSON file, or repository ID.
            revision: Optional repository revision for remote loading.
            **kwargs: Values that override serialized fields.

        Returns:
            An initialized config instance.
        """
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            config_path = path / "config.json"
        elif path.is_file():
            config_path = path
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(
                    str(pretrained_model_name_or_path),
                    "config.json",
                    revision=revision,
                )
            )

        with config_path.open(encoding="utf-8") as file:
            values = json.load(file)
        if cls.base_config_key and cls.base_config_key in values:
            values = values[cls.base_config_key]
        values.update(kwargs)
        config = cls.from_dict(values)
        config.name_or_path = str(pretrained_model_name_or_path)
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively serialized checkpoint dictionary.

        Private (``_``-prefixed) attributes such as ``_name_or_path`` are
        runtime-only state, not checkpoint fields, so they are left out.
        ``torch_dtype`` is an input alias already normalized into ``dtype``.
        """
        output = {
            key: _serialize(value)
            for key, value in self.__dict__.items()
            if key != "torch_dtype" and not key.startswith("_")
        }
        output["model_type"] = self.model_type
        return output

    def to_json_string(self) -> str:
        """Return the serialized config as formatted JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike[str]) -> None:
        """Serialize the config to ``json_file_path``."""
        Path(json_file_path).write_text(self.to_json_string(), encoding="utf-8")

    def get_text_config(self, *args: Any, **kwargs: Any) -> "BaseConfig":
        """Return a nested text config, or this config for text-only models."""
        return getattr(self, "text_config", self)

    def validate_architecture(self) -> None:
        """Hook for model-specific dimension validation."""

    def validate_layer_type(self) -> None:
        """Validate attention and MLP dispatch lists when present."""
        for field in ("layer_types", "mlp_layer_types"):
            layers = getattr(self, field, None)
            if layers is None:
                continue
            invalid = [layer for layer in layers if layer not in _ALLOWED_LAYER_TYPES]
            if invalid:
                raise ValueError(
                    f"The `{field}` entries must be in {_ALLOWED_LAYER_TYPES}, "
                    f"got {invalid}"
                )
            num_hidden_layers = getattr(self, "num_hidden_layers", None)
            if num_hidden_layers is not None and len(layers) != num_hidden_layers:
                raise ValueError(
                    f"`num_hidden_layers` ({num_hidden_layers}) must equal the "
                    f"number of `{field}` entries ({len(layers)})"
                )

    def validate_token_ids(self) -> None:
        """Warn when special token ids fall outside the vocabulary range."""
        text_config = self.get_text_config()
        vocab_size = getattr(text_config, "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            return
        for name in text_config:
            if not name.endswith("_token_id"):
                continue
            value = getattr(text_config, name, None)
            token_ids = value if isinstance(value, (list, tuple)) else [value]
            for token_id in token_ids:
                if isinstance(token_id, int) and not 0 <= token_id < vocab_size:
                    logger.warning(
                        "Model config: %s must be None or an integer within the "
                        "vocabulary [0, %d), got %r.",
                        name,
                        vocab_size,
                        token_id,
                    )

    def __iter__(self):
        yield from self.__dict__

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BaseConfig) and self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"


class TextConfigBase(BaseConfig):
    """Common base for text-decoder backbone configurations.

    Resolves grouped-query key/value heads and the per-head attention
    dimension, mirroring how transformers resolves those in each text
    decoder's ``__post_init__``. Kept out of ``BaseConfig`` so vision and
    audio encoders that declare ``hidden_size``/``num_attention_heads`` for
    their own towers don't inherit decoder-only attention geometry.
    """

    def __post_init__(self, **kwargs: Any) -> None:
        # Resolve text-decoder attention defaults (GQA key/value heads and the
        # per-head dimension). Reads ``self.__dict__`` rather than ``getattr`` so
        # composite configs that forward missing attributes to a nested text
        # config (via ``__getattr__``) and audio encoders that alias these names
        # through ``attribute_map`` are both left untouched.
        num_attention_heads = self.__dict__.get("num_attention_heads")
        num_key_value_heads = self.__dict__.get("num_key_value_heads")
        if num_attention_heads is not None and num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        head_dim = self.__dict__.get("head_dim")
        hidden_size = self.__dict__.get("hidden_size")
        if head_dim is None and hidden_size is not None and num_attention_heads:
            self.head_dim = hidden_size // num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["BaseConfig", "TextConfigBase"]
