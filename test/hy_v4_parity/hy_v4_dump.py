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

"""Dump HY V4 layer activations from an engine, for cross-engine comparison.

Run inside the process that owns the model, or inject the companion
``sitecustomize.py`` into an engine's subprocesses so this module attaches
automatically.

Two ways to use it:

* **sitecustomize injection** (works for both engines, no patches to either).
  Copy ``hy_v4_dump_sitecustomize.py`` onto ``PYTHONPATH`` and set
  ``HY_V4_DUMP_PATH``; every engine subprocess then attaches a dumper to its
  model as soon as the loader finishes.
* **direct** (in-process, e.g. a unit test) by calling ``dump_layers`` as a
  context manager around the forward pass.

The dump records sub-block inputs and outputs, so a divergence is reported at
the first sub-block that produced it: an iHC boundary bug shows up at
``hc_attn``/``hc_mlp``, a sink or gate bug at ``attn``, an expert bug at ``mlp``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from test.hy_v4_parity.hy_v4_parity_hook import (
    LayerDumper,
    ModelDumpSession,
    first_divergence,
    format_report,
)

import torch

DUMP_PATH_ENV = "HY_V4_DUMP_PATH"
DUMP_LAYERS_ENV = "HY_V4_DUMP_LAYERS"
DUMP_TAG_ENV = "HY_V4_DUMP_TAG"
TARGETS_ENV = "HY_V4_DUMP_TARGETS"
DEFAULT_DUMP_PATH = "/tmp/hy_v4_parity_dump.json"

# Known ``module:class`` locations of the HY V4 target model: TokenSpeed's and
# vLLM's. Missing ones are skipped.
_TARGETS = (
    ("tokenspeed", "tokenspeed.runtime.models.hy_v4", "HYV4ForCausalLM"),
    ("vllm", "vllm.models.hy_v4.nvidia.model", "HYV4ForCausalLM"),
)


def dump_path() -> Path:
    """Where the active dump should be written."""
    return Path(os.environ.get(DUMP_PATH_ENV, DEFAULT_DUMP_PATH))


def attach_to_model(
    model: torch.nn.Module,
    *,
    num_layers: int | None = None,
    dumper: LayerDumper | None = None,
) -> LayerDumper | None:
    """Attach a dumper to a HY V4 model, or return None if it is not one.

    The check is deliberately explicit rather than duck-typed: a dump file that
    is silently empty would make a parity run look like a pass, so a model that
    cannot be instrumented must be reported as such.

    Args:
        model: Candidate model; must expose a ``layers`` list and a HY V4 config.
        num_layers: Layers to tap; defaults to the config's count.
        dumper: Optional existing dumper to reuse.

    Returns:
        The attached dumper, or None when ``model`` is not a HY V4 target.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "hy_v4":
        return None
    layers = getattr(model, "layers", None)
    container = "layers"
    if (
        layers is None
        and getattr(getattr(model, "model", None), "layers", None) is not None
    ):
        # ``*ForCausalLM`` wrappers keep the decoder stack under ``.model``.
        layers = model.model.layers
        container = "model.layers"
    if layers is None:
        # The MTP draft holds its single block under ``decoder``; the target
        # stack is what both engines are compared on.
        return None
    if num_layers is None:
        num_layers = getattr(config, "num_hidden_layers", len(layers))
    dumper = dumper or LayerDumper()
    dumper.attach(model, num_layers=num_layers, layer_container=container)
    return dumper


def instrument_hy_v4_classes(
    *,
    path: str | os.PathLike[str] | None = None,
    num_layers: int | None = None,
    tags: list[str] | None = None,
    extra_targets: Iterable[str] | None = None,
) -> list[str]:
    """Wrap every HY V4 target class that this interpreter can import.

    The engines construct their model inside the process that will run it, and
    the two do so through different hooks (TokenSpeed builds it in
    ``BaseCausalLM``'s initializer, vLLM builds it in the loader), so this
    patches the model classes themselves: the one hook both engines share.

    The imports are eager because a class cannot be patched before its module is
    loaded, and this function is only ever called from an explicitly armed dump
    session where that cost is acceptable.

    Args:
        path: Dump destination; defaults to ``HY_V4_DUMP_PATH``.
        num_layers: Layers to tap; defaults to each model's own config.
        tags: Labels written beside the dump.
        extra_targets: Additional ``module:class`` pairs to instrument, for a
            reference checkout laid out differently from the two known paths.
            ``HY_V4_DUMP_TARGETS`` (comma separated) is appended to these.

    Returns:
        The class targets that were instrumented (empty when no HY V4 model class
        is importable in this interpreter).
    """
    target = Path(path) if path else dump_path()
    label = tags if tags is not None else [f"tag={os.environ.get(DUMP_TAG_ENV, '')}"]
    names = [f"{module_name}:{class_name}" for _, module_name, class_name in _TARGETS]
    names.extend(extra_targets or [])
    from_env = os.environ.get(TARGETS_ENV, "")
    names.extend(part for part in from_env.split(",") if part.strip())

    instrumented: list[str] = []
    for name in names:
        module_name, _, class_name = name.strip().partition(":")
        if not module_name or not class_name:
            continue
        try:
            module = __import__(module_name, fromlist=[class_name])
        except Exception:
            continue
        model_class = getattr(module, class_name, None)
        if model_class is None:
            continue
        if getattr(model_class, "_hy_v4_dump_patched", False):
            # Already armed (e.g. by the sitecustomize injection); still report
            # it so callers do not mistake a pre-armed class for a missing one.
            instrumented.append(f"{module_name}:{class_name}")
            continue
        if class_name not in _patch_model_class(model_class, target, num_layers, label):
            continue
        instrumented.append(f"{module_name}:{class_name}")
    return instrumented


def _patch_model_class(
    model_class: type,
    target: Path,
    num_layers: int | None,
    tags: list[str],
) -> list[str]:
    """Attach a dumper to every instance of ``model_class`` after construction.

    Returns:
        ``[model_class.__name__]`` once patched, so a caller can report which
        targets actually took effect.
    """
    original_init = model_class.__init__
    state: dict = {"session": None, "dumper": None}

    import functools

    # ``wraps`` keeps ``__wrapped__`` so ``inspect.signature`` still reports the
    # original keyword-only ``vllm_config``; vLLM picks its constructor calling
    # convention from that signature.
    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        if state["dumper"] is not None:
            # Only the first constructed model (the target) is dumped; a draft
            # model would otherwise overwrite the target's taps.
            return
        dumper = attach_to_model(self, num_layers=num_layers)
        if dumper is None:
            return
        session = ModelDumpSession(dumper)
        session._dump_path = target  # noqa: SLF001
        session._tags = tags  # noqa: SLF001
        session._dumper = dumper  # noqa: SLF001
        state["dumper"] = dumper
        state["session"] = session
        print(
            f"[hy_v4_dump] attached to {type(self).__name__} "
            f"({len(dumper.records)} taps so far)",
            file=__import__("sys").stderr,
        )

    model_class.__init__ = patched_init
    model_class._hy_v4_dump_patched = True  # type: ignore[attr-defined]

    import atexit

    atexit.register(lambda: _flush(state, target))
    return [model_class.__name__]


def _flush(state: dict, target: Path) -> None:
    """Write the recorded dump (plus its tags) at interpreter exit.

    Silent when nothing was attached, so an interpreter that merely imported the
    instrumentation does not leave a misleading empty file behind.
    """
    import sys

    dumper = state.get("dumper")
    if dumper is None or not dumper.records:
        return
    try:
        dumper.save(target)
        session = state.get("session")
        tags = getattr(session, "_tags", None) if session is not None else None
        if tags:
            sidecar = target.with_suffix(target.suffix + ".meta.json")
            sidecar.write_text(json.dumps({"tags": tags}, indent=2))
        print(
            f"[hy_v4_dump] wrote {target} ({len(dumper.records)} taps)",
            file=sys.stderr,
        )
    except Exception as error:  # pragma: no cover - diagnostics only
        print(f"[hy_v4_dump] dump failed: {error}", file=sys.stderr)


def dump_layers(
    model: torch.nn.Module,
    *,
    num_layers: int | None = None,
    layer_container: str = "layers",
    path: str | os.PathLike[str] | None = None,
    tags: list[str] | None = None,
) -> ModelDumpSession:
    """Return a session that dumps ``model``'s sub-block taps on exit.

    Args:
        model: The model to inspect; the session walks its layer list.
        num_layers: Layers to tap; defaults to the config's layer count.
        layer_container: Attribute path to the decoder layer list from ``model``.
        path: Destination JSON; defaults to ``HY_V4_DUMP_PATH``.
        tags: Optional free-form labels stored next to the dump, so a run can
            record which engine and config produced it.

    Returns:
        A context manager; exiting it detaches the hooks and writes the dump.
    """
    if num_layers is None:
        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("num_layers is required when the config has no count")
    dumper = LayerDumper()
    session = ModelDumpSession(dumper)
    session.attach(model, num_layers=num_layers, layer_container=layer_container)
    session._dump_path = Path(path) if path else dump_path()  # noqa: SLF001
    session._tags = tags or []  # noqa: SLF001
    session._dumper = dumper  # noqa: SLF001
    return session


def write_dump(session: ModelDumpSession) -> Path:
    """Persist a session's records, with its tags, and return the path."""
    dumper = getattr(session, "_dumper")
    tags = getattr(session, "_tags", [])
    destination = getattr(session, "_dump_path")
    dumper.save(destination)
    if tags:
        sidecar = destination.with_suffix(destination.suffix + ".meta.json")
        import json

        sidecar.write_text(json.dumps({"tags": tags}, indent=2))
    return destination


@contextmanager
def capture_layers(
    model: torch.nn.Module,
    *,
    num_layers: int | None = None,
    layer_container: str = "layers",
    path: str | os.PathLike[str] | None = None,
    tags: list[str] | None = None,
):
    """Context manager wrapper around :func:`dump_layers` + :func:`write_dump`."""
    session = dump_layers(
        model,
        num_layers=num_layers,
        layer_container=layer_container,
        path=path,
        tags=tags,
    )
    try:
        yield session
    finally:
        session.detach()
        write_dump(session)


def compare_dumps(
    reference_path: str | os.PathLike[str],
    candidate_path: str | os.PathLike[str],
    *,
    atol: float = 1e-2,
    rtol: float = 2e-2,
) -> tuple[bool, str]:
    """Compare two dump files and render the verdict.

    Args:
        reference_path: Dump from the known-good engine.
        candidate_path: Dump from the engine under test.
        atol: Absolute tolerance on sampled activations.
        rtol: Relative tolerance on sampled activations.

    Returns:
        ``(passed, report)`` where ``report`` is a human-readable table.
    """
    import json

    reference = json.loads(Path(reference_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    report = first_divergence(reference, candidate, atol=atol, rtol=rtol)
    return (all(entry["status"] == "ok" for entry in report), format_report(report))
