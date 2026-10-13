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

"""Dump per-layer activations from either engine, then compare them.

The hook is engine-agnostic: it attaches to whatever module tree is handed to it
and wraps the sub-blocks of each decoder layer whose names it is told to look
for. TokenSpeed and vLLM name those sub-modules
``self_attn``/``mlp`` and ``hc_attn_layer``/``hc_mlp_layer`` identically, so one
implementation covers both and a dump cannot silently come from a different
place than the other engine's.

Why per-layer dumps rather than only end-to-end logits: a full-stack logprob
comparison tells you *that* something diverged, while the first diverging layer
and sub-block tells you *what*. With iHC in the stack, a wrong boundary formula
and a wrong sink both corrupt the final logits, and only the layer-wise view
separates them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch

# Sub-blocks to tap, in execution order. Each entry is the attribute path below
# a decoder layer plus the stage name used in dump keys.
_DEFAULT_TAPS = (
    ("hc_attn_layer", "hc_attn"),
    ("self_attn", "attn"),
    ("hc_mlp_layer", "hc_mlp"),
    ("mlp", "mlp"),
)


class LayerDumper:
    """Records a summary of each tapped sub-block's input and output.

    Storing the full activation would be needlessly large, so each tap keeps
    statistics that localize a divergence: the max absolute difference against
    the other engine needs the values themselves, so a fixed-size sample of
    positions and channels is stored alongside the statistics.
    """

    def __init__(
        self,
        *,
        sample_tokens: int = 8,
        sample_channels: int = 8,
    ) -> None:
        self.sample_tokens = sample_tokens
        self.sample_channels = sample_channels
        self.records: dict[str, dict] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pass_index = 0

    # -- attachment --------------------------------------------------------
    def attach(
        self,
        root: torch.nn.Module,
        *,
        num_layers: int,
        taps: Iterable[tuple[str, str]] = _DEFAULT_TAPS,
        layer_container: str = "layers",
    ) -> None:
        """Wrap every tapped sub-block of ``root``'s decoder layers.

        Args:
            root: The model (or module) holding the decoder layer list.
            num_layers: Number of layers to tap.
            taps: ``(attribute_path, stage_name)`` pairs per layer.
            layer_container: Attribute holding the layer list; the MTP draft puts
                its single block at ``model.decoder``, hence the override.
        """
        layers = self._resolve_layers(root, layer_container)
        self.detach()
        for layer_id, layer in enumerate(list(layers)[:num_layers]):
            for attribute, stage in taps:
                target = layer
                for part in attribute.split("."):
                    target = getattr(target, part, None)
                    if target is None:
                        break
                if target is None:
                    # iHC boundaries are absent when the model runs without them.
                    continue
                self._wrap(target, layer_id, stage)

    @staticmethod
    def _resolve_layers(root: torch.nn.Module, container: str) -> torch.nn.ModuleList:
        node = root
        for part in container.split("."):
            node = getattr(node, part)
        return node

    def _wrap(self, module: torch.nn.Module, layer_id: int, stage: str) -> None:
        def pre_hook(_module, args, layer_id=layer_id, stage=stage):
            if args:
                self._record(layer_id, stage, "in", args[0])

        def post_hook(_module, _args, output, layer_id=layer_id, stage=stage):
            self._record(layer_id, stage, "out", output)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- recording ---------------------------------------------------------
    def _record(self, layer_id: int, stage: str, position: str, value) -> None:
        if isinstance(value, (tuple, list)):
            if not value:
                return
            value = value[0]
        if not torch.is_tensor(value) or value.numel() == 0:
            return
        key = f"p{self._pass_index}_layer{layer_id}_{stage}_{position}"
        tensor = value.detach()
        # Engines also run warmup / autotune / graph-capture forwards on dummy
        # batches; keep only the real prompt prefill when its length is known.
        only_tokens = int(os.environ.get("HY_V4_DUMP_ONLY_TOKENS", "0") or 0)
        if only_tokens and tensor.reshape(-1, tensor.shape[-1]).shape[0] != only_tokens:
            return
        if only_tokens and key in self.records:
            return
        if tensor.is_cuda:
            tensor = tensor.float().cpu()
        else:
            tensor = tensor.float()
        flat = tensor.reshape(-1, tensor.shape[-1])
        tokens = flat.shape[0]
        channels = flat.shape[1]
        token_idx = torch.arange(min(tokens, self.sample_tokens))
        channel_idx = torch.arange(min(channels, self.sample_channels))
        self.records[key] = {
            "shape": list(tensor.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "abs_max": float(tensor.abs().max()),
            "abs_mean": float(tensor.abs().mean()),
            "norm": float(tensor.norm()),
            "sample": flat[token_idx][:, channel_idx].tolist(),
            "sample_tokens": [int(i) for i in token_idx],
            "sample_channels": [int(i) for i in channel_idx],
        }
        eager_path = os.environ.get("HY_V4_DUMP_EAGER_PATH")
        if eager_path:
            # Engine workers are often killed rather than exited, so atexit
            # never runs; persist after every tap instead.
            self.save(eager_path)

    def next_pass(self) -> None:
        """Advance the pass counter so prefill and decode dumps stay distinct."""
        self._pass_index += 1

    # -- io ----------------------------------------------------------------
    def save(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.records, indent=2, sort_keys=True))
        return destination


def first_divergence(
    reference: dict,
    candidate: dict,
    *,
    atol: float,
    rtol: float,
    norm_rtol: float | None = None,
) -> list[dict]:
    """Compare two dump files and return per-key deviations, worst first.

    Three independent signals are checked, because an 8x8 corner sample alone can
    miss a divergence that concentrates elsewhere in the tensor:

    * the sampled values, against ``atol + rtol * |reference|``;
    * the Frobenius norm ratio, which summarizes the whole tensor;
    * the peak magnitude ratio, which catches a localized blow-up.

    Args:
        reference: Dump from the trusted reference engine.
        candidate: Dump from the engine under test.
        atol: Absolute tolerance on the sampled values.
        rtol: Relative tolerance on the sampled values.
        norm_rtol: Relative tolerance on the whole-tensor statistics; defaults to
            ``rtol``.

    Returns:
        One entry per compared key, ordered by decreasing error.
    """
    norm_rtol = rtol if norm_rtol is None else norm_rtol
    report: list[dict] = []
    for key in sorted(set(reference) | set(candidate)):
        left, right = reference.get(key), candidate.get(key)
        if left is None or right is None:
            report.append(
                {
                    "key": key,
                    "status": "missing",
                    "error": float("inf"),
                    "detail": "present in only one dump",
                }
            )
            continue
        if left["shape"] != right["shape"]:
            report.append(
                {
                    "key": key,
                    "status": "shape",
                    "error": float("inf"),
                    "detail": f"{left['shape']} vs {right['shape']}",
                }
            )
            continue

        left_sample = torch.tensor(left["sample"], dtype=torch.float32)
        right_sample = torch.tensor(right["sample"], dtype=torch.float32)
        deviation = (left_sample - right_sample).abs()
        tolerance = atol + rtol * left_sample.abs()
        sample_excess = float((deviation - tolerance).max())

        norm_delta = _relative_delta(left["norm"], right["norm"])
        peak_delta = _relative_delta(left["abs_max"], right["abs_max"])
        # Normalize each signal by its own tolerance so the ranking reflects how
        # badly each one exceeded, not which units it happens to be in.
        magnitude = max(abs(left["abs_mean"]), 1e-6)
        sample_score = float(deviation.max()) / magnitude if sample_excess > 0 else 0.0
        score = max(
            sample_score,
            norm_delta / norm_rtol if norm_delta > norm_rtol else 0.0,
            peak_delta / norm_rtol if peak_delta > norm_rtol else 0.0,
        )
        breached = sample_excess > 0 or norm_delta > norm_rtol or peak_delta > norm_rtol
        if not breached:
            error = max(sample_score, norm_delta / norm_rtol, peak_delta / norm_rtol)
        else:
            error = max(score, 1e-12)
        report.append(
            {
                "key": key,
                "status": "diverge" if breached else "ok",
                "error": error,
                "max_abs_diff": float(deviation.max()),
                "ref_abs_mean": left["abs_mean"],
                "cand_abs_mean": right["abs_mean"],
                "norm_delta": norm_delta,
                "peak_delta": peak_delta,
                "detail": "",
            }
        )
    report.sort(key=lambda entry: -entry["error"])
    return report


def _relative_delta(left: float, right: float) -> float:
    scale = max(abs(left), 1e-6)
    return abs(left - right) / scale


def format_report(report: list[dict], *, limit: int = 14) -> str:
    """Render the worst deviations as a table for test output."""
    lines = [
        "key                              status    err      max|diff|   norm_d   peak_d"
    ]
    for entry in report[:limit]:
        lines.append(
            f"{entry['key'][:32]:32s} {entry['status']:9s} "
            f"{entry['error']:8.2e} {entry.get('max_abs_diff', float('nan')):10.2e} "
            f"{entry.get('norm_delta', float('nan')):8.1e} "
            f"{entry.get('peak_delta', float('nan')):8.1e}"
        )
    diverged = [e for e in report if e["status"] != "ok"]
    lines.append(
        f"{len(diverged)}/{len(report)} taps outside tolerance"
        + (f"; worst: {diverged[0]['key']}" if diverged else "")
    )
    return "\n".join(lines)


def worst_status(report: list[dict]) -> tuple[str, float, str]:
    """Return ``(status, error, detail)`` of the largest deviation."""
    if not report:
        return ("ok", 0.0, "no overlapping taps")
    worst = max(report, key=lambda entry: entry["error"])
    return (worst["status"], worst["error"], worst.get("detail", ""))


class ModelDumpSession:
    """Attaches one dumper to model instances and detaches them together.

    The engines build their model inside a factory, so the session is handed to
    that factory instead of reaching into a live process.
    """

    def __init__(self, dumper: LayerDumper) -> None:
        self.dumper = dumper
        self._attached: list[torch.nn.Module] = []

    def attach(self, module: torch.nn.Module, **kwargs) -> None:
        """Attach the dumper to ``module``; kwargs go to ``LayerDumper.attach``."""
        self.dumper.attach(module, **kwargs)
        self._attached.append(module)

    def mark_pass(self) -> None:
        """Advance the dumper's pass counter between prefill and decode."""
        self.dumper.next_pass()

    def detach(self) -> None:
        self.dumper.detach()
        self._attached.clear()

    def __enter__(self) -> ModelDumpSession:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.detach()
