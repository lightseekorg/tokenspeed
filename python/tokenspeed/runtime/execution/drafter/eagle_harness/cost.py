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

"""EAGLE-Harness cost data: the measured verify (target) and draft-step cost
per batch size and context interval that the harness's decisions read
(:func:`load_cost_data`), and its processing from a measurement run's samples.

A measurement (``measure_run``) records two live stage costs at short
contexts, ``model_slm`` (one draft step) and ``model_llm`` (one verify step)
per query width, and an attention microbenchmark (``cache_slm`` /
``cache_llm``, one attention layer) over the whole context range. The
context-free part of a stage, its query cost, is the stage minus its
attention layers at the sample's context; a stage at context interval ``I``
is then estimated as::

    model(I, q) = query(q) + cache(I, q) * attention_layers

with sliding-window layers frozen at the window's interval. Every average is
a robust mean (iterated z-score outlier removal); a run whose averages do
not converge is unstable and should be repeated.

Layout under the run directory (serving's ``--speculative-data-path``)::

    test/bs_<n>/{model,cache}_{slm,llm}/{model,cache}_qo_len_<q>.jsonl  raw samples
    cost/bs_<n>/cache_{slm,llm}/cache_interval_<i>.json
    cost/bs_<n>/query_{slm,llm}/query_<q>.json
    cost/bs_<n>/model_{slm,llm}/model_interval_<i>.json                 serving input

Interval ``i`` covers contexts ``[step * (i + 1), step * (i + 2))``, with
``step = cache_length_step``. A ``model_interval`` file holds its stage's cost
per query width (``index``: widths minus one, ``value``: microseconds,
``ratio``: cost relative to the first width). The processing steps share
their other arguments: the measured ``batch_size`` and query ``widths``
(ascending), the widest swept width ``max_query_length`` (each cache row's
length), and a stage's ``attention_layers`` (draft 1, target its
full-attention layers), of which ``window_layers`` see a sliding ``window``
(0: none). The file formats (keys, rounding, ``indent=4``) follow EAGLE-4's
cost data format.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "CHAIN_RULE_MAX_BATCH_SIZE",
    "COST_BATCH_SIZES",
    "BatchCost",
    "IntervalCost",
    "MeasureRecord",
    "load_cost_data",
    "load_records",
    "max_batch_size",
    "measured_batch_sizes",
    "process_cache",
    "process_model",
    "process_query",
    "robust_mean",
    "save_records",
]

# Batch sizes the cost measurement samples.
COST_BATCH_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
# Largest chain round the confidence rule decides; the cost data and its measurement stop here.
CHAIN_RULE_MAX_BATCH_SIZE = COST_BATCH_SIZES[-1]


def max_batch_size(compute_intensity: str) -> int:
    """Largest batch the harness drafts for at this intensity: 32 for trees,
    512 for chains, whose width table decides every round above
    ``CHAIN_RULE_MAX_BATCH_SIZE``."""
    return 512 if compute_intensity == "chain_intensity" else 32


@dataclass(frozen=True)
class IntervalCost:
    """One model's cost at one context interval, per query width."""

    value: list[float]  # microseconds
    ratio: list[float]  # relative to the first width
    index: list[int]  # query width minus one


@dataclass(frozen=True)
class BatchCost:
    """Every interval's ``(target, draft)`` cost at one measured batch size."""

    intervals: list[tuple[IntervalCost, IntervalCost]]
    cache_length_step: int
    # Target over draft cost at width one, per interval (-1.0 without draft cost).
    model_ratio: list[float]


def _interval_cost(data: dict) -> IntervalCost:
    return IntervalCost(value=data["value"], ratio=data["ratio"], index=data["index"])


def _load_intervals(directory: str, kind: str) -> list[dict]:
    """Every ``<kind>_interval_<i>.json`` under ``directory``, by interval.

    Raises:
        ValueError: There is none, or an interval is missing.
    """
    prefix = f"{kind}_interval_"
    ids = sorted(
        int(name[len(prefix) : -len(".json")])
        for name in os.listdir(directory)
        if name.startswith(prefix) and name.endswith(".json")
    )
    if not ids or ids != list(range(len(ids))):
        raise ValueError(f"{directory}: needs {prefix}<i>.json for i = 0, 1, ...")
    rows = []
    for i in ids:
        with open(os.path.join(directory, f"{prefix}{i}.json"), encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def measured_batch_sizes(cost_dir: str) -> list[int]:
    """The batch sizes ``cost_dir`` (``<data_path>/cost``) holds cost data for
    (its ``bs_<N>`` directories), ascending."""
    return sorted(
        int(entry[len("bs_") :])
        for entry in os.listdir(cost_dir)
        if entry.startswith("bs_") and os.path.isdir(os.path.join(cost_dir, entry))
    )


def load_cost_data(cost_dir: str) -> dict[int, BatchCost]:
    """Every measured batch size's cost under ``cost_dir`` (``<data_path>/cost``).

    Raises:
        ValueError: A batch directory misses its target or draft cost, or an
            interval the other model has.
    """
    batches = {}
    for entry in sorted(os.listdir(cost_dir)):
        path = os.path.join(cost_dir, entry)
        if not (entry.startswith("bs_") and os.path.isdir(path)):
            continue
        batch_size = int(entry[len("bs_") :])
        target = _load_intervals(os.path.join(path, "model_llm"), "model")
        draft = _load_intervals(os.path.join(path, "model_slm"), "model")
        if len(target) != len(draft):
            raise ValueError(f"{path}: target and draft cost differ in intervals")
        intervals, model_ratio = [], []
        for target_data, draft_data in zip(target, draft):
            target_cost = _interval_cost(target_data)
            draft_cost = _interval_cost(draft_data)
            intervals.append((target_cost, draft_cost))
            target_v0 = target_cost.value[0] if target_cost.value else 0
            draft_v0 = draft_cost.value[0] if draft_cost.value else 0
            model_ratio.append(round(target_v0 / draft_v0, 4) if draft_v0 > 0 else -1.0)
        batches[batch_size] = BatchCost(
            intervals=intervals,
            cache_length_step=target[0]["cache_length_step"],
            model_ratio=model_ratio,
        )
    return batches


@dataclass(frozen=True)
class MeasureRecord:
    """One raw sample: ``time_intervals`` microseconds over ``measure_count``
    repetitions (draft steps, verify steps or attention layers)."""

    time_intervals: int
    measure_count: int
    query_length: int
    cache_length: int
    task_category: str  # model_slm, model_llm, cache_slm or cache_llm
    batch_size: int


def save_records(test_dir: str, records: list[MeasureRecord]) -> None:
    """Write ``records`` under ``test_dir`` (``test/bs_<n>``), one file per
    category and query width."""
    by_file: dict[str, list[MeasureRecord]] = defaultdict(list)
    for r in records:
        name = f"{r.task_category.split('_')[0]}_qo_len_{r.query_length}.jsonl"
        by_file[os.path.join(test_dir, r.task_category, name)].append(r)
    for path, group in by_file.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in group:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_records(path: str) -> list[dict]:
    """The records of a ``save_records`` JSONL file, one dict per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def robust_mean(data: list[float], convergence: float) -> tuple[float, bool]:
    """Mean after iterated z-score outlier removal.

    The z threshold starts at 6 and drops by 0.2 per pass (to at least 1)
    until the kept samples span at most ``convergence`` of their mean, at
    most 50 passes, stopping early if fewer than 75% of the samples remain.

    Returns:
        The mean, and whether it converged with at least 75% of the samples
        kept.
    """
    kept = list(data)
    threshold = 6.0
    range_rate = 1.0
    mean = np.mean(kept)
    for _ in range(50):
        values = np.asarray(kept, dtype=np.float64)
        if values.std() == 0:
            # Identical samples span nothing; a zero mean still never converges.
            range_rate = 0.0 if mean != 0 else np.inf
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs((values - values.mean()) / values.std())
        kept = [x for i, x in enumerate(kept) if z[i] < threshold]
        if kept:
            mean = np.mean(kept)
        if len(kept) < 0.75 * len(data):
            break
        range_rate = (np.max(kept) - np.min(kept)) / mean
        if mean != 0 and range_rate <= convergence:
            break
        threshold = max(threshold - 0.2, 1.0)
    converged = range_rate <= convergence and len(kept) >= 0.75 * len(data)
    return mean, converged


def _dump(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def process_cache(
    source_dir: str,
    target_dir: str,
    *,
    widths: list[int],
    max_query_length: int,
    cache_length_step: int,
    cache_length_stop: int,
    batch_size: int,
    fast: bool,
) -> float:
    """Each interval's attention cost per layer and width, from the raw
    ``cache_qo_len_<q>.jsonl`` in ``source_dir`` to ``cache_interval_<i>.json``
    in ``target_dir``, up to the context ``cache_length_stop``; ``fast``
    averages with the tighter convergence (0.25 instead of 0.35).

    Returns:
        The share of (interval, width) means that did not converge; at least
        0.20 (fast) / 0.25 makes the run unstable.
    """
    step = cache_length_step
    samples: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in widths:
        for record in load_records(os.path.join(source_dir, f"cache_qo_len_{q}.jsonl")):
            kv_len = record["cache_length"]
            if not step <= kv_len < cache_length_stop:
                continue
            samples[(kv_len - step) // step][q].append(
                record["time_intervals"] / record["measure_count"]
            )
    means: dict[int, list] = defaultdict(lambda: [0.0] * max_query_length)
    total = unstable = 0
    for interval_id, per_width in samples.items():
        for q, values in per_width.items():
            total += 1
            mean, converged = robust_mean(values, 0.25 if fast else 0.35)
            unstable += not converged
            means[interval_id][q - 1] = mean
    instability = unstable / total if total else 0.0
    for interval_id in range((cache_length_stop - step) // step):
        row = means[interval_id]
        ratio = [round(v / row[0], 4) for v in row] if row and row[0] > 0 else []
        _dump(
            os.path.join(target_dir, f"cache_interval_{interval_id}.json"),
            {
                "value": [round(v, 4) for v in row],
                "index": list(range(len(row))),
                "ratio": ratio,
                "array": [round(v, 4) for v in row],
                "batch_size": batch_size,
                "max_query_length": max_query_length,
                "cache_length_start": step,
                "cache_length_step": step,
                "cache_length_stop": cache_length_stop,
                "instability_rate": round(instability, 2),
            },
        )
    return instability


def process_query(
    model_dir: str,
    cache_dir: str,
    target_dir: str,
    *,
    widths: list[int],
    cache_length_step: int,
    attention_layers: int,
    batch_size: int,
    loose: bool,
    window_layers: int,
    window: int,
) -> float:
    """Each width's query cost ``query_<q>.json`` in ``target_dir``: the raw
    stage samples in ``model_dir`` minus their attention layers at each
    sample's interval (``cache_interval_<i>.json`` in ``cache_dir``);
    ``loose`` averages with the looser convergence (0.50 instead of 0.30).

    Returns:
        The share of widths whose mean did not converge; at least 0.30 makes
        the run unstable.
    """
    step = cache_length_step
    cache = [row["array"] for row in _load_intervals(cache_dir, "cache")]
    costs: dict[int, list[float]] = defaultdict(list)
    for q in widths:
        for record in load_records(os.path.join(model_dir, f"model_qo_len_{q}.jsonl")):
            kv_len = record["cache_length"]
            if kv_len < step:
                continue
            cost_model = record["time_intervals"] / record["measure_count"]
            interval_id = (kv_len - step) // step
            if interval_id < len(cache):
                cost_cache = cache[interval_id][q - 1]
                cost_query = cost_model - cost_cache * attention_layers
                if window_layers > 0 and kv_len > window:
                    # The window layers saw only the window's keys.
                    window_id = max((window - step) // step, 0)
                    cost_query += window_layers * (cost_cache - cache[window_id][q - 1])
                costs[q].append(cost_query)
    means, unstable = {}, 0
    for q in widths:
        if not costs[q]:
            raise RuntimeError(f"no query cost samples for width {q}")
        means[q], converged = robust_mean(costs[q], 0.50 if loose else 0.30)
        unstable += not converged
    instability = unstable / len(widths)
    for q in widths:
        _dump(
            os.path.join(target_dir, f"query_{q}.json"),
            {
                "mean": round(means[q], 4),
                "array": [round(v, 4) for v in costs[q]],
                "batch_size": batch_size,
                "instability_rate": round(instability, 4),
            },
        )
    return instability


def process_model(
    query_dir: str,
    cache_dir: str,
    target_dir: str,
    *,
    widths: list[int],
    max_query_length: int,
    cache_length_step: int,
    attention_layers: int,
    batch_size: int,
    window_layers: int,
    window: int,
) -> None:
    """Each interval's stage cost estimate ``model_interval_<i>.json`` in
    ``target_dir``, from the stage's ``query_<q>.json`` in ``query_dir`` and
    ``cache_interval_<i>.json`` in ``cache_dir``."""
    query = []
    for q in widths:
        with open(os.path.join(query_dir, f"query_{q}.json"), encoding="utf-8") as f:
            query.append(json.load(f)["mean"])
    rows = _load_intervals(cache_dir, "cache")
    window_id = max(window // cache_length_step - 1, 0)
    for interval_id, row in enumerate(rows):
        cache = row["array"]
        if len(cache) != max_query_length:
            raise RuntimeError(
                f"cache interval {interval_id} has no {max_query_length}-wide row"
            )
        estimate = [0] * len(widths)
        for k, q in enumerate(widths):
            if window_id == 0 or interval_id <= window_id:
                estimate[k] = query[k] + cache[q - 1] * attention_layers
            else:
                estimate[k] = (
                    query[k]
                    + cache[q - 1] * (attention_layers - window_layers)
                    + rows[window_id]["array"][q - 1] * window_layers
                )
        ratio = (
            [round(v / estimate[0], 4) for v in estimate]
            if estimate and estimate[0] > 0
            else []
        )
        _dump(
            os.path.join(target_dir, f"model_interval_{interval_id}.json"),
            {
                "notes": "Estimated model_cost = query_cost + cache_cost",
                "value": [round(v, 4) for v in estimate],
                "index": [q - 1 for q in widths],
                "ratio": ratio,
                "array_model": [round(v, 4) for v in estimate],
                "array_query": [round(v, 4) for v in query],
                "array_cache": [round(v * attention_layers, 4) for v in cache],
                "batch_size": batch_size,
                "max_query_length_query": max_query_length,
                "max_query_length_cache": max_query_length,
                "cache_length_start": cache_length_step,
                "cache_length_step": cache_length_step,
                "cache_layers_num": attention_layers,
                "instability_rate": row["instability_rate"],
            },
        )
