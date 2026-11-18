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

"""Profile MLP activity or watch a frozen channel list on eager Qwen3 forwards.

The input is a JSON array of records with ``id`` and explicit ``token_ids``.
An optional ``text`` must encode to exactly those IDs with no added special
tokens. The IDs are the model input; this example never changes or truncates
them. ``--rows all`` measures every input position; ``--rows last`` measures
only each input's final, next-token decision position. It uses the same
cold-prefill ModelRunner path as inspect_final_mlp.py.

Example input::

    [{"id": "request-1", "token_ids": [9707]}]

This is a single-process analysis example, not a serving or tool policy. The
caller-supplied model identity describes the loaded weights; it is not a
verified signature. Profiling and watching support eager, unquantized,
single-device Qwen3 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mlp_channel_inputs import (
    InputCase,
)
from mlp_channel_inputs import _nonnegative as _nonnegative
from mlp_channel_inputs import build_parser as _build_parser
from mlp_channel_inputs import parse_histogram_edges as parse_histogram_edges
from mlp_channel_inputs import (
    validate_cases,
    validate_mode_options,
)


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(description=__doc__)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_inputs(path: Path, max_tokens: int) -> tuple[InputCase, ...]:
    return validate_cases(
        json.loads(path.read_text(encoding="utf-8")), max_tokens=max_tokens
    )


def _check_tokenizer(runtime: object, cases: Sequence[InputCase]) -> None:
    vocab_size = runtime.model.config.vocab_size
    for case in cases:
        if any(token >= vocab_size for token in case.token_ids):
            raise ValueError(
                f"Input {case.id} has a token ID outside this model's vocabulary"
            )
        if (
            case.text is not None
            and tuple(runtime.tokenizer.encode(case.text, add_special_tokens=False))
            != case.token_ids
        ):
            raise ValueError(f"Input {case.id} text does not encode to its token IDs")


def _selection_reasons(row: dict, thresholds: dict) -> list[str]:
    reasons = []
    if (
        row["active_fraction"] <= thresholds["max_active_fraction"]
        and row["mean_contribution"] <= thresholds["max_mean_contribution"]
        and row["max_contribution"] <= thresholds["max_observed_contribution"]
    ):
        reasons.append("quiet_activity")
    ratio = row["down_norm_ratio_to_layer_median"]
    if (
        ratio is not None
        and ratio >= thresholds["min_down_norm_ratio"]
        and row["active_fraction"] <= thresholds["max_outlier_active_fraction"]
    ):
        reasons.append("large_down_column_with_low_activity")
    return reasons


def _candidate_preview(
    ranked: Sequence[dict],
    watchlist: dict,
    thresholds: dict,
    *,
    norm_outliers: Sequence[dict] = (),
) -> dict:
    """Show quiet and large-down-column ranks with explicit selection reasons."""
    selected = {
        (entry["layer"], entry["channel"]): entry["contribution_limit"]
        for entry in watchlist["entries"]
    }

    def preview_rows(rows: Sequence[dict]) -> list[dict]:
        entries = []
        for rank, row in enumerate(rows, start=1):
            key = (row["layer"], row["channel"])
            reasons = _selection_reasons(row, thresholds)
            if (key in selected) != bool(reasons):
                raise RuntimeError("Candidate selection differs from frozen watchlist")
            entries.append(
                {
                    "rank": rank,
                    "layer": row["layer"],
                    "channel": row["channel"],
                    "sample_count": row["sample_count"],
                    "active_fraction": row["active_fraction"],
                    "mean_abs": row["mean_abs"],
                    "max_abs": row["max_abs"],
                    "down_column_norm": row["down_column_norm"],
                    "down_norm_ratio_to_layer_median": row[
                        "down_norm_ratio_to_layer_median"
                    ],
                    "mean_contribution": row["mean_contribution"],
                    "max_contribution": row["max_contribution"],
                    "selected_for_watch": key in selected,
                    "selection_reasons": reasons,
                    "contribution_limit": selected.get(key),
                }
            )
        return entries

    return {
        "schema": "channel-candidate-preview-v1",
        "model_identity": watchlist["model_identity"],
        "ranking": "active_fraction, mean_contribution, layer, channel",
        "entries": preview_rows(ranked),
        "norm_outlier_ranking": "down_norm_ratio_to_layer_median descending",
        "norm_outlier_entries": preview_rows(norm_outliers),
    }


def run(args: argparse.Namespace) -> dict:
    edges = validate_mode_options(args)
    cases = _read_inputs(args.inputs, args.max_tokens)
    input_sha256 = hashlib.sha256(args.inputs.read_bytes()).hexdigest()

    # Import GPU dependencies only after validating the input and CLI options.
    import torch
    from inspect_final_mlp import ColdPrefill

    from tokenspeed.runtime.inspection.channel_profile import ChannelActivityProfiler
    from tokenspeed.runtime.inspection.channel_watch import ChannelWatch, RowIdentity

    runtime = ColdPrefill(args)
    _check_tokenizer(runtime, cases)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    properties = torch.cuda.get_device_properties(runtime.device)
    common = {
        "model": args.model,
        "revision": args.revision,
        "model_identity": args.model_identity,
        "model_identity_verified": False,
        "input_sha256": input_sha256,
        "input_count": len(cases),
        "gpu_arch": getattr(properties, "gcnArchName", None),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dtype": "bfloat16",
        "norm_mode": args.norm_mode,
        "attention_backend": args.attention_backend,
        "rows": args.rows,
        "scope": "Single-device eager cold prefill; no scheduler, decode, or serving lifecycle",
    }
    if args.mode == "profile":
        with ChannelActivityProfiler(
            runtime.model,
            model_identity=args.model_identity,
            execution="eager",
            speculative=False,
            monokernel=False,
            activation_threshold=args.activation_threshold,
            histogram_edges=edges,
        ) as profiler:
            for case in cases:
                ids = torch.tensor(
                    case.token_ids, device=runtime.device, dtype=torch.int32
                )
                row_indices = None if args.rows == "all" else [len(case.token_ids) - 1]
                with profiler.record(row_indices=row_indices):
                    runtime.forward(ids)
        profile = profiler.profile()
        thresholds = {
            "activation_threshold": args.activation_threshold,
            "histogram_edges": list(edges),
            "max_active_fraction": args.max_active_fraction,
            "max_mean_contribution": args.max_mean_contribution,
            "max_observed_contribution": args.max_observed_contribution,
            "min_observed_rows": args.min_observed_rows,
            "runtime_multiplier": args.runtime_multiplier,
            "contribution_floor": args.contribution_floor,
            "min_down_norm_ratio": args.min_down_norm_ratio,
            "max_outlier_active_fraction": args.max_outlier_active_fraction,
        }
        watchlist = profiler.freeze_watchlist(
            max_active_fraction=args.max_active_fraction,
            max_mean_contribution=args.max_mean_contribution,
            max_observed_contribution=args.max_observed_contribution,
            min_observed_rows=args.min_observed_rows,
            runtime_multiplier=args.runtime_multiplier,
            contribution_floor=args.contribution_floor,
            min_down_norm_ratio=args.min_down_norm_ratio,
            max_outlier_active_fraction=args.max_outlier_active_fraction,
        )
        norm_outliers = sorted(
            (
                row
                for layer in profile["layers"]
                for row in layer["channels"]
                if "large_down_column_with_low_activity"
                in _selection_reasons(row, thresholds)
            ),
            key=lambda row: (
                -row["down_norm_ratio_to_layer_median"],
                row["active_fraction"],
                row["layer"],
                row["channel"],
            ),
        )[:100]
        preview = _candidate_preview(
            profiler.rank_candidates(limit=100),
            watchlist,
            thresholds,
            norm_outliers=norm_outliers,
        )
        _write_json(args.output_dir / "profile.json", profile)
        _write_json(args.output_dir / "watchlist.json", watchlist)
        _write_json(args.output_dir / "candidate-preview.json", preview)
        report = {
            **common,
            "mode": "profile",
            "thresholds": thresholds,
            "profile_path": "profile.json",
            "watchlist_path": "watchlist.json",
            "candidate_preview_path": "candidate-preview.json",
            "candidate_preview_count": len(preview["entries"]),
            "norm_outlier_preview_count": len(preview["norm_outlier_entries"]),
            "profiled_channel_count": sum(
                len(layer["channels"]) for layer in profile["layers"]
            ),
            "watched_channel_count": len(watchlist["entries"]),
            "physical_token_rows": sum(len(case.token_ids) for case in cases),
            "sampled_token_rows": sum(
                len(case.token_ids) if args.rows == "all" else 1 for case in cases
            ),
        }
    else:
        watchlist_bytes = args.watchlist.read_bytes()
        watchlist = json.loads(watchlist_bytes)
        if not isinstance(watchlist, dict):
            raise ValueError("Watchlist must be a JSON object")
        report = {
            **common,
            "mode": "watch",
            "watchlist_sha256": hashlib.sha256(watchlist_bytes).hexdigest(),
            "watched_channel_count": len(watchlist.get("entries", [])),
            "cases": [],
        }
        event_count = 0
        with (args.output_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
            with ChannelWatch(
                runtime.model,
                watchlist=watchlist,
                model_identity=args.model_identity,
                execution="eager",
                speculative=False,
                monokernel=False,
            ) as watcher:
                for case in cases:
                    ids = torch.tensor(
                        case.token_ids, device=runtime.device, dtype=torch.int32
                    )
                    baseline = runtime.forward(ids)
                    rows = tuple(
                        RowIdentity(case.id, position)
                        for position in range(len(case.token_ids))
                    )
                    selected_rows = (
                        None if args.rows == "all" else [len(case.token_ids) - 1]
                    )
                    with watcher.record(rows, selected_rows=selected_rows):
                        observed = runtime.forward(ids)
                    snapshot = watcher.snapshot()
                    events = watcher.events(snapshot)
                    for event in events:
                        stream.write(json.dumps(asdict(event), allow_nan=False) + "\n")
                    event_count += len(events)
                    finite = bool(
                        torch.isfinite(baseline).all()
                        and torch.isfinite(observed).all()
                    )
                    difference = observed.float() - baseline.float()
                    identical = bool(finite and torch.equal(baseline, observed))
                    top_values, top_ids = torch.topk(
                        observed.float().flatten(), min(args.top_k, observed.numel())
                    )
                    report["cases"].append(
                        {
                            "id": case.id,
                            "input_token_count": len(case.token_ids),
                            "sampled_token_rows": (
                                len(case.token_ids) if args.rows == "all" else 1
                            ),
                            "input_token_ids_sha256": hashlib.sha256(
                                json.dumps(
                                    case.token_ids, separators=(",", ":")
                                ).encode("ascii")
                            ).hexdigest(),
                            "event_count": len(events),
                            "finite_logits": finite,
                            "watcher_on_off_exact_equal": identical,
                            "max_abs_logit_difference": (
                                float(difference.abs().max()) if finite else None
                            ),
                            "next_token_top": (
                                [
                                    {"token_id": int(token_id), "logit": float(value)}
                                    for token_id, value in zip(
                                        top_ids.tolist(), top_values.tolist()
                                    )
                                ]
                                if finite
                                else []
                            ),
                        }
                    )
        report["events_path"] = "events.jsonl"
        report["event_count"] = event_count
        report["passed"] = all(
            case["watcher_on_off_exact_equal"] for case in report["cases"]
        )
    _write_json(args.output_dir / "report.json", report)
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "output_dir": str(args.output_dir),
                "input_count": result["input_count"],
                "watched_channel_count": result["watched_channel_count"],
                "event_count": result.get("event_count"),
                "passed": result.get("passed"),
            },
            allow_nan=False,
        )
    )
    if result.get("passed") is False:
        raise SystemExit("Watched and unwatched logits differed; see report.json")


if __name__ == "__main__":
    main()
