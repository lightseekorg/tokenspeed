"""Input contract for the public MLP channel inspection example."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES))
import profile_mlp_channels as example  # noqa: E402


def test_input_cases_and_hash_match_file_bytes(tmp_path) -> None:
    path = tmp_path / "inputs.json"
    contents = b'[{"id":"sample","token_ids":[1,2]}]\n'
    path.write_bytes(contents)
    cases, digest = example._read_inputs(path, max_tokens=2)
    assert cases == (example.InputCase("sample", (1, 2)),)
    assert digest == hashlib.sha256(contents).hexdigest()


@pytest.mark.parametrize(
    "entry",
    [
        (str(EXAMPLES / "profile_mlp_channels.py"),),
        ("-m", "examples.profile_mlp_channels"),
    ],
)
def test_script_and_module_help(entry: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=EXAMPLES.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--mode {profile,watch}" in result.stdout


@pytest.mark.parametrize(
    (
        "baseline",
        "observed",
        "top_k",
        "finite",
        "equal",
        "max_difference",
        "top_ids",
    ),
    [
        ([1.0, 2.0], [1.0, 2.0], 1, True, True, 0.0, [1]),
        ([1.0, 2.0], [2.0, 1.0], 10, True, False, 1.0, [0, 1]),
        ([1.0, 2.0], [1.0, float("nan")], 1, False, False, None, []),
    ],
)
def test_logit_comparison_preserves_finite_and_exact_equal_contract(
    baseline, observed, top_k, finite, equal, max_difference, top_ids
) -> None:
    torch = pytest.importorskip("torch")
    comparison = example._compare_logits(
        torch.tensor(baseline), torch.tensor(observed), top_k=top_k
    )
    assert comparison["finite_logits"] is finite
    assert comparison["watcher_on_off_exact_equal"] is equal
    assert comparison["max_abs_logit_difference"] == max_difference
    assert [row["token_id"] for row in comparison["next_token_top"]] == top_ids


def test_explicit_ids_are_required_and_preserved() -> None:
    records = [
        {"id": "a", "token_ids": [1, 2], "text": "example"},
        {"id": "b", "token_ids": [3]},
    ]
    cases = example.validate_cases(records, max_tokens=2)
    assert [(case.id, case.token_ids) for case in cases] == [("a", (1, 2)), ("b", (3,))]
    assert records[0]["token_ids"] == [1, 2]


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"id": "a", "text": "implicit tokenization"}],
        [{"id": "a", "token_ids": []}],
        [{"id": "a", "token_ids": [1, 2, 3]}],
        [{"id": "a", "token_ids": [True]}],
        [{"id": "a", "token_ids": [-1]}],
        [{"id": "a", "token_ids": [1], "unexpected": 1}],
        [{"id": "a", "token_ids": [1]}, {"id": "a", "token_ids": [2]}],
    ],
)
def test_malformed_or_ambiguous_inputs_fail(records: object) -> None:
    with pytest.raises(ValueError):
        example.validate_cases(records, max_tokens=2)


def test_optional_text_must_match_exact_ids_and_vocab() -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [1, 2] if text == "known text" else [9]

    runtime = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=10)),
        tokenizer=Tokenizer(),
    )
    example._check_tokenizer(
        runtime,
        example.validate_cases(
            [{"id": "match", "token_ids": [1, 2], "text": "known text"}],
            max_tokens=3,
        ),
    )
    with pytest.raises(ValueError, match="does not encode"):
        example._check_tokenizer(
            runtime,
            example.validate_cases(
                [{"id": "mismatch", "token_ids": [1], "text": "known text"}],
                max_tokens=3,
            ),
        )
    with pytest.raises(ValueError, match="outside"):
        example._check_tokenizer(
            runtime,
            example.validate_cases(
                [{"id": "invalid", "token_ids": [10]}], max_tokens=3
            ),
        )


def test_profile_thresholds_must_be_explicit_and_finite() -> None:
    args = Namespace(
        mode="profile",
        rows="last",
        model_identity="model-1",
        gpu_id=0,
        max_tokens=512,
        cache_mib=128,
        top_k=5,
        watchlist=None,
        activation_threshold=0.1,
        histogram_edges="0.01,1,10",
        max_active_fraction=0.01,
        max_mean_contribution=0.2,
        max_observed_contribution=1.0,
        min_observed_rows=100,
        runtime_multiplier=2.0,
        contribution_floor=0.5,
        min_down_norm_ratio=2.0,
        max_outlier_active_fraction=0.02,
    )
    assert example.validate_mode_options(args) == (0.01, 1.0, 10.0)
    args.max_mean_contribution = float("nan")
    with pytest.raises(ValueError, match="finite"):
        example.validate_mode_options(args)
    args.max_mean_contribution = 0.2
    args.histogram_edges = "1,1"
    with pytest.raises(ValueError, match="ascending"):
        example.validate_mode_options(args)
    args.histogram_edges = "0.01,1,10"
    args.min_down_norm_ratio = 0.0
    with pytest.raises(ValueError, match="positive"):
        example.validate_mode_options(args)
    args.min_down_norm_ratio = 2.0
    args.max_outlier_active_fraction = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        example.validate_mode_options(args)


def test_row_mode_is_required_and_validated() -> None:
    parser = example.build_parser()
    common = [
        "--mode",
        "watch",
        "--model",
        "model",
        "--revision",
        "revision",
        "--model-identity",
        "identity",
        "--inputs",
        "inputs.json",
        "--output-dir",
        "output",
        "--watchlist",
        "watchlist.json",
        "--norm-mode",
        "fused",
        "--attention-backend",
        "mha",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    args = parser.parse_args([*common, "--rows", "last"])
    assert example.validate_mode_options(args) is None
    args.rows = "decision"
    with pytest.raises(ValueError, match="rows must be"):
        example.validate_mode_options(args)


def test_candidate_preview_marks_nominations_without_reordering() -> None:
    ranked = [
        {
            "layer": 1,
            "channel": 5,
            "sample_count": 100,
            "active_fraction": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.01,
            "mean_contribution": 0.0,
            "max_contribution": 0.5,
            "down_column_norm": 1.2,
            "down_norm_ratio_to_layer_median": 0.8,
        },
        {
            "layer": 0,
            "channel": 2,
            "sample_count": 100,
            "active_fraction": 0.5,
            "mean_abs": 0.1,
            "max_abs": 1.0,
            "mean_contribution": 0.2,
            "max_contribution": 3.0,
            "down_column_norm": 1.0,
            "down_norm_ratio_to_layer_median": 1.0,
        },
        {
            "layer": 0,
            "channel": 3,
            "sample_count": 100,
            "active_fraction": 0.01,
            "mean_abs": 0.1,
            "max_abs": 1.0,
            "mean_contribution": 5.0,
            "max_contribution": 10.0,
            "down_column_norm": 4.0,
            "down_norm_ratio_to_layer_median": 4.0,
        },
    ]
    watchlist = {
        "model_identity": "model-1",
        "entries": [
            {"layer": 0, "channel": 3, "contribution_limit": 20.0},
            {"layer": 1, "channel": 5, "contribution_limit": 1.0},
        ],
    }
    thresholds = {
        "max_active_fraction": 0.0,
        "max_mean_contribution": 1.0,
        "max_observed_contribution": 1.0,
        "min_down_norm_ratio": 2.0,
        "max_outlier_active_fraction": 0.02,
    }
    preview = example._candidate_preview(
        ranked, watchlist, thresholds, norm_outliers=[ranked[2]]
    )
    assert [
        (row["rank"], row["layer"], row["channel"]) for row in preview["entries"]
    ] == [(1, 1, 5), (2, 0, 2), (3, 0, 3)]
    assert preview["entries"][0]["selected_for_watch"] is True
    assert preview["entries"][0]["selection_reasons"] == ["quiet_activity"]
    assert preview["entries"][0]["contribution_limit"] == 1.0
    assert preview["entries"][1]["selected_for_watch"] is False
    assert preview["entries"][1]["contribution_limit"] is None
    assert preview["entries"][2]["selection_reasons"] == [
        "large_down_column_with_low_activity"
    ]
    assert preview["norm_outlier_entries"][0]["channel"] == 3
