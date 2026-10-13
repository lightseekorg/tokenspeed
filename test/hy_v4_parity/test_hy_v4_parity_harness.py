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

"""Tests for the HY V4 parity harness itself (CPU only, no checkpoint download).

A parity run that silently compares nothing is worse than no run at all, so the
hook, the comparator and the truncation rewrite are all exercised here: the hook
is attached to a stand-in model, the comparator is fed dumps that differ in one
tap, and the config rewrite is checked against a synthetic per-layer vector.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test.hy_v4_parity.hy_v4_fixtures import (  # noqa: E402
    Truncation,
    _keep,
    truncated_config,
)
from test.hy_v4_parity.hy_v4_parity_hook import (  # noqa: E402
    LayerDumper,
    ModelDumpSession,
    first_divergence,
    format_report,
    worst_status,
)


class _SubBlock(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


class _Layer(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.self_attn = _SubBlock(scale)
        self.mlp = _SubBlock(scale)
        self.hc_attn_layer = _SubBlock(1.0)
        self.hc_mlp_layer = _SubBlock(1.0)


class _StandInModel(torch.nn.Module):
    def __init__(self, num_layers: int, scale: float = 1.0) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(scale) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer.hc_attn_layer(hidden_states)
            hidden_states = layer.self_attn(hidden_states)
            hidden_states = layer.hc_mlp_layer(hidden_states)
            hidden_states = layer.mlp(hidden_states)
        return hidden_states


class LayerDumperTests(unittest.TestCase):
    def _run(self, model: _StandInModel) -> LayerDumper:
        dumper = LayerDumper()
        with ModelDumpSession(dumper) as session:
            session.attach(model, num_layers=len(model.layers))
            model(torch.ones(4, 8))
        return dumper

    def test_records_every_tapped_sub_block(self):
        dumper = self._run(_StandInModel(num_layers=2))
        for layer_id in (0, 1):
            for stage in ("hc_attn", "attn", "hc_mlp", "mlp"):
                for position in ("in", "out"):
                    key = f"p0_layer{layer_id}_{stage}_{position}"
                    self.assertIn(key, dumper.records)
        self.assertEqual(len(dumper.records), 2 * 4 * 2)

    def test_captures_shape_and_values(self):
        model = _StandInModel(num_layers=1, scale=3.0)
        dumper = self._run(model)
        record = dumper.records["p0_layer0_attn_in"]
        self.assertEqual(record["shape"], [4, 8])
        self.assertEqual(record["dtype"], "float32")
        self.assertAlmostEqual(record["abs_max"], 1.0)
        record_out = dumper.records["p0_layer0_attn_out"]
        self.assertAlmostEqual(record_out["abs_max"], 3.0)

    def test_hooks_are_removed_on_detach(self):
        model = _StandInModel(num_layers=1)
        dumper = LayerDumper()
        with ModelDumpSession(dumper) as session:
            session.attach(model, num_layers=1)
        after_detach = len(dumper.records)
        model(torch.ones(2, 4))
        self.assertEqual(len(dumper.records), after_detach)

    def test_missing_boundary_is_skipped(self):
        class _PlainLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = _SubBlock(1.0)
                self.mlp = _SubBlock(1.0)

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.mlp(self.self_attn(hidden_states))

        class _PlainModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([_PlainLayer()])

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                return hidden_states

        dumper = LayerDumper()
        model = _PlainModel()
        with ModelDumpSession(dumper) as session:
            session.attach(model, num_layers=1)
            model(torch.ones(2, 4))
        self.assertTrue(any("attn" in key for key in dumper.records))
        self.assertFalse(any("hc_" in key for key in dumper.records))

    def test_pass_counter_separates_prefill_and_decode(self):
        model = _StandInModel(num_layers=1)
        dumper = LayerDumper()
        with ModelDumpSession(dumper) as session:
            session.attach(model, num_layers=1)
            model(torch.ones(2, 4))
            session.mark_pass()
            model(torch.ones(2, 4))
        self.assertTrue(any(key.startswith("p0_") for key in dumper.records))
        self.assertTrue(any(key.startswith("p1_") for key in dumper.records))


class ComparatorTests(unittest.TestCase):
    def _dump(self, records: dict) -> Path:
        handle, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(handle, "w") as stream:
            json.dump(records, stream)
        self.addCleanup(os.unlink, path)
        return Path(path)

    def _record(self, value: float, shape=None) -> dict:
        return {
            "shape": shape or [4, 8],
            "dtype": "float32",
            "abs_max": value,
            "abs_mean": value,
            "norm": value,
            "sample": [[value] * 8] * 4,
            "sample_tokens": list(range(4)),
            "sample_channels": list(range(8)),
        }

    def test_identical_dumps_pass(self):
        records = {"p0_layer0_attn_out": self._record(1.0)}
        report = first_divergence(records, records, atol=1e-3, rtol=1e-3)
        self.assertEqual(worst_status(report)[0], "ok")

    def test_bf16_scale_noise_is_tolerated(self):
        # A sub-percent wiggle is what bf16 kernel noise looks like; it must not
        # be reported, or the harness would be unusable on real runs.
        reference = {"k": self._record(100.0)}
        candidate = {"k": self._record(100.05)}
        self.assertEqual(
            worst_status(first_divergence(reference, candidate, atol=1e-4, rtol=1e-3))[
                0
            ],
            "ok",
        )

    def test_divergence_is_flagged_and_ranked_first(self):
        reference = {
            "p0_layer0_attn_out": self._record(1.0),
            "p0_layer1_attn_out": self._record(1.0),
        }
        candidate = {
            "p0_layer0_attn_out": self._record(1.0),
            "p0_layer1_attn_out": self._record(2.0),
        }
        report = first_divergence(reference, candidate, atol=1e-3, rtol=1e-3)
        self.assertEqual(worst_status(report)[0], "diverge")
        # The diverging tap must sort ahead of the matching one.
        self.assertEqual(report[0]["key"], "p0_layer1_attn_out")
        self.assertEqual(report[0]["status"], "diverge")
        self.assertEqual(report[1]["status"], "ok")
        self.assertIn("1/2", format_report(report))

    def test_whole_tensor_norm_breach_is_caught(self):
        # A difference that lies outside the sampled corner must still fail, via
        # the whole-tensor norm.
        reference = {"k": self._record(1.0)}
        candidate = {"k": self._record(1.0)}
        candidate["k"]["norm"] = reference["k"]["norm"] * 1.5
        report = first_divergence(reference, candidate, atol=1e-3, rtol=1e-3)
        self.assertEqual(worst_status(report)[0], "diverge")
        self.assertEqual(report[0]["norm_delta"], 0.5)

    def test_peak_breach_is_caught(self):
        reference = {"k": self._record(1.0)}
        candidate = {"k": self._record(1.0)}
        candidate["k"]["abs_max"] = 4.0
        report = first_divergence(reference, candidate, atol=1e-3, rtol=1e-3)
        self.assertEqual(worst_status(report)[0], "diverge")

    def test_shape_mismatch_is_detected(self):
        reference = {"k": self._record(1.0)}
        candidate = {"k": self._record(1.0, shape=[4, 4])}
        report = first_divergence(reference, candidate, atol=1e-3, rtol=1e-3)
        self.assertEqual(report[0]["status"], "shape")

    def test_missing_tap_is_detected(self):
        report = first_divergence({"a": self._record(1.0)}, {}, atol=1e-3, rtol=1e-3)
        self.assertEqual(report[0]["status"], "missing")

    def test_tolerance_scales_with_magnitude(self):
        # The same absolute deviation must pass on a large activation and fail on
        # a tiny one, because tolerance is atol + rtol * |reference|.
        atol, rtol = 1e-6, 1e-3
        large_ref = {"k": self._record(100.0)}
        large_cand = {"k": self._record(100.001)}
        self.assertEqual(
            worst_status(first_divergence(large_ref, large_cand, atol=atol, rtol=rtol))[
                0
            ],
            "ok",
        )
        small_ref = {"k": self._record(0.001)}
        small_cand = {"k": self._record(0.002)}
        self.assertEqual(
            worst_status(first_divergence(small_ref, small_cand, atol=atol, rtol=rtol))[
                0
            ],
            "diverge",
        )


class TruncationTests(unittest.TestCase):
    def test_keep_selects_globals_layers_and_mtp(self):
        truncation = Truncation(num_layers=3)
        self.assertTrue(_keep("model.embed_tokens.weight", truncation))
        self.assertTrue(_keep("model.norm.weight", truncation))
        self.assertTrue(_keep("model.hc_head.hc_head_fn", truncation))
        self.assertTrue(_keep("lm_head.weight", truncation))
        self.assertTrue(_keep("model.mtp_layers.0.eh_proj.weight", truncation))
        self.assertTrue(_keep("model.layers.2.mlp.gate.weight", truncation))
        self.assertFalse(_keep("model.layers.3.mlp.gate.weight", truncation))

    def test_config_rewrite_slices_every_per_layer_vector(self):
        config = {
            "num_hidden_layers": 6,
            "n_routed_experts": 256,
            "mlp_layer_types": [
                "dense",
                "sparse",
                "sparse",
                "sparse",
                "sparse",
                "sparse",
            ],
            "layer_types": ["a"] * 6,
            "indexer_types": ["full", "shared", "full", "shared", "full", "shared"],
            "hidden_size": 6144,
        }
        updated = truncated_config(config, Truncation(num_layers=3, num_experts=8))
        self.assertEqual(updated["num_hidden_layers"], 3)
        self.assertEqual(updated["mlp_layer_types"], ["dense", "sparse", "sparse"])
        self.assertEqual(updated["layer_types"], ["a"] * 3)
        self.assertEqual(updated["indexer_types"], ["full", "shared", "full"])
        self.assertEqual(updated["n_routed_experts"], 8)
        # Scalars must survive untouched.
        self.assertEqual(updated["hidden_size"], 6144)
        # The source dict must not be mutated.
        self.assertEqual(config["num_hidden_layers"], 6)

    def test_rejects_out_of_range_truncation(self):
        config = {"num_hidden_layers": 4, "n_routed_experts": 8}
        with self.assertRaises(ValueError):
            truncated_config(config, Truncation(num_layers=5))
        with self.assertRaises(ValueError):
            truncated_config(config, Truncation(num_layers=1, num_experts=9))

    def test_released_config_geometry_survives_truncation(self):
        """The family's kernel-relevant geometry must not be edited away."""
        released = {
            "num_hidden_layers": 78,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "hc_mult": 4,
            "mlp_layer_types": ["dense"] + ["sparse"] * 77,
            "indexer_types": [
                "full" if i == 0 or (i - 1) % 4 == 0 else "shared" for i in range(78)
            ],
        }
        updated = truncated_config(released, Truncation(num_layers=3, num_experts=8))
        for key in (
            "num_experts_per_tok",
            "qk_nope_head_dim",
            "kv_lora_rank",
            "hc_mult",
        ):
            self.assertEqual(updated[key], released[key])


class DraftGraphTensorEvidenceTests(unittest.TestCase):
    """Exercise persisted evidence, not only the in-memory tensor comparator."""

    def _compare(self, mutation=None):
        from test.hy_v4_parity.draft_graph_check import _compare_tensor_evidence
        from types import SimpleNamespace

        def payload():
            return {
                "inputs": {
                    name: torch.ones(2, 3)
                    for name in (
                        "input_ids",
                        "positions",
                        "hidden",
                        "gather_ids",
                        "frontier",
                    )
                },
                "outputs": {
                    name: torch.ones(2, 3)
                    for name in ("logits", "latent_kv", "index_k")
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            graph_file = Path(directory) / "graph.pt"
            eager_file = Path(directory) / "eager.pt"
            graph_payload = payload()
            if mutation == "kv":
                graph_payload["outputs"]["latent_kv"][1, 2] = 10.0
            elif mutation == "inputs":
                graph_payload["inputs"]["frontier"][1, 2] = 9.0
            elif mutation == "missing_tensor":
                del graph_payload["outputs"]["index_k"]
            if mutation != "missing_file":
                torch.save(graph_payload, graph_file)
            torch.save(payload(), eager_file)
            graph = [{"first_step": {"path": "graph", "artifact": str(graph_file)}}]
            eager = [{"first_step": {"path": "eager", "artifact": str(eager_file)}}]
            tolerances = SimpleNamespace(
                logit_atol=0.05, logit_rtol=0.01, kv_atol=0.004, kv_rtol=0.02
            )
            return _compare_tensor_evidence(graph, eager, tolerances)

    def test_identical_saved_evidence_passes(self):
        self.assertEqual(self._compare(), [])

    def test_missing_artifact_fails(self):
        errors = self._compare("missing_file")
        self.assertTrue(any("unreadable tensor evidence" in error for error in errors))

    def test_saved_kv_corruption_fails(self):
        errors = self._compare("kv")
        self.assertTrue(any("latent_kv" in error for error in errors))

    def test_saved_inputs_and_tensor_coverage_are_checked(self):
        for mutation in ("inputs", "missing_tensor"):
            with self.subTest(mutation=mutation):
                self.assertTrue(self._compare(mutation))


class ServingBenchmarkStreamTests(unittest.TestCase):
    """Client metrics need visible output and actual usage, not requested counts."""

    def _run_stream(self, frames):
        import asyncio
        from test.hy_v4_parity import serve_bench
        from types import SimpleNamespace
        from unittest import mock

        clock = [100.0]

        class Content:
            async def __aiter__(self):
                for when, payload in frames:
                    clock[0] = 100.0 + when
                    data = payload if isinstance(payload, str) else json.dumps(payload)
                    yield f"data: {data}\n".encode()

        class Response:
            status = 200
            content = Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        session = SimpleNamespace(post=lambda *args, **kwargs: Response())
        with mock.patch.object(serve_bench.time, "perf_counter", lambda: clock[0]):
            return asyncio.run(
                serve_bench.one_request(session, "unused", "model", "prompt", 2)
            )

    def _frames(self):
        return [
            (1.0, {"choices": [{"text": ""}]}),
            (5.0, {"choices": [{"text": "a"}]}),
            (7.0, {"choices": [{"text": "b"}]}),
            (10.0, {"usage": {"completion_tokens": 2, "prompt_tokens": 3}}),
            (11.0, "[DONE]"),
        ]

    def test_empty_frames_and_delayed_usage_do_not_bias_token_latency(self):
        self.assertEqual(self._run_stream(self._frames()), (5.0, 2.0, 11.0, 2, 3))

    def test_whitespace_is_visible_output_not_an_empty_frame(self):
        frames = self._frames()
        frames[1] = (5.0, {"choices": [{"text": " "}]})
        self.assertEqual(self._run_stream(frames), (5.0, 2.0, 11.0, 2, 3))

    def test_incomplete_or_unaccounted_responses_fail(self):
        cases = [
            "usage_missing",
            "done_missing",
            "zero",
            "short",
            "float",
            "boolean",
            "error",
            "empty",
        ]
        for case in cases:
            with self.subTest(case=case):
                frames = self._frames()
                if case == "usage_missing":
                    frames.pop(3)
                elif case == "done_missing":
                    frames.pop()
                elif case in {"zero", "short", "float", "boolean"}:
                    value = {"zero": 0, "short": 1, "float": 2.0, "boolean": True}[case]
                    frames[3][1]["usage"]["completion_tokens"] = value
                elif case == "error":
                    frames.insert(1, (2.0, {"error": {"message": "generation failed"}}))
                else:
                    frames[1] = (5.0, {"choices": [{"text": ""}]})
                    frames[2] = (7.0, {"choices": [{"text": ""}]})
                with self.assertRaises(RuntimeError):
                    self._run_stream(frames)


class TruncatedCheckpointSafetyTests(unittest.TestCase):
    def _small_source(self, root):
        from safetensors.torch import save_file

        source = root / "source"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps({"num_hidden_layers": 1, "n_routed_experts": 8})
        )
        key = "model.embed_tokens.weight"
        save_file({key: torch.ones(2, 2)}, str(source / "source.safetensors"))
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "source.safetensors"}})
        )
        return source

    def test_nonempty_output_requires_explicit_overwrite(self):
        from test.hy_v4_parity import hy_v4_fixtures as fixtures
        from unittest import mock

        for marker_kind in ("missing", "mismatched", "incomplete"):
            with self.subTest(marker_kind=marker_kind):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = self._small_source(root)
                    target = root / "output"
                    target.mkdir()
                    (target / "config.json").write_text("previous config")
                    (target / "model.safetensors").write_bytes(b"previous weights")
                    if marker_kind != "missing":
                        (target / ".hy_v4_truncated.json").write_text(
                            json.dumps(
                                {
                                    "repo_id": str(source),
                                    "truncation": (
                                        "1L/allE"
                                        if marker_kind == "incomplete"
                                        else "1L/4E"
                                    ),
                                }
                            )
                        )
                    before = {p.name: p.read_bytes() for p in target.iterdir()}
                    with mock.patch("safetensors.torch.save_file") as save:
                        with self.assertRaisesRegex(FileExistsError, "overwrite"):
                            fixtures.build_truncated_checkpoint(
                                target, Truncation(1, None), repo_id=str(source)
                            )
                        save.assert_not_called()
                    self.assertEqual(
                        before, {p.name: p.read_bytes() for p in target.iterdir()}
                    )

    def test_empty_directory_export_reuse_and_explicit_rebuild(self):
        from test.hy_v4_parity import hy_v4_fixtures as fixtures
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._small_source(root)
            target = root / "output"
            target.mkdir()
            selection = Truncation(1, None)
            fixtures.build_truncated_checkpoint(target, selection, repo_id=str(source))
            before = {p.name: p.read_bytes() for p in target.iterdir()}
            with mock.patch("safetensors.torch.save_file") as save:
                self.assertEqual(
                    fixtures.build_truncated_checkpoint(
                        target, selection, repo_id=str(source)
                    ),
                    target,
                )
                save.assert_not_called()
            fixtures.build_truncated_checkpoint(
                target, selection, repo_id=str(source), overwrite=True
            )
            self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})

    def test_output_symlink_cannot_modify_another_directory(self):
        from test.hy_v4_parity import hy_v4_fixtures as fixtures

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._small_source(root)
            unrelated = root / "other_checkpoint"
            unrelated.mkdir()
            sentinel = unrelated / "config.json"
            sentinel.write_text("must not change")
            target = root / "output"
            target.symlink_to(unrelated, target_is_directory=True)
            for overwrite in (False, True):
                with self.subTest(overwrite=overwrite):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        fixtures.build_truncated_checkpoint(
                            target,
                            Truncation(1, None),
                            repo_id=str(source),
                            overwrite=overwrite,
                        )
                    self.assertEqual(sentinel.read_text(), "must not change")

    def test_rebuild_cannot_remove_the_source_or_its_parent(self):
        from test.hy_v4_parity import hy_v4_fixtures as fixtures
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            sentinel = source / "keep.txt"
            sentinel.write_text("source data")
            alias = root / "alias"
            alias.symlink_to(source, target_is_directory=True)
            for target in (source, root, alias):
                with self.subTest(target=target.name):
                    with mock.patch.object(fixtures.shutil, "rmtree") as remove:
                        with self.assertRaisesRegex(ValueError, "source checkpoint"):
                            fixtures.build_truncated_checkpoint(
                                target, repo_id=str(source), overwrite=True
                            )
                        remove.assert_not_called()
                    self.assertEqual(sentinel.read_text(), "source data")

    def test_invalid_truncation_keeps_the_previous_export(self):
        from test.hy_v4_parity import hy_v4_fixtures as fixtures
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "output"
            source.mkdir()
            target.mkdir()
            (source / "config.json").write_text(
                json.dumps({"num_hidden_layers": 2, "n_routed_experts": 8})
            )
            sentinel = target / "keep.txt"
            sentinel.write_text("previous export")
            with mock.patch.object(fixtures.shutil, "rmtree") as remove:
                with self.assertRaisesRegex(ValueError, "num_layers"):
                    fixtures.build_truncated_checkpoint(
                        target, Truncation(3, 8), repo_id=str(source), overwrite=True
                    )
                remove.assert_not_called()
            self.assertEqual(sentinel.read_text(), "previous export")


if __name__ == "__main__":
    unittest.main()
