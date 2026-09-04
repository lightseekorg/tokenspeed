"""Tests for NextN/MTP draft shard preselection.

Draft models embedded in the target checkpoint expose
``checkpoint_weight_name_filter``; the loader uses it with the safetensors
index to skip shards holding no draft weights (see DefaultModelLoader).
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
from tokenspeed.runtime.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_safetensors_files_by_weight_names,
)


class TestFilterDuplicateSafetensorsFiles(unittest.TestCase):
    def test_keeps_modelopt_input_scales_outside_weight_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = os.path.join(tmpdir, "model-00001-of-00001.safetensors")
            scales = os.path.join(tmpdir, "input_scales.safetensors")
            duplicate = os.path.join(tmpdir, "consolidated.safetensors")
            for path in (shard, scales, duplicate):
                with open(path, "wb"):
                    pass
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"), "w"
            ) as index:
                json.dump(
                    {"weight_map": {"model.weight": os.path.basename(shard)}}, index
                )

            kept = filter_duplicate_safetensors_files(
                [shard, scales, duplicate],
                tmpdir,
                "model.safetensors.index.json",
            )

            self.assertEqual(kept, [shard, scales])


class TestFilterSafetensorsFilesByWeightNames(unittest.TestCase):
    def _write_index(self, tmpdir, weight_map):
        index_path = os.path.join(tmpdir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump({"weight_map": weight_map}, f)
        return index_path

    def _touch(self, tmpdir, *names):
        paths = []
        for name in names:
            path = os.path.join(tmpdir, name)
            with open(path, "wb"):
                pass
            paths.append(path)
        return paths

    def test_keeps_only_matching_shards_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._touch(tmpdir, "a.safetensors", "b.safetensors")
            self._write_index(
                tmpdir,
                {
                    "model.layers.0.w": "a.safetensors",
                    "mtp.fc.weight": "b.safetensors",
                    "mtp.norm.weight": "b.safetensors",
                },
            )
            kept = filter_safetensors_files_by_weight_names(
                files, tmpdir, "model.safetensors.index.json", lambda n: "mtp" in n
            )
            self.assertEqual(kept, files[1:])

    def test_keeps_shards_absent_from_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._touch(tmpdir, "a.safetensors", "extra.safetensors")
            self._write_index(tmpdir, {"mtp.fc.weight": "a.safetensors"})
            kept = filter_safetensors_files_by_weight_names(
                files, tmpdir, "model.safetensors.index.json", lambda n: "mtp" in n
            )
            self.assertEqual(kept, files)

    def test_missing_index_returns_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._touch(tmpdir, "a.safetensors")
            kept = filter_safetensors_files_by_weight_names(
                files, tmpdir, "model.safetensors.index.json", lambda n: False
            )
            self.assertEqual(kept, files)

    def test_no_match_falls_back_to_all_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._touch(tmpdir, "a.safetensors")
            self._write_index(tmpdir, {"model.layers.0.w": "a.safetensors"})
            kept = filter_safetensors_files_by_weight_names(
                files, tmpdir, "model.safetensors.index.json", lambda n: "mtp" in n
            )
            self.assertEqual(kept, files)


class TestLoaderSkipsFilteredShards(unittest.TestCase):
    def test_get_all_weights_reads_only_draft_shards(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(
                {"model.layers.0.w": torch.zeros(2)},
                os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            )
            save_file(
                {"model.mtp.fc.weight": torch.zeros(2)},
                os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
            )
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(
                    {
                        "weight_map": {
                            "model.layers.0.w": "model-00001-of-00002.safetensors",
                            "model.mtp.fc.weight": "model-00002-of-00002.safetensors",
                        }
                    },
                    f,
                )

            loader = DefaultModelLoader(LoadConfig())
            model = SimpleNamespace(
                checkpoint_weight_name_filter=lambda name: "mtp" in name,
                fall_back_to_pt_during_load=False,
                secondary_weights=(),
            )
            model_config = SimpleNamespace(model_path=tmpdir, revision=None)

            names = [name for name, _ in loader._get_all_weights(model_config, model)]
            self.assertEqual(names, ["model.mtp.fc.weight"])

    def test_target_model_without_filter_reads_everything(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(
                {"model.layers.0.w": torch.zeros(2)},
                os.path.join(tmpdir, "model.safetensors"),
            )
            loader = DefaultModelLoader(LoadConfig())
            model = SimpleNamespace(
                fall_back_to_pt_during_load=False,
                secondary_weights=(),
            )
            model_config = SimpleNamespace(model_path=tmpdir, revision=None)

            names = [name for name, _ in loader._get_all_weights(model_config, model)]
            self.assertEqual(names, ["model.layers.0.w"])


class TestNextNModelFilters(unittest.TestCase):
    """The predicates must accept every name each ``load_weights`` consumes."""

    def test_qwen3_5_nextn(self):
        from tokenspeed.runtime.models.qwen3_5_nextn import (
            Qwen3_5ForConditionalGenerationNextN,
        )

        model = object.__new__(Qwen3_5ForConditionalGenerationNextN)
        self.assertTrue(model.checkpoint_weight_name_filter("mtp.fc.weight"))
        self.assertTrue(
            model.checkpoint_weight_name_filter("model.mtp.self_attn.q_proj.weight")
        )
        self.assertFalse(
            model.checkpoint_weight_name_filter("model.layers.0.mlp.gate.weight")
        )

    def test_deepseek_nextn(self):
        from tokenspeed.runtime.models.deepseek_nextn import DeepseekV3ForCausalLMNextN

        model = object.__new__(DeepseekV3ForCausalLMNextN)
        model.config = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=61)
        self.assertTrue(
            model.checkpoint_weight_name_filter("model.layers.61.eh_proj.weight")
        )
        self.assertFalse(
            model.checkpoint_weight_name_filter("model.layers.60.mlp.gate.weight")
        )
        self.assertFalse(
            model.checkpoint_weight_name_filter("model.embed_tokens.weight")
        )

        # Standalone draft checkpoint keeps everything.
        model.config = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=1)
        self.assertTrue(
            model.checkpoint_weight_name_filter("model.layers.0.eh_proj.weight")
        )

    def test_glm_moe_dsa_nextn(self):
        from tokenspeed.runtime.models.glm_moe_dsa_nextn import (
            GlmMoeDsaForCausalLMNextN,
        )

        model = object.__new__(GlmMoeDsaForCausalLMNextN)
        model.config = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=92)
        self.assertTrue(
            model.checkpoint_weight_name_filter("model.layers.92.eh_proj.weight")
        )
        self.assertFalse(
            model.checkpoint_weight_name_filter("model.layers.10.mlp.gate.weight")
        )
        self.assertFalse(model.checkpoint_weight_name_filter("lm_head.weight"))

    def test_kimi_k3_nextn(self):
        from tokenspeed.runtime.models.kimi_k3_nextn import (
            KimiK3ForConditionalGenerationNextN,
            KimiK3NextNForCausalLM,
        )

        inner = object.__new__(KimiK3NextNForCausalLM)
        inner.config = SimpleNamespace(num_hidden_layers=48)
        self.assertTrue(
            inner.checkpoint_weight_name_filter("model.layers.48.eh_proj.weight")
        )
        self.assertFalse(
            inner.checkpoint_weight_name_filter("model.layers.47.mlp.gate.weight")
        )

        wrapper = object.__new__(KimiK3ForConditionalGenerationNextN)
        # Bypass nn.Module.__setattr__: the wrapper is deliberately not
        # initialized as a module here.
        object.__setattr__(wrapper, "language_model", inner)
        self.assertTrue(
            wrapper.checkpoint_weight_name_filter(
                "language_model.model.layers.48.eh_proj.weight"
            )
        )
        self.assertFalse(
            wrapper.checkpoint_weight_name_filter("model.layers.48.eh_proj.weight")
        )

    def test_inkling_nextn(self):
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )

        model = object.__new__(InklingForConditionalGenerationNextN)
        self.assertTrue(
            model.checkpoint_weight_name_filter("model.mtp.layers.0.qkv.weight")
        )
        self.assertTrue(
            model.checkpoint_weight_name_filter("model.llm.embed_norm.weight")
        )
        self.assertFalse(
            model.checkpoint_weight_name_filter("model.layers.5.qkv.weight")
        )


if __name__ == "__main__":
    unittest.main()
