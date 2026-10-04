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

"""Runtime offload configuration stays separate from model-specific options."""

import argparse
import json
import os
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class KVOffloadArgsTest(unittest.TestCase):
    def setUp(self):
        self.options = dict(layers=[1, 3], hot_tokens=4096, host_gb=1.5, overlap=True)

    def args(self, options, *, overrides="{}"):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = ["--model", "x", "--hf-overrides", overrides]
        if options is not None:
            argv.extend(["--kv-offload-config", options])
        with patch.object(ServerArgs, "__post_init__", return_value=None):
            args = ServerArgs.from_cli_args(parser.parse_args(argv))
        args.resolve_kv_offload()
        return args

    def test_cli_normalizes_once_and_preserves_worker_configuration(self):
        overrides = (
            '{"kv_offload_window_ring": true, "index_selection_deterministic": true}'
        )
        args = self.args(json.dumps(self.options), overrides=overrides)
        self.assertEqual(args.kv_offload_config, self.options)
        self.assertEqual(args.hf_overrides, overrides)
        self.assertEqual(
            pickle.loads(pickle.dumps(args)).kv_offload_config, self.options
        )
        args.resolve_kv_offload()
        self.assertEqual(args.kv_offload_config, self.options)

    def test_omission_disables_offload(self):
        self.assertIsNone(self.args(None).kv_offload_config)

    def test_invalid_deployment_options_are_rejected(self):
        invalid = [
            "{",
            "[]",
            "null",
            "{}",
            json.dumps(dict(self.options, window_ring=True)),
        ]
        for field, values in {
            "layers": [[], [1, 1], [-1], [True], "1", [[1]]],
            "hot_tokens": [0, 4095, True, "4096"],
            "host_gb": [0, -1, True, "1", float("nan"), float("inf")],
            "overlap": [1, "true", None],
        }.items():
            invalid.extend(json.dumps(dict(self.options, **{field: v})) for v in values)
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.args(config)

    def test_legacy_source_never_overrides_runtime_configuration(self):
        legacy = json.dumps({"kv_offload": dict(self.options, window_ring=True)})
        for options in [None, json.dumps(self.options)]:
            with self.subTest(options=options):
                args = self.args(options, overrides=legacy)
                self.assertEqual(
                    args.kv_offload_config,
                    None if options is None else self.options,
                )
                self.assertEqual(args.hf_overrides, legacy)

    def test_unsupported_recipe_rejects_enabled_configuration(self):
        recipe = SimpleNamespace(
            server_args=SimpleNamespace(kv_offload_config=self.options)
        )
        with self.assertRaisesRegex(ValueError, "does not support"):
            CacheRecipe.offload_policy(recipe)
        recipe.server_args.kv_offload_config = None
        self.assertIsNone(CacheRecipe.offload_policy(recipe))


if __name__ == "__main__":
    unittest.main()
