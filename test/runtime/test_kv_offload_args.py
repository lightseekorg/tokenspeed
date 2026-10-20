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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class TestKVOffloadArgsTest:
    def setup_method(self):
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
        assert (args.kv_offload_config) == (self.options)
        assert (args.hf_overrides) == (overrides)
        assert (pickle.loads(pickle.dumps(args)).kv_offload_config) == (self.options)
        args.resolve_kv_offload()
        assert (args.kv_offload_config) == (self.options)

    def test_offloading_disables_snapshot_allocation_without_extra_flags(self):
        args = ServerArgs(
            model="x",
            disaggregation_mode="decode",
            enable_kvstore=False,
            kv_offload_config=self.options,
        )
        assert args.retraction_snapshot_pool_disabled
        assert (args.retraction_snapshot_host_gb) == (0)
        assert (args.retraction_snapshot_max_requests) == (0)

    def test_offloading_rejects_recovery_options_instead_of_ignoring_them(
        self, subtests
    ):
        for override in (
            dict(retraction_snapshot_host_gb=1),
            dict(retraction_snapshot_ratio=0.5),
            dict(retraction_snapshot_max_requests=2),
            dict(debug_force_retraction_interval=1),
        ):
            with subtests.test(override=override):
                with pytest.raises((ValueError), match=("KV offloading")):
                    ServerArgs(
                        model="x",
                        disaggregation_mode="decode",
                        enable_kvstore=False,
                        kv_offload_config=self.options,
                        **override,
                    )

        args = ServerArgs(
            model="x",
            disaggregation_mode="decode",
            enable_kvstore=False,
            kv_offload_config=self.options,
            retraction_snapshot_ratio=0,
        )
        assert args.retraction_snapshot_pool_disabled

    def test_omission_disables_offload(self):
        assert (self.args(None).kv_offload_config) is None

    def test_invalid_deployment_options_are_rejected(self, subtests):
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
            with subtests.test(config=config), pytest.raises((ValueError)):
                self.args(config)

    def test_legacy_source_never_overrides_runtime_configuration(self, subtests):
        legacy = json.dumps({"kv_offload": dict(self.options, window_ring=True)})
        for options in [None, json.dumps(self.options)]:
            with subtests.test(options=options):
                args = self.args(options, overrides=legacy)
                assert (args.kv_offload_config) == (
                    None if options is None else self.options
                )
                assert (args.hf_overrides) == (legacy)

    def test_unsupported_recipe_rejects_enabled_configuration(self):
        recipe = SimpleNamespace(
            server_args=SimpleNamespace(kv_offload_config=self.options)
        )
        with pytest.raises((ValueError), match=("does not support")):
            CacheRecipe.offload_policy(recipe)
        recipe.server_args.kv_offload_config = None
        assert (CacheRecipe.offload_policy(recipe)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
