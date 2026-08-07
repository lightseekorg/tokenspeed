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

"""Regression tests for single-snapshot Hugging Face config loading.

``get_config`` downloads once under the TokenSpeed cross-process lock and
parses ``config.json`` from that local snapshot, so ``--revision`` pins every
consumer at the same commit. A second remote lookup can silently construct
defaults, so every consumer must read from the resolved snapshot.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.llama_config import LlamaConfig
from tokenspeed.runtime.configs.utils import get_config

_MODULE = "tokenspeed.runtime.configs.utils"


class _RecordingLock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self):
        self.events.append("acquire")

    def __exit__(self, *exc_info):
        self.events.append("release")


def _write_llama_config(path: str) -> None:
    raw = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
    }
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as file:
        json.dump(raw, file)


class ConfigSnapshotResolutionTests(unittest.TestCase):
    def test_config_uses_revision_pinned_snapshot(self) -> None:
        events: list[str] = []

        with tempfile.TemporaryDirectory() as snapshot:
            _write_llama_config(snapshot)

            def download_snapshot(*args, **kwargs):
                events.append("snapshot")
                return snapshot

            with (
                patch(
                    "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                    return_value=_RecordingLock(events),
                ),
                patch(
                    f"{_MODULE}.snapshot_download", side_effect=download_snapshot
                ) as download,
            ):
                config = get_config("org/model", revision="revision-tag")

        download.assert_called_once_with(
            "org/model",
            revision="revision-tag",
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        # Parsing happens after the lock is released: config.json is read from
        # the already-resolved snapshot, never a second remote lookup.
        self.assertEqual(events, ["acquire", "snapshot", "release"])
        self.assertIsInstance(config, LlamaConfig)
        self.assertEqual(config.name_or_path, "org/model")

    def test_config_local_dir_does_not_resolve_remote_snapshot(self) -> None:
        with (
            tempfile.TemporaryDirectory() as snapshot,
            patch(
                f"{_MODULE}.snapshot_download",
                side_effect=AssertionError("local configs must not download"),
            ) as download,
        ):
            _write_llama_config(snapshot)
            config = get_config(snapshot)

        download.assert_not_called()
        self.assertIsInstance(config, LlamaConfig)
        self.assertEqual(config.name_or_path, snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
