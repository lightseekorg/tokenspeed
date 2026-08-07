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

"""Regression tests for single-snapshot Hugging Face tokenizer loading.

``get_tokenizer`` downloads once under the TokenSpeed cross-process lock and
parses from that local snapshot, so ``--revision`` pins every consumer at the
same commit. Custom tokenizer code is the one exception: it parses from the
original repo at the snapshot's immutable commit, still under the lock, so
Transformers can resolve sibling imports without racing a peer TP worker.
"""

import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.utils.tokenizer_utils import (
    _find_deepseek_v4_encoding_file,
    _snapshot_commit_hash,
    get_tokenizer,
)

_MODULE = "tokenspeed.runtime.utils.tokenizer_utils"


class _RecordingLock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self):
        self.events.append("acquire")

    def __exit__(self, *exc_info):
        self.events.append("release")


class TokenizerSnapshotResolutionTests(unittest.TestCase):
    def test_snapshot_commit_hash_requires_standard_hf_layout(self) -> None:
        commit = "c" * 40
        self.assertEqual(
            _snapshot_commit_hash(f"/cache/repo/snapshots/{commit}/"), commit
        )
        self.assertIsNone(_snapshot_commit_hash("/cache/repo/snapshots/main"))

    def test_tokenizer_uses_revision_pinned_snapshot(self) -> None:
        events: list[str] = []

        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})

        def download_snapshot(*args, **kwargs):
            events.append("snapshot")
            return "/resolved/tokenizer-snapshot"

        def load_tokenizer(*args, **kwargs):
            events.append("parse")
            return tokenizer

        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=_RecordingLock(events),
            ),
            patch(
                f"{_MODULE}.snapshot_download", side_effect=download_snapshot
            ) as download,
            patch(
                f"{_MODULE}.AutoTokenizer.from_pretrained",
                side_effect=load_tokenizer,
            ) as from_pretrained,
            patch(f"{_MODULE}.warnings.warn"),
        ):
            result = get_tokenizer(
                "org/tokenizer",
                trust_remote_code=False,
                revision="tokenizer-revision",
            )

        download.assert_called_once_with(
            "org/tokenizer",
            revision="tokenizer-revision",
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        from_pretrained.assert_called_once_with(
            "/resolved/tokenizer-snapshot",
            trust_remote_code=False,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(events, ["acquire", "snapshot", "release", "parse"])
        self.assertIs(result, tokenizer)
        self.assertEqual(result.name_or_path, "org/tokenizer")
        self.assertEqual(result.init_kwargs["name_or_path"], "org/tokenizer")

    def test_remote_code_tokenizer_uses_repo_and_commit_under_lock(self) -> None:
        commit = "b" * 40
        snapshot = f"/cache/models--org--tokenizer/snapshots/{commit}"
        events: list[str] = []

        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})

        def download_snapshot(*args, **kwargs):
            events.append("snapshot")
            return snapshot

        def load_tokenizer(*args, **kwargs):
            events.append("parse")
            return tokenizer

        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=_RecordingLock(events),
            ),
            patch(f"{_MODULE}.snapshot_download", side_effect=download_snapshot),
            patch(
                f"{_MODULE}.AutoTokenizer.from_pretrained",
                side_effect=load_tokenizer,
            ) as from_pretrained,
            patch(f"{_MODULE}.warnings.warn"),
        ):
            result = get_tokenizer(
                "org/tokenizer",
                trust_remote_code=True,
                revision="tokenizer-revision",
            )

        from_pretrained.assert_called_once_with(
            "org/tokenizer",
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
            revision=commit,
        )
        # The parse happens before the lock is released: a peer TP worker must
        # not write transformers_modules concurrently.
        self.assertEqual(events, ["acquire", "snapshot", "parse", "release"])
        self.assertIs(result, tokenizer)

    def test_remote_code_tokenizer_without_commit_falls_back_to_snapshot(self) -> None:
        events: list[str] = []

        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})

        def download_snapshot(*args, **kwargs):
            events.append("snapshot")
            return "/resolved/nonstandard-layout"

        def load_tokenizer(*args, **kwargs):
            events.append("parse")
            return tokenizer

        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=_RecordingLock(events),
            ),
            patch(f"{_MODULE}.snapshot_download", side_effect=download_snapshot),
            patch(
                f"{_MODULE}.AutoTokenizer.from_pretrained",
                side_effect=load_tokenizer,
            ) as from_pretrained,
            patch(f"{_MODULE}.logger.warning") as warn,
            patch(f"{_MODULE}.warnings.warn"),
        ):
            get_tokenizer(
                "org/tokenizer",
                trust_remote_code=True,
                revision="moving-branch",
            )

        from_pretrained.assert_called_once_with(
            "/resolved/nonstandard-layout",
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(events, ["acquire", "snapshot", "release", "parse"])
        warn.assert_called_once()

    def test_local_tokenizer_does_not_resolve_remote_snapshot(self) -> None:
        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})
        with (
            tempfile.TemporaryDirectory() as snapshot,
            patch(
                f"{_MODULE}.snapshot_download",
                side_effect=AssertionError("local tokenizers must not download"),
            ) as download,
            patch(
                f"{_MODULE}.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_pretrained,
            patch(f"{_MODULE}.warnings.warn"),
        ):
            get_tokenizer(snapshot, trust_remote_code=True)

        download.assert_not_called()
        from_pretrained.assert_called_once_with(
            snapshot,
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
        )

    def test_tokenizer_rejects_conflicting_revisions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            get_tokenizer(
                "org/tokenizer",
                tokenizer_revision="tokenizer-revision",
                revision="model-revision",
            )

    def test_revision_is_the_alias_callers_actually_pass(self) -> None:
        """``--revision`` reaches the download; ``tokenizer_revision`` is the
        internal name. Both engine call sites pass the former, so dropping it
        from the signature would silently strand it in ``**kwargs``."""
        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})
        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=nullcontext(),
            ),
            patch(
                f"{_MODULE}.snapshot_download", return_value="/resolved/snapshot"
            ) as download,
            patch(
                f"{_MODULE}.AutoTokenizer.from_pretrained", return_value=tokenizer
            ) as from_pretrained,
            patch(f"{_MODULE}.warnings.warn"),
        ):
            get_tokenizer("org/tokenizer", revision="pinned-tag")

        self.assertEqual(download.call_args.kwargs["revision"], "pinned-tag")
        # The snapshot path carries the pin; no revision kwarg leaks through.
        self.assertNotIn("revision", from_pretrained.call_args.kwargs)
        self.assertNotIn("tokenizer_revision", from_pretrained.call_args.kwargs)

    def test_deepseek_encoding_missing_from_snapshot_fails_locally(self) -> None:
        with (
            tempfile.TemporaryDirectory() as snapshot,
            patch(
                f"{_MODULE}.cached_file",
                side_effect=AssertionError("must not re-resolve a local snapshot"),
            ) as cached,
            self.assertRaisesRegex(RuntimeError, "encoding/encoding_dsv4.py"),
        ):
            _find_deepseek_v4_encoding_file(snapshot, None)

        cached.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
