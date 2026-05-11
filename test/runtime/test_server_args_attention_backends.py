"""Regression tests for --attention-backend / --drafter-attention-backend choices.

Guards against the bug where --drafter-attention-backend rejected valid main-model
backends (e.g. trtllm_mla) because its argparse `choices` was a narrower subset
of --attention-backend's.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs, prepare_server_args


class TestAttentionBackendChoices(unittest.TestCase):
    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser

    def _action(self, parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
        for action in parser._actions:
            if action.dest == dest:
                return action
        raise AssertionError(f"no action with dest={dest!r}")

    def test_attention_backend_accepts_trtllm_mla(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--attention-backend", "trtllm_mla"]
        )
        self.assertEqual(args.attention_backend, "trtllm_mla")

    def test_attention_backend_accepts_fa4(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--attention-backend", "fa4"]
        )
        self.assertEqual(args.attention_backend, "fa4")

    def test_drafter_attention_backend_accepts_fa4(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--drafter-attention-backend", "fa4"]
        )
        self.assertEqual(args.drafter_attention_backend, "fa4")

    def test_drafter_attention_backend_accepts_trtllm_mla(self):
        """Regression: trtllm_mla must be accepted here too."""
        args = self._build_parser().parse_args(
            ["--model", "x", "--drafter-attention-backend", "trtllm_mla"]
        )
        self.assertEqual(args.drafter_attention_backend, "trtllm_mla")

    def test_drafter_choices_match_main_choices(self):
        parser = self._build_parser()
        main = set(self._action(parser, "attention_backend").choices)
        drafter = set(self._action(parser, "drafter_attention_backend").choices)
        self.assertEqual(main, drafter)

    def test_invalid_backend_rejected_on_both_flags(self):
        for flag in ("--attention-backend", "--drafter-attention-backend"):
            parser = self._build_parser()
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--model", "x", flag, "bogus"])

    def test_inline_detokenizer_flag_removed_from_cli(self):
        parser = self._build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--model", "x", "--enable-inline-detokenizer"])

    def test_inline_detokenizer_is_forced_on(self):
        args = prepare_server_args(["--model", "x"])
        self.assertTrue(args.enable_inline_detokenizer)

    def test_defaults_to_mha_for_mha(self):
        self.assertEqual(registry._get_default_backend_name(AttentionArch.MHA), "mha")

    def test_fa4_resolves_to_mha_backend(self):
        from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend

        self.assertIs(
            registry._get_backend_cls("fa4", AttentionArch.MHA), MHAAttnBackend
        )

    def test_sm90_defaults_to_flashmla_for_mla(self):
        platform = SimpleNamespace(is_blackwell=False, is_hopper=True)
        with mock.patch.object(registry, "current_platform", return_value=platform):
            self.assertEqual(
                registry._get_default_backend_name(AttentionArch.MLA), "flashmla"
            )

    def test_mha_config_propagates_speculative_settings(self):
        server_args = SimpleNamespace(
            device="cuda",
            attention_backend=None,
            drafter_attention_backend=None,
            attn_tp_size=None,
            mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=2, dp_size=1)),
            kv_cache_dtype="auto",
            max_num_seqs=8,
            data_parallel_size=None,
            block_size=64,
            max_cudagraph_capture_size=4,
            kv_cache_quant_method="none",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        model_config = SimpleNamespace(
            context_len=4096,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
        )

        config = MHAConfig.generate(server_args, model_config)

        self.assertEqual(config.speculative_num_steps, 3)
        self.assertEqual(config.speculative_num_draft_tokens, 4)


if __name__ == "__main__":
    unittest.main()
