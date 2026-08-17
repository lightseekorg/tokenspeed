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

    def test_attention_backend_accepts_mla(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--attention-backend", "mla"]
        )
        self.assertEqual(args.attention_backend, "mla")

    def test_attention_backend_accepts_mha_kernel_solutions(self):
        for backend in ("fa3", "fa4", "triton", "flashinfer"):
            args = self._build_parser().parse_args(
                ["--model", "x", "--attention-backend", backend]
            )
            self.assertEqual(args.attention_backend, backend)

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
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(["--model", "x", flag, "bogus"])

    def test_inline_detokenizer_flag_removed_from_cli(self):
        parser = self._build_parser()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(["--model", "x", "--enable-inline-detokenizer"])

    def test_inline_detokenizer_is_forced_on(self):
        args = prepare_server_args(["--model", "x"])
        self.assertTrue(args.enable_inline_detokenizer)

    def test_model_path_alias_sets_model(self):
        args = self._build_parser().parse_args(["--model-path", "x"])
        self.assertEqual(args.model, "x")

    def test_prepare_server_args_accepts_model_path_alias(self):
        args = prepare_server_args(["--model-path", "x"])
        self.assertEqual(args.model, "x")

    def test_defaults_to_mha_for_mha(self):
        self.assertEqual(registry._get_default_backend_name(AttentionArch.MHA), "mha")

    def test_mha_kernel_solution_backends_use_mha_backend(self):
        from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend

        for backend in ("mha", "fa3", "fa4", "triton", "flashinfer"):
            self.assertIs(
                registry._get_backend_cls(backend, AttentionArch.MHA),
                MHAAttnBackend,
            )

    def test_mla_backend_registered_for_mla(self):
        from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend

        self.assertIs(
            registry._get_backend_cls("mla", AttentionArch.MLA),
            MLAAttnBackend,
        )

    def test_defaults_to_mla_for_mla(self):
        self.assertEqual(registry._get_default_backend_name(AttentionArch.MLA), "mla")

    def test_lcm_kernel_page_size_is_validated_without_rewriting_it(self):
        config = SimpleNamespace(kernel_page_size=64)

        registry._validate_lcm_page_size(config, prefix_granularity=128)

        self.assertEqual(config.kernel_page_size, 64)
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            registry._validate_lcm_page_size(
                SimpleNamespace(kernel_page_size=96),
                prefix_granularity=128,
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
            prefix_granularity=64,
            kernel_page_size=None,
            max_cudagraph_capture_size=4,
            kv_cache_quant_method="none",
            speculative_algorithm="EAGLE3",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            spec_context_pad=12,  # 3 overshoot spans * 4 draft tokens
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
        self.assertEqual(config.context_len, 4108)


class TestDecodeHostL2(unittest.TestCase):
    def test_decode_enables_host_l2_without_prefix_matching(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "decode"
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = False
        args.kvstore_storage_backend = None
        args.kvstore_mem_layout = "layer_first"
        args.kvstore_io_backend = "kernel"

        args._handle_kvstore()
        args.validate_cache_options()

        self.assertTrue(args.enable_kvstore)


class TestL3StorageBackend(unittest.TestCase):
    def test_cli_accepts_mooncake_and_memory(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        action = None
        for candidate in parser._actions:
            if candidate.dest == "kvstore_storage_backend":
                action = candidate
                break
        self.assertIsNotNone(action)
        self.assertEqual(set(action.choices), {"mooncake", "memory"})

    def test_memory_backend_keeps_host_layout_and_io(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = "memory"
        args.kvstore_mem_layout = "layer_first"
        args.kvstore_io_backend = "direct"

        args._handle_kvstore()

        self.assertTrue(args.enable_kvstore)
        self.assertEqual(args.kvstore_mem_layout, "layer_first")
        self.assertEqual(args.kvstore_io_backend, "direct")

    def test_l3_requires_host_l2(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.disable_kvstore = True
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = "mooncake"
        args.kvstore_mem_layout = "layer_first"
        args.kvstore_io_backend = "direct"

        with self.assertRaisesRegex(ValueError, "requires Host L2"):
            args._handle_kvstore()

    def test_mooncake_backend_switches_layout(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = "mooncake"
        args.kvstore_mem_layout = "layer_first"
        args.kvstore_io_backend = "direct"

        args._handle_kvstore()

        self.assertTrue(args.enable_kvstore)
        self.assertEqual(args.kvstore_mem_layout, "page_first")
        self.assertEqual(args.kvstore_io_backend, "kernel")


if __name__ == "__main__":
    unittest.main()
