"""Tests for vLLM-style CLI configuration arguments.

Verifies that vLLM-style CLI args are correctly parsed and mapped
to TokenSpeed's internal ServerArgs configuration.
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
from unittest.mock import patch

from tokenspeed.runtime.utils.server_args import ServerArgs


class TestCLIConfigCompat(unittest.TestCase):
    """Test that vLLM-style CLI arguments map to TokenSpeed config."""

    def _parse_args(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser.parse_args(argv)

    def _from_cli_args_no_init(self, args: argparse.Namespace) -> ServerArgs:
        with patch.object(ServerArgs, "__post_init__"):
            return ServerArgs.from_cli_args(args)

    def _resolve_speculative_config(
        self,
        model: str,
        config: str,
        *,
        enable_prefix_caching: bool = True,
    ) -> ServerArgs:
        argv = ["--model", model]
        if not enable_prefix_caching:
            argv.append("--no-enable-prefix-caching")
        argv.extend(["--speculative-config", config])
        args = self._parse_args(argv)
        server_args = self._from_cli_args_no_init(args)
        server_args.resolve_basic_defaults()
        server_args.resolve_speculative_decoding()
        return server_args

    def _parallelism_snapshot(self, argv: list[str]) -> tuple[int, ...]:
        args = self._parse_args(argv)
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        sa.resolve_parallelism()
        mapping = sa.mapping
        return (
            mapping.world_size,
            mapping.attn.tp_size,
            mapping.attn.cp_size,
            mapping.attn.dp_size,
            mapping.dense.tp_size,
            mapping.dense.dp_size,
            mapping.moe.tp_size,
            mapping.moe.ep_size,
            mapping.moe.dp_size,
        )

    # ---- Positional model arg ----

    def test_positional_model_arg(self):
        args = self._parse_args(["deepseek-ai/DeepSeek-V3"])
        self.assertEqual(args.model_path, "deepseek-ai/DeepSeek-V3")
        self.assertIsNone(args.model)

    def test_model_flag(self):
        args = self._parse_args(["--model", "deepseek-ai/DeepSeek-V3"])
        self.assertIsNone(args.model_path)
        self.assertEqual(args.model, "deepseek-ai/DeepSeek-V3")

    def test_positional_model_resolved_in_from_cli_args(self):
        args = self._parse_args(["deepseek-ai/DeepSeek-V3"])
        sa = self._from_cli_args_no_init(args)
        self.assertEqual(sa.model, "deepseek-ai/DeepSeek-V3")

    def test_both_positional_and_model_raises(self):
        args = self._parse_args(["deepseek-ai/DeepSeek-V3", "--model", "other/model"])
        with self.assertRaises(ValueError):
            self._from_cli_args_no_init(args)

    def test_no_model_raises(self):
        args = self._parse_args([])
        with self.assertRaises(ValueError):
            self._from_cli_args_no_init(args)

    # ---- Tensor parallel size ----

    def test_tensor_parallel_size_maps_to_attn_tp_size(self):
        args = self._parse_args(
            ["--model", "test/model", "--tensor-parallel-size", "8"]
        )
        sa = self._from_cli_args_no_init(args)
        self.assertEqual(sa.attn_tp_size, 8)

    def test_tp_long_alias(self):
        args = self._parse_args(["--model", "test/model", "--tp", "4"])
        sa = self._from_cli_args_no_init(args)
        self.assertEqual(sa.attn_tp_size, 4)

    def test_tensor_parallel_aliases_match_explicit_attn_moe_tp(self):
        explicit = self._parallelism_snapshot(
            [
                "--model",
                "nvidia/Kimi-K2.5-NVFP4",
                "--attn-tp-size",
                "4",
                "--moe-tp-size",
                "4",
            ]
        )
        tensor_parallel_size = self._parallelism_snapshot(
            [
                "--model",
                "nvidia/Kimi-K2.5-NVFP4",
                "--tensor-parallel-size",
                "4",
            ]
        )
        tp = self._parallelism_snapshot(
            [
                "--model",
                "nvidia/Kimi-K2.5-NVFP4",
                "--tp",
                "4",
            ]
        )

        self.assertEqual(tensor_parallel_size, explicit)
        self.assertEqual(tp, explicit)

    def test_tensor_parallel_size_conflicts_with_attn_tp_size(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--tensor-parallel-size",
                "8",
                "--attn-tp-size",
                "4",
            ]
        )
        with self.assertRaises(ValueError):
            self._from_cli_args_no_init(args)

    # ---- Enable expert parallel ----

    def test_enable_expert_parallel_flag(self):
        args = self._parse_args(["--model", "test/model", "--enable-expert-parallel"])
        sa = self._from_cli_args_no_init(args)
        self.assertTrue(sa.enable_expert_parallel)

    def test_enable_expert_parallel_default_false(self):
        args = self._parse_args(["--model", "test/model"])
        sa = self._from_cli_args_no_init(args)
        self.assertFalse(sa.enable_expert_parallel)

    # ---- Dense TP default ----

    def test_dense_tp_defaults_to_full_world_without_dp(self):
        # No DP attention: attn_tp == world, so the dense default is unchanged
        # (still the full world). This is the no-op case for the new default.
        world, attn_tp = self._parallelism_snapshot(
            ["--model", "test/model", "--attn-tp-size", "8"]
        )[:2]
        dense_tp = self._parallelism_snapshot(
            ["--model", "test/model", "--attn-tp-size", "8"]
        )[4]
        self.assertEqual((world, attn_tp), (8, 8))
        self.assertEqual(dense_tp, 8)

    def test_dense_tp_defaults_to_attn_replica_width_with_dp(self):
        # DP attention: the dense default follows the attention replica width
        # (attn_tp), not the full world, keeping each dense all-reduce local.
        snap = self._parallelism_snapshot(
            [
                "--model",
                "test/model",
                "--attn-tp-size",
                "4",
                "--data-parallel-size",
                "2",
            ]
        )
        world, attn_tp, _attn_cp, attn_dp, dense_tp = snap[:5]
        self.assertEqual((world, attn_tp, attn_dp), (8, 4, 2))
        self.assertEqual(dense_tp, 4)

    def test_dense_tp_explicit_override_is_respected_under_dp(self):
        # An explicit --dense-tp-size still wins over the replica-width default.
        dense_tp = self._parallelism_snapshot(
            [
                "--model",
                "test/model",
                "--attn-tp-size",
                "4",
                "--data-parallel-size",
                "2",
                "--dense-tp-size",
                "8",
            ]
        )[4]
        self.assertEqual(dense_tp, 8)

    def test_dense_tp_default_tracks_replica_width_under_cp(self):
        # Under ENABLE_CP the attention TP size is reinterpreted as CP, so the
        # replica width is attn_tp x attn_cp; the dense default must use the
        # product, not the post-swap attn_tp (which is 1 here).
        import tokenspeed.runtime.utils.server_args as server_args_mod

        with patch.object(server_args_mod, "ENABLE_CP", True):
            snap = self._parallelism_snapshot(
                [
                    "--model",
                    "test/model",
                    "--attn-tp-size",
                    "4",
                    "--data-parallel-size",
                    "2",
                ]
            )
        world, attn_tp, attn_cp, attn_dp, dense_tp = snap[:5]
        self.assertEqual((world, attn_tp, attn_cp, attn_dp), (8, 1, 4, 2))
        self.assertEqual(dense_tp, 4)

    # ---- vLLM config names ----

    def test_tokenizer_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--tokenizer", "my/tokenizer"]
        )
        self.assertEqual(args.tokenizer, "my/tokenizer")

    def test_max_model_len_arg(self):
        args = self._parse_args(["--model", "test/model", "--max-model-len", "4096"])
        self.assertEqual(args.max_model_len, 4096)

    def test_gpu_memory_utilization_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--gpu-memory-utilization", "0.9"]
        )
        self.assertEqual(args.gpu_memory_utilization, 0.9)

    def test_seed_arg(self):
        args = self._parse_args(["--model", "test/model", "--seed", "42"])
        self.assertEqual(args.seed, 42)

    def test_max_num_seqs_arg(self):
        args = self._parse_args(["--model", "test/model", "--max-num-seqs", "256"])
        self.assertEqual(args.max_num_seqs, 256)

    def test_dp_sampling_backend_arg_removed(self):
        with self.assertRaises(SystemExit):
            self._parse_args(
                ["--model", "test/model", "--dp-sampling-backend", "onesided"]
            )

    def test_max_prefill_tokens_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--max-prefill-tokens", "4096"]
        )
        self.assertEqual(args.max_prefill_tokens, 4096)

    def test_chunked_prefill_size_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--chunked-prefill-size", "2048"]
        )
        self.assertEqual(args.chunked_prefill_size, 2048)

    def test_prefill_token_defaults(self):
        args = self._parse_args(["--model", "test/model"])
        self.assertEqual(args.max_prefill_tokens, 8192)
        self.assertIsNone(args.chunked_prefill_size)
        self.assertFalse(args.enable_mixed_batch)

        sa = self._from_cli_args_no_init(args)
        sa.mapping = SimpleNamespace(world_size=1)
        platform = SimpleNamespace(is_amd=False, is_nvidia=False)
        with patch(
            "tokenspeed.runtime.utils.server_args.current_platform",
            return_value=platform,
        ):
            sa.resolve_memory_and_scheduling()

        self.assertEqual(sa.max_prefill_tokens, 8192)
        self.assertEqual(sa.chunked_prefill_size, 8192)
        self.assertFalse(sa.enable_mixed_batch)

    def test_mixed_batch_can_be_enabled(self):
        args = self._parse_args(["--model", "test/model", "--enable-mixed-batch"])
        self.assertTrue(args.enable_mixed_batch)

    def test_distributed_timeout_seconds_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--distributed-timeout-seconds", "600"]
        )
        self.assertEqual(args.distributed_timeout_seconds, 600)

    def test_enforce_eager_arg(self):
        args = self._parse_args(["--model", "test/model", "--enforce-eager"])
        self.assertTrue(args.enforce_eager)

    def test_cudagraph_capture_size_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--max-cudagraph-capture-size", "32"]
        )
        self.assertEqual(args.max_cudagraph_capture_size, 32)

    def test_cudagraph_capture_sizes_arg(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--cudagraph-capture-sizes",
                "1",
                "2",
                "4",
            ]
        )
        self.assertEqual(args.cudagraph_capture_sizes, [1, 2, 4])

    def test_block_size_arg(self):
        args = self._parse_args(["--model", "test/model", "--block-size", "128"])
        self.assertEqual(args.prefix_granularity, 128)

    def test_moe_backend_arg(self):
        args = self._parse_args(["--model", "test/model", "--moe-backend", "triton"])
        self.assertEqual(args.moe_backend, "triton")

    def test_sampling_backend_arg(self):
        for backend in (
            "greedy",
            "flashinfer",
            "flashinfer_full",
            "triton",
            "triton_full",
        ):
            args = self._parse_args(
                ["--model", "test/model", "--sampling-backend", backend]
            )
            self.assertEqual(args.sampling_backend, backend)

    def test_all2all_backend_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--all2all-backend", "deepep"]
        )
        self.assertEqual(args.all2all_backend, "deepep")

    def test_recipe_all2all_backend_alias_arg(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--all2all-backend",
                "flashinfer_nvlink_one_sided",
            ]
        )
        self.assertEqual(args.all2all_backend, "flashinfer_nvlink_one_sided")

    def test_recipe_moe_backend_alias_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--moe-backend", "deep_gemm_mega_moe"]
        )
        self.assertEqual(args.moe_backend, "deep_gemm_mega_moe")

    def test_kv_cache_dtype_fp8_alias_arg(self):
        args = self._parse_args(["--model", "test/model", "--kv-cache-dtype", "fp8"])
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        self.assertEqual(sa.kv_cache_dtype, "fp8_e4m3")

    def test_tokenizer_mode_deepseek_v4_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--tokenizer-mode", "deepseek_v4"]
        )
        self.assertEqual(args.tokenizer_mode, "deepseek_v4")

    def test_hf_overrides_arg(self):
        args = self._parse_args(
            ["--model", "test/model", "--hf-overrides", '{"rope_scaling": null}']
        )
        self.assertEqual(args.hf_overrides, '{"rope_scaling": null}')

    def test_enable_log_requests_arg(self):
        args = self._parse_args(["--model", "test/model", "--enable-log-requests"])
        self.assertTrue(args.enable_log_requests)

    def test_no_enable_log_requests_arg(self):
        args = self._parse_args(["--model", "test/model", "--no-enable-log-requests"])
        self.assertFalse(args.enable_log_requests)

    def test_no_trust_remote_code_arg(self):
        args = self._parse_args(["--model", "test/model", "--no-trust-remote-code"])
        self.assertFalse(args.trust_remote_code)

    def test_enable_prefix_caching_arg(self):
        args = self._parse_args(["--model", "test/model", "--enable-prefix-caching"])
        self.assertTrue(args.enable_prefix_caching)

    def test_no_enable_prefix_caching_arg(self):
        args = self._parse_args(["--model", "test/model", "--no-enable-prefix-caching"])
        self.assertFalse(args.enable_prefix_caching)

    def test_kv_events_config_arg(self):
        config = (
            '{"publisher":"zmq","endpoint":"tcp://*:5557",'
            '"topic":"kv-events","enable_kv_cache_events":true}'
        )
        args = self._parse_args(["--model", "test/model", "--kv-events-config", config])
        sa = self._from_cli_args_no_init(args)
        self.assertEqual(sa.kv_events_config, config)

    def test_speculative_draft_quantization_defaults_to_unquant(self):
        args = self._parse_args(["--model", "test/model", "--quantization", "nvfp4"])
        self.assertEqual(args.speculative_draft_model_quantization, "unquant")

        sa = self._from_cli_args_no_init(args)
        sa.resolve_speculative_decoding()
        self.assertIsNone(sa.speculative_draft_model_quantization)

    def test_replay_ssm_defaults_to_disabled(self):
        args = self._parse_args(["--model", "test/model"])
        self.assertFalse(self._from_cli_args_no_init(args).enable_replay_ssm)

    def test_replay_ssm_can_be_enabled(self):
        args = self._parse_args(["--model", "test/model", "--enable-replay-ssm"])
        self.assertTrue(self._from_cli_args_no_init(args).enable_replay_ssm)

    def test_mxfp4_quantization_arg(self):
        args = self._parse_args(["--model", "test/model", "--quantization", "mxfp4"])
        self.assertEqual(args.quantization, "mxfp4")

    def test_dotted_attention_config_args(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--attention_config.use_fp4_indexer_cache=True",
                "--attention-config.use_trtllm_ragged_deepseek_prefill=True",
            ]
        )
        self.assertTrue(args.attention_use_fp4_indexer_cache)
        self.assertTrue(args.use_trtllm_ragged_deepseek_prefill)

    def test_vllm_recipe_speculative_config_arg(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-config",
                '{"method": "mtp", "model": "draft/model", "num_speculative_tokens": 3}',
            ]
        )
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        self.assertEqual(sa.speculative_algorithm, "MTP")
        self.assertEqual(sa.speculative_draft_model_path, "draft/model")
        self.assertEqual(sa.speculative_num_steps, 3)
        self.assertEqual(sa.speculative_num_draft_tokens, 4)

    def test_speculative_config_matches_explicit_eagle3_args(self):
        draft_model = "lightseekorg/kimi-k2.5-eagle3-mla"

        config_args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-config",
                (
                    f'{{"model":"{draft_model}",'
                    '"method":"eagle3",'
                    '"num_speculative_tokens":1}'
                ),
            ]
        )
        explicit_args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                draft_model,
                "--speculative-num-steps",
                "1",
            ]
        )

        config_server_args = self._from_cli_args_no_init(config_args)
        explicit_server_args = self._from_cli_args_no_init(explicit_args)
        config_server_args.resolve_basic_defaults()
        explicit_server_args.resolve_basic_defaults()

        self.assertEqual(
            config_server_args.speculative_algorithm,
            explicit_server_args.speculative_algorithm,
        )
        self.assertEqual(
            config_server_args.speculative_draft_model_path,
            explicit_server_args.speculative_draft_model_path,
        )
        self.assertEqual(
            config_server_args.speculative_num_steps,
            explicit_server_args.speculative_num_steps,
        )
        self.assertEqual(
            config_server_args.speculative_num_draft_tokens,
            explicit_server_args.speculative_num_draft_tokens,
        )

    def test_speculative_config_matches_explicit_mtp_args(self):
        target_model = "nvidia/Qwen3.5-397B-A17B-NVFP4"

        config_args = self._parse_args(
            [
                "--model",
                target_model,
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":3}',
            ]
        )
        explicit_args = self._parse_args(
            [
                "--model",
                target_model,
                "--speculative-algorithm",
                "MTP",
                "--speculative-num-steps",
                "3",
            ]
        )

        config_server_args = self._from_cli_args_no_init(config_args)
        explicit_server_args = self._from_cli_args_no_init(explicit_args)
        config_server_args.resolve_basic_defaults()
        explicit_server_args.resolve_basic_defaults()
        config_server_args.resolve_speculative_decoding()
        explicit_server_args.resolve_speculative_decoding()

        self.assertEqual(
            config_server_args.speculative_algorithm,
            explicit_server_args.speculative_algorithm,
        )
        self.assertEqual(
            config_server_args.speculative_draft_model_path,
            explicit_server_args.speculative_draft_model_path,
        )
        self.assertEqual(
            config_server_args.speculative_draft_model_path,
            target_model,
        )
        self.assertTrue(explicit_server_args.draft_model_path_use_base)
        self.assertEqual(
            config_server_args.speculative_num_steps,
            explicit_server_args.speculative_num_steps,
        )
        self.assertEqual(
            config_server_args.speculative_num_draft_tokens,
            explicit_server_args.speculative_num_draft_tokens,
        )

    def test_dspark_same_checkpoint_config_uses_block_plus_bonus_width(self):
        server_args = self._resolve_speculative_config(
            "same-checkpoint",
            (
                '{"method":"dspark","model":"same-checkpoint",'
                '"num_speculative_tokens":5}'
            ),
            enable_prefix_caching=False,
        )

        self.assertTrue(server_args.draft_model_path_use_base)
        self.assertEqual(server_args.speculative_draft_model_path, "same-checkpoint")
        self.assertEqual(server_args.speculative_num_steps, 5)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_dspark_redirected_same_checkpoint_uses_block_plus_bonus_width(self):
        with patch(
            "tokenspeed.runtime.utils.server_args.maybe_model_redirect",
            side_effect=lambda model: (
                "resolved-checkpoint" if model == "model-alias" else model
            ),
        ):
            server_args = self._resolve_speculative_config(
                "model-alias",
                (
                    '{"method":"dspark","model":"model-alias",'
                    '"num_speculative_tokens":5}'
                ),
                enable_prefix_caching=False,
            )

        self.assertTrue(server_args.draft_model_path_use_base)
        self.assertEqual(
            server_args.speculative_draft_model_path, "resolved-checkpoint"
        )
        self.assertEqual(server_args.speculative_num_steps, 5)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_dspark_external_checkpoint_config_uses_verify_width(self):
        server_args = self._resolve_speculative_config(
            "target-checkpoint",
            (
                '{"method":"dspark","model":"draft-checkpoint",'
                '"num_speculative_tokens":6}'
            ),
        )

        self.assertFalse(server_args.draft_model_path_use_base)
        self.assertEqual(server_args.speculative_draft_model_path, "draft-checkpoint")
        self.assertEqual(server_args.speculative_num_steps, 5)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_speculative_config_must_be_json_object(self):
        args = self._parse_args(["--model", "test/model", "--speculative-config", "[]"])
        sa = self._from_cli_args_no_init(args)
        with self.assertRaisesRegex(
            ValueError, "--speculative-config must be a JSON object"
        ):
            sa.resolve_basic_defaults()

    def test_speculative_defaults(self):
        args = self._parse_args(["--model", "test/model"])
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        self.assertEqual(sa.speculative_num_steps, 3)
        self.assertEqual(sa.speculative_eagle_topk, 1)
        self.assertEqual(sa.speculative_num_draft_tokens, 4)

    def test_speculative_draft_tokens_default_to_steps_plus_one(self):
        args = self._parse_args(
            ["--model", "test/model", "--speculative-num-steps", "1"]
        )
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        self.assertEqual(sa.speculative_num_steps, 1)
        self.assertEqual(sa.speculative_num_draft_tokens, 2)

    def test_speculative_eagle_topk_cli_rejects_non_1(self):
        # Only chain spec (topk=1) is wired end-to-end; the CLI choices
        # set is the gate, so non-1 values must fail at parse time.
        with self.assertRaises(SystemExit):
            self._parse_args(["--model", "test/model", "--speculative-eagle-topk", "4"])

    def test_speculative_eagle_topk_runtime_rejects_non_1_when_spec_on(self):
        # ServerArgs can be built programmatically (e.g. by smg_grpc_servicer),
        # bypassing argparse — keep the resolve-time defensive check covered.
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-algorithm",
                "EAGLE3",
            ]
        )
        sa = self._from_cli_args_no_init(args)
        sa.speculative_eagle_topk = 4
        sa.resolve_basic_defaults()
        with self.assertRaisesRegex(ValueError, "speculative_eagle_topk"):
            sa.resolve_speculative_decoding()

    def test_dp_sampling_is_opt_in(self):
        args = self._parse_args(["--model", "test/model"])
        sa = self._from_cli_args_no_init(args)
        self.assertFalse(sa.dp_sampling)
        self.assertIsNone(sa.dp_sampling_min_bs)

        args = self._parse_args(["--model", "test/model", "--dp-sampling"])
        sa = self._from_cli_args_no_init(args)
        self.assertTrue(sa.dp_sampling)

    def test_dp_sampling_min_bs_arg(self):
        args = self._parse_args(["--model", "test/model", "--dp-sampling-min-bs", "16"])
        sa = self._from_cli_args_no_init(args)
        self.assertEqual(sa.dp_sampling_min_bs, 16)

    # ---- Full server command example ----

    def test_full_server_command(self):
        """Test a full server command example:
        tokenspeed serve deepseek-ai/DeepSeek-V3.1 \\
          --enable-expert-parallel \\
          --tensor-parallel-size 8 \\
          --served-model-name ds31
        """
        args = self._parse_args(
            [
                "deepseek-ai/DeepSeek-V3.1",
                "--enable-expert-parallel",
                "--tensor-parallel-size",
                "8",
                "--served-model-name",
                "ds31",
            ]
        )
        sa = self._from_cli_args_no_init(args)

        self.assertEqual(sa.model, "deepseek-ai/DeepSeek-V3.1")
        self.assertEqual(sa.attn_tp_size, 8)
        self.assertTrue(sa.enable_expert_parallel)
        self.assertEqual(sa.served_model_name, "ds31")

    def test_data_parallel_size_arg(self):
        args = self._parse_args(["--model", "test/model", "--data-parallel-size", "2"])
        sa = self._from_cli_args_no_init(args)
        self.assertEqual(sa.data_parallel_size, 2)

    def test_help_uses_expected_metavars(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit):
                parser.parse_args(["--help"])

        help_text = stdout.getvalue()
        self.assertIn("--max-num-seqs MAX_NUM_SEQS", help_text)
        self.assertIn("--max-prefill-tokens MAX_PREFILL_TOKENS", help_text)
        self.assertIn("--chunked-prefill-size CHUNKED_PREFILL_SIZE", help_text)
        self.assertIn("--gpu-memory-utilization GPU_MEMORY_UTILIZATION", help_text)
        self.assertIn(
            "--distributed-timeout-seconds DISTRIBUTED_TIMEOUT_SECONDS", help_text
        )
        self.assertIn("--all2all-backend ALL2ALL_BACKEND", help_text)
        self.assertIn("--hf-overrides HF_OVERRIDES", help_text)
        self.assertNotIn("MAX_RUNNING_REQUESTS", help_text)
        self.assertNotIn("MEM_FRACTION_STATIC", help_text)
        self.assertNotIn("DIST_TIMEOUT", help_text)
        self.assertNotIn("MOE_A2A_BACKEND", help_text)
        self.assertNotIn("JSON_MODEL_OVERRIDE_ARGS", help_text)


if __name__ == "__main__":
    unittest.main()
