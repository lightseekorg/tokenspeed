"""Tests for vLLM-style CLI configuration arguments.

Verifies that vLLM-style CLI args are correctly parsed and mapped
to TokenSpeed's internal ServerArgs configuration.
"""

import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import argparse
import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.spec_block_geometry import BLOCK_SPEC_RULES


class TestCLIConfigCompat:
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
        argv = ["--model", model, "--sampling-backend", "greedy"]
        if not enable_prefix_caching:
            argv.append("--disable-prefix-caching")
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
            mapping.attn.dp_size,
            mapping.dense.tp_size,
            mapping.dense.dp_size,
            mapping.moe.tp_size,
            mapping.moe.ep_size,
            mapping.moe.dp_size,
        )

    @pytest.mark.parametrize(
        "contents, expected",
        [
            ('{"org/model": "/local/model"}', "/local/model"),
            ('{"org/model": ""}', "org/model"),
            ("org/model /local/model\ninvalid\nother /other\n", "/local/model"),
            ("", "org/model"),
            ("[]", "org/model"),
            ("null", "org/model"),
            ('{"org/model": 1}', "org/model"),
            ('{"org/model": {"path": "/local/model"}}', "org/model"),
            ('{"org/model": "/local/model", "other": null}', "org/model"),
        ],
    )
    def test_model_redirect_file(self, tmp_path, monkeypatch, contents, expected):
        redirect_path = tmp_path / "redirect"
        redirect_path.write_text(contents)
        monkeypatch.setenv("TOKENSPEED_MODEL_REDIRECT_PATH", str(redirect_path))
        server_args = self._from_cli_args_no_init(
            self._parse_args(["--model", "org/model"])
        )

        server_args.resolve_basic_defaults()

        assert server_args.model == expected
        assert server_args.tokenizer == expected
        assert server_args.served_model_name == expected

    # ---- Positional model arg ----

    def test_positional_model_arg(self):
        args = self._parse_args(["deepseek-ai/DeepSeek-V3"])
        assert (args.model_path) == ("deepseek-ai/DeepSeek-V3")
        assert (args.model) is None

    def test_positional_model_resolved_in_from_cli_args(self):
        args = self._parse_args(["deepseek-ai/DeepSeek-V3"])
        sa = self._from_cli_args_no_init(args)
        assert (sa.model) == ("deepseek-ai/DeepSeek-V3")

    def test_both_positional_and_model_raises(self):
        args = self._parse_args(["deepseek-ai/DeepSeek-V3", "--model", "other/model"])
        with pytest.raises((ValueError)):
            self._from_cli_args_no_init(args)

    # ---- Tensor parallel size ----

    def test_tensor_parallel_size_maps_to_attn_tp_size(self):
        args = self._parse_args(
            ["--model", "test/model", "--tensor-parallel-size", "8"]
        )
        sa = self._from_cli_args_no_init(args)
        assert (sa.attn_tp_size) == (8)

    def test_tp_long_alias(self):
        args = self._parse_args(["--model", "test/model", "--tp", "4"])
        sa = self._from_cli_args_no_init(args)
        assert (sa.attn_tp_size) == (4)

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

        assert (tensor_parallel_size) == (explicit)
        assert (tp) == (explicit)

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
        with pytest.raises((ValueError)):
            self._from_cli_args_no_init(args)

    # ---- Enable expert parallel ----

    def test_enable_expert_parallel_default_false(self):
        args = self._parse_args(["--model", "test/model"])
        sa = self._from_cli_args_no_init(args)
        assert not (sa.enable_expert_parallel)

    # ---- Dense TP default ----

    def test_dense_tp_defaults_to_full_world_without_dp(self):
        # No DP attention: attn_tp == world, so the dense default is unchanged
        # (still the full world). This is the no-op case for the new default.
        world, attn_tp = self._parallelism_snapshot(
            ["--model", "test/model", "--attn-tp-size", "8"]
        )[:2]
        dense_tp = self._parallelism_snapshot(
            ["--model", "test/model", "--attn-tp-size", "8"]
        )[3]
        assert ((world, attn_tp)) == ((8, 8))
        assert (dense_tp) == (8)

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
        world, attn_tp, attn_dp, dense_tp = snap[:4]
        assert ((world, attn_tp, attn_dp)) == ((8, 4, 2))
        assert (dense_tp) == (4)

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
        )[3]
        assert (dense_tp) == (8)

    # ---- vLLM config names ----

    def test_vllm_flags_parse_to_their_destinations(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--tokenizer",
                "my/tokenizer",
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.9",
                "--seed",
                "42",
                "--max-num-seqs",
                "256",
                "--max-prefill-tokens",
                "4096",
                "--chunked-prefill-size",
                "2048",
                "--enable-mixed-batch",
                "--distributed-timeout-seconds",
                "600",
                "--enforce-eager",
                "--max-cudagraph-capture-size",
                "32",
                "--cudagraph-capture-sizes",
                "1",
                "2",
                "4",
                "--moe-backend",
                "deep_gemm_mega_moe",
                "--sampling-backend",
                "triton_full",
                "--tokenizer-mode",
                "slow",
                "--hf-overrides",
                '{"rope_scaling": null}',
                "--no-trust-remote-code",
                "--disable-prefix-caching",
                "--quantization",
                "mxfp4",
                "--enable-expert-parallel",
                "--data-parallel-size",
                "2",
                "--dp-sampling-min-bs",
                "16",
            ]
        )
        expected = {
            "tokenizer": "my/tokenizer",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "seed": 42,
            "max_num_seqs": 256,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 2048,
            "enable_mixed_batch": True,
            "distributed_timeout_seconds": 600,
            "enforce_eager": True,
            "max_cudagraph_capture_size": 32,
            "cudagraph_capture_sizes": [1, 2, 4],
            "moe_backend": "deep_gemm_mega_moe",
            "sampling_backend": "triton_full",
            "tokenizer_mode": "slow",
            "hf_overrides": '{"rope_scaling": null}',
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "quantization": "mxfp4",
            "enable_expert_parallel": True,
            "data_parallel_size": 2,
            "dp_sampling_min_bs": 16,
        }
        assert {name: getattr(args, name) for name in expected} == expected

    def test_gpu_memory_utilization_default_is_one_value_at_every_world_size(self):
        for tp in ("1", "2", "8", "16"):
            sa = self._from_cli_args_no_init(
                self._parse_args(
                    ["--model", "test/model", "--tensor-parallel-size", tp]
                )
            )
            sa.resolve_basic_defaults()
            sa.resolve_parallelism()
            sa.resolve_memory_and_scheduling()
            _pytest_first_1 = sa.gpu_memory_utilization
            _pytest_second_2 = 0.95
            _pytest_msg_3 = tp
            assert _pytest_first_1 == _pytest_second_2, _pytest_msg_3
        sa = self._from_cli_args_no_init(
            self._parse_args(
                [
                    "--model",
                    "test/model",
                    "--tensor-parallel-size",
                    "16",
                    "--gpu-memory-utilization",
                    "0.8",
                ]
            )
        )
        sa.resolve_basic_defaults()
        sa.resolve_parallelism()
        sa.resolve_memory_and_scheduling()
        assert (sa.gpu_memory_utilization) == (0.8)

    def test_prefill_token_defaults(self):
        args = self._parse_args(["--model", "test/model"])
        assert (args.max_prefill_tokens) == (8192)
        assert (args.chunked_prefill_size) is None
        assert not (args.enable_mixed_batch)

        sa = self._from_cli_args_no_init(args)
        sa.mapping = SimpleNamespace(world_size=1)
        platform = SimpleNamespace(is_amd=False, is_nvidia=False)
        with patch(
            "tokenspeed.runtime.utils.server_args.current_platform",
            return_value=platform,
        ):
            sa.resolve_memory_and_scheduling()

        assert (sa.max_prefill_tokens) == (8192)
        assert (sa.chunked_prefill_size) == (8192)
        assert not (sa.enable_mixed_batch)

    def test_constrained_decoding_is_internal_and_defaults_to_none(self):
        args = self._parse_args(["--model", "test/model"])
        assert self._from_cli_args_no_init(args).constrained_decoding is None
        with pytest.raises(SystemExit):
            self._parse_args(
                ["--model", "test/model", "--constrained-decoding", "always"]
            )

    def test_kv_cache_dtype_fp8_alias_arg(self):
        args = self._parse_args(["--model", "test/model", "--kv-cache-dtype", "fp8"])
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        assert (sa.kv_cache_dtype) == ("fp8_e4m3")

    def test_kv_events_config_arg(self):
        config = (
            '{"publisher":"zmq","endpoint":"tcp://*:5557",'
            '"topic":"kv-events","enable_kv_cache_events":true}'
        )
        args = self._parse_args(["--model", "test/model", "--kv-events-config", config])
        sa = self._from_cli_args_no_init(args)
        assert (sa.kv_events_config) == (config)

    def test_l2_cache_options_across_engine_roles(self, subtests):
        for role in ("null", "prefill", "decode", "encode"):
            for io_backend in ("direct", "kernel"):
                for disabled in (False, True):
                    with subtests.test(
                        role=role, io_backend=io_backend, disabled=disabled
                    ):
                        argv = [
                            "--model",
                            "test/model",
                            "--disaggregation-mode",
                            role,
                            "--kvstore-io-backend",
                            io_backend,
                            "--kvstore-ratio",
                            "3",
                            "--kvstore-size",
                            "8",
                            "--max-num-seqs",
                            "16",
                        ]
                        if disabled:
                            argv.append("--disable-kvstore")
                        if role == "decode":
                            argv.append("--disable-prefix-caching")
                        sa = self._from_cli_args_no_init(self._parse_args(argv))
                        # resolve_cache derives the retraction snapshot request
                        # cap from max_num_seqs per attention-DP rank, so the
                        # mapping must be resolved first, as in __post_init__.
                        sa.resolve_parallelism()
                        sa.resolve_cache()

                        assert (sa.enable_kvstore) == (
                            not disabled and role != "encode"
                        )
                        assert (sa.kvstore_io_backend) == (io_backend)
                        assert (sa.kvstore_ratio) == (3)
                        assert (sa.kvstore_size) == (8)

    def test_speculative_draft_quantization_defaults_to_unquant(self):
        args = self._parse_args(["--model", "test/model", "--quantization", "nvfp4"])
        assert (args.speculative_draft_model_quantization) == ("unquant")

        sa = self._from_cli_args_no_init(args)
        sa.resolve_speculative_decoding()
        assert (sa.speculative_draft_model_quantization) is None

    def test_replay_ssm_defaults_to_enabled(self):
        args = self._parse_args(["--model", "test/model"])
        assert self._from_cli_args_no_init(args).enable_replay_ssm

    def test_replay_ssm_can_be_disabled(self):
        args = self._parse_args(["--model", "test/model", "--disable-replay-ssm"])
        assert not (self._from_cli_args_no_init(args).enable_replay_ssm)

    def test_replay_ssm_can_be_enabled(self):
        args = self._parse_args(["--model", "test/model", "--enable-replay-ssm"])
        assert self._from_cli_args_no_init(args).enable_replay_ssm

    def test_dotted_attention_config_args(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--attention_config.use_fp4_indexer_cache=True",
                "--attention-config.use_trtllm_ragged_deepseek_prefill=True",
            ]
        )
        assert args.attention_use_fp4_indexer_cache
        assert args.use_trtllm_ragged_deepseek_prefill

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
        assert (sa.speculative_algorithm) == ("MTP")
        assert (sa.speculative_draft_model_path) == ("draft/model")
        assert (sa.speculative_num_steps) == (3)
        assert (sa.speculative_num_draft_tokens) == (4)

    def test_synthetic_acceptance_length_cli_and_json(
        self, subtests, caplog, monkeypatch
    ):
        monkeypatch.setattr(server_args_module.logger, "propagate", True)
        for length in (1, 2.6, 4):
            for source in ("cli", "json", "both"):
                with subtests.test(length=length, source=source):
                    config = {"method": "mtp", "num_speculative_tokens": 3}
                    if source in ("json", "both"):
                        config["synthetic_acceptance_length"] = length
                    argv = [
                        "--model",
                        "test/model",
                        "--sampling-backend",
                        "greedy",
                        "--speculative-config",
                        json.dumps(config),
                    ]
                    if source in ("cli", "both"):
                        argv += ["--synthetic-acceptance-length", str(length)]
                    sa = self._from_cli_args_no_init(self._parse_args(argv))
                    sa.resolve_basic_defaults()
                    caplog.clear()
                    with caplog.at_level(
                        "WARNING", logger="tokenspeed.runtime.utils.server_args"
                    ):
                        sa.resolve_speculative_decoding()
                    assert (sa.synthetic_acceptance_length) == (length)
                    records = [
                        record
                        for record in caplog.records
                        if record.name == "tokenspeed.runtime.utils.server_args"
                        or record.name.startswith(
                            "tokenspeed.runtime.utils.server_args."
                        )
                    ]
                    assert records
                    assert (
                        "correctness or accuracy evaluation" in records[0].getMessage()
                    )

    def test_synthetic_acceptance_length_defaults_disabled(self):
        sa = self._resolve_speculative_config(
            "test/model", '{"method":"mtp","num_speculative_tokens":3}'
        )
        assert (sa.synthetic_acceptance_length) is None

    def test_synthetic_acceptance_length_rejects_unsupported_modes(self, subtests):
        unsupported = [
            ([], "requires --sampling-backend greedy"),
            *[
                (["--sampling-backend", name], "requires --sampling-backend greedy")
                for name in ("triton", "triton_full", "flashinfer", "flashinfer_full")
            ],
            *[
                (["--sampling-backend", "greedy", flag], "does not support")
                for flag in ("--dp-sampling", "--enable-output-logprobs")
            ],
            (
                [
                    "--sampling-backend",
                    "greedy",
                    "--speculative-eagle-topk",
                    "2",
                    "--speculative-num-draft-tokens",
                    "4",
                ],
                "does not support draft trees",
            ),
        ]
        for extra, message in unsupported:
            with subtests.test(extra=extra):
                args = self._parse_args(
                    [
                        "--model",
                        "test/model",
                        "--speculative-algorithm",
                        "MTP",
                        "--synthetic-acceptance-length",
                        "2.6",
                        *extra,
                    ]
                )
                sa = self._from_cli_args_no_init(args)
                sa.resolve_basic_defaults()
                with pytest.raises((ValueError), match=(message)):
                    sa.resolve_speculative_decoding()

    def test_synthetic_acceptance_length_conflicting_sources(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--synthetic-acceptance-length",
                "2.6",
                "--speculative-config",
                '{"method":"mtp","synthetic_acceptance_length":3}',
            ]
        )
        with pytest.raises((ValueError), match=("conflicts")):
            self._from_cli_args_no_init(args).resolve_basic_defaults()

    def test_synthetic_acceptance_length_rejects_invalid_values(self, subtests):
        for length in (
            True,
            False,
            "2.6",
            "bad",
            float("nan"),
            float("inf"),
            -float("inf"),
        ):
            with subtests.test(length=length, source="json"):
                with pytest.raises((ValueError), match=("finite number")):
                    self._resolve_speculative_config(
                        "test/model",
                        json.dumps(
                            {"method": "mtp", "synthetic_acceptance_length": length}
                        ),
                    )
            with subtests.test(length=length, source="python"):
                args = self._parse_args(
                    ["--model", "test/model", "--speculative-algorithm", "MTP"]
                )
                sa = self._from_cli_args_no_init(args)
                sa.synthetic_acceptance_length = length
                sa.resolve_basic_defaults()
                with pytest.raises((ValueError), match=("finite number")):
                    sa.resolve_speculative_decoding()

    def test_synthetic_acceptance_length_rejects_outside_verify_width(self, subtests):
        for length in (0, 0.9, -1, 4.1, 5):
            with subtests.test(length=length):
                with pytest.raises((ValueError), match=(r"must be in \[1, 4\]")):
                    self._resolve_speculative_config(
                        "test/model",
                        json.dumps(
                            {
                                "method": "mtp",
                                "num_speculative_tokens": 3,
                                "synthetic_acceptance_length": length,
                            }
                        ),
                    )

    def test_synthetic_acceptance_length_requires_speculation(self):
        args = self._parse_args(
            ["--model", "test/model", "--synthetic-acceptance-length", "1"]
        )
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        with pytest.raises((ValueError), match=("requires speculative decoding")):
            sa.resolve_speculative_decoding()

    def test_synthetic_acceptance_length_respects_block_verify_width(self, subtests):
        for method in ("dflash", "dspark"):
            with subtests.test(method=method):
                sa = self._resolve_speculative_config(
                    "test/model",
                    json.dumps(
                        {
                            "method": method,
                            "model": "draft/model",
                            "num_speculative_tokens": 4,
                            "synthetic_acceptance_length": 4,
                        }
                    ),
                )
                assert (sa.synthetic_acceptance_length) == (4)
                assert (sa.speculative_num_draft_tokens) == (4)

    def test_synthetic_acceptance_length_defers_implicit_block_width(self, subtests):
        for method in ("dflash", "dspark"):
            with subtests.test(method=method):
                config = {
                    "method": method,
                    "model": "draft/model",
                    "synthetic_acceptance_length": 5,
                }
                sa = self._resolve_speculative_config("test/model", json.dumps(config))
                assert not (sa._speculative_widths_explicit)
                assert (sa.synthetic_acceptance_length) == (5)

                config["num_speculative_tokens"] = 4
                with pytest.raises((ValueError), match=(r"must be in \[1, 4\]")):
                    self._resolve_speculative_config("test/model", json.dumps(config))

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

        assert (config_server_args.speculative_algorithm) == (
            explicit_server_args.speculative_algorithm
        )
        assert (config_server_args.speculative_draft_model_path) == (
            explicit_server_args.speculative_draft_model_path
        )
        assert (config_server_args.speculative_num_steps) == (
            explicit_server_args.speculative_num_steps
        )
        assert (config_server_args.speculative_num_draft_tokens) == (
            explicit_server_args.speculative_num_draft_tokens
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

        assert (config_server_args.speculative_algorithm) == (
            explicit_server_args.speculative_algorithm
        )
        assert (config_server_args.speculative_draft_model_path) == (
            explicit_server_args.speculative_draft_model_path
        )
        assert (config_server_args.speculative_draft_model_path) == (target_model)
        assert explicit_server_args.draft_model_path_use_base
        assert (config_server_args.speculative_num_steps) == (
            explicit_server_args.speculative_num_steps
        )
        assert (config_server_args.speculative_num_draft_tokens) == (
            explicit_server_args.speculative_num_draft_tokens
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

        assert server_args.draft_model_path_use_base
        assert (server_args.speculative_draft_model_path) == ("same-checkpoint")
        assert (server_args.speculative_num_steps) == (5)
        assert (server_args.speculative_num_draft_tokens) == (6)

    def test_dspark_redirected_same_checkpoint_uses_block_plus_bonus_width(
        self, tmp_path, monkeypatch
    ):
        redirect_path = tmp_path / "redirect.json"
        redirect_path.write_text('{"model-alias": "resolved-checkpoint"}')
        monkeypatch.setenv("TOKENSPEED_MODEL_REDIRECT_PATH", str(redirect_path))
        server_args = self._resolve_speculative_config(
            "model-alias",
            ('{"method":"dspark","model":"model-alias",' '"num_speculative_tokens":5}'),
            enable_prefix_caching=False,
        )

        assert server_args.draft_model_path_use_base
        assert (server_args.speculative_draft_model_path) == ("resolved-checkpoint")
        assert (server_args.speculative_num_steps) == (5)
        assert (server_args.speculative_num_draft_tokens) == (6)

    def test_dspark_external_checkpoint_config_uses_verify_width(self):
        server_args = self._resolve_speculative_config(
            "target-checkpoint",
            (
                '{"method":"dspark","model":"draft-checkpoint",'
                '"num_speculative_tokens":6}'
            ),
        )

        assert not (server_args.draft_model_path_use_base)
        assert (server_args.speculative_draft_model_path) == ("draft-checkpoint")
        assert (server_args.speculative_num_steps) == (5)
        assert (server_args.speculative_num_draft_tokens) == (6)

    def test_block_spec_widths_record_whether_they_were_asked_for(self, subtests):
        """ModelConfig only overwrites the widths the user left defaulted."""
        for extra, explicit in (
            ([], False),
            (["--speculative-num-steps", "7"], True),
            (["--speculative-num-draft-tokens", "8"], True),
            (
                [
                    "--speculative-config",
                    '{"method":"dflash","num_speculative_tokens":8}',
                ],
                True,
            ),
        ):
            with subtests.test(extra=extra):
                args = self._parse_args(
                    ["--model", "test/model", "--speculative-algorithm", "DFLASH"]
                    + extra
                )
                sa = self._from_cli_args_no_init(args)
                sa.resolve_basic_defaults()

                assert (sa._speculative_widths_explicit) == (explicit)

    def test_block_spec_step_mismatch_states_the_family_rules(self):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-num-steps",
                "7",
                "--speculative-num-draft-tokens",
                "7",
            ]
        )
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()

        with pytest.raises((ValueError)) as ctx:
            sa.resolve_speculative_decoding()
        assert (BLOCK_SPEC_RULES) in (str(ctx.value))

    def test_speculative_defaults(self):
        args = self._parse_args(["--model", "test/model"])
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        assert (sa.speculative_num_steps) == (3)
        assert (sa.speculative_eagle_topk) == (1)
        assert (sa.speculative_num_draft_tokens) == (4)

    def test_speculative_draft_tokens_default_to_steps_plus_one(self):
        args = self._parse_args(
            ["--model", "test/model", "--speculative-num-steps", "1"]
        )
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        assert (sa.speculative_num_steps) == (1)
        assert (sa.speculative_num_draft_tokens) == (2)

    def _resolve_tree(self, algorithm, topk, steps, nodes, **overrides):
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-algorithm",
                algorithm,
                "--speculative-eagle-topk",
                str(topk),
                "--speculative-num-steps",
                str(steps),
                "--speculative-num-draft-tokens",
                str(nodes),
            ]
        )
        sa = self._from_cli_args_no_init(args)
        sa.resolve_basic_defaults()
        for name, value in overrides.items():
            setattr(sa, name, value)
        sa.resolve_parallelism()
        sa.resolve_speculative_decoding()
        return sa

    def test_speculative_eagle_topk_accepts_draft_trees(self):
        for algorithm in ("EAGLE3", "MTP"):
            sa = self._resolve_tree(algorithm, 4, 5, 16)
            assert (sa.speculative_eagle_topk) == (4)
        # Largest budgets: 16 children per node; 64 nodes and 64 lane slots.
        self._resolve_tree("EAGLE3", 16, 5, 64)
        self._resolve_tree("EAGLE3", 8, 9, 64)

    def test_speculative_eagle_topk_rejects_unsupported_trees(self, subtests):
        cases = [
            (("DFLASH", 4, 15, 16), {}, "needs --speculative-algorithm EAGLE3, EAGLE4"),
            (("EAGLE3", 17, 2, 16), {}, "1..16 children"),
            (("EAGLE3", 0, 3, 4), {}, "1..16 children"),
            (("EAGLE3", 4, 11, 16), {}, "1..10 steps"),
            (("EAGLE3", 16, 6, 64), {}, "lane slots"),
            (("EAGLE3", 4, 5, 8), {}, "lane slots"),
            (("EAGLE3", 4, 5, 65), {}, "speculative_num_draft_tokens=65"),
            (("EAGLE3", 2, 2, 8), {}, "speculative_num_draft_tokens=8"),
            (("EAGLE3", 4, 5, 16), {"grammar_backend": "xgrammar"}, "structured"),
            (("EAGLE3", 4, 5, 16), {"enable_mixed_batch": True}, "mixed batches"),
            (("EAGLE3", 4, 5, 16), {"disaggregation_mode": "decode"}, "disaggregation"),
            (("EAGLE3", 4, 5, 16), {"pipeline_parallel_size": 2}, "pipeline stages"),
            (
                ("EAGLE3", 4, 5, 16),
                {"world_size": 2, "data_parallel_size": 2},
                "attention data parallelism",
            ),
            # Attention DP derived from the world and attention TP sizes, not set explicitly.
            (
                ("EAGLE3", 4, 5, 16),
                {"world_size": 4, "attn_tp_size": 2},
                "attention data parallelism",
            ),
        ]
        for (algorithm, topk, steps, nodes), overrides, message in cases:
            with subtests.test(
                algorithm=algorithm, topk=topk, steps=steps, nodes=nodes
            ):
                with pytest.raises((ValueError), match=(message)):
                    self._resolve_tree(algorithm, topk, steps, nodes, **overrides)

    def test_speculative_eagle_topk_requires_algorithm_and_node_budget(self):
        args = self._parse_args(
            ["--model", "test/model", "--speculative-eagle-topk", "4"]
        )
        sa = self._from_cli_args_no_init(args)
        with pytest.raises((ValueError), match=("needs --speculative-algorithm")):
            sa.resolve_basic_defaults()
        args = self._parse_args(
            [
                "--model",
                "test/model",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-eagle-topk",
                "4",
            ]
        )
        sa = self._from_cli_args_no_init(args)
        with pytest.raises((ValueError), match=("node budget")):
            sa.resolve_basic_defaults()

    def test_speculative_chain_rejects_mismatched_draft_tokens(self):
        with pytest.raises((ValueError), match=("draft chain verifies")):
            self._resolve_tree("EAGLE3", 1, 3, 8)
        self._resolve_tree("EAGLE3", 1, 3, 4)

    def test_dp_sampling_is_opt_in(self):
        args = self._parse_args(["--model", "test/model"])
        sa = self._from_cli_args_no_init(args)
        assert not (sa.dp_sampling)
        assert (sa.dp_sampling_min_bs) is None

        args = self._parse_args(["--model", "test/model", "--dp-sampling"])
        sa = self._from_cli_args_no_init(args)
        assert sa.dp_sampling

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

        assert (sa.model) == ("deepseek-ai/DeepSeek-V3.1")
        assert (sa.attn_tp_size) == (8)
        assert sa.enable_expert_parallel
        assert (sa.served_model_name) == ("ds31")

    def test_help_uses_expected_metavars(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with pytest.raises((SystemExit)):
                parser.parse_args(["--help"])

        help_text = stdout.getvalue()
        assert ("--max-num-seqs MAX_NUM_SEQS") in (help_text)
        assert ("--max-prefill-tokens MAX_PREFILL_TOKENS") in (help_text)
        assert ("--chunked-prefill-size CHUNKED_PREFILL_SIZE") in (help_text)
        assert ("--gpu-memory-utilization GPU_MEMORY_UTILIZATION") in (help_text)
        assert ("--distributed-timeout-seconds DISTRIBUTED_TIMEOUT_SECONDS") in (
            help_text
        )
        assert ("--all2all-backend ALL2ALL_BACKEND") in (help_text)
        assert ("--hf-overrides HF_OVERRIDES") in (help_text)
        assert ("MAX_RUNNING_REQUESTS") not in (help_text)
        assert ("MEM_FRACTION_STATIC") not in (help_text)
        assert ("DIST_TIMEOUT") not in (help_text)
        assert ("MOE_A2A_BACKEND") not in (help_text)
        assert ("JSON_MODEL_OVERRIDE_ARGS") not in (help_text)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
