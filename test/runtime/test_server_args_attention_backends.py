"""Regression tests for --attention-backend / --drafter-attention-backend names.

Guards against the bug where --drafter-attention-backend rejected valid main-model
backends (e.g. trtllm_mla) because its argparse `choices` was a narrower subset
of --attention-backend's. Both flags are now validated against one backend
registry after plugin discovery, so they accept exactly the same names.

Also covers the cache-producer identity the resolved names produce: the
helper-level resolution and the AttentionBuild fields the real builder exports
for L3 namespacing.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import argparse
import json
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch, ModelConfig
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.attention.backends.state import kda
from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.registry import (
    _cache_backend_name,
    _get_default_backend_name,
)
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

    def test_attention_backend_accepts_gluon(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--attention-backend", "gluon"]
        )
        self.assertEqual(args.attention_backend, "gluon")

    def test_attention_backend_accepts_mha_kernel_solutions(self):
        for backend in ("fa3", "fa4", "triton", "flashinfer"):
            args = self._build_parser().parse_args(
                ["--model", "x", "--attention-backend", backend]
            )
            self.assertEqual(args.attention_backend, backend)

    def test_attention_backend_uses_generic_mha_for_ascend(self):
        import tokenspeed.runtime.layers.attention.backends  # noqa: F401

        registry.validate_attention_backend_name("mha", flag="--attention-backend")
        for name in ("ascend_mha", "npu"):
            with self.assertRaisesRegex(ValueError, "Unknown --attention-backend"):
                registry.validate_attention_backend_name(
                    name, flag="--attention-backend"
                )

    def test_drafter_attention_backend_accepts_trtllm_mla(self):
        """Regression: trtllm_mla must be accepted here too."""
        args = self._build_parser().parse_args(
            ["--model", "x", "--drafter-attention-backend", "trtllm_mla"]
        )
        self.assertEqual(args.drafter_attention_backend, "trtllm_mla")

    def test_drafter_attention_backend_accepts_gluon(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--drafter-attention-backend", "gluon"]
        )
        self.assertEqual(args.drafter_attention_backend, "gluon")

    def test_flags_defer_backend_names_to_the_registry(self):
        parser = self._build_parser()
        # Plugins register backends after argument parsing, so neither flag
        # may carry a closed argparse choice list.
        self.assertIsNone(self._action(parser, "attention_backend").choices)
        self.assertIsNone(self._action(parser, "drafter_attention_backend").choices)

    def test_invalid_backend_rejected_on_both_flags(self):
        import tokenspeed.runtime.layers.attention.backends  # noqa: F401

        for flag in ("--attention-backend", "--drafter-attention-backend"):
            registry.validate_attention_backend_name("trtllm_mla", flag=flag)
            registry.validate_attention_backend_name(
                registry.HYBRID_LINEAR_ATTN_BACKEND, flag=flag
            )
            registry.validate_attention_backend_name(None, flag=flag)
            with self.assertRaisesRegex(ValueError, f"Unknown {flag} 'bogus'"):
                registry.validate_attention_backend_name("bogus", flag=flag)

    def test_inline_detokenizer_is_forced_on(self):
        args = prepare_server_args(["--model", "x"])
        self.assertTrue(args.enable_inline_detokenizer)

    def test_kda_prefill_graph_uses_shared_args_not_worker_environment(self):
        spec = MHAConfig(
            num_attention_heads=4, num_kv_heads=4, head_dim=128, attn_tp_size=1
        )
        components = {
            SoftmaxAttnConfig: spec,
            LinearAttnConfig: SimpleNamespace(
                layer_ids=(0,),
                replay_ssm=False,
                draft_tree=False,
                tree_node_state_workspace=True,
            ),
        }
        config = SimpleNamespace(
            device="cpu",
            dtype=None,
            is_draft=False,
            speculative_num_draft_tokens=1,
            max_bs=4,
            component=components.get,
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(full_attention_layer_ids=[1]),
            attention_arch=AttentionArch.MLA,
        )
        for disabled in (False, True):
            argv = ["--model", "x"]
            if disabled:
                argv.append("--disable-kda-prefill-graph")
            shared_args = prepare_server_args(argv)
            self.assertIs(shared_args.disable_kda_prefill_graph, disabled)
            self.assertFalse(shared_args.disable_prefill_graph)
            self.assertFalse(shared_args.enforce_eager)
            # Simulate workers receiving the same serialized server arguments
            # despite conflicting legacy environment settings. Keep the real
            # KDA constructor so reintroducing a local env read fails this test.
            for worker_env in ("0", "1"):
                with (
                    self.subTest(disabled=disabled, worker_env=worker_env),
                    mock.patch.dict(
                        os.environ, {"TOKENSPEED_KDA_PREFILL_GRAPH": worker_env}
                    ),
                    mock.patch.object(
                        registry,
                        "_create_attn_backend_with_name",
                        return_value=SimpleNamespace(device="cpu"),
                    ),
                    mock.patch.object(
                        kda, "resolve_kda_backend", return_value="cutedsl_kda"
                    ),
                    mock.patch.object(registry, "is_qwen4_exp", return_value=False),
                ):
                    backend = registry._create_hybrid_linear_attn_backend(
                        pickle.loads(pickle.dumps(shared_args)),
                        model_config,
                        config,
                        pool=SimpleNamespace(state_group_by_layer={0: "state"}),
                        full_attn_backend_name="mla",
                        linear_attention="kda",
                    )
                    self.assertIs(
                        backend.linear_attn_backend._prefill_graph_enabled, not disabled
                    )
                    self.assertEqual(
                        backend.linear_attn_backend.kda_backend, "cutedsl_kda"
                    )

    def test_model_path_alias_sets_model(self):
        args = self._build_parser().parse_args(["--model-path", "x"])
        self.assertEqual(args.model, "x")

    def test_prepare_server_args_accepts_model_path_alias(self):
        args = prepare_server_args(["--model-path", "x"])
        self.assertEqual(args.model, "x")

    def test_defaults_to_mha_for_mha(self):
        self.assertEqual(registry._get_default_backend_name(AttentionArch.MHA), "mha")

    def test_mha_kernel_solution_backends_use_mha_backend(self):
        from tokenspeed.runtime.layers.attention.backends.paged.mha import (
            MHAAttnBackend,
        )

        for backend in ("mha", "fa3", "fa4", "triton", "flashinfer"):
            self.assertIs(
                registry._get_backend_cls(backend, AttentionArch.MHA),
                MHAAttnBackend,
            )

    def test_mla_backend_registered_for_mla(self):
        from tokenspeed.runtime.layers.attention.backends.paged.mla import (
            MLAAttnBackend,
        )

        self.assertIs(
            registry._get_backend_cls("mla", AttentionArch.MLA),
            MLAAttnBackend,
        )

    def test_gluon_backend_registered_for_mla(self):
        from tokenspeed.runtime.layers.attention.backends.paged.mla import (
            MLAAttnBackend,
        )

        self.assertIs(
            registry._get_backend_cls("gluon", AttentionArch.MLA),
            MLAAttnBackend,
        )

    def test_gluon_mla_backend_forces_gluon_kernel_solution(self):
        import torch

        from tokenspeed.runtime.layers.attention.backends.paged.mla import (
            MLAAttnBackend,
        )
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

        spec = MLAConfig(
            backend_name="gluon",
            num_attention_heads=128,
            num_kv_heads=1,
            head_dim=576,
            attn_tp_size=8,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            kv_cache_dim=576,
            scaling=192**-0.5,
        )
        config = AttnConfig(
            device="cpu",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_quant_method="none",
            prefix_granularity=128,
            is_draft=True,
            speculative_num_draft_tokens=4,
            context_len=65536,
            kernel_page_size=64,
            max_bs=8,
            components=(spec,),
        )

        self.assertEqual(
            MLAAttnBackend(config, spec, kernel_page_size=64).kernel_solution,
            "gluon",
        )

    def test_dsa_routes_dense_attention_by_platform(self):
        import dataclasses

        from tokenspeed.runtime.layers.attention.backends.paged import (
            dsa as dsa_backend,
        )
        from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig

        config = object()
        # The dense delegate interprets backend_name itself, so _make_dense_leaf
        # must hand it a spec cleared of the wrapper-selecting 'dsa' name.
        spec = SoftmaxAttnConfig(
            backend_name="dsa",
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            attn_tp_size=1,
        )
        dense_backend = object()

        for platform, cls_name in (
            (SimpleNamespace(is_nvidia=True, is_amd=False), "TRTLLMMLABackend"),
            (SimpleNamespace(is_nvidia=False, is_amd=True), "MLAAttnBackend"),
        ):
            with (
                self.subTest(name=cls_name),
                mock.patch.object(
                    dsa_backend, cls_name, return_value=dense_backend
                ) as create,
            ):
                self.assertIs(
                    dsa_backend._make_dense_leaf(config, spec, platform, 64),
                    dense_backend,
                )
                create.assert_called_once_with(
                    config,
                    dataclasses.replace(spec, backend_name=None),
                    kernel_page_size=64,
                )

    def test_named_backend_routing_does_not_mutate_source_spec(self):
        from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig

        source = SoftmaxAttnConfig(
            backend_name="parent",
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            attn_tp_size=1,
        )
        config = SimpleNamespace(component=lambda _: source)
        routed = {}

        class ProbeBackend:
            def __init__(self, _config, spec):
                routed["source_name"] = source.backend_name
                routed["spec"] = spec

        with mock.patch.object(registry, "_get_backend_cls", return_value=ProbeBackend):
            registry._create_attn_backend_with_name("child", AttentionArch.MHA, config)

        self.assertEqual(routed["source_name"], "parent")
        self.assertEqual(routed["spec"].backend_name, "child")
        self.assertIsNot(routed["spec"], source)

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
        # world_size=2 resolves mapping.attn.tp_size=2; spec_context_pad
        # derives from the speculative widths (3 overshoot spans * 4 tokens).
        server_args = ServerArgs(
            model="x",
            world_size=2,
            attention_backend=None,
            drafter_attention_backend=None,
            attn_tp_size=None,
            kv_cache_dtype="auto",
            max_num_seqs=8,
            data_parallel_size=None,
            prefix_granularity=64,
            disaggregation_mode="null",
            kv_cache_quant_method="none",
            speculative_algorithm="EAGLE3",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            skip_softmax_threshold=0.0,
        )
        server_args.mapping.rank = 0
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            context_len=4096,
            num_attention_layers=2,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
        )

        config = MHAConfig.generate(server_args, model_config)

        self.assertEqual(config.speculative_num_steps, 3)
        self.assertEqual(config.speculative_num_draft_tokens, 4)
        self.assertEqual(config.context_len, 4108)


class _GpuBackendStandIn:
    """Stands in for GPU backend construction, injected through the
    ``reuse_*_backend`` seam of ``create_attn_components``."""

    def __init__(self):
        self.cache_pool = None

    def set_cache_pool(self, pool):
        self.cache_pool = pool


class TestCacheBackendIdentity(unittest.TestCase):
    """The cache-producer identity the runtime exports for L3 namespacing:
    auto choices must resolve, the MSA hybrid wrapper must keep its dense
    sub-backend, and a build without a draft must export an empty draft
    identity."""

    def test_attention_build_exports_resolved_backend_names(self):
        # The AttentionBuild identity is what _cache_backend_name returns at
        # the build site: an unset (auto) request resolves to the
        # architecture's default, and the draft identity is empty when no
        # draft backend was built.
        def mha_config():
            return MHAConfig(
                num_attention_heads=1, num_kv_heads=1, head_dim=1, attn_tp_size=1
            )

        def msa_config(dense_name):
            return MSAConfig(
                num_attention_heads=1,
                num_kv_heads=1,
                head_dim=1,
                attn_tp_size=1,
                full_attn_backend_name=dense_name,
            )

        cases = [
            (None, None, True, ("mha", "mla")),
            ("fa3", "fa4", True, ("fa3", "fa4")),
            ("tokenspeed_mla", "flashinfer", True, ("tokenspeed_mla", "flashinfer")),
            ("fa3", "fa4", False, ("fa3", "")),
        ]
        for target_name, draft_name, has_draft, expected in cases:
            with self.subTest(
                target=target_name, draft=draft_name, has_draft=has_draft
            ):
                target = _cache_backend_name(
                    mha_config(), target_name, AttentionArch.MHA
                )
                draft = (
                    _cache_backend_name(mha_config(), draft_name, AttentionArch.MLA)
                    if has_draft
                    else ""
                )
                self.assertEqual((target, draft), expected)

        # Auto resolution is the architecture default.
        self.assertEqual(_get_default_backend_name(AttentionArch.MHA), "mha")
        self.assertEqual(_get_default_backend_name(AttentionArch.MLA), "mla")
        self.assertEqual(_get_default_backend_name(AttentionArch.MSA), "msa")

        # Outer MSA names stay identical while its dense implementations
        # differ. Both target and draft identities must retain that choice.
        for target_dense, draft_dense in (
            ("trtllm", "fa3"),
            ("fa3", "trtllm"),
            (None, None),
        ):
            with self.subTest(target_dense=target_dense, draft_dense=draft_dense):
                target = _cache_backend_name(
                    msa_config(target_dense), "msa", AttentionArch.MSA
                )
                draft = _cache_backend_name(
                    msa_config(draft_dense), "msa", AttentionArch.MSA
                )
                self.assertEqual(
                    (target, draft),
                    (f"msa:{target_dense or 'mha'}", f"msa:{draft_dense or 'mha'}"),
                )

    _LLAMA_CONFIG = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "dtype": "bfloat16",
        "vocab_size": 32000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
    }

    _MINIMAX_M3_CONFIG = {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "dtype": "bfloat16",
        "text_config": {
            "vocab_size": 32000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 512,
            "layer_types": ["full_attention", "minimax_m3_sparse"],
            "index_head_dim": 16,
            "index_n_heads": 2,
            "index_topk_blocks": 4,
            "index_init_blocks": 1,
            "index_local_blocks": 4,
        },
    }

    def _runtime(self, directory, *, attention_backend, drafter_attention_backend):
        args = ServerArgs(
            model=directory,
            device="cpu",
            dtype="bfloat16",
            trust_remote_code=False,
            revision=None,
            max_model_len=512,
            attention_backend=attention_backend,
            drafter_attention_backend=drafter_attention_backend,
            prefix_granularity=64,
            load_format="auto",
            speculative_algorithm=None,
            world_size=1,
            attn_tp_size=None,
            data_parallel_size=None,
            pipeline_parallel_size=1,
            max_num_seqs=2,
            chunked_prefill_size=384,
            max_total_tokens=512,
            kv_cache_quant_method="none",
            disaggregation_mode="null",
            enforce_eager=True,
            disable_prefill_graph=True,
            seed=0,
        )
        args.mapping.rank = 0
        model = ModelConfig(
            model_path=args.model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            context_length=args.max_model_len,
            model_override_args="{}",
            dtype=args.dtype,
            quantization=args.quantization,
            override_config_file=None,
            is_draft_worker=False,
            server_args=args,
        )
        return args, model

    def _build(self, args, model, *, draft_model=None):
        # The real builder end to end on CPU arenas; only the GPU backend
        # construction is substituted, through the designed reuse seam.
        target = _GpuBackendStandIn()
        draft = _GpuBackendStandIn() if draft_model is not None else None
        build = registry.create_attn_components(
            args,
            model,
            gpu_id=0,
            rank=0,
            gpu_memory=0,
            draft_model_config=draft_model,
            graph_reserve_bytes=0,
            post_profile_bytes=0,
            probe_batch_rows=None,
            profiled_cache_bytes=64 * 1024 * 1024,
            reuse_target_backend=target,
            reuse_draft_backend=draft,
        )
        return build, target, draft

    def _write_config(self, directory, raw):
        with open(f"{directory}/config.json", "w", encoding="utf-8") as handle:
            json.dump(raw, handle)

    def test_create_attn_components_resolves_auto_and_empty_draft(self):
        import tokenspeed.runtime.layers.attention.backends  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            self._write_config(directory, self._LLAMA_CONFIG)
            args, model = self._runtime(
                directory, attention_backend=None, drafter_attention_backend=None
            )
            build, target, _ = self._build(args, model)

        self.assertEqual(build.attention_backend_name, "mha")
        self.assertEqual(build.draft_attention_backend_name, "")
        self.assertIs(build.attn_backend, target)
        self.assertIsNone(build.draft_attn_backend)
        self.assertIs(target.cache_pool, build.token_to_kv_pool)

    def test_create_attn_components_keeps_explicit_target_and_draft_names(self):
        import tokenspeed.runtime.layers.attention.backends  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            self._write_config(directory, self._LLAMA_CONFIG)
            args, model = self._runtime(
                directory,
                attention_backend="fa3",
                drafter_attention_backend="fa4",
            )
            build, target, draft = self._build(args, model, draft_model=model)

        self.assertEqual(build.attention_backend_name, "fa3")
        self.assertEqual(build.draft_attention_backend_name, "fa4")
        self.assertIs(build.attn_backend, target)
        self.assertIs(build.draft_attn_backend, draft)
        self.assertIs(draft.cache_pool, build.draft_token_to_kv_pool)

    @unittest.skipUnless(
        current_platform().is_nvidia,
        "the trtllm backend registers only on NVIDIA (backends/__init__.py)",
    )
    def test_create_attn_components_keeps_msa_dense_sub_backend(self):
        import tokenspeed.runtime.layers.attention.backends  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            self._write_config(directory, self._MINIMAX_M3_CONFIG)
            args, model = self._runtime(
                directory,
                attention_backend="trtllm",
                drafter_attention_backend="fa3",
            )
            build, _, _ = self._build(args, model, draft_model=model)
        self.assertEqual(build.attention_backend_name, "msa:trtllm")
        self.assertEqual(build.draft_attention_backend_name, "msa:fa3")

    def test_create_attn_components_msa_auto_names_dense_default(self):
        import tokenspeed.runtime.layers.attention.backends  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            self._write_config(directory, self._MINIMAX_M3_CONFIG)
            args, model = self._runtime(
                directory, attention_backend=None, drafter_attention_backend=None
            )
            build, _, _ = self._build(args, model, draft_model=model)
        self.assertEqual(build.attention_backend_name, "msa:mha")
        self.assertEqual(build.draft_attention_backend_name, "msa:mha")


class TestKdaBackendPolicy(unittest.TestCase):
    def test_prefill_backend_selection(self):
        cases = (
            # AMD keeps kernel-registry selection, including explicit NVIDIA policies.
            (True, False, False, "auto", "auto"),
            (True, False, False, "cutedsl_kda", "auto"),
            # NVIDIA auto prefers CuteDSL, then FlashKDA, then FLA.
            (False, True, True, "auto", "cutedsl_kda"),
            (False, True, False, "auto", "flashkda"),
            (False, False, False, "auto", "fla"),
            # Explicit NVIDIA choices remain authoritative.
            (False, True, True, "fla", "fla"),
            (False, True, True, "flashkda", "flashkda"),
            (False, True, True, "cutedsl_kda", "cutedsl_kda"),
        )
        for is_amd, is_hopper_plus, cute_supported, requested, expected in cases:
            with (
                self.subTest(is_amd=is_amd, requested=requested, expected=expected),
                mock.patch.object(
                    kda,
                    "current_platform",
                    return_value=SimpleNamespace(
                        is_amd=is_amd, is_hopper_plus=is_hopper_plus
                    ),
                ),
                mock.patch(
                    "tokenspeed_kernel.ops.attention.kda.cute_dsl.cutedsl_kda_supported",
                    return_value=cute_supported,
                ) as supports_cute,
            ):
                self.assertEqual(kda.resolve_kda_backend(requested), expected)
                if is_amd:
                    supports_cute.assert_not_called()

    def test_unsupported_explicit_cutedsl_is_rejected(self):
        with (
            mock.patch.object(
                kda,
                "current_platform",
                return_value=SimpleNamespace(is_amd=False, is_hopper_plus=True),
            ),
            mock.patch(
                "tokenspeed_kernel.ops.attention.kda.cute_dsl.cutedsl_kda_supported",
                return_value=False,
            ),
            self.assertRaisesRegex(ValueError, "requires an NVIDIA sm_100 or sm_103"),
        ):
            kda.resolve_kda_backend("cutedsl_kda")


class TestPagedRouterNameResolution(unittest.TestCase):
    """Names reaching create_paged_router select the LEAF; wrapper names
    must not leak into leaf constructors that interpret backend_name
    themselves (MHA/MLA kernel-solution maps)."""

    def _config(self):
        import torch

        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig

        spec = MHAConfig(
            backend_name="hybrid_linear_attn",  # the composite sentinel
            num_attention_heads=2,
            num_kv_heads=2,
            head_dim=8,
            attn_tp_size=1,
        )
        return AttnConfig(
            device="cpu",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            prefix_granularity=64,
            kernel_page_size=64,
            context_len=256,
            max_bs=2,
            kv_cache_quant_method="none",
            components=(spec,),
        )

    def test_hybrid_sentinel_resolves_the_arch_default_leaf(self):
        # 'hybrid_linear_attn' names the WRAPPER; the paged leaf under it
        # auto-resolves from the arch. Startup used to die with
        # "Unknown attention backend: 'hybrid_linear_attn'".
        config = self._config()
        router = registry.create_paged_router(config, AttentionArch.MHA)
        leaf = router._leaf_factory("full_attention", 64)
        self.assertEqual(type(leaf).__name__, "MHAAttnBackend")
        # And the leaf's own spec resolved a kernel solution (backend_name
        # was cleared, not passed through as the sentinel).
        self.assertNotEqual(
            getattr(leaf, "kernel_solution", "unset"), "hybrid_linear_attn"
        )

    def test_leaf_factory_does_not_mutate_the_shared_spec(self):
        config = self._config()
        spec = config.component(registry.SoftmaxAttnConfig)
        router = registry.create_paged_router(
            config, AttentionArch.MHA, backend_name="mha"
        )
        router._leaf_factory("full_attention", 64)
        # Lazy leaf construction must not leave a mutated shared component.
        self.assertEqual(spec.backend_name, "hybrid_linear_attn")


class TestDecodeHostL2(unittest.TestCase):
    def test_decode_enables_host_l2_without_prefix_matching(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "decode"
        args.kv_parallel_size = 1
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = False
        args.kvstore_io_backend = "kernel"

        args._handle_kvstore()
        args.validate_cache_options()

        self.assertTrue(args.enable_kvstore)


class TestDisaggregationGraphFlags(unittest.TestCase):
    """The prefill role is a role with no decode step, not an eager role."""

    def test_prefill_role_keeps_the_ordinary_graph_flags(self):
        args = prepare_server_args(["--model", "x", "--disaggregation-mode", "prefill"])
        self.assertFalse(args.enforce_eager)
        self.assertFalse(args.disable_prefill_graph)

    def test_prefill_role_honours_explicit_eager(self):
        args = prepare_server_args(
            ["--model", "x", "--disaggregation-mode", "prefill", "--enforce-eager"]
        )
        self.assertTrue(args.enforce_eager)

    def test_pipeline_parallelism_forces_eager(self):
        args = prepare_server_args(
            [
                "--model",
                "x",
                "--disaggregation-mode",
                "prefill",
                "--pipeline-parallel-size",
                "2",
            ]
        )
        self.assertTrue(args.enforce_eager)

    def test_pipeline_debug_without_pd_forces_eager(self):
        with mock.patch.dict(os.environ, {"TS_PP_DEBUG_ALLOW_NON_PREFILL": "1"}):
            args = prepare_server_args(
                ["--model", "x", "--pipeline-parallel-size", "2"]
            )
        self.assertTrue(args.enforce_eager)

    @staticmethod
    def _pipeline_prefill_args(algorithm: str, *extra: str) -> list[str]:
        return [
            "--model",
            "x",
            "--disaggregation-mode",
            "prefill",
            "--pipeline-parallel-size",
            "2",
            "--speculative-algorithm",
            algorithm,
            *extra,
        ]

    def test_pipeline_prefill_accepts_last_stage_drafters(self):
        # The drafter runs on the last stage, the only stage that samples;
        # DSPARK additionally produces context across stages.
        for algorithm in ("MTP", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                args = prepare_server_args(self._pipeline_prefill_args(algorithm))
                self.assertEqual(args.speculative_algorithm, algorithm)
                self.assertTrue(args.enforce_eager)

    def test_pipeline_speculation_requires_the_prefill_role(self):
        with self.assertRaisesRegex(ValueError, "disaggregation-mode prefill"):
            prepare_server_args(
                [
                    "--model",
                    "x",
                    "--disaggregation-mode",
                    "decode",
                    "--pipeline-parallel-size",
                    "2",
                    "--speculative-algorithm",
                    "MTP",
                ]
            )
        # The PP debug escape hatch runs without PD; it has no decode token
        # feedback to draft against either.
        with (
            mock.patch.dict(os.environ, {"TS_PP_DEBUG_ALLOW_NON_PREFILL": "1"}),
            self.assertRaisesRegex(ValueError, "only on a prefill server"),
        ):
            prepare_server_args(
                [
                    "--model",
                    "x",
                    "--pipeline-parallel-size",
                    "2",
                    "--speculative-algorithm",
                    "MTP",
                ]
            )

    def test_pipeline_rejects_drafts_that_read_taps_from_several_stages(self):
        # DFLASH has no cross-stage context production; EAGLE3's aux taps
        # are not carried through the stage boundary.
        for algorithm in ("DFLASH", "EAGLE3"):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(ValueError, f"{algorithm} is not supported"),
            ):
                prepare_server_args(self._pipeline_prefill_args(algorithm))

    def test_pipeline_dspark_keeps_matching_dense_and_attention_tp(self):
        # The DSPARK draft reduces attention-TP embedding partials over the
        # dense TP group; MTP embeds with a reduced lookup and carries no
        # such rule.
        narrow_dense = (
            "--world-size",
            "4",
            "--attn-tp-size",
            "2",
            "--dense-tp-size",
            "1",
        )
        with self.assertRaisesRegex(ValueError, "matching dense/attention TP"):
            prepare_server_args(self._pipeline_prefill_args("DSPARK", *narrow_dense))
        args = prepare_server_args(self._pipeline_prefill_args("MTP", *narrow_dense))
        self.assertEqual(args.mapping.dense.tp_size, 1)
        self.assertEqual(args.mapping.attn.tp_size, 2)


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

    def test_memory_backend_keeps_host_io(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = "memory"
        args.kvstore_io_backend = "direct"

        args._handle_kvstore()

        self.assertTrue(args.enable_kvstore)
        self.assertEqual(args.kvstore_io_backend, "direct")

    def test_l3_requires_host_l2(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.disable_kvstore = True
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = "mooncake"
        args.kvstore_io_backend = "direct"

        with self.assertRaisesRegex(ValueError, "requires Host L2"):
            args._handle_kvstore()

    def test_mooncake_backend_keeps_host_io(self):
        args = object.__new__(ServerArgs)
        args.disaggregation_mode = "null"
        args.disable_kvstore = False
        args.enable_kvstore = False
        args.enable_prefix_caching = True
        args.kvstore_storage_backend = "mooncake"
        args.kvstore_io_backend = "direct"

        args._handle_kvstore()

        self.assertTrue(args.enable_kvstore)
        self.assertEqual(args.kvstore_io_backend, "direct")


if __name__ == "__main__":
    unittest.main()
