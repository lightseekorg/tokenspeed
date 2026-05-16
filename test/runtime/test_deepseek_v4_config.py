import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    configure_deepseek_v4_attention,
    is_deepseek_v4,
)
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import (
    deepseek_v4 as deepseek_v4_backend,
)
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DeepseekV4AttentionOpUnavailable,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_indexer_topk_reference,
    fused_qnorm_rope_kv_insert,
    has_fused_qnorm_rope_kv_insert,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4ForwardMetadata,
    DeepseekV4TokenToKVPool,
    _group_slot_mapping_from_raw,
    deepseek_v4_cache_layout_from_config,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.moe.backends.mxfp4.flashinfer import (
    _get_flashinfer_mxfp4_device_permute_indices,
    _reorder_w1w3_to_w3w1,
)
from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.models import deepseek_v4 as deepseek_v4_model
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4MLP,
    DeepseekV4MoEGate,
    _deepseek_v4_fused_select_experts,
    _deepseek_v4_gather_indexer_mxfp4_cache,
    _deepseek_v4_get_fp8_linear_deep_gemm,
    _deepseek_v4_indexer_decode_max_len,
    _deepseek_v4_indexer_prefill_gather_plan,
    _deepseek_v4_indexer_prefill_max_logits_bytes,
    _deepseek_v4_indexer_prefill_metadata,
    _deepseek_v4_indexer_prefill_request_chunks,
    _deepseek_v4_indexer_prefill_request_gather_plan,
    _deepseek_v4_indexer_prefill_topk_chunks,
    _deepseek_v4_indexer_topk_from_cache_batched,
    _deepseek_v4_indexer_topk_from_logits,
    _deepseek_v4_prefill_topk_op_available,
    _deepseek_v4_reorder_c4_ape_2604,
    _DeepseekV4TopKBuffer,
    _fp8_act_quant_dequant,
    _fp8_linear,
    deepseek_v4_attention_layout,
    deepseek_v4_rope_config,
    deepseek_v4_select_experts,
    hc_head,
    mhc_post,
    mhc_pre,
    pack_topk_as_router_logits,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import (
    _CONFIG_REGISTRY,
    _wrap_deepseek_v4_tokenizer,
    get_tokenizer,
    prefers_deepseek_v4_tokenizer,
)


class TestDeepseekV4Config(unittest.TestCase):
    quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
    }

    def test_config_registry(self):
        self.assertEqual(DeepseekV4Config.model_type, "deepseek_v4")
        self.assertIs(_CONFIG_REGISTRY["deepseek_v4"], DeepseekV4Config)

    def test_forward_mode_mixed_predicate(self):
        self.assertTrue(ForwardMode.MIXED.is_mixed())
        self.assertFalse(ForwardMode.EXTEND.is_mixed())
        self.assertFalse(ForwardMode.DECODE.is_mixed())
        self.assertEqual(ForwardMode.from_num_extends(0, 0), ForwardMode.IDLE)
        self.assertEqual(ForwardMode.from_num_extends(0, 2), ForwardMode.DECODE)
        self.assertEqual(
            ForwardMode.from_num_extends(0, 2, has_drafter=True),
            ForwardMode.TARGET_VERIFY,
        )
        self.assertEqual(ForwardMode.from_num_extends(2, 2), ForwardMode.EXTEND)
        self.assertEqual(ForwardMode.from_num_extends(1, 2), ForwardMode.MIXED)

    def test_cuda_graph_group_table_padding_uses_dummy_page_rows(self):
        table = torch.tensor([[5, -1]], dtype=torch.int32)
        padded = CudaGraphWrapper._pad_block_tables_to_padded_bs(
            {"v4.swa": table},
            actual_bs=1,
            padded_bs=3,
        )

        self.assertEqual(padded["v4.swa"].tolist(), [[5, -1], [0, 0], [0, 0]])

    def test_cuda_graph_replay_keeps_idle_actual_bs_with_padded_group_tables(self):
        captured = {}

        class FakeBackend:
            uses_paged_cache_groups = True
            uses_padded_decode_token_mask = True

            def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.attn_backend = FakeBackend()
        wrapper.draft_attn_backend = None

        wrapper._init_replay_metadata(
            padded_bs=4,
            actual_bs=0,
            req_pool_indices=torch.zeros(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            req_to_page=torch.zeros((1, 1), dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            paged_cache_block_tables={
                "v4.swa": torch.zeros((4, 1), dtype=torch.int32),
            },
        )

        self.assertEqual(captured["kwargs"]["actual_bs"], 0)
        self.assertEqual(
            captured["kwargs"]["paged_cache_block_tables"]["v4.swa"].shape,
            (4, 1),
        )

    def test_deepseek_v4_tokenizer_wrapper_uses_model_encoder(self):
        calls = []

        class DummyTokenizer:
            vocab_size = 5

            def __call__(self, text, add_special_tokens=False, **kwargs):
                self.last_call = (text, add_special_tokens, kwargs)
                return {"input_ids": [len(text)]}

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {"<extra>": 5}

        def encode_messages(messages, **kwargs):
            calls.append((messages, kwargs))
            return "<encoded>"

        tokenizer = _wrap_deepseek_v4_tokenizer(DummyTokenizer(), encode_messages)

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            truncation=True,
            max_length=16,
        )

        self.assertEqual(prompt, "<encoded>")
        self.assertEqual(token_ids, [9])
        self.assertEqual(len(tokenizer), 6)
        self.assertEqual(calls[0][1]["thinking_mode"], "thinking")
        self.assertIsNone(calls[0][1]["reasoning_effort"])
        self.assertEqual(calls[1][1]["thinking_mode"], "chat")
        self.assertEqual(
            tokenizer.last_call,
            ("<encoded>", False, {"truncation": True, "max_length": 16}),
        )

    def test_deepseek_v4_tokenizer_is_auto_selected_by_architecture(self):
        self.assertTrue(prefers_deepseek_v4_tokenizer(["DeepseekV4ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(["KimiK2ForCausalLM"]))
        self.assertFalse(prefers_deepseek_v4_tokenizer(None))

    def test_auto_tokenizer_mode_wraps_deepseek_v4_architecture(self):
        class DummyTokenizer:
            vocab_size = 5

            def __call__(self, text, add_special_tokens=False, **kwargs):
                return {"input_ids": [len(text)]}

            def encode(self, text, add_special_tokens=False, **kwargs):
                return [len(text)]

            def get_added_vocab(self):
                return {}

        def encode_messages(messages, **kwargs):
            return "<encoded>"

        with (
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils._load_deepseek_v4_encode_messages",
                return_value=encode_messages,
            ),
        ):
            tokenizer = get_tokenizer(
                "deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_mode="auto",
                architectures=["DeepseekV4ForCausalLM"],
            )

        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
            ),
            [9],
        )

    def test_deepseek_v4_server_args_cli_flags_round_trip(self):
        from tokenspeed.runtime.utils.env import (
            global_server_args_dict,
            global_server_args_dict_update,
        )
        from tokenspeed.runtime.utils.server_args import ServerArgs

        # Defaults match dataclass declaration
        self.assertFalse(ServerArgs.disable_deepseek_v4_fast_mhc)
        self.assertEqual(ServerArgs.deepseek_v4_mega_moe_max_num_tokens, 0)
        self.assertEqual(ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb, 512)
        self.assertEqual(ServerArgs.deepseek_v4_prefill_chunk_size, 4)

        # CLI flags parse
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(
            [
                "--model=stub",
                "--disable-deepseek-v4-fast-mhc",
                "--deepseek-v4-mega-moe-max-num-tokens=128",
                "--deepseek-v4-indexer-prefill-max-logits-mb=256",
                "--deepseek-v4-prefill-chunk-size=8",
            ]
        )
        args = ServerArgs.from_cli_args(ns)
        self.assertTrue(args.disable_deepseek_v4_fast_mhc)
        self.assertEqual(args.deepseek_v4_mega_moe_max_num_tokens, 128)
        self.assertEqual(args.deepseek_v4_indexer_prefill_max_logits_mb, 256)
        self.assertEqual(args.deepseek_v4_prefill_chunk_size, 8)

        # Propagation into global_server_args_dict
        snapshot = dict(global_server_args_dict)
        try:
            global_server_args_dict_update(args)
            self.assertTrue(global_server_args_dict["disable_deepseek_v4_fast_mhc"])
            self.assertEqual(
                global_server_args_dict["deepseek_v4_mega_moe_max_num_tokens"], 128
            )
            self.assertEqual(
                global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"],
                256,
            )
            self.assertEqual(
                global_server_args_dict["deepseek_v4_prefill_chunk_size"], 8
            )
        finally:
            global_server_args_dict.clear()
            global_server_args_dict.update(snapshot)

    def test_deepseek_v4_indexer_prefill_max_logits_uses_server_arg(self):
        snapshot = dict(global_server_args_dict)
        try:
            global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"] = 7

            self.assertEqual(
                _deepseek_v4_indexer_prefill_max_logits_bytes(),
                7 * 1024 * 1024,
            )
        finally:
            global_server_args_dict.clear()
            global_server_args_dict.update(snapshot)

    def test_fp8_quantization_config(self):
        quantization = QUANTIZATION_METHODS["fp8"]

        config = quantization.from_config(self.quant_config)

        self.assertEqual(quantization.get_name(), "fp8")
        self.assertIsNone(
            quantization.override_quantization_method(self.quant_config, None)
        )
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertTrue(config.is_checkpoint_fp8_serialized)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fp8_linear_deep_gemm_pads_partial_n_block(self):
        deep_gemm_module = _deepseek_v4_get_fp8_linear_deep_gemm()
        if deep_gemm_module is None:
            self.skipTest("DeepGEMM FP8 linear is unavailable")

        torch.manual_seed(0)
        device = torch.device("cuda")
        rows, out_dim, in_dim = 8, 192, 256
        block_n, block_k = 128, 128
        weight = torch.randn(out_dim, in_dim, device=device) * 0.05
        scale_rows = []
        quantized_rows = []
        for n_start in range(0, out_dim, block_n):
            scale_cols = []
            quantized_cols = []
            for k_start in range(0, in_dim, block_k):
                block = weight[n_start : n_start + block_n, k_start : k_start + block_k]
                scale = torch.pow(
                    torch.tensor(2.0, device=device),
                    torch.ceil(torch.log2(block.abs().amax().clamp_min(1e-4) / 448.0)),
                )
                scale_cols.append(scale)
                quantized_cols.append(
                    (block / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                )
            scale_rows.append(torch.stack(scale_cols))
            quantized_rows.append(torch.cat(quantized_cols, dim=1))

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.cat(quantized_rows, dim=0), requires_grad=False
        )
        layer.weight_scale_inv = torch.stack(scale_rows).contiguous()
        layer.quant_config = SimpleNamespace(weight_block_size=(block_n, block_k))

        x = torch.randn(rows, in_dim, device=device, dtype=torch.bfloat16)
        actual = DeepseekV4Attention._deep_gemm_fp8_linear(
            object(),
            deep_gemm_module,
            layer,
            x,
            (out_dim, in_dim),
        )
        expected = _fp8_linear(layer, x, (out_dim, in_dim))

        torch.cuda.synchronize()
        self.assertEqual(tuple(actual.shape), (rows, out_dim))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(float((actual - expected).abs().max()), 0.25)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fp8_linear_deep_gemm_runtime_failure_falls_back(self):
        class FailingDeepGemm:
            @staticmethod
            def transform_sf_into_required_layout(**kwargs):
                del kwargs
                raise RuntimeError("ptxas failed")

        device = torch.device("cuda")
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.zeros(128, 128, device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.ones(1, 1, device=device)
        layer.quant_config = SimpleNamespace(weight_block_size=(128, 128))
        x = torch.zeros(1, 128, device=device, dtype=torch.bfloat16)

        actual = DeepseekV4Attention._deep_gemm_fp8_linear(
            object(),
            FailingDeepGemm(),
            layer,
            x,
            (128, 128),
        )

        self.assertIsNone(actual)
        self.assertTrue(layer._deepseek_v4_deep_gemm_linear_disabled)
        self.assertIsNone(layer._deepseek_v4_deep_gemm_linear_cache)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_qkv_rmsnorm_matches_separate(self):
        torch.manual_seed(0)
        q = torch.randn(8, 1536, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        q_norm = RMSNorm(1536, eps=1e-6).cuda().to(torch.bfloat16)
        kv_norm = RMSNorm(512, eps=1e-6).cuda().to(torch.bfloat16)
        fused_norm = FusedRMSNorm(q_norm, kv_norm)

        q_out = torch.empty_like(q)
        kv_out = torch.empty_like(kv)
        try:
            fused_norm(q, kv, output_q_a=q_out, output_kv_a=kv_out)
        except RuntimeError as exc:
            self.skipTest(str(exc))

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(q_out, q_norm(q)))
        self.assertTrue(torch.equal(kv_out, kv_norm(kv)))

    def test_model_config_maps_deepseek_v4_to_standard_fp8(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = None

        model_config._verify_quantization()

        self.assertEqual(model_config.quantization, "fp8")

    def test_model_config_overrides_default_block_size_for_deepseek_v4(self):
        def make_hf_config():
            return SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                model_type="deepseek_v4",
                head_dim=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                rope_scaling=None,
                hidden_size=4096,
                num_attention_heads=8,
                num_key_value_heads=8,
                num_hidden_layers=1,
                vocab_size=32000,
                quantization_config=None,
            )

        def build(block_size):
            server_args = SimpleNamespace(
                mapping=None,
                block_size=block_size,
                load_format="auto",
                ext_yaml=None,
            )
            hf_config = make_hf_config()
            with (
                patch(
                    "tokenspeed.runtime.configs.model_config.get_config",
                    return_value=hf_config,
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_generation_config",
                    return_value=SimpleNamespace(eos_token_id=None),
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_hf_text_config",
                    return_value=hf_config,
                ),
                patch(
                    "tokenspeed.runtime.configs.model_config.get_context_length",
                    return_value=4096,
                ),
                patch.object(ModelConfig, "_verify_quantization"),
            ):
                ModelConfig(
                    "stub",
                    model_override_args="{}",
                    server_args=server_args,
                )
            return server_args

        self.assertEqual(build(64).block_size, 256)
        self.assertEqual(build(128).block_size, 128)

    def test_model_config_keeps_incompatible_user_quantization_error(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4", quantization_config=self.quant_config
        )
        model_config.quantization = "mxfp4"

        with self.assertRaisesRegex(ValueError, "does not match"):
            model_config._verify_quantization()

    def test_deepseek_v4_attention_op_boundary_fails_loudly_when_missing(self):
        if has_fused_qnorm_rope_kv_insert():
            self.skipTest("DeepSeek V4 fused attention op is available in this build")

        q = torch.empty(1, 1, 512)
        kv = torch.empty(1, 512)
        cache = torch.empty(1, 584, dtype=torch.uint8)
        slots = torch.zeros(1, dtype=torch.int32)
        positions = torch.zeros(1, dtype=torch.int32)
        cos_sin = torch.empty(1, 128)

        with self.assertRaisesRegex(
            DeepseekV4AttentionOpUnavailable,
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        ):
            fused_qnorm_rope_kv_insert(
                q, kv, cache, slots, positions, cos_sin, 1e-6, 256
            )

    def test_deepseek_v4_flashmla_wrapper_exposes_required_api(self):
        try:
            from tokenspeed_kernel.ops.attention.flash_mla import (
                flash_mla_sparse_fwd,
                flash_mla_with_kvcache,
                get_mla_metadata,
            )
            from tokenspeed_kernel.registry import error_fn
        except Exception as exc:
            self.skipTest(f"FlashMLA wrapper unavailable: {exc}")
        if (
            flash_mla_with_kvcache is error_fn
            or flash_mla_sparse_fwd is error_fn
            or get_mla_metadata is error_fn
        ):
            self.skipTest("FlashMLA wrapper unavailable on this platform")

        self.assertTrue(callable(flash_mla_with_kvcache))
        self.assertTrue(callable(flash_mla_sparse_fwd))
        self.assertTrue(callable(get_mla_metadata))

    def test_deepseek_v4_model_config_uses_mla_runtime_metadata(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            rope_scaling=None,
        )

        self.assertTrue(is_deepseek_v4(model_config.hf_config))

        configure_deepseek_v4_attention(model_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 512)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.qk_nope_head_dim, 448)
        self.assertEqual(model_config.v_head_dim, 512)
        self.assertEqual(model_config.index_head_dim, 128)
        self.assertAlmostEqual(model_config.scaling, 512**-0.5)

    def test_deepseek_v4_attention_layout_matches_compressed_cache_contract(self):
        config = SimpleNamespace(
            compress_ratios=[0, 4, 128],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        swa = deepseek_v4_attention_layout(config, 0, attn_tp_size=4)
        csa = deepseek_v4_attention_layout(config, 1, attn_tp_size=4)
        csa_fp4 = deepseek_v4_attention_layout(
            config, 1, attn_tp_size=4, use_fp4_indexer_cache=True
        )
        hca = deepseek_v4_attention_layout(config, 2, attn_tp_size=4)

        self.assertEqual(swa.kind, "swa")
        self.assertEqual(swa.compress_ratio, 1)
        self.assertEqual(swa.num_local_heads, 16)
        self.assertEqual(swa.padded_heads, 64)
        self.assertEqual(swa.nope_head_dim, 448)
        self.assertEqual(swa.swa_head_bytes, 584)
        self.assertFalse(swa.needs_compressed_cache)
        self.assertFalse(swa.needs_indexer)

        self.assertEqual(csa.kind, "csa")
        self.assertEqual(csa.compress_ratio, 4)
        self.assertTrue(csa.needs_compressed_cache)
        self.assertTrue(csa.needs_indexer)
        self.assertEqual(csa.compressed_cache_alignment, 576)
        self.assertEqual(csa.indexer_cache_head_bytes, 132)
        self.assertEqual(csa_fp4.indexer_cache_head_bytes, 68)

        self.assertEqual(hca.kind, "hca")
        self.assertEqual(hca.compress_ratio, 128)
        self.assertTrue(hca.needs_compressed_cache)
        self.assertFalse(hca.needs_indexer)

    def test_deepseek_v4_attention_layout_rejects_unknown_ratio(self):
        config = SimpleNamespace(
            compress_ratios=[8],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )

        with self.assertRaisesRegex(ValueError, "compress_ratio=8"):
            deepseek_v4_attention_layout(config, 0)

    def test_deepseek_v4_rope_config_matches_layer_type(self):
        config = SimpleNamespace(
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )

        swa_base, swa_scaling = deepseek_v4_rope_config(config, compress_ratio=1)
        csa_base, csa_scaling = deepseek_v4_rope_config(config, compress_ratio=4)

        self.assertEqual(swa_base, 10000.0)
        self.assertIsNone(swa_scaling)
        self.assertEqual(csa_base, 160000.0)
        self.assertIsNot(csa_scaling, config.rope_scaling)
        self.assertEqual(csa_scaling["rope_type"], "deepseek_yarn")
        self.assertEqual(csa_scaling["factor"], 16)
        self.assertEqual(csa_scaling["mscale"], 0)
        self.assertEqual(csa_scaling["mscale_all_dim"], 0)

    def test_deepseek_v4_kv_pool_allocates_v4_cache_families(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        self.assertEqual(layout.cache_cell_size(3), 16771)

        pool = DeepseekV4TokenToKVPool(
            size=128,
            model_dtype=torch.bfloat16,
            layout=layout,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=128,
            page_size=64,
            rank=0,
            hf_config=config,
            max_scheduled_tokens=1,
        )

        self.assertEqual(tuple(pool.get_swa_kv_buffer(0).shape), (7, 37440))
        self.assertIsNone(pool.compressed_kv_buffer[0])
        self.assertEqual(tuple(pool.get_compressed_kv_buffer_2d(1).shape), (4, 37440))
        self.assertEqual(tuple(pool.get_compressor_state_buffer(1).shape), (7, 4, 2048))
        self.assertEqual(
            tuple(pool.get_compressor_state_buffer(2).shape), (35, 8, 1024)
        )
        self.assertEqual(pool.get_compressor_state_buffer(1).dtype, torch.float32)
        self.assertEqual(pool.get_compressor_state_buffer(2).dtype, torch.float32)
        self.assertEqual(tuple(pool.get_indexer_kv_buffer_2d(1).shape), (4, 64 * 68))
        self.assertEqual(tuple(pool.get_indexer_state_buffer(1).shape), (7, 4, 512))
        self.assertEqual(pool.get_indexer_state_buffer(1).dtype, torch.float32)

    def test_deepseek_v4_kv_pool_uses_compressed_storage_blocks_for_page256(self):
        config = SimpleNamespace(
            compress_ratios=[1, 4, 128],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=256,
            use_fp4_indexer_cache=True,
        )
        pool = DeepseekV4TokenToKVPool(
            size=512,
            model_dtype=torch.bfloat16,
            layout=layout,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=512,
            page_size=256,
            rank=0,
            hf_config=config,
            max_scheduled_tokens=1,
        )

        self.assertEqual(pool.swa_block_size, 64)
        self.assertEqual(pool.get_compressed_block_size(1), 64)
        self.assertEqual(pool.get_compressed_block_size(2), 2)
        self.assertEqual(tuple(pool.get_compressed_kv_buffer_2d(1).shape), (5, 37440))
        self.assertEqual(tuple(pool.get_indexer_kv_buffer_2d(1).shape), (5, 64 * 68))

    def test_deepseek_v4_kv_pool_rejects_nonpositive_size(self):
        config = SimpleNamespace(
            compress_ratios=[1],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            DeepseekV4TokenToKVPool(
                size=0,
                model_dtype=torch.bfloat16,
                layout=layout,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
                max_batch_size=2,
                max_context_len=128,
                page_size=64,
                rank=0,
                hf_config=config,
                max_scheduled_tokens=1,
            )

    def test_deepseek_v4_group_slot_mapping_consumes_compact_base_offsets(self):
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([128, 129, 192, 64], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            rows_per_page=64,
            base_offsets=torch.tensor([2, 1], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(slots, torch.tensor([640, 641, -1, 1280])))

    def test_deepseek_v4_backend_preserves_compact_paged_cache_contract(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        compact = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        base = torch.tensor([2, 1], dtype=torch.int32)

        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            paged_cache_block_tables={"v4.swa_kv": compact},
            paged_cache_block_table_base_offsets={"v4.swa_kv": base},
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(torch.equal(metadata.swa_block_table, compact))
        self.assertTrue(torch.equal(metadata.swa_base_logical_page, base))

    def test_deepseek_v4_mixed_metadata_keeps_decode_rows_single_token(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )

        backend.init_forward_metadata(
            bs=3,
            num_tokens=10,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([7, 10, 4], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            req_to_page=torch.zeros((3, 1), dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
            num_extends=1,
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.query_lens.tolist(), [7, 1, 1])
        self.assertEqual(metadata.query_lens_cpu.tolist(), [7, 1, 1])
        self.assertEqual(metadata.num_prefill_reqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 7)
        self.assertEqual(metadata.decode_req_count(), 2)
        self.assertEqual(metadata.decode_token_count(), 2)
        self.assertEqual(
            metadata.token_to_req_indices.tolist(),
            [0, 0, 0, 0, 0, 0, 0, 1, 2],
        )

    def test_deepseek_v4_cuda_graph_refresh_keeps_compact_table_columns(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        backend.init_cuda_graph_state(
            2,
            paged_cache_group_specs=(
                SimpleNamespace(
                    group_id="v4.swa_kv",
                    retention="sliding_window",
                    rows_per_page=64,
                    entry_stride_tokens=1,
                    sliding_window_tokens=128,
                ),
            ),
            max_tokens_per_req=1,
        )
        compact = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        refreshed = backend._refresh_cuda_graph_paged_cache_block_tables(
            2,
            {"v4.swa_kv": compact},
            pad_value=-1,
            paged_cache_block_table_base_offsets={
                "v4.swa_kv": torch.tensor([2, 1], dtype=torch.int32)
            },
        )

        table = refreshed["v4.swa_kv"]
        self.assertTrue(torch.equal(table[:, :2], compact))
        self.assertTrue(torch.equal(table[:, 2:], torch.full_like(table[:, 2:], -1)))

    def test_deepseek_v4_metadata_splits_named_cache_groups(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        swa = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        c4_state = torch.tensor([[30], [40]], dtype=torch.int32)
        c128_state = torch.tensor([[50], [60]], dtype=torch.int32)
        indexer_state = torch.tensor([[70], [80]], dtype=torch.int32)
        c4_state_base = torch.tensor([3, 4], dtype=torch.int32)
        c128_state_base = torch.tensor([5, 6], dtype=torch.int32)
        indexer_state_base = torch.tensor([7, 8], dtype=torch.int32)

        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([200, 80], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            paged_cache_block_tables={
                "v4.swa_kv": swa,
                "v4.c4a.compressor_state": c4_state,
                "v4.c128a.compressor_state": c128_state,
                "v4.c4a.indexer_compressor_state": indexer_state,
            },
            paged_cache_block_table_base_offsets={
                "v4.c4a.compressor_state": c4_state_base,
                "v4.c128a.compressor_state": c128_state_base,
                "v4.c4a.indexer_compressor_state": indexer_state_base,
            },
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(torch.equal(metadata.swa_block_table, swa))
        self.assertTrue(
            torch.equal(metadata.compressor_state_block_tables[4], c4_state)
        )
        self.assertTrue(
            torch.equal(metadata.compressor_state_block_tables[128], c128_state)
        )
        self.assertTrue(torch.equal(metadata.indexer_state_block_table, indexer_state))
        self.assertTrue(
            torch.equal(metadata.compressor_state_base_logical_pages[4], c4_state_base)
        )
        self.assertTrue(
            torch.equal(
                metadata.compressor_state_base_logical_pages[128],
                c128_state_base,
            )
        )
        self.assertTrue(
            torch.equal(metadata.indexer_state_base_logical_page, indexer_state_base)
        )

    def test_deepseek_v4_metadata_slice_preserves_compact_base_offsets(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=4096,
            )
        )
        swa = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
        c4_state = torch.tensor([[40], [41], [42]], dtype=torch.int32)
        c128_state = torch.tensor([[50], [51], [52]], dtype=torch.int32)
        indexer_state = torch.tensor([[60], [61], [62]], dtype=torch.int32)
        raw_offsets = {
            "v4.swa_kv": torch.tensor([100, 200, 300], dtype=torch.int32),
            "v4.c4a.compressor_state": torch.tensor([400, 500, 600], dtype=torch.int32),
            "v4.c128a.compressor_state": torch.tensor(
                [700, 800, 900], dtype=torch.int32
            ),
            "v4.c4a.indexer_compressor_state": torch.tensor(
                [1000, 1100, 1200], dtype=torch.int32
            ),
        }
        metadata = DeepseekV4ForwardMetadata(
            page_size=64,
            req_pool_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
            block_table=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
            query_lens=torch.tensor([2, 1, 3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32),
            forward_mode=ForwardMode.EXTEND,
            paged_cache_block_tables={
                "v4.swa_kv": swa,
                "v4.c4a.compressor_state": c4_state,
                "v4.c128a.compressor_state": c128_state,
                "v4.c4a.indexer_compressor_state": indexer_state,
            },
            paged_cache_block_table_base_offsets=raw_offsets,
            swa_block_table=swa,
            swa_base_logical_page=raw_offsets["v4.swa_kv"],
            compressor_state_block_tables={4: c4_state, 128: c128_state},
            compressor_state_base_logical_pages={
                4: raw_offsets["v4.c4a.compressor_state"],
                128: raw_offsets["v4.c128a.compressor_state"],
            },
            indexer_state_block_table=indexer_state,
            indexer_state_base_logical_page=raw_offsets[
                "v4.c4a.indexer_compressor_state"
            ],
        )

        sliced = backend._metadata_slice(
            metadata,
            req_start=1,
            req_end=3,
            token_start=2,
            token_end=6,
            forward_mode=ForwardMode.EXTEND,
        )

        self.assertTrue(
            torch.equal(
                sliced.token_to_req_indices,
                torch.tensor([0, 1, 1, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.query_start_loc,
                torch.tensor([0, 1, 4], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(sliced.swa_block_table, swa[1:3]))
        self.assertTrue(
            torch.equal(sliced.swa_base_logical_page, raw_offsets["v4.swa_kv"][1:3])
        )
        self.assertTrue(
            torch.equal(
                sliced.paged_cache_block_table_base_offsets["v4.swa_kv"],
                raw_offsets["v4.swa_kv"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.compressor_state_base_logical_pages[4],
                raw_offsets["v4.c4a.compressor_state"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.compressor_state_base_logical_pages[128],
                raw_offsets["v4.c128a.compressor_state"][1:3],
            )
        )
        self.assertTrue(
            torch.equal(
                sliced.indexer_state_base_logical_page,
                raw_offsets["v4.c4a.indexer_compressor_state"][1:3],
            )
        )

    def test_deepseek_v4_metadata_maps_compressed_slots(self):
        compressed_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        metadata = DeepseekV4ForwardMetadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[0, 1], [3, 4]], dtype=torch.int32),
            seq_lens=torch.tensor([70, 5], dtype=torch.int32),
            query_lens=torch.tensor([3, 5], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
            token_to_req_indices=torch.tensor(
                [0, 0, 0, 1, 1, 1, 1, 1],
                dtype=torch.int32,
            ),
            paged_cache_block_tables={"v4.c4a.compressed_kv": compressed_table},
        )

        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(metadata.compressed_block_table(4), compressed_table)
        )
        self.assertTrue(
            torch.equal(metadata.compressed_block_table(128), metadata.block_table)
        )
        slots = metadata.compressed_slot_mapping(
            torch.tensor([3, 7, 127], dtype=torch.int64),
            compress_ratio=4,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([640, 641, 671])))

        page256_metadata = DeepseekV4ForwardMetadata(
            page_size=256,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            block_table=torch.tensor([[5, 6]], dtype=torch.int32),
            seq_lens=torch.tensor([300], dtype=torch.int32),
            query_lens=torch.tensor([3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0], dtype=torch.int32),
        )
        slots = page256_metadata.compressed_slot_mapping(
            torch.tensor([255, 256, 511], dtype=torch.int64),
            compress_ratio=4,
            kv_cache_block_size=64,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([383, 384, 447])))

        grouped_metadata = DeepseekV4ForwardMetadata(
            page_size=256,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
            seq_lens=torch.tensor([300, 10], dtype=torch.int32),
            query_lens=torch.tensor([3, 2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
            paged_cache_block_tables={
                "v4.c4a.compressed_kv": torch.tensor(
                    [[20, 21], [30, -1]], dtype=torch.int32
                )
            },
        )
        slots = grouped_metadata.compressed_slot_mapping(
            torch.tensor([255, 256, 511, 2560, 4], dtype=torch.int64),
            compress_ratio=4,
            kv_cache_block_size=64,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([1343, 1344, 1407, -1, 1921])))

    def test_deepseek_v4_group_slot_mapping_from_raw(self):
        block_table = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
        slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([0, 63, 64, 9, 10], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
            block_table=block_table,
            rows_per_page=64,
            entry_stride_tokens=1,
        )
        self.assertTrue(torch.equal(slots, torch.tensor([640, 703, 704, 1289, 1290])))

        compressed_slots = _group_slot_mapping_from_raw(
            positions=torch.tensor([0, 255, 256, 511], dtype=torch.int64),
            req_indices=torch.tensor([0, 0, 0, 1], dtype=torch.int32),
            block_table=block_table,
            rows_per_page=64,
            entry_stride_tokens=4,
        )
        self.assertTrue(
            torch.equal(compressed_slots, torch.tensor([640, 703, 704, -1]))
        )

    def test_deepseek_v4_mixed_metadata_splits_prefill_and_decode(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=576,
                context_len=256,
            )
        )
        backend.init_forward_metadata(
            bs=3,
            num_tokens=5,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            req_to_page=torch.tensor([[10], [20], [30]], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([3, 1, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([2, 8, 11], dtype=torch.int32),
            num_extends=1,
        )
        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.num_prefill_reqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 3)
        self.assertEqual(metadata.decode_req_count(), 2)
        self.assertEqual(metadata.decode_token_count(), 2)
        self.assertTrue(
            torch.equal(
                metadata.token_to_req_indices,
                torch.tensor([0, 0, 0, 1, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.seq_lens_cpu,
                torch.tensor([5, 9, 12], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.query_lens_cpu,
                torch.tensor([3, 1, 1], dtype=torch.int32),
            )
        )

        prefill = backend._metadata_slice(
            metadata,
            req_start=0,
            req_end=1,
            token_start=0,
            token_end=3,
            forward_mode=ForwardMode.EXTEND,
        )
        decode = backend._metadata_slice(
            metadata,
            req_start=1,
            req_end=3,
            token_start=3,
            token_end=5,
            forward_mode=ForwardMode.DECODE,
        )

        self.assertTrue(prefill.forward_mode.is_extend())
        self.assertTrue(decode.forward_mode.is_decode())
        self.assertTrue(
            torch.equal(prefill.token_to_req_indices, torch.tensor([0, 0, 0]))
        )
        self.assertTrue(torch.equal(decode.token_to_req_indices, torch.tensor([0, 1])))
        self.assertTrue(
            torch.equal(
                decode.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
            )
        )
        self.assertTrue(torch.equal(decode.block_table[:, 0], torch.tensor([20, 30])))
        self.assertTrue(
            torch.equal(prefill.seq_lens_cpu, torch.tensor([5], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(decode.query_lens_cpu, torch.tensor([1, 1], dtype=torch.int32))
        )

    def test_deepseek_v4_mixed_metadata_accepts_prefill_prefix_lens_only(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=576,
                context_len=256,
            )
        )
        backend.init_forward_metadata(
            bs=4,
            num_tokens=8,
            req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12, 6], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            req_to_page=torch.tensor([[10], [20], [30], [40]], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([3, 4, 1, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([2, 5, 11], dtype=torch.int32),
            num_extends=3,
        )

        metadata = backend.forward_metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.num_prefill_reqs, 3)
        self.assertEqual(metadata.num_prefill_tokens, 8)
        self.assertEqual(metadata.decode_req_count(), 1)
        self.assertEqual(metadata.decode_token_count(), 1)
        self.assertTrue(
            torch.equal(
                metadata.seq_lens_cpu,
                torch.tensor([5, 9, 12, 6], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.query_lens_cpu,
                torch.tensor([3, 4, 1, 1], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_mixed_backend_slices_prefill_and_decode(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=8,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=576,
                context_len=256,
            )
        )
        backend.init_forward_metadata(
            bs=3,
            num_tokens=5,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9, 12], dtype=torch.int32),
            forward_mode=ForwardMode.MIXED,
            req_to_page=torch.tensor([[10], [20], [30]], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([3, 1, 1], dtype=torch.int32),
            num_extends=1,
        )
        calls = []

        def fake_prefill(**kwargs):
            metadata = backend.forward_metadata
            calls.append(
                (
                    "prefill",
                    kwargs["q"].shape[0],
                    kwargs["positions"].tolist(),
                    kwargs["topk_indices"].tolist(),
                    metadata.req_pool_indices.tolist(),
                    metadata.token_to_req_indices.tolist(),
                    metadata.forward_mode,
                )
            )
            return kwargs["q"].new_full((3, 2, 4), 1.0)

        def fake_decode(**kwargs):
            metadata = backend.forward_metadata
            calls.append(
                (
                    "decode",
                    kwargs["q"].shape[0],
                    kwargs["positions"].tolist(),
                    kwargs["topk_indices"].tolist(),
                    metadata.req_pool_indices.tolist(),
                    metadata.token_to_req_indices.tolist(),
                    metadata.forward_mode,
                )
            )
            return kwargs["q"].new_full((2, 2, 4), 2.0)

        backend.forward_deepseek_v4_prefill = fake_prefill
        backend.forward_deepseek_v4_decode = fake_decode
        q = torch.zeros((5, 2, 4), dtype=torch.float32)
        topk = torch.arange(10, dtype=torch.int32).view(5, 2)
        out = backend.forward_deepseek_v4_mixed(
            q=q,
            positions=torch.arange(5, dtype=torch.int32),
            token_to_kv_pool=SimpleNamespace(),
            layer_id=0,
            kind="mla",
            compress_ratio=4,
            num_local_heads=2,
            padded_heads=2,
            head_dim=4,
            window_size=4,
            softmax_scale=1.0,
            attn_sink=torch.zeros(2),
            topk_indices=topk,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "prefill")
        self.assertEqual(calls[0][1], 3)
        self.assertEqual(calls[0][2], [0, 1, 2])
        self.assertEqual(calls[0][3], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(calls[0][4], [0])
        self.assertEqual(calls[0][5], [0, 0, 0])
        self.assertTrue(calls[0][6].is_extend())
        self.assertEqual(calls[1][0], "decode")
        self.assertEqual(calls[1][1], 2)
        self.assertEqual(calls[1][2], [3, 4])
        self.assertEqual(calls[1][3], [[6, 7], [8, 9]])
        self.assertEqual(calls[1][4], [1, 2])
        self.assertEqual(calls[1][5], [0, 1])
        self.assertTrue(calls[1][6].is_decode())
        self.assertTrue(torch.equal(out[:3], torch.ones((3, 2, 4))))
        self.assertTrue(torch.equal(out[3:], torch.full((2, 2, 4), 2.0)))

    def test_deepseek_v4_decode_backend_maps_compressed_slots_batched(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=128,
            )
        )
        seq_lens = torch.tensor([70, 3], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        )
        positions = seq_lens.to(torch.int64) - 1

        topk_indices = torch.tensor(
            [[1, 65, 3, -1], [0, -1, -1, -1]],
            dtype=torch.int32,
        )
        indices, lens = backend._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=4,
            block_size=64,
            topk_indices=topk_indices,
        )
        self.assertTrue(torch.equal(lens, torch.tensor([3, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                indices[:, 0, :4],
                torch.tensor(
                    [[641, 705, 643, -1], [1280, -1, -1, -1]],
                    dtype=torch.int32,
                ),
            )
        )

        seq_lens = torch.tensor([256, 129], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]],
                dtype=torch.int32,
            ),
        )
        hca_positions = seq_lens.to(torch.int64) - 1
        indices, lens = backend._decode_compressed_attention_indices_and_lens(
            hca_positions,
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )
        self.assertTrue(torch.equal(lens, torch.tensor([2, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                indices[:, 0, :2],
                torch.tensor([[640, 641], [1280, -1]], dtype=torch.int32),
            )
        )
        cached_indices, cached_lens = (
            backend._decode_compressed_attention_indices_and_lens(
                hca_positions,
                compress_ratio=128,
                block_size=64,
                topk_indices=None,
            )
        )
        self.assertEqual(cached_indices.data_ptr(), indices.data_ptr())
        self.assertEqual(cached_lens.data_ptr(), lens.data_ptr())

    def test_deepseek_v4_decode_backend_capture_ignores_warmup_cache(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for capture cache semantics")
        device = torch.device("cuda")
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cuda",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=128,
            )
        )
        seq_lens = torch.tensor([128, 64], device=device, dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], device=device, dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor(
                [[10, 11], [20, 21]],
                device=device,
                dtype=torch.int32,
            ),
        )
        positions = seq_lens.to(torch.int64) - 1

        warmup_indices, _ = backend._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )
        metadata = backend.forward_metadata
        key = next(iter(metadata.decode_dense_compressed_indices_cache.keys()))
        metadata.decode_dense_compressed_indices_capture_safe_keys.clear()

        original_capturing = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: True
        try:
            capture_indices, _ = backend._decode_compressed_attention_indices_and_lens(
                positions,
                compress_ratio=128,
                block_size=64,
                topk_indices=None,
            )
            reused_indices, _ = backend._decode_compressed_attention_indices_and_lens(
                positions,
                compress_ratio=128,
                block_size=64,
                topk_indices=None,
            )
        finally:
            torch.cuda.is_current_stream_capturing = original_capturing

        self.assertNotEqual(capture_indices.data_ptr(), warmup_indices.data_ptr())
        self.assertEqual(reused_indices.data_ptr(), capture_indices.data_ptr())
        self.assertIn(key, metadata.decode_dense_compressed_indices_capture_safe_keys)

    def test_deepseek_v4_c128a_prefill_local_compressed_indices_contract(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=1024,
            )
        )
        self.assertEqual(backend._dense_compressed_indices_width(128), 128)

        indices = backend._dense_prefill_local_compressed_indices(
            torch.tensor([0, 127, 128, 255], dtype=torch.int64),
            compress_ratio=128,
            width=backend._dense_compressed_indices_width(128),
        )
        self.assertEqual(tuple(indices.shape), (4, 128))
        self.assertTrue(
            torch.equal(indices[0, :2], torch.tensor([-1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(indices[1, :3], torch.tensor([0, -1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(indices[2, :3], torch.tensor([0, -1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(indices[3, :4], torch.tensor([0, 1, -1, -1], dtype=torch.int32))
        )
        cached = backend._dense_prefill_local_compressed_indices(
            torch.tensor([127], dtype=torch.int64),
            compress_ratio=128,
            width=backend._dense_compressed_indices_width(128),
        )
        self.assertEqual(cached.data_ptr(), indices.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_prefill_topk_cuda_op_matches_torch_topk(self):
        try:
            from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
                has_indexer_topk_prefill,
                indexer_topk_prefill,
            )
        except Exception as exc:
            self.skipTest(str(exc))
        if not has_indexer_topk_prefill():
            self.skipTest("DeepSeek V4 prefill top-k op is unavailable")

        torch.manual_seed(0)
        lengths = torch.tensor([0, 3, 17, 33], device="cuda", dtype=torch.int32)
        logits = torch.randn((lengths.numel(), 40), device="cuda", dtype=torch.float32)
        row_starts = torch.zeros_like(lengths)
        out = torch.full((lengths.numel(), 8), -1, device="cuda", dtype=torch.int32)

        indexer_topk_prefill(logits, row_starts, lengths, out, out.shape[-1])
        torch.cuda.synchronize()

        for row, raw_len in enumerate(lengths.cpu().tolist()):
            selected = min(raw_len, out.shape[-1])
            actual = out[row, :selected].sort().values.cpu()
            if selected == 0:
                self.assertTrue(torch.equal(out[row], torch.full_like(out[row], -1)))
                continue
            expected = (
                torch.topk(
                    logits[row, :raw_len],
                    k=selected,
                    dim=-1,
                    sorted=False,
                )
                .indices.sort()
                .values.cpu()
                .to(torch.int32)
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(
                torch.equal(
                    out[row, selected:],
                    torch.full_like(out[row, selected:], -1),
                )
            )

    def test_deepseek_v4_indexer_mxfp4_gather_reuses_workspace(self):
        block_size = 2
        value_bytes = 64
        scale_bytes = 4
        page_bytes = block_size * (value_bytes + scale_bytes)
        cache = (
            torch.arange(3 * page_bytes, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .view(3, page_bytes)
        )
        slots = torch.tensor([0, 2, 5], dtype=torch.int64)

        expected_values, expected_scales = _deepseek_v4_gather_indexer_mxfp4_cache(
            cache,
            slots,
            block_size,
        )
        values_workspace = torch.empty((slots.numel(), value_bytes), dtype=torch.uint8)
        scales_workspace = torch.empty((slots.numel(), scale_bytes), dtype=torch.uint8)
        values, scales = _deepseek_v4_gather_indexer_mxfp4_cache(
            cache,
            slots,
            block_size,
            out=(values_workspace, scales_workspace),
        )

        self.assertEqual(
            values.data_ptr(), values_workspace.view(torch.int8).data_ptr()
        )
        self.assertEqual(
            scales.data_ptr(),
            scales_workspace.view(torch.int32).data_ptr(),
        )
        self.assertTrue(torch.equal(values, expected_values))
        self.assertTrue(torch.equal(scales, expected_scales))

    def test_deepseek_v4_decode_backend_masks_padding_tokens(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=128,
            )
        )
        seq_lens = torch.tensor([70, 3], dtype=torch.int32)
        backend.init_forward_metadata(
            bs=2,
            num_tokens=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        )
        metadata = backend.forward_metadata
        metadata.is_valid_token = torch.tensor([True, False])
        positions = seq_lens.to(torch.int64) - 1

        topk_indices = torch.tensor(
            [[1, 65, 3, -1], [0, -1, -1, -1]],
            dtype=torch.int32,
        )
        _, csa_lens = backend._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=4,
            block_size=64,
            topk_indices=topk_indices,
        )
        _, hca_lens = backend._decode_compressed_attention_indices_and_lens(
            torch.tensor([255, 128], dtype=torch.int64),
            compress_ratio=128,
            block_size=64,
            topk_indices=None,
        )

        self.assertTrue(torch.equal(csa_lens, torch.tensor([3, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(hca_lens, torch.tensor([2, 0], dtype=torch.int32)))

    def test_deepseek_v4_global_topk_cpu_masks_invalid_req_before_indexing(self):
        indices, lens = deepseek_v4_compute_global_topk_indices_and_lens(
            topk_indices=torch.tensor([[0, 4], [0, 1]], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 99], dtype=torch.int32),
            block_table=torch.tensor([[10]], dtype=torch.int32),
            block_size=4,
            is_valid_token=torch.tensor([True, False]),
        )

        self.assertTrue(
            torch.equal(
                indices,
                torch.tensor([[40, -1], [-1, -1]], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(lens, torch.tensor([1, 0], dtype=torch.int32)))

    def test_deepseek_v4_cuda_graph_replay_marks_padding_tokens_invalid(self):
        backend = DeepseekV4AttentionBackend(
            SimpleNamespace(
                page_size=64,
                device="cpu",
                num_attention_heads=64,
                num_kv_heads=1,
                attn_tp_size=1,
                dtype=torch.bfloat16,
                head_dim=512,
                context_len=128,
            )
        )
        backend.init_cuda_graph_state(max_bs=4)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=4,
            num_tokens=4,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )

        backend.init_forward_metadata_replay_cuda_graph(
            bs=4,
            num_tokens=4,
            actual_bs=2,
            req_pool_indices=torch.arange(4, dtype=torch.int32),
            seq_lens=torch.tensor([70, 3, 1, 1], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            req_to_page=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                    [30, 31],
                    [40, 41],
                ],
                dtype=torch.int32,
            ),
        )

        metadata = backend.forward_metadata
        self.assertTrue(
            torch.equal(
                metadata.is_valid_token,
                torch.tensor([True, True, False, False]),
            )
        )
        self.assertEqual(metadata.decode_token_count(), 4)

    def test_deepseek_v4_indexer_metadata_refresh_masks_padding_tokens(self):
        key = (4, 4, 3)
        metadata = DeepseekV4ForwardMetadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32),
            seq_lens=torch.tensor([9, 5, 3], dtype=torch.int32),
            query_lens=torch.ones(3, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            is_valid_token=torch.tensor([True, False, True]),
            forward_mode=ForwardMode.DECODE,
        )
        plan = SimpleNamespace(
            context_lens=torch.empty((3, 1), dtype=torch.int32),
            block_table=torch.empty((3, 2), dtype=torch.int32),
            max_context_len=0,
        )
        metadata.decode_indexer_plan_cache[key] = plan

        def fake_compute(**kwargs):
            kwargs["out_context_lens"].copy_(
                torch.tensor([[2], [2], [1]], dtype=torch.int32)
            )
            kwargs["out_block_tables"].copy_(
                torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
            )

        with patch.object(
            deepseek_v4_backend,
            "deepseek_v4_indexer_decode_metadata_compute",
            side_effect=fake_compute,
        ):
            deepseek_v4_backend._refresh_decode_indexer_plan_cache(
                metadata,
                max_context_len=256,
            )

        self.assertTrue(
            torch.equal(
                plan.context_lens,
                torch.tensor([[2], [0], [1]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.block_table,
                torch.tensor([[10, 11], [0, 0], [30, 31]], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_decode_metadata_accepts_sliced_valid_mask(self):
        metadata = SimpleNamespace(
            decode_indexer_plan_cache={},
            decode_indexer_plan_refreshed_keys=set(),
        )

        def fake_compute(**kwargs):
            kwargs["out_context_lens"].copy_(
                torch.tensor([[2], [2]], dtype=torch.int32)
            )
            kwargs["out_block_tables"].copy_(
                torch.tensor([[10], [20]], dtype=torch.int32)
            )

        with patch.object(
            deepseek_v4_model,
            "deepseek_v4_indexer_decode_metadata_compute",
            side_effect=fake_compute,
        ):
            plan = deepseek_v4_model._deepseek_v4_indexer_decode_metadata(
                positions=torch.tensor([8, 4], dtype=torch.int64),
                token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
                block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
                cache_block_size=4,
                compress_ratio=4,
                metadata=metadata,
                is_valid_token=torch.tensor([False, True]),
            )

        self.assertTrue(
            torch.equal(
                plan.context_lens,
                torch.tensor([[0], [2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.block_table,
                torch.tensor([[0], [20]], dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_schedule_refresh_uses_decode_plan_lens(self):
        captured = {}

        def fake_get_metadata(context_lens, cache_block_size, num_sms):
            captured["context_lens"] = context_lens.clone()
            captured["cache_block_size"] = cache_block_size
            captured["num_sms"] = num_sms
            return torch.full((2, 1), 9, dtype=torch.int32)

        fake_deep_gemm = SimpleNamespace(
            get_paged_mqa_logits_metadata=fake_get_metadata,
            get_num_sms=lambda: 123,
        )
        key = (4, 4, 2)
        metadata = DeepseekV4ForwardMetadata(
            page_size=64,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[0], [0]], dtype=torch.int32),
            seq_lens=torch.tensor([5, 1], dtype=torch.int32),
            query_lens=torch.tensor([1, 1], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
            is_valid_token=torch.tensor([True, False]),
            forward_mode=ForwardMode.DECODE,
        )
        metadata.decode_indexer_plan_cache[key] = SimpleNamespace(
            context_lens=torch.zeros((2, 1), dtype=torch.int32),
        )
        metadata.decode_indexer_schedule_metadata[key] = torch.zeros(
            (2, 1),
            dtype=torch.int32,
        )

        with patch(
            "tokenspeed_kernel.thirdparty.deep_gemm",
            fake_deep_gemm,
            create=True,
        ):
            deepseek_v4_backend._refresh_decode_indexer_schedule_metadata(metadata)

        self.assertTrue(
            torch.equal(
                captured["context_lens"], torch.zeros((2, 1), dtype=torch.int32)
            )
        )
        self.assertEqual(captured["cache_block_size"], 4)
        self.assertEqual(captured["num_sms"], 123)
        self.assertTrue(
            torch.equal(
                metadata.decode_indexer_schedule_metadata[key],
                torch.full((2, 1), 9, dtype=torch.int32),
            )
        )

    def test_deepseek_v4_indexer_decode_batches_cache_reads(self):
        torch.manual_seed(0)
        positions = torch.tensor([15, 7, 3], dtype=torch.int64)
        token_to_req_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        block_table = torch.tensor([[0], [1], [2]], dtype=torch.int32)
        cache = torch.randn(12, 128, dtype=torch.float32)
        index_q = torch.randn(3, 2, 128, dtype=torch.float32)
        weights = torch.randn(3, 2, dtype=torch.float32)

        def cache_reader(cache_2d, slot_mapping, block_size):
            del block_size
            return cache_2d[slot_mapping.long()]

        actual = _deepseek_v4_indexer_topk_from_cache_batched(
            cache_reader=cache_reader,
            cache_2d=cache,
            positions=positions,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            cache_block_size=4,
            index_q=index_q,
            weights=weights,
            compress_ratio=4,
            topk_tokens=3,
        )

        expected = torch.full((3, 3), -1, dtype=torch.int32)
        for token_idx, position in enumerate(positions.tolist()):
            num_compressed = (position + 1) // 4
            local = torch.arange(num_compressed, dtype=torch.int64)
            req_idx = int(token_to_req_indices[token_idx].item())
            pages = torch.div(local, 4, rounding_mode="floor")
            offsets = local % 4
            page_ids = block_table[req_idx, pages.long()].to(torch.int64)
            slots = page_ids * 4 + offsets
            selected = min(num_compressed, expected.shape[1])
            expected[token_idx, :selected] = deepseek_v4_indexer_topk_reference(
                index_q[token_idx : token_idx + 1],
                cache_reader(cache, slots, 4),
                weights[token_idx : token_idx + 1],
                top_k=selected,
            )[0]

        self.assertTrue(torch.equal(actual, expected))

    def test_deepseek_v4_indexer_decode_max_len_uses_context_or_cache_window(self):
        block_table = torch.zeros((2, 257), dtype=torch.int32)

        with patch.dict(global_server_args_dict, {"max_model_len": 4096}):
            self.assertEqual(
                _deepseek_v4_indexer_decode_max_len(
                    block_table,
                    cache_block_size=64,
                    compress_ratio=4,
                ),
                1024,
            )

        with patch.dict(global_server_args_dict, {"max_model_len": None}):
            self.assertEqual(
                _deepseek_v4_indexer_decode_max_len(
                    block_table,
                    cache_block_size=64,
                    compress_ratio=4,
                ),
                4112,
            )

    def test_deepseek_v4_indexer_topk_reuses_output_buffer(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf")],
                [4.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 4], dtype=torch.int32)
        out = torch.empty((2, 2), dtype=torch.int32)

        actual = _deepseek_v4_indexer_topk_from_logits(
            logits,
            lengths,
            topk_tokens=2,
            out=out,
        )

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(torch.equal(actual[0].sort().values, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(actual[1].sort().values, torch.tensor([0, 3])))

    def test_deepseek_v4_indexer_topk_accepts_decode_lens_shape(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf")],
                [4.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([[3], [4]], dtype=torch.int32)

        actual = _deepseek_v4_indexer_topk_from_logits(
            logits,
            lengths,
            topk_tokens=2,
            next_n=1,
        )

        self.assertEqual(actual.shape, (2, 2))
        self.assertTrue(torch.equal(actual[0].sort().values, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(actual[1].sort().values, torch.tensor([0, 3])))

    def test_deepseek_v4_indexer_topk_can_sort_preserved_order(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf")],
                [4.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 4], dtype=torch.int32)

        actual = _deepseek_v4_indexer_topk_from_logits(
            logits,
            lengths,
            topk_tokens=4,
            preserve_topk_order=True,
            sort_preserved_topk=True,
        )

        self.assertTrue(torch.equal(actual[0], torch.tensor([1, 2, 0, -1])))
        self.assertTrue(torch.equal(actual[1], torch.tensor([0, 3, 2, 1])))

    def test_deepseek_v4_indexer_topk_handles_shifted_prefill_rows(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf"), -float("inf"), -float("inf")],
                [-float("inf"), -float("inf"), -float("inf"), 2.0, 8.0, 5.0],
            ],
            dtype=torch.float32,
        )
        row_starts = torch.tensor([0, 3], dtype=torch.int32)
        row_ends = torch.tensor([3, 6], dtype=torch.int32)
        lengths = row_ends - row_starts

        actual = _deepseek_v4_indexer_topk_from_logits(
            logits,
            lengths,
            topk_tokens=3,
            preserve_topk_order=True,
            sort_preserved_topk=True,
            row_starts=row_starts,
            row_ends=row_ends,
        )

        self.assertTrue(torch.equal(actual[0], torch.tensor([1, 2, 0])))
        self.assertTrue(torch.equal(actual[1], torch.tensor([1, 2, 0])))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_indexer_topk_uses_local_prefill_op(self):
        if not _deepseek_v4_prefill_topk_op_available():
            self.skipTest("TRT-LLM indexer_topk_prefill is unavailable")

        logits = torch.tensor(
            [
                [0.0, 3.0, 1.0, -float("inf"), -float("inf"), -float("inf")],
                [-float("inf"), -float("inf"), -float("inf"), 2.0, 8.0, 5.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        row_starts = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        row_ends = torch.tensor([3, 6], device="cuda", dtype=torch.int32)

        actual = _deepseek_v4_indexer_topk_from_logits(
            logits,
            row_ends - row_starts,
            topk_tokens=4,
            preserve_topk_order=True,
            row_starts=row_starts,
            row_ends=row_ends,
        )

        expected = torch.tensor(
            [[0, 1, 2, -1], [0, 1, 2, -1]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_deepseek_v4_topk_buffer_grows_and_reuses(self):
        buffer = _DeepseekV4TopKBuffer(topk_tokens=3)

        first = buffer.get(2, torch.device("cpu"))
        second = buffer.get(1, torch.device("cpu"))
        third = buffer.get(4, torch.device("cpu"))

        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(second.shape, (1, 3))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(third.shape, (4, 3))
        self.assertGreaterEqual(buffer.buffer.shape[0], 4)

    def test_deepseek_v4_sparse_indexer_custom_op_registered(self):
        self.assertTrue(
            hasattr(torch.ops.tokenspeed, "deepseek_v4_sparse_attn_indexer")
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_sparse_indexer_custom_op_fallback_covers_decode_tokens(self):
        device = torch.device("cuda")
        n_head = 2
        head_dim = 4
        total_tokens = 3

        class FakeLinear:
            def __init__(self, out_features):
                self.out_features = out_features

            def __call__(self, x):
                return (
                    torch.zeros(
                        (x.shape[0], self.out_features),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    None,
                )

        self_obj = SimpleNamespace(
            use_fp4_cache=True,
            wq_b=FakeLinear(n_head * head_dim),
            weights_proj=FakeLinear(n_head),
            n_head=n_head,
            head_dim=head_dim,
            softmax_scale=1.0,
            compress_ratio=4,
            topk_tokens=2,
            topk_buffer=None,
            _prefill_gather_workspace=lambda rows, device: (
                torch.empty((0, 0), dtype=torch.uint8, device=device),
                torch.empty((0, 0), dtype=torch.uint8, device=device),
            ),
        )
        metadata = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            num_prefill_tokens=1,
            num_prefill_reqs=1,
            seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
            query_lens_cpu=torch.tensor([1], dtype=torch.int32),
            token_to_req_indices=torch.tensor(
                [0, 0, 0], dtype=torch.int32, device=device
            ),
            compressed_block_table=lambda compress_ratio, block_size: torch.zeros(
                (1, 1),
                dtype=torch.int32,
                device=device,
            ),
            decode_token_count=lambda: 2,
        )
        captured = {}

        def fake_prepare_mxfp4(**kwargs):
            index_q = kwargs["index_q"]
            rows = index_q.shape[0]
            return (
                (
                    torch.empty(
                        (rows, n_head, head_dim // 2), dtype=torch.uint8, device=device
                    ),
                    torch.empty((rows, n_head, 1), dtype=torch.uint8, device=device),
                ),
                torch.empty((rows, n_head), dtype=torch.float32, device=device),
            )

        def fake_prepare_reference(**kwargs):
            captured["reference_rows"] = kwargs["positions"].numel()
            rows = kwargs["positions"].numel()
            return (
                torch.empty(
                    (rows, n_head, head_dim), dtype=torch.float32, device=device
                ),
                torch.empty((rows, n_head), dtype=torch.float32, device=device),
            )

        def fake_sparse_indexer(**kwargs):
            captured["fallback_rows"] = kwargs["fallback_index_q"].shape[0]
            captured["has_packed_q"] = kwargs["has_packed_q"]
            captured["num_prefill_tokens"] = kwargs["num_prefill_tokens"]
            captured["num_decode_tokens"] = kwargs["num_decode_tokens"]
            return torch.full(
                (total_tokens, self_obj.topk_tokens),
                7,
                dtype=torch.int32,
                device=device,
            )

        empty_prefill_metadata = SimpleNamespace(
            chunk_bounds=torch.empty((0, 7), dtype=torch.int64, device="cpu"),
            chunk_plan=torch.empty((0, 7), dtype=torch.int64, device="cpu"),
            slots=torch.empty(0, dtype=torch.int64, device=device),
            cu_seq_lens=torch.empty(0, dtype=torch.int32, device=device),
            cu_start=torch.empty(0, dtype=torch.int32, device=device),
            cu_end=torch.empty(0, dtype=torch.int32, device=device),
            row_lens=torch.empty(0, dtype=torch.int32, device=device),
        )
        decode_metadata = SimpleNamespace(
            context_lens=torch.ones((2, 1), dtype=torch.int32, device=device),
            block_table=torch.zeros((2, 1), dtype=torch.int32, device=device),
            max_context_len=1,
        )

        with patch.object(
            deepseek_v4_model,
            "deepseek_v4_prepare_indexer_q_mxfp4",
            side_effect=fake_prepare_mxfp4,
        ), patch.object(
            deepseek_v4_model,
            "_deepseek_v4_deepgemm_fp4_indexer_available",
            return_value=False,
        ), patch.object(
            deepseek_v4_model,
            "deepseek_v4_prepare_indexer_q_reference",
            side_effect=fake_prepare_reference,
        ), patch.object(
            deepseek_v4_model,
            "_deepseek_v4_indexer_prefill_metadata",
            return_value=empty_prefill_metadata,
        ), patch.object(
            deepseek_v4_model,
            "_deepseek_v4_indexer_decode_metadata",
            return_value=decode_metadata,
        ), patch.object(
            deepseek_v4_model,
            "_deepseek_v4_indexer_decode_schedule_metadata",
            return_value=None,
        ), patch.object(
            deepseek_v4_model,
            "_deepseek_v4_sparse_attn_indexer",
            side_effect=fake_sparse_indexer,
        ):
            actual = DeepseekV4Indexer._forward_sparse_indexer_custom_op(
                self_obj,
                hidden_states=torch.zeros((total_tokens, 8), device=device),
                qr=torch.zeros((total_tokens, 8), device=device),
                positions=torch.arange(total_tokens, dtype=torch.int64, device=device),
                metadata=metadata,
                indexer_cache=torch.empty((1, 1), dtype=torch.uint8, device=device),
                indexer_block_size=1,
                cos_sin_cache=torch.empty((1, 1), device=device),
            )

        self.assertEqual(tuple(actual.shape), (total_tokens, self_obj.topk_tokens))
        self.assertEqual(captured["reference_rows"], total_tokens)
        self.assertEqual(captured["fallback_rows"], total_tokens)
        self.assertFalse(captured["has_packed_q"])
        self.assertEqual(captured["num_prefill_tokens"], 1)
        self.assertEqual(captured["num_decode_tokens"], 2)

    def test_deepseek_v4_indexer_prefill_topk_chunks_cap_logits_bytes(self):
        positions = torch.tensor([3, 7, 11, 15], dtype=torch.int64)

        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                positions,
                compress_ratio=4,
                max_logits_bytes=32,
            ),
            [(0, 2), (2, 4)],
        )
        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                positions,
                compress_ratio=4,
                max_logits_bytes=64,
            ),
            [(0, 4)],
        )
        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                torch.tensor([39], dtype=torch.int64),
                compress_ratio=4,
                max_logits_bytes=16,
            ),
            [(0, 1)],
        )

    def test_deepseek_v4_indexer_prefill_topk_chunks_use_cpu_lengths(self):
        positions = torch.zeros(6, dtype=torch.int64)

        self.assertEqual(
            _deepseek_v4_indexer_prefill_topk_chunks(
                positions,
                compress_ratio=4,
                max_logits_bytes=16,
                seq_lens_cpu=torch.tensor([12, 8], dtype=torch.int32),
                query_lens_cpu=torch.tensor([4, 2], dtype=torch.int32),
            ),
            [(0, 2), (2, 3), (3, 4), (4, 6)],
        )

    def test_deepseek_v4_mixed_indexer_fallback_uses_compressed_block_table(self):
        base_block_table = torch.tensor([[1]], dtype=torch.int32)
        indexer_block_table = torch.tensor([[7]], dtype=torch.int32)
        captured = {}

        class FakeLinear:
            def __init__(self, out_features):
                self.out_features = out_features

            def __call__(self, x):
                return (
                    torch.zeros(
                        (x.shape[0], self.out_features),
                        dtype=torch.float32,
                        device=x.device,
                    ),
                    None,
                )

        class FakeCompressor:
            def __init__(self):
                self.norm = SimpleNamespace(
                    weight=torch.ones(1),
                    variance_epsilon=1e-6,
                )

            def __call__(self, **kwargs):
                return None

        pool = SimpleNamespace(
            state_block_size=4,
            get_indexer_state_buffer=lambda layer_id: torch.empty((1, 1)),
            get_indexer_block_size=lambda layer_id: 4,
            get_indexer_kv_buffer_2d=lambda layer_id: torch.empty((8, 128)),
        )
        metadata = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            indexer_state_block_table=None,
            block_table=base_block_table,
            token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32),
            compressed_block_table=(
                lambda compress_ratio, block_size: indexer_block_table
            ),
            compressed_slot_mapping=lambda *args, **kwargs: torch.zeros(
                2, dtype=torch.int64
            ),
            decode_token_count=lambda: 0,
            num_prefill_tokens=2,
            num_prefill_reqs=1,
            seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
            query_lens_cpu=torch.tensor([2], dtype=torch.int32),
        )
        ctx = SimpleNamespace(
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(forward_metadata=metadata),
            forward_mode=ForwardMode.MIXED,
        )
        self_obj = SimpleNamespace(
            use_fp4_cache=False,
            compressor=FakeCompressor(),
            compress_ratio=4,
            n_head=1,
            head_dim=4,
            softmax_scale=1.0,
            topk_tokens=2,
            topk_buffer=None,
            wq_b=FakeLinear(4),
            weights_proj=FakeLinear(1),
            _forward_sparse_indexer_custom_op=lambda **kwargs: None,
        )

        def fake_prepare_reference(**kwargs):
            rows = kwargs["positions"].numel()
            return (
                torch.zeros((rows, 1, 4), dtype=torch.float32),
                torch.zeros((rows, 1), dtype=torch.float32),
            )

        def fake_topk_from_cache(**kwargs):
            captured["block_table"] = kwargs["block_table"]
            rows = kwargs["positions"].numel()
            return torch.full((rows, 2), 3, dtype=torch.int32)

        with patch.object(
            deepseek_v4_model,
            "deepseek_v4_csa_indexer_cache_insert",
            return_value=None,
        ), patch.object(
            deepseek_v4_model,
            "deepseek_v4_prepare_indexer_q_reference",
            side_effect=fake_prepare_reference,
        ), patch.object(
            deepseek_v4_model,
            "_deepseek_v4_indexer_topk_from_cache_batched",
            side_effect=fake_topk_from_cache,
        ):
            topk = DeepseekV4Indexer.forward(
                self_obj,
                hidden_states=torch.zeros((2, 8)),
                qr=torch.zeros((2, 8)),
                positions=torch.tensor([6, 7], dtype=torch.int64),
                ctx=ctx,
                out_cache_loc=torch.zeros(2, dtype=torch.int64),
                layer_index=0,
                cos_sin_cache=torch.empty((1, 1)),
            )

        self.assertTrue(torch.equal(captured["block_table"], indexer_block_table))
        self.assertTrue(torch.equal(topk, torch.full((2, 2), 3, dtype=torch.int32)))

    def test_deepseek_v4_indexer_prefill_gather_plan_reuses_request_k(self):
        slots, cu_start, cu_end, row_lens, max_len = (
            _deepseek_v4_indexer_prefill_gather_plan(
                positions=torch.tensor([0, 1, 5, 0, 3], dtype=torch.int64),
                token_to_req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
                block_table=torch.tensor([[10], [20]], dtype=torch.int32),
                cache_block_size=4,
                compress_ratio=2,
            )
        )

        self.assertTrue(torch.equal(slots, torch.tensor([40, 41, 42, 80, 81])))
        self.assertTrue(torch.equal(cu_start, torch.tensor([0, 0, 0, 3, 3])))
        self.assertTrue(torch.equal(cu_end, torch.tensor([0, 1, 3, 3, 5])))
        self.assertTrue(torch.equal(row_lens, torch.tensor([0, 1, 3, 0, 2])))
        self.assertEqual(max_len, 3)

    def test_deepseek_v4_indexer_prefill_request_chunks_match_reference(self):
        chunks = _deepseek_v4_indexer_prefill_request_chunks(
            seq_lens_cpu=torch.tensor([16], dtype=torch.int32),
            query_lens_cpu=torch.tensor([6], dtype=torch.int32),
            compress_ratio=4,
            num_tokens=6,
            max_logits_bytes=32,
            workspace_size=100,
        )

        self.assertEqual(
            [
                (
                    c.req_start,
                    c.req_end,
                    c.query_start,
                    c.query_end,
                    c.token_start,
                    c.token_end,
                    c.skip_kv_gather,
                )
                for c in chunks
            ],
            [
                (0, 1, 0, 2, 0, 2, False),
                (0, 1, 2, 4, 2, 4, True),
                (0, 1, 4, 6, 4, 6, True),
            ],
        )

        chunks = _deepseek_v4_indexer_prefill_request_chunks(
            seq_lens_cpu=torch.tensor([16, 8], dtype=torch.int32),
            query_lens_cpu=torch.tensor([2, 2], dtype=torch.int32),
            compress_ratio=4,
            num_tokens=4,
            max_logits_bytes=128,
            workspace_size=100,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].req_start, chunks[0].req_end), (0, 2))
        self.assertEqual((chunks[0].token_start, chunks[0].token_end), (0, 4))
        self.assertFalse(chunks[0].skip_kv_gather)

    def test_deepseek_v4_indexer_prefill_request_gather_plan_matches_reference(self):
        slots, cu_start, cu_end, row_lens, max_len = (
            _deepseek_v4_indexer_prefill_request_gather_plan(
                seq_lens_cpu=torch.tensor([16, 8], dtype=torch.int32),
                query_lens_cpu=torch.tensor([4, 2], dtype=torch.int32),
                block_table=torch.tensor([[10], [20]], dtype=torch.int32),
                cache_block_size=4,
                compress_ratio=4,
                req_start=0,
                req_end=2,
                query_start=1,
                query_end=5,
            )
        )

        self.assertTrue(torch.equal(slots, torch.tensor([40, 41, 42, 43, 80, 81])))
        self.assertTrue(torch.equal(cu_start, torch.tensor([0, 0, 0, 4])))
        self.assertTrue(torch.equal(cu_end, torch.tensor([3, 3, 4, 5])))
        self.assertTrue(torch.equal(row_lens, torch.tensor([3, 3, 4, 1])))
        self.assertEqual(max_len, 4)

    def test_deepseek_v4_indexer_prefill_metadata_packs_and_caches_plan(self):
        metadata = SimpleNamespace(
            seq_lens_cpu=torch.tensor([16, 8], dtype=torch.int32),
            query_lens_cpu=torch.tensor([4, 2], dtype=torch.int32),
            num_prefill_reqs=2,
            prefill_indexer_plan_cache={},
        )
        block_table = torch.tensor([[10], [20]], dtype=torch.int32)

        actual = _deepseek_v4_indexer_prefill_metadata(
            metadata=metadata,
            block_table=block_table,
            cache_block_size=4,
            compress_ratio=4,
            num_prefill_tokens=6,
        )
        cached = _deepseek_v4_indexer_prefill_metadata(
            metadata=metadata,
            block_table=block_table,
            cache_block_size=4,
            compress_ratio=4,
            num_prefill_tokens=6,
        )

        self.assertIs(actual, cached)
        self.assertTrue(
            torch.equal(
                actual.chunk_bounds,
                torch.tensor([[0, 6, 0, 2, 0, 6, 0]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                actual.chunk_plan,
                torch.tensor([[0, 6, 0, 6, 4, 0, 3]], dtype=torch.int64),
            )
        )
        self.assertEqual(actual.slots.numel(), 0)
        self.assertTrue(
            torch.equal(actual.cu_seq_lens, torch.tensor([0, 4, 6], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(actual.cu_start, torch.tensor([0, 0, 0, 0, 4, 4])))
        self.assertTrue(torch.equal(actual.cu_end, torch.tensor([3, 3, 3, 4, 5, 6])))
        self.assertTrue(torch.equal(actual.row_lens, torch.tensor([3, 3, 3, 4, 1, 2])))

    def test_hidden_compression_helpers_preserve_expected_shapes(self):
        import torch

        torch.manual_seed(0)
        tokens, hc_mult, hidden = 3, 4, 5
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.float32)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.zeros(mix_hc, dtype=torch.float32)

        layer_input, post, comb = mhc_pre(
            residual,
            fn,
            scale,
            base,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=2,
        )
        updated = mhc_post(layer_input, residual, post, comb)

        self.assertEqual(tuple(layer_input.shape), (tokens, hidden))
        self.assertEqual(tuple(post.shape), (tokens, hc_mult, 1))
        self.assertEqual(tuple(comb.shape), (tokens, hc_mult, hc_mult))
        self.assertEqual(tuple(updated.shape), tuple(residual.shape))

    def test_hidden_compression_pre_matches_reference_math(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(1)
        tokens, hc_mult, hidden = 2, 3, 4
        mix_hc = (2 + hc_mult) * hc_mult
        residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.bfloat16)
        fn = torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32)
        scale = torch.tensor([0.7, 1.1, 0.5], dtype=torch.float32)
        base = torch.randn(mix_hc, dtype=torch.float32)
        eps = 1e-5

        layer_input, post, comb = mhc_pre(
            residual, fn, scale, base, rms_eps=1e-6, hc_eps=eps, sinkhorn_iters=3
        )

        x = residual.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        mixes = F.linear(x, fn) * rsqrt
        pre_raw, post_raw, comb_raw = torch.split(
            mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        pre_base, post_base, comb_base = torch.split(
            base, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
        )
        expected_pre = torch.sigmoid(pre_raw * scale[0] + pre_base) + eps
        expected_post = (
            torch.sigmoid(post_raw * scale[1] + post_base) * 2.0
        ).unsqueeze(-1)
        expected_comb = (
            F.softmax(
                comb_raw.reshape(tokens, hc_mult, hc_mult) * scale[2]
                + comb_base.reshape(1, hc_mult, hc_mult),
                dim=-1,
            )
            + eps
        )
        expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(2):
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-1, keepdim=True) + eps
            )
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-2, keepdim=True) + eps
            )
        expected_layer_input = torch.sum(
            expected_pre.unsqueeze(-1) * residual.float(), dim=1
        ).to(residual.dtype)

        self.assertTrue(torch.allclose(layer_input, expected_layer_input))
        self.assertTrue(torch.allclose(post, expected_post))
        self.assertTrue(torch.allclose(comb, expected_comb))

    def test_hidden_compression_post_matches_lane_orientation(self):
        import torch

        hidden_states = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        post = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
        comb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)

        updated = mhc_post(hidden_states, residual, post, comb)

        expected = torch.empty_like(residual)
        expected[:, 0] = (
            comb[:, 0, 0:1] * residual[:, 0]
            + comb[:, 1, 0:1] * residual[:, 1]
            + post[:, 0] * hidden_states
        )
        expected[:, 1] = (
            comb[:, 0, 1:2] * residual[:, 0]
            + comb[:, 1, 1:2] * residual[:, 1]
            + post[:, 1] * hidden_states
        )
        self.assertTrue(torch.allclose(updated, expected))

    def test_hc_head_matches_shape_contract(self):
        import torch

        tokens, hc_mult, hidden = 2, 4, 6
        x = torch.randn(tokens, hc_mult, hidden)
        fn = torch.randn(hc_mult, hc_mult * hidden)
        scale = torch.ones(1)
        base = torch.zeros(hc_mult)

        y = hc_head(x, fn, scale, base, rms_norm_eps=1e-6, hc_eps=1e-6)

        self.assertEqual(tuple(y.shape), (tokens, hidden))

    def test_deepseek_v4_router_matches_noaux_bias_semantics(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [0.2, 1.0, -0.5, 0.7],
                [1.5, -0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)

        topk_weights, topk_ids, scores = deepseek_v4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=2, dim=-1, sorted=False)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_hash_router_uses_table_ids_and_gate_scores(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [0.5, 1.0, -0.5, 0.1],
                [-0.2, 0.3, 1.4, 0.0],
            ],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([3, 1], dtype=torch.long)
        table = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 0],
                [3, 1],
            ],
            dtype=torch.int32,
        )

        topk_weights, topk_ids, _ = deepseek_v4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_indices_table=table,
            input_ids=input_ids,
        )

        expected_ids = torch.tensor([[3, 1], [2, 3]], dtype=torch.int32)
        expected_scores = F.softplus(logits).sqrt()
        expected_weights = expected_scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights))

    def test_deepseek_v4_gate_fallback_returns_fp32_logits(self):
        import torch
        import torch.nn.functional as F

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            num_hash_layers=0,
            topk_method=None,
        )
        gate = DeepseekV4MoEGate(config, layer_index=1)
        with torch.no_grad():
            gate.weight.copy_(torch.randn_like(gate.weight))
        hidden_states = torch.randn(3, config.hidden_size)

        logits = gate(hidden_states)
        expected = F.linear(hidden_states, gate.weight, None).float()

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.allclose(logits, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_gate_dsv3_router_gemm_shape(self):
        import torch

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("DSV3 router GEMM requires SM90+")

        config = SimpleNamespace(
            n_routed_experts=256,
            hidden_size=4096,
            num_hash_layers=0,
            topk_method=None,
        )
        gate = DeepseekV4MoEGate(config, layer_index=1).cuda().to(torch.bfloat16)
        hidden_states = torch.randn(
            2, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        try:
            logits = gate(hidden_states)
        except RuntimeError as exc:
            if "dsv3_gemm library not found" not in str(exc):
                raise
            self.skipTest(str(exc))
        torch.cuda.synchronize()

        self.assertEqual(tuple(logits.shape), (2, config.n_routed_experts))
        self.assertEqual(logits.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_softplus_sqrt_topk_matches_reference(self):
        import torch
        import torch.nn.functional as F
        from tokenspeed_kernel.thirdparty.cuda.routing import (
            softplus_sqrt_topk_flash,
        )

        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(3, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)
        topk_weights = torch.empty(3, 6, device="cuda", dtype=torch.float32)
        topk_ids = torch.empty(3, 6, device="cuda", dtype=torch.int32)

        try:
            softplus_sqrt_topk_flash(logits, bias, topk_ids, topk_weights, 1.0, True)
        except (AttributeError, RuntimeError) as exc:
            self.skipTest(f"fused DeepSeek V4 router op unavailable: {exc}")
        torch.cuda.synchronize()

        scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_select_experts_returns_scores(self):
        import torch
        import torch.nn.functional as F

        logits = torch.linspace(
            -3.0, 3.0, 256, device="cuda", dtype=torch.float32
        ).repeat(2, 1)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)

        topk_weights, topk_ids, scores = deepseek_v4_select_experts(
            logits,
            top_k=6,
            renormalize=True,
            correction_bias=bias,
        )

        expected_scores = F.softplus(logits).sqrt()
        expected_ids = torch.topk(expected_scores + bias, k=6, dim=-1, sorted=True)[1]
        expected_weights = expected_scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(scores, expected_scores))
        self.assertTrue(torch.equal(topk_ids, expected_ids.to(torch.int32)))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_bias_fused_router_runs_by_default(self):
        import torch

        logits = torch.zeros(2, 256, device="cuda", dtype=torch.float32)
        bias = torch.linspace(0.25, -0.25, 256, device="cuda", dtype=torch.float32)

        out = _deepseek_v4_fused_select_experts(
            logits, top_k=6, renormalize=True, correction_bias=bias
        )

        if out is None:
            self.skipTest("fused DeepSeek V4 router op unavailable")
        topk_weights, topk_ids = out
        self.assertEqual(tuple(topk_weights.shape), (2, 6))
        self.assertEqual(tuple(topk_ids.shape), (2, 6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fused_hash_topk_matches_reference(self):
        import torch
        import torch.nn.functional as F
        from tokenspeed_kernel.thirdparty.cuda.routing import (
            hash_softplus_sqrt_topk_flash,
        )

        logits = torch.linspace(
            -2.0, 2.0, 256, device="cuda", dtype=torch.float32
        ).repeat(3, 1)
        input_ids = torch.tensor([1, 0, 1], device="cuda", dtype=torch.long)
        table = torch.tensor(
            [[5, 7, 11, 13, 17, 19], [23, 29, 31, 37, 41, 43]],
            device="cuda",
            dtype=torch.int32,
        )
        topk_weights = torch.empty(3, 6, device="cuda", dtype=torch.float32)
        topk_ids = torch.empty(3, 6, device="cuda", dtype=torch.int32)

        try:
            hash_softplus_sqrt_topk_flash(
                logits, input_ids, table, topk_ids, topk_weights, 1.0, True
            )
        except (AttributeError, RuntimeError) as exc:
            self.skipTest(f"fused DeepSeek V4 hash router op unavailable: {exc}")
        torch.cuda.synchronize()

        expected_ids = table[input_ids]
        scores = F.softplus(logits).sqrt()
        expected_weights = scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.equal(topk_ids, expected_ids))
        self.assertTrue(torch.allclose(topk_weights, expected_weights, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deepseek_v4_fp8_activation_quant_matches_reference(self):
        import torch

        x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16) * 3.0

        actual = _fp8_act_quant_dequant(x, 128)

        x_blocks = x.float().reshape(-1, x.shape[-1]).unflatten(-1, (-1, 128))
        amax = x_blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
        scale = scale.to(torch.float8_e8m0fnu).float()
        quantized = (
            (x_blocks / scale.unsqueeze(-1))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        expected = (quantized.float() * scale.unsqueeze(-1)).flatten(-2).reshape_as(x)

        self.assertTrue(torch.equal(actual, expected))

    def test_packed_topk_router_logits_recover_weights_after_softmax(self):
        import torch

        topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3], [0.55, 0.45]], dtype=torch.float32)

        packed = pack_topk_as_router_logits(topk_weights, topk_ids, num_experts=4)
        recovered = packed.softmax(dim=-1).gather(1, topk_ids.long())

        self.assertTrue(torch.allclose(recovered, topk_weights))

    def test_mxfp4_flashinfer_reorders_w1w3_halves_for_trtllm(self):
        import torch

        weight = torch.arange(4, dtype=torch.uint8).reshape(1, 4, 1)
        scale = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
        bias = torch.arange(4, dtype=torch.float32).reshape(1, 4)

        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(weight, -2).flatten(),
                torch.tensor([2, 3, 0, 1], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(scale, -2).flatten(),
                torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                _reorder_w1w3_to_w3w1(bias, -1).flatten(),
                torch.tensor([2, 3, 0, 1], dtype=torch.float32),
            )
        )
        if hasattr(torch, "float8_e8m0fnu"):
            scale_f8 = torch.tensor(
                [[0.0078125, 0.015625, 0.03125, 0.0625]], dtype=torch.float32
            ).to(torch.float8_e8m0fnu)
            reordered = _reorder_w1w3_to_w3w1(scale_f8, -1)
            self.assertEqual(reordered.dtype, torch.float8_e8m0fnu)
            self.assertTrue(
                torch.equal(
                    reordered.view(torch.uint8),
                    torch.tensor([[122, 123, 120, 121]], dtype=torch.uint8),
                )
            )

    def test_mxfp4_flashinfer_uses_gated_permute_for_w13(self):
        import torch
        from tokenspeed_kernel.ops.moe.flashinfer import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        x = torch.empty((4096, 2048), dtype=torch.uint8)
        expected_w13 = _maybe_get_cached_w3_w1_permute_indices({}, x, 128)
        expected_w2 = get_w2_permute_indices_with_cache({}, x, 128)

        actual_w13 = _get_flashinfer_mxfp4_device_permute_indices(x, 128, kind="w13")
        actual_w2 = _get_flashinfer_mxfp4_device_permute_indices(x, 128, kind="w2")

        self.assertTrue(torch.equal(actual_w13.cpu(), expected_w13.cpu()))
        self.assertTrue(torch.equal(actual_w2.cpu(), expected_w2.cpu()))
        self.assertFalse(torch.equal(actual_w13.cpu(), actual_w2.cpu()))

    def test_c4_ape_reorder_matches_overlap_window_layout(self):
        import torch

        ape = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

        reordered = _deepseek_v4_reorder_c4_ape_2604(ape)
        expected = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [16, 17, 18, 19, 24, 25, 26, 27],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [20, 21, 22, 23, 28, 29, 30, 31],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(reordered, expected))


if __name__ == "__main__":
    unittest.main()
