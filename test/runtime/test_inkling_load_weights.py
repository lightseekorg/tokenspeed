"""Inkling ``load_weights`` unit test against a synthetic real-name checkpoint.

Builds an in-memory checkpoint using the REAL checkpoint tensor names
(``model.llm.layers.N.attn.wq_du.weight`` etc.), with gate/up-interleaved w13
tensors and asymmetric full/SWA KV heads like the real model, then asserts
every parameter is covered and the values land where the reference
implementation puts them: qkvr fusion order, full-layer KV replication,
and w13 de-interleaving (dense, routed and shared experts).

NOTE: intentionally not registered in CI. This file is the only loader
coverage Inkling gets (the gated checkpoint rules out the
boot-real-weights CI gate other models use) — run it locally after loader
changes. Needs any CUDA GPU (weight copies only, no kernels).
"""

import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.models.inkling_fixtures import (  # noqa: E402
    TINY_MM_CONFIG,
    make_inkling_dummy_checkpoint,
)

SEED = 1234


def _build_model(ckpt: str, quant_config=None, rank=0, world_size=1, **mapping_kw):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.models.inkling import InklingForConditionalGeneration
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config

    # Same precomputed-topk backend the engine resolves for Inkling.
    moe_utils.MOE_BACKEND = moe_utils.MoeBackend.FLASHINFER_CUTLASS
    mapping = Mapping(rank=rank, world_size=world_size, **mapping_kw)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["enable_prefix_caching"] = False
    config = get_config(ckpt, trust_remote_code=False, revision=None)
    with torch.device("cuda"):
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = InklingForConditionalGeneration(
                config, mapping, quant_config=quant_config
            )
        finally:
            torch.set_default_dtype(torch.float32)
    return model.eval(), config.get_text_config()


def _deinterleave_rows(w: torch.Tensor) -> torch.Tensor:
    """Independent gate/up de-interleave: rows [g0,u0,...] -> [gate | up]."""
    return torch.cat([w[..., 0::2, :], w[..., 1::2, :]], dim=-2)


def _make_checkpoint_tensors(
    text, fp4_layers: set[int] = frozenset()
) -> dict[str, torch.Tensor]:
    """Synthesize one tensor per real-checkpoint name for the tiny config.

    Layers in ``fp4_layers`` emit their routed experts in the ModelOpt NVFP4
    format (packed U8 + F8 block scales + scale2 + input_amax) instead of
    BF16, mirroring the real FP4 snapshot's mixed precision.
    """
    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype)

    def rand_bytes(*shape, dtype=torch.uint8):
        return (
            torch.randint(0, 256, shape, generator=gen, dtype=torch.int64)
            .to(torch.uint8)
            .view(dtype)
        )

    h = text.hidden_size
    hd = text.head_dim
    heads = text.num_attention_heads
    local = set(text.local_layer_ids)
    ckpt: dict[str, torch.Tensor] = {
        "model.llm.embed.weight": rand(text.padded_vocab_size, h),
        "model.llm.embed_norm.weight": rand(h),
        "model.llm.norm.weight": rand(h),
        "model.llm.unembed.weight": rand(text.padded_vocab_size, h),
        # MTP chain: must be skipped by the loader.
        "model.mtp.chain_norm.weight": rand(h),
    }
    for i in range(text.num_hidden_layers):
        p = f"model.llm.layers.{i}"
        kv = (
            text.swa_num_key_value_heads
            if i in local
            else text.ckpt_num_key_value_heads
        )
        rel_extent = text.sliding_window_size if i in local else text.rel_extent
        ckpt[f"{p}.attn.wq_du.weight"] = rand(heads * hd, h)
        ckpt[f"{p}.attn.wk_dv.weight"] = rand(kv * hd, h)
        ckpt[f"{p}.attn.wv_dv.weight"] = rand(kv * hd, h)
        ckpt[f"{p}.attn.wr_du.weight"] = rand(heads * text.d_rel, h)
        ckpt[f"{p}.attn.wo_ud.weight"] = rand(h, heads * hd)
        ckpt[f"{p}.attn.q_norm.weight"] = rand(hd)
        ckpt[f"{p}.attn.k_norm.weight"] = rand(hd)
        ckpt[f"{p}.attn.rel_logits_proj.proj"] = rand(text.d_rel, rel_extent)
        ckpt[f"{p}.attn.k_sconv.weight"] = rand(kv * hd, 1, text.sconv_kernel_size)
        ckpt[f"{p}.attn.v_sconv.weight"] = rand(kv * hd, 1, text.sconv_kernel_size)
        ckpt[f"{p}.attn_norm.weight"] = rand(h)
        ckpt[f"{p}.mlp_norm.weight"] = rand(h)
        ckpt[f"{p}.attn_sconv.weight"] = rand(h, 1, text.sconv_kernel_size)
        ckpt[f"{p}.mlp_sconv.weight"] = rand(h, 1, text.sconv_kernel_size)
        if i < text.dense_mlp_idx:
            ckpt[f"{p}.mlp.w13_dn.weight"] = rand(2 * text.dense_intermediate_size, h)
            ckpt[f"{p}.mlp.w2_md.weight"] = rand(h, text.dense_intermediate_size)
            ckpt[f"{p}.mlp.global_scale"] = rand(1)
        else:
            n_total = text.n_routed_experts + text.n_shared_experts
            ckpt[f"{p}.mlp.gate.weight"] = rand(n_total, h)
            ckpt[f"{p}.mlp.gate.bias"] = rand(
                text.n_routed_experts, dtype=torch.float32
            )
            ckpt[f"{p}.mlp.gate.global_scale"] = rand(1, dtype=torch.float32)
            E, I2 = text.n_routed_experts, 2 * text.intermediate_size
            if i in fp4_layers:
                for w, rows, cols in (
                    ("w13", I2, h),
                    ("w2", h, text.intermediate_size),
                ):
                    q = f"{p}.mlp.experts.{w}_weight"
                    ckpt[q] = rand_bytes(E, rows, cols // 2)
                    ckpt[f"{q}.scale"] = rand_bytes(
                        E, rows, cols // 16, dtype=torch.float8_e4m3fn
                    )
                    ckpt[f"{q}.scale2"] = rand(E, dtype=torch.float32).abs()
                    ckpt[f"{q}.input_amax"] = rand(1).abs()
                    ckpt[f"{q}.original_shape"] = torch.tensor([E, rows, cols])
            else:
                ckpt[f"{p}.mlp.experts.w13_weight"] = rand(E, I2, h)
                ckpt[f"{p}.mlp.experts.w2_weight"] = rand(E, h, text.intermediate_size)
            ckpt[f"{p}.mlp.shared_experts.shared_w13_weight"] = rand(
                text.n_shared_experts, 2 * text.intermediate_size, h
            )
            ckpt[f"{p}.mlp.shared_experts.shared_w2_weight"] = rand(
                text.n_shared_experts, h, text.intermediate_size
            )
    return ckpt


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingLoadWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["INKLING_TORCH_MOE"] = "1"  # experts as plain parameters
        cls._tmpdir = tempfile.TemporaryDirectory()
        ckpt_dir = make_inkling_dummy_checkpoint(cls._tmpdir.name, tiny=True)
        cls.model, cls.text = _build_model(str(ckpt_dir))
        cls.ckpt = _make_checkpoint_tensors(cls.text)
        cls.loaded = cls.model.load_weights(iter(cls.ckpt.items()))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _param(self, name: str) -> torch.Tensor:
        return dict(self.model.named_parameters())[name].data.cpu()

    def _assert_eq(self, name: str, expected: torch.Tensor):
        got = self._param(name)
        expected = expected.to(got.dtype)
        self.assertEqual(got.shape, expected.shape, name)
        self.assertTrue(torch.equal(got, expected), f"{name} mismatch")

    def test_full_parameter_coverage(self):
        params = dict(self.model.named_parameters())
        missing = sorted(set(params) - self.loaded)
        self.assertEqual(missing, [], f"params never loaded: {missing}")

    def test_qkvr_fusion_and_kv_replication(self):
        from tokenspeed.runtime.configs.inkling_config import (
            inkling_kv_heads_for_layer,
        )

        text = self.text
        hd = text.head_dim
        hetero = True  # gate retired 2026-07-15; hetero KV is unconditional
        local = set(text.local_layer_ids)
        for i in range(text.num_hidden_layers):
            p = f"model.llm.layers.{i}"
            kv = (
                text.swa_num_key_value_heads
                if i in local
                else text.ckpt_num_key_value_heads
            )
            # Hetero (default): full layers serve their native ckpt heads
            # (factor 1, no replication); uniform mode replicates to the max.
            served = inkling_kv_heads_for_layer(text, i, hetero)
            factor = served // kv
            wk = self.ckpt[f"{p}.attn.wk_dv.weight"]
            wv = self.ckpt[f"{p}.attn.wv_dv.weight"]
            # Served head j must carry checkpoint head j // factor.
            rep_k = torch.cat(
                [wk[(j // factor) * hd : (j // factor + 1) * hd] for j in range(served)]
            )
            rep_v = torch.cat(
                [wv[(j // factor) * hd : (j // factor + 1) * hd] for j in range(served)]
            )
            expected = torch.cat(
                [
                    self.ckpt[f"{p}.attn.wq_du.weight"],
                    rep_k,
                    rep_v,
                    self.ckpt[f"{p}.attn.wr_du.weight"],
                ]
            )
            self._assert_eq(f"model.layers.{i}.attn.qkvr.weight", expected)
            # K/V sconv taps replicate the same way.
            ks = self.ckpt[f"{p}.attn.k_sconv.weight"]
            rep_ks = torch.cat(
                [ks[(j // factor) * hd : (j // factor + 1) * hd] for j in range(served)]
            )
            self._assert_eq(f"model.layers.{i}.attn.k_sconv.weight", rep_ks)

    def test_dense_mlp_deinterleave_and_scale(self):
        for i in range(self.text.dense_mlp_idx):
            p = f"model.llm.layers.{i}"
            o = f"model.layers.{i}"
            self._assert_eq(
                f"{o}.mlp.gate_up_proj.weight",
                _deinterleave_rows(self.ckpt[f"{p}.mlp.w13_dn.weight"]),
            )
            self._assert_eq(
                f"{o}.mlp.down_proj.weight", self.ckpt[f"{p}.mlp.w2_md.weight"]
            )
            self._assert_eq(f"{o}.mlp.global_scale", self.ckpt[f"{p}.mlp.global_scale"])

    def test_moe_experts_and_gate(self):
        text = self.text
        for i in range(text.dense_mlp_idx, text.num_hidden_layers):
            p = f"model.llm.layers.{i}"
            o = f"model.layers.{i}"
            self._assert_eq(f"{o}.mlp.gate.weight", self.ckpt[f"{p}.mlp.gate.weight"])
            self._assert_eq(f"{o}.mlp.gate.bias", self.ckpt[f"{p}.mlp.gate.bias"])
            self._assert_eq(
                f"{o}.mlp.gate.global_scale", self.ckpt[f"{p}.mlp.gate.global_scale"]
            )
            self._assert_eq(
                f"{o}.mlp.experts.w13_weight",
                _deinterleave_rows(self.ckpt[f"{p}.mlp.experts.w13_weight"]),
            )
            self._assert_eq(
                f"{o}.mlp.experts.w2_weight", self.ckpt[f"{p}.mlp.experts.w2_weight"]
            )
            self._assert_eq(
                f"{o}.mlp.shared_experts.w13_weight",
                _deinterleave_rows(
                    self.ckpt[f"{p}.mlp.shared_experts.shared_w13_weight"]
                ),
            )
            self._assert_eq(
                f"{o}.mlp.shared_experts.w2_weight",
                self.ckpt[f"{p}.mlp.shared_experts.shared_w2_weight"],
            )

    def test_moe_block_output_passes_through_moe_comm(self):
        """The sparse-MoE block output is a rank-local partial sum (routed:
        TP-sharded intermediate / EP-sharded experts; shared: TP-sharded
        intermediate) and must go through CommManager.post_moe_comm — the
        all-reduce at TP/EP > 1, identity at world size 1."""
        layer = next(la for la in self.model.model.layers if la.is_moe)
        block = layer.mlp
        seen = []
        orig = block.comm_manager.post_moe_comm

        def spy(hidden_states, residual, ctx):
            seen.append(hidden_states)
            return orig(hidden_states, residual, ctx)

        block.comm_manager.post_moe_comm = spy
        try:
            x = torch.randn(
                3, self.text.hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            out = block(x, ctx=None)
        finally:
            block.comm_manager.post_moe_comm = orig
        self.assertEqual(len(seen), 1)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(out, seen[0]))

    def test_nvfp4_aux_names_are_consumed_without_error(self):
        """On unquantized layers the FP4 aux tensors drop with a warning; the
        packing metadata is skipped silently. Neither may raise."""
        p = f"model.llm.layers.{self.text.dense_mlp_idx}.mlp.experts"
        extra = {
            f"{p}.w13_weight.original_shape": torch.tensor([8, 128, 256]),
            f"{p}.w13_weight.scale": torch.zeros(8, 128, 16),
            f"{p}.w13_weight.scale2": torch.zeros(8),
            f"{p}.w13_weight.input_amax": torch.zeros(1),
            # One regular tensor so the loaded-nothing guard stays out of play.
            "model.llm.norm.weight": self.ckpt["model.llm.norm.weight"],
        }
        loaded = self.model.load_weights(iter(extra.items()))
        self.assertEqual(loaded, {"model.norm.weight"})

    def test_lm_head_ties_to_embedding_when_unembed_absent(self):
        stream = {k: v for k, v in self.ckpt.items() if k != "model.llm.unembed.weight"}
        # Perturb the embedding so the tie is observable.
        stream["model.llm.embed.weight"] = (
            self.ckpt["model.llm.embed.weight"].float() + 1.0
        ).to(torch.bfloat16)
        loaded = self.model.load_weights(iter(stream.items()))
        self.assertIn("lm_head.weight", loaded)
        self._assert_eq("lm_head.weight", stream["model.llm.embed.weight"])
        # Restore the original weights for any later test.
        self.model.load_weights(iter(self.ckpt.items()))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingMoeBlockDeferredFinalize(unittest.TestCase):
    """The sparse block's deferred branch (routed trtllm do_finalize=False +
    moe_finalize_fuse_shared with the gate's full [T, k+S] weights) must
    match the fallback branch (in-kernel finalize + separate shared add) on
    the same weights."""

    def test_deferred_matches_fallback(self):
        from tokenspeed_kernel.platform import ArchVersion, current_platform

        plat = current_platform()
        if not (
            plat.is_nvidia
            and plat.is_blackwell
            and plat.arch_version == ArchVersion(10, 0)
        ):
            self.skipTest("routed trtllm MoE kernels need SM100")

        from tokenspeed.runtime.configs.inkling_config import InklingModelConfig
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.layers.moe import utils as moe_utils
        from tokenspeed.runtime.models.inkling import InklingSparseMoeBlock
        from tokenspeed.runtime.utils.env import global_server_args_dict

        torch.manual_seed(SEED)
        # The routed unquant trtllm kernel requires ispp % 128 == 0; the tiny
        # fixture's intermediate 64 doesn't qualify, so widen it here.
        cfg = dict(TINY_MM_CONFIG["text_config"])
        cfg["intermediate_size"] = 256
        text = InklingModelConfig(**cfg)

        saved_backend = moe_utils.MOE_BACKEND
        saved_torch_moe = os.environ.pop("INKLING_TORCH_MOE", None)
        moe_utils.MOE_BACKEND = moe_utils.MoeBackend.FLASHINFER_TRTLLM
        mapping = Mapping(rank=0, world_size=1)
        global_server_args_dict["mapping"] = mapping
        global_server_args_dict["enable_prefix_caching"] = False
        try:
            with torch.device("cuda"):
                torch.set_default_dtype(torch.bfloat16)
                try:
                    block = InklingSparseMoeBlock(text, mapping, layer_id=2)
                finally:
                    torch.set_default_dtype(torch.float32)
            self.assertTrue(block.experts.supports_deferred_finalize)
            self.assertFalse(block.experts.support_routing)

            with torch.no_grad():
                for p in block.parameters():
                    p.data = torch.randn_like(p.data) * 0.05
            block.experts.process_weights_after_loading(block.experts)

            num_tokens = 33
            x = torch.randn(
                num_tokens, text.hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            block.comm_manager.get_num_tokens = lambda ctx: (num_tokens, num_tokens)

            out_deferred = block(x, ctx=None)

            # Same block, same weights, forced down the fallback branch.
            block.experts.plan["supports_deferred_finalize"] = False
            try:
                out_fallback = block(x, ctx=None)
            finally:
                block.experts.plan["supports_deferred_finalize"] = True

            self.assertEqual(out_deferred.shape, out_fallback.shape)
            # The two branches round differently (f32 vs bf16 weight
            # application, gamma pre- vs post-down-proj); tolerate bf16 noise.
            torch.testing.assert_close(
                out_deferred.float(), out_fallback.float(), atol=3e-2, rtol=3e-2
            )
        finally:
            moe_utils.MOE_BACKEND = saved_backend
            if saved_torch_moe is not None:
                os.environ["INKLING_TORCH_MOE"] = saved_torch_moe


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingLoadWeightsParallel(unittest.TestCase):
    """Per-rank slice correctness under TP/EP.

    Builds one model per rank of a parallel layout in the same process (no
    collectives run during weight loading), loads the SAME checkpoint stream
    into each, and asserts every rank holds exactly its slice of the full
    tensors — attention TP sharding, KV replication when TP exceeds the KV
    head count, MoE expert-parallel partitioning, and MoE TP narrowing.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.pop("INKLING_TORCH_MOE", None)  # real MoELayer path
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.ckpt_dir = str(make_inkling_dummy_checkpoint(cls._tmpdir.name, tiny=True))
        cls.ckpt = None  # built after the first model gives us the text config

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _load_rank(self, rank, world_size, **mapping_kw):
        model, text = _build_model(
            self.ckpt_dir, rank=rank, world_size=world_size, **mapping_kw
        )
        if type(self).ckpt is None:
            type(self).ckpt = _make_checkpoint_tensors(text)
        model.load_weights(iter(self.ckpt.items()))
        return model, text

    def _param(self, model, name):
        return dict(model.named_parameters())[name].data.cpu()

    def test_attention_tp2_and_moe_tp2(self):
        from tokenspeed.runtime.configs.inkling_config import (
            inkling_kv_heads_for_layer,
        )

        hd = 32
        hetero = True  # gate retired 2026-07-15; hetero KV is unconditional
        for rank in range(2):
            model, text = self._load_rank(rank, 2)
            tp = 2
            q_heads = text.num_attention_heads // tp  # 4 per rank
            # Layer 4 is local (ckpt KV == served). Layer 5 is full: hetero
            # (default) serves its native 2 heads (1 per rank, factor 1);
            # uniform mode replicates 2 -> 4.
            for layer, ckpt_kv in ((4, 4), (5, 2)):
                served = max(inkling_kv_heads_for_layer(text, layer, hetero), tp)
                p = f"model.llm.layers.{layer}.attn"
                qkvr = self._param(model, f"model.layers.{layer}.attn.qkvr.weight")
                wq = self.ckpt[f"{p}.wq_du.weight"]
                q_rows = q_heads * hd
                self.assertTrue(
                    torch.equal(qkvr[:q_rows], wq[rank * q_rows : (rank + 1) * q_rows])
                )
                # K section: rank's served heads <- ckpt head j // factor.
                wk = self.ckpt[f"{p}.wk_dv.weight"]
                factor = served // ckpt_kv
                for j_local in range(served // tp):
                    j = rank * (served // tp) + j_local
                    src = (j // factor) * hd
                    got = qkvr[q_rows + j_local * hd : q_rows + (j_local + 1) * hd]
                    self.assertTrue(
                        torch.equal(got, wk[src : src + hd]),
                        f"layer{layer} rank{rank} kv head {j}",
                    )
                # wo_ud (row-parallel): input-dim shard.
                wo = self._param(model, f"model.layers.{layer}.attn.wo_ud.weight")
                full = self.ckpt[f"{p}.wo_ud.weight"]
                cols = full.shape[1] // tp
                self.assertTrue(
                    torch.equal(wo, full[:, rank * cols : (rank + 1) * cols])
                )
            # Dense MLP (layer 0): per-half gate_up rows, down_proj cols.
            gu = self._param(model, "model.layers.0.mlp.gate_up_proj.weight")
            full = _deinterleave_rows(self.ckpt["model.llm.layers.0.mlp.w13_dn.weight"])
            half = full.shape[0] // 2
            sh = half // tp
            expected = torch.cat(
                [
                    full[rank * sh : (rank + 1) * sh],
                    full[half + rank * sh : half + (rank + 1) * sh],
                ]
            )
            self.assertTrue(torch.equal(gu, expected))
            # Routed experts under MoE TP=2 (default moe layout for world 2).
            w13 = self._param(model, "model.layers.4.mlp.experts.w13_weight")
            full13 = _deinterleave_rows(
                self.ckpt["model.llm.layers.4.mlp.experts.w13_weight"]
            )
            ifull = full13.shape[1] // 2
            isl = ifull // tp
            expected13 = torch.cat(
                [
                    full13[:, rank * isl : (rank + 1) * isl],
                    full13[:, ifull + rank * isl : ifull + (rank + 1) * isl],
                ],
                dim=1,
            )
            self.assertTrue(torch.equal(w13, expected13), f"moe tp w13 rank{rank}")
            w2 = self._param(model, "model.layers.4.mlp.experts.w2_weight")
            full2 = self.ckpt["model.llm.layers.4.mlp.experts.w2_weight"]
            self.assertTrue(torch.equal(w2, full2[:, :, rank * isl : (rank + 1) * isl]))
            # Shared experts follow attention TP.
            sw13 = self._param(model, "model.layers.4.mlp.shared_experts.w13_weight")
            sfull = _deinterleave_rows(
                self.ckpt["model.llm.layers.4.mlp.shared_experts.shared_w13_weight"]
            )
            sifull = sfull.shape[1] // 2
            ssl = sifull // tp
            sexp = torch.cat(
                [
                    sfull[:, rank * ssl : (rank + 1) * ssl],
                    sfull[:, sifull + rank * ssl : sifull + (rank + 1) * ssl],
                ],
                dim=1,
            )
            self.assertTrue(torch.equal(sw13, sexp))
            del model
            torch.cuda.empty_cache()

    def test_attention_tp8_kv_replication(self):
        """TP (8) beyond the served KV head count (4): one KV head per rank,
        replicated block-contiguously; ckpt head = (rank * ckpt_kv) // tp."""
        hd, tp = 32, 8
        for rank in (0, 3, 5, 7):
            model, text = self._load_rank(rank, tp)
            for layer, ckpt_kv in ((4, 4), (5, 2)):
                p = f"model.llm.layers.{layer}.attn"
                qkvr = self._param(model, f"model.layers.{layer}.attn.qkvr.weight")
                q_rows = (text.num_attention_heads // tp) * hd  # 1 head
                wk = self.ckpt[f"{p}.wk_dv.weight"]
                src = ((rank * ckpt_kv) // tp) * hd
                got = qkvr[q_rows : q_rows + hd]
                self.assertTrue(
                    torch.equal(got, wk[src : src + hd]),
                    f"layer{layer} rank{rank} replicated kv",
                )
                # K sconv taps replicate identically.
                ks = self._param(model, f"model.layers.{layer}.attn.k_sconv.weight")
                full_ks = self.ckpt[f"{p}.k_sconv.weight"]
                self.assertTrue(torch.equal(ks, full_ks[src : src + hd]))
            del model
            torch.cuda.empty_cache()

    def test_moe_ep2_expert_partition(self):
        for rank in range(2):
            try:
                model, text = self._load_rank(rank, 2, moe_ep_size=2, moe_tp_size=1)
            except Exception as e:  # pragma: no cover - environment specific
                self.skipTest(f"EP-2 MoELayer construction unavailable: {e}")
            n_local = text.n_routed_experts // 2
            w13 = self._param(model, "model.layers.4.mlp.experts.w13_weight")
            self.assertEqual(w13.shape[0], n_local)
            full13 = _deinterleave_rows(
                self.ckpt["model.llm.layers.4.mlp.experts.w13_weight"]
            )
            for local in range(n_local):
                self.assertTrue(
                    torch.equal(w13[local], full13[rank * n_local + local]),
                    f"ep rank{rank} expert {local}",
                )
            w2 = self._param(model, "model.layers.4.mlp.experts.w2_weight")
            full2 = self.ckpt["model.llm.layers.4.mlp.experts.w2_weight"]
            self.assertTrue(
                torch.equal(w2, full2[rank * n_local : (rank + 1) * n_local])
            )
            del model
            torch.cuda.empty_cache()

    def test_logits_processor_gets_attn_tp_topology(self):
        """The vocab-parallel lm_head shards logits across attn-TP ranks, so
        LogitsProcessor must receive the attn-TP topology or it will never
        all-gather: sampling would only ever see rank 0's [vocab/tp]-wide
        shard, making ids >= vocab/tp (Inkling's chat framing tokens live at
        ~200k) unreachable. Regression test for the TP4 serving bug where
        greedy decode could not emit <|content_thinking|>/<|end_message|>."""
        model, _ = self._load_rank(1, 4)
        mapping = model.mapping.attn if hasattr(model, "mapping") else None
        lp = model.logits_processor
        self.assertEqual(lp.tp_size, 4)
        self.assertEqual(lp.tp_rank, 1)
        self.assertIsNotNone(lp.tp_group)
        self.assertFalse(lp.skip_all_gather)
        if mapping is not None:
            self.assertEqual(lp.tp_group, mapping.tp_group)
        del model
        torch.cuda.empty_cache()


def _checkpoint_style_exclude_list(text) -> list[str]:
    """The real FP4 snapshot's exclude_modules structure, tiny-config sized:
    everything except routed experts, and the first MoE layer's experts too."""
    exclude = [
        "model.llm.embed",
        "model.llm.embed_norm",
        "model.llm.norm",
        "model.llm.unembed",
    ]
    for i in range(text.num_hidden_layers):
        p = f"model.llm.layers.{i}"
        exclude += [
            f"{p}.attn",
            f"{p}.attn_norm",
            f"{p}.attn_sconv",
            f"{p}.mlp_norm",
            f"{p}.mlp_sconv",
        ]
        if i < text.dense_mlp_idx:
            exclude += [
                f"{p}.mlp.w13_dn",
                f"{p}.mlp.w2_md",
                f"{p}.mlp.global_scale",
            ]
        else:
            exclude += [f"{p}.mlp.gate", f"{p}.mlp.shared_experts"]
    exclude.append(f"model.llm.layers.{text.dense_mlp_idx}.mlp.experts")
    return exclude


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingLoadWeightsNvfp4(unittest.TestCase):
    """NVFP4 flavor: mixed-precision experts (first MoE layer stays BF16),
    packed U8 weights, F8 block scales, scale2/input_amax routing."""

    @classmethod
    def setUpClass(cls):
        from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

        os.environ.pop("INKLING_TORCH_MOE", None)  # real MoELayer path
        cls._tmpdir = tempfile.TemporaryDirectory()
        ckpt_dir = make_inkling_dummy_checkpoint(cls._tmpdir.name, tiny=True)
        cfg_path = os.path.join(str(ckpt_dir), "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["text_config"]["use_global_scale"] = True
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        from tokenspeed.runtime.configs.inkling_config import InklingModelConfig

        text_probe = InklingModelConfig(**cfg["text_config"])
        quant = Nvfp4Config(
            group_size=16,
            exclude_modules=_checkpoint_style_exclude_list(text_probe),
        )
        cls.model, cls.text = _build_model(str(ckpt_dir), quant_config=quant)
        cls.fp4_layers = set(
            range(cls.text.dense_mlp_idx + 1, cls.text.num_hidden_layers)
        )
        cls.ckpt = _make_checkpoint_tensors(cls.text, fp4_layers=cls.fp4_layers)
        cls.loaded = cls.model.load_weights(iter(cls.ckpt.items()))

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.cuda.empty_cache()
        cls._tmpdir.cleanup()

    def _param(self, name: str) -> torch.Tensor:
        return dict(self.model.named_parameters())[name].data.cpu()

    def test_exclusion_translation(self):
        translated = self.model.quant_config.exclude_modules
        self.assertIn("model.layers.2.mlp.experts", translated)
        self.assertIn("model.layers.2.mlp.experts.*", translated)
        self.assertIn("model.layers.0.mlp.gate_up_proj", translated)
        self.assertIn("lm_head", translated)
        self.assertFalse([e for e in translated if e.startswith("model.llm.")])

    def test_full_parameter_coverage(self):
        params = dict(self.model.named_parameters())
        missing = sorted(set(params) - self.loaded)
        self.assertEqual(missing, [], f"params never loaded: {missing}")

    def test_first_moe_layer_experts_stay_bf16(self):
        i = self.text.dense_mlp_idx
        w13 = self._param(f"model.layers.{i}.mlp.experts.w13_weight")
        self.assertEqual(w13.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                w13,
                _deinterleave_rows(
                    self.ckpt[f"model.llm.layers.{i}.mlp.experts.w13_weight"]
                ),
            )
        )

    def test_quantized_experts_packed_and_scales(self):
        i = min(self.fp4_layers)
        p = f"model.llm.layers.{i}.mlp.experts"
        o = f"model.layers.{i}.mlp.experts"
        w13 = self._param(f"{o}.w13_weight")
        self.assertEqual(w13.dtype, torch.uint8)
        self.assertTrue(
            torch.equal(w13, _deinterleave_rows(self.ckpt[f"{p}.w13_weight"]))
        )
        w2 = self._param(f"{o}.w2_weight")
        self.assertTrue(torch.equal(w2, self.ckpt[f"{p}.w2_weight"]))
        scale = self._param(f"{o}.w13_weight_scale")
        self.assertEqual(scale.dtype, torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(
                scale.view(torch.uint8),
                _deinterleave_rows(
                    self.ckpt[f"{p}.w13_weight.scale"].view(torch.uint8)
                ),
            )
        )

    def test_scale2_and_input_scale_conversion(self):
        i = min(self.fp4_layers)
        p = f"model.llm.layers.{i}.mlp.experts"
        o = f"model.layers.{i}.mlp.experts"
        scale_2 = self._param(f"{o}.w13_weight_scale_2")
        ckpt_scale2 = self.ckpt[f"{p}.w13_weight.scale2"]
        self.assertTrue(torch.equal(scale_2[:, 0], ckpt_scale2))
        self.assertTrue(torch.equal(scale_2[:, 1], ckpt_scale2))
        input_scale = self._param(f"{o}.w13_input_scale")
        expected = self.ckpt[f"{p}.w13_weight.input_amax"].float() / (448.0 * 6.0)
        self.assertTrue(torch.allclose(input_scale, expected.expand_as(input_scale)))
        w2_input_scale = self._param(f"{o}.w2_input_scale")
        expected2 = self.ckpt[f"{p}.w2_weight.input_amax"].float() / (448.0 * 6.0)
        self.assertTrue(
            torch.allclose(w2_input_scale, expected2.expand_as(w2_input_scale))
        )

    def test_non_expert_modules_stay_bf16(self):
        self.assertEqual(
            self._param("model.layers.3.attn.qkvr.weight").dtype, torch.bfloat16
        )
        self.assertEqual(
            self._param("model.layers.0.mlp.gate_up_proj.weight").dtype,
            torch.bfloat16,
        )
        self.assertEqual(
            self._param("model.layers.3.mlp.gate.bias").dtype, torch.float32
        )


if __name__ == "__main__":
    unittest.main()
