"""Inkling NextN (MTP draft) ``load_weights`` unit test.

Builds an in-memory ``model.mtp.*`` checkpoint using the REAL checkpoint
tensor names for a tiny config with ``mtp_config`` set, then asserts every
draft parameter is covered (embedding/lm_head excepted — those are shared
from the target via ``set_embed_and_head``) and that the block transforms
land like the base loader: qkvr fusion order, KV replication to the uniform
served head count, w13 gate/up de-interleave, and the depth/chain_norm
remaps. Also asserts the depth blocks come out full-attention + dense-MLP
(legacy heads) and, for ``mtp_config.local_layer_ids`` checkpoints
(4d71c3ea+), that SWA depths build local attention at the swa head count and
that depths pruned by ``--speculative-num-steps`` are skipped by the loader.

NOTE: intentionally not registered in CI; run locally after loader changes.
Needs any CUDA GPU (weight copies only, no kernels).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.models.inkling_fixtures import TINY_MM_CONFIG  # noqa: E402

SEED = 4321
NUM_DEPTHS = 2


def _tiny_nextn_config(num_depths=NUM_DEPTHS, mtp_local_layer_ids=()):
    from tokenspeed.runtime.configs.inkling_config import InklingMMConfig

    cfg_dict = {
        **TINY_MM_CONFIG,
        "architectures": ["InklingForConditionalGenerationNextN"],
        "mtp_config": {
            "num_nextn_predict_layers": num_depths,
            "chain_hidden_post_norm": True,
            "local_layer_ids": list(mtp_local_layer_ids),
        },
    }
    cfg_dict = {
        k: v for k, v in cfg_dict.items() if k not in ("model_type", "architectures")
    }
    return InklingMMConfig(**cfg_dict)


def _make_mtp_checkpoint(
    text, num_depths=NUM_DEPTHS, local_layer_ids=()
) -> dict[str, torch.Tensor]:
    """One tensor per real-checkpoint ``model.mtp.*`` name (tiny dims).

    SWA depths (``local_layer_ids``) ship the swa KV head count and a
    rel-logits extent of the sliding window, like the 4d71c3ea checkpoint.
    """
    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(
            torch.bfloat16
        )

    h = text.hidden_size
    hd = text.head_dim
    heads = text.num_attention_heads
    d_rel = text.d_rel
    dense_int = text.dense_intermediate_size
    w = text.sconv_kernel_size
    local = set(local_layer_ids)

    ckpt: dict[str, torch.Tensor] = {
        "model.mtp.chain_norm.weight": rand(h),
        # Base-model embed norm, consumed as base_embed_norm (loader remap).
        "model.llm.embed_norm.weight": rand(h),
    }
    for i in range(num_depths):
        ckpt_kv = (
            text.swa_num_key_value_heads
            if i in local
            else text.ckpt_num_key_value_heads
        )
        rel_extent = text.sliding_window_size if i in local else text.rel_extent
        p = f"model.mtp.layers.{i}."
        b = p + "transformer_block."
        ckpt.update(
            {
                p + "hidden_norm.weight": rand(h),
                p + "embed_norm.weight": rand(h),
                p + "input_proj.weight": rand(h, 2 * h),
                b + "attn.wq_du.weight": rand(heads * hd, h),
                b + "attn.wk_dv.weight": rand(ckpt_kv * hd, h),
                b + "attn.wv_dv.weight": rand(ckpt_kv * hd, h),
                b + "attn.wr_du.weight": rand(d_rel * heads, h),
                b + "attn.wo_ud.weight": rand(h, heads * hd),
                b + "attn.q_norm.weight": rand(hd),
                b + "attn.k_norm.weight": rand(hd),
                b + "attn.rel_logits_proj.proj": rand(d_rel, rel_extent),
                b + "attn.k_sconv.weight": rand(ckpt_kv * hd, 1, w),
                b + "attn.v_sconv.weight": rand(ckpt_kv * hd, 1, w),
                b + "attn_norm.weight": rand(h),
                b + "attn_sconv.weight": rand(h, 1, w),
                b + "mlp.w13_dn.weight": rand(2 * dense_int, h),
                b + "mlp.w2_md.weight": rand(h, dense_int),
                b + "mlp.global_scale": rand(1),
                b + "mlp_norm.weight": rand(h),
                b + "mlp_sconv.weight": rand(h, 1, w),
            }
        )
    return ckpt


def _deinterleave_rows(w: torch.Tensor) -> torch.Tensor:
    return torch.cat([w[..., 0::2, :], w[..., 1::2, :]], dim=-2)


def _replicate_heads(w: torch.Tensor, ckpt_heads: int, target: int, hd: int):
    x = w.reshape(ckpt_heads, hd, *w.shape[1:])
    x = torch.repeat_interleave(x, target // ckpt_heads, dim=0)
    return x.reshape(target * hd, *w.shape[1:])


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingMTPLoadWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )
        from tokenspeed.runtime.utils.env import global_server_args_dict

        mapping = Mapping(rank=0, world_size=1)
        global_server_args_dict["mapping"] = mapping
        global_server_args_dict["enable_prefix_caching"] = False
        cls.config = _tiny_nextn_config()
        cls.text = cls.config.get_text_config()
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            try:
                cls.model = InklingForConditionalGenerationNextN(cls.config, mapping)
            finally:
                torch.set_default_dtype(torch.float32)
        cls.model.eval()
        cls.ckpt = _make_mtp_checkpoint(cls.text)
        # Base-model tensors must be ignored by the NextN loader.
        cls.ckpt_with_noise = {
            "model.llm.embed.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            **cls.ckpt,
        }
        cls.loaded = cls.model.load_weights(iter(cls.ckpt_with_noise.items()))
        cls.params = dict(cls.model.named_parameters())

    def test_structure_full_attention_dense_mlp(self):
        from tokenspeed.runtime.models.inkling import InklingDenseMLP

        self.assertEqual(len(self.model.model.layers), NUM_DEPTHS)
        for layer in self.model.model.layers:
            block = layer.transformer_block
            self.assertFalse(block.is_moe)
            self.assertIsInstance(block.mlp, InklingDenseMLP)
            self.assertFalse(block.attn.is_local)
            self.assertEqual(block.attn.rel_extent, self.text.rel_extent)

    def test_all_params_loaded_except_shared(self):
        shared = {"model.embed_tokens.weight", "lm_head.weight"}
        missing = set(self.params) - self.loaded - shared
        self.assertEqual(missing, set(), f"params not loaded: {sorted(missing)}")

    def test_qkvr_fusion_with_kv_replication(self):
        text = self.text
        hd, heads = text.head_dim, text.num_attention_heads
        # Hetero (default): depth blocks are full-attention and serve their
        # native ckpt KV heads — the loader's replication is a no-op.
        hetero = True  # gate retired 2026-07-15; hetero KV is unconditional
        target_kv = (
            text.ckpt_num_key_value_heads
            if hetero
            else max(text.num_key_value_heads, 1)
        )
        for i in range(NUM_DEPTHS):
            b = f"model.mtp.layers.{i}.transformer_block."
            wq = self.ckpt[b + "attn.wq_du.weight"]
            wk = _replicate_heads(
                self.ckpt[b + "attn.wk_dv.weight"],
                text.ckpt_num_key_value_heads,
                target_kv,
                hd,
            )
            wv = _replicate_heads(
                self.ckpt[b + "attn.wv_dv.weight"],
                text.ckpt_num_key_value_heads,
                target_kv,
                hd,
            )
            wr = self.ckpt[b + "attn.wr_du.weight"]
            expect = torch.cat([wq, wk, wv, wr], dim=0)
            got = self.params[
                f"model.layers.{i}.transformer_block.attn.qkvr.weight"
            ].cpu()
            self.assertTrue(torch.equal(got, expect), f"qkvr mismatch depth {i}")

    def test_w13_deinterleave(self):
        for i in range(NUM_DEPTHS):
            src = self.ckpt[f"model.mtp.layers.{i}.transformer_block.mlp.w13_dn.weight"]
            got = self.params[
                f"model.layers.{i}.transformer_block.mlp.gate_up_proj.weight"
            ].cpu()
            self.assertTrue(torch.equal(got, _deinterleave_rows(src)))

    def test_sconv_kv_replication(self):
        text = self.text
        hetero = True  # gate retired 2026-07-15; hetero KV is unconditional
        target_kv = (
            text.ckpt_num_key_value_heads if hetero else text.num_key_value_heads
        )
        for i in range(NUM_DEPTHS):
            src = self.ckpt[
                f"model.mtp.layers.{i}.transformer_block.attn.k_sconv.weight"
            ]
            expect = _replicate_heads(
                src,
                text.ckpt_num_key_value_heads,
                target_kv,
                text.head_dim,
            )
            got = self.params[
                f"model.layers.{i}.transformer_block.attn.k_sconv.weight"
            ].cpu()
            self.assertTrue(torch.equal(got, expect))

    def test_chain_norm_and_fusion_norms(self):
        self.assertTrue(
            torch.equal(
                self.params["model.chain_norm.weight"].cpu(),
                self.ckpt["model.mtp.chain_norm.weight"],
            )
        )
        for i in range(NUM_DEPTHS):
            for leaf in ("hidden_norm", "embed_norm", "input_proj"):
                got = self.params[f"model.layers.{i}.{leaf}.weight"].cpu()
                self.assertTrue(
                    torch.equal(got, self.ckpt[f"model.mtp.layers.{i}.{leaf}.weight"])
                )

    def test_missing_depth_raises(self):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )

        mapping = Mapping(rank=0, world_size=1)
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            try:
                model = InklingForConditionalGenerationNextN(self.config, mapping)
            finally:
                torch.set_default_dtype(torch.float32)
        partial = {
            k: v
            for k, v in self.ckpt.items()
            if not k.startswith("model.mtp.layers.1.")
        }
        with self.assertRaisesRegex(ValueError, r"depth layer\(s\) \[1\]"):
            model.load_weights(iter(partial.items()))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingMTPLoadWeightsSWADepths(unittest.TestCase):
    """4d71c3ea-style head: mixed SWA/full depths + steps pruning."""

    NUM_DEPTHS = 3
    LOCAL_IDS = (0, 2)

    @classmethod
    def setUpClass(cls):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )
        from tokenspeed.runtime.utils.env import global_server_args_dict

        cls.mapping = Mapping(rank=0, world_size=1)
        global_server_args_dict["mapping"] = cls.mapping
        global_server_args_dict["enable_prefix_caching"] = False
        cls.config = _tiny_nextn_config(cls.NUM_DEPTHS, cls.LOCAL_IDS)
        cls.text = cls.config.get_text_config()
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            try:
                cls.model = InklingForConditionalGenerationNextN(
                    cls.config, cls.mapping
                )
            finally:
                torch.set_default_dtype(torch.float32)
        cls.model.eval()
        cls.ckpt = _make_mtp_checkpoint(cls.text, cls.NUM_DEPTHS, cls.LOCAL_IDS)
        cls.loaded = cls.model.load_weights(iter(cls.ckpt.items()))
        cls.params = dict(cls.model.named_parameters())

    def test_structure_mixed_local_full(self):
        self.assertEqual(len(self.model.model.layers), self.NUM_DEPTHS)
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.transformer_block.attn
            if i in self.LOCAL_IDS:
                self.assertTrue(attn.is_local, f"depth {i}")
                self.assertEqual(attn.rel_extent, self.text.sliding_window_size)
                self.assertEqual(
                    attn.num_tp_kv_heads, self.text.swa_num_key_value_heads
                )
            else:
                self.assertFalse(attn.is_local, f"depth {i}")
                self.assertEqual(attn.rel_extent, self.text.rel_extent)
                self.assertEqual(
                    attn.num_tp_kv_heads, self.text.ckpt_num_key_value_heads
                )

    def test_all_params_loaded_except_shared(self):
        shared = {"model.embed_tokens.weight", "lm_head.weight"}
        missing = set(self.params) - self.loaded - shared
        self.assertEqual(missing, set(), f"params not loaded: {sorted(missing)}")

    def test_qkvr_fusion_no_replication_per_depth(self):
        # Hetero serving: every depth serves its native ckpt head count, so
        # the loader's replication is a no-op for BOTH depth kinds.
        for i in range(self.NUM_DEPTHS):
            b = f"model.mtp.layers.{i}.transformer_block."
            expect = torch.cat(
                [
                    self.ckpt[b + "attn.wq_du.weight"],
                    self.ckpt[b + "attn.wk_dv.weight"],
                    self.ckpt[b + "attn.wv_dv.weight"],
                    self.ckpt[b + "attn.wr_du.weight"],
                ],
                dim=0,
            )
            got = self.params[
                f"model.layers.{i}.transformer_block.attn.qkvr.weight"
            ].cpu()
            self.assertTrue(torch.equal(got, expect), f"qkvr mismatch depth {i}")

    def test_pruned_depths_are_skipped(self):
        from tokenspeed.runtime.configs.inkling_config import (
            inkling_mtp_text_config,
        )
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )

        # Simulate ModelConfig's draft-worker swap at --speculative-num-steps 2.
        config = _tiny_nextn_config(self.NUM_DEPTHS, self.LOCAL_IDS)
        config.text_config = inkling_mtp_text_config(
            config.get_text_config(), num_steps=2
        )
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            try:
                model = InklingForConditionalGenerationNextN(config, self.mapping)
            finally:
                torch.set_default_dtype(torch.float32)
        self.assertEqual(len(model.model.layers), 2)
        self.assertEqual(config.text_config.local_layer_ids, [0])
        # The full 3-depth checkpoint loads cleanly; depth 2 is skipped.
        loaded = model.load_weights(iter(self.ckpt.items()))
        self.assertFalse(any(n.startswith("model.layers.2.") for n in loaded))
        missing = (
            set(dict(model.named_parameters()))
            - loaded
            - {"model.embed_tokens.weight", "lm_head.weight"}
        )
        self.assertEqual(missing, set())


if __name__ == "__main__":
    unittest.main()
