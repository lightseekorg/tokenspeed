"""Inkling NextN (MTP draft) ``load_weights`` unit test.

Builds an in-memory ``model.mtp.*`` checkpoint using the REAL checkpoint
tensor names for the released config (layer-truncated; see
``inkling_fixtures``) with the test's own MTP layout, then asserts every
draft parameter is covered (embedding/lm_head excepted — those are shared
from the target via ``set_embed_and_head``) and that the block transforms
land like the base loader: qkvr fusion order, KV replication to the uniform
served head count, w13 gate/up de-interleave, and the depth/chain_norm
remaps. Also asserts the depth blocks come out full-attention + dense-MLP
(legacy heads) and, for ``mtp_config.local_layer_ids`` checkpoints
(4d71c3ea+), that SWA depths build local attention at the swa head count and
that depths pruned by ``--speculative-num-steps`` are skipped by the loader.

Needs any CUDA GPU (weight copies only, no kernels); CI runs it on the
Blackwell runners alongside the other Inkling suites.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test.runtime.models.inkling_fixtures import load_inkling_config  # noqa: E402

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*", "h100-*"],
)

SEED = 4321
NUM_DEPTHS = 2


def _nextn_config(num_depths=NUM_DEPTHS, mtp_local_layer_ids=()):
    """The released config with the test's own MTP head layout (the same
    fields ``InklingMMConfig`` parses out of ``mtp_config``)."""
    config = load_inkling_config()
    text = config.get_text_config()
    text.num_nextn_predict_layers = num_depths
    text.chain_hidden_post_norm = True
    text.mtp_local_layer_ids = list(mtp_local_layer_ids)
    return config


def _make_mtp_checkpoint(
    text, num_depths=NUM_DEPTHS, local_layer_ids=()
) -> dict[str, torch.Tensor]:
    """One tensor per real-checkpoint ``model.mtp.*`` name.

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


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs a CUDA device"))
class TestInklingMTPLoadWeights:
    @classmethod
    def setup_class(cls):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )
        from tokenspeed.runtime.utils.env import global_server_args_dict

        mapping = Mapping(rank=0, world_size=1)
        global_server_args_dict["mapping"] = mapping
        global_server_args_dict["enable_prefix_caching"] = False
        cls.config = _nextn_config()
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

        assert (len(self.model.model.layers)) == (NUM_DEPTHS)
        # The multi-depth drafter's depth-count contract attribute.
        assert (self.model.num_mtp_layers) == (NUM_DEPTHS)
        for layer in self.model.model.layers:
            block = layer.transformer_block
            assert not (block.is_moe)
            assert isinstance((block.mlp), (InklingDenseMLP))
            assert not (block.attn.is_local)
            assert (block.attn.rel_extent) == (self.text.rel_extent)

    def test_all_params_loaded_except_shared(self):
        shared = {"model.embed_tokens.weight", "lm_head.weight"}
        missing = set(self.params) - self.loaded - shared
        _pytest_first_1 = missing
        _pytest_second_2 = set()
        _pytest_msg_3 = f"params not loaded: {sorted(missing)}"
        assert _pytest_first_1 == _pytest_second_2, _pytest_msg_3

    def test_qkvr_fusion_with_kv_replication(self):
        text = self.text
        hd, _heads = text.head_dim, text.num_attention_heads
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
            _pytest_expr_7 = torch.equal(got, expect)
            _pytest_msg_8 = f"qkvr mismatch depth {i}"
            assert _pytest_expr_7, _pytest_msg_8

    def test_w13_deinterleave(self):
        for i in range(NUM_DEPTHS):
            src = self.ckpt[f"model.mtp.layers.{i}.transformer_block.mlp.w13_dn.weight"]
            got = self.params[
                f"model.layers.{i}.transformer_block.mlp.gate_up_proj.weight"
            ].cpu()
            assert torch.equal(got, _deinterleave_rows(src))

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
            assert torch.equal(got, expect)

    def test_chain_norm_and_fusion_norms(self):
        assert torch.equal(
            self.params["model.chain_norm.weight"].cpu(),
            self.ckpt["model.mtp.chain_norm.weight"],
        )
        for i in range(NUM_DEPTHS):
            for leaf in ("hidden_norm", "embed_norm", "input_proj"):
                got = self.params[f"model.layers.{i}.{leaf}.weight"].cpu()
                assert torch.equal(
                    got, self.ckpt[f"model.mtp.layers.{i}.{leaf}.weight"]
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
        with pytest.raises((ValueError), match=(r"depth layer\(s\) \[1\]")):
            model.load_weights(iter(partial.items()))


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs a CUDA device"))
class TestInklingMTPLoadWeightsSWADepths:
    """4d71c3ea-style head: mixed SWA/full depths + steps pruning."""

    NUM_DEPTHS = 3
    LOCAL_IDS = (0, 2)

    @classmethod
    def setup_class(cls):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )
        from tokenspeed.runtime.utils.env import global_server_args_dict

        cls.mapping = Mapping(rank=0, world_size=1)
        global_server_args_dict["mapping"] = cls.mapping
        global_server_args_dict["enable_prefix_caching"] = False
        cls.config = _nextn_config(cls.NUM_DEPTHS, cls.LOCAL_IDS)
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
        assert (len(self.model.model.layers)) == (self.NUM_DEPTHS)
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.transformer_block.attn
            if i in self.LOCAL_IDS:
                _pytest_expr_11 = attn.is_local
                _pytest_msg_12 = f"depth {i}"
                assert _pytest_expr_11, _pytest_msg_12
                assert (attn.rel_extent) == (self.text.sliding_window_size)
                assert (attn.num_tp_kv_heads) == (self.text.swa_num_key_value_heads)
            else:
                _pytest_expr_13 = attn.is_local
                _pytest_msg_14 = f"depth {i}"
                assert not _pytest_expr_13, _pytest_msg_14
                assert (attn.rel_extent) == (self.text.rel_extent)
                assert (attn.num_tp_kv_heads) == (self.text.ckpt_num_key_value_heads)

    def test_all_params_loaded_except_shared(self):
        shared = {"model.embed_tokens.weight", "lm_head.weight"}
        missing = set(self.params) - self.loaded - shared
        _pytest_first_4 = missing
        _pytest_second_5 = set()
        _pytest_msg_6 = f"params not loaded: {sorted(missing)}"
        assert _pytest_first_4 == _pytest_second_5, _pytest_msg_6

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
            _pytest_expr_9 = torch.equal(got, expect)
            _pytest_msg_10 = f"qkvr mismatch depth {i}"
            assert _pytest_expr_9, _pytest_msg_10

    def test_pruned_depths_are_skipped(self):
        from tokenspeed.runtime.configs.inkling_config import (
            inkling_mtp_text_config,
        )
        from tokenspeed.runtime.models.inkling_nextn import (
            InklingForConditionalGenerationNextN,
        )

        # Simulate ModelConfig's draft-worker swap at --speculative-num-steps 2.
        config = _nextn_config(self.NUM_DEPTHS, self.LOCAL_IDS)
        config.text_config = inkling_mtp_text_config(
            config.get_text_config(), num_steps=2
        )
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            try:
                model = InklingForConditionalGenerationNextN(config, self.mapping)
            finally:
                torch.set_default_dtype(torch.float32)
        assert (len(model.model.layers)) == (2)
        assert (config.text_config.local_layer_ids) == ([0])
        # The full 3-depth checkpoint loads cleanly; depth 2 is skipped.
        loaded = model.load_weights(iter(self.ckpt.items()))
        assert not (any(n.startswith("model.layers.2.") for n in loaded))
        missing = (
            set(dict(model.named_parameters()))
            - loaded
            - {"model.embed_tokens.weight", "lm_head.weight"}
        )
        assert (missing) == (set())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
