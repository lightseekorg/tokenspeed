"""Inkling ``load_weights`` validation against a real checkpoint (BF16 or NVFP4).

Gated on ``INKLING_REAL_CKPT`` naming a real snapshot directory (config.json +
sharded safetensors + index; NVFP4 snapshots additionally carry
hf_quant_config.json, which switches on the quantized-expert checks).
Builds a 4-layer replica of the real config —
dense layers 0/1, one full-attention MoE layer, one SWA MoE layer — feeds it
the corresponding real layers' tensors straight from the shards, and checks:

1. Every unique tensor-name pattern in the snapshot index is exercised (the
   checkpoint is layer-uniform, so the chosen layers cover all patterns).
2. Every replica parameter is covered by the load.
3. Values land exactly where the reference implementation puts them (qkvr
   fusion, full-layer KV replication, w13 de-interleave, fp32 router).

Run inside the weights container, e.g.::

    INKLING_REAL_CKPT=/path/to/inkling-checkpoint \
        python3 -m pytest test/runtime/test_inkling_real_checkpoint_load.py -q

NOTE: intentionally NOT registered in CI (confidential checkpoint; needs
~70 GB of GPU memory for the replica).
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CKPT = os.environ.get("INKLING_REAL_CKPT", "")

# Replica layer <- real checkpoint layer. Real layers 0/1 are dense, 5 is the
# first full-attention MoE layer, 2 is an SWA MoE layer.
_LAYER_MAP = {0: 0, 1: 1, 2: 5, 3: 2}
_REPLICA_LOCAL_IDS = [0, 1, 3]


def _layer_of(name: str) -> int | None:
    m = re.match(r"model\.llm\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _deinterleave_rows(w: torch.Tensor) -> torch.Tensor:
    return torch.cat([w[..., 0::2, :], w[..., 1::2, :]], dim=-2)


@unittest.skipUnless(
    _CKPT and Path(_CKPT).is_dir(), "INKLING_REAL_CKPT must name a snapshot dir"
)
@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingRealCheckpointLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors import safe_open

        snapshot = Path(_CKPT)
        with open(snapshot / "model.safetensors.index.json") as f:
            cls.weight_map: dict[str, str] = json.load(f)["weight_map"]

        # NVFP4 snapshots carry hf_quant_config.json; remap its exclude
        # entries' REAL layer ids into replica ids (entries for unmapped
        # layers are irrelevant to the replica and dropped).
        quant_config = None
        cls.is_fp4 = (snapshot / "hf_quant_config.json").exists()
        if cls.is_fp4:
            from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

            with open(snapshot / "hf_quant_config.json") as f:
                quant_config = Nvfp4Config.from_config(json.load(f))
            inv = {real: rep for rep, real in _LAYER_MAP.items()}
            remapped = []
            for entry in quant_config.exclude_modules:
                m = re.match(r"model\.llm\.layers\.(\d+)\.(.*)", entry)
                if m is None:
                    remapped.append(entry)
                elif int(m.group(1)) in inv:
                    remapped.append(
                        f"model.llm.layers.{inv[int(m.group(1))]}.{m.group(2)}"
                    )
            quant_config.exclude_modules = remapped

        with open(snapshot / "config.json") as f:
            cfg = json.load(f)
        cls.real_local_ids = set(cfg["text_config"]["local_layer_ids"])
        assert set(_LAYER_MAP.values()) & cls.real_local_ids == {
            0,
            1,
            2,
        }, "layer-kind assumptions changed; update _LAYER_MAP"
        cfg["text_config"]["num_hidden_layers"] = len(_LAYER_MAP)
        cfg["text_config"]["local_layer_ids"] = _REPLICA_LOCAL_IDS
        cfg["model_type"] = "inkling_mm_model"
        cfg["text_config"]["model_type"] = "inkling_model"
        cfg["audio_config"]["model_type"] = "inkling_audio_model"
        cfg["vision_config"]["model_type"] = "inkling_vision_model"
        cfg["architectures"] = ["InklingForConditionalGeneration"]
        cls._tmpdir = tempfile.TemporaryDirectory()
        with open(Path(cls._tmpdir.name) / "config.json", "w") as f:
            json.dump(cfg, f)

        # Real names to feed: top-level + towers + the mapped layers (+ one
        # MTP tensor to verify the skip path).
        renames: dict[str, str] = {}
        for name in cls.weight_map:
            layer = _layer_of(name)
            if layer is None:
                if name.startswith(("model.llm.", "model.audio.", "model.visual.")):
                    renames[name] = name
                elif name.startswith("model.mtp.chain_norm."):
                    renames[name] = name  # loader must skip it
                continue
            for replica_id, real_id in _LAYER_MAP.items():
                if layer == real_id:
                    renames[name] = name.replace(
                        f"model.llm.layers.{real_id}.",
                        f"model.llm.layers.{replica_id}.",
                        1,
                    )

        # Pattern coverage: the fed subset must exercise every tensor-name
        # pattern in the index (MTP chain excluded by design).
        def pattern(n: str) -> str:
            return re.sub(r"\.\d+\.", ".N.", n)

        all_patterns = {
            pattern(n) for n in cls.weight_map if not n.startswith("model.mtp.")
        }
        fed_patterns = {pattern(n) for n in renames}
        cls.uncovered_patterns = sorted(all_patterns - fed_patterns)

        os.environ.pop("INKLING_TORCH_MOE", None)  # exercise the real MoELayer
        cls.model, cls.text = cls._build_model(cls._tmpdir.name, quant_config)

        def weights_iter():
            by_file: dict[str, list[str]] = {}
            for real_name in renames:
                by_file.setdefault(cls.weight_map[real_name], []).append(real_name)
            for file, names in by_file.items():
                with safe_open(snapshot / file, framework="pt") as f:
                    for real_name in names:
                        yield renames[real_name], f.get_tensor(real_name)

        cls.loaded = cls.model.load_weights(weights_iter())
        cls._safe_open = safe_open
        cls._snapshot = snapshot

    @classmethod
    def _build_model(cls, ckpt: str, quant_config=None):
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.layers.moe import utils as moe_utils
        from tokenspeed.runtime.models.inkling import InklingForConditionalGeneration
        from tokenspeed.runtime.utils.env import global_server_args_dict
        from tokenspeed.runtime.utils.hf_transformers_utils import get_config

        # Same precomputed-topk backend the engine resolves for Inkling (the
        # standalone auto default plans its own routing, which Inkling rejects).
        moe_utils.MOE_BACKEND = moe_utils.MoeBackend.FLASHINFER_CUTLASS
        mapping = Mapping(rank=0, world_size=1)
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

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.cuda.empty_cache()
        cls._tmpdir.cleanup()

    def _ckpt_tensor(self, real_name: str) -> torch.Tensor:
        with self._safe_open(
            self._snapshot / self.weight_map[real_name], framework="pt"
        ) as f:
            return f.get_tensor(real_name)

    def _param(self, name: str) -> torch.Tensor:
        return dict(self.model.named_parameters())[name].data

    def _assert_eq(self, param: torch.Tensor, expected: torch.Tensor, what: str):
        expected = expected.to(device=param.device, dtype=param.dtype)
        self.assertEqual(param.shape, expected.shape, what)
        self.assertTrue(torch.equal(param, expected), f"{what} mismatch")

    def test_index_pattern_coverage(self):
        self.assertEqual(
            self.uncovered_patterns, [], "index patterns not exercised by the load"
        )

    def test_full_parameter_coverage(self):
        params = dict(self.model.named_parameters())
        missing = sorted(set(params) - self.loaded)
        self.assertEqual(missing, [], f"replica params never loaded: {missing}")
        self.assertNotIn("model.mtp.chain_norm.weight", self.loaded)

    def test_embed_and_lm_head(self):
        embed = self._ckpt_tensor("model.llm.embed.weight")
        self._assert_eq(
            self._param("model.embed_tokens.weight")[:4096], embed[:4096], "embed"
        )
        unembed = self._ckpt_tensor("model.llm.unembed.weight")
        self._assert_eq(self._param("lm_head.weight")[:4096], unembed[:4096], "lm_head")

    def test_qkvr_fusion_and_full_layer_kv_replication(self):
        text = self.text
        hd, served = text.head_dim, text.num_key_value_heads
        q_size = text.num_attention_heads * hd
        for replica_id, real_id in _LAYER_MAP.items():
            if replica_id < 2:
                continue  # attention checks on the two MoE layers
            qkvr = self._param(f"model.layers.{replica_id}.attn.qkvr.weight")
            p = f"model.llm.layers.{real_id}.attn"
            wq = self._ckpt_tensor(f"{p}.wq_du.weight")
            self._assert_eq(qkvr[:q_size], wq, f"layer{replica_id} q")
            wk = self._ckpt_tensor(f"{p}.wk_dv.weight")
            ckpt_heads = wk.shape[0] // hd
            factor = served // ckpt_heads
            for j in (0, 1, served // 2, served - 1):
                src = (j // factor) * hd
                self._assert_eq(
                    qkvr[q_size + j * hd : q_size + (j + 1) * hd],
                    wk[src : src + hd],
                    f"layer{replica_id} kv head {j} (factor {factor})",
                )
            # K sconv taps replicate identically.
            ks = self._ckpt_tensor(f"{p}.k_sconv.weight")
            k_sconv = self._param(f"model.layers.{replica_id}.attn.k_sconv.weight")
            j = served - 1
            self._assert_eq(
                k_sconv[j * hd : (j + 1) * hd],
                ks[(j // factor) * hd : (j // factor + 1) * hd],
                f"layer{replica_id} k_sconv head {j}",
            )

    def test_dense_mlp(self):
        w13 = self._ckpt_tensor("model.llm.layers.0.mlp.w13_dn.weight")
        self._assert_eq(
            self._param("model.layers.0.mlp.gate_up_proj.weight"),
            _deinterleave_rows(w13),
            "dense w13 deinterleave",
        )
        self._assert_eq(
            self._param("model.layers.0.mlp.down_proj.weight"),
            self._ckpt_tensor("model.llm.layers.0.mlp.w2_md.weight"),
            "dense w2",
        )
        self._assert_eq(
            self._param("model.layers.0.mlp.global_scale"),
            self._ckpt_tensor("model.llm.layers.0.mlp.global_scale"),
            "dense global_scale",
        )

    def test_moe_layer(self):
        replica_id, real_id = 3, _LAYER_MAP[3]
        p = f"model.llm.layers.{real_id}.mlp"
        o = f"model.layers.{replica_id}.mlp"
        gate_bias = self._param(f"{o}.gate.bias")
        self.assertEqual(gate_bias.dtype, torch.float32, "router bias stays fp32")
        self._assert_eq(gate_bias, self._ckpt_tensor(f"{p}.gate.bias"), "gate bias")
        self._assert_eq(
            self._param(f"{o}.gate.global_scale"),
            self._ckpt_tensor(f"{p}.gate.global_scale"),
            "gate global_scale",
        )
        self._assert_eq(
            self._param(f"{o}.gate.weight"),
            self._ckpt_tensor(f"{p}.gate.weight"),
            "gate weight",
        )
        w13 = self._ckpt_tensor(f"{p}.experts.w13_weight")
        w2 = self._ckpt_tensor(f"{p}.experts.w2_weight")
        pw13 = self._param(f"{o}.experts.w13_weight")
        pw2 = self._param(f"{o}.experts.w2_weight")
        for e in (0, 100, self.text.n_routed_experts - 1):
            self._assert_eq(pw13[e], _deinterleave_rows(w13[e]), f"expert {e} w13")
            self._assert_eq(pw2[e], w2[e], f"expert {e} w2")
        self._assert_eq(
            self._param(f"{o}.shared_experts.w13_weight"),
            _deinterleave_rows(
                self._ckpt_tensor(f"{p}.shared_experts.shared_w13_weight")
            ),
            "shared w13",
        )

    def test_fp4_quantized_experts(self):
        """FP4 snapshot only: packed U8 weights + F8 scales + scalar scales
        land on the quantized layer, while the excluded first-MoE-layer
        experts (covered by test_moe_layer) stay BF16."""
        if not self.is_fp4:
            self.skipTest("BF16 snapshot: no quantized experts")
        replica_id, real_id = 2, _LAYER_MAP[2]
        p = f"model.llm.layers.{real_id}.mlp.experts"
        o = f"model.layers.{replica_id}.mlp.experts"
        pw13 = self._param(f"{o}.w13_weight")
        self.assertEqual(pw13.dtype, torch.uint8)
        w13 = self._ckpt_tensor(f"{p}.w13_weight")
        for e in (0, 100, self.text.n_routed_experts - 1):
            self._assert_eq(pw13[e], _deinterleave_rows(w13[e]), f"fp4 w13 e{e}")
        scale = self._param(f"{o}.w13_weight_scale")
        self.assertEqual(scale.dtype, torch.float8_e4m3fn)
        ckpt_scale = self._ckpt_tensor(f"{p}.w13_weight.scale")
        self._assert_eq(
            scale[0].view(torch.uint8),
            _deinterleave_rows(ckpt_scale[0].view(torch.uint8)),
            "fp4 w13 block scale e0",
        )
        scale_2 = self._param(f"{o}.w13_weight_scale_2")
        ckpt_scale2 = self._ckpt_tensor(f"{p}.w13_weight.scale2")
        self._assert_eq(scale_2[:, 0], ckpt_scale2, "fp4 scale2 gate")
        self._assert_eq(scale_2[:, 1], ckpt_scale2, "fp4 scale2 up")
        input_scale = self._param(f"{o}.w13_input_scale")
        expected = self._ckpt_tensor(f"{p}.w13_weight.input_amax").float() / (
            448.0 * 6.0
        )
        self.assertTrue(
            torch.allclose(input_scale.cpu(), expected.expand_as(input_scale.cpu())),
            "fp4 input_scale conversion",
        )
        # The excluded layer's experts must have stayed unquantized.
        self.assertEqual(
            self._param("model.layers.3.mlp.experts.w13_weight").dtype,
            torch.bfloat16,
        )

    def test_vision_tower(self):
        self._assert_eq(
            self._param("visual.vision_encoder.layers.linear_0.weight"),
            self._ckpt_tensor("model.visual.layers.linear_0.weight"),
            "vision linear_0",
        )
        self._assert_eq(
            self._param("audio.encoder.weight"),
            self._ckpt_tensor("model.audio.encoder.weight"),
            "audio encoder",
        )


if __name__ == "__main__":
    unittest.main()
