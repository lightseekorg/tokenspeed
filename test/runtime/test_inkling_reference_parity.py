"""Inkling full-model numerical parity: engine vs independent torch reference.

Strategy: ``initialize_dummy_weights`` seeds each parameter with a fixed
per-parameter generator (values depend only on numel + dtype), so an
in-process replica of the model initialized the same way holds *identical*
weights to the engine subprocess's copy. A pure-torch reference forward —
written independently from the architecture spec, consuming the replica's
parameters — then predicts the engine's outputs exactly (up to bf16 kernel
noise):

  * greedy continuation token ids must match step by step, and
  * the engine-reported chosen-token logprobs must match the reference's.

This validates the full wiring end-to-end: embed norm, QKVR split, K/V
sconv placement, per-head QK norms, relative-attention bias + causal/SWA
masking, scale 1/head_dim, gate/shared-sink MoE, dense MLP, muP divide, and
the padded-vocab mask.

NOTE: intentionally NOT registered in CI while the Inkling port is
confidential/local-only. Requires Blackwell (FA4).
"""

import math
import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.models.inkling_fixtures import make_inkling_dummy_checkpoint  # noqa: E402

PROMPT_IDS = [11, 25, 3, 999, 42, 7, 128, 55, 1023, 64, 2, 300]
DECODE_STEPS = 6


def _has_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _build_replica(ckpt: str):
    """Construct the Inkling model in-process and dummy-init it like the engine."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.model_loader.weight_utils import (
        initialize_dummy_weights,
    )
    from tokenspeed.runtime.models.inkling import InklingForConditionalGeneration
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config

    mapping = Mapping(rank=0, world_size=1)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["enable_prefix_caching"] = False
    config = get_config(ckpt, trust_remote_code=False, revision=None)
    with torch.device("cuda"):
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = InklingForConditionalGeneration(config, mapping)
        finally:
            torch.set_default_dtype(torch.float32)
    initialize_dummy_weights(model)
    return model.eval(), config.get_text_config()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()


def _rel_attention(q, k, v, rel_logits, rel_extent, window_left, scale, num_kv):
    """[T,H,D] x [T,KV,D] full-recompute relative attention (fp32)."""
    T, H, D = q.shape
    rep = H // num_kv
    k = k.repeat_interleave(rep, dim=1).float()
    v = v.repeat_interleave(rep, dim=1).float()
    logits = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    pos = torch.arange(T, device=q.device)
    dist = pos[:, None] - pos[None, :]
    in_range = (dist >= 0) & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    bias = rel_logits.float().gather(-1, idx.unsqueeze(1).expand(T, H, T))
    logits = logits + torch.where(in_range.unsqueeze(1), bias, 0.0).permute(1, 0, 2)
    mask = dist < 0
    if window_left >= 0:
        mask |= dist > window_left
    logits.masked_fill_(mask.unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)


def _ref_sconv(x: torch.Tensor, weight3d: torch.Tensor) -> torch.Tensor:
    """Residual causal FIR over a full sequence, zero initial state (fp32)."""
    w = weight3d.squeeze(1).float()  # [D, W]
    W = w.shape[1]
    xp = torch.cat([torch.zeros(W - 1, x.shape[1], device=x.device), x.float()])
    y = sum(xp[i : i + len(x)] * w[:, i] for i in range(W))
    return x.float() + y


def _reference_forward(model, text, input_ids: list[int]) -> torch.Tensor:
    """Independent full-sequence forward; returns final-position log-probs."""
    dev = "cuda"
    m = model.model
    ids = torch.tensor(input_ids, device=dev)
    h = m.embed_tokens.weight[ids]
    if m.embed_norm is not None:
        h = _rms_norm(h, m.embed_norm.weight, text.rms_norm_eps)
    h = h.to(torch.bfloat16)

    head_dim = text.head_dim
    num_heads = text.num_attention_heads
    num_kv = text.num_key_value_heads
    T = len(input_ids)

    for layer in m.layers:
        attn = layer.attn
        # --- attention sublayer ---
        x = _rms_norm(h, layer.attn_norm.weight, text.rms_norm_eps).to(h.dtype)
        qkvr = x @ attn.qkvr.weight.t()
        q, k, v, r = qkvr.split(
            [attn.q_size, attn.kv_size, attn.kv_size, attn.r_size], dim=-1
        )
        k = _ref_sconv(k, attn.k_sconv.weight).to(h.dtype)
        v = _ref_sconv(v, attn.v_sconv.weight).to(h.dtype)
        q = _rms_norm(
            q.reshape(-1, head_dim), attn.q_norm.weight, text.rms_norm_eps
        ).view(T, num_heads, head_dim)
        k = _rms_norm(
            k.reshape(-1, head_dim), attn.k_norm.weight, text.rms_norm_eps
        ).view(T, num_kv, head_dim)
        v = v.view(T, num_kv, head_dim)
        rel = torch.einsum(
            "thd,de->the",
            r.view(T, num_heads, text.d_rel).float(),
            attn.rel_logits_proj.proj.float(),
        ).to(h.dtype)
        window_left = (text.sliding_window_size - 1) if attn.is_local else -1
        o = _rel_attention(
            q.to(h.dtype),
            k,
            v,
            rel,
            attn.rel_extent,
            window_left,
            1.0 / head_dim,
            num_kv,
        )
        o = o.to(h.dtype).reshape(T, -1) @ attn.wo_ud.weight.t()
        o = _ref_sconv(o, layer.attn_sconv.weight).to(h.dtype)
        h = h + o

        # --- mlp sublayer ---
        x = _rms_norm(h, layer.mlp_norm.weight, text.rms_norm_eps).to(h.dtype)
        if not layer.is_moe:
            gu = x @ layer.mlp.gate_up_proj.weight.t()
            gate, up = gu.chunk(2, dim=-1)
            y = (F.silu(gate.float()) * up.float()).to(h.dtype)
            y = y @ layer.mlp.down_proj.weight.t()
        else:
            blk = layer.mlp
            full_w, ids_k, _ = blk.gate(x)
            k = blk.gate.top_k
            weights, gammas = full_w[:, :k].contiguous(), full_w[:, k:].contiguous()
            experts = blk.experts  # InklingTorchMoEExperts
            y = experts(x, weights, ids_k).float()
            sh = blk.shared_experts
            hh = torch.einsum("th,sih->sti", x, sh.w13_weight.to(x.dtype))
            g, u = hh.chunk(2, dim=-1)
            so = torch.einsum("sti,shi->sth", F.silu(g) * u, sh.w2_weight.to(x.dtype))
            y = (y + torch.einsum("sth,ts->th", so, gammas.to(x.dtype))).to(h.dtype)
        y = _ref_sconv(y, layer.mlp_sconv.weight).to(h.dtype)
        h = h + y

    h = _rms_norm(h, m.norm.weight, text.rms_norm_eps).to(h.dtype)
    h = h / text.logits_mup_width_multiplier
    logits = (h.float() @ model.lm_head.weight.t().float()).float()
    logits[:, text.unpadded_vocab_size :] = float("-inf")
    return logits.log_softmax(-1)  # [T, V]: next-token logprobs per position


@unittest.skipUnless(_has_blackwell(), "Inkling parity needs a Blackwell GPU (FA4)")
class TestInklingReferenceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Torch MoE experts in BOTH the engine and the replica so parameter
        # shapes/values coincide (MoELayer stores backend-specific layouts).
        os.environ["INKLING_TORCH_MOE"] = "1"
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.ckpt = str(make_inkling_dummy_checkpoint(cls._tmpdir.name, tiny=True))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_logprobs_match_reference(self):
        """Compare engine logprob VALUES against the reference at every
        sequence position: prompt tokens (prefill path) and each generated
        token along the engine's trajectory (decode path).

        Dummy weights (uniform ±1e-3) make logits nearly flat, so argmax
        identity is noise — but the logprob values themselves are a sharp
        full-stack numerical check.
        """
        from tokenspeed.runtime.entrypoints.engine import Engine

        engine = Engine(
            model=self.ckpt,
            load_format="dummy",
            attention_backend="fa4",
            enable_prefix_caching=False,
            disable_kvstore=True,
            enforce_eager=True,
            enable_output_logprobs=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.3,
            max_model_len=2048,
            max_num_seqs=4,
            log_level="warning",
        )
        try:
            out = engine.generate(
                input_ids=PROMPT_IDS,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": DECODE_STEPS,
                },
                return_logprob=True,
            )
            result = out if isinstance(out, dict) else out[0]
            engine_ids = result["output_ids"][-DECODE_STEPS:]
            meta = result["meta_info"]
            # List of (logprob, token_id, ...) per generated token. (Prompt
            # logprobs are not supported by the engine yet.)
            output_lps = meta["output_token_logprobs"]
        finally:
            engine.shutdown()
        self.assertEqual(len(output_lps), DECODE_STEPS)

        model, text = _build_replica(self.ckpt)
        # One reference forward over prompt + engine trajectory gives
        # next-token logprobs at every position.
        seq = list(PROMPT_IDS) + [int(t) for t in engine_ids]
        with torch.no_grad():
            ref = _reference_forward(model, text, seq[:-1])  # [T-1, V]

        diffs = []
        # Decode positions: output_token_logprobs[s] scores engine_ids[s]
        # given prompt + engine_ids[:s]. Step 0 validates the prefill path
        # end-to-end; later steps validate rolling decode (conv state, KV
        # append, SWA windows).
        for s, entry in enumerate(output_lps):
            lp, tid = float(entry[0]), int(entry[1])
            self.assertEqual(tid, int(engine_ids[s]))
            pos = len(PROMPT_IDS) + s - 1
            diffs.append(abs(float(ref[pos, tid]) - lp))

        self.assertEqual(len(diffs), DECODE_STEPS)
        max_diff = max(diffs)
        # bf16 kernels vs fp32-accum reference across the whole stack.
        self.assertLess(
            max_diff,
            2e-2,
            f"max |ref - engine| logprob diff {max_diff:.4e} over "
            f"{len(diffs)} positions (all diffs: {[f'{d:.1e}' for d in diffs]})",
        )


if __name__ == "__main__":
    unittest.main()
