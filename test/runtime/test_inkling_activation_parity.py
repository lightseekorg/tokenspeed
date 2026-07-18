"""Inkling per-layer ACTIVATION parity: real kernel stack vs torch reference.

Runs the Inkling model in-process through the real backends (FA4 score_mod
attention over the paged KV cache, ops/conv sconv kernels, Triton
silu_and_mul) with hand-built forward metadata, and compares the hidden
states after EVERY decoder layer — prefill and rolling decode steps —
against the independent pure-torch reference on identical dummy weights.

Unlike the end-to-end logprob parity test, this localizes any numerical
divergence to the exact layer and phase where it first appears, and leaves
no room for compensating errors.

NOTE: intentionally NOT registered in CI while the Inkling port is
confidential/local-only. Requires Blackwell (FA4).
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.models.inkling_fixtures import make_inkling_dummy_checkpoint  # noqa: E402
from runtime.test_inkling_reference_parity import (  # noqa: E402
    _build_replica,
    _has_blackwell,
    _ref_sconv,
    _rel_attention,
    _rms_norm,
)

PROMPT_IDS = [11, 25, 3, 999, 42, 7, 128, 55, 1023, 64, 2, 300, 17, 500]
DECODE_TOKENS = [123, 45, 678]
PAGE_SIZE = 64
REQ_SLOT = 1  # 1-based request pool slot (row 0 reserved), page id 1
# bf16 kernels vs fp32-accum reference; tiny dummy weights keep activations
# O(1e-3) so absolute tolerance is tight.
TOL = 5e-3


def _reference_layer_states(model, text, input_ids):
    """Yield hidden states after each decoder layer (plus embed and final)."""
    m = model.model
    dev = "cuda"
    ids = torch.tensor(input_ids, device=dev)
    h = m.embed_tokens.weight[ids]
    if m.embed_norm is not None:
        h = _rms_norm(h, m.embed_norm.weight, text.rms_norm_eps)
    h = h.to(torch.bfloat16)
    yield "embed", h

    head_dim = text.head_dim
    num_heads = text.num_attention_heads
    num_kv = text.num_key_value_heads
    T = len(input_ids)

    for li, layer in enumerate(m.layers):
        attn = layer.attn
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

        x = _rms_norm(h, layer.mlp_norm.weight, text.rms_norm_eps).to(h.dtype)
        if not layer.is_moe:
            gu = x @ layer.mlp.gate_up_proj.weight.t()
            gate, up = gu.chunk(2, dim=-1)
            y = (torch.nn.functional.silu(gate.float()) * up.float()).to(h.dtype)
            y = y @ layer.mlp.down_proj.weight.t()
        else:
            blk = layer.mlp
            full_w, ids_k, _ = blk.gate(x)
            k = blk.gate.top_k
            weights, gammas = full_w[:, :k].contiguous(), full_w[:, k:].contiguous()
            y = blk.experts(x, weights, ids_k).float()
            sh = blk.shared_experts
            hh = torch.einsum("th,sih->sti", x, sh.w13_weight.to(x.dtype))
            g, u = hh.chunk(2, dim=-1)
            so = torch.einsum(
                "sti,shi->sth",
                torch.nn.functional.silu(g) * u,
                sh.w2_weight.to(x.dtype),
            )
            y = (y + torch.einsum("sth,ts->th", so, gammas.to(x.dtype))).to(h.dtype)
        y = _ref_sconv(y, layer.mlp_sconv.weight).to(h.dtype)
        h = h + y
        yield f"layer{li}", h

    h = _rms_norm(h, m.norm.weight, text.rms_norm_eps).to(h.dtype)
    yield "final_norm", h


class _Harness:
    """Real-backend single-request driver for the in-process model."""

    def __init__(self, model, text, device="cuda"):
        from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvStatePool,
        )
        from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
        )

        self.model = model
        self.text = text
        self.device = device
        config = MHAConfig(
            device=device,
            backend_name="fa4",
            num_attention_heads=text.num_attention_heads,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            page_size=PAGE_SIZE,
            context_len=1024,
            max_bs=4,
            max_graph_bs=4,
            kv_cache_quant_method="none",
        )
        inner = MHAAttnBackend(config)
        self.kv_pool = MHATokenToKVPool(
            size=1024,
            dtype=torch.bfloat16,
            head_num=text.num_key_value_heads,
            head_dim=text.head_dim,
            layer_num=text.num_hidden_layers,
            device=device,
            enable_memory_saver=False,
            max_batch_size=4,
            max_context_len=1024,
            page_size=PAGE_SIZE,
            rank=0,
        )
        conv_pool = InklingConvStatePool(
            num_layers=text.num_hidden_layers,
            num_slots=6,
            conv_dim=inkling_conv_total_dim(text, 1),
            kernel_size=text.sconv_kernel_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.backend = InklingAttnBackend(inner, conv_pool)
        # Request slot REQ_SLOT owns pages [1, 2, ...] -> token locs 64+.
        max_pages = 1024 // PAGE_SIZE
        self.req_to_page = torch.zeros(8, max_pages, dtype=torch.int32, device=device)
        for p in range(max_pages - 1):
            self.req_to_page[REQ_SLOT, p] = p + 1
        self.seq_len = 0

    def _ctx(self, mode):
        return SimpleNamespace(
            attn_backend=self.backend,
            token_to_kv_pool=self.kv_pool,
            forward_mode=mode,
            bs=1,
        )

    def _token_locs(self, start, n):
        # Page ids start at 1: token location = page_id * page_size + offset.
        pos = torch.arange(start, start + n, device=self.device)
        return (pos // PAGE_SIZE + 1) * PAGE_SIZE + pos % PAGE_SIZE

    def prefill(self, input_ids):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        T = len(input_ids)
        dev = self.device
        req_pool_indices = torch.tensor([REQ_SLOT], dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([T], dtype=torch.int32, device=dev)
        self.backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_page=self.req_to_page,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=seq_lens,
            extend_seq_lens_cpu=torch.tensor([T]),
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=dev),
            extend_prefix_lens_cpu=torch.zeros(1, dtype=torch.int32),
        )
        self.seq_len = T
        out_cache_loc = self._token_locs(0, T)
        ids = torch.tensor(input_ids, device=dev)
        positions = torch.arange(T, device=dev)
        return self._layer_states(ids, positions, ForwardMode.EXTEND, out_cache_loc)

    def decode(self, token_id):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        dev = self.device
        self.seq_len += 1
        req_pool_indices = torch.tensor([REQ_SLOT], dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([self.seq_len], dtype=torch.int32, device=dev)
        self.backend.init_forward_metadata(
            bs=1,
            num_extends=0,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_page=self.req_to_page,
            forward_mode=ForwardMode.DECODE,
        )
        out_cache_loc = self._token_locs(self.seq_len - 1, 1)
        ids = torch.tensor([token_id], device=dev)
        positions = torch.tensor([self.seq_len - 1], device=dev)
        return self._layer_states(ids, positions, ForwardMode.DECODE, out_cache_loc)

    def _layer_states(self, ids, positions, mode, out_cache_loc):
        """Run embed + layers through the real stack, yielding per-layer h."""
        m = self.model.model
        ctx = self._ctx(mode)
        states = []
        h = m.embed_tokens(ids)
        if m.embed_norm is not None:
            h = m.embed_norm(h)
        states.append(("embed", h.clone()))
        tau = None  # log_scaling_n_floor is null in the fixture config
        for li, layer in enumerate(m.layers):
            h = layer(h, ctx, out_cache_loc, log_scaling_tau=tau)
            states.append((f"layer{li}", h.clone()))
        h = m.norm(h)
        states.append(("final_norm", h.clone()))
        return states


@unittest.skipUnless(_has_blackwell(), "Inkling parity needs a Blackwell GPU (FA4)")
class TestInklingActivationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["INKLING_TORCH_MOE"] = "1"  # experts as plain parameters
        cls._tmpdir = tempfile.TemporaryDirectory()
        ckpt = str(make_inkling_dummy_checkpoint(cls._tmpdir.name, tiny=True))
        cls.model, cls.text = _build_replica(ckpt)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _compare(self, phase, got_states, ref_states, positions):
        report = []
        for (name_g, got), (name_r, ref) in zip(got_states, ref_states):
            self.assertEqual(name_g, name_r)
            diff = (got.float() - ref.float()[positions]).abs().max().item()
            scale = ref.float().abs().max().item()
            report.append(f"{phase}/{name_g}: max_diff={diff:.2e} (scale {scale:.2e})")
            self.assertLess(
                diff, TOL, f"{phase}/{name_g} diverged:\n" + "\n".join(report)
            )
        return report

    def test_layerwise_prefill_and_decode(self):
        harness = _Harness(self.model, self.text)
        report = []
        with torch.no_grad():
            # ---- prefill: compare every position, every layer ----
            got = harness.prefill(PROMPT_IDS)
            ref = list(_reference_layer_states(self.model, self.text, PROMPT_IDS))
            report += self._compare("prefill", got, ref, slice(None))
            # ---- rolling decode: compare the new position, every layer ----
            seq = list(PROMPT_IDS)
            for step, tok in enumerate(DECODE_TOKENS):
                seq.append(tok)
                got = harness.decode(tok)
                ref = list(_reference_layer_states(self.model, self.text, seq))
                report += self._compare(f"decode{step}", got, ref, slice(-1, None))
        # Print the full per-layer report on success for eyeballing.
        print("\n".join(report))


if __name__ == "__main__":
    unittest.main()
