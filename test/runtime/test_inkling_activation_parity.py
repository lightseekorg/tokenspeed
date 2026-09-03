"""Inkling per-layer ACTIVATION parity: real kernel stack vs torch reference.

Runs the Inkling model in-process through the real backends (FA4 score_mod
attention over the paged KV cache, ops/conv sconv kernels, Triton
silu_and_mul) with hand-built forward metadata, and compares the hidden
states after EVERY decoder layer — prefill and rolling decode steps —
against the independent pure-torch reference on identical dummy weights.

Unlike the end-to-end logprob parity test, this localizes any numerical
divergence to the exact layer and phase where it first appears, and leaves
no room for compensating errors.

Runs on the released config (layer-truncated, experts shrunk; see
``inkling_fixtures``) with dummy weights. Requires Blackwell (FA4).
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci
from runtime.test_inkling_reference_parity import (
    _build_replica,
    _has_blackwell,
    _ref_sconv,
    _rel_attention,
    _rms_norm,
)

register_cuda_ci(
    est_time=90,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*", "h100-*"],
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
    T = len(input_ids)

    for li, layer in enumerate(m.layers):
        attn = layer.attn
        # Per-layer native KV width (hetero serving).
        num_kv = attn.kv_size // head_dim
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


class _ConvCheckpointPool:
    """KV pool view adding the paged sconv checkpoint buffers the backend
    now requires (paged conv is mandatory; there is no rolling fallback)."""

    def __init__(self, kv_pool, *, layer_kv_widths, num_pages, rows, hidden, device):
        self._kv_pool = kv_pool
        # Per-layer widths: kvconv fields track each layer's own K/V conv
        # width (hetero full/SWA head counts), exactly like the real recipe.
        self._kvconv = [
            tuple(
                torch.zeros(num_pages, rows, width, dtype=torch.bfloat16, device=device)
                for _ in range(2)
            )
            for width in layer_kv_widths
        ]
        self._hidden = {
            component: torch.zeros(
                len(layer_kv_widths),
                num_pages,
                rows,
                hidden,
                dtype=torch.bfloat16,
                device=device,
            )
            for component in ("attnconv", "mlpconv")
        }

    def kvconv_checkpoint_buffers(self, layer_id):
        return self._kvconv[layer_id]

    def hiddenconv_checkpoint_buffer(self, layer_id, component):
        return self._hidden[component][layer_id]

    def __getattr__(self, name):
        return getattr(self._kv_pool, name)


class _Harness:
    """Real-backend single-request driver for the in-process model."""

    def __init__(self, model, text, device="cuda"):
        from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim
        from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
            CacheGroupGeometry,
        )
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvStatePool,
        )
        from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend
        from tokenspeed.runtime.layers.attention.backends.router import (
            CacheGroupRouter,
        )
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
        )

        self.model = model
        self.text = text
        self.device = device
        spec = MHAConfig(
            backend_name="fa4",
            num_attention_heads=text.num_attention_heads,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            attn_tp_size=1,
        )
        config = AttnConfig(
            device=device,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            prefix_granularity=PAGE_SIZE,
            kernel_page_size=PAGE_SIZE,
            context_len=1024,
            max_bs=4,
            max_graph_bs=4,
            kv_cache_quant_method="none",
            components=(spec,),
        )
        # Production composition: the Inkling wrapper sits over a
        # CacheGroupRouter, one MHA leaf per attention cache group (the
        # layers' group_ids are the cache_layer_types labels; sliding layers
        # split into sub-groups). Write locations are router-owned: the layer
        # call carries none, the router derives them from block_tables.
        self.attn_groups = tuple(dict.fromkeys(text.cache_layer_types))
        leaves = {
            gid: MHAAttnBackend(config, spec, kernel_page_size=PAGE_SIZE)
            for gid in self.attn_groups
        }
        inner = CacheGroupRouter(None, is_draft=False, spec_num_tokens=1, device=device)
        inner.bind(
            CacheGroupGeometry(
                granularities={gid: PAGE_SIZE for gid in self.attn_groups},
                families={gid: "history" for gid in self.attn_groups},
                full_history_group_id=(
                    "full_attention"
                    if "full_attention" in self.attn_groups
                    else self.attn_groups[0]
                ),
            ),
            leaves,
        )
        from cache_pool_test_utils import make_mha_memory_plan, make_pool

        # One arena, one view over it: the pool owns no memory or geometry.
        _arena, self.kv_pool = make_pool(
            MHATokenToKVPool,
            make_mha_memory_plan(
                size=1024,
                prefix_granularity=PAGE_SIZE,
                layer_num=text.num_hidden_layers,
                kv_heads=text.num_key_value_heads,
                head_dim=text.head_dim,
                dtype=torch.bfloat16,
            ),
            device=device,
            dtype=torch.bfloat16,
            head_num=text.num_key_value_heads,
            head_dim=text.head_dim,
            layer_num=text.num_hidden_layers,
            rank=0,
        )
        conv_pool = InklingConvStatePool(
            num_layers=text.num_hidden_layers,
            num_slots=6,
            conv_dim=inkling_conv_total_dim(text, 1),
            kernel_size=text.sconv_kernel_size,
            # Non-spec ring: (W-1) + K(1) = W.
            ring_size=text.sconv_kernel_size,
            dtype=torch.bfloat16,
            device=device,
        )
        conv_block_tokens = 128
        num_conv_pages = 1024 // conv_block_tokens
        conv_columns = {
            "block_tokens": conv_block_tokens,
            "conv_group_of_layer": ("kvconv",) * text.num_hidden_layers,
            "hidden_group_of_layer": ("hiddenconv",) * text.num_hidden_layers,
            "group_block_tokens": {
                "kvconv": conv_block_tokens,
                "hiddenconv": conv_block_tokens,
            },
        }
        self.backend = InklingAttnBackend(inner, conv_pool, conv_columns=conv_columns)
        self.pool_view = _ConvCheckpointPool(
            self.kv_pool,
            layer_kv_widths=[
                (
                    text.swa_num_key_value_heads
                    if i in text.local_layer_ids
                    else text.ckpt_num_key_value_heads
                )
                * text.head_dim
                for i in range(text.num_hidden_layers)
            ],
            num_pages=num_conv_pages + 1,
            rows=text.sconv_kernel_size - 1,
            hidden=text.hidden_size,
            device=device,
        )
        # Dense conv page map: boundary j*128 -> page j+1 (page 0 = hole).
        conv_table = torch.arange(
            1, num_conv_pages + 1, dtype=torch.int32, device=device
        ).unsqueeze(0)
        self.conv_tables = {"kvconv": conv_table, "hiddenconv": conv_table}
        # Request slot REQ_SLOT owns pages [1, 2, ...] -> token locs 64+.
        max_pages = 1024 // PAGE_SIZE
        self.page_table = torch.zeros(8, max_pages, dtype=torch.int32, device=device)
        for p in range(max_pages - 1):
            self.page_table[REQ_SLOT, p] = p + 1
        # Router tables are batch-row-major ([bs, cols]); every attention
        # group shares this one request's page row over the single pool.
        attn_table = self.page_table[REQ_SLOT : REQ_SLOT + 1]
        self.block_tables = {
            **self.conv_tables,
            **{gid: attn_table for gid in self.attn_groups},
        }
        self.seq_len = 0
        # Unified decode path: decode metadata is refreshed into persistent
        # buffers allocated here (production allocates them unconditionally
        # at ForwardStepRunner construction, enforce-eager included). The
        # router is hand-bound above, so bind the pool to its leaves the way
        # set_cache_pool would.
        for leaf in leaves.values():
            leaf.set_cache_pool(self.kv_pool)
        self.backend.init_cuda_graph_state(max_bs=4)

    def _ctx(self, mode):
        return SimpleNamespace(
            attn_backend=self.backend,
            token_to_kv_pool=self.pool_view,
            forward_mode=mode,
            bs=1,
        )

    def _token_locs(self, start, n):
        # Page ids start at 1: token location = page_id * page_size + offset.
        pos = torch.arange(start, start + n, device=self.device)
        return (pos // PAGE_SIZE + 1) * PAGE_SIZE + pos % PAGE_SIZE

    def _check_write_locations(self, mode, start, n):
        """Pin the router-derived KV write slots to the page map above: every
        layer's group shares the request's page row, so every layer writes
        the same token locations the old explicit out_cache_loc named."""
        expected = self._token_locs(start, n)
        for layer in self.model.model.layers:
            paged = layer.attn.attn  # InklingAttention -> its PagedAttention
            got = self.backend.write_locations(paged, mode).long()
            assert torch.equal(got, expected), (
                f"layer {paged.layer_id} ({paged.group_id!r}) {mode}: "
                f"write locs {got.tolist()} != {expected.tolist()}"
            )

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
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=seq_lens,
            extend_seq_lens_cpu=torch.tensor([T]),
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=dev),
            extend_prefix_lens_cpu=torch.zeros(1, dtype=torch.int32),
            extend_with_prefix=False,
            block_tables=self.block_tables,
        )
        self.seq_len = T
        self._check_write_locations(ForwardMode.EXTEND, 0, T)
        ids = torch.tensor(input_ids, device=dev)
        positions = torch.arange(T, device=dev)
        return self._layer_states(ids, positions, ForwardMode.EXTEND)

    def decode(self, token_id):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        dev = self.device
        self.seq_len += 1
        req_pool_indices = torch.tensor([REQ_SLOT], dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([self.seq_len], dtype=torch.int32, device=dev)
        self.backend.refresh_decode_metadata(
            1,
            1,
            req_pool_indices,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self.block_tables,
        )
        self._check_write_locations(ForwardMode.DECODE, self.seq_len - 1, 1)
        ids = torch.tensor([token_id], device=dev)
        positions = torch.tensor([self.seq_len - 1], device=dev)
        return self._layer_states(ids, positions, ForwardMode.DECODE)

    def _layer_states(self, ids, positions, mode):
        """Run embed + layers through the real stack, yielding per-layer h
        (fused add-norm convention: the comparable value is output+residual).
        """
        m = self.model.model
        ctx = self._ctx(mode)
        states = []
        h = m.embed_tokens(ids)
        if m.embed_norm is not None:
            h = m.embed_norm(h)
        states.append(("embed", h.clone()))
        tau = None  # exactly 1.0 below log_scaling_n_floor; off on both sides
        residual = None
        for li, layer in enumerate(m.layers):
            h, residual = layer(h, residual, ctx, log_scaling_tau=tau)
            states.append((f"layer{li}", (h + residual).clone()))
        h, _ = m.norm(h, residual)
        states.append(("final_norm", h.clone()))
        return states


@unittest.skipUnless(_has_blackwell(), "Inkling parity needs a Blackwell GPU (FA4)")
class TestInklingActivationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["INKLING_TORCH_MOE"] = "1"  # experts as plain parameters
        cls.model, cls.text = _build_replica()

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
