# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Runner that drives the GPT-OSS decode megakernel from the engine.

Everything bound to the kernel lives at a stable address. HIP graph capture
records addresses, not objects, so a tensor reallocated per step would bake a
stale pointer into the graph and read freed memory on replay -- the runner
therefore copies INTO persistent buffers rather than binding fresh tensors.

Cache-group ordering is the other trap. ``att_gids`` is ``sorted()``, so in the
engine gid 0 is ``full_attention`` and gid 1 is ``sliding_attention`` -- the
opposite of the standalone dev harnesses. The layer->gid map is therefore built
from the pool's own per-layer labels against the backend's ordering, never
hard-coded.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.models.gpt_oss_megakernel.weights import build_kv_tables
from tokenspeed.runtime.models.gpt_oss_megakernel_mfma.kernel import NBAR, mk_model_opt
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

H = 2880
I = 2880
NH = 64
NKV = 8
DH = 64
QKV = NH * DH + 2 * NKV * DH
E = 128
TOPK = 4
FULL_WINDOW = 1 << 30


class GptOssMegakernelRunner:
    def __init__(
        self,
        attn_w,
        moe_w,
        config,
        max_bs: int = 8,
        nwg: int = 256,
        use_dot: bool = True,
    ):
        self.aw, self.mw, self.cfg = attn_w, moe_w, config
        self.max_bs, self.nwg, self.use_dot = max_bs, nwg, use_dot
        self.n_layers = int(config.num_hidden_layers)
        dev = attn_w.wqkv.device
        self.dev = dev

        props = torch.cuda.get_device_properties(dev)
        if not props.gcnArchName.startswith("gfx950"):
            raise RuntimeError(f"megakernel requires gfx950, got {props.gcnArchName}")
        if nwg > props.multi_processor_count:
            raise RuntimeError(
                f"grid {nwg} exceeds {props.multi_processor_count} CUs; a persistent "
                "kernel with grid-wide barriers would deadlock"
            )

        # per-layer sliding window; full-attention layers use a value larger than
        # any position so `lo = max(0, pos - win + 1)` clamps to 0 with no branch
        lt = list(getattr(config, "layer_types", []))
        win = [
            (
                int(getattr(config, "sliding_window", 128))
                if t == "sliding_attention"
                else FULL_WINDOW
            )
            for t in lt
        ]
        self.win = torch.tensor(win, dtype=torch.int32, device=dev)

        b = max_bs
        f32, z = torch.float32, lambda n: torch.zeros(
            n, dtype=torch.float32, device=dev
        )
        self.xnorm, self.qkv = z(b * H), z(b * QKV)
        self.attn, self.resid = z(b * NH * DH), z(b * H)
        self.rlog, self.gu = z(b * E), z(b * TOPK * 2 * I)
        self.act, self.ynorm = z(b * TOPK * I), z(b * H)
        self.tki = torch.zeros(b * TOPK, dtype=torch.int32, device=dev)
        self.tkw = z(b * TOPK)
        self.ynq = torch.zeros(b * H, dtype=torch.float8_e4m3fn, device=dev)
        self.yns = torch.zeros(b * (H // 32), dtype=torch.uint8, device=dev)
        self.actq = torch.zeros(b * TOPK * I, dtype=torch.float8_e4m3fn, device=dev)
        self.acts = torch.zeros(b * TOPK * (I // 32), dtype=torch.uint8, device=dev)

        # persistent, capture-safe inputs/outputs
        self.hs_buf, self.out_buf = z(b * H), z(b * H)
        self.posn_buf = torch.zeros(b, dtype=torch.int32, device=dev)
        # allocated before any capture => not from the graph's private pool
        self.bar = torch.zeros(1, dtype=torch.int32, device=dev)

        self.final_norm = attn_w.fnorm
        self._kv = None  # bound lazily on the first decode
        self._bound_sig = None
        logger.info(
            "megakernel runner ready: %d layers, max_bs=%d, grid=%d, "
            "%d barriers/layer, dot_scaled=%s",
            self.n_layers,
            max_bs,
            nwg,
            NBAR,
            use_dot,
        )

    # ---------------------------------------------------------------- binding

    def _bind_backend(self, ctx) -> bool:
        """Bind the pool offsets and the backend's persistent group buffers.

        All of these are stable for the life of the backend, so binding once is
        both correct and required for graph capture.
        """
        be, pool = ctx.attn_backend, ctx.token_to_kv_pool
        tab = getattr(be, "_group_tables_stack", None)
        loc = getattr(be, "_group_locs_stack", None)
        ps = getattr(be, "_group_page_sizes_tensor", None)
        gids = list(getattr(be, "_graph_group_ids", []) or [])
        if tab is None or loc is None or ps is None or not gids:
            logger.warning(
                "megakernel: backend %s does not expose cache-group "
                "buffers; falling back to the normal path",
                type(be).__name__,
            )
            return False

        kv = build_kv_tables(pool)
        if kv is None:
            return False

        # Layer -> index into the group stacks. Group ids ARE the layer-type
        # labels ("sliding_attention" / "full_attention"), and `att_gids` is
        # sorted(), so the ordering is alphabetical -- full_attention first.
        # Never hard-code that: resolve each layer's label against `gids`.
        labels = getattr(pool, "_layer_types", None) or tuple(
            getattr(self.cfg, "layer_types", ())
        )
        labels = tuple(labels)
        try:
            lgid = [gids.index(lbl) for lbl in labels]
        except ValueError as exc:
            logger.warning(
                "megakernel: layer label %s not in backend groups %s (%s)",
                labels[:2],
                gids,
                exc,
            )
            return False
        if len(lgid) != self.n_layers:
            logger.warning(
                "megakernel: %d layer labels for %d layers", len(lgid), self.n_layers
            )
            return False

        self.koff, self.voff = kv["kptr"], kv["vptr"]  # absolute addresses
        self.kbuf = kv["k_base_view"].view(-1)
        self.vbuf = kv["v_base_view"].view(-1)
        self.lgid = torch.tensor(lgid, dtype=torch.int32, device=self.dev)
        self.gids = gids
        self.ps = ps

        # The backend's _group_tables_stack / _group_locs_stack are CAPTURE
        # storage: `_fill_group_graph_buffers` populates them at graph replay
        # and they are all-zero otherwise. Binding them directly makes every
        # position read page 0. So use them only for SHAPES, and own persistent
        # buffers that `run` refills from the live decode metadata each step --
        # which is both correct eager and capture-safe (the copies become graph
        # nodes reading the metadata's recorded addresses).
        g, _, wmax = tab.shape
        self.ptab = torch.zeros(
            (g, self.max_bs, wmax), dtype=torch.int32, device=self.dev
        )
        self.wloc = torch.zeros((g, self.max_bs), dtype=torch.int32, device=self.dev)
        self.TAB_G = self.max_bs * wmax
        self.TAB_B = wmax
        self.WLOC_G = self.max_bs
        self._kv = True
        logger.info(
            "megakernel: bound backend -- groups=%s, layer0 gid=%d (%s), "
            "tables%s locs%s",
            gids,
            lgid[0],
            labels[0],
            tuple(tab.shape),
            tuple(loc.shape),
        )
        return True

    # ---------------------------------------------------------------- gating

    def can_run(self, ctx, input_embeds, layers_to_capture) -> bool:
        """Pure host-side gate, evaluated at capture time.

        cuda_graph_wrapper captures one graph per batch size, so this branch is
        baked into each graph -- there is no runtime branching inside a graph.
        """
        if input_embeds is not None:
            return False
        if layers_to_capture:  # Eagle3 aux capture
            return False
        if getattr(ctx, "spec_info", None) is not None:
            return False
        fm = getattr(ctx, "forward_mode", None)
        if fm is None or not fm.is_decode():
            return False
        if int(getattr(ctx, "bs", 0) or 0) > self.max_bs:
            return False
        if self._kv is None and not self._bind_backend(ctx):
            self._kv = False
        return bool(self._kv)

    # ---------------------------------------------------------------- launch

    def _refill_paging(self, ctx, out_cache_loc, bs: int) -> bool:
        """Copy this step's per-group page table and write slots into our own
        persistent buffers, from the same metadata the normal path reads.

        Returns False if the metadata is not in the expected shape, so the
        caller can fall through rather than serve wrong tokens.
        """
        md = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if md is None:
            return False
        for i, gid in enumerate(self.gids):
            tabs = getattr(md, "page_tables", None)
            pt = tabs.get(gid) if isinstance(tabs, dict) else None
            if pt is None:
                pt = getattr(md, "page_table", None)
            if pt is None:
                return False
            w = min(pt.shape[1], self.ptab.shape[2])
            self.ptab[i, :bs, :w].copy_(pt[:bs, :w])

            locs = getattr(md, "out_cache_locs", None)
            lc = locs.get(gid) if isinstance(locs, dict) else None
            if lc is None:
                lc = out_cache_loc
            if lc is None:
                return False
            self.wloc[i, :bs].copy_(lc.reshape(-1)[:bs])
        return True

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx,
        out_cache_loc=None,
    ) -> torch.Tensor:
        bs = int(ctx.bs)
        n = bs * H
        if not self._refill_paging(ctx, out_cache_loc, bs):
            self._fallbacks = getattr(self, "_fallbacks", 0) + 1
            if self._fallbacks in (1, 10):
                logger.warning(
                    "megakernel: fell back to the normal path (%d times)",
                    self._fallbacks,
                )
            return None
        # NOTE: no device->host syncs anywhere in this path. Decode runs under
        # the engine's graph capture, where `int(tensor.max())` and friends raise
        # "operation not permitted when stream is capturing". Diagnostics that
        # read device values must be gated on a host-side flag and disabled by
        # default.
        # copy INTO persistent buffers; never bind a fresh tensor (capture-safety)
        self.hs_buf[:n].copy_(hidden_states.reshape(-1))
        self.posn_buf[:bs].copy_(positions.reshape(-1)[:bs])
        self.bar.zero_()

        mk_model_opt[(self.nwg,)](
            self.hs_buf,
            self.out_buf,
            self.aw.an,
            self.aw.wqkv,
            self.aw.bqkv,
            self.aw.wo,
            self.aw.bo,
            self.aw.sinks,
            self.aw.cos,
            self.aw.sin,
            self.kbuf,
            self.vbuf,
            self.koff,
            self.voff,
            self.lgid,
            self.ptab,
            self.ps,
            self.wloc,
            self.win,
            self.posn_buf,
            self.aw.mn,
            self.aw.rw,
            self.aw.rb,
            self.mw.gu_blk,
            self.mw.gu_scl,
            self.mw.gu_bias,
            self.mw.dn_blk,
            self.mw.dn_scl,
            self.mw.dn_bias,
            self.final_norm,
            self.xnorm,
            self.qkv,
            self.attn,
            self.resid,
            self.rlog,
            self.gu,
            self.act,
            self.ynorm,
            self.tki,
            self.tkw,
            self.ynq,
            self.yns,
            self.actq,
            self.acts,
            self.bar,
            bs,
            self.n_layers,
            NWG=self.nwg,
            BLK_H=4096,
            BLK_D=64,
            BLK_E=128,
            TAB_G=self.TAB_G,
            TAB_B=self.TAB_B,
            WLOC_G=self.WLOC_G,
            BLK_K2=2048,
            DOT=self.use_dot,
            BLOCK_N=32,
            BLOCK_K=256,
            MTILE=16,
            EXTRA_BAR=0,
            BLOCK_T=64,
            num_warps=4,
        )
        self._runs = getattr(self, "_runs", 0) + 1
        if self._runs in (1, 10, 100, 500):
            logger.info(
                "megakernel: %d decode steps executed by the megakernel", self._runs
            )
        return self.out_buf[:n].view(bs, H).to(hidden_states.dtype)
