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

"""Snapshot raw MXFP4 expert weights for the decode megakernel.

Why this exists
---------------
``preprocess_gluon_mxfp4_gfx950_moe_weights`` rewrites the MoE weights into a
gdot128-preshuffled, CDNA4-scale-swizzled, K-packed layout and then calls
``_release_parameter`` on ``w13_weight`` / ``w2_weight`` and both scale tensors.
The prefill aliases it leaves behind are *not* plain row-major -- they carry
``is_gdot128_shuffled = True`` precisely so consumers know to use gdot128 byte
addressing. The megakernel's ``tl.dot_scaled`` GEMV cannot read that layout.

So we take a copy of the raw row-major tensors *before* the preprocessor runs.
The hook site is ``GptOssForCausalLM.process_weights_after_loading``: the model
loader iterates ``model.named_modules()``, which yields the root module first,
so the root's hook runs ahead of every ``MoELayer``'s. ``post_quant_warmup``
would be too late -- it runs after the whole loop, by which point the raw
parameters have already been released.

Layout note: for GPT-OSS ``round_up(2880, 32) == 2880``, so the raw parameters
carry no padding and are already row-major ``[E, N, K/2]`` -- exactly what the
GEMV consumes. The snapshot is a straight per-layer copy, not a re-layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

GB = 1024**3

# Temporary opt-in. Becomes a proper ``--enable-gpt-oss-megakernel`` server arg
# once the runner is wired up; an env var keeps the weight work independent of
# server_args/env plumbing for now.
_ENV_FLAG = "TOKENSPEED_GPT_OSS_MEGAKERNEL"


def megakernel_enabled() -> bool:
    """True when --enable-gpt-oss-megakernel is set (env var still overrides)."""
    if os.environ.get(_ENV_FLAG, "0") not in ("0", "", "false", "False"):
        return True
    try:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return bool(global_server_args_dict.get("enable_gpt_oss_megakernel", False))
    except Exception:
        return False


@dataclass
class MegakernelMoEWeights:
    """Layer-stacked raw MXFP4 expert weights, row-major, contiguous.

    gu_blk  [L, E, 2I, H/2]   uint8   E2M1 nibble pairs, low nibble = even elem
    gu_scl  [L, E, 2I, H/32]  uint8   E8M0, value = 2^(byte - 127)
    gu_bias [L, E, 2I]        bfloat16
    dn_blk  [L, E, H,  I/2]   uint8
    dn_scl  [L, E, H,  I/32]  uint8
    dn_bias [L, E, H]         bfloat16
    """

    gu_blk: torch.Tensor
    gu_scl: torch.Tensor
    gu_bias: torch.Tensor | None
    dn_blk: torch.Tensor
    dn_scl: torch.Tensor
    dn_bias: torch.Tensor | None

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.gu_blk,
                self.gu_scl,
                self.gu_bias,
                self.dn_blk,
                self.dn_scl,
                self.dn_bias,
            )
            if t is not None
        )


def probe_model_structure(model: torch.nn.Module) -> None:
    """Dump the live module/parameter layout the runner has to bind to.

    Enabled with TOKENSPEED_GPT_OSS_MEGAKERNEL_PROBE=1. Guessing attribute
    names and tensor layouts is the usual way an integration like this breaks,
    so read them off the real model instead.
    """
    layers = getattr(model, "layers", [])
    if not layers:
        logger.info("megakernel probe: model has no .layers")
        return
    l0 = layers[0]
    logger.info("megakernel probe: layer type %s", type(l0).__name__)
    for name, mod in l0.named_children():
        logger.info("  child %-24s %s", name, type(mod).__name__)
    for name, p in l0.named_parameters(recurse=True):
        logger.info("  param %-52s %-18s %s", name, tuple(p.shape), p.dtype)
    for name, b in l0.named_buffers(recurse=True):
        logger.info("  buffer %-51s %-18s %s", name, tuple(b.shape), b.dtype)


def probe_kv_pool(pool) -> None:
    """Dump the KV pool layout: aliasing, per-layer offsets, dtypes."""
    kb = getattr(pool, "k_buffer", None)
    if kb is None:
        logger.info("megakernel probe: pool %s has no k_buffer", type(pool).__name__)
        return
    base = min(t.data_ptr() for t in kb)
    ptrs = [t.data_ptr() for t in kb]
    esz = kb[0].element_size()
    logger.info(
        "megakernel probe: pool=%s layers=%d dtype=%s store_dtype=%s "
        "shape0=%s distinct_ptrs=%d",
        type(pool).__name__,
        len(kb),
        kb[0].dtype,
        getattr(pool, "store_dtype", None),
        tuple(kb[0].shape),
        len({p for p in ptrs}),
    )
    offs = [(p - base) // esz for p in ptrs]
    logger.info("megakernel probe: per-layer k element offsets (first 8) %s", offs[:8])
    logger.info(
        "megakernel probe: page_size=%s group_specs=%s",
        getattr(pool, "page_size", None),
        [
            getattr(s, "group_id", s)
            for s in getattr(pool, "paged_cache_group_specs", ())
        ],
    )


@dataclass
class MegakernelAttnWeights:
    """Layer-stacked attention/router weights, contiguous for affine indexing.

    Names and layouts read off the live model (see ``probe_model_structure``),
    not assumed. Notably ``qkv_proj`` is ALREADY fused [5120, 2880], which is
    exactly the kernel's ``wqkv`` layout, and ``cos_sin_cache`` is
    [max_pos, 64] holding cos and sin concatenated on the last dim with the
    YaRN mscale already applied -- so the kernel reuses the engine's table
    rather than re-deriving YaRN, which is where a silent 1e-3 drift would hide.
    """

    an: torch.Tensor  # [L, H]      fp32  input_layernorm
    mn: torch.Tensor  # [L, H]      fp32  post_attention_layernorm
    wqkv: torch.Tensor  # [L, 5120, H] bf16
    bqkv: torch.Tensor  # [L, 5120]   fp32
    wo: torch.Tensor  # [L, H, 4096] bf16
    bo: torch.Tensor  # [L, H]      fp32
    sinks: torch.Tensor  # [L, NH]     fp32
    rw: torch.Tensor  # [L, E, H]   bf16
    rb: torch.Tensor  # [L, E]      fp32
    cos: torch.Tensor  # [max_pos, 32] fp32 contiguous
    sin: torch.Tensor  # [max_pos, 32] fp32 contiguous
    fnorm: torch.Tensor  # [H] fp32, the model's final RMSNorm

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in vars(self).values())


def build_megakernel_attn_weights(
    model: torch.nn.Module,
) -> MegakernelAttnWeights | None:
    layers = list(getattr(model, "layers", []))
    if not layers:
        return None
    L = len(layers)
    a0 = layers[0].self_attn
    qkv_n, hid = a0.qkv_proj.weight.shape
    o_out, o_in = a0.o_proj.weight.shape
    nh = a0.sinks.numel()
    rt = layers[0].mlp.router
    ne = rt.weight.shape[0]
    dev = a0.qkv_proj.weight.device

    def ef(*shape, dtype=torch.float32):
        return torch.empty(*shape, dtype=dtype, device=dev)

    w = MegakernelAttnWeights(
        an=ef(L, hid),
        mn=ef(L, hid),
        wqkv=ef(L, qkv_n, hid, dtype=torch.bfloat16),
        bqkv=ef(L, qkv_n),
        wo=ef(L, o_out, o_in, dtype=torch.bfloat16),
        bo=ef(L, o_out),
        sinks=ef(L, nh),
        rw=ef(L, ne, hid, dtype=torch.bfloat16),
        rb=ef(L, ne),
        cos=torch.empty(0),
        sin=torch.empty(0),
        fnorm=torch.empty(0),
    )
    for i, lyr in enumerate(layers):
        at = lyr.self_attn
        w.an[i].copy_(lyr.input_layernorm.weight.float())
        w.mn[i].copy_(lyr.post_attention_layernorm.weight.float())
        w.wqkv[i].copy_(at.qkv_proj.weight)
        w.bqkv[i].copy_(at.qkv_proj.bias.float())
        w.wo[i].copy_(at.o_proj.weight)
        w.bo[i].copy_(at.o_proj.bias.float())
        w.sinks[i].copy_(at.sinks.float())
        w.rw[i].copy_(lyr.mlp.router.weight)
        w.rb[i].copy_(lyr.mlp.router.bias.float())

    # cos_sin_cache is [max_pos, rotary_dim] = cat(cos, sin); the kernel wants
    # them separate and contiguous.
    csc = layers[0].self_attn.rotary_emb.cos_sin_cache
    half = csc.shape[-1] // 2
    w.cos = csc[:, :half].contiguous().float()
    w.sin = csc[:, half:].contiguous().float()
    # the megakernel owns the final norm: GptOssModel.forward's contract is
    # to return POST-final-norm hidden states
    w.fnorm = model.norm.weight.detach().float().contiguous()

    logger.info(
        "megakernel: stacked attention weights for %d layers "
        "(qkv=%s, o=%s, rope=%s, %.2f GB)",
        L,
        tuple(w.wqkv.shape),
        tuple(w.wo.shape),
        tuple(w.cos.shape),
        w.nbytes() / GB,
    )
    return w


def build_kv_tables(pool) -> dict | None:
    """Per-layer KV element offsets into the single pooled allocation.

    Layers ALIAS -- gpt-oss's 36 layers share 18 slabs -- so there is no
    uniform layer*stride and the offsets must be read from the buffers
    themselves. Returns element offsets (not bytes) off the lowest base.
    """
    kb = getattr(pool, "k_buffer", None)
    vb = getattr(pool, "v_buffer", None)
    if not kb or not vb:
        logger.warning(
            "megakernel: pool %s exposes no k/v buffers", type(pool).__name__
        )
        return None
    if pool.store_dtype != torch.bfloat16:
        logger.warning(
            "megakernel: KV store dtype %s unsupported (need bfloat16)",
            pool.store_dtype,
        )
        return None
    # The offsets are relative to the LOWEST-addressed layer view, so the base
    # tensor handed to the kernel must be that same view -- layer 0 is not
    # generally the lowest, and using it silently shifts every KV address by
    # gigabytes. (Symptom: correct prefill, garbage decode.)
    ki = min(range(len(kb)), key=lambda i: kb[i].data_ptr())
    vi = min(range(len(vb)), key=lambda i: vb[i].data_ptr())
    kbase, vbase = kb[ki].data_ptr(), vb[vi].data_ptr()
    esz = kb[0].element_size()
    koff = torch.tensor(
        [(t.data_ptr() - kbase) // esz for t in kb],
        dtype=torch.int64,
        device=kb[0].device,
    )
    voff = torch.tensor(
        [(t.data_ptr() - vbase) // esz for t in vb],
        dtype=torch.int64,
        device=vb[0].device,
    )
    n_distinct = len({t.data_ptr() for t in kb})
    logger.info(
        "megakernel: KV base layer k=%d v=%d, max element offset %d "
        "(base view numel %d)",
        ki,
        vi,
        int(koff.max()),
        kb[ki].numel(),
    )
    logger.info(
        "megakernel: KV offsets built -- %d layers over %d distinct slabs, "
        "head_num=%d head_dim=%d",
        len(kb),
        n_distinct,
        kb[0].shape[1],
        kb[0].shape[2],
    )
    # Absolute per-layer addresses. base+offset arithmetic was verified correct
    # host-side yet demonstrably did NOT reach the live pool from the kernel
    # (sentinel write landed elsewhere), so hand the kernel exact pointers and
    # let it bitcast -- no giant offsets to get wrong.
    kptr = torch.tensor(
        [t.data_ptr() for t in kb], dtype=torch.int64, device=kb[0].device
    )
    vptr = torch.tensor(
        [t.data_ptr() for t in vb], dtype=torch.int64, device=vb[0].device
    )
    return {
        "koff": koff,
        "voff": voff,
        "n_slabs": n_distinct,
        "k_base_view": kb[ki],
        "v_base_view": vb[vi],
        "kptr": kptr,
        "vptr": vptr,
    }


def _moe_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Every MoELayer that still holds raw MXFP4 parameters, in layer order."""
    found = []
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        # the MoELayer may be the mlp itself or hang off it
        for cand in (mlp, getattr(mlp, "experts", None)):
            if cand is not None and hasattr(cand, "w13_weight"):
                found.append(cand)
                break
    return found


def build_megakernel_moe_weights(model: torch.nn.Module) -> MegakernelMoEWeights | None:
    """Copy raw per-layer MXFP4 expert weights into flat layer-stacked blobs.

    Returns None (with a warning) if the expected raw parameters are absent, so
    that enabling the flag on an unsupported checkpoint degrades to the normal
    path rather than crashing during weight load.
    """
    moes = _moe_layers(model)
    if not moes:
        logger.warning(
            "megakernel: no MoELayer with raw w13_weight found; "
            "skipping weight snapshot"
        )
        return None

    n_layers = len(moes)
    ref = moes[0]
    have_bias = hasattr(ref, "w13_weight_bias") and ref.w13_weight_bias is not None

    def _alloc(sample: torch.Tensor) -> torch.Tensor:
        # preallocate and fill per layer: torch.stack would spike a full extra copy
        return torch.empty(
            (n_layers, *sample.shape), dtype=sample.dtype, device=sample.device
        )

    blobs = {
        "gu_blk": _alloc(ref.w13_weight.data),
        "gu_scl": _alloc(ref.w13_weight_scale.data),
        "dn_blk": _alloc(ref.w2_weight.data),
        "dn_scl": _alloc(ref.w2_weight_scale.data),
    }
    if have_bias:
        blobs["gu_bias"] = _alloc(ref.w13_weight_bias.data)
        blobs["dn_bias"] = _alloc(ref.w2_weight_bias.data)

    src_names = {
        "gu_blk": "w13_weight",
        "gu_scl": "w13_weight_scale",
        "dn_blk": "w2_weight",
        "dn_scl": "w2_weight_scale",
        "gu_bias": "w13_weight_bias",
        "dn_bias": "w2_weight_bias",
    }
    for layer_id, moe in enumerate(moes):
        for key, blob in blobs.items():
            src = getattr(moe, src_names[key]).data
            if src.shape != blob.shape[1:]:
                raise RuntimeError(
                    f"megakernel: layer {layer_id} {src_names[key]} has shape "
                    f"{tuple(src.shape)}, expected {tuple(blob.shape[1:])}; "
                    "the checkpoint layout is not what the megakernel assumes"
                )
            blob[layer_id].copy_(src)

    out = MegakernelMoEWeights(
        gu_blk=blobs["gu_blk"],
        gu_scl=blobs["gu_scl"],
        gu_bias=blobs.get("gu_bias"),
        dn_blk=blobs["dn_blk"],
        dn_scl=blobs["dn_scl"],
        dn_bias=blobs.get("dn_bias"),
    )
    logger.info(
        "megakernel: snapshotted raw MXFP4 experts for %d layers "
        "(gu=%s, dn=%s, %.2f GB duplicated)",
        n_layers,
        tuple(out.gu_blk.shape),
        tuple(out.dn_blk.shape),
        out.nbytes() / GB,
    )
    free, total = torch.cuda.mem_get_info()
    logger.info(
        "megakernel: VRAM after snapshot -- used %.2f GB of %.2f GB (%.2f GB free)",
        (total - free) / GB,
        total / GB,
        free / GB,
    )
    return out
