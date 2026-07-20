"""GFX1250 paged grouped-query decode attention kernels and benchmark harness."""

import re

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import pytest

if not hasattr(gluon, "aggregate"):
    from triton.language.core import _aggregate

    gluon.aggregate = _aggregate


def compute_split_factor(batch, num_q_heads, seq_len_k):
    target_total_wrkgrps = 1024
    tasks = batch * num_q_heads
    if tasks == 0: return 1
    ideal_split = target_total_wrkgrps // tasks
    min_chunk_size = 64  # arbitrary lower bound
    max_possible_splits = (seq_len_k + min_chunk_size - 1) // min_chunk_size

    split_factor = min(ideal_split, max_possible_splits)
    split_factor = max(split_factor, 1)
    return split_factor


def compute_peeled_split_factor(batch, num_head_groups, seq_len_k, block_n):
    split_factor = compute_split_factor(batch, num_head_groups, seq_len_k)
    max_peeled_splits = max(seq_len_k // (4 * block_n), 1)
    return min(split_factor, max_peeled_splits)


def estimate_attention_traffic_bytes(config):
    batch = config["BATCH"]
    seqlen_q = config["SEQLEN_Q"]
    seqlen_k = config["SEQLEN_K"]
    num_q_heads = config["NUM_Q_HEADS"]
    num_k_heads = config["NUM_K_HEADS"]
    head_sz = config["HEAD_SZ"]
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    dtype = config.get("DTYPE", "bf16")
    paged_decode = config.get("PAGED_DECODE", True)
    page_block_size = config.get("PAGE_BLOCK_SIZE") or block_n
    decode_stage = config.get("DECODE_STAGE")

    dtype_bytes = 2 if dtype in ("fp16", "bf16") else 4
    q_bytes = batch * num_q_heads * seqlen_q * head_sz * dtype_bytes
    output_bytes = batch * num_q_heads * seqlen_q * head_sz * 4

    if config.get("SPLIT_FACTOR"):
        split_factor = config["SPLIT_FACTOR"]
    elif paged_decode and decode_stage == "full-gqa":
        gqa_group_size = num_q_heads // num_k_heads
        gqa_block_h = config.get("GQA_BLOCK_H") or min(gqa_group_size, block_m)
        num_head_groups = num_k_heads * (gqa_group_size // gqa_block_h)
        split_factor = compute_split_factor(batch, num_head_groups, seqlen_k)
    elif paged_decode and decode_stage in ("full-gqa-peeled", "full-gqa-peeled-direct"):
        gqa_group_size = num_q_heads // num_k_heads
        gqa_block_h = config.get("GQA_BLOCK_H") or min(gqa_group_size, block_m)
        num_head_groups = num_k_heads * (gqa_group_size // gqa_block_h)
        split_factor = compute_peeled_split_factor(batch, num_head_groups, seqlen_k, block_n)
    else:
        split_factor = compute_split_factor(batch, num_q_heads, seqlen_k)

    if paged_decode and decode_stage in ("full-gqa", "full-gqa-peeled", "full-gqa-peeled-direct"):
        gqa_group_size = num_q_heads // num_k_heads
        gqa_block_h = config.get("GQA_BLOCK_H") or min(gqa_group_size, block_m)
        kv_reader_groups = num_k_heads * (gqa_group_size // gqa_block_h)
    else:
        kv_reader_groups = num_q_heads

    kv_bytes = batch * kv_reader_groups * seqlen_k * head_sz * dtype_bytes * 2
    page_table_bytes = 0
    if paged_decode:
        pages_per_batch = (seqlen_k + page_block_size - 1) // page_block_size
        page_table_bytes = batch * kv_reader_groups * pages_per_batch * 4

    direct_output = decode_stage in ("stage1-pipeline-direct", "full-peeled-direct", "full-gqa-peeled-direct")
    intermediate_bytes = 0
    if config["ATTN_FN"] == "decode" and not direct_output:
        mid_o_bytes = batch * num_q_heads * split_factor * block_m * head_sz * 4
        mid_lm_bytes = batch * num_q_heads * split_factor * block_m * 4 * 2
        intermediate_bytes = 2 * (mid_o_bytes + mid_lm_bytes)

    return q_bytes + kv_bytes + page_table_bytes + output_bytes + intermediate_bytes


def static_profile(kernel):
    amdgcn = kernel.asm["amdgcn"]
    fields = {
        "sgpr_count": r"\.sgpr_count:\s+(\d+)",
        "sgpr_spill_count": r"\.sgpr_spill_count:\s+(\d+)",
        "vgpr_count": r"\.vgpr_count:\s+(\d+)",
        "vgpr_spill_count": r"\.vgpr_spill_count:\s+(\d+)",
        "scratch_size": r";\s+ScratchSize:\s+(\d+)",
        "code_len_in_byte": r";\s+codeLenInByte\s+=\s+(\d+)",
        "occupancy": r";\s+Occupancy:\s+(\d+)",
    }
    for name, pattern in fields.items():
        match = re.search(pattern, amdgcn)
        print(f"- {name}: {int(match.group(1)) if match else 'unknown'}")


@gluon.aggregate
class AttentionConfig:
    SEQLEN_Q: gl.constexpr
    SEQLEN_K: gl.constexpr
    HEAD_SZ: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_BUFFERS: gl.constexpr

    qk_layout: gl.constexpr
    pv_layout: gl.constexpr

    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    q_layout: gl.constexpr
    k_layout: gl.constexpr
    v_layout: gl.constexpr
    p_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS):

        # constants
        self.SEQLEN_Q = gl.constexpr(SEQLEN_Q)
        self.SEQLEN_K = gl.constexpr(SEQLEN_K)
        self.HEAD_SZ = gl.constexpr(HEAD_SZ)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)

        assert NUM_WARPS == 4 or NUM_WARPS == 8
        if NUM_WARPS == 4:
            warp_bases = [[1, 0], [2, 0]]
        else:
            warp_bases = [[1, 0], [2, 0], [4, 0]]

        # operator layouts
        self.qk_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]))
        self.pv_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]))

        # tensor layouts
        self.k_smem_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[HEAD_SZ, 8]], [BLOCK_N, HEAD_SZ], [1, 0]))
        self.v_smem_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[HEAD_SZ, 16]], [BLOCK_N, HEAD_SZ], [1, 0]))

        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, self.qk_layout, 8))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, self.qk_layout, 8))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, self.pv_layout, 8))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, self.pv_layout, 8))


@gluon.aggregate
class AttentionProgram:
    cfg: AttentionConfig

    q: gl.tensor

    k_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    k_buffer: gl.shared_memory_descriptor

    v_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    v_buffer: gl.shared_memory_descriptor

    o_ptr: gl.tensor
    o_offs: gl.tensor
    o_mask: gl.tensor

    sm_scale: gl.constexpr
    rcp_ln2: gl.constexpr
    sm_scale_dot_rcp_ln2: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg,  #
                 q,  #
                 k_desc, k_buffer,  #
                 v_desc, v_buffer,  #
                 o_ptr, o_offs, o_mask,  #
                 sm_scale):

        self.cfg = cfg
        self.q = q

        self.k_desc = k_desc
        self.k_buffer = k_buffer
        self.v_desc = v_desc
        self.v_buffer = v_buffer

        self.o_ptr = o_ptr
        self.o_offs = o_offs
        self.o_mask = o_mask

        self.sm_scale = gl.constexpr(sm_scale)
        self.rcp_ln2 = gl.constexpr(1.4426950408889634)
        self.sm_scale_dot_rcp_ln2: gl.constexpr = self.sm_scale * self.rcp_ln2

    @gluon.jit
    def initialize_decode(cfg, q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
                          stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, SM_SCALE):
        off_z = gl.program_id(0)
        off_q_head = gl.program_id(1)
        off_k_head = off_q_head
        off_m = 0  # Decode always processes the first (and only) Q block per instance

        # q [BLOCK_M, HEAD_SZ]
        q_offs = (stride_qz * off_z + stride_qh * off_q_head + stride_qm *
                  (off_m + gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] + stride_qk *
                  (gl.arange(0, cfg.HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout)))[None, :])

        # k [HEAD_SZ, BLOCK_N]
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=k_ptr + stride_kz * off_z + stride_kh * off_k_head,
                                                           shape=(cfg.SEQLEN_K, cfg.HEAD_SZ),
                                                           strides=(stride_kn, stride_kk),
                                                           block_shape=(cfg.BLOCK_N, cfg.HEAD_SZ),
                                                           layout=cfg.k_smem_layout)
        k_buffer = gl.allocate_shared_memory(k_desc.dtype, shape=[cfg.NUM_BUFFERS] + k_desc.block_shape,
                                             layout=k_desc.layout)

        # v [BLOCK_N, BLOCK_DMODEL]
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=v_ptr + stride_vz * off_z + stride_vh * off_k_head,
                                                           shape=(cfg.SEQLEN_K, cfg.HEAD_SZ),
                                                           strides=(stride_vn, stride_vk),
                                                           block_shape=(cfg.BLOCK_N, cfg.HEAD_SZ),
                                                           layout=cfg.v_smem_layout)
        v_buffer = gl.allocate_shared_memory(v_desc.dtype, shape=[cfg.NUM_BUFFERS] + v_desc.block_shape,
                                             layout=v_desc.layout)

        q_mask = (off_m + gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] < cfg.SEQLEN_Q
        q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

        # dummy values for Program struct (unused in Decode FWD)
        o_offs = gl.zeros([cfg.BLOCK_M, cfg.HEAD_SZ], dtype=gl.int32)
        o_mask = gl.zeros([cfg.BLOCK_M, 1], dtype=gl.int1)

        return AttentionProgram(cfg, q, k_desc, k_buffer, v_desc, v_buffer, o_ptr, o_offs, o_mask, SM_SCALE)

    @gluon.jit
    def initialize_decode_paged(cfg, q_ptr, k_ptr, v_ptr, o_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kp,
                                stride_kh, stride_kn, stride_kk, stride_vp, stride_vh, stride_vn, stride_vk,
                                PAGE_BLOCK_SIZE: gl.constexpr, SM_SCALE):
        off_z = gl.program_id(0)
        off_q_head = gl.program_id(1)
        off_m = 0  # Decode always processes the first (and only) Q block per instance.

        q_offs = (stride_qz * off_z + stride_qh * off_q_head + stride_qm *
                  (off_m + gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] + stride_qk *
                  (gl.arange(0, cfg.HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout)))[None, :])

        # Dummy descriptors provide dtype/layout for the program object. Actual
        # paged descriptors are rebuilt per tile from the block table.
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=k_ptr, shape=(PAGE_BLOCK_SIZE, cfg.HEAD_SZ),
                                                           strides=(stride_kn, stride_kk),
                                                           block_shape=(cfg.BLOCK_N, cfg.HEAD_SZ),
                                                           layout=cfg.k_smem_layout)
        k_buffer = gl.allocate_shared_memory(k_desc.dtype, shape=[cfg.NUM_BUFFERS] + k_desc.block_shape,
                                             layout=k_desc.layout)

        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=v_ptr, shape=(PAGE_BLOCK_SIZE, cfg.HEAD_SZ),
                                                           strides=(stride_vn, stride_vk),
                                                           block_shape=(cfg.BLOCK_N, cfg.HEAD_SZ),
                                                           layout=cfg.v_smem_layout)
        v_buffer = gl.allocate_shared_memory(v_desc.dtype, shape=[cfg.NUM_BUFFERS] + v_desc.block_shape,
                                             layout=v_desc.layout)

        q_mask = (off_m + gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] < cfg.SEQLEN_Q
        q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

        o_offs = gl.zeros([cfg.BLOCK_M, cfg.HEAD_SZ], dtype=gl.int32)
        o_mask = gl.zeros([cfg.BLOCK_M, 1], dtype=gl.int1)

        return AttentionProgram(cfg, q, k_desc, k_buffer, v_desc, v_buffer, o_ptr, o_offs, o_mask, SM_SCALE)

    @gluon.jit
    def initialize(cfg,  #
                   q_ptr, k_ptr, v_ptr, o_ptr,  #
                   stride_qz, stride_qh, stride_qm, stride_qk,  #
                   stride_kz, stride_kh, stride_kn, stride_kk,  #
                   stride_vz, stride_vh, stride_vn, stride_vk,  #
                   stride_oz, stride_oh, stride_om, stride_on,  #
                   sm_scale: gl.constexpr):
        SEQLEN_K: gl.constexpr = cfg.SEQLEN_K
        SEQLEN_Q: gl.constexpr = cfg.SEQLEN_Q
        HEAD_SZ: gl.constexpr = cfg.HEAD_SZ
        BLOCK_M: gl.constexpr = cfg.BLOCK_M
        BLOCK_N: gl.constexpr = cfg.BLOCK_N

        # workgroup offsets
        off_z = gl.program_id(0)
        off_q_head = gl.program_id(1)
        off_k_head = off_q_head
        off_m = gl.program_id(2) * BLOCK_M

        # q [BLOCK_M, HEAD_SZ]
        q_offs = (stride_qz * off_z + stride_qh * off_q_head + stride_qm *
                  (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] + stride_qk *
                  (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout)))[None, :])

        # k [HEAD_SZ, BLOCK_N]
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=k_ptr + stride_kz * off_z + stride_kh * off_k_head,  #
            shape=(SEQLEN_K, HEAD_SZ),  #
            strides=(stride_kn, stride_kk),  #
            block_shape=(BLOCK_N, HEAD_SZ),  #
            layout=cfg.k_smem_layout)
        k_buffer = gl.allocate_shared_memory(k_desc.dtype, shape=[cfg.NUM_BUFFERS] + k_desc.block_shape,
                                             layout=k_desc.layout)

        # v [BLOCK_N, BLOCK_DMODEL]
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=v_ptr + stride_vz * off_z + stride_vh * off_k_head,  #
            shape=(SEQLEN_K, HEAD_SZ),  #
            strides=(stride_vn, stride_vk),  #
            block_shape=(BLOCK_N, HEAD_SZ),  #
            layout=cfg.v_smem_layout)
        v_buffer = gl.allocate_shared_memory(v_desc.dtype, shape=[cfg.NUM_BUFFERS] + v_desc.block_shape,
                                             layout=v_desc.layout)

        q_mask = (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)))[:, None] < SEQLEN_Q
        q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

        o_offs = (stride_oz * off_z + stride_oh * off_q_head + stride_om *
                  (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] + stride_on *
                  (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout)))[None, :])

        o_mask = (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)))[:, None] < SEQLEN_Q

        # create the program
        return AttentionProgram(cfg, q,  #
                                k_desc, k_buffer,  #
                                v_desc, v_buffer,  #
                                o_ptr, o_offs, o_mask,  #
                                sm_scale)

    @gluon.jit
    def tdm_shared_load_k(self, buffer_id, wait_count):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.k_buffer.index(buffer_id).permute([1, 0]).load(layout=self.cfg.k_layout)

    @gluon.jit
    def tdm_shared_load_v(self, buffer_id, wait_count):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.v_buffer.index(buffer_id).load(layout=self.cfg.v_layout)

    @gluon.jit
    def tdm_load_global_to_shared_k(self, offset, buffer_index, cache_modifier: gl.constexpr = "", pred=True):
        gl.amd.gfx1250.tdm.async_load(self.k_desc, offset, self.k_buffer.index(buffer_index), pred=pred,
                                      cache_modifier=cache_modifier)

    @gluon.jit
    def tdm_load_global_to_shared_v(self, offset, buffer_index, cache_modifier: gl.constexpr = "", pred=True):
        gl.amd.gfx1250.tdm.async_load(self.v_desc, offset, self.v_buffer.index(buffer_index), pred=pred,
                                      cache_modifier=cache_modifier)

    @gluon.jit
    def tdm_prefetch_k(self, offset, pred=True):
        gl.amd.gfx1250.tdm.prefetch(self.k_desc, offset, pred=pred)

    @gluon.jit
    def tdm_prefetch_v(self, offset, pred=True):
        gl.amd.gfx1250.tdm.prefetch(self.v_desc, offset, pred=pred)

    @gluon.jit
    def tdm_load_paged_global_to_shared_k(self, k_ptr, block_table_ptr, logical_k, buffer_index, stride_kp, stride_kh,
                                          stride_kn, stride_kk, off_z, off_k_head, PAGES_PER_BATCH: gl.constexpr,
                                          PAGE_BLOCK_SIZE: gl.constexpr, cache_modifier: gl.constexpr = "", pred=True):
        logical_page = logical_k // PAGE_BLOCK_SIZE
        page_offset = logical_k - logical_page * PAGE_BLOCK_SIZE
        physical_page = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + logical_page)
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=k_ptr + physical_page * stride_kp + off_k_head * stride_kh, shape=(PAGE_BLOCK_SIZE, self.cfg.HEAD_SZ),
            strides=(stride_kn, stride_kk), block_shape=(self.cfg.BLOCK_N, self.cfg.HEAD_SZ),
            layout=self.cfg.k_smem_layout)
        gl.amd.gfx1250.tdm.async_load(k_desc, [page_offset, 0], self.k_buffer.index(buffer_index), pred=pred,
                                      cache_modifier=cache_modifier)

    @gluon.jit
    def tdm_load_paged_global_to_shared_v(self, v_ptr, block_table_ptr, logical_k, buffer_index, stride_vp, stride_vh,
                                          stride_vn, stride_vk, off_z, off_k_head, PAGES_PER_BATCH: gl.constexpr,
                                          PAGE_BLOCK_SIZE: gl.constexpr, cache_modifier: gl.constexpr = "", pred=True):
        logical_page = logical_k // PAGE_BLOCK_SIZE
        page_offset = logical_k - logical_page * PAGE_BLOCK_SIZE
        physical_page = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + logical_page)
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_ptr + physical_page * stride_vp + off_k_head * stride_vh, shape=(PAGE_BLOCK_SIZE, self.cfg.HEAD_SZ),
            strides=(stride_vn, stride_vk), block_shape=(self.cfg.BLOCK_N, self.cfg.HEAD_SZ),
            layout=self.cfg.v_smem_layout)
        gl.amd.gfx1250.tdm.async_load(v_desc, [page_offset, 0], self.v_buffer.index(buffer_index), pred=pred,
                                      cache_modifier=cache_modifier)

    @gluon.jit
    def compute_qk(self, k, cur_seq):
        qk = gl.zeros([self.cfg.BLOCK_M, self.cfg.BLOCK_N], dtype=gl.float32, layout=self.cfg.qk_layout)
        qk = gl.amd.gfx1250.wmma(self.q, k, qk)
        # Handle/pad unaligned M and K2 ids for QK.
        qk_mask = (
            cur_seq +
            gl.arange(0, self.cfg.BLOCK_N, layout=gl.SliceLayout(0, self.cfg.qk_layout)))[None, :] < self.cfg.SEQLEN_K
        qk = gl.where(qk_mask, qk, float("-inf"))
        return qk

    @gluon.jit
    def compute_qk_no_mask(self, k):
        qk = gl.zeros([self.cfg.BLOCK_M, self.cfg.BLOCK_N], dtype=gl.float32, layout=self.cfg.qk_layout)
        qk = gl.amd.gfx1250.wmma(self.q, k, qk)
        return qk

    @gluon.jit
    def mask_qk_to_end(self, qk, cur_seq, end_k):
        qk_mask = (cur_seq +
                   gl.arange(0, self.cfg.BLOCK_N, layout=gl.SliceLayout(0, self.cfg.qk_layout)))[None, :] < end_k
        return gl.where(qk_mask, qk, float("-inf"))

    @gluon.jit
    def softmax_part0(self, qk, m_i):
        # get max scores so far
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * self.sm_scale_dot_rcp_ln2

        # scale and subtract max
        q_shifted = self.sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = gl.exp2(q_shifted)

        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = self.sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled
        alpha = gl.exp2(m_diff_scaled)

        return p, alpha, m_ij

    @gluon.jit
    def compute_pv(self, p, v, acc):
        p = gl.convert_layout(p, self.cfg.p_layout)
        return gl.amd.gfx1250.wmma(p, v, acc)

    @gluon.jit
    def softmax_part1(self, p, l_i, acc, alpha):
        # update l_ij before applying dropout
        l_ij = gl.sum(p, 1)

        # update output accumulator
        updated_acc = acc * alpha[:, None]
        updated_p = p.to(self.v_desc.dtype, fp_downcast_rounding="rtz")

        # Update l_i
        updated_l_i = l_i * alpha + l_ij
        return updated_p, updated_l_i, updated_acc

    @gluon.jit
    def store_output(self, out):
        casted_out = out.to(self.o_ptr.dtype.element_ty)
        gl.amd.gfx1250.buffer_store(casted_out, self.o_ptr, self.o_offs, mask=self.o_mask)


@gluon.jit
def attn_decode_fwd_paged_gqa_kernel(q_ptr, k_ptr, v_ptr, block_table_ptr, mid_o_ptr, mid_l_ptr, mid_m_ptr,
                                     stride_qz, stride_qh, stride_qm, stride_qk, stride_kp, stride_kh, stride_kn,
                                     stride_kk, stride_vp, stride_vh, stride_vn, stride_vk, stride_mid_oz,
                                     stride_mid_oh, stride_mid_os, stride_mid_om, stride_mid_on, stride_mid_lz,
                                     stride_mid_lh, stride_mid_ls, stride_mid_lm, stride_mid_mz, stride_mid_mh,
                                     stride_mid_ms, stride_mid_mm, SM_SCALE: gl.constexpr, SEQLEN_Q: gl.constexpr,
                                     SEQLEN_K: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                     HEAD_SZ: gl.constexpr, GQA_GROUP_SIZE: gl.constexpr,
                                     GQA_BLOCK_H: gl.constexpr, GQA_GROUPS_PER_K_HEAD: gl.constexpr,
                                     SPLIT_FACTOR: gl.constexpr, CHUNK_SIZE: gl.constexpr,
                                     DECODE_NUM_WARPS: gl.constexpr, PAGES_PER_BATCH: gl.constexpr,
                                     PAGE_BLOCK_SIZE: gl.constexpr, TDM_CACHE_MODIFIER: gl.constexpr):
    NUM_BUFFERS: gl.constexpr = 1
    NUM_WARPS: gl.constexpr = DECODE_NUM_WARPS
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)

    off_z = gl.program_id(0)
    off_head_group = gl.program_id(1)
    split_id = gl.program_id(2)

    off_k_head = off_head_group // GQA_GROUPS_PER_K_HEAD
    q_group_in_k_head = off_head_group - off_k_head * GQA_GROUPS_PER_K_HEAD
    q_head_start = off_k_head * GQA_GROUP_SIZE + q_group_in_k_head * GQA_BLOCK_H

    q_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
    q_cols = gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout))
    q_heads = q_head_start + q_rows
    q_offs = (stride_qz * off_z + stride_qh * q_heads[:, None] + stride_qk * q_cols[None, :])
    q_mask = q_rows[:, None] < GQA_BLOCK_H
    q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=k_ptr, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
                                                       strides=(stride_kn, stride_kk),
                                                       block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.k_smem_layout)
    k_buffer = gl.allocate_shared_memory(k_desc.dtype, shape=[NUM_BUFFERS] + k_desc.block_shape, layout=k_desc.layout)
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=v_ptr, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
                                                       strides=(stride_vn, stride_vk),
                                                       block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.v_smem_layout)
    v_buffer = gl.allocate_shared_memory(v_desc.dtype, shape=[NUM_BUFFERS] + v_desc.block_shape, layout=v_desc.layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_i = gl.full([BLOCK_M], 0.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)
    sm_scale_dot_rcp_ln2: gl.constexpr = SM_SCALE * 1.4426950408889634

    start_k = split_id * CHUNK_SIZE
    end_k = min(start_k + CHUNK_SIZE, SEQLEN_K)

    for current_k in range(start_k, end_k, BLOCK_N):
        logical_page = current_k // PAGE_BLOCK_SIZE
        page_offset = current_k - logical_page * PAGE_BLOCK_SIZE
        physical_page = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + logical_page)

        k_tile_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=k_ptr + physical_page * stride_kp + off_k_head * stride_kh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
            strides=(stride_kn, stride_kk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.k_smem_layout)
        gl.amd.gfx1250.tdm.async_load(k_tile_desc, [page_offset, 0], k_buffer.index(0),
                                      cache_modifier=TDM_CACHE_MODIFIER)
        gl.amd.gfx1250.tdm.async_wait(0)
        k = k_buffer.index(0).permute([1, 0]).load(layout=cfg.k_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout)
        qk = gl.amd.gfx1250.wmma(q, k, qk)
        k_mask = (current_k + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)))[None, :] < end_k
        qk = gl.where(k_mask, qk, float("-inf"))

        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * sm_scale_dot_rcp_ln2
        p = gl.exp2(sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None])
        alpha = gl.exp2(sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled)
        l_ij = gl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        p = p.to(v_desc.dtype, fp_downcast_rounding="rtz")

        v_tile_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_ptr + physical_page * stride_vp + off_k_head * stride_vh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
            strides=(stride_vn, stride_vk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.v_smem_layout)
        gl.amd.gfx1250.tdm.async_load(v_tile_desc, [page_offset, 0], v_buffer.index(0),
                                      cache_modifier=TDM_CACHE_MODIFIER)
        gl.amd.gfx1250.tdm.async_wait(0)
        v = v_buffer.index(0).load(layout=cfg.v_layout)
        p = gl.convert_layout(p, cfg.p_layout)
        acc = gl.amd.gfx1250.wmma(p, v, acc)

    store_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
    store_cols = gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout))
    store_q_heads = q_head_start + store_rows
    store_mask = store_rows < GQA_BLOCK_H

    mid_o_offs = (off_z * stride_mid_oz + store_q_heads[:, None] * stride_mid_oh + split_id * stride_mid_os +
                  store_cols[None, :] * stride_mid_on)
    gl.amd.gfx1250.buffer_store(acc.to(mid_o_ptr.dtype.element_ty), mid_o_ptr, mid_o_offs, mask=store_mask[:, None])

    mid_l_offs = off_z * stride_mid_lz + store_q_heads * stride_mid_lh + split_id * stride_mid_ls
    mid_m_offs = off_z * stride_mid_mz + store_q_heads * stride_mid_mh + split_id * stride_mid_ms
    gl.store(mid_l_ptr + mid_l_offs, l_i, mask=store_mask)
    gl.store(mid_m_ptr + mid_m_offs, m_i, mask=store_mask)


@gluon.jit
def attn_decode_fwd_paged_pipeline_peeled_kernel(
        q_ptr, k_ptr, v_ptr, block_table_ptr, mid_o_ptr, mid_l_ptr, mid_m_ptr, stride_qz, stride_qh, stride_qm,
        stride_qk, stride_kp, stride_kh, stride_kn, stride_kk, stride_vp, stride_vh, stride_vn, stride_vk,
        stride_mid_oz, stride_mid_oh, stride_mid_os, stride_mid_om, stride_mid_on, stride_mid_lz, stride_mid_lh,
        stride_mid_ls, stride_mid_lm, stride_mid_mz, stride_mid_mh, stride_mid_ms, stride_mid_mm,
        SM_SCALE: gl.constexpr, SEQLEN_Q: gl.constexpr, SEQLEN_K: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, HEAD_SZ: gl.constexpr, GQA_GROUP_SIZE: gl.constexpr, SPLIT_FACTOR: gl.constexpr,
        GQA_BLOCK_H: gl.constexpr, GQA_GROUPS_PER_K_HEAD: gl.constexpr, CHUNK_SIZE: gl.constexpr,
        DECODE_NUM_WARPS: gl.constexpr, PAGES_PER_BATCH: gl.constexpr, PAGE_BLOCK_SIZE: gl.constexpr,
        TDM_CACHE_MODIFIER: gl.constexpr, PEELED_HOT_WAIT_COUNT: gl.constexpr,
        PEELED_SKIP_FINAL_WAIT: gl.constexpr, PEELED_DIRECT_OUTPUT: gl.constexpr):
    NUM_BUFFERS: gl.constexpr = 2
    NUM_WARPS: gl.constexpr = DECODE_NUM_WARPS
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)

    off_z = gl.program_id(0)
    off_head_group = gl.program_id(1)
    off_k_head = off_head_group // GQA_GROUPS_PER_K_HEAD
    q_group_in_k_head = off_head_group - off_k_head * GQA_GROUPS_PER_K_HEAD
    q_head_start = off_k_head * GQA_GROUP_SIZE + q_group_in_k_head * GQA_BLOCK_H
    off_m: gl.constexpr = 0

    q_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
    q_cols = gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.q_layout))
    q_heads = q_head_start + q_rows
    q_offs = stride_qz * off_z + stride_qh * q_heads[:, None] + stride_qk * q_cols[None, :]
    q_mask = q_rows[:, None] < GQA_BLOCK_H
    q = gl.amd.gfx1250.buffer_load(q_ptr, q_offs, mask=q_mask)

    if PEELED_DIRECT_OUTPUT:
        split_id: gl.constexpr = 0
        start_k: gl.constexpr = 0
        end_k: gl.constexpr = SEQLEN_K
    else:
        split_id = gl.program_id(2)
        start_k = split_id * CHUNK_SIZE
        end_k = min(start_k + CHUNK_SIZE, SEQLEN_K)

    logical_k_0 = start_k
    start_page = start_k // BLOCK_N
    page_idx_0 = start_page
    physical_page_0 = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + page_idx_0)
    k_desc_0 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + physical_page_0 * stride_kp + off_k_head * stride_kh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
        strides=(stride_kn, stride_kk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.k_smem_layout)
    k_buffer = gl.allocate_shared_memory(k_desc_0.dtype, shape=[NUM_BUFFERS] + k_desc_0.block_shape,
                                         layout=k_desc_0.layout)
    v_desc_0 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + physical_page_0 * stride_vp + off_k_head * stride_vh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
        strides=(stride_vn, stride_vk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.v_smem_layout)
    v_buffer = gl.allocate_shared_memory(v_desc_0.dtype, shape=[NUM_BUFFERS] + v_desc_0.block_shape,
                                         layout=v_desc_0.layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=cfg.pv_layout)
    sm_scale_dot_rcp_ln2: gl.constexpr = SM_SCALE * 1.4426950408889634

    gl.amd.gfx1250.tdm.async_load(k_desc_0, [0, 0], k_buffer.index(0), pred=logical_k_0 < end_k,
                                  cache_modifier=TDM_CACHE_MODIFIER)

    logical_k_1 = start_k + BLOCK_N
    page_idx_1 = start_page + 1
    physical_page_1 = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + page_idx_1)
    k_desc_1 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + physical_page_1 * stride_kp + off_k_head * stride_kh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
        strides=(stride_kn, stride_kk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.k_smem_layout)
    gl.amd.gfx1250.tdm.async_load(k_desc_1, [0, 0], k_buffer.index(1), pred=logical_k_1 < end_k,
                                  cache_modifier=TDM_CACHE_MODIFIER)
    gl.amd.gfx1250.tdm.async_load(v_desc_0, [0, 0], v_buffer.index(0), pred=logical_k_0 < end_k,
                                  cache_modifier=TDM_CACHE_MODIFIER)

    gl.amd.gfx1250.tdm.async_wait(2)
    k = k_buffer.index(0).permute([1, 0]).load(layout=cfg.k_layout)

    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout)
    qk = gl.amd.gfx1250.wmma(q, k, qk)
    qk_mask = (logical_k_0 + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)))[None, :] < end_k
    qk = gl.where(qk_mask, qk, float("-inf"))
    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    m_ij_scaled = m_ij * sm_scale_dot_rcp_ln2
    p = gl.exp2(sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None])
    alpha = gl.exp2(sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled)
    m_i = m_ij

    tile_2: gl.constexpr = 2 * BLOCK_N
    logical_k_2 = start_k + tile_2
    page_idx_2 = start_page + 2
    physical_page_2 = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + page_idx_2)
    k_desc_2 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + physical_page_2 * stride_kp + off_k_head * stride_kh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
        strides=(stride_kn, stride_kk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.k_smem_layout)
    gl.amd.gfx1250.tdm.async_load(k_desc_2, [0, 0], k_buffer.index(0), pred=logical_k_2 < end_k,
                                  cache_modifier=TDM_CACHE_MODIFIER)

    v_desc_1 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + physical_page_1 * stride_vp + off_k_head * stride_vh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
        strides=(stride_vn, stride_vk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.v_smem_layout)
    gl.amd.gfx1250.tdm.async_load(v_desc_1, [0, 0], v_buffer.index(1), pred=logical_k_1 < end_k,
                                  cache_modifier=TDM_CACHE_MODIFIER)

    gl.amd.gfx1250.tdm.async_wait(3)
    k = k_buffer.index(1).permute([1, 0]).load(layout=cfg.k_layout)

    ITERS_IN_PROLOGUE_EPILOGUE: gl.constexpr = 3
    n_blocks_n = max((CHUNK_SIZE + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1)
    block_max = n_blocks_n * BLOCK_N

    iter_id = 0
    physical_page_for_next_v = physical_page_2
    for block_id in range(0, block_max, BLOCK_N):
        t_2 = block_id + 2 * BLOCK_N
        t_3 = block_id + 3 * BLOCK_N
        qk_tile = start_k + block_id + BLOCK_N
        logical_t_2 = start_k + t_2
        logical_t_3 = start_k + t_3
        page_idx_k_next = start_page + iter_id + 3
        physical_page_k_next = gl.load(block_table_ptr + off_z * PAGES_PER_BATCH + page_idx_k_next)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout)
        qk = gl.amd.gfx1250.wmma(q, k, qk)
        if SPLIT_FACTOR != 1:
            qk_mask_loop = (qk_tile + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)))[None, :] < end_k
            qk = gl.where(qk_mask_loop, qk, float("-inf"))

        l_ij = gl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        p_fp = p.to(v_desc_0.dtype, fp_downcast_rounding="rtz")

        gl.amd.gfx1250.tdm.async_wait(PEELED_HOT_WAIT_COUNT)
        v = v_buffer.index(iter_id % NUM_BUFFERS).load(layout=cfg.v_layout)

        k_desc_next = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=k_ptr + physical_page_k_next * stride_kp + off_k_head * stride_kh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
            strides=(stride_kn, stride_kk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.k_smem_layout)
        gl.amd.gfx1250.tdm.async_load(k_desc_next, [0, 0], k_buffer.index((iter_id + 1) % NUM_BUFFERS), pred=logical_t_3
                                      < end_k, cache_modifier=TDM_CACHE_MODIFIER)

        p_dot = gl.convert_layout(p_fp, cfg.p_layout)
        acc = gl.amd.gfx1250.wmma(p_dot, v, acc)

        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * sm_scale_dot_rcp_ln2
        p = gl.exp2(sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None])
        alpha = gl.exp2(sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled)
        m_i = m_ij

        gl.amd.gfx1250.tdm.async_wait(PEELED_HOT_WAIT_COUNT)
        k = k_buffer.index(iter_id % NUM_BUFFERS).permute([1, 0]).load(layout=cfg.k_layout)

        v_desc_next = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_ptr + physical_page_for_next_v * stride_vp + off_k_head * stride_vh,
            shape=(PAGE_BLOCK_SIZE, HEAD_SZ), strides=(stride_vn, stride_vk), block_shape=(BLOCK_N, HEAD_SZ),
            layout=cfg.v_smem_layout)
        gl.amd.gfx1250.tdm.async_load(v_desc_next, [0, 0], v_buffer.index(iter_id % NUM_BUFFERS), pred=logical_t_2
                                      < end_k, cache_modifier=TDM_CACHE_MODIFIER)
        physical_page_for_next_v = physical_page_k_next
        iter_id += 1

    epilogue_offset = (iter_id - 1) * BLOCK_N
    t_2 = epilogue_offset + 2 * BLOCK_N
    t_3 = epilogue_offset + 3 * BLOCK_N
    logical_t_2 = start_k + t_2
    logical_t_3 = start_k + t_3

    l_ij = gl.sum(p, 1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    p_fp = p.to(v_desc_0.dtype, fp_downcast_rounding="rtz")
    gl.amd.gfx1250.tdm.async_wait(PEELED_HOT_WAIT_COUNT)
    v = v_buffer.index(iter_id % NUM_BUFFERS).load(layout=cfg.v_layout)
    p_dot = gl.convert_layout(p_fp, cfg.p_layout)
    acc = gl.amd.gfx1250.wmma(p_dot, v, acc)

    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout)
    qk = gl.amd.gfx1250.wmma(q, k, qk)
    qk_mask2 = (logical_t_2 + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)))[None, :] < end_k
    qk = gl.where(qk_mask2, qk, float("-inf"))
    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    m_ij_scaled = m_ij * sm_scale_dot_rcp_ln2
    p = gl.exp2(sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None])
    alpha = gl.exp2(sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled)
    m_i = m_ij

    gl.amd.gfx1250.tdm.async_wait(1)
    k = k_buffer.index(iter_id % NUM_BUFFERS).permute([1, 0]).load(layout=cfg.k_layout)

    v_desc_3 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + physical_page_for_next_v * stride_vp + off_k_head * stride_vh, shape=(PAGE_BLOCK_SIZE, HEAD_SZ),
        strides=(stride_vn, stride_vk), block_shape=(BLOCK_N, HEAD_SZ), layout=cfg.v_smem_layout)
    gl.amd.gfx1250.tdm.async_load(v_desc_3, [0, 0], v_buffer.index(iter_id % NUM_BUFFERS), pred=logical_t_3 < end_k,
                                  cache_modifier=TDM_CACHE_MODIFIER)

    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout)
    qk = gl.amd.gfx1250.wmma(q, k, qk)
    qk_mask3 = (logical_t_3 + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)))[None, :] < end_k
    qk = gl.where(qk_mask3, qk, float("-inf"))

    l_ij = gl.sum(p, 1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    p_fp = p.to(v_desc_0.dtype, fp_downcast_rounding="rtz")
    gl.amd.gfx1250.tdm.async_wait(1)
    v = v_buffer.index((iter_id + 1) % NUM_BUFFERS).load(layout=cfg.v_layout)
    p_dot = gl.convert_layout(p_fp, cfg.p_layout)
    acc = gl.amd.gfx1250.wmma(p_dot, v, acc)

    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    m_ij_scaled = m_ij * sm_scale_dot_rcp_ln2
    p = gl.exp2(sm_scale_dot_rcp_ln2 * qk - m_ij_scaled[:, None])
    alpha = gl.exp2(sm_scale_dot_rcp_ln2 * m_i - m_ij_scaled)
    m_i = m_ij

    l_ij = gl.sum(p, 1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    p_fp = p.to(v_desc_0.dtype, fp_downcast_rounding="rtz")
    gl.amd.gfx1250.tdm.async_wait(0)
    v = v_buffer.index(iter_id % NUM_BUFFERS).load(layout=cfg.v_layout)
    p_dot = gl.convert_layout(p_fp, cfg.p_layout)
    acc = gl.amd.gfx1250.wmma(p_dot, v, acc)

    if not PEELED_SKIP_FINAL_WAIT:
        gl.amd.gfx1250.tdm.async_wait(0)

    store_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
    store_cols = gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, cfg.pv_layout))
    store_q_heads = q_head_start + store_rows
    store_mask = store_rows < GQA_BLOCK_H

    if PEELED_DIRECT_OUTPUT:
        out = acc * (1.0 / l_i[:, None])
        o_offs = (stride_mid_oz * off_z + stride_mid_oh * store_q_heads[:, None] + store_cols[None, :] *
                  stride_mid_on)
        o_mask = store_mask[:, None]
        gl.amd.gfx1250.buffer_store(out.to(mid_o_ptr.dtype.element_ty), mid_o_ptr, o_offs, mask=o_mask)
    else:
        mid_o_offs = (off_z * stride_mid_oz + store_q_heads[:, None] * stride_mid_oh + split_id * stride_mid_os +
                      store_cols[None, :] * stride_mid_on)

        casted_acc = acc.to(mid_o_ptr.dtype.element_ty)
        gl.amd.gfx1250.buffer_store(casted_acc, mid_o_ptr, mid_o_offs, mask=store_mask[:, None])

        mid_l_offs = off_z * stride_mid_lz + store_q_heads * stride_mid_lh + split_id * stride_mid_ls
        mid_m_offs = off_z * stride_mid_mz + store_q_heads * stride_mid_mh + split_id * stride_mid_ms
        gl.store(mid_l_ptr + mid_l_offs, l_i, mask=store_mask)
        gl.store(mid_m_ptr + mid_m_offs, m_i, mask=store_mask)

@gluon.jit
def attn_decode_reduce_row1_kernel(mid_o_ptr, mid_l_ptr, mid_m_ptr, out_ptr, stride_mid_oz, stride_mid_oh,
                                   stride_mid_os, stride_mid_om, stride_mid_on, stride_mid_lz, stride_mid_lh,
                                   stride_mid_ls, stride_mid_lm, stride_mid_mz, stride_mid_mh, stride_mid_ms,
                                   stride_mid_mm, stride_oz, stride_oh, stride_om, stride_on, SM_SCALE: gl.constexpr,
                                   SPLIT_FACTOR: gl.constexpr, BLOCK_M: gl.constexpr, HEAD_SZ: gl.constexpr,
                                   SEQLEN_Q: gl.constexpr, SEQLEN_K: gl.constexpr, BLOCK_N: gl.constexpr):
    # Decode specialization for SEQLEN_Q == 1. Stage1 computes a BLOCK_M tile,
    # but only row 0 is semantically live, so avoid reducing the other rows.
    off_z = gl.program_id(0)
    off_h = gl.program_id(1)

    NUM_BUFFERS: gl.constexpr = 1
    NUM_WARPS: gl.constexpr = 4
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)
    head_layout: gl.constexpr = gl.SliceLayout(0, cfg.pv_layout)

    offs_n = gl.arange(0, HEAD_SZ, layout=head_layout)
    m_global = -float("inf")
    l_global = 0.0
    acc_global = gl.zeros([HEAD_SZ], dtype=gl.float32, layout=head_layout)

    rcp_ln2 = 1.4426950408889634

    for s in range(SPLIT_FACTOR):
        off_l_base = off_z * stride_mid_lz + off_h * stride_mid_lh + s * stride_mid_ls
        off_m_base = off_z * stride_mid_mz + off_h * stride_mid_mh + s * stride_mid_ms
        off_o_base = off_z * stride_mid_oz + off_h * stride_mid_oh + s * stride_mid_os

        m_s = gl.load(mid_m_ptr + off_m_base)
        l_s = gl.load(mid_l_ptr + off_l_base)
        acc_s = gl.amd.gfx1250.buffer_load(mid_o_ptr, off_o_base + offs_n * stride_mid_on)
        acc_s = acc_s.to(gl.float32)

        m_new = gl.maximum(m_global, m_s)
        alpha = gl.exp2((m_global - m_new) * SM_SCALE * rcp_ln2)
        beta = gl.exp2((m_s - m_new) * SM_SCALE * rcp_ln2)

        l_global = l_global * alpha + l_s * beta
        acc_global = acc_global * alpha + acc_s * beta
        m_global = m_new

    out = acc_global * (1.0 / l_global)
    o_offs = stride_oz * off_z + stride_oh * off_h + offs_n * stride_on
    gl.amd.gfx1250.buffer_store(out.to(out_ptr.dtype.element_ty), out_ptr, o_offs)




@gluon.jit
def attn_decode_reduce_row1_group_kernel(src_o_ptr, src_l_ptr, src_m_ptr, dst_o_ptr, dst_l_ptr, dst_m_ptr,
                                         stride_src_oz, stride_src_oh, stride_src_os, stride_src_om,
                                         stride_src_on, stride_src_lz, stride_src_lh, stride_src_ls,
                                         stride_src_lm, stride_src_mz, stride_src_mh, stride_src_ms,
                                         stride_src_mm, stride_dst_oz, stride_dst_oh, stride_dst_os,
                                         stride_dst_om, stride_dst_on, stride_dst_lz, stride_dst_lh,
                                         stride_dst_ls, stride_dst_lm, stride_dst_mz, stride_dst_mh,
                                         stride_dst_ms, stride_dst_mm, SM_SCALE: gl.constexpr,
                                         REDUCE_GROUP_SIZE: gl.constexpr, BLOCK_M: gl.constexpr,
                                         HEAD_SZ: gl.constexpr, SEQLEN_Q: gl.constexpr,
                                         SEQLEN_K: gl.constexpr, BLOCK_N: gl.constexpr):
    # First-level decode reducer. It merges a small group of split-K partials
    # into another mid-buffer, but intentionally leaves acc unnormalized so the
    # existing final row1 reducer can merge groups with the same online formula.
    off_z = gl.program_id(0)
    off_h = gl.program_id(1)
    reduce_group_id = gl.program_id(2)

    NUM_BUFFERS: gl.constexpr = 1
    NUM_WARPS: gl.constexpr = 4
    cfg = AttentionConfig(SEQLEN_Q, SEQLEN_K, HEAD_SZ, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS)
    head_layout: gl.constexpr = gl.SliceLayout(0, cfg.pv_layout)

    offs_n = gl.arange(0, HEAD_SZ, layout=head_layout)
    m_global = -float("inf")
    l_global = 0.0
    acc_global = gl.zeros([HEAD_SZ], dtype=gl.float32, layout=head_layout)

    rcp_ln2 = 1.4426950408889634
    first_split = reduce_group_id * REDUCE_GROUP_SIZE

    for i in range(REDUCE_GROUP_SIZE):
        split_id = first_split + i
        off_l_base = off_z * stride_src_lz + off_h * stride_src_lh + split_id * stride_src_ls
        off_m_base = off_z * stride_src_mz + off_h * stride_src_mh + split_id * stride_src_ms
        off_o_base = off_z * stride_src_oz + off_h * stride_src_oh + split_id * stride_src_os

        m_s = gl.load(src_m_ptr + off_m_base)
        l_s = gl.load(src_l_ptr + off_l_base)
        acc_s = gl.amd.gfx1250.buffer_load(src_o_ptr, off_o_base + offs_n * stride_src_on)
        acc_s = acc_s.to(gl.float32)

        m_new = gl.maximum(m_global, m_s)
        alpha = gl.exp2((m_global - m_new) * SM_SCALE * rcp_ln2)
        beta = gl.exp2((m_s - m_new) * SM_SCALE * rcp_ln2)

        l_global = l_global * alpha + l_s * beta
        acc_global = acc_global * alpha + acc_s * beta
        m_global = m_new

    dst_o_base = off_z * stride_dst_oz + off_h * stride_dst_oh + reduce_group_id * stride_dst_os
    dst_l_base = off_z * stride_dst_lz + off_h * stride_dst_lh + reduce_group_id * stride_dst_ls
    dst_m_base = off_z * stride_dst_mz + off_h * stride_dst_mh + reduce_group_id * stride_dst_ms

    gl.amd.gfx1250.buffer_store(acc_global.to(dst_o_ptr.dtype.element_ty), dst_o_ptr,
                                dst_o_base + offs_n * stride_dst_on)
    gl.store(dst_l_ptr + dst_l_base, l_global)
    gl.store(dst_m_ptr + dst_m_base, m_global)


def generate_configs():
    return [
        pytest.param({
            "BATCH": 1,
            "SEQLEN_Q": 1,
            "SEQLEN_K": 256,
            "NUM_Q_HEADS": 8,
            "NUM_K_HEADS": 1,
            "HEAD_SZ": 64,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "ATTN_FN": "decode",
            "DTYPE": "bf16",
            "PAGED_DECODE": True,
            "PAGE_BLOCK_SIZE": 64,
            "DECODE_STAGE": "full-gqa-peeled-direct",
            "SPLIT_FACTOR": 1,
            "GQA_BLOCK_H": 8,
            "DECODE_NUM_WARPS": 4,
        }),
        pytest.param({
            "BATCH": 1,
            "SEQLEN_Q": 1,
            "SEQLEN_K": 512,
            "NUM_Q_HEADS": 8,
            "NUM_K_HEADS": 1,
            "HEAD_SZ": 64,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "ATTN_FN": "decode",
            "DTYPE": "bf16",
            "PAGED_DECODE": True,
            "PAGE_BLOCK_SIZE": 64,
            "DECODE_STAGE": "full-gqa",
            "SPLIT_FACTOR": 2,
            "GQA_BLOCK_H": 8,
            "DECODE_NUM_WARPS": 4,
        }),
    ]


def run_paged_decode_attention(config, q, k_cache, v_cache, block_table, o, sm_scale, verbose=True):
    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    NUM_K_HEADS = config["NUM_K_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    page_block_size = config["PAGE_BLOCK_SIZE"]
    pages_per_batch = (SEQLEN_K + page_block_size - 1) // page_block_size
    decode_num_warps = config.get("DECODE_NUM_WARPS", 4)
    decode_tdm_cache_modifier = config.get("DECODE_TDM_CACHE_MODIFIER", "")
    decode_peeled_hot_wait_count = config.get("DECODE_PEELED_HOT_WAIT_COUNT", 2)
    decode_peeled_skip_final_wait = config.get("DECODE_PEELED_SKIP_FINAL_WAIT", False)
    decode_parallel_row1_reduce = config.get("DECODE_PARALLEL_ROW1_REDUCE", False)
    decode_reduce_group_size = config.get("DECODE_REDUCE_GROUP_SIZE", 4)
    gqa_block_h = config.get("GQA_BLOCK_H")

    if SEQLEN_Q != 1:
        raise RuntimeError("paged decode requires SEQLEN_Q == 1")
    if page_block_size % BLOCK_N != 0:
        raise RuntimeError("--page-block-size must be a multiple of --block-n")
    if NUM_Q_HEADS % NUM_K_HEADS != 0:
        raise RuntimeError("--num-heads-q must be divisible by --num-heads-k")

    gqa_group_size = NUM_Q_HEADS // NUM_K_HEADS
    if gqa_block_h is None:
        gqa_block_h = min(gqa_group_size, BLOCK_M)
    if gqa_block_h > BLOCK_M:
        raise RuntimeError("--gqa-block-h must be <= --block-m")
    if gqa_group_size % gqa_block_h != 0:
        raise RuntimeError("--gqa-block-h must divide the Q heads per K/V head")

    gqa_groups_per_k_head = gqa_group_size // gqa_block_h
    num_gqa_head_groups = NUM_K_HEADS * gqa_groups_per_k_head
    decode_stage = config.get("DECODE_STAGE", "full-gqa-peeled-direct")
    if decode_stage not in ("full-gqa", "full-gqa-peeled", "full-gqa-peeled-direct"):
        raise RuntimeError(
            "paged decode supports --decode-stage full-gqa, full-gqa-peeled, or full-gqa-peeled-direct"
        )

    if config.get("SPLIT_FACTOR") is not None:
        split_factor = config["SPLIT_FACTOR"]
    elif decode_stage == "full-gqa":
        split_factor = compute_split_factor(BATCH, num_gqa_head_groups, SEQLEN_K)
    elif decode_stage == "full-gqa-peeled":
        split_factor = compute_peeled_split_factor(BATCH, num_gqa_head_groups, SEQLEN_K, BLOCK_N)
    else:
        split_factor = 1

    chunk_size = (SEQLEN_K + split_factor - 1) // split_factor
    chunk_size = ((chunk_size + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    last_chunk_size = max(SEQLEN_K - (split_factor - 1) * chunk_size, 0)
    min_tiles_per_split = (last_chunk_size + BLOCK_N - 1) // BLOCK_N

    mid_o = torch.zeros((BATCH, NUM_Q_HEADS, split_factor, BLOCK_M, HEAD_SZ), dtype=torch.float32, device=q.device)
    mid_l = torch.zeros((BATCH, NUM_Q_HEADS, split_factor, BLOCK_M), dtype=torch.float32, device=q.device)
    mid_m = torch.full(
        (BATCH, NUM_Q_HEADS, split_factor, BLOCK_M), float("-inf"), dtype=torch.float32, device=q.device
    )

    if verbose:
        print(
            f"Launching Paged GQA Decode FWD: Split Factor {split_factor}, Chunk Size {chunk_size}, "
            f"Page Block Size {page_block_size}"
        )

    def launch_row1_reduce(src_o, src_l, src_m, src_split_factor):
        if decode_parallel_row1_reduce and src_split_factor > decode_reduce_group_size:
            if src_split_factor % decode_reduce_group_size != 0:
                raise RuntimeError(
                    "--decode-parallel-row1-reduce requires split_factor divisible by --decode-reduce-group-size"
                )
            reduce_groups = src_split_factor // decode_reduce_group_size
            reduce_o = torch.zeros(
                (BATCH, NUM_Q_HEADS, reduce_groups, BLOCK_M, HEAD_SZ), dtype=torch.float32, device=q.device
            )
            reduce_l = torch.zeros(
                (BATCH, NUM_Q_HEADS, reduce_groups, BLOCK_M), dtype=torch.float32, device=q.device
            )
            reduce_m = torch.full(
                (BATCH, NUM_Q_HEADS, reduce_groups, BLOCK_M),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            reduce_group_kernel = attn_decode_reduce_row1_group_kernel[(BATCH, NUM_Q_HEADS, reduce_groups)](
                src_o,
                src_l,
                src_m,
                reduce_o,
                reduce_l,
                reduce_m,
                *src_o.stride(),
                *src_l.stride(),
                *src_m.stride(),
                *reduce_o.stride(),
                *reduce_l.stride(),
                *reduce_m.stride(),
                sm_scale,
                decode_reduce_group_size,
                BLOCK_M,
                HEAD_SZ,
                SEQLEN_Q,
                SEQLEN_K,
                BLOCK_N,
                num_warps=4,
                waves_per_eu=1,
            )
            reduce_kernel = attn_decode_reduce_row1_kernel[(BATCH, NUM_Q_HEADS, 1)](
                reduce_o,
                reduce_l,
                reduce_m,
                o,
                *reduce_o.stride(),
                *reduce_l.stride(),
                *reduce_m.stride(),
                *o.stride(),
                sm_scale,
                reduce_groups,
                BLOCK_M,
                HEAD_SZ,
                SEQLEN_Q,
                SEQLEN_K,
                BLOCK_N,
                num_warps=4,
                waves_per_eu=1,
            )
            return (reduce_group_kernel, reduce_kernel)

        reduce_kernel = attn_decode_reduce_row1_kernel[(BATCH, NUM_Q_HEADS, 1)](
            src_o,
            src_l,
            src_m,
            o,
            *src_o.stride(),
            *src_l.stride(),
            *src_m.stride(),
            *o.stride(),
            sm_scale,
            src_split_factor,
            BLOCK_M,
            HEAD_SZ,
            SEQLEN_Q,
            SEQLEN_K,
            BLOCK_N,
            num_warps=4,
            waves_per_eu=1,
        )
        return (reduce_kernel, )

    if decode_stage == "full-gqa-peeled-direct":
        if split_factor != 1:
            raise RuntimeError("--decode-stage full-gqa-peeled-direct requires split_factor == 1")
        if page_block_size != BLOCK_N:
            raise RuntimeError("--decode-stage full-gqa-peeled-direct requires page_block_size == BLOCK_N")
        if SEQLEN_K < 4 * BLOCK_N:
            raise RuntimeError("--decode-stage full-gqa-peeled-direct requires at least 4 K/V tiles")
        attn_direct = attn_decode_fwd_paged_pipeline_peeled_kernel[(BATCH, num_gqa_head_groups, 1)](
            q,
            k_cache,
            v_cache,
            block_table,
            o,
            mid_l,
            mid_m,
            *q.stride(),
            *k_cache.stride(),
            *v_cache.stride(),
            o.stride(0),
            o.stride(1),
            0,
            o.stride(2),
            o.stride(3),
            *mid_l.stride(),
            *mid_m.stride(),
            sm_scale,
            SEQLEN_Q,
            SEQLEN_K,
            BLOCK_M,
            BLOCK_N,
            HEAD_SZ,
            gqa_group_size,
            1,
            gqa_block_h,
            gqa_groups_per_k_head,
            chunk_size,
            decode_num_warps,
            pages_per_batch,
            page_block_size,
            decode_tdm_cache_modifier,
            decode_peeled_hot_wait_count,
            decode_peeled_skip_final_wait,
            True,
            num_warps=decode_num_warps,
            waves_per_eu=1,
        )
        return (attn_direct, )

    if decode_stage == "full-gqa-peeled":
        if page_block_size != BLOCK_N:
            raise RuntimeError("--decode-stage full-gqa-peeled requires page_block_size == BLOCK_N")
        if min_tiles_per_split < 4:
            raise RuntimeError("--decode-stage full-gqa-peeled requires at least 4 K/V tiles per split")
        attn_stage1 = attn_decode_fwd_paged_pipeline_peeled_kernel[
            (BATCH, num_gqa_head_groups, split_factor)
        ](
            q,
            k_cache,
            v_cache,
            block_table,
            mid_o,
            mid_l,
            mid_m,
            *q.stride(),
            *k_cache.stride(),
            *v_cache.stride(),
            *mid_o.stride(),
            *mid_l.stride(),
            *mid_m.stride(),
            sm_scale,
            SEQLEN_Q,
            SEQLEN_K,
            BLOCK_M,
            BLOCK_N,
            HEAD_SZ,
            gqa_group_size,
            split_factor,
            gqa_block_h,
            gqa_groups_per_k_head,
            chunk_size,
            decode_num_warps,
            pages_per_batch,
            page_block_size,
            decode_tdm_cache_modifier,
            decode_peeled_hot_wait_count,
            decode_peeled_skip_final_wait,
            False,
            num_warps=decode_num_warps,
            waves_per_eu=1,
        )
    else:
        attn_stage1 = attn_decode_fwd_paged_gqa_kernel[(BATCH, num_gqa_head_groups, split_factor)](
            q,
            k_cache,
            v_cache,
            block_table,
            mid_o,
            mid_l,
            mid_m,
            *q.stride(),
            *k_cache.stride(),
            *v_cache.stride(),
            *mid_o.stride(),
            *mid_l.stride(),
            *mid_m.stride(),
            sm_scale,
            SEQLEN_Q,
            SEQLEN_K,
            BLOCK_M,
            BLOCK_N,
            HEAD_SZ,
            gqa_group_size,
            gqa_block_h,
            gqa_groups_per_k_head,
            split_factor,
            chunk_size,
            decode_num_warps,
            pages_per_batch,
            page_block_size,
            decode_tdm_cache_modifier,
            num_warps=decode_num_warps,
            waves_per_eu=1,
        )

    return (attn_stage1, *launch_row1_reduce(mid_o, mid_l, mid_m, split_factor))


def run_attention(config, check=True, profile=False):
    BATCH = config["BATCH"]
    SEQLEN_Q = config["SEQLEN_Q"]
    SEQLEN_K = config["SEQLEN_K"]
    NUM_Q_HEADS = config["NUM_Q_HEADS"]
    NUM_K_HEADS = config["NUM_K_HEADS"]
    HEAD_SZ = config["HEAD_SZ"]
    page_block_size = config.get("PAGE_BLOCK_SIZE") or config["BLOCK_N"]

    if config.get("ATTN_FN", "decode") != "decode":
        raise RuntimeError("This file only supports paged decode attention")
    if not config.get("PAGED_DECODE", True):
        raise RuntimeError("This file only supports paged K/V caches")
    if SEQLEN_Q != 1:
        raise RuntimeError("Paged decode requires SEQLEN_Q == 1")
    if NUM_Q_HEADS % NUM_K_HEADS != 0:
        raise RuntimeError("--num-heads-q must be divisible by --num-heads-k")
    if SEQLEN_K % page_block_size != 0:
        raise RuntimeError("--seqlen-k must be divisible by --page-block-size")

    dtype_name = config.get("DTYPE", "bf16")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    torch.manual_seed(0)

    q = torch.randn(BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ, dtype=dtype)
    o = torch.zeros(q.shape, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_SZ**0.5)

    gqa_group_size = NUM_Q_HEADS // NUM_K_HEADS
    pages_per_batch = SEQLEN_K // page_block_size
    total_pages = BATCH * pages_per_batch
    block_table = torch.arange(total_pages, dtype=torch.int32).reshape(BATCH, pages_per_batch)

    if check:
        k = torch.randn(BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ, dtype=dtype)
        v = torch.randn(BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ, dtype=dtype)
        k_cache = k.reshape(BATCH, NUM_K_HEADS, pages_per_batch, page_block_size, HEAD_SZ)
        v_cache = v.reshape(BATCH, NUM_K_HEADS, pages_per_batch, page_block_size, HEAD_SZ)
        k_cache = k_cache.permute(0, 2, 1, 3, 4).reshape(total_pages, NUM_K_HEADS, page_block_size,
                                                         HEAD_SZ).contiguous()
        v_cache = v_cache.permute(0, 2, 1, 3, 4).reshape(total_pages, NUM_K_HEADS, page_block_size,
                                                         HEAD_SZ).contiguous()
        k_ref = k.repeat_interleave(gqa_group_size, dim=1)
        v_ref = v.repeat_interleave(gqa_group_size, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref)
    else:
        k_cache = torch.randn(total_pages, NUM_K_HEADS, page_block_size, HEAD_SZ, dtype=dtype)
        v_cache = torch.randn(total_pages, NUM_K_HEADS, page_block_size, HEAD_SZ, dtype=dtype)

    q = q.cuda()
    o = o.cuda()
    k_cache = k_cache.cuda()
    v_cache = v_cache.cuda()
    block_table = block_table.cuda()

    def kernel_fn():
        return run_paged_decode_attention(
            config, q, k_cache, v_cache, block_table, o, sm_scale, verbose=not profile
        )

    attn_kernel = kernel_fn()
    ms = triton.testing.do_bench(kernel_fn) if profile else None

    torch.cuda.synchronize()
    if check:
        torch.testing.assert_close(o.cpu(), ref, rtol=0.004, atol=0.004, check_dtype=False)
    if profile:
        return attn_kernel, ms
    return attn_kernel


@pytest.mark.parametrize("config", generate_configs())
def test_attention(config):
    run_attention(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GFX1250 paged GQA decode attention")
    parser.add_argument("-b", type=int, default=64, help="batch size")
    parser.add_argument("--seqlen-q", type=int, default=1, help="Q sequence length; must be 1")
    parser.add_argument("--seqlen-k", type=int, default=2048, help="K/V sequence length")
    parser.add_argument("--num-heads-q", type=int, default=64, help="number of Q heads")
    parser.add_argument("--num-heads-k", type=int, default=8, help="number of K/V heads")
    parser.add_argument("--head-size", type=int, default=64, help="Q/K/V head size")
    parser.add_argument("--block-m", type=int, default=16, help="BLOCK_M size")
    parser.add_argument("--block-n", type=int, default=64, help="BLOCK_N size")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16", help="Q/K/V dtype")
    parser.add_argument("--no-check", action="store_true", help="Skip correctness reference and D2H copy-back")
    parser.add_argument("--profile", action="store_true",
                        help="Benchmark the selected attention path with triton.testing.do_bench")
    parser.add_argument("--page-block-size", type=int, default=None,
                        help="Paged decode KV page size; defaults to --block-n")
    parser.add_argument("--decode-stage", type=str,
                        choices=["full-gqa", "full-gqa-peeled", "full-gqa-peeled-direct"],
                        default="full-gqa-peeled-direct",
                        help="Grouped-GQA decode implementation")
    parser.add_argument("--gqa-block-h", type=int, default=None,
                        help="Q heads per grouped-GQA stage1 program; defaults to Q heads per K/V head")
    parser.add_argument("--split-factor", type=int, default=None,
                        help="Override decode split factor; default uses compute_split_factor")
    parser.add_argument("--decode-num-warps", type=int, choices=[4, 8], default=4,
                        help="Number of warps for decode stage1")
    parser.add_argument("--decode-tdm-cache-modifier", type=str, choices=["", ".ca", ".cg", ".cs", ".cv"], default="",
                        help="TDM async-load cache modifier for decode")
    parser.add_argument("--decode-peeled-hot-wait-count", type=int, default=2,
                        help="Steady-state K/V LDS wait count for peeled decode")
    parser.add_argument("--decode-peeled-skip-final-wait", action="store_true",
                        help="Skip final TDM async_wait(0) after the last PV in peeled decode")
    parser.add_argument("--decode-parallel-row1-reduce", action="store_true",
                        help="Reduce split-K row1 partials in parallel groups before the final row1 reduce")
    parser.add_argument("--decode-reduce-group-size", type=int, default=4,
                        help="Number of split-K partials per first-level row1 reduce group")
    args = parser.parse_args()
    config = {
        "BATCH": args.b,  #
        "SEQLEN_Q": args.seqlen_q, "SEQLEN_K": args.seqlen_k,  #
        "NUM_Q_HEADS": args.num_heads_q, "NUM_K_HEADS": args.num_heads_k,  #
        "HEAD_SZ": args.head_size,  #
        "BLOCK_M": args.block_m, "BLOCK_N": args.block_n,  #
        "ATTN_FN": "decode",  #
        "DTYPE": args.dtype,  #
        "PAGED_DECODE": True,  #
        "PAGE_BLOCK_SIZE": args.page_block_size or args.block_n,  #
        "DECODE_STAGE": args.decode_stage,  #
        "SPLIT_FACTOR": args.split_factor,  #
        "GQA_BLOCK_H": args.gqa_block_h,  #
        "DECODE_NUM_WARPS": args.decode_num_warps,  #
        "DECODE_TDM_CACHE_MODIFIER": args.decode_tdm_cache_modifier,  #
        "DECODE_PEELED_HOT_WAIT_COUNT": args.decode_peeled_hot_wait_count,  #
        "DECODE_PEELED_SKIP_FINAL_WAIT": args.decode_peeled_skip_final_wait,  #
        "DECODE_PARALLEL_ROW1_REDUCE": args.decode_parallel_row1_reduce,  #
        "DECODE_REDUCE_GROUP_SIZE": args.decode_reduce_group_size,  #
    }
    print(config)
    result = run_attention(config, check=not args.no_check, profile=args.profile)
    if args.profile:
        attn_kernel, ms = result
        traffic_bytes = estimate_attention_traffic_bytes(config)
        bandwidth_gbps = traffic_bytes / ms / 1.0e6
        print(f"triton.testing.do_bench: {ms:.6f} ms")
        print(f"estimated logical traffic: {traffic_bytes / 1.0e9:.6f} GB")
        print(f"estimated logical bandwidth: {bandwidth_gbps:.3f} GB/s")
    else:
        attn_kernel = result
    [static_profile(kernel) for kernel in attn_kernel]
