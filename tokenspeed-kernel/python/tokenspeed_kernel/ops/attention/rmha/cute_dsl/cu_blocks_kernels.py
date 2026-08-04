# Copyright (c) 2026, Colfax International.

# Ported verbatim from the sheared-bias FA4 fork
# (flash_attn/cute/cu_blocks_kernels.py); only the import block is adapted.

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32

from .fmha_bias_helper import get_batch_from_cu_tensor


class CuSeqlensToBlocksKernel:
    def __init__(
        self,
        tile: int = 128,
        num_threads: int = 1024,
        seqlen_multiple: int = 1,
    ):
        self.tile = tile
        self.num_threads = num_threads
        assert num_threads % 32 == 0
        self.num_warps = num_threads // cute.arch.WARP_SIZE
        self.seqlen_multiple = seqlen_multiple

    @cute.jit
    def __call__(
        self,
        mCuBlocks: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        @cute.struct
        class SharedStorage:
            warp_block_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mCuBlocks,
            mCuSeqlens,
            SharedStorage,
        ).launch(
            grid=[1, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCuBlocks: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        batch_size = mCuBlocks.shape[0] - 1
        batch_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_block_count = storage.warp_block_count.get_tensor(
            cute.make_layout(self.num_warps)
        )

        if batch_idx == 0:
            mCuBlocks[0] = 0

        seqlen = Int32(0)
        if batch_idx < batch_size:
            seqlen = mCuSeqlens[batch_idx + 1] - mCuSeqlens[batch_idx]
        seqlen *= self.seqlen_multiple
        num_blocks = (seqlen + self.tile - 1) // self.tile

        total_blocks_for_batch = num_blocks
        for delta in (1, 2, 4, 8, 16):
            other = cute.arch.shuffle_sync_up(
                total_blocks_for_batch, delta, mask_and_clamp=0
            )
            if lane_idx >= delta:
                total_blocks_for_batch += other

        if lane_idx == 31:
            warp_block_count[warp_idx] = total_blocks_for_batch

        cute.arch.sync_threads()

        if warp_idx * 32 < batch_size:
            for idx in cutlass.range(warp_idx):
                total_blocks_for_batch += warp_block_count[idx]

            if batch_idx < batch_size:
                mCuBlocks[batch_idx + 1] = total_blocks_for_batch


class CuBlocksToBatchKernel:
    def __init__(
        self,
    ):
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mCuTotalBlocks: cute.Tensor,
        mBlocksToBatchIdx: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        num_blocks = mBlocksToBatchIdx.shape[0]

        self.kernel(
            mCuTotalBlocks,
            mBlocksToBatchIdx,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCuTotalBlocks: cute.Tensor,
        mBlocksToBatchIdx: cute.Tensor,
    ):
        block_idx = cute.arch.block_idx()[0]
        batch_idx = get_batch_from_cu_tensor(block_idx, mCuTotalBlocks)
        mBlocksToBatchIdx[block_idx] = batch_idx
