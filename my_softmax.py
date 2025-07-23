

import math
from typing import Type
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.torch as cutlass_torch


# For our softmax, we will do softmax over (M, N) vectors.
# What will this concretely look like?

# We will parameterize the shape of the kernel over N.
# We will determine how many threads we will use based on the shape of the kernel.
# We will use vectorized loads to load from the GMem->Smem.

# We will then do an exponentiation on each thread.
# This means that each thread will get a chunk.

# n_blocks = N // num_threads
# for i in range(n_blocks):
#   load from GMem->Smem
#   do exponentiation
#   have an accumulator per thread.
#   add it all up within the thread.
#   do a warp-level reduction for the final sum over the N row (on each warp)
#   this produces one value per warp.
#   do a block-level reduction for the final sum over the M rows (on each block)
class RowMajorSoftmax():
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
    ):
        self.dtype = dtype
        self.N = N

    def create_tv_layout(self):
        copy_bits = 128
        vecsize = copy_bits // self.dtype.width
        # how will each thread deal with a chunk?
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        num_threads = 128
        threads_per_row = 64
        # vec size
        assert num_threads % cute.arch.WARP_SIZE == 0
        # how should we split this work?
        tiler_mn = ()

        blkN = threads_per_row * vecsize

        num_blocks_N = cute.ceil_div(self.N, blkN)
        cols_per_block = num_threads // threads_per_row

        tiler_mn = (cols_per_block, blkN * num_blocks_N)

        thread_layout = (threads_per_row, cols_per_block)
        value_layout = (vecsize, num_blocks_N)

        # you're going to receive a row-major matrix.
        # so you need to adapt to that.
        tv_layout = cute.make_layout(
            (thread_layout, value_layout),
            # it is 1 on the column here.
            stride=(
                (1, threads_per_row * vecsize),
                (threads_per_row, threads_per_row * vecsize * num_blocks_N),
            )
        )
        return tiler_mn, tv_layout

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tv_layout: cute.Layout,
    ):
        # consts
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # smem
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(mX.element_type, cute.make_layout(tiler_mn), byte_alignment=16)
        # copy to smem
        mX_block_local = cute.local_tile(mX, tiler_mn, (bidx, 0))
        mO_block_local = cute.local_tile(mO, tiler_mn, (bidx, 0))
        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        tXsX = thr_copy_X.partition_D(sX)
        tXgX = thr_copy_X.partition_S(mX_block_local)
        cute.copy(copy_atom_load_X, tXgX, tXsX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # start doing a warp-level reduction.
        tXrX: cute.Tensor = cute.make_fragment_like(tXsX)
        tXrO: cute.Tensor = cute.make_fragment_like(tXsX)
        cute.autovec_copy(tXsX, tXrX)
        x: cute.TensorSSA = tXrX.load().to(cute.Float32)
        # exponentiate then get the sum
        log2_e = math.log2(math.e)
        x_exp  = cute.math.exp2(x * log2_e)
        thread_sum = x_exp.reduce(cute.ReductionOp.ADD, 0.0, 0)
        # reduce across the warp
        for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
            thread_sum += cute.arch.shuffle_sync_bfly(thread_sum, 1 << i)
        divided_x_exp = x_exp / thread_sum
        tXrO.store(divided_x_exp)
        # store the results back to gmem.
        copy_atom_store_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mO.element_type, num_bits_per_copy=128)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)
        tXgO = thr_copy_O.partition_D(mO_block_local)
        cute.copy(copy_atom_store_O, tXrO, tXgO)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + num_warps * (cutlass.Float32.width // 8)
            + (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        # input
        mX: cute.Tensor,
        # output
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # block-level tiling (apply to get the logical tensor to operate on)
        tiler_mn_check = (1, mX.shape[1])
        # each thread will get a chunk of 32 elements.
        num_blocks = mX.shape[1] // 32
        thr_layout = cute.make_layout((1, 32), stride=(1, 1))
        val_layout = cute.make_layout((1, num_blocks), stride=(32, 1))

        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        assert tiler_mn == tiler_mn_check, f"tiler_mn {tiler_mn} != tiler_mn_check {tiler_mn_check}"

        # the tiler_mn captures the over-all shape of your tv layout.
        # so once you have this tv-layout on the warp-level, you should split the
        # block by it.

        self.kernel(mX, mO, tiler_mn, tv_layout).launch(
            grid=[mX.shape[0], 1, 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            cluster=[1, 1, 1],
            smem=self._smem_size_in_bytes(tiler_mn, 1),
            stream=stream,
        )

if __name__ == "__main__":
    M, N = 512, 32768
    softmax = RowMajorSoftmax(cutlass.Float32, N)
    torch_tensor = torch.randn(M, N, dtype=torch.float32, device="cuda")
    torch_output_tensor = torch.empty_like(torch_tensor)
    input_tensor = cutlass_torch.from_dlpack(torch_tensor, assumed_align=16)
    output_tensor = cutlass_torch.from_dlpack(torch_output_tensor, assumed_align=16)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = cute.compile(
        softmax,
        input_tensor,
        output_tensor,
        current_stream,
    )

    import time
    time_start = time.time()
    torch.cuda.synchronize()
    compiled_kernel(
        input_tensor,
        output_tensor,
        current_stream,
    )
    torch.cuda.synchronize()
    time_end = time.time()

    # Reference torch implementation
    time_start_torch = time.time()
    torch.cuda.synchronize()
    torch_output = torch.softmax(torch_tensor, dim=1)
    torch.cuda.synchronize()
    time_end_torch = time.time()

    print(f"Time taken to run cute softmax: {time_end - time_start} seconds")
    print(f"Time taken to run torch softmax: {time_end_torch - time_start_torch} seconds")

    print("Are results close?", torch.allclose(torch_output_tensor, torch_output, atol=1e-5, rtol=1e-4))
