import os
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from triton.experimental.gluon import language as gl

from gluon_memcpy_benchmark import get_throughput, memcpy_1d_impl, memcpy_2d_persistent_impl
from dist_sync import do_dist_synchronized_bench, setup, teardown

torch.manual_seed(0)
warp_size = 64
num_xcds = 8
num_cus = num_xcds * 38 * 2
# num_cus = num_xcds
masked = False
use_buffer_instructions = False

xnumel = 2 << 27
# Note: for some reason, sizes >= 2 << 28 result in errors in the 2D case with buffer instructions.
# This is despite our computing the base pointer and ensuring the offsets fit
# 2^28 -> 1 element is wrong.
# 2^29 -> 50% elements are wrong.
# 2^30 -> 75% elements are wrong.
# 2^31 -> 87.5% elements are wrong.
# 2^32 -> not enough memory for all tensors

def bench_memcpy_1d_impl(rank, world_size, tensor_size, dtype):
    device = setup(rank, world_size)

    input = torch.randn(tensor_size, dtype=dtype).to(device)
    output = torch.empty_like(input).to(device)

    def do(XBLOCK, vector_size, num_warps, check=True):
        max_vector_size = XBLOCK // (num_warps * warp_size)
        unroll_factor_per_warp = XBLOCK // (num_warps * warp_size * vector_size)
        if unroll_factor_per_warp < 1:
            return

        # Avoid loads / stores that require more than 16B / thread
        if vector_size * torch.tensor([], dtype=dtype).element_size() > 16:
            return

        # Avoid overflowing the buffer instruction offset
        if use_buffer_instructions and XBLOCK > 2 ** 12:
            return

        config_str = f"XBLOCK({XBLOCK}), num_warps({num_warps}) vector_size({vector_size})" + \
            f" (max_vector_size {max_vector_size} unroll_factor_per_warp {unroll_factor_per_warp})"
            
        layout = gl.BlockedLayout([vector_size], [warp_size], [num_warps], [0])
        impl = partial(
            memcpy_1d_impl,
            num_warps=num_warps,
            XBLOCK=XBLOCK,
            layout=layout,
            masked=masked,
            use_buffer_instructions=use_buffer_instructions,
        )

        if check:
            impl(input, output)
            torch.testing.assert_close(input, output)

        def bench_wrapper(device, impl, input, output, str_to_print):
            fn = lambda: impl(input, output)
            ms = do_dist_synchronized_bench(fn, device, warmup=10, rep=100, return_mode="median")
            throughput = get_throughput(input, ms)
            print(f"device {device} -> {throughput:.3f} TB/s in {ms:.3f}ms {str_to_print}\t")

        bench_wrapper(device, impl, input, output, config_str)

    bench_one = True
    if bench_one:
        # 4TF/s on MI300x
        XBLOCK, num_warps, vector_size = 2048, 4, 4
        do(XBLOCK, vector_size, num_warps, check=False)
    else:
        # we are voluntarily testing some outliers here where the vector size is too big
        # to make sense perf-wise
        for XBLOCK in (128, 256, 512, 1024, 2048, 4096, 8192):
            for vector_size in (1, 2, 4, 8, 16, 32):
                for num_warps in (1, 2, 4, 8):
                    do(XBLOCK, vector_size, num_warps, check=False)

    teardown(rank, device)


def bench_memcpy_2d_impl(rank, world_size, tensor_size, dtype):
    device = setup(rank, world_size)

    input = torch.randn(tensor_size, dtype=dtype).to(device)
    output = torch.empty_like(input).to(device)

    def do(BLOCK_SIZE_N, vector_size, num_warps, num_cus, row_cu_shift, check=True):
        max_vector_size = BLOCK_SIZE_N // (num_warps * warp_size)
        unroll_factor_per_warp = BLOCK_SIZE_N // (num_warps * warp_size * vector_size)
        if unroll_factor_per_warp < 1:
            # print(f"unroll_factor_per_warp {unroll_factor_per_warp} < 1 -> SKIP")
            return
        
        # Avoid loads / stores that require more than 16B / thread
        if vector_size * torch.tensor([], dtype=dtype).element_size() > 16:
            # print(f"vector_size * torch.tensor([], dtype=dtype).element_size() {vector_size * torch.tensor([], dtype=dtype).element_size()} > 16 -> SKIP")
            return

        # Avoid overflowing the buffer instruction offset
        if use_buffer_instructions and BLOCK_SIZE_N > 2 ** 12:
            return

        config_str = f"BLOCK_SIZE_N({BLOCK_SIZE_N}), num_warps({num_warps}) vector_size({vector_size})" + \
            f" (max_vector_size {max_vector_size} unroll_factor_per_warp {unroll_factor_per_warp} row_cu_shift {row_cu_shift})"
        # print(config_str)

        layout = gl.BlockedLayout([vector_size], [warp_size], [num_warps], [0])
        impl = partial(
            memcpy_2d_persistent_impl,
            num_warps=num_warps,
            num_cus=num_cus,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            layout=layout,
            masked=masked,
            use_buffer_instructions=use_buffer_instructions,
            row_cu_shift=row_cu_shift,
        )

        if check:
            impl(input, output)
            torch.testing.assert_close(input, output)

        def bench_wrapper(device, impl, input, output, str_to_print):
            fn = lambda: impl(input, output)
            ms = do_dist_synchronized_bench(fn, device, warmup=10, rep=100, return_mode="median")
            throughput = get_throughput(input, ms)
            print(f"device {device} -> {throughput:.3f} TB/s in {ms:.3f}ms {str_to_print}\t")

        bench_wrapper(device, impl, input, output, config_str)

    bench_one = True
    if bench_one:
        # Example: 2D memcpy with reasonable parameters
        BLOCK_SIZE_N, num_warps, vector_size = 4096, 4, 2
        row_cu_shift = 0
        # for row_cu_shift in range(1):
        do(BLOCK_SIZE_N, vector_size, num_warps, num_cus, row_cu_shift=row_cu_shift, check=False)
    else:
        for BLOCK_SIZE_N in (512, 1024, 2048, 4096, 8192):
            # Avoid unmasked overflows
            if not masked and input.shape[1] % BLOCK_SIZE_N != 0:
                continue
            for vector_size in (1, 2, 4, 8, 16, 32):
                for num_warps in (1, 2, 4, 8):
                    row_cu_shift = 0
                    # for row_cu_shift in range(num_xcds):
                    shift = (row_cu_shift * torch.randint(0, 101, (1,)).item()) % num_cus
                    do(BLOCK_SIZE_N, vector_size, num_warps, num_cus, row_cu_shift=shift, check=False)

    teardown(rank, device)

    
def bench_memcpy_1d():
    world_size = min(8, torch.cuda.device_count())
    
    mp.spawn(
        bench_memcpy_1d_impl,
        args=(world_size, (xnumel, ), torch.float32),
        nprocs=world_size,
        join=True
    )

    
def bench_memcpy_2d():
    world_size = min(8, torch.cuda.device_count())
    
    M = 4096
    N = xnumel // M
    mp.spawn(
        bench_memcpy_2d_impl,
        args=(world_size, (M, N), torch.float32),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    bench_memcpy_1d()
    bench_memcpy_2d()