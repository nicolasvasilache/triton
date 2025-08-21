import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gluon_memcpy_base import memcpy_1d_kernel, memcpy_2d_persistent_kernel

torch.manual_seed(0)
warp_size = 64
num_xcds = 8
num_cus = num_xcds * 38
# num_cus = num_xcds
masked = False
use_buffer_instructions = True

xnumel = 2 << 27
# Note: for some reason, sizes >= 2 << 28 result in errors in the 2D case with buffer instructions.
# This is despite our computing the base pointer and ensuring the offsets fit
# 2^28 -> 1 element is wrong.
# 2^29 -> 50% elements are wrong.
# 2^30 -> 75% elements are wrong.
# 2^31 -> 87.5% elements are wrong.
# 2^32 -> not enough memory for all tensors

def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size()) / (1024 ** 4)
    return tbytes / (ms * 1e-3)

def bench_wrapper(impl, input, output, str_to_print):
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn, warmup=10, rep=100)
    throughput = get_throughput(input, ms)
    print(f"{throughput:.3f} TB/s in {ms:.3f}ms {str_to_print}\t")

def memcpy_1d_impl(
        input,
        output,
        num_warps,
        XBLOCK,
        layout,
        masked: bool = True,
        use_buffer_instructions: bool = True,
    ):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid]( # type: ignore
        input,
        output,
        xnumel,
        XBLOCK,
        layout,
        masked,
        use_buffer_instructions,
        # JIT compile-time configuration special variable
        num_warps=num_warps
    )
    return compiled_kernel

def memcpy_2d_persistent_impl(
        input,
        output,
        num_warps,
        num_cus,
        BLOCK_SIZE_N,
        layout,
        masked: bool = True,
        use_buffer_instructions: bool = True,
        row_cu_shift: int = 0
    ):
    """
    2D memcpy implementation using a persistent kernel.
    """
    M, N = input.shape
    grid = (num_cus,)
    compiled_kernel = memcpy_2d_persistent_kernel[grid](  # type: ignore
        input,
        output,
        M,
        N,
        input.stride(0),
        output.stride(0),
        input.stride(1),
        output.stride(1),
        BLOCK_SIZE_N,
        layout,
        masked,
        use_buffer_instructions,
        num_cus,
        row_cu_shift,
        # JIT compile-time configuration special variable
        num_warps=num_warps
    )
    return compiled_kernel

def benchmark_memcpy_1d():
    print("vector_size vs. Throughput")
    print("================")

    dtype = torch.float32
    input = torch.randn(xnumel, device="cuda", dtype=dtype)
    output = torch.empty_like(input)

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

        bench_wrapper(impl, input, output, config_str)

    bench_one = False
    if bench_one:
        # 4TF/s on MI300x
        XBLOCK, num_warps, vector_size = 2048, 4, 4
        do(XBLOCK, vector_size, num_warps, check=True)
    else:
        # we are voluntarily testing some outliers here where the vector size is too big
        # to make sense perf-wise
        for XBLOCK in (128, 256, 512, 1024, 2048, 4096, 8192):
            for vector_size in (1, 2, 4, 8, 16, 32):
                for num_warps in (1, 2, 4, 8):
                    do(XBLOCK, vector_size, num_warps, check=True)


def benchmark_memcpy_2d():
    print("vector_size vs. Throughput (2D)")
    print("================")

    M = 4096
    N = xnumel // M
    dtype = torch.float32
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    output = torch.empty_like(input)

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

        bench_wrapper(impl, input, output, config_str)

    bench_one = False
    if bench_one:
        # Example: 2D memcpy with reasonable parameters
        BLOCK_SIZE_N, num_warps, vector_size = 4096, 4, 2
        row_cu_shift = 0
        # for row_cu_shift in range(1):
        do(BLOCK_SIZE_N, vector_size, num_warps, num_cus, row_cu_shift=row_cu_shift, check=False)
    else:
        for BLOCK_SIZE_N in (512, 1024, 2048, 4096, 8192):
            # Avoid unmasked overflows
            if not masked and N % BLOCK_SIZE_N != 0:
                continue
            for vector_size in (1, 2, 4, 8, 16, 32):
                for num_warps in (1, 2, 4, 8):
                    row_cu_shift = 0
                    # for row_cu_shift in range(num_xcds):
                    shift = (row_cu_shift * torch.randint(0, 101, (1,)).item()) % num_cus
                    do(BLOCK_SIZE_N, vector_size, num_warps, num_cus, row_cu_shift=shift, check=True)
  
benchmark_memcpy_1d()
benchmark_memcpy_2d()