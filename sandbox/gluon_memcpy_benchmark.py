import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gluon_memcpy_base import memcpy_1d_impl

def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size()) / (1024 ** 4)
    return tbytes / (ms * 1e-3)

def bench_memcpy(impl, xnumel, dtype):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda", dtype=dtype)
    output = torch.empty_like(input)
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn, warmup=1, rep=1)
    return compiled_kernel, get_throughput(input, ms), ms

def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps, masked: bool = True):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, masked) # type: ignore
    return compiled_kernel


if __name__ == "__main__":
    print("vector_size vs. Throughput")
    print("================")

    warp_size = 64
    masked = False

    def do(XBLOCK, vector_size, num_warps, check=True):
        max_vector_size = XBLOCK // (num_warps * warp_size)
        unroll_factor_per_warp = XBLOCK // (num_warps * warp_size * vector_size)
        if unroll_factor_per_warp < 1:
            return

        kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps, masked=masked)
        layout = gl.BlockedLayout([vector_size], [warp_size], [num_warps], [0])
        impl = partial(kernel, layout=layout)

        dtype = torch.float32
        xnumel = 2 << 30
        if check:
            input = torch.randn(xnumel, device="cuda", dtype=dtype)
            output = torch.empty_like(input)
            impl(input, output)
            torch.testing.assert_close(input, output)

        compiled_kernel, throughput, ms = bench_memcpy(impl, xnumel, dtype)
        print(f"vector_size={vector_size:<9} {throughput:.9f} TB/s\t" + f"XBLOCK, num_warps, vector_size = {XBLOCK}, {num_warps}, {vector_size} (max_vector_size {max_vector_size} unroll_factor_per_warp {unroll_factor_per_warp})")

        import gc
        gc.collect()

    bench_one = True
    if bench_one:
        # 4TF/s on MI300x
        XBLOCK, num_warps, vector_size = 2048, 8, 1
        do(XBLOCK, vector_size, num_warps, check=False)
    else:
        # we are voluntarily testing some outliers here where the vector size is too big
        # to make sense perf-wise
        for XBLOCK in (128, 256, 512, 1024, 2048, 4096, 8192):
            for vector_size in (1, 2, 4, 8, 16, 32):
                for num_warps in (1, 2, 4, 8):
                    do(XBLOCK, vector_size, num_warps)
