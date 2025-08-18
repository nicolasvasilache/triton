import torch
import triton
from triton.backends.compiler import GPUTarget

from matmul_base import matmul_kernel

BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 128, 128, 32, 1

def matmul_impl(a, b, c):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.
    def grid(META): 
      res = (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
      print(f"Grid: {res}")
      return res
    return matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_warps=1,
        num_stages=3,
    )

dump_asm = True
def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    kernel = matmul_impl(a, b, c)
    if dump_asm:
      print(kernel.asm["amdgcn"])

    return c

# Allocate Tensors
M, N, K = 8192, 8192, 8192
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.empty((M, N), device=A.device, dtype=torch.float16)
fn = lambda: matmul_impl(A, B, C)
ms = triton.testing.do_bench(fn, warmup=5, rep=10)
flops_per_ms = (M * N * K * 2) / ms
tflops_per_s = flops_per_ms / (1024 ** (4 - 1))
print(f"{tflops_per_s} TFlops/s")

# Compare against torch.matmul with torch.assert_close
C_torch = torch.matmul(A, B)
torch.testing.assert_close(C, C_torch, rtol=1e-3, atol=1e-3)
