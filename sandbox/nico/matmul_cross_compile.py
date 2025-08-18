import torch
import triton
from triton.backends.compiler import GPUTarget

from matmul_base import matmul_kernel

BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32

src = triton.compiler.ASTSource(
  fn=matmul_kernel,
  signature={
    "a_ptr": "*fp32",
    "b_ptr": "*fp32",
    "c_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "i32",
    "stride_bk": "i32",
    "stride_bn": "i32",
    "stride_cm": "i32",
    "stride_cn": "i32",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr"
  },
  constexprs={
    "BLOCK_SIZE_M": BLOCK_SIZE_M,
    "BLOCK_SIZE_N": BLOCK_SIZE_N,
    "BLOCK_SIZE_K": BLOCK_SIZE_K,
  }
)
  
# Note: this does not cross-compile as ptxas seems needed to be installed on the host machine.
# Did not investigate further as to how to stop at PTX atm.
# target = GPUTarget("cuda", 80, 32)

# AMDGPU backend is OSS so we can get asm.
target = GPUTarget("hip", 'gfx942', 64) # For AMD HIP gfx942 (e.g., MI300X GPU)

# Create backend and parse options to specify num_warps
backend = triton.compiler.make_backend(target)
options = backend.parse_options({"num_warps": 1, "num_stages": 3})

output = triton.compile(src, target=target, options=options.__dict__)
print(output.asm["amdgcn"])
