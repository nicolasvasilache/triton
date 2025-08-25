import triton

from triton.experimental.gluon._runtime import GluonASTSource
from triton.backends.compiler import GPUTarget
from triton.tutorials.gluon._03_async_copy import memcpy_1d_cpasync_kernel

XBLOCK = 64

src = GluonASTSource(
  fn=memcpy_1d_cpasync_kernel,
  signature={
    "in_ptr": "*fp32",
    "out_ptr": "*fp32",
    "xnumel": "i32",
    "XBLOCK": "constexpr",
    "threads_per_warp": "constexpr",
  },
  constexprs={
    "XBLOCK": XBLOCK,
    "threads_per_warp": 64,
  }
)

# AMDGPU backend is OSS so we can get asm.
target = GPUTarget("hip", 'gfx942', 64) # For AMD HIP gfx942 (e.g., MI300X GPU)

# Create backend and parse options to specify num_warps
backend = triton.compiler.make_backend(target)
options = backend.parse_options({"num_warps": 4, "num_stages": 3})

output = triton.compile(src, target=target, options=options.__dict__)
print(output.asm["amdgcn"])
