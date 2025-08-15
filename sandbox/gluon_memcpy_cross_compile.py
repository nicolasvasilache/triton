import triton
from triton.backends.compiler import GPUTarget
import triton.experimental
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from gluon_memcpy_base import memcpy_1d_kernel

warp_size = 64
XBLOCK, num_warps, vector_size = 2048, 2, 4
layout = gl.BlockedLayout([vector_size], [warp_size], [num_warps], [0])
masked = False

# Create Gluon AST source for cross-compilation
src = GluonASTSource(
    fn=memcpy_1d_kernel,
    signature={
        "in_ptr": "*fp32",
        "out_ptr": "*fp32", 
        "xnumel": "i32",
        "XBLOCK": "constexpr",
        "layout": "constexpr",
        "masked": "constexpr"
    },
    constexprs={
        "XBLOCK": XBLOCK,
        "layout": layout,
        "masked": masked
    }
)

target = GPUTarget("hip", 'gfx942', 64)  # For AMD HIP gfx942 (e.g., MI300X GPU)
backend = triton.compiler.make_backend(target)
options = backend.parse_options({"warp_size": warp_size, "num_warps": num_warps})

output = triton.compile(src, target=target, options=options.__dict__)
print(output.asm["amdgcn"])
