import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from gluon_memcpy_base import memcpy_1d_kernel

def memcpy_1d_cross_compile():
    warp_size = 64
    XBLOCK, num_warps, vector_size, order = 2048, 2, 4, 0
    layout = gl.BlockedLayout([vector_size], [warp_size], [num_warps], [order])
    masked = False
    use_buffer_instructions = True

    # Create Gluon AST source for cross-compilation
    src = GluonASTSource(
        fn=memcpy_1d_kernel,
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32", 
            "xnumel": "i32",
            "XBLOCK": "constexpr",
            "layout": "constexpr",
            "masked": "constexpr",
            "use_buffer_instructions": "constexpr",
        },
        constexprs={
            "XBLOCK": XBLOCK,
            "layout": layout,
            "masked": masked,
            "use_buffer_instructions": use_buffer_instructions,
        }
    )

    # AMD HIP gfx942 (e.g., MI300X GPU)
    target = GPUTarget("hip", 'gfx942', warp_size)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"warp_size": warp_size, "num_warps": num_warps})

    output = triton.compile(src, target=target, options=options.__dict__)
    print(output.asm["amdgcn"])


def memcpy_2d_cross_compile():
    warp_size = 64
    BLOCK_SIZE_N, num_warps, vector_size, order = 2048, 4, 4, 0
    layout = gl.BlockedLayout([vector_size], [warp_size], [num_warps], [order])
    masked = False
    use_buffer_instructions = True

    from gluon_memcpy_base import memcpy_2d_persistent_kernel

    src = GluonASTSource(
        fn=memcpy_2d_persistent_kernel,
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32", 
            "M": "i32",
            "N": "i32",
            "stride_in_m": "i32",
            "stride_out_m": "i32",
            "stride_in_n": "constexpr",
            "stride_out_n": "constexpr",
            "BLOCK_SIZE_N": "constexpr",
            "layout": "constexpr",
            "masked": "constexpr",
            "use_buffer_instructions": "constexpr",
            "num_cus": "constexpr",
            "row_cu_shift": "constexpr"
        },
        constexprs={
            "stride_in_n": 1,
            "stride_out_n": 1,
            "BLOCK_SIZE_N": BLOCK_SIZE_N,
            "layout": layout,
            "masked": masked,
            "use_buffer_instructions": use_buffer_instructions,
            "num_cus": 8,
            "row_cu_shift": 0,
        }
    )

    # AMD HIP gfx942 (e.g., MI300X GPU)
    target = GPUTarget("hip", 'gfx942', warp_size)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"warp_size": warp_size, "num_warps": num_warps})

    output = triton.compile(src, target=target, options=options.__dict__)
    print(output.asm["amdgcn"])

memcpy_1d_cross_compile()
memcpy_2d_cross_compile()
