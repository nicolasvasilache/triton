import math
import tempfile

import torch

import triton
import triton.language as tl
from triton.language.core import static_print
import triton.experimental.gluon as gluon

from triton._utils import canonicalize_dtype, get_primitive_bitwidth
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from tuple_helpers import delinearize, get_linear_program_id
from descriptor_helpers import TensorDescriptor
from nd_helpers import nd_offset_from_blocked_descriptor


# To accept a non-static pointer, we have to pass a_ptr as a non-constexpr.
# Our custom TensorDescriptor object is not supported and is unlikely to every be.
# Passing it as a constexpr seems enough to appease GluonASTSource and the parser.
#
# We are lying to the type system, A and B are actually TensorDescriptor that
# masquerade as a constexpr. So we sprinkle a bunch of type: ignore for now.
#
# Older code to convert int64 addresses to proper pointer types
# This is useless because we have to decouple a_ptr from A.
# a_ptr = tl.cast(A.base.data_ptr(), tl.pointer_type(A.dtype))
# b_ptr = tl.cast(B.base.data_ptr(), tl.pointer_type(B.dtype))
#
# Note: would be nice to have A and B support taking a_ptr and b_ptr.
@gluon.jit #(do_not_specialize=["A", "B"]): no effect atm
def copy_kernel(a_ptr, b_ptr, A: gl.constexpr, B: gl.constexpr):
    gl.static_assert(len(A.shape) == len(B.shape), f"A and B must have same rank: {len(A.shape)} vs {len(B.shape)}") # type: ignore

    linear_program_id = get_linear_program_id()
    starts = delinearize(linear_program_id, A.block_shape) # type: ignore
    
    a_offsets_nd = nd_offset_from_blocked_descriptor(starts, A)
    mask = gl.mask_nd(starts, A.shape, A.block_shape, A.global_layout) # type: ignore

    a = gl.load(a_ptr + a_offsets_nd, mask=mask)
    
    b_offsets_nd = nd_offset_from_blocked_descriptor(starts, B)
    mask = gl.convert_layout(mask, B.global_layout, assert_trivial=False) # type: ignore
    
    via_explicit_shared_memory: gl.constexpr = False
    if via_explicit_shared_memory:
        smem = gl.allocate_shared_memory(A.dtype, A.block_shape, layout=A.shared_layout) # type: ignore
        smem.store(a)
        b = smem.load(B.global_layout) # type: ignore
    else:
        b = gl.convert_layout(a, B.global_layout, assert_trivial=False) # type: ignore
    
    gl.store(b_ptr + b_offsets_nd, b, mask=mask)


def compile_with_ast_source(A: torch.Tensor, B: torch.Tensor, a_desc: TensorDescriptor, b_desc: TensorDescriptor, warp_size=64, num_warps=1):
    src = GluonASTSource(
        fn=copy_kernel,
        signature={
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "A": "constexpr",
            "B": "constexpr",
        },
        constexprs={
            "A": a_desc,
            "B": b_desc,
        }
    )
    # AMD HIP gfx942 (e.g., MI300X GPU)
    target = GPUTarget("hip", 'gfx942', 64)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"warp_size": warp_size, "num_warps": num_warps})
    output = triton.compile(src, target=target, options=options.__dict__)
    # print(output.asm["amdgcn"])


def compile_with_parser(A: torch.Tensor, B: torch.Tensor, a_desc: TensorDescriptor, b_desc: TensorDescriptor, warp_size=64, num_warps=1):
    # target=GPUTarget("cuda", 100, 32),
    target=GPUTarget("hip", 'gfx942', 64)
    # Run the parser
    from triton._filecheck import run_parser
    mod = run_parser(
        copy_kernel,
        args=(A, B, a_desc, b_desc),
        kwargs={"warp_size": warp_size, "num_warps": num_warps},
        target=target,
    )
    # Compile the module from file path.
    with tempfile.NamedTemporaryFile(suffix=".ttir", mode="w", delete=False) as f:
        f.write(mod.str_nodebug())
        f.flush()
        ttir_path = f.name
        print(f"ttir_path: {ttir_path}")
        # Note: for this to work, I had to hack disable `if ir_source:` in `compiler.py`
        # compiled = triton.compile(ttir_path, target=target, options={"warp_size": warp_size, "num_warps": num_warps})
        # print(compiled.asm["amdgcn"])


def run(A, B, a_desc, b_desc, grid: tuple, warp_size=64, num_warps=1):
    A = A.to("cuda")
    B = B.to("cuda")
    copy_kernel[grid](A, B, a_desc, b_desc, warp_size=warp_size, num_warps=num_warps)
    assert (A == B).all()


def test():
    num_warps = 4
    sizes    = [257, 1025, 513]
    block_sizes = [8, 16, 32]
    A = torch.randn(sizes, dtype=torch.float32)
    B = torch.empty_like(A).zero_()
    blocked_a = gl.BlockedLayout(
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 4, 16],
        warps_per_cta=[1, 1, num_warps],
        order=[2, 1, 0])
    blocked_b = gl.BlockedLayout(
        size_per_thread=[1, 4, 1],
        threads_per_warp=[1, 16, 4],
        warps_per_cta=[1, 1, num_warps],
        order=[2, 1, 0])
    shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0])
    a_desc = TensorDescriptor.from_tensor(
        A, block_sizes, blocked_a, shared_layout)
    b_desc = TensorDescriptor.from_tensor(
        B, block_sizes, blocked_b, shared_layout)

    # Run 2 emitter and 1 execution test.
    compile_with_ast_source(A, B, a_desc, b_desc, num_warps=num_warps)
    compile_with_parser(A, B, a_desc, b_desc, num_warps=num_warps)
    # grid = tuple((math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N)))
    # run(A, B, a_desc, b_desc, grid, num_warps=num_warps)


test()
