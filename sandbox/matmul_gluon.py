from dataclasses import dataclass
import math
from typing import List, Any

import torch

import triton
import triton.language as tl
import triton.experimental.gluon as gluon

from triton._utils import validate_block_shape, canonicalize_dtype, get_primitive_bitwidth
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


# Convert PyTorch dtype to Triton dtype
def torch_to_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.int32:
        return tl.int32
    elif torch_dtype == torch.int64:
        return tl.int64
    else:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")

@dataclass
class TensorDescriptor:
    base: torch.Tensor
    dtype: tl.dtype
    # WARNING: because of https://github.com/triton-lang/triton/pull/7239/files#diff-a94d24c42ec01a10430ef002dabce1e275f194ead5123b418d518fbf92a6c4a0R1306
    # we have to use tuple[int] instead of List[int]
    # Otherwise, consecutive lists get flattened into a single list of arguments
    # and calling e.g.
    # ```
    #    gl.arange_nd([0, 0], A.block_shape, A.strides, layout=A.global_layout)
    # ```
    # will fail.
    shape: tuple[int]
    strides: tuple[int]
    block_shape: tuple[int]
    global_layout: gl.BlockedLayout
    shared_layout: gl.SwizzledSharedLayout

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        dtype_str = canonicalize_dtype(self.dtype)
        elem_bytes = get_primitive_bitwidth(dtype_str) // 8
        assert isinstance(self.global_layout, gl.BlockedLayout), "Layout must be gl.BlockedLayout"
        assert isinstance(self.shared_layout, gl.SwizzledSharedLayout), "Layout must be gl.SwizzledSharedLayout"

    @staticmethod
    def from_tensor(tensor: Any,
                    block_shape: tuple[int],
                    global_layout: gl.BlockedLayout,
                    shared_layout: gl.SwizzledSharedLayout):
        return TensorDescriptor(
            tensor,
            torch_to_triton_dtype(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tuple(block_shape),
            global_layout,
            shared_layout,
        )

    def __hash__(self):
        return hash((
            self.base.dtype,
            tuple(self.shape),
            tuple(self.strides),
            tuple(self.block_shape),
            self.global_layout,
            self.shared_layout,
        ))


@gluon.jit
def arange_nd_from_blocked_descriptor(blocked_desc: gl.constexpr):
    return gl.arange_nd((0, ) * len(blocked_desc.shape), blocked_desc.block_shape, blocked_desc.strides, layout=blocked_desc.global_layout) # type: ignore

@gluon.jit
# To accept a non-static pointer, we have to pass a_ptr as a non-constexpr.
# Our custom TensorDescriptor object is not supported and is unlikely to every be.
# Passing it as a constexpr seems enough to appeaseGluonASTSource and the parser.
#
# We are lying to the type system, A and B are actually TensorDescriptor that
# masquerade as a constexpr. So we sprinkle a bunch of type: ignore.
#
# Older code to convert int64 addresses to proper pointer types
# This is useless because we have to decouple a_ptr from A.
# a_ptr = tl.cast(A.data_ptr(), tl.pointer_type(A.dtype))
# b_ptr = tl.cast(B.data_ptr(), tl.pointer_type(B.dtype))

def copy_kernel(a_ptr, b_ptr, A: gl.constexpr, B: gl.constexpr): 
    gl.static_assert(len(A.shape) == 2, f"A must be rank 2 but got {len(A.shape)} in {A}") # type: ignore
    gl.static_assert(len(B.shape) == 2, f"B must be rank 2 but got {len(B.shape)} in {B}") # type: ignore

    rank: gl.constexpr = len(A.shape) # type: ignore

    M : gl.constexpr = A.shape[0] # type: ignore
    N : gl.constexpr = A.shape[1] # type: ignore
        
    m, n = gl.program_id(0), gl.program_id(1)
    start_m = m * A.block_shape[0] # type: ignore
    start_n = n * A.block_shape[1] # type: ignore
    
    # We cannot shift start and end by dynamic quantities: the type will not be statically known.
    # Even if the type was statically known, the start and end would be dynamic SSA values but tt.make_range only takes attributes.
    # So we have to use constants for the start and end and shift by an offset separately.
    a_offsets_nd = arange_nd_from_blocked_descriptor(A)
    a_offsets_shift_1d = (start_m * A.strides[0] + start_n * A.strides[1]) # type: ignore
    a_offsets_nd = a_offsets_nd + a_offsets_shift_1d
        
    b_offsets_nd = arange_nd_from_blocked_descriptor(B)
    b_offsets_shift_1d = (start_m * B.strides[0] + start_n * B.strides[1]) # type: ignore
    b_offsets_nd = b_offsets_nd + b_offsets_shift_1d

    mask = None
    a = gl.load(a_ptr + a_offsets_nd, mask=mask)
    via_explicit_shared_memory: gl.constexpr = False
    if via_explicit_shared_memory:
        smem = gl.allocate_shared_memory(A.dtype, A.block_shape, layout=A.shared_layout) # type: ignore
        smem.store(a)
        b = smem.load(B.global_layout) # type: ignore
    else:
        b = gl.convert_layout(a, B.global_layout, assert_trivial=False) # type: ignore
    gl.store(b_ptr + b_offsets_nd, b, mask=mask)


def compile_with_ast_source(a_desc, b_desc, warp_size=64, num_warps=1):
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
    print(output.asm["amdgcn"])


def compile_with_parser(a_desc, b_desc, warp_size=64, num_warps=1):
    # Run the parser
    from triton._filecheck import run_parser
    mod = run_parser(
        copy_kernel,
        # Under regular parser flow, the ASTSource is created from a signature that is
        # inferred from the JIT arguments via `create_function_from_signature`.
        # This procedure infers the types with rules whose limitations are not immediately
        # obvious (see `create_specialize_impl` and KernelParam specialization rules).
        # compile_with_parser()
        args=(a_desc.base, b_desc.base, a_desc, b_desc),
        kwargs={"warp_size": warp_size, "num_warps": num_warps},
        target=GPUTarget("hip", 'gfx942', 64),
        # target=GPUTarget("cuda", 100, 32),
    )
    # print(mod.str_nodebug())


def run(a_desc, b_desc, grid: tuple, warp_size=64, num_warps=1):
    import math
    a_desc.base.to("cuda")
    b_desc.base.to("cuda")
    copy_kernel[grid](a_desc.base, b_desc.base, a_desc, b_desc, warp_size=warp_size, num_warps=num_warps)
    assert (a_desc.base == b_desc.base).all()

def test():
    num_warps = 4
    M, N = 257, 1025
    BLOCK_M, BLOCK_N = 16, 32
    A = torch.randn(M, N, dtype=torch.float32)
    B = torch.empty_like(A)
    blocked_a = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, num_warps],
        order=[1, 0])
    blocked_b = gl.BlockedLayout(
        size_per_thread=[4, 1],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, num_warps],
        order=[1, 0])
    shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    a_desc = TensorDescriptor.from_tensor(
        A, [BLOCK_M, BLOCK_N], blocked_a, shared_layout)
    b_desc = TensorDescriptor.from_tensor(
        B, [BLOCK_M, BLOCK_N], blocked_b, shared_layout)

    # Run 2 emitter and 1 execution test.
    compile_with_ast_source(a_desc, b_desc, num_warps=num_warps)
    # compile_with_parser(a_desc, b_desc, num_warps=num_warps)
    # grid = tuple((math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N)))
    # run(a_desc, b_desc, grid, num_warps=num_warps)


test()
