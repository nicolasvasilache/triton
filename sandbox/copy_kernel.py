import triton.language as tl
import triton.experimental.gluon as gluon
from triton.experimental.gluon import language as gl

from tuple_helpers import delinearize, get_linear_program_id
from nd_helpers import nd_offset_from_blocked_descriptor, nd_mask_from_blocked_descriptor


@gluon.jit
def copy_kernel(a_ptr, b_ptr, A: gl.constexpr, B: gl.constexpr):
    gl.static_assert(len(A.shape) == len(B.shape), f"A and B must have same rank: {len(A.shape)} vs {len(B.shape)}")

    linear_program_id = get_linear_program_id()
    start_blocks = delinearize(linear_program_id, A.num_blocks)
    
    mask_nd = None
    mask_nd_a = nd_mask_from_blocked_descriptor(start_blocks, A)
    mask_nd_b = gl.convert_layout(mask_nd_a, B.global_layout, assert_trivial=False)

    a_offsets_nd = nd_offset_from_blocked_descriptor(start_blocks, A)
    a = gl.load(a_ptr + a_offsets_nd, mask=mask_nd_a, cache_modifier=".ca")
    
    via_explicit_shared_memory: gl.constexpr = False
    if via_explicit_shared_memory:
        smem = gl.allocate_shared_memory(A.dtype, A.block_shape, layout=A.shared_layout)
        smem.store(a)
        b = smem.load(B.global_layout)
    else:
        b = gl.convert_layout(a, B.global_layout, assert_trivial=False)
    
    b_offsets_nd = nd_offset_from_blocked_descriptor(start_blocks, B)
    gl.store(b_ptr + b_offsets_nd, b, mask=mask_nd_b, cache_modifier=".cs")
