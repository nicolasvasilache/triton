"""
AMD MI300 CDNA3 Matrix Multiplication Kernel - Step 1: Memory Movement Focus

This implementation focuses on getting the memory movement patterns correct using
Gluon abstractions, specifically for AMD CDNA3 architecture.

Authors: Phil Tillet, Thomas Raoux, Jeff Niu
"""

import math

from jinja2 import debug
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna3

# Import helper utilities
from descriptor_helpers import TensorDescriptor
from tuple_helpers import linearize, delinearize, get_linear_program_id
from nd_helpers import nd_offset_from_blocked_descriptor, nd_mask_from_blocked_descriptor
from tuple_helpers import compute_strides, tuple_reduce_add, tuple_mul, tuple_any, tuple_zip_2, tuple_add



@gluon.jit
def matmul(
    # Matrix pointers
    a_ptr, b_ptr, c_ptr,
    # Tensor descriptors (must be constexpr)
    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
    mfma_layout: tl.constexpr,
):    
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=4)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=4)
    
    linear_program_id = get_linear_program_id()
    start_blocks_c = delinearize(linear_program_id, C.num_blocks)
    
    acc = gl.full(C.block_shape, 0, dtype=C.dtype, layout=mfma_layout)

    # Warning: bad surprises about if we're going out of offset bounds for buffer_load
    use_buffer_ops: gl.constexpr = False

    smem_a = gl.allocate_shared_memory(A.dtype, A.block_shape, layout=A.shared_layout)
    smem_b = gl.allocate_shared_memory(B.dtype, B.block_shape, layout=B.shared_layout)
    for k in tl.range(0, A.num_blocks[1]):
        start_blocks_a = tl.tuple([start_blocks_c[0], k])
        start_blocks_b = tl.tuple([k, start_blocks_c[1]])
        mask_nd_a = None # nd_mask_from_blocked_descriptor(start_blocks_a, A)
        mask_nd_b = None # nd_mask_from_blocked_descriptor(start_blocks_b, B)
        off_a, base_offsets_a, shift_a = nd_offset_from_blocked_descriptor(start_blocks_a, A)
        off_b, base_offsets_b, shift_b = nd_offset_from_blocked_descriptor(start_blocks_b, B)

        if use_buffer_ops:
            a = gl.amd.cdna3.buffer_load(a_ptr + shift_a, base_offsets_a, mask_nd_a)
            b = gl.amd.cdna3.buffer_load(b_ptr + shift_b, base_offsets_b, mask_nd_b)
        else:
            a = gl.load(a_ptr + off_a, mask_nd_a)
            b = gl.load(b_ptr + off_b, mask_nd_b)

        use_smem: gl.constexpr = True
        if use_smem:
            smem_a.store(a)
            smem_b.store(b)
            aa = smem_a.load(dot_a_layout)
            bb = smem_b.load(dot_b_layout)
            acc = gl.amd.cdna3.mfma(aa, bb, acc)
        else:
            a = gl.convert_layout(a, dot_a_layout)
            b = gl.convert_layout(b, dot_b_layout)        
            acc = gl.amd.cdna3.mfma(a, b, acc)


    # TODO: make trivial
    acc = gl.convert_layout(acc, C.global_layout, assert_trivial=False)

    mask_nd_c = None # nd_mask_from_blocked_descriptor(start_blocks_c, C)
    offsets_c, base_offsets_c, shift_c = nd_offset_from_blocked_descriptor(start_blocks_c, C)

    # Yes .. CDNA3 and Triton use different parameter orders
    if use_buffer_ops:
        gl.amd.cdna3.buffer_store(acc, c_ptr + shift_c, base_offsets_c, mask_nd_c)
    else:
        gl.store(c_ptr + offsets_c, acc, mask_nd_c)
 
    return


def create_tensor_descriptors(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, num_warps: int):
    size_per_thread_y = 16
    assert size_per_thread_y % num_warps == 0, f"num_warps must divide {size_per_thread_y} (got {num_warps})"
    a_global_layout = gl.BlockedLayout(
        size_per_thread=[size_per_thread_y, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[num_warps, 1],
        order=[1, 0]
    )
    b_global_layout = gl.BlockedLayout(
        size_per_thread=[size_per_thread_y, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, num_warps],
        order=[1, 0]
    )
    c_global_layout = gl.BlockedLayout(
        size_per_thread=[size_per_thread_y, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[num_warps, 1],
        order=[1, 0]
    )
    a_shared_layout = gl.SwizzledSharedLayout(
        vec=2,
        per_phase=1,
        max_phase=8,
        order=[1, 0]
    )
    b_shared_layout = gl.SwizzledSharedLayout(
        vec=2,
        per_phase=1,
        max_phase=8,
        order=[0, 1]
    )
    A_desc = TensorDescriptor.from_tensor(A, (BLOCK_M, BLOCK_K), a_global_layout, a_shared_layout)
    B_desc = TensorDescriptor.from_tensor(B, (BLOCK_K, BLOCK_N), b_global_layout, b_shared_layout)  
    C_desc = TensorDescriptor.from_tensor(C, (BLOCK_M, BLOCK_N), c_global_layout)
    
    return A_desc, B_desc, C_desc


def test():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    num_warps = 1
    num_xcd = 304
    # factor_m, factor_n, factor_k = 4 * num_xcd, 4, 256
    factor_m, factor_n, factor_k =  \
        (num_xcd * num_warps) // 8, \
                             8 * 4, \
                              512
    BLOCK_M, BLOCK_N, BLOCK_K = 64 * num_warps, 64, 32 // num_warps
    M, N, K = BLOCK_M * factor_m, BLOCK_N * factor_n, BLOCK_K * factor_k

    # Create test tensors.
    # Note: randn is 10% slower than linspace; linspace is 10% slower than ones likely due to arith/power interplay on MI300x
    # To measure vs peak, one needs to use ones.
    ver = 0
    device = 'cuda'
    if ver == 0:
        A = torch.ones(M, K, dtype=torch.float32, device=device)
        B = torch.ones(K, N, dtype=torch.float32, device=device)
    elif ver == 1:
        A = torch.linspace(0, M * K, steps=M * K, dtype=torch.float32, device=device).reshape(M, K) / (10 ** 6)
        B = torch.linspace(0, K * N, steps=K * N, dtype=torch.float32, device=device).reshape(K, N) / (10 ** 6)
    elif ver == 2:
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        B = torch.randn(K, N, dtype=torch.float32, device=device)
    C = torch.empty(M, N, dtype=torch.float32, device=device)
    
    # Create tensor descriptors
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(\
        version=3, instr_shape=[32, 32], transposed=False, warps_per_cta=[num_warps, 1])
    A_desc, B_desc, C_desc = \
        create_tensor_descriptors(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps)
    
    def check_fn(args):
        A, B, C = args[:3]
        ref = (A @ B).to(C.dtype)
        torch.testing.assert_close(C, ref, atol=3e-3, rtol=1e-4)

    
    ir_only = False
    if ir_only:
        # Test compilation with AST source
        from compile_helpers import compile_with_ast_source
        compile_with_ast_source(
            matmul,
            A, B, C,
            A_desc, B_desc, C_desc,
            mfma_layout,
            warp_size=64,
            num_warps=num_warps,
            debug=True
        )
        return
    


    grid = C_desc.num_blocks
    print(grid)

    fn = lambda: matmul[grid](A, B, C, A_desc, B_desc, C_desc, mfma_layout, warp_size=64, num_warps=num_warps)


    # Check and time by default, if we profile, don't check or time.
    profile_it = True
    profile_it = False
    time_it = not profile_it
    check = time_it
    if check:
        fn()
        check_fn([A, B, C, A_desc, B_desc, C_desc, mfma_layout])
    
    if time_it:
        extrapolated = M * N < BLOCK_M * BLOCK_N * num_xcd * num_warps
        extrapolated = " (extrapolated assuming perfect scaling)" if extrapolated else ""
        ms = triton.testing.do_bench(fn, warmup=2, rep=10)
        print(f"Running Time {ms:>9.5f} ms\n" + \
            f"\t{(M * N * K * 304 * 10 ** 3) / (ms * 10 ** 12 * num_xcd):>9.2f}TF/s{extrapolated}\n" + \
            f"\t{((M * N + M * K + M * N) * 304 * 10 ** 3) / (ms * 10 ** 12 * num_xcd):>9.2f}TB/s{extrapolated}\n")
    else:
        fn()
        fn()
        fn()

if __name__ == "__main__":
    test()
