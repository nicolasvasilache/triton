import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def memcpy_1d_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    XBLOCK: gl.constexpr,
    layout: gl.constexpr,
    masked: gl.constexpr,
    use_buffer_instructions: gl.constexpr,
):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # Locally preincrement pointers to ensure we have offsets within 2**32 so we can use buffer ops.
    in_ptr = in_ptr + start
    out_ptr = out_ptr + start

    if use_buffer_instructions:
        gl.static_assert(XBLOCK >= 0 and XBLOCK <= 2**12,
                        "offsets must (conservatively) fit within 12 bits")

    offsets = gl.arange(0, XBLOCK, layout=layout)

    mask = None
    if masked:
        mask = offsets < xnumel

    if not use_buffer_instructions:
        in_ptrs = in_ptr + offsets
        value = gl.load(in_ptrs, mask=mask, cache_modifier=".cg")
        out_ptrs = out_ptr + offsets
        gl.store(out_ptrs, value, mask=mask, cache_modifier=".cs")
    else:
        value = gl.amd.cdna3.buffer_load(in_ptr, offsets, mask=mask)
        gl.amd.cdna3.buffer_store(value, out_ptr, offsets, mask=mask)


@gluon.jit
def memcpy_2d_persistent_kernel(
    in_ptr, 
    out_ptr, 
    M, 
    N, 
    stride_in_m, 
    stride_out_m,
    stride_in_n: gl.constexpr, 
    stride_out_n: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    layout: gl.constexpr, 
    masked: gl.constexpr,
    use_buffer_instructions: gl.constexpr,
    num_cus: gl.constexpr,
    row_cu_shift: gl.constexpr,
):
    """
    2D memcpy kernel; each CU processes a row of elements in round-robin fashion.
    """
    
    gl.static_assert(stride_in_n == 1, "stride_in_n must be 1")
    gl.static_assert(stride_out_n == 1, "stride_out_n must be 1")
    gl.static_assert(row_cu_shift >= 0 and row_cu_shift < num_cus,
                     "row_cu_shift must be in range [0, num_cus)")

    # Each CU processes rows in a round-robin manner.
    cu_id = gl.program_id(0)
    m = (cu_id + row_cu_shift) % num_cus
    
    if use_buffer_instructions:
        gl.static_assert(BLOCK_SIZE_N >= 0 and BLOCK_SIZE_N <= 2**12,
                        "offsets must (conservatively) fit within 12 bits")

    # indices prescribe how a 1-D block of size BLOCK_SIZE_N is processed.
    indices = gl.arange(0, BLOCK_SIZE_N, layout=layout)

    while m < M:
        for n in range(0, N, BLOCK_SIZE_N):
            in_offset = m * stride_in_m + n + indices
            out_offset = m * stride_out_m + n + indices

            mask = None
            if masked:
                mask = (n < N)

            if not use_buffer_instructions:
                value = gl.load(in_ptr + in_offset, mask=mask, cache_modifier=".cg")
                gl.store(out_ptr + out_offset, value, mask=mask, cache_modifier=".cs")
            else:
                value = gl.amd.cdna3.buffer_load(in_ptr, in_offset, mask=mask, cache=".cg")
                gl.amd.cdna3.buffer_store(value, out_ptr, out_offset, mask=mask, cache=".cs")

        m += num_cus
