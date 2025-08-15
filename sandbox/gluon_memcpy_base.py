import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr, masked: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    if masked:
        mask = offsets < xnumel
        value = gl.load(in_ptrs, mask=mask, cache_modifier=".cg")
        out_ptrs = out_ptr + offsets
        gl.store(out_ptrs, value, mask=mask, cache_modifier=".cs")
    else:
        value = gl.load(in_ptrs, cache_modifier=".cg")
        out_ptrs = out_ptr + offsets
        gl.store(out_ptrs, value, cache_modifier=".cs")

