"""
Tuple helper functions for Triton Gluon operations.

This module provides utility functions for working with tuples in Triton Gluon,
including linearization, delinearization, and basic tuple operations.
"""

import triton.language as tl
import triton.experimental.gluon as gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def compute_strides(basis: tl.tuple):
    """
    Compute strides from basis dimensions.
    
    Args:
        basis: Tuple of basis dimensions (e.g., (b0, b1, b2, b3))
    
    Returns:
        Tuple of strides where each stride is the product of all subsequent basis dimensions.
        For basis (b0, b1, b2, b3), returns (b1*b2*b3, b2*b3, b3, 1)
    """
    strides = tl.tuple([1] * len(basis))
    for i in tl.static_range(len(basis) - 2, -1, -1):  # len-2 down to 0
        stride_val = strides[i + 1] * basis[i + 1]
        strides._setitem(i, stride_val)
    return strides


@gluon.jit
def linearize(indices: tl.tuple, basis: tl.tuple):
    """
    Convert multi-dimensional indices to a linear index using the given basis.
    
    Args:
        indices: Tuple of indices (e.g., (i, j, k, l))
        basis: Tuple of basis dimensions (e.g., (b0, b1, b2, b3))
    
    Returns:
        Linear index computed using strides: i*stride0 + j*stride1 + k*stride2 + l*stride3
        where strides = (b1*b2*b3, b2*b3, b3, 1)
    """
    gl.static_assert(len(indices) == len(basis), 
                     f"indices and basis must have same length: {len(indices)} vs {len(basis)}")
    
    strides = compute_strides(basis)
    
    linear_idx = 0
    for i in tl.static_range(len(indices)):
        linear_idx += indices[i] * strides[i]
    return linear_idx


@gluon.jit
def delinearize(linear_idx, basis: tl.tuple):
    """
    Convert a linear index back to multi-dimensional indices using the given basis.
    
    Args:
        linear_idx: Linear index to convert
        basis: Tuple of basis dimensions (e.g., (b0, b1, b2, b3))
    
    Returns:
        Tuple of indices (i, j, k, l, ...)
    """
    strides = compute_strides(basis)
    
    result = tl.tuple([0] * len(basis))
    remaining = linear_idx
    for i in tl.static_range(len(basis)):
        idx_val = remaining // strides[i]
        result._setitem(i, idx_val)
        remaining = remaining % strides[i]
    
    return result


@gluon.jit
def get_linear_program_id():
    """Get linear program ID using the generic linearize function."""
    pid0 = gl.program_id(0)
    pid1 = gl.program_id(1) 
    pid2 = gl.program_id(2)
    npg0 = gl.num_programs(0)
    npg1 = gl.num_programs(1)
    npg2 = gl.num_programs(2)
    return linearize(tl.tuple([pid0, pid1, pid2]), tl.tuple([npg0, npg1, npg2]))


@gluon.jit
def tuple_mul(ta: tl.tuple, tb: tl.tuple):
    """
    Element-wise multiplication of two tuples.
    
    Args:
        ta: First tuple
        tb: Second tuple (must have same length as ta)
    
    Returns:
        Tuple where result[i] = ta[i] * tb[i]
    """
    gl.static_assert(len(ta) == len(tb), f"tuple_mul: {ta} and {tb} must have the same length")
    # tuple is a very special flower:
    #   - this fails because only tuple comprehension is supported.
    #       return tl.tuple([ta[i] * tb[i] for i in range(len(ta))])
    #   - this fails because GeneratorExp is not supported.
    #       return tl.tuple(ta[i] * tb[i] for i in range(len(ta)))
    #   - can't use zip because it's unsupported.
    #   - can't use list + append because it's unsupported.
    # So I have to resort to _setitem which is marked with TODO: remove.
    result = tl.tuple(ta)
    for i in tl.static_range(len(ta)):
        result._setitem(i, ta[i] * tb[i])
    return result


@gluon.jit
def tuple_reduce_add(t: tl.tuple):
    """
    Sum all elements in a tuple.
    
    Args:
        t: Tuple to sum
    
    Returns:
        Sum of all elements in the tuple
    """
    res = t[0]
    for i in tl.static_range(1, len(t)):
        res += t[i]
    return res
