"""
N-Dimensional Helper Utilities for Triton Gluon

This module provides helper functions for working with n-dimensional tensor operations
in Triton's Gluon framework. It includes utilities for generating offset tensors from
blocked tensor descriptors, which are essential for memory access patterns in 
matrix multiplication and other tensor operations.

Functions:
    arange_nd_from_blocked_descriptor: Generate n-dimensional offsets from blocked tensor descriptors
"""

import triton.language as tl
import triton.experimental.gluon as gluon
from triton.experimental.gluon import language as gl

from descriptor_helpers import TensorDescriptor
from tuple_helpers import tuple_reduce_add, tuple_mul


@gluon.jit
def nd_offset_from_blocked_descriptor(starts: tl.tuple, 
                                      blocked_desc: TensorDescriptor):
    """
    Generate n-dimensional offsets from a blocked tensor descriptor.
    
    This function creates offset tensors for accessing elements in a blocked tensor layout.
    It's particularly useful for matrix multiplication kernels where memory access patterns
    need to be carefully managed for performance.
    
    The function works by:
    1. Creating base offsets using the tensor's block shape and strides
    2. Computing a 1D shift based on the starting positions and strides
    3. Adding the shift to the base offsets to get the final memory addresses
    
    Args:
        starts: Starting indices for each dimension as a Triton tuple.
                These represent the block-level coordinates (e.g., which block
                in the M and N dimensions for a 2D matrix).
        blocked_desc: A TensorDescriptor containing:
                     - shape: The full tensor shape
                     - block_shape: The shape of each block/tile
                     - strides: Memory strides for each dimension
                     - global_layout: The distributed layout for the tensor
                     
    Returns:
        tensor: An n-dimensional offset tensor that can be used for memory access.
                Each element contains the linear memory offset for that position
                within the block.
                
    Example:
        >>> # For a 2D matrix with blocks of size 16x32
        >>> starts = tl.tuple([block_m, block_n])  # Block coordinates
        >>> offsets = arange_nd_from_blocked_descriptor(starts, tensor_desc)
        >>> # offsets now contains memory addresses for the entire block
        >>> data = tl.load(base_ptr + offsets, mask=mask)
        
    Note:
        This function uses constant start and end values for the arange operation
        because Triton's tt.make_range operation only accepts compile-time constants.
        The dynamic shifting is handled separately by adding the computed shift offset.
    """
    # We cannot shift start and end by dynamic quantities: the type will not
    # be statically known. Even if the type was statically known, the start
    # and end would be dynamic SSA values but tt.make_range only takes
    # attributes. So we have to use constants for the start and end and shift
    # by an offset separately.
    base_offsets_nd = gl.arange_nd(
        (0, ) * len(blocked_desc.shape), # type: ignore
        blocked_desc.block_shape, # type: ignore
        blocked_desc.strides, # type: ignore
        layout=blocked_desc.global_layout) # type: ignore
    shift_1d = tuple_reduce_add(
        tuple_mul(starts, blocked_desc.strides)) # type: ignore
    return base_offsets_nd + shift_1d
