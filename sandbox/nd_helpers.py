"""
N-Dimensional Helper Utilities for Triton Gluon

This module provides helper functions for working with n-dimensional tensor operations
in Triton's Gluon framework. It includes utilities for generating offset tensors and
masks from blocked tensor descriptors, which are essential for memory access patterns
in matrix multiplication and other tensor operations.

Functions:
    nd_offset_from_blocked_descriptor: Generate n-dimensional offsets from blocked tensor descriptors
    nd_mask_from_blocked_descriptor: Generate n-dimensional masks from blocked tensor descriptors
"""

import triton.language as tl
import triton.experimental.gluon as gluon
from triton.experimental.gluon import language as gl

from descriptor_helpers import TensorDescriptor
from tuple_helpers import compute_strides, tuple_reduce_add, tuple_mul, tuple_any, tuple_zip_2, tuple_add


@gluon.jit
def nd_offset_from_blocked_descriptor(start_blocks: tl.tuple, 
                                      blocked_desc: TensorDescriptor):
    """
    Generate n-dimensional thread offsets from a blocked tensor descriptor.
    
    This function creates offset tensors for accessing elements in a blocked tensor layout.
    
    Args:
        start_blocks: triton.tuple: Starting block index for each dimension.
        blocked_desc: TensorDescriptor.
                     
    Returns:
        tensor: An n-dimensional offset tensor that can be used for memory access.
                Each element contains the linear memory offset for that position
                within the block.
                
    Example:
        >>> # For a 2D matrix with blocks of size 16x32
        >>> start_blocks = tl.tuple([block_m, block_n])  # Block coordinates
        >>> offsets = arange_nd_from_blocked_descriptor(start_blocks, tensor_desc)
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
        (0, ) * len(blocked_desc.shape),                     # type: ignore
        blocked_desc.block_shape,                            # type: ignore
        blocked_desc.strides,                                # type: ignore
        layout=blocked_desc.global_layout)                   # type: ignore
    shift_1d = tuple_reduce_add(tuple_mul(
        start_blocks, 
        tuple_mul(blocked_desc.block_shape, blocked_desc.strides)))   # type: ignore
    return base_offsets_nd + shift_1d


@gluon.jit
def nd_mask_from_blocked_descriptor(start_blocks: tl.tuple, 
                                    blocked_desc: TensorDescriptor):
    """
    Generate an n-dimensional mask from a blocked tensor descriptor.
        
    The function works by:
    1. Checking if any dimension of the tensor shape is not evenly divisible by the block shape
    2. Checking if the current block extends beyond tensor boundaries (start * block_shape + block_shape > shape)
    3. If masking is needed for either condition, creating a proper n-dimensional mask using gl.mask_nd
    4. If no masking is needed, creating a full mask of True values for optimal performance
    
    Args:
        start_blocks: Starting indices for each dimension as a Triton tuple.
                These represent the block-level coordinates (e.g., which block
                in the M and N dimensions for a 2D matrix).
        blocked_desc: A TensorDescriptor containing:
                     - shape: The full tensor shape
                     - block_shape: The shape of each block/tile
                     - global_layout: The distributed layout for the tensor
                     
    Returns:
        tensor: An n-dimensional mask tensor with the same shape as the block.
                Contains True for valid elements and False for out-of-bounds elements.
                When no masking is needed, returns a tensor filled with True values.
                
    Example:
        >>> # For a 2D matrix of size 100x100 with blocks of size 16x32
        >>> start_blocks = tl.tuple([block_m, block_n])  # Block coordinates
        >>> mask = nd_mask_from_blocked_descriptor(start_blocks, tensor_desc)
        >>> # mask contains False for elements beyond tensor boundaries or
        >>> # when blocks don't evenly divide tensor dimensions
        >>> data = tl.load(base_ptr + offsets, mask=mask)
        
    Note:
        This function optimizes for the common case where no masking is needed by
        creating a full True mask, which can be more efficient than computing
        boundary conditions when they're not necessary.
    """
    # Check if any dimension of block_shape does not evenly divide the tensor shape
    tup_zip = tuple_zip_2(blocked_desc.shape, blocked_desc.block_shape)
    tup_uneven = gl.tuple([shape % block_shape != 0 for shape, block_shape in tup_zip]) # type: ignore
    needs_mask_modulo = tuple_any(tup_uneven) # type: ignore
    
    # Check if the current block extends beyond tensor boundaries
    # This happens when start * block_shape + block_shape > shape for any dimension
    block_positions = tuple_mul(start_blocks, blocked_desc.block_shape) # type: ignore
    block_ends = tuple_add(block_positions, blocked_desc.block_shape) # type: ignore
    tup_zip_bounds = tuple_zip_2(block_ends, blocked_desc.shape)
    tup_bounds = gl.tuple([block_end > shape for block_end, shape in tup_zip_bounds]) # type: ignore
    needs_mask_bounds = tuple_any(tup_bounds) # type: ignore
    
    # Need mask if both conditions are true (needs_mask_modulo acts as a static
    # filter that is expected to canonicalize away when possible).
    needs_mask = needs_mask_modulo and needs_mask_bounds
    if needs_mask:
        mask = gl.mask_nd(start_blocks, blocked_desc.shape, blocked_desc.block_shape, blocked_desc.global_layout)
    else:
        mask = gl.full(blocked_desc.block_shape, gl.constexpr(True), gl.int1, blocked_desc.global_layout)
    
    return mask
