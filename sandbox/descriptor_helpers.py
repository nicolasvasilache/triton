"""
Tensor Descriptor Utilities for Triton Gluon

This module provides utilities for working with tensor descriptors in Triton's Gluon framework.
It includes type conversion utilities and a comprehensive TensorDescriptor class that encapsulates
tensor metadata including shape, strides, block shapes, and memory layouts.

Classes:
    TensorDescriptor: A comprehensive descriptor for tensors with layout information

Functions:
    torch_to_triton_dtype: Convert PyTorch dtypes to Triton dtypes
"""

from dataclasses import dataclass
import math
from typing import Any, List, Optional

import torch
import triton.language as tl
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon.language as gl
from triton._utils import is_power_of_two


def torch_to_triton_dtype(dtype: torch.dtype) -> tl.dtype:
    """
    Convert a PyTorch dtype to the corresponding Triton dtype.
    """
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float64: tl.float64,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.int16: tl.int16,
        torch.int8: tl.int8,
        torch.uint8: tl.uint8,
        torch.bool: tl.int1,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]

TRITON_MAX_TENSOR_NUMEL = 2 ** 63
def validate_shape(shape: List[tl.constexpr], is_block_shape: bool = False):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, tl.constexpr) or not isinstance(d.value, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `{type(d)}")
        if is_block_shape and not is_power_of_two(d.value):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d.value

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel

@aggregate
@dataclass
class TensorDescriptor:
    """
    A comprehensive descriptor for tensors with layout information.
    
    Attributes:
        dtype: The Triton data type of the tensor elements
        shape: The shape of the tensor as a Triton tuple
        strides: The memory strides of the tensor as a Triton tuple
        block_shape: The block shape for tiling operations as a Triton tuple
        global_layout: The global memory layout (BlockedLayout)
        shared_layout: The shared memory layout (SwizzledSharedLayout)
    """
    
    dtype: tl.dtype
    shape: tl.tuple
    strides: tl.tuple
    block_shape: tl.tuple
    num_blocks: tl.tuple
    global_layout: gl.BlockedLayout
    shared_layout: Optional[gl.SwizzledSharedLayout] = None

    def __post_init__(self):
        """
        Validate the tensor descriptor after initialization.
        
        Ensures that all dimensions are consistent and that the layouts
        are of the correct types.
        
        Raises:
            AssertionError: If validation fails
        """
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch strides: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch block_shape: {self}"
        assert len(self.num_blocks) == rank, f"rank mismatch num_blocks: {self}"
        assert rank > 0, "rank must not be zero"
        
        # Convert tuple to list for validate_block_shape
        validate_shape(list(self.shape))
        validate_shape(list(self.strides))
        validate_shape(list(self.block_shape), is_block_shape=True)
        
        assert isinstance(self.global_layout, gl.BlockedLayout), "Layout must be gl.BlockedLayout"
        assert self.shared_layout is None or isinstance(self.shared_layout, gl.SwizzledSharedLayout), "Layout must be gl.SwizzledSharedLayout"

    @staticmethod
    def from_tensor(tensor: Any,
                    block_shape: tuple[int],
                    global_layout: gl.BlockedLayout,
                    shared_layout: gl.SwizzledSharedLayout = None) -> 'TensorDescriptor':
        """
        Create a TensorDescriptor from a PyTorch tensor and layout information.
        
        Args:
            tensor: The PyTorch tensor to create a descriptor for
            block_shape: The block shape for tiling operations
            global_layout: The global memory layout
            shared_layout: The shared memory layout
            
        Returns:
            A new TensorDescriptor instance
            
        Example:
            >>> tensor = torch.randn(128, 256, dtype=torch.float32)
            >>> desc = TensorDescriptor.from_tensor(
            ...     tensor, (16, 32), blocked_layout, shared_layout
            ... )
        """
        return TensorDescriptor(
            torch_to_triton_dtype(tensor.dtype),
            tl.tuple([tl.constexpr(s) for s in tensor.shape]),
            tl.tuple([tl.constexpr(s) for s in tensor.stride()]),
            tl.tuple([tl.constexpr(s) for s in block_shape]),
            tl.tuple([tl.constexpr(math.ceil(s / bs)) for s, bs in zip(tensor.shape, block_shape)]),
            global_layout,
            shared_layout,
        )

    def __hash__(self):
        """
        Compute hash for the tensor descriptor.
        
        Returns:
            Hash value for the descriptor
        """
        return hash((
            self.dtype,
            tuple(self.shape),  # Convert tl.tuple to regular tuple for hashing
            tuple(self.strides),
            tuple(self.block_shape),
            self.global_layout,
            self.shared_layout,
        ))
