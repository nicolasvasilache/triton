#!/usr/bin/env python3
"""
Simple Copy Kernel Benchmark

Benchmarks 1D, 2D, and 3D copy operations with various layouts in the style of gluon tutorials.
"""

import itertools
import math

import torch
import triton
from triton.experimental.gluon import language as gl

from descriptor_helpers import TensorDescriptor
from copy_kernel import copy_kernel

SHIFT = 111
nelts = 304 * (2 ** 18 + SHIFT)

nelts_2d_1 = 2 ** 16 + SHIFT
nelts_2d = [math.ceil(nelts / nelts_2d_1), nelts_2d_1]

nelts_3d_1 = 2 ** 4 + SHIFT
nelts_3d_2 = 2 ** 16 + SHIFT
nelts_3d = [math.ceil(nelts / (nelts_3d_1 * nelts_3d_2)), 2 ** 4 + SHIFT, nelts_3d_2]
dtype = torch.float16

def get_throughput(tensor, ms):
    """Calculate throughput in GB/s."""
    gbytes = 2 * tensor.numel() * tensor.element_size() / 1e9
    return gbytes / (ms * 1e-3)

def benchmark_1d():
    """Benchmark 1D copy with different layouts."""
    print("1D Copy Benchmark")
    print("=================")
    
    size = nelts
    block_size = 2048
    print(f"block_sizes {block_size}")

    A = torch.randn(size, device="cuda", dtype=dtype)
    B = torch.empty_like(A)
    
    print("vector_size num_warps time (ms) throughput (GB/s)")
    
    for vector_size, num_warps in itertools.product([1, 2, 4, 8], [1, 2, 4, 8]):
        threads_per_warp = 64
        warps_per_cta = num_warps
        
        # Skip invalid configs
        if block_size // (threads_per_warp * warps_per_cta * vector_size) < 1:
            continue
            
        shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout = gl.BlockedLayout(
            size_per_thread=[vector_size],
            threads_per_warp=[threads_per_warp],
            warps_per_cta=[warps_per_cta],
            order=[0]
        )
        
        a_desc = TensorDescriptor.from_tensor(A, [block_size], blocked_layout, shared_layout)
        b_desc = TensorDescriptor.from_tensor(B, [block_size], blocked_layout, shared_layout)
        
        grid = (math.ceil(size / block_size), )
        fn = lambda: copy_kernel[grid](A, B, a_desc, b_desc, warp_size=threads_per_warp, num_warps=num_warps)
        fn()
        assert (A == B).all()

        try:
            ms = triton.testing.do_bench(fn, warmup=10, rep=100)
            throughput = get_throughput(A, ms)
            print(f"{vector_size:>11} {num_warps:>9} {ms:>9.5f} {throughput:>17.2f}")
        except Exception as e:
            print(f"{vector_size:>11} {num_warps:>9}     FAILED\n{e}")
    print()


def benchmark_2d():
    """Benchmark 2D copy with different layouts."""
    print("2D Copy Benchmark")
    print("=================")
    
    sizes = nelts_2d
    block_sizes = [2, 512]
    print(f"sizes {sizes}")
    print(f"block_sizes {block_sizes}")

    A = torch.randn(sizes, device="cuda", dtype=dtype)
    B = torch.empty_like(A)
    
    print("layout_order vector_size num_warps time (ms) throughput (GB/s)")
    
    orders = [[0, 1], [1, 0]]
    for order, vector_size, num_warps in itertools.product(orders, [1, 2, 4, 8], [1, 2, 4, 8]):
        shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=order)
        
        # Different blocked layouts for input and output
        blocked_a = gl.BlockedLayout(
            size_per_thread=[1, vector_size],
            threads_per_warp=[4, 16] if order == [1, 0] else [16, 4],
            warps_per_cta=[1, num_warps],
            order=order
        )
        
        blocked_b = gl.BlockedLayout(
            size_per_thread=[1, vector_size],
            threads_per_warp=[4, 16] if order == [1, 0] else [16, 4],
            warps_per_cta=[1, num_warps],
            order=order
        )
        
        a_desc = TensorDescriptor.from_tensor(A, block_sizes, blocked_a, shared_layout)
        b_desc = TensorDescriptor.from_tensor(B, block_sizes, blocked_b, shared_layout)
        
        # Benchmark
        grid = (math.ceil(sizes[0] / block_sizes[0]), math.ceil(sizes[1] / block_sizes[1]))
        fn = lambda: copy_kernel[grid](A, B, a_desc, b_desc, warp_size=64, num_warps=num_warps)
        fn()
        assert (A == B).all()

        try:
            ms = triton.testing.do_bench(fn, warmup=10, rep=100)
            throughput = get_throughput(A, ms)
            order_str = f"[{order[0]},{order[1]}]"
            print(f"{order_str:>12} {vector_size:>9} {num_warps:>9} {ms:>9.5f} {throughput:>17.2f}")
        except Exception as e:
            order_str = f"[{order[0]},{order[1]}]"
            print(f"{order_str:>12} {vector_size:>9} {num_warps:>9}     FAILED\n{e}")
    print()


def benchmark_3d():
    """Benchmark 3D copy with different layouts."""
    print("3D Copy Benchmark")
    print("=================")
    
    sizes = nelts_3d
    block_sizes = [2, 4, 512]
    print(f"sizes {sizes}")
    print(f"block_sizes {block_sizes}")
    
    A = torch.randn(sizes, device="cuda", dtype=dtype)
    B = torch.empty_like(A)
    
    print("layout_order num_warps time (ms) throughput (GB/s)")
    
    orders = [[2, 0, 1], [2, 1, 0]]
    for order, num_warps, vector_size in itertools.product(orders, [1, 2, 4, 8], [1, 2, 4, 8]):
        shared_layout = gl.SwizzledSharedLayout(vec=vector_size, per_phase=1, max_phase=1, order=order)
        
        size_per_thread_a = None
        # Create different blocked layouts for input and output
        if order == [2, 1, 0]:
            size_per_thread_a = [1, 1, vector_size]
            threads_per_warp_a = [1, 4, 16]
            size_per_thread_b = [1, 1, vector_size]
            threads_per_warp_b = [1, 4, 16]
        elif order == [2, 0, 1]:
            size_per_thread_a = [1, 1, vector_size]
            threads_per_warp_a = [4, 1, 16]
            size_per_thread_b = [1, 1, vector_size]
            threads_per_warp_b = [4, 1, 16]
        else:
            assert False, "unsupported num threads"
        
        blocked_a = gl.BlockedLayout(
            size_per_thread=size_per_thread_a,
            threads_per_warp=threads_per_warp_a,
            warps_per_cta=[1, 1, num_warps],
            order=order
        )
        blocked_b = gl.BlockedLayout(
            size_per_thread=size_per_thread_b,
            threads_per_warp=threads_per_warp_b,
            warps_per_cta=[1, 1, num_warps],
            order=order
        )
        
        a_desc = TensorDescriptor.from_tensor(A, block_sizes, blocked_a, shared_layout)
        b_desc = TensorDescriptor.from_tensor(B, block_sizes, blocked_b, shared_layout)
        
        # Benchmark
        grid = (math.ceil(sizes[0] / block_sizes[0]), math.ceil(sizes[1] / block_sizes[1]), math.ceil(sizes[2] / block_sizes[2]))
        fn = lambda: copy_kernel[grid](A, B, a_desc, b_desc, warp_size=64, num_warps=num_warps)
        try:
            ms = triton.testing.do_bench(fn, warmup=10, rep=100)
            throughput = get_throughput(A, ms)
            order_str = str(order)
            print(f"{order_str:>12} {num_warps:>9} {vector_size:>9} {ms:>9.5f} {throughput:>17.2f}")
        except Exception as e:
            order_str = str(order)
            print(f"{order_str:>12} {size_per_thread_a} {num_warps:>9} {vector_size:>9}     FAILED\n{e}")
    print()


if __name__ == "__main__":
    print("Copy Kernel Benchmarks")
    print("======================")
    print()
    
    benchmark_1d()
    benchmark_2d() 
    benchmark_3d()
