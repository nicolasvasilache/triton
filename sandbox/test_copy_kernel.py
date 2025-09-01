import math
import torch

from triton.experimental.gluon import language as gl

from descriptor_helpers import TensorDescriptor
from copy_kernel import copy_kernel
from compile_helpers import compile_with_ast_source, compile_with_parser, run_kernel

def test_2d():
    num_warps = 4
    sizes = [128, 256]
    block_sizes = [8, 32]
    shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    # A
    A = torch.randn(sizes, dtype=torch.float32)
    blocked_a = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, num_warps],
        order=[1, 0])
    a_desc = TensorDescriptor.from_tensor(
        A, block_sizes, blocked_a, shared_layout)

    # B
    B = torch.empty_like(A).zero_()
    blocked_b = gl.BlockedLayout(
        size_per_thread=[4, 1],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, num_warps],
        order=[1, 0])
    b_desc = TensorDescriptor.from_tensor(
        B, block_sizes, blocked_b, shared_layout)

    # Check front and IR are well-formed, not specifics of what they do.
    compile_with_ast_source(copy_kernel, A, B, a_desc, b_desc, num_warps=num_warps)
    compile_with_parser(copy_kernel, A, B, a_desc, b_desc, num_warps=num_warps)
    grid = tuple((math.ceil(sizes[0] / block_sizes[0]), math.ceil(sizes[1] / block_sizes[1])))
    run_kernel(copy_kernel, grid, A, B, a_desc, b_desc, num_warps=num_warps)


def test_3d():
    num_warps = 4
    sizes = [257, 1025, 513]
    block_sizes = [8, 16, 32]
    shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0])

    # A
    A = torch.randn(sizes, dtype=torch.float32)
    blocked_a = gl.BlockedLayout(
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 4, 16],
        warps_per_cta=[1, 1, num_warps],
        order=[2, 1, 0])
    a_desc = TensorDescriptor.from_tensor(
        A, block_sizes, blocked_a, shared_layout)

    # B
    B = torch.empty_like(A).zero_()
    blocked_b = gl.BlockedLayout(
        size_per_thread=[1, 4, 1],
        threads_per_warp=[1, 16, 4],
        warps_per_cta=[1, 1, num_warps],
        order=[2, 1, 0])
    b_desc = TensorDescriptor.from_tensor(
        B, block_sizes, blocked_b, shared_layout)

    # Check front and IR are well-formed, not specifics of what they do.
    compile_with_ast_source(copy_kernel, A, B, a_desc, b_desc, num_warps=num_warps)
    compile_with_parser(copy_kernel, A, B, a_desc, b_desc, num_warps=num_warps)
    grid = tuple((math.ceil(sizes[0] / block_sizes[0]), math.ceil(sizes[1] / block_sizes[1])))
    run_kernel(copy_kernel, grid, A, B, a_desc, b_desc, num_warps=num_warps)


if __name__ == "__main__":
    test_2d()
    test_3d()
