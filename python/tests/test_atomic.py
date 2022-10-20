import triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import torch

import pytest
import itertools

import numpy as np
import re
from numpy.random import RandomState
from typing import Optional, Union

float_dtypes = ['float16', 'float32', 'float64']
int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


@pytest.mark.parametrize("op, dtype_x_str, mode", itertools.chain.from_iterable([
    [
        ('add', 'float16', mode),
        ('add', 'uint32', mode), ('add', 'int32', mode), ('add', 'float32', mode),
        ('max', 'uint32', mode), ('max', 'int32', mode), ('max', 'float32', mode),
        ('min', 'uint32', mode), ('min', 'int32', mode), ('min', 'float32', mode),
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']]))
def test_atomic_rmw(op, dtype_x_str, mode, device='cuda'):
    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x)'})
    numpy_op = {'add': np.sum, 'max': np.max, 'min': np.min}[op]
    max_neutral = float('-inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    rs = RandomState(17)
    x = numpy_random((n_programs, ), dtype_str=dtype_x_str, rs=rs)
    if mode == 'all_neg':
        x = -np.abs(x)
    if mode == 'all_pos':
        x = np.abs(x)
    if mode == 'min_neg':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = -np.max(np.abs(x)) - 1
    if mode == 'max_pos':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = np.max(np.abs(x)) + 1
    x_tri = to_triton(x, device=device)

    z_tri = to_triton(np.array([neutral], dtype=getattr(np, dtype_x_str)), device=device)
    kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = numpy_op(x).astype(getattr(np, dtype_x_str))
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == to_numpy(z_tri).item()
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
