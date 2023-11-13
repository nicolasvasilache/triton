import torch
import triton
from triton.ops import matmul

torch.manual_seed(0)
a = torch.randn((2048, 2048), device='cpu', dtype=torch.float32)
b = torch.randn((2048, 2048), device='cpu', dtype=torch.float32)
triton_output = matmul(a, b)
