import tempfile
from mpmath.functions.bessel import ker
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from descriptor_helpers import TensorDescriptor
from triton.language.core import constexpr
from triton.runtime.jit import JITFunction


def torch_dtype_to_triton_ptr_type(dtype: torch.dtype) -> str:
    """Convert PyTorch dtype to Triton type string."""
    dtype_map = {
        torch.float32: "*fp32",
        torch.float16: "*fp16",
        torch.bfloat16: "*bf16",
        torch.float64: "*fp64",
        torch.int32: "*i32",
        torch.int64: "*i64",
        torch.int16: "*i16",
        torch.int8: "*i8",
        torch.uint8: "*u8",
        torch.bool: "*i1",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]

def compile_with_ast_source(kernel_fn: JITFunction, *args, warp_size: int = 64, num_warps: int = 1, dump: bool = False, **kwargs):
        # Filter out parameters with default values to make them optional
    required_params = []
    optional_params = []
    for param in kernel_fn.params:
        if not param.has_default:
            required_params.append(param)
        else:
            optional_params.append(param)

    if len(required_params) > len(args):
        raise ValueError(f"Kernel function expects at least {len(required_params)} required parameters but got {len(required_params)} (tensors + descriptors)")
    
    # Build signature and constexprs dynamically using actual parameter names
    signature, constexprs = {}, {}
    def dispatch_param(param, arg):
        if isinstance(arg, torch.Tensor):
            signature[param.name] = torch_dtype_to_triton_ptr_type(arg.dtype)
        elif isinstance(arg, TensorDescriptor) or param.is_constexpr:
            assert param.is_constexpr, f"TensorDescriptor parameter {param.name} must be annotated as constexpr"
            signature[param.name] = "constexpr"
            constexprs[param.name] = arg
        else:
            raise ValueError(f"Unsupported argument type: {type(arg)} for parameter {param.name}")

    for param, arg in zip(required_params, args):
        dispatch_param(param, arg)
    for param in optional_params:
        if not param.name in kwargs:
            continue
        dispatch_param(param, kwargs[param.name])
    
    src = GluonASTSource(
        fn=kernel_fn,
        signature=signature,
        constexprs=constexprs
    )

    # AMD HIP gfx942 (e.g., MI300X GPU)
    target = GPUTarget("hip", 'gfx942', 64)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"warp_size": warp_size, "num_warps": num_warps})
    output = triton.compile(src, target=target, options=options.__dict__)
    if dump:
        with tempfile.NamedTemporaryFile(suffix=".asm", mode="w", delete=False) as f:
            f.write(output.asm["amdgcn"])
            print(f"asm after compile_with_ast_source: {f.name}")


def compile_with_parser(kernel_fn: JITFunction, *args, warp_size: int = 64, num_warps: int = 1, dump: bool = False):
    target = GPUTarget("hip", 'gfx942', 64)
    from triton._filecheck import run_parser
    mod = run_parser(
        kernel_fn,
        args=args,
        kwargs={"warp_size": warp_size, "num_warps": num_warps},
        target=target,
    )
    if dump:
        with tempfile.NamedTemporaryFile(suffix=".ttir", mode="w", delete=False) as f:
            f.write(mod.str_nodebug())
            f.flush()
            print(f"ttir_path after run_parser: {f.name}")


def run_kernel(kernel_fn: JITFunction, grid: tuple, *args, warp_size: int = 64, num_warps: int = 1):
    # Move tensors to CUDA (only torch.Tensor objects)
    cuda_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cuda_args.append(arg.to("cuda"))
        else:
            cuda_args.append(arg)
    
    # Call kernel with all arguments
    kernel_fn[grid](*cuda_args, warp_size=warp_size, num_warps=num_warps)
