"""
Matrix Multiplication
======================
In this tutorial, you will write a 25-lines high-performance FP16 matrix multiplication
kernel that achieves performance on par with cuBLAS.
You will specifically learn about:

- Block-level matrix multiplications
- Multi-dimensional pointer arithmetic
- Program re-ordering for improved L2 cache hit rate
- Automatic performance tuning
"""

IR = """
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.mma<{version = 2, warpsPerCTA = [2, 4]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 8 : i32} {
  func public @matmul_kernel_0d1d2d3d4d5d6d7c8d9c10d11c(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<64> : tensor<128x64xi32, #blocked0>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    %c256_i32 = arith.constant 256 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.cmpi slt, %8, %c8_i32 : i32
    %10 = select %9, %8, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.splat %15 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0}>>
    %19 = tt.splat %15 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %20 = arith.muli %14, %c256_i32 : i32
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.splat %20 : (i32) -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %23 = arith.addi %18, %16 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0}>>
    %24 = arith.addi %19, %17 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0}>>) -> tensor<128x1xi32, #blocked0>
    %26 = tt.expand_dims %24 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %27 = tt.splat %arg6 : (i32) -> tensor<128x1xi32, #blocked0>
    %28 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0}>>
    %29 = tt.expand_dims %28 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0}>>) -> tensor<1x64xi32, #blocked0>
    %30 = tt.broadcast %29 : (tensor<1x64xi32, #blocked0>) -> tensor<128x64xi32, #blocked0>
    %31 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x64x!tt.ptr<f16>, #blocked0>
    %32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
    %34 = tt.splat %arg7 : (i32) -> tensor<64x1xi32, #blocked1>
    %35 = arith.addi %22, %21 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %36 = tt.expand_dims %35 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x256xi32, #blocked1>
    %37 = tt.broadcast %36 : (tensor<1x256xi32, #blocked1>) -> tensor<64x256xi32, #blocked1>
    %38 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %39 = arith.index_cast %arg5 : i32 to index
    %40 = arith.muli %arg7, %c64_i32 : i32
    %41 = tt.splat %40 : (i32) -> tensor<64x256xi32, #blocked1>
    %42 = arith.muli %25, %27 : tensor<128x1xi32, #blocked0>
    %43 = tt.broadcast %42 : (tensor<128x1xi32, #blocked0>) -> tensor<128x64xi32, #blocked0>
    %44 = arith.addi %43, %30 : tensor<128x64xi32, #blocked0>
    %45 = tt.addptr %31, %44 : tensor<128x64x!tt.ptr<f16>, #blocked0>
    %46 = arith.muli %33, %34 : tensor<64x1xi32, #blocked1>
    %47 = tt.broadcast %46 : (tensor<64x1xi32, #blocked1>) -> tensor<64x256xi32, #blocked1>
    %48 = arith.addi %47, %37 : tensor<64x256xi32, #blocked1>
    %49 = tt.addptr %38, %48 : tensor<64x256x!tt.ptr<f16>, #blocked1>
    %50 = arith.cmpi slt, %c0, %39 : index
    %51 = triton_gpu.alloc_tensor : tensor<3x128x64xf16, #shared>
    %52 = tt.splat %50 : (i1) -> tensor<128x64xi1, #blocked0>
    %53 = triton_gpu.insert_slice_async %45, %51, %c0_i32, %52 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64x!tt.ptr<f16>, #blocked0> -> tensor<3x128x64xf16, #shared>
    %54 = triton_gpu.alloc_tensor : tensor<3x64x256xf16, #shared>
    %55 = tt.splat %50 : (i1) -> tensor<64x256xi1, #blocked1>
    %56 = triton_gpu.insert_slice_async %49, %54, %c0_i32, %55 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256x!tt.ptr<f16>, #blocked1> -> tensor<3x64x256xf16, #shared>
    %57 = tt.addptr %45, %cst : tensor<128x64x!tt.ptr<f16>, #blocked0>
    %58 = tt.addptr %49, %41 : tensor<64x256x!tt.ptr<f16>, #blocked1>
    %59 = arith.cmpi slt, %c64, %39 : index
    %60 = tt.splat %59 : (i1) -> tensor<128x64xi1, #blocked0>
    %61 = triton_gpu.insert_slice_async %57, %53, %c1_i32, %60 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64x!tt.ptr<f16>, #blocked0> -> tensor<3x128x64xf16, #shared>
    %62 = tt.splat %59 : (i1) -> tensor<64x256xi1, #blocked1>
    %63 = triton_gpu.insert_slice_async %58, %56, %c1_i32, %62 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256x!tt.ptr<f16>, #blocked1> -> tensor<3x64x256xf16, #shared>
    %64 = tt.addptr %57, %cst : tensor<128x64x!tt.ptr<f16>, #blocked0>
    %65 = tt.addptr %58, %41 : tensor<64x256x!tt.ptr<f16>, #blocked1>
    triton_gpu.async_wait {num = 2 : i32}
    %66 = tensor.extract_slice %61[0, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<3x128x64xf16, #shared> to tensor<128x64xf16, #shared>
    %67 = tensor.extract_slice %63[0, 0, 0] [1, 64, 256] [1, 1, 1] : tensor<3x64x256xf16, #shared> to tensor<64x256xf16, #shared>
    %68 = tensor.extract_slice %66[0, 0] [128, 16] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x16xf16, #shared>
    %70 = tensor.extract_slice %67[0, 0] [16, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<16x256xf16, #shared>
    %72:14 = scf.for %arg9 = %c0 to %39 step %c128 iter_args(%arg10 = %cst_0, %arg11 = %45, %arg12 = %49, %arg13 = %61, %arg14 = %63, %arg15 = %66, %arg16 = %67, %arg17 = %64, %arg18 = %65, %arg19 = %c64, %arg20 = %c2_i32, %arg21 = %c1_i32, %arg22 = %68, %arg23 = %70) -> (tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked0>, tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<3x128x64xf16, #shared>, tensor<3x64x256xf16, #shared>, tensor<128x64xf16, #shared>, tensor<64x256xf16, #shared>, tensor<128x64x!tt.ptr<f16>, #blocked0>, tensor<64x256x!tt.ptr<f16>, #blocked1>, index, i32, i32, tensor<128x16xf16, #shared>, tensor<16x256xf16, #shared>) {
      %69 = triton_gpu.convert_layout %arg22 : (tensor<128x16xf16, #shared>) -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %71 = triton_gpu.convert_layout %arg23 : (tensor<16x256xf16, #shared>) -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %89 = tt.dot %69, %71, %arg10 {allowTF32 = true, transA = false, transB = false} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x256xf32, #mma>
      %90 = tensor.extract_slice %arg15[0, 16] [128, 32] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x32xf16, #shared>
      %91 = triton_gpu.convert_layout %90 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %92 = tensor.extract_slice %arg16[16, 0] [32, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<32x256xf16, #shared>
      %93 = triton_gpu.convert_layout %92 : (tensor<32x256xf16, #shared>) -> tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %94 = tt.dot %91, %93, %89 {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x256xf32, #mma>
      %95 = tensor.extract_slice %arg15[0, 48] [128, 16] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x16xf16, #shared>
      %96 = triton_gpu.convert_layout %95 : (tensor<128x16xf16, #shared>) -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %97 = tensor.extract_slice %arg16[48, 0] [16, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<16x256xf16, #shared>
      %98 = triton_gpu.convert_layout %97 : (tensor<16x256xf16, #shared>) -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %99 = tt.dot %96, %98, %94 {allowTF32 = true, transA = false, transB = false} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x256xf32, #mma>
      %100 = tt.addptr %arg11, %cst : tensor<128x64x!tt.ptr<f16>, #blocked0>
      %101 = tt.addptr %arg12, %41 : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %102 = arith.addi %arg19, %c64 : index
      %103 = arith.cmpi slt, %102, %39 : index
      %104 = arith.remsi %arg20, %c3_i32 : i32
      %105 = arith.remsi %arg21, %c3_i32 : i32
      %106 = arith.index_cast %105 : i32 to index
      %107 = tt.splat %103 : (i1) -> tensor<128x64xi1, #blocked0>
      %108 = triton_gpu.insert_slice_async %arg17, %arg13, %104, %107 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64x!tt.ptr<f16>, #blocked0> -> tensor<3x128x64xf16, #shared>
      %109 = tt.splat %103 : (i1) -> tensor<64x256xi1, #blocked1>
      %110 = triton_gpu.insert_slice_async %arg18, %arg14, %104, %109 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256x!tt.ptr<f16>, #blocked1> -> tensor<3x64x256xf16, #shared>
      %111 = tt.addptr %arg17, %cst : tensor<128x64x!tt.ptr<f16>, #blocked0>
      %112 = tt.addptr %arg18, %41 : tensor<64x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.async_wait {num = 2 : i32}
      %113 = tensor.extract_slice %108[%106, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<3x128x64xf16, #shared> to tensor<128x64xf16, #shared>
      %114 = tensor.extract_slice %110[%106, 0, 0] [1, 64, 256] [1, 1, 1] : tensor<3x64x256xf16, #shared> to tensor<64x256xf16, #shared>
      %115 = arith.addi %arg20, %c1_i32 : i32
      %116 = arith.addi %arg21, %c1_i32 : i32
      %117 = tensor.extract_slice %113[0, 0] [128, 16] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x16xf16, #shared>
      %119 = tensor.extract_slice %114[0, 0] [16, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<16x256xf16, #shared>
      %691 = triton_gpu.convert_layout %117 : (tensor<128x16xf16, #shared>) -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %711 = triton_gpu.convert_layout %119 : (tensor<16x256xf16, #shared>) -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %891 = tt.dot %691, %711, %99 {allowTF32 = true, transA = false, transB = false} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x256xf32, #mma>
      %901 = tensor.extract_slice %113[0, 16] [128, 32] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x32xf16, #shared>
      %911 = triton_gpu.convert_layout %901 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %921 = tensor.extract_slice %114[16, 0] [32, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<32x256xf16, #shared>
      %931 = triton_gpu.convert_layout %921 : (tensor<32x256xf16, #shared>) -> tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %941 = tt.dot %911, %931, %891 {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x256xf32, #mma>
      %951 = tensor.extract_slice %113[0, 48] [128, 16] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x16xf16, #shared>
      %961 = triton_gpu.convert_layout %951 : (tensor<128x16xf16, #shared>) -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %971 = tensor.extract_slice %114[48, 0] [16, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<16x256xf16, #shared>
      %981 = triton_gpu.convert_layout %971 : (tensor<16x256xf16, #shared>) -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %991 = tt.dot %961, %981, %941 {allowTF32 = true, transA = false, transB = false} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x256xf32, #mma>
      %1001 = tt.addptr %100, %cst : tensor<128x64x!tt.ptr<f16>, #blocked0>
      %1011 = tt.addptr %101, %41 : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %1021 = arith.addi %102, %c64 : index
      %1031 = arith.cmpi slt, %1021, %39 : index
      %1041 = arith.remsi %115, %c3_i32 : i32
      %1051 = arith.remsi %116, %c3_i32 : i32
      %1061 = arith.index_cast %1051 : i32 to index
      %1071 = tt.splat %1031 : (i1) -> tensor<128x64xi1, #blocked0>
      %1081 = triton_gpu.insert_slice_async %111, %108, %1041, %1071 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64x!tt.ptr<f16>, #blocked0> -> tensor<3x128x64xf16, #shared>
      %1091 = tt.splat %1031 : (i1) -> tensor<64x256xi1, #blocked1>
      %1101 = triton_gpu.insert_slice_async %112, %110, %1041, %1091 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256x!tt.ptr<f16>, #blocked1> -> tensor<3x64x256xf16, #shared>
      %1111 = tt.addptr %111, %cst : tensor<128x64x!tt.ptr<f16>, #blocked0>
      %1121 = tt.addptr %112, %41 : tensor<64x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.async_wait {num = 2 : i32}
      %1131 = tensor.extract_slice %1081[%1061, 0, 0] [1, 128, 64] [1, 1, 1] : tensor<3x128x64xf16, #shared> to tensor<128x64xf16, #shared>
      %1141 = tensor.extract_slice %1101[%1061, 0, 0] [1, 64, 256] [1, 1, 1] : tensor<3x64x256xf16, #shared> to tensor<64x256xf16, #shared>
      %1151 = arith.addi %115, %c1_i32 : i32
      %1161 = arith.addi %116, %c1_i32 : i32
      %1171 = tensor.extract_slice %1131[0, 0] [128, 16] [1, 1] : tensor<128x64xf16, #shared> to tensor<128x16xf16, #shared>
      %1191 = tensor.extract_slice %1141[0, 0] [16, 256] [1, 1] : tensor<64x256xf16, #shared> to tensor<16x256xf16, #shared>
      scf.yield %991, %1001, %1011, %1081, %1101, %1131, %1141, %1111, %1121, %1021, %1151, %1161, %1171, %1191 : tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked0>, tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<3x128x64xf16, #shared>, tensor<3x64x256xf16, #shared>, tensor<128x64xf16, #shared>, tensor<64x256xf16, #shared>, tensor<128x64x!tt.ptr<f16>, #blocked0>, tensor<64x256x!tt.ptr<f16>, #blocked1>, index, i32, i32, tensor<128x16xf16, #shared>, tensor<16x256xf16, #shared>
    }
    triton_gpu.async_wait {num = 0 : i32}
    %73 = triton_gpu.convert_layout %72#0 : (tensor<128x256xf32, #mma>) -> tensor<128x256xf32, #blocked1>
    %74 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked1>
    %75 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %76 = tt.broadcast %36 : (tensor<1x256xi32, #blocked1>) -> tensor<128x256xi32, #blocked1>
    %77 = tt.splat %arg3 : (i32) -> tensor<128x1xi32, #blocked1>
    %78 = tt.splat %arg4 : (i32) -> tensor<1x256xi32, #blocked1>
    %79 = "triton_gpu.cmpi"(%36, %78) {predicate = 2 : i64} : (tensor<1x256xi32, #blocked1>, tensor<1x256xi32, #blocked1>) -> tensor<1x256xi1, #blocked1>
    %80 = tt.broadcast %79 : (tensor<1x256xi1, #blocked1>) -> tensor<128x256xi1, #blocked1>
    %81 = arith.muli %74, %26 : tensor<128x1xi32, #blocked1>
    %82 = tt.addptr %75, %81 : tensor<128x1x!tt.ptr<f16>, #blocked1>
    %83 = tt.broadcast %82 : (tensor<128x1x!tt.ptr<f16>, #blocked1>) -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %84 = tt.addptr %83, %76 : tensor<128x256x!tt.ptr<f16>, #blocked1>
    %85 = arith.truncf %73 : tensor<128x256xf32, #blocked1> to tensor<128x256xf16, #blocked1>
    %86 = "triton_gpu.cmpi"(%26, %77) {predicate = 2 : i64} : (tensor<128x1xi32, #blocked1>, tensor<128x1xi32, #blocked1>) -> tensor<128x1xi1, #blocked1>
    %87 = tt.broadcast %86 : (tensor<128x1xi1, #blocked1>) -> tensor<128x256xi1, #blocked1>
    %88 = arith.andi %87, %80 : tensor<128x256xi1, #blocked1>
    tt.store %84, %85, %88 : tensor<128x256xf16, #blocked1>
    return
  }
}
"""



# %%
# Motivations
# -------------
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc;
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# ----------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetics.
#
# Pointer Arithmetics
# ~~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given b
# y :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as:
#
#  .. code-block:: python
#
#    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program. and unfortunately, a
# a simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = triton.program_id(0);
#    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M;
#    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N;
#    pid_m = pid / grid_n;
#    pid_n = pid % grid_n;
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # program ID
#    pid = tl.program_id(axis=0)
#    # number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # if `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *within groups*, programs are ordered in a column-major order
#    # row-id of the program in the *launch grid*
#    pid_m = first_pid_m + (pid % group_size_m)
#    # col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# -------------
#

import torch

import triton
import triton.language as tl
import triton.testing

# %
# :code:`triton.jit`'ed functions can be auto-tuned by using the `triton.autotune`
# decorator, which consumes:
#   - A list of :code:`triton.Config` objects that define different configurations of
#       meta-parameters (e.g., BLOCK_SIZE_M) and compilation options (e.g., num_warps) to try
#   - An autotuning *key* whose change in values will trigger evaluation of all the
#       provided configs


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # you can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION:
        accumulator = ACTIVATION(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel

ttgir_kernel = None

def matmul(a, b, activation=None):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    global ttgir_kernel
    if ttgir_kernel is None:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
            f.write(IR)
            f.flush()
            ttgir_kernel = triton.compile(f.name, num_warps=8)
    ttgir_kernel[(2048, 1, 1)](
        a.data_ptr(), b.data_ptr(), c.data_ptr(),
        M, N, K,
        a.stride(0),
        b.stride(0),
        c.stride(0)
    )
    #k = matmul_kernel[grid](
    #    a, b, c,
    #    M, N, K,
    #    a.stride(0), a.stride(1),
    #    b.stride(0), b.stride(1),
    #    c.stride(0), c.stride(1),
    #    ACTIVATION=None,
    #)
    return c


# %%
# Unit Test
# -----------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS)

torch.manual_seed(0)
a = torch.randn((8192, 8192), device='cuda', dtype=torch.float16)
b = torch.randn((8192, 8192), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b, activation=None)
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if triton.testing.allclose(triton_output, torch_output):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# %%
# Benchmark
# --------------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now compare the performance of our kernel against that of cuBLAS. Here we focus on square matrices, but feel free to arrange this script as you wish to benchmark any other matrix shape.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[
            8192
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'],
        # label name for the lines
        line_names=["cuBLAS", "Triton"],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    with triton.testing.set_gpu_clock():
        if provider == 'cublas':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=1000)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), rep=1000)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
