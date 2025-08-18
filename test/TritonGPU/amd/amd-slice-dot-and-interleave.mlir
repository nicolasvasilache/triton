// RUN: triton-opt %s -split-input-file --pass-pipeline="builtin.module(tt.func(tritonamdgpu-dot-slice-and-interleave{target-slice-mnk=32,32,32}))" | FileCheck %s --check-prefix=SLICE

//   SLICE-LABEL: slice_8_dots
// Check that we get memory slicing and local loads (interleaved pattern)
//         SLICE:   amdgpu.extract_slice {{.*}}[0, 0] : tensor<64x64xf32, {{.*}}> to tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.extract_slice {{.*}}[0, 32] : tensor<64x64xf32, {{.*}}> to tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.extract_slice {{.*}}[32, 0] : tensor<64x64xf32, {{.*}}> to tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.extract_slice {{.*}}[32, 32] : tensor<64x64xf32, {{.*}}> to tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<32x32xf16, {{.*}}> * tensor<32x32xf16, {{.*}}> -> tensor<32x32xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.concat {{.*}} : tensor<32x32xf32, {{.*}}> -> tensor<64x64xf32, {{.*}}>
#mma2 = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @slice_8_dots(%smemA: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %smemB: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma2> {
    %A = ttg.local_load %smemA : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 4}>>
    %B = ttg.local_load %smemB : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 4}>>
    %C0 = arith.constant dense<0.0> : tensor<64x64xf32, #mma2>
    %D = tt.dot %A, %B, %C0 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 4}>> -> tensor<64x64xf32, #mma2>
    tt.return %D : tensor<64x64xf32, #mma2>
  }
}

// -----

//   SLICE-LABEL: slice_multi_warp
// Check that we get memory slicing and operations (interleaved pattern)
//         SLICE:   amdgpu.extract_slice {{.*}}[0, 0] : tensor<128x128xf32, {{.*}}> to tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.extract_slice {{.*}}[0, 64] : tensor<128x128xf32, {{.*}}> to tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 64] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 64] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.extract_slice {{.*}}[64, 0] : tensor<128x128xf32, {{.*}}> to tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[64, 0] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[64, 32] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.extract_slice {{.*}}[64, 64] : tensor<128x128xf32, {{.*}}> to tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[64, 0] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[0, 64] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//
//         SLICE:   ttg.memdesc_subslice {{.*}}[64, 32] : !ttg.memdesc<128x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   ttg.memdesc_subslice {{.*}}[32, 64] : !ttg.memdesc<64x128xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//         SLICE:   ttg.local_load
//         SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
///////////
//         SLICE:   amdgpu.concat {{.*}} : tensor<64x64xf32, {{.*}}> -> tensor<128x128xf32, {{.*}}>
#mma4 = #ttg.amd_mfma<{version = 2, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem4 = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @slice_multi_warp(%smemA: !ttg.memdesc<128x64xf16, #shared4, #smem4, mutable>, %smemB: !ttg.memdesc<64x128xf16, #shared4, #smem4, mutable>) -> tensor<128x128xf32, #mma4> {
    %A = ttg.local_load %smemA : !ttg.memdesc<128x64xf16, #shared4, #smem4, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma4, kWidth = 4}>>
    %B = ttg.local_load %smemB : !ttg.memdesc<64x128xf16, #shared4, #smem4, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma4, kWidth = 4}>>
    %C0 = arith.constant dense<0.0> : tensor<128x128xf32, #mma4>
    %D = tt.dot %A, %B, %C0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma4, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma4, kWidth = 4}>> -> tensor<128x128xf32, #mma4>
    tt.return %D : tensor<128x128xf32, #mma4>
  }
}

// -----

// Test that pass doesn't modify dots with non-local_load operands
// SLICE-LABEL: no_slice_non_local_load
//       SLICE:   tt.dot {{.*}} : tensor<64x64xf16, {{.*}}> * tensor<64x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//   SLICE-NOT:   ttg.memdesc_subslice
//   SLICE-NOT:   amdgpu.extract_slice
//   SLICE-NOT:   amdgpu.concat
#mma_neg1 = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @no_slice_non_local_load(%A: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_neg1, kWidth = 4}>>, %B: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_neg1, kWidth = 4}>>) -> tensor<64x64xf32, #mma_neg1> {
    %C0 = arith.constant dense<0.0> : tensor<64x64xf32, #mma_neg1>
    %D = tt.dot %A, %B, %C0 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_neg1, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_neg1, kWidth = 4}>> -> tensor<64x64xf32, #mma_neg1>
    tt.return %D : tensor<64x64xf32, #mma_neg1>
  }
}

// -----

// Test that pass doesn't modify dots when warpsPerCTA prevents slicing.
// With warpsPerCTA=[2, 2] and instrShape=[32, 32], minimum tile is 64x64.
// But we only have 64x64 input, so no slicing is possible along M or N
// However, we are still able to slice along K.
//
// SLICE-LABEL: no_slice_warps_constraint
//   Noop extract_slice
//       SLICE:   amdgpu.extract_slice %{{.*}} [0, 0] : tensor<64x64xf32, #mma> to tensor<64x64xf32, #mma>
//
//       SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//       SLICE:   ttg.local_load
//       SLICE:   ttg.memdesc_subslice {{.*}}[0, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//       SLICE:   ttg.local_load
//       SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//
//       SLICE:   ttg.memdesc_subslice {{.*}}[0, 32] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<64x32xf16, {{.*}}>
//       SLICE:   ttg.local_load
//       SLICE:   ttg.memdesc_subslice {{.*}}[32, 0] : !ttg.memdesc<64x64xf16, {{.*}}> -> !ttg.memdesc<32x64xf16, {{.*}}>
//       SLICE:   ttg.local_load
//       SLICE:   tt.dot {{.*}} : tensor<64x32xf16, {{.*}}> * tensor<32x64xf16, {{.*}}> -> tensor<64x64xf32, {{.*}}>
//
//   SLICE-NOT:   concat
//       SLICE:   return %{{.*}} : tensor<64x64xf32, #mma>
#mma_neg_warp = #ttg.amd_mfma<{version = 2, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#shared_neg_warp = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem_neg_warp = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @no_slice_warps_constraint(%smemA: !ttg.memdesc<64x64xf16, #shared_neg_warp, #smem_neg_warp, mutable>, %smemB: !ttg.memdesc<64x64xf16, #shared_neg_warp, #smem_neg_warp, mutable>) -> tensor<64x64xf32, #mma_neg_warp> {
    %A = ttg.local_load %smemA : !ttg.memdesc<64x64xf16, #shared_neg_warp, #smem_neg_warp, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_neg_warp, kWidth = 4}>>
    %B = ttg.local_load %smemB : !ttg.memdesc<64x64xf16, #shared_neg_warp, #smem_neg_warp, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_neg_warp, kWidth = 4}>>
    %C0 = arith.constant dense<0.0> : tensor<64x64xf32, #mma_neg_warp>
    %D = tt.dot %A, %B, %C0 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_neg_warp, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_neg_warp, kWidth = 4}>> -> tensor<64x64xf32, #mma_neg_warp>
    tt.return %D : tensor<64x64xf32, #mma_neg_warp>
  }
}
