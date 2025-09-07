// -----// Input IR
#blocked = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc1 = loc("/home/nico/triton/sandbox/matmul.py":30:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
#loc41 = loc("a_ptr"(#loc1))
#loc42 = loc("b_ptr"(#loc1))
#loc43 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %base_offsets_nd = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc75)
    %shift_1d = arith.constant 131072 : i32 loc(#loc93)
    %cst = arith.constant dense<2048> : tensor<32x1xi32, #blocked> loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %c65536_i32 = arith.constant 65536 : i32 loc(#loc)
    %cst_0 = arith.constant dense<16384> : tensor<64x1xi32, #blocked> loc(#loc)
    %c1048576_i32 = arith.constant 1048576 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc6)
    %c512_i32 = arith.constant 512 : i32 loc(#loc6)
    %c0_i32 = arith.constant 0 : i32 loc(#loc6)
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc46)
    %pid0 = tt.get_program_id x : i32 loc(#loc77)
    %pid1 = tt.get_program_id y : i32 loc(#loc78)
    %pid2 = tt.get_program_id z : i32 loc(#loc79)
    %npg1 = tt.get_num_programs y : i32 loc(#loc80)
    %npg2 = tt.get_num_programs z : i32 loc(#loc81)
    %stride_val = arith.muli %npg2, %npg1 : i32 loc(#loc103)
    %linear_idx = arith.muli %pid0, %stride_val : i32 loc(#loc95)
    %linear_idx_1 = arith.muli %pid1, %npg2 : i32 loc(#loc95)
    %linear_idx_2 = arith.addi %linear_idx, %linear_idx_1 : i32 loc(#loc96)
    %linear_idx_3 = arith.addi %linear_idx_2, %pid2 : i32 loc(#loc96)
    %idx_val = arith.divsi %linear_idx_3, %c32_i32 : i32 loc(#loc83)
    %remaining = arith.remsi %linear_idx_3, %c32_i32 : i32 loc(#loc84)
    %acc_4 = scf.for %k = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%acc_16 = %acc) -> (tensor<64x64xf32, #mma>)  : i32 {
      %base_offsets_nd_17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc85)
      %base_offsets_nd_18 = tt.expand_dims %base_offsets_nd_17 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc85)
      %base_offsets_nd_19 = arith.muli %base_offsets_nd_18, %cst_0 : tensor<64x1xi32, #blocked> loc(#loc85)
      %base_offsets_nd_20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc85)
      %base_offsets_nd_21 = tt.expand_dims %base_offsets_nd_20 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc85)
      %base_offsets_nd_22 = tt.broadcast %base_offsets_nd_19 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc85)
      %base_offsets_nd_23 = tt.broadcast %base_offsets_nd_21 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc85)
      %base_offsets_nd_24 = arith.addi %base_offsets_nd_22, %base_offsets_nd_23 : tensor<64x32xi32, #blocked> loc(#loc85)
      %shift_1d_25 = arith.muli %idx_val, %c1048576_i32 : i32 loc(#loc97)
      %shift_1d_26 = arith.muli %k, %c32_i32 : i32 loc(#loc97)
      %res_27 = arith.addi %shift_1d_25, %shift_1d_26 : i32 loc(#loc98)
      %4 = tt.splat %res_27 : i32 -> tensor<64x32xi32, #blocked> loc(#loc64)
      %5 = arith.addi %base_offsets_nd_24, %4 : tensor<64x32xi32, #blocked> loc(#loc64)
      %base_offsets_nd_28 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc88)
      %base_offsets_nd_29 = tt.expand_dims %base_offsets_nd_28 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked> loc(#loc88)
      %base_offsets_nd_30 = arith.muli %base_offsets_nd_29, %cst : tensor<32x1xi32, #blocked> loc(#loc88)
      %base_offsets_nd_31 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc88)
      %base_offsets_nd_32 = tt.expand_dims %base_offsets_nd_31 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc88)
      %base_offsets_nd_33 = tt.broadcast %base_offsets_nd_30 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc88)
      %base_offsets_nd_34 = tt.broadcast %base_offsets_nd_32 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc88)
      %base_offsets_nd_35 = arith.addi %base_offsets_nd_33, %base_offsets_nd_34 : tensor<32x64xi32, #blocked> loc(#loc88)
      %shift_1d_36 = arith.muli %k, %c65536_i32 : i32 loc(#loc99)
      %shift_1d_37 = arith.muli %remaining, %c64_i32 : i32 loc(#loc99)
      %res_38 = arith.addi %shift_1d_36, %shift_1d_37 : i32 loc(#loc100)
      %6 = tt.splat %res_38 : i32 -> tensor<32x64xi32, #blocked> loc(#loc65)
      %7 = arith.addi %base_offsets_nd_35, %6 : tensor<32x64xi32, #blocked> loc(#loc65)
      %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc66)
      %a_39 = tt.addptr %a, %5 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc66)
      %a_40 = tt.load %a_39 : tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc67)
      %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc68)
      %b_41 = tt.addptr %b, %7 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked> loc(#loc68)
      %b_42 = tt.load %b_41 : tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc69)
      %a_43 = ttg.convert_layout %a_40 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc70)
      %b_44 = ttg.convert_layout %b_42 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc71)
      %acc_45 = tt.dot %a_43, %b_44, %acc_16 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma> loc(#loc72)
      scf.yield %acc_45 : tensor<64x64xf32, #mma> loc(#loc36)
    } loc(#loc60)
    %acc_5 = ttg.convert_layout %acc_4 : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked> loc(#loc73)
    %base_offsets_nd_6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc75)
    %base_offsets_nd_7 = tt.expand_dims %base_offsets_nd_6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc75)
    %base_offsets_nd_8 = arith.muli %base_offsets_nd_7, %base_offsets_nd : tensor<64x1xi32, #blocked> loc(#loc75)
    %base_offsets_nd_9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc75)
    %base_offsets_nd_10 = tt.expand_dims %base_offsets_nd_9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc75)
    %base_offsets_nd_11 = tt.broadcast %base_offsets_nd_8 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc75)
    %base_offsets_nd_12 = tt.broadcast %base_offsets_nd_10 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc75)
    %base_offsets_nd_13 = arith.addi %base_offsets_nd_11, %base_offsets_nd_12 : tensor<64x64xi32, #blocked> loc(#loc75)
    %shift_1d_14 = arith.muli %idx_val, %shift_1d : i32 loc(#loc101)
    %shift_1d_15 = arith.muli %remaining, %c64_i32 : i32 loc(#loc101)
    %res = arith.addi %shift_1d_14, %shift_1d_15 : i32 loc(#loc102)
    %0 = tt.splat %res : i32 -> tensor<64x64xi32, #blocked> loc(#loc74)
    %1 = arith.addi %base_offsets_nd_13, %0 : tensor<64x64xi32, #blocked> loc(#loc74)
    %2 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc38)
    %3 = tt.addptr %2, %1 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc38)
    tt.store %3, %acc_5 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc39)
    tt.return loc(#loc40)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/nico/triton/sandbox/nd_helpers.py":50:8)
#loc3 = loc("/home/nico/triton/sandbox/matmul.py":82:91)
#loc4 = loc("/home/nico/triton/sandbox/tuple_helpers.py":116:11)
#loc5 = loc("/home/nico/triton/sandbox/nd_helpers.py":54:44)
#loc6 = loc("/home/nico/triton/sandbox/matmul.py":50:25)
#loc7 = loc("/home/nico/triton/sandbox/matmul.py":43:33)
#loc8 = loc("/home/nico/triton/sandbox/tuple_helpers.py":83:25)
#loc9 = loc("/home/nico/triton/sandbox/matmul.py":40:24)
#loc10 = loc("/home/nico/triton/sandbox/tuple_helpers.py":84:25)
#loc11 = loc("/home/nico/triton/sandbox/tuple_helpers.py":85:25)
#loc12 = loc("/home/nico/triton/sandbox/tuple_helpers.py":87:27)
#loc13 = loc("/home/nico/triton/sandbox/tuple_helpers.py":88:27)
#loc14 = loc("/home/nico/triton/sandbox/tuple_helpers.py":27:38)
#loc15 = loc("/home/nico/triton/sandbox/tuple_helpers.py":48:30)
#loc16 = loc("/home/nico/triton/sandbox/tuple_helpers.py":89:51)
#loc17 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:35)
#loc18 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:22)
#loc19 = loc("/home/nico/triton/sandbox/tuple_helpers.py":73:31)
#loc20 = loc("/home/nico/triton/sandbox/matmul.py":41:52)
#loc21 = loc("/home/nico/triton/sandbox/tuple_helpers.py":75:32)
#loc22 = loc("/home/nico/triton/sandbox/matmul.py":55:91)
#loc23 = loc("/home/nico/triton/sandbox/tuple_helpers.py":115:35)
#loc24 = loc("/home/nico/triton/sandbox/nd_helpers.py":54:8)
#loc25 = loc("/home/nico/triton/sandbox/tuple_helpers.py":132:15)
#loc26 = loc("/home/nico/triton/sandbox/nd_helpers.py":52:32)
#loc27 = loc("/home/nico/triton/sandbox/nd_helpers.py":55:29)
#loc28 = loc("/home/nico/triton/sandbox/matmul.py":56:91)
#loc29 = loc("/home/nico/triton/sandbox/matmul.py":62:32)
#loc30 = loc("/home/nico/triton/sandbox/matmul.py":62:39)
#loc31 = loc("/home/nico/triton/sandbox/matmul.py":63:32)
#loc32 = loc("/home/nico/triton/sandbox/matmul.py":63:39)
#loc33 = loc("/home/nico/triton/sandbox/matmul.py":73:37)
#loc34 = loc("/home/nico/triton/sandbox/matmul.py":74:37)
#loc35 = loc("/home/nico/triton/sandbox/matmul.py":75:42)
#loc36 = loc("/home/nico/triton/sandbox/matmul.py":66:8)
#loc37 = loc("/home/nico/triton/sandbox/matmul.py":79:33)
#loc38 = loc("/home/nico/triton/sandbox/matmul.py":88:25)
#loc39 = loc("/home/nico/triton/sandbox/matmul.py":88:41)
#loc40 = loc("/home/nico/triton/sandbox/matmul.py":90:4)
#loc44 = loc("base_offsets_nd"(#loc2))
#loc45 = loc("shift_1d"(#loc5))
#loc46 = loc("acc"(#loc7))
#loc47 = loc("pid0"(#loc8))
#loc48 = loc("linear_program_id"(#loc9))
#loc49 = loc("pid1"(#loc10))
#loc50 = loc("pid2"(#loc11))
#loc51 = loc("npg1"(#loc12))
#loc52 = loc("npg2"(#loc13))
#loc53 = loc("stride_val"(#loc14))
#loc54 = loc("strides"(#loc15))
#loc55 = loc("linear_idx"(#loc17))
#loc56 = loc("linear_idx"(#loc18))
#loc57 = loc("idx_val"(#loc19))
#loc58 = loc("start_blocks_c"(#loc20))
#loc59 = loc("remaining"(#loc21))
#loc60 = loc("acc"(#loc6))
#loc61 = loc("shift_1d"(#loc24))
#loc62 = loc("res"(#loc25))
#loc63 = loc("shift_1d"(#loc26))
#loc64 = loc(callsite(#loc27 at #loc22))
#loc65 = loc(callsite(#loc27 at #loc28))
#loc66 = loc("a"(#loc29))
#loc67 = loc("a"(#loc30))
#loc68 = loc("b"(#loc31))
#loc69 = loc("b"(#loc32))
#loc70 = loc("a"(#loc33))
#loc71 = loc("b"(#loc34))
#loc72 = loc("acc"(#loc35))
#loc73 = loc("acc"(#loc37))
#loc74 = loc(callsite(#loc27 at #loc3))
#loc75 = loc(callsite(#loc44 at #loc3))
#loc76 = loc(callsite(#loc45 at #loc3))
#loc77 = loc(callsite(#loc47 at #loc48))
#loc78 = loc(callsite(#loc49 at #loc48))
#loc79 = loc(callsite(#loc50 at #loc48))
#loc80 = loc(callsite(#loc51 at #loc48))
#loc81 = loc(callsite(#loc52 at #loc48))
#loc82 = loc(callsite(#loc16 at #loc48))
#loc83 = loc(callsite(#loc57 at #loc58))
#loc84 = loc(callsite(#loc59 at #loc58))
#loc85 = loc(callsite(#loc44 at #loc22))
#loc86 = loc(callsite(#loc61 at #loc22))
#loc87 = loc(callsite(#loc63 at #loc22))
#loc88 = loc(callsite(#loc44 at #loc28))
#loc89 = loc(callsite(#loc61 at #loc28))
#loc90 = loc(callsite(#loc63 at #loc28))
#loc91 = loc(callsite(#loc61 at #loc3))
#loc92 = loc(callsite(#loc63 at #loc3))
#loc93 = loc(callsite(#loc4 at #loc76))
#loc94 = loc(callsite(#loc54 at #loc82))
#loc95 = loc(callsite(#loc55 at #loc82))
#loc96 = loc(callsite(#loc56 at #loc82))
#loc97 = loc(callsite(#loc23 at #loc86))
#loc98 = loc(callsite(#loc62 at #loc87))
#loc99 = loc(callsite(#loc23 at #loc89))
#loc100 = loc(callsite(#loc62 at #loc90))
#loc101 = loc(callsite(#loc23 at #loc91))
#loc102 = loc(callsite(#loc62 at #loc92))
#loc103 = loc(callsite(#loc53 at #loc94))

// -----// IR Dump Before OptimizeAMDLDSUsage (optimize-amd-lds-usage) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc1 = loc("/home/nico/triton/sandbox/matmul.py":30:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#loc37 = loc("a_ptr"(#loc1))
#loc38 = loc("b_ptr"(#loc1))
#loc39 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %acc = arith.constant 511 : i32 loc(#loc40)
    %true = arith.constant true loc(#loc)
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1048576_i32 = arith.constant 1048576 : i32 loc(#loc)
    %cst_0 = arith.constant dense<16384> : tensor<64x1xi32, #blocked> loc(#loc)
    %c65536_i32 = arith.constant 65536 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst_1 = arith.constant dense<2048> : tensor<32x1xi32, #blocked> loc(#loc)
    %c131072_i32 = arith.constant 131072 : i32 loc(#loc)
    %cst_2 = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc)
    %pid0 = tt.get_program_id x : i32 loc(#loc69)
    %pid1 = tt.get_program_id y : i32 loc(#loc70)
    %pid2 = tt.get_program_id z : i32 loc(#loc71)
    %npg1 = tt.get_num_programs y : i32 loc(#loc72)
    %npg2 = tt.get_num_programs z : i32 loc(#loc73)
    %stride_val = arith.muli %npg2, %npg1 : i32 loc(#loc95)
    %linear_idx = arith.muli %pid0, %stride_val : i32 loc(#loc87)
    %linear_idx_3 = arith.muli %pid1, %npg2 : i32 loc(#loc87)
    %linear_idx_4 = arith.addi %linear_idx, %linear_idx_3 : i32 loc(#loc88)
    %linear_idx_5 = arith.addi %linear_idx_4, %pid2 : i32 loc(#loc88)
    %idx_val = arith.divsi %linear_idx_5, %c32_i32 : i32 loc(#loc75)
    %remaining = arith.remsi %linear_idx_5, %c32_i32 : i32 loc(#loc76)
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf32, #shared, #smem, mutable> loc(#loc54)
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf32, #shared1, #smem, mutable> loc(#loc55)
    %base_offsets_nd = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc77)
    %base_offsets_nd_6 = tt.expand_dims %base_offsets_nd {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc77)
    %base_offsets_nd_7 = arith.muli %base_offsets_nd_6, %cst_0 : tensor<64x1xi32, #blocked> loc(#loc77)
    %base_offsets_nd_8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc77)
    %base_offsets_nd_9 = tt.expand_dims %base_offsets_nd_8 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc77)
    %base_offsets_nd_10 = tt.broadcast %base_offsets_nd_7 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc77)
    %base_offsets_nd_11 = tt.broadcast %base_offsets_nd_9 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc77)
    %base_offsets_nd_12 = arith.addi %base_offsets_nd_10, %base_offsets_nd_11 : tensor<64x32xi32, #blocked> loc(#loc77)
    %shift_1d = arith.muli %idx_val, %c1048576_i32 : i32 loc(#loc89)
    %res = arith.addi %shift_1d, %c0_i32 : i32 loc(#loc90)
    %0 = tt.splat %res : i32 -> tensor<64x32xi32, #blocked> loc(#loc60)
    %1 = arith.addi %base_offsets_nd_12, %0 : tensor<64x32xi32, #blocked> loc(#loc60)
    %base_offsets_nd_13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc80)
    %base_offsets_nd_14 = tt.expand_dims %base_offsets_nd_13 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked> loc(#loc80)
    %base_offsets_nd_15 = arith.muli %base_offsets_nd_14, %cst_1 : tensor<32x1xi32, #blocked> loc(#loc80)
    %base_offsets_nd_16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc80)
    %base_offsets_nd_17 = tt.expand_dims %base_offsets_nd_16 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_18 = tt.broadcast %base_offsets_nd_15 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_19 = tt.broadcast %base_offsets_nd_17 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_20 = arith.addi %base_offsets_nd_18, %base_offsets_nd_19 : tensor<32x64xi32, #blocked> loc(#loc80)
    %shift_1d_21 = arith.muli %remaining, %c64_i32 : i32 loc(#loc91)
    %res_22 = arith.addi %c0_i32, %shift_1d_21 : i32 loc(#loc92)
    %2 = tt.splat %res_22 : i32 -> tensor<32x64xi32, #blocked> loc(#loc61)
    %3 = arith.addi %base_offsets_nd_20, %2 : tensor<32x64xi32, #blocked> loc(#loc61)
    %a_23 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc62)
    %a_24 = tt.addptr %a_23, %1 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc62)
    %4 = tt.splat %true : i1 -> tensor<64x32xi1, #blocked> loc(#loc)
    %a_25 = tt.load %a_24, %4 {amd.pipeliner_part = "prologue"} : tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc54)
    %5 = tt.splat %true : i1 -> tensor<64x32xi1, #blocked> loc(#loc)
    %a_26 = tt.load %a_24, %5 {amd.pipeliner_part = "prologue"} : tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc54)
    %b_27 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc63)
    %b_28 = tt.addptr %b_27, %3 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked> loc(#loc63)
    %6 = tt.splat %true : i1 -> tensor<32x64xi1, #blocked> loc(#loc)
    %b_29 = tt.load %b_28, %6 {amd.pipeliner_part = "prologue"} : tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc55)
    %7 = tt.splat %true : i1 -> tensor<32x64xi1, #blocked> loc(#loc)
    %b_30 = tt.load %b_28, %7 {amd.pipeliner_part = "prologue"} : tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc55)
    %acc_31 = arith.cmpi slt, %c0_i32, %c1_i32 : i32 loc(#loc40)
    %acc_32 = arith.select %acc_31, %c0_i32, %c0_i32 : i32 loc(#loc40)
    %a_33 = ttg.memdesc_index %a[%acc_32] : !ttg.memdesc<1x64x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable, 1x64x32> loc(#loc54)
    ttg.local_store %a_25, %a_33 : tensor<64x32xf32, #blocked> -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable, 1x64x32> loc(#loc54)
    %b_34 = ttg.memdesc_index %b[%acc_32] : !ttg.memdesc<1x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 1x32x64> loc(#loc55)
    ttg.local_store %b_29, %b_34 : tensor<32x64xf32, #blocked> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 1x32x64> loc(#loc55)
    %a_35 = ttg.local_load %a_33 : !ttg.memdesc<64x32xf32, #shared, #smem, mutable, 1x64x32> -> tensor<64x32xf32, #blocked> loc(#loc54)
    %b_36 = ttg.local_load %b_34 : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 1x32x64> -> tensor<32x64xf32, #blocked> loc(#loc55)
    %a_37 = ttg.convert_layout %a_35 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc64)
    %b_38 = ttg.convert_layout %b_36 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc65)
    %acc_39:4 = scf.for %acc_53 = %c0_i32 to %acc step %c1_i32 iter_args(%arg4 = %cst, %acc_54 = %acc_32, %a_55 = %a_37, %b_56 = %b_38) -> (tensor<64x64xf32, #mma>, i32, tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)  : i32 {
      %base_offsets_nd_57 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc77)
      %base_offsets_nd_58 = tt.expand_dims %base_offsets_nd_57 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc77)
      %base_offsets_nd_59 = arith.muli %base_offsets_nd_58, %cst_0 : tensor<64x1xi32, #blocked> loc(#loc77)
      %base_offsets_nd_60 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc77)
      %base_offsets_nd_61 = tt.expand_dims %base_offsets_nd_60 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc77)
      %base_offsets_nd_62 = tt.broadcast %base_offsets_nd_59 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc77)
      %base_offsets_nd_63 = tt.broadcast %base_offsets_nd_61 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc77)
      %base_offsets_nd_64 = arith.addi %base_offsets_nd_62, %base_offsets_nd_63 : tensor<64x32xi32, #blocked> loc(#loc77)
      %shift_1d_65 = arith.muli %idx_val, %c1048576_i32 : i32 loc(#loc89)
      %acc_66 = arith.addi %acc_53, %c1_i32 : i32 loc(#loc40)
      %shift_1d_67 = arith.muli %acc_66, %c32_i32 : i32 loc(#loc89)
      %res_68 = arith.addi %shift_1d_65, %shift_1d_67 : i32 loc(#loc90)
      %12 = tt.splat %res_68 : i32 -> tensor<64x32xi32, #blocked> loc(#loc60)
      %13 = arith.addi %base_offsets_nd_64, %12 : tensor<64x32xi32, #blocked> loc(#loc60)
      %base_offsets_nd_69 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc80)
      %base_offsets_nd_70 = tt.expand_dims %base_offsets_nd_69 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked> loc(#loc80)
      %base_offsets_nd_71 = arith.muli %base_offsets_nd_70, %cst_1 : tensor<32x1xi32, #blocked> loc(#loc80)
      %base_offsets_nd_72 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc80)
      %base_offsets_nd_73 = tt.expand_dims %base_offsets_nd_72 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc80)
      %base_offsets_nd_74 = tt.broadcast %base_offsets_nd_71 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc80)
      %base_offsets_nd_75 = tt.broadcast %base_offsets_nd_73 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc80)
      %base_offsets_nd_76 = arith.addi %base_offsets_nd_74, %base_offsets_nd_75 : tensor<32x64xi32, #blocked> loc(#loc80)
      %acc_77 = arith.addi %acc_53, %c1_i32 : i32 loc(#loc40)
      %shift_1d_78 = arith.muli %acc_77, %c65536_i32 : i32 loc(#loc91)
      %shift_1d_79 = arith.muli %remaining, %c64_i32 : i32 loc(#loc91)
      %res_80 = arith.addi %shift_1d_78, %shift_1d_79 : i32 loc(#loc92)
      %14 = tt.splat %res_80 : i32 -> tensor<32x64xi32, #blocked> loc(#loc61)
      %15 = arith.addi %base_offsets_nd_76, %14 : tensor<32x64xi32, #blocked> loc(#loc61)
      %a_81 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc62)
      %a_82 = tt.addptr %a_81, %13 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc62)
      %a_83 = tt.load %a_82 : tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc54)
      %b_84 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc63)
      %b_85 = tt.addptr %b_84, %15 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked> loc(#loc63)
      %b_86 = tt.load %b_85 : tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc55)
      %acc_87 = tt.dot %a_55, %b_56, %arg4 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma> loc(#loc66)
      %acc_88 = arith.addi %acc_54, %c1_i32 : i32 loc(#loc40)
      %acc_89 = arith.cmpi slt, %acc_88, %c1_i32 : i32 loc(#loc40)
      %acc_90 = arith.select %acc_89, %acc_88, %c0_i32 : i32 loc(#loc40)
      %a_91 = ttg.memdesc_index %a[%acc_90] : !ttg.memdesc<1x64x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable, 1x64x32> loc(#loc54)
      ttg.local_store %a_83, %a_91 : tensor<64x32xf32, #blocked> -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable, 1x64x32> loc(#loc54)
      %b_92 = ttg.memdesc_index %b[%acc_90] : !ttg.memdesc<1x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 1x32x64> loc(#loc55)
      ttg.local_store %b_86, %b_92 : tensor<32x64xf32, #blocked> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 1x32x64> loc(#loc55)
      %a_93 = ttg.local_load %a_91 : !ttg.memdesc<64x32xf32, #shared, #smem, mutable, 1x64x32> -> tensor<64x32xf32, #blocked> loc(#loc54)
      %b_94 = ttg.local_load %b_92 : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 1x32x64> -> tensor<32x64xf32, #blocked> loc(#loc55)
      %a_95 = ttg.convert_layout %a_93 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc64)
      %b_96 = ttg.convert_layout %b_94 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc65)
      scf.yield %acc_87, %acc_90, %a_95, %b_96 : tensor<64x64xf32, #mma>, i32, tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc40)
    } loc(#loc40)
    %acc_40 = scf.if %true -> (tensor<64x64xf32, #mma>) {
      %acc_53 = tt.dot %acc_39#2, %acc_39#3, %acc_39#0 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma> loc(#loc66)
      scf.yield %acc_53 : tensor<64x64xf32, #mma> loc(#loc66)
    } else {
      scf.yield %acc_39#0 : tensor<64x64xf32, #mma> loc(#loc66)
    } loc(#loc66)
    ttg.local_dealloc %b : !ttg.memdesc<1x32x64xf32, #shared1, #smem, mutable> loc(#loc40)
    ttg.local_dealloc %a : !ttg.memdesc<1x64x32xf32, #shared, #smem, mutable> loc(#loc40)
    %acc_41 = ttg.convert_layout %acc_40 : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked> loc(#loc67)
    %base_offsets_nd_42 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc83)
    %base_offsets_nd_43 = tt.expand_dims %base_offsets_nd_42 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc83)
    %base_offsets_nd_44 = arith.muli %base_offsets_nd_43, %cst_2 : tensor<64x1xi32, #blocked> loc(#loc83)
    %base_offsets_nd_45 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc83)
    %base_offsets_nd_46 = tt.expand_dims %base_offsets_nd_45 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc83)
    %base_offsets_nd_47 = tt.broadcast %base_offsets_nd_44 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc83)
    %base_offsets_nd_48 = tt.broadcast %base_offsets_nd_46 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc83)
    %base_offsets_nd_49 = arith.addi %base_offsets_nd_47, %base_offsets_nd_48 : tensor<64x64xi32, #blocked> loc(#loc83)
    %shift_1d_50 = arith.muli %idx_val, %c131072_i32 : i32 loc(#loc93)
    %shift_1d_51 = arith.muli %remaining, %c64_i32 : i32 loc(#loc93)
    %res_52 = arith.addi %shift_1d_50, %shift_1d_51 : i32 loc(#loc94)
    %8 = tt.splat %res_52 : i32 -> tensor<64x64xi32, #blocked> loc(#loc68)
    %9 = arith.addi %base_offsets_nd_49, %8 : tensor<64x64xi32, #blocked> loc(#loc68)
    %10 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc34)
    %11 = tt.addptr %10, %9 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc34)
    tt.store %11, %acc_41 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc35)
    tt.return loc(#loc36)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/nico/triton/sandbox/matmul.py":50:25)
#loc3 = loc("/home/nico/triton/sandbox/tuple_helpers.py":83:25)
#loc4 = loc("/home/nico/triton/sandbox/matmul.py":40:24)
#loc5 = loc("/home/nico/triton/sandbox/tuple_helpers.py":84:25)
#loc6 = loc("/home/nico/triton/sandbox/tuple_helpers.py":85:25)
#loc7 = loc("/home/nico/triton/sandbox/tuple_helpers.py":87:27)
#loc8 = loc("/home/nico/triton/sandbox/tuple_helpers.py":88:27)
#loc9 = loc("/home/nico/triton/sandbox/tuple_helpers.py":27:38)
#loc10 = loc("/home/nico/triton/sandbox/tuple_helpers.py":48:30)
#loc11 = loc("/home/nico/triton/sandbox/tuple_helpers.py":89:51)
#loc12 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:35)
#loc13 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:22)
#loc14 = loc("/home/nico/triton/sandbox/tuple_helpers.py":73:31)
#loc15 = loc("/home/nico/triton/sandbox/matmul.py":41:52)
#loc16 = loc("/home/nico/triton/sandbox/tuple_helpers.py":75:32)
#loc17 = loc("/home/nico/triton/sandbox/matmul.py":62:39)
#loc18 = loc("/home/nico/triton/sandbox/matmul.py":63:39)
#loc19 = loc("/home/nico/triton/sandbox/nd_helpers.py":50:8)
#loc20 = loc("/home/nico/triton/sandbox/matmul.py":55:91)
#loc21 = loc("/home/nico/triton/sandbox/tuple_helpers.py":115:35)
#loc22 = loc("/home/nico/triton/sandbox/nd_helpers.py":54:8)
#loc23 = loc("/home/nico/triton/sandbox/tuple_helpers.py":132:15)
#loc24 = loc("/home/nico/triton/sandbox/nd_helpers.py":52:32)
#loc25 = loc("/home/nico/triton/sandbox/nd_helpers.py":55:29)
#loc26 = loc("/home/nico/triton/sandbox/matmul.py":56:91)
#loc27 = loc("/home/nico/triton/sandbox/matmul.py":62:32)
#loc28 = loc("/home/nico/triton/sandbox/matmul.py":63:32)
#loc29 = loc("/home/nico/triton/sandbox/matmul.py":73:37)
#loc30 = loc("/home/nico/triton/sandbox/matmul.py":74:37)
#loc31 = loc("/home/nico/triton/sandbox/matmul.py":75:42)
#loc32 = loc("/home/nico/triton/sandbox/matmul.py":79:33)
#loc33 = loc("/home/nico/triton/sandbox/matmul.py":82:91)
#loc34 = loc("/home/nico/triton/sandbox/matmul.py":88:25)
#loc35 = loc("/home/nico/triton/sandbox/matmul.py":88:41)
#loc36 = loc("/home/nico/triton/sandbox/matmul.py":90:4)
#loc40 = loc("acc"(#loc2))
#loc41 = loc("pid0"(#loc3))
#loc42 = loc("linear_program_id"(#loc4))
#loc43 = loc("pid1"(#loc5))
#loc44 = loc("pid2"(#loc6))
#loc45 = loc("npg1"(#loc7))
#loc46 = loc("npg2"(#loc8))
#loc47 = loc("stride_val"(#loc9))
#loc48 = loc("strides"(#loc10))
#loc49 = loc("linear_idx"(#loc12))
#loc50 = loc("linear_idx"(#loc13))
#loc51 = loc("idx_val"(#loc14))
#loc52 = loc("start_blocks_c"(#loc15))
#loc53 = loc("remaining"(#loc16))
#loc54 = loc("a"(#loc17))
#loc55 = loc("b"(#loc18))
#loc56 = loc("base_offsets_nd"(#loc19))
#loc57 = loc("shift_1d"(#loc22))
#loc58 = loc("res"(#loc23))
#loc59 = loc("shift_1d"(#loc24))
#loc60 = loc(callsite(#loc25 at #loc20))
#loc61 = loc(callsite(#loc25 at #loc26))
#loc62 = loc("a"(#loc27))
#loc63 = loc("b"(#loc28))
#loc64 = loc("a"(#loc29))
#loc65 = loc("b"(#loc30))
#loc66 = loc("acc"(#loc31))
#loc67 = loc("acc"(#loc32))
#loc68 = loc(callsite(#loc25 at #loc33))
#loc69 = loc(callsite(#loc41 at #loc42))
#loc70 = loc(callsite(#loc43 at #loc42))
#loc71 = loc(callsite(#loc44 at #loc42))
#loc72 = loc(callsite(#loc45 at #loc42))
#loc73 = loc(callsite(#loc46 at #loc42))
#loc74 = loc(callsite(#loc11 at #loc42))
#loc75 = loc(callsite(#loc51 at #loc52))
#loc76 = loc(callsite(#loc53 at #loc52))
#loc77 = loc(callsite(#loc56 at #loc20))
#loc78 = loc(callsite(#loc57 at #loc20))
#loc79 = loc(callsite(#loc59 at #loc20))
#loc80 = loc(callsite(#loc56 at #loc26))
#loc81 = loc(callsite(#loc57 at #loc26))
#loc82 = loc(callsite(#loc59 at #loc26))
#loc83 = loc(callsite(#loc56 at #loc33))
#loc84 = loc(callsite(#loc57 at #loc33))
#loc85 = loc(callsite(#loc59 at #loc33))
#loc86 = loc(callsite(#loc48 at #loc74))
#loc87 = loc(callsite(#loc49 at #loc74))
#loc88 = loc(callsite(#loc50 at #loc74))
#loc89 = loc(callsite(#loc21 at #loc78))
#loc90 = loc(callsite(#loc58 at #loc79))
#loc91 = loc(callsite(#loc21 at #loc81))
#loc92 = loc(callsite(#loc58 at #loc82))
#loc93 = loc(callsite(#loc21 at #loc84))
#loc94 = loc(callsite(#loc58 at #loc85))
#loc95 = loc(callsite(#loc47 at #loc86))

// -----// AMDGCN Dump //----- //
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul                          ; -- Begin function matmul
	.p2align	8
	.type	matmul,@function
matmul:                                 ; @matmul
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.5:
	.file	1 "/home/nico/triton/sandbox" "matmul.py"
	.loc	1 30 0 prologue_end             ; matmul.py:30:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_load_dword s14, s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.6:
.LBB0_0:
.Ltmp1:
	.file	2 "/home/nico/triton/sandbox" "tuple_helpers.py"
	.loc	2 87 27 is_stmt 1               ; tuple_helpers.py:87:27 @[ matmul.py:40:24 ]
	s_load_dword s0, s[0:1], 0x3c
.Ltmp2:
	.file	3 "/home/nico/triton/sandbox" "nd_helpers.py"
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_and_b32_e32 v42, 48, v0
	v_and_b32_e32 v26, 7, v0
	v_lshlrev_b32_e32 v1, 14, v42
	v_lshlrev_b32_e32 v27, 2, v26
.Ltmp3:
	.loc	2 87 27                         ; tuple_helpers.py:87:27 @[ matmul.py:40:24 ]
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s8, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_cmp_lg_u32 s0, 0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lg_u64 s[0:1], 0
	s_addc_u32 s9, s13, 0
	.loc	2 88 27                         ; tuple_helpers.py:88:27 @[ matmul.py:40:24 ]
	s_cmp_lg_u32 s8, 0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lg_u64 s[0:1], 0
	.loc	2 52 35                         ; tuple_helpers.py:52:35 @[ matmul.py:40:24 ]
	s_mul_i32 s1, s9, s15
	.loc	2 88 27                         ; tuple_helpers.py:88:27 @[ matmul.py:40:24 ]
	s_addc_u32 s0, s14, 0
	.loc	2 52 22                         ; tuple_helpers.py:52:22 @[ matmul.py:40:24 ]
	s_add_i32 s1, s1, s16
	s_mul_i32 s8, s1, s0
	s_add_i32 s8, s8, s17
.Ltmp4:
	.loc	2 73 31                         ; tuple_helpers.py:73:31 @[ matmul.py:41:52 ]
	s_ashr_i32 s0, s8, 31
	s_lshr_b32 s0, s0, 27
	s_add_i32 s0, s8, s0
	s_ashr_i32 s10, s0, 5
.Ltmp5:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:55:91 ]
	s_lshl_b32 s1, s10, 20
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v28, v1, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v1, 8, v42
	v_accvgpr_write_b32 a93, v1
	v_lshlrev_b32_e32 v1, 14, v1
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v1, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v1, 1, v42
.Ltmp6:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_accvgpr_write_b32 a95, v1
.Ltmp7:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v1
.Ltmp8:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshl_add_u64 v[10:11], v[2:3], 2, s[2:3]
.Ltmp9:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v12, 9, v42
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v1, v27, s1
.Ltmp10:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[6:7], v[28:29], 2, s[2:3]
	v_ashrrev_i32_e32 v3, 31, v2
.Ltmp11:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v12
.Ltmp12:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[8:9], v[2:3], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[18:21], v[6:7], off
	global_load_dwordx4 v[2:5], v[8:9], off
.Ltmp13:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v1, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v1, 2, v42
.Ltmp14:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_accvgpr_write_b32 a97, v1
.Ltmp15:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v1
	v_accvgpr_write_b32 a94, v12
.Ltmp16:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[12:13], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[22:25], v[10:11], off
	global_load_dwordx4 v[6:9], v[12:13], off
.Ltmp17:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v10, v1, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v1, 10, v42
.Ltmp18:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v11, 31, v10
	v_accvgpr_write_b32 a96, v1
.Ltmp19:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v1
.Ltmp20:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[30:31], v[10:11], 2, s[2:3]
.Ltmp21:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v10, v1, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v1, 3, v42
.Ltmp22:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v11, 31, v10
	v_accvgpr_write_b32 a98, v1
.Ltmp23:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v1
.Ltmp24:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[32:33], v[10:11], 2, s[2:3]
.Ltmp25:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v10, v1, v27, s1
.Ltmp26:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[34:35], v[10:11], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[14:17], v[30:31], off
	global_load_dwordx4 v[10:13], v[34:35], off
.Ltmp27:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v31, 6, v42
	v_or_b32_e32 v30, 11, v42
	v_or_b32_e32 v34, 12, v42
	v_accvgpr_write_b32 a105, v31
	v_lshlrev_b32_e32 v38, 14, v31
	v_lshlrev_b32_e32 v31, 14, v30
	v_or_b32_e32 v1, 4, v42
	v_or_b32_e32 v35, 13, v42
	v_accvgpr_write_b32 a100, v34
	v_lshlrev_b32_e32 v36, 14, v34
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v34, v31, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v29, 5, v42
	v_accvgpr_write_b32 a103, v1
	v_lshlrev_b32_e32 v1, 14, v1
	v_accvgpr_write_b32 a101, v35
	v_lshlrev_b32_e32 v37, 14, v35
.Ltmp28:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v35, 31, v34
	v_accvgpr_write_b32 a104, v29
.Ltmp29:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v29, 14, v29
	v_accvgpr_write_b32 a99, v30
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v30, v1, v27, s1
.Ltmp30:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[46:49], v[32:33], off
	global_load_dwordx4 v[50:53], v[34:35], off
.Ltmp31:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v32, v29, v27, s1
	v_or3_b32 v34, v36, v27, s1
.Ltmp32:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v31, 31, v30
.Ltmp33:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v36, v37, v27, s1
.Ltmp34:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[2:3]
	v_ashrrev_i32_e32 v33, 31, v32
	v_ashrrev_i32_e32 v35, 31, v34
	v_ashrrev_i32_e32 v37, 31, v36
.Ltmp35:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v40, 7, v42
.Ltmp36:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[2:3]
	v_lshl_add_u64 v[32:33], v[32:33], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[54:57], v[30:31], off
	global_load_dwordx4 v[58:61], v[32:33], off
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[30:31], v[36:37], 2, s[2:3]
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[66:69], v[34:35], off
	global_load_dwordx4 v[70:73], v[30:31], off
.Ltmp37:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v30, v38, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v40
	v_or_b32_e32 v39, 14, v42
.Ltmp38:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v31, 31, v30
.Ltmp39:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v34, v1, v27, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v1, 15, v42
	v_accvgpr_write_b32 a102, v39
	v_lshlrev_b32_e32 v39, 14, v39
.Ltmp40:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_accvgpr_write_b32 a107, v1
.Ltmp41:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v1, 14, v1
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v32, v39, v27, s1
.Ltmp42:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[74:77], v[30:31], off
	global_load_dwordx4 v[78:81], v[34:35], off
.Ltmp43:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v30, v1, v27, s1
.Ltmp44:
	.loc	2 75 32                         ; tuple_helpers.py:75:32 @[ matmul.py:41:52 ]
	s_and_b32 s0, s0, 0x3ffffe0
.Ltmp45:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_and_b32_e32 v65, 16, v0
	v_and_b32_e32 v29, 15, v0
.Ltmp46:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v33, 31, v32
	v_ashrrev_i32_e32 v31, 31, v30
.Ltmp47:
	.loc	2 75 32                         ; tuple_helpers.py:75:32 @[ matmul.py:41:52 ]
	s_sub_i32 s0, s8, s0
.Ltmp48:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshlrev_b32_e32 v1, 11, v65
	v_lshlrev_b32_e32 v43, 2, v29
.Ltmp49:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[32:33], v[32:33], 2, s[2:3]
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[2:3]
.Ltmp50:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_or_b32_e32 v27, v1, v43
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:56:91 ]
	s_lshl_b32 s11, s0, 6
.Ltmp51:
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[82:85], v[32:33], off
	global_load_dwordx4 v[86:89], v[30:31], off
.Ltmp52:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v30, s11, v27
.Ltmp53:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u64 v[32:33], v[30:31], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[90:93], v[32:33], off
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	v_lshlrev_b32_e32 v26, 4, v26
	v_lshl_or_b32 v26, v42, 7, v26
	v_add_u32_e32 v44, 0, v26
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v33, 0
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(16)
	ds_write_b128 v44, v[18:21]
	s_waitcnt vmcnt(14)
	ds_write_b128 v44, v[22:25] offset:1024
.Ltmp54:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v32, 0x800, v30
	v_add_u32_e32 v18, 0x1000, v30
	v_add_u32_e32 v22, 0x1800, v30
.Ltmp55:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v19, v33
	v_mov_b32_e32 v23, v33
	v_xor_b32_e32 v27, 16, v26
.Ltmp56:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v24, 0x2000, v30
.Ltmp57:
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[20:21], v[32:33], 2, s[4:5]
	v_lshl_add_u64 v[34:35], v[18:19], 2, s[4:5]
	v_lshl_add_u64 v[22:23], v[22:23], 2, s[4:5]
	v_mov_b32_e32 v25, v33
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[18:21], v[20:21], off
	s_nop 0
	global_load_dwordx4 v[94:97], v[34:35], off
	global_load_dwordx4 v[98:101], v[22:23], off
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[22:23], v[24:25], 2, s[4:5]
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	v_add_u32_e32 v27, 0, v27
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[102:105], v[22:23], off
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v27, v[2:5] offset:128
	s_waitcnt vmcnt(17)
	ds_write_b128 v27, v[6:9] offset:1152
	v_xor_b32_e32 v6, 32, v26
	v_add_u32_e32 v24, 0, v6
	v_xor_b32_e32 v6, 48, v26
.Ltmp58:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, 0x2800, v30
.Ltmp59:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v33
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v25, 0, v6
	v_xor_b32_e32 v6, 64, v26
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v45, 0, v6
	v_xor_b32_e32 v6, 0x50, v26
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[2:5], v[2:3], off
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(17)
	ds_write_b128 v24, v[14:17] offset:256
	s_waitcnt vmcnt(15)
	ds_write_b128 v24, v[46:49] offset:1280
	v_add_u32_e32 v46, 0, v6
	v_xor_b32_e32 v6, 0x60, v26
	v_add_u32_e32 v47, 0, v6
	v_xor_b32_e32 v6, 0x70, v26
	.loc	1 63 39                         ; matmul.py:63:39
	v_lshlrev_b32_e32 v7, 8, v65
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v48, 0, v6
	.loc	1 63 39                         ; matmul.py:63:39
	v_lshlrev_b32_e32 v6, 4, v29
	v_accvgpr_write_b32 a109, v7
	v_add_u32_e32 v7, 0, v7
	v_add_u32_e32 v49, v7, v6
.Ltmp60:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, 0x3000, v30
	v_add_u32_e32 v8, 0x3800, v30
	v_add_u32_e32 v14, 0x4000, v30
	v_add_u32_e32 v34, 0x4800, v30
.Ltmp61:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v33
	v_mov_b32_e32 v9, v33
	v_mov_b32_e32 v15, v33
	v_mov_b32_e32 v35, v33
	v_accvgpr_write_b32 a106, v40
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v25, v[10:13] offset:384
	s_waitcnt vmcnt(14)
	ds_write_b128 v25, v[50:53] offset:1408
	s_waitcnt vmcnt(13)
	ds_write_b128 v45, v[54:57] offset:512
	s_waitcnt vmcnt(11)
	ds_write_b128 v45, v[66:69] offset:1536
	ds_write_b128 v46, v[58:61] offset:640
	s_waitcnt vmcnt(10)
	ds_write_b128 v46, v[70:73] offset:1664
	s_waitcnt vmcnt(9)
	ds_write_b128 v47, v[74:77] offset:768
	s_waitcnt vmcnt(7)
	ds_write_b128 v47, v[82:85] offset:1792
	ds_write_b128 v48, v[78:81] offset:896
	s_waitcnt vmcnt(6)
	ds_write_b128 v48, v[86:89] offset:1920
.Ltmp62:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v36, 0x5000, v30
	v_add_u32_e32 v38, 0x5800, v30
	v_add_u32_e32 v40, 0x6000, v30
.Ltmp63:
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(5)
	ds_write_b128 v49, v[90:93] offset:8192
.Ltmp64:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x6800, v30
	v_add_u32_e32 v52, 0x7000, v30
	v_add_u32_e32 v30, 0x7800, v30
.Ltmp65:
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[16:17], v[6:7], 2, s[4:5]
	v_lshl_add_u64 v[54:55], v[8:9], 2, s[4:5]
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[4:5]
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[4:5]
	v_mov_b32_e32 v37, v33
	v_mov_b32_e32 v39, v33
	v_mov_b32_e32 v41, v33
	v_mov_b32_e32 v51, v33
	v_mov_b32_e32 v53, v33
	v_mov_b32_e32 v31, v33
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[16:17], off
	global_load_dwordx4 v[10:13], v[54:55], off
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[36:37], v[36:37], 2, s[4:5]
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[14:17], v[14:15], off
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[38:39], v[38:39], 2, s[4:5]
	v_lshl_add_u64 v[40:41], v[40:41], 2, s[4:5]
	v_lshl_add_u64 v[62:63], v[50:51], 2, s[4:5]
	v_lshl_add_u64 v[82:83], v[52:53], 2, s[4:5]
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[4:5]
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[34:35], off
	global_load_dwordx4 v[54:57], v[36:37], off
	global_load_dwordx4 v[58:61], v[38:39], off
	global_load_dwordx4 v[66:69], v[40:41], off
	global_load_dwordx4 v[70:73], v[62:63], off
	global_load_dwordx4 v[74:77], v[82:83], off
	global_load_dwordx4 v[78:81], v[30:31], off
	.loc	1 73 37 is_stmt 1               ; matmul.py:73:37
	v_and_b32_e32 v22, 6, v0
	v_and_b32_e32 v23, 1, v0
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(14)
	ds_write_b128 v49, v[18:21] offset:8448
	s_waitcnt vmcnt(13)
	ds_write_b128 v49, v[94:97] offset:8704
	s_waitcnt vmcnt(12)
	ds_write_b128 v49, v[98:101] offset:8960
	s_waitcnt vmcnt(11)
	ds_write_b128 v49, v[102:105] offset:9216
	.loc	1 73 37                         ; matmul.py:73:37
	v_lshlrev_b32_e32 v18, 4, v22
	v_accvgpr_write_b32 a92, v42
	v_lshlrev_b32_e32 v19, 3, v42
	v_lshlrev_b32_e32 v42, 11, v23
	v_or3_b32 v26, v18, v19, v42
	s_add_i32 s0, 0, 0x4000
	v_accvgpr_write_b32 a63, v33
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(10)
	ds_write_b128 v49, v[2:5] offset:9472
	s_waitcnt vmcnt(9)
	ds_write_b128 v49, v[6:9] offset:9728
	s_waitcnt vmcnt(8)
	ds_write_b128 v49, v[10:13] offset:9984
	s_waitcnt vmcnt(7)
	ds_write_b128 v49, v[14:17] offset:10240
	s_waitcnt vmcnt(6)
	ds_write_b128 v49, v[50:53] offset:10496
	s_waitcnt vmcnt(5)
	ds_write_b128 v49, v[54:57] offset:10752
	s_waitcnt vmcnt(4)
	ds_write_b128 v49, v[58:61] offset:11008
	s_waitcnt vmcnt(3)
	ds_write_b128 v49, v[66:69] offset:11264
	s_waitcnt vmcnt(2)
	ds_write_b128 v49, v[70:73] offset:11520
	s_waitcnt vmcnt(1)
	ds_write_b128 v49, v[74:77] offset:11776
	s_waitcnt vmcnt(0)
	ds_write_b128 v49, v[78:81] offset:12032
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt lgkmcnt(0)
	; wave barrier
	ds_read_b128 v[2:5], v44
	ds_read_b128 v[6:9], v44 offset:1024
	ds_read_b128 v[10:13], v27 offset:128
	ds_read_b128 v[14:17], v27 offset:1152
	.loc	1 73 37                         ; matmul.py:73:37
	v_add_u32_e32 v55, 0, v26
	s_waitcnt lgkmcnt(3)
	ds_write_b128 v55, v[2:5] offset:16384
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[2:5], v24 offset:256
	ds_read_b128 v[18:21], v24 offset:1280
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(4)
	ds_write_b128 v55, v[10:13] offset:20480
	ds_write_b128 v55, v[6:9] offset:16400
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[6:9], v25 offset:384
	ds_read_b128 v[10:13], v25 offset:1408
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(7)
	ds_write_b128 v55, v[14:17] offset:20496
	v_xor_b32_e32 v14, 32, v26
	v_add_u32_e32 v29, s0, v14
	s_waitcnt lgkmcnt(6)
	ds_write_b128 v29, v[2:5] offset:512
	v_accvgpr_write_b32 a115, v14
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[2:5], v45 offset:512
	ds_read_b128 v[14:17], v45 offset:1536
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(5)
	ds_write_b128 v29, v[6:9] offset:4608
	ds_write_b128 v29, v[18:21] offset:528
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[6:9], v46 offset:640
	ds_read_b128 v[18:21], v46 offset:1664
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(8)
	ds_write_b128 v29, v[10:13] offset:4624
	v_xor_b32_e32 v10, 64, v26
	v_add_u32_e32 v29, s0, v10
	s_waitcnt lgkmcnt(6)
	ds_write_b128 v29, v[2:5] offset:1024
	v_accvgpr_write_b32 a116, v10
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[2:5], v47 offset:768
	ds_read_b128 v[10:13], v47 offset:1792
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(5)
	ds_write_b128 v29, v[6:9] offset:5120
	ds_write_b128 v29, v[14:17] offset:1040
	v_accvgpr_write_b32 a62, v33
	v_accvgpr_write_b32 a61, v33
	v_accvgpr_write_b32 a60, v33
	v_accvgpr_write_b32 a59, v33
	v_accvgpr_write_b32 a58, v33
	v_accvgpr_write_b32 a57, v33
	v_accvgpr_write_b32 a56, v33
	v_accvgpr_write_b32 a55, v33
	v_accvgpr_write_b32 a54, v33
	v_accvgpr_write_b32 a53, v33
	v_accvgpr_write_b32 a52, v33
	v_accvgpr_write_b32 a51, v33
	v_accvgpr_write_b32 a50, v33
	v_accvgpr_write_b32 a49, v33
	v_accvgpr_write_b32 a48, v33
	v_accvgpr_write_b32 a47, v33
	v_accvgpr_write_b32 a46, v33
	v_accvgpr_write_b32 a45, v33
	v_accvgpr_write_b32 a44, v33
	v_accvgpr_write_b32 a43, v33
	v_accvgpr_write_b32 a42, v33
	v_accvgpr_write_b32 a41, v33
	v_accvgpr_write_b32 a40, v33
	v_accvgpr_write_b32 a39, v33
	v_accvgpr_write_b32 a38, v33
	v_accvgpr_write_b32 a37, v33
	v_accvgpr_write_b32 a36, v33
	v_accvgpr_write_b32 a35, v33
	v_accvgpr_write_b32 a34, v33
	v_accvgpr_write_b32 a33, v33
	v_accvgpr_write_b32 a32, v33
	v_accvgpr_write_b32 a31, v33
	v_accvgpr_write_b32 a30, v33
	v_accvgpr_write_b32 a29, v33
	v_accvgpr_write_b32 a28, v33
	v_accvgpr_write_b32 a27, v33
	v_accvgpr_write_b32 a26, v33
	v_accvgpr_write_b32 a25, v33
	v_accvgpr_write_b32 a24, v33
	v_accvgpr_write_b32 a23, v33
	v_accvgpr_write_b32 a22, v33
	v_accvgpr_write_b32 a21, v33
	v_accvgpr_write_b32 a20, v33
	v_accvgpr_write_b32 a19, v33
	v_accvgpr_write_b32 a18, v33
	v_accvgpr_write_b32 a17, v33
	v_accvgpr_write_b32 a16, v33
	v_accvgpr_write_b32 a15, v33
	v_accvgpr_write_b32 a14, v33
	v_accvgpr_write_b32 a13, v33
	v_accvgpr_write_b32 a12, v33
	v_accvgpr_write_b32 a11, v33
	v_accvgpr_write_b32 a10, v33
	v_accvgpr_write_b32 a9, v33
	v_accvgpr_write_b32 a8, v33
	v_accvgpr_write_b32 a7, v33
	v_accvgpr_write_b32 a6, v33
	v_accvgpr_write_b32 a5, v33
	v_accvgpr_write_b32 a4, v33
	v_accvgpr_write_b32 a3, v33
	v_accvgpr_write_b32 a2, v33
	v_accvgpr_write_b32 a1, v33
	v_accvgpr_write_b32 a0, v33
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[6:9], v48 offset:896
	ds_read_b128 v[14:17], v48 offset:1920
	.loc	1 63 39                         ; matmul.py:63:39
	ds_read_b128 v[30:33], v49 offset:8192
	ds_read_b128 v[34:37], v49 offset:8448
	ds_read_b128 v[38:41], v49 offset:8704
	ds_read_b128 v[50:53], v49 offset:8960
	ds_read_b128 v[56:59], v49 offset:9216
	ds_read_b128 v[66:69], v49 offset:9472
	ds_read_b128 v[70:73], v49 offset:9728
	ds_read_b128 v[74:77], v49 offset:9984
	ds_read_b128 v[84:87], v49 offset:10240
	ds_read_b128 v[88:91], v49 offset:10496
	ds_read_b128 v[92:95], v49 offset:10752
	ds_read_b128 v[220:223], v49 offset:11008
	ds_read_b128 v[96:99], v49 offset:11264
	ds_read_b128 v[100:103], v49 offset:11520
	ds_read_b128 v[104:107], v49 offset:11776
	ds_read_b128 v[212:215], v49 offset:12032
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(14)
	ds_write_b128 v29, v[18:21] offset:5136
	v_xor_b32_e32 v18, 0x60, v26
	v_accvgpr_write_b32 a117, v18
	v_add_u32_e32 v18, s0, v18
	ds_write_b128 v18, v[2:5] offset:1536
	ds_write_b128 v18, v[6:9] offset:5632
	ds_write_b128 v18, v[10:13] offset:1552
	ds_write_b128 v18, v[14:17] offset:5648
	v_lshlrev_b32_e32 v5, 6, v0
	v_and_b32_e32 v3, 8, v0
	v_and_b32_e32 v5, 0x800, v5
	v_lshlrev_b32_e32 v2, 12, v23
	v_lshlrev_b32_e32 v4, 1, v3
	v_lshl_or_b32 v5, v65, 3, v5
	v_or3_b32 v2, v2, v4, v5
	s_movk_i32 s1, 0x110
	v_mad_u32_u24 v2, v22, s1, v2
	v_add_u32_e32 v4, 0, v2
	; wave barrier
	ds_read_b128 v[176:179], v4 offset:16384
	ds_read_b128 v[180:183], v4 offset:16640
	v_accvgpr_write_b32 a118, v4
	v_xor_b32_e32 v4, 32, v2
	v_add_u32_e32 v82, 0, v4
	v_xor_b32_e32 v4, 64, v2
	v_xor_b32_e32 v2, 0x60, v2
	v_add_u32_e32 v54, 0, v4
	v_add_u32_e32 v62, 0, v2
	.loc	1 74 37                         ; matmul.py:74:37
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v4, 4, v0
	v_accvgpr_write_b32 a111, v2
	v_and_b32_e32 v2, 0x130, v2
	v_accvgpr_write_b32 a112, v4
	v_lshlrev_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v3, 3, v3
	v_or3_b32 v6, v4, v2, v3
	v_accvgpr_write_b32 a113, v3
	v_add_u32_e32 v63, 0, v6
	v_mov_b32_e32 v2, v30
	v_mov_b32_e32 v3, v34
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v4, v38
	v_mov_b32_e32 v5, v50
	.loc	1 73 37                         ; matmul.py:73:37
	ds_read_b128 v[192:195], v82 offset:16384
	ds_read_b128 v[196:199], v82 offset:16640
	ds_read_b128 v[200:203], v54 offset:16384
	ds_read_b128 v[204:207], v54 offset:16640
	ds_read_b128 v[208:211], v62 offset:16384
	ds_read_b128 v[216:219], v62 offset:16640
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt lgkmcnt(0)
	; wave barrier
	ds_write_b128 v63, v[2:5] offset:16384
	v_mov_b32_e32 v2, v31
	v_mov_b32_e32 v3, v35
	v_mov_b32_e32 v4, v39
	v_mov_b32_e32 v5, v51
	ds_write_b128 v63, v[2:5] offset:18432
	v_mov_b32_e32 v2, v56
	v_mov_b32_e32 v3, v66
	v_mov_b32_e32 v4, v70
	v_mov_b32_e32 v5, v74
	ds_write_b128 v63, v[2:5] offset:16512
	v_mov_b32_e32 v2, v57
	v_mov_b32_e32 v3, v67
	v_mov_b32_e32 v4, v71
	v_mov_b32_e32 v5, v75
	v_xor_b32_e32 v64, 64, v6
	ds_write_b128 v63, v[2:5] offset:18560
	v_add_u32_e32 v6, s0, v64
	v_mov_b32_e32 v2, v32
	v_mov_b32_e32 v3, v36
	v_mov_b32_e32 v4, v40
	v_mov_b32_e32 v5, v52
	ds_write_b128 v6, v[2:5] offset:512
	v_mov_b32_e32 v2, v58
	v_mov_b32_e32 v3, v68
	v_mov_b32_e32 v4, v72
	v_mov_b32_e32 v5, v76
	ds_write_b128 v6, v[2:5] offset:640
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v3, 0xb0, v2
	v_and_b32_e32 v2, 2, v0
	v_bfe_i32 v0, v0, 1, 1
	v_cmp_eq_u32_e64 s[0:1], 0, v2
	v_lshlrev_b32_e32 v2, 6, v65
	s_movk_i32 s9, 0x240
	v_and_or_b32 v0, v0, s9, v2
	v_or3_b32 v0, v0, v42, v3
	v_mov_b32_e32 v76, v73
	v_add_u32_e32 v73, 0, v0
	v_xor_b32_e32 v0, 64, v0
	v_mov_b32_e32 v50, v33
	v_mov_b32_e32 v51, v37
	v_mov_b32_e32 v52, v41
	v_mov_b32_e32 v74, v59
	v_mov_b32_e32 v75, v69
	v_add_u32_e32 v78, 0, v0
	ds_write_b128 v6, v[50:53] offset:2560
	ds_write_b128 v6, v[74:77] offset:2688
	; wave barrier
	ds_read_b128 v[224:227], v73 offset:16384
	ds_read_b128 v[228:231], v73 offset:16640
	ds_read_b128 v[232:235], v78 offset:16384
	v_lshl_add_u32 v0, s8, 6, v1
	v_or_b32_e32 v0, v0, v43
	s_lshl_b32 s8, s10, 11
	v_subrev_u32_e32 v0, s8, v0
	s_mov_b32 s12, 0
	v_accvgpr_write_b32 a110, v42
	v_accvgpr_write_b32 a114, v3
	v_accvgpr_write_b32 a108, v43
	v_accvgpr_write_b32 a119, v0
	v_or_b32_e32 v20, 0x3c020, v28
	v_mov_b32_e32 v184, v84
	v_mov_b32_e32 v185, v88
	v_mov_b32_e32 v186, v92
	v_mov_b32_e32 v187, v220
	v_mov_b32_e32 v248, v85
	v_mov_b32_e32 v249, v89
	v_mov_b32_e32 v250, v93
	v_mov_b32_e32 v251, v221
	v_mov_b32_e32 v188, v96
	v_mov_b32_e32 v189, v100
	v_mov_b32_e32 v190, v104
	v_mov_b32_e32 v191, v212
	v_mov_b32_e32 v244, v97
	v_mov_b32_e32 v245, v101
	v_mov_b32_e32 v246, v105
	v_mov_b32_e32 v247, v213
	v_mov_b32_e32 v240, v86
	v_mov_b32_e32 v241, v90
	v_mov_b32_e32 v242, v94
	v_mov_b32_e32 v243, v222
	v_mov_b32_e32 v220, v87
	v_mov_b32_e32 v221, v91
	v_mov_b32_e32 v222, v95
	v_mov_b32_e32 v236, v98
	v_mov_b32_e32 v237, v102
	v_mov_b32_e32 v238, v106
	v_mov_b32_e32 v239, v214
	v_mov_b32_e32 v212, v99
	v_mov_b32_e32 v213, v103
	v_mov_b32_e32 v214, v107
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 75 42                         ; matmul.py:75:42
	v_accvgpr_write_b32 a0, v174
	v_accvgpr_write_b32 a1, v173
	v_accvgpr_write_b32 a2, v172
	v_accvgpr_write_b32 a3, v171
	v_accvgpr_write_b32 a4, v170
	v_accvgpr_write_b32 a5, v169
	v_accvgpr_write_b32 a6, v168
	v_accvgpr_write_b32 a7, v167
	v_accvgpr_write_b32 a8, v166
	v_accvgpr_write_b32 a9, v165
	v_accvgpr_write_b32 a10, v164
	v_accvgpr_write_b32 a11, v163
	v_accvgpr_write_b32 a12, v162
	v_accvgpr_write_b32 a13, v161
	v_accvgpr_write_b32 a14, v160
	v_accvgpr_write_b32 a15, v159
	v_accvgpr_write_b32 a16, v158
	v_accvgpr_write_b32 a17, v157
	v_accvgpr_write_b32 a18, v156
	v_accvgpr_write_b32 a19, v155
	v_accvgpr_write_b32 a20, v154
	v_accvgpr_write_b32 a21, v153
	v_accvgpr_write_b32 a22, v152
	v_accvgpr_write_b32 a23, v151
	v_accvgpr_write_b32 a24, v150
	v_accvgpr_write_b32 a25, v149
	v_accvgpr_write_b32 a26, v148
	v_accvgpr_write_b32 a27, v147
	v_accvgpr_write_b32 a28, v146
	v_accvgpr_write_b32 a29, v145
	v_accvgpr_write_b32 a30, v144
	v_accvgpr_write_b32 a31, v143
	v_accvgpr_write_b32 a32, v142
	v_accvgpr_write_b32 a33, v141
	v_accvgpr_write_b32 a34, v140
	v_accvgpr_write_b32 a35, v139
	v_accvgpr_write_b32 a36, v138
	v_accvgpr_write_b32 a37, v137
	v_accvgpr_write_b32 a38, v136
	v_accvgpr_write_b32 a39, v135
	v_accvgpr_write_b32 a40, v134
	v_accvgpr_write_b32 a41, v133
	v_accvgpr_write_b32 a42, v132
	v_accvgpr_write_b32 a43, v131
	v_accvgpr_write_b32 a44, v130
	v_accvgpr_write_b32 a45, v129
	v_accvgpr_write_b32 a46, v128
	v_accvgpr_write_b32 a47, v127
	v_accvgpr_write_b32 a48, v126
	v_accvgpr_write_b32 a49, v125
	v_accvgpr_write_b32 a50, v124
	v_accvgpr_write_b32 a51, v123
	v_accvgpr_write_b32 a52, v122
	v_accvgpr_write_b32 a53, v121
	v_accvgpr_write_b32 a54, v120
	v_accvgpr_write_b32 a55, v119
	v_accvgpr_write_b32 a56, v118
	v_accvgpr_write_b32 a57, v117
	v_accvgpr_write_b32 a58, v115
	v_accvgpr_write_b32 a59, v114
	v_accvgpr_write_b32 a60, v112
	v_accvgpr_write_b32 a61, v110
	v_accvgpr_write_b32 a62, v109
	v_accvgpr_write_b32 a63, v107
	v_mfma_f32_32x32x2_f32 a[0:15], v103, v116, a[0:15]
.Ltmp66:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v30, 0xfffc4000, v20
	v_add_u32_e32 v180, 0xfffdc000, v20
	v_add_u32_e32 v184, 0xfffe4000, v20
.Ltmp67:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v31, 31, v30
.Ltmp68:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v32, 0xfffc8000, v20
	v_add_u32_e32 v182, 0xfffe0000, v20
.Ltmp69:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[2:3]
	v_ashrrev_i32_e32 v181, 31, v180
	v_ashrrev_i32_e32 v185, 31, v184
.Ltmp70:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v186, 0xfffe8000, v20
.Ltmp71:
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[176:179], v[30:31], off
	.loc	1 62 32 is_stmt 0               ; matmul.py:62:32
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshl_add_u64 v[216:217], v[180:181], 2, s[2:3]
	v_ashrrev_i32_e32 v183, 31, v182
	v_lshl_add_u64 v[180:181], v[184:185], 2, s[2:3]
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v103, v92, a[16:31]
.Ltmp72:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v34, 0xfffcc000, v20
	v_add_u32_e32 v188, 0xfffec000, v20
	v_add_u32_e32 v190, 0xffff0000, v20
.Ltmp73:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[32:33], v[32:33], 2, s[2:3]
	v_lshl_add_u64 v[220:221], v[182:183], 2, s[2:3]
	v_ashrrev_i32_e32 v187, 31, v186
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[180:183], v[180:181], off
.Ltmp74:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v192, 0xffff4000, v20
.Ltmp75:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[194:195], v[186:187], 2, s[2:3]
	v_ashrrev_i32_e32 v189, 31, v188
	v_ashrrev_i32_e32 v191, 31, v190
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[184:187], v[32:33], off
.Ltmp76:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v36, 0xfffd0000, v20
	v_add_u32_e32 v196, 0xffff8000, v20
.Ltmp77:
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v81, v116, a[32:47]
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[2:3]
	v_lshl_add_u64 v[198:199], v[188:189], 2, s[2:3]
	v_lshl_add_u64 v[32:33], v[190:191], 2, s[2:3]
	v_ashrrev_i32_e32 v193, 31, v192
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[188:191], v[194:195], off
.Ltmp78:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v200, 0xffffc000, v20
.Ltmp79:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[224:225], v[192:193], 2, s[2:3]
	v_ashrrev_i32_e32 v197, 31, v196
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[192:195], v[34:35], off
	global_load_dwordx4 v[204:207], v[32:33], off
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[36:37], v[36:37], 2, s[2:3]
	v_lshl_add_u64 v[34:35], v[196:197], 2, s[2:3]
	v_ashrrev_i32_e32 v201, 31, v200
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[196:199], v[198:199], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v81, v92, a[48:63]
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[232:233], v[200:201], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[200:203], v[36:37], off
.Ltmp80:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v38, 0xfffd4000, v20
.Ltmp81:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[38:39], v[38:39], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[208:211], v[38:39], off
.Ltmp82:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v40, 0xfffd8000, v20
	v_accvgpr_read_b32 v21, a119
.Ltmp83:
	.loc	3 55 29 is_stmt 0               ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v21, s12, v21
.Ltmp84:
	.loc	1 62 32 is_stmt 1               ; matmul.py:62:32
	v_ashrrev_i32_e32 v41, 31, v40
.Ltmp85:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v236, 0x10000, v21
.Ltmp86:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[40:41], v[40:41], 2, s[2:3]
.Ltmp87:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v30, 0x10800, v21
	v_add_u32_e32 v238, 0x11000, v21
	v_add_u32_e32 v240, 0x11800, v21
.Ltmp88:
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v101, v113, a[0:15]
.Ltmp89:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v242, 0x12000, v21
	v_add_u32_e32 v244, 0x12800, v21
	v_add_u32_e32 v246, 0x13000, v21
	v_add_u32_e32 v248, 0x13800, v21
	v_add_u32_e32 v250, 0x14000, v21
	v_add_u32_e32 v252, 0x14800, v21
	v_add_u32_e32 v254, 0x15000, v21
	v_add_u32_e32 v52, 0x15800, v21
	v_add_u32_e32 v50, 0x16000, v21
	v_add_u32_e32 v28, 0x16800, v21
	v_add_u32_e32 v42, 0x17000, v21
.Ltmp90:
	.loc	1 62 32                         ; matmul.py:62:32
	v_add_u32_e32 v56, 0x17800, v21
	v_ashrrev_i32_e32 v21, 31, v20
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[212:215], v[40:41], off
	s_nop 0
	global_load_dwordx4 v[216:219], v[216:217], off
	s_nop 0
	global_load_dwordx4 v[220:223], v[220:221], off
	s_nop 0
	global_load_dwordx4 v[224:227], v[224:225], off
	s_nop 0
	global_load_dwordx4 v[228:231], v[34:35], off
	.loc	1 63 32 is_stmt 1               ; matmul.py:63:32
	v_ashrrev_i32_e32 v237, 31, v236
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v101, v90, a[16:31]
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[36:37], v[20:21], 2, s[2:3]
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[32:33], v[236:237], 2, s[4:5]
	v_ashrrev_i32_e32 v31, 31, v30
	v_ashrrev_i32_e32 v239, 31, v238
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[4:5]
	v_lshl_add_u64 v[34:35], v[238:239], 2, s[4:5]
	v_ashrrev_i32_e32 v241, 31, v240
	v_ashrrev_i32_e32 v243, 31, v242
	v_lshl_add_u64 v[38:39], v[242:243], 2, s[4:5]
	v_ashrrev_i32_e32 v245, 31, v244
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(13)
	ds_write_b128 v44, v[176:179]
	global_load_dwordx4 v[176:179], v[232:233], off
	s_nop 0
	global_load_dwordx4 v[232:235], v[36:37], off
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[36:37], v[240:241], 2, s[4:5]
	v_ashrrev_i32_e32 v247, 31, v246
	v_ashrrev_i32_e32 v249, 31, v248
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v80, v113, a[32:47]
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[40:41], v[244:245], 2, s[4:5]
	v_lshl_add_u64 v[58:59], v[246:247], 2, s[4:5]
	v_lshl_add_u64 v[248:249], v[248:249], 2, s[4:5]
	v_ashrrev_i32_e32 v251, 31, v250
	v_ashrrev_i32_e32 v253, 31, v252
	v_lshl_add_u64 v[60:61], v[250:251], 2, s[4:5]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(14)
	ds_write_b128 v44, v[180:183] offset:1024
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[180:183], v[32:33], off
	global_load_dwordx4 v[236:239], v[30:31], off
	.loc	1 63 32 is_stmt 0               ; matmul.py:63:32
	v_lshl_add_u64 v[252:253], v[252:253], 2, s[4:5]
	v_ashrrev_i32_e32 v255, 31, v254
	v_lshl_add_u64 v[22:23], v[254:255], 2, s[4:5]
	v_ashrrev_i32_e32 v53, 31, v52
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	s_waitcnt vmcnt(15)
	ds_write_b128 v27, v[184:187] offset:128
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[184:187], v[34:35], off
	global_load_dwordx4 v[240:243], v[36:37], off
	.loc	1 63 32 is_stmt 0               ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v80, v90, a[48:63]
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[52:53], v[52:53], 2, s[4:5]
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshl_add_u64 v[28:29], v[28:29], 2, s[4:5]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(16)
	ds_write_b128 v27, v[188:191] offset:1152
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[188:191], v[38:39], off
	global_load_dwordx4 v[244:247], v[40:41], off
	.loc	1 63 32 is_stmt 0               ; matmul.py:63:32
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[4:5]
	v_ashrrev_i32_e32 v57, 31, v56
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	s_waitcnt vmcnt(17)
	ds_write_b128 v24, v[192:195] offset:256
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[192:195], v[58:59], off
	s_nop 0
	global_load_dwordx4 v[248:251], v[248:249], off
	.loc	1 63 32 is_stmt 0               ; matmul.py:63:32
	v_lshl_add_u64 v[56:57], v[56:57], 2, s[4:5]
	v_accvgpr_read_b32 v21, a115
	.loc	1 73 37 is_stmt 1               ; matmul.py:73:37
	v_add_u32_e32 v21, 0, v21
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(17)
	ds_write_b128 v24, v[196:199] offset:1280
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v100, v111, a[0:15]
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[196:199], v[60:61], off
	s_nop 0
	global_load_dwordx4 v[252:255], v[252:253], off
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(18)
	ds_write_b128 v25, v[200:203] offset:384
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[200:203], v[22:23], off
	global_load_dwordx4 v[30:33], v[52:53], off
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v25, v[204:207] offset:1408
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[204:207], v[50:51], off
	global_load_dwordx4 v[34:37], v[28:29], off
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(21)
	ds_write_b128 v45, v[208:211] offset:512
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[208:211], v[42:43], off
	global_load_dwordx4 v[38:41], v[56:57], off
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(19)
	ds_write_b128 v45, v[224:227] offset:1536
	ds_write_b128 v46, v[212:215] offset:640
	s_waitcnt vmcnt(18)
	ds_write_b128 v46, v[228:231] offset:1664
	ds_write_b128 v47, v[216:219] offset:768
	s_waitcnt vmcnt(17)
	ds_write_b128 v47, v[176:179] offset:1792
	ds_write_b128 v48, v[220:223] offset:896
	s_waitcnt vmcnt(16)
	ds_write_b128 v48, v[232:235] offset:1920
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(15)
	ds_write_b128 v49, v[180:183] offset:8192
	s_waitcnt vmcnt(14)
	ds_write_b128 v49, v[236:239] offset:8448
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v100, v88, a[16:31]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(13)
	ds_write_b128 v49, v[184:187] offset:8704
	s_waitcnt vmcnt(12)
	ds_write_b128 v49, v[240:243] offset:8960
	s_waitcnt vmcnt(11)
	ds_write_b128 v49, v[188:191] offset:9216
	s_waitcnt vmcnt(10)
	ds_write_b128 v49, v[244:247] offset:9472
	s_waitcnt vmcnt(9)
	ds_write_b128 v49, v[192:195] offset:9728
	s_waitcnt vmcnt(8)
	ds_write_b128 v49, v[248:251] offset:9984
	s_waitcnt vmcnt(7)
	ds_write_b128 v49, v[196:199] offset:10240
	s_waitcnt vmcnt(6)
	ds_write_b128 v49, v[252:255] offset:10496
	s_waitcnt vmcnt(5)
	ds_write_b128 v49, v[200:203] offset:10752
	s_waitcnt vmcnt(4)
	ds_write_b128 v49, v[30:33] offset:11008
	s_waitcnt vmcnt(3)
	ds_write_b128 v49, v[204:207] offset:11264
	s_waitcnt vmcnt(2)
	ds_write_b128 v49, v[34:37] offset:11520
	s_waitcnt vmcnt(1)
	ds_write_b128 v49, v[208:211] offset:11776
	s_waitcnt vmcnt(0)
	ds_write_b128 v49, v[38:41] offset:12032
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v79, v111, a[32:47]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt lgkmcnt(0)
	; wave barrier
	ds_read_b128 v[30:33], v44
	ds_read_b128 v[34:37], v44 offset:1024
	ds_read_b128 v[38:41], v27 offset:128
	ds_read_b128 v[176:179], v27 offset:1152
	ds_read_b128 v[180:183], v24 offset:256
	ds_read_b128 v[192:195], v24 offset:1280
	ds_read_b128 v[196:199], v25 offset:384
	ds_read_b128 v[200:203], v25 offset:1408
	ds_read_b128 v[204:207], v45 offset:512
	ds_read_b128 v[208:211], v45 offset:1536
	ds_read_b128 v[212:215], v46 offset:640
	ds_read_b128 v[216:219], v46 offset:1664
	ds_read_b128 v[220:223], v47 offset:768
	ds_read_b128 v[224:227], v47 offset:1792
	ds_read_b128 v[228:231], v48 offset:896
	ds_read_b128 v[232:235], v48 offset:1920
	.loc	1 63 39                         ; matmul.py:63:39
	ds_read_b128 v[50:53], v49 offset:8192
	ds_read_b128 v[56:59], v49 offset:8448
	ds_read_b128 v[236:239], v49 offset:8704
	ds_read_b128 v[240:243], v49 offset:8960
	ds_read_b128 v[244:247], v49 offset:9216
	ds_read_b128 v[248:251], v49 offset:9472
	ds_read_b128 v[252:255], v49 offset:9728
	ds_read_b128 a[64:67], v49 offset:9984
	ds_read_b128 v[184:187], v49 offset:10240
	ds_read_b128 a[68:71], v49 offset:10496
	ds_read_b128 a[72:75], v49 offset:10752
	ds_read_b128 a[76:79], v49 offset:11008
	ds_read_b128 v[188:191], v49 offset:11264
	ds_read_b128 a[80:83], v49 offset:11520
	ds_read_b128 a[84:87], v49 offset:11776
	ds_read_b128 a[88:91], v49 offset:12032
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(14)
	ds_write_b128 v55, v[30:33] offset:16384
	ds_write_b128 v55, v[38:41] offset:20480
	ds_write_b128 v55, v[34:37] offset:16400
	ds_write_b128 v55, v[176:179] offset:20496
	ds_write_b128 v21, v[180:183] offset:16896
	ds_write_b128 v21, v[196:199] offset:20992
	ds_write_b128 v21, v[192:195] offset:16912
	ds_write_b128 v21, v[200:203] offset:21008
	v_accvgpr_read_b32 v21, a116
	v_add_u32_e32 v21, 0, v21
	ds_write_b128 v21, v[204:207] offset:17408
	ds_write_b128 v21, v[212:215] offset:21504
	ds_write_b128 v21, v[208:211] offset:17424
	ds_write_b128 v21, v[216:219] offset:21520
	v_accvgpr_read_b32 v21, a117
	v_add_u32_e32 v21, 0, v21
	ds_write_b128 v21, v[220:223] offset:17920
	ds_write_b128 v21, v[228:231] offset:22016
	ds_write_b128 v21, v[224:227] offset:17936
	ds_write_b128 v21, v[232:235] offset:22032
	v_accvgpr_read_b32 v21, a118
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v79, v88, a[48:63]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v30, v50
	v_mov_b32_e32 v31, v56
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v32, v236
	v_mov_b32_e32 v33, v240
	; wave barrier
	.loc	1 73 37                         ; matmul.py:73:37
	ds_read_b128 v[176:179], v21 offset:16384
	ds_read_b128 v[180:183], v21 offset:16640
	ds_read_b128 v[192:195], v82 offset:16384
	ds_read_b128 v[196:199], v82 offset:16640
	ds_read_b128 v[200:203], v54 offset:16384
	ds_read_b128 v[204:207], v54 offset:16640
	ds_read_b128 v[208:211], v62 offset:16384
	ds_read_b128 v[216:219], v62 offset:16640
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt lgkmcnt(0)
	; wave barrier
	ds_write_b128 v63, v[30:33] offset:16384
	v_mov_b32_e32 v30, v51
	v_mov_b32_e32 v31, v57
	v_mov_b32_e32 v32, v237
	v_mov_b32_e32 v33, v241
	ds_write_b128 v63, v[30:33] offset:18432
	v_mov_b32_e32 v30, v244
	v_mov_b32_e32 v31, v248
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v99, v108, a[0:15]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v32, v252
	v_accvgpr_read_b32 v33, a64
	ds_write_b128 v63, v[30:33] offset:16512
	v_mov_b32_e32 v30, v245
	v_mov_b32_e32 v31, v249
	v_mov_b32_e32 v32, v253
	v_accvgpr_read_b32 v33, a65
	ds_write_b128 v63, v[30:33] offset:18560
	v_mov_b32_e32 v30, v52
	v_mov_b32_e32 v31, v58
	v_mov_b32_e32 v32, v238
	v_mov_b32_e32 v33, v242
	ds_write_b128 v175, v[30:33] offset:16896
	v_mov_b32_e32 v240, v53
	v_mov_b32_e32 v241, v59
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v99, v86, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v242, v239
	v_mov_b32_e32 v30, v246
	v_mov_b32_e32 v31, v250
	v_mov_b32_e32 v32, v254
	v_accvgpr_read_b32 v33, a66
	v_accvgpr_write_b32 a64, v247
	v_accvgpr_write_b32 a65, v251
	v_accvgpr_write_b32 a66, v255
	ds_write_b128 v175, v[240:243] offset:18944
	ds_write_b128 v175, v[30:33] offset:17024
	ds_write_b128 v175, a[64:67] offset:19072
	; wave barrier
	ds_read_b128 v[224:227], v73 offset:16384
	ds_read_b128 v[228:231], v73 offset:16640
	ds_read_b128 v[232:235], v78 offset:16384
	s_add_i32 s12, s12, 0x10000
	v_add_u32_e32 v20, 32, v20
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v77, v108, a[32:47]
	v_accvgpr_read_b32 v215, a91
	v_accvgpr_read_b32 v214, a87
	v_accvgpr_read_b32 v213, a83
	v_mov_b32_e32 v212, v191
	v_accvgpr_read_b32 v239, a90
	v_accvgpr_read_b32 v238, a86
	v_accvgpr_read_b32 v237, a82
	v_mov_b32_e32 v236, v190
	v_accvgpr_read_b32 v223, a79
	v_accvgpr_read_b32 v222, a75
	v_accvgpr_read_b32 v221, a71
	v_mov_b32_e32 v220, v187
	v_accvgpr_read_b32 v243, a78
	v_accvgpr_read_b32 v242, a74
	v_accvgpr_read_b32 v241, a70
	v_mfma_f32_32x32x2_f32 a[48:63], v77, v86, a[48:63]
	v_mov_b32_e32 v240, v186
	v_accvgpr_read_b32 v247, a89
	v_accvgpr_read_b32 v246, a85
	v_accvgpr_read_b32 v245, a81
	v_mov_b32_e32 v244, v189
	v_accvgpr_read_b32 v191, a88
	v_accvgpr_read_b32 v190, a84
	v_accvgpr_read_b32 v189, a80
	v_accvgpr_read_b32 v251, a77
	v_accvgpr_read_b32 v250, a73
	v_accvgpr_read_b32 v249, a69
	v_mov_b32_e32 v248, v185
	v_accvgpr_read_b32 v187, a76
	v_accvgpr_read_b32 v186, a72
	v_accvgpr_read_b32 v185, a68
	v_mfma_f32_32x32x2_f32 a[0:15], v98, v16, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v98, v8, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v76, v16, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v76, v8, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v97, v17, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v97, v9, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v75, v17, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v75, v9, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v96, v18, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v96, v10, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v74, v18, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v74, v10, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v95, v19, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v95, v11, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v72, v19, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v72, v11, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v94, v106, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v94, v0, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v71, v106, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v71, v0, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v93, v105, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v93, v1, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v70, v105, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v70, v1, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v91, v104, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v91, v2, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v69, v104, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v69, v2, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v89, v102, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v89, v3, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v68, v102, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v68, v3, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v87, v12, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v87, v4, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v67, v12, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v67, v4, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v85, v13, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v85, v5, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v66, v13, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v66, v5, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v84, v14, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v84, v6, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v65, v14, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v65, v6, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v83, v15, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v83, v7, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v26, v15, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v26, v7, a[48:63]
	s_cbranch_execz .LBB0_4
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 74 37                         ; matmul.py:74:37
	v_add_u32_e32 v175, 0, v64
	s_waitcnt lgkmcnt(12)
	ds_read_b128 v[0:3], v78 offset:16640
	; wave barrier
	ds_write_b128 v63, v[184:187] offset:16384
	ds_write_b128 v63, v[248:251] offset:18432
	ds_write_b128 v63, v[188:191] offset:16512
	ds_write_b128 v63, v[244:247] offset:18560
	ds_write_b128 v175, v[240:243] offset:16896
	ds_write_b128 v175, v[220:223] offset:18944
	ds_write_b128 v175, v[236:239] offset:17024
	ds_write_b128 v175, v[212:215] offset:19072
	; wave barrier
	s_waitcnt lgkmcnt(12)
	ds_read_b128 v[16:19], v73 offset:16384
	s_waitcnt lgkmcnt(12)
	ds_read_b128 v[12:15], v73 offset:16640
	s_waitcnt lgkmcnt(12)
	ds_read_b128 v[8:11], v78 offset:16384
	s_waitcnt lgkmcnt(12)
	ds_read_b128 v[4:7], v78 offset:16640
	v_mov_b32_e32 v26, v219
	v_mov_b32_e32 v65, v218
	v_mov_b32_e32 v66, v217
	v_mov_b32_e32 v67, v216
	v_mov_b32_e32 v68, v207
	v_mov_b32_e32 v69, v206
	v_mov_b32_e32 v70, v205
	v_mov_b32_e32 v71, v204
	v_mov_b32_e32 v72, v199
	v_mov_b32_e32 v74, v198
	v_mov_b32_e32 v75, v197
	v_mov_b32_e32 v76, v196
	v_mov_b32_e32 v77, v183
	v_mov_b32_e32 v79, v182
	v_mov_b32_e32 v80, v181
	v_mov_b32_e32 v81, v180
	v_mov_b32_e32 v83, v211
	v_mov_b32_e32 v84, v210
	v_mov_b32_e32 v85, v209
	v_mov_b32_e32 v87, v208
	v_mov_b32_e32 v89, v203
	v_mov_b32_e32 v91, v202
	v_mov_b32_e32 v93, v201
	v_mov_b32_e32 v94, v200
	v_mov_b32_e32 v95, v195
	v_mov_b32_e32 v96, v194
	v_mov_b32_e32 v97, v193
	v_mov_b32_e32 v98, v192
	v_mov_b32_e32 v99, v179
	v_mov_b32_e32 v100, v178
	v_mov_b32_e32 v101, v177
	v_mov_b32_e32 v103, v176
	s_waitcnt lgkmcnt(13)
	v_mov_b32_e32 v86, v235
	v_mov_b32_e32 v88, v234
	v_mov_b32_e32 v90, v233
	v_mov_b32_e32 v92, v232
	v_mov_b32_e32 v102, v231
	v_mov_b32_e32 v104, v230
	v_mov_b32_e32 v105, v229
	v_mov_b32_e32 v106, v228
	v_mov_b32_e32 v108, v227
	v_mov_b32_e32 v111, v226
	v_mov_b32_e32 v113, v225
	v_mov_b32_e32 v116, v224
	v_accvgpr_read_b32 v107, a63
	v_accvgpr_read_b32 v109, a62
	v_accvgpr_read_b32 v110, a61
	v_accvgpr_read_b32 v112, a60
	v_accvgpr_read_b32 v114, a59
	v_accvgpr_read_b32 v115, a58
	v_accvgpr_read_b32 v117, a57
	v_accvgpr_read_b32 v118, a56
	v_accvgpr_read_b32 v119, a55
	v_accvgpr_read_b32 v120, a54
	v_accvgpr_read_b32 v121, a53
	v_accvgpr_read_b32 v122, a52
	v_accvgpr_read_b32 v123, a51
	v_accvgpr_read_b32 v124, a50
	v_accvgpr_read_b32 v125, a49
	v_accvgpr_read_b32 v126, a48
	v_accvgpr_read_b32 v127, a47
	v_accvgpr_read_b32 v128, a46
	v_accvgpr_read_b32 v129, a45
	v_accvgpr_read_b32 v130, a44
	v_accvgpr_read_b32 v131, a43
	v_accvgpr_read_b32 v132, a42
	v_accvgpr_read_b32 v133, a41
	v_accvgpr_read_b32 v134, a40
	v_accvgpr_read_b32 v135, a39
	v_accvgpr_read_b32 v136, a38
	v_accvgpr_read_b32 v137, a37
	v_accvgpr_read_b32 v138, a36
	v_accvgpr_read_b32 v139, a35
	v_accvgpr_read_b32 v140, a34
	v_accvgpr_read_b32 v141, a33
	v_accvgpr_read_b32 v142, a32
	v_accvgpr_read_b32 v143, a31
	v_accvgpr_read_b32 v144, a30
	v_accvgpr_read_b32 v145, a29
	v_accvgpr_read_b32 v146, a28
	v_accvgpr_read_b32 v147, a27
	v_accvgpr_read_b32 v148, a26
	v_accvgpr_read_b32 v149, a25
	v_accvgpr_read_b32 v150, a24
	v_accvgpr_read_b32 v151, a23
	v_accvgpr_read_b32 v152, a22
	v_accvgpr_read_b32 v153, a21
	v_accvgpr_read_b32 v154, a20
	v_accvgpr_read_b32 v155, a19
	v_accvgpr_read_b32 v156, a18
	v_accvgpr_read_b32 v157, a17
	v_accvgpr_read_b32 v158, a16
	v_accvgpr_read_b32 v159, a15
	v_accvgpr_read_b32 v160, a14
	v_accvgpr_read_b32 v161, a13
	v_accvgpr_read_b32 v162, a12
	v_accvgpr_read_b32 v163, a11
	v_accvgpr_read_b32 v164, a10
	v_accvgpr_read_b32 v165, a9
	v_accvgpr_read_b32 v166, a8
	v_accvgpr_read_b32 v167, a7
	v_accvgpr_read_b32 v168, a6
	v_accvgpr_read_b32 v169, a5
	v_accvgpr_read_b32 v170, a4
	v_accvgpr_read_b32 v171, a3
	v_accvgpr_read_b32 v172, a2
	v_accvgpr_read_b32 v173, a1
	.loc	1 50 25                         ; matmul.py:50:25
	s_cmp_eq_u32 s12, 0x1ff0000
	v_accvgpr_read_b32 v174, a0
	s_cbranch_scc0 .LBB0_1
; %bb.3:
                                        ; implicit-def: $vgpr219
                                        ; implicit-def: $vgpr207
                                        ; implicit-def: $vgpr199
                                        ; implicit-def: $vgpr183
                                        ; implicit-def: $vgpr211
                                        ; implicit-def: $vgpr203
                                        ; implicit-def: $vgpr195
                                        ; implicit-def: $vgpr179
                                        ; implicit-def: $agpr63
                                        ; implicit-def: $agpr47
                                        ; implicit-def: $agpr31
                                        ; implicit-def: $agpr15
                                        ; implicit-def: $vgpr235
                                        ; implicit-def: $vgpr231
                                        ; implicit-def: $vgpr227
                                        ; implicit-def: $vgpr215
                                        ; implicit-def: $vgpr239
                                        ; implicit-def: $vgpr223
                                        ; implicit-def: $vgpr243
                                        ; implicit-def: $vgpr247
                                        ; implicit-def: $vgpr191
                                        ; implicit-def: $vgpr251
                                        ; implicit-def: $vgpr187
                                        ; implicit-def: $sgpr12
                                        ; implicit-def: $vgpr20
.LBB0_4:
	.loc	1 75 42                         ; matmul.py:75:42
	s_nop 7
	s_nop 5
	v_accvgpr_write_b32 a0, v174
	v_accvgpr_write_b32 a1, v173
	v_accvgpr_write_b32 a2, v172
	v_accvgpr_write_b32 a3, v171
	v_accvgpr_write_b32 a4, v170
	v_accvgpr_write_b32 a5, v169
	v_accvgpr_write_b32 a6, v168
	v_accvgpr_write_b32 a7, v167
	v_accvgpr_write_b32 a8, v166
	v_accvgpr_write_b32 a9, v165
	v_accvgpr_write_b32 a10, v164
	v_accvgpr_write_b32 a11, v163
	v_accvgpr_write_b32 a12, v162
	v_accvgpr_write_b32 a13, v161
	v_accvgpr_write_b32 a14, v160
	v_accvgpr_write_b32 a15, v159
	v_accvgpr_write_b32 a16, v158
	v_accvgpr_write_b32 a17, v157
	v_accvgpr_write_b32 a18, v156
	v_accvgpr_write_b32 a19, v155
	v_accvgpr_write_b32 a20, v154
	v_accvgpr_write_b32 a21, v153
	v_accvgpr_write_b32 a22, v152
	v_accvgpr_write_b32 a23, v151
	v_accvgpr_write_b32 a24, v150
	v_accvgpr_write_b32 a25, v149
	v_accvgpr_write_b32 a26, v148
	v_accvgpr_write_b32 a27, v147
	v_accvgpr_write_b32 a28, v146
	v_accvgpr_write_b32 a29, v145
	v_accvgpr_write_b32 a30, v144
	v_accvgpr_write_b32 a31, v143
	v_accvgpr_write_b32 a48, v126
	v_accvgpr_write_b32 a49, v125
	v_accvgpr_write_b32 a50, v124
	v_accvgpr_write_b32 a51, v123
	v_accvgpr_write_b32 a52, v122
	v_accvgpr_write_b32 a53, v121
	v_accvgpr_write_b32 a54, v120
	v_accvgpr_write_b32 a55, v119
	v_accvgpr_write_b32 a56, v118
	v_accvgpr_write_b32 a57, v117
	v_accvgpr_write_b32 a58, v115
	v_accvgpr_write_b32 a59, v114
	v_accvgpr_write_b32 a60, v112
	v_accvgpr_write_b32 a61, v110
	v_accvgpr_write_b32 a62, v109
	v_accvgpr_write_b32 a63, v107
	v_mfma_f32_32x32x2_f32 a[0:15], v103, v116, a[0:15]
	v_accvgpr_write_b32 a32, v142
	v_accvgpr_write_b32 a33, v141
	v_accvgpr_write_b32 a34, v140
	v_accvgpr_write_b32 a35, v139
	v_accvgpr_write_b32 a36, v138
	v_accvgpr_write_b32 a37, v137
	v_accvgpr_write_b32 a38, v136
	v_accvgpr_write_b32 a39, v135
	v_accvgpr_write_b32 a40, v134
	v_accvgpr_write_b32 a41, v133
	v_accvgpr_write_b32 a42, v132
	v_accvgpr_write_b32 a43, v131
	v_accvgpr_write_b32 a44, v130
	v_accvgpr_write_b32 a45, v129
	v_accvgpr_write_b32 a46, v128
	v_mfma_f32_32x32x2_f32 a[16:31], v103, v92, a[16:31]
	v_accvgpr_write_b32 a47, v127
	v_mfma_f32_32x32x2_f32 a[48:63], v81, v92, a[48:63]
	s_nop 0
	v_mfma_f32_32x32x2_f32 a[32:47], v81, v116, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v101, v113, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v101, v90, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v80, v90, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v80, v113, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v100, v111, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v100, v88, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v79, v88, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v79, v111, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v99, v108, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v99, v86, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v77, v86, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v77, v108, a[32:47]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[0:15], v98, v16, a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[16:31], v98, v8, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v76, v8, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v76, v16, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v97, v17, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v97, v9, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v75, v9, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v75, v17, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v96, v18, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v96, v10, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v74, v10, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v74, v18, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v95, v19, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v95, v11, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v72, v11, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v72, v19, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v94, v106, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v94, v0, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v71, v0, a[48:63]
	.loc	1 79 33                         ; matmul.py:79:33
	v_mov_b32_e32 v0, 0x440
	v_cndmask_b32_e64 v0, v0, 0, s[0:1]
.Ltmp91:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:82:91 ]
	s_lshl_b32 s0, s10, 17
	.loc	2 132 15                        ; tuple_helpers.py:132:15 @[ matmul.py:82:91 ]
	s_add_i32 s0, s0, s11
.Ltmp92:
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v71, v106, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v93, v105, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v93, v1, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v70, v1, a[48:63]
	v_accvgpr_read_b32 v1, a110
	.loc	1 79 33                         ; matmul.py:79:33
	v_or_b32_e32 v0, v1, v0
	v_accvgpr_read_b32 v1, a109
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v70, v105, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v91, v104, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v91, v2, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v69, v2, a[48:63]
	v_accvgpr_read_b32 v2, a114
	.loc	1 79 33                         ; matmul.py:79:33
	v_or3_b32 v0, v0, v2, v1
	v_accvgpr_read_b32 v2, a111
	v_and_b32_e32 v2, 0x330, v2
	v_add_u32_e32 v1, 0, v0
	v_xad_u32 v0, v0, 64, 0
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v69, v104, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v89, v102, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v89, v3, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v68, v3, a[48:63]
	v_accvgpr_read_b32 v3, a112
	.loc	1 79 33                         ; matmul.py:79:33
	v_lshlrev_b32_e32 v3, 10, v3
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v68, v102, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v87, v12, a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x2_f32 a[16:31], v87, v4, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v67, v4, a[48:63]
	v_accvgpr_read_b32 v4, a113
	.loc	1 79 33                         ; matmul.py:79:33
	v_or3_b32 v2, v3, v2, v4
	v_add_u32_e32 v3, 0, v2
	v_xad_u32 v2, v2, 64, 0
	v_accvgpr_read_b32 v4, a108
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v67, v12, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v85, v13, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v85, v5, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v66, v5, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v66, v13, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v84, v14, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v84, v6, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v65, v6, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v65, v14, a[32:47]
	v_mfma_f32_32x32x2_f32 a[0:15], v83, v15, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v83, v7, a[16:31]
	v_mfma_f32_32x32x2_f32 a[48:63], v26, v7, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v26, v15, a[32:47]
	.loc	1 79 33                         ; matmul.py:79:33
	s_nop 7
	s_nop 6
	ds_write_b128 v1, a[0:3]
	ds_write_b128 v1, a[8:11] offset:256
	s_nop 0
	ds_write_b128 v1, a[32:35] offset:512
	ds_write_b128 v1, a[40:43] offset:768
	ds_write_b128 v0, a[16:19]
	ds_write_b128 v0, a[24:27] offset:256
	ds_write_b128 v0, a[48:51] offset:512
	ds_write_b128 v0, a[56:59] offset:768
	; wave barrier
	ds_read_b128 v[36:39], v3
	ds_read_b128 v[40:43], v3 offset:128
	ds_read_b128 v[44:47], v3 offset:2048
	ds_read_b128 v[48:51], v3 offset:2176
	ds_read_b128 v[52:55], v2 offset:1024
	ds_read_b128 v[56:59], v2 offset:1152
	ds_read_b128 v[60:63], v2 offset:3072
	ds_read_b128 v[64:67], v2 offset:3200
	; wave barrier
	ds_write_b128 v1, a[4:7]
	ds_write_b128 v1, a[12:15] offset:256
	ds_write_b128 v1, a[36:39] offset:512
	ds_write_b128 v1, a[44:47] offset:768
	ds_write_b128 v0, a[20:23]
	ds_write_b128 v0, a[28:31] offset:256
	ds_write_b128 v0, a[52:55] offset:512
	ds_write_b128 v0, a[60:63] offset:768
	; wave barrier
	ds_read_b128 v[68:71], v3
	ds_read_b128 v[72:75], v3 offset:128
	ds_read_b128 v[76:79], v3 offset:2048
	ds_read_b128 v[80:83], v3 offset:2176
	ds_read_b128 v[84:87], v2 offset:1024
	ds_read_b128 v[88:91], v2 offset:1152
	ds_read_b128 v[92:95], v2 offset:3072
	ds_read_b128 v[96:99], v2 offset:3200
	v_accvgpr_read_b32 v2, a97
.Ltmp93:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	v_lshl_or_b32 v3, v2, 11, v4
	v_accvgpr_read_b32 v2, a98
	v_lshl_or_b32 v5, v2, 11, v4
	v_accvgpr_read_b32 v2, a103
	v_lshl_or_b32 v7, v2, 11, v4
	v_accvgpr_read_b32 v2, a104
	v_lshl_or_b32 v9, v2, 11, v4
	v_accvgpr_read_b32 v2, a105
	v_lshl_or_b32 v11, v2, 11, v4
	v_accvgpr_read_b32 v2, a106
	v_lshl_or_b32 v13, v2, 11, v4
	v_accvgpr_read_b32 v2, a93
	v_lshl_or_b32 v15, v2, 11, v4
	v_accvgpr_read_b32 v2, a94
	v_lshl_or_b32 v17, v2, 11, v4
	v_accvgpr_read_b32 v2, a96
	v_lshl_or_b32 v19, v2, 11, v4
	v_accvgpr_read_b32 v2, a99
	v_lshl_or_b32 v21, v2, 11, v4
	v_accvgpr_read_b32 v2, a100
	v_lshl_or_b32 v23, v2, 11, v4
	v_accvgpr_read_b32 v2, a101
	v_accvgpr_read_b32 v0, a92
	v_accvgpr_read_b32 v1, a95
	v_lshl_or_b32 v25, v2, 11, v4
	v_accvgpr_read_b32 v2, a102
	v_lshl_or_b32 v0, v0, 11, v4
	v_lshl_or_b32 v1, v1, 11, v4
	v_lshl_or_b32 v27, v2, 11, v4
	v_accvgpr_read_b32 v2, a107
	v_lshl_or_b32 v29, v2, 11, v4
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v0, s0, v0
	v_add_u32_e32 v2, s0, v1
	v_add_u32_e32 v4, s0, v3
.Ltmp94:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v1, 31, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[32:33], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[34:35], v[2:3], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v0, v36
	v_mov_b32_e32 v1, v44
	v_mov_b32_e32 v2, v52
	v_mov_b32_e32 v3, v60
.Ltmp95:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v6, s0, v5
	v_add_u32_e32 v8, s0, v7
.Ltmp96:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	global_store_dwordx4 v[32:33], v[0:3], off
.Ltmp97:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v10, s0, v9
.Ltmp98:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	v_mov_b32_e32 v0, v37
	v_mov_b32_e32 v1, v45
	v_mov_b32_e32 v2, v53
	v_mov_b32_e32 v3, v61
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[34:35], v[0:3], off
.Ltmp99:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v12, s0, v11
.Ltmp100:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	v_mov_b32_e32 v0, v38
	v_mov_b32_e32 v1, v46
	v_mov_b32_e32 v2, v54
	v_mov_b32_e32 v3, v62
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[8:9], v[8:9], 2, s[6:7]
	v_ashrrev_i32_e32 v11, 31, v10
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[4:5], v[0:3], off
	v_mov_b32_e32 v60, v39
	v_mov_b32_e32 v61, v47
	v_mov_b32_e32 v62, v55
	v_mov_b32_e32 v0, v40
	v_mov_b32_e32 v1, v48
	v_mov_b32_e32 v2, v56
	v_mov_b32_e32 v3, v64
.Ltmp101:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v14, s0, v13
	v_add_u32_e32 v16, s0, v15
.Ltmp102:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_ashrrev_i32_e32 v13, 31, v12
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	global_store_dwordx4 v[6:7], v[60:63], off
	global_store_dwordx4 v[8:9], v[0:3], off
.Ltmp103:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v18, s0, v17
.Ltmp104:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[12:13], v[12:13], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	v_mov_b32_e32 v0, v41
	v_mov_b32_e32 v1, v49
	v_mov_b32_e32 v2, v57
	v_mov_b32_e32 v3, v65
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[10:11], v[0:3], off
.Ltmp105:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v20, s0, v19
.Ltmp106:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	v_mov_b32_e32 v0, v42
	v_mov_b32_e32 v1, v50
	v_mov_b32_e32 v2, v58
	v_mov_b32_e32 v3, v66
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[6:7]
	v_ashrrev_i32_e32 v19, 31, v18
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[12:13], v[0:3], off
	v_mov_b32_e32 v64, v43
	v_mov_b32_e32 v65, v51
	v_mov_b32_e32 v66, v59
	s_waitcnt lgkmcnt(7)
	v_mov_b32_e32 v0, v68
	s_waitcnt lgkmcnt(5)
	v_mov_b32_e32 v1, v76
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v2, v84
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v3, v92
.Ltmp107:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v22, s0, v21
	v_add_u32_e32 v24, s0, v23
.Ltmp108:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[6:7]
	v_ashrrev_i32_e32 v21, 31, v20
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	global_store_dwordx4 v[14:15], v[64:67], off
	global_store_dwordx4 v[16:17], v[0:3], off
.Ltmp109:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v26, s0, v25
.Ltmp110:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	v_mov_b32_e32 v0, v69
	v_mov_b32_e32 v1, v77
	v_mov_b32_e32 v2, v85
	v_mov_b32_e32 v3, v93
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v25, 31, v24
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[18:19], v[0:3], off
.Ltmp111:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v28, s0, v27
	v_add_u32_e32 v30, s0, v29
.Ltmp112:
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v70
	v_mov_b32_e32 v1, v78
	v_mov_b32_e32 v2, v86
	v_mov_b32_e32 v3, v94
	.loc	1 88 25 is_stmt 0               ; matmul.py:88:25
	v_lshl_add_u64 v[22:23], v[22:23], 2, s[6:7]
	v_lshl_add_u64 v[24:25], v[24:25], 2, s[6:7]
	v_ashrrev_i32_e32 v27, 31, v26
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[20:21], v[0:3], off
	v_mov_b32_e32 v92, v71
	v_mov_b32_e32 v93, v79
	v_mov_b32_e32 v94, v87
	v_mov_b32_e32 v0, v72
	v_mov_b32_e32 v1, v80
	v_mov_b32_e32 v2, v88
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, v96
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[6:7]
	v_ashrrev_i32_e32 v29, 31, v28
	v_ashrrev_i32_e32 v31, 31, v30
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[22:23], v[92:95], off
	global_store_dwordx4 v[24:25], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[28:29], v[28:29], 2, s[6:7]
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[6:7]
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v73
	v_mov_b32_e32 v1, v81
	v_mov_b32_e32 v2, v89
	v_mov_b32_e32 v3, v97
	global_store_dwordx4 v[26:27], v[0:3], off
	v_mov_b32_e32 v96, v75
	v_mov_b32_e32 v97, v83
	v_mov_b32_e32 v0, v74
	v_mov_b32_e32 v1, v82
	v_mov_b32_e32 v2, v90
	v_mov_b32_e32 v3, v98
	v_mov_b32_e32 v98, v91
	global_store_dwordx4 v[28:29], v[0:3], off
	global_store_dwordx4 v[30:31], v[96:99], off
	.loc	1 90 4 is_stmt 1                ; matmul.py:90:4
	s_endpgm
.Ltmp113:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 13
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 376
		.amdhsa_next_free_sgpr 18
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	matmul, .Lfunc_end0-matmul
	.cfi_endproc
                                        ; -- End function
	.set matmul.num_vgpr, 256
	.set matmul.num_agpr, 120
	.set matmul.numbered_sgpr, 18
	.set matmul.private_seg_size, 0
	.set matmul.uses_vcc, 0
	.set matmul.uses_flat_scratch, 0
	.set matmul.has_dyn_sized_stack, 0
	.set matmul.has_recursion, 0
	.set matmul.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10228
; TotalNumSgprs: 24
; NumVgprs: 256
; NumAgprs: 120
; TotalNumVgprs: 376
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 46
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 376
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x74 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x4e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	40                              ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	55                              ; DW_AT_call_line
	.byte	91                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x59:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	41                              ; DW_AT_call_line
	.byte	52                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x65:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	56                              ; DW_AT_call_line
	.byte	91                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x71:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	82                              ; DW_AT_call_line
	.byte	91                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul.py"                     ; string offset=7
.Linfo_string2:
	.asciz	"/home/nico/triton/sandbox"     ; string offset=17
.Linfo_string3:
	.asciz	"matmul"                        ; string offset=43
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     120
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         160
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .max_flat_workgroup_size: 64
    .name:           matmul
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         matmul.kd
    .uses_dynamic_stack: false
    .vgpr_count:     376
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:

Running Time   4.78888 ms
	    17.04TF/s
	     0.01TB/s
