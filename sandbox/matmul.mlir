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
#loc38 = loc("a_ptr"(#loc1))
#loc39 = loc("b_ptr"(#loc1))
#loc40 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c512_i32 = arith.constant 512 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1048576_i32 = arith.constant 1048576 : i32 loc(#loc)
    %cst_0 = arith.constant dense<16384> : tensor<64x1xi32, #blocked> loc(#loc)
    %c65536_i32 = arith.constant 65536 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst_1 = arith.constant dense<2048> : tensor<32x1xi32, #blocked> loc(#loc)
    %c131072_i32 = arith.constant 131072 : i32 loc(#loc)
    %cst_2 = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc)
    %pid0 = tt.get_program_id x : i32 loc(#loc70)
    %pid1 = tt.get_program_id y : i32 loc(#loc71)
    %pid2 = tt.get_program_id z : i32 loc(#loc72)
    %npg1 = tt.get_num_programs y : i32 loc(#loc73)
    %npg2 = tt.get_num_programs z : i32 loc(#loc74)
    %stride_val = arith.muli %npg2, %npg1 : i32 loc(#loc96)
    %linear_idx = arith.muli %pid0, %stride_val : i32 loc(#loc88)
    %linear_idx_3 = arith.muli %pid1, %npg2 : i32 loc(#loc88)
    %linear_idx_4 = arith.addi %linear_idx, %linear_idx_3 : i32 loc(#loc89)
    %linear_idx_5 = arith.addi %linear_idx_4, %pid2 : i32 loc(#loc89)
    %idx_val = arith.divsi %linear_idx_5, %c32_i32 : i32 loc(#loc76)
    %remaining = arith.remsi %linear_idx_5, %c32_i32 : i32 loc(#loc77)
    %acc = scf.for %k = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%acc_15 = %cst) -> (tensor<64x64xf32, #mma>)  : i32 {
      %base_offsets_nd_16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc78)
      %base_offsets_nd_17 = tt.expand_dims %base_offsets_nd_16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc78)
      %base_offsets_nd_18 = arith.muli %base_offsets_nd_17, %cst_0 : tensor<64x1xi32, #blocked> loc(#loc78)
      %base_offsets_nd_19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc78)
      %base_offsets_nd_20 = tt.expand_dims %base_offsets_nd_19 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc78)
      %base_offsets_nd_21 = tt.broadcast %base_offsets_nd_18 : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc78)
      %base_offsets_nd_22 = tt.broadcast %base_offsets_nd_20 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc78)
      %base_offsets_nd_23 = arith.addi %base_offsets_nd_21, %base_offsets_nd_22 : tensor<64x32xi32, #blocked> loc(#loc78)
      %shift_1d_24 = arith.muli %idx_val, %c1048576_i32 : i32 loc(#loc90)
      %shift_1d_25 = arith.muli %k, %c32_i32 : i32 loc(#loc90)
      %res_26 = arith.addi %shift_1d_24, %shift_1d_25 : i32 loc(#loc91)
      %4 = tt.splat %res_26 : i32 -> tensor<64x32xi32, #blocked> loc(#loc59)
      %5 = arith.addi %base_offsets_nd_23, %4 : tensor<64x32xi32, #blocked> loc(#loc59)
      %base_offsets_nd_27 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc81)
      %base_offsets_nd_28 = tt.expand_dims %base_offsets_nd_27 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked> loc(#loc81)
      %base_offsets_nd_29 = arith.muli %base_offsets_nd_28, %cst_1 : tensor<32x1xi32, #blocked> loc(#loc81)
      %base_offsets_nd_30 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc81)
      %base_offsets_nd_31 = tt.expand_dims %base_offsets_nd_30 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc81)
      %base_offsets_nd_32 = tt.broadcast %base_offsets_nd_29 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc81)
      %base_offsets_nd_33 = tt.broadcast %base_offsets_nd_31 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc81)
      %base_offsets_nd_34 = arith.addi %base_offsets_nd_32, %base_offsets_nd_33 : tensor<32x64xi32, #blocked> loc(#loc81)
      %shift_1d_35 = arith.muli %k, %c65536_i32 : i32 loc(#loc92)
      %shift_1d_36 = arith.muli %remaining, %c64_i32 : i32 loc(#loc92)
      %res_37 = arith.addi %shift_1d_35, %shift_1d_36 : i32 loc(#loc93)
      %6 = tt.splat %res_37 : i32 -> tensor<32x64xi32, #blocked> loc(#loc60)
      %7 = arith.addi %base_offsets_nd_34, %6 : tensor<32x64xi32, #blocked> loc(#loc60)
      %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc61)
      %a_38 = tt.addptr %a, %5 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc61)
      %a_39 = tt.load %a_38 : tensor<64x32x!tt.ptr<f32>, #blocked> loc(#loc62)
      %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc63)
      %b_40 = tt.addptr %b, %7 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked> loc(#loc63)
      %b_41 = tt.load %b_40 : tensor<32x64x!tt.ptr<f32>, #blocked> loc(#loc64)
      %a_42 = ttg.convert_layout %a_39 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc65)
      %b_43 = ttg.convert_layout %b_41 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc66)
      %acc_44 = tt.dot %a_42, %b_43, %acc_15 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma> loc(#loc67)
      scf.yield %acc_44 : tensor<64x64xf32, #mma> loc(#loc32)
    } loc(#loc54)
    %acc_6 = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked> loc(#loc68)
    %base_offsets_nd = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc84)
    %base_offsets_nd_7 = tt.expand_dims %base_offsets_nd {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc84)
    %base_offsets_nd_8 = arith.muli %base_offsets_nd_7, %cst_2 : tensor<64x1xi32, #blocked> loc(#loc84)
    %base_offsets_nd_9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc84)
    %base_offsets_nd_10 = tt.expand_dims %base_offsets_nd_9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc84)
    %base_offsets_nd_11 = tt.broadcast %base_offsets_nd_8 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc84)
    %base_offsets_nd_12 = tt.broadcast %base_offsets_nd_10 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc84)
    %base_offsets_nd_13 = arith.addi %base_offsets_nd_11, %base_offsets_nd_12 : tensor<64x64xi32, #blocked> loc(#loc84)
    %shift_1d = arith.muli %idx_val, %c131072_i32 : i32 loc(#loc94)
    %shift_1d_14 = arith.muli %remaining, %c64_i32 : i32 loc(#loc94)
    %res = arith.addi %shift_1d, %shift_1d_14 : i32 loc(#loc95)
    %0 = tt.splat %res : i32 -> tensor<64x64xi32, #blocked> loc(#loc69)
    %1 = arith.addi %base_offsets_nd_13, %0 : tensor<64x64xi32, #blocked> loc(#loc69)
    %2 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc35)
    %3 = tt.addptr %2, %1 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc35)
    tt.store %3, %acc_6 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc36)
    tt.return loc(#loc37)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/nico/triton/sandbox/tuple_helpers.py":83:25)
#loc3 = loc("/home/nico/triton/sandbox/matmul.py":40:24)
#loc4 = loc("/home/nico/triton/sandbox/tuple_helpers.py":84:25)
#loc5 = loc("/home/nico/triton/sandbox/tuple_helpers.py":85:25)
#loc6 = loc("/home/nico/triton/sandbox/tuple_helpers.py":87:27)
#loc7 = loc("/home/nico/triton/sandbox/tuple_helpers.py":88:27)
#loc8 = loc("/home/nico/triton/sandbox/tuple_helpers.py":27:38)
#loc9 = loc("/home/nico/triton/sandbox/tuple_helpers.py":48:30)
#loc10 = loc("/home/nico/triton/sandbox/tuple_helpers.py":89:51)
#loc11 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:35)
#loc12 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:22)
#loc13 = loc("/home/nico/triton/sandbox/tuple_helpers.py":73:31)
#loc14 = loc("/home/nico/triton/sandbox/matmul.py":41:52)
#loc15 = loc("/home/nico/triton/sandbox/tuple_helpers.py":75:32)
#loc16 = loc("/home/nico/triton/sandbox/matmul.py":50:25)
#loc17 = loc("/home/nico/triton/sandbox/nd_helpers.py":50:8)
#loc18 = loc("/home/nico/triton/sandbox/matmul.py":55:91)
#loc19 = loc("/home/nico/triton/sandbox/tuple_helpers.py":115:35)
#loc20 = loc("/home/nico/triton/sandbox/nd_helpers.py":54:8)
#loc21 = loc("/home/nico/triton/sandbox/tuple_helpers.py":132:15)
#loc22 = loc("/home/nico/triton/sandbox/nd_helpers.py":52:32)
#loc23 = loc("/home/nico/triton/sandbox/nd_helpers.py":55:29)
#loc24 = loc("/home/nico/triton/sandbox/matmul.py":56:91)
#loc25 = loc("/home/nico/triton/sandbox/matmul.py":62:32)
#loc26 = loc("/home/nico/triton/sandbox/matmul.py":62:39)
#loc27 = loc("/home/nico/triton/sandbox/matmul.py":63:32)
#loc28 = loc("/home/nico/triton/sandbox/matmul.py":63:39)
#loc29 = loc("/home/nico/triton/sandbox/matmul.py":73:37)
#loc30 = loc("/home/nico/triton/sandbox/matmul.py":74:37)
#loc31 = loc("/home/nico/triton/sandbox/matmul.py":75:42)
#loc32 = loc("/home/nico/triton/sandbox/matmul.py":66:8)
#loc33 = loc("/home/nico/triton/sandbox/matmul.py":79:33)
#loc34 = loc("/home/nico/triton/sandbox/matmul.py":82:91)
#loc35 = loc("/home/nico/triton/sandbox/matmul.py":88:25)
#loc36 = loc("/home/nico/triton/sandbox/matmul.py":88:41)
#loc37 = loc("/home/nico/triton/sandbox/matmul.py":90:4)
#loc41 = loc("pid0"(#loc2))
#loc42 = loc("linear_program_id"(#loc3))
#loc43 = loc("pid1"(#loc4))
#loc44 = loc("pid2"(#loc5))
#loc45 = loc("npg1"(#loc6))
#loc46 = loc("npg2"(#loc7))
#loc47 = loc("stride_val"(#loc8))
#loc48 = loc("strides"(#loc9))
#loc49 = loc("linear_idx"(#loc11))
#loc50 = loc("linear_idx"(#loc12))
#loc51 = loc("idx_val"(#loc13))
#loc52 = loc("start_blocks_c"(#loc14))
#loc53 = loc("remaining"(#loc15))
#loc54 = loc("acc"(#loc16))
#loc55 = loc("base_offsets_nd"(#loc17))
#loc56 = loc("shift_1d"(#loc20))
#loc57 = loc("res"(#loc21))
#loc58 = loc("shift_1d"(#loc22))
#loc59 = loc(callsite(#loc23 at #loc18))
#loc60 = loc(callsite(#loc23 at #loc24))
#loc61 = loc("a"(#loc25))
#loc62 = loc("a"(#loc26))
#loc63 = loc("b"(#loc27))
#loc64 = loc("b"(#loc28))
#loc65 = loc("a"(#loc29))
#loc66 = loc("b"(#loc30))
#loc67 = loc("acc"(#loc31))
#loc68 = loc("acc"(#loc33))
#loc69 = loc(callsite(#loc23 at #loc34))
#loc70 = loc(callsite(#loc41 at #loc42))
#loc71 = loc(callsite(#loc43 at #loc42))
#loc72 = loc(callsite(#loc44 at #loc42))
#loc73 = loc(callsite(#loc45 at #loc42))
#loc74 = loc(callsite(#loc46 at #loc42))
#loc75 = loc(callsite(#loc10 at #loc42))
#loc76 = loc(callsite(#loc51 at #loc52))
#loc77 = loc(callsite(#loc53 at #loc52))
#loc78 = loc(callsite(#loc55 at #loc18))
#loc79 = loc(callsite(#loc56 at #loc18))
#loc80 = loc(callsite(#loc58 at #loc18))
#loc81 = loc(callsite(#loc55 at #loc24))
#loc82 = loc(callsite(#loc56 at #loc24))
#loc83 = loc(callsite(#loc58 at #loc24))
#loc84 = loc(callsite(#loc55 at #loc34))
#loc85 = loc(callsite(#loc56 at #loc34))
#loc86 = loc(callsite(#loc58 at #loc34))
#loc87 = loc(callsite(#loc48 at #loc75))
#loc88 = loc(callsite(#loc49 at #loc75))
#loc89 = loc(callsite(#loc50 at #loc75))
#loc90 = loc(callsite(#loc19 at #loc79))
#loc91 = loc(callsite(#loc57 at #loc80))
#loc92 = loc(callsite(#loc19 at #loc82))
#loc93 = loc(callsite(#loc57 at #loc83))
#loc94 = loc(callsite(#loc19 at #loc85))
#loc95 = loc(callsite(#loc57 at #loc86))
#loc96 = loc(callsite(#loc47 at #loc87))

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
; %bb.3:
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
; %bb.4:
.LBB0_0:
.Ltmp1:
	.file	2 "/home/nico/triton/sandbox" "tuple_helpers.py"
	.loc	2 87 27 is_stmt 1               ; tuple_helpers.py:87:27 @[ matmul.py:40:24 ]
	s_load_dword s0, s[0:1], 0x3c
	v_lshlrev_b32_e32 v15, 6, v0
	v_and_b32_e32 v6, 16, v0
	v_and_b32_e32 v9, 1, v0
	v_and_b32_e32 v13, 8, v0
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s8, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_cmp_lg_u32 s0, 0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lg_u64 s[0:1], 0
	s_addc_u32 s10, s13, 0
	.loc	2 88 27                         ; tuple_helpers.py:88:27 @[ matmul.py:40:24 ]
	s_cmp_lg_u32 s8, 0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lg_u64 s[0:1], 0
	.loc	2 52 35                         ; tuple_helpers.py:52:35 @[ matmul.py:40:24 ]
	s_mul_i32 s1, s10, s15
	.loc	2 88 27                         ; tuple_helpers.py:88:27 @[ matmul.py:40:24 ]
	s_addc_u32 s0, s14, 0
	.loc	2 52 22                         ; tuple_helpers.py:52:22 @[ matmul.py:40:24 ]
	s_add_i32 s1, s1, s16
	s_mul_i32 s10, s1, s0
	v_and_b32_e32 v15, 0x800, v15
	s_add_i32 s10, s10, s17
	v_lshlrev_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v10, 11, v9
	v_lshlrev_b32_e32 v9, 12, v9
	v_lshlrev_b32_e32 v14, 1, v13
	v_lshl_or_b32 v15, v6, 3, v15
	v_lshlrev_b32_e32 v40, 4, v0
	v_lshlrev_b32_e32 v41, 8, v0
.Ltmp2:
	.loc	2 73 31                         ; tuple_helpers.py:73:31 @[ matmul.py:41:52 ]
	s_ashr_i32 s0, s10, 31
	v_and_b32_e32 v5, 28, v4
	v_and_b32_e32 v1, 60, v4
	v_and_b32_e32 v7, 6, v0
	v_or3_b32 v9, v9, v14, v15
	s_movk_i32 s1, 0x110
	v_and_b32_e32 v16, 0x130, v40
	v_and_b32_e32 v17, 0x400, v41
	v_lshlrev_b32_e32 v13, 3, v13
	v_and_b32_e32 v42, 0xb0, v4
	v_and_b32_e32 v4, 2, v0
	s_lshr_b32 s0, s0, 27
	v_and_b32_e32 v2, 48, v0
	v_lshlrev_b32_e32 v8, 4, v7
	v_mad_u32_u24 v7, v7, s1, v9
	v_or3_b32 v13, v17, v13, v16
	v_bfe_i32 v17, v0, 1, 1
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_lshlrev_b32_e32 v4, 6, v6
	s_movk_i32 s1, 0x240
	s_add_i32 s0, s10, s0
	v_lshlrev_b32_e32 v3, 14, v2
	v_lshlrev_b32_e32 v2, 3, v2
	v_and_or_b32 v4, v17, s1, v4
.Ltmp3:
	.loc	1 50 25                         ; matmul.py:50:25
	s_lshl_b32 s1, s10, 6
.Ltmp4:
	.loc	2 73 31                         ; tuple_helpers.py:73:31 @[ matmul.py:41:52 ]
	s_ashr_i32 s8, s0, 5
	v_or3_b32 v2, v8, v2, v10
	v_or3_b32 v4, v4, v42, v10
.Ltmp5:
	.loc	1 50 25                         ; matmul.py:50:25
	v_lshl_add_u32 v6, v6, 11, s1
	s_lshl_b32 s0, s8, 20
	v_xor_b32_e32 v8, 32, v2
	v_xor_b32_e32 v11, 64, v2
	v_xor_b32_e32 v12, 0x60, v2
	v_xor_b32_e32 v9, 32, v7
	v_xor_b32_e32 v14, 64, v7
	v_xor_b32_e32 v15, 0x60, v7
	v_xor_b32_e32 v16, 64, v13
	v_xor_b32_e32 v10, 64, v4
	v_or_b32_e32 v6, v6, v1
	s_lshl_b32 s1, s8, 11
	s_mov_b32 s9, 0
	v_subrev_u32_e32 v43, s1, v6
	v_or3_b32 v38, s0, v3, v5
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	s_mov_b32 s11, 0x10000
	s_mov_b32 s12, 0x20000
	s_mov_b32 s13, 0x30000
	s_mov_b32 s14, 0x40000
	s_mov_b32 s15, 0x50000
	s_mov_b32 s16, 0x60000
	s_mov_b32 s17, 0x70000
	s_mov_b32 s18, 0x80000
	s_mov_b32 s19, 0x90000
	s_mov_b32 s20, 0xa0000
	s_mov_b32 s21, 0xb0000
	s_mov_b32 s22, 0xc0000
	s_mov_b32 s23, 0xd0000
	s_mov_b32 s24, 0xe0000
	s_mov_b32 s25, 0xf0000
	v_add_u32_e32 v44, 0, v2
	v_add_u32_e32 v45, 0, v8
	v_add_u32_e32 v46, 0, v11
	v_add_u32_e32 v47, 0, v12
	v_add_u32_e32 v48, 0, v7
	v_add_u32_e32 v49, 0, v9
	v_add_u32_e32 v50, 0, v14
	v_add_u32_e32 v51, 0, v15
	v_add_u32_e32 v52, 0, v13
	v_add_u32_e32 v53, 0, v16
	v_add_u32_e32 v54, 0, v4
	v_add_u32_e32 v55, 0, v10
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[32:33], v[38:39], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	v_add_co_u32_e64 v58, s[0:1], s11, v32
.Ltmp6:
	.file	3 "/home/nico/triton/sandbox" "nd_helpers.py"
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v34, s9, v43
.Ltmp7:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v59, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v60, s[0:1], s12, v32
.Ltmp8:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v30, 0x800, v34
.Ltmp9:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v61, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v62, s[0:1], s13, v32
.Ltmp10:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v28, 0x1000, v34
.Ltmp11:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v63, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v72, s[0:1], s14, v32
.Ltmp12:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v26, 0x1800, v34
.Ltmp13:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v73, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v76, s[0:1], s15, v32
.Ltmp14:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v24, 0x2000, v34
.Ltmp15:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v77, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v88, s[0:1], s16, v32
.Ltmp16:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v22, 0x2800, v34
.Ltmp17:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v89, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v92, s[0:1], s17, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v35, 31, v34
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v93, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v64, s[0:1], s18, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[56:57], v[34:35], 2, s[4:5]
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v65, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v66, s[0:1], s19, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v31, 31, v30
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v67, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v68, s[0:1], s20, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v29, 31, v28
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v69, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v70, s[0:1], s21, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v27, 31, v26
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v71, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v80, s[0:1], s22, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v25, 31, v24
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v81, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v84, s[0:1], s23, v32
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v23, 31, v22
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v85, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v96, s[0:1], s24, v32
.Ltmp18:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, 0x3000, v34
.Ltmp19:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v97, s[0:1], 0, v33, s[0:1]
	v_add_co_u32_e64 v100, s[0:1], s25, v32
.Ltmp20:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v4, 0x3800, v34
.Ltmp21:
	.loc	1 62 39                         ; matmul.py:62:39
	s_nop 0
	v_addc_co_u32_e64 v101, s[0:1], 0, v33, s[0:1]
.Ltmp22:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, 0x4000, v34
	v_add_u32_e32 v8, 0x4800, v34
	v_add_u32_e32 v10, 0x5000, v34
	v_add_u32_e32 v12, 0x5800, v34
	v_add_u32_e32 v14, 0x6000, v34
	v_add_u32_e32 v16, 0x6800, v34
	v_add_u32_e32 v18, 0x7000, v34
.Ltmp23:
	.loc	1 62 32                         ; matmul.py:62:32
	v_add_u32_e32 v20, 0x7800, v34
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[34:37], v[32:33], off
	.loc	1 63 39 is_stmt 1               ; matmul.py:63:39
	global_load_dwordx4 v[114:117], v[56:57], off
	.loc	1 63 32 is_stmt 0               ; matmul.py:63:32
	v_lshl_add_u64 v[104:105], v[30:31], 2, s[4:5]
	v_lshl_add_u64 v[106:107], v[28:29], 2, s[4:5]
	v_lshl_add_u64 v[108:109], v[26:27], 2, s[4:5]
	v_lshl_add_u64 v[110:111], v[24:25], 2, s[4:5]
	v_lshl_add_u64 v[112:113], v[22:23], 2, s[4:5]
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	global_load_dwordx4 v[22:25], v[58:59], off
	global_load_dwordx4 v[26:29], v[64:65], off
	global_load_dwordx4 v[30:33], v[66:67], off
	s_nop 0
	global_load_dwordx4 v[56:59], v[60:61], off
	s_nop 0
	global_load_dwordx4 v[60:63], v[62:63], off
	s_nop 0
	global_load_dwordx4 v[64:67], v[68:69], off
	s_nop 0
	global_load_dwordx4 v[68:71], v[70:71], off
	s_nop 0
	global_load_dwordx4 v[72:75], v[72:73], off
	s_nop 0
	global_load_dwordx4 v[76:79], v[76:77], off
	s_nop 0
	global_load_dwordx4 v[80:83], v[80:81], off
	s_nop 0
	global_load_dwordx4 v[84:87], v[84:85], off
	s_nop 0
	global_load_dwordx4 v[88:91], v[88:89], off
	s_nop 0
	global_load_dwordx4 v[92:95], v[92:93], off
	s_nop 0
	global_load_dwordx4 v[96:99], v[96:97], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v17, 31, v16
	v_ashrrev_i32_e32 v19, 31, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[4:5]
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	v_lshl_add_u64 v[8:9], v[8:9], 2, s[4:5]
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[4:5]
	v_lshl_add_u64 v[12:13], v[12:13], 2, s[4:5]
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[4:5]
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[4:5]
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[4:5]
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[118:121], v[104:105], off
	global_load_dwordx4 v[122:125], v[106:107], off
	global_load_dwordx4 v[126:129], v[108:109], off
	global_load_dwordx4 v[130:133], v[110:111], off
	s_nop 0
	global_load_dwordx4 v[110:113], v[112:113], off
	s_nop 0
	global_load_dwordx4 v[134:137], v[2:3], off
	global_load_dwordx4 v[138:141], v[4:5], off
	global_load_dwordx4 v[142:145], v[6:7], off
	global_load_dwordx4 v[146:149], v[8:9], off
	global_load_dwordx4 v[150:153], v[10:11], off
	global_load_dwordx4 v[154:157], v[12:13], off
	global_load_dwordx4 v[158:161], v[14:15], off
	global_load_dwordx4 v[162:165], v[16:17], off
	global_load_dwordx4 v[166:169], v[18:19], off
	global_load_dwordx4 v[170:173], v[20:21], off
	; wave barrier
	.loc	1 50 25 is_stmt 1               ; matmul.py:50:25
	s_add_i32 s9, s9, 0x10000
	s_cmp_lg_u32 s9, 0x2000000
	v_add_u32_e32 v38, 32, v38
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(31)
	ds_write_b128 v44, v[34:37]
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt vmcnt(30)
	v_mov_b32_e32 v34, v114
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(29)
	ds_write_b128 v44, v[22:25] offset:4096
	s_waitcnt vmcnt(28)
	ds_write_b128 v44, v[26:29] offset:16
	s_waitcnt vmcnt(27)
	ds_write_b128 v44, v[30:33] offset:4112
	s_waitcnt vmcnt(26)
	ds_write_b128 v45, v[56:59] offset:512
	s_waitcnt vmcnt(25)
	ds_write_b128 v45, v[60:63] offset:4608
	s_waitcnt vmcnt(24)
	ds_write_b128 v45, v[64:67] offset:528
	s_waitcnt vmcnt(23)
	ds_write_b128 v45, v[68:71] offset:4624
	s_waitcnt vmcnt(22)
	ds_write_b128 v46, v[72:75] offset:1024
	s_waitcnt vmcnt(21)
	ds_write_b128 v46, v[76:79] offset:5120
	s_waitcnt vmcnt(20)
	ds_write_b128 v46, v[80:83] offset:1040
	s_waitcnt vmcnt(19)
	ds_write_b128 v46, v[84:87] offset:5136
	s_waitcnt vmcnt(18)
	ds_write_b128 v47, v[88:91] offset:1536
	s_waitcnt vmcnt(17)
	ds_write_b128 v47, v[92:95] offset:5632
	s_waitcnt vmcnt(16)
	ds_write_b128 v47, v[96:99] offset:1552
	s_waitcnt vmcnt(15)
	ds_write_b128 v47, v[100:103] offset:5648
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v104, v115
	v_mov_b32_e32 v108, v116
	; wave barrier
	.loc	1 73 37                         ; matmul.py:73:37
	ds_read_b128 v[30:33], v48
	ds_read_b128 v[26:29], v48 offset:256
	ds_read_b128 v[22:25], v49
	ds_read_b128 v[18:21], v49 offset:256
	ds_read_b128 v[14:17], v50
	ds_read_b128 v[10:13], v50 offset:256
	ds_read_b128 v[6:9], v51
	ds_read_b128 v[2:5], v51 offset:256
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt vmcnt(14) lgkmcnt(0)
	; wave barrier
	v_mov_b32_e32 v35, v118
	s_waitcnt vmcnt(13)
	v_mov_b32_e32 v36, v122
	s_waitcnt vmcnt(12)
	v_mov_b32_e32 v37, v126
	s_waitcnt vmcnt(11)
	v_mov_b32_e32 v56, v130
	s_waitcnt vmcnt(10)
	v_mov_b32_e32 v57, v110
	s_waitcnt vmcnt(9)
	v_mov_b32_e32 v58, v134
	s_waitcnt vmcnt(8)
	v_mov_b32_e32 v59, v138
	v_mov_b32_e32 v60, v131
	v_mov_b32_e32 v61, v111
	v_mov_b32_e32 v62, v135
	v_mov_b32_e32 v63, v139
	v_mov_b32_e32 v64, v132
	v_mov_b32_e32 v65, v112
	v_mov_b32_e32 v66, v136
	v_mov_b32_e32 v67, v140
	v_mov_b32_e32 v105, v119
	v_mov_b32_e32 v106, v123
	v_mov_b32_e32 v107, v127
	v_mov_b32_e32 v109, v120
	v_mov_b32_e32 v110, v124
	v_mov_b32_e32 v111, v128
	v_mov_b32_e32 v126, v117
	v_mov_b32_e32 v127, v121
	v_mov_b32_e32 v128, v125
	v_mov_b32_e32 v138, v133
	v_mov_b32_e32 v139, v113
	v_mov_b32_e32 v140, v137
	ds_write_b128 v52, v[34:37]
	ds_write_b128 v52, v[104:107] offset:2048
	ds_write_b128 v52, v[56:59] offset:128
	ds_write_b128 v52, v[60:63] offset:2176
	ds_write_b128 v53, v[108:111] offset:512
	ds_write_b128 v53, v[126:129] offset:2560
	ds_write_b128 v53, v[64:67] offset:640
	ds_write_b128 v53, v[138:141] offset:2688
	; wave barrier
	ds_read_b128 v[34:37], v54
	ds_read_b128 v[56:59], v54 offset:256
	ds_read_b128 v[60:63], v55
	ds_read_b128 v[64:67], v55 offset:256
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[48:63], v30, v34, a[48:63]
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt vmcnt(7)
	v_mov_b32_e32 v68, v143
	s_waitcnt vmcnt(6)
	v_mov_b32_e32 v69, v147
	s_waitcnt vmcnt(5)
	v_mov_b32_e32 v70, v151
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v71, v155
	s_waitcnt vmcnt(3)
	v_mov_b32_e32 v72, v158
	s_waitcnt vmcnt(2)
	v_mov_b32_e32 v73, v162
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v74, v166
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v75, v170
	v_mov_b32_e32 v76, v144
	v_mov_b32_e32 v77, v148
	v_mov_b32_e32 v78, v152
	v_mov_b32_e32 v79, v156
	v_mov_b32_e32 v155, v149
	v_mov_b32_e32 v156, v153
	v_mov_b32_e32 v80, v160
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[32:47], v30, v60, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v30, v142
	v_mov_b32_e32 v81, v164
	v_mov_b32_e32 v82, v168
	v_mov_b32_e32 v83, v172
	v_mov_b32_e32 v170, v161
	v_mov_b32_e32 v172, v169
	; wave barrier
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v26, v34, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v34, v159
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v26, v60, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v31, v35, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v31, v61, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v31, v146
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v27, v35, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v35, v163
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v27, v61, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v32, v36, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v32, v62, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v32, v150
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v28, v36, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v36, v167
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v28, v62, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v33, v37, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v33, v63, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v33, v154
	v_mov_b32_e32 v154, v145
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v29, v37, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v37, v171
	v_mov_b32_e32 v171, v165
	ds_write_b128 v52, v[30:33]
	ds_write_b128 v52, v[68:71] offset:2048
	ds_write_b128 v52, v[72:75] offset:128
	ds_write_b128 v52, v[34:37] offset:2176
	ds_write_b128 v53, v[76:79] offset:512
	ds_write_b128 v53, v[154:157] offset:2560
	ds_write_b128 v53, v[80:83] offset:640
	ds_write_b128 v53, v[170:173] offset:2688
	; wave barrier
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v29, v63, a[0:15]
	.loc	1 74 37                         ; matmul.py:74:37
	ds_read_b128 v[26:29], v54
	ds_read_b128 v[30:33], v54 offset:256
	ds_read_b128 v[34:37], v55
	ds_read_b128 v[60:63], v55 offset:256
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[48:63], v22, v26, a[48:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[32:47], v22, v34, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v18, v26, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v18, v34, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v23, v27, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v23, v35, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v19, v27, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v19, v35, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v24, v28, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v24, v36, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v20, v28, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v20, v36, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v25, v29, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v25, v37, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v21, v29, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v21, v37, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v14, v56, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v14, v64, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v10, v56, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v10, v64, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v15, v57, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v15, v65, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v11, v57, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v11, v65, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v16, v58, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v16, v66, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v12, v58, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v12, v66, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v17, v59, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v17, v67, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v13, v59, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v13, v67, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v6, v30, a[48:63]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x2_f32 a[32:47], v6, v60, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v2, v30, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v2, v60, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v7, v31, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v7, v61, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v3, v31, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v3, v61, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v8, v32, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v8, v62, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v4, v32, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v4, v62, a[0:15]
	v_mfma_f32_32x32x2_f32 a[48:63], v9, v33, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v9, v63, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v5, v33, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v5, v63, a[0:15]
	.loc	1 50 25                         ; matmul.py:50:25
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	.loc	1 79 33                         ; matmul.py:79:33
	v_lshlrev_b32_e32 v2, 11, v0
	v_and_b32_e32 v3, 0x800, v2
	v_mov_b32_e32 v4, 0x440
	s_movk_i32 s1, 0x1000
	v_lshlrev_b32_e32 v6, 3, v0
	v_cndmask_b32_e64 v4, v4, 0, vcc
	v_and_or_b32 v3, v41, s1, v3
	v_and_b32_e32 v5, 0x330, v40
	v_lshl_or_b32 v0, v0, 10, v6
	s_movk_i32 s1, 0x1040
.Ltmp24:
	.loc	2 75 32                         ; tuple_helpers.py:75:32 @[ matmul.py:41:52 ]
	s_lshl_b32 s0, s8, 5
.Ltmp25:
	.loc	1 79 33                         ; matmul.py:79:33
	v_or3_b32 v3, v3, v4, v42
	v_and_or_b32 v0, v0, s1, v5
.Ltmp26:
	.loc	2 75 32                         ; tuple_helpers.py:75:32 @[ matmul.py:41:52 ]
	s_sub_i32 s0, s10, s0
.Ltmp27:
	.loc	1 79 33                         ; matmul.py:79:33
	v_add_u32_e32 v4, 0, v3
	v_xad_u32 v3, v3, 64, 0
	v_add_u32_e32 v5, 0, v0
	v_xad_u32 v0, v0, 64, 0
	s_mov_b32 s1, 0x18000
	s_lshl_b32 s0, s0, 6
	; wave barrier
	ds_write_b128 v4, a[48:51]
	ds_write_b128 v4, a[56:59] offset:256
	ds_write_b128 v4, a[16:19] offset:512
	ds_write_b128 v4, a[24:27] offset:768
	ds_write_b128 v3, a[32:35]
	ds_write_b128 v3, a[40:43] offset:256
	ds_write_b128 v3, a[0:3] offset:512
	ds_write_b128 v3, a[8:11] offset:768
	; wave barrier
	ds_read_b128 v[36:39], v5
	ds_read_b128 v[40:43], v5 offset:128
	ds_read_b128 v[44:47], v5 offset:2048
	ds_read_b128 v[48:51], v5 offset:2176
	ds_read_b128 v[52:55], v0 offset:1024
	ds_read_b128 v[56:59], v0 offset:1152
	ds_read_b128 v[60:63], v0 offset:3072
	ds_read_b128 v[64:67], v0 offset:3200
	; wave barrier
	ds_write_b128 v4, a[52:55]
	ds_write_b128 v4, a[60:63] offset:256
	ds_write_b128 v4, a[20:23] offset:512
	ds_write_b128 v4, a[28:31] offset:768
	ds_write_b128 v3, a[36:39]
	ds_write_b128 v3, a[44:47] offset:256
	ds_write_b128 v3, a[4:7] offset:512
	ds_write_b128 v3, a[12:15] offset:768
	; wave barrier
	ds_read_b128 v[68:71], v5
	ds_read_b128 v[72:75], v5 offset:128
	ds_read_b128 v[76:79], v5 offset:2048
	ds_read_b128 v[80:83], v5 offset:2176
	ds_read_b128 v[84:87], v0 offset:1024
	ds_read_b128 v[88:91], v0 offset:1152
	ds_read_b128 v[92:95], v0 offset:3072
	ds_read_b128 v[96:99], v0 offset:3200
.Ltmp28:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	v_and_or_b32 v0, v2, s1, v1
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:82:91 ]
	s_lshl_b32 s1, s8, 17
	.loc	2 132 15                        ; tuple_helpers.py:132:15 @[ matmul.py:82:91 ]
	s_add_i32 s1, s1, s0
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v0, s1, v0
	v_add_u32_e32 v2, 0x800, v0
.Ltmp29:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v1, 31, v0
	v_ashrrev_i32_e32 v3, 31, v2
.Ltmp30:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v4, 0x1000, v0
	v_add_u32_e32 v6, 0x1800, v0
	v_add_u32_e32 v8, 0x2000, v0
	v_add_u32_e32 v10, 0x2800, v0
	v_add_u32_e32 v12, 0x3000, v0
	v_add_u32_e32 v14, 0x3800, v0
	v_add_u32_e32 v16, 0x4000, v0
	v_add_u32_e32 v18, 0x4800, v0
	v_add_u32_e32 v20, 0x5000, v0
	v_add_u32_e32 v22, 0x5800, v0
	v_add_u32_e32 v24, 0x6000, v0
	v_add_u32_e32 v26, 0x6800, v0
	v_add_u32_e32 v28, 0x7000, v0
	v_add_u32_e32 v30, 0x7800, v0
.Ltmp31:
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[32:33], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[34:35], v[2:3], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v0, v36
	v_mov_b32_e32 v1, v44
	v_mov_b32_e32 v2, v52
	v_mov_b32_e32 v3, v60
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[32:33], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_ashrrev_i32_e32 v7, 31, v6
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v37
	v_mov_b32_e32 v1, v45
	v_mov_b32_e32 v2, v53
	v_mov_b32_e32 v3, v61
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[34:35], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[6:7]
	v_lshl_add_u64 v[8:9], v[8:9], 2, s[6:7]
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v38
	v_mov_b32_e32 v1, v46
	v_mov_b32_e32 v2, v54
	v_mov_b32_e32 v3, v62
	.loc	1 88 25                         ; matmul.py:88:25
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
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_ashrrev_i32_e32 v13, 31, v12
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[6:7], v[60:63], off
	global_store_dwordx4 v[8:9], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[12:13], v[12:13], 2, s[6:7]
	v_ashrrev_i32_e32 v15, 31, v14
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v41
	v_mov_b32_e32 v1, v49
	v_mov_b32_e32 v2, v57
	v_mov_b32_e32 v3, v65
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[10:11], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[6:7]
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[6:7]
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v42
	v_mov_b32_e32 v1, v50
	v_mov_b32_e32 v2, v58
	v_mov_b32_e32 v3, v66
	.loc	1 88 25                         ; matmul.py:88:25
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
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[6:7]
	v_ashrrev_i32_e32 v21, 31, v20
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[14:15], v[64:67], off
	global_store_dwordx4 v[16:17], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[6:7]
	v_ashrrev_i32_e32 v23, 31, v22
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v69
	v_mov_b32_e32 v1, v77
	v_mov_b32_e32 v2, v85
	v_mov_b32_e32 v3, v93
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v25, 31, v24
	.loc	1 88 41                         ; matmul.py:88:41
	global_store_dwordx4 v[18:19], v[0:3], off
	.loc	1 88 25                         ; matmul.py:88:25
	v_lshl_add_u64 v[22:23], v[22:23], 2, s[6:7]
	v_lshl_add_u64 v[24:25], v[24:25], 2, s[6:7]
	.loc	1 88 41                         ; matmul.py:88:41
	v_mov_b32_e32 v0, v70
	v_mov_b32_e32 v1, v78
	v_mov_b32_e32 v2, v86
	v_mov_b32_e32 v3, v94
	.loc	1 88 25                         ; matmul.py:88:25
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
.Ltmp32:
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
		.amdhsa_next_free_vgpr 240
		.amdhsa_next_free_sgpr 26
		.amdhsa_accum_offset 176
		.amdhsa_reserve_vcc 1
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
	.set matmul.num_vgpr, 174
	.set matmul.num_agpr, 64
	.set matmul.numbered_sgpr, 26
	.set matmul.private_seg_size, 0
	.set matmul.uses_vcc, 1
	.set matmul.uses_flat_scratch, 0
	.set matmul.has_dyn_sized_stack, 0
	.set matmul.has_recursion, 0
	.set matmul.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4648
; TotalNumSgprs: 32
; NumVgprs: 174
; NumAgprs: 64
; TotalNumVgprs: 240
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 240
; AccumOffset: 176
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 43
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
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xb:0x70 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x4a DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	40                              ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	41                              ; DW_AT_call_line
	.byte	52                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x61:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	56                              ; DW_AT_call_line
	.byte	91                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	82                              ; DW_AT_call_line
	.byte	91                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
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
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
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
  - .agpr_count:     64
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
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         matmul.kd
    .uses_dynamic_stack: false
    .vgpr_count:     240
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

Running Time   2.75965 ms
	    29.57TF/s
	     0.02TB/s

