// --- Input IR 
#blocked = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc1 = loc("/home/nico/triton/sandbox/matmul.py":30:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
#loc41 = loc("a_ptr"(#loc1))
#loc42 = loc("b_ptr"(#loc1))
#loc43 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %base_offsets_nd = arith.constant dense<2048> : tensor<128x1xi32, #blocked> loc(#loc73)
    %shift_1d = arith.constant 262144 : i32 loc(#loc92)
    %cst = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc)
    %c131072_i32 = arith.constant 131072 : i32 loc(#loc)
    %cst_0 = arith.constant dense<32768> : tensor<128x1xi32, #blocked> loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %c4194304_i32 = arith.constant 4194304 : i32 loc(#loc)
    %stride_val = arith.constant 32 : i32 loc(#loc93)
    %c1_i32 = arith.constant 1 : i32 loc(#loc9)
    %c512_i32 = arith.constant 512 : i32 loc(#loc9)
    %c0_i32 = arith.constant 0 : i32 loc(#loc9)
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma> loc(#loc49)
    %pid0 = tt.get_program_id x : i32 loc(#loc76)
    %pid1 = tt.get_program_id y : i32 loc(#loc77)
    %pid2 = tt.get_program_id z : i32 loc(#loc78)
    %npg1 = tt.get_num_programs y : i32 loc(#loc79)
    %npg2 = tt.get_num_programs z : i32 loc(#loc80)
    %stride_val_1 = arith.muli %npg2, %npg1 : i32 loc(#loc103)
    %linear_idx = arith.muli %pid0, %stride_val_1 : i32 loc(#loc95)
    %linear_idx_2 = arith.muli %pid1, %npg2 : i32 loc(#loc95)
    %linear_idx_3 = arith.addi %linear_idx, %linear_idx_2 : i32 loc(#loc96)
    %linear_idx_4 = arith.addi %linear_idx_3, %pid2 : i32 loc(#loc96)
    %idx_val = arith.divsi %linear_idx_4, %stride_val : i32 loc(#loc82)
    %remaining = arith.remsi %linear_idx_4, %stride_val : i32 loc(#loc83)
    %acc_5 = scf.for %k = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%acc_17 = %acc) -> (tensor<128x64xf32, #mma>)  : i32 {
      %base_offsets_nd_18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc84)
      %base_offsets_nd_19 = tt.expand_dims %base_offsets_nd_18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc84)
      %base_offsets_nd_20 = arith.muli %base_offsets_nd_19, %cst_0 : tensor<128x1xi32, #blocked> loc(#loc84)
      %base_offsets_nd_21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc84)
      %base_offsets_nd_22 = tt.expand_dims %base_offsets_nd_21 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc84)
      %base_offsets_nd_23 = tt.broadcast %base_offsets_nd_20 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc84)
      %base_offsets_nd_24 = tt.broadcast %base_offsets_nd_22 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc84)
      %base_offsets_nd_25 = arith.addi %base_offsets_nd_23, %base_offsets_nd_24 : tensor<128x64xi32, #blocked> loc(#loc84)
      %shift_1d_26 = arith.muli %idx_val, %c4194304_i32 : i32 loc(#loc97)
      %shift_1d_27 = arith.muli %k, %c64_i32 : i32 loc(#loc97)
      %res_28 = arith.addi %shift_1d_26, %shift_1d_27 : i32 loc(#loc98)
      %base_offsets_nd_29 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc87)
      %base_offsets_nd_30 = tt.expand_dims %base_offsets_nd_29 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc87)
      %base_offsets_nd_31 = arith.muli %base_offsets_nd_30, %cst : tensor<64x1xi32, #blocked> loc(#loc87)
      %base_offsets_nd_32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc87)
      %base_offsets_nd_33 = tt.expand_dims %base_offsets_nd_32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc87)
      %base_offsets_nd_34 = tt.broadcast %base_offsets_nd_31 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc87)
      %base_offsets_nd_35 = tt.broadcast %base_offsets_nd_33 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc87)
      %base_offsets_nd_36 = arith.addi %base_offsets_nd_34, %base_offsets_nd_35 : tensor<64x64xi32, #blocked> loc(#loc87)
      %shift_1d_37 = arith.muli %k, %c131072_i32 : i32 loc(#loc99)
      %shift_1d_38 = arith.muli %remaining, %c64_i32 : i32 loc(#loc99)
      %res_39 = arith.addi %shift_1d_37, %shift_1d_38 : i32 loc(#loc100)
      %a = tt.addptr %a_ptr, %res_28 : !tt.ptr<f32>, i32 loc(#loc65)
      %a_40 = amdgpu.buffer_load %a[%base_offsets_nd_25] : tensor<128x64xf32, #blocked> loc(#loc66)
      %b = tt.addptr %b_ptr, %res_39 : !tt.ptr<f32>, i32 loc(#loc67)
      %b_41 = amdgpu.buffer_load %b[%base_offsets_nd_36] : tensor<64x64xf32, #blocked> loc(#loc68)
      %a_42 = ttg.convert_layout %a_40 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc69)
      %b_43 = ttg.convert_layout %b_41 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc70)
      %acc_44 = tt.dot %a_42, %b_43, %acc_17 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma> loc(#loc71)
      scf.yield %acc_44 : tensor<128x64xf32, #mma> loc(#loc36)
    } loc(#loc61)
    %acc_6 = ttg.convert_layout %acc_5 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked> loc(#loc72)
    %base_offsets_nd_7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc73)
    %base_offsets_nd_8 = tt.expand_dims %base_offsets_nd_7 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc73)
    %base_offsets_nd_9 = arith.muli %base_offsets_nd_8, %base_offsets_nd : tensor<128x1xi32, #blocked> loc(#loc73)
    %base_offsets_nd_10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc73)
    %base_offsets_nd_11 = tt.expand_dims %base_offsets_nd_10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc73)
    %base_offsets_nd_12 = tt.broadcast %base_offsets_nd_9 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc73)
    %base_offsets_nd_13 = tt.broadcast %base_offsets_nd_11 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc73)
    %base_offsets_nd_14 = arith.addi %base_offsets_nd_12, %base_offsets_nd_13 : tensor<128x64xi32, #blocked> loc(#loc73)
    %shift_1d_15 = arith.muli %idx_val, %shift_1d : i32 loc(#loc101)
    %shift_1d_16 = arith.muli %remaining, %c64_i32 : i32 loc(#loc101)
    %res = arith.addi %shift_1d_15, %shift_1d_16 : i32 loc(#loc102)
    %0 = tt.addptr %c_ptr, %res : !tt.ptr<f32>, i32 loc(#loc38)
    amdgpu.buffer_store %acc_6, %0[%base_offsets_nd_14] : tensor<128x64xf32, #blocked> loc(#loc39)
    tt.return loc(#loc40)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/nico/triton/sandbox/nd_helpers.py":50:8)
#loc3 = loc("/home/nico/triton/sandbox/matmul.py":82:91)
#loc4 = loc("/home/nico/triton/sandbox/tuple_helpers.py":116:11)
#loc5 = loc("/home/nico/triton/sandbox/nd_helpers.py":54:44)
#loc6 = loc("/home/nico/triton/sandbox/tuple_helpers.py":27:38)
#loc7 = loc("/home/nico/triton/sandbox/tuple_helpers.py":68:30)
#loc8 = loc("/home/nico/triton/sandbox/matmul.py":41:52)
#loc9 = loc("/home/nico/triton/sandbox/matmul.py":50:25)
#loc10 = loc("/home/nico/triton/sandbox/matmul.py":43:33)
#loc11 = loc("/home/nico/triton/sandbox/tuple_helpers.py":83:25)
#loc12 = loc("/home/nico/triton/sandbox/matmul.py":40:24)
#loc13 = loc("/home/nico/triton/sandbox/tuple_helpers.py":84:25)
#loc14 = loc("/home/nico/triton/sandbox/tuple_helpers.py":85:25)
#loc15 = loc("/home/nico/triton/sandbox/tuple_helpers.py":87:27)
#loc16 = loc("/home/nico/triton/sandbox/tuple_helpers.py":88:27)
#loc17 = loc("/home/nico/triton/sandbox/tuple_helpers.py":48:30)
#loc18 = loc("/home/nico/triton/sandbox/tuple_helpers.py":89:51)
#loc19 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:35)
#loc20 = loc("/home/nico/triton/sandbox/tuple_helpers.py":52:22)
#loc21 = loc("/home/nico/triton/sandbox/tuple_helpers.py":73:31)
#loc22 = loc("/home/nico/triton/sandbox/tuple_helpers.py":75:32)
#loc23 = loc("/home/nico/triton/sandbox/matmul.py":55:91)
#loc24 = loc("/home/nico/triton/sandbox/tuple_helpers.py":115:35)
#loc25 = loc("/home/nico/triton/sandbox/nd_helpers.py":54:8)
#loc26 = loc("/home/nico/triton/sandbox/tuple_helpers.py":132:15)
#loc27 = loc("/home/nico/triton/sandbox/nd_helpers.py":52:32)
#loc28 = loc("/home/nico/triton/sandbox/matmul.py":56:91)
#loc29 = loc("/home/nico/triton/sandbox/matmul.py":59:49)
#loc30 = loc("/home/nico/triton/sandbox/matmul.py":59:74)
#loc31 = loc("/home/nico/triton/sandbox/matmul.py":60:49)
#loc32 = loc("/home/nico/triton/sandbox/matmul.py":60:74)
#loc33 = loc("/home/nico/triton/sandbox/matmul.py":73:37)
#loc34 = loc("/home/nico/triton/sandbox/matmul.py":74:37)
#loc35 = loc("/home/nico/triton/sandbox/matmul.py":75:42)
#loc36 = loc("/home/nico/triton/sandbox/matmul.py":66:8)
#loc37 = loc("/home/nico/triton/sandbox/matmul.py":79:33)
#loc38 = loc("/home/nico/triton/sandbox/matmul.py":86:47)
#loc39 = loc("/home/nico/triton/sandbox/matmul.py":86:72)
#loc40 = loc("/home/nico/triton/sandbox/matmul.py":90:4)
#loc44 = loc("base_offsets_nd"(#loc2))
#loc45 = loc("shift_1d"(#loc5))
#loc46 = loc("stride_val"(#loc6))
#loc47 = loc("strides"(#loc7))
#loc48 = loc("start_blocks_c"(#loc8))
#loc49 = loc("acc"(#loc10))
#loc50 = loc("pid0"(#loc11))
#loc51 = loc("linear_program_id"(#loc12))
#loc52 = loc("pid1"(#loc13))
#loc53 = loc("pid2"(#loc14))
#loc54 = loc("npg1"(#loc15))
#loc55 = loc("npg2"(#loc16))
#loc56 = loc("strides"(#loc17))
#loc57 = loc("linear_idx"(#loc19))
#loc58 = loc("linear_idx"(#loc20))
#loc59 = loc("idx_val"(#loc21))
#loc60 = loc("remaining"(#loc22))
#loc61 = loc("acc"(#loc9))
#loc62 = loc("shift_1d"(#loc25))
#loc63 = loc("res"(#loc26))
#loc64 = loc("shift_1d"(#loc27))
#loc65 = loc("a"(#loc29))
#loc66 = loc("a"(#loc30))
#loc67 = loc("b"(#loc31))
#loc68 = loc("b"(#loc32))
#loc69 = loc("a"(#loc33))
#loc70 = loc("b"(#loc34))
#loc71 = loc("acc"(#loc35))
#loc72 = loc("acc"(#loc37))
#loc73 = loc(callsite(#loc44 at #loc3))
#loc74 = loc(callsite(#loc45 at #loc3))
#loc75 = loc(callsite(#loc47 at #loc48))
#loc76 = loc(callsite(#loc50 at #loc51))
#loc77 = loc(callsite(#loc52 at #loc51))
#loc78 = loc(callsite(#loc53 at #loc51))
#loc79 = loc(callsite(#loc54 at #loc51))
#loc80 = loc(callsite(#loc55 at #loc51))
#loc81 = loc(callsite(#loc18 at #loc51))
#loc82 = loc(callsite(#loc59 at #loc48))
#loc83 = loc(callsite(#loc60 at #loc48))
#loc84 = loc(callsite(#loc44 at #loc23))
#loc85 = loc(callsite(#loc62 at #loc23))
#loc86 = loc(callsite(#loc64 at #loc23))
#loc87 = loc(callsite(#loc44 at #loc28))
#loc88 = loc(callsite(#loc62 at #loc28))
#loc89 = loc(callsite(#loc64 at #loc28))
#loc90 = loc(callsite(#loc62 at #loc3))
#loc91 = loc(callsite(#loc64 at #loc3))
#loc92 = loc(callsite(#loc4 at #loc74))
#loc93 = loc(callsite(#loc46 at #loc75))
#loc94 = loc(callsite(#loc56 at #loc81))
#loc95 = loc(callsite(#loc57 at #loc81))
#loc96 = loc(callsite(#loc58 at #loc81))
#loc97 = loc(callsite(#loc24 at #loc85))
#loc98 = loc(callsite(#loc63 at #loc86))
#loc99 = loc(callsite(#loc24 at #loc88))
#loc100 = loc(callsite(#loc63 at #loc89))
#loc101 = loc(callsite(#loc24 at #loc90))
#loc102 = loc(callsite(#loc63 at #loc91))
#loc103 = loc(callsite(#loc46 at #loc94))

// --- IR After Pipelining
#blocked = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc1 = loc("/home/nico/triton/sandbox/matmul.py":30:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
#loc37 = loc("a_ptr"(#loc1))
#loc38 = loc("b_ptr"(#loc1))
#loc39 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c512_i32 = arith.constant 512 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c4194304_i32 = arith.constant 4194304 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst_0 = arith.constant dense<32768> : tensor<128x1xi32, #blocked> loc(#loc)
    %c131072_i32 = arith.constant 131072 : i32 loc(#loc)
    %cst_1 = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc)
    %c262144_i32 = arith.constant 262144 : i32 loc(#loc)
    %cst_2 = arith.constant dense<2048> : tensor<128x1xi32, #blocked> loc(#loc)
    %pid0 = tt.get_program_id x : i32 loc(#loc66)
    %pid1 = tt.get_program_id y : i32 loc(#loc67)
    %pid2 = tt.get_program_id z : i32 loc(#loc68)
    %npg1 = tt.get_num_programs y : i32 loc(#loc69)
    %npg2 = tt.get_num_programs z : i32 loc(#loc70)
    %stride_val = arith.muli %npg2, %npg1 : i32 loc(#loc92)
    %linear_idx = arith.muli %pid0, %stride_val : i32 loc(#loc84)
    %linear_idx_3 = arith.muli %pid1, %npg2 : i32 loc(#loc84)
    %linear_idx_4 = arith.addi %linear_idx, %linear_idx_3 : i32 loc(#loc85)
    %linear_idx_5 = arith.addi %linear_idx_4, %pid2 : i32 loc(#loc85)
    %idx_val = arith.divsi %linear_idx_5, %c32_i32 : i32 loc(#loc72)
    %remaining = arith.remsi %linear_idx_5, %c32_i32 : i32 loc(#loc73)
    %acc = scf.for %k = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%acc_15 = %cst) -> (tensor<128x64xf32, #mma>)  : i32 {
      %base_offsets_nd_16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc74)
      %base_offsets_nd_17 = tt.expand_dims %base_offsets_nd_16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc74)
      %base_offsets_nd_18 = arith.muli %base_offsets_nd_17, %cst_0 : tensor<128x1xi32, #blocked> loc(#loc74)
      %base_offsets_nd_19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc74)
      %base_offsets_nd_20 = tt.expand_dims %base_offsets_nd_19 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc74)
      %base_offsets_nd_21 = tt.broadcast %base_offsets_nd_18 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc74)
      %base_offsets_nd_22 = tt.broadcast %base_offsets_nd_20 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc74)
      %base_offsets_nd_23 = arith.addi %base_offsets_nd_21, %base_offsets_nd_22 : tensor<128x64xi32, #blocked> loc(#loc74)
      %shift_1d_24 = arith.muli %idx_val, %c4194304_i32 : i32 loc(#loc86)
      %shift_1d_25 = arith.muli %k, %c64_i32 : i32 loc(#loc86)
      %res_26 = arith.addi %shift_1d_24, %shift_1d_25 : i32 loc(#loc87)
      %base_offsets_nd_27 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc77)
      %base_offsets_nd_28 = tt.expand_dims %base_offsets_nd_27 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc77)
      %base_offsets_nd_29 = arith.muli %base_offsets_nd_28, %cst_1 : tensor<64x1xi32, #blocked> loc(#loc77)
      %base_offsets_nd_30 = tt.broadcast %base_offsets_nd_29 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc77)
      %base_offsets_nd_31 = tt.broadcast %base_offsets_nd_20 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc77)
      %base_offsets_nd_32 = arith.addi %base_offsets_nd_30, %base_offsets_nd_31 : tensor<64x64xi32, #blocked> loc(#loc77)
      %shift_1d_33 = arith.muli %k, %c131072_i32 : i32 loc(#loc88)
      %shift_1d_34 = arith.muli %remaining, %c64_i32 : i32 loc(#loc88)
      %res_35 = arith.addi %shift_1d_33, %shift_1d_34 : i32 loc(#loc89)
      %a = tt.addptr %a_ptr, %res_26 : !tt.ptr<f32>, i32 loc(#loc58)
      %a_36 = amdgpu.buffer_load %a[%base_offsets_nd_23] : tensor<128x64xf32, #blocked> loc(#loc59)
      %b = tt.addptr %b_ptr, %res_35 : !tt.ptr<f32>, i32 loc(#loc60)
      %b_37 = amdgpu.buffer_load %b[%base_offsets_nd_32] : tensor<64x64xf32, #blocked> loc(#loc61)
      %a_38 = ttg.convert_layout %a_36 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc62)
      %b_39 = ttg.convert_layout %b_37 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc63)
      %acc_40 = tt.dot %a_38, %b_39, %acc_15 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma> loc(#loc64)
      scf.yield %acc_40 : tensor<128x64xf32, #mma> loc(#loc31)
    } loc(#loc53)
    %acc_6 = ttg.convert_layout %acc : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked> loc(#loc65)
    %base_offsets_nd = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc80)
    %base_offsets_nd_7 = tt.expand_dims %base_offsets_nd {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc80)
    %base_offsets_nd_8 = arith.muli %base_offsets_nd_7, %cst_2 : tensor<128x1xi32, #blocked> loc(#loc80)
    %base_offsets_nd_9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc80)
    %base_offsets_nd_10 = tt.expand_dims %base_offsets_nd_9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_11 = tt.broadcast %base_offsets_nd_8 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_12 = tt.broadcast %base_offsets_nd_10 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_13 = arith.addi %base_offsets_nd_11, %base_offsets_nd_12 : tensor<128x64xi32, #blocked> loc(#loc80)
    %shift_1d = arith.muli %idx_val, %c262144_i32 : i32 loc(#loc90)
    %shift_1d_14 = arith.muli %remaining, %c64_i32 : i32 loc(#loc90)
    %res = arith.addi %shift_1d, %shift_1d_14 : i32 loc(#loc91)
    %0 = tt.addptr %c_ptr, %res : !tt.ptr<f32>, i32 loc(#loc34)
    amdgpu.buffer_store %acc_6, %0[%base_offsets_nd_13] : tensor<128x64xf32, #blocked> loc(#loc35)
    tt.return loc(#loc36)
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
#loc23 = loc("/home/nico/triton/sandbox/matmul.py":56:91)
#loc24 = loc("/home/nico/triton/sandbox/matmul.py":59:49)
#loc25 = loc("/home/nico/triton/sandbox/matmul.py":59:74)
#loc26 = loc("/home/nico/triton/sandbox/matmul.py":60:49)
#loc27 = loc("/home/nico/triton/sandbox/matmul.py":60:74)
#loc28 = loc("/home/nico/triton/sandbox/matmul.py":73:37)
#loc29 = loc("/home/nico/triton/sandbox/matmul.py":74:37)
#loc30 = loc("/home/nico/triton/sandbox/matmul.py":75:42)
#loc31 = loc("/home/nico/triton/sandbox/matmul.py":66:8)
#loc32 = loc("/home/nico/triton/sandbox/matmul.py":79:33)
#loc33 = loc("/home/nico/triton/sandbox/matmul.py":82:91)
#loc34 = loc("/home/nico/triton/sandbox/matmul.py":86:47)
#loc35 = loc("/home/nico/triton/sandbox/matmul.py":86:72)
#loc36 = loc("/home/nico/triton/sandbox/matmul.py":90:4)
#loc40 = loc("pid0"(#loc2))
#loc41 = loc("linear_program_id"(#loc3))
#loc42 = loc("pid1"(#loc4))
#loc43 = loc("pid2"(#loc5))
#loc44 = loc("npg1"(#loc6))
#loc45 = loc("npg2"(#loc7))
#loc46 = loc("stride_val"(#loc8))
#loc47 = loc("strides"(#loc9))
#loc48 = loc("linear_idx"(#loc11))
#loc49 = loc("linear_idx"(#loc12))
#loc50 = loc("idx_val"(#loc13))
#loc51 = loc("start_blocks_c"(#loc14))
#loc52 = loc("remaining"(#loc15))
#loc53 = loc("acc"(#loc16))
#loc54 = loc("base_offsets_nd"(#loc17))
#loc55 = loc("shift_1d"(#loc20))
#loc56 = loc("res"(#loc21))
#loc57 = loc("shift_1d"(#loc22))
#loc58 = loc("a"(#loc24))
#loc59 = loc("a"(#loc25))
#loc60 = loc("b"(#loc26))
#loc61 = loc("b"(#loc27))
#loc62 = loc("a"(#loc28))
#loc63 = loc("b"(#loc29))
#loc64 = loc("acc"(#loc30))
#loc65 = loc("acc"(#loc32))
#loc66 = loc(callsite(#loc40 at #loc41))
#loc67 = loc(callsite(#loc42 at #loc41))
#loc68 = loc(callsite(#loc43 at #loc41))
#loc69 = loc(callsite(#loc44 at #loc41))
#loc70 = loc(callsite(#loc45 at #loc41))
#loc71 = loc(callsite(#loc10 at #loc41))
#loc72 = loc(callsite(#loc50 at #loc51))
#loc73 = loc(callsite(#loc52 at #loc51))
#loc74 = loc(callsite(#loc54 at #loc18))
#loc75 = loc(callsite(#loc55 at #loc18))
#loc76 = loc(callsite(#loc57 at #loc18))
#loc77 = loc(callsite(#loc54 at #loc23))
#loc78 = loc(callsite(#loc55 at #loc23))
#loc79 = loc(callsite(#loc57 at #loc23))
#loc80 = loc(callsite(#loc54 at #loc33))
#loc81 = loc(callsite(#loc55 at #loc33))
#loc82 = loc(callsite(#loc57 at #loc33))
#loc83 = loc(callsite(#loc47 at #loc71))
#loc84 = loc(callsite(#loc48 at #loc71))
#loc85 = loc(callsite(#loc49 at #loc71))
#loc86 = loc(callsite(#loc19 at #loc75))
#loc87 = loc(callsite(#loc56 at #loc76))
#loc88 = loc(callsite(#loc19 at #loc78))
#loc89 = loc(callsite(#loc56 at #loc79))
#loc90 = loc(callsite(#loc19 at #loc81))
#loc91 = loc(callsite(#loc56 at #loc82))
#loc92 = loc(callsite(#loc46 at #loc83))


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
	v_and_b32_e32 v2, 48, v0
	v_lshlrev_b32_e32 v19, 4, v0
	v_or_b32_e32 v3, 1, v2
	v_and_b32_e32 v1, 0xf0, v19
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
	s_mul_i32 s0, s1, s0
	s_add_i32 s0, s0, s17
.Ltmp2:
	.loc	2 73 31                         ; tuple_helpers.py:73:31 @[ matmul.py:41:52 ]
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	s_add_i32 s1, s0, s1
	s_ashr_i32 s14, s1, 5
	.loc	2 75 32                         ; tuple_helpers.py:75:32 @[ matmul.py:41:52 ]
	s_and_b32 s1, s1, 0x3ffffe0
	s_sub_i32 s1, s0, s1
	v_or_b32_e32 v4, 2, v2
	v_lshl_or_b32 v147, v3, 17, v1
	v_lshl_or_b32 v165, v3, 13, v1
	v_and_b32_e32 v3, 1, v0
	v_or_b32_e32 v5, 3, v2
	v_or_b32_e32 v6, 4, v2
	v_or_b32_e32 v7, 5, v2
	v_or_b32_e32 v8, 6, v2
	v_or_b32_e32 v9, 7, v2
	v_or_b32_e32 v10, 8, v2
	v_or_b32_e32 v11, 9, v2
	v_or_b32_e32 v12, 10, v2
	v_or_b32_e32 v13, 11, v2
	v_or_b32_e32 v14, 12, v2
	v_or_b32_e32 v15, 13, v2
	v_or_b32_e32 v16, 14, v2
	v_or_b32_e32 v17, 15, v2
	s_lshl_b32 s15, s1, 6
	v_lshl_or_b32 v146, v2, 17, v1
	v_lshl_or_b32 v148, v4, 17, v1
	v_lshl_or_b32 v164, v2, 13, v1
	v_lshl_or_b32 v166, v4, 13, v1
	v_lshlrev_b32_e32 v2, 3, v0
	v_lshlrev_b32_e32 v4, 12, v3
	s_movk_i32 s1, 0x1f0
	v_lshlrev_b32_e32 v18, 2, v0
	v_and_or_b32 v181, v2, s1, v4
	v_lshlrev_b32_e32 v180, 8, v0
	v_and_b32_e32 v162, 0x330, v19
	v_lshl_or_b32 v19, v0, 9, v2
	s_movk_i32 s1, 0x840
	s_lshl_b32 s0, s14, 22
	v_lshl_or_b32 v156, v12, 17, v1
	v_lshl_or_b32 v157, v13, 17, v1
	v_lshl_or_b32 v174, v12, 13, v1
	v_lshl_or_b32 v175, v13, 13, v1
	v_or_b32_e32 v12, v180, v2
	v_lshlrev_b32_e32 v13, 7, v0
	v_and_or_b32 v185, v19, s1, v162
	v_and_b32_e32 v183, 0xb0, v18
	v_bfe_i32 v18, v0, 1, 1
	s_movk_i32 s1, 0x800
	v_lshl_or_b32 v158, v14, 17, v1
	v_lshl_or_b32 v176, v14, 13, v1
	v_lshlrev_b32_e32 v3, 13, v3
	v_and_b32_e32 v14, 0x1000, v13
	v_and_b32_e32 v12, 0xef0, v12
	v_and_b32_e32 v184, 0x440, v18
	v_and_or_b32 v13, v13, s1, v183
.Ltmp3:
	.loc	1 50 25                         ; matmul.py:50:25
	s_ashr_i32 s1, s0, 31
	v_or3_b32 v182, v3, v14, v12
	v_or3_b32 v186, v13, v184, v4
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshl_or_b32 v149, v5, 17, v1
	v_lshl_or_b32 v150, v6, 17, v1
	v_lshl_or_b32 v151, v7, 17, v1
	v_lshl_or_b32 v152, v8, 17, v1
	v_lshl_or_b32 v153, v9, 17, v1
	v_lshl_or_b32 v154, v10, 17, v1
	v_lshl_or_b32 v155, v11, 17, v1
	v_lshl_or_b32 v159, v15, 17, v1
	v_lshl_or_b32 v160, v16, 17, v1
	v_lshl_or_b32 v161, v17, 17, v1
	v_lshl_or_b32 v167, v5, 13, v1
	v_lshl_or_b32 v168, v6, 13, v1
	v_lshl_or_b32 v169, v7, 13, v1
	v_lshl_or_b32 v170, v8, 13, v1
	v_lshl_or_b32 v171, v9, 13, v1
	v_lshl_or_b32 v172, v10, 13, v1
	v_lshl_or_b32 v173, v11, 13, v1
	v_lshl_or_b32 v177, v15, 13, v1
	v_lshl_or_b32 v178, v16, 13, v1
	v_lshl_or_b32 v179, v17, 13, v1
	v_xor_b32_e32 v5, 16, v181
	v_xor_b32_e32 v6, 32, v181
	v_xor_b32_e32 v7, 48, v181
	v_xor_b32_e32 v8, 64, v181
	v_xor_b32_e32 v9, 0x50, v181
	v_xor_b32_e32 v10, 0x60, v181
	v_xor_b32_e32 v11, 0x70, v181
	v_xor_b32_e32 v3, 16, v182
	v_xor_b32_e32 v12, 32, v182
	v_xor_b32_e32 v14, 48, v182
	v_xor_b32_e32 v15, 64, v182
	v_xor_b32_e32 v16, 0x50, v182
	v_xor_b32_e32 v17, 0x60, v182
	v_xor_b32_e32 v20, 0x70, v182
	v_and_b32_e32 v163, 64, v2
	v_xor_b32_e32 v2, 64, v185
	v_xor_b32_e32 v4, 64, v186
	s_add_u32 s16, s2, s0
	s_addc_u32 s17, s3, s1
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
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	s_mov_b64 s[8:9], 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_u32_e32 v187, 0, v5
	v_add_u32_e32 v188, 0, v6
	v_add_u32_e32 v189, 0, v7
	v_add_u32_e32 v190, 0, v8
	v_add_u32_e32 v191, 0, v9
	v_add_u32_e32 v192, 0, v10
	v_add_u32_e32 v193, 0, v11
	v_add_u32_e32 v194, 0, v3
	v_add_u32_e32 v195, 0, v12
	v_add_u32_e32 v196, 0, v14
	v_add_u32_e32 v197, 0, v15
	v_add_u32_e32 v198, 0, v16
	v_add_u32_e32 v199, 0, v17
	v_add_u32_e32 v200, 0, v20
	v_add_u32_e32 v201, 0, v2
	v_add_u32_e32 v202, 0, v4
	s_mov_b32 s10, s15
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 59 74                         ; matmul.py:59:74
	s_add_u32 s0, s16, s8
	s_addc_u32 s1, s17, s9
	s_and_b32 s1, s1, 0xffff
	v_or_b32_e32 v66, 0x800000, v146
	buffer_load_dwordx4 v[2:5], v146, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v147, s[0:3], 0 offen
	buffer_load_dwordx4 v[10:13], v148, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v149, s[0:3], 0 offen
	buffer_load_dwordx4 v[18:21], v150, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v151, s[0:3], 0 offen
	buffer_load_dwordx4 v[26:29], v152, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v153, s[0:3], 0 offen
	buffer_load_dwordx4 v[34:37], v154, s[0:3], 0 offen
	buffer_load_dwordx4 v[38:41], v155, s[0:3], 0 offen
	buffer_load_dwordx4 v[42:45], v156, s[0:3], 0 offen
	buffer_load_dwordx4 v[46:49], v157, s[0:3], 0 offen
	buffer_load_dwordx4 v[50:53], v158, s[0:3], 0 offen
	buffer_load_dwordx4 v[54:57], v159, s[0:3], 0 offen
	buffer_load_dwordx4 v[58:61], v160, s[0:3], 0 offen
	buffer_load_dwordx4 v[62:65], v161, s[0:3], 0 offen
	buffer_load_dwordx4 v[74:77], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x820000, v146
	buffer_load_dwordx4 v[78:81], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x840000, v146
	buffer_load_dwordx4 v[90:93], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x860000, v146
	buffer_load_dwordx4 v[94:97], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x880000, v146
	buffer_load_dwordx4 v[106:109], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x8a0000, v146
	buffer_load_dwordx4 v[110:113], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x8c0000, v146
	buffer_load_dwordx4 v[114:117], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x8e0000, v146
	buffer_load_dwordx4 v[118:121], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x900000, v146
	buffer_load_dwordx4 v[130:133], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x920000, v146
	buffer_load_dwordx4 v[134:137], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x940000, v146
	buffer_load_dwordx4 v[138:141], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x960000, v146
	buffer_load_dwordx4 v[142:145], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x980000, v146
	buffer_load_dwordx4 v[238:241], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x9a0000, v146
	.loc	1 60 49                         ; matmul.py:60:49
	s_ashr_i32 s11, s10, 31
	.loc	1 59 74                         ; matmul.py:59:74
	buffer_load_dwordx4 v[242:245], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x9c0000, v146
	.loc	1 60 49                         ; matmul.py:60:49
	s_lshl_b64 s[12:13], s[10:11], 2
	.loc	1 59 74                         ; matmul.py:59:74
	buffer_load_dwordx4 v[246:249], v66, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x9e0000, v146
	buffer_load_dwordx4 v[250:253], v66, s[0:3], 0 offen
	.loc	1 60 49                         ; matmul.py:60:49
	s_add_u32 s0, s4, s12
	s_addc_u32 s1, s5, s13
	.loc	1 60 74 is_stmt 0               ; matmul.py:60:74
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 a[128:131], v164, s[0:3], 0 offen
	buffer_load_dwordx4 a[132:135], v165, s[0:3], 0 offen
	buffer_load_dwordx4 a[136:139], v166, s[0:3], 0 offen
	buffer_load_dwordx4 a[140:143], v167, s[0:3], 0 offen
	buffer_load_dwordx4 a[144:147], v168, s[0:3], 0 offen
	buffer_load_dwordx4 a[148:151], v169, s[0:3], 0 offen
	buffer_load_dwordx4 a[152:155], v170, s[0:3], 0 offen
	buffer_load_dwordx4 a[156:159], v171, s[0:3], 0 offen
	buffer_load_dwordx4 v[226:229], v172, s[0:3], 0 offen
	buffer_load_dwordx4 v[230:233], v173, s[0:3], 0 offen
	buffer_load_dwordx4 v[234:237], v174, s[0:3], 0 offen
	buffer_load_dwordx4 v[206:209], v175, s[0:3], 0 offen
	buffer_load_dwordx4 v[214:217], v176, s[0:3], 0 offen
	buffer_load_dwordx4 v[218:221], v177, s[0:3], 0 offen
	buffer_load_dwordx4 v[222:225], v178, s[0:3], 0 offen
	buffer_load_dwordx4 v[210:213], v179, s[0:3], 0 offen
	.loc	1 73 37 is_stmt 1               ; matmul.py:73:37
	v_add_u32_e32 v205, 0, v181
	v_add_u32_e32 v254, 0, v182
	.loc	1 74 37                         ; matmul.py:74:37
	v_add_u32_e32 v204, 0, v185
	v_add_u32_e32 v203, 0, v186
	; wave barrier
	.loc	1 50 25                         ; matmul.py:50:25
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s9, s9, 0
	s_add_i32 s10, s10, 0x20000
	s_cmp_lg_u32 s8, 0x20000
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(47)
	ds_write_b128 v205, v[2:5]
	s_waitcnt vmcnt(46)
	ds_write_b128 v205, v[6:9] offset:8192
	s_waitcnt vmcnt(45)
	ds_write_b128 v187, v[10:13] offset:512
	s_waitcnt vmcnt(44)
	ds_write_b128 v187, v[14:17] offset:8704
	s_waitcnt vmcnt(43)
	ds_write_b128 v188, v[18:21] offset:1024
	s_waitcnt vmcnt(42)
	ds_write_b128 v188, v[22:25] offset:9216
	s_waitcnt vmcnt(41)
	ds_write_b128 v189, v[26:29] offset:1536
	s_waitcnt vmcnt(40)
	ds_write_b128 v189, v[30:33] offset:9728
	s_waitcnt vmcnt(39)
	ds_write_b128 v190, v[34:37] offset:2048
	s_waitcnt vmcnt(38)
	ds_write_b128 v190, v[38:41] offset:10240
	s_waitcnt vmcnt(37)
	ds_write_b128 v191, v[42:45] offset:2560
	s_waitcnt vmcnt(36)
	ds_write_b128 v191, v[46:49] offset:10752
	s_waitcnt vmcnt(35)
	ds_write_b128 v192, v[50:53] offset:3072
	s_waitcnt vmcnt(34)
	ds_write_b128 v192, v[54:57] offset:11264
	s_waitcnt vmcnt(33)
	ds_write_b128 v193, v[58:61] offset:3584
	s_waitcnt vmcnt(32)
	ds_write_b128 v193, v[62:65] offset:11776
	; wave barrier
	ds_read_b128 v[126:129], v254
	ds_read_b128 v[122:125], v254 offset:256
	ds_read_b128 v[102:105], v194
	ds_read_b128 v[98:101], v194 offset:256
	ds_read_b128 v[86:89], v195
	ds_read_b128 v[82:85], v195 offset:256
	ds_read_b128 v[70:73], v196
	ds_read_b128 v[66:69], v196 offset:256
	ds_read_b128 v[54:57], v197
	ds_read_b128 v[50:53], v197 offset:256
	ds_read_b128 v[38:41], v198
	ds_read_b128 v[34:37], v198 offset:256
	ds_read_b128 v[22:25], v199
	ds_read_b128 v[18:21], v199 offset:256
	ds_read_b128 v[6:9], v200
	ds_read_b128 v[2:5], v200 offset:256
	; wave barrier
	s_waitcnt vmcnt(31)
	ds_write_b128 v205, v[74:77]
	s_waitcnt vmcnt(30)
	ds_write_b128 v205, v[78:81] offset:8192
	s_waitcnt vmcnt(29)
	ds_write_b128 v187, v[90:93] offset:512
	s_waitcnt vmcnt(28)
	ds_write_b128 v187, v[94:97] offset:8704
	s_waitcnt vmcnt(27)
	ds_write_b128 v188, v[106:109] offset:1024
	s_waitcnt vmcnt(26)
	ds_write_b128 v188, v[110:113] offset:9216
	s_waitcnt vmcnt(25)
	ds_write_b128 v189, v[114:117] offset:1536
	s_waitcnt vmcnt(24)
	ds_write_b128 v189, v[118:121] offset:9728
	s_waitcnt vmcnt(23)
	ds_write_b128 v190, v[130:133] offset:2048
	s_waitcnt vmcnt(22)
	ds_write_b128 v190, v[134:137] offset:10240
	s_waitcnt vmcnt(21)
	ds_write_b128 v191, v[138:141] offset:2560
	s_waitcnt vmcnt(20)
	ds_write_b128 v191, v[142:145] offset:10752
	s_waitcnt vmcnt(19)
	ds_write_b128 v192, v[238:241] offset:3072
	s_waitcnt vmcnt(18)
	ds_write_b128 v192, v[242:245] offset:11264
	s_waitcnt vmcnt(17)
	ds_write_b128 v193, v[246:249] offset:3584
	s_waitcnt vmcnt(16)
	ds_write_b128 v193, v[250:253] offset:11776
	; wave barrier
	ds_read_b128 v[130:133], v254
	ds_read_b128 v[118:121], v254 offset:256
	ds_read_b128 v[110:113], v194
	ds_read_b128 v[106:109], v194 offset:256
	ds_read_b128 v[94:97], v195
	ds_read_b128 v[90:93], v195 offset:256
	ds_read_b128 v[78:81], v196
	ds_read_b128 v[74:77], v196 offset:256
	ds_read_b128 v[62:65], v197
	ds_read_b128 v[58:61], v197 offset:256
	ds_read_b128 v[46:49], v198
	ds_read_b128 v[42:45], v198 offset:256
	ds_read_b128 v[30:33], v199
	ds_read_b128 v[26:29], v199 offset:256
	ds_read_b128 v[14:17], v200
	ds_read_b128 v[10:13], v200 offset:256
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt vmcnt(15) lgkmcnt(0)
	; wave barrier
	v_accvgpr_read_b32 v114, a128
	s_waitcnt vmcnt(14)
	v_accvgpr_read_b32 v115, a132
	s_waitcnt vmcnt(13)
	v_accvgpr_read_b32 v116, a136
	s_waitcnt vmcnt(12)
	v_accvgpr_read_b32 v117, a140
	v_accvgpr_read_b32 v134, a129
	v_accvgpr_read_b32 v135, a133
	v_accvgpr_read_b32 v136, a137
	v_accvgpr_read_b32 v137, a141
	s_waitcnt vmcnt(11)
	v_accvgpr_read_b32 v138, a144
	s_waitcnt vmcnt(10)
	v_accvgpr_read_b32 v139, a148
	s_waitcnt vmcnt(9)
	v_accvgpr_read_b32 v140, a152
	s_waitcnt vmcnt(8)
	v_accvgpr_read_b32 v141, a156
	v_accvgpr_read_b32 v142, a145
	v_accvgpr_read_b32 v143, a149
	v_accvgpr_read_b32 v144, a153
	v_accvgpr_read_b32 v145, a157
	v_accvgpr_read_b32 v238, a130
	v_accvgpr_read_b32 v239, a134
	v_accvgpr_read_b32 v240, a138
	v_accvgpr_read_b32 v241, a142
	v_accvgpr_mov_b32 a140, a131
	v_accvgpr_mov_b32 a141, a135
	v_accvgpr_mov_b32 a142, a139
	v_accvgpr_read_b32 v242, a146
	v_accvgpr_read_b32 v243, a150
	v_accvgpr_read_b32 v244, a154
	v_accvgpr_read_b32 v245, a158
	v_accvgpr_mov_b32 a156, a147
	v_accvgpr_mov_b32 a157, a151
	v_accvgpr_mov_b32 a158, a155
	ds_write_b128 v204, v[114:117]
	ds_write_b128 v204, v[134:137] offset:4096
	ds_write_b128 v204, v[138:141] offset:128
	ds_write_b128 v204, v[142:145] offset:4224
	ds_write_b128 v201, v[238:241] offset:1024
	ds_write_b128 v201, a[140:143] offset:5120
	ds_write_b128 v201, v[242:245] offset:1152
	ds_write_b128 v201, a[156:159] offset:5248
	; wave barrier
	ds_read_b128 v[138:141], v203
	ds_read_b128 v[114:117], v203 offset:256
	ds_read_b128 v[142:145], v202
	ds_read_b128 v[134:137], v202 offset:256
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[112:127], v126, v138, a[112:127]
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt vmcnt(3)
	v_mov_b32_e32 v238, v215
	s_waitcnt vmcnt(2)
	v_mov_b32_e32 v239, v219
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v240, v223
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v241, v211
	v_mov_b32_e32 v211, v221
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[96:111], v126, v142, a[96:111]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v126, v227
	v_mov_b32_e32 v227, v220
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[80:95], v122, v138, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v122, v142, a[64:79]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v122, v226
	v_mov_b32_e32 v226, v216
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v130, v138, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v130, v142, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v130, v214
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v118, v138, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v138, v228
	v_mov_b32_e32 v228, v224
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v118, v142, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v127, v139, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v127, v143, a[96:111]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v127, v231
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[80:95], v123, v139, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v123, v143, a[64:79]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v123, v230
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v131, v139, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v131, v143, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v131, v218
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v119, v139, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v139, v232
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v119, v143, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v128, v140, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v128, v144, a[96:111]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v128, v235
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[80:95], v124, v140, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v124, v144, a[64:79]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v124, v234
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v132, v140, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v132, v144, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v132, v222
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v120, v140, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v140, v236
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v120, v144, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v129, v141, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v129, v145, a[96:111]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v129, v207
	v_mov_b32_e32 v207, v233
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[80:95], v125, v141, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v125, v145, a[64:79]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v125, v206
	v_mov_b32_e32 v206, v229
	v_mov_b32_e32 v229, v212
	v_mov_b32_e32 v212, v225
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v133, v141, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v133, v145, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v133, v210
	v_mov_b32_e32 v210, v217
	ds_read_b128 v[214:217], v203 offset:512
	ds_read_b128 v[218:221], v203 offset:768
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v121, v141, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v141, v208
	v_mov_b32_e32 v208, v237
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v121, v145, a[0:15]
	.loc	1 74 37                         ; matmul.py:74:37
	ds_read_b128 v[118:121], v202 offset:512
	ds_read_b128 v[142:145], v202 offset:768
	; wave barrier
	ds_write_b128 v204, v[122:125]
	ds_write_b128 v204, v[126:129] offset:4096
	ds_write_b128 v204, v[130:133] offset:128
	ds_write_b128 v204, v[238:241] offset:4224
	ds_write_b128 v201, v[138:141] offset:1024
	ds_write_b128 v201, v[206:209] offset:5120
	ds_write_b128 v201, v[226:229] offset:1152
	ds_write_b128 v201, v[210:213] offset:5248
	; wave barrier
	ds_read_b128 v[122:125], v203
	ds_read_b128 v[126:129], v203 offset:256
	ds_read_b128 v[130:133], v202
	ds_read_b128 v[138:141], v202 offset:256
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[112:127], v102, v122, a[112:127]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[96:111], v102, v130, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v98, v122, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v98, v130, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v110, v122, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v110, v130, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v106, v122, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v106, v130, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v103, v123, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v103, v131, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v99, v123, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v99, v131, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v111, v123, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v111, v131, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v107, v123, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v107, v131, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v104, v124, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v104, v132, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v100, v124, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v100, v132, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v112, v124, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v112, v132, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v108, v124, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v108, v132, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v105, v125, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v105, v133, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v101, v125, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v101, v133, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v113, v125, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v113, v133, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v109, v125, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v109, v133, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v86, v114, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v86, v134, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v82, v114, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v82, v134, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v94, v114, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v94, v134, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v90, v114, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v90, v134, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v87, v115, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v87, v135, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v83, v115, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v83, v135, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v95, v115, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v95, v135, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v91, v115, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v91, v135, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v88, v116, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v88, v136, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v84, v116, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v84, v136, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v96, v116, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v96, v136, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v92, v116, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v92, v136, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v89, v117, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v89, v137, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v85, v117, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v85, v137, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v97, v117, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v97, v137, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v93, v117, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v93, v137, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v70, v126, a[112:127]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x2_f32 a[96:111], v70, v138, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v66, v126, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v66, v138, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v78, v126, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v78, v138, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v74, v126, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v74, v138, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v71, v127, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v71, v139, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v67, v127, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v67, v139, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v79, v127, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v79, v139, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v75, v127, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v75, v139, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v72, v128, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v72, v140, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v68, v128, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v68, v140, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v80, v128, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v80, v140, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v76, v128, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v76, v140, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v73, v129, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v73, v141, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v69, v129, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v69, v141, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v81, v129, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v81, v141, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v77, v129, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v77, v141, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v54, v214, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v54, v118, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v50, v214, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v50, v118, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v62, v214, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v62, v118, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v58, v214, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v58, v118, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v55, v215, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v55, v119, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v51, v215, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v51, v119, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v63, v215, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v63, v119, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v59, v215, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v59, v119, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v56, v216, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v56, v120, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v52, v216, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v52, v120, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v64, v216, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v64, v120, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v60, v216, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v60, v120, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v57, v217, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v57, v121, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v53, v217, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v53, v121, a[64:79]
	.loc	1 74 37                         ; matmul.py:74:37
	ds_read_b128 v[50:53], v203 offset:512
	ds_read_b128 v[54:57], v203 offset:768
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v65, v217, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v65, v121, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v61, v217, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v61, v121, a[0:15]
	.loc	1 74 37                         ; matmul.py:74:37
	ds_read_b128 v[58:61], v202 offset:512
	ds_read_b128 v[62:65], v202 offset:768
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[112:127], v38, v50, a[112:127]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[96:111], v38, v58, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v34, v50, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v34, v58, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v46, v50, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v46, v58, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v42, v50, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v42, v58, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v39, v51, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v39, v59, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v35, v51, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v35, v59, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v47, v51, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v47, v59, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v43, v51, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v43, v59, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v40, v52, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v40, v60, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v36, v52, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v36, v60, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v48, v52, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v48, v60, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v44, v52, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v44, v60, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v41, v53, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v41, v61, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v37, v53, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v37, v61, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v49, v53, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v49, v61, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v45, v53, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v45, v61, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v22, v218, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v22, v142, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v18, v218, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v18, v142, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v30, v218, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v30, v142, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v26, v218, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v26, v142, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v23, v219, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v23, v143, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v19, v219, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v19, v143, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v31, v219, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v31, v143, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v27, v219, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v27, v143, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v24, v220, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v24, v144, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v20, v220, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v20, v144, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v32, v220, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v32, v144, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v28, v220, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v28, v144, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v25, v221, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v25, v145, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v21, v221, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v21, v145, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v33, v221, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v33, v145, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v29, v221, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v29, v145, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v6, v54, a[112:127]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x2_f32 a[96:111], v6, v62, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v2, v54, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v2, v62, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v14, v54, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v14, v62, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v10, v54, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v10, v62, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v7, v55, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v7, v63, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v3, v55, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v3, v63, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v15, v55, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v15, v63, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v11, v55, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v11, v63, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v8, v56, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v8, v64, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v4, v56, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v4, v64, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v16, v56, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v16, v64, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v12, v56, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v12, v64, a[0:15]
	v_mfma_f32_32x32x2_f32 a[112:127], v9, v57, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v9, v65, a[96:111]
	v_mfma_f32_32x32x2_f32 a[80:95], v5, v57, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v5, v65, a[64:79]
	v_mfma_f32_32x32x2_f32 a[48:63], v17, v57, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v17, v65, a[32:47]
	v_mfma_f32_32x32x2_f32 a[16:31], v13, v57, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v13, v65, a[0:15]
	.loc	1 50 25                         ; matmul.py:50:25
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	.loc	1 79 33                         ; matmul.py:79:33
	v_lshlrev_b32_e32 v2, 11, v0
	v_and_b32_e32 v2, 0x800, v2
	s_movk_i32 s0, 0x1000
	v_lshlrev_b32_e32 v4, 10, v0
	v_and_or_b32 v2, v180, s0, v2
	v_and_b32_e32 v4, 0x1000, v4
	v_or3_b32 v2, v2, v183, v184
	v_or3_b32 v4, v4, v163, v162
.Ltmp4:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:82:91 ]
	s_lshl_b32 s0, s14, 18
.Ltmp5:
	.loc	1 79 33                         ; matmul.py:79:33
	v_add_u32_e32 v3, 0, v2
	v_xad_u32 v2, v2, 64, 0
	v_add_u32_e32 v5, 0, v4
	v_xad_u32 v4, v4, 64, 0
.Ltmp6:
	.loc	2 132 15                        ; tuple_helpers.py:132:15 @[ matmul.py:82:91 ]
	s_add_i32 s0, s0, s15
.Ltmp7:
	; wave barrier
	.loc	1 79 33                         ; matmul.py:79:33
	ds_write_b128 v3, a[112:115]
	ds_write_b128 v3, a[120:123] offset:256
	ds_write_b128 v3, a[80:83] offset:512
	ds_write_b128 v3, a[88:91] offset:768
	ds_write_b128 v2, a[96:99]
	ds_write_b128 v2, a[104:107] offset:256
	ds_write_b128 v2, a[64:67] offset:512
	ds_write_b128 v2, a[72:75] offset:768
	; wave barrier
	ds_read_b128 v[6:9], v5
	ds_read_b128 v[10:13], v5 offset:128
	ds_read_b128 v[14:17], v5 offset:2048
	ds_read_b128 v[18:21], v5 offset:2176
	ds_read_b128 v[22:25], v4 offset:1024
	ds_read_b128 v[26:29], v4 offset:1152
	ds_read_b128 v[30:33], v4 offset:3072
	ds_read_b128 v[34:37], v4 offset:3200
	.loc	1 86 47                         ; matmul.py:86:47
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s7, s1
	.loc	1 86 72 is_stmt 0               ; matmul.py:86:72
	v_lshlrev_b32_e32 v0, 13, v0
	s_mov_b32 s4, 0x60000
	; wave barrier
	.loc	1 79 33 is_stmt 1               ; matmul.py:79:33
	ds_write_b128 v3, a[116:119]
	ds_write_b128 v3, a[124:127] offset:256
	ds_write_b128 v3, a[84:87] offset:512
	ds_write_b128 v3, a[92:95] offset:768
	ds_write_b128 v2, a[100:103]
	ds_write_b128 v2, a[108:111] offset:256
	ds_write_b128 v2, a[68:71] offset:512
	ds_write_b128 v2, a[76:79] offset:768
	; wave barrier
	ds_read_b128 v[38:41], v5
	ds_read_b128 v[42:45], v5 offset:128
	ds_read_b128 v[46:49], v5 offset:2048
	ds_read_b128 v[50:53], v5 offset:2176
	ds_read_b128 v[54:57], v4 offset:1024
	ds_read_b128 v[58:61], v4 offset:1152
	ds_read_b128 v[62:65], v4 offset:3072
	ds_read_b128 v[66:69], v4 offset:3200
	; wave barrier
	ds_write_b128 v3, a[48:51]
	ds_write_b128 v3, a[56:59] offset:256
	ds_write_b128 v3, a[16:19] offset:512
	ds_write_b128 v3, a[24:27] offset:768
	ds_write_b128 v2, a[32:35]
	ds_write_b128 v2, a[40:43] offset:256
	ds_write_b128 v2, a[0:3] offset:512
	ds_write_b128 v2, a[8:11] offset:768
	; wave barrier
	ds_read_b128 v[70:73], v5
	ds_read_b128 v[74:77], v5 offset:128
	ds_read_b128 v[78:81], v5 offset:2048
	ds_read_b128 v[82:85], v5 offset:2176
	ds_read_b128 v[86:89], v4 offset:1024
	ds_read_b128 v[90:93], v4 offset:1152
	ds_read_b128 v[94:97], v4 offset:3072
	ds_read_b128 v[98:101], v4 offset:3200
	; wave barrier
	ds_write_b128 v3, a[52:55]
	ds_write_b128 v3, a[60:63] offset:256
	ds_write_b128 v3, a[20:23] offset:512
	ds_write_b128 v3, a[28:31] offset:768
	ds_write_b128 v2, a[36:39]
	ds_write_b128 v2, a[44:47] offset:256
	ds_write_b128 v2, a[4:7] offset:512
	ds_write_b128 v2, a[12:15] offset:768
	; wave barrier
	ds_read_b128 v[102:105], v5
	ds_read_b128 v[106:109], v5 offset:128
	ds_read_b128 v[110:113], v5 offset:2048
	ds_read_b128 v[114:117], v5 offset:2176
	ds_read_b128 v[118:121], v4 offset:1024
	ds_read_b128 v[122:125], v4 offset:1152
	ds_read_b128 v[126:129], v4 offset:3072
	ds_read_b128 v[130:133], v4 offset:3200
	.loc	1 86 72                         ; matmul.py:86:72
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v2, v6
	v_mov_b32_e32 v3, v14
	v_mov_b32_e32 v4, v22
	v_mov_b32_e32 v5, v30
	v_and_or_b32 v6, v0, s4, v1
	buffer_store_dwordx4 v[2:5], v6, s[0:3], 0 offen
	v_mov_b32_e32 v0, v7
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v31
	v_or_b32_e32 v4, 0x2000, v6
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x4000, v6
	v_mov_b32_e32 v30, v9
	v_mov_b32_e32 v0, v8
	v_mov_b32_e32 v1, v16
	v_mov_b32_e32 v2, v24
	v_mov_b32_e32 v3, v32
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v31, v17
	v_mov_b32_e32 v32, v25
	v_or_b32_e32 v0, 0x6000, v6
	buffer_store_dwordx4 v[30:33], v0, s[0:3], 0 offen
	v_mov_b32_e32 v0, v10
	v_mov_b32_e32 v1, v18
	v_mov_b32_e32 v2, v26
	v_mov_b32_e32 v3, v34
	v_or_b32_e32 v4, 0x8000, v6
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0xa000, v6
	v_mov_b32_e32 v34, v13
	v_mov_b32_e32 v0, v11
	v_mov_b32_e32 v1, v19
	v_mov_b32_e32 v2, v27
	v_mov_b32_e32 v3, v35
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0xc000, v6
	v_mov_b32_e32 v35, v21
	v_mov_b32_e32 v0, v12
	v_mov_b32_e32 v1, v20
	v_mov_b32_e32 v2, v28
	v_mov_b32_e32 v3, v36
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v36, v29
	v_or_b32_e32 v4, 0x10000, v6
	v_or_b32_e32 v0, 0xe000, v6
	buffer_store_dwordx4 v[34:37], v0, s[0:3], 0 offen
	v_mov_b32_e32 v0, v38
	v_mov_b32_e32 v1, v46
	v_mov_b32_e32 v2, v54
	v_mov_b32_e32 v3, v62
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x12000, v6
	v_mov_b32_e32 v62, v41
	v_mov_b32_e32 v0, v39
	v_mov_b32_e32 v1, v47
	v_mov_b32_e32 v2, v55
	v_mov_b32_e32 v3, v63
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x14000, v6
	v_mov_b32_e32 v63, v49
	v_mov_b32_e32 v0, v40
	v_mov_b32_e32 v1, v48
	v_mov_b32_e32 v2, v56
	v_mov_b32_e32 v3, v64
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v64, v57
	v_or_b32_e32 v4, 0x18000, v6
	v_or_b32_e32 v0, 0x16000, v6
	buffer_store_dwordx4 v[62:65], v0, s[0:3], 0 offen
	v_mov_b32_e32 v0, v42
	v_mov_b32_e32 v1, v50
	v_mov_b32_e32 v2, v58
	v_mov_b32_e32 v3, v66
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x1a000, v6
	v_mov_b32_e32 v66, v45
	v_mov_b32_e32 v0, v43
	v_mov_b32_e32 v1, v51
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v3, v67
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x1c000, v6
	v_mov_b32_e32 v67, v53
	v_mov_b32_e32 v0, v44
	v_mov_b32_e32 v1, v52
	v_mov_b32_e32 v2, v60
	v_mov_b32_e32 v3, v68
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v68, v61
	v_or_b32_e32 v4, 0x80000, v6
	v_or_b32_e32 v0, 0x1e000, v6
	buffer_store_dwordx4 v[66:69], v0, s[0:3], 0 offen
	v_mov_b32_e32 v0, v70
	v_mov_b32_e32 v1, v78
	v_mov_b32_e32 v2, v86
	v_mov_b32_e32 v3, v94
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x82000, v6
	v_mov_b32_e32 v94, v73
	v_mov_b32_e32 v0, v71
	v_mov_b32_e32 v1, v79
	v_mov_b32_e32 v2, v87
	v_mov_b32_e32 v3, v95
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x84000, v6
	v_mov_b32_e32 v95, v81
	v_mov_b32_e32 v0, v72
	v_mov_b32_e32 v1, v80
	v_mov_b32_e32 v2, v88
	v_mov_b32_e32 v3, v96
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v96, v89
	v_or_b32_e32 v4, 0x88000, v6
	v_or_b32_e32 v0, 0x86000, v6
	buffer_store_dwordx4 v[94:97], v0, s[0:3], 0 offen
	v_mov_b32_e32 v0, v74
	v_mov_b32_e32 v1, v82
	v_mov_b32_e32 v2, v90
	v_mov_b32_e32 v3, v98
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x8a000, v6
	v_mov_b32_e32 v98, v77
	v_mov_b32_e32 v0, v75
	v_mov_b32_e32 v1, v83
	v_mov_b32_e32 v2, v91
	v_mov_b32_e32 v3, v99
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x8c000, v6
	v_mov_b32_e32 v99, v85
	v_mov_b32_e32 v0, v76
	v_mov_b32_e32 v1, v84
	v_mov_b32_e32 v2, v92
	v_mov_b32_e32 v3, v100
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v100, v93
	v_or_b32_e32 v4, 0x90000, v6
	v_or_b32_e32 v0, 0x8e000, v6
	buffer_store_dwordx4 v[98:101], v0, s[0:3], 0 offen
	s_waitcnt lgkmcnt(7)
	v_mov_b32_e32 v0, v102
	s_waitcnt lgkmcnt(5)
	v_mov_b32_e32 v1, v110
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v2, v118
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v3, v126
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x92000, v6
	v_mov_b32_e32 v126, v105
	v_mov_b32_e32 v0, v103
	v_mov_b32_e32 v1, v111
	v_mov_b32_e32 v2, v119
	v_mov_b32_e32 v3, v127
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x94000, v6
	v_mov_b32_e32 v127, v113
	v_mov_b32_e32 v0, v104
	v_mov_b32_e32 v1, v112
	v_mov_b32_e32 v2, v120
	v_mov_b32_e32 v3, v128
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v128, v121
	v_or_b32_e32 v4, 0x98000, v6
	v_or_b32_e32 v0, 0x96000, v6
	buffer_store_dwordx4 v[126:129], v0, s[0:3], 0 offen
	v_mov_b32_e32 v0, v106
	v_mov_b32_e32 v1, v114
	v_mov_b32_e32 v2, v122
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, v130
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x9a000, v6
	v_mov_b32_e32 v130, v109
	v_mov_b32_e32 v0, v107
	v_mov_b32_e32 v1, v115
	v_mov_b32_e32 v2, v123
	v_mov_b32_e32 v3, v131
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_or_b32_e32 v4, 0x9c000, v6
	v_mov_b32_e32 v131, v117
	v_mov_b32_e32 v0, v108
	v_mov_b32_e32 v1, v116
	v_mov_b32_e32 v2, v124
	v_mov_b32_e32 v3, v132
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_mov_b32_e32 v132, v125
	s_nop 0
	v_or_b32_e32 v0, 0x9e000, v6
	buffer_store_dwordx4 v[130:133], v0, s[0:3], 0 offen
	.loc	1 90 4                          ; matmul.py:90:4
	s_endpgm
.Ltmp8:
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
		.amdhsa_next_free_vgpr 416
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
	.set matmul.num_vgpr, 255
	.set matmul.num_agpr, 160
	.set matmul.numbered_sgpr, 18
	.set matmul.private_seg_size, 0
	.set matmul.uses_vcc, 0
	.set matmul.uses_flat_scratch, 0
	.set matmul.has_dyn_sized_stack, 0
	.set matmul.has_recursion, 0
	.set matmul.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7688
; TotalNumSgprs: 24
; NumVgprs: 255
; NumAgprs: 160
; TotalNumVgprs: 416
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 51
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 416
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
	.byte	1                               ; Abbrev [1] 0xb:0x6c DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x46 DW_TAG_subprogram
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
	.byte	4                               ; Abbrev [4] 0x55:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	41                              ; DW_AT_call_line
	.byte	52                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x69:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	82                              ; DW_AT_call_line
	.byte	91                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
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
  - .agpr_count:     160
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
    .vgpr_count:     416
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

Running Time   5.38985 ms
	    60.56TF/s
	     0.03TB/s

