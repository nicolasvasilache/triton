// -----// Input IR
#blocked = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc1 = loc("/home/nico/triton/sandbox/matmul.py":30:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
#loc42 = loc("a_ptr"(#loc1))
#loc43 = loc("b_ptr"(#loc1))
#loc44 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %base_offsets_nd = arith.constant dense<2048> : tensor<128x1xi32, #blocked> loc(#loc77)
    %shift_1d = arith.constant 262144 : i32 loc(#loc96)
    %cst = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc)
    %c131072_i32 = arith.constant 131072 : i32 loc(#loc)
    %cst_0 = arith.constant dense<32768> : tensor<128x1xi32, #blocked> loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %c4194304_i32 = arith.constant 4194304 : i32 loc(#loc)
    %stride_val = arith.constant 32 : i32 loc(#loc97)
    %c1_i32 = arith.constant 1 : i32 loc(#loc9)
    %c512_i32 = arith.constant 512 : i32 loc(#loc9)
    %c0_i32 = arith.constant 0 : i32 loc(#loc9)
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma> loc(#loc50)
    %pid0 = tt.get_program_id x : i32 loc(#loc80)
    %pid1 = tt.get_program_id y : i32 loc(#loc81)
    %pid2 = tt.get_program_id z : i32 loc(#loc82)
    %npg1 = tt.get_num_programs y : i32 loc(#loc83)
    %npg2 = tt.get_num_programs z : i32 loc(#loc84)
    %stride_val_1 = arith.muli %npg2, %npg1 : i32 loc(#loc107)
    %linear_idx = arith.muli %pid0, %stride_val_1 : i32 loc(#loc99)
    %linear_idx_2 = arith.muli %pid1, %npg2 : i32 loc(#loc99)
    %linear_idx_3 = arith.addi %linear_idx, %linear_idx_2 : i32 loc(#loc100)
    %linear_idx_4 = arith.addi %linear_idx_3, %pid2 : i32 loc(#loc100)
    %idx_val = arith.divsi %linear_idx_4, %stride_val : i32 loc(#loc86)
    %remaining = arith.remsi %linear_idx_4, %stride_val : i32 loc(#loc87)
    %acc_5 = scf.for %k = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%acc_17 = %acc) -> (tensor<128x64xf32, #mma>)  : i32 {
      %base_offsets_nd_18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc88)
      %base_offsets_nd_19 = tt.expand_dims %base_offsets_nd_18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc88)
      %base_offsets_nd_20 = arith.muli %base_offsets_nd_19, %cst_0 : tensor<128x1xi32, #blocked> loc(#loc88)
      %base_offsets_nd_21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc88)
      %base_offsets_nd_22 = tt.expand_dims %base_offsets_nd_21 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc88)
      %base_offsets_nd_23 = tt.broadcast %base_offsets_nd_20 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc88)
      %base_offsets_nd_24 = tt.broadcast %base_offsets_nd_22 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc88)
      %base_offsets_nd_25 = arith.addi %base_offsets_nd_23, %base_offsets_nd_24 : tensor<128x64xi32, #blocked> loc(#loc88)
      %shift_1d_26 = arith.muli %idx_val, %c4194304_i32 : i32 loc(#loc101)
      %shift_1d_27 = arith.muli %k, %c64_i32 : i32 loc(#loc101)
      %res_28 = arith.addi %shift_1d_26, %shift_1d_27 : i32 loc(#loc102)
      %4 = tt.splat %res_28 : i32 -> tensor<128x64xi32, #blocked> loc(#loc66)
      %5 = arith.addi %base_offsets_nd_25, %4 : tensor<128x64xi32, #blocked> loc(#loc66)
      %base_offsets_nd_29 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc91)
      %base_offsets_nd_30 = tt.expand_dims %base_offsets_nd_29 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc91)
      %base_offsets_nd_31 = arith.muli %base_offsets_nd_30, %cst : tensor<64x1xi32, #blocked> loc(#loc91)
      %base_offsets_nd_32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc91)
      %base_offsets_nd_33 = tt.expand_dims %base_offsets_nd_32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc91)
      %base_offsets_nd_34 = tt.broadcast %base_offsets_nd_31 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc91)
      %base_offsets_nd_35 = tt.broadcast %base_offsets_nd_33 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc91)
      %base_offsets_nd_36 = arith.addi %base_offsets_nd_34, %base_offsets_nd_35 : tensor<64x64xi32, #blocked> loc(#loc91)
      %shift_1d_37 = arith.muli %k, %c131072_i32 : i32 loc(#loc103)
      %shift_1d_38 = arith.muli %remaining, %c64_i32 : i32 loc(#loc103)
      %res_39 = arith.addi %shift_1d_37, %shift_1d_38 : i32 loc(#loc104)
      %6 = tt.splat %res_39 : i32 -> tensor<64x64xi32, #blocked> loc(#loc67)
      %7 = arith.addi %base_offsets_nd_36, %6 : tensor<64x64xi32, #blocked> loc(#loc67)
      %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc68)
      %a_40 = tt.addptr %a, %5 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc68)
      %a_41 = tt.load %a_40 : tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc69)
      %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc70)
      %b_42 = tt.addptr %b, %7 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc70)
      %b_43 = tt.load %b_42 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc71)
      %a_44 = ttg.convert_layout %a_41 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc72)
      %b_45 = ttg.convert_layout %b_43 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc73)
      %acc_46 = tt.dot %a_44, %b_45, %acc_17 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma> loc(#loc74)
      scf.yield %acc_46 : tensor<128x64xf32, #mma> loc(#loc37)
    } loc(#loc62)
    %acc_6 = ttg.convert_layout %acc_5 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked> loc(#loc75)
    %base_offsets_nd_7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc77)
    %base_offsets_nd_8 = tt.expand_dims %base_offsets_nd_7 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc77)
    %base_offsets_nd_9 = arith.muli %base_offsets_nd_8, %base_offsets_nd : tensor<128x1xi32, #blocked> loc(#loc77)
    %base_offsets_nd_10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc77)
    %base_offsets_nd_11 = tt.expand_dims %base_offsets_nd_10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc77)
    %base_offsets_nd_12 = tt.broadcast %base_offsets_nd_9 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc77)
    %base_offsets_nd_13 = tt.broadcast %base_offsets_nd_11 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc77)
    %base_offsets_nd_14 = arith.addi %base_offsets_nd_12, %base_offsets_nd_13 : tensor<128x64xi32, #blocked> loc(#loc77)
    %shift_1d_15 = arith.muli %idx_val, %shift_1d : i32 loc(#loc105)
    %shift_1d_16 = arith.muli %remaining, %c64_i32 : i32 loc(#loc105)
    %res = arith.addi %shift_1d_15, %shift_1d_16 : i32 loc(#loc106)
    %0 = tt.splat %res : i32 -> tensor<128x64xi32, #blocked> loc(#loc76)
    %1 = arith.addi %base_offsets_nd_14, %0 : tensor<128x64xi32, #blocked> loc(#loc76)
    %2 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc39)
    %3 = tt.addptr %2, %1 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc39)
    tt.store %3, %acc_6 : tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc40)
    tt.return loc(#loc41)
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
#loc28 = loc("/home/nico/triton/sandbox/nd_helpers.py":55:29)
#loc29 = loc("/home/nico/triton/sandbox/matmul.py":56:91)
#loc30 = loc("/home/nico/triton/sandbox/matmul.py":62:32)
#loc31 = loc("/home/nico/triton/sandbox/matmul.py":62:39)
#loc32 = loc("/home/nico/triton/sandbox/matmul.py":63:32)
#loc33 = loc("/home/nico/triton/sandbox/matmul.py":63:39)
#loc34 = loc("/home/nico/triton/sandbox/matmul.py":73:37)
#loc35 = loc("/home/nico/triton/sandbox/matmul.py":74:37)
#loc36 = loc("/home/nico/triton/sandbox/matmul.py":75:42)
#loc37 = loc("/home/nico/triton/sandbox/matmul.py":66:8)
#loc38 = loc("/home/nico/triton/sandbox/matmul.py":79:33)
#loc39 = loc("/home/nico/triton/sandbox/matmul.py":88:25)
#loc40 = loc("/home/nico/triton/sandbox/matmul.py":88:41)
#loc41 = loc("/home/nico/triton/sandbox/matmul.py":90:4)
#loc45 = loc("base_offsets_nd"(#loc2))
#loc46 = loc("shift_1d"(#loc5))
#loc47 = loc("stride_val"(#loc6))
#loc48 = loc("strides"(#loc7))
#loc49 = loc("start_blocks_c"(#loc8))
#loc50 = loc("acc"(#loc10))
#loc51 = loc("pid0"(#loc11))
#loc52 = loc("linear_program_id"(#loc12))
#loc53 = loc("pid1"(#loc13))
#loc54 = loc("pid2"(#loc14))
#loc55 = loc("npg1"(#loc15))
#loc56 = loc("npg2"(#loc16))
#loc57 = loc("strides"(#loc17))
#loc58 = loc("linear_idx"(#loc19))
#loc59 = loc("linear_idx"(#loc20))
#loc60 = loc("idx_val"(#loc21))
#loc61 = loc("remaining"(#loc22))
#loc62 = loc("acc"(#loc9))
#loc63 = loc("shift_1d"(#loc25))
#loc64 = loc("res"(#loc26))
#loc65 = loc("shift_1d"(#loc27))
#loc66 = loc(callsite(#loc28 at #loc23))
#loc67 = loc(callsite(#loc28 at #loc29))
#loc68 = loc("a"(#loc30))
#loc69 = loc("a"(#loc31))
#loc70 = loc("b"(#loc32))
#loc71 = loc("b"(#loc33))
#loc72 = loc("a"(#loc34))
#loc73 = loc("b"(#loc35))
#loc74 = loc("acc"(#loc36))
#loc75 = loc("acc"(#loc38))
#loc76 = loc(callsite(#loc28 at #loc3))
#loc77 = loc(callsite(#loc45 at #loc3))
#loc78 = loc(callsite(#loc46 at #loc3))
#loc79 = loc(callsite(#loc48 at #loc49))
#loc80 = loc(callsite(#loc51 at #loc52))
#loc81 = loc(callsite(#loc53 at #loc52))
#loc82 = loc(callsite(#loc54 at #loc52))
#loc83 = loc(callsite(#loc55 at #loc52))
#loc84 = loc(callsite(#loc56 at #loc52))
#loc85 = loc(callsite(#loc18 at #loc52))
#loc86 = loc(callsite(#loc60 at #loc49))
#loc87 = loc(callsite(#loc61 at #loc49))
#loc88 = loc(callsite(#loc45 at #loc23))
#loc89 = loc(callsite(#loc63 at #loc23))
#loc90 = loc(callsite(#loc65 at #loc23))
#loc91 = loc(callsite(#loc45 at #loc29))
#loc92 = loc(callsite(#loc63 at #loc29))
#loc93 = loc(callsite(#loc65 at #loc29))
#loc94 = loc(callsite(#loc63 at #loc3))
#loc95 = loc(callsite(#loc65 at #loc3))
#loc96 = loc(callsite(#loc4 at #loc78))
#loc97 = loc(callsite(#loc47 at #loc79))
#loc98 = loc(callsite(#loc57 at #loc85))
#loc99 = loc(callsite(#loc58 at #loc85))
#loc100 = loc(callsite(#loc59 at #loc85))
#loc101 = loc(callsite(#loc24 at #loc89))
#loc102 = loc(callsite(#loc64 at #loc90))
#loc103 = loc(callsite(#loc24 at #loc92))
#loc104 = loc(callsite(#loc64 at #loc93))
#loc105 = loc(callsite(#loc24 at #loc94))
#loc106 = loc(callsite(#loc64 at #loc95))
#loc107 = loc(callsite(#loc47 at #loc98))

// -----// After pipelining
#blocked = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc1 = loc("/home/nico/triton/sandbox/matmul.py":30:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#loc37 = loc("a_ptr"(#loc1))
#loc38 = loc("b_ptr"(#loc1))
#loc39 = loc("c_ptr"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"(#loc1)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"(#loc1)), %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"(#loc1))) attributes {noinline = false} {
    %acc = arith.constant 511 : i32 loc(#loc40)
    %true = arith.constant true loc(#loc)
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c4194304_i32 = arith.constant 4194304 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst_0 = arith.constant dense<32768> : tensor<128x1xi32, #blocked> loc(#loc)
    %c131072_i32 = arith.constant 131072 : i32 loc(#loc)
    %cst_1 = arith.constant dense<2048> : tensor<64x1xi32, #blocked> loc(#loc)
    %c262144_i32 = arith.constant 262144 : i32 loc(#loc)
    %cst_2 = arith.constant dense<2048> : tensor<128x1xi32, #blocked> loc(#loc)
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
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf32, #shared, #smem, mutable> loc(#loc54)
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf32, #shared1, #smem, mutable> loc(#loc55)
    %base_offsets_nd = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc77)
    %base_offsets_nd_6 = tt.expand_dims %base_offsets_nd {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc77)
    %base_offsets_nd_7 = arith.muli %base_offsets_nd_6, %cst_0 : tensor<128x1xi32, #blocked> loc(#loc77)
    %base_offsets_nd_8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc77)
    %base_offsets_nd_9 = tt.expand_dims %base_offsets_nd_8 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc77)
    %base_offsets_nd_10 = tt.broadcast %base_offsets_nd_7 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc77)
    %base_offsets_nd_11 = tt.broadcast %base_offsets_nd_9 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc77)
    %base_offsets_nd_12 = arith.addi %base_offsets_nd_10, %base_offsets_nd_11 : tensor<128x64xi32, #blocked> loc(#loc77)
    %shift_1d = arith.muli %idx_val, %c4194304_i32 : i32 loc(#loc89)
    %res = arith.addi %shift_1d, %c0_i32 : i32 loc(#loc90)
    %0 = tt.splat %res : i32 -> tensor<128x64xi32, #blocked> loc(#loc60)
    %1 = arith.addi %base_offsets_nd_12, %0 : tensor<128x64xi32, #blocked> loc(#loc60)
    %base_offsets_nd_13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc80)
    %base_offsets_nd_14 = tt.expand_dims %base_offsets_nd_13 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc80)
    %base_offsets_nd_15 = arith.muli %base_offsets_nd_14, %cst_1 : tensor<64x1xi32, #blocked> loc(#loc80)
    %base_offsets_nd_16 = tt.broadcast %base_offsets_nd_15 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_17 = tt.broadcast %base_offsets_nd_9 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc80)
    %base_offsets_nd_18 = arith.addi %base_offsets_nd_16, %base_offsets_nd_17 : tensor<64x64xi32, #blocked> loc(#loc80)
    %shift_1d_19 = arith.muli %remaining, %c64_i32 : i32 loc(#loc91)
    %res_20 = arith.addi %c0_i32, %shift_1d_19 : i32 loc(#loc92)
    %2 = tt.splat %res_20 : i32 -> tensor<64x64xi32, #blocked> loc(#loc61)
    %3 = arith.addi %base_offsets_nd_18, %2 : tensor<64x64xi32, #blocked> loc(#loc61)
    %a_21 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc62)
    %a_22 = tt.addptr %a_21, %1 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc62)
    %4 = tt.splat %true : i1 -> tensor<128x64xi1, #blocked> loc(#loc)
    %a_23 = tt.load %a_22, %4 {amd.pipeliner_part = "prologue"} : tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc54)
    %5 = tt.splat %true : i1 -> tensor<128x64xi1, #blocked> loc(#loc)
    %a_24 = tt.load %a_22, %5 {amd.pipeliner_part = "prologue"} : tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc54)
    %b_25 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc63)
    %b_26 = tt.addptr %b_25, %3 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc63)
    %6 = tt.splat %true : i1 -> tensor<64x64xi1, #blocked> loc(#loc)
    %b_27 = tt.load %b_26, %6 {amd.pipeliner_part = "prologue"} : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc55)
    %7 = tt.splat %true : i1 -> tensor<64x64xi1, #blocked> loc(#loc)
    %b_28 = tt.load %b_26, %7 {amd.pipeliner_part = "prologue"} : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc55)
    %acc_29 = arith.cmpi slt, %c0_i32, %c1_i32 : i32 loc(#loc40)
    %acc_30 = arith.select %acc_29, %c0_i32, %c0_i32 : i32 loc(#loc40)
    %a_31 = ttg.memdesc_index %a[%acc_30] : !ttg.memdesc<1x128x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable, 1x128x64> loc(#loc54)
    ttg.local_store %a_23, %a_31 : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable, 1x128x64> loc(#loc54)
    %b_32 = ttg.memdesc_index %b[%acc_30] : !ttg.memdesc<1x64x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable, 1x64x64> loc(#loc55)
    ttg.local_store %b_27, %b_32 : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable, 1x64x64> loc(#loc55)
    %a_33 = ttg.local_load %a_31 : !ttg.memdesc<128x64xf32, #shared, #smem, mutable, 1x128x64> -> tensor<128x64xf32, #blocked> loc(#loc54)
    %b_34 = ttg.local_load %b_32 : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable, 1x64x64> -> tensor<64x64xf32, #blocked> loc(#loc55)
    %a_35 = ttg.convert_layout %a_33 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc64)
    %b_36 = ttg.convert_layout %b_34 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc65)
    %acc_37:4 = scf.for %acc_51 = %c0_i32 to %acc step %c1_i32 iter_args(%arg4 = %cst, %acc_52 = %acc_30, %a_53 = %a_35, %b_54 = %b_36) -> (tensor<128x64xf32, #mma>, i32, tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)  : i32 {
      %base_offsets_nd_55 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc77)
      %base_offsets_nd_56 = tt.expand_dims %base_offsets_nd_55 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc77)
      %base_offsets_nd_57 = arith.muli %base_offsets_nd_56, %cst_0 : tensor<128x1xi32, #blocked> loc(#loc77)
      %base_offsets_nd_58 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc77)
      %base_offsets_nd_59 = tt.expand_dims %base_offsets_nd_58 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc77)
      %base_offsets_nd_60 = tt.broadcast %base_offsets_nd_57 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc77)
      %base_offsets_nd_61 = tt.broadcast %base_offsets_nd_59 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc77)
      %base_offsets_nd_62 = arith.addi %base_offsets_nd_60, %base_offsets_nd_61 : tensor<128x64xi32, #blocked> loc(#loc77)
      %shift_1d_63 = arith.muli %idx_val, %c4194304_i32 : i32 loc(#loc89)
      %acc_64 = arith.addi %acc_51, %c1_i32 : i32 loc(#loc40)
      %shift_1d_65 = arith.muli %acc_64, %c64_i32 : i32 loc(#loc89)
      %res_66 = arith.addi %shift_1d_63, %shift_1d_65 : i32 loc(#loc90)
      %12 = tt.splat %res_66 : i32 -> tensor<128x64xi32, #blocked> loc(#loc60)
      %13 = arith.addi %base_offsets_nd_62, %12 : tensor<128x64xi32, #blocked> loc(#loc60)
      %base_offsets_nd_67 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc80)
      %base_offsets_nd_68 = tt.expand_dims %base_offsets_nd_67 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc80)
      %base_offsets_nd_69 = arith.muli %base_offsets_nd_68, %cst_1 : tensor<64x1xi32, #blocked> loc(#loc80)
      %base_offsets_nd_70 = tt.broadcast %base_offsets_nd_69 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc80)
      %base_offsets_nd_71 = tt.broadcast %base_offsets_nd_59 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked> loc(#loc80)
      %base_offsets_nd_72 = arith.addi %base_offsets_nd_70, %base_offsets_nd_71 : tensor<64x64xi32, #blocked> loc(#loc80)
      %acc_73 = arith.addi %acc_51, %c1_i32 : i32 loc(#loc40)
      %shift_1d_74 = arith.muli %acc_73, %c131072_i32 : i32 loc(#loc91)
      %shift_1d_75 = arith.muli %remaining, %c64_i32 : i32 loc(#loc91)
      %res_76 = arith.addi %shift_1d_74, %shift_1d_75 : i32 loc(#loc92)
      %14 = tt.splat %res_76 : i32 -> tensor<64x64xi32, #blocked> loc(#loc61)
      %15 = arith.addi %base_offsets_nd_72, %14 : tensor<64x64xi32, #blocked> loc(#loc61)
      %a_77 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc62)
      %a_78 = tt.addptr %a_77, %13 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc62)
      %a_79 = tt.load %a_78 : tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc54)
      %b_80 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc63)
      %b_81 = tt.addptr %b_80, %15 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc63)
      %b_82 = tt.load %b_81 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc55)
      %acc_83 = tt.dot %a_53, %b_54, %arg4 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma> loc(#loc66)
      %acc_84 = arith.addi %acc_52, %c1_i32 : i32 loc(#loc40)
      %acc_85 = arith.cmpi slt, %acc_84, %c1_i32 : i32 loc(#loc40)
      %acc_86 = arith.select %acc_85, %acc_84, %c0_i32 : i32 loc(#loc40)
      %a_87 = ttg.memdesc_index %a[%acc_86] : !ttg.memdesc<1x128x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable, 1x128x64> loc(#loc54)
      ttg.local_store %a_79, %a_87 : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable, 1x128x64> loc(#loc54)
      %b_88 = ttg.memdesc_index %b[%acc_86] : !ttg.memdesc<1x64x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable, 1x64x64> loc(#loc55)
      ttg.local_store %b_82, %b_88 : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable, 1x64x64> loc(#loc55)
      %a_89 = ttg.local_load %a_87 : !ttg.memdesc<128x64xf32, #shared, #smem, mutable, 1x128x64> -> tensor<128x64xf32, #blocked> loc(#loc54)
      %b_90 = ttg.local_load %b_88 : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable, 1x64x64> -> tensor<64x64xf32, #blocked> loc(#loc55)
      %a_91 = ttg.convert_layout %a_89 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc64)
      %b_92 = ttg.convert_layout %b_90 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc65)
      scf.yield %acc_83, %acc_86, %a_91, %b_92 : tensor<128x64xf32, #mma>, i32, tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc40)
    } loc(#loc40)
    %acc_38 = scf.if %true -> (tensor<128x64xf32, #mma>) {
      %acc_51 = tt.dot %acc_37#2, %acc_37#3, %acc_37#0 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma> loc(#loc66)
      scf.yield %acc_51 : tensor<128x64xf32, #mma> loc(#loc66)
    } else {
      scf.yield %acc_37#0 : tensor<128x64xf32, #mma> loc(#loc66)
    } loc(#loc66)
    ttg.local_dealloc %b : !ttg.memdesc<1x64x64xf32, #shared1, #smem, mutable> loc(#loc40)
    ttg.local_dealloc %a : !ttg.memdesc<1x128x64xf32, #shared, #smem, mutable> loc(#loc40)
    %acc_39 = ttg.convert_layout %acc_38 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked> loc(#loc67)
    %base_offsets_nd_40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc83)
    %base_offsets_nd_41 = tt.expand_dims %base_offsets_nd_40 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc83)
    %base_offsets_nd_42 = arith.muli %base_offsets_nd_41, %cst_2 : tensor<128x1xi32, #blocked> loc(#loc83)
    %base_offsets_nd_43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc83)
    %base_offsets_nd_44 = tt.expand_dims %base_offsets_nd_43 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc83)
    %base_offsets_nd_45 = tt.broadcast %base_offsets_nd_42 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc83)
    %base_offsets_nd_46 = tt.broadcast %base_offsets_nd_44 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc83)
    %base_offsets_nd_47 = arith.addi %base_offsets_nd_45, %base_offsets_nd_46 : tensor<128x64xi32, #blocked> loc(#loc83)
    %shift_1d_48 = arith.muli %idx_val, %c262144_i32 : i32 loc(#loc93)
    %shift_1d_49 = arith.muli %remaining, %c64_i32 : i32 loc(#loc93)
    %res_50 = arith.addi %shift_1d_48, %shift_1d_49 : i32 loc(#loc94)
    %8 = tt.splat %res_50 : i32 -> tensor<128x64xi32, #blocked> loc(#loc68)
    %9 = arith.addi %base_offsets_nd_47, %8 : tensor<128x64xi32, #blocked> loc(#loc68)
    %10 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc34)
    %11 = tt.addptr %10, %9 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc34)
    tt.store %11, %acc_39 : tensor<128x64x!tt.ptr<f32>, #blocked> loc(#loc35)
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
	v_and_b32_e32 v30, 48, v0
	v_and_b32_e32 v38, 15, v0
	v_lshlrev_b32_e32 v1, 15, v30
	v_lshlrev_b32_e32 v51, 2, v38
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
	s_mul_i32 s0, s1, s0
	s_add_i32 s0, s0, s17
.Ltmp4:
	.loc	2 73 31                         ; tuple_helpers.py:73:31 @[ matmul.py:41:52 ]
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	s_add_i32 s9, s0, s1
	s_ashr_i32 s8, s9, 5
.Ltmp5:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:55:91 ]
	s_lshl_b32 s1, s8, 22
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v1, v51, s1
.Ltmp6:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp7:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v6, 64, v30
	scratch_store_dword off, v6, off offset:956 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v6, 15, v6
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v6, v51, s1
.Ltmp8:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	v_lshlrev_b32_e32 v42, 4, v38
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
.Ltmp9:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v37, 1, v30
	v_or_b32_e32 v40, 0x44, v30
.Ltmp10:
	.loc	1 62 39                         ; matmul.py:62:39
	v_lshl_or_b32 v50, v30, 8, v42
	global_load_dwordx4 v[14:17], v[6:7], off
.Ltmp11:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v6, 15, v37
	v_lshlrev_b32_e32 v48, 15, v40
.Ltmp12:
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v70, 0, v50
.Ltmp13:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v6, v51, s1
.Ltmp14:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[26:29], v[6:7], off
.Ltmp15:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v33, 5, v30
	v_lshlrev_b32_e32 v46, 15, v33
	v_or_b32_e32 v34, 2, v30
	v_or_b32_e32 v32, 6, v30
	v_or_b32_e32 v52, 0x46, v30
	v_lshlrev_b32_e32 v47, 15, v32
	scratch_store_dword off, v52, off offset:980 ; 4-byte Folded Spill
	v_or_b32_e32 v41, 0x45, v30
	v_lshlrev_b32_e32 v49, 15, v41
	v_or_b32_e32 v36, 3, v30
	v_or_b32_e32 v39, 0x43, v30
	v_lshlrev_b32_e32 v38, 15, v39
	v_or_b32_e32 v35, 4, v30
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v38, v38, v51, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v43, 15, v35
	scratch_store_dword off, v39, off offset:968 ; 4-byte Folded Spill
.Ltmp16:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[38:39], v[38:39], 2, s[2:3]
.Ltmp17:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v42, v43, v51, s1
	scratch_store_dword off, v40, off offset:972 ; 4-byte Folded Spill
	scratch_store_dword off, v41, off offset:976 ; 4-byte Folded Spill
.Ltmp18:
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[38:41], v[38:39], off
	.loc	1 62 32 is_stmt 0               ; matmul.py:62:32
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[2:3]
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[42:45], v[42:43], off
.Ltmp19:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v31, 7, v30
.Ltmp20:
	.loc	2 75 32                         ; tuple_helpers.py:75:32 @[ matmul.py:41:52 ]
	s_and_b32 s9, s9, 0x3ffffe0
	s_sub_i32 s9, s0, s9
.Ltmp21:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:56:91 ]
	s_lshl_b32 s9, s9, 6
.Ltmp22:
	.loc	1 73 37                         ; matmul.py:73:37
	v_and_b32_e32 v82, 1, v0
	s_movk_i32 s10, 0x1f0
	.loc	1 74 37                         ; matmul.py:74:37
	v_or_b32_e32 v255, s1, v1
	.loc	1 63 39                         ; matmul.py:63:39
	v_mov_b32_e32 v205, 0x1ff
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(9)
	ds_write_b128 v70, v[2:5]
.Ltmp23:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v48, v51, s1
.Ltmp24:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp25:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v6, 0x41, v30
	scratch_store_dword off, v6, off offset:960 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v6, 15, v6
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v6, v51, s1
.Ltmp26:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[22:25], v[6:7], off
	s_waitcnt vmcnt(10)
	ds_write_b128 v70, v[14:17] offset:16384
.Ltmp27:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v14, v46, v51, s1
.Ltmp28:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[14:17], v[14:15], off
.Ltmp29:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v6, 15, v34
	v_xor_b32_e32 v46, 16, v50
	v_or_b32_e32 v7, 0x42, v30
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v6, v51, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v48, 15, v52
.Ltmp30:
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v52, 0, v46
.Ltmp31:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v46, v47, v51, s1
	scratch_store_dword off, v7, off offset:964 ; 4-byte Folded Spill
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v8, 15, v7
.Ltmp32:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v47, 31, v46
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[18:21], v[6:7], off
	s_waitcnt vmcnt(12)
	ds_write_b128 v52, v[26:29] offset:256
.Ltmp33:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v26, v49, v51, s1
.Ltmp34:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[26:29], v[26:27], off
	s_waitcnt vmcnt(4)
	ds_write_b128 v52, v[22:25] offset:16640
.Ltmp35:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v22, v48, v51, s1
.Ltmp36:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[24:25], v[46:47], 2, s[2:3]
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[46:47], v[22:23], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[22:25], v[24:25], off
.Ltmp37:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v8, v51, s1
.Ltmp38:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[10:13], v[6:7], off
.Ltmp39:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v6, 15, v36
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v6, v51, s1
.Ltmp40:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[6:9], v[6:7], off
	v_xor_b32_e32 v48, 32, v50
	v_add_u32_e32 v48, 0, v48
	s_waitcnt vmcnt(4)
	ds_write_b128 v48, v[18:21] offset:512
	global_load_dwordx4 v[18:21], v[46:47], off
.Ltmp41:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v46, 15, v31
	v_or_b32_e32 v47, 0x47, v30
	scratch_store_dword off, v47, off offset:984 ; 4-byte Folded Spill
.Ltmp42:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(3)
	ds_write_b128 v48, v[10:13] offset:16896
.Ltmp43:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v10, v46, v51, s1
.Ltmp44:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[2:3]
	v_xor_b32_e32 v46, 48, v50
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[10:13], v[10:11], off
	v_add_u32_e32 v49, 0, v46
	s_waitcnt vmcnt(3)
	ds_write_b128 v49, v[6:9] offset:768
.Ltmp45:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v6, 15, v47
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v6, v51, s1
.Ltmp46:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[6:9], v[6:7], off
	ds_write_b128 v49, v[38:41] offset:17152
	v_xor_b32_e32 v41, 64, v50
	v_add_u32_e32 v47, 0, v41
.Ltmp47:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v39, 9, v30
.Ltmp48:
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v47, v[42:45] offset:1024
.Ltmp49:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v44, 0x48, v30
.Ltmp50:
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v47, v[2:5] offset:17408
	v_xor_b32_e32 v2, 0x50, v50
.Ltmp51:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v38, 8, v30
	v_lshlrev_b32_e32 v42, 15, v39
	scratch_store_dword off, v44, off offset:988 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v44, 15, v44
.Ltmp52:
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v53, 0, v2
.Ltmp53:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v41, 15, v38
.Ltmp54:
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v53, v[14:17] offset:1280
.Ltmp55:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v14, v44, v51, s1
.Ltmp56:
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v53, v[26:29] offset:17664
.Ltmp57:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v26, v42, v51, s1
	v_or3_b32 v2, v41, v51, s1
.Ltmp58:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v27, 31, v26
	v_xor_b32_e32 v28, 0x60, v50
.Ltmp59:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v45, 0x49, v30
.Ltmp60:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[2:3]
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	v_add_u32_e32 v41, 0, v28
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[2:3]
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[14:17], v[14:15], off
	ds_write_b128 v41, v[22:25] offset:1536
	global_load_dwordx4 v[22:25], v[26:27], off
.Ltmp61:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v27, 15, v45
	v_or_b32_e32 v40, 10, v30
.Ltmp62:
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[2:5], v[2:3], off
	s_waitcnt vmcnt(7)
	ds_write_b128 v41, v[18:21] offset:17920
.Ltmp63:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v18, v27, v51, s1
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v43, 15, v40
.Ltmp64:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v19, 31, v18
.Ltmp65:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v26, v43, v51, s1
.Ltmp66:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[18:21], v[18:19], off
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v27, 31, v26
	v_xor_b32_e32 v28, 0x70, v50
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[2:3]
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v42, 0, v28
.Ltmp67:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v46, 0x4a, v30
	scratch_store_dword off, v46, off offset:996 ; 4-byte Folded Spill
	v_or_b32_e32 v29, 0x4b, v30
	scratch_store_dword off, v45, off offset:992 ; 4-byte Folded Spill
	scratch_store_dword off, v29, off offset:1000 ; 4-byte Folded Spill
.Ltmp68:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(9)
	ds_write_b128 v42, v[10:13] offset:1792
	global_load_dwordx4 v[10:13], v[26:27], off
.Ltmp69:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v26, 15, v46
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v26, v26, v51, s1
.Ltmp70:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	s_waitcnt vmcnt(9)
	ds_write_b128 v42, v[6:9] offset:18176
	global_load_dwordx4 v[6:9], v[26:27], off
	v_xor_b32_e32 v27, 0x80, v50
	v_add_u32_e32 v43, 0, v27
.Ltmp71:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v26, 11, v30
	v_or_b32_e32 v27, 12, v30
	v_lshlrev_b32_e32 v28, 15, v27
.Ltmp72:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(8)
	ds_write_b128 v43, v[14:17] offset:18432
	v_xor_b32_e32 v14, 0x90, v50
	v_add_u32_e32 v46, 0, v14
.Ltmp73:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v14, 15, v29
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v14, v14, v51, s1
.Ltmp74:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(6)
	ds_write_b128 v43, v[2:5] offset:2048
.Ltmp75:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v2, 15, v26
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v2, v51, s1
.Ltmp76:
	.loc	1 62 39                         ; matmul.py:62:39
	ds_write_b128 v46, v[22:25] offset:2304
.Ltmp77:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v22, v28, v51, s1
.Ltmp78:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[2:3]
	v_ashrrev_i32_e32 v15, 31, v14
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	s_waitcnt vmcnt(5)
	ds_write_b128 v46, v[18:21] offset:18688
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[18:19], v[22:23], 2, s[2:3]
	v_xor_b32_e32 v22, 0xa0, v50
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[2:5], v[2:3], off
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[2:3]
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v54, 0, v22
	global_load_dwordx4 v[14:17], v[14:15], off
.Ltmp79:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v25, 13, v30
.Ltmp80:
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[18:21], v[18:19], off
.Ltmp81:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshlrev_b32_e32 v24, 11, v30
.Ltmp82:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v23, 0
	v_accvgpr_write_b32 a127, v23
	v_accvgpr_write_b32 a126, v23
	v_accvgpr_write_b32 a125, v23
	v_accvgpr_write_b32 a124, v23
	v_accvgpr_write_b32 a123, v23
	v_accvgpr_write_b32 a122, v23
	v_accvgpr_write_b32 a121, v23
	v_accvgpr_write_b32 a120, v23
	v_accvgpr_write_b32 a119, v23
	v_accvgpr_write_b32 a118, v23
	v_accvgpr_write_b32 a117, v23
	v_accvgpr_write_b32 a116, v23
	v_accvgpr_write_b32 a115, v23
	v_accvgpr_write_b32 a114, v23
	v_accvgpr_write_b32 a113, v23
	v_accvgpr_write_b32 a112, v23
	v_accvgpr_write_b32 a111, v23
	v_accvgpr_write_b32 a110, v23
	v_accvgpr_write_b32 a109, v23
	v_accvgpr_write_b32 a108, v23
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(4)
	ds_write_b128 v54, v[10:13] offset:2560
.Ltmp83:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v10, 0x4c, v30
	scratch_store_dword off, v10, off offset:1004 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v10, 15, v10
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v10, v10, v51, s1
.Ltmp84:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	s_waitcnt vmcnt(4)
	ds_write_b128 v54, v[6:9] offset:18944
	global_load_dwordx4 v[6:9], v[10:11], off
	v_xor_b32_e32 v10, 0xb0, v50
.Ltmp85:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v11, 0x4d, v30
.Ltmp86:
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v55, 0, v10
	v_xor_b32_e32 v12, 0xc0, v50
	v_add_u32_e32 v58, 0, v12
	scratch_store_dword off, v11, off offset:1008 ; 4-byte Folded Spill
	v_accvgpr_write_b32 a107, v23
	v_accvgpr_write_b32 a106, v23
	v_accvgpr_write_b32 a105, v23
	v_accvgpr_write_b32 a104, v23
	v_accvgpr_write_b32 a103, v23
	v_accvgpr_write_b32 a102, v23
	v_accvgpr_write_b32 a101, v23
	v_accvgpr_write_b32 a100, v23
	v_accvgpr_write_b32 a99, v23
	v_accvgpr_write_b32 a98, v23
	v_accvgpr_write_b32 a97, v23
	v_accvgpr_write_b32 a96, v23
	v_accvgpr_write_b32 a95, v23
	v_accvgpr_write_b32 a94, v23
	v_accvgpr_write_b32 a93, v23
	v_accvgpr_write_b32 a92, v23
	v_accvgpr_write_b32 a91, v23
	v_accvgpr_write_b32 a90, v23
	v_accvgpr_write_b32 a89, v23
	v_accvgpr_write_b32 a88, v23
	v_accvgpr_write_b32 a87, v23
	v_accvgpr_write_b32 a86, v23
	v_accvgpr_write_b32 a85, v23
	v_accvgpr_write_b32 a84, v23
	v_accvgpr_write_b32 a83, v23
	v_accvgpr_write_b32 a82, v23
	v_accvgpr_write_b32 a81, v23
	v_accvgpr_write_b32 a80, v23
	v_accvgpr_write_b32 a79, v23
	v_accvgpr_write_b32 a78, v23
	v_accvgpr_write_b32 a77, v23
	v_accvgpr_write_b32 a76, v23
	v_accvgpr_write_b32 a75, v23
	v_accvgpr_write_b32 a74, v23
	v_accvgpr_write_b32 a73, v23
	v_accvgpr_write_b32 a72, v23
	v_accvgpr_write_b32 a71, v23
	s_waitcnt vmcnt(5)
	ds_write_b128 v55, v[2:5] offset:2816
.Ltmp87:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v2, 15, v25
	v_lshlrev_b32_e32 v3, 15, v11
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v2, v51, s1
	v_or3_b32 v10, v3, v51, s1
.Ltmp88:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(3)
	ds_write_b128 v58, v[18:21] offset:3072
.Ltmp89:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v18, 14, v30
.Ltmp90:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v11, 31, v10
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	ds_write_b128 v55, v[14:17] offset:19200
.Ltmp91:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v14, 15, v18
.Ltmp92:
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[2:3]
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[2:3]
.Ltmp93:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v14, v14, v51, s1
.Ltmp94:
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[2:5], v[2:3], off
	.loc	1 62 32 is_stmt 0               ; matmul.py:62:32
	v_ashrrev_i32_e32 v15, 31, v14
	.loc	1 62 39                         ; matmul.py:62:39
	global_load_dwordx4 v[10:13], v[10:11], off
	.loc	1 62 32                         ; matmul.py:62:32
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[2:3]
.Ltmp95:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v19, 15, v30
	v_accvgpr_write_b32 a70, v23
	v_accvgpr_write_b32 a69, v23
	v_accvgpr_write_b32 a68, v23
	v_accvgpr_write_b32 a67, v23
	v_accvgpr_write_b32 a66, v23
	v_accvgpr_write_b32 a65, v23
	v_accvgpr_write_b32 a64, v23
	v_accvgpr_write_b32 a63, v23
	v_accvgpr_write_b32 a62, v23
	v_accvgpr_write_b32 a61, v23
	v_accvgpr_write_b32 a60, v23
	v_accvgpr_write_b32 a59, v23
	v_accvgpr_write_b32 a58, v23
	v_accvgpr_write_b32 a57, v23
	v_accvgpr_write_b32 a56, v23
	v_accvgpr_write_b32 a55, v23
	v_accvgpr_write_b32 a54, v23
.Ltmp96:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(3)
	ds_write_b128 v58, v[6:9] offset:19456
	global_load_dwordx4 v[6:9], v[14:15], off
	v_xor_b32_e32 v14, 0xd0, v50
	v_add_u32_e32 v62, 0, v14
.Ltmp97:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v15, 0x4f, v30
	scratch_store_dword off, v15, off offset:1016 ; 4-byte Folded Spill
	v_accvgpr_write_b32 a53, v23
	v_accvgpr_write_b32 a52, v23
	v_accvgpr_write_b32 a51, v23
	v_accvgpr_write_b32 a50, v23
	v_accvgpr_write_b32 a49, v23
	v_accvgpr_write_b32 a48, v23
	v_accvgpr_write_b32 a47, v23
	v_accvgpr_write_b32 a46, v23
	v_accvgpr_write_b32 a45, v23
	v_accvgpr_write_b32 a44, v23
	v_accvgpr_write_b32 a43, v23
	v_accvgpr_write_b32 a42, v23
	v_accvgpr_write_b32 a41, v23
	v_accvgpr_write_b32 a40, v23
	v_accvgpr_write_b32 a39, v23
	v_accvgpr_write_b32 a38, v23
	v_accvgpr_write_b32 a37, v23
	v_accvgpr_write_b32 a36, v23
	v_accvgpr_write_b32 a35, v23
	v_accvgpr_write_b32 a34, v23
	v_accvgpr_write_b32 a33, v23
	v_accvgpr_write_b32 a32, v23
	v_accvgpr_write_b32 a31, v23
	v_accvgpr_write_b32 a30, v23
	v_accvgpr_write_b32 a29, v23
	v_accvgpr_write_b32 a28, v23
	v_accvgpr_write_b32 a27, v23
	v_accvgpr_write_b32 a26, v23
	v_accvgpr_write_b32 a25, v23
	v_accvgpr_write_b32 a24, v23
	v_accvgpr_write_b32 a23, v23
	v_accvgpr_write_b32 a22, v23
	v_accvgpr_write_b32 a21, v23
	v_accvgpr_write_b32 a20, v23
	v_accvgpr_write_b32 a19, v23
	v_accvgpr_write_b32 a18, v23
	v_accvgpr_write_b32 a17, v23
	v_accvgpr_write_b32 a16, v23
	v_accvgpr_write_b32 a15, v23
	v_accvgpr_write_b32 a14, v23
	v_accvgpr_write_b32 a13, v23
	v_accvgpr_write_b32 a12, v23
	v_accvgpr_write_b32 a11, v23
	v_accvgpr_write_b32 a10, v23
	v_accvgpr_write_b32 a9, v23
	v_accvgpr_write_b32 a8, v23
	v_accvgpr_write_b32 a7, v23
	v_accvgpr_write_b32 a6, v23
	v_accvgpr_write_b32 a5, v23
	v_accvgpr_write_b32 a4, v23
	v_accvgpr_write_b32 a3, v23
	v_accvgpr_write_b32 a2, v23
.Ltmp98:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(3)
	ds_write_b128 v62, v[2:5] offset:3328
.Ltmp99:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_or_b32_e32 v3, 0x4e, v30
.Ltmp100:
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(2)
	ds_write_b128 v62, v[10:13] offset:19712
	v_xor_b32_e32 v10, 0xe0, v50
.Ltmp101:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v2, 15, v19
	v_lshlrev_b32_e32 v14, 15, v3
.Ltmp102:
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v67, 0, v10
.Ltmp103:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v2, v2, v51, s1
	scratch_store_dword off, v3, off offset:1012 ; 4-byte Folded Spill
.Ltmp104:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[2:5], v[2:3], off
	v_accvgpr_write_b32 a1, v23
	v_accvgpr_write_b32 a0, v23
	s_waitcnt vmcnt(3)
	ds_write_b128 v67, v[6:9] offset:3584
.Ltmp105:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:55:91 ]
	v_lshlrev_b32_e32 v7, 15, v15
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_or3_b32 v6, v14, v51, s1
	v_or3_b32 v8, v7, v51, s1
.Ltmp106:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[14:15], v[6:7], 2, s[2:3]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[16:17], v[8:9], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[6:9], v[14:15], off
	global_load_dwordx4 v[10:13], v[16:17], off
	s_waitcnt vmcnt(1)
	ds_write_b128 v67, v[6:9] offset:19968
	v_xor_b32_e32 v6, 0xf0, v50
.Ltmp107:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_or_b32_e32 v7, v24, v51
.Ltmp108:
	.loc	1 62 39                         ; matmul.py:62:39
	v_add_u32_e32 v72, 0, v6
	ds_write_b128 v72, v[2:5] offset:3840
.Ltmp109:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v7
.Ltmp110:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
.Ltmp111:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v8, v37, 11, v51
.Ltmp112:
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp113:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v22, s9, v8
	scratch_store_dword off, v7, off offset:1020 ; 4-byte Folded Spill
.Ltmp114:
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[6:7], v[22:23], 2, s[4:5]
	scratch_store_dword off, v8, off offset:1024 ; 4-byte Folded Spill
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	s_waitcnt vmcnt(4)
	ds_write_b128 v72, v[10:13] offset:20224
.Ltmp115:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v10, v25, 11, v51
	scratch_store_dword off, v10, off offset:1072 ; 4-byte Folded Spill
	scratch_store_dword off, v51, off offset:828 ; 4-byte Folded Spill
.Ltmp116:
	.loc	1 73 37                         ; matmul.py:73:37
	v_lshlrev_b32_e32 v25, 3, v0
	v_lshlrev_b32_e32 v22, 12, v82
	v_and_or_b32 v66, v25, s10, v22
	v_add_u32_e32 v114, 0, v66
	s_add_i32 s10, 0, 0xc000
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(5)
	ds_write_b128 v70, v[2:5] offset:32768
.Ltmp117:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v34, 11, v51
	scratch_store_dword off, v2, off offset:1028 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
.Ltmp118:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(3)
	ds_write_b128 v70, v[6:9] offset:33024
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp119:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v6, v36, 11, v51
	scratch_store_dword off, v6, off offset:1032 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, s9, v6
.Ltmp120:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v23
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(2)
	ds_write_b128 v70, v[2:5] offset:33280
.Ltmp121:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v35, 11, v51
	scratch_store_dword off, v2, off offset:1036 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
.Ltmp122:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[6:9] offset:33536
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp123:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v6, v33, 11, v51
	scratch_store_dword off, v6, off offset:1040 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, s9, v6
.Ltmp124:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v23
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(2)
	ds_write_b128 v70, v[2:5] offset:33792
.Ltmp125:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v32, 11, v51
	scratch_store_dword off, v2, off offset:1044 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
.Ltmp126:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[6:9] offset:34048
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp127:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v6, v31, 11, v51
	scratch_store_dword off, v6, off offset:1048 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, s9, v6
.Ltmp128:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v23
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(2)
	ds_write_b128 v70, v[2:5] offset:34304
.Ltmp129:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v38, 11, v51
	scratch_store_dword off, v2, off offset:1052 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
.Ltmp130:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[6:9] offset:34560
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp131:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v6, v39, 11, v51
	scratch_store_dword off, v6, off offset:1056 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, s9, v6
.Ltmp132:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v23
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(2)
	ds_write_b128 v70, v[2:5] offset:34816
.Ltmp133:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v40, 11, v51
	scratch_store_dword off, v2, off offset:1060 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
.Ltmp134:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[6:9] offset:35072
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp135:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v6, v26, 11, v51
	scratch_store_dword off, v6, off offset:1064 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, s9, v6
.Ltmp136:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v23
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(2)
	ds_write_b128 v70, v[2:5] offset:35328
.Ltmp137:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v27, 11, v51
	scratch_store_dword off, v2, off offset:1068 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
.Ltmp138:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[6:9] offset:35584
	global_load_dwordx4 v[2:5], v[2:3], off
.Ltmp139:
	.loc	3 55 29 is_stmt 1               ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v6, s9, v10
.Ltmp140:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v7, v23
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[6:9], v[6:7], off
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[2:5] offset:35840
.Ltmp141:
	.loc	3 50 8 is_stmt 1                ; nd_helpers.py:50:8 @[ matmul.py:56:91 ]
	v_lshl_or_b32 v2, v18, 11, v51
	v_lshl_or_b32 v3, v19, 11, v51
	scratch_store_dword off, v2, off offset:1076 ; 4-byte Folded Spill
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v2, s9, v2
	scratch_store_dword off, v3, off offset:1080 ; 4-byte Folded Spill
	v_add_u32_e32 v4, s9, v3
.Ltmp142:
	.loc	1 63 32                         ; matmul.py:63:32
	v_mov_b32_e32 v3, v23
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	s_waitcnt vmcnt(2)
	ds_write_b128 v70, v[6:9] offset:36096
	.loc	1 63 32                         ; matmul.py:63:32
	v_lshl_add_u64 v[10:11], v[2:3], 2, s[4:5]
	v_mov_b32_e32 v5, v23
	v_lshl_add_u64 v[12:13], v[4:5], 2, s[4:5]
	.loc	1 63 39                         ; matmul.py:63:39
	global_load_dwordx4 v[2:5], v[10:11], off
	global_load_dwordx4 v[6:9], v[12:13], off
	s_waitcnt vmcnt(1)
	ds_write_b128 v70, v[2:5] offset:36352
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[6:9] offset:36608
	.loc	1 62 39 is_stmt 1               ; matmul.py:62:39
	s_waitcnt lgkmcnt(0)
	; wave barrier
	ds_read_b128 v[6:9], v70
	ds_read_b128 v[2:5], v70 offset:16384
	ds_read_b128 v[10:13], v52 offset:256
	ds_read_b128 v[30:33], v53 offset:17664
	ds_read_b128 v[18:21], v49 offset:768
	ds_read_b128 v[14:17], v48 offset:512
	ds_read_b128 v[26:29], v47 offset:1024
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(4)
	ds_write_b128 v114, v[10:13] offset:57344
	v_xor_b32_e32 v10, 16, v66
	scratch_store_dword off, v10, off offset:896 ; 4-byte Folded Spill
	v_add_u32_e32 v71, s10, v10
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[10:13], v48 offset:16896
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(4)
	ds_write_b128 v71, v[18:21] offset:8704
	v_xor_b32_e32 v18, 32, v66
	v_add_u32_e32 v76, s10, v18
	ds_write_b128 v114, v[6:9] offset:49152
	s_waitcnt lgkmcnt(5)
	ds_write_b128 v71, v[14:17] offset:512
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[14:17], v49 offset:17152
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(6)
	ds_write_b128 v76, v[26:29] offset:1024
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v53 offset:1280
	ds_read_b128 v[34:37], v41 offset:17920
	scratch_store_dword off, v41, off offset:852 ; 4-byte Folded Spill
	scratch_store_dword off, v18, off offset:900 ; 4-byte Folded Spill
	ds_read_b128 v[18:21], v47 offset:17408
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(2)
	ds_write_b128 v76, v[26:29] offset:9216
	v_xor_b32_e32 v26, 48, v66
	scratch_store_dword off, v26, off offset:904 ; 4-byte Folded Spill
	v_add_u32_e32 v77, s10, v26
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v41 offset:1536
	ds_read_b128 v[38:41], v42 offset:18176
	scratch_store_dword off, v42, off offset:856 ; 4-byte Folded Spill
	scratch_store_dword off, v43, off offset:860 ; 4-byte Folded Spill
	scratch_store_dword off, v52, off offset:832 ; 4-byte Folded Spill
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v77, v[26:29] offset:1536
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v42 offset:1792
	ds_read_b128 v[6:9], v52 offset:16640
	scratch_store_dword off, v48, off offset:836 ; 4-byte Folded Spill
	scratch_store_dword off, v49, off offset:840 ; 4-byte Folded Spill
	scratch_store_dword off, v47, off offset:844 ; 4-byte Folded Spill
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v77, v[26:29] offset:9728
	v_xor_b32_e32 v26, 64, v66
	scratch_store_dword off, v26, off offset:908 ; 4-byte Folded Spill
	v_add_u32_e32 v78, s10, v26
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v43 offset:2048
	ds_read_b128 v[42:45], v43 offset:18432
	scratch_store_dword off, v53, off offset:848 ; 4-byte Folded Spill
	scratch_store_dword off, v46, off offset:864 ; 4-byte Folded Spill
	scratch_store_dword off, v54, off offset:868 ; 4-byte Folded Spill
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v78, v[26:29] offset:2048
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v46 offset:2304
	ds_read_b128 v[46:49], v46 offset:18688
	ds_read_b128 v[50:53], v54 offset:18944
	scratch_store_dword off, v55, off offset:872 ; 4-byte Folded Spill
	scratch_store_dword off, v58, off offset:876 ; 4-byte Folded Spill
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(2)
	ds_write_b128 v78, v[26:29] offset:10240
	v_xor_b32_e32 v26, 0x50, v66
	scratch_store_dword off, v26, off offset:912 ; 4-byte Folded Spill
	v_add_u32_e32 v79, s10, v26
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v54 offset:2560
	scratch_store_dword off, v62, off offset:880 ; 4-byte Folded Spill
	scratch_store_dword off, v67, off offset:884 ; 4-byte Folded Spill
	scratch_store_dword off, v72, off offset:888 ; 4-byte Folded Spill
	scratch_store_dword off, v82, off offset:1084 ; 4-byte Folded Spill
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v79, v[26:29] offset:2560
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v55 offset:2816
	ds_read_b128 v[54:57], v55 offset:19200
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v79, v[26:29] offset:10752
	v_xor_b32_e32 v26, 0x60, v66
	scratch_store_dword off, v26, off offset:916 ; 4-byte Folded Spill
	v_add_u32_e32 v80, s10, v26
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v58 offset:3072
	ds_read_b128 v[58:61], v58 offset:19456
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v80, v[26:29] offset:3072
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v62 offset:3328
	ds_read_b128 v[62:65], v62 offset:19712
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v80, v[26:29] offset:11264
	v_xor_b32_e32 v26, 0x70, v66
	scratch_store_dword off, v26, off offset:920 ; 4-byte Folded Spill
	v_add_u32_e32 v81, s10, v26
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v67 offset:3584
	ds_read_b128 v[66:69], v67 offset:19968
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v81, v[26:29] offset:3584
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[26:29], v72 offset:3840
	ds_read_b128 v[72:75], v72 offset:20224
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v81, v[26:29] offset:11776
	v_lshlrev_b32_e32 v26, 7, v0
	v_lshl_or_b32 v27, v0, 8, v25
	v_and_b32_e32 v26, 0x1000, v26
	v_and_b32_e32 v27, 0xef0, v27
	v_lshlrev_b32_e32 v28, 13, v82
	v_or3_b32 v26, v28, v26, v27
	v_xor_b32_e32 v27, 16, v26
	v_add_u32_e32 v29, 0, v27
	v_xor_b32_e32 v27, 32, v26
	v_add_u32_e32 v115, 0, v27
	v_xor_b32_e32 v27, 48, v26
	v_add_u32_e32 v116, 0, v27
	v_xor_b32_e32 v27, 64, v26
	v_add_u32_e32 v117, 0, v27
	v_xor_b32_e32 v27, 0x50, v26
	v_add_u32_e32 v28, 0, v26
	v_add_u32_e32 v118, 0, v27
	v_xor_b32_e32 v27, 0x60, v26
	v_xor_b32_e32 v26, 0x70, v26
	v_add_u32_e32 v27, 0, v27
	v_add_u32_e32 v26, 0, v26
	.loc	1 63 39                         ; matmul.py:63:39
	ds_read_b128 v[82:85], v70 offset:32768
	ds_read_b128 v[86:89], v70 offset:33024
	ds_read_b128 v[90:93], v70 offset:33280
	ds_read_b128 v[94:97], v70 offset:33536
	ds_read_b128 v[98:101], v70 offset:33792
	ds_read_b128 v[102:105], v70 offset:34048
	ds_read_b128 v[106:109], v70 offset:34304
	ds_read_b128 v[110:113], v70 offset:34560
	ds_read_b128 v[132:135], v70 offset:34816
	ds_read_b128 v[136:139], v70 offset:35072
	ds_read_b128 v[144:147], v70 offset:35328
	ds_read_b128 v[158:161], v70 offset:35584
	ds_read_b128 v[170:173], v70 offset:35840
	ds_read_b128 v[174:177], v70 offset:36096
	ds_read_b128 v[178:181], v70 offset:36352
	ds_read_b128 v[166:169], v70 offset:36608
	; wave barrier
	.loc	1 73 37                         ; matmul.py:73:37
	ds_read_b128 a[128:131], v28 offset:49152
	ds_read_b128 a[132:135], v28 offset:49408
	ds_read_b128 a[136:139], v29 offset:49152
	ds_read_b128 a[140:143], v29 offset:49408
	ds_read_b128 a[144:147], v115 offset:49152
	ds_read_b128 a[148:151], v115 offset:49408
	ds_read_b128 a[152:155], v116 offset:49152
	ds_read_b128 a[156:159], v116 offset:49408
	ds_read_b128 a[160:163], v117 offset:49152
	ds_read_b128 a[164:167], v117 offset:49408
	ds_read_b128 a[168:171], v118 offset:49152
	ds_read_b128 a[172:175], v118 offset:49408
	ds_read_b128 a[180:183], v27 offset:49152
	ds_read_b128 a[188:191], v27 offset:49408
	ds_read_b128 a[192:195], v26 offset:49152
	ds_read_b128 a[196:199], v26 offset:49408
	; wave barrier
	ds_write_b128 v114, v[2:5] offset:49152
	scratch_store_dword off, v114, off offset:892 ; 4-byte Folded Spill
	ds_write_b128 v114, v[6:9] offset:57344
	ds_write_b128 v71, v[10:13] offset:512
	ds_write_b128 v71, v[14:17] offset:8704
	ds_write_b128 v76, v[18:21] offset:1024
	ds_write_b128 v76, v[30:33] offset:9216
	ds_write_b128 v77, v[34:37] offset:1536
	ds_write_b128 v77, v[38:41] offset:9728
	ds_write_b128 v78, v[42:45] offset:2048
	ds_write_b128 v78, v[46:49] offset:10240
	ds_write_b128 v79, v[50:53] offset:2560
	ds_write_b128 v79, v[54:57] offset:10752
	ds_write_b128 v80, v[58:61] offset:3072
	ds_write_b128 v80, v[62:65] offset:11264
	ds_write_b128 v81, v[66:69] offset:3584
	s_waitcnt lgkmcnt(14)
	ds_write_b128 v81, v[72:75] offset:11776
	; wave barrier
	scratch_store_dword off, v28, off offset:924 ; 4-byte Folded Spill
	ds_read_b128 a[200:203], v28 offset:49152
	ds_read_b128 a[204:207], v28 offset:49408
	ds_read_b128 a[208:211], v29 offset:49152
	scratch_store_dword off, v29, off offset:928 ; 4-byte Folded Spill
	ds_read_b128 a[212:215], v29 offset:49408
	ds_read_b128 a[216:219], v115 offset:49152
	scratch_store_dword off, v115, off offset:932 ; 4-byte Folded Spill
	ds_read_b128 a[220:223], v115 offset:49408
	ds_read_b128 a[224:227], v116 offset:49152
	scratch_store_dword off, v116, off offset:936 ; 4-byte Folded Spill
	ds_read_b128 a[228:231], v116 offset:49408
	ds_read_b128 a[232:235], v117 offset:49152
	scratch_store_dword off, v117, off offset:940 ; 4-byte Folded Spill
	ds_read_b128 a[236:239], v117 offset:49408
	ds_read_b128 a[240:243], v118 offset:49152
	scratch_store_dword off, v118, off offset:944 ; 4-byte Folded Spill
	.loc	1 74 37                         ; matmul.py:74:37
	v_lshlrev_b32_e32 v2, 4, v0
	.loc	1 73 37                         ; matmul.py:73:37
	ds_read_b128 a[244:247], v118 offset:49408
	ds_read_b128 a[248:251], v27 offset:49152
	scratch_store_dword off, v27, off offset:948 ; 4-byte Folded Spill
	.loc	1 74 37                         ; matmul.py:74:37
	v_and_b32_e32 v3, 0x330, v2
	v_and_b32_e32 v2, 4, v0
	.loc	1 73 37                         ; matmul.py:73:37
	ds_read_b128 a[252:255], v27 offset:49408
	ds_read_b128 v[240:243], v26 offset:49152
	scratch_store_dword off, v26, off offset:952 ; 4-byte Folded Spill
	ds_read_b128 v[248:251], v26 offset:49408
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt lgkmcnt(0)
	; wave barrier
	v_and_b32_e32 v4, 64, v25
	scratch_store_dword off, v2, off offset:1096 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v2, 9, v2
	v_or3_b32 v6, v2, v4, v3
	scratch_store_dword off, v3, off offset:1088 ; 4-byte Folded Spill
	scratch_store_dword off, v4, off offset:1092 ; 4-byte Folded Spill
	v_mov_b32_e32 v2, v82
	v_mov_b32_e32 v3, v86
	v_mov_b32_e32 v4, v90
	v_mov_b32_e32 v5, v94
	v_add_u32_e32 v140, 0, v6
	ds_write_b128 v140, v[2:5] offset:49152
	v_mov_b32_e32 v2, v83
	v_mov_b32_e32 v3, v87
	v_mov_b32_e32 v4, v91
	v_mov_b32_e32 v5, v95
	ds_write_b128 v140, v[2:5] offset:53248
	v_mov_b32_e32 v2, v98
	v_mov_b32_e32 v3, v102
	v_mov_b32_e32 v4, v106
	v_mov_b32_e32 v5, v110
	ds_write_b128 v140, v[2:5] offset:49280
	v_mov_b32_e32 v2, v99
	v_mov_b32_e32 v3, v103
	v_mov_b32_e32 v4, v107
	v_mov_b32_e32 v5, v111
	v_xor_b32_e32 v6, 64, v6
	ds_write_b128 v140, v[2:5] offset:53376
	v_mov_b32_e32 v2, v84
	v_mov_b32_e32 v3, v88
	v_mov_b32_e32 v4, v92
	v_mov_b32_e32 v5, v96
	scratch_store_dword off, v6, off offset:116 ; 4-byte Folded Spill
	v_add_u32_e32 v6, s10, v6
	ds_write_b128 v6, v[2:5] offset:1024
	v_mov_b32_e32 v2, v100
	v_mov_b32_e32 v3, v104
	v_mov_b32_e32 v4, v108
	v_mov_b32_e32 v5, v112
	ds_write_b128 v6, v[2:5] offset:1152
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v3, 0xb0, v2
	v_bfe_i32 v2, v0, 1, 1
	v_mov_b32_e32 v94, v85
	v_mov_b32_e32 v95, v89
	v_mov_b32_e32 v96, v93
	v_mov_b32_e32 v110, v101
	v_mov_b32_e32 v111, v105
	v_mov_b32_e32 v112, v109
	v_and_b32_e32 v2, 0x440, v2
	v_and_b32_e32 v0, 16, v0
	ds_write_b128 v6, v[94:97] offset:5120
	ds_write_b128 v6, v[110:113] offset:5248
	scratch_store_dword off, v2, off offset:1104 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:1108 ; 4-byte Folded Spill
	v_lshl_or_b32 v0, v0, 7, v2
	v_or3_b32 v0, v0, v22, v3
	v_add_u32_e32 v191, 0, v0
	v_xor_b32_e32 v0, 64, v0
	scratch_store_dword off, v3, off offset:1100 ; 4-byte Folded Spill
	; wave barrier
	ds_read_b128 v[54:57], v191 offset:49152
	ds_read_b128 v[58:61], v191 offset:49408
	ds_read_b128 v[106:109], v191 offset:49664
	ds_read_b128 v[112:115], v191 offset:49920
	v_add_u32_e32 v212, 0, v0
	ds_read_b128 v[116:119], v212 offset:49152
	ds_read_b128 v[122:125], v212 offset:49408
	ds_read_b128 v[128:131], v212 offset:49664
	v_lshl_add_u32 v0, s0, 6, v24
	s_lshl_b32 s0, s8, 11
	v_subrev_u32_e32 v252, s0, v0
	v_accvgpr_write_b32 a184, v132
	v_accvgpr_write_b32 a185, v136
	v_accvgpr_write_b32 a186, v144
	v_accvgpr_write_b32 a187, v158
	v_mov_b32_e32 v150, v133
	v_mov_b32_e32 v151, v137
	v_mov_b32_e32 v152, v145
	v_mov_b32_e32 v153, v159
	v_accvgpr_write_b32 a176, v170
	v_accvgpr_write_b32 a177, v174
	v_accvgpr_write_b32 a178, v178
	v_accvgpr_write_b32 a179, v166
	v_mov_b32_e32 v142, v171
	v_mov_b32_e32 v143, v175
	v_mov_b32_e32 v144, v179
	v_mov_b32_e32 v145, v167
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v138
	v_mov_b32_e32 v156, v146
	v_mov_b32_e32 v157, v160
	v_mov_b32_e32 v158, v135
	v_mov_b32_e32 v159, v139
	v_mov_b32_e32 v160, v147
	v_mov_b32_e32 v162, v172
	v_mov_b32_e32 v163, v176
	v_mov_b32_e32 v164, v180
	v_mov_b32_e32 v165, v168
	v_mov_b32_e32 v166, v173
	v_mov_b32_e32 v167, v177
	v_mov_b32_e32 v168, v181
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 75 42                         ; matmul.py:75:42
	v_accvgpr_write_b32 a94, v253
	v_accvgpr_write_b32 a95, v251
	v_mfma_f32_32x32x2_f32 a[64:79], v220, v232, a[64:79]
	v_accvgpr_write_b32 a32, v155
	v_accvgpr_write_b32 a33, v154
	v_accvgpr_write_b32 a34, v153
	v_accvgpr_write_b32 a35, v152
	v_accvgpr_write_b32 a36, v151
	v_accvgpr_write_b32 a37, v150
	v_accvgpr_write_b32 a38, v145
	v_accvgpr_write_b32 a39, v144
	v_accvgpr_write_b32 a40, v143
	v_accvgpr_write_b32 a41, v142
	v_accvgpr_write_b32 a111, v58
	v_accvgpr_write_b32 a110, v106
	v_accvgpr_write_b32 a109, v112
	v_accvgpr_write_b32 a108, v116
	v_accvgpr_write_b32 a107, v122
	v_mfma_f32_32x32x2_f32 a[80:95], v220, v185, a[80:95]
	v_accvgpr_write_b32 a103, v128
	v_accvgpr_write_b32 a31, v156
	v_accvgpr_write_b32 a27, v160
	v_accvgpr_write_b32 a23, v164
	v_accvgpr_write_b32 a19, v168
	v_accvgpr_write_b32 a18, v169
	v_accvgpr_write_b32 a127, v54
	v_accvgpr_write_b32 a96, v250
	v_accvgpr_write_b32 a97, v249
	v_accvgpr_write_b32 a98, v248
	v_accvgpr_write_b32 a99, v247
	v_accvgpr_write_b32 a100, v131
	v_accvgpr_write_b32 a101, v130
	v_accvgpr_write_b32 a102, v129
	v_accvgpr_write_b32 a104, v125
	v_mfma_f32_32x32x2_f32 a[64:79], v219, v231, a[64:79]
	v_accvgpr_write_b32 a105, v124
	v_accvgpr_write_b32 a106, v123
	v_accvgpr_write_b32 a201, v211
	v_accvgpr_write_b32 a112, v119
	v_accvgpr_write_b32 a113, v118
	v_accvgpr_write_b32 a114, v117
	v_accvgpr_write_b32 a115, v115
	v_accvgpr_write_b32 a116, v114
	v_accvgpr_write_b32 a117, v113
	v_accvgpr_write_b32 a118, v109
	v_accvgpr_write_b32 a119, v108
	v_accvgpr_write_b32 a120, v107
	v_accvgpr_write_b32 a121, v61
	v_accvgpr_write_b32 a122, v60
	v_accvgpr_write_b32 a123, v59
	v_mfma_f32_32x32x2_f32 a[80:95], v219, v184, a[80:95]
	v_accvgpr_write_b32 a124, v57
	v_accvgpr_write_b32 a125, v56
	v_accvgpr_write_b32 a126, v55
	v_accvgpr_write_b32 a200, v74
	v_accvgpr_write_b32 a20, v167
	v_accvgpr_write_b32 a21, v166
	v_accvgpr_write_b32 a22, v165
	v_accvgpr_write_b32 a24, v163
	v_accvgpr_write_b32 a25, v162
	v_accvgpr_write_b32 a26, v161
	v_accvgpr_write_b32 a28, v159
	v_accvgpr_write_b32 a29, v158
	v_accvgpr_write_b32 a30, v157
	.loc	1 74 37                         ; matmul.py:74:37
	v_add_u32_e32 v205, -1, v205
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[64:79], v218, v229, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v218, v183, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v216, v227, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v216, v182, a[80:95]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x2_f32 a[64:79], v215, v32, a[64:79]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[80:95], v215, v16, a[80:95]
	v_mov_b32_e32 v215, v37
	v_mfma_f32_32x32x2_f32 a[64:79], v213, v33, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v213, v17, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v48, v232, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v48, v185, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v208, v34, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v208, v18, a[80:95]
	v_mov_b32_e32 v208, v44
	v_mfma_f32_32x32x2_f32 a[32:47], v90, v231, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v90, v184, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v170, v35, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v170, v19, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v201, v229, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v201, v183, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v149, v217, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v149, v181, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v254, v227, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v254, v182, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v148, v214, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v148, v180, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v110, v32, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v110, v16, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v141, v209, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v141, v179, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v49, v33, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v49, v17, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v139, v206, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v139, v178, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v222, v34, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v222, v18, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v138, v28, a[64:79]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x2_f32 a[80:95], v138, v12, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v147, v35, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v147, v19, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v137, v29, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v137, v13, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v102, v217, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v102, v181, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v136, v30, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v136, v14, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v199, v214, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v199, v180, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v135, v31, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v135, v15, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], a150, v209, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], a150, v179, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v134, v196, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v134, v177, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], a151, v206, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], a151, v178, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v133, v193, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v133, v176, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v246, v28, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v246, v12, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v132, v192, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v132, v174, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v245, v29, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v245, v13, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v101, v190, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v101, v172, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v244, v30, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v244, v14, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v100, v24, a[64:79]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[80:95], v100, v8, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v243, v31, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v243, v15, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v99, v25, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v99, v9, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v242, v196, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v242, v177, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v98, v26, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v98, v10, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v241, v193, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v241, v176, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v97, v27, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v97, v11, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v240, v192, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v240, v174, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v96, v189, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v96, v4, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v239, v190, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v239, v172, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v95, v188, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v95, v5, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v238, v24, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v238, v8, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v53, v187, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v53, v6, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v237, v25, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v237, v9, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v52, v186, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v52, v7, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v236, v26, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v236, v10, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v51, v20, a[64:79]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x2_f32 a[80:95], v51, v0, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v235, v27, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v235, v11, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v50, v21, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v50, v1, a[80:95]
	v_mfma_f32_32x32x2_f32 a[32:47], v234, v189, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v234, v4, a[48:63]
	v_mfma_f32_32x32x2_f32 a[64:79], v36, v22, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v36, v2, a[80:95]
	scratch_load_dword v36, off, off offset:828 ; 4-byte Folded Reload
.Ltmp143:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v54, v36, v252
.Ltmp144:
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v233, v188, a[32:47]
	.loc	1 74 37                         ; matmul.py:74:37
	v_add_u32_e32 v252, 0x20000, v252
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v233, v5, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v230, v187, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v230, v6, a[48:63]
	v_mov_b32_e32 v230, v40
	v_mfma_f32_32x32x2_f32 a[32:47], v228, v186, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v228, v7, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v226, v20, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v226, v0, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v225, v21, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v225, v1, a[48:63]
.Ltmp145:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v225, v36, v255
	v_add_u32_e32 v50, 64, v225
.Ltmp146:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 74 37 is_stmt 1               ; matmul.py:74:37
	v_add_u32_e32 v255, 64, v255
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53]
.Ltmp147:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x200040, v225
.Ltmp148:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v93, v232, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:16384
.Ltmp149:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x8040, v225
.Ltmp150:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v58, off, off offset:832 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v89, v231, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v58, v[50:53] offset:256
.Ltmp151:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x208040, v225
.Ltmp152:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v88, v229, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v58, v[50:53] offset:16640
.Ltmp153:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x10040, v225
.Ltmp154:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v106, off, off offset:836 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v87, v227, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[50:53] offset:512
.Ltmp155:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x210040, v225
.Ltmp156:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v86, v32, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[50:53] offset:16896
.Ltmp157:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x18040, v225
.Ltmp158:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v112, off, off offset:840 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v80, v33, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v112, v[50:53] offset:768
.Ltmp159:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x218040, v225
.Ltmp160:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v79, v34, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v112, v[50:53] offset:17152
.Ltmp161:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x20040, v225
.Ltmp162:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v116, off, off offset:844 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v77, v35, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v116, v[50:53] offset:1024
.Ltmp163:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x220040, v225
.Ltmp164:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v76, v217, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v116, v[50:53] offset:17408
.Ltmp165:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x28040, v225
.Ltmp166:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v122, off, off offset:848 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v69, v214, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v122, v[50:53] offset:1280
.Ltmp167:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x228040, v225
.Ltmp168:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v68, v209, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v122, v[50:53] offset:17664
.Ltmp169:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x30040, v225
.Ltmp170:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v128, off, off offset:852 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v67, v206, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v128, v[50:53] offset:1536
.Ltmp171:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x230040, v225
.Ltmp172:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v66, v28, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v128, v[50:53] offset:17920
.Ltmp173:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x38040, v225
.Ltmp174:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v142, off, off offset:856 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v65, v29, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v142, v[50:53] offset:1792
.Ltmp175:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x238040, v225
.Ltmp176:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v64, v30, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v142, v[50:53] offset:18176
.Ltmp177:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x40040, v225
.Ltmp178:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v156, off, off offset:860 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v63, v31, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v156, v[50:53] offset:2048
.Ltmp179:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x240040, v225
.Ltmp180:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v62, v196, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v156, v[50:53] offset:18432
.Ltmp181:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x48040, v225
.Ltmp182:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v160, off, off offset:864 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v47, v193, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v160, v[50:53] offset:2304
.Ltmp183:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x248040, v225
.Ltmp184:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v46, v192, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v160, v[50:53] offset:18688
.Ltmp185:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x50040, v225
.Ltmp186:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v164, off, off offset:868 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v45, v190, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v164, v[50:53] offset:2560
.Ltmp187:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x250040, v225
.Ltmp188:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v43, v24, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v164, v[50:53] offset:18944
.Ltmp189:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x58040, v225
.Ltmp190:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v168, off, off offset:872 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v42, v25, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v168, v[50:53] offset:2816
.Ltmp191:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x258040, v225
.Ltmp192:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v171, v26, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v168, v[50:53] offset:19200
.Ltmp193:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x60040, v225
.Ltmp194:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v169, off, off offset:876 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v210, v27, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v169, v[50:53] offset:3072
.Ltmp195:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x260040, v225
.Ltmp196:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v211, v189, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v169, v[50:53] offset:19456
.Ltmp197:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x68040, v225
.Ltmp198:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v170, off, off offset:880 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v93, v185, a[112:127]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v170, v[50:53] offset:3328
.Ltmp199:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x268040, v225
.Ltmp200:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v89, v184, a[112:127]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v170, v[50:53] offset:19712
.Ltmp201:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x70040, v225
.Ltmp202:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v218, off, off offset:884 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v88, v183, a[112:127]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v218, v[50:53] offset:3584
.Ltmp203:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x270040, v225
.Ltmp204:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v87, v182, a[112:127]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v218, v[50:53] offset:19968
.Ltmp205:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:55:91 ]
	v_add_u32_e32 v50, 0x78040, v225
.Ltmp206:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	s_nop 0
	scratch_load_dword v234, off, off offset:888 ; 4-byte Folded Reload
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v86, v16, a[112:127]
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v234, v[50:53] offset:3840
.Ltmp207:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x278040, v225
.Ltmp208:
	.loc	1 62 32                         ; matmul.py:62:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[2:3]
	.loc	1 62 39 is_stmt 0               ; matmul.py:62:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v80, v17, a[112:127]
	v_mov_b32_e32 v80, v41
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v234, v[50:53] offset:20224
.Ltmp209:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x20000, v54
.Ltmp210:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v79, v18, a[112:127]
	v_mov_b32_e32 v79, v39
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:32768
.Ltmp211:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x20800, v54
.Ltmp212:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v77, v19, a[112:127]
	v_mov_b32_e32 v77, v38
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:33024
.Ltmp213:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x21000, v54
.Ltmp214:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v76, v181, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:33280
.Ltmp215:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x21800, v54
.Ltmp216:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v69, v180, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:33536
.Ltmp217:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x22000, v54
.Ltmp218:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v68, v179, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:33792
.Ltmp219:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x22800, v54
.Ltmp220:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v67, v178, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:34048
.Ltmp221:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x23000, v54
.Ltmp222:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v66, v12, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:34304
.Ltmp223:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x23800, v54
.Ltmp224:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v65, v13, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:34560
.Ltmp225:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x24000, v54
.Ltmp226:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v64, v14, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:34816
.Ltmp227:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x24800, v54
.Ltmp228:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v63, v15, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:35072
.Ltmp229:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x25000, v54
.Ltmp230:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v74, v188, a[96:111]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:35328
.Ltmp231:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x25800, v54
.Ltmp232:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v62, v177, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:35584
.Ltmp233:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x26000, v54
.Ltmp234:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v44, v187, a[96:111]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:35840
.Ltmp235:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x26800, v54
.Ltmp236:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v47, v176, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:36096
.Ltmp237:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:56:91 ]
	v_add_u32_e32 v50, 0x27000, v54
.Ltmp238:
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v41, v186, a[96:111]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:36352
	.loc	1 62 32                         ; matmul.py:62:32
	v_add_u32_e32 v50, 0x27800, v54
	.loc	1 63 32                         ; matmul.py:63:32
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[4:5]
	.loc	1 63 39 is_stmt 0               ; matmul.py:63:39
	global_load_dwordx4 v[50:53], v[50:51], off
	.loc	1 75 42 is_stmt 1               ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v46, v174, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	s_waitcnt vmcnt(0)
	ds_write_b128 v70, v[50:53] offset:36608
	.loc	1 62 39                         ; matmul.py:62:39
	s_waitcnt lgkmcnt(0)
	; wave barrier
	scratch_load_dword v211, off, off offset:892 ; 4-byte Folded Reload
	.loc	1 73 37                         ; matmul.py:73:37
	scratch_load_dword v36, off, off offset:896 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v40, v20, a[96:111]
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v70
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v225, 0, v36
	scratch_load_dword v36, off, off offset:900 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v45, v172, a[112:127]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v216, 0, v36
	scratch_load_dword v36, off, off offset:904 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v39, v21, a[96:111]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v141, 0, v36
	scratch_load_dword v36, off, off offset:908 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v43, v8, a[112:127]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v253, 0, v36
	scratch_load_dword v36, off, off offset:912 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v38, v22, a[96:111]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v93, 0, v36
	scratch_load_dword v36, off, off offset:916 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v42, v9, a[112:127]
	.loc	1 63 39                         ; matmul.py:63:39
	ds_read_b128 v[40:43], v70 offset:33024
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v211, v[50:53] offset:49152
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v58 offset:256
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v74, 0, v36
	scratch_load_dword v36, off, off offset:920 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v37, v23, a[96:111]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v76, 0, v36
	.loc	1 63 39                         ; matmul.py:63:39
	ds_read_b128 v[36:39], v70 offset:32768
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(1)
	ds_write_b128 v211, v[50:53] offset:57344
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v106 offset:512
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v224, v22, a[32:47]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v225, v[50:53] offset:49664
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v112 offset:768
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v225, v[50:53] offset:57856
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v116 offset:1024
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v216, v[50:53] offset:50176
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v122 offset:1280
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v224, v2, a[48:63]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v216, v[50:53] offset:58368
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v128 offset:1536
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v141, v[50:53] offset:50688
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v142 offset:1792
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v141, v[50:53] offset:58880
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v156 offset:2048
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[32:47], v221, v23, a[32:47]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v253, v[50:53] offset:51200
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v160 offset:2304
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v253, v[50:53] offset:59392
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v164 offset:2560
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v93, v[50:53] offset:51712
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v168 offset:2816
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[48:63], v221, v3, a[48:63]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v93, v[50:53] offset:59904
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v169 offset:3072
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v74, v[50:53] offset:52224
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v170 offset:3328
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v74, v[50:53] offset:60416
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v218 offset:3584
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[64:79], v94, v23, a[64:79]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v76, v[50:53] offset:52736
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[50:53], v234 offset:3840
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(0)
	ds_write_b128 v76, v[50:53] offset:60928
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[80:95], v94, v3, a[80:95]
	.loc	1 63 39                         ; matmul.py:63:39
	ds_read_b128 v[44:47], v70 offset:33280
	ds_read_b128 v[62:65], v70 offset:33536
	ds_read_b128 v[66:69], v70 offset:33792
	ds_read_b128 v[86:89], v70 offset:34048
	ds_read_b128 v[94:97], v70 offset:34304
	ds_read_b128 v[98:101], v70 offset:34560
	ds_read_b128 a[184:187], v70 offset:34816
	ds_read_b128 v[148:151], v70 offset:35072
	ds_read_b128 v[136:139], v70 offset:35328
	ds_read_b128 v[50:53], v70 offset:35584
	ds_read_b128 a[176:179], v70 offset:35840
	ds_read_b128 v[152:155], v70 offset:36096
	ds_read_b128 v[132:135], v70 offset:36352
	ds_read_b128 v[244:247], v70 offset:36608
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v171, v10, a[112:127]
	.loc	1 62 39                         ; matmul.py:62:39
	ds_read_b128 v[54:57], v70 offset:16384
	ds_read_b128 v[58:61], v58 offset:16640
	ds_read_b128 v[106:109], v106 offset:16896
	ds_read_b128 v[112:115], v112 offset:17152
	ds_read_b128 v[116:119], v116 offset:17408
	ds_read_b128 v[122:125], v122 offset:17664
	ds_read_b128 v[128:131], v128 offset:17920
	ds_read_b128 v[142:145], v142 offset:18176
	ds_read_b128 v[156:159], v156 offset:18432
	ds_read_b128 v[160:163], v160 offset:18688
	ds_read_b128 v[164:167], v164 offset:18944
	ds_read_b128 v[240:243], v168 offset:19200
	ds_read_b128 v[248:251], v169 offset:19456
	ds_read_b128 v[168:171], v170 offset:19712
	ds_read_b128 v[218:221], v218 offset:19968
	ds_read_b128 v[234:237], v234 offset:20224
	; wave barrier
	scratch_load_dword v224, off, off offset:924 ; 4-byte Folded Reload
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	ds_read_b128 a[128:131], v224 offset:49152
	ds_read_b128 a[132:135], v224 offset:49408
	scratch_load_dword v226, off, off offset:928 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 a[136:139], v226 offset:49152
	ds_read_b128 a[140:143], v226 offset:49408
	scratch_load_dword v228, off, off offset:932 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 a[144:147], v228 offset:49152
	ds_read_b128 a[148:151], v228 offset:49408
	scratch_load_dword v233, off, off offset:936 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v203, v232, a[0:15]
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	ds_read_b128 a[152:155], v233 offset:49152
	ds_read_b128 a[156:159], v233 offset:49408
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v203, v185, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v202, v231, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v202, v184, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v200, v229, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v200, v183, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v198, v227, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v198, v182, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v210, v11, a[112:127]
	scratch_load_dword v210, off, off offset:940 ; 4-byte Folded Reload
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	ds_read_b128 a[160:163], v210 offset:49152
	ds_read_b128 a[164:167], v210 offset:49408
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v197, v32, a[0:15]
	scratch_load_dword v213, off, off offset:944 ; 4-byte Folded Reload
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	ds_read_b128 a[168:171], v213 offset:49152
	ds_read_b128 a[172:175], v213 offset:49408
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v197, v16, a[16:31]
	scratch_load_dword v238, off, off offset:948 ; 4-byte Folded Reload
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	ds_read_b128 a[180:183], v238 offset:49152
	ds_read_b128 a[188:191], v238 offset:49408
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v194, v33, a[0:15]
	scratch_load_dword v239, off, off offset:952 ; 4-byte Folded Reload
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt vmcnt(0)
	ds_read_b128 a[192:195], v239 offset:49152
	ds_read_b128 a[196:199], v239 offset:49408
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v194, v17, a[16:31]
	; wave barrier
	.loc	1 73 37                         ; matmul.py:73:37
	s_waitcnt lgkmcnt(14)
	ds_write_b128 v211, v[54:57] offset:49152
	ds_write_b128 v211, v[58:61] offset:57344
	v_accvgpr_read_b32 v211, a201
	ds_write_b128 v225, v[106:109] offset:49664
	ds_write_b128 v225, v[112:115] offset:57856
	ds_write_b128 v216, v[116:119] offset:50176
	ds_write_b128 v216, v[122:125] offset:58368
	ds_write_b128 v141, v[128:131] offset:50688
	ds_write_b128 v141, v[142:145] offset:58880
	ds_write_b128 v253, v[156:159] offset:51200
	ds_write_b128 v253, v[160:163] offset:59392
	ds_write_b128 v93, v[164:167] offset:51712
	ds_write_b128 v93, v[240:243] offset:59904
	ds_write_b128 v74, v[248:251] offset:52224
	ds_write_b128 v74, v[168:171] offset:60416
	v_accvgpr_read_b32 v74, a200
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v56, v44
	v_mov_b32_e32 v44, v208
	v_mov_b32_e32 v54, v36
	v_mov_b32_e32 v55, v40
	v_mov_b32_e32 v57, v62
	.loc	1 73 37                         ; matmul.py:73:37
	ds_write_b128 v76, v[218:221] offset:52736
	ds_write_b128 v76, v[234:237] offset:60928
	; wave barrier
	ds_read_b128 a[200:203], v224 offset:49152
	ds_read_b128 a[204:207], v224 offset:49408
	ds_read_b128 a[208:211], v226 offset:49152
	ds_read_b128 a[212:215], v226 offset:49408
	ds_read_b128 a[216:219], v228 offset:49152
	ds_read_b128 a[220:223], v228 offset:49408
	ds_read_b128 a[224:227], v233 offset:49152
	ds_read_b128 a[228:231], v233 offset:49408
	ds_read_b128 a[232:235], v210 offset:49152
	ds_read_b128 a[236:239], v210 offset:49408
	ds_read_b128 a[240:243], v213 offset:49152
	ds_read_b128 a[244:247], v213 offset:49408
	ds_read_b128 a[248:251], v238 offset:49152
	ds_read_b128 a[252:255], v238 offset:49408
	ds_read_b128 v[240:243], v239 offset:49152
	ds_read_b128 v[248:251], v239 offset:49408
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt lgkmcnt(0)
	; wave barrier
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v126, v34, a[0:15]
	.loc	1 74 37                         ; matmul.py:74:37
	ds_write_b128 v140, v[54:57] offset:49152
	v_mov_b32_e32 v55, v41
	v_mov_b32_e32 v41, v80
	v_mov_b32_e32 v40, v230
	v_mov_b32_e32 v62, v39
	v_mov_b32_e32 v39, v79
	v_mov_b32_e32 v54, v37
	v_mov_b32_e32 v56, v45
	v_mov_b32_e32 v57, v63
	ds_write_b128 v140, v[54:57] offset:53248
	v_mov_b32_e32 v54, v66
	v_mov_b32_e32 v55, v86
	v_mov_b32_e32 v56, v94
	v_mov_b32_e32 v57, v98
	ds_write_b128 v140, v[54:57] offset:49280
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v126, v18, a[16:31]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v54, v67
	v_mov_b32_e32 v55, v87
	v_mov_b32_e32 v56, v95
	v_mov_b32_e32 v57, v99
	ds_write_b128 v140, v[54:57] offset:53376
	v_mov_b32_e32 v54, v38
	v_mov_b32_e32 v38, v77
	v_mov_b32_e32 v37, v215
	v_mov_b32_e32 v55, v42
	v_mov_b32_e32 v56, v46
	v_mov_b32_e32 v57, v64
	ds_write_b128 v223, v[54:57] offset:50176
	v_mov_b32_e32 v54, v68
	v_mov_b32_e32 v55, v88
	v_mov_b32_e32 v56, v96
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[0:15], v121, v35, a[0:15]
	.loc	1 74 37                         ; matmul.py:74:37
	v_mov_b32_e32 v57, v100
	v_mov_b32_e32 v63, v43
	v_mov_b32_e32 v64, v47
	v_mov_b32_e32 v98, v69
	v_mov_b32_e32 v99, v89
	v_mov_b32_e32 v100, v97
	ds_write_b128 v223, v[54:57] offset:50304
	ds_write_b128 v223, v[62:65] offset:54272
	ds_write_b128 v223, v[98:101] offset:54400
	; wave barrier
	ds_read_b128 v[54:57], v191 offset:49152
	ds_read_b128 v[58:61], v191 offset:49408
	ds_read_b128 v[106:109], v191 offset:49664
	ds_read_b128 v[112:115], v191 offset:49920
	ds_read_b128 v[116:119], v212 offset:49152
	ds_read_b128 v[122:125], v212 offset:49408
	ds_read_b128 v[128:131], v212 offset:49664
	v_mov_b32_e32 v169, v247
	v_mov_b32_e32 v168, v135
	v_mov_b32_e32 v167, v155
	v_accvgpr_read_b32 v166, a179
	v_mov_b32_e32 v165, v246
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v121, v19, a[16:31]
	v_mov_b32_e32 v164, v134
	v_mov_b32_e32 v163, v154
	v_accvgpr_read_b32 v162, a178
	v_mov_b32_e32 v161, v53
	v_mov_b32_e32 v160, v139
	v_mov_b32_e32 v159, v151
	v_accvgpr_read_b32 v158, a187
	v_mov_b32_e32 v157, v52
	v_mov_b32_e32 v156, v138
	v_mov_b32_e32 v155, v150
	v_accvgpr_read_b32 v154, a186
	v_mov_b32_e32 v145, v245
	v_mov_b32_e32 v144, v133
	v_mov_b32_e32 v143, v153
	v_accvgpr_read_b32 v142, a177
	v_mfma_f32_32x32x2_f32 a[0:15], v111, v217, a[0:15]
	v_accvgpr_write_b32 a179, v244
	v_accvgpr_write_b32 a178, v132
	v_accvgpr_write_b32 a177, v152
	v_mov_b32_e32 v153, v51
	v_mov_b32_e32 v152, v137
	v_mov_b32_e32 v151, v149
	v_accvgpr_read_b32 v150, a185
	v_accvgpr_write_b32 a187, v50
	v_accvgpr_write_b32 a186, v136
	v_accvgpr_write_b32 a185, v148
	v_mfma_f32_32x32x2_f32 a[16:31], v111, v181, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v105, v214, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v105, v180, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v104, v209, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v104, v179, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v103, v206, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v103, v178, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v92, v28, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v92, v12, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v85, v29, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v85, v13, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v84, v30, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v84, v14, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v83, v31, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v83, v15, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v82, v196, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v82, v177, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v81, v193, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v81, v176, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v78, v192, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v78, v174, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v75, v190, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v75, v172, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v72, v24, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v72, v8, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v146, v25, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v146, v9, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v207, v26, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v207, v10, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v195, v27, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v195, v11, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v91, v189, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v91, v4, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v211, v4, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v71, v188, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v71, v5, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v74, v5, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v175, v187, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v175, v6, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v44, v6, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v127, v186, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v127, v7, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v41, v7, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v204, v20, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v204, v0, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v40, v0, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v173, v21, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v173, v1, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v39, v1, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v73, v22, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v73, v2, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v38, v2, a[112:127]
	v_mfma_f32_32x32x2_f32 a[0:15], v120, v23, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v120, v3, a[16:31]
	v_mfma_f32_32x32x2_f32 a[112:127], v37, v3, a[112:127]
	s_cbranch_execz .LBB0_4
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 74 37                         ; matmul.py:74:37
	ds_read_b128 v[4:7], v212 offset:49920
	; wave barrier
	ds_write_b128 v140, a[184:187] offset:49152
	ds_write_b128 v140, v[150:153] offset:53248
	ds_write_b128 v140, a[176:179] offset:49280
	ds_write_b128 v140, v[142:145] offset:53376
	s_waitcnt lgkmcnt(5)
	scratch_load_dword v0, off, off offset:116 ; 4-byte Folded Reload
	.loc	1 50 25                         ; matmul.py:50:25
	v_cmp_eq_u32_e32 vcc, 0, v205
	v_mov_b32_e32 v37, v251
	v_mov_b32_e32 v38, v250
	v_mov_b32_e32 v39, v249
	v_mov_b32_e32 v40, v248
	v_accvgpr_read_b32 v41, a255
	v_accvgpr_read_b32 v44, a254
	v_accvgpr_read_b32 v74, a253
	v_accvgpr_read_b32 v211, a252
	v_accvgpr_read_b32 v210, a247
	v_accvgpr_read_b32 v171, a246
	v_accvgpr_read_b32 v42, a245
	v_accvgpr_read_b32 v43, a244
	v_accvgpr_read_b32 v45, a239
	v_accvgpr_read_b32 v46, a238
	v_accvgpr_read_b32 v47, a237
	v_accvgpr_read_b32 v62, a236
	v_accvgpr_read_b32 v63, a231
	v_accvgpr_read_b32 v64, a230
	v_accvgpr_read_b32 v65, a229
	v_accvgpr_read_b32 v66, a228
	v_accvgpr_read_b32 v67, a223
	v_accvgpr_read_b32 v68, a222
	v_accvgpr_read_b32 v69, a221
	v_accvgpr_read_b32 v76, a220
	v_accvgpr_read_b32 v77, a215
	v_accvgpr_read_b32 v79, a214
	v_accvgpr_read_b32 v80, a213
	v_accvgpr_read_b32 v86, a212
	v_accvgpr_read_b32 v87, a207
	v_accvgpr_read_b32 v88, a206
	v_accvgpr_read_b32 v89, a205
	v_accvgpr_read_b32 v93, a204
	v_mov_b32_e32 v94, v243
	v_mov_b32_e32 v36, v242
	v_mov_b32_e32 v50, v241
	v_mov_b32_e32 v51, v240
	v_accvgpr_read_b32 v52, a251
	v_accvgpr_read_b32 v53, a250
	v_accvgpr_read_b32 v95, a249
	v_accvgpr_read_b32 v96, a248
	v_accvgpr_read_b32 v97, a243
	v_accvgpr_read_b32 v98, a242
	v_accvgpr_read_b32 v99, a241
	v_accvgpr_read_b32 v100, a240
	v_accvgpr_read_b32 v101, a235
	v_accvgpr_read_b32 v132, a234
	v_accvgpr_read_b32 v133, a233
	v_accvgpr_read_b32 v134, a232
	v_accvgpr_read_b32 v135, a227
	v_accvgpr_read_b32 v136, a226
	v_accvgpr_read_b32 v137, a225
	v_accvgpr_read_b32 v138, a224
	v_accvgpr_read_b32 v139, a219
	v_accvgpr_read_b32 v141, a218
	v_accvgpr_read_b32 v148, a217
	v_accvgpr_read_b32 v149, a216
	v_accvgpr_read_b32 v170, a211
	v_accvgpr_read_b32 v208, a210
	v_accvgpr_read_b32 v213, a209
	v_accvgpr_read_b32 v215, a208
	v_accvgpr_read_b32 v216, a203
	v_accvgpr_read_b32 v218, a202
	v_accvgpr_read_b32 v219, a201
	v_accvgpr_read_b32 v220, a200
	v_accvgpr_read_b32 v221, a199
	v_accvgpr_read_b32 v224, a198
	v_accvgpr_read_b32 v225, a197
	v_accvgpr_read_b32 v226, a196
	v_accvgpr_read_b32 v228, a191
	v_accvgpr_read_b32 v230, a190
	v_accvgpr_read_b32 v233, a189
	v_accvgpr_read_b32 v234, a188
	v_accvgpr_read_b32 v235, a175
	v_accvgpr_read_b32 v236, a174
	v_accvgpr_read_b32 v237, a173
	v_accvgpr_read_b32 v238, a172
	v_accvgpr_read_b32 v239, a167
	v_accvgpr_read_b32 v240, a166
	.loc	1 74 37                         ; matmul.py:74:37
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v223, 0, v0
	ds_write_b128 v223, v[154:157] offset:50176
	ds_write_b128 v223, v[158:161] offset:54272
	ds_write_b128 v223, v[162:165] offset:50304
	ds_write_b128 v223, v[166:169] offset:54400
	; wave barrier
	ds_read_b128 v[32:35], v191 offset:49152
	ds_read_b128 v[28:31], v191 offset:49408
	ds_read_b128 v[24:27], v191 offset:49664
	ds_read_b128 v[20:23], v191 offset:49920
	ds_read_b128 v[16:19], v212 offset:49152
	ds_read_b128 v[12:15], v212 offset:49408
	ds_read_b128 v[8:11], v212 offset:49664
	ds_read_b128 v[0:3], v212 offset:49920
	v_accvgpr_read_b32 v241, a165
	v_accvgpr_read_b32 v242, a164
	v_accvgpr_read_b32 v243, a159
	v_accvgpr_read_b32 v244, a158
	v_accvgpr_read_b32 v245, a157
	v_accvgpr_read_b32 v246, a156
	v_accvgpr_read_b32 v199, a149
	v_accvgpr_read_b32 v102, a148
	v_accvgpr_read_b32 v147, a143
	v_accvgpr_read_b32 v222, a142
	v_accvgpr_read_b32 v49, a141
	v_accvgpr_read_b32 v110, a140
	v_accvgpr_read_b32 v254, a135
	v_accvgpr_read_b32 v201, a134
	v_accvgpr_read_b32 v90, a133
	v_accvgpr_read_b32 v48, a132
	v_accvgpr_read_b32 v120, a195
	v_accvgpr_read_b32 v73, a194
	v_accvgpr_read_b32 v173, a193
	v_accvgpr_read_b32 v204, a192
	v_accvgpr_read_b32 v127, a183
	v_accvgpr_read_b32 v175, a182
	v_accvgpr_read_b32 v71, a181
	v_accvgpr_read_b32 v91, a180
	v_accvgpr_read_b32 v195, a171
	v_accvgpr_read_b32 v207, a170
	v_accvgpr_read_b32 v146, a169
	v_accvgpr_read_b32 v72, a168
	v_accvgpr_read_b32 v75, a163
	v_accvgpr_read_b32 v78, a162
	v_accvgpr_read_b32 v81, a161
	v_accvgpr_read_b32 v82, a160
	v_accvgpr_read_b32 v83, a155
	v_accvgpr_read_b32 v84, a154
	v_accvgpr_read_b32 v85, a153
	v_accvgpr_read_b32 v92, a152
	v_accvgpr_read_b32 v103, a147
	v_accvgpr_read_b32 v104, a146
	v_accvgpr_read_b32 v105, a145
	v_accvgpr_read_b32 v111, a144
	v_accvgpr_read_b32 v121, a139
	v_accvgpr_read_b32 v126, a138
	v_accvgpr_read_b32 v194, a137
	v_accvgpr_read_b32 v197, a136
	v_accvgpr_read_b32 v198, a131
	v_accvgpr_read_b32 v200, a130
	v_accvgpr_read_b32 v202, a129
	v_accvgpr_read_b32 v203, a128
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v172, v131
	v_mov_b32_e32 v174, v130
	v_mov_b32_e32 v176, v129
	v_mov_b32_e32 v177, v128
	v_mov_b32_e32 v178, v125
	v_mov_b32_e32 v179, v124
	v_mov_b32_e32 v180, v123
	v_mov_b32_e32 v181, v122
	v_mov_b32_e32 v182, v119
	v_mov_b32_e32 v183, v118
	v_mov_b32_e32 v184, v117
	v_mov_b32_e32 v185, v116
	v_mov_b32_e32 v186, v115
	v_mov_b32_e32 v187, v114
	v_mov_b32_e32 v188, v113
	v_mov_b32_e32 v189, v112
	v_mov_b32_e32 v190, v109
	v_mov_b32_e32 v192, v108
	v_mov_b32_e32 v193, v107
	v_mov_b32_e32 v196, v106
	v_mov_b32_e32 v206, v61
	v_mov_b32_e32 v209, v60
	v_mov_b32_e32 v214, v59
	v_mov_b32_e32 v217, v58
	v_mov_b32_e32 v227, v57
	v_mov_b32_e32 v229, v56
	v_mov_b32_e32 v231, v55
	v_mov_b32_e32 v232, v54
	v_accvgpr_read_b32 v54, a127
	v_accvgpr_read_b32 v55, a126
	v_accvgpr_read_b32 v56, a125
	v_accvgpr_read_b32 v57, a124
	v_accvgpr_read_b32 v59, a123
	v_accvgpr_read_b32 v60, a122
	v_accvgpr_read_b32 v61, a121
	v_accvgpr_read_b32 v107, a120
	v_accvgpr_read_b32 v108, a119
	v_accvgpr_read_b32 v109, a118
	v_accvgpr_read_b32 v113, a117
	v_accvgpr_read_b32 v114, a116
	v_accvgpr_read_b32 v115, a115
	v_accvgpr_read_b32 v117, a114
	v_accvgpr_read_b32 v118, a113
	v_accvgpr_read_b32 v119, a112
	v_accvgpr_read_b32 v58, a111
	v_accvgpr_read_b32 v106, a110
	v_accvgpr_read_b32 v112, a109
	v_accvgpr_read_b32 v116, a108
	v_accvgpr_read_b32 v122, a107
	v_accvgpr_read_b32 v123, a106
	v_accvgpr_read_b32 v124, a105
	v_accvgpr_read_b32 v125, a104
	v_accvgpr_read_b32 v128, a103
	v_accvgpr_read_b32 v129, a102
	v_accvgpr_read_b32 v130, a101
	v_accvgpr_read_b32 v131, a100
	v_accvgpr_read_b32 v247, a99
	v_accvgpr_read_b32 v248, a98
	v_accvgpr_read_b32 v249, a97
	v_accvgpr_read_b32 v250, a96
	v_accvgpr_read_b32 v251, a95
	v_accvgpr_read_b32 v253, a94
	.loc	1 50 25                         ; matmul.py:50:25
	s_and_b64 vcc, exec, vcc
	v_accvgpr_read_b32 v142, a41
	v_accvgpr_read_b32 v143, a40
	v_accvgpr_read_b32 v144, a39
	v_accvgpr_read_b32 v145, a38
	v_accvgpr_read_b32 v150, a37
	v_accvgpr_read_b32 v151, a36
	v_accvgpr_read_b32 v152, a35
	v_accvgpr_read_b32 v153, a34
	v_accvgpr_read_b32 v154, a33
	v_accvgpr_read_b32 v155, a32
	v_accvgpr_read_b32 v156, a31
	v_accvgpr_read_b32 v157, a30
	v_accvgpr_read_b32 v158, a29
	v_accvgpr_read_b32 v159, a28
	v_accvgpr_read_b32 v160, a27
	v_accvgpr_read_b32 v161, a26
	v_accvgpr_read_b32 v162, a25
	v_accvgpr_read_b32 v163, a24
	v_accvgpr_read_b32 v164, a23
	v_accvgpr_read_b32 v165, a22
	v_accvgpr_read_b32 v166, a21
	v_accvgpr_read_b32 v167, a20
	v_accvgpr_read_b32 v168, a19
	v_accvgpr_read_b32 v169, a18
	scratch_store_dword off, v210, off      ; 4-byte Folded Spill
	scratch_store_dword off, v171, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v42, off offset:8 ; 4-byte Folded Spill
	scratch_store_dword off, v43, off offset:12 ; 4-byte Folded Spill
	scratch_store_dword off, v45, off offset:16 ; 4-byte Folded Spill
	scratch_store_dword off, v46, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v47, off offset:24 ; 4-byte Folded Spill
	scratch_store_dword off, v62, off offset:28 ; 4-byte Folded Spill
	scratch_store_dword off, v63, off offset:32 ; 4-byte Folded Spill
	scratch_store_dword off, v64, off offset:36 ; 4-byte Folded Spill
	scratch_store_dword off, v65, off offset:40 ; 4-byte Folded Spill
	scratch_store_dword off, v66, off offset:44 ; 4-byte Folded Spill
	scratch_store_dword off, v67, off offset:48 ; 4-byte Folded Spill
	scratch_store_dword off, v68, off offset:52 ; 4-byte Folded Spill
	scratch_store_dword off, v69, off offset:56 ; 4-byte Folded Spill
	scratch_store_dword off, v76, off offset:60 ; 4-byte Folded Spill
	scratch_store_dword off, v77, off offset:64 ; 4-byte Folded Spill
	scratch_store_dword off, v79, off offset:68 ; 4-byte Folded Spill
	scratch_store_dword off, v80, off offset:72 ; 4-byte Folded Spill
	scratch_store_dword off, v86, off offset:76 ; 4-byte Folded Spill
	scratch_store_dword off, v87, off offset:80 ; 4-byte Folded Spill
	scratch_store_dword off, v88, off offset:84 ; 4-byte Folded Spill
	scratch_store_dword off, v89, off offset:88 ; 4-byte Folded Spill
	scratch_store_dword off, v93, off offset:92 ; 4-byte Folded Spill
	scratch_store_dword off, v94, off offset:96 ; 4-byte Folded Spill
	scratch_store_dword off, v36, off offset:100 ; 4-byte Folded Spill
	scratch_store_dword off, v50, off offset:104 ; 4-byte Folded Spill
	scratch_store_dword off, v51, off offset:108 ; 4-byte Folded Spill
	scratch_store_dword off, v52, off offset:112 ; 4-byte Folded Spill
	scratch_store_dword off, v53, off offset:120 ; 4-byte Folded Spill
	scratch_store_dword off, v95, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v96, off offset:128 ; 4-byte Folded Spill
	scratch_store_dword off, v97, off offset:132 ; 4-byte Folded Spill
	scratch_store_dword off, v98, off offset:136 ; 4-byte Folded Spill
	scratch_store_dword off, v99, off offset:140 ; 4-byte Folded Spill
	scratch_store_dword off, v100, off offset:144 ; 4-byte Folded Spill
	scratch_store_dword off, v101, off offset:148 ; 4-byte Folded Spill
	scratch_store_dword off, v132, off offset:152 ; 4-byte Folded Spill
	scratch_store_dword off, v133, off offset:156 ; 4-byte Folded Spill
	scratch_store_dword off, v134, off offset:160 ; 4-byte Folded Spill
	scratch_store_dword off, v135, off offset:164 ; 4-byte Folded Spill
	scratch_store_dword off, v136, off offset:168 ; 4-byte Folded Spill
	scratch_store_dword off, v137, off offset:172 ; 4-byte Folded Spill
	scratch_store_dword off, v138, off offset:176 ; 4-byte Folded Spill
	scratch_store_dword off, v139, off offset:180 ; 4-byte Folded Spill
	scratch_store_dword off, v141, off offset:184 ; 4-byte Folded Spill
	scratch_store_dword off, v148, off offset:188 ; 4-byte Folded Spill
	scratch_store_dword off, v149, off offset:192 ; 4-byte Folded Spill
	scratch_store_dword off, v170, off offset:196 ; 4-byte Folded Spill
	scratch_store_dword off, v208, off offset:200 ; 4-byte Folded Spill
	scratch_store_dword off, v213, off offset:204 ; 4-byte Folded Spill
	scratch_store_dword off, v215, off offset:208 ; 4-byte Folded Spill
	scratch_store_dword off, v216, off offset:212 ; 4-byte Folded Spill
	scratch_store_dword off, v218, off offset:216 ; 4-byte Folded Spill
	scratch_store_dword off, v219, off offset:220 ; 4-byte Folded Spill
	scratch_store_dword off, v220, off offset:224 ; 4-byte Folded Spill
	scratch_store_dword off, v54, off offset:228 ; 4-byte Folded Spill
	scratch_store_dword off, v221, off offset:232 ; 4-byte Folded Spill
	scratch_store_dword off, v55, off offset:236 ; 4-byte Folded Spill
	scratch_store_dword off, v56, off offset:240 ; 4-byte Folded Spill
	scratch_store_dword off, v224, off offset:244 ; 4-byte Folded Spill
	scratch_store_dword off, v57, off offset:248 ; 4-byte Folded Spill
	scratch_store_dword off, v225, off offset:252 ; 4-byte Folded Spill
	scratch_store_dword off, v59, off offset:256 ; 4-byte Folded Spill
	scratch_store_dword off, v60, off offset:260 ; 4-byte Folded Spill
	scratch_store_dword off, v226, off offset:264 ; 4-byte Folded Spill
	scratch_store_dword off, v61, off offset:268 ; 4-byte Folded Spill
	scratch_store_dword off, v228, off offset:272 ; 4-byte Folded Spill
	scratch_store_dword off, v107, off offset:276 ; 4-byte Folded Spill
	scratch_store_dword off, v108, off offset:280 ; 4-byte Folded Spill
	scratch_store_dword off, v230, off offset:284 ; 4-byte Folded Spill
	scratch_store_dword off, v109, off offset:288 ; 4-byte Folded Spill
	scratch_store_dword off, v233, off offset:292 ; 4-byte Folded Spill
	scratch_store_dword off, v113, off offset:296 ; 4-byte Folded Spill
	scratch_store_dword off, v114, off offset:300 ; 4-byte Folded Spill
	scratch_store_dword off, v234, off offset:304 ; 4-byte Folded Spill
	scratch_store_dword off, v115, off offset:308 ; 4-byte Folded Spill
	scratch_store_dword off, v235, off offset:312 ; 4-byte Folded Spill
	scratch_store_dword off, v117, off offset:316 ; 4-byte Folded Spill
	scratch_store_dword off, v118, off offset:320 ; 4-byte Folded Spill
	scratch_store_dword off, v236, off offset:324 ; 4-byte Folded Spill
	scratch_store_dword off, v119, off offset:328 ; 4-byte Folded Spill
	scratch_store_dword off, v237, off offset:332 ; 4-byte Folded Spill
	scratch_store_dword off, v238, off offset:336 ; 4-byte Folded Spill
	scratch_store_dword off, v239, off offset:340 ; 4-byte Folded Spill
	scratch_store_dword off, v240, off offset:344 ; 4-byte Folded Spill
	scratch_store_dword off, v241, off offset:348 ; 4-byte Folded Spill
	scratch_store_dword off, v242, off offset:352 ; 4-byte Folded Spill
	scratch_store_dword off, v243, off offset:356 ; 4-byte Folded Spill
	scratch_store_dword off, v244, off offset:360 ; 4-byte Folded Spill
	scratch_store_dword off, v245, off offset:364 ; 4-byte Folded Spill
	scratch_store_dword off, v246, off offset:368 ; 4-byte Folded Spill
	scratch_store_dword off, v58, off offset:372 ; 4-byte Folded Spill
	scratch_store_dword off, v106, off offset:376 ; 4-byte Folded Spill
	scratch_store_dword off, v112, off offset:380 ; 4-byte Folded Spill
	scratch_store_dword off, v116, off offset:384 ; 4-byte Folded Spill
	scratch_store_dword off, v122, off offset:388 ; 4-byte Folded Spill
	scratch_store_dword off, v123, off offset:392 ; 4-byte Folded Spill
	scratch_store_dword off, v124, off offset:396 ; 4-byte Folded Spill
	scratch_store_dword off, v125, off offset:400 ; 4-byte Folded Spill
	scratch_store_dword off, v128, off offset:404 ; 4-byte Folded Spill
	scratch_store_dword off, v129, off offset:408 ; 4-byte Folded Spill
	scratch_store_dword off, v130, off offset:412 ; 4-byte Folded Spill
	scratch_store_dword off, v131, off offset:416 ; 4-byte Folded Spill
	scratch_store_dword off, v247, off offset:420 ; 4-byte Folded Spill
	scratch_store_dword off, v248, off offset:424 ; 4-byte Folded Spill
	scratch_store_dword off, v249, off offset:428 ; 4-byte Folded Spill
	scratch_store_dword off, v250, off offset:432 ; 4-byte Folded Spill
	scratch_store_dword off, v251, off offset:436 ; 4-byte Folded Spill
	scratch_store_dword off, v253, off offset:440 ; 4-byte Folded Spill
	scratch_store_dword off, a93, off offset:444 ; 4-byte Folded Spill
	scratch_store_dword off, a92, off offset:448 ; 4-byte Folded Spill
	scratch_store_dword off, a91, off offset:452 ; 4-byte Folded Spill
	scratch_store_dword off, a90, off offset:456 ; 4-byte Folded Spill
	scratch_store_dword off, a89, off offset:460 ; 4-byte Folded Spill
	scratch_store_dword off, a88, off offset:464 ; 4-byte Folded Spill
	scratch_store_dword off, a87, off offset:468 ; 4-byte Folded Spill
	scratch_store_dword off, a86, off offset:472 ; 4-byte Folded Spill
	scratch_store_dword off, a85, off offset:476 ; 4-byte Folded Spill
	scratch_store_dword off, a84, off offset:480 ; 4-byte Folded Spill
	scratch_store_dword off, a83, off offset:484 ; 4-byte Folded Spill
	scratch_store_dword off, a82, off offset:488 ; 4-byte Folded Spill
	scratch_store_dword off, a81, off offset:492 ; 4-byte Folded Spill
	scratch_store_dword off, a80, off offset:496 ; 4-byte Folded Spill
	scratch_store_dword off, a79, off offset:500 ; 4-byte Folded Spill
	scratch_store_dword off, a78, off offset:504 ; 4-byte Folded Spill
	scratch_store_dword off, a77, off offset:508 ; 4-byte Folded Spill
	scratch_store_dword off, a76, off offset:512 ; 4-byte Folded Spill
	scratch_store_dword off, a75, off offset:516 ; 4-byte Folded Spill
	scratch_store_dword off, a74, off offset:520 ; 4-byte Folded Spill
	scratch_store_dword off, a73, off offset:524 ; 4-byte Folded Spill
	scratch_store_dword off, a72, off offset:528 ; 4-byte Folded Spill
	scratch_store_dword off, a71, off offset:532 ; 4-byte Folded Spill
	scratch_store_dword off, a70, off offset:536 ; 4-byte Folded Spill
	scratch_store_dword off, a69, off offset:540 ; 4-byte Folded Spill
	scratch_store_dword off, a68, off offset:544 ; 4-byte Folded Spill
	scratch_store_dword off, a67, off offset:548 ; 4-byte Folded Spill
	scratch_store_dword off, a66, off offset:552 ; 4-byte Folded Spill
	scratch_store_dword off, a65, off offset:556 ; 4-byte Folded Spill
	scratch_store_dword off, a64, off offset:560 ; 4-byte Folded Spill
	scratch_store_dword off, a151, off offset:564 ; 4-byte Folded Spill
	scratch_store_dword off, a150, off offset:568 ; 4-byte Folded Spill
	scratch_store_dword off, a63, off offset:572 ; 4-byte Folded Spill
	scratch_store_dword off, a62, off offset:576 ; 4-byte Folded Spill
	scratch_store_dword off, a61, off offset:580 ; 4-byte Folded Spill
	scratch_store_dword off, a60, off offset:584 ; 4-byte Folded Spill
	scratch_store_dword off, a59, off offset:588 ; 4-byte Folded Spill
	scratch_store_dword off, a58, off offset:592 ; 4-byte Folded Spill
	scratch_store_dword off, a57, off offset:596 ; 4-byte Folded Spill
	scratch_store_dword off, a56, off offset:600 ; 4-byte Folded Spill
	scratch_store_dword off, a55, off offset:604 ; 4-byte Folded Spill
	scratch_store_dword off, a54, off offset:608 ; 4-byte Folded Spill
	scratch_store_dword off, a53, off offset:612 ; 4-byte Folded Spill
	scratch_store_dword off, a52, off offset:616 ; 4-byte Folded Spill
	scratch_store_dword off, a51, off offset:620 ; 4-byte Folded Spill
	scratch_store_dword off, a50, off offset:624 ; 4-byte Folded Spill
	scratch_store_dword off, a49, off offset:628 ; 4-byte Folded Spill
	scratch_store_dword off, a48, off offset:632 ; 4-byte Folded Spill
	scratch_store_dword off, a47, off offset:636 ; 4-byte Folded Spill
	scratch_store_dword off, a46, off offset:640 ; 4-byte Folded Spill
	scratch_store_dword off, a45, off offset:644 ; 4-byte Folded Spill
	scratch_store_dword off, a44, off offset:648 ; 4-byte Folded Spill
	scratch_store_dword off, a43, off offset:652 ; 4-byte Folded Spill
	scratch_store_dword off, a42, off offset:656 ; 4-byte Folded Spill
	scratch_store_dword off, v142, off offset:660 ; 4-byte Folded Spill
	scratch_store_dword off, v143, off offset:664 ; 4-byte Folded Spill
	scratch_store_dword off, v144, off offset:668 ; 4-byte Folded Spill
	scratch_store_dword off, v145, off offset:672 ; 4-byte Folded Spill
	scratch_store_dword off, v150, off offset:676 ; 4-byte Folded Spill
	scratch_store_dword off, v151, off offset:680 ; 4-byte Folded Spill
	scratch_store_dword off, v152, off offset:684 ; 4-byte Folded Spill
	scratch_store_dword off, v153, off offset:688 ; 4-byte Folded Spill
	scratch_store_dword off, v154, off offset:692 ; 4-byte Folded Spill
	scratch_store_dword off, v155, off offset:696 ; 4-byte Folded Spill
	scratch_store_dword off, v156, off offset:700 ; 4-byte Folded Spill
	scratch_store_dword off, v157, off offset:704 ; 4-byte Folded Spill
	scratch_store_dword off, v158, off offset:708 ; 4-byte Folded Spill
	scratch_store_dword off, v159, off offset:712 ; 4-byte Folded Spill
	scratch_store_dword off, v160, off offset:716 ; 4-byte Folded Spill
	scratch_store_dword off, v161, off offset:720 ; 4-byte Folded Spill
	scratch_store_dword off, v162, off offset:724 ; 4-byte Folded Spill
	scratch_store_dword off, v163, off offset:728 ; 4-byte Folded Spill
	scratch_store_dword off, v164, off offset:732 ; 4-byte Folded Spill
	scratch_store_dword off, v165, off offset:736 ; 4-byte Folded Spill
	scratch_store_dword off, v166, off offset:740 ; 4-byte Folded Spill
	scratch_store_dword off, v167, off offset:744 ; 4-byte Folded Spill
	scratch_store_dword off, v168, off offset:748 ; 4-byte Folded Spill
	scratch_store_dword off, v169, off offset:752 ; 4-byte Folded Spill
	scratch_store_dword off, a17, off offset:756 ; 4-byte Folded Spill
	scratch_store_dword off, a16, off offset:760 ; 4-byte Folded Spill
	scratch_store_dword off, a15, off offset:764 ; 4-byte Folded Spill
	scratch_store_dword off, a14, off offset:768 ; 4-byte Folded Spill
	scratch_store_dword off, a13, off offset:772 ; 4-byte Folded Spill
	scratch_store_dword off, a12, off offset:776 ; 4-byte Folded Spill
	scratch_store_dword off, a11, off offset:780 ; 4-byte Folded Spill
	scratch_store_dword off, a10, off offset:784 ; 4-byte Folded Spill
	scratch_store_dword off, a9, off offset:788 ; 4-byte Folded Spill
	scratch_store_dword off, a8, off offset:792 ; 4-byte Folded Spill
	scratch_store_dword off, a7, off offset:796 ; 4-byte Folded Spill
	scratch_store_dword off, a6, off offset:800 ; 4-byte Folded Spill
	scratch_store_dword off, a5, off offset:804 ; 4-byte Folded Spill
	scratch_store_dword off, a4, off offset:808 ; 4-byte Folded Spill
	scratch_store_dword off, a3, off offset:812 ; 4-byte Folded Spill
	scratch_store_dword off, a2, off offset:816 ; 4-byte Folded Spill
	scratch_store_dword off, a1, off offset:820 ; 4-byte Folded Spill
	scratch_store_dword off, a0, off offset:824 ; 4-byte Folded Spill
	s_cbranch_vccz .LBB0_1
; %bb.3:
                                        ; implicit-def: $vgpr251
                                        ; implicit-def: $agpr255
                                        ; implicit-def: $agpr247
                                        ; implicit-def: $agpr239
                                        ; implicit-def: $agpr231
                                        ; implicit-def: $agpr223
                                        ; implicit-def: $agpr215
                                        ; implicit-def: $agpr207
                                        ; implicit-def: $vgpr243
                                        ; implicit-def: $agpr251
                                        ; implicit-def: $agpr243
                                        ; implicit-def: $agpr235
                                        ; implicit-def: $agpr227
                                        ; implicit-def: $agpr219
                                        ; implicit-def: $agpr211
                                        ; implicit-def: $agpr203
                                        ; implicit-def: $agpr199
                                        ; implicit-def: $agpr191
                                        ; implicit-def: $agpr175
                                        ; implicit-def: $agpr167
                                        ; implicit-def: $agpr159
                                        ; implicit-def: $agpr151
                                        ; implicit-def: $agpr143
                                        ; implicit-def: $agpr135
                                        ; implicit-def: $agpr195
                                        ; implicit-def: $agpr183
                                        ; implicit-def: $agpr171
                                        ; implicit-def: $agpr163
                                        ; implicit-def: $agpr155
                                        ; implicit-def: $agpr147
                                        ; implicit-def: $agpr139
                                        ; implicit-def: $agpr131
                                        ; implicit-def: $agpr127
                                        ; implicit-def: $agpr111
                                        ; implicit-def: $agpr95
                                        ; implicit-def: $agpr79
                                        ; implicit-def: $agpr63
                                        ; implicit-def: $agpr47
                                        ; implicit-def: $agpr31
                                        ; implicit-def: $agpr15
                                        ; implicit-def: $vgpr131
                                        ; implicit-def: $vgpr125
                                        ; implicit-def: $vgpr119
                                        ; implicit-def: $vgpr115
                                        ; implicit-def: $vgpr109
                                        ; implicit-def: $vgpr61
                                        ; implicit-def: $vgpr57
                                        ; implicit-def: $vgpr169
                                        ; implicit-def: $vgpr165
                                        ; implicit-def: $vgpr161
                                        ; implicit-def: $vgpr157
                                        ; implicit-def: $vgpr145
                                        ; implicit-def: $agpr179
                                        ; implicit-def: $vgpr153
                                        ; implicit-def: $agpr187
                                        ; implicit-def: $vgpr252
                                        ; implicit-def: $vgpr255
                                        ; implicit-def: $vgpr205
.LBB0_4:
	.loc	1 75 42                         ; matmul.py:75:42
	s_nop 7
	s_nop 6
	scratch_load_dword a0, off, off offset:824 ; 4-byte Folded Reload
	scratch_load_dword a1, off, off offset:820 ; 4-byte Folded Reload
	scratch_load_dword a2, off, off offset:816 ; 4-byte Folded Reload
	scratch_load_dword a3, off, off offset:812 ; 4-byte Folded Reload
	scratch_load_dword a4, off, off offset:808 ; 4-byte Folded Reload
	scratch_load_dword a5, off, off offset:804 ; 4-byte Folded Reload
	scratch_load_dword a6, off, off offset:800 ; 4-byte Folded Reload
	scratch_load_dword a7, off, off offset:796 ; 4-byte Folded Reload
	scratch_load_dword a8, off, off offset:792 ; 4-byte Folded Reload
	scratch_load_dword a9, off, off offset:788 ; 4-byte Folded Reload
	scratch_load_dword a10, off, off offset:784 ; 4-byte Folded Reload
	scratch_load_dword a11, off, off offset:780 ; 4-byte Folded Reload
	scratch_load_dword a12, off, off offset:776 ; 4-byte Folded Reload
	scratch_load_dword a13, off, off offset:772 ; 4-byte Folded Reload
	scratch_load_dword a14, off, off offset:768 ; 4-byte Folded Reload
	scratch_load_dword a15, off, off offset:764 ; 4-byte Folded Reload
	scratch_load_dword a16, off, off offset:760 ; 4-byte Folded Reload
	scratch_load_dword a17, off, off offset:756 ; 4-byte Folded Reload
	scratch_load_dword a18, off, off offset:752 ; 4-byte Folded Reload
	scratch_load_dword a19, off, off offset:748 ; 4-byte Folded Reload
	scratch_load_dword a20, off, off offset:744 ; 4-byte Folded Reload
	scratch_load_dword a21, off, off offset:740 ; 4-byte Folded Reload
	scratch_load_dword a22, off, off offset:736 ; 4-byte Folded Reload
	scratch_load_dword a23, off, off offset:732 ; 4-byte Folded Reload
	scratch_load_dword a24, off, off offset:728 ; 4-byte Folded Reload
	scratch_load_dword a25, off, off offset:724 ; 4-byte Folded Reload
	scratch_load_dword a26, off, off offset:720 ; 4-byte Folded Reload
	scratch_load_dword a27, off, off offset:716 ; 4-byte Folded Reload
	scratch_load_dword a28, off, off offset:712 ; 4-byte Folded Reload
	scratch_load_dword a29, off, off offset:708 ; 4-byte Folded Reload
	scratch_load_dword a30, off, off offset:704 ; 4-byte Folded Reload
	scratch_load_dword a31, off, off offset:700 ; 4-byte Folded Reload
	scratch_load_dword a32, off, off offset:696 ; 4-byte Folded Reload
	scratch_load_dword a33, off, off offset:692 ; 4-byte Folded Reload
	scratch_load_dword a34, off, off offset:688 ; 4-byte Folded Reload
	scratch_load_dword a35, off, off offset:684 ; 4-byte Folded Reload
	scratch_load_dword a36, off, off offset:680 ; 4-byte Folded Reload
	scratch_load_dword a37, off, off offset:676 ; 4-byte Folded Reload
	scratch_load_dword a38, off, off offset:672 ; 4-byte Folded Reload
	scratch_load_dword a39, off, off offset:668 ; 4-byte Folded Reload
	scratch_load_dword a40, off, off offset:664 ; 4-byte Folded Reload
	scratch_load_dword a41, off, off offset:660 ; 4-byte Folded Reload
	scratch_load_dword a42, off, off offset:656 ; 4-byte Folded Reload
	scratch_load_dword a43, off, off offset:652 ; 4-byte Folded Reload
	scratch_load_dword a44, off, off offset:648 ; 4-byte Folded Reload
	scratch_load_dword a45, off, off offset:644 ; 4-byte Folded Reload
	scratch_load_dword a46, off, off offset:640 ; 4-byte Folded Reload
	scratch_load_dword a47, off, off offset:636 ; 4-byte Folded Reload
	scratch_load_dword a48, off, off offset:632 ; 4-byte Folded Reload
	scratch_load_dword a49, off, off offset:628 ; 4-byte Folded Reload
	scratch_load_dword a50, off, off offset:624 ; 4-byte Folded Reload
	scratch_load_dword a51, off, off offset:620 ; 4-byte Folded Reload
	scratch_load_dword a52, off, off offset:616 ; 4-byte Folded Reload
	scratch_load_dword a53, off, off offset:612 ; 4-byte Folded Reload
	scratch_load_dword a54, off, off offset:608 ; 4-byte Folded Reload
	scratch_load_dword a55, off, off offset:604 ; 4-byte Folded Reload
	scratch_load_dword a56, off, off offset:600 ; 4-byte Folded Reload
	scratch_load_dword a57, off, off offset:596 ; 4-byte Folded Reload
	scratch_load_dword a58, off, off offset:592 ; 4-byte Folded Reload
	scratch_load_dword a59, off, off offset:588 ; 4-byte Folded Reload
	scratch_load_dword a60, off, off offset:584 ; 4-byte Folded Reload
	scratch_load_dword a61, off, off offset:580 ; 4-byte Folded Reload
	scratch_load_dword a62, off, off offset:576 ; 4-byte Folded Reload
	scratch_load_dword a63, off, off offset:572 ; 4-byte Folded Reload
	scratch_load_dword v36, off, off offset:568 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(6)
	scratch_load_dword v57, off, off offset:352 ; 4-byte Folded Reload
	scratch_load_dword v56, off, off offset:348 ; 4-byte Folded Reload
	scratch_load_dword v55, off, off offset:344 ; 4-byte Folded Reload
	scratch_load_dword v54, off, off offset:340 ; 4-byte Folded Reload
	scratch_load_dword v53, off, off offset:336 ; 4-byte Folded Reload
	scratch_load_dword v52, off, off offset:332 ; 4-byte Folded Reload
	scratch_load_dword v51, off, off offset:324 ; 4-byte Folded Reload
	scratch_load_dword v50, off, off offset:312 ; 4-byte Folded Reload
.Ltmp239:
	.loc	2 115 35                        ; tuple_helpers.py:115:35 @[ matmul.py:82:91 ]
	s_lshl_b32 s0, s8, 18
	.loc	2 132 15                        ; tuple_helpers.py:132:15 @[ matmul.py:82:91 ]
	s_add_i32 s0, s0, s9
.Ltmp240:
	.loc	1 75 42                         ; matmul.py:75:42
	s_waitcnt vmcnt(25)
	v_mfma_f32_32x32x2_f32 a[32:47], v48, v232, a[32:47]
	scratch_load_dword v47, off, off offset:284 ; 4-byte Folded Reload
	scratch_load_dword v46, off, off offset:272 ; 4-byte Folded Reload
	scratch_load_dword v45, off, off offset:264 ; 4-byte Folded Reload
	scratch_load_dword v43, off, off offset:252 ; 4-byte Folded Reload
	scratch_load_dword v42, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(5)
	scratch_load_dword v61, off, off offset:368 ; 4-byte Folded Reload
	scratch_load_dword v60, off, off offset:364 ; 4-byte Folded Reload
	scratch_load_dword v59, off, off offset:360 ; 4-byte Folded Reload
	scratch_load_dword v58, off, off offset:356 ; 4-byte Folded Reload
	s_waitcnt vmcnt(18)
	v_mfma_f32_32x32x2_f32 a[48:63], v48, v185, a[48:63]
	scratch_load_dword v48, off, off offset:292 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[32:47], v90, v231, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v90, v184, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v201, v229, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v201, v183, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v254, v227, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v254, v182, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v110, v32, a[32:47]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x2_f32 a[48:63], v110, v16, a[48:63]
	v_mfma_f32_32x32x2_f32 a[32:47], v49, v33, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v49, v17, a[48:63]
	scratch_load_dword v49, off, off offset:304 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[0:15], v203, v232, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v203, v185, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v222, v34, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v222, v18, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v202, v231, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v202, v184, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v147, v35, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v147, v19, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v200, v229, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v200, v183, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v102, v217, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v102, v181, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v198, v227, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v198, v182, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v199, v214, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v199, v180, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v197, v32, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v197, v16, a[16:31]
	s_waitcnt vmcnt(19)
	v_mfma_f32_32x32x2_f32 a[32:47], v36, v209, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v36, v179, a[48:63]
	scratch_load_dword v36, off, off offset:564 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[0:15], v194, v33, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v194, v17, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v126, v34, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v126, v18, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v121, v35, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v121, v19, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v111, v217, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v111, v181, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v105, v214, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v105, v180, a[16:31]
	v_mfma_f32_32x32x2_f32 a[0:15], v104, v209, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v104, v179, a[16:31]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x2_f32 a[32:47], v36, v206, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v36, v178, a[48:63]
	scratch_load_dword v36, off, off offset:232 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[0:15], v103, v206, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v103, v178, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v61, v28, a[32:47]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x2_f32 a[48:63], v61, v12, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v92, v28, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v92, v12, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v60, v29, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v60, v13, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v85, v29, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v85, v13, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v59, v30, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v59, v14, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v84, v30, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v84, v14, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v58, v31, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v58, v15, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v83, v31, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v83, v15, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v57, v196, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v57, v177, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v82, v196, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v82, v177, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v56, v193, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v56, v176, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v81, v193, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v81, v176, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v55, v192, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v55, v174, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v78, v192, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v78, v174, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v54, v190, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v54, v172, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v75, v190, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v75, v172, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v53, v24, a[32:47]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x2_f32 a[48:63], v53, v8, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v72, v24, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v72, v8, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v52, v25, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v52, v9, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v146, v25, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v146, v9, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v51, v26, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v51, v10, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v207, v26, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v207, v10, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v50, v27, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v50, v11, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v195, v27, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v195, v11, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v49, v189, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v49, v4, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v91, v189, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v91, v4, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v48, v188, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v48, v5, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v71, v188, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v71, v5, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v47, v187, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v47, v6, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v175, v187, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v175, v6, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v46, v186, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v46, v7, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v127, v186, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v127, v7, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v45, v20, a[32:47]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x2_f32 a[48:63], v45, v0, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v204, v20, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v204, v0, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v43, v21, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v43, v1, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v173, v21, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v173, v1, a[16:31]
	v_mfma_f32_32x32x2_f32 a[32:47], v42, v22, a[32:47]
	v_mfma_f32_32x32x2_f32 a[48:63], v42, v2, a[48:63]
	v_mfma_f32_32x32x2_f32 a[0:15], v73, v22, a[0:15]
	v_mfma_f32_32x32x2_f32 a[16:31], v73, v2, a[16:31]
	scratch_load_dword a64, off, off offset:560 ; 4-byte Folded Reload
	scratch_load_dword a65, off, off offset:556 ; 4-byte Folded Reload
	scratch_load_dword a66, off, off offset:552 ; 4-byte Folded Reload
	scratch_load_dword a67, off, off offset:548 ; 4-byte Folded Reload
	scratch_load_dword a68, off, off offset:544 ; 4-byte Folded Reload
	scratch_load_dword a69, off, off offset:540 ; 4-byte Folded Reload
	scratch_load_dword a70, off, off offset:536 ; 4-byte Folded Reload
	scratch_load_dword a71, off, off offset:532 ; 4-byte Folded Reload
	scratch_load_dword a72, off, off offset:528 ; 4-byte Folded Reload
	scratch_load_dword a73, off, off offset:524 ; 4-byte Folded Reload
	scratch_load_dword a74, off, off offset:520 ; 4-byte Folded Reload
	scratch_load_dword a75, off, off offset:516 ; 4-byte Folded Reload
	scratch_load_dword a76, off, off offset:512 ; 4-byte Folded Reload
	scratch_load_dword a77, off, off offset:508 ; 4-byte Folded Reload
	scratch_load_dword a78, off, off offset:504 ; 4-byte Folded Reload
	scratch_load_dword a79, off, off offset:500 ; 4-byte Folded Reload
	scratch_load_dword v73, off, off offset:224 ; 4-byte Folded Reload
	scratch_load_dword v72, off, off offset:220 ; 4-byte Folded Reload
	scratch_load_dword v71, off, off offset:216 ; 4-byte Folded Reload
	scratch_load_dword v70, off, off offset:212 ; 4-byte Folded Reload
	scratch_load_dword v69, off, off offset:208 ; 4-byte Folded Reload
	scratch_load_dword v68, off, off offset:204 ; 4-byte Folded Reload
	scratch_load_dword v67, off, off offset:200 ; 4-byte Folded Reload
	scratch_load_dword v66, off, off offset:196 ; 4-byte Folded Reload
	scratch_load_dword v65, off, off offset:192 ; 4-byte Folded Reload
	scratch_load_dword v64, off, off offset:188 ; 4-byte Folded Reload
	scratch_load_dword v63, off, off offset:184 ; 4-byte Folded Reload
	scratch_load_dword v62, off, off offset:180 ; 4-byte Folded Reload
	scratch_load_dword v61, off, off offset:176 ; 4-byte Folded Reload
	scratch_load_dword v60, off, off offset:172 ; 4-byte Folded Reload
	scratch_load_dword v59, off, off offset:168 ; 4-byte Folded Reload
	scratch_load_dword v58, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(32)
	v_mfma_f32_32x32x2_f32 a[32:47], v36, v23, a[32:47]
	scratch_load_dword v57, off, off offset:160 ; 4-byte Folded Reload
	scratch_load_dword v56, off, off offset:156 ; 4-byte Folded Reload
	scratch_load_dword v55, off, off offset:152 ; 4-byte Folded Reload
	scratch_load_dword v54, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v53, off, off offset:144 ; 4-byte Folded Reload
	scratch_load_dword v52, off, off offset:140 ; 4-byte Folded Reload
	scratch_load_dword v51, off, off offset:136 ; 4-byte Folded Reload
	scratch_load_dword v50, off, off offset:132 ; 4-byte Folded Reload
	scratch_load_dword v49, off, off offset:128 ; 4-byte Folded Reload
	scratch_load_dword v48, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v47, off, off offset:120 ; 4-byte Folded Reload
	scratch_load_dword v46, off, off offset:112 ; 4-byte Folded Reload
	scratch_load_dword v45, off, off offset:108 ; 4-byte Folded Reload
	scratch_load_dword v43, off, off offset:104 ; 4-byte Folded Reload
	scratch_load_dword v42, off, off offset:100 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[48:63], v36, v3, a[48:63]
	scratch_load_dword v36, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword a80, off, off offset:496 ; 4-byte Folded Reload
	scratch_load_dword a81, off, off offset:492 ; 4-byte Folded Reload
	scratch_load_dword a82, off, off offset:488 ; 4-byte Folded Reload
	scratch_load_dword a83, off, off offset:484 ; 4-byte Folded Reload
	scratch_load_dword a84, off, off offset:480 ; 4-byte Folded Reload
	scratch_load_dword a85, off, off offset:476 ; 4-byte Folded Reload
	scratch_load_dword a86, off, off offset:472 ; 4-byte Folded Reload
	scratch_load_dword a87, off, off offset:468 ; 4-byte Folded Reload
	scratch_load_dword a88, off, off offset:464 ; 4-byte Folded Reload
	scratch_load_dword a89, off, off offset:460 ; 4-byte Folded Reload
	scratch_load_dword a90, off, off offset:456 ; 4-byte Folded Reload
	scratch_load_dword a91, off, off offset:452 ; 4-byte Folded Reload
	scratch_load_dword a92, off, off offset:448 ; 4-byte Folded Reload
	scratch_load_dword a93, off, off offset:444 ; 4-byte Folded Reload
	scratch_load_dword a94, off, off offset:440 ; 4-byte Folded Reload
	scratch_load_dword a95, off, off offset:436 ; 4-byte Folded Reload
	s_waitcnt vmcnt(47)
	v_mfma_f32_32x32x2_f32 a[64:79], v73, v232, a[64:79]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x2_f32 a[80:95], v73, v185, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v72, v231, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v72, v184, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v71, v229, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v71, v183, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v70, v227, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v70, v182, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v69, v32, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v69, v16, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v68, v33, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v68, v17, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v67, v34, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v67, v18, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v66, v35, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v66, v19, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v65, v217, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v65, v181, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v64, v214, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v64, v180, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v63, v209, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v63, v179, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v62, v206, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v62, v178, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v61, v28, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v61, v12, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v60, v29, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v60, v13, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v59, v30, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v59, v14, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v58, v31, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v58, v15, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v57, v196, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v57, v177, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v56, v193, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v56, v176, a[80:95]
	v_mfma_f32_32x32x2_f32 a[64:79], v55, v192, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v55, v174, a[80:95]
	scratch_load_dword a96, off, off offset:432 ; 4-byte Folded Reload
	scratch_load_dword a97, off, off offset:428 ; 4-byte Folded Reload
	scratch_load_dword a98, off, off offset:424 ; 4-byte Folded Reload
	scratch_load_dword a99, off, off offset:420 ; 4-byte Folded Reload
	scratch_load_dword a100, off, off offset:416 ; 4-byte Folded Reload
	scratch_load_dword a101, off, off offset:412 ; 4-byte Folded Reload
	scratch_load_dword a102, off, off offset:408 ; 4-byte Folded Reload
	scratch_load_dword a103, off, off offset:404 ; 4-byte Folded Reload
	scratch_load_dword a104, off, off offset:400 ; 4-byte Folded Reload
	scratch_load_dword a105, off, off offset:396 ; 4-byte Folded Reload
	scratch_load_dword a106, off, off offset:392 ; 4-byte Folded Reload
	scratch_load_dword a107, off, off offset:388 ; 4-byte Folded Reload
	scratch_load_dword a108, off, off offset:384 ; 4-byte Folded Reload
	scratch_load_dword a109, off, off offset:380 ; 4-byte Folded Reload
	scratch_load_dword a110, off, off offset:376 ; 4-byte Folded Reload
	scratch_load_dword a111, off, off offset:372 ; 4-byte Folded Reload
	scratch_load_dword v55, off, off offset:92 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v54, v190, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v54, v172, a[80:95]
	scratch_load_dword v54, off, off offset:88 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v53, v24, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v53, v8, a[80:95]
	scratch_load_dword v53, off, off offset:84 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v52, v25, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v52, v9, a[80:95]
	scratch_load_dword v52, off, off offset:80 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v51, v26, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v51, v10, a[80:95]
	scratch_load_dword v51, off, off offset:76 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v50, v27, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v50, v11, a[80:95]
	scratch_load_dword v50, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x2_f32 a[96:111], v55, v232, a[96:111]
	v_mfma_f32_32x32x2_f32 a[64:79], v49, v189, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v49, v4, a[80:95]
	scratch_load_dword v49, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x2_f32 a[96:111], v54, v231, a[96:111]
	v_mfma_f32_32x32x2_f32 a[64:79], v48, v188, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v48, v5, a[80:95]
	scratch_load_dword v48, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x2_f32 a[96:111], v53, v229, a[96:111]
	v_mfma_f32_32x32x2_f32 a[64:79], v47, v187, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v47, v6, a[80:95]
	scratch_load_dword v47, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x2_f32 a[96:111], v52, v227, a[96:111]
	v_mfma_f32_32x32x2_f32 a[64:79], v46, v186, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v46, v7, a[80:95]
	scratch_load_dword v46, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x2_f32 a[96:111], v51, v32, a[96:111]
	scratch_load_dword v32, off, off offset:24 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v45, v20, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v45, v0, a[80:95]
	scratch_load_dword v45, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(6)
	v_mfma_f32_32x32x2_f32 a[96:111], v50, v33, a[96:111]
	scratch_load_dword v33, off, off offset:28 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v43, v21, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v43, v1, a[80:95]
	scratch_load_dword v43, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x2_f32 a[96:111], v49, v34, a[96:111]
	scratch_load_dword v34, off, off offset:32 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v42, v22, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v42, v2, a[80:95]
	scratch_load_dword v42, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(8)
	v_mfma_f32_32x32x2_f32 a[96:111], v48, v35, a[96:111]
	scratch_load_dword v35, off, off offset:36 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[64:79], v36, v23, a[64:79]
	v_mfma_f32_32x32x2_f32 a[80:95], v36, v3, a[80:95]
	scratch_load_dword v36, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(9)
	v_mfma_f32_32x32x2_f32 a[96:111], v47, v217, a[96:111]
	s_waitcnt vmcnt(8)
	v_mfma_f32_32x32x2_f32 a[96:111], v46, v214, a[96:111]
	s_waitcnt vmcnt(6)
	v_mfma_f32_32x32x2_f32 a[96:111], v45, v209, a[96:111]
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x2_f32 a[96:111], v43, v206, a[96:111]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x2_f32 a[96:111], v42, v28, a[96:111]
	scratch_load_dword v28, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x2_f32 a[96:111], v36, v29, a[96:111]
	scratch_load_dword v29, off, off offset:12 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[96:111], v35, v30, a[96:111]
	scratch_load_dword v30, off, off offset:16 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[96:111], v34, v31, a[96:111]
	scratch_load_dword v31, off, off offset:20 ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[96:111], v33, v196, a[96:111]
	v_mfma_f32_32x32x2_f32 a[96:111], v32, v193, a[96:111]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x2_f32 a[96:111], v31, v192, a[96:111]
	v_mfma_f32_32x32x2_f32 a[96:111], v30, v190, a[96:111]
	v_mfma_f32_32x32x2_f32 a[96:111], v29, v24, a[96:111]
	scratch_load_dword v24, off, off        ; 4-byte Folded Reload
	v_mfma_f32_32x32x2_f32 a[96:111], v28, v25, a[96:111]
	scratch_load_dword v25, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword a112, off, off offset:328 ; 4-byte Folded Reload
	scratch_load_dword a113, off, off offset:320 ; 4-byte Folded Reload
	scratch_load_dword a114, off, off offset:316 ; 4-byte Folded Reload
	scratch_load_dword a115, off, off offset:308 ; 4-byte Folded Reload
	scratch_load_dword a116, off, off offset:300 ; 4-byte Folded Reload
	scratch_load_dword a117, off, off offset:296 ; 4-byte Folded Reload
	scratch_load_dword a118, off, off offset:288 ; 4-byte Folded Reload
	scratch_load_dword a119, off, off offset:280 ; 4-byte Folded Reload
	scratch_load_dword a120, off, off offset:276 ; 4-byte Folded Reload
	scratch_load_dword a121, off, off offset:268 ; 4-byte Folded Reload
	scratch_load_dword a122, off, off offset:260 ; 4-byte Folded Reload
	scratch_load_dword a123, off, off offset:256 ; 4-byte Folded Reload
	scratch_load_dword a124, off, off offset:248 ; 4-byte Folded Reload
	scratch_load_dword a125, off, off offset:240 ; 4-byte Folded Reload
	scratch_load_dword a126, off, off offset:236 ; 4-byte Folded Reload
	scratch_load_dword a127, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x2_f32 a[112:127], v55, v185, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v54, v184, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v53, v183, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v52, v182, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v51, v16, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v50, v17, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v49, v18, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v48, v19, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v47, v181, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v46, v180, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v45, v179, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v43, v178, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v42, v12, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v36, v13, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v35, v14, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v34, v15, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v33, v177, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v32, v176, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v31, v174, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v30, v172, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v29, v8, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v28, v9, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v25, v10, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v24, v11, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v211, v4, a[112:127]
	.loc	1 79 33                         ; matmul.py:79:33
	scratch_load_dword v4, off, off offset:1108 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[112:127], v74, v5, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v44, v6, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v41, v7, a[112:127]
	v_mfma_f32_32x32x2_f32 a[112:127], v40, v0, a[112:127]
	.loc	1 79 33                         ; matmul.py:79:33
	scratch_load_dword v0, off, off offset:1084 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v4, 8, v4
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v25, v26, a[96:111]
	.loc	1 79 33                         ; matmul.py:79:33
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v0, v0, 11, v4
	scratch_load_dword v4, off, off offset:1100 ; 4-byte Folded Reload
	scratch_load_dword v5, off, off offset:1104 ; 4-byte Folded Reload
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v24, v27, a[96:111]
	.loc	1 79 33                         ; matmul.py:79:33
	s_waitcnt vmcnt(0)
	v_or3_b32 v0, v0, v5, v4
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v211, v189, a[96:111]
	.loc	1 79 33                         ; matmul.py:79:33
	v_add_u32_e32 v4, 0, v0
	v_xad_u32 v0, v0, 64, 0
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v74, v188, a[96:111]
	v_mfma_f32_32x32x2_f32 a[0:15], v120, v23, a[0:15]
	.loc	1 79 33                         ; matmul.py:79:33
	s_nop 7
	s_nop 7
	s_nop 1
	ds_write_b128 v4, a[0:3]
	ds_write_b128 v4, a[8:11] offset:256
	ds_write_b128 v4, a[32:35] offset:512
	ds_write_b128 v4, a[40:43] offset:768
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[16:31], v120, v3, a[16:31]
	.loc	1 79 33                         ; matmul.py:79:33
	s_nop 7
	s_nop 7
	s_nop 1
	ds_write_b128 v0, a[16:19]
	ds_write_b128 v0, a[24:27] offset:256
	ds_write_b128 v0, a[48:51] offset:512
	ds_write_b128 v0, a[56:59] offset:768
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v44, v187, a[96:111]
	; wave barrier
	.loc	1 79 33                         ; matmul.py:79:33
	scratch_load_dword v5, off, off offset:1096 ; 4-byte Folded Reload
	scratch_load_dword v6, off, off offset:1088 ; 4-byte Folded Reload
	scratch_load_dword v7, off, off offset:1092 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v5, 10, v5
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v41, v186, a[96:111]
	.loc	1 79 33                         ; matmul.py:79:33
	s_waitcnt vmcnt(0)
	v_or3_b32 v5, v5, v7, v6
	v_add_u32_e32 v6, 0, v5
	ds_read_b128 v[68:71], v6
	ds_read_b128 v[72:75], v6 offset:128
	ds_read_b128 v[76:79], v6 offset:2048
	ds_read_b128 v[80:83], v6 offset:2176
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v40, v20, a[96:111]
	v_mfma_f32_32x32x2_f32 a[112:127], v39, v1, a[112:127]
	.loc	1 79 33                         ; matmul.py:79:33
	v_xad_u32 v1, v5, 64, 0
	ds_read_b128 v[84:87], v1 offset:1024
	ds_read_b128 v[88:91], v1 offset:1152
	ds_read_b128 v[92:95], v1 offset:3072
	ds_read_b128 v[96:99], v1 offset:3200
	; wave barrier
	ds_write_b128 v4, a[4:7]
	ds_write_b128 v4, a[12:15] offset:256
	ds_write_b128 v4, a[36:39] offset:512
	ds_write_b128 v4, a[44:47] offset:768
	ds_write_b128 v0, a[20:23]
	ds_write_b128 v0, a[28:31] offset:256
	ds_write_b128 v0, a[52:55] offset:512
	ds_write_b128 v0, a[60:63] offset:768
	; wave barrier
	ds_read_b128 v[100:103], v6
	ds_read_b128 v[104:107], v6 offset:128
	ds_read_b128 v[108:111], v6 offset:2048
	ds_read_b128 v[112:115], v6 offset:2176
	ds_read_b128 v[116:119], v1 offset:1024
	ds_read_b128 v[120:123], v1 offset:1152
	ds_read_b128 v[124:127], v1 offset:3072
	ds_read_b128 v[128:131], v1 offset:3200
	; wave barrier
	.loc	1 75 42                         ; matmul.py:75:42
	v_mfma_f32_32x32x2_f32 a[96:111], v39, v21, a[96:111]
	v_mfma_f32_32x32x2_f32 a[112:127], v38, v2, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v38, v22, a[96:111]
	v_mfma_f32_32x32x2_f32 a[112:127], v37, v3, a[112:127]
	v_mfma_f32_32x32x2_f32 a[96:111], v37, v23, a[96:111]
	.loc	1 79 33                         ; matmul.py:79:33
	ds_write_b128 v4, a[64:67]
	ds_write_b128 v4, a[72:75] offset:256
	s_nop 7
	s_nop 7
	ds_write_b128 v4, a[96:99] offset:512
	ds_write_b128 v4, a[104:107] offset:768
	ds_write_b128 v0, a[80:83]
	ds_write_b128 v0, a[88:91] offset:256
	ds_write_b128 v0, a[112:115] offset:512
	ds_write_b128 v0, a[120:123] offset:768
	; wave barrier
	ds_read_b128 v[132:135], v6
	ds_read_b128 v[136:139], v6 offset:128
	ds_read_b128 v[140:143], v6 offset:2048
	ds_read_b128 v[144:147], v6 offset:2176
	ds_read_b128 v[148:151], v1 offset:1024
	ds_read_b128 v[152:155], v1 offset:1152
	ds_read_b128 v[156:159], v1 offset:3072
	ds_read_b128 v[160:163], v1 offset:3200
	; wave barrier
	ds_write_b128 v4, a[68:71]
	ds_write_b128 v4, a[76:79] offset:256
	ds_write_b128 v4, a[100:103] offset:512
	ds_write_b128 v4, a[108:111] offset:768
	ds_write_b128 v0, a[84:87]
	ds_write_b128 v0, a[92:95] offset:256
	ds_write_b128 v0, a[116:119] offset:512
	ds_write_b128 v0, a[124:127] offset:768
	; wave barrier
	ds_read_b128 v[164:167], v6
	ds_read_b128 v[168:171], v6 offset:128
	ds_read_b128 v[172:175], v6 offset:2048
	ds_read_b128 v[176:179], v6 offset:2176
	ds_read_b128 v[180:183], v1 offset:1024
	ds_read_b128 v[184:187], v1 offset:1152
	ds_read_b128 v[188:191], v1 offset:3072
	ds_read_b128 v[192:195], v1 offset:3200
	scratch_load_dword v0, off, off offset:828 ; 4-byte Folded Reload
.Ltmp241:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	scratch_load_dword v1, off, off offset:956 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:960 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	scratch_load_dword v4, off, off offset:1028 ; 4-byte Folded Reload
	scratch_load_dword v6, off, off offset:1032 ; 4-byte Folded Reload
	scratch_load_dword v8, off, off offset:1036 ; 4-byte Folded Reload
	scratch_load_dword v10, off, off offset:1040 ; 4-byte Folded Reload
	scratch_load_dword v12, off, off offset:1044 ; 4-byte Folded Reload
	scratch_load_dword v14, off, off offset:1048 ; 4-byte Folded Reload
	scratch_load_dword v16, off, off offset:1052 ; 4-byte Folded Reload
	scratch_load_dword v18, off, off offset:1056 ; 4-byte Folded Reload
	scratch_load_dword v20, off, off offset:1060 ; 4-byte Folded Reload
	scratch_load_dword v22, off, off offset:1064 ; 4-byte Folded Reload
	scratch_load_dword v24, off, off offset:1068 ; 4-byte Folded Reload
	scratch_load_dword v26, off, off offset:1072 ; 4-byte Folded Reload
	scratch_load_dword v28, off, off offset:1076 ; 4-byte Folded Reload
	scratch_load_dword v30, off, off offset:1080 ; 4-byte Folded Reload
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(15)
	v_lshl_or_b32 v1, v1, 11, v0
	s_waitcnt vmcnt(14)
	v_lshl_or_b32 v3, v2, 11, v0
	scratch_load_dword v2, off, off offset:964 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v32, s0, v1
	v_add_u32_e32 v34, s0, v3
	s_waitcnt vmcnt(14)
	v_add_u32_e32 v4, s0, v4
	s_waitcnt vmcnt(13)
	v_add_u32_e32 v6, s0, v6
	s_waitcnt vmcnt(12)
	v_add_u32_e32 v8, s0, v8
	s_waitcnt vmcnt(11)
	v_add_u32_e32 v10, s0, v10
	s_waitcnt vmcnt(10)
	v_add_u32_e32 v12, s0, v12
	s_waitcnt vmcnt(9)
	v_add_u32_e32 v14, s0, v14
	s_waitcnt vmcnt(8)
	v_add_u32_e32 v16, s0, v16
	s_waitcnt vmcnt(7)
	v_add_u32_e32 v18, s0, v18
	s_waitcnt vmcnt(6)
	v_add_u32_e32 v20, s0, v20
	s_waitcnt vmcnt(5)
	v_add_u32_e32 v22, s0, v22
	s_waitcnt vmcnt(4)
	v_add_u32_e32 v24, s0, v24
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v26, s0, v26
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v28, s0, v28
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v30, s0, v30
.Ltmp242:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshl_add_u64 v[32:33], v[32:33], 2, s[6:7]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[6:7]
.Ltmp243:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v5, v2, 11, v0
	scratch_load_dword v2, off, off offset:968 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v36, s0, v5
.Ltmp244:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[36:37], 2, s[6:7]
.Ltmp245:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v7, v2, 11, v0
	scratch_load_dword v2, off, off offset:972 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v38, s0, v7
.Ltmp246:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 2, s[6:7]
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[38:39], v[38:39], 2, s[6:7]
.Ltmp247:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v9, v2, 11, v0
	scratch_load_dword v2, off, off offset:976 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v40, s0, v9
.Ltmp248:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], v[8:9], 2, s[6:7]
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[40:41], v[40:41], 2, s[6:7]
.Ltmp249:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v11, v2, 11, v0
	scratch_load_dword v2, off, off offset:980 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v42, s0, v11
.Ltmp250:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[6:7]
.Ltmp251:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v13, v2, 11, v0
	scratch_load_dword v2, off, off offset:984 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v44, s0, v13
.Ltmp252:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 2, s[6:7]
	v_ashrrev_i32_e32 v45, 31, v44
	v_lshl_add_u64 v[44:45], v[44:45], 2, s[6:7]
.Ltmp253:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v15, v2, 11, v0
	scratch_load_dword v2, off, off offset:988 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v46, s0, v15
.Ltmp254:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[6:7]
	v_ashrrev_i32_e32 v47, 31, v46
	v_lshl_add_u64 v[46:47], v[46:47], 2, s[6:7]
.Ltmp255:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v17, v2, 11, v0
	scratch_load_dword v2, off, off offset:992 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v48, s0, v17
.Ltmp256:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[6:7]
	v_ashrrev_i32_e32 v49, 31, v48
	v_lshl_add_u64 v[48:49], v[48:49], 2, s[6:7]
.Ltmp257:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v19, v2, 11, v0
	scratch_load_dword v2, off, off offset:996 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v50, s0, v19
.Ltmp258:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[6:7]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[6:7]
.Ltmp259:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v21, v2, 11, v0
	scratch_load_dword v2, off, off offset:1000 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v52, s0, v21
.Ltmp260:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[6:7]
	v_ashrrev_i32_e32 v53, 31, v52
	v_lshl_add_u64 v[52:53], v[52:53], 2, s[6:7]
.Ltmp261:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v23, v2, 11, v0
	scratch_load_dword v2, off, off offset:1004 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v54, s0, v23
.Ltmp262:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], v[22:23], 2, s[6:7]
	v_ashrrev_i32_e32 v55, 31, v54
	v_lshl_add_u64 v[54:55], v[54:55], 2, s[6:7]
.Ltmp263:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v25, v2, 11, v0
	scratch_load_dword v2, off, off offset:1008 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v56, s0, v25
.Ltmp264:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u64 v[24:25], v[24:25], 2, s[6:7]
	v_ashrrev_i32_e32 v57, 31, v56
	v_lshl_add_u64 v[56:57], v[56:57], 2, s[6:7]
.Ltmp265:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v27, v2, 11, v0
	scratch_load_dword v2, off, off offset:1012 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v58, s0, v27
.Ltmp266:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[6:7]
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[6:7]
.Ltmp267:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v29, v2, 11, v0
	scratch_load_dword v2, off, off offset:1016 ; 4-byte Folded Reload
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	v_add_u32_e32 v60, s0, v29
.Ltmp268:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshl_add_u64 v[28:29], v[28:29], 2, s[6:7]
	v_ashrrev_i32_e32 v61, 31, v60
	v_lshl_add_u64 v[60:61], v[60:61], 2, s[6:7]
.Ltmp269:
	.loc	3 50 8                          ; nd_helpers.py:50:8 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v31, v2, 11, v0
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	scratch_load_dword v0, off, off offset:1020 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:1024 ; 4-byte Folded Reload
	v_add_u32_e32 v62, s0, v31
.Ltmp270:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[6:7]
	v_ashrrev_i32_e32 v63, 31, v62
	v_lshl_add_u64 v[62:63], v[62:63], 2, s[6:7]
.Ltmp271:
	.loc	3 55 29                         ; nd_helpers.py:55:29 @[ matmul.py:82:91 ]
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v0, s0, v0
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v2, s0, v2
.Ltmp272:
	.loc	1 88 25                         ; matmul.py:88:25
	v_ashrrev_i32_e32 v1, 31, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[64:65], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[66:67], v[2:3], 2, s[6:7]
	.loc	1 88 41 is_stmt 0               ; matmul.py:88:41
	s_waitcnt lgkmcnt(14)
	v_mov_b32_e32 v0, v68
	v_mov_b32_e32 v1, v76
	v_mov_b32_e32 v2, v84
	v_mov_b32_e32 v3, v92
	global_store_dwordx4 v[64:65], v[0:3], off
	v_mov_b32_e32 v92, v71
	s_nop 0
	v_mov_b32_e32 v0, v69
	v_mov_b32_e32 v1, v77
	v_mov_b32_e32 v2, v85
	v_mov_b32_e32 v3, v93
	global_store_dwordx4 v[66:67], v[0:3], off
	v_mov_b32_e32 v93, v79
	s_nop 0
	v_mov_b32_e32 v0, v70
	v_mov_b32_e32 v1, v78
	v_mov_b32_e32 v2, v86
	v_mov_b32_e32 v3, v94
	global_store_dwordx4 v[4:5], v[0:3], off
	v_mov_b32_e32 v94, v87
	global_store_dwordx4 v[6:7], v[92:95], off
	v_mov_b32_e32 v0, v72
	v_mov_b32_e32 v1, v80
	v_mov_b32_e32 v2, v88
	v_mov_b32_e32 v3, v96
	global_store_dwordx4 v[8:9], v[0:3], off
	v_mov_b32_e32 v96, v75
	s_nop 0
	v_mov_b32_e32 v0, v73
	v_mov_b32_e32 v1, v81
	v_mov_b32_e32 v2, v89
	v_mov_b32_e32 v3, v97
	global_store_dwordx4 v[10:11], v[0:3], off
	v_mov_b32_e32 v97, v83
	s_nop 0
	v_mov_b32_e32 v0, v74
	v_mov_b32_e32 v1, v82
	v_mov_b32_e32 v2, v90
	v_mov_b32_e32 v3, v98
	global_store_dwordx4 v[12:13], v[0:3], off
	v_mov_b32_e32 v98, v91
	global_store_dwordx4 v[14:15], v[96:99], off
	v_mov_b32_e32 v0, v100
	v_mov_b32_e32 v1, v108
	v_mov_b32_e32 v2, v116
	v_mov_b32_e32 v3, v124
	global_store_dwordx4 v[16:17], v[0:3], off
	v_mov_b32_e32 v124, v103
	s_nop 0
	v_mov_b32_e32 v0, v101
	v_mov_b32_e32 v1, v109
	v_mov_b32_e32 v2, v117
	v_mov_b32_e32 v3, v125
	global_store_dwordx4 v[18:19], v[0:3], off
	v_mov_b32_e32 v125, v111
	s_nop 0
	v_mov_b32_e32 v0, v102
	v_mov_b32_e32 v1, v110
	v_mov_b32_e32 v2, v118
	v_mov_b32_e32 v3, v126
	global_store_dwordx4 v[20:21], v[0:3], off
	v_mov_b32_e32 v126, v119
	global_store_dwordx4 v[22:23], v[124:127], off
	v_mov_b32_e32 v0, v104
	v_mov_b32_e32 v1, v112
	v_mov_b32_e32 v2, v120
	v_mov_b32_e32 v3, v128
	global_store_dwordx4 v[24:25], v[0:3], off
	v_mov_b32_e32 v128, v107
	s_nop 0
	v_mov_b32_e32 v0, v105
	v_mov_b32_e32 v1, v113
	v_mov_b32_e32 v2, v121
	v_mov_b32_e32 v3, v129
	global_store_dwordx4 v[26:27], v[0:3], off
	v_mov_b32_e32 v129, v115
	s_nop 0
	v_mov_b32_e32 v0, v106
	v_mov_b32_e32 v1, v114
	v_mov_b32_e32 v2, v122
	v_mov_b32_e32 v3, v130
	global_store_dwordx4 v[28:29], v[0:3], off
	v_mov_b32_e32 v130, v123
	global_store_dwordx4 v[30:31], v[128:131], off
	v_mov_b32_e32 v0, v132
	v_mov_b32_e32 v1, v140
	v_mov_b32_e32 v2, v148
	v_mov_b32_e32 v3, v156
	global_store_dwordx4 v[32:33], v[0:3], off
	v_mov_b32_e32 v156, v135
	s_nop 0
	v_mov_b32_e32 v0, v133
	v_mov_b32_e32 v1, v141
	v_mov_b32_e32 v2, v149
	v_mov_b32_e32 v3, v157
	global_store_dwordx4 v[34:35], v[0:3], off
	v_mov_b32_e32 v157, v143
	s_nop 0
	v_mov_b32_e32 v0, v134
	v_mov_b32_e32 v1, v142
	v_mov_b32_e32 v2, v150
	v_mov_b32_e32 v3, v158
	global_store_dwordx4 v[36:37], v[0:3], off
	v_mov_b32_e32 v158, v151
	global_store_dwordx4 v[38:39], v[156:159], off
	v_mov_b32_e32 v0, v136
	v_mov_b32_e32 v1, v144
	v_mov_b32_e32 v2, v152
	v_mov_b32_e32 v3, v160
	global_store_dwordx4 v[40:41], v[0:3], off
	v_mov_b32_e32 v160, v139
	s_nop 0
	v_mov_b32_e32 v0, v137
	v_mov_b32_e32 v1, v145
	v_mov_b32_e32 v2, v153
	v_mov_b32_e32 v3, v161
	global_store_dwordx4 v[42:43], v[0:3], off
	v_mov_b32_e32 v161, v147
	s_nop 0
	v_mov_b32_e32 v0, v138
	v_mov_b32_e32 v1, v146
	v_mov_b32_e32 v2, v154
	v_mov_b32_e32 v3, v162
	global_store_dwordx4 v[44:45], v[0:3], off
	v_mov_b32_e32 v162, v155
	global_store_dwordx4 v[46:47], v[160:163], off
	s_waitcnt lgkmcnt(7)
	v_mov_b32_e32 v0, v164
	s_waitcnt lgkmcnt(5)
	v_mov_b32_e32 v1, v172
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v2, v180
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v3, v188
	global_store_dwordx4 v[48:49], v[0:3], off
	v_mov_b32_e32 v188, v167
	s_nop 0
	v_mov_b32_e32 v0, v165
	v_mov_b32_e32 v1, v173
	v_mov_b32_e32 v2, v181
	v_mov_b32_e32 v3, v189
	global_store_dwordx4 v[50:51], v[0:3], off
	v_mov_b32_e32 v189, v175
	s_nop 0
	v_mov_b32_e32 v0, v166
	v_mov_b32_e32 v1, v174
	v_mov_b32_e32 v2, v182
	v_mov_b32_e32 v3, v190
	global_store_dwordx4 v[52:53], v[0:3], off
	v_mov_b32_e32 v190, v183
	global_store_dwordx4 v[54:55], v[188:191], off
	v_mov_b32_e32 v0, v168
	v_mov_b32_e32 v1, v176
	v_mov_b32_e32 v2, v184
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, v192
	global_store_dwordx4 v[56:57], v[0:3], off
	v_mov_b32_e32 v192, v171
	s_nop 0
	v_mov_b32_e32 v0, v169
	v_mov_b32_e32 v1, v177
	v_mov_b32_e32 v2, v185
	v_mov_b32_e32 v3, v193
	global_store_dwordx4 v[58:59], v[0:3], off
	v_mov_b32_e32 v193, v179
	s_nop 0
	v_mov_b32_e32 v0, v170
	v_mov_b32_e32 v1, v178
	v_mov_b32_e32 v2, v186
	v_mov_b32_e32 v3, v194
	v_mov_b32_e32 v194, v187
	global_store_dwordx4 v[60:61], v[0:3], off
	global_store_dwordx4 v[62:63], v[192:195], off
	.loc	1 90 4 is_stmt 1                ; matmul.py:90:4
	s_endpgm
.Ltmp273:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 1116
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 18
		.amdhsa_accum_offset 256
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
	.set matmul.num_vgpr, 256
	.set matmul.num_agpr, 256
	.set matmul.numbered_sgpr, 18
	.set matmul.private_seg_size, 1116
	.set matmul.uses_vcc, 1
	.set matmul.uses_flat_scratch, 0
	.set matmul.has_dyn_sized_stack, 0
	.set matmul.has_recursion, 0
	.set matmul.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 22208
; TotalNumSgprs: 24
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 1116
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
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
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
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
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp221-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp232-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp243-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp247-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp250-.Lfunc_begin0
	.quad	.Ltmp251-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp253-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp258-.Lfunc_begin0
	.quad	.Ltmp259-.Lfunc_begin0
	.quad	.Ltmp260-.Lfunc_begin0
	.quad	.Ltmp261-.Lfunc_begin0
	.quad	.Ltmp262-.Lfunc_begin0
	.quad	.Ltmp263-.Lfunc_begin0
	.quad	.Ltmp264-.Lfunc_begin0
	.quad	.Ltmp265-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp268-.Lfunc_begin0
	.quad	.Ltmp269-.Lfunc_begin0
	.quad	.Ltmp270-.Lfunc_begin0
	.quad	.Ltmp271-.Lfunc_begin0
	.quad	.Ltmp272-.Lfunc_begin0
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
  - .agpr_count:     256
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
    .private_segment_fixed_size: 1116
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         matmul.kd
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 278
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:

Running Time  57.17849 ms
	     5.71TF/s
	     0.00TB/s

