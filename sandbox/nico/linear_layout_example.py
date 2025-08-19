#!/usr/bin/env python3
"""
Example usage of LinearLayout Python bindings with FileCheck verification.

See include/triton/Tools/LinearLayout.h for the class and basic description.
Paper at: https://arxiv.org/html/2505.23819v1
"""

from triton._C.libtriton import ir
from triton._C.libtriton.gluon_ir import LinearLayout
from triton._filecheck import run_filecheck

def test_linear_layout_constructor():
    ctx = ir.context()
    # Create a custom layout from explicit bases
    # L(in1=1, in2=0) = (0, 1), L(in1=2, in2=0) = (0, 2)
    # L(in1=0, in2=1) = (0, 4), L(in1=0, in2=2) = (1, 1)
    bases = [
        ("in1", [[0, 1], [0, 2]]),  # L(in1=1)=(0,1), L(in1=2)=(0,2)
        ("in2", [[0, 4], [1, 1]])   # L(in2=1)=(0,4), L(in2=2)=(1,1)
    ]
    out_dims = ["out1", "out2"]
    layout = LinearLayout(bases, out_dims, ctx)
    check_template = """
    # CHECK: in1=1 -> (0, 1)
    # CHECK: in1=2 -> (0, 2)
    # CHECK: in2=1 -> (0, 4)
    # CHECK: in2=2 -> (1, 1)
    # CHECK: where out dims are: [out1 (size 2), out2 (size 8)]
    """
    run_filecheck("test_linear_layout_constructor", str(layout), check_template)

def test_to_pretty_binary_string():
    ctx = ir.context()
    
    empty = LinearLayout.empty()
    empty_binary = empty.to_pretty_binary_string()
    check_template = """
    # CHECK: (empty binary matrix)
    """
    run_filecheck("test_to_pretty_binary_string_empty", empty_binary, check_template)
    
    layout = LinearLayout.identity1D(16, "thread", "offset", ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (4 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1,2,3] 
    # CHECK: offset[0]:       1000
    # CHECK: offset[1]:       0100
    # CHECK: offset[2]:       0010
    # CHECK: offset[3]:       0001
    """
    assert layout.is_surjective() == True
    assert layout.is_injective() == True
    assert layout.is_invertible() == True
    run_filecheck("test_to_pretty_binary_string_identity", binary_str, check_template)
    
    # Note: the example below shows the binary matrix representation of the
    # layout where the basis are column vectors:
    #
    #     1 2|4 1 2 8
    #     ---+-------
    #     1 2|0 2 0 1
    #
    # and the dimensions of the quadrants (in numbits are):
    #
    #     (log2 8 + 1) x rank(reg) | (log2 8 + 1) x rank(lane)
    #     -----------------------+------------------------
    #     (log2 2 + 1) x rank(reg) | (log2 2 + 1) x rank(lane)
    #
    # LinearLayouts care about invertibility, surjectivity, projections and 
    # permutations in GF(2)^n.
    bases = [("reg", [[1, 1], [2, 2]]), ("lane", [[4, 0], [1, 2], [2, 0], [8, 1]])]
    layout2d = LinearLayout(bases, ["dim0", "dim1"], ctx)
    binary_str2d = layout2d.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (6 rows x 6 cols):
    # CHECK: Input dimensions (bit positions): reg[0,1] lane[2,3,4,5] 
    # CHECK: dim0[0]:         10|0100
    # CHECK: dim0[1]:         01|0010
    # CHECK: dim0[2]:         00|1000
    # CHECK: dim0[3]:         00|0001
    # CHECK:                  --+----
    # CHECK: dim1[0]:         10|0001
    # CHECK: dim1[1]:         01|0100
    """
    assert layout.is_surjective() == True
    assert layout.is_injective() == True
    assert layout.is_invertible() == True
    run_filecheck("test_to_pretty_binary_string_2d", binary_str2d, check_template)

def test_linear_layout_constructor_with_out_dims():
    ctx = ir.context()
    
    bases = [
        ("thread", [[1, 0], [2, 0], [4, 0]]),  # thread dimension with 3 bits (size 8)
        ("warp", [[0, 1], [0, 2]])              # warp dimension with 2 bits (size 4)
    ]
    out_dims = [ ("x", 16), ("y", 32)]
    
    # Create layout that does not require surjectivity
    # Note: the example below shows the binary matrix representation of the
    # layout where the basis are column vectors:
    #
    #     1 2 4|0 0
    #     -----+---
    #     0 0 0|1 2
    #
    # and the dimensions of the quadrants (in numbits are):
    #
    #     (log2 x_max + 1) x rank(thread) | (log2 x_max + 1) x rank(warp)
    #     -----------------------+------------------------
    #     (log2 y_max + 1) x rank(thread) | (log2 y_max + 1) x rank(warp)
    #
    # LinearLayouts care about invertibility, surjectivity, projections and 
    # permutations in GF(2)^n.
    layout_non_surjective = LinearLayout(bases, out_dims, False, ctx)
    check_template = """
    # CHECK: Binary matrix representation (9 rows x 5 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1,2] warp[3,4] 
    # CHECK: x[0]:            100|00
    # CHECK: x[1]:            010|00
    # CHECK: x[2]:            001|00
    # CHECK: x[3]:            000|00
    # CHECK:                  ---+--
    # CHECK: y[0]:            000|10
    # CHECK: y[1]:            000|01
    # CHECK: y[2]:            000|00
    # CHECK: y[3]:            000|00
    # CHECK: y[4]:            000|00
    """
    assert layout_non_surjective.is_surjective() == False
    assert layout_non_surjective.is_injective() == True
    assert layout_non_surjective.is_invertible() == False
    run_filecheck("test_linear_layout_constructor", layout_non_surjective.to_pretty_binary_string(), check_template)

def test_empty_layout():
    layout = LinearLayout.empty()
    check_template = """
    # CHECK: (empty layout)
    """
    run_filecheck("test_empty_layout", str(layout), check_template)

def test_strided1D_layout():
    ctx = ir.context()
    # Create a strided layout: L(x) = 2*x for x in [0, 8)
    # Maps input dimension "thread" to output dimension "offset"
    layout = LinearLayout.strided1D(8, 2, "thread", "offset", ctx)
    check_template = """
    # CHECK:   thread=1 -> (2)
    # CHECK:   thread=2 -> (4)
    # CHECK:   thread=4 -> (8)
    # CHECK: where out dims are: [offset (size 16)]
    """
    run_filecheck("test_strided1D_layout", str(layout), check_template)
    
    # Note this is not a surjective layout and cannot be built with the default
    # constructor.
    binary_str = layout.to_pretty_binary_string()
    binary_check_template = """
    # CHECK: Binary matrix representation (4 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1,2] 
    # CHECK: offset[0]:       000
    # CHECK: offset[1]:       100
    # CHECK: offset[2]:       010
    # CHECK: offset[3]:       001
    """
    run_filecheck("test_strided1D_layout_binary", binary_str, binary_check_template)

def test_identity1D_layout():
    ctx = ir.context()
    # Create an identity layout: L(x) = x for x in [0, 16)
    # Maps input dimension "lane" to output dimension "index"
    layout = LinearLayout.identity1D(16, "lane", "index", ctx)
    check_template = """
    # CHECK: lane=1 -> (1)
    # CHECK: lane=2 -> (2)
    # CHECK: lane=4 -> (4)
    # CHECK: lane=8 -> (8)
    # CHECK: where out dims are: [index (size 16)]
    """
    run_filecheck("test_identity1D_layout", str(layout), check_template)

def test_zeros1D_layout():
    ctx = ir.context()
    # Create a zeros layout: L(x) = 0 for x in [0, 8)
    # Maps input dimension "thread" to output dimension "result" (all map to 0)
    layout = LinearLayout.zeros1D(8, "thread", "result", ctx)
    check_template = """
    # CHECK: thread=1 -> (0)
    # CHECK: thread=2 -> (0)
    # CHECK: thread=4 -> (0)
    # CHECK: where out dims are: [result (size 1)]
    """
    run_filecheck("test_zeros1D_layout", str(layout), check_template)
    
def test_query():
    ctx = ir.context()
    empty = LinearLayout.empty()
    assert empty.get_num_in_dims() == 0
    assert empty.get_num_out_dims() == 0
    
    identity = LinearLayout.identity1D(8, "thread", "offset", ctx)
    assert identity.get_num_in_dims() == 1
    assert identity.get_num_out_dims() == 1
    
    bases = [("in1", [[0, 1]]), ("in2", [[1, 0]])]
    custom = LinearLayout(bases, ["out1", "out2"], ctx)
    assert custom.get_num_in_dims() == 2
    assert custom.get_num_out_dims() == 2

def test_has_dim():
    ctx = ir.context()
    layout = LinearLayout.identity1D(8, "thread", "offset", ctx)
    assert layout.has_in_dim("thread", ctx) == True
    assert layout.has_out_dim("offset", ctx) == True
    assert layout.has_in_dim("warp", ctx) == False
    assert layout.has_out_dim("address", ctx) == False

def test_dim_size():
    ctx = ir.context()
    strided = LinearLayout.strided1D(8, 2, "lane", "address", ctx)
    assert strided.get_in_dim_size("lane", ctx) == 8
    assert strided.get_out_dim_size("address", ctx) == 16

def test_apply():
    ctx = ir.context()
    bases = [("in1", [[1, 0], [0, 2]]), ("in2", [[0, 1]])]
    layout2d = LinearLayout(bases, ["out1", "out2"], ctx)
    result = layout2d.apply([("in1", 3), ("in2", 1)], ctx)
    assert result == [("out1", 1), ("out2", 3)]

def test_compose():
    ctx = ir.context()
    
    # Note: LinearLayout.strided1D does not need to be surjective but 
    # LinearLayout constructor does.by default
    # bases1 = [("a", [[2], [4]])]
    # layout1 = LinearLayout(bases1, ["b"], ctx)
    layout1 = LinearLayout.strided1D(4, 2, "a", "b", ctx)
    check_template = """
    # CHECK:   a=1 -> (2)
    # CHECK:   a=2 -> (4)
    # CHECK: where out dims are: [b (size 8)]
    """
    run_filecheck("test_compose", str(layout1), check_template)
  
    # Second layout: L2(b) = b + 1
    bases2 = [("b", [[1], [8], [4], [2]])]
    layout2 = LinearLayout(bases2, ["c"], ctx)
    check_template = """
    # CHECK:   b=1 -> (1)
    # CHECK:   b=2 -> (8)
    # CHECK:   b=4 -> (4)
    # CHECK:   b=8 -> (2)
    # CHECK: where out dims are: [c (size 16)]
    """
    run_filecheck("test_compose", str(layout2), check_template)
    
    # Compose: (L2 ∘ L1)(a) = L2(L1(a)) = L2(2*a) = 2*a + 1
    composed = layout1.compose(layout2)
    check_template = """
    # CHECK:   a=1 -> (8)
    # CHECK:   a=2 -> (4)
    # CHECK: where out dims are: [c (size 16)]
    """
    run_filecheck("test_compose", str(composed), check_template)
    
    # Test: (L2 ∘ L1)(1) = L2(L1(1)) = L2(2) = 3
    result = composed.apply([("a", 1)], ctx)
    assert result == [("c", 8)]

def test_linear_layout_unsqueeze_out():
    ctx = ir.context()
    
    bases = [
        ("in1", [[1, 0, 1], [2, 0, 0]]),  # 1 bit
        ("in2", [[2, 0, 2]])   # 1 bit
    ]
    out_dims = [
        ("out1", 16),
        ("out2", 1),
        ("out3", 8),
    ]
    layout = LinearLayout(bases, out_dims, False, ctx)
    assert layout.has_out_dim("out2", ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (7 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): in1[0,1] in2[2] 
    # CHECK: out1[0]:         10|0
    # CHECK: out1[1]:         01|1
    # CHECK: out1[2]:         00|0
    # CHECK: out1[3]:         00|0
    # CHECK:                  --+-
    # CHECK:                  --+-
    # CHECK: out3[0]:         10|0
    # CHECK: out3[1]:         00|1
    # CHECK: out3[2]:         00|0
    """
    run_filecheck("test_linear_layout_unsqueeze", binary_str, check_template)
    
    layout = layout.unsqueeze_out("out2", ctx)
    assert not layout.has_out_dim("out2", ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (7 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): in1[0,1] in2[2] 
    # CHECK: out1[0]:         10|0
    # CHECK: out1[1]:         01|1
    # CHECK: out1[2]:         00|0
    # CHECK: out1[3]:         00|0
    # CHECK:                  --+-
    # CHECK: out3[0]:         10|0
    # CHECK: out3[1]:         00|1
    # CHECK: out3[2]:         00|0
    """
    run_filecheck("test_linear_layout_unsqueeze", binary_str, check_template)
    
def test_linear_layout_unsqueeze_in():
    ctx = ir.context()
    
    bases = [
        ("in1", [[1, 0]]), 
        ("in2", []),        
        ("in3", [[1, 1], [0, 1]])         
    ]
    out_dims_simple = [
        ("out1", 2),
        ("out2", 2)
    ]
    layout = LinearLayout(bases, out_dims_simple, False, ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (2 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): in1[0] in2[] in3[1,2] 
    # CHECK: out1[0]:         1|10
    # CHECK:                  -+--
    # CHECK: out2[0]:         0|11
    """
    run_filecheck("test_linear_layout_unsqueeze", binary_str, check_template)
    
    assert layout.has_in_dim("in2", ctx)
    layout = layout.unsqueeze_in("in2", ctx)
    assert not layout.has_in_dim("in2", ctx)

def test_get_basis():
    ctx = ir.context()
    
    bases = [
        ("thread", [[1, 0, 0], [2, 0, 0]]),
        ("warp", [[0, 1, 0], [0, 2, 0]]),     
        ("block", [[0, 0, 1], [0, 0, 2]])     
    ]
    out_dims = [
        ("x", 4),
        ("y", 4),
        ("z", 4)
    ]
    
    layout = LinearLayout(bases, out_dims, True, ctx)
    
    thread_basis_0 = layout.get_basis("thread", 0, ctx)
    assert thread_basis_0 == [1, 0, 0]
    
    thread_basis_1 = layout.get_basis("thread", 1, ctx)
    assert thread_basis_1 == [2, 0, 0]
    
    assert layout.get_basis_component("thread", 0, "x", ctx) == 1
    assert layout.get_basis_component("thread", 0, "y", ctx) == 0
    assert layout.get_basis_component("thread", 0, "z", ctx) == 0
    
    warp_basis_0 = layout.get_basis("warp", 0, ctx)
    assert warp_basis_0 == [0, 1, 0]
    
    block_basis_0 = layout.get_basis("block", 0, ctx)
    assert block_basis_0 == [0, 0, 1]

def test_get_out_dims_3d():
    """Test getOutDims method with a 3D layout."""
    ctx = ir.context()
    bases = [
        ("thread", [[1, 0, 0], [2, 0, 0]]),
        ("warp", [[0, 1, 0], [0, 2, 0]]),     
        ("block", [[0, 0, 1], [0, 0, 2]])     
    ]
    out_dims = [("x", 4), ("y", 8), ("z", 16) ]
    layout = LinearLayout(bases, out_dims, False, ctx)
    assert layout.get_out_dims() == [("x", 4), ("y", 8), ("z", 16)]

def test_transpose_ins():
    ctx = ir.context()
    bases = [
        ("thread", [[1, 0], [2, 0]]),  # thread dimension with 2 bits
        ("warp", [[0, 1], [0, 2]])     # warp dimension with 2 bits
    ]
    out_dims = [("x", 4), ("y", 4)]
    layout = LinearLayout(bases, out_dims, True, ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (4 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1] warp[2,3] 
    # CHECK: x[0]:            10|00
    # CHECK: x[1]:            01|00
    # CHECK:                  --+--
    # CHECK: y[0]:            00|10
    # CHECK: y[1]:            00|01
    """
    run_filecheck("test_transpose_ins", binary_str, check_template)
    
    transposed = layout.transpose_ins(["warp", "thread"], ctx)
    binary_str = transposed.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (4 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): warp[0,1] thread[2,3] 
    # CHECK: x[0]:            00|10
    # CHECK: x[1]:            00|01
    # CHECK:                  --+--
    # CHECK: y[0]:            10|00
    # CHECK: y[1]:            01|00
    """
    run_filecheck("test_transpose_ins", binary_str, check_template)
        
    # The layout should still work the same functionally
    result = transposed.apply([("warp", 1), ("thread", 2)], ctx)
    expected = layout.apply([("thread", 2), ("warp", 1)], ctx)
    assert result == expected

def test_transpose_outs():
    ctx = ir.context()
    bases = [
        ("thread", [[1, 0], [2, 0]]),  # thread dimension with 2 bits
        ("warp", [[0, 1], [0, 2]])     # warp dimension with 2 bits
    ]
    out_dims = [("x", 4), ("y", 4)]
    layout = LinearLayout(bases, out_dims, True, ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (4 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1] warp[2,3] 
    # CHECK: x[0]:            10|00
    # CHECK: x[1]:            01|00
    # CHECK:                  --+--
    # CHECK: y[0]:            00|10
    # CHECK: y[1]:            00|01
    """
    run_filecheck("test_transpose_outs", binary_str, check_template)

    transposed = layout.transpose_outs(["y", "x"], ctx)
    binary_str = transposed.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (4 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1] warp[2,3] 
    # CHECK: y[0]:            00|10
    # CHECK: y[1]:            00|01
    # CHECK:                  --+--
    # CHECK: x[0]:            10|00
    # CHECK: x[1]:            01|00
    """
    run_filecheck("test_transpose_outs", binary_str, check_template)

    result = transposed.apply([("thread", 2), ("warp", 1)], ctx)
    expected = layout.apply([("thread", 2), ("warp", 1)], ctx)
    assert result[0] == expected[1] and result[1] == expected[0]

def test_reshape_ins():
    ctx = ir.context()
    bases = [
        ("a", [[1, 2], [2, 1]]), 
        ("b", [[2, 1], [0, 4]]),
        ("c", [[2, 1], [2, 4], [0, 0]]),
        ("d", [[1, 0]]),
    ]
    out_dims = [("x", 16), ("y", 8)]
    layout = LinearLayout(bases, out_dims, False, ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (7 rows x 8 cols):
    # CHECK: Input dimensions (bit positions): a[0,1] b[2,3] c[4,5,6] d[7] 
    # CHECK: x[0]:            10|00|000|1
    # CHECK: x[1]:            01|10|110|0
    # CHECK: x[2]:            00|00|000|0
    # CHECK: x[3]:            00|00|000|0
    # CHECK:                  --+--+---+-
    # CHECK: y[0]:            01|10|100|0
    # CHECK: y[1]:            10|00|000|0
    # CHECK: y[2]:            00|01|010|0
    """
    run_filecheck("test_reshape_ins_before", binary_str, check_template)

    # The following will fail with assertion if the total size does not match.
    # Let's compute the total input size and use that for the reshape.
    transposed = layout.transpose_ins(["c", "d", "a", "b"], ctx)
    total_in_size = 1
    for name in ["c", "d", "a"]:
        total_in_size *= transposed.get_in_dim_size(name, ctx)
    #                        c      d      a
    assert total_in_size == 2**3 * 2**1 * 2**2
    reshaped = transposed.reshape_ins([("CDA", total_in_size), ("B", 4)], ctx)
    binary_str = reshaped.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (7 rows x 8 cols):
    # CHECK: Input dimensions (bit positions): CDA[0,1,2,3,4,5] B[6,7] 
    # CHECK: x[0]:            000110|00
    # CHECK: x[1]:            110001|10
    # CHECK: x[2]:            000000|00
    # CHECK: x[3]:            000000|00
    # CHECK:                  ------+--
    # CHECK: y[0]:            100001|10
    # CHECK: y[1]:            000010|00
    # CHECK: y[2]:            010000|01          
    """
    run_filecheck("test_reshape_ins_after", binary_str, check_template)
    # Validate mapping for a grid of inputs
    for a in range(layout.get_in_dim_size("a", ctx)):
        for b in range(layout.get_in_dim_size("b", ctx)):
            for c in range(layout.get_in_dim_size("c", ctx)):
                for d in range(layout.get_in_dim_size("d", ctx)):
                    # Get sizes for flattening
                    c_s = transposed.get_in_dim_size("c", ctx)
                    d_s = transposed.get_in_dim_size("d", ctx)
                    
                    # Note: the order CDA is C minor and A major
                    combined_cda = c + d * c_s + a * (c_s * d_s)
                    
                    # Apply to reshaped layout
                    result = reshaped.apply([("CDA", combined_cda), ("B", b)], ctx)
                    
                    # The expected output is from the original layout with the original names
                    expected = layout.apply([("a", a), ("b", b), ("c", c), ("d", d)], ctx)
                    
                    assert result == expected, f"mismatch for a={a}, b={b}, c={c}, d={d}: {result} != {expected}"
                    
                    # Check we can invert back to the original components
                    a_s = layout.get_in_dim_size("a", ctx)
                    c2 = combined_cda % c_s
                    d2 = (combined_cda // c_s) % d_s
                    a2 = (combined_cda // (c_s * d_s)) % a_s
                    assert (a, c, d) == (a2, c2, d2), f"mismatch: {a, c, d} != {a2, c2, d2} for combined_cda={combined_cda}"


def test_reshape_outs():
    ctx = ir.context()
    bases = [
        ("a", [[1, 2, 0, 0, 0], [2, 1, 0, 0, 0]]), 
        ("b", [[2, 1, 1, 0, 0], [0, 4, 0, 1, 0]])
    ]
    out_dims = [("x", 4), ("y", 8), ("z", 2), ("w", 2), ("v", 2)]
    layout = LinearLayout(bases, out_dims, False, ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (8 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): a[0,1] b[2,3] 
    # CHECK: x[0]:            10|00
    # CHECK: x[1]:            01|10
    # CHECK:                  --+--
    # CHECK: y[0]:            01|10
    # CHECK: y[1]:            10|00
    # CHECK: y[2]:            00|01
    # CHECK:                  --+--
    # CHECK: z[0]:            00|10
    # CHECK:                  --+--
    # CHECK: w[0]:            00|01
    # CHECK:                  --+--
    # CHECK: v[0]:            00|00
    """
    run_filecheck("test_reshape_outs_before", binary_str, check_template)

    transposed = layout.transpose_outs(["w", "x", "z", "v", "y"], ctx)
    binary_str = transposed.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (8 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): a[0,1] b[2,3] 
    # CHECK: w[0]:            00|01
    # CHECK:                  --+--
    # CHECK: x[0]:            10|00
    # CHECK: x[1]:            01|10
    # CHECK:                  --+--
    # CHECK: z[0]:            00|10
    # CHECK:                  --+--
    # CHECK: v[0]:            00|00
    # CHECK:                  --+--
    # CHECK: y[0]:            01|10
    # CHECK: y[1]:            10|00
    # CHECK: y[2]:            00|01
    """
    run_filecheck("test_reshape_outs_before", binary_str, check_template)
    #           w      x      z
    WXZ_size = 2**1 * 2**2 * 2**1
    #           v      y
    VY_size = 2**1 * 2**3
    reshaped = transposed.reshape_outs([("WXZ", WXZ_size), ("VY", VY_size)], ctx)
    binary_str = reshaped.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (8 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): a[0,1] b[2,3] 
    # CHECK: WXZ[0]:          00|01
    # CHECK: WXZ[1]:          10|00
    # CHECK: WXZ[2]:          01|10
    # CHECK: WXZ[3]:          00|10
    # CHECK:                  --+--
    # CHECK: VY[0]:           00|00
    # CHECK: VY[1]:           01|10
    # CHECK: VY[2]:           10|00
    # CHECK: VY[3]:           00|01
    """
    run_filecheck("test_reshape_outs_after", binary_str, check_template)

    w_s = transposed.get_out_dim_size("w", ctx)
    x_s = transposed.get_out_dim_size("x", ctx)
    z_s = transposed.get_out_dim_size("z", ctx)
    v_s = transposed.get_out_dim_size("v", ctx)
    y_s = transposed.get_out_dim_size("y", ctx)
    
    # Validate mapping for a grid of inputs
    for a in range(layout.get_in_dim_size("a", ctx)):
        for b in range(layout.get_in_dim_size("b", ctx)):
            out = dict(transposed.apply([("a", a), ("b", b)], ctx))
            w, x, z, v, y = out["w"], out["x"], out["z"], out["v"], out["y"]

            # Least-significant-first within each group
            expected_WXZ = w + x * w_s + z * (w_s * x_s)
            expected_VY  = v + y * v_s

            res = dict(reshaped.apply([("a", a), ("b", b)], ctx))
            assert res["WXZ"] == expected_WXZ, f"mismatch: {res['WXZ']} != {expected_WXZ} for a={a}, b={b}"
            assert res["VY"] == expected_VY, f"mismatch: {res['VY']} != {expected_VY} for a={a}, b={b}"
            
            # Check we can invert back to the original components
            w2 = res["WXZ"] % w_s
            x2 = (res["WXZ"] // w_s) % x_s
            z2 = (res["WXZ"] // (w_s * x_s)) % z_s
            v2 = res["VY"] % v_s
            y2 = (res["VY"] // v_s) % y_s
            assert (w, x, z, v, y) == (w2, x2, z2, v2, y2), f"mismatch: {w, x, z, v, y} != {w2, x2, z2, v2, y2} for a={a}, b={b}"

def test_invert():
    ctx = ir.context()
    bases = [
        ("a", [ [1, 2, 0, 0], [0, 0, 1, 2] ]),
        ("b", [ [0, 1, 2, 0], [0, 0, 1, 1] ]),
        ("c", [ [0, 0, 1, 0], [1, 0, 0, 1] ]),
        ("d", [ [0, 0, 4, 1], [1, 1, 0, 0] ]),
    ]
    out_dims = [("o1", 2), ("o2", 4), ("o3", 8), ("o4", 4)]
    layout = LinearLayout(bases, out_dims, True, ctx)
    assert layout.invert().invert() == layout

def test_pseudoinvert():
    ctx = ir.context()
    # This is obtained from the test_invert above by dropping the last 
    # column / output dimension.
    # Note: this is still a full row-rank matrix.
    # We cannot do the same with a full column-rank matric (i.e. drop the last 
    # row) because the lstsq function fails when Im(B) is not a subset of Im(A).
    # I.e. the implementation of pseudoinvert is only defined for full row-rank 
    # / surjective matrices.
    bases = [
        ("a", [ [1, 2, 0], [0, 0, 1] ]),
        ("b", [ [0, 1, 2], [0, 0, 1] ]),
        ("c", [ [0, 0, 1], [1, 0, 0] ]),
        ("d", [ [0, 0, 4], [1, 1, 0] ]),
    ]
    out_dims = [("o1", 2), ("o2", 4), ("o3", 8)]
    layout = LinearLayout(bases, out_dims, True, ctx)

    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (6 rows x 8 cols):
    # CHECK: Input dimensions (bit positions): a[0,1] b[2,3] c[4,5] d[6,7] 
    # CHECK: o1[0]:           10|00|01|01
    # CHECK:                  --+--+--+--
    # CHECK: o2[0]:           00|10|00|01
    # CHECK: o2[1]:           10|00|00|00
    # CHECK:                  --+--+--+--
    # CHECK: o3[0]:           01|01|10|00
    # CHECK: o3[1]:           00|10|00|00
    # CHECK: o3[2]:           00|00|00|10
    """
    run_filecheck("test_pseudoinvert", binary_str, check_template)

    binary_str = layout.pseudoinvert().to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (8 rows x 6 cols):
    # CHECK: Input dimensions (bit positions): o1[0] o2[1,2] o3[3,4,5] 
    # CHECK: a[0]:            0|01|000
    # CHECK: a[1]:            0|00|100
    # CHECK:                  -+--+---
    # CHECK: b[0]:            0|00|010
    # CHECK: b[1]:            0|00|000
    # CHECK:                  -+--+---
    # CHECK: c[0]:            0|00|000
    # CHECK: c[1]:            1|11|010
    # CHECK:                  -+--+---
    # CHECK: d[0]:            0|00|001
    # CHECK: d[1]:            0|10|010
    """
    run_filecheck("test_pseudoinvert", binary_str, check_template)

def test_invert_and_compose():
    ctx = ir.context()
    
    bases = [
        ("i", [[0], [2], [4], [1]]),
    ]
    out_dims = ["o"]
    layout = LinearLayout(bases, out_dims, ctx)
    binary_str = layout.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (3 rows x 4 cols):
    # CHECK: Input dimensions (bit positions): i[0,1,2,3] 
    # CHECK: o[0]:            0001
    # CHECK: o[1]:            0100
    # CHECK: o[2]:            0010
    """
    run_filecheck("test_pseudoinvert", binary_str, check_template)
    
    bases = [
        ("p", [[2], [1], [0]]),
    ]
    out_dims = ["o"]
    layout2 = LinearLayout(bases, out_dims, ctx)
    binary_str = layout2.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (2 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): p[0,1,2] 
    # CHECK: o[0]:            010
    # CHECK: o[1]:            100
    """
    run_filecheck("test_pseudoinvert", binary_str, check_template)
    
    binary_str = layout2.invert_and_compose(layout).to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (4 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): p[0,1,2] 
    # CHECK: i[0]:            000
    # CHECK: i[1]:            100
    # CHECK: i[2]:            000
    # CHECK: i[3]:            010
    """
    run_filecheck("test_pseudoinvert", binary_str, check_template)

def test_direct_sum_operator():
    ctx = ir.context()
    layout1 = LinearLayout.identity1D(4, "thread", "offset", ctx)
    layout2 = LinearLayout.strided1D(2, 4, "warp", "address", ctx)
    combined = layout1 * layout2

    binary_str = combined.to_pretty_binary_string()
    check_template = """
    # CHECK: Binary matrix representation (5 rows x 3 cols):
    # CHECK: Input dimensions (bit positions): thread[0,1] warp[2] 
    # CHECK: offset[0]:       10|0
    # CHECK: offset[1]:       01|0
    # CHECK:                  --+-
    # CHECK: address[0]:      00|0
    # CHECK: address[1]:      00|0
    # CHECK: address[2]:      00|1
    """
    run_filecheck("test_direct_sum_operator", binary_str, check_template)
    
    layout1 *= layout2
    binary_str = layout1.to_pretty_binary_string()
    run_filecheck("test_direct_sum_operator", binary_str, check_template)

if __name__ == "__main__":
    test_linear_layout_constructor()
    test_to_pretty_binary_string()
    test_linear_layout_constructor_with_out_dims()
    test_empty_layout()
    test_strided1D_layout() 
    test_identity1D_layout()
    test_zeros1D_layout()
    test_query()
    test_has_dim()
    test_dim_size()
    test_apply()
    test_compose()
    test_linear_layout_unsqueeze_out()
    test_linear_layout_unsqueeze_in()
    test_get_basis()
    test_get_out_dims_3d()
    test_transpose_ins()
    test_transpose_outs()
    test_reshape_ins()
    test_reshape_outs()
    test_invert()
    test_pseudoinvert()
    test_invert_and_compose()
    test_direct_sum_operator()
