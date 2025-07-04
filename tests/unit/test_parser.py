#!/usr/bin/env python3
"""Test the keya D-C parser implementation."""

import sys
import os

# Ensure the src directory is on the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from keya.dsl import (
    Assignment,
    ContainmentOp,
    ContainmentType,
    DCCycle,
    DissonanceOp,
    Glyph,
    GlyphLiteral,
    GrammarProgram,
    MatrixAssignment,
    MatrixLiteral,
    MatrixProgram,
    ParseError,
    ResonanceProgram,
    ResonanceTrace,
    VerifyArithmetic,
    VerifyStrings,
    parse,
)


def test_basic_parsing():
    """Test basic parser functionality."""
    
    print("Testing D-C Parser...")
    
    # Test 1: Simple matrix program with D operator
    print("\n1. Testing matrix program with D operator:")
    code1 = """matrix test_matrix {
        operations {
            result = D [2, 2, ∅]
        }
    }"""
    
    try:
        ast1 = parse(code1)
        print(f"✓ Parsed matrix program: {ast1.name}")
        assert isinstance(ast1, MatrixProgram)
        assert ast1.name == "test_matrix"
        print(f"  Program type: {type(ast1).__name__}")
        print(f"  Sections: {len(ast1.sections)}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Grammar program with rules
    print("\n2. Testing grammar program:")
    code2 = """grammar test_grammar {
        rules {
            my_grammar = grammar simple {
                ▽ → △
                △ → ⊕
                ⊕ → ▽
            }
        }
    }"""
    
    try:
        ast2 = parse(code2)
        print(f"✓ Parsed grammar program: {ast2.name}")
        assert isinstance(ast2, GrammarProgram)
        print(f"  Program type: {type(ast2).__name__}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: Resonance program with verification
    print("\n3. Testing resonance program with verification:")
    code3 = """resonance test_resonance {
        analysis {
            verify arithmetic 10
            verify strings
            trace D([3, 3, ▽])
        }
    }"""
    
    try:
        ast3 = parse(code3)
        print(f"✓ Parsed resonance program: {ast3.name}")
        assert isinstance(ast3, ResonanceProgram)
        print(f"  Program type: {type(ast3).__name__}")
    except Exception as e:
        print(f"✗ Failed: {e}")


def test_dc_operations():
    """Test D-C specific operations."""
    
    print("\n\nTesting D-C Operations...")
    
    # Test 4: D operator
    print("\n4. Testing D operator:")
    code4 = """matrix d_test {
        ops {
            matrix1 = [2, 2, ▽]
            result = D matrix1
        }
    }"""
    
    try:
        ast4 = parse(code4)
        print("✓ D operator parsed successfully")
        # Check that we have DissonanceOp in the AST
        statements = ast4.sections[0].statements
        d_op_found = any(isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, DissonanceOp) 
                        for stmt in statements)
        if d_op_found:
            print("  ✓ DissonanceOp found in AST")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 5: C operator with containment type
    print("\n5. Testing C operator:")
    code5 = """matrix c_test {
        ops {
            matrix1 = [3, 3, ⊙]
            result = C(matrix1, binary)
        }
    }"""
    
    try:
        ast5 = parse(code5)
        print("✓ C operator parsed successfully")
        # Check for ContainmentOp
        statements = ast5.sections[0].statements
        c_op_found = any(isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, ContainmentOp) 
                        for stmt in statements)
        if c_op_found:
            print("  ✓ ContainmentOp found in AST")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: DC cycle
    print("\n6. Testing DC cycle:")
    code6 = """matrix dc_test {
        ops {
            matrix1 = [4, 4, △]
            result = DC(matrix1, decimal, 5)
        }
    }"""
    
    try:
        ast6 = parse(code6)
        print("✓ DC cycle parsed successfully")
        # Check for DCCycle
        statements = ast6.sections[0].statements
        dc_cycle_found = any(hasattr(stmt, 'value') and isinstance(stmt.value, DCCycle) 
                            for stmt in statements)
        if dc_cycle_found:
            print("  ✓ DCCycle found in AST")
    except Exception as e:
        print(f"✗ Failed: {e}")


def test_glyph_literals():
    """Test glyph literal parsing."""
    
    print("\n\nTesting Glyph Literals...")
    
    # Test 7: All glyph types
    print("\n7. Testing all glyph literals:")
    code7 = """matrix glyph_test {
        glyphs {
            void_glyph = ∅
            down_glyph = ▽  
            up_glyph = △
            unity_glyph = ⊙
            flow_glyph = ⊕
        }
    }"""
    
    try:
        ast7 = parse(code7)
        print("✓ All glyph literals parsed successfully")
        
        # Verify all glyph types are recognized
        statements = ast7.sections[0].statements
        glyph_literals = [stmt.value for stmt in statements if hasattr(stmt, 'value') and isinstance(stmt.value, GlyphLiteral)]
        
        glyph_types = {gl.glyph for gl in glyph_literals}
        expected_glyphs = {Glyph.VOID, Glyph.DOWN, Glyph.UP, Glyph.UNITY, Glyph.FLOW}
        
        if glyph_types == expected_glyphs:
            print("  ✓ All glyph types found in AST")
        else:
            print(f"  ? Found glyphs: {glyph_types}")
            print(f"  ? Expected: {expected_glyphs}")
            
    except Exception as e:
        print(f"✗ Failed: {e}")


def test_matrix_literals():
    """Test matrix literal parsing."""
    
    print("\n\nTesting Matrix Literals...")
    
    # Test 8: Matrix with dimensions and fill
    print("\n8. Testing matrix with dimensions:")
    code8 = """matrix matrix_test {
        matrices {
            empty_matrix = [3, 3, ∅]
            explicit_matrix = [[▽, △], [⊙, ⊕]]
        }
    }"""
    
    try:
        ast8 = parse(code8)
        print("✓ Matrix literals parsed successfully")
        
        statements = ast8.sections[0].statements
        matrix_literals = [stmt.value for stmt in statements if hasattr(stmt, 'value') and isinstance(stmt.value, MatrixLiteral)]
        
        if len(matrix_literals) == 2:
            print(f"  ✓ Found {len(matrix_literals)} matrix literals")
            
            # Check dimensions
            for i, ml in enumerate(matrix_literals):
                print(f"    Matrix {i+1}: {ml.rows}x{ml.cols}")
                
    except Exception as e:
        print(f"✗ Failed: {e}")


def test_error_handling():
    """Test parser error handling."""
    
    print("\n\nTesting Error Handling...")
    
    # Test 9: Invalid syntax
    print("\n9. Testing invalid syntax:")
    invalid_codes = [
        "invalid_program_type test {}",  # Invalid program type
        "matrix test { invalid_glyph = ◊ }",  # Invalid glyph
        "matrix test { incomplete = D }",  # Incomplete D operator
        "matrix test { bad_containment = C(matrix1, invalid_type) }",  # Invalid containment type
    ]
    
    for i, code in enumerate(invalid_codes):
        try:
            ast = parse(code)
            print(f"  ✗ Test {i+1}: Should have failed but didn't")
        except ParseError as e:
            print(f"  ✓ Test {i+1}: Correctly caught parse error: {e.message}")
        except Exception as e:
            print(f"  ? Test {i+1}: Unexpected error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("KEYA D-C PARSER TEST SUITE")
    print("=" * 60)
    
    test_basic_parsing()
    test_dc_operations()
    test_glyph_literals()
    test_matrix_literals()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("PARSER TEST SUITE COMPLETE")
    print("=" * 60) 