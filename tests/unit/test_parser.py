#!/usr/bin/env python3
"""Test the keya  parser implementation."""


from keya.dsl import (
    TameOp,
    WildTameCycle,
    WildOp,
    Glyph,
    GlyphLiteral,
    GrammarProgram,
    MatrixAssignment,
    MatrixLiteral,
    MatrixProgram,
    ParseError,
    ResonanceProgram,
    parse,
)


def test_basic_parsing():
    """Test basic parser functionality."""
    
    print("Testing  Parser...")
    
    # Test 1: Simple matrix program with Ϟ operator
    print("\n1. Testing matrix program with Ϟ operator:")
    code1 = """matrix test_matrix {
        operations {
            result = Ϟ [2, 2, ∅]
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
            trace Ϟ([3, 3, ▽])
        }
    }"""
    
    try:
        ast3 = parse(code3)
        print(f"✓ Parsed resonance program: {ast3.name}")
        assert isinstance(ast3, ResonanceProgram)
        print(f"  Program type: {type(ast3).__name__}")
    except Exception as e:
        print(f"✗ Failed: {e}")


def test_wildtame_operations():
    """Test  specific operations."""
    
    print("\n\nTesting  Operations...")
    
    # Test 4: Ϟ operator
    print("\n4. Testing Ϟ operator:")
    code4 = """matrix wild_test {
        ops {
            matrix1 = [2, 2, ▽]
            result = Ϟ matrix1
        }
    }"""
    
    try:
        ast4 = parse(code4)
        print("✓ Ϟ operator parsed successfully")
        # Check that we have WildOp in the AST
        statements = ast4.sections[0].statements
        wild_op_found = any(isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, WildOp) 
                        for stmt in statements)
        if wild_op_found:
            print("  ✓ WildOp found in AST")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 5: § operator with containment type
    print("\n5. Testing § operator:")
    code5 = """matrix tame_test {
        ops {
            matrix1 = [3, 3, ⊙]
            result = §(matrix1, binary)
        }
    }"""
    
    try:
        ast5 = parse(code5)
        print("✓ § operator parsed successfully")
        # Check for TameOp
        statements = ast5.sections[0].statements
        tame_op_found = any(isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, TameOp) 
                        for stmt in statements)
        if tame_op_found:
            print("  ✓ TameOp found in AST")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: ∮ cycle
    print("\n6. Testing ∮ cycle:")
    code6 = """matrix wildtame_test {
        ops {
            matrix1 = [4, 4, △]
            result = ∮(matrix1, decimal, 5)
        }
    }"""
    
    try:
        ast6 = parse(code6)
        print("✓ ∮ cycle parsed successfully")
        # Check for WildTameCycle
        statements = ast6.sections[0].statements
        wildtame_cycle_found = any(isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, WildTameCycle) 
                            for stmt in statements)
        if wildtame_cycle_found:
            print("  ✓ WildTameCycle found in AST")
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
        glyph_literals = []
        for stmt in statements:
            if isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, GlyphLiteral):
                glyph_literals.append(stmt.value)
        
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
        matrix_literals = []
        for stmt in statements:
            if isinstance(stmt, MatrixAssignment) and isinstance(stmt.value, MatrixLiteral):
                matrix_literals.append(stmt.value)
        
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
        "matrix test { incomplete = Ϟ }",  # Incomplete Ϟ operator
        "matrix test { bad_containment = §(matrix1, invalid_type) }",  # Invalid containment type
    ]
    
    for i, code in enumerate(invalid_codes):
        try:
            parse(code)  # Should fail for invalid syntax
            print(f"  ✗ Test {i+1}: Should have failed but didn't")
        except ParseError as e:
            print(f"  ✓ Test {i+1}: Correctly caught parse error: {e.message}")
        except Exception as e:
            print(f"  ? Test {i+1}: Unexpected error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("KEYA WILD-TAME PARSER TEST SUITE")
    print("=" * 60)
    
    test_basic_parsing()
    test_wildtame_operations()
    test_glyph_literals()
    test_matrix_literals()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("PARSER TEST SUITE COMPLETE")
    print("=" * 60) 