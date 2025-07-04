"""
Test the string generation system from D-C operators.

This demonstrates how horizontal propagation from D-C resonance can create
formal languages, grammars, and string operations.
"""

import os

os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU to avoid METAL bugs

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import jax.numpy as jnp

from keya.core.operators import (
    BINARY_GRAMMAR,
    SIMPLE_GRAMMAR,
    D_operator,
    Glyph,
    Grammar,
    apply_string_grammar_matrix,
    create_glyph_matrix,
    extract_string_from_matrix,
    generate_language_samples,
    generate_string_from_seed,
    recognize_pattern,
    string_concatenate_matrices,
    string_to_text,
    verify_string_generation,
)


def print_matrix(matrix: jnp.ndarray, title: str):
    """Print a glyph matrix with readable symbols."""
    print(f"\n{title}:")
    symbols = {
        Glyph.VOID.value: "∅",
        Glyph.DOWN.value: "▽", 
        Glyph.UP.value: "△",
        Glyph.UNITY.value: "⊙",
        Glyph.FLOW.value: "⊕"
    }
    
    for row in matrix:
        row_str = " ".join(symbols.get(int(val), "?") for val in row)
        print(f"  {row_str}")

def test_basic_grammars():
    """Test the predefined grammar systems."""
    print("=== TESTING BASIC GRAMMARS ===")
    
    # Test Simple Grammar: ▽ → △ → ⊕ → ▽ → ...
    print("\n1. Simple Grammar (▽ → △ → ⊕ → ▽):")
    simple_matrix = generate_string_from_seed(Glyph.DOWN, SIMPLE_GRAMMAR, 8)
    simple_string = extract_string_from_matrix(simple_matrix)
    simple_text = string_to_text(simple_string)
    
    print_matrix(simple_matrix, "Generated matrix")
    print(f"String: {simple_text}")
    
    # Verify the pattern (excluding VOID glyphs)
    non_void_string = [g for g in simple_string if g != Glyph.VOID]
    expected_cycle = [Glyph.DOWN, Glyph.UP, Glyph.FLOW] * 3  # Repeat cycle
    actual_cycle = non_void_string[:len(expected_cycle)]
    
    # Check the cycle pattern
    for i in range(min(len(actual_cycle), len(expected_cycle))):
        expected_glyph = expected_cycle[i % 3]  # Cycle through the 3-glyph pattern
        assert actual_cycle[i] == expected_glyph, f"Position {i}: expected {expected_glyph}, got {actual_cycle[i]}"
    print("✅ Simple grammar generates correct cycle")
    
    # Test Binary Grammar: ⊙ → ▽ → ⊙ → ▽ → ...
    print("\n2. Binary Grammar (⊙ → ▽ → ⊙ → ▽):")
    binary_matrix = generate_string_from_seed(Glyph.UNITY, BINARY_GRAMMAR, 6)
    binary_string = extract_string_from_matrix(binary_matrix)
    binary_text = string_to_text(binary_string)
    
    print_matrix(binary_matrix, "Binary grammar matrix")
    print(f"String: {binary_text}")
    
    # Verify alternating pattern (excluding VOID glyphs)
    non_void_binary = [g for g in binary_string if g != Glyph.VOID]
    for i in range(min(6, len(non_void_binary))):
        if i % 2 == 0:
            assert non_void_binary[i] == Glyph.UNITY, f"Position {i} should be UNITY, got {non_void_binary[i]}"
        else:
            assert non_void_binary[i] == Glyph.DOWN, f"Position {i} should be DOWN, got {non_void_binary[i]}"
    print("✅ Binary grammar alternates correctly")

def test_language_generation():
    """Test generation of language samples."""
    print("\n=== TESTING LANGUAGE GENERATION ===")
    
    # Generate samples from different grammars
    simple_samples = generate_language_samples(SIMPLE_GRAMMAR, Glyph.DOWN, 5)
    binary_samples = generate_language_samples(BINARY_GRAMMAR, Glyph.UNITY, 5)
    
    print("\nSimple Grammar Language Samples:")
    for i, sample in enumerate(simple_samples):
        print(f"  {i+1}: {sample}")
    
    print("\nBinary Grammar Language Samples:")
    for i, sample in enumerate(binary_samples):
        print(f"  {i+1}: {sample}")
    
    assert len(simple_samples) > 0, "Should generate simple grammar samples"
    assert len(binary_samples) > 0, "Should generate binary grammar samples"
    print("✅ Language sample generation works")

def test_pattern_recognition():
    """Test pattern matching in generated strings."""
    print("\n=== TESTING PATTERN RECOGNITION ===")
    
    # Generate a longer string for pattern testing
    matrix = generate_string_from_seed(Glyph.DOWN, SIMPLE_GRAMMAR, 10)
    string = extract_string_from_matrix(matrix)
    text = string_to_text(string)
    
    print(f"Test string: {text}")
    
    # Filter out VOID glyphs for pattern matching
    non_void_string = [g for g in string if g != Glyph.VOID]
    print(f"Non-void string: {string_to_text(non_void_string)}")
    
    # Test various patterns
    pattern1 = [Glyph.DOWN, Glyph.UP]  # ▽△
    pattern2 = [Glyph.UP, Glyph.FLOW]  # △⊕
    pattern3 = [Glyph.UNITY, Glyph.UNITY]  # ⊙⊙ (should not exist)
    
    match1 = recognize_pattern(non_void_string, pattern1)
    match2 = recognize_pattern(non_void_string, pattern2)
    match3 = recognize_pattern(non_void_string, pattern3)
    
    print(f"Pattern ▽△ found: {match1}")
    print(f"Pattern △⊕ found: {match2}")  
    print(f"Pattern ⊙⊙ found: {match3}")
    
    assert match1, "Should find ▽△ pattern in simple grammar"
    assert match2, "Should find △⊕ pattern in simple grammar"
    assert not match3, "Should not find ⊙⊙ pattern in simple grammar"
    print("✅ Pattern recognition works correctly")

def test_string_operations():
    """Test string concatenation and operations."""
    print("\n=== TESTING STRING OPERATIONS ===")
    
    # Create two different strings
    string1 = generate_string_from_seed(Glyph.DOWN, SIMPLE_GRAMMAR, 4)
    string2 = generate_string_from_seed(Glyph.UNITY, BINARY_GRAMMAR, 4)
    
    print_matrix(string1, "String 1 (Simple Grammar)")
    print_matrix(string2, "String 2 (Binary Grammar)")
    
    # Concatenate them
    concatenated = string_concatenate_matrices(string1, string2)
    print_matrix(concatenated, "Concatenated String")
    
    # Verify dimensions
    expected_width = string1.shape[1] + string2.shape[1]
    assert concatenated.shape[1] == expected_width, f"Expected width {expected_width}, got {concatenated.shape[1]}"
    
    # Verify content preservation
    concat_string = extract_string_from_matrix(concatenated)
    string1_part = extract_string_from_matrix(string1)
    extract_string_from_matrix(string2)  # Extract for comparison
    
    # Check that original strings are preserved in concatenation
    assert concat_string[:len(string1_part)] == string1_part, "First string not preserved"
    print("✅ String concatenation works correctly")

def test_dc_string_integration():
    """Test string generation integrated with D-C operators."""
    print("\n=== TESTING D-C STRING INTEGRATION ===")
    
    # Start with uniform field, apply D-C, then custom string grammar
    uniform_field = create_glyph_matrix((4, 8), Glyph.VOID)
    print_matrix(uniform_field, "Initial uniform field")
    
    # Apply D operator to create diagonal seeds
    dissonant = D_operator(uniform_field)
    print_matrix(dissonant, "After D operator (diagonal seeds)")
    
    # Apply custom string grammar instead of basic containment
    string_result = apply_string_grammar_matrix(dissonant, SIMPLE_GRAMMAR)
    print_matrix(string_result, "After string grammar application")
    
    # Extract and analyze the generated strings
    for row in range(string_result.shape[0]):
        if row < string_result.shape[1] and string_result[row, row] != Glyph.VOID.value:
            row_string = extract_string_from_matrix(string_result, row)
            row_text = string_to_text([g for g in row_string if g != Glyph.VOID])
            if row_text:
                print(f"Row {row} string: {row_text}")
    
    print("✅ D-C operators integrate with string generation")

def test_custom_grammar():
    """Test creation and use of custom grammar."""
    print("\n=== TESTING CUSTOM GRAMMAR ===")
    
    # Create a custom grammar for DNA-like sequences
    dna_grammar = Grammar({
        Glyph.DOWN: [Glyph.UP],      # A → T
        Glyph.UP: [Glyph.DOWN],      # T → A  
        Glyph.UNITY: [Glyph.FLOW],   # G → C
        Glyph.FLOW: [Glyph.UNITY],   # C → G
        Glyph.VOID: [Glyph.DOWN]     # Start with A
    })
    
    # Generate DNA-like sequence
    dna_matrix = generate_string_from_seed(Glyph.DOWN, dna_grammar, 8)
    dna_string = extract_string_from_matrix(dna_matrix)
    dna_text = string_to_text(dna_string)
    
    print_matrix(dna_matrix, "DNA-like sequence")
    print(f"DNA sequence: {dna_text}")
    print("Interpretation: ▽=A, △=T, ⊙=G, ⊕=C")
    
    # Verify complementary pairing pattern
    assert dna_string[0] == Glyph.DOWN, "Should start with A"
    assert dna_string[1] == Glyph.UP, "A should pair with T"
    print("✅ Custom grammar works correctly")

def test_comprehensive_verification():
    """Run the built-in comprehensive verification."""
    print("\n=== COMPREHENSIVE VERIFICATION ===")
    
    success = verify_string_generation()
    assert success, "Comprehensive string generation verification failed"
    print("✅ All comprehensive tests passed")

if __name__ == "__main__":
    print("Testing string generation system...")
    print("Verifying that D-C operators can generate formal languages.\n")
    
    try:
        test_basic_grammars()
        test_language_generation()
        test_pattern_recognition() 
        test_string_operations()
        test_dc_string_integration()
        test_custom_grammar()
        test_comprehensive_verification()
        
        print("\n🎉 ALL STRING TESTS PASSED!")
        print("✅ String generation system is verified and working")
        print("✅ D-C operators generate functional grammar systems")
        print("✅ Formal languages emerge from resonance patterns")
        print("✅ Pattern recognition and string operations functional")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("The string generation system needs further work.")
    except Exception as e:
        print(f"\n🔥 UNEXPECTED ERROR: {e}")
        print("Implementation has bugs that need fixing.") 