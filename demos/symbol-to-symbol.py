#!/usr/bin/env python3
"""
Demo: Symbol-to-Symbol Mapping in Kéya

This demonstrates that the symbol system can map symbols to other symbols,
not just symbols to numbers. We show three approaches:

1. Built-in omega (ω) transformation: symbol → symbol
2. Grammar-based string generation: symbol → [symbols]  
3. Custom transformation functions: symbol → symbol

The key insight: symbols are first-class entities that can transform into other symbols!
"""

import sys
sys.path.append('src')

import jax.numpy as jnp
from keya.core.operators import (
    Glyph, omega, create_glyph_matrix, 
    SIMPLE_GRAMMAR, BINARY_GRAMMAR, Grammar,
    generate_string_from_seed, string_to_text, extract_string_from_matrix,
    apply_glyph_transform
)

def print_glyph_matrix(matrix, title):
    """Pretty print a glyph matrix with symbols."""
    print(f"\n{title}:")
    symbols = {
        0: "∅", 1: "▽", 2: "△", 3: "⊙", 4: "⊕"
    }
    for row in matrix:
        row_str = " ".join(symbols.get(int(val), "?") for val in row)
        print(f"  {row_str}")

def custom_transform(glyph_value):
    """Custom symbol-to-symbol transformation function."""
    # ∅→⊕, ▽→⊙, △→∅, ⊙→▽, ⊕→△
    transform_map = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}
    return transform_map.get(glyph_value, glyph_value)

def main():
    print("🔥 Symbol-to-Symbol Mapping Demo")
    print("=" * 50)
    
    # 1. SYMBOL-TO-NUMBER MAPPING (traditional)
    print("\n1️⃣ SYMBOL-TO-NUMBER MAPPING")
    print("Symbols get interpreted as numeric values:")
    
    mixed_matrix = jnp.array([[0, 1], [2, 3]])  # ∅, ▽, △, ⊙
    print_glyph_matrix(mixed_matrix, "Original Matrix (symbols as numbers)")
    
    print("\nNumeric interpretation:")
    print(f"  As numbers: {mixed_matrix.tolist()}")
    
    # 2. SYMBOL-TO-SYMBOL MAPPING (omega transformation)
    print("\n2️⃣ SYMBOL-TO-SYMBOL MAPPING (Omega ω)")
    print("Symbols transform into other symbols!")
    
    print("\nOmega transformation rules:")
    print("  ∅ → ▽  (void becomes down)")
    print("  ▽ → △  (down becomes up)")  
    print("  △ → ▽  (up becomes down)")
    print("  ⊙ → ⊙  (unity stays unity)")
    print("  ⊕ → ⊕  (flow stays flow)")
    
    # Apply omega transformation element-wise
    def apply_omega_to_matrix(matrix):
        # Convert to regular numpy first, then back to JAX
        result = jnp.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                glyph = Glyph(int(matrix[i, j]))
                transformed = omega(glyph)
                result = result.at[i, j].set(transformed.value)
        return result
    
    omega_result = apply_omega_to_matrix(mixed_matrix)
    print_glyph_matrix(omega_result, "After Omega Transformation")
    
    # 3. CUSTOM SYMBOL-TO-SYMBOL TRANSFORMATION
    print("\n3️⃣ CUSTOM SYMBOL-TO-SYMBOL MAPPING")
    print("Custom transformation rules:")
    print("  ∅→⊕, ▽→⊙, △→∅, ⊙→▽, ⊕→△")
    
    # Apply custom transformation manually to avoid JAX issues
    custom_result = jnp.zeros_like(mixed_matrix)
    for i in range(mixed_matrix.shape[0]):
        for j in range(mixed_matrix.shape[1]):
            val = int(mixed_matrix[i, j])
            transformed_val = custom_transform(val)
            custom_result = custom_result.at[i, j].set(transformed_val)
    print_glyph_matrix(custom_result, "After Custom Transformation")
    
    # 4. GRAMMAR-BASED SYMBOL-TO-SYMBOL GENERATION
    print("\n4️⃣ GRAMMAR-BASED SYMBOL GENERATION")
    print("Symbols generate sequences of other symbols!")
    
    print("\nSimple Grammar Rules:")
    print("  ▽ → △  (down becomes up)")
    print("  △ → ⊕  (up becomes flow)")
    print("  ⊕ → ▽  (flow becomes down)")
    print("  ∅ → ▽  (void starts sequence)")
    
    # Generate symbol strings using grammar
    string_matrix = generate_string_from_seed(Glyph.VOID, SIMPLE_GRAMMAR, 8)
    glyph_string = extract_string_from_matrix(string_matrix)
    symbol_text = string_to_text(glyph_string)
    
    print(f"\nGenerated symbol sequence from ∅:")
    print(f"  {symbol_text}")
    
    print("\nBinary Grammar (alternating):")
    binary_string_matrix = generate_string_from_seed(Glyph.VOID, BINARY_GRAMMAR, 6)
    binary_glyph_string = extract_string_from_matrix(binary_string_matrix)
    binary_text = string_to_text(binary_glyph_string)
    print(f"  {binary_text}")
    
    # 5. FIBONACCI-LIKE SYMBOL SEQUENCE
    print("\n5️⃣ FIBONACCI-LIKE SYMBOL SEQUENCES")
    fibonacci_grammar = Grammar({
        Glyph.VOID: [Glyph.DOWN],           # Start: ∅ → ▽
        Glyph.DOWN: [Glyph.UP],             # A: ▽ → △  
        Glyph.UP: [Glyph.DOWN, Glyph.UP],   # B: △ → ▽△ (Fibonacci!)
        Glyph.UNITY: [Glyph.UNITY],         # ⊙ → ⊙
        Glyph.FLOW: [Glyph.VOID]            # ⊕ → ∅
    })
    
    fib_matrix = generate_string_from_seed(Glyph.VOID, fibonacci_grammar, 10)
    fib_string = extract_string_from_matrix(fib_matrix)
    fib_text = string_to_text(fib_string)
    print(f"Fibonacci symbol sequence: {fib_text}")
    
    print("\n✅ CONCLUSION:")
    print("Symbol-translation can map symbols to symbols through:")
    print("  • Built-in transformations (omega)")
    print("  • Custom transformation functions") 
    print("  • Grammar-based symbol generation")
    print("  • Context-dependent evolution rules")
    print("\nSymbols are not just numbers - they're autonomous entities!")

if __name__ == "__main__":
    main() 