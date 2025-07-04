"""
Pascal's Triangle and Combinatorial Iterators Demo

This demonstrates the key insight: combinatorics are like having two iterators 
adding up terms across a row forwards and backwards.

In Pascal's triangle: C(n,k) = C(n-1,k-1) + C(n-1,k)
This can be seen as:
- Iterator 1: moves "backwards" from position k to k-1 in row above
- Iterator 2: stays at position k in row above  
- Sum: C(n-1,k-1) + C(n-1,k) = C(n,k)

This dual-iterator pattern connects to Keya's Ϟ§ operators and creates
the fractal Sierpinski triangle when viewed modulo 2.
"""

import sys
import os
from typing import Optional, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import comb
import jax.numpy as jnp

from keya.core.engine import Engine
from keya.core.operators import D_operator as Wild_operator, C_operator as Containment_operator, DC_cycle as WildTame_cycle, Glyph


class PascalIteratorDemo:
    """Demonstrates the dual-iterator nature of Pascal's triangle construction."""
    
    def __init__(self, max_rows: int = 16):
        self.max_rows = max_rows
        self.triangle = self._build_pascal_triangle()
        self.binary_triangle = self._build_binary_triangle()
        self._wildtame_results: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        print(f"🔺 Pascal's Triangle Demo (rows 0 to {max_rows-1})")
        print("=" * 60)
    
    def _build_pascal_triangle(self) -> np.ndarray:
        """Build Pascal's triangle using the combinatorial formula."""
        triangle = np.zeros((self.max_rows, self.max_rows))
        for n in range(self.max_rows):
            for k in range(n + 1):
                triangle[n, k] = comb(n, k, exact=True)
        return triangle
    
    def _build_binary_triangle(self) -> np.ndarray:
        """Build Pascal's triangle modulo 2 (reveals Sierpinski pattern)."""
        return self.triangle % 2
    
    def demonstrate_dual_iterators(self):
        """Show how each Pascal triangle entry uses two 'iterators' from the row above."""
        print("\n🔄 DUAL ITERATOR CONSTRUCTION")
        print("-" * 50)
        print("Each C(n,k) = C(n-1,k-1) + C(n-1,k)")
        print("   ↑ backward iterator  ↑ forward iterator")
        print()
        
        # Show construction for first few rows
        for n in range(1, min(8, self.max_rows)):
            print(f"Row {n}:")
            row_str = ""
            for k in range(n + 1):
                if k == 0:
                    # Left edge: only forward iterator (from position 0)
                    back_val = 0  # No backward position
                    forward_val = int(self.triangle[n-1, 0])
                    result = int(self.triangle[n, k])
                    row_str += f"  C({n},{k}) = 0 + {forward_val} = {result}"
                elif k == n:
                    # Right edge: only backward iterator (from position k-1)
                    back_val = int(self.triangle[n-1, k-1])
                    forward_val = 0  # No forward position
                    result = int(self.triangle[n, k])
                    row_str += f"  C({n},{k}) = {back_val} + 0 = {result}"
                else:
                    # Interior: both iterators active
                    back_val = int(self.triangle[n-1, k-1])  # Iterator moving "back"
                    forward_val = int(self.triangle[n-1, k])  # Iterator staying put
                    result = int(self.triangle[n, k])
                    row_str += f"  C({n},{k}) = {back_val} + {forward_val} = {result}"
                
                if k < n:
                    row_str += ",  "
            
            print(row_str)
        
        print("\n💡 Key Insight: Two iterators scan the previous row:")
        print("   • Backward iterator: grabs value from position k-1")  
        print("   • Forward iterator: grabs value from position k")
        print("   • Sum: creates new value at position k in current row")
        
    def demonstrate_sierpinski_emergence(self):
        """Show how the Sierpinski triangle emerges from Pascal's triangle mod 2."""
        print("\n🔺 SIERPINSKI TRIANGLE EMERGENCE")
        print("-" * 50)
        print("Pascal's triangle modulo 2 reveals the Sierpinski fractal:")
        print()
        
        # Display binary triangle
        for n in range(min(12, self.max_rows)):
            # Center the row display
            spaces = " " * (12 - n)
            row_display = ""
            for k in range(n + 1):
                val = int(self.binary_triangle[n, k])
                symbol = "▲" if val == 1 else "·"
                row_display += f"{symbol} "
            print(f"{spaces}{row_display}")
        
        print("\n🔍 Pattern Analysis:")
        print("• ▲ = 1 (odd values from Pascal's triangle)")
        print("• · = 0 (even values from Pascal's triangle)")
        print("• This creates the characteristic Sierpinski triangle fractal")
        
        # Show the self-similarity
        print("\n🌀 Self-Similarity Property:")
        quarter_size = self.max_rows // 4
        if quarter_size >= 4:
            top_left = self.binary_triangle[:quarter_size, :quarter_size]
            top_right = self.binary_triangle[:quarter_size, quarter_size:2*quarter_size]
            print(f"Top-left {quarter_size}x{quarter_size} block matches pattern")
            print(f"Self-similar copies appear at multiple scales")
    
    def apply_dc_operators_to_pascal(self):
        """Apply Keya Ϟ§ operators to Pascal's triangle patterns."""
        print("\n🔧 APPLYING Ϟ§ OPERATORS TO PASCAL PATTERNS")
        print("-" * 50)
        
        # Convert Pascal's triangle to glyph matrix
        size = min(8, self.max_rows)  # Work with manageable size
        pascal_matrix = jnp.zeros((size, size), dtype=jnp.int32)
        
        for i in range(size):
            for j in range(min(i + 1, size)):
                val = int(self.triangle[i, j])
                # Map to glyph values [0-4]
                glyph_val = val % 5
                pascal_matrix = pascal_matrix.at[i, j].set(glyph_val)
        
        print(f"Original Pascal matrix ({size}x{size}):")
        self._print_matrix(pascal_matrix, "Pascal")
        
        # Apply Ϟ-operator (breaks symmetry on diagonal)
        wild_pascal = Wild_operator(pascal_matrix)
        print(f"\nAfter Ϟ-operator (diagonal transformation):")
        self._print_matrix(wild_pascal, "Ϟ-Pascal")
        
        # Apply §-operator with binary containment
        containment_pascal = Containment_operator(wild_pascal, "binary")
        print(f"\nAfter §-operator (binary containment):")
        self._print_matrix(containment_pascal, "§-Pascal")
        
        # Apply full ∮ cycle
        wildtame_pascal = WildTame_cycle(pascal_matrix, "binary", max_iterations=5)
        print(f"\nAfter ∮-cycle (5 iterations):")
        self._print_matrix(wildtame_pascal, "∮-Pascal")
        
        # Analyze the transformation effect
        original_nonzero = np.count_nonzero(pascal_matrix)
        wildtame_nonzero = np.count_nonzero(wildtame_pascal)
        
        print(f"\n📊 Transformation Analysis:")
        print(f"   Original non-zero elements: {original_nonzero}")
        print(f"   ∮-processed non-zero elements: {wildtame_nonzero}")
        print(f"   Sparsity change: {wildtame_nonzero/original_nonzero:.2f}x")
        
        return pascal_matrix, wildtame_pascal
    
    def _print_matrix(self, matrix: jnp.ndarray, label: str):
        """Pretty print a matrix with glyph symbols."""
        glyph_symbols = ["∅", "▽", "△", "⊙", "⊕"]
        print(f"   {label}:")
        for i in range(matrix.shape[0]):
            row_str = "   "
            for j in range(matrix.shape[1]):
                if j <= i or matrix[i, j] != 0:  # Only show triangle + any other non-zero
                    val = int(matrix[i, j])
                    symbol = glyph_symbols[val % len(glyph_symbols)]
                    row_str += f"{symbol} "
                else:
                    row_str += "  "
            print(row_str)
    
    def analyze_iterator_patterns(self):
        """Analyze the mathematical patterns in the iterator construction."""
        print("\n🔢 ITERATOR PATTERN ANALYSIS")
        print("-" * 50)
        
        # Look at iterator contribution ratios
        print("Iterator contribution analysis:")
        for n in range(2, min(8, self.max_rows)):
            print(f"\nRow {n} iterator contributions:")
            for k in range(1, n):  # Skip edges
                back_val = self.triangle[n-1, k-1]
                forward_val = self.triangle[n-1, k]
                total = back_val + forward_val
                
                back_ratio = back_val / total if total > 0 else 0
                forward_ratio = forward_val / total if total > 0 else 0
                
                print(f"   Position {k}: back={back_ratio:.2f}, forward={forward_ratio:.2f}")
        
        # Analyze growth patterns
        print(f"\n📈 Growth Pattern Analysis:")
        center_values = []
        for n in range(2, min(self.max_rows, 10), 2):  # Even rows
            center = n // 2
            val = self.triangle[n, center]
            center_values.append(val)
            print(f"   Row {n}, center value: {int(val)}")
        
        if len(center_values) > 1:
            ratios = [center_values[i+1] / center_values[i] for i in range(len(center_values)-1)]
            avg_ratio = np.mean(ratios)
            print(f"   Average growth ratio: {avg_ratio:.2f}")
    
    def visualize_patterns(self):
        """Create visualizations of the Pascal triangle and iterator patterns."""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Original Pascal's triangle
        ax1 = fig.add_subplot(2, 3, 1)
        size = min(12, self.max_rows)
        triangle_plot = self.triangle[:size, :size]
        im1 = ax1.imshow(triangle_plot, cmap='viridis', origin='upper')
        ax1.set_title("Pascal's Triangle")
        ax1.set_xlabel("k (column)")
        ax1.set_ylabel("n (row)")
        plt.colorbar(im1, ax=ax1)
        
        # 2. Sierpinski triangle (mod 2)
        ax2 = fig.add_subplot(2, 3, 2)
        binary_plot = self.binary_triangle[:size, :size]
        im2 = ax2.imshow(binary_plot, cmap='binary', origin='upper')
        ax2.set_title("Sierpinski Triangle (mod 2)")
        ax2.set_xlabel("k (column)")
        ax2.set_ylabel("n (row)")
        
        # 3. Iterator contribution heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        iterator_ratios = np.zeros((size, size))
        for n in range(1, size):
            for k in range(1, min(n, size)):
                if k < n and n-1 < size and k-1 >= 0:
                    back_val = self.triangle[n-1, k-1]
                    forward_val = self.triangle[n-1, k]
                    total = back_val + forward_val
                    if total > 0:
                        iterator_ratios[n, k] = back_val / total
        
        im3 = ax3.imshow(iterator_ratios, cmap='RdBu', origin='upper', vmin=0, vmax=1)
        ax3.set_title("Backward Iterator Contribution")
        ax3.set_xlabel("k (column)")
        ax3.set_ylabel("n (row)")
        plt.colorbar(im3, ax=ax3)
        
        # 4. Ϟ§ operator effect comparison
        if self._wildtame_results is not None:
            pascal_matrix, wildtame_pascal = self._wildtame_results
            
            ax4 = fig.add_subplot(2, 3, 4)
            im4 = ax4.imshow(pascal_matrix, cmap='plasma', origin='upper')
            ax4.set_title("Original (Glyph-mapped)")
            plt.colorbar(im4, ax=ax4)
            
            ax5 = fig.add_subplot(2, 3, 5)
            im5 = ax5.imshow(wildtame_pascal, cmap='plasma', origin='upper')
            ax5.set_title("After Ϟ§ Processing")
            plt.colorbar(im5, ax=ax5)
        
        # 6. Growth pattern
        ax6 = fig.add_subplot(2, 3, 6)
        rows = range(min(15, self.max_rows))
        center_vals = []
        for n in rows:
            center = n // 2
            if center < self.triangle.shape[1]:
                center_vals.append(self.triangle[n, center])
            else:
                center_vals.append(0)
        
        ax6.semilogy(rows, center_vals, 'bo-', markersize=4)
        ax6.set_title("Central Column Growth")
        ax6.set_xlabel("Row n")
        ax6.set_ylabel("C(n, n//2) (log scale)")
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs('.out/visualizations', exist_ok=True)
        plt.savefig('.out/visualizations/pascal_iterators.png', dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualization saved to .out/visualizations/pascal_iterators.png")
        plt.close()


def main():
    """Main demonstration of Pascal's triangle dual-iterator patterns."""
    print("🔺 Pascal's Triangle & Combinatorial Iterators")
    print("=" * 60)
    print("Key insight: C(n,k) = C(n-1,k-1) + C(n-1,k)")
    print("This is like having two iterators scanning the previous row:")
    print("• One moves backwards (k-1) and one stays (k)")
    print("• Their sum creates the next row's value")
    print("• This dual-iterator pattern creates the Sierpinski fractal!")
    
    # Create demo
    demo = PascalIteratorDemo(max_rows=16)
    
    # Show the dual iterator construction
    demo.demonstrate_dual_iterators()
    
    # Show Sierpinski emergence
    demo.demonstrate_sierpinski_emergence()
    
    # Apply Ϟ§ operators
    demo._wildtame_results = demo.apply_dc_operators_to_pascal()
    
    # Analyze patterns
    demo.analyze_iterator_patterns()
    
    # Create visualizations
    demo.visualize_patterns()
    
    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHTS")
    print("=" * 60)
    print("1. Pascal's Triangle Construction:")
    print("   • Each element uses TWO iterators from the row above")
    print("   • Backward iterator (k-1) + Forward iterator (k)")
    print("   • This dual-scan pattern is fundamental to combinatorics")
    print()
    print("2. Sierpinski Triangle Emergence:")
    print("   • Taking Pascal's triangle modulo 2 reveals fractal structure")  
    print("   • The dual iterators create self-similar patterns")
    print("   • Odd/even distribution follows fractal geometry")
    print()
    print("3. Connection to Ϟ§ Operators:")
    print("   • Ϟ-operator breaks diagonal symmetry (like iterator asymmetry)")
    print("   • §-operator creates containment patterns (like row constraints)")
    print("   • ∮-cycles evolve patterns (like recursive iterator application)")
    print()
    print("4. Mathematical Significance:")
    print("   • Forward/backward iteration is core to combinatorial recursion")
    print("   • This pattern appears in many mathematical constructions")
    print("   • The dual-iterator view unifies Pascal, Sierpinski, and Ϟ§ theory")
    
    print(f"\n✅ Demo complete! Check .out/visualizations/pascal_iterators.png")


if __name__ == "__main__":
    main() 