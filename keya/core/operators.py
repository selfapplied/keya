"""
Core symbolic operators for the kéya engine.

Each function will be designed to operate on JAX arrays,
allowing for JIT compilation and hardware acceleration.
"""

import math
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp

from keya.dsl.ast import Glyph

# --- Foundational Attractors ---

def get_pi() -> float:
    """Get π constant."""
    return float(jnp.pi)


def get_golden_ratio() -> float:
    """Get φ (golden ratio) constant."""
    return float((1 + jnp.sqrt(5)) / 2)


# --- Glyph System ---

# Mapping from symbolic glyphs to integer representation for computation
GLYPH_TO_INT: Dict[Glyph, int] = {
    Glyph.VOID: 0,
    Glyph.DOWN: 1,
    Glyph.UP: 2,
    Glyph.UNITY: 3,
    Glyph.FLOW: 4,
}

# Reverse mapping for converting back to symbolic glyphs
INT_TO_GLYPH: Dict[int, Glyph] = {v: k for k, v in GLYPH_TO_INT.items()}


# Glyph transformation function ω
OMEGA_TRANSFORMS: Dict[Glyph, Glyph] = {
    Glyph.VOID: Glyph.DOWN,
    Glyph.DOWN: Glyph.UP,
    Glyph.UP: Glyph.DOWN,
    Glyph.UNITY: Glyph.UNITY,  # Fixed point
    Glyph.FLOW: Glyph.FLOW,  # Fixed point
}


def omega(glyph: Glyph) -> Glyph:
    """The fundamental glyph transformation ω."""
    return OMEGA_TRANSFORMS[glyph]


# --- Matrix Utilities ---


def create_glyph_matrix(shape: Tuple[int, int], fill_glyph: Glyph = Glyph.VOID) -> jnp.ndarray:
    """Create a matrix filled with a specific glyph."""
    return jnp.full(shape, GLYPH_TO_INT[fill_glyph], dtype=jnp.int32)


def apply_glyph_transform(matrix: jnp.ndarray, transform_fn: Callable[[int], int]) -> jnp.ndarray:
    """Apply a glyph transformation function to all elements in a matrix."""
    vectorized_transform = jnp.vectorize(transform_fn)
    return vectorized_transform(matrix)


# --- Fundamental Operators ---


def Wild_operator(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    The Wild operator (Ϟ): Breaks symmetry by applying ω to diagonal elements.

    This is the fundamental symmetry-breaking operation that seeds asymmetry
    in uniform glyph fields, creating the potential for pattern formation.
    """
    result = matrix.copy()
    rows, cols = matrix.shape
    min_dim = min(rows, cols)

    # Apply ω transformation to diagonal elements
    for i in range(min_dim):
        diagonal_int = int(matrix[i, i])
        diagonal_glyph = INT_TO_GLYPH[diagonal_int]
        transformed_glyph = omega(diagonal_glyph)
        result = result.at[i, i].set(GLYPH_TO_INT[transformed_glyph])

    return result


def Tame_operator(matrix: jnp.ndarray, containment_rule: str = "binary") -> jnp.ndarray:
    """
    The Tame operator (Ϙ): Creates resonance by organizing wildness into stable patterns.

    This operator heals the fractures created by the Wild operator, establishing equilibrium
    and generating attractor patterns (like number bases, strings, etc.).

    Args:
        matrix: Input glyph matrix
        containment_rule: Type of containment ("binary", "decimal", "string", etc.)
    """
    if containment_rule == "binary":
        return _binary_containment(matrix)
    elif containment_rule == "decimal":
        return _decimal_containment(matrix)
    elif containment_rule == "string":
        return _string_containment(matrix)
    else:
        return _general_containment(matrix)


def _binary_containment(matrix: jnp.ndarray) -> jnp.ndarray:
    """Binary containment: Forces equilibrium into 2×2 attractor blocks."""
    rows, cols = matrix.shape
    result = jnp.zeros_like(matrix)

    # Create 2×2 binary attractor pattern
    binary_attractor = jnp.array(
        [[GLYPH_TO_INT[Glyph.UNITY], GLYPH_TO_INT[Glyph.DOWN]], 
         [GLYPH_TO_INT[Glyph.DOWN], GLYPH_TO_INT[Glyph.UNITY]]]
    )

    # Tile the attractor across the matrix
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            end_i = min(i + 2, rows)
            end_j = min(j + 2, cols)
            result = result.at[i:end_i, j:end_j].set(binary_attractor[: end_i - i, : end_j - j])

    return result


def _decimal_containment(matrix: jnp.ndarray) -> jnp.ndarray:
    """Decimal containment: Forces equilibrium into 10×10 attractor blocks."""
    rows, cols = matrix.shape
    result = jnp.zeros_like(matrix)

    # Create 10×10 decimal attractor (simplified - diagonal pattern)
    decimal_attractor = jnp.full((10, 10), GLYPH_TO_INT[Glyph.DOWN])
    for k in range(10):
        decimal_attractor = decimal_attractor.at[k, k].set(GLYPH_TO_INT[Glyph.UNITY])

    # Tile across matrix
    for i in range(0, rows, 10):
        for j in range(0, cols, 10):
            end_i = min(i + 10, rows)
            end_j = min(j + 10, cols)
            result = result.at[i:end_i, j:end_j].set(decimal_attractor[: end_i - i, : end_j - j])

    return result


def _string_containment(matrix: jnp.ndarray) -> jnp.ndarray:
    """String containment: Propagates glyphs horizontally to form string sequences."""
    result = matrix.copy()
    rows, cols = matrix.shape

    for i in range(rows):
        # Start from diagonal and propagate right using string grammar
        if i < cols:
            current_glyph_int = int(matrix[i, i])
            current_glyph = INT_TO_GLYPH[current_glyph_int]
            
            for j in range(i + 1, cols):
                # Simple string grammar: DOWN → UP → FLOW → DOWN...
                if current_glyph == Glyph.DOWN:
                    current_glyph = Glyph.UP
                elif current_glyph == Glyph.UP:
                    current_glyph = Glyph.FLOW
                elif current_glyph == Glyph.FLOW:
                    current_glyph = Glyph.DOWN

                result = result.at[i, j].set(GLYPH_TO_INT[current_glyph])

    return result


def _general_containment(matrix: jnp.ndarray) -> jnp.ndarray:
    """General containment: Applies local averaging to create smooth resonance."""
    # Simple averaging with neighbors to create equilibrium
    kernel = jnp.array([[0.1, 0.2, 0.1], [0.2, 0.0, 0.2], [0.1, 0.2, 0.1]])

    # Apply convolution to smooth out sharp dissonances
    padded = jnp.pad(matrix.astype(jnp.float32), 1, mode="edge")
    result = jax.scipy.signal.convolve2d(padded, kernel, mode="valid")

    # Quantize back to glyph values
    return jnp.round(jnp.clip(result, 0, len(GLYPH_TO_INT) - 1)).astype(jnp.int32)


# --- Resonance Analysis ---


def compute_resonance_trace(matrix: jnp.ndarray) -> float:
    """
    Compute the resonance trace ℜ_ω of a glyph matrix.

    This measures how much dissonance remains in the system.
    ℜ_ω = 0 indicates perfect resonance/equilibrium.
    """
    rows, cols = matrix.shape
    total_dissonance = 0.0

    # Check diagonal-neighbor dissonance
    min_dim = min(rows, cols)
    for i in range(min_dim):
        diagonal_glyph = INT_TO_GLYPH[int(matrix[i, i])]
        expected_glyph = omega(diagonal_glyph)

        # Check if neighbors match expected transformation
        neighbors = []
        if i + 1 < cols:
            neighbors.append(matrix[i, i + 1])
        if i + 1 < rows:
            neighbors.append(matrix[i + 1, i])

        for neighbor_val in neighbors:
            if neighbor_val != GLYPH_TO_INT[expected_glyph]:
                total_dissonance += 1.0

    # Normalize by total number of checked positions
    max_positions = min_dim * 2  # Up to 2 neighbors per diagonal element
    return total_dissonance / max(max_positions, 1)


# --- Composite Operations ---


def Wild_closure(matrix: jnp.ndarray, containment_rule: str = "binary", max_iterations: int = 100) -> jnp.ndarray:
    """
    Apply Wild-Tame cycles until resonance equilibrium is reached.

    This is the fundamental process that generates stable symbolic structures
    from initial uniform fields.
    """
    current = matrix

    for iteration in range(max_iterations):
        # Apply wildness
        wild = Wild_operator(current)

        # Apply containment
        tamed = Tame_operator(wild, containment_rule)

        # Check for equilibrium
        if jnp.array_equal(tamed, current):
            print(f"Equilibrium reached after {iteration + 1} iterations.")
            return tamed
        
        current = tamed

    print(f"Max iterations ({max_iterations}) reached without equilibrium.")
    return current


# --- Existing Core Operators ---

def fuse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Fusion operator (⊕): Combines two elements."""
    # Defaulting to simple addition for now.
    return a + b


def tensor(a: jax.Array, b: jax.Array) -> jax.Array:
    """Tensor operator (⊗): Binds two elements."""
    # Defaulting to multiplication.
    return a * b


def reflect(a: jax.Array) -> jax.Array:
    """Reflection operator (~): Inverts phase or orientation."""
    # Defaulting to negation.
    return -a


def descent(a: jax.Array) -> jax.Array:
    """Descent operator (ℓ): Folds or regularizes a process."""
    # This will eventually be a complex function, e.g., Ramanujan summation proxy.
    # For now, a simple identity or compression.
    return a


def growth(a: jax.Array, n: Any) -> jax.Array:
    """Growth operator (↑): Increases dimensionality or scale."""
    # Defaulting to exponentiation.
    return a**n


def curvature(a: jax.Array) -> jax.Array:
    """
    Curvature operator (κ): Computes the discrete Laplacian across all dimensions
    using periodic boundary conditions.
    """
    dim = a.ndim
    if dim == 0:
        return jnp.zeros_like(a)
    
    lap = jnp.zeros_like(a)
    for i in range(dim):
        lap += jnp.roll(a, shift=1, axis=i) + jnp.roll(a, shift=-1, axis=i) - 2 * a
    return lap

def taylorphasewalk(f: Callable[[jnp.ndarray], jnp.ndarray],
                     x0: jnp.ndarray,
                     order: int) -> jnp.ndarray:
    """
    Compute the univariate Taylor series coefficients of f around x0 up to the given order.
    Returns an array of shape (order+1,) where coeffs[k] = f^{(k)}(x0) / k!.
    """
    deriv_fn = f
    coeffs: List[jnp.ndarray] = []
    for k in range(order + 1):
        if k == 0:
            val = deriv_fn(x0)
        else:
            deriv_fn = jax.grad(deriv_fn)
            val = deriv_fn(x0)
        coeffs.append(val / math.factorial(k))
    return jnp.stack(coeffs)


def taylorphasewalk_inverse(coeffs: jnp.ndarray) -> jnp.ndarray:
    """
    Given univariate Taylor coefficients coeffs of f at x0 (coeffs[k] = f^{(k)}(x0)/k!),
    compute the coefficients of 1/f up to the same order via formal power series inversion.
    """
    order = coeffs.shape[0] - 1
    inv_coeffs: List[jnp.ndarray] = [1.0 / coeffs[0]]
    for n in range(1, order + 1):
        s = 0.0
        for k in range(1, n + 1):
            s += coeffs[k] * inv_coeffs[n - k]
        inv_coeffs.append(-s / coeffs[0])
    return jnp.stack(inv_coeffs)


def taylorphasewalk_multivariate(f: Callable,
                                 x0: jnp.ndarray,
                                 order: int) -> List[jnp.ndarray]:
    """
    Compute multivariate Taylor series of f around x0 up to total degree `order`.
    Returns a list of arrays where the k-th element is the k-th order derivative tensor
    f^{(k)}(x0)/k!.
    """
    deriv_fn = f
    coeffs: List[jnp.ndarray] = []
    # zeroth order
    coeffs.append(deriv_fn(x0))
    for k in range(1, order + 1):
        deriv_fn = jax.jacfwd(deriv_fn)
        coeffs.append(deriv_fn(x0) / math.factorial(k))
    return coeffs

# --- Base System Arithmetic ---


def extract_binary_blocks(matrix: jnp.ndarray) -> List[Tuple[int, int, int]]:
    """
    Extract 2x2 binary blocks from a matrix and convert to decimal values.

    Returns list of (row, col, decimal_value) tuples for each 2x2 block.
    ⊙ = 0, ▽ = 1 in binary interpretation.
    """
    rows, cols = matrix.shape
    blocks = []

    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            if i + 1 < rows and j + 1 < cols:
                # Extract 2x2 block
                block = matrix[i : i + 2, j : j + 2]

                # Convert to binary (⊙=0, ▽=1, others=0)
                binary_block = jnp.where(
                    block == GLYPH_TO_INT[Glyph.UNITY], 0, jnp.where(block == GLYPH_TO_INT[Glyph.DOWN], 1, 0)
                )

                # Interpret as 2-bit binary number (row-major order)
                # [[a,b], [c,d]] -> a*8 + b*4 + c*2 + d*1
                decimal_value = (
                    int(binary_block[0, 0]) * 8
                    + int(binary_block[0, 1]) * 4
                    + int(binary_block[1, 0]) * 2
                    + int(binary_block[1, 1]) * 1
                )

                blocks.append((i, j, int(decimal_value)))

    return blocks


def matrix_to_binary_number(matrix: jnp.ndarray) -> int:
    """
    Convert an entire matrix of binary patterns to a single binary number.

    Reads the first row only, rightmost column = least significant bit.
    Each column represents a power of 2.
    """
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return 0

    binary_value = 0

    # Read only the first row, from right to left (least to most significant)
    for col in range(cols):
        bit_position = cols - 1 - col  # Rightmost column = position 0

        if matrix[0, col] == GLYPH_TO_INT[Glyph.DOWN]:
            binary_value += 2 ** bit_position
        # UNITY and others count as 0, so no addition needed

    return binary_value


def binary_number_to_matrix(number: int, rows: int, cols: int) -> jnp.ndarray:
    """
    Convert a binary number back to a glyph matrix.

    Creates a matrix where columns represent powers of 2,
    and ⊙=0, ▽=1 encode the binary digits.
    """
    result = create_glyph_matrix((rows, cols), Glyph.UNITY)  # Start with all 0s

    # Convert number to binary string
    binary_str = bin(number)[2:]  # Remove '0b' prefix

    # Place binary digits in rightmost columns
    for i, bit_char in enumerate(reversed(binary_str)):
        if i < cols:  # Don't exceed matrix width
            bit_value = int(bit_char)
            glyph_value = GLYPH_TO_INT[Glyph.DOWN] if bit_value == 1 else GLYPH_TO_INT[Glyph.UNITY]

            # Fill entire column with this bit pattern
            # For simplicity, just set the first row
            if rows > 0:
                result = result.at[0, cols - 1 - i].set(glyph_value)

    return result


def matrix_binary_add(matrix_a: jnp.ndarray, matrix_b: jnp.ndarray) -> jnp.ndarray:
    """
    Add two matrices as binary numbers using emergent base system.

    This demonstrates actual arithmetic using the generated patterns.
    """
    # Convert matrices to binary numbers
    num_a = matrix_to_binary_number(matrix_a)
    num_b = matrix_to_binary_number(matrix_b)

    # Perform addition
    result_num = num_a + num_b

    # Convert back to matrix format
    rows, cols = matrix_a.shape
    return binary_number_to_matrix(result_num, rows, cols)


def matrix_binary_multiply(matrix_a: jnp.ndarray, matrix_b: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two matrices as binary numbers.
    """
    num_a = matrix_to_binary_number(matrix_a)
    num_b = matrix_to_binary_number(matrix_b)
    result_num = num_a * num_b

    rows, cols = matrix_a.shape
    return binary_number_to_matrix(result_num, rows, cols)


def verify_base_emergence(test_size: int = 4) -> bool:
    """
    Verify that the emergent base system actually performs correct arithmetic.

    Tests whether the generated patterns can represent numbers and
    perform arithmetic operations correctly.
    """
    # Create two small numbers in matrix form
    matrix_3 = binary_number_to_matrix(3, test_size, test_size)  # Binary: 11
    matrix_5 = binary_number_to_matrix(5, test_size, test_size)  # Binary: 101

    # Test addition: 3 + 5 = 8
    result_add = matrix_binary_add(matrix_3, matrix_5)
    computed_sum = matrix_to_binary_number(result_add)

    # Test multiplication: 3 * 5 = 15
    result_mult = matrix_binary_multiply(matrix_3, matrix_5)
    computed_product = matrix_to_binary_number(result_mult)

    # Verify correctness
    addition_correct = computed_sum == 8
    multiplication_correct = computed_product == 15

    return addition_correct and multiplication_correct

# --- String Generation System ---

class Grammar:
    """Production rules for string generation."""
    
    def __init__(self, rules: Dict[Glyph, List[Glyph]]):
        self.rules = rules
    
    def apply_rule(self, glyph: Glyph) -> Glyph:
        """Apply production rule to a single glyph."""
        if glyph in self.rules and self.rules[glyph]:
            # For simplicity, take first production
            return self.rules[glyph][0] 
        return glyph  # No rule, return unchanged

# Predefined grammars
SIMPLE_GRAMMAR = Grammar({
    Glyph.DOWN: [Glyph.UP],
    Glyph.UP: [Glyph.FLOW], 
    Glyph.FLOW: [Glyph.DOWN],
    Glyph.VOID: [Glyph.DOWN],
    Glyph.UNITY: [Glyph.UNITY]  # Fixed point
})

BINARY_GRAMMAR = Grammar({
    Glyph.UNITY: [Glyph.DOWN],  # 0 → 1
    Glyph.DOWN: [Glyph.UNITY],  # 1 → 0  
    Glyph.VOID: [Glyph.UNITY]   # Start with 0
})

FIBONACCI_GRAMMAR = Grammar({
    Glyph.DOWN: [Glyph.UP],           # A → B
    Glyph.UP: [Glyph.DOWN, Glyph.UP], # B → AB  
    Glyph.VOID: [Glyph.DOWN]          # Start with A
})

def generate_string_from_seed(seed_glyph: Glyph, grammar: Grammar, 
                             length: int, matrix_width: int = 12) -> jnp.ndarray:
    """
    Generate a string by applying grammar rules starting from a seed glyph.
    
    Creates a matrix where the first row contains the generated string.
    """
    result = create_glyph_matrix((3, matrix_width), Glyph.VOID)
    
    if length == 0 or matrix_width == 0:
        return result
    
    # Start with seed glyph
    current_glyph = seed_glyph
    result = result.at[0, 0].set(GLYPH_TO_INT[current_glyph])
    
    # Generate string by applying grammar rules
    for pos in range(1, min(length, matrix_width)):
        current_glyph = grammar.apply_rule(current_glyph)
        result = result.at[0, pos].set(GLYPH_TO_INT[current_glyph])
    
    return result

def extract_string_from_matrix(matrix: jnp.ndarray, row: int = 0) -> List[Glyph]:
    """Extract a string (list of glyphs) from a matrix row."""
    rows, cols = matrix.shape
    if row >= rows:
        return []
    
    string = []
    for col in range(cols):
        glyph_value = int(matrix[row, col])
        try:
            glyph = INT_TO_GLYPH[glyph_value]
            string.append(glyph)
        except ValueError:
            string.append(Glyph.VOID)  # Default for invalid values
    
    return string

def string_to_text(glyph_string: List[Glyph]) -> str:
    """Convert a glyph string to readable text."""
    symbol_map = {
        Glyph.VOID: "∅",
        Glyph.DOWN: "▽",
        Glyph.UP: "△", 
        Glyph.UNITY: "⊙",
        Glyph.FLOW: "⊕"
    }
    return "".join(symbol_map.get(g, "?") for g in glyph_string)

def apply_string_grammar_matrix(matrix: jnp.ndarray, grammar: Grammar) -> jnp.ndarray:
    """
    Apply string containment with custom grammar rules.
    
    Enhanced version of _string_containment with configurable grammar.
    """
    result = matrix.copy()
    rows, cols = matrix.shape
    
    for i in range(rows):
        # Start from diagonal and propagate right using grammar
        if i < cols and matrix[i, i] != GLYPH_TO_INT[Glyph.VOID]:
            current_glyph_int = int(matrix[i, i])
            current_glyph = INT_TO_GLYPH[current_glyph_int]
            for j in range(i + 1, cols):
                current_glyph = grammar.apply_rule(current_glyph)
                result = result.at[i, j].set(GLYPH_TO_INT[current_glyph])
    
    return result

def generate_language_samples(grammar: Grammar, start_glyph: Glyph, 
                            num_samples: int = 5, max_length: int = 10) -> List[str]:
    """
    Generate multiple string samples from a grammar.
    
    Returns list of string representations.
    """
    samples = []
    
    for i in range(num_samples):
        # Vary the length for diversity
        length = min(max_length, 3 + i * 2)
        matrix = generate_string_from_seed(start_glyph, grammar, length)
        glyph_string = extract_string_from_matrix(matrix)
        
        # Only take non-void characters 
        non_void = [g for g in glyph_string if g != Glyph.VOID]
        if non_void:
            text = string_to_text(non_void)
            samples.append(text)
    
    return samples

def recognize_pattern(string: List[Glyph], pattern: List[Glyph]) -> bool:
    """
    Check if a string contains a specific pattern.
    
    Simple pattern matching for language recognition.
    """
    if len(pattern) > len(string):
        return False
    
    for i in range(len(string) - len(pattern) + 1):
        if string[i:i+len(pattern)] == pattern:
            return True
    
    return False

def string_concatenate_matrices(matrix_a: jnp.ndarray, matrix_b: jnp.ndarray) -> jnp.ndarray:
    """
    Concatenate two string matrices horizontally.
    
    Demonstrates string operations using matrix operations.
    """
    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape
    
    # Use minimum rows to avoid dimension mismatch
    rows = min(rows_a, rows_b)
    result = create_glyph_matrix((rows, cols_a + cols_b), Glyph.VOID)
    
    # Copy first matrix
    result = result.at[:rows, :cols_a].set(matrix_a[:rows, :])
    
    # Copy second matrix
    result = result.at[:rows, cols_a:].set(matrix_b[:rows, :])
    
    return result

def verify_string_generation() -> bool:
    """
    Verify that string generation system works correctly.
    
    Tests grammar application, pattern recognition, and string operations.
    """
    # Test 1: Simple grammar generates expected sequence
    matrix = generate_string_from_seed(Glyph.DOWN, SIMPLE_GRAMMAR, 6)
    string = extract_string_from_matrix(matrix)
    expected = [Glyph.DOWN, Glyph.UP, Glyph.FLOW, Glyph.DOWN, Glyph.UP, Glyph.FLOW]
    
    if string[:len(expected)] != expected:
        return False
    
    # Test 2: Binary grammar alternates correctly  
    binary_matrix = generate_string_from_seed(Glyph.UNITY, BINARY_GRAMMAR, 4)
    binary_string = extract_string_from_matrix(binary_matrix)
    expected_binary = [Glyph.UNITY, Glyph.DOWN, Glyph.UNITY, Glyph.DOWN]
    
    if binary_string[:len(expected_binary)] != expected_binary:
        return False
    
    # Test 3: Pattern recognition works
    pattern = [Glyph.DOWN, Glyph.UP]
    if not recognize_pattern(string, pattern):
        return False
    
    # Test 4: String concatenation works
    concat_result = string_concatenate_matrices(matrix, binary_matrix)
    if concat_result.shape[1] != matrix.shape[1] + binary_matrix.shape[1]:
        return False
    
    return True

# --- Existing operators continue...
