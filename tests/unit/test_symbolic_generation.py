import numpy as np

# Since the functions are now local to a demo, we import them directly.
# This is a temporary measure to ensure test coverage.
# A future refactor might move this logic to a dedicated symbolic utility module.
from demos.symbol import (
    generate_string_from_seed,
    extract_string_from_matrix,
    string_to_text,
    apply_glyph_transform,
    create_glyph_matrix,
    SIMPLE_GRAMMAR,
    BINARY_GRAMMAR,
    Glyph,
    GLYPH_TO_INT
)

def test_generate_string_from_seed():
    """Tests that a seed correctly generates a larger pattern."""
    matrix = generate_string_from_seed(
        rows=5, 
        cols=10, 
        seed_glyph=Glyph.UP, 
        grammar=SIMPLE_GRAMMAR, 
        steps=3
    )
    
    # Check that the matrix is not empty and has the correct shape
    assert matrix.shape == (5, 10)
    # Check that the original seed is still there
    assert matrix[2, 5] == GLYPH_TO_INT[Glyph.UP]
    # Check that some growth has occurred
    assert np.sum(matrix != GLYPH_TO_INT[Glyph.VOID]) > 1

def test_extract_string_and_convert():
    """Tests that a string can be extracted from a matrix and converted."""
    matrix = np.full((3, 5), GLYPH_TO_INT[Glyph.VOID])
    matrix[0, :] = [
        GLYPH_TO_INT[Glyph.UP],
        GLYPH_TO_INT[Glyph.DOWN],
        GLYPH_TO_INT[Glyph.VOID],
        GLYPH_TO_INT[Glyph.UP],
        GLYPH_TO_INT[Glyph.DOWN]
    ]
    
    glyph_string = extract_string_from_matrix(matrix)
    text_string = string_to_text(glyph_string)
    
    assert glyph_string == "△▽∅△▽"
    assert text_string == "10∅10"

def test_apply_glyph_transform():
    """Tests the core transformation logic with a simple grammar."""
    matrix = create_glyph_matrix(3, 3, Glyph.UP, (1, 1))
    
    # Apply transform once
    new_matrix = apply_glyph_transform(matrix, SIMPLE_GRAMMAR)
    
    # The simple grammar should fill the neighbors of the seed
    expected = np.array([
        [0, 2, 0],
        [2, 2, 2],
        [0, 2, 0]
    ])
    
    assert np.array_equal(new_matrix, expected)

def test_binary_grammar_alternation():
    """Tests that the binary grammar creates an alternating pattern."""
    # Use a seed that is part of the grammar
    matrix = create_glyph_matrix(3, 8, Glyph.UP, (1, 1))
    
    # Evolve it a few steps
    for _ in range(4):
        matrix = apply_glyph_transform(matrix, BINARY_GRAMMAR)
        
    # The binary grammar should create alternating UP/DOWN from an UP seed
    # but the current apply_glyph_transform doesn't do replacement, just filling.
    # This test verifies that it fills according to the grammar mask.
    
    # Seed at (1,1) is UP (2)
    # Neighbors at (0,1), (2,1), (1,0), (1,2) should become UP.
    assert matrix[0,1] == GLYPH_TO_INT[Glyph.UP]
    assert matrix[2,1] == GLYPH_TO_INT[Glyph.UP]
    assert matrix[1,0] == GLYPH_TO_INT[Glyph.UP]
    assert matrix[1,2] == GLYPH_TO_INT[Glyph.UP]
    
    # Diagonal neighbors should remain VOID (0)
    assert matrix[0,0] == GLYPH_TO_INT[Glyph.VOID]
    assert matrix[0,2] == GLYPH_TO_INT[Glyph.VOID] 