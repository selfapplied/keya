import pytest
import jax.numpy as jnp

from keya.kernel.kernel import PascalKernel
from keya.kernel.operators import Fuse, Diff, Identity, Operator

@pytest.fixture
def kernel():
    """Provides a PascalKernel instance for tests."""
    return PascalKernel()

def test_identity_operator(kernel):
    """The Identity operator should not change the state."""
    state = jnp.array([1, 0, 1, 1, 0], dtype=jnp.int32)
    identity_op = Identity()
    
    new_state = kernel.apply_polynomial(state, identity_op.coeffs)
    
    # After convolution with [1], the state should be unchanged.
    assert jnp.array_equal(state, new_state)

def test_fuse_operator(kernel):
    """
    The Fuse operator is equivalent to multiplication by (1+x),
    which should perform a left-shift and add.
    [a, b, c] -> [a, a+b, b+c, c]
    """
    state = jnp.array([1, 1, 0, 1], dtype=jnp.int32)
    fuse_op = Fuse() # Corresponds to polynomial 1 + 1*x
    
    new_state = kernel.apply_polynomial(state, fuse_op.coeffs)
    
    # Expected result: [1, 1, 0, 1] convolved with [1, 1] -> [1, 2, 1, 1, 1]
    expected = jnp.array([1, 2, 1, 1, 1], dtype=jnp.int32)
    
    assert jnp.array_equal(new_state, expected)

def test_diff_operator(kernel):
    """
    The Diff operator is the inverse of Fuse for this field. Applying 
    Diff then Fuse should result in the Identity transformation.
    """
    state = jnp.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=jnp.int32)
    
    # Create operators
    diff_op = Diff()
    fuse_op = Fuse()

    # Apply Diff then Fuse
    state_after_diff = kernel.apply_polynomial(state, diff_op.coeffs)
    state_after_both = kernel.apply_polynomial(state_after_diff, fuse_op.coeffs)
    
    # The result of (x-1)(x+1) = x^2 - 1. So we expect a shift and a subtraction.
    # state_after_both should equal state convolved with [-1, 0, 1]
    identity_op = Operator("x^2-1", [-1, 0, 1])
    expected_state = kernel.apply_polynomial(state, identity_op.coeffs)

    assert jnp.array_equal(state_after_both, expected_state)


def test_operator_chaining(kernel):
    """Tests applying a sequence of operators."""
    state = jnp.array([1, 1, 1], dtype=jnp.int32)
    
    # Chain: Fuse() -> Fuse() -> Diff()
    # This should be equivalent to just applying Fuse() once convolved with (x^2-1)
    fuse = Fuse()
    diff = Diff()
    
    intermediate_1 = kernel.apply_polynomial(state, fuse.coeffs)
    intermediate_2 = kernel.apply_polynomial(intermediate_1, fuse.coeffs)
    final_state = kernel.apply_polynomial(intermediate_2, diff.coeffs)
    
    # The equivalent single operation is state * (1+x) * (1+x) * (1-x)
    # = state * (1+x) * (1-x^2) = state * (1 -x^2 + x - x^3)
    # So the operator is [1, 1, -1, -1]
    equiv_op = Operator("equiv", [1, 1, -1, -1])
    expected_state = kernel.apply_polynomial(state, equiv_op.coeffs)
    
    assert jnp.array_equal(final_state, expected_state) 