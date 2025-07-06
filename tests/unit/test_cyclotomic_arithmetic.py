import jax.numpy as jnp
from keya.kernel.cyclotomic import CyclotomicBinary

def test_number_representation():
    """Tests that integers are correctly represented as CyclotomicBinary objects."""
    # Test a few integers
    num_3 = CyclotomicBinary(3)
    assert num_3.value == 3
    assert num_3.components.tolist() == [1, 1]

    num_10 = CyclotomicBinary(10)
    assert num_10.value == 10
    assert num_10.components.tolist() == [0, 1, 0, 1]
    
    num_0 = CyclotomicBinary(0)
    assert num_0.value == 0
    assert num_0.components.tolist() == [0]

def test_from_vector():
    """Tests creating a CyclotomicBinary object from a coefficient vector."""
    vector = [1, 0, 1, 1] # Represents 1*1 + 0*2 + 1*4 + 1*8 = 13
    num = CyclotomicBinary.from_vector(jnp.array(vector))
    assert num.value == 13
    assert num.components.tolist() == vector

def test_addition():
    """Tests the __add__ method of the CyclotomicBinary class."""
    # 3 (0b11) + 5 (0b101) = 8 (0b1000)
    num_3 = CyclotomicBinary(3, kernel_depth=8)
    num_5 = CyclotomicBinary(5, kernel_depth=8)
    result = num_3 + num_5
    assert result.value == 8
    
    # 7 (0b111) + 1 (0b1) = 8 (0b1000)
    num_7 = CyclotomicBinary(7, kernel_depth=8)
    num_1 = CyclotomicBinary(1, kernel_depth=8)
    result_2 = num_7 + num_1
    assert result_2.value == 8

    # Test with carries
    # 1 (0b1) + 1 (0b1) = 2 (0b10)
    num_1_a = CyclotomicBinary(1, kernel_depth=8)
    num_1_b = CyclotomicBinary(1, kernel_depth=8)
    result_3 = num_1_a + num_1_b
    assert result_3.value == 2

def test_multiplication():
    """Tests the __mul__ method of the CyclotomicBinary class."""
    # 3 * 5 = 15
    num_3 = CyclotomicBinary(3)
    num_5 = CyclotomicBinary(5)
    result = num_3 * num_5
    assert result.value == 15
    
    # 7 * 0 = 0
    num_7 = CyclotomicBinary(7)
    num_0 = CyclotomicBinary(0)
    result_2 = num_7 * num_0
    assert result_2.value == 0
    
    # 10 * 10 = 100
    num_10 = CyclotomicBinary(10)
    result_3 = num_10 * num_10
    assert result_3.value == 100 