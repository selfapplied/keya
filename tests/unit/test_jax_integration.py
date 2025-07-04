"""
Simple JAX test to verify hardware acceleration is working.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np


def test_jax_backends():
    """Test which JAX backends are available and working."""
    print("=== JAX Backend Test ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available backends: {jax.local_devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Test basic operations
    try:
        x = jnp.array([1, 2, 3, 4, 5])
        y = jnp.array([2, 3, 4, 5, 6])
        result = x + y
        print(f"Basic operation test: {x} + {y} = {result}")
        print("✅ JAX basic operations working")
    except Exception as e:
        print(f"❌ JAX basic operations failed: {e}")
        return False
    
    return True

def test_jax_acceleration():
    """Test JAX JIT compilation and acceleration."""
    print("\n=== JAX Acceleration Test ===")
    
    @jax.jit
    def matrix_multiply(a, b):
        return jnp.dot(a, b)
    
    # Create test matrices
    size = 1000
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (size, size))
    b = jax.random.normal(key, (size, size))
    
    # Test JAX JIT
    try:
        print("Testing JAX JIT compilation...")
        start_time = time.time()
        result_jax = matrix_multiply(a, b)
        jax_time = time.time() - start_time
        print(f"JAX JIT time: {jax_time:.4f}s")
        print(f"Result shape: {result_jax.shape}")
        print("✅ JAX JIT acceleration working")
        
        # Compare with numpy
        a_np = np.array(a)
        b_np = np.array(b)
        start_time = time.time()
        np.dot(a_np, b_np)  # Perform computation for timing
        numpy_time = time.time() - start_time
        print(f"NumPy time: {numpy_time:.4f}s")
        
        speedup = numpy_time / jax_time
        print(f"JAX speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX JIT failed: {e}")
        return False

def test_simple_operators():
    """Test simple JAX operations that we'll need for keya."""
    print("\n=== JAX Operations Test ===")
    
    try:
        # Test matrix creation
        matrix = jnp.zeros((5, 5), dtype=jnp.int32)
        print(f"Matrix creation: {matrix.shape}")
        
        # Test element assignment (JAX style)
        matrix = matrix.at[0, 0].set(42)
        matrix = matrix.at[1, 1].set(99)
        print(f"Element assignment: matrix[0,0]={matrix[0,0]}, matrix[1,1]={matrix[1,1]}")
        
        # Test array operations
        ones = jnp.ones((3, 3))
        twos = jnp.full((3, 3), 2)
        result = ones + twos
        print(f"Array operations: ones + twos = \n{result}")
        
        print("✅ All JAX operations working")
        return True
        
    except Exception as e:
        print(f"❌ JAX operations failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing JAX setup for keya...")
    
    backends_ok = test_jax_backends()
    if not backends_ok:
        print("❌ JAX backends not working properly")
        exit(1)
    
    acceleration_ok = test_jax_acceleration()
    if not acceleration_ok:
        print("⚠️  JAX acceleration may not be optimal")
    
    operations_ok = test_simple_operators()
    if not operations_ok:
        print("❌ JAX operations not working")
        exit(1)
    
    print("\n🎉 JAX is ready for keya development!")
    print("Hardware acceleration:", "✅" if acceleration_ok else "⚠️") 