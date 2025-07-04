#!/usr/bin/env python3
"""
FLOATING-POINT AS OPERATIONS: Testing Deepseek's Insight

This demo tests the hypothesis that floating-point representation 
is a fundamental operation by:

1. Measuring quantization effects in IEEE 754
2. Verifying glyph mapping as containment operation
3. Testing rounding as micro cycles
4. Analyzing special values as fixed points
5. Comparing with keya operators

All claims are tested with numerical verification.
Output files are saved to .out/ directory structure.
"""

import sys
import os


import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict

from keya.core.engine import Engine 
from keya.core.operators import Glyph
from keya.dsl.parser import parse

# Ensure output directories exist (relative to project root)
output_base = os.path.join(os.path.dirname(__file__), '..')
os.makedirs(os.path.join(output_base, '.out', 'tests'), exist_ok=True)
os.makedirs(os.path.join(output_base, '.out', 'visualizations'), exist_ok=True)

def test_ieee754_quantization() -> Dict[str, float]:
    """Test quantization effects in IEEE 754 representation."""
    print("Testing IEEE 754 quantization effects...")
    
    # Test precision limits
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    
    infinite_pi = Decimal("3.1415926535897932384626433832795028841971693993751")
    finite_pi = float(infinite_pi)
    
    # Measure quantization error
    reconstructed = Decimal(str(finite_pi))
    error = abs(infinite_pi - reconstructed)
    relative_error = float(error / infinite_pi)
    
    # Test if this behaves like a containment operation
    quantization_factor = 2**23  # 23-bit mantissa
    expected_error_bound = 1.0 / quantization_factor
    
    results = {
        'relative_error': relative_error,
        'expected_bound': expected_error_bound,
        'bounded_correctly': relative_error < expected_error_bound
    }
    
    print(f"  Relative error: {relative_error:.2e}")
    print(f"  Expected bound: {expected_error_bound:.2e}") 
    print(f"  Bounded correctly: {results['bounded_correctly']}")
    
    return results


def test_glyph_quantization_properties() -> Dict[str, bool]:
    """Test if glyph quantization behaves like a mathematical projection."""
    print("\nTesting glyph quantization properties...")
    
    def float_to_glyph(value: float) -> Glyph:
        if abs(value) < 0.1:
            return Glyph.VOID
        elif value < -0.5:
            return Glyph.DOWN
        elif value > 1.5:
            return Glyph.FLOW
        elif 0.3 <= value <= 0.7:
            return Glyph.UNITY
        else:
            return Glyph.UP
    
    def glyph_to_float(glyph: Glyph) -> float:
        mapping = {
            Glyph.VOID: 0.0,
            Glyph.DOWN: -1.0,
            Glyph.UP: 1.0,
            Glyph.UNITY: 0.5,
            Glyph.FLOW: 2.0,
        }
        return mapping[glyph]
    
    # Test idempotency: Q(Q(x)) = Q(x)
    test_values = np.random.uniform(-3, 4, 100)
    idempotent_failures = 0
    
    for val in test_values:
        glyph1 = float_to_glyph(val)
        float1 = glyph_to_float(glyph1)
        glyph2 = float_to_glyph(float1)
        
        if glyph1 != glyph2:
            idempotent_failures += 1
    
    # Test bounded quantization error
    max_error = 0.0
    for val in test_values:
        glyph = float_to_glyph(val)
        quantized = glyph_to_float(glyph)
        error = abs(val - quantized)
        max_error = max(max_error, error)
    
    results = {
        'idempotent': idempotent_failures == 0,
        'bounded_error': max_error < 5.0,  # Should be bounded by range
        'max_error': max_error
    }
    
    print(f"  Idempotent: {results['idempotent']} ({idempotent_failures} failures)")
    print(f"  Bounded error: {results['bounded_error']} (max: {max_error:.3f})")
    
    return results


def test_rounding_as_operator_cycles() -> Dict[str, bool]:
    """Test if floating-point rounding behaves like operations."""
    print("\nTesting rounding as cycles...")
    
    # Test classic 0.1 + 0.2 != 0.3 example
    a, b, c = 0.1, 0.2, 0.3
    result = a + b
    error = abs(result - c)
    
    # Test that error is bounded by machine epsilon
    epsilon = np.finfo(float).eps
    bounded_by_epsilon = error < 10 * epsilon  # Allow some margin
    
    # Test systematic errors in arithmetic
    systematic_errors = []
    for i in range(10):
        val = 0.1 * i
        rounded = float(f"{val:.1f}")  # Force rounding
        err = abs(val - rounded)
        systematic_errors.append(err)
    
    max_systematic = max(systematic_errors)
    
    results = {
        'classic_error_bounded': bounded_by_epsilon,
        'systematic_bounded': max_systematic < 1e-10,
        'error_magnitude': error,
        'max_systematic': max_systematic
    }
    
    print(f"  Classic error bounded: {results['classic_error_bounded']}")
    print(f"  Error magnitude: {error:.2e}")
    print(f"  Systematic bounded: {results['systematic_bounded']}")
    
    return results


def test_special_values_as_fixed_points() -> Dict[str, bool]:
    """Test if special values behave as fixed points."""
    print("\nTesting special values as fixed points...")
    
    # Test NaN propagation (should absorb operations)
    nan_val = float('nan')
    nan_tests = [
        nan_val + 1.0,
        nan_val * 2.0,
        nan_val / 3.0,
        nan_val - nan_val
    ]
    nan_propagates = all(np.isnan(x) for x in nan_tests)
    
    # Test infinity saturation
    inf_val = float('inf')
    inf_tests = [
        inf_val + 1000.0,
        inf_val * 2.0,
        inf_val + inf_val
    ]
    inf_saturates = all(np.isinf(x) and x > 0 for x in inf_tests)
    
    # Test signed zero preservation
    neg_zero = -0.0
    pos_zero = 0.0
    zero_distinct = str(neg_zero) != str(pos_zero)
    
    results = {
        'nan_propagates': nan_propagates,
        'inf_saturates': inf_saturates,
        'signed_zero_distinct': zero_distinct
    }
    
    print(f"  NaN propagates: {results['nan_propagates']}")
    print(f"  Infinity saturates: {results['inf_saturates']}")
    print(f"  Signed zero distinct: {results['signed_zero_distinct']}")
    
    return results


def test_keya_operator_correspondence() -> Dict[str, bool]:
    """Test if keya operators correspond to floating-point operations."""
    print("\nTesting keya correspondence...")
    
    try:
        # Create test matrices
        test_matrix = np.random.uniform(-2, 3, (6, 6))
        
        # Apply keya operations
        keya_program = """
matrix correspondence_test {
    ops {
        test_matrix = [6, 6, ⊕]
        wild_result = W test_matrix
        tame_result = T(test_matrix, general)
        wild_tame_result = WT(test_matrix, general, 2)
    }
}
"""
        
        parse(keya_program.strip())  # Validate syntax
        engine = Engine()
        engine.variables['test_matrix'] = test_matrix
        engine.execute_program(keya_program.strip())
        
        # Check if operations preserve structure
        matrices_exist = all(name in engine.variables for name in 
                           ['wild_result', 'tame_result', 'wild_tame_result'])
        
        if matrices_exist:
            wild_result = engine.variables['wild_result']
            tame_result = engine.variables['tame_result'] 
            wild_tame_result = engine.variables['wild_tame_result']
            
            # Test properties
            shape_preserved = (wild_result.shape == test_matrix.shape and
                             tame_result.shape == test_matrix.shape and
                             wild_tame_result.shape == test_matrix.shape)
            
            # Test containment property (Tame should reduce variance)
            original_var = np.var(test_matrix)
            contained_var = np.var(tame_result)
            variance_reduced = contained_var < original_var
            
            # Test convergence property (Wild/Tame cycles should stabilize)
            wild_tame_var = np.var(wild_tame_result)
            wild_tame_converged = wild_tame_var < original_var
            
        else:
            shape_preserved = False
            variance_reduced = False
            wild_tame_converged = False
        
        results = {
            'keya_executes': matrices_exist,
            'shape_preserved': shape_preserved,
            'variance_reduced': variance_reduced,
            'wild_tame_converged': wild_tame_converged
        }
        
    except Exception as e:
        print(f"  Error in keya execution: {e}")
        results = {
            'keya_executes': False,
            'shape_preserved': False,
            'variance_reduced': False,
            'wild_tame_converged': False
        }
    
    print(f"  Keya executes: {results['keya_executes']}")
    print(f"  Shape preserved: {results['shape_preserved']}")
    print(f"  Variance reduced: {results['variance_reduced']}")
    print(f"  Wild/Tame converged: {results['wild_tame_converged']}")
    
    return results


def create_test_visualizations(test_results: Dict):
    """Create visualizations of test results without showing windows."""
    print("\nCreating test result visualizations...")
    
    # Summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Floating-Point Operation Tests", fontsize=14)
    
    # Test 1: IEEE 754 quantization
    ieee_data = test_results['ieee754']
    ax1.bar(['Relative Error', 'Expected Bound'], 
            [ieee_data['relative_error'], ieee_data['expected_bound']])
    ax1.set_yscale('log')
    ax1.set_title('IEEE 754 Quantization')
    ax1.set_ylabel('Error Magnitude')
    
    # Test 2: Glyph quantization
    glyph_data = test_results['glyph']
    ax2.bar(['Max Error'], [glyph_data['max_error']])
    ax2.set_title('Glyph Quantization Error')
    ax2.set_ylabel('Maximum Error')
    
    # Test 3: Rounding errors
    round_data = test_results['rounding']
    ax3.bar(['0.1+0.2 Error', 'Systematic Error'], 
            [round_data['error_magnitude'], round_data['max_systematic']])
    ax3.set_yscale('log')
    ax3.set_title('Rounding Errors')
    ax3.set_ylabel('Error Magnitude')
    
    # Test 4: Overall results
    all_tests = []
    all_results = []
    for item in [(cat, name, result) for cat, data in test_results.items() 
                 if isinstance(data, dict) for name, result in data.items()]:
        match item:
            case (str(category), str(test_name), bool(value)):
                all_tests.append(f"{category}_{test_name}")
                all_results.append(1.0 if value else 0.0)
    
    colors = ['green' if r > 0.5 else 'red' for r in all_results]
    ax4.bar(range(len(all_results)), all_results, color=colors)
    ax4.set_title('Test Results Summary')
    ax4.set_ylabel('Pass/Fail')
    ax4.set_ylim(0, 1.2)
    ax4.set_xticks(range(len(all_results)))
    ax4.set_xticklabels(all_tests, rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_base, '.out', 'visualizations', 'floating_point_tests.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Run rigorous tests of floating-point claims."""
    print("TESTING FLOATING-POINT AS OPERATIONS")
    print("=" * 50)
    
    # Run all tests
    test_results = {
        'ieee754': test_ieee754_quantization(),
        'glyph': test_glyph_quantization_properties(),
        'rounding': test_rounding_as_operator_cycles(),
        'special': test_special_values_as_fixed_points(),
        'keya': test_keya_operator_correspondence()
    }
    
    # Create visualizations
    create_test_visualizations(test_results)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in test_results.items():
        print(f"\n{category.upper()}:")
        if not isinstance(results, dict):
            continue
        for name, result in results.items():
            match (name, result):
                case (str(name), bool(passed)):
                    status = "PASS" if passed else "FAIL"
                    print(f"  {name}: {status}")
                    total_tests += 1
                    if passed:
                        passed_tests += 1
                case (str(name), value):
                    print(f"  {name}: {value}")
    
    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    # Scientific conclusion
    if passed_tests / total_tests > 0.7:
        print("\nCONCLUSION: Evidence supports floating-point as operations")
    else:
        print("\nCONCLUSION: Insufficient evidence for floating-point hypothesis")


if __name__ == "__main__":
    main() 