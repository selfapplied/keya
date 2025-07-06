#!/usr/bin/env python3
"""Test infinity support in keya  language."""


from keya.dsl import parse, MatrixAssignment, WildTameCycle
from keya.core.engine import Engine
from typing import cast


def test_infinity_parsing():
    """Test that infinity (∞) can be parsed in  cycles."""
    
    # Test program with infinite iterations
    program = """
matrix CellularAutomata {
    evolution {
        grid = [3, 3, ∅]
        result = ∮(grid, binary, ∞)
    }
}
"""
    
    print("🧪 Testing infinity parsing...")
    try:
        ast = parse(program.strip())
        print("✅ Infinity parsing successful!")
        
        # Check that the AST contains the infinity value
        assignment = ast.sections[0].statements[1]  # result = ∮(...)
        assert isinstance(assignment, MatrixAssignment)
        assert isinstance(assignment.value, WildTameCycle)
        wildtame_cycle = cast(WildTameCycle, assignment.value)
        assert wildtame_cycle.max_iterations == float('inf')
        print(f"   Max iterations: {wildtame_cycle.max_iterations}")
        print(f"   Is infinity: {wildtame_cycle.max_iterations == float('inf')}")
        
    except Exception as e:
        assert False, f"❌ Infinity parsing failed: {e}"


def test_infinity_execution():
    """Test that infinite  cycles execute properly."""
    
    print("\n🔥 Testing infinity execution...")
    
    engine = Engine()
    
    # Simple test with small number first
    finite_program = """
matrix Test {
    demo {
        grid = [2, 2, △]
        result = ∮(grid, binary, 3)
    }
}
"""
    
    print("   Testing finite iterations first...")
    result = engine.execute_program(finite_program.strip())
    assert result, "Finite execution failed"
    print("✅ Finite execution works!")
    
    # Now test with infinity (will run 1000 iterations in demo)
    infinite_program = """
matrix CellularTest {
    infinite_evolution {
        grid = [2, 2, △]  
        cellular = ∮(grid, binary, ∞)
    }
}
"""
    
    print("   Testing infinite iterations...")
    result = engine.execute_program(infinite_program.strip())
    assert result, "Infinite execution failed"
    print("✅ Infinite execution works!") 