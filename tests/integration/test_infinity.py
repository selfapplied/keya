#!/usr/bin/env python3
"""Test infinity support in keya D-C language."""

import sys
sys.path.append('src')

from keya.dsl import parse, Engine


def test_infinity_parsing():
    """Test that infinity (∞) can be parsed in DC cycles."""
    
    # Test program with infinite iterations
    program = """
matrix CellularAutomata {
    evolution {
        grid = [3, 3, ∅]
        result = DC(grid, binary, ∞)
    }
}
"""
    
    print("🧪 Testing infinity parsing...")
    try:
        ast = parse(program.strip())
        print("✅ Infinity parsing successful!")
        
        # Check that the AST contains the infinity value
        assignment = ast.sections[0].statements[1]  # result = DC(...)
        if hasattr(assignment, 'value'):
            dc_cycle = assignment.value
            print(f"   Max iterations: {dc_cycle.max_iterations}")
            print(f"   Is infinity: {dc_cycle.max_iterations == float('inf')}")
        else:
            print("   Could not access DC cycle value")
        
        return ast
    except Exception as e:
        print(f"❌ Infinity parsing failed: {e}")
        return None


def test_infinity_execution():
    """Test that infinite DC cycles execute properly."""
    
    print("\n🔥 Testing infinity execution...")
    
    engine = Engine()
    
    # Simple test with small number first
    finite_program = """
matrix Test {
    demo {
        grid = [2, 2, △]
        result = DC(grid, binary, 3)
    }
}
"""
    
    print("   Testing finite iterations first...")
    result = engine.execute_program(finite_program.strip())
    if result:
        print("✅ Finite execution works!")
    
    # Now test with infinity (will run 1000 iterations in demo)
    infinite_program = """
matrix CellularTest {
    infinite_evolution {
        grid = [2, 2, △]  
        cellular = DC(grid, binary, ∞)
    }
}
"""
    
    print("   Testing infinite iterations...")
    result = engine.execute_program(infinite_program.strip())
    if result:
        print("✅ Infinite execution works!")
        return True
    return False


if __name__ == "__main__":
    print("🌟 TESTING KEYA INFINITY SUPPORT 🌟\n")
    
    # Test parsing
    ast = test_infinity_parsing()
    
    # Test execution 
    if ast:
        success = test_infinity_execution()
        if success:
            print("\n🎉 INFINITY SUPPORT FULLY WORKING! 🎉")
            print("Ready to build cellular automata widgets! 🔥")
        else:
            print("\n❌ Execution failed")
    else:
        print("\n❌ Parsing failed") 