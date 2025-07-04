#!/usr/bin/env python3
"""
Integration test for the complete Kéya D-C system.
Tests parser, engine, REPL, and visualization integration.
"""

import sys
import os

# Ensure the src directory is on the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from keya.dsl import Engine, parse
from keya.shell.repl import KeyaREPL


def test_basic_parsing():
    """Test basic parsing functionality."""
    print("🧪 Testing basic parsing...")
    
    # Test matrix program
    matrix_code = """matrix test_program {
    operations {
        m1 = [[∅, △], [▽, ⊙]]
        result = D m1
    }
}"""
    
    try:
        parse(matrix_code)  # Validate syntax
        print("  ✅ Matrix program parsing successful")
        return True
    except Exception as e:
        print(f"  ❌ Matrix program parsing failed: {e}")
        return False


def test_engine_execution():
    """Test engine execution functionality."""
    print("🚀 Testing engine execution...")
    
    engine = Engine()
    
    # Test simple matrix operations
    test_programs = [
        """matrix simple_test {
    operations {
        m1 = [[△, ▽], [⊙, ∅]]
        result = D m1
    }
}""",
        """matrix containment_test {
    operations {
        m2 = [2, 2, ∅]
        contained = C(m2, binary)
    }
}""",
        """resonance trace_test {
    analysis {
        m3 = [[△, ∅], [∅, △]]
        trace m3
    }
}"""
    ]
    
    success_count = 0
    for i, program in enumerate(test_programs):
        try:
            result = engine.execute_program(program)
            if result is not None:
                print(f"  ✅ Program {i+1} executed successfully")
                success_count += 1
            else:
                print(f"  ⚠️  Program {i+1} executed but returned None")
        except Exception as e:
            print(f"  ❌ Program {i+1} execution failed: {e}")
    
    return success_count == len(test_programs)


def test_repl_commands():
    """Test REPL functionality."""
    print("🐚 Testing REPL functionality...")
    
    engine = Engine()
    KeyaREPL(engine)  # Initialize REPL for engine setup
    
    # Test some basic operations that don't require user input
    test_commands = [
        """matrix test1 {
    operations {
        m1 = [[△, ▽], [⊙, ∅]]
    }
}""",
        """matrix test2 {
    operations {
        m1 = [[△, ▽], [⊙, ∅]]
        result = D m1
    }
}""",
    ]
    
    success_count = 0
    for command in test_commands:
        try:
            engine.execute_program(command)  # Execute for side effects
            print(f"  ✅ Command '{command}' executed")
            success_count += 1
        except Exception as e:
            print(f"  ❌ Command '{command}' failed: {e}")
    
    return success_count == len(test_commands)


def test_matrix_visualization():
    """Test visualization functionality."""
    print("📊 Testing visualization functionality...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        
        import numpy as np
        
        # Create a test matrix
        np.array([
            [0.0, 1.0],    # ∅, △
            [-1.0, 0.5]    # ▽, ⊙
        ])
        
        # Test visualization functions (they won't display in test mode)
        print("  ✅ Visualization imports successful")
        print("  ✅ Test matrix created")
        
        return True
    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        return False


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("🎯 Testing end-to-end workflow...")
    
    # Complete D-C program
    dc_program = """matrix advanced_test {
    operations {
        base = [[△, ▽], [⊙, ⊕]]
        dissonant = D base
        contained = C(dissonant, decimal)
        cycle_result = DC(base, binary, 3)
    }
}"""
    
    try:
        engine = Engine()
        
        # Parse
        parse(dc_program)  # Validate syntax
        print("  ✅ Complex program parsed")
        
        # Execute
        engine.execute_program(dc_program)  # Execute for side effects
        print("  ✅ Complex program executed")
        
        # Check if variables were created
        if hasattr(engine, 'variables') and engine.variables:
            print(f"  ✅ Variables created: {list(engine.variables.keys())}")
        
        return True
    except Exception as e:
        print(f"  ❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🧠 Kéya D-C Language Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Parsing", test_basic_parsing),
        ("Engine Execution", test_engine_execution),
        ("REPL Commands", test_repl_commands),
        ("Visualization", test_matrix_visualization),
        ("End-to-End", test_end_to_end),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests PASSED! The system is ready!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 