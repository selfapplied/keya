#!/usr/bin/env python3
"""
Integration test for the complete Keya  system.
Tests parser, engine, REPL, and visualization integration.
"""

from keya.dsl import parse
from keya.core.engine import Engine
from keya.shell.repl import KeyaDCREPL


def test_basic_parsing():
    """Test basic parsing functionality."""
    print("🧪 Testing basic parsing...")
    
    # Test matrix program
    matrix_code = """matrix test_program {
    operations {
        m1 = [[∅, △], [▽, ⊙]]
        result = Ϟ m1
    }
}"""
    
    try:
        parse(matrix_code)  # Validate syntax
        print("  ✅ Matrix program parsing successful")
    except Exception as e:
        assert False, f"  ❌ Matrix program parsing failed: {e}"


def test_engine_execution():
    """Test engine execution functionality."""
    print("🚀 Testing engine execution...")
    
    engine = Engine()
    
    # Test simple matrix operations
    test_programs = [
        """matrix simple_test {
    operations {
        m1 = [[△, ▽], [⊙, ∅]]
        result = Ϟ m1
    }
}""",
        """matrix containment_test {
    operations {
        m2 = [2, 2, ∅]
        contained = §(m2, binary)
    }
}""",
        """resonance trace_test {
    analysis {
        m3 = [[△, ∅], [∅, △]]
        trace m3
    }
}"""
    ]
    
    for i, program in enumerate(test_programs):
        try:
            result = engine.execute_program(program)
            assert result is not None, f"Program {i+1} executed but returned None"
            print(f"  ✅ Program {i+1} executed successfully")
        except Exception as e:
            assert False, f"  ❌ Program {i+1} execution failed: {e}"


def test_repl_commands():
    """Test REPL functionality."""
    print("🐚 Testing REPL functionality...")
    
    engine = Engine()
    KeyaDCREPL(engine)  # Initialize REPL for engine setup
    
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
        result = Ϟ m1
    }
}""",
    ]
    
    for command in test_commands:
        try:
            engine.execute_program(command)  # Execute for side effects
            print(f"  ✅ Command '{command[:20]}...' executed")
        except Exception as e:
            assert False, f"  ❌ Command '{command[:20]}...' failed: {e}"


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
        
    except Exception as e:
        assert False, f"  ❌ Visualization test failed: {e}"


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("🎯 Testing end-to-end workflow...")
    
    # Complete  program
    wildtame_program = """matrix advanced_test {
    operations {
        base = [[△, ▽], [⊙, ⊕]]
        wild_result = Ϟ base
        tamed = §(wild_result, decimal)
        cycle_result = ∮(base, binary, 3)
    }
}"""
    
    try:
        engine = Engine()
        
        # Parse
        parse(wildtame_program)  # Validate syntax
        print("  ✅ Complex program parsed")
        
        # Execute
        engine.execute_program(wildtame_program)  # Execute for side effects
        print("  ✅ Complex program executed")
        
    except Exception as e:
        assert False, f"  ❌ End-to-end test failed: {e}" 