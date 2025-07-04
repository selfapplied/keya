#!/usr/bin/env python3
"""Debug infinity parsing."""

import sys
import os

from keya.dsl import parse


def test_simple_infinity():
    """Test the simplest possible infinity case."""
    
    program = """
matrix test {
    basic {
        result = DC([2, 2], binary, ∞)
    }
}
"""
    
    print("🔍 Testing simple infinity...")
    try:
        parse(program.strip())  # Validate syntax
        print("✅ Simple parsing works!")
        return True
    except Exception as e:
        print(f"❌ Simple parsing failed: {e}")
        return False


def test_regular_number():
    """Test that regular numbers still work."""
    
    program = """
matrix test {
    basic {
        result = DC([2, 2], binary, 5)
    }
}
"""
    
    print("🔍 Testing regular number...")
    try:
        parse(program.strip())  # Validate syntax
        print("✅ Regular number works!")
        return True
    except Exception as e:
        print(f"❌ Regular number failed: {e}")
        return False


if __name__ == "__main__":
    print("🐛 DEBUGGING INFINITY PARSING 🐛\n")
    
    # Test regular numbers first
    if test_regular_number():
        # Then test infinity
        test_simple_infinity()
    else:
        print("❌ Basic parsing is broken") 