#!/usr/bin/env python3
"""
Demo Runner for Keya D-C Language

Simple script to run demos with proper output organization.
All outputs are saved to .out/ directory structure.
Run from demos/ directory: python run_demo.py <demo_name>
"""

import sys
import os
import subprocess
from datetime import datetime

def ensure_output_dirs():
    """Create output directory structure."""
    os.makedirs('../.out/tests', exist_ok=True)
    os.makedirs('../.out/visualizations', exist_ok=True)

def run_demo(demo_name: str, save_output: bool = True):
    """Run a demo and optionally save output."""
    demo_path = demo_name
    
    if not os.path.exists(demo_path):
        print(f"❌ Demo not found: {demo_path}")
        return False
    
    print(f"🚀 Running {demo_name}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if save_output:
            output_file = f"../.out/tests/{demo_name.replace('.py', '')}_{timestamp}.out"
            with open(output_file, 'w') as f:
                result = subprocess.run([sys.executable, demo_path], 
                                      stdout=f, stderr=subprocess.STDOUT, 
                                      cwd='.')
            print(f"✅ Complete. Output saved to: {output_file}")
        else:
            result = subprocess.run([sys.executable, demo_path], cwd='.')
            print("✅ Complete.")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running {demo_name}: {e}")
        return False

def list_demos():
    """List available demos."""
    demos_dir = "."
    demos = [f for f in os.listdir(demos_dir) 
             if f.endswith('.py') and not f.startswith('README') and f != 'run-demo.py']
    return sorted(demos)

def main():
    """Main entry point."""
    ensure_output_dirs()
    
    if len(sys.argv) < 2:
        print("Keya Demo Runner")
        print("================")
        print("\nUsage:")
        print("  python run_demo.py <demo_name>")
        print("  python run_demo.py list")
        print("  python run_demo.py all")
        print("\nAvailable demos:")
        
        demos = list_demos()
        for demo in demos:
            print(f"  - {demo}")
        
        print("\nOutput saved to:")
        print("  - Test logs: ../.out/tests/")
        print("  - Visualizations: ../.out/visualizations/")
        
        return
    
    command = sys.argv[1]
    
    if command == "list":
        demos = list_demos()
        print("Available demos:")
        for demo in demos:
            print(f"  - {demo}")
    
    elif command == "all":
        print("Running all demos...")
        demos = list_demos()
        success_count = 0
        
        for demo in demos:
            if run_demo(demo):
                success_count += 1
        
        print(f"\n📊 Results: {success_count}/{len(demos)} demos completed successfully")
    
    elif command.endswith('.py'):
        run_demo(command)
    
    else:
        # Try adding .py extension
        demo_name = f"demo_{command}.py"
        if not run_demo(demo_name):
            # Try exact name
            demo_name = f"{command}.py"
            if not run_demo(demo_name):
                print(f"❌ Demo not found: {command}")
                print("Use 'python run_demo.py list' to see available demos")

if __name__ == "__main__":
    main() 