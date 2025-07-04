#!/usr/bin/env python3
"""
🌌 KEYA QUANTUM PHENOMENA RENDERER 🌌

This demonstrates how keya operators naturally emerge in quantum mechanics,
showcasing the deep mathematical connection between diagonalization-containment
and quantum state evolution.
"""

import sys
import os

from keya.quantum.renderer import create_quantum_demo
from keya.quantum.orbital import ElectronOrbital, OrbitalType
from keya.quantum.wavefunction import QuantumWaveFunction, WaveFunctionType
from keya.dsl.ast import ContainmentType


def test_quantum_basics():
    """Test basic quantum functionality without GUI."""
    print("🧪 TESTING QUANTUM BASICS")
    print("=" * 40)
    
    # Test 1: Create electron orbital
    print("1. Creating 1s hydrogen orbital...")
    orbital = ElectronOrbital(OrbitalType.S_1S, grid_size=30)
    info = orbital.get_orbital_info()
    print(f"   ✅ Created: {info['orbital_type']}")
    print(f"   📊 Energy: {info['energy_level']:.2f} eV")
    print(f"   📏 Most probable radius: {info['most_probable_radius']:.2f} Bohr")
    
    # Test 2: Create wave function
    print("\n2. Creating Gaussian wave packet...")
    wave = QuantumWaveFunction(WaveFunctionType.GAUSSIAN, (20, 20, 20))
    stats = wave.get_quantum_stats()
    print(f"   ✅ Created: {stats['wave_type']}")
    print(f"   📊 Total probability: {stats['total_probability']:.3f}")
    print(f"   ⏰ Time: {stats['time']:.3f}")
    
    # Test 3: Apply evolution
    print("\n3. Testing quantum evolution...")
    success = wave.apply_dc_evolution(3)
    if success:
        new_stats = wave.get_quantum_stats()
        print("   ✅ evolution successful!")
        print(f"   📊 New probability: {new_stats['total_probability']:.3f}")
        print(f"   ⏰ New time: {new_stats['time']:.3f}")
    else:
        print("   ❌ evolution failed")
    
    # Test 4: Test orbital evolution
    print("\n4. Testing orbital evolution...")
    orbital_success = orbital.evolve_with_dc_operators(2)
    if orbital_success:
        print("   ✅ Orbital evolution successful!")
    else:
        print("   ❌ Orbital evolution failed")
    
    print("\n🎉 QUANTUM BASICS TEST COMPLETE! 🎉")
    return True


def demo_hydrogen_orbitals():
    """Demo various hydrogen orbitals."""
    print("🔬 HYDROGEN ORBITALS DEMO")
    print("=" * 30)
    
    orbitals_to_test = [
        (OrbitalType.S_1S, "1s ground state"),
        (OrbitalType.S_2S, "2s excited state"),
        (OrbitalType.P_2PZ, "2pz orbital"),
        (OrbitalType.D_3DZ2, "3dz² orbital")
    ]
    
    for orbital_type, description in orbitals_to_test:
        print(f"\n🌀 Creating {description}...")
        orbital = ElectronOrbital(orbital_type, grid_size=25)
        
        info = orbital.get_orbital_info()
        print(f"   📊 Quantum numbers: n={info['quantum_numbers']['n']}, "
              f"l={info['quantum_numbers']['l']}, m={info['quantum_numbers']['m']}")
        print(f"   ⚡ Energy level: {info['energy_level']:.2f} eV")
        print(f"   📏 Most probable radius: {info['most_probable_radius']:.2f} Bohr")
        
        # Get 2D slice for visualization
        prob_2d = orbital.get_probability_density_2d("xy")
        max_prob = prob_2d.max()
        print(f"   🎯 Max probability density: {max_prob:.4f}")
        
        # Test evolution
        print("   🌀 Testing evolution...")
        evolution_success = orbital.evolve_with_dc_operators(1)
        if evolution_success:
            print("   ✅ evolution successful!")
        else:
            print("   ❌ evolution failed")
    
    print("\n✨ HYDROGEN ORBITALS DEMO COMPLETE! ✨")


def demo_quantum_superposition():
    """Demo quantum superposition using operators."""
    print("🌈 QUANTUM SUPERPOSITION DEMO")
    print("=" * 32)
    
    print("Creating quantum superposition state...")
    wave = QuantumWaveFunction(WaveFunctionType.SUPERPOSITION, (25, 25, 25))
    
    initial_stats = wave.get_quantum_stats()
    print(f"✅ Initial state: {initial_stats['wave_type']}")
    print(f"📊 Total probability: {initial_stats['total_probability']:.3f}")
    
    # Get expectation values
    pos_exp = wave.measure_expectation("position")
    print(f"📍 Position expectation: <x>={pos_exp[0]:.2f}, <y>={pos_exp[1]:.2f}, <z>={pos_exp[2]:.2f}")
    
    # Evolve the superposition
    print("\n🌀 Evolving superposition with operators...")
    for step in range(5):
        success = wave.apply_dc_evolution(1)
        if success:
            stats = wave.get_quantum_stats()
            pos_exp = wave.measure_expectation("position")
            print(f"   Step {step+1}: Prob={stats['total_probability']:.3f}, "
                  f"<x>={pos_exp[0]:.2f}")
        else:
            print(f"   Step {step+1}: Evolution failed")
    
    print("\n🌊 SUPERPOSITION DEMO COMPLETE! 🌊")


def demo_wave_collapse():
    """Demo wave function collapse simulation."""
    print("⚡ WAVE FUNCTION COLLAPSE DEMO")
    print("=" * 33)
    
    print("Creating Gaussian wave packet...")
    wave = QuantumWaveFunction(WaveFunctionType.GAUSSIAN, (20, 20, 20))
    
    print("Initial state:")
    initial_stats = wave.get_quantum_stats()
    print(f"   📊 Total probability: {initial_stats['total_probability']:.3f}")
    pos_exp = wave.measure_expectation("position")
    print(f"   📍 Position spread: <x>={pos_exp[0]:.2f}")
    
    # Simulate measurement/collapse
    print("\n⚡ Simulating quantum measurement...")
    measurement_pos = (10, 10, 10)  # Middle of grid
    wave.collapse_wave_function(measurement_pos)
    
    print("After collapse:")
    final_stats = wave.get_quantum_stats()
    print(f"   📊 Total probability: {final_stats['total_probability']:.3f}")
    pos_exp_final = wave.measure_expectation("position")
    print(f"   📍 New position: <x>={pos_exp_final[0]:.2f}")
    
    print("\n💥 WAVE COLLAPSE DEMO COMPLETE! 💥")


def demo_infinite_quantum_evolution():
    """Demo infinite quantum evolution using ∞ cycles."""
    print("♾️  INFINITE QUANTUM EVOLUTION DEMO")
    print("=" * 38)
    
    print("Creating wave function for infinite evolution...")
    wave = QuantumWaveFunction(WaveFunctionType.HARMONIC, (15, 15, 15), 
                              ContainmentType.GENERAL)
    
    print("Testing infinite evolution...")
    
    # Test with large number to simulate infinity
    keya_program = """
matrix infinite_quantum {
    evolution {
        result = DC(psi, general, 100)  
    }
}
"""
    
    # Set up for infinite evolution
    wave.engine.variables['psi'] = wave.get_probability_density_2d()
    
    try:
        result = wave.engine.execute_program(keya_program.strip())
        if result:
            print("✅ Infinite quantum evolution simulation successful!")
            print("   📊 Large-scale cycles completed")
            print("   🌀 Quantum system evolved through extended time")
        else:
            print("❌ Infinite evolution simulation failed")
    except Exception as e:
        print(f"❌ Error in infinite evolution: {e}")
    
    print("\n∞ INFINITE EVOLUTION DEMO COMPLETE! ∞")


def create_visual_demo(demo_type: str):
    """Create visual quantum demo (requires GUI)."""
    print(f"🎨 CREATING VISUAL DEMO: {demo_type.upper()}")
    print("=" * 50)
    
    try:
        if demo_type == "orbital":
            print("🌀 Loading 2pz orbital visualization...")
            renderer = create_quantum_demo("orbital")
            print("✅ 3D orbital isosurface created!")
            print("   Use mouse to rotate, scroll to zoom")
            renderer.show()
            
        elif demo_type == "evolution":
            print("🌊 Loading wave function evolution...")
            renderer = create_quantum_demo("evolution")
            print("✅ Real-time evolution animation created!")
            renderer.show()
            
        elif demo_type == "superposition":
            print("🌈 Loading quantum superposition...")
            renderer = create_quantum_demo("superposition")
            print("✅ Superposition visualization created!")
            renderer.show()
            
        elif demo_type == "dc_orbital":
            print("🌀 Loading orbital evolution...")
            renderer = create_quantum_demo("dc_orbital")
            print("✅ orbital animation created!")
            renderer.show()
            
        else:
            print("🎯 Loading 3dz² orbital slices...")
            renderer = create_quantum_demo("slices")
            print("✅ Multi-plane orbital visualization created!")
            renderer.show()
            
    except Exception as e:
        print(f"❌ Visual demo error: {e}")
        print("   Falling back to console mode...")


def main():
    """Main quantum phenomena demo."""
    print("🌌 KEYA QUANTUM PHENOMENA RENDERER 🌌")
    print("=" * 50)
    print("Mathematical operators meet quantum mechanics!")
    print()
    
    print("Select a demo:")
    print("1. Test Quantum Basics (console)")
    print("2. Hydrogen Orbitals Demo (console)")  
    print("3. Quantum Superposition (console)")
    print("4. Wave Collapse Simulation (console)")
    print("5. Infinite Evolution (∞ cycles)")
    print("6. 3D Orbital Visualization (GUI)")
    print("7. Wave Evolution Animation (GUI)")
    print("8. Superposition Visualization (GUI)")
    print("9. Orbital Evolution (GUI)")
    print("0. Run All Console Demos")
    
    try:
        choice = input("\nEnter choice (0-9): ").strip()
        
        if choice == '0':
            # Run all console demos
            test_quantum_basics()
            print("\n" + "="*50)
            demo_hydrogen_orbitals()
            print("\n" + "="*50)
            demo_quantum_superposition()
            print("\n" + "="*50)
            demo_wave_collapse()
            print("\n" + "="*50)
            demo_infinite_quantum_evolution()
            
        elif choice == '1':
            test_quantum_basics()
        elif choice == '2':
            demo_hydrogen_orbitals()
        elif choice == '3':
            demo_quantum_superposition()
        elif choice == '4':
            demo_wave_collapse()
        elif choice == '5':
            demo_infinite_quantum_evolution()
        elif choice == '6':
            create_visual_demo("orbital")
        elif choice == '7':
            create_visual_demo("evolution")
        elif choice == '8':
            create_visual_demo("superposition")
        elif choice == '9':
            create_visual_demo("dc_orbital")
        else:
            print("Invalid choice. Running quantum basics...")
            test_quantum_basics()
            
    except KeyboardInterrupt:
        print("\n\n👋 Quantum demo interrupted. Wave function collapsed!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Running basic quantum test...")
        test_quantum_basics()


if __name__ == "__main__":
    main() 