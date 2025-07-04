#!/usr/bin/env python3
"""
🌌 KEYA QUANTUM PHENOMENA RENDERER 🌌

This demonstrates how keya operators naturally emerge in quantum mechanics,
showcasing the deep mathematical connection between diagonalization-containment
and quantum state evolution.
"""

import sys
import os
import matplotlib.pyplot as plt

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
    success = wave.apply_wild_tame_evolution(3)
    if success:
        new_stats = wave.get_quantum_stats()
        print("   ✅ evolution successful!")
        print(f"   📊 New probability: {new_stats['total_probability']:.3f}")
        print(f"   ⏰ New time: {new_stats['time']:.3f}")
    else:
        print("   ❌ evolution failed")
    
    # Test 4: Test orbital evolution
    print("\n4. Testing orbital evolution...")
    orbital_success = orbital.evolve_with_wild_tame_operators(2)
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
        evolution_success = orbital.evolve_with_wild_tame_operators(1)
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
        success = wave.apply_wild_tame_evolution(1)
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
        result = WT(psi, general, 100)  
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


def create_visualization():
    """Create a non-interactive visualization of quantum phenomena."""
    print("🎨 CREATING VISUALIZATION: Quantum Phenomena")
    print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    fig.suptitle("Keya Quantum Phenomena", fontsize=16)

    # 1. 1s Orbital
    orb_1s = ElectronOrbital(OrbitalType.S_1S)
    orb_1s.plot_orbital(axes[0, 0], title="1s Hydrogen Orbital")

    # 2. 2pz Orbital
    orb_2pz = ElectronOrbital(OrbitalType.P_2PZ)
    orb_2pz.plot_orbital(axes[0, 1], title="2pz Hydrogen Orbital")

    # 3. Gaussian Wave Packet
    wave = QuantumWaveFunction(WaveFunctionType.GAUSSIAN, dimensions=(25, 25, 25))
    wave.plot_wave_function(axes[1, 0], title="Gaussian Wave Packet")
    
    # 4. Evolved Wave Packet
    wave.apply_wild_tame_evolution(5)
    wave.plot_wave_function(axes[1, 1], title="Evolved Wave Packet")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # Save to file
    output_path = os.path.join(".out", "visualizations", "quantum_phenomena.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Visualization saved to {output_path}")


def main():
    """Main quantum phenomena demo."""
    print("🌌 KEYA QUANTUM PHENOMENA RENDERER 🌌")
    print("=" * 50)
    print("Mathematical operators meet quantum mechanics!")
    print()

    # Run all console demos sequentially
    test_quantum_basics()
    print("\n" + "=" * 50)
    demo_hydrogen_orbitals()
    print("\n" + "=" * 50)
    demo_quantum_superposition()
    print("\n" + "=" * 50)
    demo_wave_collapse()
    print("\n" + "=" * 50)
    demo_infinite_quantum_evolution()

    # Create the non-interactive visualization
    create_visualization()

    print("\n🎉 ALL QUANTUM DEMOS COMPLETE! 🎉")


if __name__ == "__main__":
    main() 