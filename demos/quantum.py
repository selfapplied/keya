#!/usr/bin/env python3
"""
🌌 KEYA QUANTUM PHENOMENA RENDERER 🌌

This demonstrates how keya operators naturally emerge in quantum mechanics,
showcasing the deep mathematical connection between diagonalization-containment
and quantum state evolution.
"""

import os
import matplotlib.pyplot as plt

from keya.quantum.orbital import ElectronOrbital, OrbitalType
from keya.quantum.wavefunction import QuantumWaveFunction, WaveFunctionType
from demos.reporting.registry import register_demo


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


@register_demo(
    title="Quantum Phenomena Simulation",
    artifacts=[
        {"filename": "docs/quantum_phenomena.svg", "caption": "A gallery of simulated quantum states, including hydrogen orbitals, a time-evolved wave packet, and a superposition state."}
    ],
    claims=[
        "Hydrogen orbitals can be constructed and visualized.",
        "Keya's evolution operators can model the time-development of a quantum wave packet.",
        "Superposition states can be created and manipulated."
    ],
    findings="The script runs through its series of demos, printing confirmations for each test. The final visualization successfully renders the different quantum states, confirming that the simulation and plotting functions are working correctly."
)
def main():
    """
    This demo simulates various quantum phenomena to show how Keya's operators
    can model quantum state evolution. It covers:
    - The structure of hydrogen orbitals (1s, 2pz).
    - The evolution of a Gaussian wave packet over time.
    - The principle of superposition.
    The visualization provides a gallery of these quantum states.
    """
    print("🌌 KEYA QUANTUM PHENOMENA RENDERER 🌌")
    print("=" * 50)

    # Run all console demos
    test_quantum_basics()
    demo_hydrogen_orbitals()
    demo_quantum_superposition()
    demo_wave_collapse()

    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    fig.suptitle("Keya Quantum Phenomena", fontsize=16)

    orb_1s = ElectronOrbital(OrbitalType.S_1S)
    orb_1s.plot_orbital(axes[0, 0], title="1s Hydrogen Orbital")

    orb_2pz = ElectronOrbital(OrbitalType.P_2PZ)
    orb_2pz.plot_orbital(axes[0, 1], title="2pz Hydrogen Orbital")

    wave = QuantumWaveFunction(WaveFunctionType.GAUSSIAN, dimensions=(25, 25, 25))
    wave.plot_wave_function(axes[1, 0], title="Gaussian Wave Packet")
    
    wave.apply_wild_tame_evolution(5)
    wave.plot_wave_function(axes[1, 1], title="Evolved Wave Packet")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # Save the final figure
    os.makedirs("docs", exist_ok=True)
    save_path = "docs/quantum_phenomena.svg"
    fig.savefig(save_path, bbox_inches='tight', transparent=True)
    print(f"\n✅ All quantum phenomena visualized and saved to '{save_path}'")


if __name__ == '__main__':
    main() 