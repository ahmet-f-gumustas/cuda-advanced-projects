#!/usr/bin/env python3
"""
Simple demo script for N-Body Python bindings
Quick demonstration of the main features
"""

import sys
sys.path.insert(0, '../build')

try:
    import nbody_cuda
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("\nMake sure you have built the Python bindings:")
    print("  cd build")
    print("  cmake .. -DBUILD_PYTHON_BINDINGS=ON")
    print("  make nbody_python")
    sys.exit(1)

def main():
    print("\n" + "="*60)
    print("  GPU N-Body Simulation - Python Demo")
    print("="*60 + "\n")

    # Create simulation
    print("Creating simulation with 2048 bodies...")
    sim = nbody_cuda.NBodySimulation(2048, dt=0.01, softening=0.1)
    print(f"✓ Simulation created")

    # Initialize with galaxy collision
    print("\nInitializing galaxy collision...")
    sim.initialize_galaxy_collision()

    # Get initial statistics
    stats_initial = sim.get_statistics()
    print(f"✓ Initialized")
    print(f"  Total mass: {stats_initial['total_mass']:.2f}")
    print(f"  Initial energy: {stats_initial['total_energy']:.2f}")

    # Run simulation
    print("\nRunning 200 iterations (tiled kernel)...")
    sim.simulate(200, kernel='tiled')
    print(f"✓ Simulation complete")

    # Get final statistics
    stats_final = sim.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Final energy: {stats_final['total_energy']:.2f}")
    print(f"  Energy drift: {abs(stats_final['total_energy'] - stats_initial['total_energy']) / abs(stats_initial['total_energy']) * 100:.2f}%")
    print(f"  Center of mass: ({stats_final['center_of_mass'][0]:.3f}, "
          f"{stats_final['center_of_mass'][1]:.3f}, {stats_final['center_of_mass'][2]:.3f})")

    # Benchmark
    print("\nRunning performance benchmark (100 iterations each)...")
    results = sim.benchmark(iterations=100)

    print(f"\nPerformance Results:")
    print(f"  Naive kernel:          {results['naive_ms']:.3f} ms  ({results['naive_fps']:.0f} FPS)")
    print(f"  Shared memory kernel:  {results['shared_ms']:.3f} ms  ({results['shared_fps']:.0f} FPS)  [{results['speedup_shared']:.2f}x]")
    print(f"  Register-tiled kernel: {results['tiled_ms']:.3f} ms  ({results['tiled_fps']:.0f} FPS)  [{results['speedup_tiled']:.2f}x]")

    # Get some position data
    positions = sim.get_positions()
    print(f"\nPosition data shape: {positions.shape}")
    print(f"Position range: [{positions.min():.2f}, {positions.max():.2f}]")

    print("\n" + "="*60)
    print("  Demo completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
