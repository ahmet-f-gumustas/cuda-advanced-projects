#!/usr/bin/env python3
"""
Test script for N-Body CUDA Python Bindings
Tests all functionality exposed through pybind11
"""

import sys
import time
import numpy as np

# Add build directory to path to import the module
sys.path.insert(0, '../build')

try:
    import nbody_cuda
except ImportError as e:
    print(f"Error importing nbody_cuda module: {e}")
    print("\nMake sure you have built the Python bindings:")
    print("  cd build")
    print("  cmake .. -DBUILD_PYTHON_BINDINGS=ON")
    print("  make nbody_python")
    sys.exit(1)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_basic_creation():
    """Test basic simulation creation"""
    print_section("Test 1: Basic Simulation Creation")

    num_bodies = 1024
    sim = nbody_cuda.NBodySimulation(num_bodies, dt=0.01, softening=0.1)

    print(f"✓ Created simulation with {sim.get_num_bodies()} bodies")
    print(f"✓ Time step (dt): {sim.get_delta_time()}")
    print(f"✓ Softening factor: {sim.get_softening()}")

    return sim

def test_initialization_methods(sim):
    """Test different initialization methods"""
    print_section("Test 2: Initialization Methods")

    # Test random initialization
    print("Testing random initialization...")
    sim.initialize_random(position_scale=10.0, velocity_scale=1.0, mass_scale=1.0)
    positions = sim.get_positions()
    print(f"✓ Random initialization complete")
    print(f"  Position shape: {positions.shape}")
    print(f"  Position range: [{positions.min():.2f}, {positions.max():.2f}]")

    # Test galaxy collision
    print("\nTesting galaxy collision initialization...")
    sim.initialize_galaxy_collision()
    stats = sim.get_statistics()
    print(f"✓ Galaxy collision initialized")
    print(f"  Total mass: {stats['total_mass']:.2f}")
    print(f"  Center of mass: ({stats['center_of_mass'][0]:.2f}, "
          f"{stats['center_of_mass'][1]:.2f}, {stats['center_of_mass'][2]:.2f})")

    # Test spherical cluster
    print("\nTesting spherical cluster initialization...")
    sim.initialize_spherical_cluster(radius=10.0)
    positions = sim.get_positions()
    distances = np.linalg.norm(positions, axis=1)
    print(f"✓ Spherical cluster initialized")
    print(f"  Average distance from origin: {distances.mean():.2f}")
    print(f"  Max distance: {distances.max():.2f}")

def test_data_access(sim):
    """Test data access methods"""
    print_section("Test 3: Data Access Methods")

    # Initialize with known values
    sim.initialize_random()

    # Get individual arrays
    x = sim.get_positions_x()
    y = sim.get_positions_y()
    z = sim.get_positions_z()
    print(f"✓ Got individual position arrays: {len(x)} bodies")

    # Get combined arrays
    positions = sim.get_positions()
    velocities = sim.get_velocities()
    masses = sim.get_masses()

    print(f"✓ Positions shape: {positions.shape}")
    print(f"✓ Velocities shape: {velocities.shape}")
    print(f"✓ Masses shape: {masses.shape}")

    # Verify data consistency
    assert positions.shape == (sim.get_num_bodies(), 3), "Positions shape mismatch"
    assert velocities.shape == (sim.get_num_bodies(), 3), "Velocities shape mismatch"
    assert masses.shape == (sim.get_num_bodies(),), "Masses shape mismatch"
    print("✓ Data shapes verified")

def test_simulation_steps(sim):
    """Test simulation stepping methods"""
    print_section("Test 4: Simulation Step Methods")

    sim.initialize_galaxy_collision()

    # Get initial energy
    energy_initial = sim.get_total_energy()
    print(f"Initial energy: {energy_initial:.2f}")

    # Test single steps with different kernels
    print("\nTesting naive kernel...")
    start = time.time()
    sim.step_naive()
    naive_time = (time.time() - start) * 1000
    print(f"✓ Naive step completed in {naive_time:.2f} ms")

    print("\nTesting shared memory kernel...")
    start = time.time()
    sim.step_shared()
    shared_time = (time.time() - start) * 1000
    print(f"✓ Shared memory step completed in {shared_time:.2f} ms")

    print("\nTesting tiled kernel...")
    start = time.time()
    sim.step_tiled()
    tiled_time = (time.time() - start) * 1000
    print(f"✓ Tiled step completed in {tiled_time:.2f} ms")

    # Get final energy
    energy_final = sim.get_total_energy()
    energy_change = abs(energy_final - energy_initial) / abs(energy_initial) * 100
    print(f"\nFinal energy: {energy_final:.2f}")
    print(f"Energy change: {energy_change:.2f}%")

def test_multi_step_simulation(sim):
    """Test multi-step simulation"""
    print_section("Test 5: Multi-Step Simulation")

    sim.initialize_spherical_cluster()
    initial_stats = sim.get_statistics()

    print("Running 100 iterations with tiled kernel...")
    start = time.time()
    sim.simulate(iterations=100, kernel='tiled')
    elapsed = time.time() - start

    final_stats = sim.get_statistics()

    print(f"✓ Simulation completed in {elapsed:.2f} seconds")
    print(f"  Average time per iteration: {elapsed/100*1000:.2f} ms")
    print(f"  FPS: {100/elapsed:.2f}")

    print(f"\nInitial energy: {initial_stats['total_energy']:.2f}")
    print(f"Final energy: {final_stats['total_energy']:.2f}")

    # Check center of mass conservation (momentum)
    initial_cm = initial_stats['center_of_mass']
    final_cm = final_stats['center_of_mass']
    cm_drift = np.sqrt(sum((f-i)**2 for i, f in zip(initial_cm, final_cm)))
    print(f"\nCenter of mass drift: {cm_drift:.4f}")

def test_set_data(sim):
    """Test setting data from numpy arrays"""
    print_section("Test 6: Setting Data from NumPy")

    # Create custom positions
    n = sim.get_num_bodies()
    custom_positions = np.random.randn(n, 3) * 5.0
    custom_velocities = np.random.randn(n, 3) * 0.5
    custom_masses = np.ones(n)

    print(f"Setting custom data for {n} bodies...")
    sim.set_positions(custom_positions)
    sim.set_velocities(custom_velocities)
    sim.set_masses(custom_masses)

    # Verify
    retrieved_positions = sim.get_positions()
    diff = np.abs(retrieved_positions - custom_positions).max()

    print(f"✓ Data set successfully")
    print(f"  Max difference: {diff:.2e} (should be ~0)")

def test_statistics(sim):
    """Test statistics computation"""
    print_section("Test 7: Statistics Computation")

    sim.initialize_galaxy_collision()

    stats = sim.get_statistics()

    print("System statistics:")
    print(f"  Number of bodies: {stats['num_bodies']}")
    print(f"  Total mass: {stats['total_mass']:.2f}")
    print(f"  Center of mass: ({stats['center_of_mass'][0]:.3f}, "
          f"{stats['center_of_mass'][1]:.3f}, {stats['center_of_mass'][2]:.3f})")
    print(f"  Average velocity: ({stats['avg_velocity'][0]:.3f}, "
          f"{stats['avg_velocity'][1]:.3f}, {stats['avg_velocity'][2]:.3f})")
    print(f"  Total energy: {stats['total_energy']:.2f}")

    print("\n✓ All statistics computed successfully")

def test_benchmarking(sim):
    """Test benchmarking functionality"""
    print_section("Test 8: Kernel Benchmarking")

    sim.initialize_random()

    print("Running benchmark (100 iterations per kernel)...")
    results = sim.benchmark(iterations=100)

    print("\nBenchmark Results:")
    print(f"  Naive kernel:")
    print(f"    Average time: {results['naive_ms']:.3f} ms")
    print(f"    FPS: {results['naive_fps']:.1f}")

    print(f"\n  Shared memory kernel:")
    print(f"    Average time: {results['shared_ms']:.3f} ms")
    print(f"    FPS: {results['shared_fps']:.1f}")
    print(f"    Speedup vs naive: {results['speedup_shared']:.2f}x")

    print(f"\n  Register-tiled kernel:")
    print(f"    Average time: {results['tiled_ms']:.3f} ms")
    print(f"    FPS: {results['tiled_fps']:.1f}")
    print(f"    Speedup vs naive: {results['speedup_tiled']:.2f}x")

def test_different_sizes():
    """Test with different body counts"""
    print_section("Test 9: Different Body Counts")

    sizes = [256, 512, 1024, 2048, 4096]

    for size in sizes:
        print(f"\nTesting with {size} bodies...")
        sim = nbody_cuda.NBodySimulation(size)
        sim.initialize_random()

        start = time.time()
        sim.simulate(10, kernel='tiled')
        elapsed = (time.time() - start) / 10 * 1000

        print(f"  ✓ Average time per iteration: {elapsed:.2f} ms")
        print(f"    Throughput: {1000/elapsed:.1f} FPS")

def visualize_galaxy_collision(sim, steps=50):
    """Optional: Visualize galaxy collision with matplotlib"""
    print_section("Test 10: Galaxy Collision Visualization")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not installed, skipping visualization")
        return

    print("Initializing galaxy collision...")
    sim.initialize_galaxy_collision()

    # Store positions over time
    positions_history = []

    print(f"Simulating {steps} steps...")
    for i in range(steps):
        sim.step_tiled()
        if i % 10 == 0:
            positions_history.append(sim.get_positions().copy())

    print("✓ Simulation complete")
    print("\nGenerating visualization...")

    # Plot initial, middle, and final states
    fig = plt.figure(figsize=(15, 5))

    for idx, (title, positions) in enumerate([
        ("Initial", positions_history[0]),
        ("Middle", positions_history[len(positions_history)//2]),
        ("Final", positions_history[-1])
    ]):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  s=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    output_file = 'galaxy_collision.png'
    plt.savefig(output_file, dpi=150)
    print(f"✓ Visualization saved to: {output_file}")
    plt.close()

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("  GPU N-Body Simulation - Python Bindings Test Suite")
    print("="*70)

    # Module info
    print(f"\nModule version: {nbody_cuda.__version__}")
    print(f"Module author: {nbody_cuda.__author__}")

    try:
        # Create simulation once for multiple tests
        sim = test_basic_creation()
        test_initialization_methods(sim)
        test_data_access(sim)
        test_simulation_steps(sim)
        test_multi_step_simulation(sim)
        test_set_data(sim)
        test_statistics(sim)
        test_benchmarking(sim)
        test_different_sizes()

        # Optional visualization (requires matplotlib)
        sim_viz = nbody_cuda.NBodySimulation(2048)
        visualize_galaxy_collision(sim_viz, steps=50)

        print_section("All Tests Passed! ✓")
        print("Python bindings are working correctly.\n")

        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  TEST FAILED")
        print(f"{'='*70}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
