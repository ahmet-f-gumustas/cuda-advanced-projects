#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nbody.h"
#include "cuda_utils.h"

namespace py = pybind11;

// ============================================================================
// Python-friendly wrapper class for N-Body simulation
// ============================================================================

class NBodySimulation {
private:
    Bodies* host_bodies_;
    DeviceBodies* device_bodies_;
    int num_bodies_;
    float dt_;
    float softening_;

public:
    NBodySimulation(int numBodies, float dt = 0.01f, float softening = 0.1f)
        : num_bodies_(numBodies), dt_(dt), softening_(softening) {

        host_bodies_ = createBodies(numBodies);
        device_bodies_ = createDeviceBodies(numBodies);
    }

    ~NBodySimulation() {
        if (host_bodies_) freeBodies(host_bodies_);
        if (device_bodies_) freeDeviceBodies(device_bodies_);
    }

    // Initialization methods
    void initializeRandom(float positionScale = 10.0f,
                         float velocityScale = 1.0f,
                         float massScale = 1.0f) {
        initializeRandomBodies(host_bodies_, positionScale, velocityScale, massScale);
        copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    void initializeGalaxyCollision() {
        ::initializeGalaxyCollision(host_bodies_);
        copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    void initializeSphericalCluster(float radius = 10.0f) {
        ::initializeSphericalCluster(host_bodies_, radius);
        copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    // Simulation step methods
    void stepNaive() {
        NBodyKernels::computeNaive(*device_bodies_, num_bodies_, dt_, softening_);
        copyBodiesFromDevice(host_bodies_, device_bodies_);
    }

    void stepShared() {
        NBodyKernels::computeSharedMemory(*device_bodies_, num_bodies_, dt_, softening_);
        copyBodiesFromDevice(host_bodies_, device_bodies_);
    }

    void stepTiled() {
        NBodyKernels::computeRegisterTiled(*device_bodies_, num_bodies_, dt_, softening_);
        copyBodiesFromDevice(host_bodies_, device_bodies_);
    }

    // Multi-step simulation
    void simulate(int iterations, const std::string& kernel = "tiled") {
        for (int i = 0; i < iterations; ++i) {
            if (kernel == "naive") {
                NBodyKernels::computeNaive(*device_bodies_, num_bodies_, dt_, softening_);
            } else if (kernel == "shared") {
                NBodyKernels::computeSharedMemory(*device_bodies_, num_bodies_, dt_, softening_);
            } else {
                NBodyKernels::computeRegisterTiled(*device_bodies_, num_bodies_, dt_, softening_);
            }
        }
        copyBodiesFromDevice(host_bodies_, device_bodies_);
    }

    // Data access methods (return numpy arrays)
    py::array_t<float> getPositionsX() const {
        return py::array_t<float>(num_bodies_, host_bodies_->x);
    }

    py::array_t<float> getPositionsY() const {
        return py::array_t<float>(num_bodies_, host_bodies_->y);
    }

    py::array_t<float> getPositionsZ() const {
        return py::array_t<float>(num_bodies_, host_bodies_->z);
    }

    py::array_t<float> getVelocitiesX() const {
        return py::array_t<float>(num_bodies_, host_bodies_->vx);
    }

    py::array_t<float> getVelocitiesY() const {
        return py::array_t<float>(num_bodies_, host_bodies_->vy);
    }

    py::array_t<float> getVelocitiesZ() const {
        return py::array_t<float>(num_bodies_, host_bodies_->vz);
    }

    py::array_t<float> getMasses() const {
        return py::array_t<float>(num_bodies_, host_bodies_->mass);
    }

    // Get all positions as Nx3 array
    py::array_t<float> getPositions() const {
        auto result = py::array_t<float>({num_bodies_, 3});
        auto r = result.mutable_unchecked<2>();

        for (int i = 0; i < num_bodies_; ++i) {
            r(i, 0) = host_bodies_->x[i];
            r(i, 1) = host_bodies_->y[i];
            r(i, 2) = host_bodies_->z[i];
        }

        return result;
    }

    // Get all velocities as Nx3 array
    py::array_t<float> getVelocities() const {
        auto result = py::array_t<float>({num_bodies_, 3});
        auto r = result.mutable_unchecked<2>();

        for (int i = 0; i < num_bodies_; ++i) {
            r(i, 0) = host_bodies_->vx[i];
            r(i, 1) = host_bodies_->vy[i];
            r(i, 2) = host_bodies_->vz[i];
        }

        return result;
    }

    // Set positions from numpy array
    void setPositions(py::array_t<float> positions) {
        auto pos = positions.unchecked<2>();
        if (pos.shape(0) != num_bodies_ || pos.shape(1) != 3) {
            throw std::runtime_error("Invalid positions shape");
        }

        for (int i = 0; i < num_bodies_; ++i) {
            host_bodies_->x[i] = pos(i, 0);
            host_bodies_->y[i] = pos(i, 1);
            host_bodies_->z[i] = pos(i, 2);
        }

        copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    // Set velocities from numpy array
    void setVelocities(py::array_t<float> velocities) {
        auto vel = velocities.unchecked<2>();
        if (vel.shape(0) != num_bodies_ || vel.shape(1) != 3) {
            throw std::runtime_error("Invalid velocities shape");
        }

        for (int i = 0; i < num_bodies_; ++i) {
            host_bodies_->vx[i] = vel(i, 0);
            host_bodies_->vy[i] = vel(i, 1);
            host_bodies_->vz[i] = vel(i, 2);
        }

        copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    // Set masses from numpy array
    void setMasses(py::array_t<float> masses) {
        auto m = masses.unchecked<1>();
        if (m.shape(0) != num_bodies_) {
            throw std::runtime_error("Invalid masses shape");
        }

        for (int i = 0; i < num_bodies_; ++i) {
            host_bodies_->mass[i] = m(i);
        }

        copyBodiesToDevice(host_bodies_, device_bodies_);
    }

    // Statistics
    float getTotalEnergy() const {
        return computeTotalEnergy(host_bodies_);
    }

    py::dict getStatistics() const {
        // Compute center of mass
        float cmx = 0.0f, cmy = 0.0f, cmz = 0.0f;
        float totalMass = 0.0f;

        for (int i = 0; i < num_bodies_; ++i) {
            cmx += host_bodies_->mass[i] * host_bodies_->x[i];
            cmy += host_bodies_->mass[i] * host_bodies_->y[i];
            cmz += host_bodies_->mass[i] * host_bodies_->z[i];
            totalMass += host_bodies_->mass[i];
        }

        cmx /= totalMass;
        cmy /= totalMass;
        cmz /= totalMass;

        // Compute average velocity
        float avgVx = 0.0f, avgVy = 0.0f, avgVz = 0.0f;
        for (int i = 0; i < num_bodies_; ++i) {
            avgVx += host_bodies_->vx[i];
            avgVy += host_bodies_->vy[i];
            avgVz += host_bodies_->vz[i];
        }
        avgVx /= num_bodies_;
        avgVy /= num_bodies_;
        avgVz /= num_bodies_;

        py::dict stats;
        stats["num_bodies"] = num_bodies_;
        stats["total_mass"] = totalMass;
        stats["center_of_mass"] = py::make_tuple(cmx, cmy, cmz);
        stats["avg_velocity"] = py::make_tuple(avgVx, avgVy, avgVz);
        stats["total_energy"] = computeTotalEnergy(host_bodies_);

        return stats;
    }

    // Benchmarking
    py::dict benchmark(int iterations = 100) {
        // Benchmark naive kernel
        double naiveTime = benchmarkKernel(NBodyKernels::computeNaive,
                                          *device_bodies_, num_bodies_,
                                          dt_, softening_, iterations);

        // Benchmark shared memory kernel
        double sharedTime = benchmarkKernel(NBodyKernels::computeSharedMemory,
                                           *device_bodies_, num_bodies_,
                                           dt_, softening_, iterations);

        // Benchmark register-tiled kernel
        double tiledTime = benchmarkKernel(NBodyKernels::computeRegisterTiled,
                                          *device_bodies_, num_bodies_,
                                          dt_, softening_, iterations);

        py::dict results;
        results["naive_ms"] = naiveTime;
        results["shared_ms"] = sharedTime;
        results["tiled_ms"] = tiledTime;
        results["naive_fps"] = 1000.0 / naiveTime;
        results["shared_fps"] = 1000.0 / sharedTime;
        results["tiled_fps"] = 1000.0 / tiledTime;
        results["speedup_shared"] = naiveTime / sharedTime;
        results["speedup_tiled"] = naiveTime / tiledTime;
        results["iterations"] = iterations;

        return results;
    }

    // Getters for parameters
    int getNumBodies() const { return num_bodies_; }
    float getDeltaTime() const { return dt_; }
    float getSoftening() const { return softening_; }

    // Setters for parameters
    void setDeltaTime(float dt) { dt_ = dt; }
    void setSoftening(float softening) { softening_ = softening; }
};

// ============================================================================
// Pybind11 Module Definition
// ============================================================================

PYBIND11_MODULE(nbody_cuda, m) {
    m.doc() = "GPU N-Body Simulation Python Bindings - CUDA accelerated gravitational simulation";

    // Main simulation class
    py::class_<NBodySimulation>(m, "NBodySimulation")
        .def(py::init<int, float, float>(),
             py::arg("num_bodies"),
             py::arg("dt") = 0.01f,
             py::arg("softening") = 0.1f,
             "Initialize N-Body simulation\n\n"
             "Parameters:\n"
             "  num_bodies: Number of bodies in simulation\n"
             "  dt: Time step for integration (default: 0.01)\n"
             "  softening: Softening factor to prevent singularities (default: 0.1)")

        // Initialization methods
        .def("initialize_random", &NBodySimulation::initializeRandom,
             py::arg("position_scale") = 10.0f,
             py::arg("velocity_scale") = 1.0f,
             py::arg("mass_scale") = 1.0f,
             "Initialize bodies with random positions, velocities, and masses")

        .def("initialize_galaxy_collision", &NBodySimulation::initializeGalaxyCollision,
             "Initialize two galaxies on collision course")

        .def("initialize_spherical_cluster", &NBodySimulation::initializeSphericalCluster,
             py::arg("radius") = 10.0f,
             "Initialize bodies in a spherical cluster")

        // Simulation step methods
        .def("step_naive", &NBodySimulation::stepNaive,
             "Execute one simulation step using naive O(N²) kernel")

        .def("step_shared", &NBodySimulation::stepShared,
             "Execute one simulation step using shared memory kernel")

        .def("step_tiled", &NBodySimulation::stepTiled,
             "Execute one simulation step using register-tiled kernel (fastest)")

        .def("simulate", &NBodySimulation::simulate,
             py::arg("iterations"),
             py::arg("kernel") = "tiled",
             "Run multiple simulation steps\n\n"
             "Parameters:\n"
             "  iterations: Number of steps to simulate\n"
             "  kernel: Kernel to use ('naive', 'shared', or 'tiled')")

        // Data access
        .def("get_positions_x", &NBodySimulation::getPositionsX,
             "Get X coordinates as numpy array")
        .def("get_positions_y", &NBodySimulation::getPositionsY,
             "Get Y coordinates as numpy array")
        .def("get_positions_z", &NBodySimulation::getPositionsZ,
             "Get Z coordinates as numpy array")
        .def("get_velocities_x", &NBodySimulation::getVelocitiesX,
             "Get X velocities as numpy array")
        .def("get_velocities_y", &NBodySimulation::getVelocitiesY,
             "Get Y velocities as numpy array")
        .def("get_velocities_z", &NBodySimulation::getVelocitiesZ,
             "Get Z velocities as numpy array")
        .def("get_masses", &NBodySimulation::getMasses,
             "Get masses as numpy array")

        .def("get_positions", &NBodySimulation::getPositions,
             "Get all positions as Nx3 numpy array")
        .def("get_velocities", &NBodySimulation::getVelocities,
             "Get all velocities as Nx3 numpy array")

        .def("set_positions", &NBodySimulation::setPositions,
             py::arg("positions"),
             "Set positions from Nx3 numpy array")
        .def("set_velocities", &NBodySimulation::setVelocities,
             py::arg("velocities"),
             "Set velocities from Nx3 numpy array")
        .def("set_masses", &NBodySimulation::setMasses,
             py::arg("masses"),
             "Set masses from 1D numpy array")

        // Statistics and analysis
        .def("get_total_energy", &NBodySimulation::getTotalEnergy,
             "Compute total energy (kinetic + potential)")
        .def("get_statistics", &NBodySimulation::getStatistics,
             "Get simulation statistics (mass, center of mass, energy, etc.)")

        // Benchmarking
        .def("benchmark", &NBodySimulation::benchmark,
             py::arg("iterations") = 100,
             "Benchmark all three kernel implementations")

        // Property access
        .def("get_num_bodies", &NBodySimulation::getNumBodies,
             "Get number of bodies")
        .def("get_delta_time", &NBodySimulation::getDeltaTime,
             "Get time step")
        .def("get_softening", &NBodySimulation::getSoftening,
             "Get softening factor")

        .def("set_delta_time", &NBodySimulation::setDeltaTime,
             py::arg("dt"),
             "Set time step")
        .def("set_softening", &NBodySimulation::setSoftening,
             py::arg("softening"),
             "Set softening factor");

    // Module-level information
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "GPU N-Body Simulation Project";
}
