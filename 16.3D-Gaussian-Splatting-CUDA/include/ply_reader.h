#ifndef PLY_READER_H
#define PLY_READER_H

#include <string>
#include <vector>
#include <cstdint>

// ============================================================
// Parsed PLY data — host-side, ready to upload to GaussianModel
// ============================================================

struct PLYData {
    std::vector<float> positions;     // (N x 3)
    std::vector<float> scales;        // (N x 3) log-scale
    std::vector<float> rotations;     // (N x 4) quaternion (w, x, y, z), normalized
    std::vector<float> sh_coeffs;     // (N x C x 3) coefficient-major interleaved
    std::vector<float> opacities;     // (N) pre-sigmoid
    int num_points = 0;
    int sh_degree  = 0;
    int num_sh_coeffs = 0;
};

// ============================================================
// PLY reader/writer — supports the 3DGS canonical .ply layout
// (binary_little_endian, properties: x,y,z, f_dc_*, f_rest_*,
//  opacity, scale_*, rot_*)
// ============================================================

class PLYReader {
public:
    static bool readFile(const std::string& path, PLYData& out);

    // Generate a synthetic 3DGS-format .ply for testing.
    // Returns true on success.
    static bool writeSyntheticPLY(const std::string& path,
                                   int num_points,
                                   int sh_degree,
                                   unsigned seed = 42);
};

#endif // PLY_READER_H
