#include "ply_reader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace {

struct PropertyInfo {
    std::string name;
    std::string type;
    int byte_offset = 0;
    int byte_size   = 0;
};

int sizeOfType(const std::string& t) {
    if (t == "float"  || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    if (t == "uchar"  || t == "uint8")   return 1;
    if (t == "ushort" || t == "uint16")  return 2;
    if (t == "uint"   || t == "uint32")  return 4;
    if (t == "char"   || t == "int8")    return 1;
    if (t == "short"  || t == "int16")   return 2;
    if (t == "int"    || t == "int32")   return 4;
    return 0;
}

bool parseHeader(std::ifstream& fin,
                  std::vector<PropertyInfo>& props,
                  int& num_points,
                  bool& binary_le) {
    std::string line;
    bool has_ply = false;
    bool has_format = false;
    binary_le = false;
    num_points = 0;
    int offset = 0;

    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line == "ply") { has_ply = true; continue; }
        if (line.rfind("format ", 0) == 0) {
            has_format = true;
            if (line.find("binary_little_endian") != std::string::npos) binary_le = true;
            continue;
        }
        if (line.rfind("element vertex ", 0) == 0) {
            num_points = std::stoi(line.substr(15));
            continue;
        }
        if (line.rfind("property ", 0) == 0) {
            std::istringstream iss(line);
            std::string token, type, name;
            iss >> token >> type >> name;
            int sz = sizeOfType(type);
            props.push_back({name, type, offset, sz});
            offset += sz;
            continue;
        }
        if (line == "end_header") return has_ply && has_format;
    }
    return false;
}

int findProperty(const std::vector<PropertyInfo>& props, const std::string& name) {
    for (size_t i = 0; i < props.size(); i++) {
        if (props[i].name == name) return (int)i;
    }
    return -1;
}

float readFloatAt(const uint8_t* row, int offset) {
    float v;
    std::memcpy(&v, row + offset, sizeof(float));
    return v;
}

void writeFloat(std::ofstream& fout, float v) {
    fout.write(reinterpret_cast<const char*>(&v), sizeof(float));
}

} // namespace

// ============================================================
// PLYReader::readFile
// ============================================================

bool PLYReader::readFile(const std::string& path, PLYData& out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::cerr << "PLYReader: failed to open " << path << std::endl;
        return false;
    }

    std::vector<PropertyInfo> props;
    int N = 0;
    bool binary_le = false;
    if (!parseHeader(fin, props, N, binary_le)) {
        std::cerr << "PLYReader: invalid header" << std::endl;
        return false;
    }
    if (!binary_le) {
        std::cerr << "PLYReader: only binary_little_endian is supported" << std::endl;
        return false;
    }
    if (N <= 0) {
        std::cerr << "PLYReader: bad vertex count " << N << std::endl;
        return false;
    }

    int stride = 0;
    for (auto& p : props) stride += p.byte_size;

    int idx_x      = findProperty(props, "x");
    int idx_y      = findProperty(props, "y");
    int idx_z      = findProperty(props, "z");
    int idx_dc[3]  = { findProperty(props, "f_dc_0"),
                        findProperty(props, "f_dc_1"),
                        findProperty(props, "f_dc_2") };
    int idx_opa    = findProperty(props, "opacity");
    int idx_sca[3] = { findProperty(props, "scale_0"),
                        findProperty(props, "scale_1"),
                        findProperty(props, "scale_2") };
    int idx_rot[4] = { findProperty(props, "rot_0"),
                        findProperty(props, "rot_1"),
                        findProperty(props, "rot_2"),
                        findProperty(props, "rot_3") };

    if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
        std::cerr << "PLYReader: missing x/y/z" << std::endl;
        return false;
    }

    // Count f_rest_* properties → infer SH degree
    int max_rest = -1;
    for (auto& p : props) {
        if (p.name.rfind("f_rest_", 0) == 0) {
            int n = std::stoi(p.name.substr(7));
            if (n > max_rest) max_rest = n;
        }
    }
    int num_rest = max_rest + 1;
    std::vector<int> idx_rest(num_rest, -1);
    for (int n = 0; n < num_rest; n++) {
        idx_rest[n] = findProperty(props, "f_rest_" + std::to_string(n));
    }

    int rest_per_channel = num_rest / 3;
    int total_coeffs = 1 + rest_per_channel;
    int sh_degree = 0;
    if (total_coeffs >= 16) sh_degree = 3;
    else if (total_coeffs >=  9) sh_degree = 2;
    else if (total_coeffs >=  4) sh_degree = 1;
    else                          sh_degree = 0;
    int target_coeffs = (sh_degree + 1) * (sh_degree + 1);

    // Read all vertex data
    std::vector<uint8_t> blob((size_t)N * stride);
    fin.read(reinterpret_cast<char*>(blob.data()), blob.size());
    if ((size_t)fin.gcount() != blob.size()) {
        std::cerr << "PLYReader: short read on vertex data" << std::endl;
        return false;
    }

    out.positions.resize((size_t)N * 3);
    out.scales.assign((size_t)N * 3, 0.0f);
    out.rotations.assign((size_t)N * 4, 0.0f);
    out.opacities.assign(N, 0.0f);
    out.sh_coeffs.assign((size_t)N * target_coeffs * 3, 0.0f);
    out.num_points = N;
    out.sh_degree = sh_degree;
    out.num_sh_coeffs = target_coeffs;

    for (int i = 0; i < N; i++) {
        const uint8_t* row = blob.data() + (size_t)i * stride;

        out.positions[i*3+0] = readFloatAt(row, props[idx_x].byte_offset);
        out.positions[i*3+1] = readFloatAt(row, props[idx_y].byte_offset);
        out.positions[i*3+2] = readFloatAt(row, props[idx_z].byte_offset);

        if (idx_sca[0] >= 0) {
            out.scales[i*3+0] = readFloatAt(row, props[idx_sca[0]].byte_offset);
            out.scales[i*3+1] = readFloatAt(row, props[idx_sca[1]].byte_offset);
            out.scales[i*3+2] = readFloatAt(row, props[idx_sca[2]].byte_offset);
        }

        if (idx_rot[0] >= 0) {
            float w = readFloatAt(row, props[idx_rot[0]].byte_offset);
            float x = readFloatAt(row, props[idx_rot[1]].byte_offset);
            float y = readFloatAt(row, props[idx_rot[2]].byte_offset);
            float z = readFloatAt(row, props[idx_rot[3]].byte_offset);
            float n = std::sqrt(w*w + x*x + y*y + z*z);
            if (n > 1e-9f) {
                out.rotations[i*4+0] = w/n;
                out.rotations[i*4+1] = x/n;
                out.rotations[i*4+2] = y/n;
                out.rotations[i*4+3] = z/n;
            } else {
                out.rotations[i*4+0] = 1.0f;
            }
        } else {
            out.rotations[i*4+0] = 1.0f;
        }

        if (idx_opa >= 0) {
            out.opacities[i] = readFloatAt(row, props[idx_opa].byte_offset);
        }

        // DC term (coefficient 0)
        if (idx_dc[0] >= 0) {
            out.sh_coeffs[i*target_coeffs*3 + 0] = readFloatAt(row, props[idx_dc[0]].byte_offset);
            out.sh_coeffs[i*target_coeffs*3 + 1] = readFloatAt(row, props[idx_dc[1]].byte_offset);
            out.sh_coeffs[i*target_coeffs*3 + 2] = readFloatAt(row, props[idx_dc[2]].byte_offset);
        }

        // Rest terms — 3DGS PLY uses channel-major layout:
        // f_rest_0..(K-1)        = R for coeffs 1..K
        // f_rest_K..(2K-1)       = G for coeffs 1..K
        // f_rest_2K..(3K-1)      = B for coeffs 1..K
        // We need coefficient-major interleaved (c1_R, c1_G, c1_B, c2_R, ...).
        for (int c = 1; c < target_coeffs; c++) {
            int rpc = c - 1;
            int r_idx = rpc;
            int g_idx = rpc + rest_per_channel;
            int b_idx = rpc + 2 * rest_per_channel;
            if (r_idx < num_rest && idx_rest[r_idx] >= 0)
                out.sh_coeffs[i*target_coeffs*3 + c*3 + 0] = readFloatAt(row, props[idx_rest[r_idx]].byte_offset);
            if (g_idx < num_rest && idx_rest[g_idx] >= 0)
                out.sh_coeffs[i*target_coeffs*3 + c*3 + 1] = readFloatAt(row, props[idx_rest[g_idx]].byte_offset);
            if (b_idx < num_rest && idx_rest[b_idx] >= 0)
                out.sh_coeffs[i*target_coeffs*3 + c*3 + 2] = readFloatAt(row, props[idx_rest[b_idx]].byte_offset);
        }
    }

    std::cout << "Loaded PLY: " << path << std::endl;
    std::cout << "  Vertices: " << N << std::endl;
    std::cout << "  SH degree: " << sh_degree << " (" << target_coeffs << " coeffs)" << std::endl;
    return true;
}

// ============================================================
// PLYReader::writeSyntheticPLY
// ============================================================

bool PLYReader::writeSyntheticPLY(const std::string& path,
                                   int num_points,
                                   int sh_degree,
                                   unsigned seed) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        std::cerr << "PLYReader: cannot write " << path << std::endl;
        return false;
    }

    int total_coeffs   = (sh_degree + 1) * (sh_degree + 1);
    int rest_per_chan  = total_coeffs - 1;
    int num_rest       = rest_per_chan * 3;

    // ---- Header ----
    fout << "ply\n";
    fout << "format binary_little_endian 1.0\n";
    fout << "element vertex " << num_points << "\n";
    fout << "property float x\n";
    fout << "property float y\n";
    fout << "property float z\n";
    fout << "property float nx\n";
    fout << "property float ny\n";
    fout << "property float nz\n";
    fout << "property float f_dc_0\n";
    fout << "property float f_dc_1\n";
    fout << "property float f_dc_2\n";
    for (int i = 0; i < num_rest; i++) fout << "property float f_rest_" << i << "\n";
    fout << "property float opacity\n";
    fout << "property float scale_0\n";
    fout << "property float scale_1\n";
    fout << "property float scale_2\n";
    fout << "property float rot_0\n";
    fout << "property float rot_1\n";
    fout << "property float rot_2\n";
    fout << "property float rot_3\n";
    fout << "end_header\n";

    // ---- Body ----
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> sca_dist(-5.0f, -3.0f);
    std::normal_distribution<float>      rot_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> col_dist(-0.5f, 0.5f);
    std::normal_distribution<float>      shr_dist(0.0f, 0.05f);
    std::uniform_real_distribution<float> opa_dist(0.5f, 2.0f);

    for (int i = 0; i < num_points; i++) {
        // Position
        writeFloat(fout, pos_dist(rng));
        writeFloat(fout, pos_dist(rng));
        writeFloat(fout, pos_dist(rng));
        // Normals (zero)
        writeFloat(fout, 0.0f);
        writeFloat(fout, 0.0f);
        writeFloat(fout, 0.0f);
        // f_dc_* (DC color, RGB)
        writeFloat(fout, col_dist(rng));
        writeFloat(fout, col_dist(rng));
        writeFloat(fout, col_dist(rng));
        // f_rest_* (channel-major: all R rest, then G rest, then B rest)
        for (int c = 0; c < num_rest; c++) writeFloat(fout, shr_dist(rng));
        // Opacity
        writeFloat(fout, opa_dist(rng));
        // Scales (log)
        writeFloat(fout, sca_dist(rng));
        writeFloat(fout, sca_dist(rng));
        writeFloat(fout, sca_dist(rng));
        // Rotation (random unit quat: w,x,y,z)
        float w = rot_dist(rng), x = rot_dist(rng), y = rot_dist(rng), z = rot_dist(rng);
        float n = std::sqrt(w*w + x*x + y*y + z*z);
        if (n < 1e-9f) { w = 1.0f; x = y = z = 0.0f; n = 1.0f; }
        writeFloat(fout, w/n);
        writeFloat(fout, x/n);
        writeFloat(fout, y/n);
        writeFloat(fout, z/n);
    }

    std::cout << "Wrote synthetic PLY: " << path << " (" << num_points
              << " pts, SH deg " << sh_degree << ")\n";
    return true;
}
