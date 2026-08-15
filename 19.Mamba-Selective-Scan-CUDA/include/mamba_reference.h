#ifndef MAMBA_REFERENCE_H
#define MAMBA_REFERENCE_H

#include "selective_scan.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

inline float stable_softplus(float value) {
    if (value > 20.0f) {
        return value;
    }
    if (value < -20.0f) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

inline void selective_scan_reference(const float* u,
                                     const float* delta,
                                     const float* A,
                                     const float* B,
                                     const float* C,
                                     const float* D,
                                     float* y,
                                     const SelectiveScanConfig& config) {
    const int length = config.seq_len;
    const int dim = config.dim;
    const int states = config.state_size;

    std::vector<float> hidden(static_cast<size_t>(dim) * states);
    for (int batch = 0; batch < config.batch; ++batch) {
        std::fill(hidden.begin(), hidden.end(), 0.0f);
        for (int token = 0; token < length; ++token) {
            for (int channel = 0; channel < dim; ++channel) {
                const size_t token_index =
                    (static_cast<size_t>(batch) * length + token) * dim + channel;
                const float step = stable_softplus(delta[token_index] + config.delta_bias);
                float output = D[channel] * u[token_index];
                for (int state = 0; state < states; ++state) {
                    const size_t parameter_index =
                        (static_cast<size_t>(batch) * length + token) * states + state;
                    const size_t hidden_index = static_cast<size_t>(channel) * states + state;
                    const float transition = std::exp(step * A[hidden_index]);
                    const float input = step * B[parameter_index] * u[token_index];
                    hidden[hidden_index] = transition * hidden[hidden_index] + input;
                    output += C[parameter_index] * hidden[hidden_index];
                }
                y[token_index] = output;
            }
        }
    }
}

#endif
