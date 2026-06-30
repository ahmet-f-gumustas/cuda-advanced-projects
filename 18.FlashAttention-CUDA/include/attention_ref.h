#ifndef ATTENTION_REF_H
#define ATTENTION_REF_H

// ============================================================================
// CPU reference implementations of multi-head attention (forward + backward).
// Plain, slow, numerically-stable O(N^2 * d) loops — the ground truth the CUDA
// kernels are checked against. Header-only so tests, benchmark and demo share
// the exact same definition.
//
// Tensor layout matches the kernels: [batch, num_heads, seq_len, head_dim].
// ============================================================================

#include <vector>
#include <cmath>
#include <cstddef>

// One attention head's worth of indexing helper.
struct RefDims {
    int batch, heads, seq, dim;
    bool causal;
    float scale;
    size_t bh_stride() const { return (size_t)seq * dim; }
};

// ----------------------------------------------------------------------------
// Forward. Produces O and (optionally) the per-row log-sum-exp L.
//   Q,K,V,O : [B,H,N,D]   L : [B,H,N]
// ----------------------------------------------------------------------------
inline void ref_attention_forward(const std::vector<float>& Q,
                                  const std::vector<float>& K,
                                  const std::vector<float>& V,
                                  std::vector<float>& O,
                                  std::vector<float>& L,
                                  const RefDims& d)
{
    const int B = d.batch, H = d.heads, N = d.seq, D = d.dim;
    O.assign((size_t)B * H * N * D, 0.0f);
    L.assign((size_t)B * H * N, 0.0f);

    std::vector<float> scores(N);
    for (int bh = 0; bh < B * H; ++bh) {
        const float* Qb = Q.data() + (size_t)bh * N * D;
        const float* Kb = K.data() + (size_t)bh * N * D;
        const float* Vb = V.data() + (size_t)bh * N * D;
        float* Ob = O.data() + (size_t)bh * N * D;
        float* Lb = L.data() + (size_t)bh * N;

        for (int i = 0; i < N; ++i) {
            int jmax = d.causal ? i : (N - 1);
            // scaled scores + running max
            float m = -1e30f;
            for (int j = 0; j <= jmax; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < D; ++k)
                    dot += Qb[(size_t)i * D + k] * Kb[(size_t)j * D + k];
                dot *= d.scale;
                scores[j] = dot;
                if (dot > m) m = dot;
            }
            // softmax denominator
            float l = 0.0f;
            for (int j = 0; j <= jmax; ++j) {
                float e = std::exp(scores[j] - m);
                scores[j] = e;
                l += e;
            }
            // weighted sum of V
            float inv = 1.0f / l;
            for (int k = 0; k < D; ++k) {
                float acc = 0.0f;
                for (int j = 0; j <= jmax; ++j)
                    acc += scores[j] * Vb[(size_t)j * D + k];
                Ob[(size_t)i * D + k] = acc * inv;
            }
            Lb[i] = m + std::log(l);
        }
    }
}

// ----------------------------------------------------------------------------
// Backward. Given dO, compute dQ, dK, dV (recompute P from Q,K and the stable
// max; does not require L but is consistent with it).
// ----------------------------------------------------------------------------
inline void ref_attention_backward(const std::vector<float>& Q,
                                   const std::vector<float>& K,
                                   const std::vector<float>& V,
                                   const std::vector<float>& O,
                                   const std::vector<float>& dO,
                                   std::vector<float>& dQ,
                                   std::vector<float>& dK,
                                   std::vector<float>& dV,
                                   const RefDims& d)
{
    const int B = d.batch, H = d.heads, N = d.seq, D = d.dim;
    dQ.assign((size_t)B * H * N * D, 0.0f);
    dK.assign((size_t)B * H * N * D, 0.0f);
    dV.assign((size_t)B * H * N * D, 0.0f);

    std::vector<float> P(N);
    for (int bh = 0; bh < B * H; ++bh) {
        const float* Qb  = Q.data()  + (size_t)bh * N * D;
        const float* Kb  = K.data()  + (size_t)bh * N * D;
        const float* Vb  = V.data()  + (size_t)bh * N * D;
        const float* Ob  = O.data()  + (size_t)bh * N * D;
        const float* dOb = dO.data() + (size_t)bh * N * D;
        float* dQb = dQ.data() + (size_t)bh * N * D;
        float* dKb = dK.data() + (size_t)bh * N * D;
        float* dVb = dV.data() + (size_t)bh * N * D;

        for (int i = 0; i < N; ++i) {
            int jmax = d.causal ? i : (N - 1);
            // recompute softmax row P[i, :]
            float m = -1e30f;
            for (int j = 0; j <= jmax; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < D; ++k)
                    dot += Qb[(size_t)i * D + k] * Kb[(size_t)j * D + k];
                dot *= d.scale;
                P[j] = dot;
                if (dot > m) m = dot;
            }
            float l = 0.0f;
            for (int j = 0; j <= jmax; ++j) { P[j] = std::exp(P[j] - m); l += P[j]; }
            float inv = 1.0f / l;
            for (int j = 0; j <= jmax; ++j) P[j] *= inv;

            // D_i = sum_k dO_i . O_i
            float Di = 0.0f;
            for (int k = 0; k < D; ++k)
                Di += dOb[(size_t)i * D + k] * Ob[(size_t)i * D + k];

            for (int j = 0; j <= jmax; ++j) {
                // dP_ij = dO_i . V_j
                float dp = 0.0f;
                for (int k = 0; k < D; ++k)
                    dp += dOb[(size_t)i * D + k] * Vb[(size_t)j * D + k];
                float ds = P[j] * (dp - Di) * d.scale;   // grad wrt scaled score
                for (int k = 0; k < D; ++k) {
                    // dV_j += P_ij * dO_i
                    dVb[(size_t)j * D + k] += P[j] * dOb[(size_t)i * D + k];
                    // dQ_i += ds * K_j ; dK_j += ds * Q_i
                    dQb[(size_t)i * D + k] += ds * Kb[(size_t)j * D + k];
                    dKb[(size_t)j * D + k] += ds * Qb[(size_t)i * D + k];
                }
            }
        }
    }
}

#endif // ATTENTION_REF_H
