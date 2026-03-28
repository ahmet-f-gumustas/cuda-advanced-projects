#ifndef ENCODER_H
#define ENCODER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

struct EncoderConfig {
    int d_model = 512;
    int n_heads = 8;
    int n_layers = 4;
    int ffn_dim = 2048;
    int n_mels = 80;
    int max_seq_len = 1500;   // Max audio frames (~30s)
    int conv1_kernel = 3;
    int conv2_kernel = 3;
    int conv2_stride = 2;     // Downsample by 2x
};

struct EncoderLayerWeights {
    // Self-attention: Q, K, V, O projections
    float* d_wq;    // [d_model, d_model]
    float* d_wk;
    float* d_wv;
    float* d_wo;
    float* d_bq;    // [d_model]
    float* d_bk;
    float* d_bv;
    float* d_bo;

    // LayerNorm 1 (pre-attention)
    float* d_ln1_gamma;  // [d_model]
    float* d_ln1_beta;

    // FFN
    float* d_w1;    // [d_model, ffn_dim]
    float* d_b1;    // [ffn_dim]
    float* d_w2;    // [ffn_dim, d_model]
    float* d_b2;    // [d_model]

    // LayerNorm 2 (pre-FFN)
    float* d_ln2_gamma;
    float* d_ln2_beta;
};

class WhisperEncoder {
public:
    WhisperEncoder(const EncoderConfig& config);
    ~WhisperEncoder();

    // Forward pass: d_mel[num_frames, n_mels] -> d_output[out_len, d_model]
    void forward(const float* d_mel, int num_frames, float* d_output, int& out_len);

    // Initialize with random weights (Xavier)
    void init_random_weights();

    // Get output sequence length after conv downsampling
    int get_output_length(int num_frames) const;

    EncoderConfig config;

private:
    cublasHandle_t cublas_;

    // Conv stem weights
    float* d_conv1_weight;   // [d_model, n_mels, conv1_kernel]
    float* d_conv1_bias;     // [d_model]
    float* d_conv2_weight;   // [d_model, d_model, conv2_kernel]
    float* d_conv2_bias;     // [d_model]

    // Positional encoding
    float* d_pe;             // [max_seq_len, d_model]

    // Final LayerNorm
    float* d_ln_final_gamma;
    float* d_ln_final_beta;

    // Transformer layers
    EncoderLayerWeights* layers_;

    // Workspace buffers
    float* d_conv1_out;
    float* d_conv2_out;
    float* d_ln_out;
    float* d_q;
    float* d_k;
    float* d_v;
    float* d_q_heads;
    float* d_k_heads;
    float* d_v_heads;
    float* d_attn_scores;
    float* d_attn_out;
    float* d_attn_proj;
    float* d_residual;
    float* d_ffn_mid;
    float* d_ffn_out;

    void allocate_layer_weights(EncoderLayerWeights& layer);
    void free_layer_weights(EncoderLayerWeights& layer);
    void allocate_workspace(int max_seq);

    // Self-attention forward for one layer
    void self_attention(const float* d_input, float* d_output,
                        EncoderLayerWeights& layer, int seq_len);

    // FFN forward for one layer
    void ffn(const float* d_input, float* d_output,
             EncoderLayerWeights& layer, int seq_len);
};

#endif // ENCODER_H
