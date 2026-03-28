#include "cuda_utils.h"
#include "wav_reader.h"
#include "tokenizer.h"
#include "whisper_kernels.cuh"
#include "mel_spectrogram.h"
#include "encoder.h"
#include "decoder.h"
#include "beam_search.h"
#include "pipeline.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n[TEST] %s...\n", name)
#define PASS(name) do { printf("[PASS] %s\n", name); tests_passed++; } while(0)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); tests_failed++; } while(0)

// ============================================================
// Test 1: WAV Reader
// ============================================================
void test_wav_reader() {
    TEST("WAV Reader - generate and save/load");

    WavData wav;
    generate_test_audio(wav, 16000, 1.0f);
    assert(wav.num_samples == 16000);
    assert(wav.sample_rate == 16000);

    // Save and reload
    bool saved = save_wav("/tmp/test_whisper.wav", wav.samples.data(),
                           wav.num_samples, wav.sample_rate);
    assert(saved);

    WavData loaded;
    bool loaded_ok = load_wav("/tmp/test_whisper.wav", loaded);
    assert(loaded_ok);
    assert(loaded.num_samples == wav.num_samples);
    assert(loaded.sample_rate == wav.sample_rate);

    // Verify samples are close (16-bit quantization)
    float max_err = 0.0f;
    for (int i = 0; i < wav.num_samples; i++) {
        float err = fabsf(wav.samples[i] - loaded.samples[i]);
        max_err = fmaxf(max_err, err);
    }
    printf("  Max quantization error: %.6f\n", max_err);
    assert(max_err < 0.001f);

    PASS("WAV Reader");
}

// ============================================================
// Test 2: Tokenizer
// ============================================================
void test_tokenizer() {
    TEST("Tokenizer - encode/decode");

    CharTokenizer tok;
    printf("  Vocab size: %d\n", tok.vocab_size());

    std::string text = "hello world 123";
    auto tokens = tok.encode(text);
    printf("  Encoded '%s' -> %d tokens: [", text.c_str(), (int)tokens.size());
    for (int t : tokens) printf("%d ", t);
    printf("]\n");

    assert(tokens.front() == tok.bos_token());
    assert(tokens.back() == tok.eos_token());

    std::string decoded = tok.decode(tokens);
    printf("  Decoded: '%s'\n", decoded.c_str());
    assert(decoded == text);

    PASS("Tokenizer");
}

// ============================================================
// Test 3: CUDA Kernels
// ============================================================
void test_kernels() {
    TEST("CUDA Kernels");

    // Test GELU
    {
        float h_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        int n = 5;
        float* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
        launch_gelu(d_data, n);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_data);

        // GELU(0) should be 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        printf("  GELU results: ");
        for (int i = 0; i < n; i++) printf("%.3f ", h_data[i]);
        printf("\n");
        assert(fabsf(h_data[2]) < 0.001f);        // GELU(0) = 0
        assert(h_data[3] > 0.8f && h_data[3] < 0.9f); // GELU(1) ≈ 0.841
    }

    // Test LayerNorm
    {
        int rows = 2, cols = 4;
        float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float h_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float h_beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
        float h_output[8];

        float *d_in, *d_gamma, *d_beta, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, 8 * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_in, h_input, 8 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, 4 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta, 4 * sizeof(float), cudaMemcpyHostToDevice));

        launch_layer_norm(d_in, d_gamma, d_beta, d_out, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_output, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost));

        printf("  LayerNorm row 0: ");
        for (int i = 0; i < 4; i++) printf("%.3f ", h_output[i]);
        printf("\n");

        // Normalized output should have mean ≈ 0, std ≈ 1
        float sum = 0;
        for (int i = 0; i < 4; i++) sum += h_output[i];
        assert(fabsf(sum / 4.0f) < 0.01f);

        cudaFree(d_in); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_out);
    }

    // Test Softmax
    {
        int rows = 1, cols = 4;
        float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, 4 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, 4 * sizeof(float), cudaMemcpyHostToDevice));
        launch_softmax(d_data, rows, cols);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_data);

        float sum = 0;
        printf("  Softmax: ");
        for (int i = 0; i < 4; i++) {
            printf("%.4f ", h_data[i]);
            sum += h_data[i];
        }
        printf("(sum=%.4f)\n", sum);
        assert(fabsf(sum - 1.0f) < 0.001f);
    }

    // Test Conv1D
    {
        int in_ch = 2, out_ch = 3, length = 5, kernel = 3;
        int out_len = (length + 2 * 1 - kernel) / 1 + 1; // stride=1, padding=1
        std::vector<float> h_input(in_ch * length, 1.0f);
        std::vector<float> h_weight(out_ch * in_ch * kernel, 0.1f);
        std::vector<float> h_bias(out_ch, 0.0f);
        std::vector<float> h_output(out_ch * out_len, 0.0f);

        float *d_in, *d_w, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, h_input.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, h_weight.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, h_bias.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, h_output.size() * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

        launch_conv1d(d_in, d_w, d_b, d_out, in_ch, out_ch, length, kernel, 1, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        printf("  Conv1D output[0]: ");
        for (int i = 0; i < out_len; i++) printf("%.3f ", h_output[i]);
        printf("\n");

        // Middle elements should have sum of all weights (2 channels * 3 kernel * 0.1)
        assert(fabsf(h_output[2] - 0.6f) < 0.01f);

        cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    }

    PASS("CUDA Kernels");
}

// ============================================================
// Test 4: Mel Spectrogram
// ============================================================
void test_mel_spectrogram() {
    TEST("Mel Spectrogram");

    MelConfig mc;
    mc.sample_rate = 16000;
    mc.n_fft = 400;
    mc.hop_length = 160;
    mc.n_mels = 80;
    mc.normalize = true;

    MelSpectrogram mel(mc);

    // Generate 1 second of test audio
    WavData wav;
    generate_test_audio(wav, 16000, 1.0f);

    int expected_frames = mel.get_num_frames(wav.num_samples);
    printf("  Expected frames: %d\n", expected_frames);

    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, expected_frames * mc.n_mels * sizeof(float)));

    int num_frames = mel.compute(wav.samples.data(), wav.num_samples, d_output);
    printf("  Actual frames: %d\n", num_frames);
    assert(num_frames == expected_frames);

    // Check output is not all zeros
    std::vector<float> h_output(num_frames * mc.n_mels);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                           h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float min_val = h_output[0], max_val = h_output[0];
    for (float v : h_output) {
        min_val = fminf(min_val, v);
        max_val = fmaxf(max_val, v);
    }
    printf("  Mel range: [%.3f, %.3f]\n", min_val, max_val);
    assert(max_val > min_val); // Ensure non-constant output

    cudaFree(d_output);
    PASS("Mel Spectrogram");
}

// ============================================================
// Test 5: Encoder Forward Pass
// ============================================================
void test_encoder() {
    TEST("Encoder Forward Pass");

    EncoderConfig ec;
    ec.d_model = 256;
    ec.n_heads = 4;
    ec.n_layers = 2;
    ec.ffn_dim = 512;
    ec.n_mels = 80;
    ec.max_seq_len = 200;

    WhisperEncoder encoder(ec);
    encoder.init_random_weights();

    // Create fake mel input [n_mels, num_frames] (channels-first for conv)
    int num_frames = 100;
    std::vector<float> h_mel(ec.n_mels * num_frames);
    for (float& v : h_mel) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    float* d_mel;
    CUDA_CHECK(cudaMalloc(&d_mel, h_mel.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mel, h_mel.data(), h_mel.size() * sizeof(float), cudaMemcpyHostToDevice));

    int out_len = encoder.get_output_length(num_frames);
    printf("  Input frames: %d, Expected output: %d\n", num_frames, out_len);

    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, out_len * ec.d_model * sizeof(float)));

    int actual_out;
    encoder.forward(d_mel, num_frames, d_output, actual_out);
    printf("  Actual output length: %d\n", actual_out);
    assert(actual_out == out_len);

    // Verify output is valid (no NaN/Inf)
    std::vector<float> h_output(out_len * ec.d_model);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                           h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool valid = true;
    for (float v : h_output) {
        if (isnan(v) || isinf(v)) { valid = false; break; }
    }
    printf("  Output valid (no NaN/Inf): %s\n", valid ? "yes" : "NO");
    assert(valid);

    cudaFree(d_mel);
    cudaFree(d_output);
    PASS("Encoder Forward Pass");
}

// ============================================================
// Test 6: Decoder Forward Pass
// ============================================================
void test_decoder() {
    TEST("Decoder Forward Pass");

    DecoderConfig dc;
    dc.d_model = 256;
    dc.n_heads = 4;
    dc.n_layers = 2;
    dc.ffn_dim = 512;
    dc.vocab_size = 52;
    dc.max_seq_len = 100;

    WhisperDecoder decoder(dc);
    decoder.init_random_weights();

    // Fake encoder output [enc_len, d_model]
    int enc_len = 50;
    std::vector<float> h_enc(enc_len * dc.d_model);
    for (float& v : h_enc) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    float* d_enc;
    CUDA_CHECK(cudaMalloc(&d_enc, h_enc.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_enc, h_enc.data(), h_enc.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Input tokens [BOS, token1, token2]
    int h_tokens[] = {1, 10, 15};
    int num_tokens = 3;
    int* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens, num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, num_tokens * dc.vocab_size * sizeof(float)));

    decoder.forward(d_tokens, num_tokens, d_enc, enc_len, d_logits);

    // Check last token logits
    std::vector<float> h_logits(dc.vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(),
                           d_logits + (num_tokens - 1) * dc.vocab_size,
                           dc.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    bool valid = true;
    for (float v : h_logits) {
        if (isnan(v) || isinf(v)) { valid = false; break; }
    }
    printf("  Logits valid: %s\n", valid ? "yes" : "NO");
    printf("  Logits sample [0..4]: %.3f %.3f %.3f %.3f %.3f\n",
           h_logits[0], h_logits[1], h_logits[2], h_logits[3], h_logits[4]);
    assert(valid);

    // Test single-step decoding
    decoder.reset_kv_cache();
    float* d_step_logits;
    CUDA_CHECK(cudaMalloc(&d_step_logits, dc.vocab_size * sizeof(float)));

    decoder.forward_step(1, 0, d_enc, enc_len, d_step_logits); // BOS
    decoder.forward_step(10, 1, d_enc, enc_len, d_step_logits); // token
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_step_logits,
                           dc.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    valid = true;
    for (float v : h_logits) {
        if (isnan(v) || isinf(v)) { valid = false; break; }
    }
    printf("  Step logits valid: %s\n", valid ? "yes" : "NO");
    assert(valid);

    cudaFree(d_enc); cudaFree(d_tokens); cudaFree(d_logits); cudaFree(d_step_logits);
    PASS("Decoder Forward Pass");
}

// ============================================================
// Test 7: Greedy Decoding
// ============================================================
void test_greedy_decode() {
    TEST("Greedy Decoding");

    DecoderConfig dc;
    dc.d_model = 256;
    dc.n_heads = 4;
    dc.n_layers = 2;
    dc.ffn_dim = 512;
    dc.vocab_size = 52;
    dc.max_seq_len = 100;

    WhisperDecoder decoder(dc);
    decoder.init_random_weights();

    BeamConfig bc;
    bc.beam_size = 1;
    bc.max_length = 20;
    bc.eos_token = 2;
    bc.bos_token = 1;
    BeamSearch beam(bc);

    // Fake encoder output
    int enc_len = 30;
    float* d_enc;
    CUDA_CHECK(cudaMalloc(&d_enc, enc_len * dc.d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_enc, 0, enc_len * dc.d_model * sizeof(float)));
    // Fill with small random values
    std::vector<float> h_enc(enc_len * dc.d_model);
    for (float& v : h_enc) v = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    CUDA_CHECK(cudaMemcpy(d_enc, h_enc.data(), h_enc.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto tokens = beam.greedy_decode(decoder, d_enc, enc_len);
    printf("  Generated %d tokens: [", (int)tokens.size());
    for (int t : tokens) printf("%d ", t);
    printf("]\n");

    assert(tokens.size() >= 1);
    assert(tokens[0] == bc.bos_token);

    CharTokenizer tok;
    std::string text = tok.decode(tokens);
    printf("  Decoded text: '%s'\n", text.c_str());

    cudaFree(d_enc);
    PASS("Greedy Decoding");
}

// ============================================================
// Test 8: Full Pipeline
// ============================================================
void test_pipeline() {
    TEST("Full Pipeline");

    WhisperConfig cfg;
    cfg.d_model = 256;
    cfg.n_heads = 4;
    cfg.encoder_layers = 2;
    cfg.decoder_layers = 2;
    cfg.ffn_dim = 512;
    cfg.use_greedy = true;
    cfg.beam_size = 1;

    WhisperPipeline pipeline(cfg);
    pipeline.init_random_weights();

    // Generate test audio
    WavData wav;
    generate_test_audio(wav, 16000, 2.0f);
    printf("  Audio: %.1f sec, %d samples\n",
           (float)wav.num_samples / wav.sample_rate, wav.num_samples);

    auto result = pipeline.transcribe(wav.samples.data(), wav.num_samples);

    printf("  Text: '%s'\n", result.text.c_str());
    printf("  Tokens: %d\n", result.num_tokens);
    printf("  Mel: %.2f ms\n", result.mel_time_ms);
    printf("  Encoder: %.2f ms\n", result.encoder_time_ms);
    printf("  Decoder: %.2f ms\n", result.decoder_time_ms);
    printf("  Total: %.2f ms\n", result.total_time_ms);

    assert(result.total_time_ms > 0);
    assert(result.num_tokens >= 1);

    PASS("Full Pipeline");
}

// ============================================================
// Test 9: Performance Benchmark
// ============================================================
void test_benchmark() {
    TEST("Performance Benchmark");

    WhisperConfig cfg;
    cfg.d_model = 512;
    cfg.n_heads = 8;
    cfg.encoder_layers = 4;
    cfg.decoder_layers = 4;
    cfg.ffn_dim = 2048;
    cfg.use_greedy = true;

    printf("  Model config: d=%d, h=%d, enc_L=%d, dec_L=%d, ff=%d\n",
           cfg.d_model, cfg.n_heads, cfg.encoder_layers, cfg.decoder_layers, cfg.ffn_dim);

    WhisperPipeline pipeline(cfg);
    pipeline.init_random_weights();

    // 3 second audio
    WavData wav;
    generate_speech_like_audio(wav, 16000, 3.0f);

    // Warmup
    printf("  Warming up...\n");
    pipeline.transcribe(wav.samples.data(), wav.num_samples);

    // Benchmark
    printf("  Running benchmark...\n");
    const int runs = 3;
    float total_mel = 0, total_enc = 0, total_dec = 0, total_all = 0;

    for (int i = 0; i < runs; i++) {
        auto result = pipeline.transcribe(wav.samples.data(), wav.num_samples);
        total_mel += result.mel_time_ms;
        total_enc += result.encoder_time_ms;
        total_dec += result.decoder_time_ms;
        total_all += result.total_time_ms;
    }

    printf("\n  === Benchmark Results (avg of %d runs) ===\n", runs);
    printf("  Audio duration: 3.0 sec\n");
    printf("  Mel spectrogram:  %.2f ms\n", total_mel / runs);
    printf("  Encoder:          %.2f ms\n", total_enc / runs);
    printf("  Decoder:          %.2f ms\n", total_dec / runs);
    printf("  Total:            %.2f ms\n", total_all / runs);
    printf("  Real-time factor: %.2fx\n", 3000.0f / (total_all / runs));

    PASS("Performance Benchmark");
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("==========================================\n");
    printf("  Whisper Speech Recognition CUDA Tests\n");
    printf("==========================================\n");
    print_gpu_info();

    test_wav_reader();
    test_tokenizer();
    test_kernels();
    test_mel_spectrogram();
    test_encoder();
    test_decoder();
    test_greedy_decode();
    test_pipeline();
    test_benchmark();

    printf("\n==========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("==========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
