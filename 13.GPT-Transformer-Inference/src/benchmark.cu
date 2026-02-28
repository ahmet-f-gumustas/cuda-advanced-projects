#include "../include/transformer.h"
#include "../include/transformer_kernels.cuh"
#include "../include/kv_cache.h"
#include "../include/cuda_utils.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>

// ============================================================================
// Benchmark: measures tokens/sec for each quantization mode
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --layers   N     Number of decoder layers (default: 6)\n"
              << "  --d-model  N     Model dimension (default: 512)\n"
              << "  --n-heads  N     Query head count (default: 8)\n"
              << "  --n-kv-heads N   KV head count (default: 2)\n"
              << "  --ff-dim   N     FFN dimension (default: 2048)\n"
              << "  --vocab    N     Vocabulary size (default: 256)\n"
              << "  --seq-len  N     Prompt length (default: 128)\n"
              << "  --tokens   N     Tokens to generate (default: 100)\n"
              << "  --warmup   N     Warmup iterations (default: 3)\n"
              << "  --mode     MODE  fp16 | int8 | all (default: all)\n"
              << "  --model    PATH  Weight file (optional; random weights if omitted)\n"
              << "  --seed     N     Random seed (default: 42)\n";
}

struct BenchmarkConfig {
    int         n_layers   = 6;
    int         d_model    = 512;
    int         n_heads    = 8;
    int         n_kv_heads = 2;
    int         ff_dim     = 2048;
    int         vocab_size = 256;
    int         seq_len    = 128;
    int         n_tokens   = 100;
    int         warmup     = 3;
    std::string mode       = "all";
    std::string model_path = "";
    unsigned    seed       = 42;
};

struct BenchResult {
    std::string name;
    float       ms_per_token;
    float       tokens_per_sec;
    float       speedup;    // vs FP16 baseline
};

// Run one benchmark pass: prefill seq_len tokens, then generate n_tokens
BenchResult run_benchmark(const BenchmarkConfig& bc, QuantMode quant, const std::string& name)
{
    TransformerConfig cfg;
    cfg.n_layers   = bc.n_layers;
    cfg.d_model    = bc.d_model;
    cfg.n_heads    = bc.n_heads;
    cfg.n_kv_heads = bc.n_kv_heads;
    cfg.head_dim   = bc.d_model / bc.n_heads;
    cfg.ff_dim     = bc.ff_dim;
    cfg.vocab_size = bc.vocab_size;
    cfg.max_seq_len = 2048;
    cfg.quant      = quant;

    TransformerModel model(cfg);
    if (bc.model_path.empty()) {
        model.initRandom(bc.seed);
    } else {
        if (!model.loadWeights(bc.model_path)) {
            std::cerr << "[Benchmark] Failed to load weights!" << std::endl;
            return {name, 0.f, 0.f, 0.f};
        }
    }

    KVCache kv_cache = KVCache::create(cfg.n_layers, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim);
    float* d_logits  = cudaMallocDevice<float>(cfg.vocab_size);
    int*   d_token   = cudaMallocDevice<int>(1);
    float* d_scratch = cudaMallocDevice<float>(1);

    // Build prompt (cyclic char tokens)
    std::vector<int> prompt(bc.seq_len);
    for (int i = 0; i < bc.seq_len; ++i) prompt[i] = (i % 26) + 'a';

    // Prefill (not timed)
    model.prefill(prompt, kv_cache);

    // Warmup
    int cur_token = prompt.back();
    for (int w = 0; w < bc.warmup; ++w) {
        model.forward(cur_token, d_logits, kv_cache, kv_cache.current_pos);
        // Don't advance cache during warmup — reset
        kv_cache.current_pos = bc.seq_len;
    }

    // Timed generation
    CudaTimer timer;
    kv_cache.current_pos = bc.seq_len;
    cur_token = prompt.back();

    timer.start();
    for (int t = 0; t < bc.n_tokens; ++t) {
        model.forward(cur_token, d_logits, kv_cache, kv_cache.current_pos);
        // Greedy sample
        int smem = PT_BLOCK_SIZE * 2 * sizeof(float);
        argmax_kernel<<<1, PT_BLOCK_SIZE, smem>>>(d_logits, d_token, d_scratch, cfg.vocab_size);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&cur_token, d_token, sizeof(int), cudaMemcpyDeviceToHost));
        kv_cache.current_pos++;
    }
    timer.stop();

    float total_ms    = timer.elapsed();
    float ms_per_tok  = total_ms / (float)bc.n_tokens;
    float toks_per_sec = 1000.0f / ms_per_tok;

    cudaFree(d_logits);
    cudaFree(d_token);
    cudaFree(d_scratch);
    kv_cache.destroy();

    return {name, ms_per_tok, toks_per_sec, 0.f};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    BenchmarkConfig bc;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
#define INTARG(flag, field) \
        else if (std::strcmp(argv[i], flag) == 0 && i+1 < argc) { bc.field = std::atoi(argv[++i]); }
        INTARG("--layers",    n_layers)
        INTARG("--d-model",   d_model)
        INTARG("--n-heads",   n_heads)
        INTARG("--n-kv-heads",n_kv_heads)
        INTARG("--ff-dim",    ff_dim)
        INTARG("--vocab",     vocab_size)
        INTARG("--seq-len",   seq_len)
        INTARG("--tokens",    n_tokens)
        INTARG("--warmup",    warmup)
        INTARG("--seed",      seed)
#undef INTARG
        else if (std::strcmp(argv[i], "--mode") == 0 && i+1 < argc)
            bc.mode = argv[++i];
        else if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc)
            bc.model_path = argv[++i];
    }

    printDeviceInfo();

    std::cout << "\n========================================\n";
    std::cout << "  GPT Transformer Inference Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "  Config: layers=" << bc.n_layers
              << " d_model=" << bc.d_model
              << " n_heads=" << bc.n_heads
              << " n_kv_heads=" << bc.n_kv_heads << " (GQA)\n";
    std::cout << "  ff_dim=" << bc.ff_dim
              << " vocab=" << bc.vocab_size
              << " prompt_len=" << bc.seq_len
              << " generate=" << bc.n_tokens << " tokens\n";
    std::cout << "  Warmup=" << bc.warmup << " iterations\n\n";

    std::vector<BenchResult> results;

    if (bc.mode == "fp16" || bc.mode == "all") {
        std::cout << "[*] Running FP16 benchmark..." << std::flush;
        auto r = run_benchmark(bc, QuantMode::FP16, "FP16");
        std::cout << " done\n";
        results.push_back(r);
    }

    if (bc.mode == "int8" || bc.mode == "all") {
        std::cout << "[*] Running INT8 benchmark..." << std::flush;
        auto r = run_benchmark(bc, QuantMode::INT8, "INT8");
        std::cout << " done\n";
        results.push_back(r);
    }

    // Compute speedup relative to first result
    float baseline = (results.empty()) ? 1.0f : results[0].ms_per_token;
    for (auto& r : results) r.speedup = baseline / r.ms_per_token;

    // Print table
    std::cout << "\n";
    std::cout << std::left << std::setw(8)  << "Mode"
              << std::right << std::setw(18) << "Tokens/sec"
              << std::setw(16) << "ms/token"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(54, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left  << std::setw(8)  << r.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1)
                  << r.tokens_per_sec << " tok/s"
                  << std::setw(12) << std::setprecision(2) << r.ms_per_token << " ms"
                  << std::setw(10) << std::setprecision(2) << r.speedup << "x"
                  << "\n";
    }
    std::cout << "\n";

    return 0;
}
