#include "../include/transformer.h"
#include "../include/transformer_kernels.cuh"
#include "../include/kv_cache.h"
#include "../include/tokenizer.h"
#include "../include/cuda_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <chrono>

// ============================================================================
// Inference: interactive text generation
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model     PATH   Weight file (default: data/weights.bin)\n"
              << "  --prompt    TEXT   Input prompt (default: interactive mode)\n"
              << "  --max-tokens N     Max tokens to generate (default: 200)\n"
              << "  --layers    N      Layers (default: 6, must match model)\n"
              << "  --d-model   N      Model dim (default: 512)\n"
              << "  --n-heads   N      Query heads (default: 8)\n"
              << "  --n-kv-heads N     KV heads (default: 2)\n"
              << "  --ff-dim    N      FFN dim (default: 2048)\n"
              << "  --vocab     N      Vocab size (default: 256)\n"
              << "  --mode      MODE   fp16 | int8 (default: fp16)\n"
              << "  --sampling  STRAT  greedy | top-k (default: greedy)\n"
              << "  --top-k     N      K for top-k sampling (default: 40)\n"
              << "  --temp      F      Temperature (default: 0.8)\n"
              << "  --seed      N      Random seed (default: 42)\n"
              << "  --save      PATH   Save random weights to file\n"
              << "  --load-random      Use random weights (don't load file)\n";
}

struct InferenceConfig {
    std::string model_path   = "data/weights.bin";
    std::string prompt       = "";
    int         max_tokens   = 200;
    int         n_layers     = 6;
    int         d_model      = 512;
    int         n_heads      = 8;
    int         n_kv_heads   = 2;
    int         ff_dim       = 2048;
    int         vocab_size   = 256;
    std::string mode         = "fp16";
    std::string sampling     = "greedy";
    int         top_k        = 40;
    float       temperature  = 0.8f;
    unsigned    seed         = 42;
    std::string save_path    = "";
    bool        load_random  = false;
};

// Generate tokens given a model, KV cache, logit buffer, and tokenizer config
std::vector<int> generate(TransformerModel& model,
                           KVCache&          kv,
                           const std::vector<int>& prompt_ids,
                           int max_new_tokens,
                           const std::string& sampling,
                           int top_k, float temperature,
                           int eos_id,
                           bool stream_output)
{
    // Prefill
    model.prefill(prompt_ids, kv);

    float*       d_logits  = cudaMallocDevice<float>(model.config().vocab_size);
    int*         d_token   = cudaMallocDevice<int>(1);
    float*       d_scratch = cudaMallocDevice<float>(1);
    curandState* d_rng     = cudaMallocDevice<curandState>(1);
    init_curand_state_kernel<<<1, 1>>>(d_rng, 42ULL, 1);
    CUDA_CHECK_LAST_ERROR();

    std::vector<int> output;
    int cur_token = prompt_ids.empty() ? 1 : prompt_ids.back();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < max_new_tokens; ++t) {
        model.forward(cur_token, d_logits, kv, kv.current_pos);
        kv.current_pos++;

        if (sampling == "top-k") {
            top_k_sampling_kernel<<<1, 1>>>(
                d_logits, d_token, model.config().vocab_size,
                top_k, temperature, d_rng);
        } else {
            // Greedy
            int smem = PT_BLOCK_SIZE * 2 * sizeof(float);
            argmax_kernel<<<1, PT_BLOCK_SIZE, smem>>>(
                d_logits, d_token, d_scratch, model.config().vocab_size);
        }
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        int next_token;
        CUDA_CHECK(cudaMemcpy(&next_token, d_token, sizeof(int), cudaMemcpyDeviceToHost));

        if (next_token == eos_id) break;

        output.push_back(next_token);
        cur_token = next_token;

        if (stream_output && next_token > 0 && next_token < 256) {
            std::cout << static_cast<char>(next_token) << std::flush;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    float tps = (float)output.size() / (elapsed_ms / 1000.f);

    if (stream_output) std::cout << "\n";
    std::cerr << "[Inference] Generated " << output.size()
              << " tokens in " << elapsed_ms << " ms ("
              << tps << " tokens/sec)\n";

    cudaFree(d_logits);
    cudaFree(d_token);
    cudaFree(d_scratch);
    cudaFree(d_rng);

    return output;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    InferenceConfig ic;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
#define STRARG(flag, field) \
        else if (std::strcmp(argv[i], flag) == 0 && i+1 < argc) { ic.field = argv[++i]; }
#define INTARG(flag, field) \
        else if (std::strcmp(argv[i], flag) == 0 && i+1 < argc) { ic.field = std::atoi(argv[++i]); }
#define FLTARG(flag, field) \
        else if (std::strcmp(argv[i], flag) == 0 && i+1 < argc) { ic.field = std::atof(argv[++i]); }

        STRARG("--model",     model_path)
        STRARG("--prompt",    prompt)
        INTARG("--max-tokens",max_tokens)
        INTARG("--layers",    n_layers)
        INTARG("--d-model",   d_model)
        INTARG("--n-heads",   n_heads)
        INTARG("--n-kv-heads",n_kv_heads)
        INTARG("--ff-dim",    ff_dim)
        INTARG("--vocab",     vocab_size)
        STRARG("--mode",      mode)
        STRARG("--sampling",  sampling)
        INTARG("--top-k",     top_k)
        FLTARG("--temp",      temperature)
        INTARG("--seed",      seed)
        STRARG("--save",      save_path)
        else if (std::strcmp(argv[i], "--load-random") == 0) ic.load_random = true;
#undef STRARG
#undef INTARG
#undef FLTARG
    }

    // Build config
    TransformerConfig cfg;
    cfg.n_layers   = ic.n_layers;
    cfg.d_model    = ic.d_model;
    cfg.n_heads    = ic.n_heads;
    cfg.n_kv_heads = ic.n_kv_heads;
    cfg.head_dim   = ic.d_model / ic.n_heads;
    cfg.ff_dim     = ic.ff_dim;
    cfg.vocab_size = ic.vocab_size;
    cfg.max_seq_len = 2048;
    cfg.quant      = (ic.mode == "int8") ? QuantMode::INT8 : QuantMode::FP16;

    printDeviceInfo();

    std::cout << "\n=== GPT Transformer Inference ===\n";
    std::cout << "  layers=" << cfg.n_layers << " d_model=" << cfg.d_model
              << " n_heads=" << cfg.n_heads << " n_kv_heads=" << cfg.n_kv_heads << " (GQA)\n";
    std::cout << "  ff_dim=" << cfg.ff_dim << " vocab=" << cfg.vocab_size
              << " quant=" << ic.mode << " sampling=" << ic.sampling << "\n\n";

    TransformerModel model(cfg);

    if (ic.load_random || !std::ifstream(ic.model_path).is_open()) {
        std::cout << "[*] Initializing with random weights (seed=" << ic.seed << ")\n";
        model.initRandom(ic.seed);
        if (!ic.save_path.empty()) {
            model.saveWeights(ic.save_path);
        }
    } else {
        if (!model.loadWeights(ic.model_path)) {
            std::cerr << "[!] Failed to load weights. Using random.\n";
            model.initRandom(ic.seed);
        }
    }

    CharTokenizer tokenizer;
    KVCache kv = KVCache::create(cfg.n_layers, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim);

    std::cout << "\nKV Cache: " << kv.bytes() / (1024 * 1024) << " MB\n";

    // ── Interactive or single-prompt mode ────────────────────────────────
    if (ic.prompt.empty()) {
        // Interactive mode
        std::cout << "\nInteractive mode — type a prompt and press Enter (Ctrl+C to quit)\n";
        std::string line;
        while (true) {
            std::cout << "\nPrompt> " << std::flush;
            if (!std::getline(std::cin, line) || line.empty()) break;

            kv.reset();
            auto prompt_ids = tokenizer.encode(line);

            std::cout << "Output: " << std::flush;
            generate(model, kv, prompt_ids, ic.max_tokens, ic.sampling,
                     ic.top_k, ic.temperature, tokenizer.eos_id(), true);
        }
    } else {
        // Single prompt mode
        auto prompt_ids = tokenizer.encode(ic.prompt);
        std::cout << "Prompt (" << prompt_ids.size() << " tokens): " << ic.prompt << "\n";
        std::cout << "Output: " << std::flush;

        auto output_ids = generate(model, kv, prompt_ids, ic.max_tokens, ic.sampling,
                                    ic.top_k, ic.temperature, tokenizer.eos_id(), true);
        std::string output = tokenizer.decode(output_ids);
        (void)output; // already streamed above
    }

    kv.destroy();
    return 0;
}
