#include "pipeline.h"
#include <cstring>

struct BenchConfig {
    int latent_h = 16;
    int latent_w = 16;
    int num_steps = 10;
    int warmup_runs = 1;
    int bench_runs = 3;
    bool small_model = false;
    float guidance_scale = 7.5f;
};

BenchConfig parseBenchArgs(int argc, char** argv) {
    BenchConfig bc;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--latent-h") == 0 && i + 1 < argc) bc.latent_h = atoi(argv[++i]);
        else if (strcmp(argv[i], "--latent-w") == 0 && i + 1 < argc) bc.latent_w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) bc.num_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) bc.warmup_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) bc.bench_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--small") == 0) bc.small_model = true;
        else if (strcmp(argv[i], "--no-cfg") == 0) bc.guidance_scale = 1.0f;
    }
    return bc;
}

void runBenchmark(const BenchConfig& bc, const char* label) {
    PipelineConfig cfg;

    if (bc.small_model) {
        cfg.clip.d_model = 256;
        cfg.clip.n_layers = 2;
        cfg.clip.n_heads = 4;
        cfg.clip.ff_dim = 1024;

        cfg.unet.base_channels = 64;
        cfg.unet.channel_mult[0] = 1;
        cfg.unet.channel_mult[1] = 2;
        cfg.unet.channel_mult[2] = 4;
        cfg.unet.n_heads = 4;
        cfg.unet.context_dim = 256;
        cfg.unet.time_embed_dim = 256;
        cfg.unet.num_groups = 16;

        cfg.vae.base_channels = 64;
        cfg.vae.channels[0] = 256;
        cfg.vae.channels[1] = 256;
        cfg.vae.channels[2] = 128;
        cfg.vae.channels[3] = 64;
        cfg.vae.num_groups = 16;
    }

    cfg.latent_h = bc.latent_h;
    cfg.latent_w = bc.latent_w;
    cfg.guidance_scale = bc.guidance_scale;

    int out_h = cfg.latent_h * 8;
    int out_w = cfg.latent_w * 8;

    printf("\n--- %s ---\n", label);
    printf("  Latent: %dx%d, Output: %dx%d\n", cfg.latent_h, cfg.latent_w, out_w, out_h);
    printf("  Steps: %d, Guidance: %.1f\n", bc.num_steps, bc.guidance_scale);
    printf("  Model: %s\n", bc.small_model ? "small" : "standard");

    StableDiffusionPipeline pipeline(cfg);
    pipeline.initRandom(42);

    std::string prompt = "a cat sitting on a windowsill";

    // Warmup
    printf("  Warming up (%d runs)...\n", bc.warmup_runs);
    for (int i = 0; i < bc.warmup_runs; i++) {
        pipeline.generate(prompt, bc.num_steps, bc.guidance_scale);
    }

    // Benchmark
    float total_clip = 0, total_denoise = 0, total_vae = 0, total_all = 0;
    printf("  Benchmarking (%d runs)...\n", bc.bench_runs);
    for (int i = 0; i < bc.bench_runs; i++) {
        auto result = pipeline.generate(prompt, bc.num_steps, bc.guidance_scale);
        total_clip += result.clip_time_ms;
        total_denoise += result.denoise_time_ms;
        total_vae += result.vae_time_ms;
        total_all += result.total_time_ms;
    }

    float n = (float)bc.bench_runs;
    printf("\n  === Results (avg of %d runs) ===\n", bc.bench_runs);
    printf("  %-20s %8.1f ms\n", "CLIP encoding:", total_clip / n);
    printf("  %-20s %8.1f ms (%.1f ms/step)\n", "Denoising:",
           total_denoise / n, total_denoise / n / bc.num_steps);
    printf("  %-20s %8.1f ms\n", "VAE decode:", total_vae / n);
    printf("  %-20s %8.1f ms\n", "Total:", total_all / n);

    int unet_calls = bc.num_steps * (bc.guidance_scale > 1.0f ? 2 : 1);
    printf("  %-20s %8.1f ms\n", "Per UNet forward:", total_denoise / n / unet_calls);
    printf("  %-20s %8d px\n", "Output pixels:", out_h * out_w);
    printf("  %-20s %8.1f Kpx/s\n", "Throughput:",
           (out_h * out_w) / (total_all / n) * 1000.0f / 1000.0f);
}

int main(int argc, char** argv) {
    printf("============================================\n");
    printf("  Stable Diffusion CUDA Benchmark\n");
    printf("============================================\n");

    printDeviceInfo();

    BenchConfig bc = parseBenchArgs(argc, argv);

    // Run benchmark
    runBenchmark(bc, "FP16 Stable Diffusion");

    // If not small model, also run small model for comparison
    if (!bc.small_model) {
        BenchConfig small_bc = bc;
        small_bc.small_model = true;
        runBenchmark(small_bc, "FP16 Stable Diffusion (Small Model)");
    }

    // Run without classifier-free guidance for comparison
    BenchConfig no_cfg_bc = bc;
    no_cfg_bc.guidance_scale = 1.0f;
    runBenchmark(no_cfg_bc, "FP16 SD (No Guidance)");

    printf("\n============================================\n");
    printf("  Benchmark Complete\n");
    printf("============================================\n");

    return 0;
}
