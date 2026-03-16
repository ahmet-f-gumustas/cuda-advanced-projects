#include "pipeline.h"
#include <string>
#include <cstring>

struct InferenceArgs {
    std::string prompt = "a beautiful sunset over mountains";
    int num_steps = 20;
    float guidance_scale = 7.5f;
    int latent_h = 16;
    int latent_w = 16;
    unsigned long long seed = 42;
    std::string output_file = "output/generated.ppm";
    bool small_model = false;
};

void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --prompt TEXT      Text prompt (default: 'a beautiful sunset over mountains')\n");
    printf("  --steps N          Number of diffusion steps (default: 20)\n");
    printf("  --guidance F       Guidance scale (default: 7.5)\n");
    printf("  --latent-h N       Latent height (default: 16, output = 8x)\n");
    printf("  --latent-w N       Latent width (default: 16, output = 8x)\n");
    printf("  --seed N           Random seed (default: 42)\n");
    printf("  --output FILE      Output PPM file (default: output/generated.ppm)\n");
    printf("  --small            Use smaller model config\n");
    printf("  --help             Show this help\n");
}

InferenceArgs parseArgs(int argc, char** argv) {
    InferenceArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            args.num_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--guidance") == 0 && i + 1 < argc) {
            args.guidance_scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "--latent-h") == 0 && i + 1 < argc) {
            args.latent_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--latent-w") == 0 && i + 1 < argc) {
            args.latent_w = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (strcmp(argv[i], "--small") == 0) {
            args.small_model = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    InferenceArgs args = parseArgs(argc, argv);

    printf("============================================\n");
    printf("  Stable Diffusion CUDA Inference Engine\n");
    printf("============================================\n\n");

    printDeviceInfo();

    // Configure pipeline
    PipelineConfig cfg;

    if (args.small_model) {
        // Small model for testing
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

    cfg.latent_h = args.latent_h;
    cfg.latent_w = args.latent_w;
    cfg.guidance_scale = args.guidance_scale;
    cfg.seed = args.seed;

    int out_h = cfg.latent_h * 8;
    int out_w = cfg.latent_w * 8;

    printf("Configuration:\n");
    printf("  Prompt: \"%s\"\n", args.prompt.c_str());
    printf("  Steps: %d\n", args.num_steps);
    printf("  Guidance scale: %.1f\n", args.guidance_scale);
    printf("  Latent size: %dx%d\n", cfg.latent_h, cfg.latent_w);
    printf("  Output size: %dx%d\n", out_w, out_h);
    printf("  Seed: %llu\n", args.seed);
    printf("  Model: %s\n", args.small_model ? "small" : "standard");
    printf("\n");

    // Create and initialize pipeline
    printf("Initializing pipeline...\n");
    StableDiffusionPipeline pipeline(cfg);
    pipeline.initRandom(args.seed);
    printf("\n");

    // Generate
    printf("Generating image...\n");
    GenerationResult result = pipeline.generate(
        args.prompt, args.num_steps, args.guidance_scale);

    // Save output
    savePPM(args.output_file, result.image.data(), result.width, result.height);

    // Print summary
    printf("\n============================================\n");
    printf("  Generation Complete\n");
    printf("============================================\n");
    printf("  Output: %s (%dx%d)\n", args.output_file.c_str(), result.width, result.height);
    printf("  CLIP encoding:  %8.1f ms\n", result.clip_time_ms);
    printf("  Denoising:      %8.1f ms (%d steps, %.1f ms/step)\n",
           result.denoise_time_ms, result.num_steps,
           result.denoise_time_ms / result.num_steps);
    printf("  VAE decode:     %8.1f ms\n", result.vae_time_ms);
    printf("  Total:          %8.1f ms\n", result.total_time_ms);
    printf("============================================\n");

    return 0;
}
