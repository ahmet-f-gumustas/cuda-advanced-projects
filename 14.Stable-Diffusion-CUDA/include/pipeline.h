#ifndef PIPELINE_H
#define PIPELINE_H

#include "cuda_utils.h"
#include "clip.h"
#include "unet.h"
#include "vae.h"
#include "scheduler.h"

// ============================================================
// Stable Diffusion Pipeline
// Text -> CLIP encode -> UNet denoise loop -> VAE decode -> Image
// ============================================================

struct PipelineConfig {
    CLIPConfig clip;
    UNetConfig unet;
    VAEConfig vae;
    SchedulerConfig scheduler;

    int latent_h = 32;          // Latent spatial height
    int latent_w = 32;          // Latent spatial width
    float guidance_scale = 7.5f; // Classifier-free guidance scale
    unsigned long long seed = 42;
};

struct GenerationResult {
    std::vector<float> image;   // RGB [3, H, W] in [0, 1]
    int height, width;
    float total_time_ms;
    float clip_time_ms;
    float denoise_time_ms;
    float vae_time_ms;
    int num_steps;
};

class StableDiffusionPipeline {
public:
    StableDiffusionPipeline(const PipelineConfig& cfg);
    ~StableDiffusionPipeline();

    void initRandom(unsigned long long seed = 42);

    // Generate image from text prompt
    GenerationResult generate(const std::string& prompt,
                              int num_inference_steps = 20,
                              float guidance_scale = 7.5f);

    // Generate with pre-set random latent (for benchmarking)
    GenerationResult generateFromLatent(const half* d_latent,
                                        const half* d_context,
                                        int context_len,
                                        int num_inference_steps = 20,
                                        float guidance_scale = 7.5f);

    const PipelineConfig& config() const { return cfg_; }

    // Access sub-components
    CLIPEncoder& clip() { return *clip_; }
    UNet& unet() { return *unet_; }
    VAEDecoder& vae() { return *vae_; }
    DDIMScheduler& scheduler() { return scheduler_; }

private:
    PipelineConfig cfg_;

    CLIPEncoder* clip_;
    UNet* unet_;
    VAEDecoder* vae_;
    DDIMScheduler scheduler_;

    // Device buffers for pipeline
    half* d_latent_;            // [1, 4, latent_h, latent_w]
    half* d_noise_pred_;        // [1, 4, latent_h, latent_w]
    half* d_noise_pred_uncond_; // [1, 4, latent_h, latent_w]
    half* d_context_;           // [seq_len, context_dim]
    half* d_uncond_context_;    // [seq_len, context_dim] (empty prompt)
    half* d_image_;             // [1, 3, out_h, out_w]

    curandState* d_rng_states_;
    int latent_size_;           // Total elements in latent
    int image_size_;            // Total elements in output image
};

// Save image to PPM file (simple format, no external libs)
void savePPM(const std::string& filename,
             const float* rgb_data,
             int width, int height);

#endif // PIPELINE_H
