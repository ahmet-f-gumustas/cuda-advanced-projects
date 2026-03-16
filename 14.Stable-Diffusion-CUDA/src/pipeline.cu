#include "pipeline.h"
#include "diffusion_kernels.cuh"

StableDiffusionPipeline::StableDiffusionPipeline(const PipelineConfig& cfg)
    : cfg_(cfg)
{
    clip_ = new CLIPEncoder(cfg.clip);
    unet_ = new UNet(cfg.unet);
    vae_ = new VAEDecoder(cfg.vae);
    scheduler_.init(cfg.scheduler);

    int lat_h = cfg.latent_h;
    int lat_w = cfg.latent_w;
    latent_size_ = cfg.unet.latent_channels * lat_h * lat_w;
    int out_h = lat_h * 8;
    int out_w = lat_w * 8;
    image_size_ = cfg.vae.output_channels * out_h * out_w;

    d_latent_ = cudaMallocDevice<half>(latent_size_);
    d_noise_pred_ = cudaMallocDevice<half>(latent_size_);
    d_noise_pred_uncond_ = cudaMallocDevice<half>(latent_size_);
    d_context_ = cudaMallocDevice<half>(cfg.clip.max_seq_len * cfg.clip.d_model);
    d_uncond_context_ = cudaMallocDevice<half>(cfg.clip.max_seq_len * cfg.clip.d_model);
    d_image_ = cudaMallocDevice<half>(image_size_);

    // Initialize curand states for noise generation
    int n_states = 1024;
    CUDA_CHECK(cudaMalloc(&d_rng_states_, n_states * sizeof(curandState)));
    init_curand_states_kernel<<<(n_states + 255) / 256, 256>>>(
        d_rng_states_, cfg.seed, n_states);
    CUDA_CHECK_LAST_ERROR();
}

StableDiffusionPipeline::~StableDiffusionPipeline() {
    delete clip_;
    delete unet_;
    delete vae_;
    cudaFree(d_latent_);
    cudaFree(d_noise_pred_);
    cudaFree(d_noise_pred_uncond_);
    cudaFree(d_context_);
    cudaFree(d_uncond_context_);
    cudaFree(d_image_);
    cudaFree(d_rng_states_);
}

void StableDiffusionPipeline::initRandom(unsigned long long seed) {
    clip_->initRandom(seed);
    unet_->initRandom(seed + 1000);
    vae_->initRandom(seed + 2000);
}

GenerationResult StableDiffusionPipeline::generate(
    const std::string& prompt,
    int num_inference_steps,
    float guidance_scale)
{
    GenerationResult result;
    CudaTimer total_timer, step_timer;
    total_timer.start();

    int lat_h = cfg_.latent_h;
    int lat_w = cfg_.latent_w;
    int out_h = lat_h * 8;
    int out_w = lat_w * 8;

    // 1. CLIP encode prompt
    step_timer.start();
    std::vector<int> token_ids;
    for (char c : prompt) token_ids.push_back((unsigned char)c);
    if ((int)token_ids.size() > cfg_.clip.max_seq_len)
        token_ids.resize(cfg_.clip.max_seq_len);
    int context_len = (int)token_ids.size();

    clip_->encode(token_ids, d_context_);

    // Unconditional embedding (empty prompt)
    std::vector<int> empty_tokens(context_len, 0);
    clip_->encode(empty_tokens, d_uncond_context_);

    step_timer.stop();
    result.clip_time_ms = step_timer.elapsed_ms();
    printf("[Pipeline] CLIP encoding: %.1f ms (seq_len=%d)\n",
           result.clip_time_ms, context_len);

    // 2. Generate initial random noise
    generate_gaussian_noise_kernel<<<(latent_size_ + 255) / 256, 256>>>(
        d_latent_, latent_size_, d_rng_states_);
    CUDA_CHECK_LAST_ERROR();

    // 3. Denoise loop
    step_timer.start();
    scheduler_.setTimesteps(num_inference_steps);

    printf("[Pipeline] Starting denoising (%d steps, guidance=%.1f)...\n",
           num_inference_steps, guidance_scale);

    for (int i = 0; i < num_inference_steps; i++) {
        int t = scheduler_.getTimestep(i);
        int t_prev = (i + 1 < num_inference_steps) ? scheduler_.getTimestep(i + 1) : -1;

        // Conditional prediction
        unet_->forward(d_latent_, d_context_, context_len, t,
                       d_noise_pred_, lat_h, lat_w);

        if (guidance_scale > 1.0f) {
            // Unconditional prediction
            unet_->forward(d_latent_, d_uncond_context_, context_len, t,
                           d_noise_pred_uncond_, lat_h, lat_w);

            // Classifier-free guidance:
            // noise_pred = uncond + guidance_scale * (cond - uncond)
            //            = (1 - guidance_scale) * uncond + guidance_scale * cond
            float a = guidance_scale;
            float b = 1.0f - guidance_scale;
            linear_combine_kernel<<<(latent_size_ + 255) / 256, 256>>>(
                d_noise_pred_,
                d_noise_pred_, a,
                d_noise_pred_uncond_, b,
                latent_size_);
            CUDA_CHECK_LAST_ERROR();
        }

        // DDIM step
        scheduler_.step(d_noise_pred_, t, t_prev, d_latent_, d_latent_, latent_size_);

        if ((i + 1) % 5 == 0 || i == 0) {
            printf("  Step %d/%d (t=%d)\n", i + 1, num_inference_steps, t);
        }
    }

    step_timer.stop();
    result.denoise_time_ms = step_timer.elapsed_ms();
    printf("[Pipeline] Denoising: %.1f ms (%.1f ms/step)\n",
           result.denoise_time_ms, result.denoise_time_ms / num_inference_steps);

    // 4. VAE decode
    step_timer.start();
    vae_->decode(d_latent_, d_image_, lat_h, lat_w);
    step_timer.stop();
    result.vae_time_ms = step_timer.elapsed_ms();
    printf("[Pipeline] VAE decode: %.1f ms\n", result.vae_time_ms);

    // 5. Copy result to host
    result.image.resize(image_size_);
    std::vector<half> h_image(image_size_);
    cudaMemcpyD2H(h_image.data(), d_image_, image_size_);
    for (int i = 0; i < image_size_; i++) {
        result.image[i] = __half2float(h_image[i]);
    }

    result.height = out_h;
    result.width = out_w;
    result.num_steps = num_inference_steps;

    total_timer.stop();
    result.total_time_ms = total_timer.elapsed_ms();
    printf("[Pipeline] Total: %.1f ms\n", result.total_time_ms);

    return result;
}

GenerationResult StableDiffusionPipeline::generateFromLatent(
    const half* d_latent,
    const half* d_context,
    int context_len,
    int num_inference_steps,
    float guidance_scale)
{
    GenerationResult result;
    CudaTimer total_timer, step_timer;
    total_timer.start();

    int lat_h = cfg_.latent_h;
    int lat_w = cfg_.latent_w;
    int out_h = lat_h * 8;
    int out_w = lat_w * 8;

    result.clip_time_ms = 0.0f;

    // Copy initial latent
    cudaMemcpyD2D(d_latent_, d_latent, latent_size_);

    // Encode empty prompt for unconditional
    std::vector<int> empty_tokens(context_len, 0);
    clip_->encode(empty_tokens, d_uncond_context_);
    cudaMemcpyD2D(d_context_, d_context, context_len * cfg_.clip.d_model);

    // Denoise
    step_timer.start();
    scheduler_.setTimesteps(num_inference_steps);

    for (int i = 0; i < num_inference_steps; i++) {
        int t = scheduler_.getTimestep(i);
        int t_prev = (i + 1 < num_inference_steps) ? scheduler_.getTimestep(i + 1) : -1;

        unet_->forward(d_latent_, d_context_, context_len, t,
                       d_noise_pred_, lat_h, lat_w);

        if (guidance_scale > 1.0f) {
            unet_->forward(d_latent_, d_uncond_context_, context_len, t,
                           d_noise_pred_uncond_, lat_h, lat_w);

            float a = guidance_scale;
            float b = 1.0f - guidance_scale;
            linear_combine_kernel<<<(latent_size_ + 255) / 256, 256>>>(
                d_noise_pred_,
                d_noise_pred_, a,
                d_noise_pred_uncond_, b,
                latent_size_);
            CUDA_CHECK_LAST_ERROR();
        }

        scheduler_.step(d_noise_pred_, t, t_prev, d_latent_, d_latent_, latent_size_);
    }

    step_timer.stop();
    result.denoise_time_ms = step_timer.elapsed_ms();

    // VAE decode
    step_timer.start();
    vae_->decode(d_latent_, d_image_, lat_h, lat_w);
    step_timer.stop();
    result.vae_time_ms = step_timer.elapsed_ms();

    // Copy to host
    result.image.resize(image_size_);
    std::vector<half> h_image(image_size_);
    cudaMemcpyD2H(h_image.data(), d_image_, image_size_);
    for (int i = 0; i < image_size_; i++) {
        result.image[i] = __half2float(h_image[i]);
    }

    result.height = out_h;
    result.width = out_w;
    result.num_steps = num_inference_steps;

    total_timer.stop();
    result.total_time_ms = total_timer.elapsed_ms();

    return result;
}

// ============================================================
// PPM image saving (simple, no external libs)
// ============================================================

void savePPM(const std::string& filename,
             const float* rgb_data,
             int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: cannot open %s for writing\n", filename.c_str());
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";

    // rgb_data is [3, H, W] in CHW format, values in [0, 1]
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float val = rgb_data[c * height * width + y * width + x];
                val = std::max(0.0f, std::min(1.0f, val));
                unsigned char byte = (unsigned char)(val * 255.0f + 0.5f);
                file.write((char*)&byte, 1);
            }
        }
    }

    file.close();
    printf("Saved image: %s (%dx%d)\n", filename.c_str(), width, height);
}
