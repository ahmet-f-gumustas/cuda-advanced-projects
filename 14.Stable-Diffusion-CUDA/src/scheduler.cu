#include "scheduler.h"
#include "diffusion_kernels.cuh"
#include <cmath>
#include <algorithm>

DDIMScheduler::DDIMScheduler() : num_inference_steps_(20) {}

DDIMScheduler::~DDIMScheduler() {}

void DDIMScheduler::init(const SchedulerConfig& cfg) {
    cfg_ = cfg;

    // Scaled linear beta schedule (used in Stable Diffusion)
    betas_.resize(cfg_.num_train_timesteps);
    float sqrt_beta_start = sqrtf(cfg_.beta_start);
    float sqrt_beta_end = sqrtf(cfg_.beta_end);
    for (int i = 0; i < cfg_.num_train_timesteps; i++) {
        float t = (float)i / (float)(cfg_.num_train_timesteps - 1);
        float sqrt_beta = sqrt_beta_start + t * (sqrt_beta_end - sqrt_beta_start);
        betas_[i] = sqrt_beta * sqrt_beta;
    }

    // Compute alphas and cumulative product
    alphas_.resize(cfg_.num_train_timesteps);
    alpha_cumprod_.resize(cfg_.num_train_timesteps);
    for (int i = 0; i < cfg_.num_train_timesteps; i++) {
        alphas_[i] = 1.0f - betas_[i];
    }
    alpha_cumprod_[0] = alphas_[0];
    for (int i = 1; i < cfg_.num_train_timesteps; i++) {
        alpha_cumprod_[i] = alpha_cumprod_[i - 1] * alphas_[i];
    }

    setTimesteps(cfg_.num_inference_steps);
}

void DDIMScheduler::setTimesteps(int num_inference_steps) {
    num_inference_steps_ = num_inference_steps;
    timesteps_.resize(num_inference_steps);

    // Evenly spaced timesteps in reverse order
    float step_ratio = (float)cfg_.num_train_timesteps / (float)num_inference_steps;
    for (int i = 0; i < num_inference_steps; i++) {
        timesteps_[i] = (int)((num_inference_steps - 1 - i) * step_ratio);
    }
}

int DDIMScheduler::getTimestep(int step_index) const {
    return timesteps_[step_index];
}

int DDIMScheduler::numSteps() const {
    return num_inference_steps_;
}

void DDIMScheduler::step(
    const half* d_model_output,
    int timestep, int prev_timestep,
    const half* d_sample,
    half* d_prev_sample,
    int size)
{
    // DDIM step formula:
    // alpha_bar_t = alpha_cumprod[t]
    // alpha_bar_prev = alpha_cumprod[t-1] (or 1.0 if t-1 < 0)
    //
    // Predicted x_0:
    //   x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    //
    // DDIM formula (eta = 0, deterministic):
    //   x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred
    //           + sqrt(1 - alpha_bar_{t-1}) * eps_pred

    float alpha_bar_t = alpha_cumprod_[timestep];
    float alpha_bar_prev = (prev_timestep >= 0) ? alpha_cumprod_[prev_timestep] : 1.0f;

    float sqrt_alpha_bar_t = sqrtf(alpha_bar_t);
    float sqrt_one_minus_alpha_bar_t = sqrtf(1.0f - alpha_bar_t);
    float sqrt_alpha_bar_prev = sqrtf(alpha_bar_prev);
    float sqrt_one_minus_alpha_bar_prev = sqrtf(1.0f - alpha_bar_prev);

    // Predicted x_0 coefficient: 1 / sqrt(alpha_bar_t)
    // Noise removal coefficient: -sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t)
    // Combined into x_{t-1}:
    //   x_{t-1} = sqrt(alpha_bar_prev) / sqrt(alpha_bar_t) * x_t
    //           + (sqrt(1-alpha_bar_prev) - sqrt(alpha_bar_prev) * sqrt(1-alpha_bar_t) / sqrt(alpha_bar_t)) * eps

    // Simpler: compute x_0_pred first, then combine
    // a = sqrt(alpha_bar_prev) / sqrt(alpha_bar_t)  (coefficient for x_t in x_0 direction)
    // b = sqrt(1-alpha_bar_prev) - a * sqrt(1-alpha_bar_t)  (coefficient for eps)

    // Actually let's use the standard formulation:
    // x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    // x_{t-1} = sqrt(alpha_bar_prev) * x_0_pred + sqrt(1 - alpha_bar_prev) * eps

    // This translates to:
    // x_{t-1} = (sqrt(alpha_bar_prev) / sqrt(alpha_bar_t)) * x_t
    //         + (sqrt(1 - alpha_bar_prev) - sqrt(alpha_bar_prev) * sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t)) * eps

    float coeff_x = sqrt_alpha_bar_prev / sqrt_alpha_bar_t;
    float coeff_eps = sqrt_one_minus_alpha_bar_prev
                    - sqrt_alpha_bar_prev * sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // d_prev_sample = coeff_x * d_sample + coeff_eps * d_model_output
    linear_combine_kernel<<<blocks, threads>>>(
        d_prev_sample,
        d_sample, coeff_x,
        d_model_output, coeff_eps,
        size);
    CUDA_CHECK_LAST_ERROR();
}

void DDIMScheduler::addNoise(
    const half* d_original,
    const half* d_noise,
    int timestep,
    half* d_noisy,
    int size)
{
    float alpha_bar = alpha_cumprod_[timestep];
    float sqrt_alpha_bar = sqrtf(alpha_bar);
    float sqrt_one_minus_alpha_bar = sqrtf(1.0f - alpha_bar);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    linear_combine_kernel<<<blocks, threads>>>(
        d_noisy,
        d_original, sqrt_alpha_bar,
        d_noise, sqrt_one_minus_alpha_bar,
        size);
    CUDA_CHECK_LAST_ERROR();
}

void DDIMScheduler::scaleModelInput(half* d_sample, int timestep, int size) {
    // DDIM: model input = sample (no scaling needed)
    (void)d_sample;
    (void)timestep;
    (void)size;
}
