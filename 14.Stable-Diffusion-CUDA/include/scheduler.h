#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "cuda_utils.h"

// ============================================================
// DDIM Noise Scheduler
// Implements Denoising Diffusion Implicit Models scheduling
// ============================================================

struct SchedulerConfig {
    int num_train_timesteps = 1000;
    int num_inference_steps = 20;
    float beta_start = 0.00085f;
    float beta_end = 0.012f;
};

class DDIMScheduler {
public:
    DDIMScheduler();
    ~DDIMScheduler();

    void init(const SchedulerConfig& cfg);

    // Set number of inference steps and compute timestep schedule
    void setTimesteps(int num_inference_steps);

    // Get timestep at step index
    int getTimestep(int step_index) const;

    // Get total number of inference steps
    int numSteps() const;

    // DDIM step: given model output (predicted noise) and current sample,
    // compute the previous sample x_{t-1}
    // d_sample: [size] current noisy sample x_t (FP16)
    // d_model_output: [size] predicted noise eps_theta (FP16)
    // d_prev_sample: [size] output x_{t-1} (FP16)
    void step(const half* d_model_output,
              int timestep, int prev_timestep,
              const half* d_sample,
              half* d_prev_sample,
              int size);

    // Add noise to clean sample: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    void addNoise(const half* d_original,
                  const half* d_noise,
                  int timestep,
                  half* d_noisy,
                  int size);

    // Scale model input (for DDIM, just returns input unchanged, but needed for API compatibility)
    void scaleModelInput(half* d_sample, int timestep, int size);

private:
    SchedulerConfig cfg_;

    // Host arrays
    std::vector<float> betas_;           // [num_train_timesteps]
    std::vector<float> alphas_;          // [num_train_timesteps]
    std::vector<float> alpha_cumprod_;   // [num_train_timesteps]
    std::vector<int> timesteps_;         // [num_inference_steps] selected timesteps

    int num_inference_steps_;
};

#endif // SCHEDULER_H
