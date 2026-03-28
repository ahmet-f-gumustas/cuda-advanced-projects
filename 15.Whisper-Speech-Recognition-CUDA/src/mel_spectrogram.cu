#include "mel_spectrogram.h"
#include "whisper_kernels.cuh"
#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

MelSpectrogram::MelSpectrogram(const MelConfig& config)
    : config_(config), plan_created_(false),
      d_frames_(nullptr), d_fft_(nullptr), d_power_(nullptr), d_mel_filters_(nullptr),
      max_frames_(0) {
    freq_bins_ = config_.n_fft / 2 + 1;

    // Pre-allocate for ~30 seconds of audio
    max_frames_ = (config_.sample_rate * 30) / config_.hop_length + 1;

    CUDA_CHECK(cudaMalloc(&d_frames_, max_frames_ * config_.n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_, max_frames_ * freq_bins_ * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_power_, max_frames_ * freq_bins_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mel_filters_, config_.n_mels * freq_bins_ * sizeof(float)));

    build_mel_filterbank();
}

MelSpectrogram::~MelSpectrogram() {
    if (d_frames_) cudaFree(d_frames_);
    if (d_fft_) cudaFree(d_fft_);
    if (d_power_) cudaFree(d_power_);
    if (d_mel_filters_) cudaFree(d_mel_filters_);
    if (plan_created_) cufftDestroy(fft_plan_);
}

void MelSpectrogram::build_mel_filterbank() {
    float mel_min = hz_to_mel(config_.fmin);
    float mel_max = hz_to_mel(config_.fmax);

    // Create n_mels + 2 evenly spaced mel points
    std::vector<float> mel_points(config_.n_mels + 2);
    for (int i = 0; i < config_.n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (config_.n_mels + 1);
    }

    // Convert back to Hz
    std::vector<float> hz_points(config_.n_mels + 2);
    for (int i = 0; i < config_.n_mels + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Convert to FFT bin indices
    std::vector<int> bin_points(config_.n_mels + 2);
    for (int i = 0; i < config_.n_mels + 2; i++) {
        bin_points[i] = (int)floorf((config_.n_fft + 1) * hz_points[i] / config_.sample_rate);
    }

    // Build triangular filters
    std::vector<float> filters(config_.n_mels * freq_bins_, 0.0f);
    for (int m = 0; m < config_.n_mels; m++) {
        int f_left = bin_points[m];
        int f_center = bin_points[m + 1];
        int f_right = bin_points[m + 2];

        for (int f = f_left; f < f_center && f < freq_bins_; f++) {
            if (f_center > f_left) {
                filters[m * freq_bins_ + f] = (float)(f - f_left) / (f_center - f_left);
            }
        }
        for (int f = f_center; f < f_right && f < freq_bins_; f++) {
            if (f_right > f_center) {
                filters[m * freq_bins_ + f] = (float)(f_right - f) / (f_right - f_center);
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_mel_filters_, filters.data(),
                           config_.n_mels * freq_bins_ * sizeof(float),
                           cudaMemcpyHostToDevice));
}

int MelSpectrogram::get_num_frames(int num_samples) const {
    if (num_samples < config_.n_fft) return 1;
    return (num_samples - config_.n_fft) / config_.hop_length + 1;
}

void MelSpectrogram::frame_audio(const float* h_audio, int num_samples, int num_frames) {
    // Create overlapping frames on host, then copy to device
    std::vector<float> frames(num_frames * config_.n_fft, 0.0f);

    for (int i = 0; i < num_frames; i++) {
        int start = i * config_.hop_length;
        int copy_len = std::min(config_.n_fft, num_samples - start);
        if (copy_len > 0) {
            memcpy(frames.data() + i * config_.n_fft,
                   h_audio + start, copy_len * sizeof(float));
        }
    }

    CUDA_CHECK(cudaMemcpy(d_frames_, frames.data(),
                           num_frames * config_.n_fft * sizeof(float),
                           cudaMemcpyHostToDevice));
}

int MelSpectrogram::compute(const float* h_audio, int num_samples, float* d_output) {
    int num_frames = get_num_frames(num_samples);
    if (num_frames > max_frames_) {
        // Reallocate if needed
        cudaFree(d_frames_);
        cudaFree(d_fft_);
        cudaFree(d_power_);
        max_frames_ = num_frames;
        CUDA_CHECK(cudaMalloc(&d_frames_, max_frames_ * config_.n_fft * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fft_, max_frames_ * freq_bins_ * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_power_, max_frames_ * freq_bins_ * sizeof(float)));
        if (plan_created_) { cufftDestroy(fft_plan_); plan_created_ = false; }
    }

    // Step 1: Frame the audio
    frame_audio(h_audio, num_samples, num_frames);

    // Step 2: Apply Hanning window
    launch_hanning_window(d_frames_, num_frames, config_.n_fft);

    // Step 3: Batch FFT (R2C)
    if (!plan_created_ || true) { // Recreate plan for current batch size
        if (plan_created_) cufftDestroy(fft_plan_);
        int n[] = {config_.n_fft};
        CUFFT_CHECK(cufftPlanMany(&fft_plan_, 1, n,
                                   NULL, 1, config_.n_fft,    // input stride, dist
                                   NULL, 1, freq_bins_,       // output stride, dist
                                   CUFFT_R2C, num_frames));
        plan_created_ = true;
    }
    CUFFT_CHECK(cufftExecR2C(fft_plan_, d_frames_, d_fft_));

    // Step 4: Power spectrum
    launch_power_spectrum(d_fft_, d_power_, num_frames, freq_bins_);

    // Step 5: Mel filterbank
    launch_mel_filterbank(d_power_, d_mel_filters_, d_output, num_frames, freq_bins_,
                          config_.n_mels);

    // Step 6: Log mel
    launch_log_mel(d_output, num_frames, config_.n_mels);

    // Step 7: Normalize (optional)
    if (config_.normalize) {
        launch_mel_normalize(d_output, num_frames, config_.n_mels);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return num_frames;
}
