#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <cuda_runtime.h>
#include <cufft.h>

struct MelConfig {
    int sample_rate = 16000;
    int n_fft = 400;         // 25ms window at 16kHz
    int hop_length = 160;    // 10ms hop
    int n_mels = 80;         // Number of mel bands
    float fmin = 0.0f;       // Min frequency
    float fmax = 8000.0f;    // Max frequency (Nyquist for 16kHz)
    bool normalize = true;   // Normalize per-feature
};

class MelSpectrogram {
public:
    MelSpectrogram(const MelConfig& config);
    ~MelSpectrogram();

    // Compute mel spectrogram from audio samples
    // h_audio: host audio samples [num_samples]
    // d_output: device output [num_frames, n_mels]
    // Returns number of frames
    int compute(const float* h_audio, int num_samples, float* d_output);

    // Get expected number of frames for given audio length
    int get_num_frames(int num_samples) const;

    // Get output feature dimension
    int get_n_mels() const { return config_.n_mels; }

private:
    MelConfig config_;
    int freq_bins_;          // n_fft / 2 + 1

    // cuFFT plan for batched FFT
    cufftHandle fft_plan_;
    bool plan_created_;

    // Device buffers
    float* d_frames_;        // [num_frames, n_fft] windowed frames
    cufftComplex* d_fft_;    // [num_frames, freq_bins] FFT output
    float* d_power_;         // [num_frames, freq_bins] power spectrum
    float* d_mel_filters_;   // [n_mels, freq_bins] mel filterbank

    int max_frames_;

    // Build mel filterbank matrix on host, then copy to device
    void build_mel_filterbank();

    // Frame the audio signal into overlapping windows
    void frame_audio(const float* h_audio, int num_samples, int num_frames);
};

#endif // MEL_SPECTROGRAM_H
