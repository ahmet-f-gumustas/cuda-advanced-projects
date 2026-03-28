#ifndef WAV_READER_H
#define WAV_READER_H

#include <vector>
#include <cstdint>

struct WavData {
    std::vector<float> samples;
    int sample_rate;
    int num_channels;
    int num_samples;
};

// Load a WAV file (16-bit PCM, mono/stereo -> mono conversion)
bool load_wav(const char* filename, WavData& wav);

// Save audio as 16-bit PCM WAV
bool save_wav(const char* filename, const float* samples, int num_samples, int sample_rate);

// Generate test audio: sine wave mix for testing
void generate_test_audio(WavData& wav, int sample_rate, float duration_sec);

// Generate speech-like test audio with formants
void generate_speech_like_audio(WavData& wav, int sample_rate, float duration_sec);

// Resample to target sample rate (simple linear interpolation)
void resample(const WavData& input, WavData& output, int target_rate);

#endif // WAV_READER_H
