#include "wav_reader.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct WavHeader {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

bool load_wav(const char* filename, WavData& wav) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open WAV file: %s\n", filename);
        return false;
    }

    WavHeader header;
    if (fread(&header, sizeof(WavHeader), 1, f) != 1) {
        fprintf(stderr, "Failed to read WAV header\n");
        fclose(f);
        return false;
    }

    if (memcmp(header.riff, "RIFF", 4) != 0 || memcmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a valid WAV file\n");
        fclose(f);
        return false;
    }

    if (header.audio_format != 1) {
        fprintf(stderr, "Only PCM WAV supported (format=%d)\n", header.audio_format);
        fclose(f);
        return false;
    }

    // Find data chunk
    char chunk_id[4];
    uint32_t chunk_size;
    while (true) {
        if (fread(chunk_id, 4, 1, f) != 1) {
            fprintf(stderr, "Data chunk not found\n");
            fclose(f);
            return false;
        }
        if (fread(&chunk_size, 4, 1, f) != 1) {
            fprintf(stderr, "Failed to read chunk size\n");
            fclose(f);
            return false;
        }
        if (memcmp(chunk_id, "data", 4) == 0) break;
        fseek(f, chunk_size, SEEK_CUR);
    }

    int bytes_per_sample = header.bits_per_sample / 8;
    int total_samples = chunk_size / bytes_per_sample;
    int num_channels = header.num_channels;
    int samples_per_channel = total_samples / num_channels;

    wav.sample_rate = header.sample_rate;
    wav.num_channels = 1; // Always output mono
    wav.num_samples = samples_per_channel;
    wav.samples.resize(samples_per_channel);

    if (header.bits_per_sample == 16) {
        std::vector<int16_t> raw(total_samples);
        fread(raw.data(), sizeof(int16_t), total_samples, f);
        for (int i = 0; i < samples_per_channel; i++) {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c] / 32768.0f;
            }
            wav.samples[i] = sum / num_channels;
        }
    } else if (header.bits_per_sample == 32) {
        std::vector<int32_t> raw(total_samples);
        fread(raw.data(), sizeof(int32_t), total_samples, f);
        for (int i = 0; i < samples_per_channel; i++) {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c] / 2147483648.0f;
            }
            wav.samples[i] = sum / num_channels;
        }
    } else {
        fprintf(stderr, "Unsupported bits per sample: %d\n", header.bits_per_sample);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

bool save_wav(const char* filename, const float* samples, int num_samples, int sample_rate) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    uint32_t data_size = num_samples * sizeof(int16_t);
    WavHeader header;
    memcpy(header.riff, "RIFF", 4);
    header.file_size = 36 + data_size;
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    header.fmt_size = 16;
    header.audio_format = 1;
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * sizeof(int16_t);
    header.block_align = sizeof(int16_t);
    header.bits_per_sample = 16;

    fwrite(&header, sizeof(WavHeader), 1, f);
    fwrite("data", 4, 1, f);
    fwrite(&data_size, 4, 1, f);

    std::vector<int16_t> pcm(num_samples);
    for (int i = 0; i < num_samples; i++) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        pcm[i] = (int16_t)(s * 32767.0f);
    }
    fwrite(pcm.data(), sizeof(int16_t), num_samples, f);

    fclose(f);
    return true;
}

void generate_test_audio(WavData& wav, int sample_rate, float duration_sec) {
    wav.sample_rate = sample_rate;
    wav.num_channels = 1;
    wav.num_samples = (int)(sample_rate * duration_sec);
    wav.samples.resize(wav.num_samples);

    // Mix of sine waves at different frequencies
    float freqs[] = {440.0f, 880.0f, 1320.0f, 220.0f};
    float amps[] = {0.4f, 0.2f, 0.1f, 0.3f};

    for (int i = 0; i < wav.num_samples; i++) {
        float t = (float)i / sample_rate;
        float val = 0.0f;
        for (int f = 0; f < 4; f++) {
            val += amps[f] * sinf(2.0f * M_PI * freqs[f] * t);
        }
        // Apply envelope
        float env = 1.0f;
        float fade_time = 0.05f;
        if (t < fade_time) env = t / fade_time;
        if (t > duration_sec - fade_time) env = (duration_sec - t) / fade_time;
        wav.samples[i] = val * env;
    }
}

void generate_speech_like_audio(WavData& wav, int sample_rate, float duration_sec) {
    wav.sample_rate = sample_rate;
    wav.num_channels = 1;
    wav.num_samples = (int)(sample_rate * duration_sec);
    wav.samples.resize(wav.num_samples);

    // Simulate speech with formants and pitch variation
    float f0_base = 120.0f; // Fundamental frequency

    for (int i = 0; i < wav.num_samples; i++) {
        float t = (float)i / sample_rate;

        // Slowly varying pitch
        float f0 = f0_base * (1.0f + 0.1f * sinf(2.0f * M_PI * 3.0f * t));

        // Glottal pulse (sum of harmonics)
        float glottal = 0.0f;
        for (int h = 1; h <= 10; h++) {
            float amp = 1.0f / (h * h);
            glottal += amp * sinf(2.0f * M_PI * f0 * h * t);
        }

        // Formant filters (simplified)
        float formant1 = 0.5f * sinf(2.0f * M_PI * 500.0f * t);
        float formant2 = 0.3f * sinf(2.0f * M_PI * 1500.0f * t);
        float formant3 = 0.1f * sinf(2.0f * M_PI * 2500.0f * t);

        float val = glottal * 0.5f + formant1 + formant2 + formant3;

        // Amplitude modulation to simulate syllables
        float syllable_rate = 4.0f; // ~4 syllables per second
        float env = 0.5f + 0.5f * sinf(2.0f * M_PI * syllable_rate * t);

        // Fade in/out
        float fade = 0.05f;
        if (t < fade) env *= t / fade;
        if (t > duration_sec - fade) env *= (duration_sec - t) / fade;

        wav.samples[i] = val * env * 0.3f;
    }
}

void resample(const WavData& input, WavData& output, int target_rate) {
    output.sample_rate = target_rate;
    output.num_channels = 1;
    double ratio = (double)target_rate / input.sample_rate;
    output.num_samples = (int)(input.num_samples * ratio);
    output.samples.resize(output.num_samples);

    for (int i = 0; i < output.num_samples; i++) {
        double src_pos = i / ratio;
        int idx = (int)src_pos;
        double frac = src_pos - idx;
        if (idx + 1 < input.num_samples) {
            output.samples[i] = (float)((1.0 - frac) * input.samples[idx] +
                                         frac * input.samples[idx + 1]);
        } else {
            output.samples[i] = input.samples[std::min(idx, input.num_samples - 1)];
        }
    }
}
