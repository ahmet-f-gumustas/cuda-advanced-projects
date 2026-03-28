#ifndef PIPELINE_H
#define PIPELINE_H

#include "mel_spectrogram.h"
#include "encoder.h"
#include "decoder.h"
#include "beam_search.h"
#include "tokenizer.h"
#include "wav_reader.h"
#include <string>

struct WhisperConfig {
    // Audio
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 80;

    // Model
    int d_model = 512;
    int n_heads = 8;
    int encoder_layers = 4;
    int decoder_layers = 4;
    int ffn_dim = 2048;
    int max_audio_len = 1500;
    int max_text_len = 448;

    // Decoding
    int beam_size = 5;
    float length_penalty = 1.0f;
    float temperature = 1.0f;
    bool use_greedy = true;

    MelConfig get_mel_config() const;
    EncoderConfig get_encoder_config() const;
    DecoderConfig get_decoder_config() const;
    BeamConfig get_beam_config() const;
};

struct TranscriptionResult {
    std::string text;
    float audio_duration_ms;
    float mel_time_ms;
    float encoder_time_ms;
    float decoder_time_ms;
    float total_time_ms;
    int num_tokens;
    float tokens_per_sec;
};

class WhisperPipeline {
public:
    WhisperPipeline(const WhisperConfig& config);
    ~WhisperPipeline();

    // Transcribe from audio samples
    TranscriptionResult transcribe(const float* audio, int num_samples);

    // Transcribe from WAV file
    TranscriptionResult transcribe_file(const char* wav_path);

    // Initialize random weights (for testing)
    void init_random_weights();

    WhisperConfig config;

private:
    MelSpectrogram* mel_;
    WhisperEncoder* encoder_;
    WhisperDecoder* decoder_;
    BeamSearch* beam_search_;
    CharTokenizer tokenizer_;

    // Device buffers
    float* d_mel_output_;
    float* d_encoder_output_;
    float* d_logits_;
};

#endif // PIPELINE_H
