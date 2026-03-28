#include "cuda_utils.h"
#include "pipeline.h"
#include "wav_reader.h"
#include <cstdio>
#include <cstring>

void print_usage(const char* prog) {
    printf("Whisper Speech Recognition - CUDA C++ Implementation\n\n");
    printf("Usage:\n");
    printf("  %s [options] <wav_file>\n", prog);
    printf("  %s --demo                    Run with generated audio\n\n", prog);
    printf("Options:\n");
    printf("  --d_model <int>       Model dimension (default: 512)\n");
    printf("  --n_heads <int>       Attention heads (default: 8)\n");
    printf("  --enc_layers <int>    Encoder layers (default: 4)\n");
    printf("  --dec_layers <int>    Decoder layers (default: 4)\n");
    printf("  --ffn_dim <int>       FFN dimension (default: 2048)\n");
    printf("  --beam_size <int>     Beam size (default: 1, greedy)\n");
    printf("  --temperature <float> Decoding temperature (default: 1.0)\n");
    printf("  --max_tokens <int>    Max output tokens (default: 200)\n");
    printf("  --demo                Run with generated test audio\n");
    printf("  --help                Show this message\n");
}

int main(int argc, char** argv) {
    printf("==========================================\n");
    printf("  Whisper Speech Recognition CUDA\n");
    printf("==========================================\n");
    print_gpu_info();
    printf("\n");

    WhisperConfig config;
    const char* wav_path = nullptr;
    bool demo_mode = false;
    float demo_duration = 3.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "--demo") == 0) demo_mode = true;
        else if (strcmp(argv[i], "--d_model") == 0 && i + 1 < argc) config.d_model = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_heads") == 0 && i + 1 < argc) config.n_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--enc_layers") == 0 && i + 1 < argc) config.encoder_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dec_layers") == 0 && i + 1 < argc) config.decoder_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ffn_dim") == 0 && i + 1 < argc) config.ffn_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--beam_size") == 0 && i + 1 < argc) {
            config.beam_size = atoi(argv[++i]);
            config.use_greedy = (config.beam_size <= 1);
        }
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) config.temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) config.max_text_len = atoi(argv[++i]);
        else if (argv[i][0] != '-') wav_path = argv[i];
        else { printf("Unknown option: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (!wav_path && !demo_mode) {
        printf("No input specified. Running demo mode.\n\n");
        demo_mode = true;
    }

    printf("Model Configuration:\n");
    printf("  d_model:        %d\n", config.d_model);
    printf("  n_heads:        %d\n", config.n_heads);
    printf("  encoder_layers: %d\n", config.encoder_layers);
    printf("  decoder_layers: %d\n", config.decoder_layers);
    printf("  ffn_dim:        %d\n", config.ffn_dim);
    printf("  beam_size:      %d (%s)\n", config.beam_size,
           config.use_greedy ? "greedy" : "beam search");
    printf("  temperature:    %.2f\n", config.temperature);
    printf("\n");

    printf("Initializing pipeline...\n");
    WhisperPipeline pipeline(config);
    pipeline.init_random_weights();
    printf("Pipeline ready.\n\n");

    TranscriptionResult result;

    if (demo_mode) {
        printf("Generating %.1f sec test audio...\n", demo_duration);
        WavData wav;
        generate_speech_like_audio(wav, config.sample_rate, demo_duration);
        save_wav("/tmp/whisper_demo.wav", wav.samples.data(),
                 wav.num_samples, wav.sample_rate);
        printf("Demo audio saved to /tmp/whisper_demo.wav\n\n");

        printf("Transcribing...\n");
        result = pipeline.transcribe(wav.samples.data(), wav.num_samples);
    } else {
        printf("Loading: %s\n", wav_path);
        result = pipeline.transcribe_file(wav_path);
    }

    printf("\n==========================================\n");
    printf("  Transcription Result\n");
    printf("==========================================\n");
    printf("  Text:     '%s'\n", result.text.c_str());
    printf("  Tokens:   %d\n", result.num_tokens);
    printf("\n");
    printf("  Timing Breakdown:\n");
    printf("    Audio duration:  %.1f ms (%.2f sec)\n",
           result.audio_duration_ms, result.audio_duration_ms / 1000.0f);
    printf("    Mel spectrogram: %.2f ms\n", result.mel_time_ms);
    printf("    Encoder:         %.2f ms\n", result.encoder_time_ms);
    printf("    Decoder:         %.2f ms\n", result.decoder_time_ms);
    printf("    Total inference: %.2f ms\n", result.total_time_ms);
    printf("\n");
    if (result.decoder_time_ms > 0) {
        printf("  Decoder throughput: %.1f tokens/sec\n", result.tokens_per_sec);
    }
    if (result.total_time_ms > 0) {
        printf("  Real-time factor:   %.2fx\n",
               result.audio_duration_ms / result.total_time_ms);
    }
    printf("==========================================\n");

    return 0;
}
