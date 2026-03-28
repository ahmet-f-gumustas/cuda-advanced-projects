#include "pipeline.h"
#include "cuda_utils.h"
#include <cstdio>

MelConfig WhisperConfig::get_mel_config() const {
    MelConfig mc;
    mc.sample_rate = sample_rate;
    mc.n_fft = n_fft;
    mc.hop_length = hop_length;
    mc.n_mels = n_mels;
    mc.fmin = 0.0f;
    mc.fmax = (float)(sample_rate / 2);
    mc.normalize = true;
    return mc;
}

EncoderConfig WhisperConfig::get_encoder_config() const {
    EncoderConfig ec;
    ec.d_model = d_model;
    ec.n_heads = n_heads;
    ec.n_layers = encoder_layers;
    ec.ffn_dim = ffn_dim;
    ec.n_mels = n_mels;
    ec.max_seq_len = max_audio_len;
    ec.conv1_kernel = 3;
    ec.conv2_kernel = 3;
    ec.conv2_stride = 2;
    return ec;
}

DecoderConfig WhisperConfig::get_decoder_config() const {
    CharTokenizer tok;
    DecoderConfig dc;
    dc.d_model = d_model;
    dc.n_heads = n_heads;
    dc.n_layers = decoder_layers;
    dc.ffn_dim = ffn_dim;
    dc.vocab_size = tok.vocab_size();
    dc.max_seq_len = max_text_len;
    return dc;
}

BeamConfig WhisperConfig::get_beam_config() const {
    BeamConfig bc;
    bc.beam_size = beam_size;
    bc.max_length = max_text_len;
    bc.length_penalty = length_penalty;
    bc.temperature = temperature;
    bc.eos_token = 2;
    bc.bos_token = 1;
    return bc;
}

WhisperPipeline::WhisperPipeline(const WhisperConfig& cfg) : config(cfg) {
    MelConfig mc = cfg.get_mel_config();
    EncoderConfig ec = cfg.get_encoder_config();
    // Need tokenizer for vocab size
    DecoderConfig dc;
    dc.d_model = cfg.d_model;
    dc.n_heads = cfg.n_heads;
    dc.n_layers = cfg.decoder_layers;
    dc.ffn_dim = cfg.ffn_dim;
    dc.vocab_size = tokenizer_.vocab_size();
    dc.max_seq_len = cfg.max_text_len;
    BeamConfig bc;
    bc.beam_size = cfg.beam_size;
    bc.max_length = cfg.max_text_len;
    bc.length_penalty = cfg.length_penalty;
    bc.temperature = cfg.temperature;
    bc.eos_token = tokenizer_.eos_token();
    bc.bos_token = tokenizer_.bos_token();

    mel_ = new MelSpectrogram(mc);
    encoder_ = new WhisperEncoder(ec);
    decoder_ = new WhisperDecoder(dc);
    beam_search_ = new BeamSearch(bc);

    // Allocate output buffers
    int max_mel_frames = cfg.max_audio_len;
    CUDA_CHECK(cudaMalloc(&d_mel_output_, max_mel_frames * cfg.n_mels * sizeof(float)));
    int max_enc_out = encoder_->get_output_length(max_mel_frames);
    CUDA_CHECK(cudaMalloc(&d_encoder_output_, max_enc_out * cfg.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, cfg.max_text_len * dc.vocab_size * sizeof(float)));
}

WhisperPipeline::~WhisperPipeline() {
    delete mel_;
    delete encoder_;
    delete decoder_;
    delete beam_search_;
    cudaFree(d_mel_output_);
    cudaFree(d_encoder_output_);
    cudaFree(d_logits_);
}

void WhisperPipeline::init_random_weights() {
    encoder_->init_random_weights();
    decoder_->init_random_weights();
}

TranscriptionResult WhisperPipeline::transcribe(const float* audio, int num_samples) {
    TranscriptionResult result;
    result.audio_duration_ms = (float)num_samples / config.sample_rate * 1000.0f;

    CudaTimer timer;

    // Step 1: Mel spectrogram
    timer.start();
    int num_frames = mel_->compute(audio, num_samples, d_mel_output_);
    result.mel_time_ms = timer.stop();
    printf("  Mel spectrogram: %d frames (%.2f ms)\n", num_frames, result.mel_time_ms);

    // The mel output is [num_frames, n_mels] but conv1d expects [n_mels, num_frames]
    // Need to transpose
    float* d_mel_transposed;
    CUDA_CHECK(cudaMalloc(&d_mel_transposed,
                           num_frames * config.n_mels * sizeof(float)));
    // Use cuBLAS geam for transpose
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasSgeam(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              config.n_mels, num_frames,
                              &alpha, d_mel_output_, num_frames,
                              &beta_val, d_mel_output_, config.n_mels,
                              d_mel_transposed, config.n_mels));
    cublasDestroy(cublas);

    // Step 2: Encoder
    timer.start();
    int enc_out_len;
    encoder_->forward(d_mel_transposed, num_frames, d_encoder_output_, enc_out_len);
    result.encoder_time_ms = timer.stop();
    printf("  Encoder: %d -> %d frames (%.2f ms)\n",
           num_frames, enc_out_len, result.encoder_time_ms);

    cudaFree(d_mel_transposed);

    // Step 3: Decoder (greedy or beam search)
    timer.start();
    std::vector<int> tokens;
    if (config.use_greedy) {
        tokens = beam_search_->greedy_decode(*decoder_, d_encoder_output_, enc_out_len);
    } else {
        tokens = beam_search_->search(*decoder_, d_encoder_output_, enc_out_len);
    }
    result.decoder_time_ms = timer.stop();
    result.num_tokens = (int)tokens.size();
    printf("  Decoder: %d tokens (%.2f ms)\n", result.num_tokens, result.decoder_time_ms);

    // Step 4: Decode tokens to text
    result.text = tokenizer_.decode(tokens);
    result.total_time_ms = result.mel_time_ms + result.encoder_time_ms + result.decoder_time_ms;
    result.tokens_per_sec = result.num_tokens / (result.decoder_time_ms / 1000.0f);

    return result;
}

TranscriptionResult WhisperPipeline::transcribe_file(const char* wav_path) {
    WavData wav;
    if (!load_wav(wav_path, wav)) {
        printf("Failed to load WAV file: %s\n", wav_path);
        TranscriptionResult empty;
        empty.text = "[ERROR: Failed to load audio]";
        return empty;
    }

    // Resample to target rate if needed
    if (wav.sample_rate != config.sample_rate) {
        printf("  Resampling %d -> %d Hz\n", wav.sample_rate, config.sample_rate);
        WavData resampled;
        resample(wav, resampled, config.sample_rate);
        return transcribe(resampled.samples.data(), resampled.num_samples);
    }

    return transcribe(wav.samples.data(), wav.num_samples);
}
