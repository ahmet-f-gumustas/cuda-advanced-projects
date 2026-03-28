#include "beam_search.h"
#include "cuda_utils.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <numeric>

BeamSearch::BeamSearch(const BeamConfig& cfg) : config(cfg) {
    // Logits buffer on device
    CUDA_CHECK(cudaMalloc(&d_logits_, 512 * sizeof(float))); // max vocab size
}

BeamSearch::~BeamSearch() {
    cudaFree(d_logits_);
}

void BeamSearch::get_top_k(const float* h_logits, int vocab_size, int k,
                            std::vector<int>& indices, std::vector<float>& scores) {
    // Apply temperature
    std::vector<float> logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = h_logits[i] / config.temperature;
    }

    // Log-softmax
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = logf(logits[i] / sum);
    }

    // Find top-k
    std::vector<int> sorted_idx(vocab_size);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + k, sorted_idx.end(),
                      [&logits](int a, int b) { return logits[a] > logits[b]; });

    indices.resize(k);
    scores.resize(k);
    for (int i = 0; i < k; i++) {
        indices[i] = sorted_idx[i];
        scores[i] = logits[sorted_idx[i]];
    }
}

std::vector<int> BeamSearch::greedy_decode(WhisperDecoder& decoder,
                                            const float* d_encoder_out, int enc_len) {
    decoder.reset_kv_cache();

    int vocab_size = decoder.config.vocab_size;
    std::vector<float> h_logits(vocab_size);
    std::vector<int> tokens;

    int current_token = config.bos_token;
    tokens.push_back(current_token);

    for (int step = 0; step < config.max_length; step++) {
        decoder.forward_step(current_token, step, d_encoder_out, enc_len, d_logits_);

        CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits_,
                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Find argmax
        int best_token = 0;
        float best_score = -FLT_MAX;
        for (int i = 0; i < vocab_size; i++) {
            if (h_logits[i] > best_score) {
                best_score = h_logits[i];
                best_token = i;
            }
        }

        if (best_token == config.eos_token) break;

        tokens.push_back(best_token);
        current_token = best_token;
    }

    return tokens;
}

std::vector<int> BeamSearch::search(WhisperDecoder& decoder,
                                     const float* d_encoder_out, int enc_len) {
    if (config.beam_size <= 1) {
        return greedy_decode(decoder, d_encoder_out, enc_len);
    }

    int vocab_size = decoder.config.vocab_size;
    int beam_size = config.beam_size;
    std::vector<float> h_logits(vocab_size);

    // Initialize beams
    std::vector<BeamHypothesis> beams(1);
    beams[0].tokens = {config.bos_token};
    beams[0].score = 0.0f;

    std::vector<BeamHypothesis> completed;

    for (int step = 0; step < config.max_length; step++) {
        std::vector<BeamHypothesis> candidates;

        for (int b = 0; b < (int)beams.size(); b++) {
            auto& beam = beams[b];
            int last_token = beam.tokens.back();

            // For beam search we do full forward passes (no KV cache for simplicity)
            // This is slower but correct for multi-beam
            decoder.reset_kv_cache();

            // Feed all tokens so far
            int num_tokens = (int)beam.tokens.size();
            std::vector<int> h_tokens = beam.tokens;
            int* d_tokens;
            CUDA_CHECK(cudaMalloc(&d_tokens, num_tokens * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(),
                                   num_tokens * sizeof(int), cudaMemcpyHostToDevice));

            // Allocate logits for full sequence
            float* d_full_logits;
            CUDA_CHECK(cudaMalloc(&d_full_logits,
                                   num_tokens * vocab_size * sizeof(float)));
            decoder.forward(d_tokens, num_tokens, d_encoder_out, enc_len, d_full_logits);

            // Get logits for last token
            CUDA_CHECK(cudaMemcpy(h_logits.data(),
                                   d_full_logits + (num_tokens - 1) * vocab_size,
                                   vocab_size * sizeof(float),
                                   cudaMemcpyDeviceToHost));

            cudaFree(d_tokens);
            cudaFree(d_full_logits);

            // Get top-k candidates
            std::vector<int> top_indices;
            std::vector<float> top_scores;
            get_top_k(h_logits.data(), vocab_size, beam_size * 2, top_indices, top_scores);

            for (int k = 0; k < beam_size * 2 && k < (int)top_indices.size(); k++) {
                BeamHypothesis candidate;
                candidate.tokens = beam.tokens;
                candidate.tokens.push_back(top_indices[k]);
                candidate.score = beam.score + top_scores[k];

                if (top_indices[k] == config.eos_token) {
                    // Apply length penalty
                    float penalty = powf((5.0f + candidate.tokens.size()) / 6.0f,
                                         config.length_penalty);
                    candidate.score /= penalty;
                    completed.push_back(candidate);
                } else {
                    candidates.push_back(candidate);
                }
            }
        }

        if (candidates.empty()) break;

        // Select top beam_size candidates
        std::sort(candidates.begin(), candidates.end(),
                  [](const BeamHypothesis& a, const BeamHypothesis& b) {
                      return a.score > b.score;
                  });

        beams.clear();
        for (int i = 0; i < beam_size && i < (int)candidates.size(); i++) {
            beams.push_back(candidates[i]);
        }

        // Early stopping if we have enough completed hypotheses
        if ((int)completed.size() >= beam_size) break;
    }

    // Add remaining beams to completed
    for (auto& beam : beams) {
        float penalty = powf((5.0f + beam.tokens.size()) / 6.0f, config.length_penalty);
        beam.score /= penalty;
        completed.push_back(beam);
    }

    // Return best hypothesis
    if (completed.empty()) {
        return {config.bos_token};
    }

    auto best = std::max_element(completed.begin(), completed.end(),
                                  [](const BeamHypothesis& a, const BeamHypothesis& b) {
                                      return a.score < b.score;
                                  });
    return best->tokens;
}
