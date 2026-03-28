#ifndef BEAM_SEARCH_H
#define BEAM_SEARCH_H

#include "decoder.h"
#include <vector>

struct BeamConfig {
    int beam_size = 5;
    int max_length = 200;
    float length_penalty = 1.0f;
    float temperature = 1.0f;
    int eos_token = 2;
    int bos_token = 1;
};

struct BeamHypothesis {
    std::vector<int> tokens;
    float score;
};

class BeamSearch {
public:
    BeamSearch(const BeamConfig& config);
    ~BeamSearch();

    // Run beam search decoding
    // Returns the best hypothesis token sequence
    std::vector<int> search(WhisperDecoder& decoder,
                            const float* d_encoder_out, int enc_len);

    // Greedy decoding (beam_size=1 equivalent, faster)
    std::vector<int> greedy_decode(WhisperDecoder& decoder,
                                    const float* d_encoder_out, int enc_len);

    BeamConfig config;

private:
    float* d_logits_;  // [vocab_size] workspace

    // Apply temperature and get top-k tokens from logits
    void get_top_k(const float* h_logits, int vocab_size, int k,
                   std::vector<int>& indices, std::vector<float>& scores);
};

#endif // BEAM_SEARCH_H
