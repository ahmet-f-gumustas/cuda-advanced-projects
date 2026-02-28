#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

// ============================================================================
// CharTokenizer - Simple character-level tokenizer (vocab_size = 256)
// ============================================================================
// Each byte (0-255) maps directly to a token ID.
// Special tokens:
//   0 = <EOS> / <PAD>   (null byte is treated as end-of-sequence)
//   1-255 = ASCII / UTF-8 bytes
//
// This allows inference testing without a real BPE tokenizer.
// For real GPT-2 weights, replace with a proper tiktoken/BPE implementation.
// ============================================================================

class CharTokenizer {
public:
    static constexpr int VOCAB_SIZE = 256;
    static constexpr int EOS_ID     = 0;
    static constexpr int BOS_ID     = 1;    // '\x01' — unused in practice

    CharTokenizer() = default;

    // Encode text to token IDs (one byte → one token)
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<int>(c));
        }
        return ids;
    }

    // Decode token IDs back to string
    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        text.reserve(ids.size());
        for (int id : ids) {
            if (id == EOS_ID) break;
            if (id > 0 && id < 256) {
                text += static_cast<char>(id);
            }
        }
        return text;
    }

    // Decode a single token (returns empty string for EOS)
    std::string decode_token(int id) const {
        if (id <= 0 || id >= 256) return "";
        return std::string(1, static_cast<char>(id));
    }

    int vocab_size() const { return VOCAB_SIZE; }
    int eos_id()     const { return EOS_ID; }
    int bos_id()     const { return BOS_ID; }

    // Print vocabulary info
    void print_info() const {
        std::cout << "CharTokenizer: vocab_size=" << VOCAB_SIZE
                  << ", EOS=" << EOS_ID << std::endl;
    }
};

#endif // TOKENIZER_H
