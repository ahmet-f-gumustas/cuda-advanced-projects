#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

// Char-level tokenizer for Whisper ASR
// Vocab: 0=PAD, 1=BOS, 2=EOS, 3=UNK, 4-29=a-z, 30=space, 31-40=0-9, 41+=punctuation
class CharTokenizer {
public:
    CharTokenizer();

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode_single(int token) const;

    int vocab_size() const { return vocab_size_; }
    int pad_token() const { return 0; }
    int bos_token() const { return 1; }
    int eos_token() const { return 2; }
    int unk_token() const { return 3; }

private:
    int vocab_size_;
    int char_to_id_[256];
    char id_to_char_[128];
    int num_chars_;
};

#endif // TOKENIZER_H
