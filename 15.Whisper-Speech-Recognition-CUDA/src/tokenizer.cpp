#include "tokenizer.h"
#include <cctype>
#include <cstring>

CharTokenizer::CharTokenizer() {
    memset(char_to_id_, 0, sizeof(char_to_id_));
    memset(id_to_char_, 0, sizeof(id_to_char_));

    // Special tokens: 0=PAD, 1=BOS, 2=EOS, 3=UNK
    int id = 4;

    // a-z
    for (char c = 'a'; c <= 'z'; c++) {
        char_to_id_[(unsigned char)c] = id;
        id_to_char_[id] = c;
        id++;
    }
    // space
    char_to_id_[(unsigned char)' '] = id;
    id_to_char_[id] = ' ';
    id++;
    // 0-9
    for (char c = '0'; c <= '9'; c++) {
        char_to_id_[(unsigned char)c] = id;
        id_to_char_[id] = c;
        id++;
    }
    // Punctuation
    const char* punct = ".,!?;:'-\"()";
    for (int i = 0; punct[i]; i++) {
        char_to_id_[(unsigned char)punct[i]] = id;
        id_to_char_[id] = punct[i];
        id++;
    }

    num_chars_ = id;
    vocab_size_ = id;
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_token());
    for (char c : text) {
        char lower = (char)tolower((unsigned char)c);
        int id = char_to_id_[(unsigned char)lower];
        if (id == 0 && lower != '\0') {
            // Check if it's a space mapped to 0 or truly unknown
            if (lower == ' ') {
                // Find space id
                for (int i = 4; i < num_chars_; i++) {
                    if (id_to_char_[i] == ' ') { id = i; break; }
                }
            }
            if (id == 0) id = unk_token();
        }
        tokens.push_back(id);
    }
    tokens.push_back(eos_token());
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int tok : tokens) {
        if (tok == pad_token() || tok == bos_token() || tok == eos_token()) continue;
        if (tok == unk_token()) { text += '?'; continue; }
        if (tok >= 4 && tok < num_chars_) {
            text += id_to_char_[tok];
        }
    }
    return text;
}

std::string CharTokenizer::decode_single(int token) const {
    if (token == pad_token()) return "<pad>";
    if (token == bos_token()) return "<bos>";
    if (token == eos_token()) return "<eos>";
    if (token == unk_token()) return "<unk>";
    if (token >= 4 && token < num_chars_) {
        return std::string(1, id_to_char_[token]);
    }
    return "<?>";
}
