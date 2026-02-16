#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include <vector>
#include <string>
#include <memory>

class LSTMLayer {
public:
    LSTMLayer(int input_size, int hidden_size, int batch_size);
    ~LSTMLayer();

    // Forward pass
    void forward(const float* x);

    // Backward pass
    void backward(const float* dh_next, const float* dc_next);

    // Get outputs
    const float* get_hidden_state() const { return d_h_next_; }
    const float* get_cell_state() const { return d_c_next_; }

    // Get gradients
    const float* get_dx() const { return d_dx_; }

    // Update weights (simple SGD)
    void update_weights(float learning_rate);

    // Reset states
    void reset_states();

    // Save and load weights
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);

    // Getters
    int get_input_size() const { return input_size_; }
    int get_hidden_size() const { return hidden_size_; }
    int get_batch_size() const { return batch_size_; }

private:
    int input_size_;
    int hidden_size_;
    int batch_size_;
    int concat_size_;

    // Device memory - Parameters
    float* d_W_i_;  // Input gate weights
    float* d_W_f_;  // Forget gate weights
    float* d_W_g_;  // Cell gate weights
    float* d_W_o_;  // Output gate weights
    float* d_b_i_;  // Input gate bias
    float* d_b_f_;  // Forget gate bias
    float* d_b_g_;  // Cell gate bias
    float* d_b_o_;  // Output gate bias

    // Device memory - States
    float* d_h_prev_;
    float* d_h_next_;
    float* d_c_prev_;
    float* d_c_next_;
    float* d_x_;

    // Device memory - Gates
    float* d_i_gate_;
    float* d_f_gate_;
    float* d_g_gate_;
    float* d_o_gate_;

    // Device memory - Gradients
    float* d_dW_i_;
    float* d_dW_f_;
    float* d_dW_g_;
    float* d_dW_o_;
    float* d_db_i_;
    float* d_db_f_;
    float* d_db_g_;
    float* d_db_o_;
    float* d_dh_prev_;
    float* d_dc_prev_;
    float* d_dx_;

    void allocate_memory();
    void free_memory();
    void initialize_weights();
};

#endif // LSTM_LAYER_H
