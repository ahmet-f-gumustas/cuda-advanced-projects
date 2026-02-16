#include "lstm_layer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

int main() {
    std::cout << "=== LSTM Model Testing and Inference ===" << std::endl;

    // Model parameters (should match training)
    const int input_size = 1;
    const int hidden_size = 32;
    const int batch_size = 1;
    const int sequence_length = 20;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input size: " << input_size << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Sequence length: " << sequence_length << std::endl;
    std::cout << std::endl;

    // Load LSTM model
    std::cout << "Loading LSTM model..." << std::endl;
    LSTMLayer lstm(input_size, hidden_size, batch_size);
    lstm.load_weights("lstm_model.bin");

    // Load output layer
    std::vector<float> W_out(hidden_size);
    std::vector<float> b_out(1);

    std::ifstream in_file("output_layer.bin", std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Error: Could not open output_layer.bin" << std::endl;
        std::cerr << "Please run training first!" << std::endl;
        return 1;
    }

    in_file.read(reinterpret_cast<char*>(W_out.data()), W_out.size() * sizeof(float));
    in_file.read(reinterpret_cast<char*>(b_out.data()), b_out.size() * sizeof(float));
    in_file.close();

    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << std::endl;

    // Test 1: Sine wave prediction
    std::cout << "=== Test 1: Sine Wave Prediction ===" << std::endl;
    lstm.reset_states();

    std::vector<float> input(input_size);
    float frequency = 0.05f;

    // Feed initial sequence
    std::cout << "Feeding initial sequence..." << std::endl;
    for (int t = 0; t < sequence_length; t++) {
        input[0] = std::sin(2.0f * M_PI * frequency * t);
        lstm.forward(input.data());
    }

    // Predict next 20 timesteps
    std::cout << "Predicting next 20 timesteps:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    float total_error = 0.0f;
    for (int t = 0; t < 20; t++) {
        float final_hidden[hidden_size];
        cudaMemcpy(final_hidden, lstm.get_hidden_state(),
                  hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        float prediction = b_out[0];
        for (int i = 0; i < hidden_size; i++) {
            prediction += final_hidden[i] * W_out[i];
        }

        float actual = std::sin(2.0f * M_PI * frequency * (sequence_length + t));
        float error = std::abs(prediction - actual);
        total_error += error;

        if (t < 10 || t >= 15) {  // Print first 10 and last 5
            std::cout << "  t=" << std::setw(2) << t
                      << " | Predicted: " << std::setw(7) << prediction
                      << " | Actual: " << std::setw(7) << actual
                      << " | Error: " << std::setw(7) << error
                      << std::endl;
        } else if (t == 10) {
            std::cout << "  ..." << std::endl;
        }

        // Use prediction as next input (teacher forcing disabled)
        input[0] = prediction;
        lstm.forward(input.data());
    }

    float avg_error = total_error / 20.0f;
    std::cout << std::endl;
    std::cout << "Average prediction error: " << avg_error << std::endl;
    std::cout << std::endl;

    // Test 2: Different frequency
    std::cout << "=== Test 2: Different Frequency (0.08) ===" << std::endl;
    lstm.reset_states();

    float frequency2 = 0.08f;
    for (int t = 0; t < sequence_length; t++) {
        input[0] = std::sin(2.0f * M_PI * frequency2 * t);
        lstm.forward(input.data());
    }

    std::cout << "Predicting next 10 timesteps:" << std::endl;
    total_error = 0.0f;
    for (int t = 0; t < 10; t++) {
        float final_hidden[hidden_size];
        cudaMemcpy(final_hidden, lstm.get_hidden_state(),
                  hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        float prediction = b_out[0];
        for (int i = 0; i < hidden_size; i++) {
            prediction += final_hidden[i] * W_out[i];
        }

        float actual = std::sin(2.0f * M_PI * frequency2 * (sequence_length + t));
        float error = std::abs(prediction - actual);
        total_error += error;

        std::cout << "  t=" << std::setw(2) << t
                  << " | Predicted: " << std::setw(7) << prediction
                  << " | Actual: " << std::setw(7) << actual
                  << " | Error: " << std::setw(7) << error
                  << std::endl;

        input[0] = prediction;
        lstm.forward(input.data());
    }

    avg_error = total_error / 10.0f;
    std::cout << std::endl;
    std::cout << "Average prediction error: " << avg_error << std::endl;
    std::cout << std::endl;

    // Test 3: Hidden state visualization
    std::cout << "=== Test 3: Hidden State Analysis ===" << std::endl;
    lstm.reset_states();

    // Feed a few timesteps and observe hidden state
    for (int t = 0; t < 5; t++) {
        input[0] = std::sin(2.0f * M_PI * frequency * t);
        lstm.forward(input.data());

        float hidden[hidden_size];
        cudaMemcpy(hidden, lstm.get_hidden_state(),
                  hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "After timestep " << t << ":" << std::endl;
        std::cout << "  Hidden state (first 8 units): ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::setw(6) << hidden[i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Testing Complete ===" << std::endl;

    return 0;
}
