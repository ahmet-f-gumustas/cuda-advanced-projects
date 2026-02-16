#include "lstm_layer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>

// Generate sine wave data for training
void generate_sine_wave_data(std::vector<float>& data, int length, float frequency = 0.05f) {
    data.resize(length);
    for (int i = 0; i < length; i++) {
        data[i] = std::sin(2.0f * M_PI * frequency * i);
    }
}

// Create sequences from time series data
void create_sequences(
    const std::vector<float>& data,
    std::vector<std::vector<float>>& X,
    std::vector<float>& y,
    int sequence_length
) {
    X.clear();
    y.clear();

    for (size_t i = 0; i < data.size() - sequence_length; i++) {
        std::vector<float> seq(sequence_length);
        for (int j = 0; j < sequence_length; j++) {
            seq[j] = data[i + j];
        }
        X.push_back(seq);
        y.push_back(data[i + sequence_length]);
    }
}

// Mean squared error loss
float compute_mse_loss(const std::vector<float>& predictions, const std::vector<float>& targets) {
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    return loss / predictions.size();
}

int main() {
    std::cout << "=== LSTM Sine Wave Prediction Training ===" << std::endl;

    // Hyperparameters
    const int input_size = 1;
    const int hidden_size = 32;
    const int batch_size = 1;
    const int sequence_length = 20;
    const int num_epochs = 100;
    const float learning_rate = 0.01f;
    const int data_length = 1000;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input size: " << input_size << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Sequence length: " << sequence_length << std::endl;
    std::cout << "  Epochs: " << num_epochs << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << std::endl;

    // Generate training data
    std::cout << "Generating training data..." << std::endl;
    std::vector<float> sine_data;
    generate_sine_wave_data(sine_data, data_length);

    std::vector<std::vector<float>> X_train;
    std::vector<float> y_train;
    create_sequences(sine_data, X_train, y_train, sequence_length);

    std::cout << "Training samples: " << X_train.size() << std::endl;
    std::cout << std::endl;

    // Create LSTM layer
    std::cout << "Initializing LSTM layer..." << std::endl;
    LSTMLayer lstm(input_size, hidden_size, batch_size);
    std::cout << "LSTM layer initialized successfully" << std::endl;
    std::cout << std::endl;

    // Add output layer weights (simple linear layer)
    int output_size = 1;
    std::vector<float> W_out(hidden_size * output_size);
    std::vector<float> b_out(output_size, 0.0f);

    // Initialize output weights
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / hidden_size);
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (auto& w : W_out) {
        w = dist(gen);
    }

    // Training loop
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;

        for (size_t sample_idx = 0; sample_idx < X_train.size(); sample_idx++) {
            // Reset LSTM states
            lstm.reset_states();

            // Forward pass through sequence
            std::vector<float> input(input_size);
            float final_hidden[hidden_size];

            for (int t = 0; t < sequence_length; t++) {
                input[0] = X_train[sample_idx][t];
                lstm.forward(input.data());
            }

            // Get final hidden state
            cudaMemcpy(final_hidden, lstm.get_hidden_state(),
                      hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

            // Output layer forward pass
            float prediction = b_out[0];
            for (int i = 0; i < hidden_size; i++) {
                prediction += final_hidden[i] * W_out[i];
            }

            // Compute loss
            float target = y_train[sample_idx];
            float error = prediction - target;
            total_loss += error * error;

            // Backward pass - output layer
            std::vector<float> dh_out(hidden_size);
            for (int i = 0; i < hidden_size; i++) {
                dh_out[i] = error * W_out[i];
            }

            // Update output weights
            for (int i = 0; i < hidden_size; i++) {
                W_out[i] -= learning_rate * error * final_hidden[i];
            }
            b_out[0] -= learning_rate * error;

            // LSTM backward pass
            std::vector<float> dc_zero(hidden_size, 0.0f);
            lstm.backward(dh_out.data(), dc_zero.data());

            // Update LSTM weights
            lstm.update_weights(learning_rate);
        }

        float avg_loss = total_loss / X_train.size();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << std::setw(3) << (epoch + 1)
                      << "/" << num_epochs
                      << " - Loss: " << std::fixed << std::setprecision(6) << avg_loss
                      << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Training completed!" << std::endl;

    // Save model
    std::string model_path = "lstm_model.bin";
    lstm.save_weights(model_path);

    // Save output layer weights
    std::ofstream out_file("output_layer.bin", std::ios::binary);
    out_file.write(reinterpret_cast<char*>(W_out.data()), W_out.size() * sizeof(float));
    out_file.write(reinterpret_cast<char*>(b_out.data()), b_out.size() * sizeof(float));
    out_file.close();
    std::cout << "Output layer weights saved to output_layer.bin" << std::endl;

    std::cout << std::endl;
    std::cout << "=== Testing on new sequence ===" << std::endl;

    // Test prediction
    lstm.reset_states();
    std::vector<float> test_input(input_size);

    std::cout << "Predicting next 10 values..." << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // Use last sequence as seed
    for (int t = 0; t < sequence_length; t++) {
        test_input[0] = sine_data[data_length - sequence_length + t];
        lstm.forward(test_input.data());
    }

    // Predict next values
    for (int t = 0; t < 10; t++) {
        float final_hidden[hidden_size];
        cudaMemcpy(final_hidden, lstm.get_hidden_state(),
                  hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        float prediction = b_out[0];
        for (int i = 0; i < hidden_size; i++) {
            prediction += final_hidden[i] * W_out[i];
        }

        float actual = std::sin(2.0f * M_PI * 0.05f * (data_length + t));

        std::cout << "Step " << (t + 1) << " - Predicted: " << prediction
                  << ", Actual: " << actual
                  << ", Error: " << std::abs(prediction - actual) << std::endl;

        // Use prediction as next input
        test_input[0] = prediction;
        lstm.forward(test_input.data());
    }

    return 0;
}
