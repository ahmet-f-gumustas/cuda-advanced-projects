#pragma once

#include "core/layer.h"

namespace deep_engine {

// Helper layers for testing
class IdentityLayer : public Layer {
public:
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override {
        return input.clone();
    }
    std::string type() const override { return "Identity"; }
};

class ConcatLayer : public Layer {
public:
    explicit ConcatLayer(int axis) : axis_(axis) {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs, ExecutionContext& ctx) override {
        return {cat(inputs, axis_)};
    }
    
    std::string type() const override { return "Concat"; }
    
private:
    int axis_;
};

class SplitLayer : public Layer {
public:
    SplitLayer(int chunks, int axis) : chunks_(chunks), axis_(axis) {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs, ExecutionContext& ctx) override {
        return split(inputs[0], chunks_, axis_);
    }
    
    std::string type() const override { return "Split"; }
    
private:
    int chunks_;
    int axis_;
};

class AddLayer : public Layer {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs, ExecutionContext& ctx) override {
        if (inputs.size() != 2) {
            throw std::runtime_error("AddLayer expects exactly 2 inputs");
        }
        return {inputs[0] + inputs[1]};
    }
    
    std::string type() const override { return "Add"; }
};

} // namespace deep_engine