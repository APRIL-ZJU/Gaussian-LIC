#pragma once

#include "rasterizer/rasterize_points.h"
#include <unordered_map>
#include <vector>
#include <torch/torch.h>

// State structure
struct State 
{
    int64_t step = 0;
    torch::Tensor exp_avg;
    torch::Tensor exp_avg_sq;
    bool initialized = false;
};

// Custom options class for SparseGaussianAdam
struct SparseGaussianAdamOptions : public torch::optim::OptimizerOptions 
{
public:
    // Constructor
    SparseGaussianAdamOptions(double lr = 1e-3, double eps = 1e-8)
        : lr_(lr), eps_(eps) {}

    // Hyperparameters
    double lr_;
    double eps_;

    // Implement the clone() method
    std::unique_ptr<OptimizerOptions> clone() const override 
    {
        return std::make_unique<SparseGaussianAdamOptions>(*this);
    }

    // Override get_lr() getter
    double get_lr() const override 
    {
        return lr_;
    }

    // Override set_lr() setter
    void set_lr(const double lr) override 
    {
        lr_ = lr;
    }

    // Getter for eps
    double get_eps() const 
    {
        return eps_;
    }

    // Setter for eps if needed
    void set_eps(const double eps) 
    {
        eps_ = eps;
    }
};

class SparseGaussianAdam : public torch::optim::Optimizer 
{
public:
    // Constructor
    SparseGaussianAdam(const std::vector<torch::Tensor>& params, double lr, double eps)
        : torch::optim::Optimizer(
              {torch::optim::OptimizerParamGroup(params)},
              std::make_unique<SparseGaussianAdamOptions>(lr, eps)) {}

    // Setter for visibility and N
    void set_visibility_and_N(const torch::Tensor& visibility, int64_t N) 
    {
        visibility_ = visibility;
        N_ = N;
    }

    // Override the pure virtual step function
    torch::Tensor step(LossClosure closure = nullptr) override 
    {
        torch::Tensor loss;
        if (closure != nullptr) 
        {
            loss = closure();
        }

        // Call the custom step function
        custom_step();

        return loss;
    }

    std::unordered_map<torch::TensorImpl*, State>& get_state() 
    {
        return state_;
    }

private:
    void custom_step() 
    {
        for (auto& group : param_groups_) 
        {
            // Correctly access the options
            auto& options = static_cast<SparseGaussianAdamOptions&>(group.options());
            double lr = options.get_lr();
            double eps = options.eps_;  // Access eps_ directly or use options.get_eps();

            TORCH_CHECK(group.params().size() == 1, "More than one tensor in group");
            auto& param = group.params()[0];
            if (!param.grad().defined()) 
            {
                continue;
            }

            // Lazy state initialization
            auto& state = state_[param.unsafeGetTensorImpl()];
            if (!state.initialized) 
            {
                state.step = 0;
                state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                state.initialized = true;
            }

            auto& exp_avg = state.exp_avg;
            auto& exp_avg_sq = state.exp_avg_sq;
            int64_t M = param.numel() / N_;

            // Create a mutable copy of param.grad()
            torch::Tensor grad = param.grad().clone();

            // Call the adamUpdate function with mutable grad
            adamUpdate(param, grad, exp_avg, exp_avg_sq, visibility_,
                    lr, 0.9, 0.999, eps, N_, M);

            state.step += 1;
        }
    }

    // Member variables
    torch::Tensor visibility_;
    int64_t N_;
    std::unordered_map<torch::TensorImpl*, State> state_;
};