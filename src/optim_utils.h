/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "rasterizer/rasterize_points.h"
#include <unordered_map>
#include <vector>
#include <torch/torch.h>

struct State 
{
    int64_t step = 0;
    torch::Tensor exp_avg;
    torch::Tensor exp_avg_sq;
    bool initialized = false;
};

struct SparseGaussianAdamOptions : public torch::optim::OptimizerOptions 
{
public:
    SparseGaussianAdamOptions(double lr = 1e-3, double eps = 1e-8)
        : lr_(lr), eps_(eps) {}

    double lr_;
    double eps_;

    std::unique_ptr<OptimizerOptions> clone() const override 
    {
        return std::make_unique<SparseGaussianAdamOptions>(*this);
    }

    double get_lr() const override 
    {
        return lr_;
    }

    void set_lr(const double lr) override 
    {
        lr_ = lr;
    }

    double get_eps() const 
    {
        return eps_;
    }

    void set_eps(const double eps) 
    {
        eps_ = eps;
    }
};

class SparseGaussianAdam : public torch::optim::Optimizer 
{
public:
    SparseGaussianAdam(const std::vector<torch::Tensor>& params, double lr, double eps)
        : torch::optim::Optimizer(
              {torch::optim::OptimizerParamGroup(params)},
              std::make_unique<SparseGaussianAdamOptions>(lr, eps)) {}

    void set_visibility_and_N(const torch::Tensor& visibility, int64_t N) 
    {
        visibility_ = visibility;
        N_ = N;
    }

    torch::Tensor step(LossClosure closure = nullptr) override 
    {
        torch::Tensor loss;
        if (closure != nullptr) 
        {
            loss = closure();
        }

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
            auto& options = static_cast<SparseGaussianAdamOptions&>(group.options());
            double lr = options.get_lr();
            double eps = options.eps_;

            TORCH_CHECK(group.params().size() == 1, "More than one tensor in group");
            auto& param = group.params()[0];
            if (!param.grad().defined()) 
            {
                continue;
            }

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

            torch::Tensor grad = param.grad().clone();

            adamUpdate(param, grad, exp_avg, exp_avg_sq, visibility_,
                    lr, 0.9, 0.999, eps, N_, M);

            state.step += 1;
        }
    }

    torch::Tensor visibility_;
    int64_t N_;
    std::unordered_map<torch::TensorImpl*, State> state_;
};