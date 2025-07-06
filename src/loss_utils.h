/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <vector>

#include <torch/torch.h>

#include <fused-ssim/ssim.h>

namespace loss_utils
{

inline torch::Tensor l1_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::abs(network_output - gt).mean();
}

inline torch::Tensor psnr(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).mean();
    return 10.0f * torch::log10(1.0f / mse);
}

/** def psnr(img1, img2):
 *     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
 *     return 20 * torch.log10(1.0 / torch.sqrt(mse))
 */
inline torch::Tensor psnr_gaussian_splatting(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).view({img1.size(0) , -1}).mean(1, /*keepdim=*/true);
    return 20.0f * torch::log10(1.0f / torch::sqrt(mse)).mean();
}

inline torch::Tensor gaussian(
    int window_size,
    float sigma,
    torch::DeviceType device_type = torch::kCUDA)
{
    std::vector<float> gauss_values(window_size);
    for (int x = 0; x < window_size; ++x) {
        int temp = x - window_size / 2;
        gauss_values[x] = std::exp(-temp * temp / (2.0f * sigma * sigma));
    }
    torch::Tensor gauss = torch::tensor(
        gauss_values,
        torch::TensorOptions().device(device_type));
    return gauss / gauss.sum();
}

inline torch::autograd::Variable create_window(
    int window_size,
    int64_t channel,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto _1D_window = gaussian(window_size, 1.5f, device_type).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).to(torch::kFloat).unsqueeze(0).unsqueeze(0);
    auto window = torch::autograd::Variable(_2D_window.expand({channel, 1, window_size, window_size}).contiguous());
    return window;
}

/*ssim*/

inline torch::Tensor _ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::autograd::Variable &window,
    int window_size,
    int64_t channel,
    bool size_average = true)
{
    int window_size_half = window_size / 2;
    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));
    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_sq;
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu2_sq;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_mu2;

    auto C1 = 0.01 * 0.01;
    auto C2 = 0.03 * 0.03;

    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average)
        return ssim_map.mean();
    else
        return ssim_map.mean(1).mean(1).mean(1);
}

inline torch::Tensor ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true)
{
    auto channel = img1.size(-3);
    auto window = create_window(window_size, channel, device_type);

    // window = window.to(img1.device());
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

const float C1 = std::pow(0.01, 2);
const float C2 = std::pow(0.03, 2);

/*fused-ssim*/

class FusedSSIMMap : public torch::autograd::Function<FusedSSIMMap> 
{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, float C1, float C2, 
                                 torch::Tensor& img1, torch::Tensor& img2) 
    {
        std::string padding = "same";
        bool train = true;

        auto result = fusedssim(C1, C2, img1, img2, train);
        torch::Tensor ssim_map = std::get<0>(result);
        torch::Tensor dm_dmu1 = std::get<1>(result);
        torch::Tensor dm_dsigma1_sq = std::get<2>(result);
        torch::Tensor dm_dsigma12 = std::get<3>(result);

        if (padding == "valid") 
        {
            ssim_map = ssim_map.slice(2, 5, -5).slice(3, 5, -5);
        }

        ctx->save_for_backward({img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
        ctx->saved_data["C1"] = C1;
        ctx->saved_data["C2"] = C2;
        ctx->saved_data["padding"] = padding;

        return ssim_map;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        torch::Tensor img1 = saved[0];
        torch::Tensor img2 = saved[1];
        torch::Tensor dm_dmu1 = saved[2];
        torch::Tensor dm_dsigma1_sq = saved[3];
        torch::Tensor dm_dsigma12 = saved[4];

        float C1 = static_cast<float>(ctx->saved_data["C1"].toDouble());
        float C2 = static_cast<float>(ctx->saved_data["C2"].toDouble());
        std::string padding = ctx->saved_data["padding"].toStringRef();

        torch::Tensor dL_dmap = grad_output[0];
        if (padding == "valid") 
        {
            dL_dmap = torch::zeros_like(img1);
            dL_dmap.slice(2, 5, -5).slice(3, 5, -5) = grad_output[0];
        }

        torch::Tensor grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);

        return {torch::Tensor(), torch::Tensor(), grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

inline torch::Tensor fused_ssim(torch::Tensor& img1, torch::Tensor& img2) 
{    
    torch::Tensor map = FusedSSIMMap::apply(C1, C2, img1, img2);
    return map.mean();
}

}
