/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>

#include <cooperative_groups.h>
#include <algorithm>

namespace cg = cooperative_groups;

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) 
{
    auto lambda = [&t](size_t N) 
    {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const float limx_neg,
    const float limx_pos,
    const float limy_neg,
    const float limy_pos,
    const torch::Tensor& dc,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug, const bool no_color) 
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) 
    { 
        AT_ERROR("means3D must have dimensions (num_points, 3)"); 
    }

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;
    int M = 0;
    if(sh.size(0) != 0) 
    { 
        M = sh.size(1);  // M = 15
    }

    int rendered = 0;
    int num_buckets = 0;
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts = means3D.options().dtype(torch::kInt32);
    torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
    torch::Tensor out_final_T = torch::full({H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, int_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
    std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

    if(P != 0) 
    {
        auto tup = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            sampleFunc,
            P, degree, M,
            background.contiguous().data<float>(),
            W, H,
            means3D.contiguous().data<float>(),
            dc.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data<float>(), 
            opacity.contiguous().data<float>(), 
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(), 
            viewmatrix.contiguous().data<float>(), 
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            limx_neg,
            limx_pos,
            limy_neg,
            limy_pos,
            prefiltered,
            out_color.contiguous().data<float>(),
            out_final_T.contiguous().data<float>(),
            radii.contiguous().data<int>(),
            debug, no_color);
            
        rendered = std::get<0>(tup);
        num_buckets = std::get<1>(tup);
    }

    return std::make_tuple(rendered, num_buckets, out_color, out_final_T, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float limx_neg,
    const float limx_pos,
    const float limy_neg,
    const float limy_pos,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dc,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const int B,
    const torch::Tensor& sampleBuffer,
    const float lambda_erank,
    const bool debug) 
{
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);
    int M = 0;
    if(sh.size(0) != 0) 
    {	
        M = sh.size(1); 
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors_precomp = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacities = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3Ds_precomp = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

    if(P != 0) 
    {  
        CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
            background.contiguous().data<float>(),
            W, H, 
            means3D.contiguous().data<float>(),
            dc.contiguous().data<float>(),
            sh.contiguous().data<float>(),
            colors.contiguous().data<float>(),
            scales.data_ptr<float>(),
            scale_modifier,
            rotations.data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(),
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            limx_neg,
            limx_pos,
            limy_neg,
            limy_pos,
            radii.contiguous().data<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data<float>(),
            dL_dmeans2D.contiguous().data<float>(),
            dL_dconic.contiguous().data<float>(),  
            dL_dopacities.contiguous().data<float>(),
            dL_dcolors_precomp.contiguous().data<float>(),
            dL_dmeans3D.contiguous().data<float>(),
            dL_dcov3Ds_precomp.contiguous().data<float>(),
            dL_ddc.contiguous().data<float>(),
            dL_dsh.contiguous().data<float>(),
            dL_dscales.contiguous().data<float>(),
            dL_drotations.contiguous().data<float>(),
            lambda_erank,
            debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors_precomp, dL_dopacities, dL_dmeans3D, dL_dcov3Ds_precomp, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

void adamUpdate(
    torch::Tensor &param,
    torch::Tensor &param_grad,
    torch::Tensor &exp_avg,
    torch::Tensor &exp_avg_sq,
    torch::Tensor &visible,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
){
    ADAM::adamUpdate(
        param.contiguous().data<float>(),
        param_grad.contiguous().data<float>(),
        exp_avg.contiguous().data<float>(),
        exp_avg_sq.contiguous().data<float>(),
        visible.contiguous().data<bool>(),
        lr,
        b1,
        b2,
        eps,
        N,
        M);
}