#pragma once

#include <tuple>

#include <torch/torch.h>

#include "rasterize_points.h"

struct GaussianRasterizationSettings
{
    GaussianRasterizationSettings(
        int image_height,
        int image_width,
        float tanfovx,
        float tanfovy,
        float limx_neg,
        float limx_pos,
        float limy_neg,
        float limy_pos,
        torch::Tensor& bg,
        float scale_modifier,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix,
        int sh_degree,
        torch::Tensor& campos,
        bool prefiltered,
        bool debug,
        bool no_color,
        float lambda_erank)
        : image_height_(image_height), image_width_(image_width), tanfovx_(tanfovx), tanfovy_(tanfovy),
          limx_neg_(limx_neg), limx_pos_(limx_pos), limy_neg_(limy_neg), limy_pos_(limy_pos),
          bg_(bg), scale_modifier_(scale_modifier), viewmatrix_(viewmatrix), projmatrix_(projmatrix),
          sh_degree_(sh_degree), campos_(campos), prefiltered_(prefiltered), debug_(debug), 
          no_color_(no_color), lambda_erank_(lambda_erank)
    {}

    int image_height_;
    int image_width_;
    float tanfovx_;
    float tanfovy_;
    float limx_neg_;
    float limx_pos_;
    float limy_neg_;
    float limy_pos_;
    torch::Tensor bg_;
    float scale_modifier_;
    torch::Tensor viewmatrix_;
    torch::Tensor projmatrix_;
    int sh_degree_;
    torch::Tensor campos_;
    bool prefiltered_;
    bool debug_;
    bool no_color_;
    float lambda_erank_;
};

class GaussianRasterizerFunction : public torch::autograd::Function<GaussianRasterizerFunction>
{
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor dc,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3Ds_precomp,
        GaussianRasterizationSettings raster_settings);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs);
};

class GaussianRasterizer : public torch::nn::Module
{
public:
    GaussianRasterizer(GaussianRasterizationSettings& raster_settings)
        : raster_settings_(raster_settings){}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        torch::Tensor dc,
        torch::Tensor shs,
        torch::Tensor colors_precomp,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3D_precomp);

public:
    GaussianRasterizationSettings raster_settings_;
};