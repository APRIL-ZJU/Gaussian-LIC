#include "rasterizer.h"

torch::autograd::tensor_list
GaussianRasterizerFunction::forward(
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
    GaussianRasterizationSettings raster_settings)
{
    auto rasterization_result = RasterizeGaussiansCUDA(
        raster_settings.bg_,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier_,
        cov3Ds_precomp,
        raster_settings.viewmatrix_,
        raster_settings.projmatrix_,
        raster_settings.tanfovx_,
        raster_settings.tanfovy_,
        raster_settings.image_height_,
        raster_settings.image_width_,
        raster_settings.limx_neg_,
        raster_settings.limx_pos_,
        raster_settings.limy_neg_,
        raster_settings.limy_pos_,
        dc,
        sh,
        raster_settings.sh_degree_,
        raster_settings.campos_,
        raster_settings.prefiltered_,
        raster_settings.debug_,
        raster_settings.no_color_
    );
    auto num_rendered = std::get<0>(rasterization_result);
    auto num_buckets = std::get<1>(rasterization_result);
    auto color = std::get<2>(rasterization_result);
    auto final_T = std::get<3>(rasterization_result);
    auto radii = std::get<4>(rasterization_result);
    auto geomBuffer = std::get<5>(rasterization_result);
    auto binningBuffer = std::get<6>(rasterization_result);
    auto imgBuffer = std::get<7>(rasterization_result);
    auto sampleBuffer = std::get<8>(rasterization_result);

    ctx->saved_data["num_rendered"] = num_rendered;
    ctx->saved_data["num_buckets"] = num_buckets;
    ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier_;
    ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
    ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
    ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;
    ctx->saved_data["limx_neg"] = raster_settings.limx_neg_;
    ctx->saved_data["limx_pos"] = raster_settings.limx_pos_;
    ctx->saved_data["limy_neg"] = raster_settings.limy_neg_;
    ctx->saved_data["limy_pos"] = raster_settings.limy_pos_;
    ctx->saved_data["lambda_erank"] = raster_settings.lambda_erank_;
    ctx->save_for_backward({raster_settings.bg_,
                            raster_settings.viewmatrix_,
                            raster_settings.projmatrix_,
                            raster_settings.campos_,
                            colors_precomp,
                            means3D,
                            scales,
                            rotations,
                            cov3Ds_precomp,
                            radii,
                            dc,
                            sh,
                            geomBuffer,
                            binningBuffer,
                            imgBuffer,
                            sampleBuffer});
    return {color, radii, final_T};
}

torch::autograd::tensor_list
GaussianRasterizerFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs)
{
    auto num_rendered = ctx->saved_data["num_rendered"].toInt();
    auto num_buckets = ctx->saved_data["num_buckets"].toInt();
    auto scale_modifier = static_cast<float>(ctx->saved_data["scale_modifier"].toDouble());
    auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
    auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
    auto sh_degree = ctx->saved_data["sh_degree"].toInt();
    auto limx_neg = static_cast<float>(ctx->saved_data["limx_neg"].toDouble());
    auto limx_pos = static_cast<float>(ctx->saved_data["limx_pos"].toDouble());
    auto limy_neg = static_cast<float>(ctx->saved_data["limy_neg"].toDouble());
    auto limy_pos = static_cast<float>(ctx->saved_data["limy_pos"].toDouble());
    auto lambda_erank = static_cast<float>(ctx->saved_data["lambda_erank"].toDouble());
    auto saved = ctx->get_saved_variables();
    auto bg = saved[0];
    auto viewmatrix = saved[1];
    auto projmatrix = saved[2];
    auto campos = saved[3];
    auto colors_precomp = saved[4];
    auto means3D = saved[5];
    auto scales = saved[6];
    auto rotations = saved[7];
    auto cov3Ds_precomp = saved[8];
    auto radii = saved[9];
    auto dc = saved[10];
    auto sh = saved[11];
    auto geomBuffer = saved[12];
    auto binningBuffer = saved[13];
    auto imgBuffer = saved[14];
    auto sampleBuffer = saved[15];

    auto dL_dcolor = grad_outputs[0];
    // auto dL_dradii = grad_outputs[1];
    // auto dL_dfinal_T = grad_outputs[2];
    auto rasterization_backward_result = RasterizeGaussiansBackwardCUDA(
        bg,
        means3D,
        radii,
        colors_precomp,
        scales,
        rotations,
        scale_modifier,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        limx_neg,
        limx_pos,
        limy_neg,
        limy_pos,
        dL_dcolor,
        dc,
        sh,
        sh_degree,
        campos,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        num_buckets,
        sampleBuffer,
        lambda_erank,
        false
    );

    return {
        std::get<3>(rasterization_backward_result)/*dL_dmeans3D*/,
        std::get<0>(rasterization_backward_result)/*dL_dmeans2D*/,
        std::get<5>(rasterization_backward_result)/*dL_ddc*/,
        std::get<6>(rasterization_backward_result)/*dL_dsh*/,
        std::get<1>(rasterization_backward_result)/*dL_dcolors_precomp*/,
        std::get<2>(rasterization_backward_result)/*dL_dopacities*/,
        std::get<7>(rasterization_backward_result)/*dL_dscales*/,
        std::get<8>(rasterization_backward_result)/*dL_drotations*/,
        std::get<4>(rasterization_backward_result)/*dL_dcov3Ds_precomp*/,
        torch::Tensor()/*dL_draster_setting*/
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizer::forward(
    torch::Tensor means3D,
    torch::Tensor means2D,
    torch::Tensor opacities,
    torch::Tensor dc,
    torch::Tensor shs,
    torch::Tensor colors_precomp,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor cov3D_precomp)

{
    auto raster_settings = this->raster_settings_;
    torch::TensorOptions options;
    colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
    cov3D_precomp = torch::tensor({}, options.device(torch::kCUDA));

    auto result = GaussianRasterizerFunction::apply(
        means3D,
        means2D,
        dc,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        raster_settings
    );
    return std::make_tuple(result[0], result[1], result[2]);
}