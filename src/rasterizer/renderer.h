#pragma once

#include <tuple>
#include <torch/torch.h>

#include "camera.h"
#include "gaussian.h"
#include "rasterizer.h"

class GaussianModel;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render(const std::shared_ptr<Camera>& viewpoint_camera,
       std::shared_ptr<GaussianModel> pc,
       torch::Tensor& bg_color,
       bool use_trained_exposure = false,
       bool no_color = false,
       float scaling_modifier = 1.0);