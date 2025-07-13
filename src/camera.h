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

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <torch/torch.h>
#include <Eigen/Eigen>

#include "tensor_utils.h"

class Camera
{
public:
    Camera(){}

    void setIntrinsic(double w, double h, 
                      double fx, double fy,
                      double cx, double cy)
    {
        image_width_ = w;
        image_height_ = h;
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        FoVx_ = 2.0 * std::atan(w / (2.0 * fx));
        FoVy_ = 2.0 * std::atan(h / (2.0 * fy));
    }

    void setPose(const Eigen::Matrix3d& R_wc, 
                 const Eigen::Vector3d& t_wc)
    {
        R_cw_ = R_wc.transpose();
        t_cw_ = - R_wc.transpose() * t_wc;

        setWorldViewTransform();
        setProjectionMatrix();
        full_proj_transform_ = (world_view_transform_.unsqueeze(0).bmm(projection_matrix_.unsqueeze(0))).squeeze(0);
        camera_center_ = world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});

        limx_neg_ = - 0.15 * image_width_ / fx_ - cx_ / fx_;
        limx_pos_ = 1.15 * image_width_ / fx_ - cx_ / fx_;
        limy_neg_ = - 0.15 * image_height_ / fy_ - cy_ / fy_;
        limy_pos_ = 1.15 * image_height_ / fy_ - cy_ / fy_;
    }

    void setWorldViewTransform()
    {
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = R_cw_.cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = t_cw_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f C2W = Rt.inverse();
        Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
        cam_center += trans_;
        cam_center *= scale_;
        C2W.block<3, 1>(0, 3) = cam_center;
        Rt = C2W.inverse();  // Tcw

        world_view_transform_ = tensor_utils::EigenMatrix2TorchTensor(Rt, torch::kCUDA).transpose(0, 1);
    }

    void setProjectionMatrix()
    {
        torch::Tensor P = torch::zeros({4, 4}, torch::kCUDA);

        float W = image_width_;
        float H = image_height_;
        float cx = cx_;
        float cy = cy_;
        float fx = fx_;
        float fy = fy_;
        float znear = znear_;
        float zfar = zfar_;
        P.index({0, 0}) = 1.0 / std::tan(FoVx_ / 2);
        P.index({1, 1}) = 1.0 / std::tan(FoVy_ / 2);
        P.index({0, 2}) = (2 * cx - W) / W;
        P.index({1, 2}) = (2 * cy - H) / H;
        P.index({3, 2}) = 1.0f;
        P.index({2, 2}) = zfar / (zfar - znear);
        P.index({2, 3}) = -(zfar * znear) / (zfar - znear);

        projection_matrix_ = P.transpose(0, 1);
    }

public:
    std::string image_name_;

    int image_width_;              
    int image_height_;
    torch::Tensor original_image_;
          
    float fx_;
    float fy_;
    float cx_; 
    float cy_;   
    float FoVx_; 
    float FoVy_;

    float zfar_ = 100.0f;
    float znear_ = 0.01f;

    Eigen::Vector3f trans_ = Eigen::Vector3f::Zero();
    float scale_ = 1.0;

    Eigen::Matrix3d R_cw_;
    Eigen::Vector3d t_cw_;

    torch::Tensor world_view_transform_; 
    torch::Tensor projection_matrix_;    
    torch::Tensor full_proj_transform_;   
    torch::Tensor camera_center_;      

    float limx_neg_;
    float limx_pos_;
    float limy_neg_;
    float limy_pos_;
};