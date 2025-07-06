#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/cudaimgproc.hpp>
#include <torch/torch.h>

namespace tensor_utils
{

inline void deleter(void* arg) {}

/**
 * @brief 
 * 
 * @param mat  {rows, cols, channels}
 * @param device_type 
 * @return torch::Tensor {channels, rows, cols}
 */
inline torch::Tensor cvMat2TorchTensor_Float32(
    cv::Mat& mat,
    torch::DeviceType device_type,
    bool use_pinned_memory = false)
{
    torch::Tensor mat_tensor, tensor;

    switch (mat.channels())
    {
    case 1:
    {
        mat_tensor = torch::from_blob(mat.data, /*sizes=*/{mat.rows, mat.cols}, torch::kFloat32);
        tensor = mat_tensor.clone();
    }
    break;

    case 3:
    {
        mat_tensor = torch::from_blob(mat.data, /*sizes=*/{mat.rows, mat.cols, mat.channels()}, torch::kFloat32);
        tensor = mat_tensor.clone();
        tensor = tensor.permute({2, 0, 1});
    }
    break;
    
    default:
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    break;
    }

    if (use_pinned_memory && device_type == torch::kCPU) 
    {
        tensor = tensor.pin_memory();
    }

    tensor = tensor.to(device_type, /*non_blocking=*/use_pinned_memory);

    return tensor.contiguous();
}

inline cv::Mat torchTensor2CvMat_Float32(torch::Tensor& tensor)
{
    cv::Mat mat;
    torch::Tensor mat_tensor = tensor.clone();

    switch (mat_tensor.ndimension())
    {
    case 2:
    {
        mat = cv::Mat(/*rows=*/mat_tensor.size(0),
                      /*cols=*/mat_tensor.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/mat_tensor.data_ptr<float>());
    }
    break;

    case 3:
    {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
        mat_tensor = mat_tensor.to(torch::kCPU);
        mat = cv::Mat(/*rows=*/mat_tensor.size(0),
                      /*cols=*/mat_tensor.size(1),
                      /*type=*/CV_32FC3,
                      /*data=*/mat_tensor.data_ptr<float>());
    }
    break;
    
    default:
        std::cerr << "The tensor has unsupported number of dimensions!" << std::endl;
    break;
    }

    return mat.clone();
}

inline torch::Tensor cvGpuMat2TorchTensor_Float32(cv::cuda::GpuMat& mat)
{
    torch::Tensor mat_tensor, tensor;
    int64_t step = mat.step / sizeof(float);

    switch (mat.channels())
    {
    case 1:
    {
        std::vector<int64_t> strides = {step, 1};
        mat_tensor = torch::from_blob(
            mat.data,
            /*sizes=*/{mat.rows, mat.cols},
            strides,
            deleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone();
    }
    break;

    case 3:
    {
        std::vector<int64_t> strides = {step, static_cast<int64_t>(mat.channels()), 1};
        mat_tensor = torch::from_blob(
            mat.data,
            /*sizes=*/{mat.rows, mat.cols, mat.channels()},
            strides,
            deleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone().permute({2, 0, 1});
    }
    break;
    
    default:
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    break;
    }

    return tensor.contiguous();
}

inline cv::cuda::GpuMat torchTensor2CvGpuMat_Float32(torch::Tensor& tensor)
{
    cv::cuda::GpuMat mat;
    torch::Tensor mat_tensor = tensor.clone();

    switch (mat_tensor.ndimension())
    {
    case 2:
    {
        mat = cv::cuda::GpuMat(/*rows=*/mat_tensor.size(0),
                               /*cols=*/mat_tensor.size(1),
                               /*type=*/CV_32FC1,
                               /*data=*/mat_tensor.data_ptr<float>());
    }
    break;

    case 3:
    {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
        mat = cv::cuda::GpuMat(/*rows=*/mat_tensor.size(0),
                               /*cols=*/mat_tensor.size(1),
                               /*type=*/CV_32FC3,
                               /*data=*/mat_tensor.data_ptr<float>());
    }
    break;

    default:
        std::cerr << "The tensor has unsupported number of channels!" << std::endl;
    break;
    }

    return mat.clone();
}

inline torch::Tensor EigenMatrix2TorchTensor(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto eigen_matrix_T = eigen_matrix;
    eigen_matrix_T.transposeInPlace();
    torch::Tensor tensor = torch::from_blob(
        /*data=*/eigen_matrix_T.data(),
        /*sizes=*/{eigen_matrix.rows(), eigen_matrix.cols()},
        /*options=*/torch::TensorOptions().dtype(torch::kFloat)
    ).clone();

    tensor = tensor.to(device_type);
    return tensor;
}

}
