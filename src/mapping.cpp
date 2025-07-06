#include "mapping.h"
#include "gaussian.h"

#include <atomic>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;

std::atomic<bool> exit_flag(false);
void checkESC() 
{
    struct termios oldt, newt;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    while (!exit_flag) 
    {
        int ch = getchar();
        if (ch == 27) 
        {
            exit_flag = true;
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
}

void pointCallback(const sensor_msgs::PointCloud2ConstPtr& point_msg) 
{
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}

void poseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg) 
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg) 
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

bool getAlignedData(Frame& cur_frame)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty()) 
    {
        return false;
    }

    double frame_time = point_buf.front()->header.stamp.toSec();

    while (1) 
    {
        if (pose_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            pose_buf.pop();
            if (pose_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (pose_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (image_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            image_buf.pop();
            if (image_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (image_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.image_msg = cur_image;

    point_buf.pop();
    pose_buf.pop();
    image_buf.pop();

    return true;
}

void mapping(const YAML::Node &node)
{
    torch::jit::setGraphExecutorOptimize(false);

    std::thread esc_thread(checkESC);

    Params prm(node);
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;

    Frame cur_frame;
    while (!exit_flag)
    {
        /// [1]
        m_buf.lock();
        bool flag = getAlignedData(cur_frame);
        m_buf.unlock();
        if (!flag) continue;
        
        /// [2]
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        double tadd = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << "\033[33m--------- Add Frame " 
                  << dataset->all_frame_num_ - 1 
                  << " | "
                  << tadd * 1000 << "ms"
                  << " ---------\033[0m" << std::endl;
        total_adding_time += tadd;

        if (!dataset->is_keyframe_current_) continue; 

        if (!gaussians->is_init_)
        {
            /// [3]
            gaussians->is_init_ = true;
            gaussians->init(dataset);
            gaussians->trainingSetup();
        }
        else 
        {
            /// [4]
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
            double texd = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            total_extending_time += texd;
        }

        /// [5]
        t_start = std::chrono::steady_clock::now();
        int updated_num = optimize(dataset, gaussians);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        double topt = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << "\033[32m--------- Optimize Gauss Map" 
                  << " | "
                  << updated_num << " updated"
                  << " | "
                  << topt * 1000 << "ms"
                  << " ---------\033[0m" << std::endl << std::endl;
        total_mapping_time += topt;
    }

    esc_thread.join();

    std::cout << "\n🌟 Total Mapping Time " << total_mapping_time << std::endl;
    std::cout << "\n   👉 Forward Time " << gaussians->t_forward_ << std::endl;
    std::cout << "\n   👉 Backward Time " << gaussians->t_backward_ << std::endl;
    std::cout << "\n   👉 Step Time " << gaussians->t_step_ << std::endl;
    std::cout << "\n   👉 Optlist Time " << gaussians->t_optlist_ << std::endl;
    std::cout << "\n   👉 Tocuda Time " << gaussians->t_tocuda_ << std::endl;
    std::cout << "\n🌟 Total Adding Time " << total_adding_time << std::endl;
    std::cout << "\n🌟 Total Extending Time " << total_extending_time << std::endl;

    /// [6]
    torch::NoGradGuard no_grad;
    evaluateVisualQuality(dataset, gaussians);

    /// [7]
    gaussians->saveMap();

    std::cout << "\n😋 Gaussian-LIC done!\n";
}

int main(int argc, char** argv)
{
    std::cout << "\n😋 Gaussian-LIC\n";
    ros::init(argc, argv, "gaussianlic");
    ros::NodeHandle nh("~");
    ros::Rate loop_rate(1000);
    image_transport::ImageTransport it_(nh);

    ros::Subscriber sub_point = nh.subscribe("/keyframe_points", 10000, pointCallback);
    ros::Subscriber sub_pose = nh.subscribe("/keyframe_pose", 10000, poseCallback);
    image_transport::Subscriber image_sub = it_.subscribe("/keyframe_image", 10000, imageCallback);

    std::string config_path;
    nh.param<std::string>("config_path", config_path, "");
    YAML::Node config_node = YAML::LoadFile(config_path);

    std::thread mapping_process(mapping, config_node);
    
    ros::spin();
    
    return 0;
}