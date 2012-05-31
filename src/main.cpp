#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/timer.hpp>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <tf/transform_broadcaster.h>

#include "optical_flow/KeyframeTracker.h"

// ######################################################################
class OpticalFlow
{

  public:
    OpticalFlow();
    ~OpticalFlow();

  protected:
    void imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr);
    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);
    void sonarCallback(const sensor_msgs::Range::ConstPtr &msg);

  private:
    ros::NodeHandle nh_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber imu_sub_;
    ros::Subscriber sonar_sub_;
    ros::Publisher cam_pose_pub_;
    tf::TransformBroadcaster tf_pub_;

    cv::Mat key_image_;
    std::vector<cv::Point2f> key_corners_;

    cv::Mat global_transform_;

    Eigen::Quaternionf imu_quat_;

    KeyframeTracker tracker_;

    // Parameters
    int num_keypoints_param_;
    double matchscore_thresh_param_;
    bool backwards_filter_param_;
    int downsample_times_param_;
    double backwards_threshold_param_;
    double roll_param_;
    double pitch_param_;
    double yaw_param_;

    double avg_time_;
};

// ######################################################################
OpticalFlow::OpticalFlow() :
  it_(nh_),
  avg_time_(-1.0)
{
  // Subscriptions/Advertisements
  image_sub_    = it_.subscribe("image", 1, &OpticalFlow::imageCallback, this);
  imu_sub_      = nh_.subscribe("imu",   1, &OpticalFlow::imuCallback,    this);
  sonar_sub_    = nh_.subscribe("sonar", 1, &OpticalFlow::sonarCallback, this);
  cam_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("cam_pose", 10);

  
  // Parameters
  ros::NodeHandle private_nh("~");
  private_nh.param("num_keypoints",       num_keypoints_param_, 50);
  private_nh.param("matchscore_thresh",   matchscore_thresh_param_, 10e8);
  private_nh.param("backwards_filter",    backwards_filter_param_, true);
  private_nh.param("downsample_times",    downsample_times_param_, 2);
  private_nh.param("backwards_threshold", backwards_threshold_param_, 10.0);

  private_nh.param("roll",   roll_param_,  0.0);
  private_nh.param("pitch",  pitch_param_, 0.0);
  private_nh.param("yaw",    yaw_param_,   0.0);

  ROS_WARN("Downsample Times>>>> %d", downsample_times_param_);
  ROS_WARN("numkeypoints %d", num_keypoints_param_);

  global_transform_ = cv::Mat_<double>::eye(3,3);
}

// ######################################################################
OpticalFlow::~OpticalFlow() 
{ }

// ######################################################################
void OpticalFlow::imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{
  imu_quat_ = Eigen::Quaternionf(
      msg->orientation.w, 
      msg->orientation.x, 
      msg->orientation.y, 
      msg->orientation.z); 
}

// ######################################################################
void OpticalFlow::sonarCallback(const sensor_msgs::Range::ConstPtr &msg)
{
  //double range = msg->range;
}

template <class Precision>
cv::Mat eigen2cv(Eigen::Matrix<Precision, Eigen::Dynamic, Eigen::Dynamic> const & eigen_matrix)
{
  cv::Mat_<Precision> cv_matrix(eigen_matrix.rows(), eigen_matrix.cols());
  for(int r=0; r<eigen_matrix.rows(); ++r)
    for(int c=0; c<eigen_matrix.cols(); ++c)
      cv_matrix(r, c) = eigen_matrix(r, c);

  return cv_matrix;
}

// ######################################################################
void OpticalFlow::imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(input_img_ptr, sensor_msgs::image_encodings::MONO8);

    cv::Mat input_image = cv_ptr->image;


    double const focal_length_x = 49.3804;
    double const focal_length_y = 49.3804;

    Eigen::Quaternionf imu2camera = 
      Eigen::AngleAxisf(M_PI/180.0*yaw_param_,   Eigen::Vector3f::UnitZ()) * 
      Eigen::AngleAxisf(M_PI/180.0*pitch_param_, Eigen::Vector3f::UnitY()) * 
      Eigen::AngleAxisf(M_PI/180.0*roll_param_,  Eigen::Vector3f::UnitX());

    Eigen::Quaternionf camera_rotation = imu_quat_ * imu2camera.inverse();
    tf::Transform camera_transform;
    camera_transform.setOrigin(tf::Vector3(0, 0, 0));
    camera_transform.setRotation(
        tf::Quaternion(
          camera_rotation.x(), camera_rotation.y(),
          camera_rotation.z(), camera_rotation.w()));
    tf_pub_.sendTransform(tf::StampedTransform(camera_transform, ros::Time::now(), "world", "camera"));


    Eigen::Matrix3f uv2xy;
    uv2xy << 0, 1, 0,
             1, 0, 0,
             0, 0, 1;

    Eigen::Matrix3f K;
    K << focal_length_x, 0,              input_image.cols/2,
         0,              focal_length_y, input_image.rows/2,
         0,              0,              1;

    // http://en.wikipedia.org/wiki/Homography#3D_plane_to_plane_equation
    Eigen::Matrix3f warp_matrix_eigen = uv2xy * K * camera_rotation * K.inverse() * uv2xy;



    cv::Mat warp_matrix = eigen2cv<float>(warp_matrix_eigen);
    cv::Mat warped_image;
    cv::warpPerspective(input_image, warped_image, warp_matrix, input_image.size());

    //cv::imshow("input_image", input_image);
    cv::imshow("warped_image", warped_image);

    cv::waitKey(2);

  }
  catch(cv_bridge::Exception & e)
  {
    ROS_ERROR("%s:%d: Exception: %s", __FILE__, __LINE__, e.what());  
    return;
  }
}

// ######################################################################
int main(int argc, char ** argv)
{
  ros::init(argc, argv, "contrast_enhancer");
  OpticalFlow of;
  ros::spin();
  return 0;
}

