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
#include <rcv.hpp>

//#include "optical_flow/KeyframeTracker.h"

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

    Eigen::Quaternionf imu_quat_;

    //KeyframeTracker tracker_;

    // Parameters
    int num_keypoints_param_;
    double matchscore_thresh_param_;
    bool backwards_filter_param_;
    int downsample_times_param_;
    double backwards_threshold_param_;
    double cam2imu_roll_param_;
    double cam2imu_pitch_param_;
    double cam2imu_yaw_param_;

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

  // The camera to IMU transform, applied as yaw*pitch*roll.
  // These parameters are all in degrees.
  private_nh.param("cam2imu_roll",   cam2imu_roll_param_,  0.0);
  private_nh.param("cam2imu_pitch",  cam2imu_pitch_param_, 0.0);
  private_nh.param("cam2imu_yaw",    cam2imu_yaw_param_,   0.0);

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

// ######################################################################
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
tf::Transform eigen2tf(Eigen::Quaternionf const & rotation)
{
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(0, 0, 0));
  transform.setRotation(
      tf::Quaternion(
        rotation.x(), rotation.y(),
        rotation.z(), rotation.w()));

  return transform;
}

// ######################################################################
void OpticalFlow::imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(input_img_ptr, sensor_msgs::image_encodings::MONO8);

    cv::Mat input_image = cv_ptr->image;

    double const focal_length_x = 49.3804*10;
    double const focal_length_y = 49.3804*10;

    // Transformation to flip the axis into the camera coordinate system
    Eigen::Matrix3f ENUfromNED;
    ENUfromNED << 0,  1,  0,
                  1,  0,  0,
                  0,  0, -1;

    // The IMU to Camera transformation
    Eigen::Quaternionf imu2camera = 
      Eigen::AngleAxisf(M_PI/180.0*cam2imu_yaw_param_,   Eigen::Vector3f::UnitZ()) * 
      Eigen::AngleAxisf(M_PI/180.0*cam2imu_pitch_param_, Eigen::Vector3f::UnitY()) * 
      Eigen::AngleAxisf(M_PI/180.0*cam2imu_roll_param_,  Eigen::Vector3f::UnitX());

    // The camera orientation in world coordinates
    Eigen::Quaternionf camera_rotation(ENUfromNED * imu_quat_ * imu2camera.inverse());

    // The camera intrinsic parameters
    Eigen::Matrix3f K;
    K << focal_length_x, 0,              input_image.cols/2,
         0,              focal_length_y, input_image.rows/2,
         0,              0,              1;

    // http://en.wikipedia.org/wiki/Homography#3D_plane_to_plane_equation
    Eigen::Matrix3f warp_matrix_eigen = K * camera_rotation * K.inverse();

    // Warp the image
    cv::Mat warp_matrix = eigen2cv<float>(warp_matrix_eigen);
    cv::Mat warped_image;
    cv::warpPerspective(input_image, warped_image, warp_matrix, input_image.size());

    // Send a tf frame to show where the camera is pointing
    tf_pub_.sendTransform(tf::StampedTransform(eigen2tf(camera_rotation), ros::Time::now(), "world", "camera"));

    Eigen::Quaternionf warp_to(ENUfromNED * Eigen::Quaternionf::Identity());
    tf_pub_.sendTransform(tf::StampedTransform(eigen2tf(warp_to), ros::Time::now(), "world", "warp_to"));


    // Display the image
    cv::imshow("display", rcv::hcat(input_image, warped_image));

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

