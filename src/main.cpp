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

#include "optical_flow/KeyframeTracker.h"

// ######################################################################
class OpticalFlow
{
  typedef double EigenPrecision;
  typedef Eigen::Quaternion<EigenPrecision> Quaternion;

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

    cv::Mat key_image_;
    std::vector<cv::Point2f> key_corners_;

    cv::Mat global_transform_;

    Quaternion imu_quat_;

    // Parameters
    int num_keypoints_param_;
    double matchscore_thresh_param_;
    bool backwards_filter_param_;
    int downsample_times_param_;
    double backwards_threshold_param_;


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
  imu_quat_ = Quaternion(
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
//! Draw the features onto the image, and draw lines between matches
void drawFeatures(cv::Mat & image,
    std::vector<cv::Point2f> const & old_corners, std::vector<cv::Point2f> const & new_corners)
{
  assert(old_corners.size() == new_corners.size());
  for(size_t i=0; i<new_corners.size(); ++i)
  {
    cv::rectangle(image, old_corners[i]-cv::Point2f(3,3), old_corners[i]+cv::Point2f(2,2), 255, 5);
    cv::rectangle(image, old_corners[i]-cv::Point2f(3,3), old_corners[i]+cv::Point2f(3,3), 0,   1);

    cv::rectangle(image, new_corners[i]-cv::Point2f(3,3), new_corners[i]+cv::Point2f(2,2), 0,   5);
    cv::rectangle(image, new_corners[i]-cv::Point2f(3,3), new_corners[i]+cv::Point2f(3,3), 255, 1);

    cv::line(image, old_corners[i], new_corners[i], 0,   5); 
    cv::line(image, old_corners[i], new_corners[i], 255, 1); 
  }
}

// ######################################################################
//! Filter out all points from p1 and p2 with corresponding status byte == 0
void filterPoints(std::vector<cv::Point2f> & p1, std::vector<cv::Point2f> & p2,
    std::vector<unsigned char> const & status)
{
  std::vector<cv::Point2f> p1_filt;     p1_filt.reserve(p1.size());
  std::vector<cv::Point2f> p2_filt; p2_filt.reserve(p1.size());

  std::vector<cv::Point2f>::iterator p1_it = p1.begin();
  std::vector<cv::Point2f>::iterator p2_it = p2.begin();
  std::vector<unsigned char>::const_iterator status_it = status.begin();

  while(status_it != status.end())
  {
    if(*status_it > 0)
    {
      p1_filt.push_back(*p1_it);
      p2_filt.push_back(*p2_it);
    }
    ++p1_it;
    ++p2_it;
    ++status_it;
  }
  p1 = p1_filt;
  p2 = p2_filt;
}

// ######################################################################
//! Perform forward/backward LK tracking.
/*! @param key_image The old keyframe image
    @param curr_image The new incoming image
    @param corners The old corners, found (e.g. by cv::goodFeaturesToTrack) in the key_image
    @param new_corners A vector which will be filled with the locations of the corners in curr_image
    @param backwards_filter Should we perform the backwards filtering step? (see note below)
    @param backwards_threshold How far away can a point travel during the backwards step before it is considered a bad track?
    
    \note This method tries very hard to filter out bad tracks by performing both a forwards, and a backwards LK tracking step.
          All untrackable points will be removed from the corners vector, so that corners and new_corners will have the same size
          after trackFeatures() is finished. */
void trackFeatures(cv::Mat key_image, cv::Mat curr_image,
    std::vector<cv::Point2f> & corners, std::vector<cv::Point2f> & new_corners,
    bool backwards_filter = true, double backwards_threshold = 10.0)
{
  cv::Size const searchWindow(15, 15);

  new_corners.resize(corners.size());

  if(corners.size() == 0) return;

  // Perform the forward LK step
  std::vector<unsigned char> status(corners.size());
  std::vector<float> error(corners.size());
  calcOpticalFlowPyrLK(key_image, curr_image, corners, new_corners, status, error, searchWindow, 5);

  // Filter out any untrackable points
  filterPoints(corners, new_corners, status);

  if(corners.size() == 0) return;

  if(backwards_filter == false) return;

  // Perform the backwards LK step
  std::vector<cv::Point2f> back_corners;
  calcOpticalFlowPyrLK(curr_image, key_image, new_corners, back_corners, status, error, searchWindow, 5);
  
  // Filter out points that either:
  // a) Are untrackable by LK (their status byte == 0)
  // or 
  // b) Were tracked by the backward LK step to a position that was too far away from the original point. This indicates
  //    that even though LK though it had a good track, the point path was ambiguous.
  std::vector<cv::Point2f> filt_corners;
  std::vector<cv::Point2f> filt_old_corners;
  std::vector<cv::Point2f>::iterator corners_it = corners.begin();
  std::vector<cv::Point2f>::iterator new_corners_it = new_corners.begin();
  std::vector<cv::Point2f>::iterator back_corners_it = back_corners.begin();
  std::vector<unsigned char>::iterator status_it = status.begin();
  while(status_it != status.end())
  {
    float const dist = sqrt(pow(new_corners_it->y - back_corners_it->y, 2) + pow(new_corners_it->y - back_corners_it->y, 2));
    if(*status_it && dist < backwards_threshold)
    {
      filt_old_corners.push_back(*corners_it);
      filt_corners.push_back(*new_corners_it);
    }
    ++corners_it;
    ++new_corners_it;
    ++back_corners_it;
    ++status_it;
  }
  new_corners = filt_corners;
  corners     = filt_old_corners;
}

// ######################################################################
void OpticalFlow::imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(input_img_ptr, sensor_msgs::image_encodings::MONO8);

    cv::Mat input_image = cv_ptr->image;

    // @@@@@@@@@@@@@@@ START TIMING @@@@@@@@@@@@@@@ 
    ros::Time start_time = ros::Time::now();

    // Scale the incoming image down
    for(int i=0; i<downsample_times_param_; ++i)
      cv::pyrDown(input_image, input_image);

    // Grab a new keyframe whenever we have lost more than half of our tracks
    if(key_corners_.size() < size_t(num_keypoints_param_ / 2))
    {
      cv::goodFeaturesToTrack(input_image, key_corners_, num_keypoints_param_, 0.01, 10);
      key_image_ = input_image.clone();
    }

    // Track the features from the keyframe to the current frame
    std::vector<cv::Point2f> new_corners;
    trackFeatures(key_image_, input_image,
        key_corners_, new_corners,
        backwards_filter_param_, backwards_threshold_param_);

    cv::Mat warped_key_image;
    if(new_corners.size() < size_t(num_keypoints_param_/2))
    {
      // If we didn't track enough points, then kill the keyframe
      key_corners_.clear();
    }
    else
    {
      // Find the homography between the last frame and the current frame
      cv::Mat homography = cv::findHomography(key_corners_, new_corners, CV_RANSAC);

      // Warp the keyframe image to the new image, and find the squared difference
      cv::warpPerspective(key_image_, warped_key_image, homography, key_image_.size());
      double matchError = cv::norm(input_image, warped_key_image, cv::NORM_L2);

      // If the difference between the warped template and the new frame is too large,
      // then kill the keyframe
      if(matchError > matchscore_thresh_param_)
        key_corners_.clear();
    }

    ros::Time end_time = ros::Time::now();
    // @@@@@@@@@@@@@@@ END TIMING @@@@@@@@@@@@@@@ 

    double duration_ms = (end_time - start_time).toSec() * 1000.0;
    if(avg_time_ < 0) avg_time_ = duration_ms;
    else avg_time_ = 0.2 * duration_ms + 0.8 * avg_time_;
    ROS_INFO("Processing took: %0.4fms (%0.4fhz)",  avg_time_, 1000.0 / avg_time_); 

    // Draw the warped keyframe
    if(key_corners_.size())
      cv::imshow("keyframe", warped_key_image);
    else
      cv::imshow("keyframe", input_image);

    // Draw the features on the input image
    if(key_corners_.size())
      drawFeatures(input_image, key_corners_, new_corners);
    cv::imshow("tracker (press key to force keyframe)", input_image);

    if(cv::waitKey(2) != -1) key_corners_.clear();

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

