#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/timer.hpp>

class OpticalFlow
{
  public:
    OpticalFlow();
    ~OpticalFlow();

  protected:
    void imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr);

  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    cv::Mat key_image_;
    std::vector<cv::Point2f> key_corners_;

    int num_keypoints_param_;
    double matchscore_thresh_param_;
};

// ######################################################################
OpticalFlow::OpticalFlow() :
  it_(nh_)
{
  // Subscriptions/Advertisements
  image_sub_ = it_.subscribe("image", 1, &OpticalFlow::imageCallback, this);

  nh_.param("num_keypoints", num_keypoints_param_, 50);
  nh_.param("matchscore_thresh", matchscore_thresh_param_, 10e8);
}

// ######################################################################
OpticalFlow::~OpticalFlow() 
{ 
}

// ######################################################################
void drawFeatures(cv::Mat & image, std::vector<cv::Point2f> const & old_corners, std::vector<cv::Point2f> const & new_corners)
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

void filterPoints(std::vector<cv::Point2f> & p1, std::vector<cv::Point2f> & p2, std::vector<unsigned char> const & status)
{
  std::vector<cv::Point2f> p1_filt;     p1_filt.reserve(p1.size());
  std::vector<cv::Point2f> p2_filt; p2_filt.reserve(p1.size());

  std::vector<cv::Point2f>::iterator p1_it = p1.begin();
  std::vector<cv::Point2f>::iterator p2_it = p2.begin();
  std::vector<unsigned char>::const_iterator status_it = status.begin();

  // Filter out bad tracks
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
void trackFeatures(cv::Mat key_image, cv::Mat curr_image, std::vector<cv::Point2f> & corners, std::vector<cv::Point2f> & new_corners)
{
  cv::Size const searchWindow(15, 15);

  new_corners.resize(corners.size());

  if(corners.size() == 0) return;

  std::vector<unsigned char> status(corners.size());
  std::vector<float> error(corners.size());
  calcOpticalFlowPyrLK(key_image, curr_image, corners, new_corners, status, error, searchWindow, 5);
  filterPoints(corners, new_corners, status);

  if(corners.size() == 0) return;

  std::vector<cv::Point2f> back_corners;
  calcOpticalFlowPyrLK(curr_image, key_image, new_corners, back_corners, status, error, searchWindow, 5);
  
  std::vector<cv::Point2f> filt_corners;
  std::vector<cv::Point2f> filt_old_corners;
  std::vector<cv::Point2f>::iterator corners_it = corners.begin();
  std::vector<cv::Point2f>::iterator new_corners_it = new_corners.begin();
  std::vector<cv::Point2f>::iterator back_corners_it = back_corners.begin();
  std::vector<unsigned char>::iterator status_it = status.begin();

  while(status_it != status.end())
  {
    if(*status_it &&
      sqrt(pow(new_corners_it->y - back_corners_it->y, 2) + pow(new_corners_it->y - back_corners_it->y, 2)) < 10)
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
  corners = filt_old_corners;


}

// ######################################################################
void OpticalFlow::imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(input_img_ptr, sensor_msgs::image_encodings::MONO8);

    cv::Mat input_image = cv_ptr->image;
    cv::pyrDown(input_image, input_image);

    // Grab a new keyframe whenever we have lost more than half of our tracks
    if(key_corners_.size() < size_t(num_keypoints_param_ / 2))
    {
      cv::goodFeaturesToTrack(input_image, key_corners_, num_keypoints_param_, 0.01, 30);
      key_image_ = input_image.clone();
    }

    // Track the features from the keyframe to the current frame
    std::vector<cv::Point2f> new_corners;
    trackFeatures(key_image_, input_image, key_corners_, new_corners);

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
      cv::Mat matchScore;
      cv::matchTemplate(input_image, warped_key_image, matchScore, CV_TM_SQDIFF);
      ROS_INFO("Match Score: %f", matchScore.at<float>(0,0));

      // If the difference between the warped template and the new frame is too large,
      // then kill the keyframe
      if(matchScore.at<float>(0,0) > matchscore_thresh_param_)
        key_corners_.clear();
    }

    // Draw the results
    if(key_corners_.size())
      drawFeatures(input_image, key_corners_, new_corners);

    if(key_corners_.size())
      cv::imshow("homography", warped_key_image);

    cv::imshow("tracker (press key for keyframe)", input_image);

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

